//! Reverse mode automatic differentiation.
//!
//! `value_and_grad(f, args...)` needs a function that computes the loss of
//! `f` and the gradient with respect to each of its tensor parameters in one
//! go. This pass derives it from the IR of `f`: the forward operations stay
//! (their results are the activations of the backward pass) and the backward
//! chain is appended, one gradient runtime call per forward operation,
//! walking from the loss back to the parameters.
//!
//! The gradient of a value that flows into several operations is the sum of
//! the contributions of its users (`mlr.add`).
//!
//! Restricted to straight line functions whose operations have a known
//! adjoint; the checker only accepts `value_and_grad` on such functions.

use crate::ir::{Attr, Builder, FuncOp, Op, Type, Value};
use std::collections::HashMap;

/// The operations that have an adjoint, plus the ones that pass through.
const FORWARD: &[&str] = &[
    "arith.constant",
    "mlr.matmul",
    "mlr.relu",
    "mlr.add_bias",
    "mlr.cross_entropy",
    "mlr.sum",
    "mlr.conv2d",
    "mlr.maxpool2d",
    "mlr.embed",
    "mlr.embed_seq",
    "mlr.flatten",
    "func.return",
];

/// Builds `f.value_and_grad` from the IR of `f`: the loss and one gradient
/// slot per tensor parameter. Returns the function and the runtime functions
/// it calls (for the declarations of the module).
pub fn value_and_grad_of(f: &FuncOp) -> (FuncOp, Vec<&'static str>) {
    let mut b = Builder::new();
    let mut used: Vec<&'static str> = Vec::new();

    let mut body = f.body.blocks[0].ops.clone();
    // The loss is the value that the function returns.
    let loss = match body.last() {
        Some(op) if op.name == "func.return" && op.operands.len() == 1 => {
            let v = op.operands[0].clone();
            body.pop();
            v
        }
        _ => panic!("value_and_grad: the function must return one value"),
    };

    // The adjoint of every value: the gradient that has accumulated so far.
    // The loss starts with one.
    let mut adj: HashMap<usize, Value> = HashMap::new();
    let mut ops: Vec<Op> = body.clone();

    let mut one = Op::new("arith.constant");
    one.attrs.push(("value".into(), Attr::Float(1.0)));
    let seed = b.fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
    one.results.push(seed.clone());
    ops.push(one);
    adj.insert(loss.id, seed);

    // The backward pass, from the loss back to the parameters.
    for op in body.iter().rev() {
        let y = match op.results.first() {
            Some(r) => r.clone(),
            None => continue,
        };
        let dy = match adj.get(&y.id) {
            Some(v) => v.clone(),
            // Nothing depends on this value.
            None => continue,
        };
        match op.name.as_str() {
            "mlr.matmul" => {
                // dx = dy @ w^T, dw = x^T @ dy.
                used.push("mlr_matmul_grad");
                let x = op.operands[0].clone();
                let w = op.operands[1].clone();
                let mut g = Op::new("mlr.matmul_grad");
                g.operands = vec![x.clone(), w.clone(), dy];
                let dx = b.fresh_value(x.ty.clone());
                let dw = b.fresh_value(w.ty.clone());
                g.results = vec![dx.clone(), dw.clone()];
                g.loc = op.loc.clone();
                ops.push(g);
                accumulate(&mut ops, &mut b, &mut adj, &x, dx);
                accumulate(&mut ops, &mut b, &mut adj, &w, dw);
            }
            "mlr.add_bias" => {
                // dm = dy, db = the sum of the rows of dy.
                used.push("mlr_add_bias_grad");
                let m = op.operands[0].clone();
                let bias = op.operands[1].clone();
                let mut g = Op::new("mlr.add_bias_grad");
                g.operands = vec![dy];
                let dm = b.fresh_value(m.ty.clone());
                let db = b.fresh_value(bias.ty.clone());
                g.results = vec![dm.clone(), db.clone()];
                g.loc = op.loc.clone();
                ops.push(g);
                accumulate(&mut ops, &mut b, &mut adj, &m, dm);
                accumulate(&mut ops, &mut b, &mut adj, &bias, db);
            }
            "mlr.relu" => {
                // dx = dy where x > 0.
                used.push("mlr_relu_grad");
                let x = op.operands[0].clone();
                let mut g = Op::new("mlr.relu_grad");
                g.operands = vec![x.clone(), dy];
                let dx = b.fresh_value(x.ty.clone());
                g.results = vec![dx.clone()];
                g.loc = op.loc.clone();
                ops.push(g);
                accumulate(&mut ops, &mut b, &mut adj, &x, dx);
            }
            "mlr.sum" => {
                // dx is the incoming gradient in every element.
                used.push("mlr_sum_grad");
                let x = op.operands[0].clone();
                let mut g = Op::new("mlr.sum_grad");
                g.operands = vec![x.clone(), dy];
                let dx = b.fresh_value(x.ty.clone());
                g.results = vec![dx.clone()];
                ops.push(g);
                accumulate(&mut ops, &mut b, &mut adj, &x, dx);
            }
            "mlr.flatten" => {
                // The inverse of the flatten is the reshape of the same
                // layout: a copy.
                used.push("mlr_copy");
                let x = op.operands[0].clone();
                let mut g = Op::new("mlr.flatten_grad");
                g.operands = vec![dy];
                let dx = b.fresh_value(x.ty.clone());
                g.results = vec![dx.clone()];
                ops.push(g);
                accumulate(&mut ops, &mut b, &mut adj, &x, dx);
            }
            "mlr.conv2d" => {
                used.push("mlr_conv2d_grad");
                let x = op.operands[0].clone();
                let w = op.operands[1].clone();
                let bias = op.operands[2].clone();
                let mut g = Op::new("mlr.conv2d_grad");
                g.operands = vec![x.clone(), w.clone(), dy];
                let dx = b.fresh_value(x.ty.clone());
                let dw = b.fresh_value(w.ty.clone());
                let db = b.fresh_value(bias.ty.clone());
                g.results = vec![dx.clone(), dw.clone(), db.clone()];
                g.attrs = op.attrs.clone();
                ops.push(g);
                accumulate(&mut ops, &mut b, &mut adj, &x, dx);
                accumulate(&mut ops, &mut b, &mut adj, &w, dw);
                accumulate(&mut ops, &mut b, &mut adj, &bias, db);
            }
            "mlr.maxpool2d" => {
                used.push("mlr_maxpool2d_grad");
                let x = op.operands[0].clone();
                let mut g = Op::new("mlr.maxpool2d_grad");
                g.operands = vec![x.clone(), dy];
                let dx = b.fresh_value(x.ty.clone());
                g.results = vec![dx.clone()];
                g.attrs = op.attrs.clone();
                ops.push(g);
                accumulate(&mut ops, &mut b, &mut adj, &x, dx);
            }
            "mlr.embed" => {
                used.push("mlr_embed_grad");
                let table = op.operands[0].clone();
                let mut g = Op::new("mlr.embed_grad");
                g.operands = vec![table.clone(), op.operands[1].clone(), dy];
                let dtable = b.fresh_value(table.ty.clone());
                g.results = vec![dtable.clone()];
                ops.push(g);
                accumulate(&mut ops, &mut b, &mut adj, &table, dtable);
            }
            "mlr.embed_seq" => {
                used.push("mlr_embed_seq_grad");
                let table = op.operands[0].clone();
                let mut g = Op::new("mlr.embed_seq_grad");
                g.operands = vec![table.clone(), op.operands[1].clone(), dy];
                let dtable = b.fresh_value(table.ty.clone());
                g.results = vec![dtable.clone()];
                ops.push(g);
                accumulate(&mut ops, &mut b, &mut adj, &table, dtable);
            }
            "mlr.cross_entropy" => {
                // dlogits = (softmax(logits) - one hot(labels)) / rows.
                used.push("mlr_cross_entropy_grad");
                let logits = op.operands[0].clone();
                let labels = op.operands[1].clone();
                let mut g = Op::new("mlr.cross_entropy_grad");
                g.operands = vec![logits.clone(), labels.clone(), dy];
                let dlogits = b.fresh_value(logits.ty.clone());
                g.results = vec![dlogits.clone()];
                g.loc = op.loc.clone();
                ops.push(g);
                accumulate(&mut ops, &mut b, &mut adj, &logits, dlogits);
            }
            "arith.constant" => {}
            other => panic!(
                "value_and_grad: the operation `{}` has no gradient yet",
                other
            ),
        }
    }

    // One result per tensor parameter: its gradient, or a zero tensor when
    // the parameter does not reach the loss.
    let mut results = vec![Type::Scalar(crate::ir::ScalarKind::F32)];
    let mut returns = vec![loss];
    for p in &f.params {
        if !matches!(p.ty, Type::Tensor { .. }) {
            continue;
        }
        let grad = match adj.remove(&p.id) {
            Some(v) => v,
            None => {
                let mut z = Op::new("mlr.zeros");
                let zero = b.fresh_value(p.ty.clone());
                z.results.push(zero.clone());
                z.loc = f.loc.clone();
                ops.push(z);
                zero
            }
        };
        results.push(grad.ty.clone());
        returns.push(grad);
    }

    let mut term = Op::new("func.return");
    term.operands = returns;
    ops.push(term);

    let func = FuncOp {
        name: format!("{}.value_and_grad", f.name),
        params: f.params.clone(),
        results,
        body: crate::ir::Region {
            blocks: vec![crate::ir::Block {
                args: vec![],
                ops,
            }],
        },
        attrs: vec![(
            "effects".to_string(),
            Attr::Str("pure".to_string()),
        )],
        loc: f.loc.clone(),
    };
    (func, used)
}

/// Adds a contribution to the adjoint of a value: the first one becomes the
/// adjoint, the others are summed into it with `mlr.add`.
fn accumulate(
    ops: &mut Vec<Op>,
    b: &mut Builder,
    adj: &mut HashMap<usize, Value>,
    target: &Value,
    contribution: Value,
) {
    let new = match adj.get(&target.id) {
        Some(old) => {
            let mut sum = Op::new("mlr.add");
            sum.operands = vec![old.clone(), contribution];
            let res = b.fresh_value(old.ty.clone());
            sum.results.push(res.clone());
            ops.push(sum);
            res
        }
        None => contribution,
    };
    adj.insert(target.id, new);
}

/// Verifies that a function can be differentiated.
pub fn check_differentiable(f: &FuncOp) -> Result<(), String> {
    for op in &f.body.blocks[0].ops {
        if !FORWARD.contains(&op.name.as_str()) {
            return Err(format!(
                "the operation `{}` has no gradient yet",
                op.name
            ));
        }
        if !op.regions.is_empty() {
            return Err(
                "the function has a region (`map`, `for` or `if`), which cannot \
                 be differentiated yet"
                    .to_string(),
            );
        }
    }
    Ok(())
}
