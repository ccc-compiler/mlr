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
//! Control flow:
//!
//! * an `scf.if` is differentiated branch by branch: each region is reversed
//!   with the adjoint of the result as the seed, and the contributions to
//!   the values outside the region are reported through extra results of the
//!   `scf.if`, where the two branches agree;
//! * an `scf.for` is differentiated when its carried values are only
//!   *accumulated* (`yield acc + expr`, also with `-` and a constant
//!   factor): the backward pass is a loop over the same range that replays
//!   the parts of the body that do not read the carried values, reverses
//!   them with the carried adjoint as the seed, and accumulates the
//!   contributions to the values outside the loop into extra iter args;
//! * `tensor.extract` scatters its adjoint into a zero tensor with
//!   `tensor.insert`; `tensor.insert` passes the adjoint through with the
//!   written position removed.
//!
//! The checker (`check_differentiable`) accepts exactly what the transform
//! below implements; everything else is rejected with a message.

use crate::ir::{Attr, Builder, Dim, FuncOp, Op, Type, Value};
use std::collections::{HashMap, HashSet};

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
    "mlr.layernorm",
    "mlr.transpose",
    "mlr.softmax",
    "mlr.softmax_tri",
    "mlr.add",
    "mlr.tmin",
    "mlr.tmax",
    "mlr.tclamp",
    "mlr.tpow",
    "mlr.texp",
    "mlr.tln",
    "mlr.mean",
    "mlr.norm1",
    "mlr.norm2",
    "mlr.var",
    "mlr.std",
    "mlr.dot",
    "mlr.reshape",
    "mlr.concat",
    "mlr.rows",
    "mlr.reverse",
    "mlr.iota",
    "mlr.linspace",
    "mlr.onehot",
    "mlr.gather",
    "mlr.gather_grad",
    "mlr.dropout",
    "mlr.map",
    "mlr.zeros",
    "tensor.extract",
    "tensor.insert",
    "arith.addf",
    "arith.subf",
    "arith.mulf",
    "arith.divf",
    "arith.truncf",
    "arith.extf",
    "arith.sitofp",
    "arith.index_cast",
    "arith.trunci",
    "arith.extui",
    // The comparisons produce i1: they are part of the forward pass (the
    // condition of an `if`) and never receive an adjoint.
    "arith.cmpf",
    "arith.cmpi",
    "arith.andi",
    "arith.ori",
    "arith.xori",
    "func.return",
];

/// The state of the backward walk: where the adjoint operations go, and the
/// adjoint of every value that has one.
struct Bwd {
    b: Builder,
    /// The adjoint operations, in emission order. Swapped for the
    /// operations of a region while one is being reversed.
    ops: Vec<Op>,
    /// The adjoint of every value that has one, keyed by the id of the value.
    adj: HashMap<usize, Value>,
    /// The runtime functions that the adjoints call.
    used: Vec<&'static str>,
    /// Set while a region (an `scf.if` branch or a backward loop body) is
    /// being reversed: contributions to the values outside it are
    /// accumulated inside and reported through its yield.
    local: Option<Local>,
}

/// The reversal of one region.
struct Local {
    /// The values defined inside the region: their adjoints stay inside.
    defined: HashSet<usize>,
    /// The contributions to the values from outside the region, accumulated
    /// with adds inside it; reported by the yield, in `outer_order`.
    outer: HashMap<usize, Value>,
    outer_order: Vec<usize>,
}

/// Builds `f.value_and_grad` from the IR of `f`: the loss and one gradient
/// slot per tensor parameter. Returns the function and the runtime functions
/// it calls (for the declarations of the module).
pub fn value_and_grad_of(f: &FuncOp, funcs: &[FuncOp]) -> (FuncOp, Vec<&'static str>) {
    let mut bwd = Bwd {
        b: Builder::new(),
        ops: Vec::new(),
        adj: HashMap::new(),
        used: Vec::new(),
        local: None,
    };

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

    // The calls of pure functions are inlined, so that the backward walk
    // sees their operations.
    let forward = expand_calls(body, funcs, 0);
    bwd.ops = forward.clone();

    let mut one = Op::new("arith.constant");
    one.attrs.push(("value".into(), Attr::Float(1.0)));
    let seed = bwd
        .b
        .fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
    one.results.push(seed.clone());
    bwd.ops.push(one);
    bwd.adj.insert(loss.id, seed);

    // The backward pass, from the loss back to the parameters.
    for (index, op) in forward.iter().enumerate().rev() {
        match op.name.as_str() {
            "scf.if" => bwd.diff_if(op, index),
            "scf.for" => bwd.diff_for(op),
            _ => {
                let y = match op.results.first() {
                    Some(r) => r.clone(),
                    None => continue,
                };
                let dy = match bwd.adj.get(&y.id) {
                    Some(v) => v.clone(),
                    // Nothing depends on this value.
                    None => continue,
                };
                bwd.emit_adjoint(op, &y, &dy);
            }
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
        let grad = match bwd.adj.remove(&p.id) {
            Some(v) => v,
            None => bwd.zero(&p.ty),
        };
        results.push(grad.ty.clone());
        returns.push(grad);
    }

    let mut term = Op::new("func.return");
    term.operands = returns;
    bwd.ops.push(term);

    let func = FuncOp {
        name: format!("{}.value_and_grad", f.name),
        params: f.params.clone(),
        results,
        body: crate::ir::Region {
            blocks: vec![crate::ir::Block {
                args: vec![],
                ops: bwd.ops,
            }],
        },
        attrs: vec![("effects".to_string(), Attr::Str("pure".to_string()))],
        loc: f.loc.clone(),
    };
    (func, bwd.used)
}

impl Bwd {
    /// A zero of the given type, in the current operation list: `mlr.zeros`
    /// for a tensor, a constant for an f32 scalar.
    fn zero(&mut self, ty: &Type) -> Value {
        zero_in(&mut self.ops, &mut self.b, ty)
    }

    /// Adds a contribution to the adjoint of a value. Inside a region, a
    /// contribution to a value from outside it is accumulated inside and
    /// reported through its yield. Gradients only flow through f32 scalars
    /// and f32 tensors; every other target (indices, integers) is dropped.
    fn accumulate(&mut self, target: &Value, contribution: Value) {
        if !is_differentiable_type(&target.ty) {
            return;
        }
        if let Some(local) = self.local.as_ref() {
            if !local.defined.contains(&target.id) {
                let current = match local.outer.get(&target.id) {
                    Some(old) => {
                        let old = old.clone();
                        let ty = old.ty.clone();
                        let mut sum = Op::new("mlr.add");
                        sum.operands = vec![old, contribution];
                        let res = self.b.fresh_value(ty);
                        sum.results.push(res.clone());
                        self.ops.push(sum);
                        res
                    }
                    None => contribution,
                };
                let local = self.local.as_mut().unwrap();
                if !local.outer_order.contains(&target.id) {
                    local.outer_order.push(target.id);
                }
                local.outer.insert(target.id, current);
                return;
            }
        }
        match self.adj.get(&target.id) {
            Some(old) => {
                let old = old.clone();
                let ty = old.ty.clone();
                let mut sum = Op::new("mlr.add");
                sum.operands = vec![old, contribution];
                let res = self.b.fresh_value(ty);
                sum.results.push(res.clone());
                self.ops.push(sum);
                self.adj.insert(target.id, res);
            }
            None => {
                self.adj.insert(target.id, contribution);
            }
        }
    }

    /// Seeds the adjoint of a value while a region is being reversed: a
    /// value of the region receives it directly, a value from outside it
    /// receives it as a contribution.
    fn seed(&mut self, value: &Value, dy: Value) {
        let is_local = self
            .local
            .as_ref()
            .map(|l| l.defined.contains(&value.id))
            .unwrap_or(false);
        if is_local {
            match self.adj.get(&value.id) {
                Some(old) => {
                    let old = old.clone();
                    let ty = old.ty.clone();
                    let mut sum = Op::new("mlr.add");
                    sum.operands = vec![old, dy];
                    let res = self.b.fresh_value(ty);
                    sum.results.push(res.clone());
                    self.ops.push(sum);
                    self.adj.insert(value.id, res);
                }
                None => {
                    self.adj.insert(value.id, dy);
                }
            }
        } else {
            self.accumulate(value, dy);
        }
    }

    /// The adjoint of one straight-line operation, appended to `self.ops`.
    fn emit_adjoint(&mut self, op: &Op, y: &Value, dy: &Value) {
        let dy = dy.clone();
        match op.name.as_str() {
            "mlr.matmul" => {
                // dx = dy @ w^T, dw = x^T @ dy.
                self.used.push("mlr_matmul_grad");
                let x = op.operands[0].clone();
                let w = op.operands[1].clone();
                let mut g = Op::new("mlr.matmul_grad");
                g.operands = vec![x.clone(), w.clone(), dy];
                let dx = self.b.fresh_value(x.ty.clone());
                let dw = self.b.fresh_value(w.ty.clone());
                g.results = vec![dx.clone(), dw.clone()];
                g.loc = op.loc.clone();
                self.ops.push(g);
                self.accumulate(&x, dx);
                self.accumulate(&w, dw);
            }
            "mlr.add_bias" => {
                // dm = dy, db = the sum of the rows of dy.
                self.used.push("mlr_add_bias_grad");
                let m = op.operands[0].clone();
                let bias = op.operands[1].clone();
                let mut g = Op::new("mlr.add_bias_grad");
                g.operands = vec![dy];
                let dm = self.b.fresh_value(m.ty.clone());
                let db = self.b.fresh_value(bias.ty.clone());
                g.results = vec![dm.clone(), db.clone()];
                g.loc = op.loc.clone();
                self.ops.push(g);
                self.accumulate(&m, dm);
                self.accumulate(&bias, db);
            }
            "mlr.relu" => {
                // dx = dy where x > 0.
                self.used.push("mlr_relu_grad");
                let x = op.operands[0].clone();
                let mut g = Op::new("mlr.relu_grad");
                g.operands = vec![x.clone(), dy];
                let dx = self.b.fresh_value(x.ty.clone());
                g.results = vec![dx.clone()];
                g.loc = op.loc.clone();
                self.ops.push(g);
                self.accumulate(&x, dx);
            }
            "mlr.sum" => {
                // dx is the incoming gradient in every element.
                self.used.push("mlr_sum_grad");
                let x = op.operands[0].clone();
                let mut g = Op::new("mlr.sum_grad");
                g.operands = vec![x.clone(), dy];
                let dx = self.b.fresh_value(x.ty.clone());
                g.results = vec![dx.clone()];
                self.ops.push(g);
                self.accumulate(&x, dx);
            }
            "mlr.flatten" => {
                // The inverse of the flatten is the reshape of the same
                // layout: a copy.
                self.used.push("mlr_copy");
                let x = op.operands[0].clone();
                let mut g = Op::new("mlr.flatten_grad");
                g.operands = vec![dy];
                let dx = self.b.fresh_value(x.ty.clone());
                g.results = vec![dx.clone()];
                self.ops.push(g);
                self.accumulate(&x, dx);
            }
            "mlr.conv2d" => {
                self.used.push("mlr_conv2d_grad");
                let x = op.operands[0].clone();
                let w = op.operands[1].clone();
                let bias = op.operands[2].clone();
                let mut g = Op::new("mlr.conv2d_grad");
                g.operands = vec![x.clone(), w.clone(), dy];
                let dx = self.b.fresh_value(x.ty.clone());
                let dw = self.b.fresh_value(w.ty.clone());
                let db = self.b.fresh_value(bias.ty.clone());
                g.results = vec![dx.clone(), dw.clone(), db.clone()];
                g.attrs = op.attrs.clone();
                self.ops.push(g);
                self.accumulate(&x, dx);
                self.accumulate(&w, dw);
                self.accumulate(&bias, db);
            }
            "mlr.maxpool2d" => {
                self.used.push("mlr_maxpool2d_grad");
                let x = op.operands[0].clone();
                let mut g = Op::new("mlr.maxpool2d_grad");
                g.operands = vec![x.clone(), dy];
                let dx = self.b.fresh_value(x.ty.clone());
                g.results = vec![dx.clone()];
                g.attrs = op.attrs.clone();
                self.ops.push(g);
                self.accumulate(&x, dx);
            }
            "mlr.embed" => {
                self.used.push("mlr_embed_grad");
                let table = op.operands[0].clone();
                let mut g = Op::new("mlr.embed_grad");
                g.operands = vec![table.clone(), op.operands[1].clone(), dy];
                let dtable = self.b.fresh_value(table.ty.clone());
                g.results = vec![dtable.clone()];
                self.ops.push(g);
                self.accumulate(&table, dtable);
            }
            "mlr.embed_seq" => {
                self.used.push("mlr_embed_seq_grad");
                let table = op.operands[0].clone();
                let mut g = Op::new("mlr.embed_seq_grad");
                g.operands = vec![table.clone(), op.operands[1].clone(), dy];
                let dtable = self.b.fresh_value(table.ty.clone());
                g.results = vec![dtable.clone()];
                self.ops.push(g);
                self.accumulate(&table, dtable);
            }
            "mlr.tmin" | "mlr.tmax" if op.operands.len() == 2 => {
                // The winner of the comparison receives the gradient.
                let grad = if op.name == "mlr.tmin" {
                    "mlr.tmin_grad"
                } else {
                    "mlr.tmax_grad"
                };
                let lhs = op.operands[0].clone();
                let rhs = op.operands[1].clone();
                let mut g_op = Op::new(grad);
                g_op.operands = vec![lhs.clone(), rhs.clone(), dy];
                let da = self.b.fresh_value(lhs.ty.clone());
                let db = self.b.fresh_value(rhs.ty.clone());
                g_op.results = vec![da.clone(), db.clone()];
                self.ops.push(g_op);
                self.accumulate(&lhs, da);
                self.accumulate(&rhs, db);
            }
            "mlr.tmax" | "mlr.tmin" => {
                // The reduction to one value: the largest (smallest) element
                // receives the gradient.
                let x = op.operands[0].clone();
                let grad = if op.name == "mlr.tmax" {
                    "mlr.tmax_grad_value"
                } else {
                    "mlr.tmin_grad_value"
                };
                let mut g_op = Op::new(grad);
                g_op.operands = vec![x.clone(), dy];
                let dx = self.b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                self.ops.push(g_op);
                self.accumulate(&x, dx);
            }
            "mlr.tclamp" => {
                let x = op.operands[0].clone();
                let mut g_op = Op::new("mlr.tclamp_grad");
                g_op.operands =
                    vec![x.clone(), op.operands[1].clone(), op.operands[2].clone(), dy];
                let dx = self.b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                self.ops.push(g_op);
                self.accumulate(&x, dx);
            }
            "mlr.tpow" => {
                let x = op.operands[0].clone();
                let mut g_op = Op::new("mlr.tpow_grad");
                g_op.operands = vec![x.clone(), op.operands[1].clone(), dy];
                let dx = self.b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                self.ops.push(g_op);
                self.accumulate(&x, dx);
            }
            "mlr.texp" | "mlr.tln" => {
                let x = op.operands[0].clone();
                let grad = if op.name == "mlr.texp" {
                    "mlr.texp_grad"
                } else {
                    "mlr.tln_grad"
                };
                let mut g_op = Op::new(grad);
                g_op.operands = vec![x.clone(), dy];
                let dx = self.b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                self.ops.push(g_op);
                self.accumulate(&x, dx);
            }
            "mlr.mean" => {
                // The gradient of the mean is the gradient of the sum with
                // the seed divided by the number of the elements.
                self.used.push("mlr_sum_grad");
                let x = op.operands[0].clone();
                let n = element_count(&x.ty);
                let mut cst = Op::new("arith.constant");
                cst.attrs.push(("value".into(), Attr::Float(n as f64)));
                let cval = self
                    .b
                    .fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
                cst.results.push(cval.clone());
                self.ops.push(cst);
                let mut div = Op::new("arith.divf");
                div.operands = vec![dy, cval.clone()];
                let seed = self
                    .b
                    .fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
                div.results.push(seed.clone());
                self.ops.push(div);
                let mut g_op = Op::new("mlr.sum_grad");
                g_op.operands = vec![x.clone(), seed];
                let dx = self.b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                self.ops.push(g_op);
                self.accumulate(&x, dx);
            }
            "mlr.norm1" => {
                let x = op.operands[0].clone();
                let mut g_op = Op::new("mlr.norm1_grad");
                g_op.operands = vec![x.clone(), dy];
                let dx = self.b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                self.ops.push(g_op);
                self.accumulate(&x, dx);
            }
            "mlr.norm2" => {
                let x = op.operands[0].clone();
                let mut g_op = Op::new("mlr.norm2_grad");
                g_op.operands = vec![x.clone(), y.clone(), dy];
                let dx = self.b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                self.ops.push(g_op);
                self.accumulate(&x, dx);
            }
            "mlr.var" | "mlr.std" => {
                let x = op.operands[0].clone();
                let grad = if op.name == "mlr.var" {
                    "mlr.var_grad"
                } else {
                    "mlr.std_grad"
                };
                let mut g_op = Op::new(grad);
                g_op.operands = vec![x.clone(), dy];
                let dx = self.b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                self.ops.push(g_op);
                self.accumulate(&x, dx);
            }
            "mlr.dot" => {
                let lhs = op.operands[0].clone();
                let rhs = op.operands[1].clone();
                let mut g_op = Op::new("mlr.dot_grad");
                g_op.operands = vec![lhs.clone(), rhs.clone(), dy];
                let da = self.b.fresh_value(lhs.ty.clone());
                let db = self.b.fresh_value(rhs.ty.clone());
                g_op.results = vec![da.clone(), db.clone()];
                self.ops.push(g_op);
                self.accumulate(&lhs, da);
                self.accumulate(&rhs, db);
            }
            "mlr.reshape" => {
                // The inverse of the reshape is a copy of the original shape.
                self.used.push("mlr_copy");
                let x = op.operands[0].clone();
                let mut g_op = Op::new("mlr.reshape_grad");
                g_op.operands = vec![dy];
                let dx = self.b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                self.ops.push(g_op);
                self.accumulate(&x, dx);
            }
            "mlr.concat" => {
                let lhs = op.operands[0].clone();
                let rhs = op.operands[1].clone();
                let mut g_op = Op::new("mlr.concat_grad");
                g_op.operands = vec![dy];
                let da = self.b.fresh_value(lhs.ty.clone());
                let db = self.b.fresh_value(rhs.ty.clone());
                g_op.results = vec![da.clone(), db.clone()];
                self.ops.push(g_op);
                self.accumulate(&lhs, da);
                self.accumulate(&rhs, db);
            }
            "mlr.rows" => {
                let x = op.operands[0].clone();
                let mut g_op = Op::new("mlr.rows_grad");
                g_op.operands = vec![dy, op.operands[1].clone(), op.operands[2].clone()];
                let dx = self.b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                self.ops.push(g_op);
                self.accumulate(&x, dx);
            }
            "mlr.dropout" => {
                // The mask is recomputed from the same seed and rate.
                self.used.push("mlr_dropout_grad");
                let x = op.operands[0].clone();
                let mut g_op = Op::new("mlr.dropout_grad");
                g_op.operands = vec![
                    x.clone(),
                    dy,
                    op.operands[1].clone(),
                    op.operands[2].clone(),
                ];
                let dx = self.b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                self.ops.push(g_op);
                self.accumulate(&x, dx);
            }
            "mlr.reverse" => {
                // The reverse is its own inverse.
                let x = op.operands[0].clone();
                let mut g_op = Op::new("mlr.reverse");
                g_op.operands = vec![dy];
                let dx = self.b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                self.ops.push(g_op);
                self.accumulate(&x, dx);
            }
            "mlr.layernorm" => {
                // dx, dg and db; the statistics are recomputed from x.
                self.used.push("mlr_layernorm_grad");
                let x = op.operands[0].clone();
                let g = op.operands[1].clone();
                let bias = op.operands[2].clone();
                let mut g_op = Op::new("mlr.layernorm_grad");
                g_op.operands = vec![x.clone(), g.clone(), dy];
                let dx = self.b.fresh_value(x.ty.clone());
                let dg = self.b.fresh_value(g.ty.clone());
                let db = self.b.fresh_value(bias.ty.clone());
                g_op.results = vec![dx.clone(), dg.clone(), db.clone()];
                self.ops.push(g_op);
                self.accumulate(&x, dx);
                self.accumulate(&g, dg);
                self.accumulate(&bias, db);
            }
            "mlr.transpose" => {
                // The adjoint of the transpose is the transpose.
                let x = op.operands[0].clone();
                let mut t = Op::new("mlr.transpose");
                t.operands = vec![dy];
                let dx = self.b.fresh_value(x.ty.clone());
                t.results = vec![dx.clone()];
                self.ops.push(t);
                self.accumulate(&x, dx);
            }
            "mlr.softmax" | "mlr.softmax_tri" => {
                // dx = y * (dy - sum(dy * y)) with y the output of the
                // forward pass; the mask of the triangular version lives in
                // the kernel.
                self.used.push(if op.name == "mlr.softmax" {
                    "mlr_softmax_grad"
                } else {
                    "mlr_softmax_tri_grad"
                });
                let x = op.operands[0].clone();
                let name = if op.name == "mlr.softmax" {
                    "mlr.softmax_grad"
                } else {
                    "mlr.softmax_tri_grad"
                };
                let mut g_op = Op::new(name);
                g_op.operands = vec![y.clone(), dy];
                let dx = self.b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                self.ops.push(g_op);
                self.accumulate(&x, dx);
            }
            "mlr.add" => {
                // Both operands receive the incoming gradient.
                self.used.push("mlr_add_elem");
                let lhs = op.operands[0].clone();
                let rhs = op.operands[1].clone();
                self.accumulate(&lhs, dy.clone());
                self.accumulate(&rhs, dy);
            }
            "mlr.map" => {
                // An element wise function of one tensor.
                let x = op.operands[0].clone();
                let dx = match elementwise_adjoint(op) {
                    // A known non linear function: `map(t) |v| gelu(v)`.
                    Some(grad) => {
                        let mut g_op = Op::new(grad);
                        g_op.operands = vec![x.clone(), dy];
                        let dx = self.b.fresh_value(x.ty.clone());
                        g_op.results = vec![dx.clone()];
                        self.ops.push(g_op);
                        self.used.push(match grad {
                            "mlr.gelu_grad" => "mlr_gelu_grad",
                            other => panic!("unknown gradient {other}"),
                        });
                        dx
                    }
                    // A linear map: `map(t) |v| v * 0.25`. The gradient of a
                    // linear function is the same function of the incoming
                    // gradient.
                    None => {
                        let scale = linear_scale(op).expect("a known linear map");
                        map_scale(&mut self.ops, &mut self.b, dy, scale)
                    }
                };
                self.accumulate(&x, dx);
            }
            "mlr.cross_entropy" => {
                // dlogits = (softmax(logits) - one hot(labels)) / rows.
                self.used.push("mlr_cross_entropy_grad");
                let logits = op.operands[0].clone();
                let labels = op.operands[1].clone();
                let mut g = Op::new("mlr.cross_entropy_grad");
                g.operands = vec![logits.clone(), labels.clone(), dy];
                let dlogits = self.b.fresh_value(logits.ty.clone());
                g.results = vec![dlogits.clone()];
                g.loc = op.loc.clone();
                self.ops.push(g);
                self.accumulate(&logits, dlogits);
            }
            "tensor.extract" => {
                // dt is the incoming gradient scattered into a zero tensor
                // at the index.
                let t = op.operands[0].clone();
                let zeros = self.zero(&t.ty);
                let mut ins = Op::new("tensor.insert");
                ins.operands.push(dy);
                ins.operands.push(zeros);
                ins.operands.extend(op.operands[1..].iter().cloned());
                let dt = self.b.fresh_value(t.ty.clone());
                ins.results.push(dt.clone());
                self.ops.push(ins);
                self.accumulate(&t, dt);
            }
            "tensor.insert" => {
                // dt is the incoming gradient with the written position
                // removed; the written value receives it.
                let t = op.operands[1].clone();
                let v = op.operands[0].clone();
                let mut zero = Op::new("arith.constant");
                zero.attrs.push(("value".into(), Attr::Float(0.0)));
                let zv = self
                    .b
                    .fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
                zero.results.push(zv.clone());
                self.ops.push(zero);
                let mut ins = Op::new("tensor.insert");
                ins.operands.push(zv);
                ins.operands.push(dy.clone());
                ins.operands.extend(op.operands[2..].iter().cloned());
                let dt = self.b.fresh_value(t.ty.clone());
                ins.results.push(dt.clone());
                self.ops.push(ins);
                let mut ext = Op::new("tensor.extract");
                ext.operands.push(dy);
                ext.operands.extend(op.operands[2..].iter().cloned());
                let dv = self.b.fresh_value(v.ty.clone());
                ext.results.push(dv.clone());
                self.ops.push(ext);
                self.accumulate(&t, dt);
                self.accumulate(&v, dv);
            }
            "arith.constant" | "mlr.zeros" => {}
            // The scalar arithmetic on the way to the loss: the adjoints
            // are the usual ones, expressed with scalar operations.
            "arith.addf" => {
                let lhs = op.operands[0].clone();
                let rhs = op.operands[1].clone();
                self.accumulate(&lhs, dy.clone());
                self.accumulate(&rhs, dy);
            }
            "arith.subf" => {
                let lhs = op.operands[0].clone();
                let rhs = op.operands[1].clone();
                self.accumulate(&lhs, dy.clone());
                let mut neg = Op::new("arith.negf");
                neg.operands = vec![dy];
                let ndy = self
                    .b
                    .fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
                neg.results.push(ndy.clone());
                self.ops.push(neg);
                self.accumulate(&rhs, ndy);
            }
            "arith.mulf" => {
                let lhs = op.operands[0].clone();
                let rhs = op.operands[1].clone();
                let mut pa = Op::new("arith.mulf");
                pa.operands = vec![dy.clone(), rhs.clone()];
                let pva = self
                    .b
                    .fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
                pa.results.push(pva.clone());
                self.ops.push(pa);
                let mut pb = Op::new("arith.mulf");
                pb.operands = vec![dy.clone(), lhs.clone()];
                let pvb = self
                    .b
                    .fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
                pb.results.push(pvb.clone());
                self.ops.push(pb);
                self.accumulate(&lhs, pva);
                self.accumulate(&rhs, pvb);
            }
            "arith.divf" => {
                let lhs = op.operands[0].clone();
                let rhs = op.operands[1].clone();
                let mut da = Op::new("arith.divf");
                da.operands = vec![dy.clone(), rhs.clone()];
                let dva = self
                    .b
                    .fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
                da.results.push(dva.clone());
                self.ops.push(da);
                self.accumulate(&lhs, dva);
                let mut aa = Op::new("arith.mulf");
                aa.operands = vec![dy.clone(), lhs.clone()];
                let t = self
                    .b
                    .fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
                aa.results.push(t.clone());
                self.ops.push(aa);
                let mut bb = Op::new("arith.mulf");
                bb.operands = vec![rhs.clone(), rhs.clone()];
                let b2v = self
                    .b
                    .fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
                bb.results.push(b2v.clone());
                self.ops.push(bb);
                let mut db = Op::new("arith.divf");
                db.operands = vec![t, b2v];
                let ndy = self
                    .b
                    .fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
                db.results.push(ndy.clone());
                self.ops.push(db);
                let mut neg = Op::new("arith.negf");
                neg.operands = vec![ndy];
                let nndy = self
                    .b
                    .fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
                neg.results.push(nndy.clone());
                self.ops.push(neg);
                self.accumulate(&rhs, nndy);
            }
            // The conversions of the scalars pass the gradient through.
            "arith.truncf" | "arith.extf" | "arith.sitofp" | "arith.index_cast"
            | "arith.trunci" | "arith.extui" => {
                if let Some(a) = op.operands.first() {
                    let a = a.clone();
                    self.accumulate(&a, dy);
                }
            }
            other => panic!(
                "value_and_grad: the operation `{}` has no gradient yet",
                other
            ),
        }
    }

    /// The adjoint of an `scf.if`: each branch is reversed with the adjoint
    /// of the result as the seed. A branch contributes to values from
    /// outside it conditionally, so a *second* `scf.if` with the same
    /// condition is appended after the backward chain so far, and its
    /// branches report the contributions through one extra result per
    /// outside value. (The regions of the original `if` cannot reference the
    /// adjoints, which are defined after it.)
    fn diff_if(&mut self, op: &Op, _index: usize) {
        let dys: Vec<Option<Value>> = op
            .results
            .iter()
            .map(|r| self.adj.get(&r.id).cloned())
            .collect();
        if dys.iter().all(|d| d.is_none()) {
            // Nothing depends on any result of the `if`.
            return;
        }
        // Reverse both branches, each into its own list of operations. The
        // operations are copied with fresh ids: the values of the original
        // region are not visible in the new one.
        let mut reversed: Vec<(Vec<Op>, HashMap<usize, Value>, Vec<usize>)> = Vec::new();
        for region in &op.regions {
            let block = &region.blocks[0];
            let mut map: HashMap<usize, Value> = HashMap::new();
            let cloned: Vec<Op> = block.ops[..block.ops.len() - 1]
                .iter()
                .map(|o| clone_op(o, &mut map, &mut self.b))
                .collect();
            let saved_ops = std::mem::take(&mut self.ops);
            let saved_local = self.local.take();
            self.ops = Vec::new();
            self.ops.extend(cloned.iter().cloned());
            self.local = Some(Local {
                defined: cloned
                    .iter()
                    .flat_map(|o| o.results.iter().map(|r| r.id))
                    .collect(),
                outer: HashMap::new(),
                outer_order: Vec::new(),
            });
            // The seeds: the operands of the yield, which stand for the
            // results of the `scf.if`.
            let seeds: Vec<(Value, Value)> = block
                .ops
                .last()
                .expect("a yield")
                .operands
                .iter()
                .cloned()
                .zip(dys.iter().cloned())
                .filter_map(|(v, d)| d.map(|d| (v, d)))
                .collect();
            for (v, d) in seeds {
                let target = map.get(&v.id).cloned().unwrap_or_else(|| v.clone());
                self.seed(&target, d);
            }
            for o in cloned.iter().rev() {
                let y = match o.results.first() {
                    Some(r) => r.clone(),
                    None => continue,
                };
                let dy = match self.adj.get(&y.id) {
                    Some(v) => v.clone(),
                    None => continue,
                };
                self.emit_adjoint(o, &y, &dy);
            }
            let local = self.local.take().unwrap();
            self.local = saved_local;
            let ops = std::mem::replace(&mut self.ops, saved_ops);
            reversed.push((ops, local.outer, local.outer_order));
        }
        // The outside values that at least one branch contributed to, in a
        // fixed order over both branches.
        let mut order: Vec<(usize, Type)> = Vec::new();
        for (_, outer, ids) in &reversed {
            for id in ids {
                if !order.iter().any(|(i, _)| i == id) {
                    order.push((*id, outer[id].ty.clone()));
                }
            }
        }
        if order.is_empty() {
            // The gradient stayed inside the branches; nothing to append.
            return;
        }
        // The new `scf.if`: one result per outside value, and one yield
        // operand per branch, which is the adjoint so far plus the
        // contribution of the branch.
        let mut new_op = Op::new("scf.if");
        new_op.operands.push(op.operands[0].clone());
        new_op.loc = op.loc.clone();
        let extras: Vec<Value> = order
            .iter()
            .map(|(_, ty)| self.b.fresh_value(ty.clone()))
            .collect();
        new_op.results = extras.clone();
        for (_r, (mut ops, outer, ids)) in reversed.into_iter().enumerate() {
            let mut yields: Vec<Value> = Vec::new();
            for (_i, (id, ty)) in order.iter().enumerate() {
                let base = match self.adj.get(id) {
                    Some(v) => v.clone(),
                    None => zero_in(&mut ops, &mut self.b, ty),
                };
                let value = match ids.iter().position(|j| j == id) {
                    Some(_) => {
                        let contribution = outer[id].clone();
                        let mut sum = Op::new("mlr.add");
                        sum.operands = vec![base, contribution];
                        let res = self.b.fresh_value(ty.clone());
                        sum.results.push(res.clone());
                        ops.push(sum);
                        res
                    }
                    None => base,
                };
                yields.push(value);
            }
            let mut term = Op::new("scf.yield");
            term.operands = yields;
            ops.push(term);
            new_op.regions.push(crate::ir::Region {
                blocks: vec![crate::ir::Block { args: vec![], ops }],
            });
        }
        self.ops.push(new_op.clone());
        // The results of the new `if` are the new adjoints: the branches
        // already added the adjoint so far.
        for (i, (id, _)) in order.iter().enumerate() {
            self.adj.insert(*id, extras[i].clone());
        }
    }

    /// The adjoint of an `scf.for` whose carried values are accumulated
    /// (`yield acc + expr`): a backward loop over the same range that
    /// replays the body, reverses it, and accumulates the contributions to
    /// the values outside the loop into extra iter args.
    fn diff_for(&mut self, op: &Op) {
        let block = &op.regions[0].blocks[0];
        let iv_fwd = block.args[0].clone();
        let iter_args: Vec<Value> = block.args[1..].to_vec();
        let body = &block.ops;
        let yield_op = body.last().expect("a yield");
        let inits: Vec<Value> = op.operands[3..].to_vec();
        // The carried values that a gradient can flow through.
        let tracked: Vec<usize> = (0..iter_args.len())
            .filter(|k| is_differentiable_type(&iter_args[*k].ty))
            .collect();
        let dys: Vec<Option<Value>> = op
            .results
            .iter()
            .map(|r| self.adj.get(&r.id).cloned())
            .collect();
        if !tracked.iter().any(|k| dys[*k].is_some()) {
            // Nothing depends on any carried value.
            return;
        }
        let defs: HashMap<usize, &Op> = body
            .iter()
            .flat_map(|o| o.results.iter().map(move |r| (r.id, o)))
            .collect();
        // The parts of the body that do not read the carried values: they
        // are replayed inside the backward loop, which has no forward state.
        let replay: Vec<Op> = body
            .iter()
            .filter(|o| o.name != "scf.yield" && !touches_arg(o, &iter_args, &defs))
            .cloned()
            .collect();
        // The affine form of every tracked yield operand.
        let mut visited: HashSet<usize> = HashSet::new();
        let mut affines: Vec<Affine> = Vec::new();
        for k in &tracked {
            let aff = parse_affine(
                &yield_op.operands[*k],
                &iter_args[*k],
                &iter_args,
                &defs,
                &mut visited,
            )
            .expect("validated by check_differentiable");
            affines.push(aff);
        }
        // The backward loop: the same range; one iter arg per tracked
        // carried value (its adjoint), plus one per outside value that the
        // body contributes to.
        let bwd_iv = self.b.fresh_value(Type::Index);
        let dargs: Vec<Value> = tracked
            .iter()
            .map(|k| self.b.fresh_value(iter_args[*k].ty.clone()))
            .collect();
        // Replay the body with a fresh set of values; the induction variable
        // of the backward loop stands for the forward one.
        let mut map: HashMap<usize, Value> = HashMap::new();
        map.insert(iv_fwd.id, bwd_iv.clone());
        let replayed: Vec<Op> = replay
            .iter()
            .map(|o| clone_op(o, &mut map, &mut self.b))
            .collect();
        let saved_ops = std::mem::take(&mut self.ops);
        let saved_local = self.local.take();
        self.ops = Vec::new();
        self.ops.extend(replayed.iter().cloned());
        self.local = Some(Local {
            defined: replayed
                .iter()
                .flat_map(|o| o.results.iter().map(|r| r.id))
                .collect(),
            outer: HashMap::new(),
            outer_order: Vec::new(),
        });
        // The seeds: the affine terms of every tracked yield operand, each
        // scaled by the incoming adjoint of the carried value.
        for (j, (k, aff)) in tracked.iter().zip(affines.iter()).enumerate() {
            let dy = match &dys[*k] {
                Some(v) => v.clone(),
                // The result is not used: its adjoint is zero and nothing
                // flows through this chain.
                None => continue,
            };
            for (scale, term) in &aff.terms {
                let value = scaled(&mut self.ops, &mut self.b, dy.clone(), *scale);
                let target = map
                    .get(&term.id)
                    .cloned()
                    .unwrap_or_else(|| term.clone());
                self.seed(&target, value);
            }
            let _ = j;
        }
        // The backward walk over the replayed operations.
        for o in replayed.iter().rev() {
            let y = match o.results.first() {
                Some(r) => r.clone(),
                None => continue,
            };
            let dy = match self.adj.get(&y.id) {
                Some(v) => v.clone(),
                None => continue,
            };
            self.emit_adjoint(o, &y, &dy);
        }
        let local = self.local.take().unwrap();
        let mut region_ops = std::mem::replace(&mut self.ops, saved_ops);
        self.local = saved_local;
        // The outside accumulators of the backward loop.
        let outer_args: Vec<Value> = local
            .outer_order
            .iter()
            .map(|id| {
                let ty = local.outer[id].ty.clone();
                self.b.fresh_value(ty)
            })
            .collect();
        // The yield: the scaled adjoints of the carried values, then the
        // accumulated contributions to the outside values.
        let mut yields: Vec<Value> = Vec::new();
        for (j, (k, aff)) in tracked.iter().zip(affines.iter()).enumerate() {
            let acc = dargs[j].clone();
            let value = if aff.coeff == 1.0 {
                acc
            } else if aff.coeff == 0.0 {
                zero_in(&mut region_ops, &mut self.b, &iter_args[*k].ty)
            } else {
                scaled(&mut region_ops, &mut self.b, acc, aff.coeff)
            };
            yields.push(value);
        }
        for (j, id) in local.outer_order.iter().enumerate() {
            let acc = outer_args[j].clone();
            let value = match local.outer.get(id) {
                Some(v) => {
                    let mut sum = Op::new("mlr.add");
                    sum.operands = vec![acc, v.clone()];
                    let res = self.b.fresh_value(sum.operands[0].ty.clone());
                    sum.results.push(res.clone());
                    region_ops.push(sum);
                    res
                }
                None => acc,
            };
            yields.push(value);
        }
        let mut term = Op::new("scf.yield");
        term.operands = yields;
        region_ops.push(term);
        // The initial values of the backward loop: the adjoints of the
        // results, and zeros for the outside accumulators.
        let mut operands = vec![
            op.operands[0].clone(),
            op.operands[1].clone(),
            op.operands[2].clone(),
        ];
        for (k, dy) in tracked.iter().zip(dys.iter()) {
            let init = match dy {
                Some(v) => v.clone(),
                None => self.zero(&iter_args[*k].ty),
            };
            operands.push(init);
        }
        for (j, _) in local.outer_order.iter().enumerate() {
            let ty = outer_args[j].ty.clone();
            let zero = self.zero(&ty);
            operands.push(zero);
        }
        let mut for_op = Op::new("scf.for");
        for_op.operands = operands;
        for_op
            .regions
            .push(crate::ir::Region {
                blocks: vec![crate::ir::Block {
                    args: {
                        let mut args = vec![bwd_iv];
                        args.extend(dargs.iter().cloned());
                        args.extend(outer_args.iter().cloned());
                        args
                    },
                    ops: region_ops,
                }],
            });
        for_op.results = {
            let mut results: Vec<Value> = tracked
                .iter()
                .map(|k| self.b.fresh_value(iter_args[*k].ty.clone()))
                .collect();
            for a in &outer_args {
                results.push(self.b.fresh_value(a.ty.clone()));
            }
            results
        };
        for_op.loc = op.loc.clone();
        self.ops.push(for_op.clone());
        // The gradient of the initial value of a carried value is the
        // accumulated adjoint; the same for the outside values.
        for (j, k) in tracked.iter().enumerate() {
            if dys[*k].is_some() {
                let res = for_op.results[j].clone();
                self.accumulate(&inits[*k], res);
            }
        }
        for (j, id) in local.outer_order.iter().enumerate() {
            let ty = outer_args[j].ty.clone();
            let target = Value {
                id: *id,
                ty,
            };
            let res = for_op.results[tracked.len() + j].clone();
            self.accumulate(&target, res);
        }
    }
}

/// A zero of the given type, appended to `ops`.
fn zero_in(ops: &mut Vec<Op>, b: &mut Builder, ty: &Type) -> Value {
    match ty {
        Type::Tensor { .. } => {
            let mut z = Op::new("mlr.zeros");
            let zero = b.fresh_value(ty.clone());
            z.results.push(zero.clone());
            ops.push(z);
            zero
        }
        Type::Scalar(crate::ir::ScalarKind::F32) => {
            let mut c = Op::new("arith.constant");
            c.attrs.push(("value".into(), Attr::Float(0.0)));
            let zero = b.fresh_value(ty.clone());
            c.results.push(zero.clone());
            ops.push(c);
            zero
        }
        other => panic!("value_and_grad: no zero of the type {:?}", other),
    }
}

/// The value scaled by a constant: itself for 1, a negation or a `map` for
/// -1, a multiplication otherwise.
fn scaled(ops: &mut Vec<Op>, b: &mut Builder, v: Value, scale: f64) -> Value {
    if scale == 1.0 {
        return v;
    }
    if scale == -1.0 {
        if matches!(v.ty, Type::Tensor { .. }) {
            return map_scale(ops, b, v, -1.0);
        }
        let mut neg = Op::new("arith.negf");
        neg.operands = vec![v];
        let res = b.fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
        neg.results.push(res.clone());
        ops.push(neg);
        return res;
    }
    if matches!(v.ty, Type::Tensor { .. }) {
        return map_scale(ops, b, v, scale);
    }
    let mut c = Op::new("arith.constant");
    c.attrs.push(("value".into(), Attr::Float(scale)));
    let cv = b.fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
    c.results.push(cv.clone());
    ops.push(c);
    let mut mul = Op::new("arith.mulf");
    mul.operands = vec![v, cv];
    let res = b.fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
    mul.results.push(res.clone());
    ops.push(mul);
    res
}

/// The number of the elements of a tensor type.
fn element_count(ty: &Type) -> i64 {
    match ty {
        Type::Tensor { shape, .. } | Type::MemRef { shape, .. } => shape.iter().fold(
            1i64,
            |acc, d| match d {
                Dim::Static(v) => acc * v,
                Dim::Dynamic => acc,
            },
        ),
        _ => 1,
    }
}

/// Whether a gradient can flow through the type: f32 scalars and tensors.
fn is_differentiable_type(ty: &Type) -> bool {
    match ty {
        Type::Scalar(crate::ir::ScalarKind::F32) => true,
        Type::Tensor { elem, .. } => matches!(**elem, Type::Scalar(crate::ir::ScalarKind::F32)),
        _ => false,
    }
}

/// Replaces the calls of the pure functions of the program with their
/// bodies: the values of the parameters become the values of the arguments
/// and every value receives a fresh id. The calls of the runtime (their
/// names start with `mlr_`) stay.
fn expand_calls(ops: Vec<Op>, funcs: &[FuncOp], depth: usize) -> Vec<Op> {
    if depth > 20 {
        panic!("value_and_grad: the calls of the functions are too deep");
    }
    let mut out: Vec<Op> = Vec::new();
    // The value that replaces a result of a call.
    let mut redirect: HashMap<usize, Value> = HashMap::new();
    let value_of = |v: &Value, redirect: &HashMap<usize, Value>| -> Value {
        redirect.get(&v.id).cloned().unwrap_or_else(|| v.clone())
    };
    for op in ops {
        if op.name == "func.call" {
            let callee_name = match op.attr("callee") {
                Some(Attr::Str(c)) => c.clone(),
                _ => String::new(),
            };
            let is_runtime = callee_name.starts_with("mlr_");
            if !is_runtime {
                let callee = funcs
                    .iter()
                    .find(|f| f.name == callee_name)
                    .unwrap_or_else(|| panic!("value_and_grad: unknown function `{}`", callee_name));
                let mut body = callee.body.blocks[0].ops.clone();
                let returns = match body.last() {
                    Some(o) if o.name == "func.return" => {
                        let v = o.operands.clone();
                        body.pop();
                        v
                    }
                    _ => panic!("value_and_grad: `{}` has no return", callee_name),
                };
                let expanded = expand_calls(body, funcs, depth + 1);
                // The parameters of the callee are the arguments of the call.
                let mut map: HashMap<usize, Value> = HashMap::new();
                for (p, a) in callee.params.iter().zip(op.operands.iter()) {
                    map.insert(p.id, value_of(a, &redirect));
                }
                let mut b = Builder::new();
                for o in expanded {
                    out.push(clone_op(&o, &mut map, &mut b));
                }
                // The first returned value stands for the result of the call.
                if let (Some(r), Some(v)) = (op.results.first(), returns.first()) {
                    let val = map.get(&v.id).cloned().unwrap_or_else(|| v.clone());
                    redirect.insert(r.id, val);
                }
                continue;
            }
        }
        let mut o2 = op.clone();
        remap_values(&mut o2, &redirect);
        out.push(o2);
    }
    out
}

/// Replaces the operands of an operation (and of the operations of its
/// regions) with the values of the redirect.
fn remap_values(op: &mut Op, redirect: &HashMap<usize, Value>) {
    for o in &mut op.operands {
        if let Some(v) = redirect.get(&o.id) {
            *o = v.clone();
        }
    }
    for region in &mut op.regions {
        for block in &mut region.blocks {
            for o in &mut block.ops {
                remap_values(o, redirect);
            }
        }
    }
}

/// A deep copy of an operation: every value receives a fresh id, and the
/// references to the values of the map are replaced.
fn clone_op(op: &Op, map: &mut HashMap<usize, Value>, b: &mut Builder) -> Op {
    let mut new = Op::new(op.name.as_str());
    new.generic = op.generic;
    new.attrs = op.attrs.clone();
    new.loc = op.loc.clone();
    for o in &op.operands {
        let v = map.get(&o.id).cloned().unwrap_or_else(|| o.clone());
        new.operands.push(v);
    }
    for r in &op.results {
        let nv = b.fresh_value(r.ty.clone());
        map.insert(r.id, nv.clone());
        new.results.push(nv);
    }
    for region in &op.regions {
        let mut nr = crate::ir::Region::default();
        for block in &region.blocks {
            let mut args = Vec::new();
            for a in &block.args {
                let nv = b.fresh_value(a.ty.clone());
                map.insert(a.id, nv.clone());
                args.push(nv);
            }
            let mut nops = Vec::new();
            for o in &block.ops {
                nops.push(clone_op(o, map, b));
            }
            nr.blocks.push(crate::ir::Block { args, ops: nops });
        }
        new.regions.push(nr);
    }
    new
}

/// The factor of the derivative of a linear `map`, if its body is one
/// arithmetic operation of the argument and a constant.
fn linear_scale(op: &Op) -> Option<f64> {
    let block = op.regions.first()?.blocks.first()?;
    if block.args.len() != 1 {
        return None;
    }
    let arith = block
        .ops
        .iter()
        .find(|o| {
            matches!(
                o.name.as_str(),
                "arith.mulf" | "arith.addf" | "arith.subf" | "arith.divf"
            )
        })?;
    let const_op = block.ops.iter().find(|o| o.name == "arith.constant")?;
    let c = match const_op.attr("value") {
        Some(Attr::Float(v)) => *v,
        Some(Attr::Int(v)) => *v as f64,
        _ => return None,
    };
    let is_left = arith
        .operands
        .first()
        .map(|v| v.id == block.args[0].id)
        .unwrap_or(false);
    match (arith.name.as_str(), is_left) {
        ("arith.mulf", _) => Some(c),
        ("arith.addf", _) => Some(1.0),
        ("arith.subf", true) => Some(1.0),
        ("arith.subf", false) => Some(-1.0),
        ("arith.divf", true) => Some(1.0 / c),
        _ => None,
    }
}

/// A `map` that multiplies its input by a constant.
fn map_scale(ops: &mut Vec<Op>, b: &mut Builder, input: Value, scale: f64) -> Value {
    let mut m = Op::new("mlr.map");
    m.generic = true;
    m.operands = vec![input.clone()];
    let arg = b.fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
    let mut cst = Op::new("arith.constant");
    cst.attrs.push(("value".into(), Attr::Float(scale)));
    let cv = b.fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
    cst.results.push(cv.clone());
    let mut mul = Op::new("arith.mulf");
    mul.operands = vec![arg.clone(), cv];
    let r = b.fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
    mul.results.push(r.clone());
    let mut y = Op::new("mlr.yield");
    y.operands = vec![r];
    m.regions.push(crate::ir::Region {
        blocks: vec![crate::ir::Block {
            args: vec![arg],
            ops: vec![cst, mul, y],
        }],
    });
    let res = b.fresh_value(input.ty.clone());
    m.results.push(res.clone());
    ops.push(m);
    res
}

/// The name of the gradient operation of an element wise `map`, when the
/// body is one call of a known function of the argument.
fn elementwise_adjoint(op: &Op) -> Option<&'static str> {
    if op.operands.len() != 1 {
        return None;
    }
    let region = op.regions.first()?;
    let block = region.blocks.first()?;
    if block.args.len() != 1 || block.ops.len() != 2 {
        return None;
    }
    let call = &block.ops[0];
    let yield_op = &block.ops[1];
    if call.name != "func.call" || yield_op.name != "mlr.yield" {
        return None;
    }
    if call.operands.len() != 1 || call.operands[0].id != block.args[0].id {
        return None;
    }
    match call.attr("callee") {
        Some(Attr::Str(c)) if c == "mlr_gelu" => Some("mlr.gelu_grad"),
        _ => None,
    }
}

/// Verifies that a function can be differentiated.
pub fn check_differentiable(f: &FuncOp, funcs: &[FuncOp]) -> Result<(), String> {
    check_ops(&f.body.blocks[0].ops, funcs, true)
}

/// `top` is false inside the regions of an `if` or of a `for`, where no
/// further control flow is supported.
fn check_ops(ops: &[Op], funcs: &[FuncOp], top: bool) -> Result<(), String> {
    for op in ops {
        if op.name == "scf.yield" || op.name == "mlr.yield" || op.name == "func.return" {
            continue;
        }
        if op.name == "func.call" {
            if let Some(Attr::Str(callee)) = op.attr("callee") {
                if !callee.starts_with("mlr_") {
                    let callee_fn = funcs
                        .iter()
                        .find(|g| g.name == *callee)
                        .ok_or_else(|| format!("unknown function `{}`", callee))?;
                    check_differentiable(callee_fn, funcs)?;
                    continue;
                }
            }
        }
        match op.name.as_str() {
            "scf.if" => {
                if !top {
                    return Err("nested control flow cannot be differentiated yet".to_string());
                }
                if op.results.is_empty() {
                    return Err("an `if` without a value cannot be differentiated".to_string());
                }
                for region in &op.regions {
                    check_ops(&region.blocks[0].ops, funcs, false)?;
                }
            }
            "scf.for" => {
                if !top {
                    return Err("nested control flow cannot be differentiated yet".to_string());
                }
                check_for(op)?;
            }
            _ => check_straight_op(op)?,
        }
    }
    Ok(())
}

/// One straight-line operation: it must have a known adjoint, and a `map`
/// must be a known element wise function or a linear one.
fn check_straight_op(op: &Op) -> Result<(), String> {
    if !FORWARD.contains(&op.name.as_str()) {
        return Err(format!("the operation `{}` has no gradient yet", op.name));
    }
    if op.name == "mlr.map" && elementwise_adjoint(op).is_none() && linear_scale(op).is_none() {
        return Err(
            "the gradient of this `map` is not known; only \
             `map(t) |v| { yield gelu(v) }` can be differentiated"
                .to_string(),
        );
    }
    Ok(())
}

/// A loop can be differentiated when its carried values are only
/// accumulated: every use of an iter arg is a linear `yield acc + expr`
/// (also `-` and a constant factor), and the rest of the body does not read
/// the carried values at all.
fn check_for(op: &Op) -> Result<(), String> {
    let block = &op.regions[0].blocks[0];
    let iter_args = &block.args[1..];
    let body = &block.ops;
    let yield_op = body
        .last()
        .filter(|o| o.name == "scf.yield")
        .ok_or_else(|| "the loop has no yield".to_string())?;
    if yield_op.operands.len() != iter_args.len() {
        return Err("the loop does not yield one value per carried value".to_string());
    }
    let defs: HashMap<usize, &Op> = body
        .iter()
        .flat_map(|o| o.results.iter().map(move |r| (r.id, o)))
        .collect();
    let mut visited: HashSet<usize> = HashSet::new();
    for (operand, arg) in yield_op.operands.iter().zip(iter_args.iter()) {
        let aff = parse_affine(operand, arg, iter_args, &defs, &mut visited)?;
        if matches!(arg.ty, Type::Tensor { .. })
            && aff.coeff != 0.0
            && aff.coeff != 1.0
            && aff.coeff != -1.0
        {
            return Err(
                "a tensor can only be accumulated with the factor 1: \
                 `yield acc + expr`"
                    .to_string(),
            );
        }
    }
    // Every operation that the parses did not visit must be free of the
    // carried values; those are replayed and reversed by the backward loop.
    for o in body {
        if o.name == "scf.yield" {
            continue;
        }
        if touches_arg(o, iter_args, &defs) {
            let covered = o.results.iter().all(|r| visited.contains(&r.id));
            if !covered {
                return Err(
                    "the carried values of the loop may only be read by the \
                     accumulation in `yield`: `yield acc + expr`"
                        .to_string(),
                );
            }
        } else {
            check_straight_op(o)?;
        }
    }
    Ok(())
}

/// One operand of a yield, parsed as `coeff * arg + sum(scale * term)`.
struct Affine {
    coeff: f64,
    terms: Vec<(f64, Value)>,
    /// Whether the parsed value reads the carried value at all.
    touched: bool,
}

fn parse_affine(
    v: &Value,
    arg: &Value,
    args: &[Value],
    defs: &HashMap<usize, &Op>,
    visited: &mut HashSet<usize>,
) -> Result<Affine, String> {
    if v.id == arg.id {
        return Ok(Affine {
            coeff: 1.0,
            terms: Vec::new(),
            touched: true,
        });
    }
    if args.iter().any(|a| a.id == v.id) {
        return Err(
            "each `yield` operand may only accumulate one carried value".to_string()
        );
    }
    let op = match defs.get(&v.id) {
        Some(op) => *op,
        // A value from outside the loop.
        None => {
            return Ok(Affine {
                coeff: 0.0,
                terms: vec![(1.0, v.clone())],
                touched: false,
            });
        }
    };
    visited.insert(v.id);
    match op.name.as_str() {
        "arith.constant" => Ok(Affine {
            coeff: 0.0,
            terms: vec![(1.0, v.clone())],
            touched: false,
        }),
        "arith.addf" | "arith.subf" => {
            let minus = op.name == "arith.subf";
            let a = parse_affine(&op.operands[0], arg, args, defs, visited)?;
            let b = parse_affine(&op.operands[1], arg, args, defs, visited)?;
            let mut coeff = a.coeff;
            let mut terms = a.terms;
            for (s, t) in b.terms {
                terms.push((if minus { -s } else { s }, t));
            }
            if minus {
                coeff -= b.coeff;
            } else {
                coeff += b.coeff;
            }
            Ok(Affine {
                coeff,
                terms,
                touched: a.touched || b.touched,
            })
        }
        "arith.mulf" => {
            let a = parse_affine(&op.operands[0], arg, args, defs, visited)?;
            let b = parse_affine(&op.operands[1], arg, args, defs, visited)?;
            match (a.touched, b.touched) {
                (true, true) => Err(
                    "the carried value may only be multiplied by a constant"
                        .to_string(),
                ),
                (true, false) => {
                    let c = constant_of(&b, defs)
                        .ok_or_else(|| {
                            "the carried value may only be multiplied by a constant"
                                .to_string()
                        })?;
                    let terms = a
                        .terms
                        .into_iter()
                        .map(|(s, t)| (s * c, t))
                        .collect();
                    Ok(Affine {
                        coeff: a.coeff * c,
                        terms,
                        touched: true,
                    })
                }
                (false, true) => {
                    let c = constant_of(&a, defs).ok_or_else(|| {
                        "the carried value may only be multiplied by a constant"
                            .to_string()
                    })?;
                    let terms = b
                        .terms
                        .into_iter()
                        .map(|(s, t)| (s * c, t))
                        .collect();
                    Ok(Affine {
                        coeff: b.coeff * c,
                        terms,
                        touched: true,
                    })
                }
                (false, false) => Ok(Affine {
                    coeff: 0.0,
                    terms: vec![(1.0, v.clone())],
                    touched: false,
                }),
            }
        }
        _ => {
            if touches_arg(op, args, defs) {
                return Err(
                    "the carried values of the loop may only be read by the \
                     accumulation in `yield`: `yield acc + expr`"
                        .to_string(),
                );
            }
            Ok(Affine {
                coeff: 0.0,
                terms: vec![(1.0, v.clone())],
                touched: false,
            })
        }
    }
}

/// The constant value of an affine form that is a single unscaled constant.
fn constant_of(aff: &Affine, defs: &HashMap<usize, &Op>) -> Option<f64> {
    if aff.coeff != 0.0 || aff.terms.len() != 1 || aff.terms[0].0 != 1.0 {
        return None;
    }
    let op = defs.get(&aff.terms[0].1.id)?;
    if op.name != "arith.constant" {
        return None;
    }
    match op.attr("value") {
        Some(Attr::Float(v)) => Some(*v),
        Some(Attr::Int(v)) => Some(*v as f64),
        _ => None,
    }
}

/// Whether the operation reads one of the carried values, directly or
/// through the operations of the block that define its operands.
fn touches_arg(op: &Op, args: &[Value], defs: &HashMap<usize, &Op>) -> bool {
    let mut seen: HashSet<usize> = HashSet::new();
    fn walk(
        v: &Value,
        args: &[Value],
        defs: &HashMap<usize, &Op>,
        seen: &mut HashSet<usize>,
    ) -> bool {
        if args.iter().any(|a| a.id == v.id) {
            return true;
        }
        if !seen.insert(v.id) {
            return false;
        }
        match defs.get(&v.id) {
            Some(op) => op.operands.iter().any(|o| walk(o, args, defs, seen)),
            None => false,
        }
    }
    op.operands.iter().any(|o| walk(o, args, defs, &mut seen))
}
