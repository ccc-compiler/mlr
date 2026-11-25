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

use crate::ir::{Attr, Builder, Dim, FuncOp, Op, Type, Value};
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
    "func.return",
];

/// Builds `f.value_and_grad` from the IR of `f`: the loss and one gradient
/// slot per tensor parameter. Returns the function and the runtime functions
/// it calls (for the declarations of the module).
pub fn value_and_grad_of(f: &FuncOp, funcs: &[FuncOp]) -> (FuncOp, Vec<&'static str>) {
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
    let body = body;

    // The adjoint of every value: the gradient that has accumulated so far.
    // The loss starts with one.
    let mut adj: HashMap<usize, Value> = HashMap::new();
    // The calls of pure functions are inlined, so that the backward walk
    // sees their operations.
    let forward = expand_calls(body, funcs, 0);
    let mut ops: Vec<Op> = forward.clone();

    let mut one = Op::new("arith.constant");
    one.attrs.push(("value".into(), Attr::Float(1.0)));
    let seed = b.fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
    one.results.push(seed.clone());
    ops.push(one);
    adj.insert(loss.id, seed);

    // The backward pass, from the loss back to the parameters.
    for op in forward.iter().rev() {
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
                let da = b.fresh_value(lhs.ty.clone());
                let db = b.fresh_value(rhs.ty.clone());
                g_op.results = vec![da.clone(), db.clone()];
                ops.push(g_op);
                accumulate(&mut ops, &mut b, &mut adj, &lhs, da);
                accumulate(&mut ops, &mut b, &mut adj, &rhs, db);
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
                let dx = b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                ops.push(g_op);
                accumulate(&mut ops, &mut b, &mut adj, &x, dx);
            }
            "mlr.tclamp" => {
                let x = op.operands[0].clone();
                let mut g_op = Op::new("mlr.tclamp_grad");
                g_op.operands = vec![x.clone(), op.operands[1].clone(), op.operands[2].clone(), dy];
                let dx = b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                ops.push(g_op);
                accumulate(&mut ops, &mut b, &mut adj, &x, dx);
            }
            "mlr.tpow" => {
                let x = op.operands[0].clone();
                let mut g_op = Op::new("mlr.tpow_grad");
                g_op.operands = vec![x.clone(), op.operands[1].clone(), dy];
                let dx = b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                ops.push(g_op);
                accumulate(&mut ops, &mut b, &mut adj, &x, dx);
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
                let dx = b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                ops.push(g_op);
                accumulate(&mut ops, &mut b, &mut adj, &x, dx);
            }
            "mlr.mean" => {
                // The gradient of the mean is the gradient of the sum with
                // the seed divided by the number of the elements.
                used.push("mlr_sum_grad");
                let x = op.operands[0].clone();
                let n = match &x.ty {
                    Type::Tensor { shape, .. } | Type::MemRef { shape, .. } => shape
                        .iter()
                        .fold(1i64, |acc, d| match d {
                            Dim::Static(v) => acc * v,
                            Dim::Dynamic => acc,
                        }),
                    _ => 1,
                };
                let mut cst = Op::new("arith.constant");
                cst.attrs.push(("value".into(), Attr::Float(n as f64)));
                let cval = b.fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
                cst.results.push(cval.clone());
                ops.push(cst);
                let mut div = Op::new("arith.divf");
                div.operands = vec![dy, cval.clone()];
                let seed = b.fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
                div.results.push(seed.clone());
                ops.push(div);
                let mut g_op = Op::new("mlr.sum_grad");
                g_op.operands = vec![x.clone(), seed];
                let dx = b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                ops.push(g_op);
                accumulate(&mut ops, &mut b, &mut adj, &x, dx);
            }
            "mlr.norm1" => {
                let x = op.operands[0].clone();
                let mut g_op = Op::new("mlr.norm1_grad");
                g_op.operands = vec![x.clone(), dy];
                let dx = b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                ops.push(g_op);
                accumulate(&mut ops, &mut b, &mut adj, &x, dx);
            }
            "mlr.norm2" => {
                let x = op.operands[0].clone();
                let mut g_op = Op::new("mlr.norm2_grad");
                g_op.operands = vec![x.clone(), y.clone(), dy];
                let dx = b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                ops.push(g_op);
                accumulate(&mut ops, &mut b, &mut adj, &x, dx);
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
                let dx = b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                ops.push(g_op);
                accumulate(&mut ops, &mut b, &mut adj, &x, dx);
            }
            "mlr.dot" => {
                let lhs = op.operands[0].clone();
                let rhs = op.operands[1].clone();
                let mut g_op = Op::new("mlr.dot_grad");
                g_op.operands = vec![lhs.clone(), rhs.clone(), dy];
                let da = b.fresh_value(lhs.ty.clone());
                let db = b.fresh_value(rhs.ty.clone());
                g_op.results = vec![da.clone(), db.clone()];
                ops.push(g_op);
                accumulate(&mut ops, &mut b, &mut adj, &lhs, da);
                accumulate(&mut ops, &mut b, &mut adj, &rhs, db);
            }
            "mlr.reshape" => {
                // The inverse of the reshape is a copy of the original shape.
                used.push("mlr_copy");
                let x = op.operands[0].clone();
                let mut g_op = Op::new("mlr.reshape_grad");
                g_op.operands = vec![dy];
                let dx = b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                ops.push(g_op);
                accumulate(&mut ops, &mut b, &mut adj, &x, dx);
            }
            "mlr.concat" => {
                let lhs = op.operands[0].clone();
                let rhs = op.operands[1].clone();
                let mut g_op = Op::new("mlr.concat_grad");
                g_op.operands = vec![dy];
                let da = b.fresh_value(lhs.ty.clone());
                let db = b.fresh_value(rhs.ty.clone());
                g_op.results = vec![da.clone(), db.clone()];
                ops.push(g_op);
                accumulate(&mut ops, &mut b, &mut adj, &lhs, da);
                accumulate(&mut ops, &mut b, &mut adj, &rhs, db);
            }
            "mlr.rows" => {
                let x = op.operands[0].clone();
                let mut g_op = Op::new("mlr.rows_grad");
                g_op.operands = vec![
                    dy,
                    op.operands[1].clone(),
                    op.operands[2].clone(),
                ];
                let dx = b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                ops.push(g_op);
                accumulate(&mut ops, &mut b, &mut adj, &x, dx);
            }
            "mlr.dropout" => {
                // The mask is recomputed from the same seed and rate.
                used.push("mlr_dropout_grad");
                let x = op.operands[0].clone();
                let mut g_op = Op::new("mlr.dropout_grad");
                g_op.operands = vec![
                    x.clone(),
                    dy,
                    op.operands[1].clone(),
                    op.operands[2].clone(),
                ];
                let dx = b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                ops.push(g_op);
                accumulate(&mut ops, &mut b, &mut adj, &x, dx);
            }
            "mlr.reverse" => {
                // The reverse is its own inverse.
                let x = op.operands[0].clone();
                let mut g_op = Op::new("mlr.reverse");
                g_op.operands = vec![dy];
                let dx = b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                ops.push(g_op);
                accumulate(&mut ops, &mut b, &mut adj, &x, dx);
            }
            "mlr.layernorm" => {
                // dx, dg and db; the statistics are recomputed from x.
                used.push("mlr_layernorm_grad");
                let x = op.operands[0].clone();
                let g = op.operands[1].clone();
                let bias = op.operands[2].clone();
                let mut g_op = Op::new("mlr.layernorm_grad");
                g_op.operands = vec![x.clone(), g.clone(), dy];
                let dx = b.fresh_value(x.ty.clone());
                let dg = b.fresh_value(g.ty.clone());
                let db = b.fresh_value(bias.ty.clone());
                g_op.results = vec![dx.clone(), dg.clone(), db.clone()];
                ops.push(g_op);
                accumulate(&mut ops, &mut b, &mut adj, &x, dx);
                accumulate(&mut ops, &mut b, &mut adj, &g, dg);
                accumulate(&mut ops, &mut b, &mut adj, &bias, db);
            }
            "mlr.transpose" => {
                // The adjoint of the transpose is the transpose.
                let x = op.operands[0].clone();
                let mut t = Op::new("mlr.transpose");
                t.operands = vec![dy];
                let dx = b.fresh_value(x.ty.clone());
                t.results = vec![dx.clone()];
                ops.push(t);
                accumulate(&mut ops, &mut b, &mut adj, &x, dx);
            }
            "mlr.softmax" | "mlr.softmax_tri" => {
                // dx = y * (dy - sum(dy * y)) with y the output of the
                // forward pass; the mask of the triangular version lives in
                // the kernel.
                used.push(if op.name == "mlr.softmax" {
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
                g_op.operands = vec![y, dy];
                let dx = b.fresh_value(x.ty.clone());
                g_op.results = vec![dx.clone()];
                ops.push(g_op);
                accumulate(&mut ops, &mut b, &mut adj, &x, dx);
            }
            "mlr.add" => {
                // Both operands receive the incoming gradient.
                used.push("mlr_add_elem");
                let lhs = op.operands[0].clone();
                let rhs = op.operands[1].clone();
                accumulate(&mut ops, &mut b, &mut adj, &lhs, dy.clone());
                accumulate(&mut ops, &mut b, &mut adj, &rhs, dy);
            }
            "mlr.map" => {
                // An element wise function of one tensor.
                let x = op.operands[0].clone();
                let dx = match elementwise_adjoint(op) {
                    // A known non linear function: `map(t) |v| gelu(v)`.
                    Some(grad) => {
                        let mut g_op = Op::new(grad);
                        g_op.operands = vec![x.clone(), dy];
                        let dx = b.fresh_value(x.ty.clone());
                        g_op.results = vec![dx.clone()];
                        ops.push(g_op);
                        used.push(match grad {
                            "mlr.gelu_grad" => "mlr_gelu_grad",
                            other => panic!("unknown gradient {other}"),
                        });
                        dx
                    }
                    // A linear map: `map(t) |v| v * 0.25`. The gradient of a
                    // linear function is the same function of the incoming
                    // gradient.
                    None => linear_adjoint(&mut ops, &mut b, op, dy),
                };
                accumulate(&mut ops, &mut b, &mut adj, &x, dx);
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
            // The scalar arithmetic on the way to the loss: the adjoints
            // are the usual ones, expressed with scalar operations.
            "arith.addf" => {
                let lhs = op.operands[0].clone();
                let rhs = op.operands[1].clone();
                accumulate(&mut ops, &mut b, &mut adj, &lhs, dy.clone());
                accumulate(&mut ops, &mut b, &mut adj, &rhs, dy);
            }
            "arith.subf" => {
                let lhs = op.operands[0].clone();
                let rhs = op.operands[1].clone();
                accumulate(&mut ops, &mut b, &mut adj, &lhs, dy.clone());
                let mut neg = Op::new("arith.negf");
                neg.operands = vec![dy];
                let ndy = b.fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
                neg.results.push(ndy.clone());
                ops.push(neg);
                accumulate(&mut ops, &mut b, &mut adj, &rhs, ndy);
            }
            "arith.mulf" => {
                let lhs = op.operands[0].clone();
                let rhs = op.operands[1].clone();
                let mut pa = Op::new("arith.mulf");
                pa.operands = vec![dy.clone(), rhs.clone()];
                let pva = b.fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
                pa.results.push(pva.clone());
                ops.push(pa);
                let mut pb = Op::new("arith.mulf");
                pb.operands = vec![dy.clone(), lhs.clone()];
                let pvb = b.fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
                pb.results.push(pvb.clone());
                ops.push(pb);
                accumulate(&mut ops, &mut b, &mut adj, &lhs, pva);
                accumulate(&mut ops, &mut b, &mut adj, &rhs, pvb);
            }
            "arith.divf" => {
                let lhs = op.operands[0].clone();
                let rhs = op.operands[1].clone();
                let mut da = Op::new("arith.divf");
                da.operands = vec![dy.clone(), rhs.clone()];
                let dva = b.fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
                da.results.push(dva.clone());
                ops.push(da);
                accumulate(&mut ops, &mut b, &mut adj, &lhs, dva);
                let mut aa = Op::new("arith.mulf");
                aa.operands = vec![dy.clone(), lhs.clone()];
                let t = b.fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
                aa.results.push(t.clone());
                ops.push(aa);
                let mut bb = Op::new("arith.mulf");
                bb.operands = vec![rhs.clone(), rhs.clone()];
                let b2v = b.fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
                bb.results.push(b2v.clone());
                ops.push(bb);
                let mut db = Op::new("arith.divf");
                db.operands = vec![t, b2v];
                let ndy = b.fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
                db.results.push(ndy.clone());
                ops.push(db);
                let mut neg = Op::new("arith.negf");
                neg.operands = vec![ndy];
                let nndy = b.fresh_value(Type::Scalar(crate::ir::ScalarKind::F32));
                neg.results.push(nndy.clone());
                ops.push(neg);
                accumulate(&mut ops, &mut b, &mut adj, &rhs, nndy);
            }
            // The conversions of the scalars pass the gradient through.
            "arith.truncf" | "arith.extf" | "arith.sitofp" | "arith.index_cast"
            | "arith.trunci" | "arith.extui" => {
                if let Some(a) = op.operands.first() {
                    let a = a.clone();
                    accumulate(&mut ops, &mut b, &mut adj, &a, dy);
                }
            }
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

/// The gradient of a linear `map`: the body computes one arithmetic
/// operation of the argument and a constant, and the gradient applies the
/// scaling of the derivative to the incoming gradient.
fn linear_adjoint(ops: &mut Vec<Op>, b: &mut Builder, op: &Op, dy: Value) -> Value {
    // The operation of the body, and the constant on either side of it.
    let scale = linear_scale(op).expect("a known linear map");
    map_scale(ops, b, dy, scale)
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
    for op in &f.body.blocks[0].ops {
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
        if !FORWARD.contains(&op.name.as_str()) {
            return Err(format!(
                "the operation `{}` has no gradient yet",
                op.name
            ));
        }
        if op.name == "mlr.map" {
            if elementwise_adjoint(op).is_none() && linear_scale(op).is_none() {
                return Err(
                    "the gradient of this `map` is not known; only \
                     `map(t) |v| { yield gelu(v) }` can be differentiated"
                        .to_string(),
                );
            }
            continue;
        }
        if !op.regions.is_empty() {
            return Err(
                "the function has a region (`for` or `if`), which cannot \
                 be differentiated yet"
                    .to_string(),
            );
        }
    }
    Ok(())
}
