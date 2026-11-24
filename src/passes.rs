//! The lowering pipeline.
//!
//! ```text
//! Stage 1  typed AST  ->  high level IR   (func / scf / arith / tensor / mlr.*)
//! Stage 2  canonicalization + shape propagation
//! Stage 3  domain lowering               (mlr.map -> linalg.generic, mlr.reduce -> scf.for)
//! Stage 4  bufferization                 (tensor -> memref, linalg.generic -> loops)
//! ```
//!
//! Every stage only does one thing, so a failure points at a single
//! transformation.

use std::collections::HashMap;

use crate::ir::{Attr, Builder, Op, Region, Type, Value};

// ---------------------------------------------------------------------------
// Stage 2: canonicalization
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Const {
    Int(i64),
    Float(f64),
    Bool(bool),
}

pub fn canonicalize(module: &mut crate::ir::Module) {
    for f in &mut module.funcs {
        canonicalize_ops(&mut f.body.blocks[0].ops);
    }
}

fn canonicalize_ops(ops: &mut Vec<Op>) {
    for op in ops.iter_mut() {
        for region in &mut op.regions {
            for block in &mut region.blocks {
                canonicalize_ops(&mut block.ops);
            }
        }
    }
    fold_constants(ops);
    eliminate_dead_code(ops);
}

fn collect_constants(ops: &[Op]) -> HashMap<usize, Const> {
    let mut consts = HashMap::new();
    for op in ops {
        if op.name == "arith.constant" {
            if let Some(r) = op.results.first() {
                let c = match op.attr("value") {
                    Some(Attr::Int(v)) => Const::Int(*v),
                    Some(Attr::Float(v)) => Const::Float(*v),
                    Some(Attr::Bool(v)) => Const::Bool(*v),
                    _ => continue,
                };
                consts.insert(r.id, c);
            }
        }
    }
    consts
}

fn fold_constants(ops: &mut [Op]) {
    let consts = collect_constants(ops);
    if consts.is_empty() {
        return;
    }
    for op in ops.iter_mut() {
        let folded = match (op.name.as_str(), op.operands.len()) {
            ("arith.addi", 2)
            | ("arith.subi", 2)
            | ("arith.muli", 2)
            | ("arith.divsi", 2)
            | ("arith.remsi", 2) => {
                let (a, b) = (
                    consts.get(&op.operands[0].id),
                    consts.get(&op.operands[1].id),
                );
                match (a, b) {
                    (Some(Const::Int(a)), Some(Const::Int(b))) => {
                        let v = match op.name.as_str() {
                            "arith.addi" => a.wrapping_add(*b),
                            "arith.subi" => a.wrapping_sub(*b),
                            "arith.muli" => a.wrapping_mul(*b),
                            "arith.divsi" => {
                                if *b == 0 {
                                    continue;
                                } else {
                                    a / b
                                }
                            }
                            _ => {
                                if *b == 0 {
                                    continue;
                                } else {
                                    a % b
                                }
                            }
                        };
                        Some(Const::Int(v))
                    }
                    _ => None,
                }
            }
            ("arith.addf", 2) | ("arith.subf", 2) | ("arith.mulf", 2) | ("arith.divf", 2) => {
                let (a, b) = (
                    consts.get(&op.operands[0].id),
                    consts.get(&op.operands[1].id),
                );
                match (a, b) {
                    (Some(Const::Float(a)), Some(Const::Float(b))) => {
                        let v = match op.name.as_str() {
                            "arith.addf" => a + b,
                            "arith.subf" => a - b,
                            "arith.mulf" => a * b,
                            _ => a / b,
                        };
                        Some(Const::Float(v))
                    }
                    _ => None,
                }
            }
            ("arith.cmpi", 2) | ("arith.cmpf", 2) => {
                let (a, b) = (
                    consts.get(&op.operands[0].id),
                    consts.get(&op.operands[1].id),
                );
                let pred = match op.attr("predicate") {
                    Some(Attr::Str(p)) => p.clone(),
                    _ => String::new(),
                };
                match (a, b) {
                    (Some(Const::Int(a)), Some(Const::Int(b))) => {
                        let r = match pred.as_str() {
                            "eq" => a == b,
                            "ne" => a != b,
                            "slt" => a < b,
                            "sle" => a <= b,
                            "sgt" => a > b,
                            "sge" => a >= b,
                            _ => continue,
                        };
                        Some(Const::Bool(r))
                    }
                    (Some(Const::Float(a)), Some(Const::Float(b))) => {
                        let r = match pred.as_str() {
                            "oeq" => a == b,
                            "one" => a != b,
                            "olt" => a < b,
                            "ole" => a <= b,
                            "ogt" => a > b,
                            "oge" => a >= b,
                            _ => continue,
                        };
                        Some(Const::Bool(r))
                    }
                    _ => None,
                }
            }
            _ => None,
        };
        if let Some(c) = folded {
            let ty = op.results.first().map(|r| r.ty.clone());
            op.name = "arith.constant".into();
            op.operands.clear();
            op.regions.clear();
            op.attrs.clear();
            op.attrs.push((
                "value".into(),
                match c {
                    Const::Int(v) => Attr::Int(v),
                    Const::Float(v) => Attr::Float(v),
                    Const::Bool(v) => Attr::Bool(v),
                },
            ));
            if let (Some(r), Some(t)) = (op.results.first_mut(), ty) {
                r.ty = t;
            }
        }
    }
}

fn is_pure(op: &Op) -> bool {
    op.name.starts_with("arith.")
        || op.name.starts_with("tensor.")
        || op.name == "memref.alloc"
        || op.name == "memref.load"
        || op.name == "mlr.map"
        || op.name == "mlr.reduce"
        || op.name == "linalg.generic"
}

fn eliminate_dead_code(ops: &mut Vec<Op>) {
    let mut used = std::collections::HashSet::new();
    for op in ops.iter() {
        for o in &op.operands {
            used.insert(o.id);
        }
        for region in &op.regions {
            for block in &region.blocks {
                for op in &block.ops {
                    for o in &op.operands {
                        used.insert(o.id);
                    }
                }
            }
        }
    }
    ops.retain(|op| {
        if !is_pure(op) {
            return true;
        }
        !op.results.is_empty() && !op.results.iter().all(|r| !used.contains(&r.id))
    });
}

// ---------------------------------------------------------------------------
// Stage 3: domain lowering
// ---------------------------------------------------------------------------

/// Lowers the `mlr` domain operations into standard dialects:
/// `mlr.map` becomes `linalg.generic`, `mlr.reduce` becomes a structured
/// accumulation loop.
pub fn lower_domain(module: &mut crate::ir::Module) {
    for f in &mut module.funcs {
        lower_domain_ops(&mut f.body.blocks[0].ops);
    }
}

fn lower_domain_ops(ops: &mut Vec<Op>) {
    for op in ops.iter_mut() {
        for region in &mut op.regions {
            for block in &mut region.blocks {
                lower_domain_ops(&mut block.ops);
            }
        }
    }
    let mut out = Vec::new();
    for op in ops.drain(..) {
        match op.name.as_str() {
            "mlr.map" => out.extend(map_to_linalg(op)),
            "mlr.reduce" => out.extend(reduce_to_loop(op)),
            _ => out.push(op),
        }
    }
    *ops = out;
}

/// `map(t) |v| { ... }` becomes an element wise `linalg.generic` over the
/// tensor, which keeps the iteration space and the region intact.
fn map_to_linalg(op: Op) -> Vec<Op> {
    let ins = op.operands.clone();
    let result_ty = op
        .results
        .first()
        .map(|r| r.ty.clone())
        .unwrap_or(Type::Void);
    let rank = match &result_ty {
        Type::Tensor { shape, .. } => shape.len(),
        _ => 1,
    };
    let body = op
        .regions
        .first()
        .and_then(|r| r.blocks.first())
        .cloned()
        .unwrap_or_default();
    let body_args = body.args.clone();

    // The output tensor is created with `tensor.empty`.
    let mut b = Builder::new();
    let mut empty = Op::new("tensor.empty");
    let out = b.fresh_value(result_ty.clone());
    empty.results.push(out.clone());

    let mut new = Op::new("linalg.generic");
    let mut maps = Vec::new();
    for _ in 0..ins.len() + 1 {
        maps.push(crate::ir::AffineMap::identity(rank));
    }
    new.attrs
        .push(("indexing_maps".into(), Attr::AffineMaps(maps)));
    new.attrs.push((
        "iterator_types".into(),
        Attr::StrList(vec!["parallel".into(); rank]),
    ));
    new.attrs
        .push(("ins_count".into(), Attr::Int(ins.len() as i64)));
    new.attrs.push(("outs_count".into(), Attr::Int(1)));
    new.operands = ins;
    new.operands.push(out.clone());
    new.results = op.results.clone();

    // The body of the region: the block arguments are the elements of the
    // inputs, followed by the (unused) element of the output.
    let mut body_ops = body.ops.clone();
    for op in &mut body_ops {
        if op.name == "mlr.yield" {
            op.name = "linalg.yield".into();
        }
    }
    let mut args = body_args;
    let out_elem = match &result_ty {
        Type::Tensor { elem, .. } => (**elem).clone(),
        _ => Type::Scalar(crate::ir::ScalarKind::F32),
    };
    args.push(b.fresh_value(out_elem));
    new.regions.push(Region {
        blocks: vec![crate::ir::Block {
            args,
            ops: body_ops,
        }],
    });
    vec![empty, new]
}

/// `reduce(init, t) |acc, v| { ... }` becomes a structured accumulation loop.
fn reduce_to_loop(op: Op) -> Vec<Op> {
    let tensor = op.operands[0].clone();
    let init = op.operands[1].clone();
    let result_ty = op
        .results
        .first()
        .map(|r| r.ty.clone())
        .unwrap_or(Type::Void);
    let body = op
        .regions
        .first()
        .and_then(|r| r.blocks.first())
        .cloned()
        .unwrap_or_default();
    let body_args = body.args.clone();

    let mut b = Builder::new();
    let mut lb = Op::new("arith.constant");
    lb.attrs.push(("value".into(), Attr::Int(0)));
    let lb_v = b.fresh_value(Type::Index);
    lb.results.push(lb_v.clone());

    let mut step = Op::new("arith.constant");
    step.attrs.push(("value".into(), Attr::Int(1)));
    let step_v = b.fresh_value(Type::Index);
    step.results.push(step_v.clone());

    // The upper bound is the size of the reduced dimension.
    let mut ub = Op::new("mlr.dim");
    ub.operands.push(tensor.clone());
    ub.attrs.push(("dim".into(), Attr::Int(0)));
    let ub_v = b.fresh_value(Type::Index);
    ub.results.push(ub_v.clone());

    let mut for_op = Op::new("scf.for");
    for_op.operands.push(lb_v);
    for_op.operands.push(ub_v);
    for_op.operands.push(step_v);
    for_op.operands.push(init.clone());
    let iv = b.fresh_value(Type::Index);
    let acc_arg = b.fresh_value(result_ty.clone());

    // Extract the element and run the body.
    let mut extract = Op::new("tensor.extract");
    extract.operands.push(tensor);
    extract.operands.push(iv.clone());
    let elem = b.fresh_value(match &init.ty {
        Type::Scalar(k) => Type::Scalar(*k),
        _ => Type::Scalar(crate::ir::ScalarKind::F32),
    });
    extract.results.push(elem.clone());

    let mut body_ops = vec![extract];
    let mut subst: HashMap<usize, Value> = HashMap::new();
    if let Some(a) = body_args.first() {
        subst.insert(a.id, acc_arg.clone());
    }
    if let Some(a) = body_args.get(1) {
        subst.insert(a.id, elem.clone());
    }
    for mut op in body.ops {
        substitute(&mut op, &subst);
        if op.name == "mlr.yield" {
            op.name = "scf.yield".into();
        }
        body_ops.push(op);
    }
    for_op.regions.push(Region {
        blocks: vec![crate::ir::Block {
            args: vec![iv, acc_arg],
            ops: body_ops,
        }],
    });
    // The loop produces the value of the reduction.
    for_op.results = op.results.clone();
    vec![lb, step, ub, for_op]
}

/// Replaces every use of a value with another one.
fn substitute(op: &mut Op, subst: &HashMap<usize, Value>) {
    for operand in &mut op.operands {
        if let Some(v) = subst.get(&operand.id) {
            *operand = v.clone();
        }
    }
    for result in &mut op.results {
        if let Some(v) = subst.get(&result.id) {
            *result = v.clone();
        }
    }
    for region in &mut op.regions {
        for block in &mut region.blocks {
            for op in &mut block.ops {
                substitute(op, subst);
            }
        }
    }
}

pub mod bufferize;
pub mod runtime_ops;
