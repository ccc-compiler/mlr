//! Stage 5: lowering of the domain operations.
//!
//! The domain operations (`matmul`, `conv2d`, `maxpool2d`, `relu`, `flatten`,
//! `row` and `argmax`) are lowered to calls into the runtime library. The
//! runtime has two implementations:
//!
//! * `runtime/mlr_runtime.c` computes them with plain loops on the CPU,
//! * `runtime/mlr_runtime_mps.mm` computes the heavy ones (matrix
//!   multiplication and convolution) with Metal Performance Shaders on Apple
//!   GPUs.
//!
//! Because the two runtimes share the same interface, the very same program
//! runs on the CPU and on the GPU.

use crate::ir::{Attr, Builder, Dim, Op, Type, Value};

/// The runtime interface.
#[derive(Clone, Debug)]
pub struct RuntimeFn {
    pub name: &'static str,
    pub params: &'static [&'static str],
    pub result: Option<&'static str>,
}

const RUNTIME: &[RuntimeFn] = &[
    RuntimeFn {
        name: "mlr_matmul",
        params: &["index", "index", "index", "index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_conv2d",
        params: &[
            "index", "index", "index", "index", "index", "index", "index", "index",
        ],
        result: None,
    },
    RuntimeFn {
        name: "mlr_maxpool2d",
        params: &["index", "index", "index", "index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_relu",
        params: &["index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_flatten",
        params: &["index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_row",
        params: &["index", "index", "index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_add_bias",
        params: &["index", "index", "index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_argmax",
        params: &["index", "index"],
        result: Some("i32"),
    },
];

pub fn lower_runtime_ops(module: &mut crate::ir::Module) {
    let mut used: Vec<&'static str> = Vec::new();
    for f in &mut module.funcs {
        lower_ops(&mut f.body.blocks[0].ops, &mut used);
    }
    // Every runtime function is declared once.
    let mut names: Vec<&'static str> = Vec::new();
    for name in used {
        if !names.contains(&name) {
            names.push(name);
        }
    }
    for name in names {
        let f = RUNTIME
            .iter()
            .find(|f| f.name == name)
            .expect("runtime function");
        let params = f.params.iter().map(|p| Type::scalar(p).unwrap()).collect();
        let results = f
            .result
            .map(|r| vec![Type::scalar(r).unwrap()])
            .unwrap_or_default();
        module.decls.push(crate::ir::FuncDecl {
            name: name.to_string(),
            params,
            results,
        });
    }
}

fn lower_ops(ops: &mut Vec<Op>, used: &mut Vec<&'static str>) {
    for op in ops.iter_mut() {
        for region in &mut op.regions {
            for block in &mut region.blocks {
                lower_ops(&mut block.ops, used);
            }
        }
    }
    let mut out = Vec::new();
    for op in ops.drain(..) {
        match op.name.as_str() {
            "mlr.matmul" => lower_matmul(op, used, &mut out),
            "mlr.conv2d" => lower_conv2d(op, used, &mut out),
            "mlr.maxpool2d" => lower_maxpool(op, used, &mut out),
            "mlr.relu" => lower_unary(op, used, &mut out, "mlr_relu"),
            "mlr.flatten" => lower_unary(op, used, &mut out, "mlr_flatten"),
            "mlr.row" => lower_row(op, used, &mut out),
            "mlr.add_bias" => lower_add_bias(op, used, &mut out),
            "mlr.argmax" => lower_argmax(op, used, &mut out),
            _ => out.push(op),
        }
    }
    *ops = out;
}

/// The number of elements of a memref.
fn count(ty: &Type) -> i64 {
    match ty {
        Type::MemRef { shape, .. } | Type::Tensor { shape, .. } => {
            shape.iter().fold(1i64, |acc, d| {
                acc * match d {
                    Dim::Static(v) => *v,
                    Dim::Dynamic => 1,
                }
            })
        }
        _ => 1,
    }
}

/// The dimensions of a memref.
fn dims(ty: &Type) -> Vec<i64> {
    match ty {
        Type::MemRef { shape, .. } | Type::Tensor { shape, .. } => shape
            .iter()
            .map(|d| match d {
                Dim::Static(v) => *v,
                Dim::Dynamic => 0,
            })
            .collect(),
        _ => vec![],
    }
}

struct Lowerer {
    b: Builder,
}

impl Lowerer {
    fn const_index(&mut self, v: i64) -> Op {
        let mut op = Op::new("arith.constant");
        op.attrs.push(("value".into(), Attr::Int(v)));
        let res = self.b.fresh_value(Type::Index);
        op.results.push(res.clone());
        op
    }

    /// The address of a buffer as an index.
    fn pointer(&mut self, buf: &Value) -> Op {
        let mut op = Op::new("memref.extract_aligned_pointer_as_index");
        op.operands.push(buf.clone());
        let res = self.b.fresh_value(Type::Index);
        op.results.push(res.clone());
        op
    }

    fn alloc(&mut self, ty: Type, result: &Value) -> Op {
        let mut op = Op::new("memref.alloc");
        let res = Value { id: result.id, ty };
        op.results.push(res);
        op
    }
}

fn lower_matmul(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_matmul");
    let a = op.operands[0].clone();
    let b = op.operands[1].clone();
    let result = op.results[0].clone();
    let da = dims(&a.ty);
    let db = dims(&b.ty);
    // A vector is a matrix with a single row.
    let (m, k, n) = if da.len() == 1 {
        (1, da[0], db[1])
    } else {
        (da[0], da[1], db[1])
    };
    let mut l = Lowerer { b: Builder::new() };
    let out_ty = match &result.ty {
        Type::MemRef { shape, .. } => Type::MemRef {
            shape: shape.clone(),
            elem: Box::new(Type::Scalar(crate::ir::ScalarKind::F32)),
        },
        _ => Type::MemRef {
            shape: vec![Dim::Static(m), Dim::Static(n)],
            elem: Box::new(Type::Scalar(crate::ir::ScalarKind::F32)),
        },
    };
    out.push(l.alloc(out_ty, &result));
    let pa = l.pointer(&a);
    let pb = l.pointer(&b);
    let pc = l.pointer(&result);
    let cm = l.const_index(m);
    let cn = l.const_index(n);
    let ck = l.const_index(k);
    out.push(pa.clone());
    out.push(pb.clone());
    out.push(pc.clone());
    out.push(cm.clone());
    out.push(cn.clone());
    out.push(ck.clone());
    let mut call = Op::new("func.call");
    call.attrs
        .push(("callee".into(), Attr::Str("mlr_matmul".into())));
    call.operands = vec![
        pa.results[0].clone(),
        pb.results[0].clone(),
        pc.results[0].clone(),
        cm.results[0].clone(),
        cn.results[0].clone(),
        ck.results[0].clone(),
    ];
    out.push(call);
}

fn lower_conv2d(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_conv2d");
    let x = op.operands[0].clone();
    let w = op.operands[1].clone();
    let bias = op.operands[2].clone();
    let result = op.results[0].clone();
    let dx = dims(&x.ty);
    let dw = dims(&w.ty);
    // An image without the channel dimension has a single channel.
    let (h, width) = if dx.len() == 2 {
        (dx[0], dx[1])
    } else {
        (dx[1], dx[2])
    };
    let (o, k) = (dw[0], dw[2]);
    let mut l = Lowerer { b: Builder::new() };
    let out_ty = Type::MemRef {
        shape: vec![Dim::Static(o), Dim::Static(h), Dim::Static(width)],
        elem: Box::new(Type::Scalar(crate::ir::ScalarKind::F32)),
    };
    out.push(l.alloc(out_ty, &result));
    let px = l.pointer(&x);
    let pw = l.pointer(&w);
    let pb = l.pointer(&bias);
    let pout = l.pointer(&result);
    let co = l.const_index(o);
    let ch = l.const_index(h);
    let cw = l.const_index(width);
    let ck = l.const_index(k);
    out.push(px.clone());
    out.push(pw.clone());
    out.push(pb.clone());
    out.push(pout.clone());
    out.push(co.clone());
    out.push(ch.clone());
    out.push(cw.clone());
    out.push(ck.clone());
    let mut call = Op::new("func.call");
    call.attrs
        .push(("callee".into(), Attr::Str("mlr_conv2d".into())));
    call.operands = vec![
        px.results[0].clone(),
        pw.results[0].clone(),
        pb.results[0].clone(),
        pout.results[0].clone(),
        co.results[0].clone(),
        ch.results[0].clone(),
        cw.results[0].clone(),
        ck.results[0].clone(),
    ];
    out.push(call);
}

fn lower_maxpool(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_maxpool2d");
    let x = op.operands[0].clone();
    let result = op.results[0].clone();
    let k = match op.attr("p0") {
        Some(Attr::Int(v)) => *v,
        _ => 1,
    };
    let dx = dims(&x.ty);
    let (o, h, width) = (dx[0], dx[1], dx[2]);
    let mut l = Lowerer { b: Builder::new() };
    let out_ty = Type::MemRef {
        shape: vec![Dim::Static(o), Dim::Static(h / k), Dim::Static(width / k)],
        elem: Box::new(Type::Scalar(crate::ir::ScalarKind::F32)),
    };
    out.push(l.alloc(out_ty, &result));
    let px = l.pointer(&x);
    let pout = l.pointer(&result);
    let co = l.const_index(o);
    let ch = l.const_index(h);
    let cw = l.const_index(width);
    let ck = l.const_index(k);
    out.push(px.clone());
    out.push(pout.clone());
    out.push(co.clone());
    out.push(ch.clone());
    out.push(cw.clone());
    out.push(ck.clone());
    let mut call = Op::new("func.call");
    call.attrs
        .push(("callee".into(), Attr::Str("mlr_maxpool2d".into())));
    call.operands = vec![
        px.results[0].clone(),
        pout.results[0].clone(),
        co.results[0].clone(),
        ch.results[0].clone(),
        cw.results[0].clone(),
        ck.results[0].clone(),
    ];
    out.push(call);
}

/// `relu` and `flatten` have the same shape: an element wise operation.
fn lower_unary(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>, name: &'static str) {
    used.push(name);
    let x = op.operands[0].clone();
    let result = op.results[0].clone();
    let n = count(&x.ty);
    let mut l = Lowerer { b: Builder::new() };
    // `flatten` changes the shape, the other operations keep it.
    let out_ty = match &result.ty {
        Type::MemRef { shape, .. } => Type::MemRef {
            shape: shape.clone(),
            elem: Box::new(Type::Scalar(crate::ir::ScalarKind::F32)),
        },
        _ => Type::MemRef {
            shape: match &x.ty {
                Type::MemRef { shape, .. } => shape.clone(),
                _ => vec![Dim::Static(n)],
            },
            elem: Box::new(Type::Scalar(crate::ir::ScalarKind::F32)),
        },
    };
    out.push(l.alloc(out_ty, &result));
    let px = l.pointer(&x);
    let pout = l.pointer(&result);
    let cn = l.const_index(n);
    out.push(px.clone());
    out.push(pout.clone());
    out.push(cn.clone());
    let mut call = Op::new("func.call");
    call.attrs
        .push(("callee".into(), Attr::Str(name.to_string())));
    call.operands = vec![
        px.results[0].clone(),
        pout.results[0].clone(),
        cn.results[0].clone(),
    ];
    out.push(call);
}

fn lower_row(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_row");
    let t = op.operands[0].clone();
    let index = op.operands[1].clone();
    let result = op.results[0].clone();
    let dt = dims(&t.ty);
    let stride: i64 = dt[1..].iter().product();
    let n = count(&result.ty);
    let mut l = Lowerer { b: Builder::new() };
    let out_ty = Type::MemRef {
        shape: match &result.ty {
            Type::MemRef { shape, .. } => shape.clone(),
            _ => vec![Dim::Static(n)],
        },
        elem: Box::new(Type::Scalar(crate::ir::ScalarKind::F32)),
    };
    out.push(l.alloc(out_ty, &result));
    let pt = l.pointer(&t);
    let pout = l.pointer(&result);
    let cn = l.const_index(n);
    let cstride = l.const_index(stride);
    out.push(pt.clone());
    out.push(pout.clone());
    out.push(cn.clone());
    out.push(cstride.clone());
    let mut call = Op::new("func.call");
    call.attrs
        .push(("callee".into(), Attr::Str("mlr_row".into())));
    call.operands = vec![
        pt.results[0].clone(),
        pout.results[0].clone(),
        cn.results[0].clone(),
        cstride.results[0].clone(),
        index,
    ];
    out.push(call);
}

/// A bias vector is added to every row of a matrix.
fn lower_add_bias(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_add_bias");
    let x = op.operands[0].clone();
    let bias = op.operands[1].clone();
    let result = op.results[0].clone();
    let dx = dims(&x.ty);
    // A vector is a matrix with a single row.
    let (m, n) = if dx.len() == 1 {
        (1, dx[0])
    } else {
        (dx[0], dx[1])
    };
    let mut l = Lowerer { b: Builder::new() };
    let out_ty = match &result.ty {
        Type::MemRef { shape, .. } => Type::MemRef {
            shape: shape.clone(),
            elem: Box::new(Type::Scalar(crate::ir::ScalarKind::F32)),
        },
        _ => Type::MemRef {
            shape: vec![Dim::Static(m), Dim::Static(n)],
            elem: Box::new(Type::Scalar(crate::ir::ScalarKind::F32)),
        },
    };
    out.push(l.alloc(out_ty, &result));
    let px = l.pointer(&x);
    let pb = l.pointer(&bias);
    let pout = l.pointer(&result);
    let cm = l.const_index(m);
    let cn = l.const_index(n);
    out.push(px.clone());
    out.push(pb.clone());
    out.push(pout.clone());
    out.push(cm.clone());
    out.push(cn.clone());
    let mut call = Op::new("func.call");
    call.attrs
        .push(("callee".into(), Attr::Str("mlr_add_bias".into())));
    call.operands = vec![
        px.results[0].clone(),
        pb.results[0].clone(),
        pout.results[0].clone(),
        cm.results[0].clone(),
        cn.results[0].clone(),
    ];
    out.push(call);
}

fn lower_argmax(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_argmax");
    let x = op.operands[0].clone();
    let result = op.results[0].clone();
    let n = count(&x.ty);
    let mut l = Lowerer { b: Builder::new() };
    let px = l.pointer(&x);
    let cn = l.const_index(n);
    out.push(px.clone());
    out.push(cn.clone());
    let mut call = Op::new("func.call");
    call.attrs
        .push(("callee".into(), Attr::Str("mlr_argmax".into())));
    call.operands = vec![px.results[0].clone(), cn.results[0].clone()];
    call.results.push(result);
    out.push(call);
}
