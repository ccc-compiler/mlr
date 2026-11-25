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
        name: "mlr_embed",
        params: &["index", "i32", "index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_embed_seq",
        params: &["index", "index", "index", "index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_transpose",
        params: &["index", "index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_layernorm",
        params: &["index", "index", "index", "index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_softmax_causal",
        params: &["index", "i32", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_sample",
        params: &["index", "f32", "i32", "index"],
        result: Some("i32"),
    },
    RuntimeFn {
        name: "mlr_argmax",
        params: &["index", "index"],
        result: Some("i32"),
    },
    RuntimeFn {
        name: "mlr_cross_entropy",
        params: &["index", "index", "index", "index"],
        result: Some("f32"),
    },
    RuntimeFn {
        name: "mlr_cross_entropy_grad",
        params: &["index", "index", "f32", "index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_softmax",
        params: &["index", "index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_zero",
        params: &["index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_add_elem",
        params: &["index", "index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_matmul_grad",
        params: &["index", "index", "index", "index", "index", "index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_add_bias_grad",
        params: &["index", "index", "index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_sum",
        params: &["index", "index"],
        result: Some("f32"),
    },
    RuntimeFn {
        name: "mlr_sum_grad",
        params: &["index", "f32", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_copy",
        params: &["index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_conv2d_grad",
        params: &[
            "index", "index", "index", "index", "index", "index", "index", "index", "index",
            "index",
        ],
        result: None,
    },
    RuntimeFn {
        name: "mlr_maxpool2d_grad",
        params: &["index", "index", "index", "index", "index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_embed_grad",
        params: &["i32", "index", "index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_embed_seq_grad",
        params: &["index", "index", "index", "index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_layernorm_grad",
        params: &[
            "index", "index", "index", "index", "index", "index", "index", "index",
        ],
        result: None,
    },
    RuntimeFn {
        name: "mlr_softmax_grad",
        params: &["index", "index", "index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_softmax_tri",
        params: &["index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_softmax_tri_grad",
        params: &["index", "index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_relu_grad",
        params: &["index", "index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_gelu_grad",
        params: &["index", "index", "index", "index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_save_pt_begin",
        params: &["index"],
        result: None,
    },
    RuntimeFn {
        name: "mlr_save_pt_add",
        params: &[
            "index", "index", "index", "index", "index", "index", "index", "index", "index",
        ],
        result: None,
    },
    RuntimeFn {
        name: "mlr_save_pt_end",
        params: &[],
        result: None,
    },
];

pub fn lower_runtime_ops(module: &mut crate::ir::Module) {
    let mut used: Vec<&'static str> = Vec::new();
    let mut globals: Vec<crate::ir::Global> = Vec::new();
    let mut counter = 0usize;
    for f in &mut module.funcs {
        lower_ops(&mut f.body.blocks[0].ops, &mut used, &mut globals, &mut counter);
    }
    module.globals.extend(globals);
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

fn lower_ops(
    ops: &mut Vec<Op>,
    used: &mut Vec<&'static str>,
    globals: &mut Vec<crate::ir::Global>,
    counter: &mut usize,
) {
    for op in ops.iter_mut() {
        for region in &mut op.regions {
            for block in &mut region.blocks {
                lower_ops(&mut block.ops, used, globals, counter);
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
            "mlr.embed" => lower_embed(op, used, &mut out),
            "mlr.embed_seq" => lower_embed_seq(op, used, &mut out),
            "mlr.transpose" => lower_transpose(op, used, &mut out),
            "mlr.layernorm" => lower_layernorm(op, used, &mut out),
            "mlr.softmax_causal" => lower_softmax_causal(op, used, &mut out),
            "mlr.sample" => lower_sample(op, used, &mut out),
            "mlr.argmax" => lower_argmax(op, used, &mut out),
            "mlr.save_pt" => lower_save_pt(op, used, &mut out, globals, counter),
            "mlr.cross_entropy" => lower_cross_entropy(op, used, &mut out),
            "mlr.sum" => lower_sum(op, used, &mut out),
            "mlr.sum_grad" => lower_sum_grad(op, used, &mut out),
            "mlr.flatten_grad" => lower_copy_grad(op, used, &mut out, "mlr_copy"),
            "mlr.conv2d_grad" => lower_conv2d_grad(op, used, &mut out),
            "mlr.maxpool2d_grad" => lower_maxpool_grad(op, used, &mut out),
            "mlr.embed_grad" => lower_embed_grad(op, used, &mut out),
            "mlr.embed_seq_grad" => lower_embed_seq_grad(op, used, &mut out),
            "mlr.cross_entropy_grad" => lower_cross_entropy_grad(op, used, &mut out),
            "mlr.softmax" => lower_softmax(op, used, &mut out),
            "mlr.softmax_tri" => lower_softmax_tri(op, used, &mut out),
            "mlr.softmax_grad" => lower_softmax_grad(op, used, &mut out),
            "mlr.softmax_tri_grad" => lower_softmax_tri_grad(op, used, &mut out),
            "mlr.layernorm_grad" => lower_layernorm_grad(op, used, &mut out),
            "mlr.zeros" => lower_zeros(op, used, &mut out),
            "mlr.add" => lower_add_elem(op, used, &mut out),
            "mlr.matmul_grad" => lower_matmul_grad(op, used, &mut out),
            "mlr.add_bias_grad" => lower_add_bias_grad(op, used, &mut out),
            "mlr.relu_grad" => lower_relu_grad(op, used, &mut out, "mlr_relu_grad"),
            "mlr.gelu_grad" => lower_relu_grad(op, used, &mut out, "mlr_gelu_grad"),
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

    /// A NUL terminated string in a global buffer, returned as a pointer.
    fn string_ptr(
        &mut self,
        globals: &mut Vec<crate::ir::Global>,
        counter: &mut usize,
        text: &str,
        out: &mut Vec<Op>,
    ) -> Value {
        let mut bytes = text.as_bytes().to_vec();
        bytes.push(0);
        let name = format!("__mlr_str{}", *counter);
        *counter += 1;
        let ty = Type::MemRef {
            shape: vec![Dim::Static(bytes.len() as i64)],
            elem: Box::new(Type::Scalar(crate::ir::ScalarKind::I8)),
        };
        globals.push(crate::ir::Global {
            name: name.clone(),
            ty: ty.clone(),
            value: Attr::DenseI8(bytes),
        });
        let mut get = Op::new("memref.get_global");
        get.attrs.push(("name".into(), Attr::Str(name)));
        let g = self.b.fresh_value(ty);
        get.results.push(g.clone());
        out.push(get);
        let p = self.pointer(&g);
        let r = p.results[0].clone();
        out.push(p);
        r
    }
}

fn lower_matmul(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_matmul");
    let a = op.operands[0].clone();
    let b = op.operands[1].clone();
    let result = op.results[0].clone();
    let da = dims(&a.ty);
    let db = dims(&b.ty);
    // A vector is a matrix with a single row (or column).
    let (m, k, n) = match (da.len(), db.len()) {
        (1, 2) => (1, da[0], db[1]),
        (2, 2) => (da[0], da[1], db[1]),
        (2, 1) => (da[0], da[1], 1),
        _ => (da[0], da[1], db[0]),
    };
    let mut l = Lowerer { b: Builder::new() };
    let out_ty = match &result.ty {
        Type::MemRef { shape, .. } => Type::MemRef {
            shape: shape.clone(),
            elem: Box::new(Type::Scalar(crate::ir::ScalarKind::F32)),
        },
        _ => Type::MemRef {
            shape: if da.len() == 2 && db.len() == 1 {
                vec![Dim::Static(m)]
            } else {
                vec![Dim::Static(m), Dim::Static(n)]
            },
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
    // The row keeps the element type of the tensor: the rows of a table of
    // indices are indices.
    let out_ty = match &result.ty {
        Type::MemRef { .. } => result.ty.clone(),
        t => t.clone(),
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

/// The row of a table that belongs to an index.
fn lower_embed(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_embed");
    let table = op.operands[0].clone();
    let id = op.operands[1].clone();
    let result = op.results[0].clone();
    let dt = dims(&table.ty);
    let (rows, width) = (dt[0], dt[1]);
    let mut l = Lowerer { b: Builder::new() };
    let out_ty = Type::MemRef {
        shape: vec![Dim::Static(width)],
        elem: Box::new(Type::Scalar(crate::ir::ScalarKind::F32)),
    };
    out.push(l.alloc(out_ty, &result));
    let pt = l.pointer(&table);
    let pout = l.pointer(&result);
    let crows = l.const_index(rows);
    let cwidth = l.const_index(width);
    out.push(pt.clone());
    out.push(pout.clone());
    out.push(crows.clone());
    out.push(cwidth.clone());
    let mut call = Op::new("func.call");
    call
        .attrs
        .push(("callee".into(), Attr::Str("mlr_embed".into())));
    call.operands = vec![
        pt.results[0].clone(),
        id,
        pout.results[0].clone(),
        crows.results[0].clone(),
        cwidth.results[0].clone(),
    ];
    out.push(call);
}

/// The rows of a table that belong to a tensor of indices.
fn lower_embed_seq(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_embed_seq");
    let table = op.operands[0].clone();
    let ids = op.operands[1].clone();
    let result = op.results[0].clone();
    let dt = dims(&table.ty);
    let di = dims(&ids.ty);
    let (rows, width, count) = (dt[0], dt[1], di[0]);
    let mut l = Lowerer { b: Builder::new() };
    let out_ty = Type::MemRef {
        shape: vec![Dim::Static(count), Dim::Static(width)],
        elem: Box::new(Type::Scalar(crate::ir::ScalarKind::F32)),
    };
    out.push(l.alloc(out_ty, &result));
    let pt = l.pointer(&table);
    let pi = l.pointer(&ids);
    let pout = l.pointer(&result);
    let crows = l.const_index(rows);
    let cwidth = l.const_index(width);
    let ccount = l.const_index(count);
    out.push(pt.clone());
    out.push(pi.clone());
    out.push(pout.clone());
    out.push(crows.clone());
    out.push(cwidth.clone());
    out.push(ccount.clone());
    let mut call = Op::new("func.call");
    call
        .attrs
        .push(("callee".into(), Attr::Str("mlr_embed_seq".into())));
    call.operands = vec![
        pt.results[0].clone(),
        pi.results[0].clone(),
        pout.results[0].clone(),
        crows.results[0].clone(),
        cwidth.results[0].clone(),
        ccount.results[0].clone(),
    ];
    out.push(call);
}

/// Every row of a matrix is normalized.
fn lower_layernorm(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_layernorm");
    let x = op.operands[0].clone();
    let gamma = op.operands[1].clone();
    let beta = op.operands[2].clone();
    let result = op.results[0].clone();
    let dx = dims(&x.ty);
    let (rows, width) = (dx[0], dx[1]);
    let mut l = Lowerer { b: Builder::new() };
    let out_ty = match &result.ty {
        Type::MemRef { shape, .. } => Type::MemRef {
            shape: shape.clone(),
            elem: Box::new(Type::Scalar(crate::ir::ScalarKind::F32)),
        },
        _ => Type::MemRef {
            shape: vec![Dim::Static(rows), Dim::Static(width)],
            elem: Box::new(Type::Scalar(crate::ir::ScalarKind::F32)),
        },
    };
    out.push(l.alloc(out_ty, &result));
    let px = l.pointer(&x);
    let pg = l.pointer(&gamma);
    let pb = l.pointer(&beta);
    let pout = l.pointer(&result);
    let crows = l.const_index(rows);
    let cwidth = l.const_index(width);
    out.push(px.clone());
    out.push(pg.clone());
    out.push(pb.clone());
    out.push(pout.clone());
    out.push(crows.clone());
    out.push(cwidth.clone());
    let mut call = Op::new("func.call");
    call
        .attrs
        .push(("callee".into(), Attr::Str("mlr_layernorm".into())));
    call.operands = vec![
        px.results[0].clone(),
        pg.results[0].clone(),
        pb.results[0].clone(),
        pout.results[0].clone(),
        crows.results[0].clone(),
        cwidth.results[0].clone(),
    ];
    out.push(call);
}

/// The softmax of the rows of a matrix, where the columns beyond the length
/// are masked out.
fn lower_softmax_causal(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_softmax_causal");
    let scores = op.operands[0].clone();
    let length = op.operands[1].clone();
    let result = op.results[0].clone();
    let ds = dims(&scores.ty);
    let size = ds[0];
    let mut l = Lowerer { b: Builder::new() };
    let out_ty = match &result.ty {
        Type::MemRef { shape, .. } => Type::MemRef {
            shape: shape.clone(),
            elem: Box::new(Type::Scalar(crate::ir::ScalarKind::F32)),
        },
        _ => Type::MemRef {
            shape: vec![Dim::Static(size), Dim::Static(size)],
            elem: Box::new(Type::Scalar(crate::ir::ScalarKind::F32)),
        },
    };
    out.push(l.alloc(out_ty, &result));
    let ps = l.pointer(&scores);
    let pout = l.pointer(&result);
    let csize = l.const_index(size);
    out.push(ps.clone());
    out.push(pout.clone());
    out.push(csize.clone());
    let mut call = Op::new("func.call");
    call
        .attrs
        .push(("callee".into(), Attr::Str("mlr_softmax_causal".into())));
    call.operands = vec![
        ps.results[0].clone(),
        length,
        pout.results[0].clone(),
        csize.results[0].clone(),
    ];
    out.push(call);
}

/// A token is drawn from the distribution of the logits.
fn lower_sample(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_sample");
    let logits = op.operands[0].clone();
    let temperature = op.operands[1].clone();
    let seed = op.operands[2].clone();
    let result = op.results[0].clone();
    let n = count(&logits.ty);
    let mut l = Lowerer { b: Builder::new() };
    let px = l.pointer(&logits);
    let cn = l.const_index(n);
    out.push(px.clone());
    out.push(cn.clone());
    let mut call = Op::new("func.call");
    call
        .attrs
        .push(("callee".into(), Attr::Str("mlr_sample".into())));
    call.operands = vec![
        px.results[0].clone(),
        temperature,
        seed,
        cn.results[0].clone(),
    ];
    call.results.push(result);
    out.push(call);
}

/// The transpose of a matrix.
fn lower_transpose(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_transpose");
    let x = op.operands[0].clone();
    let result = op.results[0].clone();
    let dx = dims(&x.ty);
    let (rows, width) = (dx[0], dx[1]);
    let mut l = Lowerer { b: Builder::new() };
    let out_ty = Type::MemRef {
        shape: vec![Dim::Static(width), Dim::Static(rows)],
        elem: Box::new(Type::Scalar(crate::ir::ScalarKind::F32)),
    };
    out.push(l.alloc(out_ty, &result));
    let px = l.pointer(&x);
    let pout = l.pointer(&result);
    let crows = l.const_index(rows);
    let cwidth = l.const_index(width);
    out.push(px.clone());
    out.push(pout.clone());
    out.push(crows.clone());
    out.push(cwidth.clone());
    let mut call = Op::new("func.call");
    call
        .attrs
        .push(("callee".into(), Attr::Str("mlr_transpose".into())));
    call.operands = vec![
        px.results[0].clone(),
        pout.results[0].clone(),
        crows.results[0].clone(),
        cwidth.results[0].clone(),
    ];
    out.push(call);
}

/// The element kind of a tensor, as the runtime names it.
fn elem_kind(ty: &Type) -> i64 {
    let elem = match ty {
        Type::MemRef { elem, .. } | Type::Tensor { elem, .. } => &**elem,
        _ => return 0,
    };
    match elem {
        Type::Scalar(crate::ir::ScalarKind::I32) => 1,
        Type::Scalar(crate::ir::ScalarKind::I64) => 2,
        _ => 0,
    }
}

/// Tensors are written into a PyTorch checkpoint. The runtime collects the
/// tensors between `mlr_save_pt_begin` and `mlr_save_pt_end` and writes the
/// archive when the checkpoint is closed.
fn lower_save_pt(
    op: Op,
    used: &mut Vec<&'static str>,
    out: &mut Vec<Op>,
    globals: &mut Vec<crate::ir::Global>,
    counter: &mut usize,
) {
    used.push("mlr_save_pt_begin");
    used.push("mlr_save_pt_add");
    used.push("mlr_save_pt_end");
    let path = match op.attr("path") {
        Some(Attr::Str(s)) => s.clone(),
        _ => String::new(),
    };
    let keys = match op.attr("keys") {
        Some(Attr::StrList(k)) => k.clone(),
        _ => Vec::new(),
    };
    let mut l = Lowerer { b: Builder::new() };

    let pp = l.string_ptr(globals, counter, &path, out);
    let mut begin = Op::new("func.call");
    begin
        .attrs
        .push(("callee".into(), Attr::Str("mlr_save_pt_begin".into())));
    begin.operands = vec![pp];
    out.push(begin);

    for (tensor, key) in op.operands.iter().zip(keys.iter()) {
        let kp = l.string_ptr(globals, counter, key, out);
        let dp = l.pointer(tensor);
        let n = count(&tensor.ty);
        let ds = dims(&tensor.ty);
        let ndim = ds.len() as i64;
        let d = |i: usize| ds.get(i).copied().unwrap_or(1);
        let ccount = l.const_index(n);
        let cndim = l.const_index(ndim);
        let c0 = l.const_index(d(0));
        let c1 = l.const_index(d(1));
        let c2 = l.const_index(d(2));
        let c3 = l.const_index(d(3));
        let ce = l.const_index(elem_kind(&tensor.ty));
        out.push(dp.clone());
        out.push(ccount.clone());
        out.push(cndim.clone());
        out.push(c0.clone());
        out.push(c1.clone());
        out.push(c2.clone());
        out.push(c3.clone());
        out.push(ce.clone());
        let mut call = Op::new("func.call");
        call.attrs
            .push(("callee".into(), Attr::Str("mlr_save_pt_add".into())));
        call.operands = vec![
            kp,
            dp.results[0].clone(),
            ccount.results[0].clone(),
            cndim.results[0].clone(),
            c0.results[0].clone(),
            c1.results[0].clone(),
            c2.results[0].clone(),
            c3.results[0].clone(),
            ce.results[0].clone(),
        ];
        out.push(call);
    }

    let mut end = Op::new("func.call");
    end.attrs
        .push(("callee".into(), Attr::Str("mlr_save_pt_end".into())));
    out.push(end);
}

/// The mean cross entropy of the rows: (logits, labels) -> f32.
fn lower_cross_entropy(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_cross_entropy");
    let logits = op.operands[0].clone();
    let labels = op.operands[1].clone();
    let result = op.results[0].clone();
    let dl = dims(&logits.ty);
    // A vector of logits is a single row.
    let (rows, vocab) = if dl.len() == 1 { (1, dl[0]) } else { (dl[0], dl[1]) };
    let mut l = Lowerer { b: Builder::new() };
    let pl = l.pointer(&logits);
    let py = l.pointer(&labels);
    let crows = l.const_index(rows);
    let cvocab = l.const_index(vocab);
    out.push(pl.clone());
    out.push(py.clone());
    out.push(crows.clone());
    out.push(cvocab.clone());
    let mut call = Op::new("func.call");
    call.attrs
        .push(("callee".into(), Attr::Str("mlr_cross_entropy".into())));
    call.operands = vec![
        pl.results[0].clone(),
        py.results[0].clone(),
        crows.results[0].clone(),
        cvocab.results[0].clone(),
    ];
    call.results.push(result);
    out.push(call);
}

/// The softmax of the rows: (x) -> y of the same shape.
fn lower_softmax(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_softmax");
    let x = op.operands[0].clone();
    let result = op.results[0].clone();
    let d = dims(&x.ty);
    let (rows, cols) = (d[0], d[1]);
    let mut l = Lowerer { b: Builder::new() };
    let out_ty = match &result.ty {
        Type::MemRef { .. } => result.ty.clone(),
        t => t.clone(),
    };
    out.push(l.alloc(out_ty, &result));
    let px = l.pointer(&x);
    let pout = l.pointer(&result);
    let crows = l.const_index(rows);
    let ccols = l.const_index(cols);
    out.push(px.clone());
    out.push(pout.clone());
    out.push(crows.clone());
    out.push(ccols.clone());
    let mut call = Op::new("func.call");
    call.attrs
        .push(("callee".into(), Attr::Str("mlr_softmax".into())));
    call.operands = vec![
        px.results[0].clone(),
        pout.results[0].clone(),
        crows.results[0].clone(),
        ccols.results[0].clone(),
    ];
    out.push(call);
}

/// A tensor filled with zeros: the count comes from the type.
fn lower_zeros(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_zero");
    let result = op.results[0].clone();
    let n = count(&result.ty);
    let mut l = Lowerer { b: Builder::new() };
    let out_ty = match &result.ty {
        Type::MemRef { .. } => result.ty.clone(),
        t => t.clone(),
    };
    out.push(l.alloc(out_ty, &result));
    let p = l.pointer(&result);
    let cn = l.const_index(n);
    out.push(p.clone());
    out.push(cn.clone());
    let mut call = Op::new("func.call");
    call.attrs.push(("callee".into(), Attr::Str("mlr_zero".into())));
    call.operands = vec![p.results[0].clone(), cn.results[0].clone()];
    out.push(call);
}

/// The element wise sum of two tensors.
fn lower_add_elem(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_add_elem");
    let a = op.operands[0].clone();
    let b = op.operands[1].clone();
    let result = op.results[0].clone();
    let n = count(&result.ty);
    let mut l = Lowerer { b: Builder::new() };
    let out_ty = match &result.ty {
        Type::MemRef { .. } => result.ty.clone(),
        t => t.clone(),
    };
    out.push(l.alloc(out_ty, &result));
    let pa = l.pointer(&a);
    let pb = l.pointer(&b);
    let pout = l.pointer(&result);
    let cn = l.const_index(n);
    out.push(pa.clone());
    out.push(pb.clone());
    out.push(pout.clone());
    out.push(cn.clone());
    let mut call = Op::new("func.call");
    call
        .attrs
        .push(("callee".into(), Attr::Str("mlr_add_elem".into())));
    call.operands = vec![
        pa.results[0].clone(),
        pb.results[0].clone(),
        pout.results[0].clone(),
        cn.results[0].clone(),
    ];
    out.push(call);
}

/// The gradients of a matrix multiplication:
/// (x, w, dy) -> (dx, dw).
fn lower_matmul_grad(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_matmul_grad");
    let x = op.operands[0].clone();
    let w = op.operands[1].clone();
    let dy = op.operands[2].clone();
    let dx = op.results[0].clone();
    let dw = op.results[1].clone();
    let dxr = dims(&x.ty);
    let dwr = dims(&w.ty);
    // A vector is a matrix with a single row.
    let (m, k, n) = if dxr.len() == 1 {
        (1, dxr[0], dwr[1])
    } else {
        (dxr[0], dxr[1], dwr[1])
    };
    let mut l = Lowerer { b: Builder::new() };
    let dx_ty = match &dx.ty {
        Type::MemRef { .. } => dx.ty.clone(),
        t => t.clone(),
    };
    let dw_ty = match &dw.ty {
        Type::MemRef { .. } => dw.ty.clone(),
        t => t.clone(),
    };
    out.push(l.alloc(dx_ty, &dx));
    out.push(l.alloc(dw_ty, &dw));
    let px = l.pointer(&x);
    let pw = l.pointer(&w);
    let pdy = l.pointer(&dy);
    let pdx = l.pointer(&dx);
    let pdw = l.pointer(&dw);
    let cm = l.const_index(m);
    let ck = l.const_index(k);
    let cn = l.const_index(n);
    out.push(px.clone());
    out.push(pw.clone());
    out.push(pdy.clone());
    out.push(pdx.clone());
    out.push(pdw.clone());
    out.push(cm.clone());
    out.push(ck.clone());
    out.push(cn.clone());
    let mut call = Op::new("func.call");
    call
        .attrs
        .push(("callee".into(), Attr::Str("mlr_matmul_grad".into())));
    call.operands = vec![
        px.results[0].clone(),
        pw.results[0].clone(),
        pdy.results[0].clone(),
        pdx.results[0].clone(),
        pdw.results[0].clone(),
        cm.results[0].clone(),
        ck.results[0].clone(),
        cn.results[0].clone(),
    ];
    out.push(call);
}

/// The gradients of a bias: (dy) -> (dm, db).
fn lower_add_bias_grad(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_add_bias_grad");
    let dy = op.operands[0].clone();
    let dm = op.results[0].clone();
    let db = op.results[1].clone();
    let d = dims(&dy.ty);
    let (rows, cols) = if d.len() == 1 { (1, d[0]) } else { (d[0], d[1]) };
    let mut l = Lowerer { b: Builder::new() };
    let dm_ty = match &dm.ty {
        Type::MemRef { .. } => dm.ty.clone(),
        t => t.clone(),
    };
    let db_ty = match &db.ty {
        Type::MemRef { .. } => db.ty.clone(),
        t => t.clone(),
    };
    out.push(l.alloc(dm_ty, &dm));
    out.push(l.alloc(db_ty, &db));
    let pdy = l.pointer(&dy);
    let pdm = l.pointer(&dm);
    let pdb = l.pointer(&db);
    let crows = l.const_index(rows);
    let ccols = l.const_index(cols);
    out.push(pdy.clone());
    out.push(pdm.clone());
    out.push(pdb.clone());
    out.push(crows.clone());
    out.push(ccols.clone());
    let mut call = Op::new("func.call");
    call
        .attrs
        .push(("callee".into(), Attr::Str("mlr_add_bias_grad".into())));
    call.operands = vec![
        pdy.results[0].clone(),
        pdm.results[0].clone(),
        pdb.results[0].clone(),
        crows.results[0].clone(),
        ccols.results[0].clone(),
    ];
    out.push(call);
}

/// An element wise gradient: (x, dy) -> dx.
fn lower_relu_grad(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>, name: &'static str) {
    used.push(name);
    let x = op.operands[0].clone();
    let dy = op.operands[1].clone();
    let dx = op.results[0].clone();
    let n = count(&x.ty);
    let mut l = Lowerer { b: Builder::new() };
    let dx_ty = match &dx.ty {
        Type::MemRef { .. } => dx.ty.clone(),
        t => t.clone(),
    };
    out.push(l.alloc(dx_ty, &dx));
    let px = l.pointer(&x);
    let pdy = l.pointer(&dy);
    let pdx = l.pointer(&dx);
    let cn = l.const_index(n);
    out.push(px.clone());
    out.push(pdy.clone());
    out.push(pdx.clone());
    out.push(cn.clone());
    let mut call = Op::new("func.call");
    call.attrs.push(("callee".into(), Attr::Str(name.to_string())));
    call.operands = vec![
        px.results[0].clone(),
        pdy.results[0].clone(),
        pdx.results[0].clone(),
        cn.results[0].clone(),
    ];
    out.push(call);
}

/// The gradient of the cross entropy: (logits, labels, scale) -> dlogits.
fn lower_cross_entropy_grad(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_cross_entropy_grad");
    let logits = op.operands[0].clone();
    let labels = op.operands[1].clone();
    let scale = op.operands[2].clone();
    let result = op.results[0].clone();
    let dl = dims(&logits.ty);
    // A vector of logits is a single row.
    let (rows, vocab) = if dl.len() == 1 { (1, dl[0]) } else { (dl[0], dl[1]) };
    let mut l = Lowerer { b: Builder::new() };
    let out_ty = match &result.ty {
        Type::MemRef { .. } => result.ty.clone(),
        t => t.clone(),
    };
    out.push(l.alloc(out_ty, &result));
    let pl = l.pointer(&logits);
    let py = l.pointer(&labels);
    let pout = l.pointer(&result);
    let crows = l.const_index(rows);
    let cvocab = l.const_index(vocab);
    out.push(pl.clone());
    out.push(py.clone());
    out.push(pout.clone());
    out.push(crows.clone());
    out.push(cvocab.clone());
    let mut call = Op::new("func.call");
    call.attrs
        .push(("callee".into(), Attr::Str("mlr_cross_entropy_grad".into())));
    call.operands = vec![
        pl.results[0].clone(),
        py.results[0].clone(),
        scale,
        pout.results[0].clone(),
        crows.results[0].clone(),
        cvocab.results[0].clone(),
    ];
    out.push(call);
}

/// The sum of the elements of a tensor: (x) -> f32.
fn lower_sum(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_sum");
    let x = op.operands[0].clone();
    let result = op.results[0].clone();
    let n = count(&x.ty);
    let mut l = Lowerer { b: Builder::new() };
    let px = l.pointer(&x);
    let cn = l.const_index(n);
    out.push(px.clone());
    out.push(cn.clone());
    let mut call = Op::new("func.call");
    call.attrs.push(("callee".into(), Attr::Str("mlr_sum".into())));
    call.operands = vec![px.results[0].clone(), cn.results[0].clone()];
    call.results.push(result);
    out.push(call);
}

/// The gradient of the sum: (x, dy) -> dx filled with dy.
fn lower_sum_grad(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_sum_grad");
    let x = op.operands[0].clone();
    let dy = op.operands[1].clone();
    let result = op.results[0].clone();
    let n = count(&x.ty);
    let mut l = Lowerer { b: Builder::new() };
    let out_ty = match &result.ty {
        Type::MemRef { .. } => result.ty.clone(),
        t => t.clone(),
    };
    out.push(l.alloc(out_ty, &result));
    let px = l.pointer(&x);
    let pout = l.pointer(&result);
    let cn = l.const_index(n);
    out.push(px.clone());
    out.push(pout.clone());
    out.push(cn.clone());
    let mut call = Op::new("func.call");
    call.attrs.push(("callee".into(), Attr::Str("mlr_sum_grad".into())));
    call.operands = vec![
        px.results[0].clone(),
        dy,
        pout.results[0].clone(),
        cn.results[0].clone(),
    ];
    out.push(call);
}

/// A copy, used as the gradient of the flatten.
fn lower_copy_grad(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>, name: &'static str) {
    used.push(name);
    let x = op.operands[0].clone();
    let result = op.results[0].clone();
    let n = count(&result.ty);
    let mut l = Lowerer { b: Builder::new() };
    let out_ty = match &result.ty {
        Type::MemRef { .. } => result.ty.clone(),
        t => t.clone(),
    };
    out.push(l.alloc(out_ty, &result));
    let px = l.pointer(&x);
    let pout = l.pointer(&result);
    let cn = l.const_index(n);
    out.push(px.clone());
    out.push(pout.clone());
    out.push(cn.clone());
    let mut call = Op::new("func.call");
    call.attrs.push(("callee".into(), Attr::Str(name.to_string())));
    call.operands = vec![
        px.results[0].clone(),
        pout.results[0].clone(),
        cn.results[0].clone(),
    ];
    out.push(call);
}

/// The gradients of the convolution: (x, w, dy) -> (dx, dw, db).
fn lower_conv2d_grad(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_conv2d_grad");
    let x = op.operands[0].clone();
    let w = op.operands[1].clone();
    let dy = op.operands[2].clone();
    let dx = op.results[0].clone();
    let dw = op.results[1].clone();
    let db = op.results[2].clone();
    let dxr = dims(&x.ty);
    let dwr = dims(&w.ty);
    let (h, width, o, k) = if dxr.len() == 2 {
        (dxr[0], dxr[1], dwr[0], dwr[2])
    } else {
        (dxr[1], dxr[2], dwr[0], dwr[2])
    };
    let mut l = Lowerer { b: Builder::new() };
    let mut alloc_of = |v: &Value| -> Op {
        let ty = match &v.ty {
            Type::MemRef { .. } => v.ty.clone(),
            t => t.clone(),
        };
        l.alloc(ty, v)
    };
    out.push(alloc_of(&dx));
    out.push(alloc_of(&dw));
    out.push(alloc_of(&db));
    let px = l.pointer(&x);
    let pw = l.pointer(&w);
    let pdy = l.pointer(&dy);
    let pdx = l.pointer(&dx);
    let pdw = l.pointer(&dw);
    let pdb = l.pointer(&db);
    let co = l.const_index(o);
    let ch = l.const_index(h);
    let cw = l.const_index(width);
    let ck = l.const_index(k);
    out.push(px.clone());
    out.push(pw.clone());
    out.push(pdy.clone());
    out.push(pdx.clone());
    out.push(pdw.clone());
    out.push(pdb.clone());
    out.push(co.clone());
    out.push(ch.clone());
    out.push(cw.clone());
    out.push(ck.clone());
    let mut call = Op::new("func.call");
    call
        .attrs
        .push(("callee".into(), Attr::Str("mlr_conv2d_grad".into())));
    call.operands = vec![
        px.results[0].clone(),
        pw.results[0].clone(),
        pdy.results[0].clone(),
        pdx.results[0].clone(),
        pdw.results[0].clone(),
        pdb.results[0].clone(),
        co.results[0].clone(),
        ch.results[0].clone(),
        cw.results[0].clone(),
        ck.results[0].clone(),
    ];
    out.push(call);
}

/// The gradient of the max pooling: (x, dy) -> dx; the window comes from the
/// `p0` attribute of the forward operation.
fn lower_maxpool_grad(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_maxpool2d_grad");
    let x = op.operands[0].clone();
    let dy = op.operands[1].clone();
    let result = op.results[0].clone();
    let k = match op.attr("p0") {
        Some(Attr::Int(v)) => *v,
        _ => 1,
    };
    let d = dims(&x.ty);
    let (o, h, width) = if d.len() == 2 {
        (1, d[0], d[1])
    } else {
        (d[0], d[1], d[2])
    };
    let mut l = Lowerer { b: Builder::new() };
    let out_ty = match &result.ty {
        Type::MemRef { .. } => result.ty.clone(),
        t => t.clone(),
    };
    out.push(l.alloc(out_ty, &result));
    let px = l.pointer(&x);
    let pdy = l.pointer(&dy);
    let pdx = l.pointer(&result);
    let co = l.const_index(o);
    let ch = l.const_index(h);
    let cw = l.const_index(width);
    let ck = l.const_index(k);
    out.push(px.clone());
    out.push(pdy.clone());
    out.push(pdx.clone());
    out.push(co.clone());
    out.push(ch.clone());
    out.push(cw.clone());
    out.push(ck.clone());
    let mut call = Op::new("func.call");
    call
        .attrs
        .push(("callee".into(), Attr::Str("mlr_maxpool2d_grad".into())));
    call.operands = vec![
        px.results[0].clone(),
        pdy.results[0].clone(),
        pdx.results[0].clone(),
        co.results[0].clone(),
        ch.results[0].clone(),
        cw.results[0].clone(),
        ck.results[0].clone(),
    ];
    out.push(call);
}

/// The gradient of the embedding of one token: (table, id, dy) -> dtable.
fn lower_embed_grad(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_embed_grad");
    let table = op.operands[0].clone();
    let id = op.operands[1].clone();
    let dy = op.operands[2].clone();
    let result = op.results[0].clone();
    let d = dims(&table.ty);
    let (rows, width) = (d[0], d[1]);
    let mut l = Lowerer { b: Builder::new() };
    let out_ty = match &result.ty {
        Type::MemRef { .. } => result.ty.clone(),
        t => t.clone(),
    };
    out.push(l.alloc(out_ty, &result));
    let pid = id;
    let pdy = l.pointer(&dy);
    let ptable = l.pointer(&result);
    let crows = l.const_index(rows);
    let cwidth = l.const_index(width);
    out.push(pdy.clone());
    out.push(ptable.clone());
    out.push(crows.clone());
    out.push(cwidth.clone());
    let mut call = Op::new("func.call");
    call
        .attrs
        .push(("callee".into(), Attr::Str("mlr_embed_grad".into())));
    call.operands = vec![
        pid,
        pdy.results[0].clone(),
        ptable.results[0].clone(),
        crows.results[0].clone(),
        cwidth.results[0].clone(),
    ];
    out.push(call);
}

/// The gradient of the embedding of a sequence:
/// (table, ids, dy) -> dtable.
fn lower_embed_seq_grad(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_embed_seq_grad");
    let table = op.operands[0].clone();
    let ids = op.operands[1].clone();
    let dy = op.operands[2].clone();
    let result = op.results[0].clone();
    let d = dims(&table.ty);
    let di = dims(&ids.ty);
    let (rows, width, count_v) = (d[0], d[1], di[0]);
    let mut l = Lowerer { b: Builder::new() };
    let out_ty = match &result.ty {
        Type::MemRef { .. } => result.ty.clone(),
        t => t.clone(),
    };
    out.push(l.alloc(out_ty, &result));
    let pids = l.pointer(&ids);
    let pdy = l.pointer(&dy);
    let ptable = l.pointer(&result);
    let crows = l.const_index(rows);
    let cwidth = l.const_index(width);
    let ccount = l.const_index(count_v);
    out.push(pids.clone());
    out.push(pdy.clone());
    out.push(ptable.clone());
    out.push(crows.clone());
    out.push(cwidth.clone());
    out.push(ccount.clone());
    let mut call = Op::new("func.call");
    call
        .attrs
        .push(("callee".into(), Attr::Str("mlr_embed_seq_grad".into())));
    call.operands = vec![
        pids.results[0].clone(),
        pdy.results[0].clone(),
        ptable.results[0].clone(),
        crows.results[0].clone(),
        cwidth.results[0].clone(),
        ccount.results[0].clone(),
    ];
    out.push(call);
}

/// The masked softmax of the rows: (x) -> y, square matrix.
fn lower_softmax_tri(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_softmax_tri");
    let x = op.operands[0].clone();
    let result = op.results[0].clone();
    let size = dims(&x.ty)[0];
    let mut l = Lowerer { b: Builder::new() };
    let out_ty = match &result.ty {
        Type::MemRef { .. } => result.ty.clone(),
        t => t.clone(),
    };
    out.push(l.alloc(out_ty, &result));
    let px = l.pointer(&x);
    let pout = l.pointer(&result);
    let csize = l.const_index(size);
    out.push(px.clone());
    out.push(pout.clone());
    out.push(csize.clone());
    let mut call = Op::new("func.call");
    call
        .attrs
        .push(("callee".into(), Attr::Str("mlr_softmax_tri".into())));
    call.operands = vec![
        px.results[0].clone(),
        pout.results[0].clone(),
        csize.results[0].clone(),
    ];
    out.push(call);
}

/// The gradient of a row softmax: (y, dy) -> dx, with y the output of the
/// forward pass.
fn lower_softmax_grad(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_softmax_grad");
    let y = op.operands[0].clone();
    let dy = op.operands[1].clone();
    let result = op.results[0].clone();
    let d = dims(&y.ty);
    let (rows, cols) = (d[0], d[1]);
    let mut l = Lowerer { b: Builder::new() };
    let out_ty = match &result.ty {
        Type::MemRef { .. } => result.ty.clone(),
        t => t.clone(),
    };
    out.push(l.alloc(out_ty, &result));
    let py = l.pointer(&y);
    let pdy = l.pointer(&dy);
    let pout = l.pointer(&result);
    let crows = l.const_index(rows);
    let ccols = l.const_index(cols);
    out.push(py.clone());
    out.push(pdy.clone());
    out.push(pout.clone());
    out.push(crows.clone());
    out.push(ccols.clone());
    let mut call = Op::new("func.call");
    call
        .attrs
        .push(("callee".into(), Attr::Str("mlr_softmax_grad".into())));
    call.operands = vec![
        py.results[0].clone(),
        pdy.results[0].clone(),
        pout.results[0].clone(),
        crows.results[0].clone(),
        ccols.results[0].clone(),
    ];
    out.push(call);
}

/// The gradient of the masked softmax of a square matrix: (y, dy) -> dx.
fn lower_softmax_tri_grad(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_softmax_tri_grad");
    let y = op.operands[0].clone();
    let dy = op.operands[1].clone();
    let result = op.results[0].clone();
    let size = dims(&y.ty)[0];
    let mut l = Lowerer { b: Builder::new() };
    let out_ty = match &result.ty {
        Type::MemRef { .. } => result.ty.clone(),
        t => t.clone(),
    };
    out.push(l.alloc(out_ty, &result));
    let py = l.pointer(&y);
    let pdy = l.pointer(&dy);
    let pout = l.pointer(&result);
    let csize = l.const_index(size);
    out.push(py.clone());
    out.push(pdy.clone());
    out.push(pout.clone());
    out.push(csize.clone());
    let mut call = Op::new("func.call");
    call
        .attrs
        .push(("callee".into(), Attr::Str("mlr_softmax_tri_grad".into())));
    call.operands = vec![
        py.results[0].clone(),
        pdy.results[0].clone(),
        pout.results[0].clone(),
        csize.results[0].clone(),
    ];
    out.push(call);
}

/// The gradients of the layer normalization: (x, g, dy) -> (dx, dg, db).
fn lower_layernorm_grad(op: Op, used: &mut Vec<&'static str>, out: &mut Vec<Op>) {
    used.push("mlr_layernorm_grad");
    #[allow(unused_mut)]
    let x = op.operands[0].clone();
    let g = op.operands[1].clone();
    let dy = op.operands[2].clone();
    let dx = op.results[0].clone();
    let dg = op.results[1].clone();
    let db = op.results[2].clone();
    let d = dims(&x.ty);
    let (rows, width) = (d[0], d[1]);
    let mut l = Lowerer { b: Builder::new() };
    let mut alloc_of = |v: &Value| -> Op {
        let ty = match &v.ty {
            Type::MemRef { .. } => v.ty.clone(),
            t => t.clone(),
        };
        l.alloc(ty, v)
    };
    out.push(alloc_of(&dx));
    out.push(alloc_of(&dg));
    out.push(alloc_of(&db));
    let px = l.pointer(&x);
    let pg = l.pointer(&g);
    let pdy = l.pointer(&dy);
    let pdx = l.pointer(&dx);
    let pdg = l.pointer(&dg);
    let pdb = l.pointer(&db);
    let crows = l.const_index(rows);
    let cwidth = l.const_index(width);
    out.push(px.clone());
    out.push(pg.clone());
    out.push(pdy.clone());
    out.push(pdx.clone());
    out.push(pdg.clone());
    out.push(pdb.clone());
    out.push(crows.clone());
    out.push(cwidth.clone());
    let mut call = Op::new("func.call");
    call
        .attrs
        .push(("callee".into(), Attr::Str("mlr_layernorm_grad".into())));
    call.operands = vec![
        px.results[0].clone(),
        pg.results[0].clone(),
        pdy.results[0].clone(),
        pdx.results[0].clone(),
        pdg.results[0].clone(),
        pdb.results[0].clone(),
        crows.results[0].clone(),
        cwidth.results[0].clone(),
    ];
    out.push(call);
}
