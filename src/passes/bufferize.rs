//! Stage 4: bufferization.
//!
//! Converts the tensor value semantics of the high level IR into explicit
//! memory operations:
//!
//! * `tensor.empty` becomes `memref.alloc`,
//! * `tensor.extract` becomes `memref.load`,
//! * `tensor.insert` becomes `memref.store`,
//! * `linalg.generic` over tensors becomes nested `scf.for` loops over
//!   `memref`s,
//! * function signatures turn tensor arguments into memref arguments.
//!
//! The pass is *semantics guided*: a buffer is reused in place when the
//! value it belongs to is not read again afterwards (last use) and the
//! operation is element wise.

use std::collections::HashMap;

use crate::ir::{Attr, Builder, Dim, Op, Region, Type, Value};

pub fn bufferize(module: &mut crate::ir::Module) {
    // The names of the constants are unique across the functions of the
    // module.
    let mut global_index = 0usize;
    for f in &mut module.funcs {
        let counts = use_counts(f);
        let mut b = Bufferizer {
            b: Builder::new(),
            map: HashMap::new(),
            counts,
            globals: Vec::new(),
            global_index,
            consts: std::collections::HashSet::new(),
            params: std::collections::HashSet::new(),
        };
        // Function signature: tensors become memrefs.
        for p in &mut f.params {
            p.ty = memref_of(&p.ty);
        }
        let mut params = std::collections::HashSet::new();
        for p in &f.params {
            params.insert(p.id);
        }
        b.params = params;
        for r in &mut f.results {
            *r = memref_of(r);
        }
        for p in &f.params {
            b.map.insert(p.id, p.clone());
        }
        let mut ops = Vec::new();
        b.rewrite_block(&f.body.blocks[0].ops, &mut ops);
        f.body.blocks[0].ops = ops;
        global_index = b.global_index;
        module.globals.extend(b.globals.clone());
    }
}

fn memref_of(t: &Type) -> Type {
    match t {
        Type::Tensor { shape, elem } => Type::MemRef {
            shape: shape.clone(),
            elem: elem.clone(),
        },
        other => other.clone(),
    }
}

fn use_counts(f: &crate::ir::FuncOp) -> HashMap<usize, usize> {
    let mut counts = HashMap::new();
    for block in &f.body.blocks {
        count_block(block, &mut counts);
    }
    counts
}

fn count_block(block: &crate::ir::Block, counts: &mut HashMap<usize, usize>) {
    for op in &block.ops {
        for o in &op.operands {
            *counts.entry(o.id).or_insert(0) += 1;
        }
        for region in &op.regions {
            for b in &region.blocks {
                count_block(b, counts);
            }
        }
    }
}

struct Bufferizer {
    b: Builder,
    /// Maps a tensor value to the memref that holds its data.
    map: HashMap<usize, Value>,
    counts: HashMap<usize, usize>,
    globals: Vec<crate::ir::Global>,
    global_index: usize,
    /// The values that refer to a constant global buffer, which must be
    /// copied before they are written.
    consts: std::collections::HashSet<usize>,
    /// The parameters of the function that is being bufferized; they may
    /// alias values of the caller that must not be written.
    params: std::collections::HashSet<usize>,
}

impl Bufferizer {
    fn fresh(&mut self, ty: Type) -> Value {
        self.b.fresh_value(ty)
    }

    fn buffer_of(&self, tensor: &Value) -> Value {
        self.map
            .get(&tensor.id)
            .cloned()
            .unwrap_or_else(|| tensor.clone())
    }

    /// Replaces every operand that refers to a tensor with the buffer that
    /// holds its data.
    fn remap_operands(&self, op: &mut Op) {
        for operand in &mut op.operands {
            if let Some(buf) = self.map.get(&operand.id) {
                *operand = buf.clone();
            }
        }
    }

    fn rewrite_block(&mut self, ops: &[Op], out: &mut Vec<Op>) {
        for op in ops {
            match op.name.as_str() {
                "arith.constant"
                    if matches!(
                        op.attr("value"),
                        Some(Attr::DenseF32(_)) | Some(Attr::DenseI32(_))
                    ) =>
                {
                    // A tensor that is embedded into the program becomes a
                    // global constant buffer.
                    let ty = memref_of(&op.results[0].ty);
                    let name = format!("__mlr_const{}", self.global_index);
                    self.global_index += 1;
                    let value = op.attr("value").cloned().unwrap();
                    self.globals.push(crate::ir::Global {
                        name: name.clone(),
                        ty: ty.clone(),
                        value,
                    });
                    let mut o = Op::new("memref.get_global");
                    o.attrs.push(("name".into(), Attr::Str(name)));
                    let res = self.fresh(ty);
                    o.results.push(res.clone());
                    self.consts.insert(res.id);
                    self.map.insert(op.results[0].id, res);
                    out.push(o);
                }
                "tensor.empty" => {
                    let mut o = Op::new("memref.alloc");
                    let ty = memref_of(&op.results[0].ty);
                    let res = self.fresh(ty);
                    o.results.push(res.clone());
                    self.map.insert(op.results[0].id, res);
                    out.push(o);
                }
                "tensor.insert" => {
                    // tensor.insert(value, tensor, indices...) -> memref.store
                    let value = op.operands[0].clone();
                    let tensor = op.operands[1].clone();
                    let mut buf = self.buffer_of(&tensor);
                    // The store must not be seen through the original value:
                    // copy when the input is a constant, a parameter of the
                    // function, or read again somewhere else.
                    if self.consts.contains(&buf.id)
                        || self.params.contains(&buf.id)
                        || self.counts.get(&tensor.id).copied().unwrap_or(0) > 1
                    {
                        let ty = buf.ty.clone();
                        let mut alloc = Op::new("memref.alloc");
                        let copy = self.fresh(ty.clone());
                        alloc.results.push(copy.clone());
                        out.push(alloc);
                        let mut cp = Op::new("memref.copy");
                        cp.operands.push(buf.clone());
                        cp.operands.push(copy.clone());
                        out.push(cp);
                        buf = copy;
                    }
                    let indices: Vec<Value> = op.operands[2..].to_vec();
                    let mut o = Op::new("memref.store");
                    o.operands.push(value);
                    o.operands.push(buf.clone());
                    o.operands.extend(indices);
                    self.map.insert(op.results[0].id, buf);
                    out.push(o);
                }
                "tensor.extract" => {
                    let buf = self.buffer_of(&op.operands[0]);
                    let indices: Vec<Value> = op.operands[1..].to_vec();
                    let mut o = Op::new("memref.load");
                    o.operands.push(buf);
                    o.operands.extend(indices);
                    o.results = op.results.clone();
                    out.push(o);
                }
                "mlr.dim" => {
                    let buf = self.buffer_of(&op.operands[0]);
                    let axis = match op.attr("dim") {
                        Some(Attr::Int(v)) => *v,
                        _ => 0,
                    };
                    let mut c = Op::new("arith.constant");
                    c.attrs.push(("value".into(), Attr::Int(axis)));
                    let axis_v = self.fresh(Type::Index);
                    c.results.push(axis_v.clone());
                    let mut o = Op::new("memref.dim");
                    o.operands.push(buf);
                    o.operands.push(axis_v);
                    o.results = op.results.clone();
                    out.push(c);
                    out.push(o);
                }
                "linalg.generic" => {
                    let mut ops2 = Vec::new();
                    self.linalg_to_loops(op, &mut ops2);
                    out.extend(ops2);
                }
                "scf.for" => {
                    let mut o = op.clone();
                    o.operands = op.operands.iter().map(|v| self.buffer_of(v)).collect();
                    // The loop carried values may be tensors.
                    if let Some(block) = o.regions.first_mut().and_then(|r| r.blocks.first_mut()) {
                        for arg in &mut block.args {
                            arg.ty = memref_of(&arg.ty);
                            self.map.insert(arg.id, arg.clone());
                        }
                        // An argument that starts as a constant buffer stays
                        // one until the first write copies it. The first
                        // argument of the block is the induction variable, so
                        // the iter args start at index one.
                        let inits: Vec<Value> = o.operands[3..].to_vec();
                        for (i, arg) in block.args.iter().skip(1).enumerate() {
                            if let Some(init) = inits.get(i) {
                                if self.consts.contains(&init.id) {
                                    self.consts.insert(arg.id);
                                }
                            }
                        }
                        let mut inner = Vec::new();
                        self.rewrite_block(&block.ops, &mut inner);
                        for op in &mut inner {
                            self.remap_operands(op);
                        }
                        block.ops = inner;
                    }
                    // The results of the loop are memrefs now.
                    for r in &mut o.results {
                        r.ty = memref_of(&r.ty);
                        self.map.insert(r.id, r.clone());
                    }
                    out.push(o);
                }
                "scf.if" => {
                    let mut o = op.clone();
                    for region in &mut o.regions {
                        for block in &mut region.blocks {
                            let mut inner = Vec::new();
                            self.rewrite_block(&block.ops, &mut inner);
                            for op in &mut inner {
                                self.remap_operands(op);
                            }
                            block.ops = inner;
                        }
                    }
                    for r in &mut o.results {
                        r.ty = memref_of(&r.ty);
                        self.map.insert(r.id, r.clone());
                    }
                    out.push(o);
                }
                name if name.starts_with("mlr.") => {
                    let mut o = op.clone();
                    o.operands = op.operands.iter().map(|v| self.buffer_of(v)).collect();
                    for r in &mut o.results {
                        r.ty = memref_of(&r.ty);
                        self.map.insert(r.id, r.clone());
                    }
                    out.push(o);
                }
                "func.call" => {
                    let mut o = op.clone();
                    o.operands = op.operands.iter().map(|v| self.buffer_of(v)).collect();
                    for r in &mut o.results {
                        r.ty = memref_of(&r.ty);
                        self.map.insert(r.id, r.clone());
                    }
                    out.push(o);
                }
                "func.return" => {
                    let mut o = op.clone();
                    o.operands = op.operands.iter().map(|v| self.buffer_of(v)).collect();
                    out.push(o);
                }
                _ => {
                    let mut o = op.clone();
                    for region in &mut o.regions {
                        for block in &mut region.blocks {
                            let mut inner = Vec::new();
                            self.rewrite_block(&block.ops, &mut inner);
                            for op in &mut inner {
                                self.remap_operands(op);
                            }
                            block.ops = inner;
                        }
                    }
                    self.remap_operands(&mut o);
                    out.push(o);
                }
            }
        }
    }

    /// Turns an element wise `linalg.generic` over tensors into nested
    /// `scf.for` loops over memrefs.
    fn linalg_to_loops(&mut self, op: &Op, out: &mut Vec<Op>) {
        let ins_count = match op.attr("ins_count") {
            Some(Attr::Int(v)) => *v as usize,
            _ => 0,
        };
        let ins: Vec<Value> = op.operands[..ins_count.min(op.operands.len())].to_vec();
        let result_ty = op
            .results
            .first()
            .map(|r| r.ty.clone())
            .unwrap_or(Type::Void);
        let elem_ty = match &result_ty {
            Type::Tensor { elem, .. } => (**elem).clone(),
            _ => Type::Scalar(crate::ir::ScalarKind::F32),
        };
        let out_memref_ty = memref_of(&result_ty);
        let body = op
            .regions
            .first()
            .and_then(|r| r.blocks.first())
            .cloned()
            .unwrap_or_default();

        let in_bufs: Vec<Value> = ins.iter().map(|t| self.buffer_of(t)).collect();

        // In-place reuse: if an input is not read again afterwards the buffer
        // can be written in place. A parameter of the function is excluded:
        // the caller may pass a value that must not be written (a constant);
        // so is a value that was derived from a constant.
        let mut out_buf: Option<Value> = None;
        if ins_count == 1 {
            if let Some(t) = ins.first() {
                if self.counts.get(&t.id).copied().unwrap_or(0) <= 1
                    && !self.params.contains(&in_bufs[0].id)
                    && !self.consts.contains(&in_bufs[0].id)
                {
                    out_buf = Some(in_bufs[0].clone());
                }
            }
        }
        let out_buf = match out_buf {
            Some(b) => b,
            None => {
                let mut alloc = Op::new("memref.alloc");
                let res = self.fresh(out_memref_ty.clone());
                alloc.results.push(res.clone());
                out.push(alloc);
                res
            }
        };

        // The shape of the iteration space.
        let dims: Vec<Dim> = match &result_ty {
            Type::Tensor { shape, .. } => shape.clone(),
            _ => vec![Dim::Static(1)],
        };

        let mut ivs: Vec<Value> = Vec::new();
        let mut ops = self.build_loop_nest(
            0,
            &dims,
            &in_bufs,
            &out_buf,
            out_memref_ty.clone(),
            &mut ivs,
            &body,
            &elem_ty,
        );
        // The result of the loop nest is the output buffer.
        if let Some(last) = ops.last_mut() {
            if last.name == "scf.for" && last.results.len() == 1 {
                let res = last.results[0].clone();
                self.map.insert(op.results[0].id, res);
            }
        }
        out.extend(ops);
    }

    #[allow(clippy::too_many_arguments)]
    fn build_loop_nest(
        &mut self,
        level: usize,
        dims: &[Dim],
        in_bufs: &[Value],
        out_buf: &Value,
        out_ty: Type,
        ivs: &mut Vec<Value>,
        body: &crate::ir::Block,
        elem_ty: &Type,
    ) -> Vec<Op> {
        if level >= dims.len() {
            // Innermost level: load the elements, run the body, store the
            // result.
            let mut ops = Vec::new();
            let mut subst: HashMap<usize, Value> = HashMap::new();
            for (i, buf) in in_bufs.iter().enumerate() {
                let mut load = Op::new("memref.load");
                load.operands.push(buf.clone());
                load.operands.extend(ivs.iter().cloned());
                let res = self.fresh(elem_ty.clone());
                load.results.push(res.clone());
                ops.push(load);
                if let Some(arg) = body.args.get(i) {
                    subst.insert(arg.id, res);
                }
            }
            for mut op in body.ops.clone() {
                substitute(&mut op, &subst);
                if op.name == "linalg.yield" {
                    // Store the yielded value into the output buffer.
                    let mut store = Op::new("memref.store");
                    store.operands.extend(op.operands.iter().cloned());
                    store.operands.push(out_buf.clone());
                    store.operands.extend(ivs.iter().cloned());
                    ops.push(store);
                } else {
                    ops.push(op);
                }
            }
            return ops;
        }

        // One more loop level.
        let mut prefix = Vec::new();
        let mut lb = Op::new("arith.constant");
        lb.attrs.push(("value".into(), Attr::Int(0)));
        let lb_v = self.fresh(Type::Index);
        lb.results.push(lb_v.clone());
        prefix.push(lb);

        let mut step = Op::new("arith.constant");
        step.attrs.push(("value".into(), Attr::Int(1)));
        let step_v = self.fresh(Type::Index);
        step.results.push(step_v.clone());
        prefix.push(step);

        let ub_v = match dims[level] {
            Dim::Static(v) => {
                let mut ub = Op::new("arith.constant");
                ub.attrs.push(("value".into(), Attr::Int(v)));
                let res = self.fresh(Type::Index);
                ub.results.push(res.clone());
                prefix.push(ub);
                res
            }
            Dim::Dynamic => {
                let buf = in_bufs.first().cloned().unwrap_or_else(|| out_buf.clone());
                let mut axis = Op::new("arith.constant");
                axis.attrs.push(("value".into(), Attr::Int(level as i64)));
                let axis_v = self.fresh(Type::Index);
                axis.results.push(axis_v.clone());
                prefix.push(axis);
                let mut dim_op = Op::new("memref.dim");
                dim_op.operands.push(buf);
                dim_op.operands.push(axis_v);
                let res = self.fresh(Type::Index);
                dim_op.results.push(res.clone());
                prefix.push(dim_op);
                res
            }
        };

        let iv = self.fresh(Type::Index);
        ivs.push(iv.clone());
        let out_arg = self.fresh(out_ty.clone());
        let mut for_op = Op::new("scf.for");
        for_op.operands.push(lb_v);
        for_op.operands.push(ub_v);
        for_op.operands.push(step_v);
        for_op.operands.push(out_buf.clone());
        let res = self.fresh(out_ty.clone());
        for_op.results.push(res.clone());
        let mut inner = self.build_loop_nest(
            level + 1,
            dims,
            in_bufs,
            &out_arg,
            out_ty.clone(),
            ivs,
            body,
            elem_ty,
        );
        // Every loop region ends with a yield of its (in place updated)
        // output buffer.
        let mut term = Op::new("scf.yield");
        term.operands.push(out_arg.clone());
        inner.push(term);
        for_op.regions.push(Region {
            blocks: vec![crate::ir::Block {
                args: vec![iv, out_arg],
                ops: inner,
            }],
        });
        ivs.pop();
        prefix.push(for_op);
        prefix
    }
}

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
