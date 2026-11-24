//! Stage 1: builds the high level IR from the typed AST.
//!
//! The mapping is purely structural: every region in the source becomes a
//! region in the IR, every `yield` becomes the region's terminator.

use std::collections::HashMap;

use crate::ast::{AttrValue, BinOp};
use crate::check::{CType, TBlock, TExpr, TFunc, TProgram, TStmt};
use crate::ir::{self, Attr, Builder, Dim, Loc, Op, Region, Type, Value};

/// Converts a checked type into an IR type.
pub fn ir_type(t: &CType) -> Type {
    match t {
        CType::Scalar(k) => Type::Scalar(*k),
        CType::Index => Type::Index,
        CType::Void => Type::Void,
        CType::Tensor { dims, elem } => {
            let shape = dims
                .iter()
                .map(|d| match d {
                    crate::check::CDim::Static(v) => Dim::Static(*v),
                    crate::check::CDim::Dynamic => Dim::Dynamic,
                })
                .collect();
            Type::Tensor {
                shape,
                elem: Box::new(ir_type(elem)),
            }
        }
    }
}

#[derive(Clone)]
pub struct FuncSig {
    #[allow(dead_code)]
    pub params: Vec<CType>,
    pub ret: Option<CType>,
}

pub struct IrBuilder<'a> {
    b: Builder,
    sigs: &'a HashMap<String, FuncSig>,
    file: String,
    scopes: Vec<HashMap<String, Value>>,
    /// The operation list of the region that is currently being built.
    stack: Vec<Vec<Op>>,
    /// The source location of the statement that is currently being built.
    cur_loc: Loc,
    /// Runtime functions that have to be declared in the module.
    pub runtime: Vec<String>,
    /// Errors that were collected while building the IR.
    pub errors: Vec<String>,
}

impl<'a> IrBuilder<'a> {
    pub fn new(sigs: &'a HashMap<String, FuncSig>, file: &str) -> Self {
        IrBuilder {
            b: Builder::new(),
            sigs,
            file: file.to_string(),
            scopes: Vec::new(),
            stack: Vec::new(),
            cur_loc: Loc::default(),
            runtime: Vec::new(),
            errors: Vec::new(),
        }
    }

    fn emit(&mut self, mut op: Op) {
        // Every operation inherits the location of the statement it belongs
        // to, so that diagnostics can point back into the source.
        if op.loc.file.is_empty() {
            op.loc = self.cur_loc.clone();
        }
        self.stack.last_mut().expect("open block").push(op);
    }

    fn fresh(&mut self, ty: Type) -> Value {
        self.b.fresh_value(ty)
    }

    fn lookup(&self, name: &str) -> Option<Value> {
        self.scopes.iter().rev().find_map(|s| s.get(name)).cloned()
    }

    fn define(&mut self, name: &str, v: Value) {
        self.scopes
            .last_mut()
            .expect("scope")
            .insert(name.to_string(), v);
    }

    fn ty(&self, t: &CType) -> Type {
        ir_type(t)
    }

    // -----------------------------------------------------------------
    // Functions
    // -----------------------------------------------------------------

    pub fn build_program(&mut self, prog: &TProgram) -> ir::Module {
        let mut module = ir::Module::default();
        for f in &prog.funcs {
            module.funcs.push(self.build_func(f));
        }
        for r in &self.runtime {
            let (arg, res) = match r.as_str() {
                "mlr_print_i32" => ("i32", vec![]),
                "mlr_print_i64" => ("i64", vec![]),
                "mlr_print_f32" => ("f32", vec![]),
                "mlr_print_f64" => ("f64", vec![]),
                _ => ("i1", vec![]),
            };
            module.decls.push(ir::FuncDecl {
                name: r.clone(),
                params: vec![Type::scalar(arg).unwrap()],
                results: res,
            });
        }
        module
    }

    fn build_func(&mut self, f: &TFunc) -> ir::FuncOp {
        self.scopes.clear();
        self.scopes.push(HashMap::new());
        let mut params = Vec::new();
        for (name, ty) in &f.params {
            let v = self.fresh(self.ty(ty));
            self.define(name, v.clone());
            params.push(v);
        }
        let results: Vec<Type> = f.ret.as_ref().map(|t| vec![self.ty(t)]).unwrap_or_default();
        let impure = body_is_impure(&f.body);
        let (mut ops, yields) = self.build_block(&f.body, results.len());
        // The body of a function is terminated by `func.return`, not by the
        // `scf.yield` of the region.
        if let Some(last) = ops.last() {
            if last.name == "scf.yield" {
                ops.pop();
            }
        }
        let mut term = Op::new("func.return");
        if results.len() == 1 {
            match yields.first() {
                Some(v) => term.operands.push(v.clone()),
                None => {
                    let v = self.fresh(results[0].clone());
                    term.operands.push(v);
                }
            }
        }
        ops.push(term);
        let attrs = vec![(
            "effects".to_string(),
            Attr::Str(if impure { "io".into() } else { "pure".into() }),
        )];
        ir::FuncOp {
            name: f.name.clone(),
            params: params.clone(),
            results,
            body: Region {
                blocks: vec![ir::Block { args: vec![], ops }],
            },
            attrs,
            loc: Loc::default(),
        }
    }

    // -----------------------------------------------------------------
    // Blocks
    // -----------------------------------------------------------------

    /// Builds the operations of a region; the last operation is the terminator
    /// that yields the region's values.
    fn build_block(&mut self, block: &TBlock, _results: usize) -> (Vec<Op>, Vec<Value>) {
        self.stack.push(Vec::new());
        self.scopes.push(HashMap::new());
        for stmt in &block.stmts {
            match stmt {
                TStmt::Let {
                    name,
                    ty,
                    value,
                    span,
                } => {
                    self.cur_loc = Loc::new(&self.file, span.line, span.col);
                    let v = self.expr(value);
                    let v = self.coerce_to(v, ty);
                    self.define(name, v);
                }
                TStmt::Effect(e, span) => {
                    self.cur_loc = Loc::new(&self.file, span.line, span.col);
                    self.expr(e);
                }
            }
        }
        let mut yielded = Vec::new();
        self.cur_loc = Loc::new(&self.file, block.span.line, block.span.col);
        if let Some(values) = &block.tail {
            for v in values {
                let val = self.expr(v);
                // A region that yields a value without one (a `print` call)
                // contributes nothing to the terminator.
                if val.ty != Type::Void {
                    yielded.push(val);
                }
            }
        }
        self.scopes.pop();
        let mut term = Op::new("scf.yield");
        term.operands = yielded.clone();
        self.emit(term);
        let ops = self.stack.pop().unwrap();
        (ops, yielded)
    }

    fn coerce_to(&mut self, v: Value, ty: &CType) -> Value {
        let want = self.ty(ty);
        if v.ty == want {
            return v;
        }
        if let Type::Scalar(k) = v.ty {
            if k.is_int() && want == Type::Index {
                let mut op = Op::new("arith.index_cast");
                op.operands.push(v.clone());
                let res = self.fresh(want.clone());
                op.results.push(res.clone());
                self.emit(op);
                return res;
            }
        }
        v
    }

    // -----------------------------------------------------------------
    // Expressions
    // -----------------------------------------------------------------

    fn expr(&mut self, e: &TExpr) -> Value {
        match e {
            TExpr::ConstInt(v, t) => {
                let ty = self.ty(t);
                let mut op = Op::new("arith.constant");
                op.attrs.push(("value".into(), Attr::Int(*v)));
                let res = self.fresh(ty);
                op.results.push(res.clone());
                self.emit(op);
                res
            }
            TExpr::ConstFloat(v, t) => {
                let ty = self.ty(t);
                let mut op = Op::new("arith.constant");
                op.attrs.push(("value".into(), Attr::Float(*v)));
                let res = self.fresh(ty);
                op.results.push(res.clone());
                self.emit(op);
                res
            }
            TExpr::ConstBool(b) => {
                let mut op = Op::new("arith.constant");
                op.attrs.push(("value".into(), Attr::Bool(*b)));
                let res = self.fresh(Type::Scalar(crate::ir::ScalarKind::I1));
                op.results.push(res.clone());
                self.emit(op);
                res
            }
            TExpr::Var(name, _) => self
                .lookup(name)
                .unwrap_or_else(|| panic!("unbound variable `{}`", name)),
            TExpr::Unary(_op, operand, ty) => {
                let v = self.expr(operand);
                let ity = self.ty(ty);
                let mut o;
                if ity.is_float() {
                    o = Op::new("arith.negf");
                    o.operands.push(v);
                } else {
                    let mut zero = Op::new("arith.constant");
                    zero.attrs.push(("value".into(), Attr::Int(0)));
                    let z = self.fresh(ity.clone());
                    zero.results.push(z.clone());
                    self.emit(zero);
                    o = Op::new("arith.subi");
                    o.operands.push(z);
                    o.operands.push(v);
                }
                let res = self.fresh(ity);
                o.results.push(res.clone());
                self.emit(o);
                res
            }
            TExpr::Binary(op, l, r, ty) => {
                let lv = self.expr(l);
                let rv = self.expr(r);
                let lty = lv.ty.clone();
                let ity = self.ty(ty);
                let name = match op {
                    BinOp::Add => {
                        if lty.is_float() {
                            "arith.addf"
                        } else {
                            "arith.addi"
                        }
                    }
                    BinOp::Sub => {
                        if lty.is_float() {
                            "arith.subf"
                        } else {
                            "arith.subi"
                        }
                    }
                    BinOp::Mul => {
                        if lty.is_float() {
                            "arith.mulf"
                        } else {
                            "arith.muli"
                        }
                    }
                    BinOp::Div => {
                        if lty.is_float() {
                            "arith.divf"
                        } else {
                            "arith.divsi"
                        }
                    }
                    BinOp::Rem => "arith.remsi",
                    BinOp::And => "arith.andi",
                    BinOp::Or => "arith.ori",
                    _ => "",
                };
                let mut o = Op::new(name);
                if matches!(
                    op,
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
                ) {
                    let float = lty.is_float();
                    o.name = if float {
                        "arith.cmpf".into()
                    } else {
                        "arith.cmpi".into()
                    };
                    let pred = match (op, float) {
                        (BinOp::Eq, false) => "eq",
                        (BinOp::Ne, false) => "ne",
                        (BinOp::Lt, false) => "slt",
                        (BinOp::Le, false) => "sle",
                        (BinOp::Gt, false) => "sgt",
                        (BinOp::Ge, false) => "sge",
                        (BinOp::Eq, true) => "oeq",
                        (BinOp::Ne, true) => "one",
                        (BinOp::Lt, true) => "olt",
                        (BinOp::Le, true) => "ole",
                        (BinOp::Gt, true) => "ogt",
                        (BinOp::Ge, true) => "oge",
                        _ => "eq",
                    };
                    o.attrs.push(("predicate".into(), Attr::Str(pred.into())));
                }
                o.operands.push(lv);
                o.operands.push(rv);
                let res = self.fresh(ity);
                o.results.push(res.clone());
                self.emit(o);
                res
            }
            TExpr::Call {
                name,
                type_args,
                args,
                ..
            } => self.call(name, type_args, args),
            TExpr::If {
                cond,
                then_block,
                else_block,
                ty,
            } => self.build_if(cond, then_block, else_block, ty),
            TExpr::For {
                var,
                lb,
                ub,
                step,
                iter,
                body,
                ..
            } => self.build_for(var, lb, ub, step, iter, body),
            TExpr::Map {
                inputs,
                attrs,
                params,
                body,
                ty,
            } => self.build_map(inputs, attrs, params, body, ty),
            TExpr::Reduce {
                input,
                init,
                attrs,
                params,
                body,
                ty,
            } => self.build_reduce(input, init, attrs, params, body, ty),
            TExpr::Domain {
                op,
                args,
                params,
                ty,
                ..
            } => {
                let mut o = Op::new(&format!("mlr.{}", op.name()));
                o.generic = true;
                for a in args {
                    o.operands.push(self.expr(a));
                }
                for (i, p) in params.iter().enumerate() {
                    o.attrs.push((format!("p{}", i), Attr::Int(*p)));
                }
                let res = self.fresh(self.ty(ty));
                o.results.push(res.clone());
                self.emit(o);
                res
            }
            TExpr::Load { path, ty } => {
                let ty = self.ty(ty);
                let count = match &ty {
                    Type::Tensor { shape, .. } => shape.iter().fold(1i64, |acc, d| {
                        acc * match d {
                            Dim::Static(v) => *v,
                            Dim::Dynamic => 1,
                        }
                    }),
                    _ => 0,
                };
                let elem = ty
                    .elem_type()
                    .unwrap_or(Type::Scalar(crate::ir::ScalarKind::F32));
                let bytes = match std::fs::read(path) {
                    Ok(b) => b,
                    Err(e) => {
                        self.errors.push(format!("cannot read `{}`: {}", path, e));
                        let res = self.fresh(ty);
                        return res;
                    }
                };
                let mut o = Op::new("arith.constant");
                match &elem {
                    Type::Scalar(crate::ir::ScalarKind::I32) => {
                        let mut values = Vec::with_capacity(bytes.len() / 4);
                        for chunk in bytes.as_chunks::<4>().0 {
                            values.push(i32::from_le_bytes(*chunk));
                        }
                        if values.len() as i64 != count {
                            self.errors.push(format!(
                                "`{}` contains {} elements but {} were expected",
                                path,
                                values.len(),
                                count
                            ));
                        }
                        o.attrs.push(("value".into(), Attr::DenseI32(values)));
                    }
                    _ => {
                        let mut values = Vec::with_capacity(bytes.len() / 4);
                        for chunk in bytes.as_chunks::<4>().0 {
                            values.push(f32::from_le_bytes(*chunk));
                        }
                        if values.len() as i64 != count {
                            self.errors.push(format!(
                                "`{}` contains {} elements but {} were expected",
                                path,
                                values.len(),
                                count
                            ));
                        }
                        o.attrs.push(("value".into(), Attr::DenseF32(values)));
                    }
                }
                let res = self.fresh(ty);
                o.results.push(res.clone());
                self.emit(o);
                res
            }
            TExpr::Cast { value, from, to } => {
                let v = self.expr(value);
                // A conversion between two numbers.
                let mut simple = |name: &str, v: Value, ty: &CType| -> Value {
                    let mut o = Op::new(name);
                    o.operands.push(v);
                    let res = self.fresh(self.ty(ty));
                    o.results.push(res.clone());
                    self.emit(o);
                    res
                };
                match (from, to) {
                    // An index becomes an integer and then a float.
                    (CType::Index, t) if t.is_float() => {
                        let i = simple(
                            "arith.index_cast",
                            v,
                            &CType::Scalar(crate::ir::ScalarKind::I64),
                        );
                        simple("arith.sitofp", i, to)
                    }
                    (CType::Index, _) => simple("arith.index_cast", v, to),
                    (_, CType::Index) => simple("arith.index_cast", v, to),
                    (f, t) if f.is_float() && t.is_float() => {
                        let name = if t.width() >= f.width() {
                            "arith.extf"
                        } else {
                            "arith.truncf"
                        };
                        simple(name, v, to)
                    }
                    (f, t) if f.is_float() && t.is_int() => simple("arith.fptosi", v, to),
                    _ => simple("arith.sitofp", v, to),
                }
            }
            TExpr::Extract {
                tensor,
                indices,
                ty,
            } => {
                let t = self.expr(tensor);
                let mut idx = Vec::new();
                for i in indices {
                    idx.push(self.expr(i));
                }
                let mut o = Op::new("tensor.extract");
                o.operands.push(t);
                o.operands.extend(idx);
                let res = self.fresh(self.ty(ty));
                o.results.push(res.clone());
                self.emit(o);
                res
            }
        }
    }

    fn call(&mut self, name: &str, type_args: &[CType], args: &[TExpr]) -> Value {
        match name {
            "print" => {
                let v = self.expr(&args[0]);
                let fname = match &v.ty {
                    Type::Scalar(k) => match k.name() {
                        "i32" => "mlr_print_i32",
                        "i64" => "mlr_print_i64",
                        "f32" => "mlr_print_f32",
                        "f64" => "mlr_print_f64",
                        _ => "mlr_print_i32",
                    },
                    _ => "mlr_print_i32",
                };
                if !self.runtime.contains(&fname.to_string()) {
                    self.runtime.push(fname.to_string());
                }
                let mut o = Op::new("func.call");
                o.attrs.push(("callee".into(), Attr::Str(fname.into())));
                o.operands.push(v);
                self.emit(o);
                return self.fresh(Type::Void);
            }
            "tensor.empty" | "tensor.zeros" => {
                let ty = self.ty(&type_args[0]);
                let mut o = Op::new("tensor.empty");
                let res = self.fresh(ty.clone());
                o.results.push(res.clone());
                self.emit(o);
                if name == "tensor.zeros" {
                    return self.fill_zero(res, ty);
                }
                return res;
            }
            _ => {}
        }
        let sig = self.sigs.get(name).cloned();
        let mut o = Op::new("func.call");
        o.attrs.push(("callee".into(), Attr::Str(name.into())));
        for a in args {
            o.operands.push(self.expr(a));
        }
        let ret = sig.and_then(|s| s.ret);
        match ret {
            Some(t) => {
                let res = self.fresh(self.ty(&t));
                o.results.push(res.clone());
                self.emit(o);
                res
            }
            None => {
                self.emit(o);
                self.fresh(Type::Void)
            }
        }
    }

    /// Fills a tensor with zeros using a structured loop.
    fn fill_zero(&mut self, tensor: Value, ty: Type) -> Value {
        let (shape, elem) = match &ty {
            Type::Tensor { shape, elem } => (shape.clone(), (**elem).clone()),
            _ => return tensor,
        };
        let mut zero = Op::new("arith.constant");
        zero.attrs.push(("value".into(), Attr::Float(0.0)));
        let z = self.fresh(elem.clone());
        zero.results.push(z.clone());
        self.emit(zero);

        let mut carried = tensor.clone();
        for dim in shape.iter() {
            let (lb, ub) = self.dim_bounds(dim);
            let step = self.index_const(1);
            let mut loop_op = Op::new("scf.for");
            let iv = self.fresh(Type::Index);
            let arg = self.fresh(ty.clone());
            loop_op.operands.push(lb);
            loop_op.operands.push(ub);
            loop_op.operands.push(step);
            loop_op.operands.push(carried.clone());
            let mut insert = Op::new("tensor.insert");
            insert.operands.push(z.clone());
            insert.operands.push(arg.clone());
            insert.operands.push(iv.clone());
            let inserted = self.fresh(ty.clone());
            insert.results.push(inserted.clone());
            let mut yield_op = Op::new("scf.yield");
            yield_op.operands.push(inserted);
            loop_op.regions.push(Region {
                blocks: vec![ir::Block {
                    args: vec![iv, arg],
                    ops: vec![insert, yield_op],
                }],
            });
            let res = self.fresh(ty.clone());
            loop_op.results.push(res.clone());
            self.emit(loop_op);
            carried = res;
        }
        carried
    }

    fn index_const(&mut self, v: i64) -> Value {
        let mut op = Op::new("arith.constant");
        op.attrs.push(("value".into(), Attr::Int(v)));
        let res = self.fresh(Type::Index);
        op.results.push(res.clone());
        self.emit(op);
        res
    }

    fn dim_bounds(&mut self, dim: &Dim) -> (Value, Value) {
        match dim {
            Dim::Static(v) => {
                let lb = self.index_const(0);
                let ub = self.index_const(*v);
                (lb, ub)
            }
            Dim::Dynamic => {
                // A dynamic dimension is bounded by the runtime size; the
                // bufferization stage replaces this with `memref.dim`.
                let lb = self.index_const(0);
                let mut op = Op::new("mlr.dim");
                op.attrs.push(("dim".into(), Attr::Int(-1)));
                let ub = self.fresh(Type::Index);
                op.results.push(ub.clone());
                self.emit(op);
                (lb, ub)
            }
        }
    }

    fn build_if(
        &mut self,
        cond: &TExpr,
        then_block: &TBlock,
        else_block: &Option<Box<TBlock>>,
        ty: &CType,
    ) -> Value {
        let c = self.expr(cond);
        let yields_value = !matches!(ty, CType::Void);
        let (then_ops, _) = self.build_block(then_block, if yields_value { 1 } else { 0 });
        let (else_ops, _) = match else_block {
            Some(b) => self.build_block(b, if yields_value { 1 } else { 0 }),
            None => (vec![Op::new("scf.yield")], vec![]),
        };
        let mut o = Op::new("scf.if");
        o.operands.push(c);
        let res = self.fresh(self.ty(ty));
        if yields_value {
            o.results.push(res.clone());
        }
        o.regions.push(Region {
            blocks: vec![ir::Block {
                args: vec![],
                ops: then_ops,
            }],
        });
        o.regions.push(Region {
            blocks: vec![ir::Block {
                args: vec![],
                ops: else_ops,
            }],
        });
        self.emit(o);
        res
    }

    fn build_for(
        &mut self,
        var: &str,
        lb: &TExpr,
        ub: &TExpr,
        step: &TExpr,
        iter: &[crate::check::TIter],
        body: &TBlock,
    ) -> Value {
        let lb_v = self.expr(lb);
        let lb = self.coerce_to(lb_v, &CType::Index);
        let ub_v = self.expr(ub);
        let ub = self.coerce_to(ub_v, &CType::Index);
        let step_v = self.expr(step);
        let step = self.coerce_to(step_v, &CType::Index);
        let mut o = Op::new("scf.for");
        o.operands.push(lb);
        o.operands.push(ub);
        o.operands.push(step);
        let iv = self.fresh(Type::Index);
        let mut iter_args = Vec::new();
        for it in iter {
            let init = self.expr(&it.init);
            o.operands.push(init);
            let arg = self.fresh(self.ty(&it.ty));
            iter_args.push((it.name.clone(), arg));
        }
        self.scopes.push(HashMap::new());
        self.scopes
            .last_mut()
            .unwrap()
            .insert(var.to_string(), iv.clone());
        for (name, arg) in &iter_args {
            self.scopes
                .last_mut()
                .unwrap()
                .insert(name.clone(), arg.clone());
        }
        let (body_ops, _) = self.build_block(body, iter.len());
        self.scopes.pop();

        let mut results = Vec::new();
        for it in iter {
            let res = self.fresh(self.ty(&it.ty));
            results.push(res.clone());
            o.results.push(res);
        }
        let mut args = vec![iv];
        args.extend(iter_args.iter().map(|(_, a)| a.clone()));
        o.regions.push(Region {
            blocks: vec![ir::Block {
                args,
                ops: body_ops,
            }],
        });
        self.emit(o);
        if results.is_empty() {
            self.fresh(Type::Void)
        } else {
            results[0].clone()
        }
    }

    fn build_map(
        &mut self,
        inputs: &[TExpr],
        attrs: &[(String, AttrValue)],
        params: &[String],
        body: &TBlock,
        ty: &CType,
    ) -> Value {
        let mut vals = Vec::new();
        for i in inputs {
            vals.push(self.expr(i));
        }
        let mut o = Op::new("mlr.map");
        o.generic = true;
        for (k, v) in attrs {
            o.attrs.push((k.clone(), attr_of(v)));
        }
        o.operands = vals.clone();
        self.scopes.push(HashMap::new());
        let mut args = Vec::new();
        for (i, p) in params.iter().enumerate() {
            let elem = match &vals[i].ty {
                Type::Tensor { elem, .. } => (**elem).clone(),
                _ => Type::Scalar(crate::ir::ScalarKind::F32),
            };
            let arg = self.fresh(elem);
            self.scopes
                .last_mut()
                .unwrap()
                .insert(p.clone(), arg.clone());
            args.push(arg);
        }
        let (body_ops, _) = self.build_block(body, 1);
        self.scopes.pop();
        let body_ops = body_ops
            .into_iter()
            .map(|mut op| {
                if op.name == "scf.yield" {
                    op.name = "mlr.yield".into();
                }
                op
            })
            .collect();
        o.regions.push(Region {
            blocks: vec![ir::Block {
                args,
                ops: body_ops,
            }],
        });
        let res = self.fresh(self.ty(ty));
        o.results.push(res.clone());
        self.emit(o);
        res
    }

    fn build_reduce(
        &mut self,
        input: &TExpr,
        init: &TExpr,
        attrs: &[(String, AttrValue)],
        params: &[String],
        body: &TBlock,
        ty: &CType,
    ) -> Value {
        let t = self.expr(input);
        let t_ty = t.ty.clone();
        let init_v = self.expr(init);
        let mut o = Op::new("mlr.reduce");
        o.generic = true;
        o.operands.push(t);
        o.operands.push(init_v);
        for (k, v) in attrs {
            o.attrs.push((k.clone(), attr_of(v)));
        }
        self.scopes.push(HashMap::new());
        let acc_ty = self.ty(ty);
        let elem_ty = match &t_ty {
            Type::Tensor { elem, .. } => (**elem).clone(),
            _ => acc_ty.clone(),
        };
        let acc = self.fresh(acc_ty.clone());
        let elem = self.fresh(elem_ty);
        self.scopes
            .last_mut()
            .unwrap()
            .insert(params[0].clone(), acc.clone());
        self.scopes
            .last_mut()
            .unwrap()
            .insert(params[1].clone(), elem.clone());
        let (body_ops, _) = self.build_block(body, 1);
        self.scopes.pop();
        let body_ops = body_ops
            .into_iter()
            .map(|mut op| {
                if op.name == "scf.yield" {
                    op.name = "mlr.yield".into();
                }
                op
            })
            .collect();
        o.regions.push(Region {
            blocks: vec![ir::Block {
                args: vec![acc, elem],
                ops: body_ops,
            }],
        });
        let res = self.fresh(acc_ty);
        o.results.push(res.clone());
        self.emit(o);
        res
    }
}

fn attr_of(v: &AttrValue) -> Attr {
    match v {
        AttrValue::Int(i) => Attr::Int(*i),
        AttrValue::Float(f) => Attr::Float(*f),
        AttrValue::Str(s) => Attr::Str(s.clone()),
        AttrValue::Bool(b) => Attr::Bool(*b),
    }
}

fn body_is_impure(block: &TBlock) -> bool {
    fn expr_impure(e: &TExpr) -> bool {
        match e {
            TExpr::Call { name, args, .. } => name == "print" || args.iter().any(expr_impure),
            TExpr::Unary(_, a, _) => expr_impure(a),
            TExpr::Binary(_, a, b, _) => expr_impure(a) || expr_impure(b),
            TExpr::If {
                cond,
                then_block,
                else_block,
                ..
            } => {
                expr_impure(cond)
                    || then_block.stmts.iter().any(stmt_impure)
                    || then_block
                        .tail
                        .as_ref()
                        .is_some_and(|v| v.iter().any(expr_impure))
                    || else_block.as_ref().is_some_and(|b| {
                        b.stmts.iter().any(stmt_impure)
                            || b.tail.as_ref().is_some_and(|v| v.iter().any(expr_impure))
                    })
            }
            TExpr::For {
                lb,
                ub,
                step,
                body,
                iter,
                ..
            } => {
                expr_impure(lb)
                    || expr_impure(ub)
                    || expr_impure(step)
                    || iter.iter().any(|i| expr_impure(&i.init))
                    || body.stmts.iter().any(stmt_impure)
                    || body
                        .tail
                        .as_ref()
                        .is_some_and(|v| v.iter().any(expr_impure))
            }
            TExpr::Map { inputs, body, .. } => {
                inputs.iter().any(expr_impure)
                    || body.stmts.iter().any(stmt_impure)
                    || body
                        .tail
                        .as_ref()
                        .is_some_and(|v| v.iter().any(expr_impure))
            }
            TExpr::Reduce {
                input, init, body, ..
            } => {
                expr_impure(input)
                    || expr_impure(init)
                    || body.stmts.iter().any(stmt_impure)
                    || body
                        .tail
                        .as_ref()
                        .is_some_and(|v| v.iter().any(expr_impure))
            }
            TExpr::Extract {
                tensor, indices, ..
            } => expr_impure(tensor) || indices.iter().any(expr_impure),
            _ => false,
        }
    }
    fn stmt_impure(s: &TStmt) -> bool {
        match s {
            TStmt::Let { value, .. } => expr_impure(value),
            TStmt::Effect(e, _) => expr_impure(e),
        }
    }
    block.stmts.iter().any(stmt_impure)
        || block
            .tail
            .as_ref()
            .is_some_and(|v| v.iter().any(expr_impure))
}
