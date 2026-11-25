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
        // Structs never reach the IR as one value: they are bundles of the
        // values of their fields (see `IrBuilder::ty`).
        CType::Struct(_) => Type::Void,
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

/// The elements of a tensor that is embedded into the program.
enum Elements {
    Floats(Vec<f32>),
    Ints(Vec<i32>),
}

impl Elements {
    fn len(&self) -> usize {
        match self {
            Elements::Floats(v) => v.len(),
            Elements::Ints(v) => v.len(),
        }
    }
}

pub struct IrBuilder {
    b: Builder,
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
    /// The struct definitions of the program.
    structs: Vec<crate::check::TStructDef>,
    /// The functions that `value_and_grad` differentiates; each one gets a
    /// gradient version.
    grad_funcs: Vec<String>,
    /// A struct value is a bundle of the values of its fields: this maps the
    /// id of the value that stands for the bundle to the values of its
    /// fields, flattened in the order of the definitions.
    struct_vals: HashMap<usize, Vec<Value>>,
}

impl IrBuilder {
    pub fn new(file: &str) -> Self {
        IrBuilder {
            b: Builder::new(),
            file: file.to_string(),
            scopes: Vec::new(),
            stack: Vec::new(),
            cur_loc: Loc::default(),
            runtime: Vec::new(),
            errors: Vec::new(),
            structs: Vec::new(),
            struct_vals: HashMap::new(),
            grad_funcs: Vec::new(),
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
        match t {
            // A struct is a bundle of the values of its fields; its type in
            // the IR is the tuple of the types of the fields, flattened.
            CType::Struct(id) => {
                let mut fields = Vec::new();
                for (_, f) in &self.structs[*id].fields {
                    for l in self.flat(f) {
                        fields.push(l);
                    }
                }
                Type::Tuple(fields)
            }
            t => ir_type(t),
        }
    }

    /// The types of the leaves of a type: a struct contributes the types of
    /// its fields, in order.
    fn flat(&self, t: &CType) -> Vec<Type> {
        match t {
            CType::Struct(id) => {
                let mut out = Vec::new();
                for (_, f) in &self.structs[*id].fields {
                    out.extend(self.flat(f));
                }
                out
            }
            t => vec![self.ty(t)],
        }
    }

    /// The number of the leaves of a type.
    fn leaf_count(&self, t: &CType) -> usize {
        self.flat(t).len()
    }

    /// The position of the first leaf of a field among the leaves of its
    /// struct.
    fn leaf_offset(&self, def: usize, field: usize) -> usize {
        let mut off = 0;
        for (i, (_, f)) in self.structs[def].fields.iter().enumerate() {
            if i == field {
                return off;
            }
            off += self.leaf_count(f);
        }
        off
    }

    /// The values that a value stands for: the fields of a struct, or the
    /// value itself.
    fn leaves(&self, v: &Value) -> Vec<Value> {
        match self.struct_vals.get(&v.id) {
            Some(leaves) => leaves.clone(),
            None => vec![v.clone()],
        }
    }

    /// Bundles the leaves into a value that stands for a struct.
    fn bundle(&mut self, leaves: Vec<Value>, ty: &CType) -> Value {
        if matches!(ty, CType::Struct(_)) {
            let marker = self.fresh(self.ty(ty));
            self.struct_vals.insert(marker.id, leaves);
            marker
        } else {
            debug_assert_eq!(leaves.len(), 1);
            leaves.into_iter().next().unwrap()
        }
    }

    // -----------------------------------------------------------------
    // Functions
    // -----------------------------------------------------------------

    pub fn build_program(&mut self, prog: &TProgram) -> ir::Module {
        self.structs = prog.structs.clone();
        let mut module = ir::Module::default();
        for f in &prog.funcs {
            module.funcs.push(self.build_func(f));
        }
        // The gradient version of every function that `value_and_grad` uses.
        for name in self.grad_funcs.clone() {
            let f = match module.funcs.iter().find(|g| g.name == name) {
                Some(f) => f.clone(),
                None => {
                    self.errors
                        .push(format!("the function `{}` cannot be differentiated", name));
                    continue;
                }
            };
            if let Err(e) =
                crate::passes::autodiff::check_differentiable(&f, &module.funcs)
            {
                self.errors
                    .push(format!("`{}` cannot be differentiated: {}", name, e));
                continue;
            }
            // The runtime declarations of the gradient functions are added
            // by the lowering of the domain operations (stage 5).
            let (grad_fn, _used) =
                crate::passes::autodiff::value_and_grad_of(&f, &module.funcs);
            module.funcs.push(grad_fn);
        }
        for r in &self.runtime {
            let t = |n: &str| Type::scalar(n).unwrap();
            let (params, results): (Vec<Type>, Vec<Type>) = match r.as_str() {
                "mlr_print_i32" => (vec![t("i32")], vec![]),
                "mlr_print_i64" => (vec![t("i64")], vec![]),
                "mlr_print_f32" => (vec![t("f32")], vec![]),
                "mlr_print_f64" => (vec![t("f64")], vec![]),
                "mlr_print_byte" => (vec![t("i32")], vec![]),
                "mlr_print_bool" => (vec![t("i1")], vec![]),
                "mlr_exp" | "mlr_ln" | "mlr_sqrt" | "mlr_tanh" | "mlr_sigmoid"
                | "mlr_gelu" | "mlr_abs" | "mlr_floor" | "mlr_ceil" | "mlr_round"
                | "mlr_sign" => (vec![t("f32")], vec![t("f32")]),
                "mlr_is_nan" | "mlr_is_finite" => (vec![t("f32")], vec![t("i1")]),
                "mlr_pow" | "mlr_min" | "mlr_max" => (vec![t("f32"), t("f32")], vec![t("f32")]),
                "mlr_clamp" => (vec![t("f32"), t("f32"), t("f32")], vec![t("f32")]),
                "mlr_now_ms" => (vec![], vec![t("i64")]),
                "mlr_exit" => (vec![t("i32")], vec![]),
                _ => (vec![t("i1")], vec![]),
            };
            module.decls.push(ir::FuncDecl {
                name: r.clone(),
                params,
                results,
            });
        }
        module
    }

    fn build_func(&mut self, f: &TFunc) -> ir::FuncOp {
        self.scopes.clear();
        self.scopes.push(HashMap::new());
        // A parameter of struct type becomes one parameter per field: the IR
        // signature is the flatten of the type. The name stands for a bundle
        // of the parameters, so that the body can use it as one value.
        let mut params = Vec::new();
        for (name, ty) in &f.params {
            if let CType::Struct(_) = ty {
                let leaves: Vec<Value> = self.flat(ty).into_iter().map(|t| self.fresh(t)).collect();
                params.extend(leaves.iter().cloned());
                let marker = self.fresh(self.ty(ty));
                self.struct_vals.insert(marker.id, leaves);
                self.define(name, marker);
            } else {
                let v = self.fresh(self.ty(ty));
                self.define(name, v.clone());
                params.push(v);
            }
        }
        let mut results: Vec<Type> = f
            .ret
            .as_ref()
            .map(|t| self.flat(t))
            .unwrap_or_default();
        // The entry point of the program is `main` of the C runtime, so it
        // returns an exit code of zero even when the program has none.
        let returns_exit_code = f.name == "main" && results.is_empty();
        if returns_exit_code {
            results = vec![Type::scalar("i32").unwrap()];
        }
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
                None if !returns_exit_code => {
                    let v = self.fresh(results[0].clone());
                    term.operands.push(v);
                }
                None => {
                    let mut zero = Op::new("arith.constant");
                    zero.attrs.push(("value".into(), Attr::Int(0)));
                    let v = self.fresh(results[0].clone());
                    zero.results.push(v.clone());
                    ops.push(zero);
                    term.operands.push(v);
                }
            }
        } else if results.len() > 1 {
            if yields.len() == results.len() {
                term.operands = yields;
            } else {
                for t in &results {
                    let v = self.fresh(t.clone());
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
                TStmt::LetMulti { names, value, span, .. } => {
                    self.cur_loc = Loc::new(&self.file, span.line, span.col);
                    // A `for` unpacks into all of its loop-carried values;
                    // every other value unpacks into the fields of its
                    // struct.
                    let leaves = match value {
                        TExpr::For {
                            var,
                            lb,
                            ub,
                            step,
                            iter,
                            body,
                            ..
                        } => self.build_for_all(var, lb, ub, step, iter, body),
                        value => {
                            let v = self.expr(value);
                            self.leaves(&v)
                        }
                    };
                    for (name, leaf) in names.iter().zip(leaves.iter()) {
                        if name != "_" {
                            self.define(name, leaf.clone());
                        }
                    }
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
                // contributes nothing to the terminator. A struct contributes
                // the values of its fields.
                if val.ty != Type::Void {
                    yielded.extend(self.leaves(&val));
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
                ret,
            } => self.call(name, type_args, args, ret),
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
            TExpr::Load { path, key, ty } => {
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
                let mut o = Op::new("arith.constant");
                let elements = match self.load_elements(path, key, &ty) {
                    Ok(v) => v,
                    Err(e) => {
                        self.errors.push(e);
                        Elements::Floats(Vec::new())
                    }
                };
                let wrong_size = elements.len() as i64 != count;
                match elements {
                    Elements::Ints(values) => {
                        if wrong_size && self.errors.is_empty() {
                            self.errors.push(format!(
                                "`{}` contains {} elements but {} were expected",
                                path,
                                values.len(),
                                count
                            ));
                        }
                        o.attrs.push(("value".into(), Attr::DenseI32(values)));
                    }
                    Elements::Floats(values) => {
                        if wrong_size && self.errors.is_empty() {
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
            TExpr::Sys { kind, cmd, args, ty } => {
                let name = match kind {
                    crate::check::SysKind::Exec => "mlr.exec",
                    crate::check::SysKind::ExecI => "mlr.exec_i",
                    crate::check::SysKind::ExecF => "mlr.exec_f",
                    crate::check::SysKind::ExecTensor => "mlr.exec_tensor",
                    crate::check::SysKind::LoadRuntime => "mlr.load_runtime",
                    crate::check::SysKind::Assert => "mlr.assert",
                };
                let mut o = Op::new(name);
                o.generic = true;
                o.attrs.push(("cmd".into(), Attr::Str(cmd.clone())));
                for a in args {
                    o.operands.push(self.expr(a));
                }
                if matches!(ty, CType::Void) {
                    self.emit(o);
                    self.fresh(Type::Void)
                } else {
                    let res = self.fresh(self.ty(ty));
                    o.results.push(res.clone());
                    self.emit(o);
                    res
                }
            }
            TExpr::Struct { def, fields, ty } => {
                let mut leaves = Vec::new();
                for f in fields {
                    let v = self.expr(f);
                    leaves.extend(self.leaves(&v));
                }
                debug_assert_eq!(leaves.len(), self.leaf_count(ty));
                let marker = self.fresh(self.ty(ty));
                self.struct_vals.insert(marker.id, leaves);
                let _ = def;
                marker
            }
            TExpr::Field { recv, index, ty } => {
                let r = self.expr(recv);
                let leaves = self.leaves(&r);
                let def = match recv.ty() {
                    CType::Struct(id) => id,
                    _ => unreachable!("a field access on a non struct"),
                };
                let off = self.leaf_offset(def, *index);
                let count = match ty {
                    CType::Struct(_) => self.leaf_count(ty),
                    _ => 1,
                };
                let picked: Vec<Value> = leaves[off..off + count].to_vec();
                self.bundle(picked, ty)
            }
            TExpr::With { recv, updates, ty } => {
                let r = self.expr(recv);
                let mut leaves = self.leaves(&r);
                let def = match recv.ty() {
                    CType::Struct(id) => id,
                    _ => unreachable!("an update of a non struct"),
                };
                for (index, value) in updates {
                    let off = self.leaf_offset(def, *index);
                    let v = self.expr(value);
                    let new_leaves = self.leaves(&v);
                    for (i, l) in new_leaves.into_iter().enumerate() {
                        leaves[off + i] = l;
                    }
                }
                self.bundle(leaves, ty)
            }
            TExpr::Save {
                path,
                tensors,
                keys,
            } => {
                let mut o = Op::new("mlr.save_pt");
                o.generic = true;
                for t in tensors {
                    o.operands.push(self.expr(t));
                }
                o.attrs.push(("path".into(), Attr::Str(path.clone())));
                o.attrs.push(("keys".into(), Attr::StrList(keys.clone())));
                self.emit(o);
                self.fresh(Type::Void)
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
            TExpr::Insert {
                tensor,
                indices,
                value,
                ty,
            } => {
                let mut o = Op::new("tensor.insert");
                o.operands.push(self.expr(value));
                o.operands.push(self.expr(tensor));
                for i in indices {
                    o.operands.push(self.expr(i));
                }
                let res = self.fresh(self.ty(ty));
                o.results.push(res.clone());
                self.emit(o);
                res
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

    /// Reads the elements of a tensor that is embedded into the program.
    ///
    /// The file can be a raw file of little endian floats (`file.bin`), a
    /// checkpoint of PyTorch (`file.pt`) or a safetensors file
    /// (`file.safetensors`). A file whose name starts with `hf://` is
    /// downloaded from HuggingFace.
    fn load_elements(
        &mut self,
        path: &str,
        key: &Option<String>,
        ty: &Type,
    ) -> Result<Elements, String> {
        let shape: Vec<i64> = match ty {
            Type::Tensor { shape, .. } => shape
                .iter()
                .map(|d| match d {
                    Dim::Static(v) => *v,
                    Dim::Dynamic => -1,
                })
                .collect(),
            _ => Vec::new(),
        };
        let integer = matches!(
            ty.elem_type(),
            Some(Type::Scalar(crate::ir::ScalarKind::I32))
        );
        // A checkpoint of PyTorch or a safetensors file.
        if path.ends_with(".pt") || path.ends_with(".safetensors") || path.starts_with("hf://") {
            let file = match crate::pt::resolve(path) {
                Ok(file) => file,
                Err(e) => return Err(e),
            };
            let tensors = crate::pt::read_checkpoint(&file)?;
            let name = match key {
                Some(name) => name.clone(),
                None => {
                    // The only tensor of the file.
                    if tensors.len() == 1 {
                        tensors.keys().next().cloned().unwrap()
                    } else {
                        return Err(format!(
                            "`{}` holds {} tensors, the name of the one to read is missing",
                            path,
                            tensors.len()
                        ));
                    }
                }
            };
            let tensor = tensors.get(&name).ok_or_else(|| {
                let mut names: Vec<&String> = tensors.keys().collect();
                names.sort();
                format!(
                    "`{}` has no tensor `{}` (it holds {})",
                    path,
                    name,
                    names
                        .iter()
                        .map(|n| n.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?;
            if tensor.shape != shape {
                return Err(format!(
                    "the tensor `{}` of `{}` has the shape {:?} but {:?} was expected",
                    name, path, tensor.shape, shape
                ));
            }
            if integer {
                Ok(Elements::Ints(tensor.as_i32()))
            } else {
                Ok(Elements::Floats(tensor.as_f32()))
            }
        } else {
            // A raw file of little endian values.
            let bytes = std::fs::read(path).map_err(|e| format!("cannot read `{}`: {}", path, e))?;
            if integer {
                let mut values = Vec::with_capacity(bytes.len() / 4);
                for chunk in bytes.as_chunks::<4>().0 {
                    values.push(i32::from_le_bytes(*chunk));
                }
                Ok(Elements::Ints(values))
            } else {
                let mut values = Vec::with_capacity(bytes.len() / 4);
                for chunk in bytes.as_chunks::<4>().0 {
                    values.push(f32::from_le_bytes(*chunk));
                }
                Ok(Elements::Floats(values))
            }
        }
    }

    fn call(&mut self, name: &str, type_args: &[CType], args: &[TExpr], ret: &CType) -> Value {
        // A call of a gradient version of a function, which the program
        // builder derives from the function itself.
        let mut name = name.to_string();
        if let Some(f) = name.strip_prefix("@value_and_grad:") {
            let f = f.to_string();
            if !self.grad_funcs.contains(&f) {
                self.grad_funcs.push(f.clone());
            }
            name = format!("{}.value_and_grad", f);
        }
        let name = name.as_str();
        match name {
            "print" => {
                let v = self.expr(&args[0]);
                let fname = match &v.ty {
                    Type::Scalar(k) => match k.name() {
                        "i1" | "bool" => "mlr_print_bool",
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
            // The scalar functions of the runtime library.
            "mlr_exp" | "mlr_ln" | "mlr_sqrt" | "mlr_tanh" | "mlr_sigmoid" | "mlr_gelu"
            | "mlr_print_byte" => {
                let v = self.expr(&args[0]);
                if !self.runtime.contains(&name.to_string()) {
                    self.runtime.push(name.to_string());
                }
                let mut o = Op::new("func.call");
                o.attrs.push(("callee".into(), Attr::Str(name.to_string())));
                o.operands.push(v);
                if *ret == CType::Void {
                    self.emit(o);
                    return self.fresh(Type::Void);
                }
                let res = self.fresh(self.ty(ret));
                o.results.push(res.clone());
                self.emit(o);
                return res;
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
        let mut o = Op::new("func.call");
        o.attrs.push(("callee".into(), Attr::Str(name.into())));
        // A call of the runtime library (its names start with `mlr_`) has to
        // be declared.
        if name.starts_with("mlr_") && !self.runtime.iter().any(|r| r == name) {
            self.runtime.push(name.to_string());
        }
        for a in args {
            let v = self.expr(a);
            o.operands.extend(self.leaves(&v));
        }
        // The type checker knows the result type of every call.
        let ret = match ret {
            CType::Void => None,
            t => Some(t.clone()),
        };
        match ret {
            Some(t) => {
                // A struct result is the bundle of the results of the call.
                let leaves: Vec<Value> = self
                    .flat(&t)
                    .into_iter()
                    .map(|ty| self.fresh(ty))
                    .collect();
                o.results = leaves.clone();
                self.emit(o);
                self.bundle(leaves, &t)
            }
            None => {
                self.emit(o);
                self.fresh(Type::Void)
            }
        }
    }

    /// Fills a tensor with zeros using nested structured loops.
    fn fill_zero(&mut self, tensor: Value, ty: Type) -> Value {
        let (shape, elem) = match &ty {
            Type::Tensor { shape, elem } => (shape.clone(), (**elem).clone()),
            _ => return tensor,
        };
        let mut zero = Op::new("arith.constant");
        // The zero has the element type of the tensor.
        match &elem {
            Type::Scalar(k) if k.is_int() => zero.attrs.push(("value".into(), Attr::Int(0))),
            _ => zero.attrs.push(("value".into(), Attr::Float(0.0))),
        }
        let z = self.fresh(elem.clone());
        zero.results.push(z.clone());
        self.emit(zero);

        let mut indices = Vec::new();
        self.fill_zero_rec(&shape, 0, &mut indices, &tensor, &z, &ty)
    }

    /// One level of the zero filling: a loop per dimension, and an insert of
    /// the zero at the full index in the innermost one.
    fn fill_zero_rec(
        &mut self,
        shape: &[Dim],
        level: usize,
        indices: &mut Vec<Value>,
        carried: &Value,
        z: &Value,
        ty: &Type,
    ) -> Value {
        if level == shape.len() {
            let mut insert = Op::new("tensor.insert");
            insert.operands.push(z.clone());
            insert.operands.push(carried.clone());
            insert.operands.extend(indices.iter().cloned());
            let inserted = self.fresh(ty.clone());
            insert.results.push(inserted.clone());
            self.emit(insert);
            return inserted;
        }
        let (lb, ub) = self.dim_bounds(&shape[level]);
        let step = self.index_const(1);
        let iv = self.fresh(Type::Index);
        let arg = self.fresh(ty.clone());
        let mut loop_op = Op::new("scf.for");
        loop_op.operands.push(lb);
        loop_op.operands.push(ub);
        loop_op.operands.push(step);
        loop_op.operands.push(carried.clone());
        indices.push(iv.clone());
        self.stack.push(Vec::new());
        let inner = self.fill_zero_rec(shape, level + 1, indices, &arg, z, ty);
        let mut ops = self.stack.pop().unwrap();
        let mut yield_op = Op::new("scf.yield");
        yield_op.operands.push(inner);
        ops.push(yield_op);
        loop_op.regions.push(Region {
            blocks: vec![ir::Block {
                args: vec![iv, arg],
                ops,
            }],
        });
        indices.pop();
        let res = self.fresh(ty.clone());
        loop_op.results.push(res.clone());
        self.emit(loop_op);
        res
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
        let (then_ops, _then_yields) =
            self.build_block(then_block, if yields_value { 1 } else { 0 });
        let (else_ops, _else_yields) = match else_block {
            Some(b) => self.build_block(b, if yields_value { 1 } else { 0 }),
            None => (vec![Op::new("scf.yield")], vec![]),
        };
        let mut o = Op::new("scf.if");
        o.operands.push(c);
        // A struct value is the bundle of the results of the branches.
        let leaves: Vec<Value> = if yields_value {
            self.flat(ty).into_iter().map(|t| self.fresh(t)).collect()
        } else {
            Vec::new()
        };
        o.results = leaves.clone();
        let res = if yields_value {
            self.bundle(leaves.clone(), ty)
        } else {
            self.fresh(Type::Void)
        };
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

    /// Builds the loop and returns ALL of its loop-carried values (the
    /// leaves of every iter binding, in order).
    fn build_for_all(
        &mut self,
        var: &str,
        lb: &TExpr,
        ub: &TExpr,
        step: &TExpr,
        iter: &[crate::check::TIter],
        body: &TBlock,
    ) -> Vec<Value> {
        self.build_for_inner(var, lb, ub, step, iter, body)
            .into_iter()
            .flat_map(|(values, _)| values)
            .collect()
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
        // The value of the loop is the first loop-carried value, bundled
        // (a struct stays a struct).
        let leaves = self.build_for_inner(var, lb, ub, step, iter, body);
        if let Some(((values, _), it)) = leaves.first().zip(iter.first()) {
            return self.bundle(values.clone(), &it.ty);
        }
        self.fresh(Type::Void)
    }

    fn build_for_inner(
        &mut self,
        var: &str,
        lb: &TExpr,
        ub: &TExpr,
        step: &TExpr,
        iter: &[crate::check::TIter],
        body: &TBlock,
    ) -> Vec<(Vec<Value>, Vec<Type>)> {
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
        // A loop-carried value of struct type is one iter arg per field.
        let mut iter_args = Vec::new();
        let mut iter_leaves: Vec<Vec<Value>> = Vec::new();
        for it in iter {
            let init = self.expr(&it.init);
            let leaves = self.leaves(&init);
            for l in &leaves {
                o.operands.push(l.clone());
            }
            let args: Vec<Value> = leaves.iter().map(|l| self.fresh(l.ty.clone())).collect();
            iter_leaves.push(args.clone());
            iter_args.push(it.name.clone());
        }
        self.scopes.push(HashMap::new());
        self.scopes
            .last_mut()
            .unwrap()
            .insert(var.to_string(), iv.clone());
        for (it, args) in iter.iter().zip(iter_leaves.iter()) {
            let marker = self.bundle(args.clone(), &it.ty);
            self.scopes
                .last_mut()
                .unwrap()
                .insert(it.name.clone(), marker);
        }
        let (body_ops, _) = self.build_block(body, iter.len());
        self.scopes.pop();

        let mut results = Vec::new();
        let mut result_leaves: Vec<Vec<Value>> = Vec::new();
        for (_, args) in iter.iter().zip(iter_leaves.iter()) {
            let leaves: Vec<Value> = args.iter().map(|a| self.fresh(a.ty.clone())).collect();
            results.extend(leaves.iter().cloned());
            o.results.extend(leaves.iter().cloned());
            result_leaves.push(leaves);
        }
        let mut args = vec![iv];
        for leaves in &iter_leaves {
            args.extend(leaves.iter().cloned());
        }
        o.regions.push(Region {
            blocks: vec![ir::Block {
                args,
                ops: body_ops,
            }],
        });
        self.emit(o);
        // The leaves of every loop-carried value, with their types.
        iter.iter()
            .zip(result_leaves.iter())
            .map(|(_it, leaves)| {
                let types = leaves.iter().map(|l| l.ty.clone()).collect();
                (leaves.clone(), types)
            })
            .collect()
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
            TExpr::Save { tensors, .. } => tensors.iter().any(expr_impure),
            TExpr::Struct { fields, .. } => fields.iter().any(expr_impure),
            TExpr::Field { recv, .. } => expr_impure(recv),
            TExpr::With { recv, updates, .. } => {
                expr_impure(recv) || updates.iter().any(|(_, v)| expr_impure(v))
            }
            TExpr::Sys { .. } => true,
            _ => false,
        }
    }
    fn stmt_impure(s: &TStmt) -> bool {
        match s {
            TStmt::Let { value, .. } => expr_impure(value),
            TStmt::LetMulti { value, .. } => expr_impure(value),
            TStmt::Effect(e, _) => expr_impure(e),
        }
    }
    block.stmts.iter().any(stmt_impure)
        || block
            .tail
            .as_ref()
            .is_some_and(|v| v.iter().any(expr_impure))
}
