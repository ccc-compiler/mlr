//! Type and shape checking.
//!
//! Types are checked statically; tensor dimensions are *symbolic*: a dynamic
//! dimension (`?`) becomes a symbol that can be unified with another dynamic
//! dimension or with a constant, which is what makes shape-aware reuse
//! decisions possible later on.

use std::collections::HashMap;

use crate::ast::*;
use crate::ir::ScalarKind;
use crate::lexer::{Error, Span};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CDim {
    Static(i64),
    Dynamic,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CType {
    Scalar(ScalarKind),
    Index,
    Tensor {
        dims: Vec<CDim>,
        elem: Box<CType>,
    },
    /// A struct by value; the number names the definition in the table of
    /// the program.
    Struct(usize),
    /// No value (a region that yields nothing).
    Void,
}

impl CType {
    pub fn is_tensor(&self) -> bool {
        matches!(self, CType::Tensor { .. })
    }

    pub fn elem(&self) -> Option<&CType> {
        match self {
            CType::Tensor { elem, .. } => Some(elem),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub fn dims(&self) -> Option<&[CDim]> {
        match self {
            CType::Tensor { dims, .. } => Some(dims),
            _ => None,
        }
    }

    pub fn is_numeric(&self) -> bool {
        match self {
            CType::Scalar(k) => k.is_float() || k.is_int(),
            CType::Index => true,
            _ => false,
        }
    }

    #[allow(dead_code)]
    pub fn is_float(&self) -> bool {
        matches!(self, CType::Scalar(k) if k.is_float())
    }

    /// Whether the type is a fixed width integer.
    pub fn is_int(&self) -> bool {
        matches!(self, CType::Scalar(k) if k.is_int())
    }

    /// The width of a scalar type in bits.
    pub fn width(&self) -> u32 {
        match self {
            CType::Scalar(k) => k.width(),
            CType::Index => 64,
            _ => 0,
        }
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, CType::Scalar(ScalarKind::I1))
    }

    pub fn print(&self) -> String {
        match self {
            CType::Scalar(k) => k.name().to_string(),
            CType::Index => "index".to_string(),
            CType::Void => "()".to_string(),
            CType::Struct(_) => "struct".to_string(),
            CType::Tensor { dims, elem } => {
                let mut s = String::from("Tensor<");
                for (i, d) in dims.iter().enumerate() {
                    if i > 0 {
                        s.push('x');
                    }
                    match d {
                        CDim::Static(v) => s.push_str(&v.to_string()),
                        CDim::Dynamic => s.push('?'),
                    }
                }
                if !dims.is_empty() {
                    s.push('x');
                }
                s.push_str(&elem.print());
                s.push('>');
                s
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Typed AST
// ---------------------------------------------------------------------------

/// The domain operations of the language. They are lowered to calls into the
/// runtime library, which either computes them with plain loops (CPU) or with
/// Metal Performance Shaders (Apple GPU).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DomainOp {
    MatMul,
    CrossEntropy,
    Softmax,
    SoftmaxTri,
    Sum,
    Add,
    Tmin,
    Tmax,
    Tclamp,
    Tpow,
    Texp,
    Tln,
    Mean,
    ArgMin,
    Norm1,
    Norm2,
    Var,
    Std,
    Dot,
    Reshape,
    Concat,
    Rows,
    Reverse,
    Iota,
    Linspace,
    Onehot,
    RandUniform,
    RandNormal,
    RandPermutation,
    Gather,
    Dropout,
    Accuracy,
    Conv2d,
    MaxPool2d,
    Relu,
    Flatten,
    Row,
    ArgMax,
    AddBias,
    Embed,
    EmbedSeq,
    Transpose,
    LayerNorm,
    SoftmaxCausal,
    Sample,
}

impl DomainOp {
    pub fn name(self) -> &'static str {
        match self {
            DomainOp::MatMul => "matmul",
            DomainOp::CrossEntropy => "cross_entropy",
            DomainOp::Softmax => "softmax",
            DomainOp::SoftmaxTri => "softmax_tri",
            DomainOp::Sum => "sum",
            DomainOp::Add => "add",
            DomainOp::Tmin => "tmin",
            DomainOp::Tmax => "tmax",
            DomainOp::Tclamp => "tclamp",
            DomainOp::Tpow => "tpow",
            DomainOp::Texp => "texp",
            DomainOp::Tln => "tln",
            DomainOp::Mean => "mean",
            DomainOp::ArgMin => "argmin",
            DomainOp::Norm1 => "norm1",
            DomainOp::Norm2 => "norm2",
            DomainOp::Var => "var",
            DomainOp::Std => "std",
            DomainOp::Dot => "dot",
            DomainOp::Reshape => "reshape",
            DomainOp::Concat => "concat",
            DomainOp::Rows => "rows",
            DomainOp::Reverse => "reverse",
            DomainOp::Iota => "iota",
            DomainOp::Linspace => "linspace",
            DomainOp::Onehot => "onehot",
            DomainOp::RandUniform => "rand_uniform",
            DomainOp::RandNormal => "rand_normal",
            DomainOp::RandPermutation => "rand_permutation",
            DomainOp::Gather => "gather",
            DomainOp::Dropout => "dropout",
            DomainOp::Accuracy => "accuracy",
            DomainOp::Conv2d => "conv2d",
            DomainOp::MaxPool2d => "maxpool2d",
            DomainOp::Relu => "relu",
            DomainOp::Flatten => "flatten",
            DomainOp::Row => "row",
            DomainOp::ArgMax => "argmax",
            DomainOp::AddBias => "add_bias",
            DomainOp::Embed => "embed",
            DomainOp::EmbedSeq => "embed_seq",
            DomainOp::Transpose => "transpose",
            DomainOp::LayerNorm => "layernorm",
            DomainOp::SoftmaxCausal => "softmax_causal",
            DomainOp::Sample => "sample",
        }
    }
}

/// One piece of a `print(...)` call: a string literal is written as is, a
/// scalar value is formatted according to its type.
#[derive(Clone, Debug)]
pub enum PrintPart {
    Str(String),
    Value(TExpr),
}

#[derive(Clone, Debug)]
pub enum TExpr {
    ConstInt(i64, CType),
    ConstFloat(f64, CType),
    ConstBool(bool),
    Var(String, CType),
    Unary(UnOp, Box<TExpr>, CType),
    Binary(BinOp, Box<TExpr>, Box<TExpr>, CType),
    Call {
        name: String,
        type_args: Vec<CType>,
        args: Vec<TExpr>,
        ret: CType,
    },
    If {
        cond: Box<TExpr>,
        then_block: Box<TBlock>,
        else_block: Option<Box<TBlock>>,
        ty: CType,
    },
    For {
        var: String,
        lb: Box<TExpr>,
        ub: Box<TExpr>,
        step: Box<TExpr>,
        iter: Vec<TIter>,
        body: Box<TBlock>,
        ty: CType,
        /// The anonymous struct of ALL the loop-carried values, when there
        /// are several: `let (a, b) = for ... iter(a = .., b = ..)` unpacks
        /// it.
        multi: Option<usize>,
    },
    Map {
        inputs: Vec<TExpr>,
        attrs: Vec<(String, AttrValue)>,
        params: Vec<String>,
        body: Box<TBlock>,
        ty: CType,
    },
    Reduce {
        input: Box<TExpr>,
        init: Box<TExpr>,
        attrs: Vec<(String, AttrValue)>,
        params: Vec<String>,
        body: Box<TBlock>,
        ty: CType,
    },
    Extract {
        tensor: Box<TExpr>,
        indices: Vec<TExpr>,
        ty: CType,
    },
    /// Writes a value into a tensor; the tensor is changed in place.
    Insert {
        tensor: Box<TExpr>,
        indices: Vec<TExpr>,
        value: Box<TExpr>,
        ty: CType,
    },
    /// An explicit numeric conversion (`index` <-> integer).
    Cast {
        value: Box<TExpr>,
        from: CType,
        to: CType,
    },
    /// A domain operation (`matmul`, `conv2d`, ...).
    Domain {
        op: DomainOp,
        args: Vec<TExpr>,
        params: Vec<i64>,
        ty: CType,
    },
    /// A tensor that is embedded into the program at compile time.
    Load {
        path: String,
        /// The name of the tensor in a checkpoint file.
        key: Option<String>,
        ty: CType,
    },
    /// Tensors that are written into a PyTorch checkpoint at run time.
    Save {
        path: String,
        tensors: Vec<TExpr>,
        keys: Vec<String>,
    },
    /// A struct value built from its fields, in the order of the definition.
    Struct {
        def: usize,
        fields: Vec<TExpr>,
        ty: CType,
    },
    /// A system call: `exec`, `exec_i`, `exec_f` or `assert`; the command
    /// (or the message) is a literal, the arguments are numbers.
    Sys {
        kind: SysKind,
        cmd: String,
        args: Vec<TExpr>,
        ty: CType,
    },
    /// A field of a struct: `recv.field`.
    Field {
        recv: Box<TExpr>,
        index: usize,
        ty: CType,
    },
    /// A functional update: `recv with { field: expr, ... }`.
    With {
        recv: Box<TExpr>,
        updates: Vec<(usize, TExpr)>,
        ty: CType,
    },
    /// `print("text", value, ...)`: a sequence of string literals and scalar
    /// values, written to the standard output followed by a newline.
    Print {
        parts: Vec<PrintPart>,
    },
}

impl TExpr {
    pub fn ty(&self) -> CType {
        match self {
            TExpr::ConstInt(_, t) => t.clone(),
            TExpr::ConstFloat(_, t) => t.clone(),
            TExpr::ConstBool(_) => CType::Scalar(ScalarKind::I1),
            TExpr::Var(_, t) => t.clone(),
            TExpr::Unary(_, _, t) => t.clone(),
            TExpr::Binary(_, _, _, t) => t.clone(),
            TExpr::Call { ret, .. } => ret.clone(),
            TExpr::If { ty, .. } => ty.clone(),
            TExpr::For { ty, .. } => ty.clone(),
            TExpr::Map { ty, .. } => ty.clone(),
            TExpr::Reduce { ty, .. } => ty.clone(),
            TExpr::Extract { ty, .. } => ty.clone(),
            TExpr::Insert { ty, .. } => ty.clone(),
            TExpr::Cast { to, .. } => to.clone(),
            TExpr::Domain { ty, .. } => ty.clone(),
            TExpr::Load { ty, .. } => ty.clone(),
            TExpr::Save { .. } => CType::Void,
            TExpr::Sys { ty, .. } => ty.clone(),
            TExpr::Struct { ty, .. } => ty.clone(),
            TExpr::Field { ty, .. } => ty.clone(),
            TExpr::With { ty, .. } => ty.clone(),
            TExpr::Print { .. } => CType::Void,
        }
    }

    #[allow(dead_code)]
    pub fn span(&self) -> Span {
        Span::default()
    }
}

/// The kind of a system call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SysKind {
    Exec,
    ExecI,
    ExecF,
    ExecTensor,
    LoadRuntime,
    Assert,
}

#[derive(Clone, Debug)]
pub struct TIter {
    pub name: String,
    pub ty: CType,
    pub init: TExpr,
}

#[derive(Clone, Debug, Default)]
pub struct TBlock {
    pub stmts: Vec<TStmt>,
    pub tail: Option<Vec<TExpr>>,
    /// The source location of the region.
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum TStmt {
    Let {
        name: String,
        ty: CType,
        value: TExpr,
        span: Span,
    },
    /// `let (a, b) = expr` unpacks the values of a bundle; `_` skips one.
    LetMulti {
        names: Vec<String>,
        /// The types of the values, in the order of the names; kept for the
        /// completeness of the typed tree.
        #[allow(dead_code)]
        types: Vec<CType>,
        value: TExpr,
        span: Span,
    },
    Effect(TExpr, Span),
}

#[derive(Clone, Debug)]
pub struct TFunc {
    pub name: String,
    pub params: Vec<(String, CType)>,
    pub ret: Option<CType>,
    pub body: TBlock,
}

/// A struct definition, with the checked types of its fields.
#[derive(Clone, Debug)]
pub struct TStructDef {
    pub name: String,
    pub fields: Vec<(String, CType)>,
}

#[derive(Clone, Debug, Default)]
pub struct TProgram {
    pub structs: Vec<TStructDef>,
    pub funcs: Vec<TFunc>,
}

#[derive(Clone, Debug)]
struct Sig {
    params: Vec<CType>,
    ret: Option<CType>,
}

pub struct Checker {
    sigs: HashMap<String, Sig>,
    scopes: Vec<HashMap<String, CType>>,
    cur_ret: Option<CType>,
    /// The struct definitions, in the order of their first appearance.
    structs: Vec<TStructDef>,
    /// Maps the name of a struct to its index in `structs`.
    struct_ids: HashMap<String, usize>,
    /// The first error that a place without a `Result` has seen.
    error: Option<Error>,
}

fn err<T>(msg: impl Into<String>, span: &Span) -> Result<T, Error> {
    Err(Error {
        msg: msg.into(),
        span: span.clone(),
    })
}

/// Literal-level conversion between numeric types.
fn coerce(expr: &TExpr, target: &CType) -> Option<TExpr> {
    let t = expr.ty();
    if t == *target {
        return Some(expr.clone());
    }
    match (expr, target) {
        (TExpr::ConstInt(v, _), CType::Scalar(k)) if k.is_int() => {
            Some(TExpr::ConstInt(*v, CType::Scalar(*k)))
        }
        (TExpr::ConstInt(v, _), CType::Index) => {
            Some(TExpr::Cast {
                value: Box::new(TExpr::ConstInt(*v, CType::Scalar(ScalarKind::I32))),
                from: CType::Scalar(ScalarKind::I32),
                to: CType::Index,
            })
        }
        (TExpr::ConstInt(v, _), CType::Scalar(k)) if k.is_float() => {
            Some(TExpr::ConstFloat(*v as f64, CType::Scalar(*k)))
        }
        (TExpr::ConstFloat(v, _), CType::Scalar(k)) if k.is_float() => {
            Some(TExpr::ConstFloat(*v, CType::Scalar(*k)))
        }
        (TExpr::ConstBool(b), CType::Scalar(ScalarKind::I1)) => Some(TExpr::ConstBool(*b)),
        _ => None,
    }
}

impl Checker {
    pub fn new() -> Self {
        Checker {
            sigs: HashMap::new(),
            scopes: Vec::new(),
            cur_ret: None,
            structs: Vec::new(),
            struct_ids: HashMap::new(),
            error: None,
        }
    }

    pub fn check(mut self, prog: &Program) -> Result<TProgram, Error> {
        // The names of the structs come first, so that the types of their
        // fields (and of the parameters of the functions) can name them.
        for s in &prog.structs {
            if self.struct_ids.contains_key(&s.name) {
                return err(
                    format!("duplicate definition of struct `{}`", s.name),
                    &s.span,
                );
            }
            self.struct_ids
                .insert(s.name.clone(), self.structs.len());
            self.structs.push(TStructDef {
                name: s.name.clone(),
                fields: Vec::new(),
            });
        }
        // The types of the fields; a struct may only contain scalars, tensors
        // and other structs.
        for s in &prog.structs {
            let id = self.struct_ids[&s.name];
            let mut fields = Vec::new();
            for (fname, fty) in &s.fields {
                let ty = self.resolve(fty, &s.span);
                if matches!(ty, CType::Void) {
                    return err(
                        format!("the field `{}.{}` has no type", s.name, fname),
                        &s.span,
                    );
                }
                fields.push((fname.clone(), ty));
            }
            self.structs[id].fields = fields;
        }
        // A struct may not contain itself, directly or indirectly.
        for (id, s) in prog.structs.iter().enumerate() {
            if self.reaches(id, id) {
                return err(
                    format!("the struct `{}` contains itself", s.name),
                    &s.span,
                );
            }
        }
        for f in &prog.funcs {
            if self.sigs.contains_key(&f.name) {
                return err(
                    format!("duplicate definition of function `{}`", f.name),
                    &f.span,
                );
            }
            self.sigs.insert(
                f.name.clone(),
                Sig {
                    params: Vec::new(),
                    ret: None,
                },
            );
        }
        // The methods are functions with the name of their struct in front.
        for block in &prog.impls {
            if !self.struct_ids.contains_key(&block.name) {
                return err(
                    format!("`impl` of the unknown struct `{}`", block.name),
                    &block.span,
                );
            }
            for m in &block.methods {
                let full = format!("{}.{}", block.name, m.name);
                if self.sigs.contains_key(&full) {
                    return err(
                        format!("duplicate definition of method `{}`", full),
                        &m.span,
                    );
                }
                self.sigs.insert(
                    full,
                    Sig {
                        params: Vec::new(),
                        ret: None,
                    },
                );
            }
        }
        // Signatures need the parameter types, which requires resolving the
        // symbolic dimensions of the parameters first.
        for f in &prog.funcs {
            let params: Vec<CType> = f
                .params
                .iter()
                .map(|p| self.resolve(&p.ty, &p.span))
                .collect();
            let ret = f.ret.as_ref().map(|r| self.resolve(r, &f.span));
            self.sigs.insert(f.name.clone(), Sig { params, ret });
        }
        for block in &prog.impls {
            for m in &block.methods {
                let full = format!("{}.{}", block.name, m.name);
                let params: Vec<CType> = m
                    .params
                    .iter()
                    .map(|p| self.resolve(&p.ty, &p.span))
                    .collect();
                let ret = m.ret.as_ref().map(|r| self.resolve(r, &m.span));
                self.sigs.insert(full, Sig { params, ret });
            }
        }
        let mut out = Vec::new();
        for f in &prog.funcs {
            out.push(self.check_func(f)?);
        }
        for block in &prog.impls {
            for m in &block.methods {
                // A method is a function with the name of its struct in
                // front, which is how it is called.
                let mut method = m.clone();
                method.name = format!("{}.{}", block.name, m.name);
                out.push(self.check_func(&method)?);
            }
        }
        if let Some(e) = self.error {
            return Err(e);
        }
        Ok(TProgram {
            structs: self.structs.clone(),
            funcs: out,
        })
    }

    /// Whether the struct `from` has a field of the struct `to`, directly or
    /// through other structs.
    fn reaches(&self, from: usize, to: usize) -> bool {
        for (_, ty) in &self.structs[from].fields {
            match ty {
                CType::Struct(next) => {
                    if *next == to || self.reaches(*next, to) {
                        return true;
                    }
                }
                CType::Tensor { elem, .. } => {
                    if matches!(**elem, CType::Struct(_)) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }

    /// Converts a source type into a checked type, creating fresh symbols for
    /// dynamic dimensions.
    fn resolve(&mut self, ty: &SrcType, span: &Span) -> CType {
        match ty {
            SrcType::Scalar(k) => CType::Scalar(*k),
            SrcType::Index => CType::Index,
            SrcType::Tensor { dims, elem } => {
                let dims = dims
                    .iter()
                    .map(|d| match d {
                        SrcDim::Static(v) => CDim::Static(*v),
                        SrcDim::Dynamic => CDim::Dynamic,
                    })
                    .collect();
                let elem = self.resolve(elem, span);
                CType::Tensor {
                    dims,
                    elem: Box::new(elem),
                }
            }
            SrcType::Named(name) => match self.struct_ids.get(name) {
                Some(id) => CType::Struct(*id),
                None => {
                    if self.error.is_none() {
                        self.error = Some(Error {
                            msg: format!("unknown type `{}`", name),
                            span: span.clone(),
                        });
                    }
                    CType::Void
                }
            },
        }
    }

    fn check_func(&mut self, f: &Func) -> Result<TFunc, Error> {
        self.scopes = vec![HashMap::new()];
        let mut params = Vec::new();
        for p in &f.params {
            let ty = self.resolve(&p.ty, &p.span);
            self.scopes[0].insert(p.name.clone(), ty.clone());
            params.push((p.name.clone(), ty));
        }
        let ret = f.ret.as_ref().map(|r| self.resolve(r, &f.span));
        self.cur_ret = ret.clone();
        let body = self.check_block(&f.body, ret.as_ref())?;
        Ok(TFunc {
            name: f.name.clone(),
            params,
            ret,
            body,
        })
    }

    fn check_block(&mut self, block: &Block, expected: Option<&CType>) -> Result<TBlock, Error> {
        self.scopes.push(HashMap::new());
        let mut out = TBlock::default();
        for stmt in &block.stmts {
            match stmt {
                Stmt::Let {
                    name,
                    ty,
                    value,
                    span,
                } => {
                    let v = self.check_expr(value)?;
                    let final_ty = match ty {
                        Some(t) => {
                            let target = self.resolve(t, span);
                            coerce(&v, &target).ok_or_else(|| Error {
                                msg: format!(
                                    "cannot initialize `{}` of type {} with a value of type {}",
                                    name,
                                    target.print(),
                                    v.ty().print()
                                ),
                                span: span.clone(),
                            })?
                        }
                        None => v.clone(),
                    };
                    if self.scopes.last().unwrap().contains_key(name) {
                        return err(
                            format!("`{}` is already declared in this scope", name),
                            span,
                        );
                    }
                    self.scopes
                        .last_mut()
                        .unwrap()
                        .insert(name.clone(), final_ty.ty());
                    out.stmts.push(TStmt::Let {
                        name: name.clone(),
                        ty: final_ty.ty(),
                        value: final_ty,
                        span: span.clone(),
                    });
                }
                Stmt::LetMulti(m) => {
                    let v = self.check_expr(&m.value)?;
                    // A `for` unpacks into all of its loop-carried values.
                    let for_id = match &v {
                        TExpr::For { multi: Some(id), .. } => Some(*id),
                        _ => None,
                    };
                    let id = match for_id {
                        Some(id) => id,
                        None => self.struct_of(&v.ty(), "the unpacked value", &m.span)?,
                    };
                    let fields = self.structs[id].fields.clone();
                    if fields.len() != m.names.len() {
                        return err(
                            format!(
                                "the value unpacks into {} value(s), {} name(s) were given",
                                fields.len(),
                                m.names.len()
                            ),
                            &m.span,
                        );
                    }
                    for (i, name) in m.names.iter().enumerate() {
                        let ty = &fields[i].1;
                        if name != "_" {
                            if self.scopes.last().unwrap().contains_key(name) {
                                return err(
                                    format!("`{}` is already declared in this scope", name),
                                    &m.span,
                                );
                            }
                            self.scopes.last_mut().unwrap().insert(name.clone(), ty.clone());
                        }
                    }
                    out.stmts.push(TStmt::LetMulti {
                        names: m.names.clone(),
                        types: fields.iter().map(|(_, t)| t.clone()).collect(),
                        value: v,
                        span: m.span.clone(),
                    });
                }
                Stmt::Effect(e) => {
                    let v = self.check_expr(e)?;
                    out.stmts.push(TStmt::Effect(v, e.span()));
                }
            }
        }
        let tail = match &block.tail {
            Some(exprs) => {
                let mut values = Vec::new();
                for e in exprs {
                    values.push(self.check_expr(e)?);
                }
                if let Some(expected) = expected {
                    if values.len() != 1 {
                        return err(
                            format!(
                                "this region yields {} value(s) but exactly one is expected",
                                values.len()
                            ),
                            &block.span,
                        );
                    }
                    let v = values.remove(0);
                    let coerced = coerce(&v, expected).ok_or_else(|| Error {
                        msg: format!(
                            "this region yields {} but {} is expected",
                            v.ty().print(),
                            expected.print()
                        ),
                        span: exprs[0].span(),
                    })?;
                    values.insert(0, coerced);
                }
                Some(values)
            }
            None => None,
        };
        if expected.is_some() && tail.is_none() {
            return err(
                "this region does not yield a value".to_string(),
                &block.span,
            );
        }
        self.scopes.pop();
        Ok(TBlock {
            stmts: out.stmts,
            tail,
            span: block.span.clone(),
        })
    }

    fn lookup(&self, name: &str) -> Option<CType> {
        self.scopes.iter().rev().find_map(|s| s.get(name)).cloned()
    }

    fn check_expr(&mut self, expr: &Expr) -> Result<TExpr, Error> {
        match expr {
            Expr::Int(v, _) => Ok(TExpr::ConstInt(*v, CType::Scalar(ScalarKind::I32))),
            Expr::Float(v, _) => Ok(TExpr::ConstFloat(*v, CType::Scalar(ScalarKind::F64))),
            Expr::Bool(b, _) => Ok(TExpr::ConstBool(*b)),
            Expr::Str(text, span) => err(
                format!(
                    "a string literal can only be used as a file name: `{}`",
                    text
                ),
                span,
            ),
            Expr::Var(name, span) => match self.lookup(name) {
                Some(t) => Ok(TExpr::Var(name.clone(), t)),
                None => err(format!("undefined variable `{}`", name), span),
            },
            Expr::Unary(op, e, span) => {
                let e = self.check_expr(e)?;
                let ty = e.ty();
                match op {
                    UnOp::Neg => {
                        if !ty.is_numeric() {
                            return err(
                                format!("cannot negate a value of type {}", ty.print()),
                                span,
                            );
                        }
                        // The negation of a constant is a constant, so that
                        // `-1.5` can be written wherever a literal goes.
                        match e {
                            TExpr::ConstInt(v, _) => {
                                Ok(TExpr::ConstInt(-v, CType::Scalar(ScalarKind::I32)))
                            }
                            TExpr::ConstFloat(v, t) => Ok(TExpr::ConstFloat(-v, t)),
                            e => Ok(TExpr::Unary(*op, Box::new(e), ty)),
                        }
                    }
                    UnOp::Not => {
                        if !ty.is_bool() {
                            return err(format!("`!` requires a bool, found {}", ty.print()), span);
                        }
                        Ok(TExpr::Unary(*op, Box::new(e), ty))
                    }
                }
            }
            Expr::Binary(op, l, r, span) => {
                let l = self.check_expr(l)?;
                let r = self.check_expr(r)?;
                self.check_binary(*op, l, r, span)
            }
            Expr::Call {
                name,
                type_args,
                args,
                span,
            } => self.check_call(name, type_args, args, span),
            Expr::If {
                cond,
                then_block,
                else_block,
                span,
            } => {
                let cond = self.check_expr(cond)?;
                if !cond.ty().is_bool() {
                    return err(
                        format!(
                            "the condition of `if` must be a bool, found {}",
                            cond.ty().print()
                        ),
                        span,
                    );
                }
                let then_block = Box::new(self.check_block(then_block, None)?);
                let else_block = match else_block {
                    Some(b) => Some(Box::new(self.check_block(b, None)?)),
                    None => None,
                };
                // A tail that yields nothing (a `print`) counts as no tail.
                let then_ty = then_block
                    .tail
                    .as_ref()
                    .filter(|v| !v.is_empty())
                    .map(|v| v[0].ty())
                    .filter(|t| !matches!(t, CType::Void));
                let else_ty = else_block
                    .as_ref()
                    .and_then(|b| b.tail.as_ref())
                    .filter(|v| !v.is_empty())
                    .map(|v| v[0].ty())
                    .filter(|t| !matches!(t, CType::Void));
                let ty = match (&then_ty, &else_ty) {
                    (Some(a), Some(b)) => {
                        if a != b {
                            return err(
                                format!(
                                    "the branches of `if` yield different types: {} and {}",
                                    a.print(),
                                    b.print()
                                ),
                                span,
                            );
                        }
                        a.clone()
                    }
                    (Some(a), None) => {
                        return err(
                            format!(
                                "the `if` expression yields {} but has no `else` branch",
                                a.print()
                            ),
                            span,
                        )
                    }
                    (None, Some(b)) => {
                        return err(
                            format!(
                                "the `else` branch yields {} but the `then` branch yields nothing",
                                b.print()
                            ),
                            span,
                        )
                    }
                    (None, None) => CType::Void,
                };
                Ok(TExpr::If {
                    cond: Box::new(cond),
                    then_block,
                    else_block,
                    ty,
                })
            }
            Expr::For {
                var,
                lb,
                ub,
                step,
                iter,
                body,
                span,
            } => {
                let lb = self.check_expr(lb)?;
                let ub = self.check_expr(ub)?;
                let step = match step {
                    Some(s) => self.check_expr(s)?,
                    None => TExpr::ConstInt(1, CType::Index),
                };
                let lb = self.as_index(lb, span)?;
                let ub = self.as_index(ub, span)?;
                let step = self.as_index(step, span)?;
                let mut titer = Vec::new();
                for it in iter {
                    let init = self.check_expr(&it.init)?;
                    let ty = init.ty();
                    titer.push(TIter {
                        name: it.name.clone(),
                        ty,
                        init,
                    });
                }
                // The body sees the induction variable and the loop-carried
                // values.
                self.scopes.push(HashMap::new());
                self.scopes
                    .last_mut()
                    .unwrap()
                    .insert(var.clone(), CType::Index);
                for it in &titer {
                    self.scopes
                        .last_mut()
                        .unwrap()
                        .insert(it.name.clone(), it.ty.clone());
                }
                let body = self.check_block(body, None)?;
                self.scopes.pop();
                // The body must yield exactly one value per loop-carried
                // binding.
                let body_types = self.yield_types(&body);
                if body_types.len() != titer.len() {
                    return err(
                        format!(
                            "the loop body yields {} value(s) but {} loop-carried value(s) were declared",
                            body_types.len(),
                            titer.len()
                        ),
                        span,
                    );
                }
                for (i, (a, b)) in body_types.iter().zip(titer.iter()).enumerate() {
                    if a != &b.ty {
                        return err(
                            format!(
                                "loop-carried value `{}` has type {} but the body yields {}",
                                b.name,
                                b.ty.print(),
                                a.print()
                            ),
                            span,
                        );
                    }
                    let _ = i;
                }
                // The value of a `for` expression is the final value of its
                // loop-carried values. When there are several, an anonymous
                // struct of all of them is available to `let (a, b) = ...`.
                let ty = titer.first().map(|it| it.ty.clone()).unwrap_or(CType::Void);
                let multi = if titer.len() > 1 {
                    let id = self.structs.len();
                    self.structs.push(TStructDef {
                        name: format!("@for:{}", var),
                        fields: titer.iter().map(|it| (it.name.clone(), it.ty.clone())).collect(),
                    });
                    Some(id)
                } else {
                    None
                };
                Ok(TExpr::For {
                    var: var.clone(),
                    lb: Box::new(lb),
                    ub: Box::new(ub),
                    step: Box::new(step),
                    iter: titer,
                    body: Box::new(body),
                    multi,
                    ty,
                })
            }
            Expr::Map {
                inputs,
                attrs,
                params,
                body,
                span,
            } => {
                let mut tinputs = Vec::new();
                for i in inputs {
                    tinputs.push(self.check_expr(i)?);
                }
                if tinputs.is_empty() {
                    return err("`map` needs at least one input tensor", span);
                }
                let first = tinputs[0].ty();
                let mut all_dims = Vec::new();
                for t in &tinputs {
                    let ty = t.ty();
                    match &ty {
                        CType::Tensor { dims, .. } => all_dims.push(dims.clone()),
                        _ => {
                            return err(
                                format!("`map` expects tensor inputs, found {}", ty.print()),
                                span,
                            )
                        }
                    }
                }
                let rank = all_dims[0].len();
                if all_dims.iter().any(|d| d.len() != rank) {
                    return err("all `map` inputs must have the same rank", span);
                }
                let dims = self.unify_dim_list(&all_dims, span)?;
                let elem = first.elem().cloned().unwrap();
                if params.len() != tinputs.len() {
                    return err(
                        format!(
                            "`map` has {} input(s) but {} parameter(s)",
                            tinputs.len(),
                            params.len()
                        ),
                        span,
                    );
                }
                self.scopes.push(HashMap::new());
                for (i, p) in params.iter().enumerate() {
                    let e = tinputs[i].ty().elem().cloned().unwrap();
                    self.scopes.last_mut().unwrap().insert(p.clone(), e);
                }
                let body = self.check_block(body, None)?;
                self.scopes.pop();
                let body_ty = match &body.tail {
                    Some(values) if values.len() == 1 => values[0].ty(),
                    Some(values) => {
                        return err(
                            format!(
                                "the body of `map` yields {} value(s), expected exactly one",
                                values.len()
                            ),
                            span,
                        )
                    }
                    None => {
                        return err("the body of `map` must yield a value", span);
                    }
                };
                if body_ty != elem {
                    return err(
                        format!(
                            "the body of `map` yields {} but the element type is {}",
                            body_ty.print(),
                            elem.print()
                        ),
                        span,
                    );
                }
                let ty = CType::Tensor {
                    dims,
                    elem: Box::new(elem),
                };
                Ok(TExpr::Map {
                    inputs: tinputs,
                    attrs: attrs.clone(),
                    params: params.clone(),
                    body: Box::new(body),
                    ty,
                })
            }
            Expr::Reduce {
                input,
                init,
                attrs,
                params,
                body,
                span,
            } => {
                let input = self.check_expr(input)?;
                let init = self.check_expr(init)?;
                let ity = input.ty();
                let dims = match &ity {
                    CType::Tensor { dims, .. } => dims.clone(),
                    _ => {
                        return err(
                            format!("`reduce` expects a tensor, found {}", ity.print()),
                            span,
                        )
                    }
                };
                if dims.len() != 1 {
                    return err("`reduce` currently only supports rank-1 tensors", span);
                }
                let elem = ity.elem().cloned().unwrap();
                // The accumulator has the element type of the tensor; a
                // literal initialiser is converted accordingly.
                let init = coerce(&init, &elem).ok_or_else(|| Error {
                    msg: format!(
                        "the accumulator of `reduce` has type {} but the elements have type {}",
                        init.ty().print(),
                        elem.print()
                    ),
                    span: span.clone(),
                })?;
                let acc_ty = init.ty();
                if !acc_ty.is_numeric() {
                    return err(
                        format!(
                            "the accumulator of `reduce` must be numeric, found {}",
                            acc_ty.print()
                        ),
                        span,
                    );
                }
                if params.len() != 2 {
                    return err(
                        format!(
                            "`reduce` takes two parameters (accumulator, element), found {}",
                            params.len()
                        ),
                        span,
                    );
                }
                self.scopes.push(HashMap::new());
                self.scopes
                    .last_mut()
                    .unwrap()
                    .insert(params[0].clone(), acc_ty.clone());
                self.scopes
                    .last_mut()
                    .unwrap()
                    .insert(params[1].clone(), elem.clone());
                let body = self.check_block(body, None)?;
                self.scopes.pop();
                let body_ty = match &body.tail {
                    Some(values) if values.len() == 1 => values[0].ty(),
                    Some(values) => {
                        return err(
                            format!(
                                "the body of `reduce` yields {} value(s), expected exactly one",
                                values.len()
                            ),
                            span,
                        )
                    }
                    None => return err("the body of `reduce` must yield a value", span),
                };
                if body_ty != acc_ty {
                    return err(
                        format!(
                            "the body of `reduce` yields {} but the accumulator has type {}",
                            body_ty.print(),
                            acc_ty.print()
                        ),
                        span,
                    );
                }
                Ok(TExpr::Reduce {
                    input: Box::new(input),
                    init: Box::new(init),
                    attrs: attrs.clone(),
                    params: params.clone(),
                    body: Box::new(body),
                    ty: acc_ty,
                })
            }
            Expr::Extract {
                tensor,
                indices,
                span,
            } => {
                let t = self.check_expr(tensor)?;
                let ty = t.ty();
                let dims = match &ty {
                    CType::Tensor { dims, .. } => dims.clone(),
                    _ => {
                        return err(
                            format!("`tensor.extract` expects a tensor, found {}", ty.print()),
                            span,
                        )
                    }
                };
                if dims.len() != indices.len() {
                    return err(
                        format!(
                            "`tensor.extract` expects {} index/indices but got {}",
                            dims.len(),
                            indices.len()
                        ),
                        span,
                    );
                }
                let mut tidx = Vec::new();
                for i in indices {
                    let ie = self.check_expr(i)?;
                    let ie = self.as_index(ie, span)?;
                    let ie = match ie.ty() {
                        CType::Index => ie,
                        t => TExpr::Cast {
                            value: Box::new(ie),
                            from: t,
                            to: CType::Index,
                        },
                    };
                    tidx.push(ie);
                }
                let elem = ty.elem().cloned().unwrap();
                Ok(TExpr::Extract {
                    tensor: Box::new(t),
                    indices: tidx,
                    ty: elem,
                })
            }
            Expr::Insert {
                tensor,
                indices,
                value,
                span,
            } => {
                let t = self.check_expr(tensor)?;
                let ty = t.ty();
                let dims = match &ty {
                    CType::Tensor { dims, .. } => dims.clone(),
                    _ => {
                        return err(
                            format!("`tensor.insert` expects a tensor, found {}", ty.print()),
                            span,
                        )
                    }
                };
                if dims.len() != indices.len() {
                    return err(
                        format!(
                            "`tensor.insert` expects {} index/indices but got {}",
                            dims.len(),
                            indices.len()
                        ),
                        span,
                    );
                }
                let mut tidx = Vec::new();
                for i in indices {
                    let ie = self.check_expr(i)?;
                    let ie = self.as_index(ie, span)?;
                    let ie = match ie.ty() {
                        CType::Index => ie,
                        t => TExpr::Cast {
                            value: Box::new(ie),
                            from: t,
                            to: CType::Index,
                        },
                    };
                    tidx.push(ie);
                }
                let elem = ty.elem().cloned().unwrap();
                let v = self.check_expr(value)?;
                let v = match coerce(&v, &elem) {
                    Some(v) => v,
                    None => {
                        return err(
                            format!(
                                "`tensor.insert` expects a value of type {} but found {}",
                                elem.print(),
                                v.ty().print()
                            ),
                            span,
                        )
                    }
                };
                Ok(TExpr::Insert {
                    tensor: Box::new(t),
                    indices: tidx,
                    value: Box::new(v),
                    ty,
                })
            }
            Expr::Field { recv, name, span } => {
                let r = self.check_expr(recv)?;
                let id = self.struct_of(&r.ty(), name, span)?;
                let index = self.field_index(id, name, span)?;
                let ty = self.structs[id].fields[index].1.clone();
                Ok(TExpr::Field {
                    recv: Box::new(r),
                    index,
                    ty,
                })
            }
            Expr::StructLit { name, fields, span } => {
                let id = self.struct_id(name, span)?;
                let def = self.structs[id].clone();
                for (i, (fname, _)) in fields.iter().enumerate() {
                    if fields[i + 1..].iter().any(|(n, _)| n == fname) {
                        return err(
                            format!("duplicate field `{}` in the literal of `{}`", fname, name),
                            span,
                        );
                    }
                }
                if fields.len() != def.fields.len() {
                    return err(
                        format!(
                            "the struct `{}` has {} fields, the literal has {}",
                            name,
                            def.fields.len(),
                            fields.len()
                        ),
                        span,
                    );
                }
                let mut values = Vec::new();
                for (fname, fty) in &def.fields {
                    let expr = fields
                        .iter()
                        .find(|(n, _)| n == fname)
                        .map(|(_, e)| e)
                        .ok_or_else(|| Error {
                            msg: format!("the field `{}.{}` is missing", name, fname),
                            span: span.clone(),
                        })?;
                    let v = self.check_expr(expr)?;
                    let v = coerce(&v, fty).ok_or_else(|| Error {
                        msg: format!(
                            "the field `{}.{}` has type {}, found {}",
                            name,
                            fname,
                            fty.print(),
                            v.ty().print()
                        ),
                        span: span.clone(),
                    })?;
                    values.push(v);
                }
                Ok(TExpr::Struct {
                    def: id,
                    fields: values,
                    ty: CType::Struct(id),
                })
            }
            Expr::With { recv, updates, span } => {
                let r = self.check_expr(recv)?;
                let id = self.struct_of(&r.ty(), "the value", span)?;
                let def = self.structs[id].clone();
                for (i, (fname, _)) in updates.iter().enumerate() {
                    if updates[i + 1..].iter().any(|(n, _)| n == fname) {
                        return err(
                            format!("duplicate field `{}` in the update", fname),
                            span,
                        );
                    }
                }
                let mut ups = Vec::new();
                for (fname, fexpr) in updates {
                    let index = self.field_index(id, fname, span)?;
                    let fty = def.fields[index].1.clone();
                    let v = self.check_expr(fexpr)?;
                    let v = coerce(&v, &fty).ok_or_else(|| Error {
                        msg: format!(
                            "the field `{}.{}` has type {}, found {}",
                            def.name,
                            fname,
                            fty.print(),
                            v.ty().print()
                        ),
                        span: span.clone(),
                    })?;
                    ups.push((index, v));
                }
                Ok(TExpr::With {
                    recv: Box::new(r),
                    updates: ups,
                    ty: CType::Struct(id),
                })
            }
            Expr::Method {
                recv,
                name,
                args,
                span,
            } => {
                let r = self.check_expr(recv)?;
                let id = self.struct_of(&r.ty(), name, span)?;
                let sname = self.structs[id].name.clone();
                let full = format!("{}.{}", sname, name);
                let sig = match self.sigs.get(&full) {
                    Some(s) => s.clone(),
                    None => {
                        return err(
                            format!("the struct `{}` has no method `{}`", sname, name),
                            span,
                        )
                    }
                };
                if sig.params.len() != args.len() + 1 {
                    return err(
                        format!(
                            "the method `{}` expects {} argument(s), found {}",
                            full,
                            sig.params.len() - 1,
                            args.len()
                        ),
                        span,
                    );
                }
                let mut targs = vec![coerce(&r, &sig.params[0]).ok_or_else(|| Error {
                    msg: format!(
                        "the receiver of `{}` has type {}, expected {}",
                        full,
                        r.ty().print(),
                        sig.params[0].print()
                    ),
                    span: span.clone(),
                })?];
                for (a, expected) in args.iter().zip(sig.params[1..].iter()) {
                    let ta = self.check_expr(a)?;
                    let ta = coerce(&ta, expected).ok_or_else(|| Error {
                        msg: format!(
                            "argument of `{}` has type {}, expected {}",
                            full,
                            ta.ty().print(),
                            expected.print()
                        ),
                        span: span.clone(),
                    })?;
                    targs.push(ta);
                }
                Ok(TExpr::Call {
                    name: full,
                    type_args: vec![],
                    args: targs,
                    ret: sig.ret.unwrap_or(CType::Void),
                })
            }
        }
    }

    /// The index of the struct that a type names, or an error.
    fn struct_of(&self, ty: &CType, what: &str, span: &Span) -> Result<usize, Error> {
        match ty {
            CType::Struct(id) => Ok(*id),
            t => err(format!("`{}` is not a struct, found {}", what, t.print()), span),
        }
    }

    fn struct_id(&self, name: &str, span: &Span) -> Result<usize, Error> {
        match self.struct_ids.get(name) {
            Some(id) => Ok(*id),
            None => err(format!("unknown struct `{}`", name), span),
        }
    }

    fn field_index(&self, id: usize, name: &str, span: &Span) -> Result<usize, Error> {
        match self.structs[id].fields.iter().position(|(n, _)| n == name) {
            Some(i) => Ok(i),
            None => err(
                format!("the struct `{}` has no field `{}`", self.structs[id].name, name),
                span,
            ),
        }
    }

    /// The types yielded by a block (the tail expression, or nothing).
    fn yield_types(&self, block: &TBlock) -> Vec<CType> {
        match &block.tail {
            Some(values) => values.iter().map(|v| v.ty()).collect(),
            None => vec![],
        }
    }

    fn as_index(&mut self, e: TExpr, span: &Span) -> Result<TExpr, Error> {
        match e.ty() {
            CType::Index => Ok(e),
            CType::Scalar(k) if k.is_int() => Ok(e),
            other => err(format!("expected an index, found {}", other.print()), span),
        }
    }

    /// Unifies the dimensions of several tensors: static dimensions have to
    /// agree, a dynamic dimension is refined by a static one.
    fn unify_dim_list(&mut self, dims: &[Vec<CDim>], span: &Span) -> Result<Vec<CDim>, Error> {
        let mut result: Vec<CDim> = Vec::new();
        let rank = dims.first().map(|d| d.len()).unwrap_or(0);
        for i in 0..rank {
            let mut unified = CDim::Dynamic;
            for d in dims {
                match d[i] {
                    CDim::Static(v) => match unified {
                        CDim::Static(w) if w != v => {
                            return err(format!("shape mismatch: dimension {} and {}", w, v), span)
                        }
                        _ => unified = CDim::Static(v),
                    },
                    CDim::Dynamic => {}
                }
            }
            result.push(unified);
        }
        Ok(result)
    }

    fn check_binary(&mut self, op: BinOp, l: TExpr, r: TExpr, span: &Span) -> Result<TExpr, Error> {
        let lt = l.ty();
        let rt = r.ty();
        match op {
            BinOp::And | BinOp::Or => {
                if !lt.is_bool() || !rt.is_bool() {
                    return err(
                        format!(
                            "`{}` requires bool operands, found {} and {}",
                            op.name(),
                            lt.print(),
                            rt.print()
                        ),
                        span,
                    );
                }
                Ok(TExpr::Binary(
                    op,
                    Box::new(l),
                    Box::new(r),
                    CType::Scalar(ScalarKind::I1),
                ))
            }
            BinOp::Eq | BinOp::Ne => {
                if lt != rt {
                    // Allow comparing a literal with a differently typed value.
                    if coerce(&l, &rt).is_some() || coerce(&r, &lt).is_some() {
                        // ok
                    } else {
                        return err(
                            format!("cannot compare {} and {}", lt.print(), rt.print()),
                            span,
                        );
                    }
                }
                Ok(TExpr::Binary(
                    op,
                    Box::new(l),
                    Box::new(r),
                    CType::Scalar(ScalarKind::I1),
                ))
            }
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                let (l, r, common) = self.unify_numeric(l, r, span, op)?;
                let _ = common;
                Ok(TExpr::Binary(
                    op,
                    Box::new(l),
                    Box::new(r),
                    CType::Scalar(ScalarKind::I1),
                ))
            }
            _ => {
                // A tensor and a scalar (or two tensors of the same shape)
                // combine element by element: `w - 0.01 * dw`.
                if lt.is_tensor() || rt.is_tensor() {
                    return self.broadcast(op, l, r, span);
                }
                let (l, r, common) = self.unify_numeric(l, r, span, op)?;
                if op == BinOp::Rem && !matches!(common, CType::Scalar(k) if k.is_int()) {
                    return err("`%` requires integer operands", span);
                }
                Ok(TExpr::Binary(op, Box::new(l), Box::new(r), common))
            }
        }
    }

    /// The element wise combination of a tensor with a scalar or with another
    /// tensor of the same shape, written as `map` with a scalar body.
    fn broadcast(
        &mut self,
        op: BinOp,
        l: TExpr,
        r: TExpr,
        span: &Span,
    ) -> Result<TExpr, Error> {
        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {}
            _ => {
                return err(
                    format!("`{}` is not defined for tensors", op.name()),
                    span,
                )
            }
        }
        let (tensor, scalar) = if l.ty().is_tensor() && r.ty().is_tensor() {
            // Two tensors of the same shape.
            let lt = l.ty();
            let rt = r.ty();
            if lt != rt {
                return err(
                    format!("cannot combine {} and {} element wise", lt.print(), rt.print()),
                    span,
                );
            }
            let elem = lt.elem().cloned().unwrap();
            let name = "$a";
            let body = TBlock {
                stmts: vec![],
                tail: Some(vec![TExpr::Binary(
                    op,
                    Box::new(TExpr::Var(name.into(), elem.clone())),
                    Box::new(TExpr::Var("$b".into(), elem.clone())),
                    elem.clone(),
                )]),
                span: span.clone(),
            };
            return Ok(TExpr::Map {
                inputs: vec![l, r],
                attrs: vec![],
                params: vec![name.into(), "$b".into()],
                body: Box::new(body),
                ty: lt,
            });
        } else if l.ty().is_tensor() {
            (l, r)
        } else if r.ty().is_tensor() {
            // The scalar comes first: `2 * x`. The operation keeps its
            // order, so the body is `2 - $a` for a subtraction.
            let ty = r.ty();
            let elem = ty.elem().cloned().unwrap();
            let scalar = coerce(&l, &elem).ok_or_else(|| Error {
                msg: format!(
                    "cannot combine {} with the element type {}",
                    l.ty().print(),
                    elem.print()
                ),
                span: span.clone(),
            })?;
            let name = "$a";
            let body = TBlock {
                stmts: vec![],
                tail: Some(vec![TExpr::Binary(
                    op,
                    Box::new(scalar),
                    Box::new(TExpr::Var(name.into(), elem.clone())),
                    elem.clone(),
                )]),
                span: span.clone(),
            };
            return Ok(TExpr::Map {
                inputs: vec![r],
                attrs: vec![],
                params: vec![name.into()],
                body: Box::new(body),
                ty,
            });
        } else {
            unreachable!("broadcast without a tensor")
        };
        let ty = tensor.ty();
        let elem = ty.elem().cloned().unwrap();
        let scalar = coerce(&scalar, &elem).ok_or_else(|| Error {
            msg: format!(
                "cannot combine {} with the element type {}",
                scalar.ty().print(),
                elem.print()
            ),
            span: span.clone(),
        })?;
        let name = "$a";
        let body = TBlock {
            stmts: vec![],
            tail: Some(vec![TExpr::Binary(
                op,
                Box::new(TExpr::Var(name.into(), elem.clone())),
                Box::new(scalar),
                elem.clone(),
            )]),
            span: span.clone(),
        };
        Ok(TExpr::Map {
            inputs: vec![tensor],
            attrs: vec![],
            params: vec![name.into()],
            body: Box::new(body),
            ty,
        })
    }

    fn unify_numeric(
        &mut self,
        l: TExpr,
        r: TExpr,
        span: &Span,
        op: BinOp,
    ) -> Result<(TExpr, TExpr, CType), Error> {
        let lt = l.ty();
        let rt = r.ty();
        if lt == rt {
            if !lt.is_numeric() {
                return err(
                    format!(
                        "operator `{}` requires numeric operands, found {}",
                        op.name(),
                        lt.print()
                    ),
                    span,
                );
            }
            return Ok((l, r, lt));
        }
        // `index` and fixed width integers mix freely: the index operand is
        // converted to the integer type.
        if lt == CType::Index && matches!(rt, CType::Scalar(k) if k.is_int()) {
            let c = TExpr::Cast {
                value: Box::new(l),
                from: CType::Index,
                to: rt.clone(),
            };
            return Ok((c, r, rt));
        }
        if rt == CType::Index && matches!(lt, CType::Scalar(k) if k.is_int()) {
            let c = TExpr::Cast {
                value: Box::new(r),
                from: CType::Index,
                to: lt.clone(),
            };
            return Ok((l, c, lt));
        }
        // Two floats of different widths: the computation runs in the
        // narrower type, which is the type of the runtime.
        if let (CType::Scalar(a), CType::Scalar(b)) = (&lt, &rt) {
            if a.is_float() && b.is_float() && a != b {
                if a.width() < b.width() {
                    let c = TExpr::Cast {
                        value: Box::new(r),
                        from: rt.clone(),
                        to: lt.clone(),
                    };
                    return Ok((l, c, lt));
                }
                let c = TExpr::Cast {
                    value: Box::new(l),
                    from: lt.clone(),
                    to: rt.clone(),
                };
                return Ok((c, r, rt));
            }
        }
        // An integer is converted to a float when it meets one.
        if let (CType::Scalar(a), CType::Scalar(b)) = (&lt, &rt) {
            if a.is_int() && b.is_float() {
                let c = TExpr::Cast {
                    value: Box::new(l),
                    from: lt.clone(),
                    to: rt.clone(),
                };
                return Ok((c, r, rt));
            }
            if b.is_int() && a.is_float() {
                let c = TExpr::Cast {
                    value: Box::new(r),
                    from: rt.clone(),
                    to: lt.clone(),
                };
                return Ok((l, c, lt));
            }
        }
        if let Some(c) = coerce(&l, &rt) {
            return Ok((c, r, rt));
        }
        if let Some(c) = coerce(&r, &lt) {
            return Ok((l, c, lt));
        }
        err(
            format!(
                "cannot apply `{}` to {} and {}",
                op.name(),
                lt.print(),
                rt.print()
            ),
            span,
        )
    }

    /// Type rules of the domain operations.
    fn check_domain(&mut self, name: &str, args: &[Expr], span: &Span) -> Result<TExpr, Error> {
        let op = match name {
            "matmul" => DomainOp::MatMul,
            "conv2d" => DomainOp::Conv2d,
            "maxpool2d" => DomainOp::MaxPool2d,
            "relu" => DomainOp::Relu,
            "flatten" => DomainOp::Flatten,
            "row" => DomainOp::Row,
            "argmax" => DomainOp::ArgMax,
            "add_bias" => DomainOp::AddBias,
            "embed" => DomainOp::Embed,
            "embed_seq" => DomainOp::EmbedSeq,
            "transpose" => DomainOp::Transpose,
            "layernorm" => DomainOp::LayerNorm,
            "softmax_causal" => DomainOp::SoftmaxCausal,
            "cross_entropy" => DomainOp::CrossEntropy,
            "softmax" => DomainOp::Softmax,
            "softmax_tri" => DomainOp::SoftmaxTri,
            "sum" => DomainOp::Sum,
            "add" => DomainOp::Add,
            "tmin" => DomainOp::Tmin,
            "tmax" => DomainOp::Tmax,
            "tclamp" => DomainOp::Tclamp,
            "tpow" => DomainOp::Tpow,
            "texp" => DomainOp::Texp,
            "tln" => DomainOp::Tln,
            "mean" => DomainOp::Mean,
            "argmin" => DomainOp::ArgMin,
            "norm1" => DomainOp::Norm1,
            "norm2" => DomainOp::Norm2,
            "var" => DomainOp::Var,
            "std" => DomainOp::Std,
            "dot" => DomainOp::Dot,
            "concat" => DomainOp::Concat,
            "rows" => DomainOp::Rows,
            "reverse" => DomainOp::Reverse,
            "iota" => DomainOp::Iota,
            "linspace" => DomainOp::Linspace,
            "onehot" => DomainOp::Onehot,
            "rand_permutation" => DomainOp::RandPermutation,
            "gather" => DomainOp::Gather,
            "dropout" => DomainOp::Dropout,
            "accuracy" => DomainOp::Accuracy,
            "sample" => DomainOp::Sample,
            _ => unreachable!(),
        };
        let mut targs = Vec::new();
        for a in args {
            targs.push(self.check_expr(a)?);
        }
        if name == "cross_entropy" {
            // The labels are integers, so the shared check does not fit; the
            // method below checks both arguments.
            let (ty, params) = self.check_cross_entropy(&targs, span)?;
            return Ok(TExpr::Domain {
                op: DomainOp::CrossEntropy,
                args: targs,
                params,
                ty,
            });
        }
        // The element type of every tensor argument.
        let elem = |t: &CType| -> Result<CType, Error> {
            match t.elem() {
                Some(e) if *e == CType::Scalar(ScalarKind::F32) => {
                    Ok(CType::Scalar(ScalarKind::F32))
                }
                _ => err(
                    format!("`{}` operates on tensors of f32, found {}", name, t.print()),
                    span,
                ),
            }
        };
        let dims = |t: &CType| -> Result<Vec<i64>, Error> {
            let d = t.dims().unwrap_or(&[]);
            let mut out = Vec::new();
            for dim in d {
                match dim {
                    CDim::Static(v) => out.push(*v),
                    CDim::Dynamic => {
                        return err(format!("`{}` needs statically shaped tensors", name), span)
                    }
                }
            }
            Ok(out)
        };
        let tensor = |t: &CType| -> Result<(Vec<i64>, CType), Error> {
            if !t.is_tensor() {
                return err(
                    format!("`{}` expects tensors, found {}", name, t.print()),
                    span,
                );
            }
            let d = dims(t)?;
            let e = elem(t)?;
            Ok((d, e))
        };
        // A row of a tensor of any element type.
        let row_tensor = |t: &CType| -> Result<(Vec<i64>, CType), Error> {
            if !t.is_tensor() {
                return err(
                    format!("`{}` expects tensors, found {}", name, t.print()),
                    span,
                );
            }
            let d = dims(t)?;
            let e = t.elem().cloned().unwrap();
            Ok((d, e))
        };
        // A tensor of integers, used for the indices of a table.
        let tensor_int = |t: &CType| -> Result<(Vec<i64>, CType), Error> {
            if !t.is_tensor() {
                return err(
                    format!("`{}` expects tensors, found {}", name, t.print()),
                    span,
                );
            }
            let d = dims(t)?;
            match t.elem() {
                Some(e) if *e == CType::Scalar(ScalarKind::I32) => {}
                _ => {
                    return err(
                        format!("`{}` expects a tensor of i32, found {}", name, t.print()),
                        span,
                    )
                }
            }
            Ok((d, CType::Scalar(ScalarKind::I32)))
        };
        // An integer argument; an index is converted to i32.
        let integer_arg = |args: &mut Vec<TExpr>,
                           index: usize,
                           what: &str,
                           span: &Span|
         -> Result<TExpr, Error> {
            let t = args[index].ty();
            match t {
                CType::Scalar(k) if k == ScalarKind::I32 => Ok(args[index].clone()),
                CType::Scalar(k) if k.is_int() => Ok(TExpr::Cast {
                    value: Box::new(args[index].clone()),
                    from: t,
                    to: CType::Scalar(ScalarKind::I32),
                }),
                CType::Index => Ok(TExpr::Cast {
                    value: Box::new(args[index].clone()),
                    from: t,
                    to: CType::Scalar(ScalarKind::I32),
                }),
                _ => err(format!("{} must be an integer", what), span),
            }
        };

        let (ty, params) = match op {
            // Checked by `check_cross_entropy`, which the dispatch calls
            // before this match.
            DomainOp::CrossEntropy => unreachable!(),
            DomainOp::SoftmaxTri => {
                // softmax_tri(x: Tensor<NxNxf32>) -> Tensor<NxNxf32>: the
                // softmax of the rows with a causal mask, the attention of a
                // decoder.
                if targs.len() != 1 {
                    return err("`softmax_tri` takes one matrix", span);
                }
                let (d, e) = tensor(&targs[0].ty())?;
                if d.len() != 2 {
                    return err("`softmax_tri` expects a matrix", span);
                }
                if e != CType::Scalar(ScalarKind::F32) {
                    return err(
                        format!("`softmax_tri` expects a matrix of f32, found {}", e.print()),
                        span,
                    );
                }
                (
                    CType::Tensor {
                        dims: d.iter().map(|v| CDim::Static(*v)).collect(),
                        elem: Box::new(CType::Scalar(ScalarKind::F32)),
                    },
                    vec![],
                )
            }
            DomainOp::Add => {
                // add(a, b): the element wise sum of two tensors of the same
                // shape.
                if targs.len() != 2 {
                    return err("`add` takes two tensors", span);
                }
                let (da, ea) = tensor(&targs[0].ty())?;
                let (db, eb) = tensor(&targs[1].ty())?;
                if ea != CType::Scalar(ScalarKind::F32) || eb != CType::Scalar(ScalarKind::F32) {
                    return err("`add` expects tensors of f32", span);
                }
                if da != db {
                    return err(
                        format!(
                            "`add` expects tensors of the same shape, found {} and {}",
                            targs[0].ty().print(),
                            targs[1].ty().print()
                        ),
                        span,
                    );
                }
                (
                    CType::Tensor {
                        dims: da.iter().map(|v| CDim::Static(*v)).collect(),
                        elem: Box::new(CType::Scalar(ScalarKind::F32)),
                    },
                    vec![],
                )
            }
            DomainOp::Tmin | DomainOp::Tmax => {
                // Two tensors: the element wise minimum/maximum. One tensor:
                // its largest (smallest) element, as a scalar.
                if targs.len() == 1 {
                    let (d, e) = tensor(&targs[0].ty())?;
                    if e != CType::Scalar(ScalarKind::F32) {
                        return err(
                            format!("`{}` expects a tensor of f32, found {}", name, e.print()),
                            span,
                        );
                    }
                    if d.is_empty() {
                        return err(format!("`{}` expects a tensor with elements", name), span);
                    }
                    return Ok(TExpr::Domain {
                        op,
                        args: targs,
                        params: vec![],
                        ty: CType::Scalar(ScalarKind::F32),
                    });
                }
                if targs.len() != 2 {
                    return err(format!("`{}` takes one or two tensors", name), span);
                }
                let (da, ea) = tensor(&targs[0].ty())?;
                let (db, eb) = tensor(&targs[1].ty())?;
                if ea != CType::Scalar(ScalarKind::F32) || eb != CType::Scalar(ScalarKind::F32) {
                    return err(format!("`{}` expects tensors of f32", name), span);
                }
                if da != db {
                    return err(
                        format!("`{}` expects tensors of the same shape", name),
                        span,
                    );
                }
                (
                    CType::Tensor {
                        dims: da.iter().map(|v| CDim::Static(*v)).collect(),
                        elem: Box::new(CType::Scalar(ScalarKind::F32)),
                    },
                    vec![],
                )
            }
            DomainOp::Tclamp | DomainOp::Tpow => {
                let expected = if name == "tclamp" { 3 } else { 2 };
                if targs.len() != expected {
                    return err(format!("`{}` takes {} arguments", name, expected), span);
                }
                let (d, e) = tensor(&targs[0].ty())?;
                if e != CType::Scalar(ScalarKind::F32) {
                    return err(
                        format!("`{}` expects a tensor of f32, found {}", name, e.print()),
                        span,
                    );
                }
                for t in targs[1..].iter_mut() {
                    if t.ty() != CType::Scalar(ScalarKind::F32) {
                        *t = TExpr::Cast {
                            value: Box::new(t.clone()),
                            from: t.ty(),
                            to: CType::Scalar(ScalarKind::F32),
                        };
                    }
                }
                (
                    CType::Tensor {
                        dims: d.iter().map(|v| CDim::Static(*v)).collect(),
                        elem: Box::new(CType::Scalar(ScalarKind::F32)),
                    },
                    vec![],
                )
            }
            DomainOp::Texp | DomainOp::Tln | DomainOp::Reverse => {
                if targs.len() != 1 {
                    return err(format!("`{}` takes one tensor", name), span);
                }
                let (d, e) = tensor(&targs[0].ty())?;
                if e != CType::Scalar(ScalarKind::F32) {
                    return err(
                        format!("`{}` expects a tensor of f32, found {}", name, e.print()),
                        span,
                    );
                }
                (
                    CType::Tensor {
                        dims: d.iter().map(|v| CDim::Static(*v)).collect(),
                        elem: Box::new(CType::Scalar(ScalarKind::F32)),
                    },
                    vec![],
                )
            }
            DomainOp::Mean | DomainOp::Norm1 | DomainOp::Norm2 | DomainOp::Var | DomainOp::Std => {
                if targs.len() != 1 {
                    return err(format!("`{}` takes one tensor", name), span);
                }
                let (d, e) = tensor(&targs[0].ty())?;
                if e != CType::Scalar(ScalarKind::F32) {
                    return err(
                        format!("`{}` expects a tensor of f32, found {}", name, e.print()),
                        span,
                    );
                }
                if d.is_empty() {
                    return err(format!("`{}` expects a tensor with elements", name), span);
                }
                (CType::Scalar(ScalarKind::F32), vec![])
            }
            DomainOp::ArgMin => {
                if targs.len() != 1 {
                    return err("`argmin` takes one tensor", span);
                }
                let (d, e) = tensor(&targs[0].ty())?;
                if e != CType::Scalar(ScalarKind::F32) {
                    return err(
                        format!("`argmin` expects a tensor of f32, found {}", e.print()),
                        span,
                    );
                }
                if d.is_empty() {
                    return err("`argmin` expects a tensor with elements", span);
                }
                (CType::Scalar(ScalarKind::I32), vec![])
            }
            DomainOp::Dot => {
                if targs.len() != 2 {
                    return err("`dot` takes two tensors", span);
                }
                let (da, ea) = tensor(&targs[0].ty())?;
                let (db, eb) = tensor(&targs[1].ty())?;
                if ea != CType::Scalar(ScalarKind::F32) || eb != CType::Scalar(ScalarKind::F32) {
                    return err("`dot` expects tensors of f32", span);
                }
                if da != db {
                    return err("`dot` expects tensors of the same shape", span);
                }
                (CType::Scalar(ScalarKind::F32), vec![])
            }
            DomainOp::Concat => {
                if targs.len() != 2 {
                    return err("`concat` takes two tensors", span);
                }
                let (da, ea) = tensor(&targs[0].ty())?;
                let (db, eb) = tensor(&targs[1].ty())?;
                if ea != CType::Scalar(ScalarKind::F32) || eb != CType::Scalar(ScalarKind::F32) {
                    return err("`concat` expects tensors of f32", span);
                }
                if da.len() < 2 || da.len() != db.len() || da[1..] != db[1..] {
                    return err(
                        "`concat` expects two tensors of the same rank and the same                          shape beyond the first axis"
                            .to_string(),
                        span,
                    );
                }
                (
                    CType::Tensor {
                        dims: std::iter::once(CDim::Static(da[0] + db[0]))
                            .chain(da[1..].iter().map(|v| CDim::Static(*v)))
                            .collect(),
                        elem: Box::new(CType::Scalar(ScalarKind::F32)),
                    },
                    vec![],
                )
            }
            DomainOp::Rows => {
                if targs.len() != 3 {
                    return err("`rows` takes a tensor, a start and a count", span);
                }
                let (d, e) = row_tensor(&targs[0].ty())?;
                if d.len() < 1 {
                    return err("`rows` expects a tensor of rank 1 or more", span);
                }
                let start = integer_arg(&mut targs, 1, "the start of `rows`", span)?;
                let count = integer_arg(&mut targs, 2, "the count of `rows`", span)?;
                let _ = start;
                // A literal count gives a static shape.
                let first = match &count {
                    TExpr::ConstInt(v, _) => CDim::Static(*v),
                    _ => CDim::Dynamic,
                };
                (
                    CType::Tensor {
                        dims: std::iter::once(first)
                            .chain(d[1..].iter().map(|v| CDim::Static(*v)))
                            .collect(),
                        elem: Box::new(e),
                    },
                    vec![],
                )
            }
            // Handled by `check_reshape`/`check_call`, which the dispatch
            // calls directly.
            DomainOp::Reshape
            | DomainOp::RandUniform
            | DomainOp::RandNormal => unreachable!(),
            DomainOp::Iota => {
                // iota(n: i32) -> Tensor<nxi32>: a literal n gives a static
                // shape.
                if targs.len() != 1 {
                    return err("`iota` takes the number of the elements", span);
                }
                let n = integer_arg(&mut targs, 0, "the count of `iota`", span)?;
                let dims = match &n {
                    TExpr::ConstInt(v, _) => vec![CDim::Static(*v)],
                    _ => vec![CDim::Dynamic],
                };
                (
                    CType::Tensor {
                        dims,
                        elem: Box::new(CType::Scalar(ScalarKind::I32)),
                    },
                    vec![],
                )
            }
            DomainOp::Linspace => {
                // linspace(a: f32, b: f32, n: i32) -> Tensor<nxf32>
                if targs.len() != 3 {
                    return err("`linspace` takes a start, an end and a count", span);
                }
                let start = self.call_f32_scalar(&mut targs, 0, span)?;
                let end = self.call_f32_scalar(&mut targs, 1, span)?;
                let _ = (start, end);
                let n = integer_arg(&mut targs, 2, "the count of `linspace`", span)?;
                let dims = match &n {
                    TExpr::ConstInt(v, _) => vec![CDim::Static(*v)],
                    _ => vec![CDim::Dynamic],
                };
                (
                    CType::Tensor {
                        dims,
                        elem: Box::new(CType::Scalar(ScalarKind::F32)),
                    },
                    vec![],
                )
            }
            DomainOp::Onehot => {
                // onehot(ids: Tensor<Mxi32>, vocab: i32) -> Tensor<MxVxf32>;
                // a literal vocab gives a static shape.
                if targs.len() != 2 {
                    return err("`onehot` takes the tokens and the vocabulary size", span);
                }
                let (d, e) = row_tensor(&targs[0].ty())?;
                if e != CType::Scalar(ScalarKind::I32) {
                    return err(
                        format!("`onehot` expects a tensor of i32, found {}", e.print()),
                        span,
                    );
                }
                if d.len() != 1 {
                    return err("`onehot` expects a tensor of rank 1", span);
                }
                let vocab = integer_arg(&mut targs, 1, "the vocabulary size", span)?;
                let vdim = match &vocab {
                    TExpr::ConstInt(v, _) => CDim::Static(*v),
                    _ => CDim::Dynamic,
                };
                (
                    CType::Tensor {
                        dims: vec![CDim::Static(d[0]), vdim],
                        elem: Box::new(CType::Scalar(ScalarKind::F32)),
                    },
                    vec![],
                )
            }
            DomainOp::RandPermutation => {
                // rand_permutation(n: i32, seed: i32) -> Tensor<nxi32>
                if targs.len() != 2 {
                    return err("`rand_permutation` takes a count and a seed", span);
                }
                let n = integer_arg(&mut targs, 0, "the count", span)?;
                let seed = integer_arg(&mut targs, 1, "the seed", span)?;
                let _ = seed;
                let dims = match &n {
                    TExpr::ConstInt(v, _) => vec![CDim::Static(*v)],
                    _ => vec![CDim::Dynamic],
                };
                (
                    CType::Tensor {
                        dims,
                        elem: Box::new(CType::Scalar(ScalarKind::I32)),
                    },
                    vec![],
                )
            }
            DomainOp::Gather => {
                // gather(t: Tensor<Rx...E>, idx: Tensor<Kxi32>) -> the rows,
                // with the element type of the tensor.
                if targs.len() != 2 {
                    return err("`gather` takes a tensor and the indices", span);
                }
                let (d, e) = row_tensor(&targs[0].ty())?;
                let (di, ei) = row_tensor(&targs[1].ty())?;
                if ei != CType::Scalar(ScalarKind::I32) {
                    return err(
                        format!("`gather` expects indices of i32, found {}", ei.print()),
                        span,
                    );
                }
                if di.len() != 1 {
                    return err("`gather` expects the indices as a tensor of rank 1", span);
                }
                (
                    CType::Tensor {
                        dims: std::iter::once(CDim::Static(di[0]))
                            .chain(d[1..].iter().map(|v| CDim::Static(*v)))
                            .collect(),
                        elem: Box::new(e),
                    },
                    vec![],
                )
            }
            DomainOp::Dropout => {
                // dropout(x: Tensor<...>, seed: i32, rate: f32)
                if targs.len() != 3 {
                    return err("`dropout` takes a tensor, a seed and a rate", span);
                }
                let (d, e) = tensor(&targs[0].ty())?;
                if e != CType::Scalar(ScalarKind::F32) {
                    return err(
                        format!("`dropout` expects a tensor of f32, found {}", e.print()),
                        span,
                    );
                }
                let seed = integer_arg(&mut targs, 1, "the seed", span)?;
                let rate = self.call_f32_scalar(&mut targs, 2, span)?;
                let _ = (seed, rate);
                (
                    CType::Tensor {
                        dims: d.iter().map(|v| CDim::Static(*v)).collect(),
                        elem: Box::new(CType::Scalar(ScalarKind::F32)),
                    },
                    vec![],
                )
            }
            DomainOp::Accuracy => {
                // accuracy(logits: Tensor<MxVxf32>, labels: Tensor<Mxi32>) -> f32
                if targs.len() != 2 {
                    return err("`accuracy` takes the logits and the labels", span);
                }
                let (d, e) = tensor(&targs[0].ty())?;
                if e != CType::Scalar(ScalarKind::F32) || d.len() != 2 {
                    return err(
                        format!("`accuracy` expects the logits as a matrix of f32, found {}", e.print()),
                        span,
                    );
                }
                let (dl, el) = row_tensor(&targs[1].ty())?;
                if el != CType::Scalar(ScalarKind::I32) || dl.len() != 1 || dl[0] != d[0] {
                    return err(
                        "`accuracy` expects one label per row of the logits".to_string(),
                        span,
                    );
                }
                (CType::Scalar(ScalarKind::F32), vec![])
            }
            DomainOp::Sum => {
                // sum(t: Tensor<...xf32>) -> f32
                if targs.len() != 1 {
                    return err("`sum` takes one tensor", span);
                }
                let (d, e) = tensor(&targs[0].ty())?;
                if e != CType::Scalar(ScalarKind::F32) {
                    return err(
                        format!("`sum` expects a tensor of f32, found {}", e.print()),
                        span,
                    );
                }
                if d.is_empty() {
                    return err("`sum` expects a tensor with at least one dimension", span);
                }
                (CType::Scalar(ScalarKind::F32), vec![])
            }
            DomainOp::MatMul => {
                if targs.len() != 2 {
                    return err("`matmul` takes two matrices", span);
                }
                let (da, _) = tensor(&targs[0].ty())?;
                let (db, _) = tensor(&targs[1].ty())?;
                if da.len() != 1 && da.len() != 2 {
                    return err("`matmul` operates on rank-1 or rank-2 tensors", span);
                }
                if db.len() != 1 && db.len() != 2 {
                    return err("`matmul` operates on rank-1 or rank-2 tensors", span);
                }
                // A vector is a matrix with a single row (or column).
                let (m, k) = if da.len() == 1 {
                    (1, da[0])
                } else {
                    (da[0], da[1])
                };
                if k != db[0] {
                    return err(
                        format!(
                            "`matmul` cannot multiply {}x{} by {}x{}",
                            m,
                            k,
                            db[0],
                            if db.len() == 2 { db[1] } else { 1 }
                        ),
                        span,
                    );
                }
                // A vector times a matrix, a matrix times a vector or two
                // matrices.
                let dims = match (da.len(), db.len()) {
                    (1, 2) => vec![CDim::Static(db[1])],
                    (2, 2) => vec![CDim::Static(m), CDim::Static(db[1])],
                    (2, 1) => vec![CDim::Static(m)],
                    _ => return err("`matmul` needs a vector or a matrix", span),
                };
                (
                    CType::Tensor {
                        dims,
                        elem: Box::new(CType::Scalar(ScalarKind::F32)),
                    },
                    vec![],
                )
            }
            DomainOp::Conv2d => {
                if targs.len() != 3 {
                    return err("`conv2d` takes an image, the filters and the bias", span);
                }
                let (dx, _) = tensor(&targs[0].ty())?;
                let (dw, _) = tensor(&targs[1].ty())?;
                let (db, _) = tensor(&targs[2].ty())?;
                if (dx.len() != 2 && dx.len() != 3) || dw.len() != 4 || db.len() != 1 {
                    return err(
                        "`conv2d` expects an image (1xHxW), filters (Ox1xKxK) and a bias (O)",
                        span,
                    );
                }
                // An image without the channel dimension has a single channel.
                let (channels, h, w) = if dx.len() == 2 {
                    (1, dx[0], dx[1])
                } else {
                    (dx[0], dx[1], dx[2])
                };
                if channels != 1 || dw[1] != 1 {
                    return err("`conv2d` currently supports a single input channel", span);
                }
                if dw[0] != db[0] {
                    return err("the number of filters and the bias do not match", span);
                }
                (
                    CType::Tensor {
                        dims: vec![CDim::Static(dw[0]), CDim::Static(h), CDim::Static(w)],
                        elem: Box::new(CType::Scalar(ScalarKind::F32)),
                    },
                    vec![],
                )
            }
            DomainOp::MaxPool2d => {
                if targs.len() != 2 {
                    return err("`maxpool2d` takes a tensor and a window size", span);
                }
                let (d, _) = tensor(&targs[0].ty())?;
                let k = match &targs[1] {
                    TExpr::ConstInt(v, _) => *v,
                    _ => return err("the window size of `maxpool2d` must be a constant", span),
                };
                if d.len() != 3 {
                    return err("`maxpool2d` expects a tensor of rank 3", span);
                }
                if k <= 0 || d[1] % k != 0 || d[2] % k != 0 {
                    return err("the window size of `maxpool2d` must divide the image", span);
                }
                (
                    CType::Tensor {
                        dims: vec![
                            CDim::Static(d[0]),
                            CDim::Static(d[1] / k),
                            CDim::Static(d[2] / k),
                        ],
                        elem: Box::new(CType::Scalar(ScalarKind::F32)),
                    },
                    vec![k],
                )
            }
            DomainOp::Relu => {
                if targs.len() != 1 {
                    return err("`relu` takes one tensor", span);
                }
                let (d, e) = tensor(&targs[0].ty())?;
                (
                    CType::Tensor {
                        dims: d.iter().map(|v| CDim::Static(*v)).collect(),
                        elem: Box::new(e),
                    },
                    vec![],
                )
            }
            DomainOp::Flatten => {
                if targs.len() != 1 {
                    return err("`flatten` takes one tensor", span);
                }
                let (d, e) = tensor(&targs[0].ty())?;
                let total: i64 = d.iter().product();
                (
                    CType::Tensor {
                        dims: vec![CDim::Static(total)],
                        elem: Box::new(e),
                    },
                    vec![],
                )
            }
            DomainOp::Row => {
                if targs.len() != 2 {
                    return err("`row` takes a tensor and an index", span);
                }
                // A row keeps the element type: the rows of a table of
                // indices are indices too.
                let (d, e) = row_tensor(&targs[0].ty())?;
                if d.len() < 2 {
                    return err("`row` expects a tensor of rank 2 or more", span);
                }
                // The index may be dynamic; it becomes a runtime argument.
                let index = match targs[1].ty() {
                    CType::Index => targs[1].clone(),
                    t if matches!(t, CType::Scalar(k) if k.is_int()) => TExpr::Cast {
                        value: Box::new(targs[1].clone()),
                        from: t,
                        to: CType::Index,
                    },
                    _ => return err("the index of `row` must be an integer", span),
                };
                targs[1] = index;
                (
                    CType::Tensor {
                        dims: d[1..].iter().map(|v| CDim::Static(*v)).collect(),
                        elem: Box::new(e),
                    },
                    vec![],
                )
            }
            DomainOp::ArgMax => {
                if targs.len() != 1 {
                    return err("`argmax` takes one tensor", span);
                }
                let (d, _) = tensor(&targs[0].ty())?;
                if d.len() != 1 {
                    return err("`argmax` expects a tensor of rank 1", span);
                }
                (CType::Scalar(ScalarKind::I32), vec![])
            }
            DomainOp::Embed => {
                // embed(w: Tensor<VxWxf32>, id: i32) -> Tensor<Wxf32>
                if targs.len() != 2 {
                    return err("`embed` takes a table and an index", span);
                }
                let (dw, _) = tensor(&targs[0].ty())?;
                if dw.len() != 2 {
                    return err("`embed` expects a table of rank 2", span);
                }
                let id = integer_arg(&mut targs, 1, "the index of `embed`", span)?;
                targs[1] = id;
                (
                    CType::Tensor {
                        dims: vec![CDim::Static(dw[1])],
                        elem: Box::new(CType::Scalar(ScalarKind::F32)),
                    },
                    vec![],
                )
            }
            DomainOp::EmbedSeq => {
                // embed_seq(w: Tensor<VxWxf32>, ids: Tensor<Nxi32>) -> Tensor<NxWxf32>
                if targs.len() != 2 {
                    return err("`embed_seq` takes a table and a tensor of indices", span);
                }
                let (dw, _) = tensor(&targs[0].ty())?;
                if dw.len() != 2 {
                    return err("`embed_seq` expects a table of rank 2", span);
                }
                let (di, _) = tensor_int(&targs[1].ty())?;
                if di.len() != 1 {
                    return err("`embed_seq` expects a tensor of rank 1", span);
                }
                (
                    CType::Tensor {
                        dims: vec![CDim::Static(di[0]), CDim::Static(dw[1])],
                        elem: Box::new(CType::Scalar(ScalarKind::F32)),
                    },
                    vec![],
                )
            }
            DomainOp::Transpose => {
                // transpose(x: Tensor<MxNxf32>) -> Tensor<NxMxf32>
                if targs.len() != 1 {
                    return err("`transpose` takes one matrix", span);
                }
                let (d, e) = tensor(&targs[0].ty())?;
                if d.len() != 2 {
                    return err("`transpose` expects a matrix", span);
                }
                (
                    CType::Tensor {
                        dims: vec![CDim::Static(d[1]), CDim::Static(d[0])],
                        elem: Box::new(e),
                    },
                    vec![],
                )
            }
            DomainOp::LayerNorm => {
                // layernorm(x: Tensor<MxNxf32>, gamma: Tensor<Nxf32>, beta: Tensor<Nxf32>)
                if targs.len() != 3 {
                    return err("`layernorm` takes a matrix, a scale and a shift", span);
                }
                let (dx, e) = tensor(&targs[0].ty())?;
                let (dg, _) = tensor(&targs[1].ty())?;
                let (db, _) = tensor(&targs[2].ty())?;
                if dx.len() != 2 {
                    return err("`layernorm` expects a matrix", span);
                }
                if dg.len() != 1 || db.len() != 1 {
                    return err("`layernorm` expects a scale and a shift of rank 1", span);
                }
                if dx[1] != dg[0] || dg[0] != db[0] {
                    return err(
                        format!(
                            "`layernorm` cannot normalize {} columns with a scale of {} elements",
                            dx[1], dg[0]
                        ),
                        span,
                    );
                }
                (
                    CType::Tensor {
                        dims: dx.iter().map(|v| CDim::Static(*v)).collect(),
                        elem: Box::new(e),
                    },
                    vec![],
                )
            }
            DomainOp::SoftmaxCausal => {
                // softmax_causal(scores: Tensor<TxTxf32>, length: i32)
                if targs.len() != 2 {
                    return err("`softmax_causal` takes a matrix and a length", span);
                }
                let (d, e) = tensor(&targs[0].ty())?;
                if d.len() != 2 || d[0] != d[1] {
                    return err("`softmax_causal` expects a square matrix", span);
                }
                let length = integer_arg(&mut targs, 1, "the length of `softmax_causal`", span)?;
                targs[1] = length;
                (
                    CType::Tensor {
                        dims: d.iter().map(|v| CDim::Static(*v)).collect(),
                        elem: Box::new(e),
                    },
                    vec![],
                )
            }
            DomainOp::Softmax => {
                // softmax(x: Tensor<MxNxf32>) -> Tensor<MxNxf32>
                if targs.len() != 1 {
                    return err("`softmax` takes a matrix", span);
                }
                let (d, e) = tensor(&targs[0].ty())?;
                if d.len() != 2 {
                    return err("`softmax` expects a matrix", span);
                }
                if e != CType::Scalar(ScalarKind::F32) {
                    return err(
                        format!("`softmax` expects a matrix of f32, found {}", e.print()),
                        span,
                    );
                }
                (
                    CType::Tensor {
                        dims: d.iter().map(|v| CDim::Static(*v)).collect(),
                        elem: Box::new(CType::Scalar(ScalarKind::F32)),
                    },
                    vec![],
                )
            }
            DomainOp::Sample => {
                // sample(logits: Tensor<Nxf32>, temperature: f32, seed: i32) -> i32
                if targs.len() != 3 {
                    return err("`sample` takes logits, a temperature and a seed", span);
                }
                let (d, _) = tensor(&targs[0].ty())?;
                if d.len() != 1 {
                    return err("`sample` expects a tensor of rank 1", span);
                }
                let temperature = match targs[1].ty() {
                    CType::Scalar(k) if k.is_float() => {
                        // The runtime works with single precision floats.
                        if k == ScalarKind::F32 {
                            targs[1].clone()
                        } else {
                            TExpr::Cast {
                                value: Box::new(targs[1].clone()),
                                from: CType::Scalar(k),
                                to: CType::Scalar(ScalarKind::F32),
                            }
                        }
                    }
                    t => {
                        return err(
                            format!("the temperature must be a float, found {}", t.print()),
                            span,
                        )
                    }
                };
                targs[1] = temperature;
                let seed = integer_arg(&mut targs, 2, "the seed of `sample`", span)?;
                targs[2] = seed;
                (CType::Scalar(ScalarKind::I32), vec![])
            }
            DomainOp::AddBias => {
                if targs.len() != 2 {
                    return err("`add_bias` takes a matrix and a bias vector", span);
                }
                let (d, e) = tensor(&targs[0].ty())?;
                let (db, _) = tensor(&targs[1].ty())?;
                if d.len() != 1 && d.len() != 2 {
                    return err("`add_bias` expects a vector or a matrix", span);
                }
                if db.len() != 1 {
                    return err("`add_bias` expects a vector as the bias", span);
                }
                // A vector is a matrix with a single row.
                let n = if d.len() == 1 { d[0] } else { d[1] };
                if n != db[0] {
                    return err(
                        format!(
                            "the bias has {} elements but the matrix has {} columns",
                            db[0], n
                        ),
                        span,
                    );
                }
                (
                    CType::Tensor {
                        dims: d.iter().map(|v| CDim::Static(*v)).collect(),
                        elem: Box::new(e),
                    },
                    vec![],
                )
            }
        };
        Ok(TExpr::Domain {
            op,
            args: targs,
            params,
            ty,
        })
    }


    /// Coerces one argument of a typed list to f32, in place.
    fn call_f32_scalar(
        &mut self,
        targs: &mut [TExpr],
        index: usize,
        span: &Span,
    ) -> Result<TExpr, Error> {
        let t = targs[index].ty();
        if !t.is_numeric() {
            return err(format!("expects a number, found {}", t.print()), span);
        }
        if t == CType::Scalar(ScalarKind::F32) {
            return Ok(targs[index].clone());
        }
        let c = TExpr::Cast {
            value: Box::new(targs[index].clone()),
            from: t,
            to: CType::Scalar(ScalarKind::F32),
        };
        targs[index] = c.clone();
        Ok(c)
    }

    /// A call of a scalar runtime function: every argument is a number,
    /// coerced to f32, and the result is f32.
    fn call_f32(&mut self, name: &str, args: &[Expr], span: &Span) -> Result<TExpr, Error> {
        let mut targs = Vec::new();
        for a in args {
            let a = self.check_expr(a)?;
            let t = a.ty();
            if !t.is_numeric() {
                return err(
                    format!("`{}` expects numbers, found {}", name, t.print()),
                    span,
                );
            }
            let a = if t == CType::Scalar(ScalarKind::F32) {
                a
            } else {
                TExpr::Cast {
                    value: Box::new(a),
                    from: t,
                    to: CType::Scalar(ScalarKind::F32),
                }
            };
            targs.push(a);
        }
        Ok(TExpr::Call {
            name: format!("mlr_{}", name),
            type_args: vec![],
            args: targs,
            ret: CType::Scalar(ScalarKind::F32),
        })
    }

    /// `reshape<Tensor<...>>(x)`: the same elements in a new shape; the
    /// number of the elements must be the same.
    fn check_reshape(
        &mut self,
        type_args: &[SrcType],
        args: &[Expr],
        span: &Span,
    ) -> Result<TExpr, Error> {
        if type_args.len() != 1 {
            return err("`reshape` takes exactly one type argument", span);
        }
        if args.len() != 1 {
            return err("`reshape` takes exactly one tensor", span);
        }
        let ty = self.resolve(&type_args[0], span);
        let a = self.check_expr(&args[0])?;
        let source = match a.ty().dims() {
            Some(dims) => {
                let mut count = 1i64;
                for dim in dims {
                    match dim {
                        CDim::Static(v) => count *= v,
                        CDim::Dynamic => {
                            return err(
                                "`reshape` needs a tensor with static dimensions",
                                span,
                            )
                        }
                    }
                }
                count
            }
            None => {
                return err(
                    format!("`reshape` expects a tensor, found {}", a.ty().print()),
                    span,
                )
            }
        };
        let target = match ty.dims() {
            Some(dims) => {
                let mut count = 1i64;
                for dim in dims {
                    match dim {
                        CDim::Static(v) => count *= v,
                        CDim::Dynamic => {
                            return err(
                                "`reshape` needs a type with static dimensions",
                                span,
                            )
                        }
                    }
                }
                count
            }
            None => {
                return err("`reshape` expects a tensor type", span);
            }
        };
        if source != target {
            return err(
                format!(
                    "`reshape` cannot turn {} elements into {}",
                    source, target
                ),
                span,
            );
        }
        Ok(TExpr::Domain {
            op: DomainOp::Reshape,
            args: vec![a],
            params: vec![],
            ty,
        })
    }

    /// `cross_entropy(logits, labels)`: the mean cross entropy of the rows;
    /// the labels are integers, so the arguments are checked here and not by
    /// the shared tensor rules.
    fn check_cross_entropy(
        &mut self,
        targs: &[TExpr],
        span: &Span,
    ) -> Result<(CType, Vec<i64>), Error> {
        if targs.len() != 2 {
            return err("`cross_entropy` takes logits and labels", span);
        }
        let lt = targs[0].ty();
        let (d, e) = match (lt.dims(), lt.elem()) {
            (Some(dims), Some(elem)) => {
                let mut dims_out = Vec::new();
                for dim in dims {
                    match dim {
                        CDim::Static(v) => dims_out.push(*v),
                        CDim::Dynamic => {
                            return err(
                                "`cross_entropy` needs statically shaped tensors",
                                span,
                            )
                        }
                    }
                }
                (dims_out, elem.clone())
            }
            _ => {
                return err(
                    format!("`cross_entropy` expects the logits as a matrix, found {}", lt.print()),
                    span,
                )
            }
        };
        // A matrix of M rows, or a single sample (a vector of logits).
        if d.len() != 2 && d.len() != 1 {
            return err("`cross_entropy` expects the logits as a matrix", span);
        }
        if e != CType::Scalar(ScalarKind::F32) {
            return err(
                format!("`cross_entropy` expects logits of f32, found {}", e.print()),
                span,
            );
        }
        let yt = targs[1].ty();
        let (dl, el) = match (yt.dims(), yt.elem()) {
            (Some(dims), Some(elem)) => {
                let mut dims_out = Vec::new();
                for dim in dims {
                    match dim {
                        CDim::Static(v) => dims_out.push(*v),
                        CDim::Dynamic => {
                            return err(
                                "`cross_entropy` needs statically shaped tensors",
                                span,
                            )
                        }
                    }
                }
                (dims_out, elem.clone())
            }
            _ => {
                return err(
                    format!("`cross_entropy` expects labels as a tensor, found {}", yt.print()),
                    span,
                )
            }
        };
        if el != CType::Scalar(ScalarKind::I32) {
            return err(
                format!("`cross_entropy` expects labels of i32, found {}", el.print()),
                span,
            );
        }
        let rows = if d.len() == 1 { 1 } else { d[0] };
        if dl.len() != 1 || dl[0] != rows {
            return err(
                format!(
                    "the labels have {} element(s) but the logits have {} rows",
                    dl.first().copied().unwrap_or(0),
                    rows
                ),
                span,
            );
        }
        Ok((CType::Scalar(ScalarKind::F32), vec![]))
    }

    fn check_call(
        &mut self,
        name: &str,
        type_args: &[SrcType],
        args: &[Expr],
        span: &Span,
    ) -> Result<TExpr, Error> {
        // Builtins.
        match name {
            "mlr.load" => {
                if type_args.len() != 1 {
                    return err("`mlr.load` takes exactly one type argument", span);
                }
                if args.is_empty() || args.len() > 2 {
                    return err(
                        "`mlr.load` takes a file name and, for a checkpoint, the name of a tensor",
                        span,
                    );
                }
                let path = match &args[0] {
                    Expr::Str(text, _) => text.clone(),
                    _ => return err("`mlr.load` expects a file name", span),
                };
                // The name of the tensor in a checkpoint.
                let key = match args.get(1) {
                    None => None,
                    Some(Expr::Str(text, _)) => Some(text.clone()),
                    Some(_) => return err("the name of a tensor must be a string", span),
                };
                let ty = self.resolve(&type_args[0], span);
                if !ty.is_tensor() {
                    return err("`mlr.load` expects a tensor type", span);
                }
                return Ok(TExpr::Load { path, key, ty });
            }
            "mlr.load_runtime" => {
                if type_args.len() != 1 {
                    return err("`mlr.load_runtime` takes exactly one type argument", span);
                }
                if args.is_empty() || args.len() > 2 {
                    return err(
                        "`mlr.load_runtime` takes the path of the file",
                        span,
                    );
                }
                let path = match &args[0] {
                    Expr::Str(text, _) => text.clone(),
                    _ => {
                        return err(
                            "`mlr.load_runtime` expects the path as a string literal",
                            span,
                        )
                    }
                };
                let ty = self.resolve(&type_args[0], span);
                // The elements are f32 or i32; the shape must be static (the
                // runtime checks it against the header of the file).
                match ty.dims() {
                    Some(dims) => {
                        for dim in dims {
                            if matches!(dim, CDim::Dynamic) {
                                return err(
                                    "`mlr.load_runtime` needs a type with static dimensions",
                                    span,
                                );
                            }
                        }
                    }
                    None => {
                        return err(
                            format!(
                                "`mlr.load_runtime` expects a tensor type, found {}",
                                ty.print()
                            ),
                            span,
                        )
                    }
                }
                match ty.elem() {
                    Some(CType::Scalar(ScalarKind::F32))
                    | Some(CType::Scalar(ScalarKind::I32)) => {}
                    _ => {
                        return err(
                            format!(
                                "`mlr.load_runtime` expects f32 or i32 elements, found {}",
                                ty.print()
                            ),
                            span,
                        )
                    }
                }
                return Ok(TExpr::Sys {
                    kind: SysKind::LoadRuntime,
                    cmd: path,
                    args: vec![],
                    ty,
                });
            }
            "exec_tensor" => {
                if type_args.len() != 1 {
                    return err("`exec_tensor` takes exactly one type argument", span);
                }
                if args.is_empty() {
                    return err("`exec_tensor` takes a command and its arguments", span);
                }
                let cmd = match &args[0] {
                    Expr::Str(text, _) => text.clone(),
                    _ => {
                        return err(
                            "`exec_tensor` expects the command as a string literal",
                            span,
                        )
                    }
                };
                let ty = self.resolve(&type_args[0], span);
                if !ty.is_tensor() {
                    return err("`exec_tensor` expects a tensor type", span);
                }
                let mut targs = Vec::new();
                for a in &args[1..] {
                    let a = self.check_expr(a)?;
                    if !a.ty().is_numeric() {
                        return err(
                            format!(
                                "`exec_tensor` expects numeric arguments, found {}",
                                a.ty().print()
                            ),
                            span,
                        );
                    }
                    targs.push(a);
                }
                return Ok(TExpr::Sys {
                    kind: SysKind::ExecTensor,
                    cmd,
                    args: targs,
                    ty,
                });
            }
            "exec" | "exec_i" | "exec_f" => {
                if args.is_empty() {
                    return err(format!("`{}` takes a command and its arguments", name), span);
                }
                let cmd = match &args[0] {
                    Expr::Str(text, _) => text.clone(),
                    _ => {
                        return err(
                            format!("`{}` expects the command as a string literal", name),
                            span,
                        )
                    }
                };
                // The placeholders of the command: {k}, counting the escapes.
                let mut slots: i64 = 0;
                let bytes: Vec<char> = cmd.chars().collect();
                let mut i = 0;
                while i < bytes.len() {
                    if bytes[i] == '{' {
                        if i + 1 < bytes.len() && bytes[i + 1] == '{' {
                            i += 2;
                            continue;
                        }
                        let mut k: i64 = 0;
                        let mut j = i + 1;
                        while j < bytes.len() && bytes[j].is_ascii_digit() {
                            k = k * 10 + (bytes[j] as u8 - b'0') as i64;
                            j += 1;
                        }
                        if j < bytes.len() && bytes[j] == '}' && k + 1 > slots {
                            slots = k + 1;
                        }
                    }
                    i += 1;
                }
                if slots != args.len() as i64 - 1 {
                    return err(
                        format!(
                            "`{}` has {} placeholder(s) but {} argument(s) were given",
                            name,
                            slots,
                            args.len() - 1
                        ),
                        span,
                    );
                }
                let mut targs = Vec::new();
                for a in &args[1..] {
                    let a = self.check_expr(a)?;
                    let t = a.ty();
                    if !t.is_numeric() {
                        return err(
                            format!(
                                "`{}` expects numeric arguments, found {}",
                                name,
                                t.print()
                            ),
                            span,
                        );
                    }
                    targs.push(a);
                }
                let kind = match name {
                    "exec" => SysKind::Exec,
                    "exec_i" => SysKind::ExecI,
                    _ => SysKind::ExecF,
                };
                let ty = match kind {
                    SysKind::Exec => CType::Scalar(ScalarKind::I32),
                    SysKind::ExecI => CType::Scalar(ScalarKind::I32),
                    _ => CType::Scalar(ScalarKind::F32),
                };
                return Ok(TExpr::Sys { kind, cmd, args: targs, ty });
            }
            "assert" => {
                if args.len() != 2 {
                    return err("`assert` takes a condition and a message", span);
                }
                let cmd = match &args[1] {
                    Expr::Str(text, _) => text.clone(),
                    _ => {
                        return err(
                            "the message of `assert` must be a string literal",
                            span,
                        )
                    }
                };
                let cond = self.check_expr(&args[0])?;
                if cond.ty() != CType::Scalar(ScalarKind::I1) {
                    return err(
                        format!(
                            "the condition of `assert` must be a bool, found {}",
                            cond.ty().print()
                        ),
                        span,
                    );
                }
                return Ok(TExpr::Sys {
                    kind: SysKind::Assert,
                    cmd,
                    args: vec![cond],
                    ty: CType::Void,
                });
            }
            "now_ms" => {
                if !args.is_empty() {
                    return err("`now_ms` takes no arguments", span);
                }
                return Ok(TExpr::Call {
                    name: "mlr_now_ms".to_string(),
                    type_args: vec![],
                    args: vec![],
                    ret: CType::Scalar(ScalarKind::I64),
                });
            }
            "exit" => {
                if args.len() != 1 {
                    return err("`exit` takes the exit code", span);
                }
                let a = self.check_expr(&args[0])?;
                let t = a.ty();
                if !t.is_numeric() {
                    return err(
                        format!("`exit` expects a number, found {}", t.print()),
                        span,
                    );
                }
                return Ok(TExpr::Call {
                    name: "mlr_exit".to_string(),
                    type_args: vec![],
                    args: vec![a],
                    ret: CType::Void,
                });
            }
            "value_and_grad" => {
                // `value_and_grad(f, args...)`: the gradient of a function
                // with respect to each of its tensor parameters.
                if args.is_empty() {
                    return err("`value_and_grad` takes a function and its arguments", span);
                }
                let fname = match &args[0] {
                    Expr::Var(n, _) => n.clone(),
                    _ => {
                        return err(
                            "`value_and_grad` expects the name of the function to differentiate",
                            span,
                        )
                    }
                };
                let sig = match self.sigs.get(&fname) {
                    Some(s) => s.clone(),
                    None => return err(format!("unknown function `{}`", fname), span),
                };
                match &sig.ret {
                    Some(CType::Scalar(ScalarKind::F32)) => {}
                    t => {
                        return err(
                            format!(
                                "`value_and_grad` needs a function that returns f32, `{}` returns {}",
                                fname,
                                t.clone().map(|t| t.print()).unwrap_or("()".into())
                            ),
                            span,
                        )
                    }
                }
                if sig.params.len() != args.len() - 1 {
                    return err(
                        format!(
                            "`{}` takes {} argument(s), {} were given",
                            fname,
                            sig.params.len(),
                            args.len() - 1
                        ),
                        span,
                    );
                }
                let mut targs = Vec::new();
                for (a, expected) in args[1..].iter().zip(sig.params.iter()) {
                    let ta = self.check_expr(a)?;
                    let ta = coerce(&ta, expected).ok_or_else(|| Error {
                        msg: format!(
                            "argument of `{}` has type {}, expected {}",
                            fname,
                            ta.ty().print(),
                            expected.print()
                        ),
                        span: span.clone(),
                    })?;
                    targs.push(ta);
                }
                // The result is the loss and a slot per tensor parameter: the
                // gradient of an f32 tensor, or zero for the others.
                let mut fields = vec![(String::new(), CType::Scalar(ScalarKind::F32))];
                for p in &sig.params {
                    if p.is_tensor() {
                        fields.push((String::new(), p.clone()));
                    }
                }
                // An anonymous struct: the names of its fields are unused,
                // the values are unpacked by position.
                let id = self.structs.len();
                self.structs.push(TStructDef {
                    name: format!("@grad:{}", fname),
                    fields,
                });
                return Ok(TExpr::Call {
                    name: format!("@value_and_grad:{}", fname),
                    type_args: vec![],
                    args: targs,
                    ret: CType::Struct(id),
                });
            }
            "mlr.save_pt" => {
                if args.len() < 3 || args.len() % 2 == 0 {
                    return err(
                        "`mlr.save_pt` takes the path of a file and pairs of a tensor and a name",
                        span,
                    );
                }
                let path = match &args[0] {
                    Expr::Str(text, _) => text.clone(),
                    _ => {
                        return err(
                            "`mlr.save_pt` expects the path of the file as a string",
                            span,
                        )
                    }
                };
                let mut tensors = Vec::new();
                let mut keys = Vec::new();
                let mut i = 1;
                while i < args.len() {
                    let t = self.check_expr(&args[i])?;
                    let ty = t.ty();
                    let dims = match ty.dims() {
                        Some(d) => d,
                        None => {
                            return err(
                                format!("`mlr.save_pt` expects tensors, found {}", ty.print()),
                                span,
                            )
                        }
                    };
                    for d in dims {
                        if matches!(d, CDim::Dynamic) {
                            return err(
                                "`mlr.save_pt` needs statically shaped tensors",
                                span,
                            );
                        }
                    }
                    match ty.elem() {
                        Some(CType::Scalar(ScalarKind::F32))
                        | Some(CType::Scalar(ScalarKind::I32)) => {}
                        _ => {
                            return err(
                                format!(
                                    "`mlr.save_pt` cannot write {}, only f32 and i32 tensors",
                                    ty.print()
                                ),
                                span,
                            )
                        }
                    }
                    let key = match &args[i + 1] {
                        Expr::Str(text, _) => text.clone(),
                        _ => return err("the name of a tensor must be a string", span),
                    };
                    tensors.push(t);
                    keys.push(key);
                    i += 2;
                }
                return Ok(TExpr::Save { path, tensors, keys });
            }
            "reshape" => {
                return self.check_reshape(type_args, args, span);
            }
            "rand_uniform" | "rand_normal" => {
                if type_args.len() != 1 {
                    return err(format!("`{}` takes exactly one type argument", name), span);
                }
                if args.len() != 3 {
                    return err(format!("`{}` takes a seed and two values", name), span);
                }
                let ty = self.resolve(&type_args[0], span);
                if !ty.is_tensor() {
                    return err(format!("`{}` expects a tensor type", name), span);
                }
                let mut targs = Vec::new();
                for a in args {
                    targs.push(self.check_expr(a)?);
                }
                let t = targs[0].ty();
                if t != CType::Scalar(ScalarKind::I32) && t != CType::Scalar(ScalarKind::I64) {
                    return err(format!("`{}` expects the seed as an integer", name), span);
                }
                if t == CType::Scalar(ScalarKind::I64) {
                    targs[0] = TExpr::Cast {
                        value: Box::new(targs[0].clone()),
                        from: t,
                        to: CType::Scalar(ScalarKind::I32),
                    };
                }
                let a = self.call_f32_scalar(&mut targs, 1, span)?;
                let b = self.call_f32_scalar(&mut targs, 2, span)?;
                let _ = (a, b);
                return Ok(TExpr::Domain {
                    op: if name == "rand_uniform" {
                        DomainOp::RandUniform
                    } else {
                        DomainOp::RandNormal
                    },
                    args: targs,
                    params: vec![],
                    ty,
                });
            }
            "matmul" | "conv2d" | "maxpool2d" | "relu" | "flatten" | "row" | "argmax"
            | "add_bias" | "embed" | "embed_seq" | "transpose" | "layernorm"
            | "softmax_causal" | "sample" | "cross_entropy" | "softmax" | "sum"
            | "softmax_tri" | "add" | "tmin" | "tmax" | "tclamp" | "tpow" | "texp"
            | "tln" | "mean" | "argmin" | "norm1" | "norm2" | "var" | "std" | "dot"
            | "concat" | "rows" | "reverse" | "iota" | "linspace" | "onehot"
            | "rand_permutation" | "gather"
            | "dropout" | "accuracy"
                if !self.sigs.contains_key(name) =>
            {
                // A function of the program shadows the builtin.
                return self.check_domain(name, args, span);
            }
            "print" => {
                if args.is_empty() {
                    return err("`print` takes at least one argument", span);
                }
                let mut parts = Vec::with_capacity(args.len());
                for arg in args {
                    // A string literal is printed as is; it never becomes a
                    // value, so it does not need a string type.
                    if let Expr::Str(text, _) = arg {
                        parts.push(PrintPart::Str(text.clone()));
                        continue;
                    }
                    let a = self.check_expr(arg)?;
                    let t = a.ty();
                    if !matches!(t, CType::Scalar(_)) {
                        return err(format!("`print` cannot print {}", t.print()), span);
                    }
                    parts.push(PrintPart::Value(a));
                }
                return Ok(TExpr::Print { parts });
            }
            // The scalar functions of the runtime library.
            // The scalar functions of the runtime library; every one has a
            // fixed signature over f32 (or bool for the predicates).
            "exp" | "ln" | "sqrt" | "tanh" | "sigmoid" | "gelu" | "abs" | "floor"
            | "ceil" | "round" | "sign" => {
                if args.len() != 1 {
                    return err(format!("`{}` takes exactly one argument", name), span);
                }
                return self.call_f32(name, args, span);
            }
            "is_nan" | "is_finite" => {
                if args.len() != 1 {
                    return err(format!("`{}` takes exactly one argument", name), span);
                }
                let a = self.call_f32(name, args, span)?;
                let ret = match a {
                    TExpr::Call { mut args, .. } => {
                        args.clear();
                        TExpr::Call {
                            name: format!("mlr_{}", name),
                            type_args: vec![],
                            args,
                            ret: CType::Scalar(ScalarKind::I1),
                        }
                    }
                    _ => unreachable!(),
                };
                return Ok(ret);
            }
            "pow" | "min" | "max" => {
                if args.len() != 2 {
                    return err(format!("`{}` takes exactly two arguments", name), span);
                }
                return self.call_f32(name, args, span);
            }
            "clamp" => {
                if args.len() != 3 {
                    return err(format!("`{}` takes exactly three arguments", name), span);
                }
                return self.call_f32(name, args, span);
            }
            "pi" | "e" => {
                if !args.is_empty() {
                    return err(format!("`{}` takes no arguments", name), span);
                }
                let value = if name == "pi" {
                    std::f64::consts::PI
                } else {
                    std::f64::consts::E
                };
                return Ok(TExpr::ConstFloat(value, CType::Scalar(ScalarKind::F32)));
            }
            "print_byte" => {
                if args.len() != 1 {
                    return err("`print_byte` takes exactly one argument", span);
                }
                let a = self.check_expr(&args[0])?;
                let t = a.ty();
                if !t.is_int() {
                    return err(
                        format!("`print_byte` expects an integer, found {}", t.print()),
                        span,
                    );
                }
                return Ok(TExpr::Call {
                    name: "mlr_print_byte".into(),
                    type_args: vec![],
                    args: vec![a],
                    ret: CType::Void,
                });
            }
            "tensor.empty" | "tensor.zeros" => {
                if type_args.len() != 1 {
                    return err(format!("`{}` takes exactly one type argument", name), span);
                }
                let ty = self.resolve(&type_args[0], span);
                if !ty.is_tensor() {
                    return err(format!("`{}` expects a tensor type", name), span);
                }
                return Ok(TExpr::Call {
                    name: name.to_string(),
                    type_args: vec![ty.clone()],
                    args: vec![],
                    ret: ty,
                });
            }
            _ => {}
        }
        // User functions.
        let sig = match self.sigs.get(name) {
            Some(s) => s.clone(),
            None => return err(format!("unknown function `{}`", name), span),
        };
        if sig.params.len() != args.len() {
            return err(
                format!(
                    "function `{}` expects {} argument(s), found {}",
                    name,
                    sig.params.len(),
                    args.len()
                ),
                span,
            );
        }
        let mut targs = Vec::new();
        for (a, expected) in args.iter().zip(sig.params.iter()) {
            let ta = self.check_expr(a)?;
            let ta = coerce(&ta, expected).ok_or_else(|| Error {
                msg: format!(
                    "argument of `{}` has type {}, expected {}",
                    name,
                    ta.ty().print(),
                    expected.print()
                ),
                span: span.clone(),
            })?;
            targs.push(ta);
        }
        Ok(TExpr::Call {
            name: name.to_string(),
            type_args: vec![],
            args: targs,
            ret: sig.ret.unwrap_or(CType::Void),
        })
    }
}

pub fn check(prog: &Program) -> Result<TProgram, Error> {
    Checker::new().check(prog)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_ok(src: &str) -> TProgram {
        let prog = crate::parser::parse(src).expect("parse error");
        check(&prog).expect("type error")
    }

    fn check_err(src: &str) -> String {
        let prog = crate::parser::parse(src).expect("parse error");
        check(&prog).expect_err("expected an error").to_string()
    }

    #[test]
    fn integer_literals_default_to_i32() {
        let p = check_ok("fn f() { let x = 1 }");
        match &p.funcs[0].body.stmts[0] {
            TStmt::Let { ty, value, .. } => {
                assert_eq!(*ty, CType::Scalar(ScalarKind::I32));
                assert!(matches!(value, TExpr::ConstInt(1, _)));
            }
            _ => panic!("expected a let statement"),
        }
    }

    #[test]
    fn literals_are_coerced() {
        let p = check_ok("fn f() { let x: i64 = 1 }");
        match &p.funcs[0].body.stmts[0] {
            TStmt::Let { ty, .. } => assert_eq!(*ty, CType::Scalar(ScalarKind::I64)),
            _ => panic!(),
        }
    }

    #[test]
    fn integers_are_converted_to_floats() {
        // An integer that meets a float is converted, the result is a float.
        let p = check_ok("fn f(a: i32, b: f64) { let x = a + b }");
        match &p.funcs[0].body.stmts[0] {
            TStmt::Let { ty, .. } => {
                assert_eq!(*ty, CType::Scalar(ScalarKind::F64));
            }
            _ => panic!("expected a let statement"),
        }
    }

    #[test]
    fn incompatible_types_are_rejected() {
        let err = check_err("fn f(a: i32, b: bool) { let x = a + b }");
        assert!(err.contains("cannot apply"), "{}", err);
    }

    #[test]
    fn conditions_must_be_bool() {
        let err = check_err("fn f() { if 1 { yield 2 } }");
        assert!(err.contains("must be a bool"), "{}", err);
    }

    #[test]
    fn branches_must_agree() {
        let err = check_err("fn f(c: bool) -> i32 { if c { yield 1 } else { yield 2.0 } }");
        assert!(err.contains("different types"), "{}", err);
    }

    #[test]
    fn map_inputs_must_have_the_same_shape() {
        let err = check_err(
            "fn f(a: Tensor<4xf32>, b: Tensor<8xf32>) -> Tensor<4xf32> { yield map(a, b) |x, y| { yield x + y } }",
        );
        assert!(err.contains("shape mismatch"), "{}", err);
    }

    #[test]
    fn dynamic_dimensions_are_unified_with_static_ones() {
        let p = check_ok(
            "fn f(a: Tensor<?xf32>, b: Tensor<4xf32>) -> Tensor<4xf32> { yield map(a, b) |x, y| { yield x + y } }",
        );
        match &p.funcs[0].body.tail {
            Some(values) => match &values[0] {
                TExpr::Map {
                    ty: CType::Tensor { dims, .. },
                    ..
                } => assert_eq!(dims[0], CDim::Static(4)),
                _ => panic!(),
            },
            None => panic!(),
        }
    }
}
