//! The MLR abstract syntax tree.
//!
//! Every compound construct in the language is an *expression* with a region:
//! `if`, `for ... iter(..)`, `map` and `reduce`. A region is a list of
//! statements followed by an optional `yield`, exactly like an MLIR region
//! ends with a terminator.

use crate::ir::ScalarKind;
use crate::lexer::Span;

#[derive(Clone, Debug, PartialEq)]
pub enum SrcDim {
    Static(i64),
    Dynamic,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SrcType {
    Scalar(ScalarKind),
    Index,
    Tensor {
        dims: Vec<SrcDim>,
        elem: Box<SrcType>,
    },
    /// The name of a struct.
    Named(String),
}

impl SrcType {
    #[allow(dead_code)]
    pub fn print(&self) -> String {
        match self {
            SrcType::Scalar(k) => k.name().to_string(),
            SrcType::Index => "index".to_string(),
            SrcType::Tensor { dims, elem } => {
                let mut s = String::from("Tensor<");
                if dims.is_empty() {
                    s.push_str(&elem.print());
                } else {
                    for (i, d) in dims.iter().enumerate() {
                        if i > 0 {
                            s.push('x');
                        }
                        match d {
                            SrcDim::Static(v) => s.push_str(&v.to_string()),
                            SrcDim::Dynamic => s.push('?'),
                        }
                    }
                    s.push('x');
                    s.push_str(&elem.print());
                }
                s.push('>');
                s
            }
            SrcType::Named(name) => name.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

impl BinOp {
    pub fn name(self) -> &'static str {
        match self {
            BinOp::Add => "add",
            BinOp::Sub => "sub",
            BinOp::Mul => "mul",
            BinOp::Div => "div",
            BinOp::Rem => "rem",
            BinOp::Eq => "eq",
            BinOp::Ne => "ne",
            BinOp::Lt => "lt",
            BinOp::Le => "le",
            BinOp::Gt => "gt",
            BinOp::Ge => "ge",
            BinOp::And => "and",
            BinOp::Or => "or",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)]
pub enum AttrValue {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

#[derive(Clone, Debug)]
pub enum Expr {
    Int(i64, Span),
    Float(f64, Span),
    Bool(bool, Span),
    Str(String, Span),
    Var(String, Span),
    Unary(UnOp, Box<Expr>, Span),
    Binary(BinOp, Box<Expr>, Box<Expr>, Span),
    Call {
        name: String,
        /// Explicit type arguments, e.g. `tensor.empty<Tensor<4xf32>>()`.
        type_args: Vec<SrcType>,
        args: Vec<Expr>,
        span: Span,
    },
    If {
        cond: Box<Expr>,
        then_block: Block,
        else_block: Option<Block>,
        span: Span,
    },
    For {
        var: String,
        lb: Box<Expr>,
        ub: Box<Expr>,
        step: Option<Box<Expr>>,
        /// Loop-carried values: `iter(acc = init)`.
        iter: Vec<IterBinding>,
        body: Block,
        span: Span,
    },
    Map {
        #[allow(clippy::vec_box)]
        inputs: Vec<Box<Expr>>,
        attrs: Vec<(String, AttrValue)>,
        params: Vec<String>,
        body: Block,
        span: Span,
    },
    Reduce {
        input: Box<Expr>,
        init: Box<Expr>,
        attrs: Vec<(String, AttrValue)>,
        params: Vec<String>,
        body: Block,
        span: Span,
    },
    Extract {
        tensor: Box<Expr>,
        #[allow(clippy::vec_box)]
        indices: Vec<Box<Expr>>,
        span: Span,
    },
    /// A field access: `recv.field`.
    Field {
        recv: Box<Expr>,
        name: String,
        span: Span,
    },
    /// A struct literal: `Name { field: expr, ... }`.
    StructLit {
        name: String,
        fields: Vec<(String, Expr)>,
        span: Span,
    },
    /// A functional update: `recv with { field: expr, ... }`.
    With {
        recv: Box<Expr>,
        updates: Vec<(String, Expr)>,
        span: Span,
    },
    /// A method call: `recv.method(args)`.
    Method {
        recv: Box<Expr>,
        name: String,
        args: Vec<Expr>,
        span: Span,
    },
    /// `tensor.insert(tensor, [index], value)` writes into a tensor.
    Insert {
        tensor: Box<Expr>,
        #[allow(clippy::vec_box)]
        indices: Vec<Box<Expr>>,
        value: Box<Expr>,
        span: Span,
    },
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Int(_, s)
            | Expr::Float(_, s)
            | Expr::Bool(_, s)
            | Expr::Str(_, s)
            | Expr::Var(_, s)
            | Expr::Unary(_, _, s)
            | Expr::Binary(_, _, _, s)
            | Expr::Call { span: s, .. }
            | Expr::If { span: s, .. }
            | Expr::For { span: s, .. }
            | Expr::Map { span: s, .. }
            | Expr::Reduce { span: s, .. }
            | Expr::Extract { span: s, .. } => s.clone(),
            | Expr::Insert { span: s, .. } => s.clone(),
            | Expr::Field { span: s, .. }
            | Expr::StructLit { span: s, .. }
            | Expr::With { span: s, .. }
            | Expr::Method { span: s, .. } => s.clone(),
        }
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct IterBinding {
    pub name: String,
    pub init: Expr,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Let {
        name: String,
        ty: Option<SrcType>,
        value: Expr,
        span: Span,
    },
    /// An expression evaluated for its side effects.
    Effect(Expr),
}

#[derive(Clone, Debug, Default)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    /// The values yielded by this region (`None` for regions that yield
    /// nothing).
    pub tail: Option<Vec<Expr>>,
    pub span: Span,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct Param {
    pub name: String,
    pub ty: SrcType,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Func {
    pub name: String,
    pub params: Vec<Param>,
    pub ret: Option<SrcType>,
    pub body: Block,
    pub span: Span,
}

/// A struct definition: `struct Name { field: Type, ... }`.
#[derive(Clone, Debug)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<(String, SrcType)>,
    pub span: Span,
}

/// An `impl Name { fn ... }` block: the methods of a struct.
#[derive(Clone, Debug)]
pub struct ImplBlock {
    /// The name of the struct.
    pub name: String,
    pub methods: Vec<Func>,
    pub span: Span,
}

#[derive(Clone, Debug, Default)]
pub struct Program {
    pub structs: Vec<StructDef>,
    pub impls: Vec<ImplBlock>,
    pub funcs: Vec<Func>,
}
