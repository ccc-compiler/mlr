//! The MLR intermediate representation.
//!
//! The IR is deliberately *isomorphic* to MLIR's model:
//!
//! * values are SSA values with a type,
//! * operations carry operands, results, attributes, regions and a location,
//! * regions contain blocks, blocks contain operations and end in a terminator.
//!
//! Everything the language expresses (`if`, `for ... iter(..)`, `map`,
//! `reduce`) becomes an operation with a region, so lowering is a structural
//! mapping instead of a guess about the original intent.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarKind {
    I1,
    I8,
    I16,
    I32,
    I64,
    F16,
    BF16,
    F32,
    F64,
}

impl ScalarKind {
    pub fn name(self) -> &'static str {
        match self {
            ScalarKind::I1 => "i1",
            ScalarKind::I8 => "i8",
            ScalarKind::I16 => "i16",
            ScalarKind::I32 => "i32",
            ScalarKind::I64 => "i64",
            ScalarKind::F16 => "f16",
            ScalarKind::BF16 => "bf16",
            ScalarKind::F32 => "f32",
            ScalarKind::F64 => "f64",
        }
    }

    pub fn is_float(self) -> bool {
        matches!(
            self,
            ScalarKind::F16 | ScalarKind::BF16 | ScalarKind::F32 | ScalarKind::F64
        )
    }

    pub fn is_int(self) -> bool {
        !self.is_float() && self != ScalarKind::I1
    }

    /// The width of the type in bits.
    pub fn width(self) -> u32 {
        match self {
            ScalarKind::I1 => 1,
            ScalarKind::I8 => 8,
            ScalarKind::I16 | ScalarKind::F16 | ScalarKind::BF16 => 16,
            ScalarKind::I32 | ScalarKind::F32 => 32,
            ScalarKind::I64 | ScalarKind::F64 => 64,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dim {
    Static(i64),
    Dynamic,
}

impl Dim {
    pub fn print(&self, out: &mut String) {
        match self {
            Dim::Static(v) => {
                let _ = write!(out, "{}", v);
            }
            Dim::Dynamic => out.push('?'),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    Scalar(ScalarKind),
    Index,
    Tensor {
        shape: Vec<Dim>,
        elem: Box<Type>,
    },
    MemRef {
        shape: Vec<Dim>,
        elem: Box<Type>,
    },
    /// The type of a function without results.
    Void,
    /// A tuple of values, used for the type of a struct value.
    Tuple(Vec<Type>),
}

#[allow(dead_code)]
impl Type {
    pub fn scalar(name: &str) -> Option<Type> {
        Some(match name {
            "index" => Type::Index,
            "i1" | "bool" => Type::Scalar(ScalarKind::I1),
            "i8" => Type::Scalar(ScalarKind::I8),
            "i16" => Type::Scalar(ScalarKind::I16),
            "i32" => Type::Scalar(ScalarKind::I32),
            "i64" => Type::Scalar(ScalarKind::I64),
            "f16" => Type::Scalar(ScalarKind::F16),
            "bf16" => Type::Scalar(ScalarKind::BF16),
            "f32" => Type::Scalar(ScalarKind::F32),
            "f64" => Type::Scalar(ScalarKind::F64),
            _ => return None,
        })
    }

    pub fn is_tensor(&self) -> bool {
        matches!(self, Type::Tensor { .. })
    }

    pub fn is_memref(&self) -> bool {
        matches!(self, Type::MemRef { .. })
    }

    pub fn is_float(&self) -> bool {
        match self {
            Type::Scalar(k) => k.is_float(),
            _ => false,
        }
    }

    pub fn is_integer(&self) -> bool {
        match self {
            Type::Scalar(k) => k.is_int(),
            Type::Index => true,
            _ => false,
        }
    }

    pub fn elem_type(&self) -> Option<Type> {
        match self {
            Type::Tensor { elem, .. } | Type::MemRef { elem, .. } => Some((**elem).clone()),
            _ => None,
        }
    }

    pub fn shape(&self) -> Option<&[Dim]> {
        match self {
            Type::Tensor { shape, .. } | Type::MemRef { shape, .. } => Some(shape),
            _ => None,
        }
    }

    pub fn print(&self, out: &mut String) {
        match self {
            Type::Scalar(k) => out.push_str(k.name()),
            Type::Index => out.push_str("index"),
            Type::Void => out.push_str("()"),
            Type::Tensor { shape, elem } => {
                out.push_str("tensor<");
                for (i, d) in shape.iter().enumerate() {
                    if i > 0 {
                        out.push('x');
                    }
                    d.print(out);
                }
                out.push('x');
                elem.print(out);
                out.push('>');
            }
            Type::MemRef { shape, elem } => {
                out.push_str("memref<");
                for (i, d) in shape.iter().enumerate() {
                    if i > 0 {
                        out.push('x');
                    }
                    d.print(out);
                }
                out.push('x');
                elem.print(out);
                out.push('>');
            }
            Type::Tuple(types) => {
                out.push_str("tuple<");
                for (i, t) in types.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    t.print(out);
                }
                out.push('>');
            }
        }
    }

    /// The MLIR spelling of the type.
    pub fn name(&self) -> String {
        let mut s = String::new();
        self.print(&mut s);
        s
    }
}

// ---------------------------------------------------------------------------
// Attributes
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum AffineExpr {
    Dim(usize),
    Sym(usize),
    Const(i64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct AffineMap {
    pub dims: usize,
    pub symbols: usize,
    pub results: Vec<AffineExpr>,
}

impl AffineMap {
    pub fn identity(dims: usize) -> AffineMap {
        AffineMap {
            dims,
            symbols: 0,
            results: (0..dims).map(AffineExpr::Dim).collect(),
        }
    }

    pub fn print(&self, out: &mut String) {
        out.push_str("affine_map<(");
        for i in 0..self.dims {
            if i > 0 {
                out.push_str(", ");
            }
            let _ = write!(out, "d{}", i);
        }
        if self.symbols > 0 {
            out.push_str(", ");
            for i in 0..self.symbols {
                if i > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "s{}", i);
            }
        }
        out.push_str(") -> (");
        for (i, r) in self.results.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            match r {
                AffineExpr::Dim(d) => {
                    let _ = write!(out, "d{}", d);
                }
                AffineExpr::Sym(s) => {
                    let _ = write!(out, "s{}", s);
                }
                AffineExpr::Const(c) => {
                    let _ = write!(out, "{}", c);
                }
            }
        }
        out.push_str(")>");
    }
}

#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)]
pub enum Attr {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Type(Type),
    DenseI64(Vec<i64>),
    AffineMap(AffineMap),
    AffineMaps(Vec<AffineMap>),
    StrList(Vec<String>),
    DenseF32(Vec<f32>),
    DenseI32(Vec<i32>),
    /// A buffer of bytes, printed in the hexadecimal form of MLIR.
    DenseI8(Vec<u8>),
}

impl Attr {
    pub fn print(&self, out: &mut String) {
        match self {
            Attr::Int(v) => {
                let _ = write!(out, "{}", v);
            }
            Attr::Float(v) => {
                let spelling = format!("{:?}", v);
                if spelling.contains('e') {
                    // MLIR reads `1e-8` as an identifier; a decimal spelling
                    // is always accepted.
                    let _ = write!(out, "{:.20}", v);
                } else {
                    out.push_str(&spelling);
                }
            }
            Attr::Str(s) => {
                let _ = write!(out, "\"{}\"", s);
            }
            Attr::Bool(b) => {
                let _ = write!(out, "{}", b);
            }
            Attr::Type(t) => t.print(out),
            Attr::DenseI64(v) => {
                out.push_str("dense<[");
                for (i, x) in v.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(out, "{}", x);
                }
                out.push_str("]>");
            }
            Attr::AffineMap(m) => m.print(out),
            Attr::AffineMaps(ms) => {
                out.push('[');
                for (i, m) in ms.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    m.print(out);
                }
                out.push(']');
            }
            Attr::DenseF32(values) => {
                // The raw bytes: a list literal would only fit a rank of one,
                // since the type of the tensor is unknown here.
                out.push_str("dense<\"0x");
                for v in values {
                    for byte in v.to_le_bytes() {
                        let _ = write!(out, "{:02X}", byte);
                    }
                }
                out.push_str("\">");
            }
            Attr::DenseI32(values) => {
                out.push_str("dense<\"0x");
                for v in values {
                    for byte in v.to_le_bytes() {
                        let _ = write!(out, "{:02X}", byte);
                    }
                }
                out.push_str("\">");
            }
            Attr::DenseI8(values) => {
                out.push_str("dense<\"0x");
                for v in values {
                    let _ = write!(out, "{:02X}", v);
                }
                out.push_str("\">");
            }
            Attr::StrList(items) => {
                out.push('[');
                for (i, s) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(out, "\"{}\"", s);
                }
                out.push(']');
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Locations
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Loc {
    pub file: String,
    pub line: u32,
    pub col: u32,
}

impl Loc {
    pub fn new(file: &str, line: u32, col: u32) -> Loc {
        Loc {
            file: file.to_string(),
            line,
            col,
        }
    }

    fn print(&self, out: &mut String) {
        let _ = write!(out, "loc(\"{}\":{}:{})", self.file, self.line, self.col);
    }
}

// ---------------------------------------------------------------------------
// Values, operations, regions
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Value {
    pub id: usize,
    pub ty: Type,
}

#[derive(Clone, Debug, Default)]
pub struct Block {
    pub args: Vec<Value>,
    pub ops: Vec<Op>,
}

#[derive(Clone, Debug, Default)]
pub struct Region {
    pub blocks: Vec<Block>,
}

#[derive(Clone, Debug)]
pub struct Op {
    /// Operation name, e.g. `arith.addi` or `mlr.map`.
    pub name: String,
    pub operands: Vec<Value>,
    pub results: Vec<Value>,
    pub attrs: Vec<(String, Attr)>,
    pub regions: Vec<Region>,
    pub loc: Loc,
    /// Print in MLIR's generic form (used for unregistered dialects).
    pub generic: bool,
}

impl Op {
    pub fn new(name: &str) -> Op {
        Op {
            name: name.to_string(),
            operands: Vec::new(),
            results: Vec::new(),
            attrs: Vec::new(),
            regions: Vec::new(),
            loc: Loc::default(),
            generic: false,
        }
    }

    pub fn attr(&self, key: &str) -> Option<&Attr> {
        self.attrs.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn is_terminator(&self) -> bool {
        matches!(
            self.name.as_str(),
            "scf.yield" | "func.return" | "linalg.yield" | "mlr.yield" | "cf.br" | "cf.cond_br"
        )
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct FuncOp {
    pub name: String,
    pub params: Vec<Value>,
    pub results: Vec<Type>,
    pub body: Region,
    pub attrs: Vec<(String, Attr)>,
    pub loc: Loc,
}

#[derive(Clone, Debug)]
pub struct FuncDecl {
    pub name: String,
    pub params: Vec<Type>,
    pub results: Vec<Type>,
}

/// A global constant buffer (`memref.global`).
#[derive(Clone, Debug)]
pub struct Global {
    pub name: String,
    pub ty: Type,
    pub value: Attr,
}

#[derive(Clone, Debug, Default)]
pub struct Module {
    pub funcs: Vec<FuncOp>,
    pub decls: Vec<FuncDecl>,
    pub globals: Vec<Global>,
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Value ids are unique across the whole program: every builder draws from
/// the same counter, so values created by different passes never collide.
static NEXT_VALUE: AtomicUsize = AtomicUsize::new(0);

/// Creates operations with fresh SSA values.
#[derive(Default)]
pub struct Builder {}

impl Builder {
    pub fn new() -> Builder {
        Builder {}
    }

    pub fn fresh_value(&mut self, ty: Type) -> Value {
        let id = NEXT_VALUE.fetch_add(1, Ordering::SeqCst);
        Value { id, ty }
    }

    #[allow(dead_code)]
    pub fn fresh_block_arg(&mut self, ty: Type) -> Value {
        self.fresh_value(ty)
    }
}

/// Renumbers the SSA values of every function so that the printed module uses
/// dense, readable names (`%0`, `%1`, ...).
pub fn renumber(module: &mut Module) {
    for f in &mut module.funcs {
        let mut map = std::collections::HashMap::new();
        let mut next = 0usize;
        // The parameters come first: they are the arguments of the entry
        // block.
        for p in &mut f.params {
            let new = next;
            next += 1;
            map.insert(p.id, new);
            p.id = new;
        }
        renumber_region(&mut f.body, &mut map, &mut next);
        renumber_uses_region(&mut f.body, &map);
    }
}

fn renumber_region(
    region: &mut Region,
    map: &mut std::collections::HashMap<usize, usize>,
    next: &mut usize,
) {
    for block in &mut region.blocks {
        for arg in &mut block.args {
            let new = *next;
            *next += 1;
            map.insert(arg.id, new);
            arg.id = new;
        }
        for op in &mut block.ops {
            for r in &mut op.results {
                let new = *next;
                *next += 1;
                map.insert(r.id, new);
                r.id = new;
            }
            for r in &mut op.regions {
                renumber_region(r, map, next);
            }
        }
    }
}

fn renumber_uses_region(region: &mut Region, map: &std::collections::HashMap<usize, usize>) {
    for block in &mut region.blocks {
        for op in &mut block.ops {
            for o in &mut op.operands {
                if let Some(n) = map.get(&o.id) {
                    o.id = *n;
                }
            }
            for r in &mut op.regions {
                renumber_uses_region(r, map);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// A light-weight structural verifier: it checks that terminators are
/// present, that SSA ids are unique, and that every operand is defined
/// somewhere in the enclosing function.
pub fn verify(module: &Module) -> Result<(), String> {
    for f in &module.funcs {
        // The types of the values of THIS function: the ids are per function
        // after the renumbering, so the map must not be shared.
        let mut types: HashMap<usize, Type> = HashMap::new();
        let mut defined = std::collections::HashSet::new();
        for p in &f.params {
            defined.insert(p.id);
            types.insert(p.id, p.ty.clone());
        }
        // The declarations of the runtime: the calls are checked against
        // them.
        let mut decls: HashMap<&str, (Vec<Type>, Vec<Type>)> = HashMap::new();
        for d in &module.decls {
            decls.insert(d.name.as_str(), (d.params.clone(), d.results.clone()));
        }
        verify_region(&f.body, &mut defined, &mut types, &decls, &f.name)?;
        // The result of the function must match its signature.
        verify_return(&f)?;
    }
    Ok(())
}

fn verify_return(f: &FuncOp) -> Result<(), String> {
    let term = match f.body.blocks[0].ops.last() {
        Some(t) if t.name == "func.return" => t,
        _ => return Ok(()),
    };
    let types: Vec<Type> = term.operands.iter().map(|o| o.ty.clone()).collect();
    if types.len() != f.results.len() {
        return Err(format!(
            "function `{}` returns {} value(s) but its signature has {}",
            f.name,
            types.len(),
            f.results.len()
        ));
    }
    for (i, (a, b)) in types.iter().zip(f.results.iter()).enumerate() {
        if a != b {
            return Err(format!(
                "function `{}`: the {}-th returned value has type {}, expected {}",
                f.name,
                i,
                a.name(),
                b.name()
            ));
        }
    }
    Ok(())
}

fn verify_region(
    region: &Region,
    defined: &mut std::collections::HashSet<usize>,
    types: &mut HashMap<usize, Type>,
    decls: &HashMap<&str, (Vec<Type>, Vec<Type>)>,
    func: &str,
) -> Result<(), String> {
    if region.blocks.is_empty() {
        return Err(format!("function `{}` has an empty region", func));
    }
    for block in &region.blocks {
        for a in &block.args {
            defined.insert(a.id);
            types.insert(a.id, a.ty.clone());
        }
        for (i, op) in block.ops.iter().enumerate() {
            for o in &op.operands {
                if !defined.contains(&o.id) {
                    return Err(format!(
                        "function `{}`: operand %{} of `{}` is not defined before use",
                        func, o.id, op.name
                    ));
                }
                // The type of every use must match the type of the value.
                if let Some(t) = types.get(&o.id) {
                    if *t != o.ty {
                        return Err(format!(
                            "function `{}`: operand %{} of `{}` has type {}, but the \
                             value has type {}",
                            func,
                            o.id,
                            op.name,
                            o.ty.name(),
                            t.name()
                        ));
                    }
                }
            }
            // A call must match the signature of its callee.
            if op.name == "func.call" {
                if let Some(Attr::Str(callee)) = op.attr("callee") {
                    if let Some((params, results)) = decls.get(callee.as_str()) {
                        if op.operands.len() != params.len() {
                            return Err(format!(
                                "function `{}`: the call of `{}` has {} operand(s), \
                                 the declaration has {}",
                                func,
                                callee,
                                op.operands.len(),
                                params.len()
                            ));
                        }
                        for (o, p) in op.operands.iter().zip(params.iter()) {
                            if o.ty != *p {
                                return Err(format!(
                                    "function `{}`: the operand of the call of `{}` has \
                                     type {}, expected {}",
                                    func,
                                    callee,
                                    o.ty.name(),
                                    p.name()
                                ));
                            }
                        }
                        if op.results.len() != results.len() {
                            return Err(format!(
                                "function `{}`: the call of `{}` returns {} value(s), \
                                 the declaration has {}",
                                func,
                                callee,
                                op.results.len(),
                                results.len()
                            ));
                        }
                        for (r, p) in op.results.iter().zip(results.iter()) {
                            if r.ty != *p {
                                return Err(format!(
                                    "function `{}`: the result of the call of `{}` has \
                                     type {}, expected {}",
                                    func,
                                    callee,
                                    r.ty.name(),
                                    p.name()
                                ));
                            }
                        }
                    }
                }
            }
            for r in &op.results {
                if !defined.insert(r.id) {
                    return Err(format!(
                        "function `{}`: SSA value %{} is defined twice",
                        func, r.id
                    ));
                }
            }
            for r in &op.regions {
                verify_region(r, defined, types, decls, func)?;
            }
            let last = i + 1 == block.ops.len();
            if last && !op.is_terminator() {
                return Err(format!(
                    "function `{}`: block ends with `{}` which is not a terminator",
                    func, op.name
                ));
            }
        }
        if let Some(last) = block.ops.last() {
            if !last.is_terminator() {
                return Err(format!(
                    "function `{}`: block ends with `{}` which is not a terminator",
                    func, last.name
                ));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Printer
// ---------------------------------------------------------------------------

/// Prints a module in MLIR's textual format.
pub fn print_module(module: &Module) -> String {
    let mut p = Printer::default();
    p.module(module)
}

#[derive(Default)]
struct Printer {
    affine_maps: Vec<AffineMap>,
}

/// Returns the module-level id of an affine map, registering it if needed.
impl Printer {
    fn map_id(&mut self, m: &AffineMap) -> usize {
        if let Some(i) = self.affine_maps.iter().position(|x| x == m) {
            return i;
        }
        self.affine_maps.push(m.clone());
        self.affine_maps.len() - 1
    }
}

impl Printer {
    fn module(&mut self, module: &Module) -> String {
        // Functions first (this also collects the affine maps that are needed).
        let mut bodies = Vec::new();
        for f in &module.funcs {
            bodies.push(self.func(f));
        }
        let mut out = String::new();
        for (i, m) in self.affine_maps.iter().enumerate() {
            let mut s = String::new();
            m.print(&mut s);
            let _ = writeln!(out, "#map{} = {}", i, s);
        }
        out.push_str("module {\n");
        for g in &module.globals {
            let mut value = String::new();
            g.value.print(&mut value);
            let _ = writeln!(
                out,
                "  memref.global \"private\" constant @{} : {} = {}",
                g.name,
                g.ty.name(),
                value
            );
        }
        for d in &module.decls {
            let mut params = Vec::new();
            for p in &d.params {
                params.push(p.name());
            }
            let mut results = Vec::new();
            for r in &d.results {
                results.push(r.name());
            }
            if results.is_empty() {
                let _ = writeln!(
                    out,
                    "  func.func private @{}({})",
                    d.name,
                    params.join(", ")
                );
            } else {
                let _ = writeln!(
                    out,
                    "  func.func private @{}({}) -> ({})",
                    d.name,
                    params.join(", "),
                    results.join(", ")
                );
            }
        }
        for b in bodies {
            out.push_str(&b);
        }
        out.push_str("}\n");
        out
    }

    fn func(&mut self, f: &FuncOp) -> String {
        let mut out = String::new();
        let mut params = Vec::new();
        for p in &f.params {
            params.push(format!("%{}: {}", p.id, p.ty.name()));
        }
        let _ = write!(out, "  func.func @{}({})", f.name, params.join(", "));
        if f.results.len() == 1 {
            let _ = write!(out, " -> {}", f.results[0].name());
        } else if !f.results.is_empty() {
            let rs: Vec<String> = f.results.iter().map(|t| t.name()).collect();
            let _ = write!(out, " -> ({})", rs.join(", "));
        }
        if !f.attrs.is_empty() {
            out.push_str(" attributes {");
            let parts: Vec<String> = f
                .attrs
                .iter()
                .map(|(k, v)| {
                    let mut s = String::new();
                    v.print(&mut s);
                    format!("{} = {}", k, s)
                })
                .collect();
            out.push_str(&parts.join(", "));
            out.push('}');
        }
        out.push_str(" {\n");
        if let Some(block) = f.body.blocks.first() {
            for op in &block.ops {
                self.op(&mut out, op, 2);
            }
        }
        out.push_str("  }\n");
        out
    }

    fn indent(&self, out: &mut String, level: usize) {
        for _ in 0..level {
            out.push_str("  ");
        }
    }

    fn value_list(&self, out: &mut String, values: &[Value]) {
        for (i, v) in values.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            let _ = write!(out, "%{}", v.id);
        }
    }

    /// Prints ` %a, %b : type, type` (the operand list with its types).
    fn value_list_typed(&self, out: &mut String, values: &[Value]) {
        for (i, v) in values.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            let _ = write!(out, "%{}", v.id);
        }
        if !values.is_empty() {
            out.push_str(" : ");
            let types: Vec<Type> = values.iter().map(|v| v.ty.clone()).collect();
            self.type_list(out, &types);
        }
    }

    fn type_list(&self, out: &mut String, types: &[Type]) {
        for (i, t) in types.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            t.print(out);
        }
    }

    fn results_prefix(&self, out: &mut String, op: &Op) {
        if !op.results.is_empty() {
            self.value_list(out, &op.results);
            out.push_str(" = ");
        }
    }

    fn op(&mut self, out: &mut String, op: &Op, level: usize) {
        self.indent(out, level);
        if op.generic {
            self.generic_op(out, op);
            out.push('\n');
            return;
        }
        match op.name.as_str() {
            "arith.constant" => {
                self.results_prefix(out, op);
                out.push_str("arith.constant ");
                match op.attr("value") {
                    Some(Attr::Int(v)) => {
                        let _ = write!(out, "{}", v);
                    }
                    Some(Attr::Float(v)) => {
                        // The same spelling as the attribute: MLIR reads
                        // `1e-8` as an identifier.
                        let spelling = format!("{:?}", v);
                        if spelling.contains('e') {
                            let _ = write!(out, "{:.20}", v);
                        } else {
                            out.push_str(&spelling);
                        }
                    }
                    Some(Attr::Bool(b)) => {
                        let _ = write!(out, "{}", if *b { 1 } else { 0 });
                    }
                    Some(Attr::DenseF32(_)) | Some(Attr::DenseI32(_)) => {
                        if let Some(a) = op.attr("value") {
                            a.print(out);
                        }
                    }
                    _ => out.push('0'),
                }
                out.push_str(" : ");
                if let Some(r) = op.results.first() {
                    r.ty.print(out);
                }
            }
            "arith.addi" | "arith.subi" | "arith.muli" | "arith.divsi" | "arith.remsi"
            | "arith.andi" | "arith.ori" | "arith.xori" | "arith.addf" | "arith.subf"
            | "arith.mulf" | "arith.divf" | "arith.remf" | "arith.negf" => {
                self.results_prefix(out, op);
                out.push_str(&op.name);
                out.push(' ');
                self.value_list(out, &op.operands);
                out.push_str(" : ");
                if let Some(o) = op.operands.first() {
                    o.ty.print(out);
                }
            }
            "arith.cmpi" | "arith.cmpf" => {
                self.results_prefix(out, op);
                out.push_str(&op.name);
                out.push(' ');
                if let Some(Attr::Str(p)) = op.attr("predicate") {
                    out.push_str(p);
                } else {
                    out.push_str("eq");
                }
                out.push_str(", ");
                self.value_list(out, &op.operands);
                out.push_str(" : ");
                if let Some(o) = op.operands.first() {
                    o.ty.print(out);
                }
            }
            "arith.index_cast" => {
                self.results_prefix(out, op);
                out.push_str("arith.index_cast ");
                self.value_list(out, &op.operands);
                out.push_str(" : ");
                if let Some(o) = op.operands.first() {
                    o.ty.print(out);
                    out.push_str(" to ");
                }
                if let Some(r) = op.results.first() {
                    r.ty.print(out);
                }
            }
            "scf.if" => {
                self.results_prefix(out, op);
                out.push_str("scf.if ");
                self.value_list(out, &op.operands);
                out.push_str(" -> (");
                for (i, r) in op.results.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    r.ty.print(out);
                }
                out.push(')');
                let regions = op.regions.clone();
                for (i, region) in regions.iter().enumerate() {
                    if i == 0 {
                        out.push_str(" {\n");
                    } else {
                        out.push_str(" else {\n");
                    }
                    self.region(out, region, level + 1, true);
                    self.indent(out, level);
                    out.push('}');
                }
            }
            "scf.for" => {
                self.results_prefix(out, op);
                out.push_str("scf.for ");
                // %iv = %lb to %ub step %step
                let body_args = op
                    .regions
                    .first()
                    .and_then(|r| r.blocks.first())
                    .map(|b| b.args.clone())
                    .unwrap_or_default();
                if let Some(iv) = body_args.first() {
                    let _ = write!(out, "%{} = ", iv.id);
                }
                if let Some(lb) = op.operands.first() {
                    let _ = write!(out, "%{}", lb.id);
                }
                out.push_str(" to ");
                if let Some(ub) = op.operands.get(1) {
                    let _ = write!(out, "%{}", ub.id);
                }
                out.push_str(" step ");
                if let Some(st) = op.operands.get(2) {
                    let _ = write!(out, "%{}", st.id);
                }
                // iter_args
                let iter_args = &body_args[body_args.len().min(1)..];
                if !iter_args.is_empty() {
                    out.push_str(" iter_args(");
                    for (i, a) in iter_args.iter().enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        let init = op.operands.get(3 + i);
                        let _ = write!(out, "%{} = %{}", a.id, init.map(|v| v.id).unwrap_or(0));
                    }
                    out.push(')');
                }
                out.push_str(" -> (");
                for (i, r) in op.results.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    r.ty.print(out);
                }
                out.push(')');
                if let Some(region) = op.regions.first() {
                    out.push_str(" {\n");
                    self.region(out, region, level + 1, false);
                    self.indent(out, level);
                    out.push('}');
                }
            }
            "scf.yield" | "linalg.yield" | "mlr.yield" => {
                out.push_str(&op.name);
                if !op.operands.is_empty() {
                    out.push(' ');
                    self.value_list(out, &op.operands);
                    out.push_str(" : ");
                    for (i, o) in op.operands.iter().enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        o.ty.print(out);
                    }
                }
            }
            "func.call" if op.results.len() > 1 => {
                // The custom syntax of `func.call` has no form for several
                // results, so the generic one is used.
                self.results_prefix(out, op);
                out.push_str("\"func.call\"(");
                self.value_list(out, &op.operands);
                out.push_str(") {callee = @");
                if let Some(Attr::Str(callee)) = op.attr("callee") {
                    out.push_str(callee);
                }
                out.push_str("} : (");
                let types: Vec<Type> = op.operands.iter().map(|o| o.ty.clone()).collect();
                self.type_list(out, &types);
                out.push_str(") -> (");
                let types: Vec<Type> = op.results.iter().map(|o| o.ty.clone()).collect();
                self.type_list(out, &types);
                out.push(')');
            }
            "func.call" => {
                self.results_prefix(out, op);
                out.push_str("func.call @");
                if let Some(Attr::Str(callee)) = op.attr("callee") {
                    out.push_str(callee);
                }
                out.push('(');
                self.value_list(out, &op.operands);
                out.push_str(") : (");
                let types: Vec<Type> = op.operands.iter().map(|o| o.ty.clone()).collect();
                self.type_list(out, &types);
                out.push_str(") -> ");
                if op.results.is_empty() {
                    out.push_str("()");
                } else {
                    let types: Vec<Type> = op.results.iter().map(|o| o.ty.clone()).collect();
                    self.type_list(out, &types);
                }
            }
            "memref.get_global" => {
                self.results_prefix(out, op);
                out.push_str("memref.get_global @");
                if let Some(Attr::Str(name)) = op.attr("name") {
                    out.push_str(name);
                }
                out.push_str(" : ");
                if let Some(r) = op.results.first() {
                    r.ty.print(out);
                }
            }
            "func.return" => {
                out.push_str("func.return");
                if !op.operands.is_empty() {
                    out.push(' ');
                    self.value_list(out, &op.operands);
                    out.push_str(" : ");
                    if op.operands.len() == 1 {
                        if let Some(o) = op.operands.first() {
                            o.ty.print(out);
                        }
                    } else {
                        let types: Vec<Type> = op.operands.iter().map(|o| o.ty.clone()).collect();
                        self.type_list(out, &types);
                    }
                }
            }
            "tensor.empty" => {
                self.results_prefix(out, op);
                out.push_str("tensor.empty() : ");
                if let Some(r) = op.results.first() {
                    r.ty.print(out);
                }
            }
            "tensor.extract" => {
                self.results_prefix(out, op);
                out.push_str("tensor.extract ");
                if let Some(t) = op.operands.first() {
                    let _ = write!(out, "%{}", t.id);
                }
                out.push('[');
                for (i, idx) in op.operands.iter().skip(1).enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(out, "%{}", idx.id);
                }
                out.push_str("] : ");
                if let Some(t) = op.operands.first() {
                    t.ty.print(out);
                }
            }
            "linalg.generic" => {
                self.results_prefix(out, op);
                out.push_str("linalg.generic {");
                let mut parts: Vec<String> = Vec::new();
                if let Some(Attr::AffineMaps(ms)) = op.attr("indexing_maps") {
                    let ids: Vec<String> = ms
                        .iter()
                        .map(|m| format!("#map{}", self.map_id(m)))
                        .collect();
                    parts.push(format!("indexing_maps = [{}]", ids.join(", ")));
                }
                if let Some(Attr::StrList(items)) = op.attr("iterator_types") {
                    let quoted: Vec<String> = items.iter().map(|s| format!("\"{}\"", s)).collect();
                    parts.push(format!("iterator_types = [{}]", quoted.join(", ")));
                }
                for (k, v) in &op.attrs {
                    if k == "iterator_types"
                        || k == "indexing_maps"
                        || k == "ins_count"
                        || k == "outs_count"
                    {
                        continue;
                    }
                    let mut s = String::new();
                    v.print(&mut s);
                    parts.push(format!("{} = {}", k, s));
                }
                out.push_str(&parts.join(", "));
                out.push('}');
                let ins_count = op
                    .attr("ins_count")
                    .and_then(|a| match a {
                        Attr::Int(v) => Some(*v as usize),
                        _ => None,
                    })
                    .unwrap_or(0);
                let outs_count = op
                    .attr("outs_count")
                    .and_then(|a| match a {
                        Attr::Int(v) => Some(*v as usize),
                        _ => None,
                    })
                    .unwrap_or(0);
                let ins_vals = &op.operands[..ins_count.min(op.operands.len())];
                let outs_vals = &op.operands[ins_count.min(op.operands.len())
                    ..(ins_count + outs_count).min(op.operands.len())];
                out.push_str(" ins(");
                self.value_list_typed(out, ins_vals);
                out.push_str(") outs(");
                self.value_list_typed(out, outs_vals);
                out.push(')');
                out.push(' ');
                if let Some(region) = op.regions.first() {
                    out.push_str("{\n");
                    self.region(out, region, level + 1, true);
                    self.indent(out, level);
                    out.push('}');
                }
                out.push_str(" -> ");
                if let Some(r) = op.results.first() {
                    r.ty.print(out);
                }
            }
            "memref.alloc" => {
                self.results_prefix(out, op);
                out.push_str("memref.alloc() : ");
                if let Some(r) = op.results.first() {
                    r.ty.print(out);
                }
            }
            "memref.load" => {
                self.results_prefix(out, op);
                out.push_str("memref.load ");
                if let Some(m) = op.operands.first() {
                    let _ = write!(out, "%{}", m.id);
                }
                out.push('[');
                for (i, idx) in op.operands.iter().skip(1).enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(out, "%{}", idx.id);
                }
                out.push_str("] : ");
                if let Some(m) = op.operands.first() {
                    m.ty.print(out);
                }
            }
            "memref.store" => {
                out.push_str("memref.store ");
                if let Some(v) = op.operands.first() {
                    let _ = write!(out, "%{}", v.id);
                }
                out.push_str(", ");
                if let Some(m) = op.operands.get(1) {
                    let _ = write!(out, "%{}", m.id);
                }
                out.push('[');
                for (i, idx) in op.operands.iter().skip(2).enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(out, "%{}", idx.id);
                }
                out.push_str("] : ");
                if let Some(m) = op.operands.get(1) {
                    m.ty.print(out);
                }
            }
            "memref.copy" => {
                out.push_str("memref.copy ");
                if let Some(s) = op.operands.first() {
                    let _ = write!(out, "%{}", s.id);
                }
                out.push_str(", ");
                if let Some(d) = op.operands.get(1) {
                    let _ = write!(out, "%{}", d.id);
                }
                out.push_str(" : ");
                if let Some(s) = op.operands.first() {
                    s.ty.print(out);
                }
                out.push_str(" to ");
                if let Some(d) = op.operands.get(1) {
                    d.ty.print(out);
                }
            }
            "memref.extract_aligned_pointer_as_index" => {
                self.results_prefix(out, op);
                out.push_str("memref.extract_aligned_pointer_as_index ");
                if let Some(m) = op.operands.first() {
                    let _ = write!(out, "%{}", m.id);
                }
                out.push_str(" : ");
                if let Some(m) = op.operands.first() {
                    m.ty.print(out);
                }
                out.push_str(" -> ");
                if let Some(r) = op.results.first() {
                    r.ty.print(out);
                }
            }
            "memref.dim" => {
                self.results_prefix(out, op);
                out.push_str("memref.dim ");
                if let Some(m) = op.operands.first() {
                    let _ = write!(out, "%{}", m.id);
                }
                out.push_str(", ");
                if let Some(d) = op.operands.get(1) {
                    let _ = write!(out, "%{}", d.id);
                }
                out.push_str(" : ");
                if let Some(m) = op.operands.first() {
                    m.ty.print(out);
                }
            }
            "memref.dealloc" => {
                out.push_str("memref.dealloc ");
                if let Some(m) = op.operands.first() {
                    let _ = write!(out, "%{}", m.id);
                }
                out.push_str(" : ");
                if let Some(m) = op.operands.first() {
                    m.ty.print(out);
                }
            }
            _ => {
                self.generic_op(out, op);
                out.push('\n');
                return;
            }
        }
        self.loc_suffix(out, op);
        out.push('\n');
    }

    fn loc_suffix(&self, out: &mut String, op: &Op) {
        if op.loc.file.is_empty() {
            return;
        }
        out.push(' ');
        op.loc.print(out);
    }

    fn generic_op(&mut self, out: &mut String, op: &Op) {
        self.results_prefix(out, op);
        let _ = write!(out, "\"{}\"", op.name);
        out.push('(');
        self.value_list(out, &op.operands);
        out.push(')');
        for region in &op.regions {
            out.push_str(" ({\n");
            self.region(out, region, 3, true);
            out.push_str("  })");
        }
        out.push_str(" : (");
        let types: Vec<Type> = op.operands.iter().map(|o| o.ty.clone()).collect();
        self.type_list(out, &types);
        out.push(')');
        if op.results.is_empty() {
            out.push_str(" -> ()");
        } else {
            out.push_str(" -> (");
            let types: Vec<Type> = op.results.iter().map(|o| o.ty.clone()).collect();
            self.type_list(out, &types);
            out.push(')');
        }
        self.loc_suffix(out, op);
    }

    /// Prints a region. `label` controls whether the block arguments are
    /// printed explicitly: operations whose region has implicit arguments
    /// (`scf.for`) take the names from the operation itself.
    fn region(&mut self, out: &mut String, region: &Region, level: usize, label: bool) {
        for (i, block) in region.blocks.iter().enumerate() {
            if label && !block.args.is_empty() {
                self.indent(out, level);
                let _ = write!(out, "^bb{}", i);
                out.push('(');
                for (j, a) in block.args.iter().enumerate() {
                    if j > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(out, "%{}: {}", a.id, a.ty.name());
                }
                out.push_str("):\n");
            }
            for op in &block.ops {
                self.op(out, op, level);
            }
        }
    }
}
