//! Recursive descent parser for MLR.
//!
//! The grammar is built around *regions*: `if`, `for ... iter(..)`, `map` and
//! `reduce` are expressions whose bodies are blocks that end in `yield`.

use crate::ast::*;
use crate::lexer::{Error, Span, Tok, Token};

pub struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

type PResult<T> = Result<T, Error>;

impl Parser {
    pub fn new(toks: Vec<Token>) -> Self {
        Parser { toks, pos: 0 }
    }

    fn peek(&self) -> &Tok {
        &self.toks[self.pos].tok
    }

    fn peek_at(&self, n: usize) -> &Tok {
        let i = (self.pos + n).min(self.toks.len() - 1);
        &self.toks[i].tok
    }

    fn span(&self) -> Span {
        self.toks[self.pos].span.clone()
    }

    fn bump(&mut self) -> Token {
        let t = self.toks[self.pos].clone();
        if self.pos + 1 < self.toks.len() {
            self.pos += 1;
        }
        t
    }

    fn err<T>(&self, msg: impl Into<String>) -> PResult<T> {
        Err(Error {
            msg: msg.into(),
            span: self.span(),
        })
    }

    fn at_punct(&self, p: &str) -> bool {
        matches!(self.peek(), Tok::Punct(x) if *x == p)
    }

    fn at_kw(&self, k: &str) -> bool {
        matches!(self.peek(), Tok::Ident(x) if x == k)
    }

    fn eat_punct(&mut self, p: &str) -> bool {
        if self.at_punct(p) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect_punct(&mut self, p: &str) -> PResult<Span> {
        if self.at_punct(p) {
            Ok(self.bump().span)
        } else {
            self.err(format!("expected `{}`, found {}", p, self.peek()))
        }
    }

    fn expect_ident(&mut self) -> PResult<(String, Span)> {
        match self.peek().clone() {
            Tok::Ident(name) => {
                let span = self.bump().span;
                Ok((name, span))
            }
            other => self.err(format!("expected identifier, found {}", other)),
        }
    }

    fn expect_kw(&mut self, k: &str) -> PResult<Span> {
        if self.at_kw(k) {
            Ok(self.bump().span)
        } else {
            self.err(format!("expected `{}`, found {}", k, self.peek()))
        }
    }

    // -----------------------------------------------------------------
    // Items
    // -----------------------------------------------------------------

    pub fn parse_program(&mut self) -> PResult<Program> {
        let mut prog = Program::default();
        while !matches!(self.peek(), Tok::Eof) {
            prog.funcs.push(self.parse_func()?);
        }
        Ok(prog)
    }

    fn parse_func(&mut self) -> PResult<Func> {
        let span = self.expect_kw("fn")?;
        let (name, _) = self.expect_ident()?;
        self.expect_punct("(")?;
        let mut params = Vec::new();
        if !self.at_punct(")") {
            loop {
                let (pname, pspan) = self.expect_ident()?;
                self.expect_punct(":")?;
                let ty = self.parse_type()?;
                params.push(Param {
                    name: pname,
                    ty,
                    span: pspan,
                });
                if !self.eat_punct(",") {
                    break;
                }
            }
        }
        self.expect_punct(")")?;
        let ret = if self.eat_punct("->") {
            Some(self.parse_type()?)
        } else {
            None
        };
        let body = self.parse_block()?;
        Ok(Func {
            name,
            params,
            ret,
            body,
            span,
        })
    }

    fn parse_type(&mut self) -> PResult<SrcType> {
        let (name, span) = self.expect_ident()?;
        match name.as_str() {
            "Tensor" => {
                self.expect_punct("<")?;
                let mut dims = Vec::new();
                let mut elem: Option<SrcType> = None;
                loop {
                    // A dimension.
                    match self.peek().clone() {
                        Tok::Int(v) => {
                            self.bump();
                            dims.push(SrcDim::Static(v));
                        }
                        Tok::Punct("?") => {
                            self.bump();
                            dims.push(SrcDim::Dynamic);
                        }
                        _ => break,
                    }
                    // The separator between the dimensions and the element
                    // type: `64x128xf32`.
                    if self.at_punct("x") {
                        if matches!(self.peek_at(1), Tok::Int(_) | Tok::Punct("?")) {
                            self.bump();
                            continue;
                        }
                        self.bump();
                        elem = Some(self.parse_type()?);
                        break;
                    } else if let Tok::Ident(n) = self.peek().clone() {
                        match n.strip_prefix('x') {
                            Some(rest) => {
                                self.bump();
                                elem = Some(self.scalar_type(rest, &span)?);
                                break;
                            }
                            None => break,
                        }
                    } else {
                        break;
                    }
                }
                let elem = match elem {
                    Some(e) => e,
                    None if dims.is_empty() => self.parse_type()?,
                    None => return self.err("expected the element type of the tensor"),
                };
                self.expect_punct(">")?;
                Ok(SrcType::Tensor {
                    dims,
                    elem: Box::new(elem),
                })
            }
            "index" => Ok(SrcType::Index),
            _ => match crate::ir::Type::scalar(&name) {
                Some(crate::ir::Type::Scalar(k)) => Ok(SrcType::Scalar(k)),
                Some(crate::ir::Type::Index) => Ok(SrcType::Index),
                _ => Err(Error {
                    msg: format!("unknown type `{}`", name),
                    span,
                }),
            },
        }
    }

    /// Whether the `<` at the current position starts a list of type
    /// arguments.
    fn looks_like_type_args(&self) -> bool {
        match self.peek_at(1) {
            Tok::Ident(name) => name == "Tensor" || crate::ir::Type::scalar(name).is_some(),
            _ => false,
        }
    }

    /// Parses a scalar type from its name.
    fn scalar_type(&self, name: &str, span: &Span) -> PResult<SrcType> {
        match crate::ir::Type::scalar(name) {
            Some(crate::ir::Type::Scalar(k)) => Ok(SrcType::Scalar(k)),
            Some(crate::ir::Type::Index) => Ok(SrcType::Index),
            _ => Err(Error {
                msg: format!("unknown type `{}`", name),
                span: span.clone(),
            }),
        }
    }

    // -----------------------------------------------------------------
    // Blocks
    // -----------------------------------------------------------------

    fn parse_block(&mut self) -> PResult<Block> {
        let span = self.expect_punct("{")?;
        let mut block = Block {
            span: span.clone(),
            ..Default::default()
        };
        loop {
            if self.at_punct("}") {
                break;
            }
            if matches!(self.peek(), Tok::Eof) {
                return self.err("unexpected end of file, expected `}`");
            }
            if self.at_kw("yield") {
                self.bump();
                let mut values = vec![self.parse_expr()?];
                while self.eat_punct(",") {
                    values.push(self.parse_expr()?);
                }
                self.eat_punct(";");
                block.tail = Some(values);
                break;
            }
            if self.at_kw("let") {
                let stmt_span = self.bump().span;
                let (name, _) = self.expect_ident()?;
                let ty = if self.eat_punct(":") {
                    Some(self.parse_type()?)
                } else {
                    None
                };
                self.expect_punct("=")?;
                let value = self.parse_expr()?;
                // The semicolon that terminates a binding is optional: the
                // binding ends where the next one starts.
                self.eat_punct(";");
                block.stmts.push(Stmt::Let {
                    name,
                    ty,
                    value,
                    span: stmt_span,
                });
                continue;
            }
            // The last expression of a block is the value of the region,
            // every other expression is an effect statement. The semicolon
            // that terminates an effect statement is optional.
            let e = self.parse_expr()?;
            if self.at_punct("}") {
                block.tail = Some(vec![e]);
                break;
            }
            self.eat_punct(";");
            block.stmts.push(Stmt::Effect(e));
        }
        self.expect_punct("}")?;
        Ok(block)
    }

    // -----------------------------------------------------------------
    // Expressions
    // -----------------------------------------------------------------

    fn parse_expr(&mut self) -> PResult<Expr> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> PResult<Expr> {
        let mut lhs = self.parse_and()?;
        while self.at_punct("||") {
            let span = self.bump().span;
            let rhs = self.parse_and()?;
            lhs = Expr::Binary(BinOp::Or, Box::new(lhs), Box::new(rhs), span);
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> PResult<Expr> {
        let mut lhs = self.parse_equality()?;
        while self.at_punct("&&") {
            let span = self.bump().span;
            let rhs = self.parse_equality()?;
            lhs = Expr::Binary(BinOp::And, Box::new(lhs), Box::new(rhs), span);
        }
        Ok(lhs)
    }

    fn parse_equality(&mut self) -> PResult<Expr> {
        let mut lhs = self.parse_comparison()?;
        loop {
            let op = if self.at_punct("==") {
                BinOp::Eq
            } else if self.at_punct("!=") {
                BinOp::Ne
            } else {
                break;
            };
            let span = self.bump().span;
            let rhs = self.parse_comparison()?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs), span);
        }
        Ok(lhs)
    }

    fn parse_comparison(&mut self) -> PResult<Expr> {
        let mut lhs = self.parse_additive()?;
        loop {
            let op = if self.at_punct("<") {
                BinOp::Lt
            } else if self.at_punct("<=") {
                BinOp::Le
            } else if self.at_punct(">") {
                BinOp::Gt
            } else if self.at_punct(">=") {
                BinOp::Ge
            } else {
                break;
            };
            let span = self.bump().span;
            let rhs = self.parse_additive()?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs), span);
        }
        Ok(lhs)
    }

    fn parse_additive(&mut self) -> PResult<Expr> {
        let mut lhs = self.parse_multiplicative()?;
        loop {
            let op = if self.at_punct("+") {
                BinOp::Add
            } else if self.at_punct("-") {
                BinOp::Sub
            } else {
                break;
            };
            let span = self.bump().span;
            let rhs = self.parse_multiplicative()?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs), span);
        }
        Ok(lhs)
    }

    fn parse_multiplicative(&mut self) -> PResult<Expr> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = if self.at_punct("*") {
                BinOp::Mul
            } else if self.at_punct("/") {
                BinOp::Div
            } else if self.at_punct("%") {
                BinOp::Rem
            } else {
                break;
            };
            let span = self.bump().span;
            let rhs = self.parse_unary()?;
            lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs), span);
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> PResult<Expr> {
        let span = self.span();
        if self.at_punct("-") {
            self.bump();
            let e = self.parse_unary()?;
            return Ok(Expr::Unary(UnOp::Neg, Box::new(e), span));
        }
        if self.at_punct("!") {
            self.bump();
            let e = self.parse_unary()?;
            return Ok(Expr::Unary(UnOp::Not, Box::new(e), span));
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> PResult<Expr> {
        let span = self.span();
        match self.peek().clone() {
            Tok::Int(v) => {
                self.bump();
                Ok(Expr::Int(v, span))
            }
            Tok::Float(v) => {
                self.bump();
                Ok(Expr::Float(v, span))
            }
            Tok::Str(text) => {
                self.bump();
                Ok(Expr::Str(text, span))
            }
            Tok::Ident(first) => {
                self.bump();
                // Dotted names: `tensor.empty`.
                let mut name = first;
                while self.at_punct(".") {
                    self.bump();
                    let (part, _) = self.expect_ident()?;
                    name.push('.');
                    name.push_str(&part);
                }
                match name.as_str() {
                    "true" | "false" => return Ok(Expr::Bool(name == "true", span)),
                    "if" => return self.parse_if(),
                    "for" => return self.parse_for(),
                    "map" => return self.parse_map(),
                    "reduce" => return self.parse_reduce(),
                    "tensor.extract" => return self.parse_extract(),
                    _ => {}
                }
                // Explicit type arguments: `tensor.empty<Tensor<4xf32>>()`.
                // The `<` is only a type argument list if a type follows, so
                // that `a < b` stays a comparison.
                let mut type_args = Vec::new();
                if self.at_punct("<") && self.looks_like_type_args() {
                    self.bump();
                    loop {
                        type_args.push(self.parse_type()?);
                        if !self.eat_punct(",") {
                            break;
                        }
                    }
                    self.expect_punct(">")?;
                }
                if self.at_punct("(") {
                    self.bump();
                    let mut args = Vec::new();
                    if !self.at_punct(")") {
                        loop {
                            args.push(self.parse_expr()?);
                            if !self.eat_punct(",") {
                                break;
                            }
                        }
                    }
                    self.expect_punct(")")?;
                    return Ok(Expr::Call {
                        name,
                        type_args,
                        args,
                        span,
                    });
                }
                Ok(Expr::Var(name, span))
            }
            Tok::Punct("(") => {
                self.bump();
                let e = self.parse_expr()?;
                self.expect_punct(")")?;
                Ok(e)
            }
            other => self.err(format!("expected expression, found {}", other)),
        }
    }

    fn parse_if(&mut self) -> PResult<Expr> {
        let span = self.span();
        let cond = self.parse_expr()?;
        let then_block = self.parse_block()?;
        let else_block = if self.at_kw("else") {
            self.bump();
            if self.at_kw("if") {
                let nested = self.parse_if()?;
                let mut blk = Block {
                    span: nested.span(),
                    ..Default::default()
                };
                blk.tail = Some(vec![nested]);
                Some(blk)
            } else {
                Some(self.parse_block()?)
            }
        } else {
            None
        };
        Ok(Expr::If {
            cond: Box::new(cond),
            then_block,
            else_block,
            span,
        })
    }

    fn parse_for(&mut self) -> PResult<Expr> {
        let span = self.span();
        let (var, _) = self.expect_ident()?;
        self.expect_kw("in")?;
        let lb = self.parse_expr()?;
        self.expect_punct("..")?;
        let ub = self.parse_expr()?;
        let step = if self.at_kw("step") {
            self.bump();
            Some(self.parse_expr()?)
        } else {
            None
        };
        let mut iter = Vec::new();
        if self.at_kw("iter") {
            self.bump();
            self.expect_punct("(")?;
            loop {
                let (name, ispan) = self.expect_ident()?;
                self.expect_punct("=")?;
                let init = self.parse_expr()?;
                iter.push(IterBinding {
                    name,
                    init,
                    span: ispan,
                });
                if !self.eat_punct(",") {
                    break;
                }
            }
            self.expect_punct(")")?;
        }
        let body = self.parse_block()?;
        Ok(Expr::For {
            var,
            lb: Box::new(lb),
            ub: Box::new(ub),
            step: step.map(Box::new),
            iter,
            body,
            span,
        })
    }

    /// Parses an optional attribute dictionary: `{unroll = 4}`.
    fn parse_attrs(&mut self) -> PResult<Vec<(String, AttrValue)>> {
        let mut attrs = Vec::new();
        if !self.at_punct("{") {
            return Ok(attrs);
        }
        self.bump();
        if self.at_punct("}") {
            self.bump();
            return Ok(attrs);
        }
        loop {
            let (key, _) = self.expect_ident()?;
            self.expect_punct("=")?;
            let value = match self.peek().clone() {
                Tok::Int(v) => {
                    self.bump();
                    AttrValue::Int(v)
                }
                Tok::Float(v) => {
                    self.bump();
                    AttrValue::Float(v)
                }
                Tok::Ident(s) => {
                    self.bump();
                    AttrValue::Str(s)
                }
                Tok::Punct("-") => {
                    self.bump();
                    match self.peek().clone() {
                        Tok::Int(v) => {
                            self.bump();
                            AttrValue::Int(-v)
                        }
                        Tok::Float(v) => {
                            self.bump();
                            AttrValue::Float(-v)
                        }
                        other => return self.err(format!("expected number, found {}", other)),
                    }
                }
                other => return self.err(format!("expected attribute value, found {}", other)),
            };
            attrs.push((key, value));
            if !self.eat_punct(",") {
                break;
            }
        }
        self.expect_punct("}")?;
        Ok(attrs)
    }

    /// Parses `|a, b|` closure parameters.
    fn parse_params(&mut self) -> PResult<Vec<String>> {
        self.expect_punct("|")?;
        let mut params = Vec::new();
        if !self.at_punct("|") {
            loop {
                let (name, _) = self.expect_ident()?;
                params.push(name);
                if !self.eat_punct(",") {
                    break;
                }
            }
        }
        self.expect_punct("|")?;
        Ok(params)
    }

    fn parse_map(&mut self) -> PResult<Expr> {
        let span = self.span();
        self.expect_punct("(")?;
        let mut inputs = Vec::new();
        if !self.at_punct(")") {
            loop {
                inputs.push(Box::new(self.parse_expr()?));
                if !self.eat_punct(",") {
                    break;
                }
            }
        }
        self.expect_punct(")")?;
        let attrs = self.parse_attrs()?;
        let params = self.parse_params()?;
        let body = self.parse_block()?;
        Ok(Expr::Map {
            inputs,
            attrs,
            params,
            body,
            span,
        })
    }

    fn parse_reduce(&mut self) -> PResult<Expr> {
        let span = self.span();
        self.expect_punct("(")?;
        let init = self.parse_expr()?;
        self.expect_punct(",")?;
        let input = self.parse_expr()?;
        self.expect_punct(")")?;
        let attrs = self.parse_attrs()?;
        let params = self.parse_params()?;
        let body = self.parse_block()?;
        Ok(Expr::Reduce {
            input: Box::new(input),
            init: Box::new(init),
            attrs,
            params,
            body,
            span,
        })
    }

    fn parse_extract(&mut self) -> PResult<Expr> {
        let span = self.span();
        self.expect_punct("(")?;
        let tensor = self.parse_expr()?;
        self.expect_punct(",")?;
        self.expect_punct("[")?;
        let mut indices = Vec::new();
        if !self.at_punct("]") {
            loop {
                indices.push(self.parse_expr()?);
                if !self.eat_punct(",") {
                    break;
                }
            }
        }
        self.expect_punct("]")?;
        self.expect_punct(")")?;
        Ok(Expr::Extract {
            tensor: Box::new(tensor),
            indices: indices.into_iter().map(Box::new).collect(),
            span,
        })
    }
}

pub fn parse(src: &str) -> Result<Program, Error> {
    let toks = crate::lexer::lex(src)?;
    let mut p = Parser::new(toks);
    p.parse_program()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ScalarKind;

    #[test]
    fn parses_a_function_with_a_region() {
        let prog = parse("fn f(a: i32) -> i32 { yield a + 1 }").unwrap();
        assert_eq!(prog.funcs.len(), 1);
        let f = &prog.funcs[0];
        assert_eq!(f.name, "f");
        assert_eq!(f.ret, Some(SrcType::Scalar(ScalarKind::I32)));
        assert!(f.body.tail.is_some());
    }

    #[test]
    fn parses_if_as_an_expression() {
        let prog = parse("fn f(c: bool) -> i32 { if c { yield 1 } else { yield 2 } }").unwrap();
        match &prog.funcs[0].body.tail {
            Some(values) => match &values[0] {
                Expr::If {
                    then_block,
                    else_block,
                    ..
                } => {
                    assert!(then_block.tail.is_some());
                    assert!(else_block.is_some());
                }
                other => panic!("unexpected expression: {:?}", other),
            },
            None => panic!("expected a tail expression"),
        }
    }

    #[test]
    fn parses_for_with_loop_carried_values() {
        let prog =
            parse("fn f(n: i32) -> i32 { yield for i in 0..n step 1 iter(acc = 0) { yield acc } }")
                .unwrap();
        match &prog.funcs[0].body.tail {
            Some(values) => match &values[0] {
                Expr::For { iter, .. } => assert_eq!(iter.len(), 1),
                other => panic!("unexpected expression: {:?}", other),
            },
            None => panic!("expected a tail expression"),
        }
    }

    #[test]
    fn parses_tensor_types() {
        let prog = parse("fn f(x: Tensor<64x128xf32>) { }").unwrap();
        match &prog.funcs[0].params[0].ty {
            SrcType::Tensor { dims, .. } => assert_eq!(dims.len(), 2),
            other => panic!("unexpected type: {:?}", other),
        }
    }

    #[test]
    fn rejects_unbalanced_blocks() {
        assert!(parse("fn f() { ").is_err());
    }
}
