//! Lexer for MLR.

use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Ident(String),
    Int(i64),
    Float(f64),
    Str(String),
    Punct(&'static str),
    Eof,
}

impl fmt::Display for Tok {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tok::Ident(s) => write!(f, "identifier `{}`", s),
            Tok::Int(v) => write!(f, "integer `{}`", v),
            Tok::Float(v) => write!(f, "float `{}`", v),
            Tok::Str(_) => write!(f, "string literal"),
            Tok::Punct(p) => write!(f, "`{}`", p),
            Tok::Eof => write!(f, "end of file"),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Span {
    pub line: u32,
    pub col: u32,
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

#[derive(Clone, Debug)]
pub struct Token {
    pub tok: Tok,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Error {
    pub msg: String,
    pub span: Span,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: error: {}", self.span, self.msg)
    }
}

const PUNCTS: &[&str] = &[
    "->", "==", "!=", "<=", ">=", "&&", "||", "..", ".", "|", "(", ")", "[", "]", "{", "}", ",",
    ":", ";", "=", "+", "-", "*", "/", "%", "!", "?", "<", ">",
];

pub fn lex(src: &str) -> Result<Vec<Token>, Error> {
    let bytes = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut line = 1u32;
    let mut col = 1u32;

    macro_rules! bump {
        ($n:expr) => {{
            for _ in 0..$n {
                if bytes[i] == b'\n' {
                    line += 1;
                    col = 1;
                } else {
                    col += 1;
                }
                i += 1;
            }
        }};
    }

    while i < bytes.len() {
        let c = bytes[i];
        if c == b' ' || c == b'\t' || c == b'\r' || c == b'\n' {
            bump!(1);
            continue;
        }
        if c == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                bump!(1);
            }
            continue;
        }
        if c == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            let start = (line, col);
            bump!(2);
            loop {
                if i >= bytes.len() {
                    return Err(Error {
                        msg: "unterminated block comment".into(),
                        span: Span {
                            line: start.0,
                            col: start.1,
                        },
                    });
                }
                if bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                    bump!(2);
                    break;
                }
                bump!(1);
            }
            continue;
        }
        let span = Span { line, col };
        if c == b'x' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
            // The dimension separator of a tensor type: `64x128xf32`.
            bump!(1);
            out.push(Token {
                tok: Tok::Punct("x"),
                span,
            });
            continue;
        }
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                bump!(1);
            }
            let text = std::str::from_utf8(&bytes[start..i]).unwrap().to_string();
            out.push(Token {
                tok: Tok::Ident(text),
                span,
            });
            continue;
        }
        if c.is_ascii_digit() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                bump!(1);
            }
            let mut is_float = false;
            if i + 1 < bytes.len() && bytes[i] == b'.' && bytes[i + 1].is_ascii_digit() {
                is_float = true;
                bump!(1);
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    bump!(1);
                }
            }
            if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                let mut j = i + 1;
                if j < bytes.len() && (bytes[j] == b'+' || bytes[j] == b'-') {
                    j += 1;
                }
                if j < bytes.len() && bytes[j].is_ascii_digit() {
                    is_float = true;
                    while i < bytes.len() && bytes[i] != b' ' && bytes[i] != b'\n' {
                        bump!(1);
                    }
                }
            }
            let text = std::str::from_utf8(&bytes[start..i]).unwrap();
            if is_float {
                let v: f64 = text.parse().map_err(|_| Error {
                    msg: format!("invalid float literal `{}`", text),
                    span: span.clone(),
                })?;
                out.push(Token {
                    tok: Tok::Float(v),
                    span,
                });
            } else {
                let v: i64 = text.parse().map_err(|_| Error {
                    msg: format!("integer literal `{}` is out of range", text),
                    span: span.clone(),
                })?;
                out.push(Token {
                    tok: Tok::Int(v),
                    span,
                });
            }
            continue;
        }
        // string literals
        if c == b'"' {
            bump!(1);
            let mut text = String::new();
            loop {
                if i >= bytes.len() {
                    return Err(Error {
                        msg: "unterminated string literal".into(),
                        span: span.clone(),
                    });
                }
                match bytes[i] {
                    b'"' => {
                        bump!(1);
                        break;
                    }
                    b'\\' => {
                        bump!(1);
                        if i >= bytes.len() {
                            return Err(Error {
                                msg: "unterminated escape sequence".into(),
                                span: span.clone(),
                            });
                        }
                        let ch = match bytes[i] {
                            b'n' => '\n',
                            b't' => '\t',
                            b'r' => '\r',
                            b'0' => '\0',
                            b'\\' => '\\',
                            b'"' => '"',
                            other => other as char,
                        };
                        text.push(ch);
                        bump!(1);
                    }
                    _ => {
                        let start = i;
                        bump!(1);
                        text.push_str(std::str::from_utf8(&bytes[start..i]).unwrap());
                    }
                }
            }
            out.push(Token {
                tok: Tok::Str(text),
                span,
            });
            continue;
        }
        let mut matched = None;
        for p in PUNCTS {
            if src[i..].starts_with(p) {
                matched = Some(*p);
                break;
            }
        }
        match matched {
            Some(p) => {
                bump!(p.len());
                out.push(Token {
                    tok: Tok::Punct(p),
                    span,
                });
            }
            None => {
                return Err(Error {
                    msg: format!("unexpected character `{}`", c as char),
                    span,
                })
            }
        }
    }

    out.push(Token {
        tok: Tok::Eof,
        span: Span { line, col },
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexes_identifiers_and_punctuation() {
        let toks = lex("fn add(a: i32) -> i32 { }").unwrap();
        let kinds: Vec<&Tok> = toks.iter().map(|t| &t.tok).collect();
        assert_eq!(kinds[0], &Tok::Ident("fn".into()));
        assert_eq!(kinds[1], &Tok::Ident("add".into()));
        assert_eq!(kinds[2], &Tok::Punct("("));
        assert_eq!(kinds[6], &Tok::Punct(")"));
        assert_eq!(kinds[7], &Tok::Punct("->"));
        assert!(matches!(kinds.last(), Some(Tok::Eof)));
    }

    #[test]
    fn lexes_numbers() {
        let toks = lex("1 2.5 0.25 1e3").unwrap();
        assert_eq!(toks[0].tok, Tok::Int(1));
        assert_eq!(toks[1].tok, Tok::Float(2.5));
        assert_eq!(toks[2].tok, Tok::Float(0.25));
        assert!(matches!(toks[3].tok, Tok::Float(_)));
    }

    #[test]
    fn lexes_dotted_names() {
        let toks = lex("tensor.empty").unwrap();
        assert_eq!(toks[0].tok, Tok::Ident("tensor".into()));
        assert_eq!(toks[1].tok, Tok::Punct("."));
        assert_eq!(toks[2].tok, Tok::Ident("empty".into()));
    }

    #[test]
    fn skips_comments() {
        let toks = lex("// line\n/* block */ 1").unwrap();
        assert_eq!(toks[0].tok, Tok::Int(1));
    }

    #[test]
    fn reports_unterminated_comment() {
        assert!(lex("/* unterminated").is_err());
    }
}
