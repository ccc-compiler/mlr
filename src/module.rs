//! Loading a program that spans several files.
//!
//! `use "relative/path.mlr"` pulls the declarations of another file into a
//! flat namespace. The files are resolved relative to the directory of the
//! file that contains the `use`, cycles are reported, and duplicate
//! definitions are caught by the checker like duplicates in one file.

use crate::ast::Program;
use crate::lexer::Error;

use std::path::{Path, PathBuf};

pub fn load(path: &Path) -> Result<Program, Error> {
    let mut stack: Vec<PathBuf> = Vec::new();
    load_one(path, &mut stack)
}

fn error(msg: String) -> Error {
    Error {
        msg,
        span: crate::lexer::Span { line: 0, col: 0 },
    }
}

fn load_one(path: &Path, stack: &mut Vec<PathBuf>) -> Result<Program, Error> {
    let canon = std::fs::canonicalize(path)
        .map_err(|e| error(format!("cannot read `{}`: {}", path.display(), e)))?;
    if stack.contains(&canon) {
        return Err(error(format!("circular `use` of `{}`", path.display())));
    }
    let src = std::fs::read_to_string(&canon)
        .map_err(|e| error(format!("cannot read `{}`: {}", path.display(), e)))?;
    let prog = crate::parser::parse(&src)?;
    let base = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    stack.push(canon);
    let mut merged = Program::default();
    // The dependencies come first, so that a file can use the structs of
    // the files it pulls in.
    for u in &prog.uses {
        let dep = load_one(&base.join(&u.path), stack)?;
        merged.structs.extend(dep.structs);
        merged.impls.extend(dep.impls);
        merged.funcs.extend(dep.funcs);
    }
    merged.structs.extend(prog.structs);
    merged.impls.extend(prog.impls);
    merged.funcs.extend(prog.funcs);
    stack.pop();
    Ok(merged)
}

/// Whether two files end up as the same file (used by the tests).
#[allow(dead_code)]
pub fn same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}
