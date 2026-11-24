//! The MLR compiler driver.
//!
//! ```text
//! MLR source -> typed AST -> high level IR -> [canonicalize] -> [domain
//! lowering] -> [bufferization] -> MLIR
//! ```

mod ast;
mod build;
mod check;
mod ir;
mod lexer;
mod parser;
mod passes;

use std::collections::HashMap;
use std::process::ExitCode;

fn usage() -> &'static str {
    "usage: mlr [options] <file.mlr>\n\n\
     Options:\n  \
     -o, --output <file>   write the generated MLIR to <file> (default: stdout)\n      \
     --stage <1-5>         stop after a lowering stage (default: 5)\n      \
     --check               only parse and type check the program\n      \
     --ast                 print the typed AST instead of MLIR\n      \
     --verify              run `mlir-opt` to verify the generated MLIR\n  \
     -h, --help            show this help\n"
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut stage = 5u32;
    let mut check_only = false;
    let mut dump_ast = false;
    let mut verify = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("mlr: error: missing argument for {}", args[i - 1]);
                    return ExitCode::FAILURE;
                }
                output = Some(args[i].clone());
            }
            "--stage" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("mlr: error: missing argument for --stage");
                    return ExitCode::FAILURE;
                }
                stage = match args[i].parse() {
                    Ok(s) if (1..=4).contains(&s) => s,
                    _ => {
                        eprintln!("mlr: error: the stage must be between 1 and 5");
                        return ExitCode::FAILURE;
                    }
                };
            }
            "--check" => check_only = true,
            "--ast" => dump_ast = true,
            "--verify" => verify = true,
            "-h" | "--help" => {
                print!("{}", usage());
                return ExitCode::SUCCESS;
            }
            other if other.starts_with('-') => {
                eprintln!("mlr: error: unknown option `{}`", other);
                eprint!("{}", usage());
                return ExitCode::FAILURE;
            }
            other => {
                if input.is_some() {
                    eprintln!("mlr: error: multiple input files given");
                    return ExitCode::FAILURE;
                }
                input = Some(other.to_string());
            }
        }
        i += 1;
    }

    let path = match input {
        Some(p) => p,
        None => {
            eprint!("{}", usage());
            return ExitCode::FAILURE;
        }
    };

    let src = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("mlr: error: cannot read `{}`: {}", path, e);
            return ExitCode::FAILURE;
        }
    };

    let prog = match parser::parse(&src) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{}", e);
            return ExitCode::FAILURE;
        }
    };

    let typed = match check::check(&prog) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{}", e);
            return ExitCode::FAILURE;
        }
    };

    if check_only {
        eprintln!("mlr: `{}` is valid", path);
        return ExitCode::SUCCESS;
    }

    if dump_ast {
        println!("{:#?}", typed);
        return ExitCode::SUCCESS;
    }

    // Signatures of the user functions, used to type the calls.
    let mut sigs: HashMap<String, build::FuncSig> = HashMap::new();
    for f in &typed.funcs {
        sigs.insert(
            f.name.clone(),
            build::FuncSig {
                params: f.params.iter().map(|(_, t)| t.clone()).collect(),
                ret: f.ret.clone(),
            },
        );
    }

    let mut b = build::IrBuilder::new(&sigs, &path);
    let mut module = b.build_program(&typed);
    if !b.errors.is_empty() {
        for e in &b.errors {
            eprintln!("mlr: error: {}", e);
        }
        return ExitCode::FAILURE;
    }

    if stage >= 2 {
        passes::canonicalize(&mut module);
    }
    if stage >= 3 {
        passes::lower_domain(&mut module);
    }
    if stage >= 4 {
        passes::bufferize::bufferize(&mut module);
        // Bufferization can leave dead allocations behind (for example when
        // a buffer is reused in place).
        passes::canonicalize(&mut module);
    }
    if stage >= 5 {
        // The domain operations become calls into the runtime library.
        passes::runtime_ops::lower_runtime_ops(&mut module);
    }

    if std::env::var("MLR_DUMP_IR").is_ok() {
        for f in &module.funcs {
            eprintln!("== {}", f.name);
            dump_ops(&f.body.blocks[0].ops, 2);
        }
    }
    ir::renumber(&mut module);

    if let Err(e) = ir::verify(&module) {
        eprintln!("mlr: error: the generated IR is invalid: {}", e);
        return ExitCode::FAILURE;
    }

    let mlir = ir::print_module(&module);

    if verify {
        if let Err(e) = verify_with_mlir_opt(&mlir) {
            eprintln!("mlr: error: {}", e);
            return ExitCode::FAILURE;
        }
    }

    match output {
        Some(out) => {
            if let Err(e) = std::fs::write(&out, &mlir) {
                eprintln!("mlr: error: cannot write `{}`: {}", out, e);
                return ExitCode::FAILURE;
            }
        }
        None => print!("{}", mlir),
    }
    ExitCode::SUCCESS
}

fn dump_ops(ops: &[ir::Op], level: usize) {
    for op in ops {
        eprintln!(
            "{}{} results={:?} args={:?}",
            "  ".repeat(level),
            op.name,
            op.results.iter().map(|r| r.id).collect::<Vec<_>>(),
            op.operands.iter().map(|r| r.id).collect::<Vec<_>>()
        );
        for region in &op.regions {
            for block in &region.blocks {
                eprintln!(
                    "{}block args={:?}",
                    "  ".repeat(level + 1),
                    block.args.iter().map(|r| r.id).collect::<Vec<_>>()
                );
                dump_ops(&block.ops, level + 2);
            }
        }
    }
}

/// Runs `mlir-opt` on the generated module to check that it is valid MLIR.
fn verify_with_mlir_opt(mlir: &str) -> Result<(), String> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let mut child = match Command::new("mlir-opt")
        .arg("--allow-unregistered-dialect")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => {
            eprintln!("mlr: note: `mlir-opt` was not found, skipping the verification");
            return Ok(());
        }
    };
    if let Some(stdin) = child.stdin.as_mut() {
        let _ = stdin.write_all(mlir.as_bytes());
    }
    let out = child
        .wait_with_output()
        .map_err(|e| format!("cannot run `mlir-opt`: {}", e))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "the generated MLIR was rejected by mlir-opt:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ))
    }
}
