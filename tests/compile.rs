//! Integration tests: compile MLR programs, check the generated MLIR and run
//! the programs end to end.

use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// Guards the end-to-end runs: each one spawns the compiler and the LLVM
/// tool chain, and on a small machine the tests in parallel run out of
/// memory (`Killed: 9`) before they finish.
fn e2e_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Whether the external LLVM tool chain that the run scripts need is present.
///
/// This is deliberately separate from "did the program run": a missing tool
/// chain is a skip, a failed run is always a test failure.
fn tools_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        ["mlir-opt", "mlir-translate", "clang"].iter().all(|tool| {
            Command::new(tool)
                .arg("--version")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok()
        })
    })
}

/// In CI the tool chain must be present; locally its absence is a skip.
fn require_e2e() -> bool {
    std::env::var_os("MLR_REQUIRE_E2E").is_some()
}

fn tmp_dir() -> std::path::PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("mlr-test-{}-{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Compiles a snippet and returns the generated MLIR.
fn compile(src: &str) -> String {
    let dir = tmp_dir();
    let input = dir.join("test.mlr");
    std::fs::write(&input, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_mlr"))
        .arg(&input)
        .output()
        .expect("failed to run mlr");
    assert!(
        out.status.success(),
        "compilation failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

/// Compiles a snippet that is expected to fail.
fn compile_err(src: &str) -> String {
    let dir = tmp_dir();
    let input = dir.join("bad.mlr");
    std::fs::write(&input, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_mlr"))
        .arg(&input)
        .output()
        .expect("failed to run mlr");
    assert!(!out.status.success(), "expected a compilation error");
    String::from_utf8(out.stderr).unwrap()
}

/// Compiles a snippet at a given lowering stage.
fn compile_stage(src: &str, stage: u32) -> String {
    let dir = tmp_dir();
    let input = dir.join("stage.mlr");
    std::fs::write(&input, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_mlr"))
        .arg(&input)
        .arg("--stage")
        .arg(stage.to_string())
        .output()
        .expect("failed to run mlr");
    assert!(
        out.status.success(),
        "compilation failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

/// Verifies the generated MLIR with `mlir-opt`, if it is available.
fn verify_mlir(mlir: &str) {
    let dir = tmp_dir();
    let file = dir.join("out.mlir");
    std::fs::write(&file, mlir).unwrap();
    match Command::new("mlir-opt")
        .arg("--allow-unregistered-dialect")
        .arg(&file)
        .arg("-o")
        .arg(std::path::Path::new("/dev/null"))
        .output()
    {
        Ok(out) => assert!(
            out.status.success(),
            "mlir-opt rejected the generated MLIR:\n{}\n{}",
            mlir,
            String::from_utf8_lossy(&out.stderr)
        ),
        Err(_) => {
            if require_e2e() {
                panic!("MLR_REQUIRE_E2E is set but mlir-opt was not found");
            }
            eprintln!("skipping MLIR verification: mlir-opt not found");
        }
    }
}

/// Compiles, lowers and runs a program; returns its output.
fn run(src: &str) -> Option<String> {
    let dir = tmp_dir();
    let input = dir.join("run.mlr");
    std::fs::write(&input, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_mlr"))
        .arg(&input)
        .arg("-o")
        .arg(dir.join("prog.mlir"))
        .output()
        .expect("failed to run mlr");
    assert!(
        out.status.success(),
        "compilation failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/run.sh");
    run_script_file(&script, &input)
}

/// Runs one of the `scripts/run*.sh` scripts on `input`.
///
/// Returns `None` only when the tool chain is genuinely unavailable (and
/// `MLR_REQUIRE_E2E` is unset). A non-zero exit while the tools are present is
/// a real failure and panics, so end-to-end breakage can no longer be silently
/// skipped.
fn run_script_file(script: &std::path::Path, input: &std::path::Path) -> Option<String> {
    if !script.exists() {
        if require_e2e() {
            panic!("MLR_REQUIRE_E2E is set but `{}` is missing", script.display());
        }
        eprintln!("skipping: `{}` is missing", script.display());
        return None;
    }
    if !tools_available() {
        if require_e2e() {
            panic!("MLR_REQUIRE_E2E is set but mlir-opt/mlir-translate/clang are missing");
        }
        eprintln!("skipping: the LLVM tool chain is not available");
        return None;
    }
    let _guard = e2e_lock().lock().unwrap_or_else(|e| e.into_inner());
    let out = Command::new("sh")
        .arg(script)
        .arg(input)
        .output()
        .expect("failed to run the script");
    assert!(
        out.status.success(),
        "the program failed to lower or run:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

// ---------------------------------------------------------------------------
// Front end
// ---------------------------------------------------------------------------

#[test]
fn compiles_a_function() {
    let mlir = compile(
        r#"
fn add(a: i32, b: i32) -> i32 {
    yield a + b
}
"#,
    );
    assert!(mlir.contains("func.func @add(%0: i32, %1: i32) -> i32"));
    assert!(mlir.contains("arith.addi"));
    verify_mlir(&mlir);
}

#[test]
fn regions_become_structured_operations() {
    let mlir = compile(
        r#"
fn sign(x: i32) -> i32 {
    if x < 0 {
        yield -1
    } else {
        yield 1
    }
}
"#,
    );
    assert!(mlir.contains("scf.if"));
    assert!(mlir.contains("scf.yield"));
    verify_mlir(&mlir);
}

#[test]
fn loops_carry_their_state_explicitly() {
    let mlir = compile(
        r#"
fn sum_to(n: i32) -> i32 {
    let total = for i in 0..n step 1 iter(acc = 0) {
        yield acc + i
    }
    yield total
}
"#,
    );
    assert!(mlir.contains("scf.for"));
    assert!(mlir.contains("iter_args"));
    verify_mlir(&mlir);
}

#[test]
fn map_is_lowered_to_a_domain_operation() {
    let mlir = compile_stage(
        r#"
fn scale(x: Tensor<4xf32>, s: f32) -> Tensor<4xf32> {
    yield map(x) |v| {
        yield v * s
    }
}
"#,
        3,
    );
    assert!(mlir.contains("linalg.generic"));
    assert!(mlir.contains("iterator_types"));
    verify_mlir(&mlir);
}

#[test]
fn bufferization_produces_memrefs() {
    let mlir = compile_stage(
        r#"
fn scale(x: Tensor<4xf32>, s: f32) -> Tensor<4xf32> {
    yield map(x) |v| {
        yield v * s
    }
}
"#,
        4,
    );
    assert!(mlir.contains("memref"));
    assert!(mlir.contains("memref.load"));
    assert!(mlir.contains("memref.store"));
    verify_mlir(&mlir);
}

#[test]
fn locations_are_attached_to_operations() {
    let mlir = compile(
        r#"
fn add(a: i32, b: i32) -> i32 {
    yield a + b
}
"#,
    );
    assert!(mlir.contains("loc("), "{}", mlir);
}

#[test]
fn effects_are_recorded() {
    let mlir = compile(
        r#"
fn pure_fn(a: i32) -> i32 {
    yield a
}
"#,
    );
    assert!(mlir.contains("effects = \"pure\""));
}

#[test]
fn rejects_type_errors() {
    let err = compile_err("fn main() { let x: i32 = 1.5 }");
    assert!(err.contains("error"), "{}", err);
}

#[test]
fn reports_errors_of_every_function() {
    // Three broken functions: the checker must not stop at the first one.
    let err = compile_err(
        r#"
fn a() { yield nope }
fn b() { let x: i32 = true }
fn c() { yield undefined_too }
"#,
    );
    assert!(err.contains("undefined variable `nope`"), "{}", err);
    assert!(err.contains("undefined variable `undefined_too`"), "{}", err);
    assert!(err.contains("cannot initialize `x`"), "{}", err);
    // The errors are reported in source order.
    let nope = err.find("undefined variable `nope`").unwrap();
    let init = err.find("cannot initialize `x`").unwrap();
    let undef = err.find("undefined variable `undefined_too`").unwrap();
    assert!(nope < init && init < undef, "{}", err);
}

#[test]
fn rejects_undefined_variables() {
    let err = compile_err("fn main() { yield y }");
    assert!(err.contains("undefined variable `y`"), "{}", err);
}

#[test]
fn rejects_unknown_functions() {
    let err = compile_err("fn main() { yield nope(1) }");
    assert!(err.contains("unknown function `nope`"), "{}", err);
}

#[test]
fn rejects_shape_mismatches() {
    let err = compile_err(
        r#"
fn f(a: Tensor<4xf32>, b: Tensor<8xf32>) -> Tensor<4xf32> {
    yield map(a, b) |x, y| {
        yield x + y
    }
}
"#,
    );
    assert!(err.contains("shape mismatch"), "{}", err);
}

#[test]
fn rejects_syntax_errors() {
    let err = compile_err("fn main( {");
    assert!(err.contains("error"), "{}", err);
}

#[test]
fn rejects_a_non_string_tensor_name() {
    let err = compile_err(
        r#"
fn main() {
    let t = tensor.zeros<Tensor<6xf32>>()
    mlr.save_pt("out.pt", t, 1)
}
"#,
    );
    assert!(err.contains("must be a string"), "{}", err);
}

// ---------------------------------------------------------------------------
// End to end
// ---------------------------------------------------------------------------

#[test]
fn runs_a_scalar_program() {
    let Some(out) = run(r#"
fn fact(n: i32) -> i32 {
    if n <= 1 {
        yield 1
    } else {
        yield n * fact(n - 1)
    }
}

fn main() {
    print(fact(5))
    print(fact(10))
}
"#)
    else {
        eprintln!("skipping: the LLVM tool chain is not available");
        return;
    };
    assert_eq!(out.lines().collect::<Vec<_>>(), vec!["120", "3628800"]);
}

#[test]
fn prints_strings_and_values() {
    let Some(out) = run(r#"
fn main() {
    let n = 3
    print("n = ", n, ", twice = ", n * 2, "!")
    print("plain")
}
"#)
    else {
        eprintln!("skipping: the LLVM tool chain is not available");
        return;
    };
    assert_eq!(
        out.lines().collect::<Vec<_>>(),
        vec!["n = 3, twice = 6!", "plain"]
    );
}

#[test]
fn runs_a_tensor_program() {
    let Some(out) = run(r#"
fn scale(x: Tensor<4xf32>, s: f32) -> Tensor<4xf32> {
    yield map(x) |v| {
        yield v * s
    }
}

fn main() {
    let t = tensor.zeros<Tensor<4xf32>>()
    let u = scale(t, 2.5)
    let e = tensor.extract(u, [0])
    print(e)
}
"#)
    else {
        eprintln!("skipping: the LLVM tool chain is not available");
        return;
    };
    assert_eq!(out.trim(), "0.000000");
}

// ---------------------------------------------------------------------------
// Tensor data and the runtime
// ---------------------------------------------------------------------------

/// Compiles a file in the repository with the given script.
fn run_script(script: &str, file: &str) -> Option<String> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    run_script_file(&root.join(script), &root.join(file))
}

#[test]
fn embeds_a_tensor_at_compile_time() {
    let dir = tmp_dir();
    let data = dir.join("data.bin");
    let mut bytes = Vec::new();
    for v in [1.0f32, 2.0, 3.0, 4.0] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&data, &bytes).unwrap();
    let src = format!(
        r#"
fn main() {{
    let t = mlr.load<Tensor<4xf32>>("{}")
    print(tensor.extract(t, [0]))
}}
"#,
        data.display()
    );
    let mlir = compile(&src);
    assert!(mlir.contains("memref.global"), "{}", mlir);
    assert!(mlir.contains("memref.get_global"), "{}", mlir);
    verify_mlir(&mlir);
}

#[test]
fn domain_operations_become_runtime_calls() {
    let mlir = compile(
        r#"
fn dot(a: Tensor<4x4xf32>, b: Tensor<4x4xf32>) -> f32 {
    yield tensor.extract(matmul(a, b), [0, 0])
}
"#,
    );
    assert!(mlir.contains("func.call @mlr_matmul"), "{}", mlir);
    assert!(mlir.contains("func.func private @mlr_matmul"), "{}", mlir);
    verify_mlir(&mlir);
}

#[test]
fn saves_a_checkpoint_with_runtime_calls() {
    let mlir = compile(
        r#"
fn main() {
    let t = tensor.zeros<Tensor<6xf32>>()
    mlr.save_pt("out.pt", t, "w")
}
"#,
    );
    assert!(mlir.contains("func.call @mlr_save_pt_begin"), "{}", mlir);
    assert!(mlir.contains("func.call @mlr_save_pt_add"), "{}", mlir);
    assert!(mlir.contains("func.call @mlr_save_pt_end"), "{}", mlir);
    assert!(mlir.contains("memref.global"), "{}", mlir);
    verify_mlir(&mlir);
}

#[test]
fn builds_a_struct_and_reads_its_fields() {
    let mlir = compile(
        r#"
struct Vec3 {
    x: f32,
    y: f32,
    z: f32
}

fn main() {
    let a = Vec3 { x: 1.0, y: 2.0, z: 3.0 }
    print(a.y)
}
"#,
    );
    // The struct is a bundle of three constants.
    assert!(mlir.contains("arith.constant 2.0"), "{}", mlir);
    verify_mlir(&mlir);
}

#[test]
fn runs_a_program_with_structs_and_methods() {
    let Some(out) = run(
        r#"
struct Vec3 {
    x: f32,
    y: f32,
    z: f32
}

impl Vec3 {
    fn dot(self, other: Vec3) -> f32 {
        yield self.x * other.x + self.y * other.y + self.z * other.z
    }
    fn scaled(self, s: f32) -> Vec3 {
        yield Vec3 { x: self.x * s, y: self.y * s, z: self.z * s }
    }
}

fn main() {
    let a = Vec3 { x: 1.0, y: 2.0, z: 3.0 }
    let b = Vec3 { x: 4.0, y: 5.0, z: 6.0 }
    print(a.dot(b))
    let c = a.scaled(10.0)
    print(c.x)
}
"#,
    ) else {
        eprintln!("skipping: the LLVM tool chain is not available");
        return;
    };
    assert_eq!(
        out.lines().collect::<Vec<_>>(),
        vec!["32.000000", "10.000000"]
    );
}

#[test]
fn runs_a_loop_that_carries_a_struct() {
    let Some(out) = run(
        r#"
struct Acc {
    total: f32
}

fn main() {
    let start = Acc { total: 1.0 }
    let r = for i in 0..4 step 1 iter(acc = start) {
        yield acc with { total: acc.total + 1.0 }
    }
    print(r.total)
}
"#,
    ) else {
        eprintln!("skipping: the LLVM tool chain is not available");
        return;
    };
    assert_eq!(out.lines().collect::<Vec<_>>(), vec!["5.000000"]);
}

#[test]
fn rejects_an_unknown_struct_field() {
    let err = compile_err(
        r#"
struct S {
    x: f32
}

fn main() {
    let a = S { x: 1.0 }
    print(a.y)
}
"#,
    );
    assert!(err.contains("has no field `y`"), "{}", err);
}

#[test]
fn rejects_a_missing_struct_field() {
    let err = compile_err(
        r#"
struct S {
    x: f32,
    y: f32
}

fn main() {
    let a = S { x: 1.0 }
}
"#,
    );
    assert!(err.contains("missing") || err.contains("has 2 fields"), "{}", err);
}

#[test]
fn rejects_a_wrong_field_type() {
    let err = compile_err(
        r#"
struct S {
    x: Tensor<4xf32>
}

fn main() {
    let t = tensor.zeros<Tensor<8xf32>>()
    let a = S { x: t }
}
"#,
    );
    assert!(err.contains("has type"), "{}", err);
}

#[test]
fn rejects_a_recursive_struct() {
    let err = compile_err(
        r#"
struct S {
    x: S
}
"#,
    );
    assert!(err.contains("contains itself"), "{}", err);
}

#[test]
fn rejects_an_unknown_method() {
    let err = compile_err(
        r#"
struct S {
    x: f32
}

fn main() {
    let a = S { x: 1.0 }
    a.nope()
}
"#,
    );
    assert!(err.contains("no method `nope`"), "{}", err);
}

#[test]
fn rejects_a_struct_without_assignment() {
    let err = compile_err(
        r#"
struct S {
    x: f32
}

impl S {
    fn set(self, v: f32) {
        self.x = v
    }
}
"#,
    );
    assert!(err.contains("error"), "{}", err);
}

#[test]
fn unpacks_the_loop_carried_values() {
    // `let (a, b) = for ...` unpacks into all of the loop-carried values.
    let Some(out) = run(
        r#"
fn main() {
    let (a, b) = for step in 0..3 step 1 iter(x = 1.0, y = 10) {
        yield x + 1.0, y + 1
    }
    print(a)
    print(b)
}
"#,
    ) else {
        eprintln!("skipping: the LLVM tool chain is not available");
        return;
    };
    assert_eq!(out.lines().collect::<Vec<_>>(), vec!["4.000000", "13"]);
}

#[test]
fn carries_a_struct_and_a_scalar_in_one_loop() {
    // The regression of the NaN bug: a struct and a scalar carried by the
    // same loop, the struct updated with a `with`.
    let Some(out) = run(
        r#"
struct St {
    w: Tensor<2x2xf32>,
    m: Tensor<2x2xf32>
}

fn main() {
    let w0 = tensor.zeros<Tensor<2x2xf32>>()
    let start = St { w: w0, m: w0 }
    let trained = for step in 0..3 step 1 iter(state = start, c = 1.0) {
        let w1 = map(state.w) |a| {
            yield a - 0.1 * c
        }
        yield state with { w: w1 }, c * 0.5
    }
    print(tensor.extract(trained.w, [0, 0]))
}
"#,
    ) else {
        eprintln!("skipping: the LLVM tool chain is not available");
        return;
    };
    // 0.0 - 0.1*1 - 0.1*0.5 - 0.1*0.25 = -0.175
    assert_eq!(out.trim(), "-0.175000", "{}", out);
}

#[test]
fn carries_tensor_insert_state_in_a_loop() {
    let Some(out) = run(
        r#"
struct St {
    w: Tensor<2x2xf32>,
    c: Tensor<1xf32>
}

fn main() {
    let start = St {
        w: tensor.zeros<Tensor<2x2xf32>>(),
        c: tensor.insert(tensor.zeros<Tensor<1xf32>>(), [0], 0.1)
    }
    let trained = for step in 0..3 step 1 iter(state = start) {
        let c1 = tensor.insert(state.c, [0], tensor.extract(state.c, [0]) * 0.9)
        print(tensor.extract(c1, [0]))
        yield state with { c: c1 }
    }
}
"#,
    ) else {
        eprintln!("skipping: the LLVM tool chain is not available");
        return;
    };
    // 0.1 * 0.9^t
    assert_eq!(
        out.lines().collect::<Vec<_>>(),
        vec!["0.090000", "0.081000", "0.072900"]
    );
}

#[test]
fn uses_the_scalar_math_of_the_library() {
    let mlir = compile(
        r#"
fn main() {
    print(abs(0.0 - 5.0))
    print(min(3.0, 7.0))
    print(clamp(10.0, 0.0, 5.0))
    print(pow(2.0, 10.0))
    print(pi())
}
"#,
    );
    assert!(mlir.contains("func.call @mlr_abs"), "{}", mlir);
    assert!(mlir.contains("func.call @mlr_clamp"), "{}", mlir);
    assert!(mlir.contains("func.call @mlr_pow"), "{}", mlir);
    verify_mlir(&mlir);
}

#[test]
fn uses_the_tensor_functions_of_the_library() {
    let mlir = compile(
        r#"
fn main() {
    let a = tensor.zeros<Tensor<2x2xf32>>()
    let b = tensor.zeros<Tensor<2x2xf32>>()
    let m = tmin(a, b)
    let n = mean(a)
    let v = norm2(a)
    let r = reshape<Tensor<4xf32>>(m)
    print(n + v)
    print(tensor.extract(r, [0]))
}
"#,
    );
    assert!(mlir.contains("mlr_tmin"), "{}", mlir);
    assert!(mlir.contains("mlr_mean"), "{}", mlir);
    assert!(mlir.contains("mlr_norm2"), "{}", mlir);
    assert!(mlir.contains("mlr_copy"), "{}", mlir);
    verify_mlir(&mlir);
}

#[test]
fn runs_the_shell_bridge() {
    let Some(out) = run(
        r#"
fn main() {
    let code = exec("true")
    print(code)
    let bad = exec("false")
    print(bad)
    let n = exec_i("echo 42")
    print(n)
    let f = exec_f("echo 2.5")
    print(f)
    let a = exec_i("echo {0}", 40)
    print(a + 2)
}
"#,
    ) else {
        eprintln!("skipping: the LLVM tool chain is not available");
        return;
    };
    assert_eq!(
        out.lines().collect::<Vec<_>>(),
        vec!["0", "1", "42", "2.500000", "42"]
    );
}

#[test]
fn rejects_a_shell_command_that_is_not_a_literal() {
    let err = compile_err(
        r#"
fn main() {
    exec(1)
}
"#,
    );
    assert!(err.contains("string literal"), "{}", err);
}

#[test]
fn rejects_exec_in_a_differentiated_function() {
    let err = compile_err(
        r#"
fn loss(w: Tensor<2x2xf32>, x: Tensor<2x2xf32>) -> f32 {
    let c = exec("true")
    yield sum(matmul(x, w))
}

fn main() {
    let w = tensor.zeros<Tensor<2x2xf32>>()
    let x = tensor.zeros<Tensor<2x2xf32>>()
    let (l, dw, _) = value_and_grad(loss, w, x)
    print(l)
}
"#,
    );
    assert!(err.contains("cannot be differentiated"), "{}", err);
}

#[test]
fn differentiates_a_loop() {
    // loss = sum_i w[i] * x[i]; the gradient is exactly the other tensor.
    let dir = tmp_dir();
    let w = dir.join("w.bin");
    let x = dir.join("x.bin");
    std::fs::write(&w, [1.0f32, 2.0, 3.0, 4.0].map(|v| v.to_le_bytes()).concat()).unwrap();
    std::fs::write(&x, [10.0f32, 20.0, 30.0, 40.0].map(|v| v.to_le_bytes()).concat()).unwrap();
    let Some(out) = run(&format!(
        r#"
fn loss(w: Tensor<4xf32>, x: Tensor<4xf32>) -> f32 {{
    let zero: f32 = 0.0
    let total = for i in 0..4 step 1 iter(acc = zero) {{
        yield acc + tensor.extract(w, [i]) * tensor.extract(x, [i])
    }}
    yield total
}}

fn main() {{
    let w = mlr.load<Tensor<4xf32>>("{}")
    let x = mlr.load<Tensor<4xf32>>("{}")
    let (l, dw, dx) = value_and_grad(loss, w, x)
    print(l)
    print(tensor.extract(dw, [0]))
    print(tensor.extract(dw, [3]))
    print(tensor.extract(dx, [1]))
}}
"#,
        w.display(),
        x.display()
    )) else {
        eprintln!("skipping: the LLVM tool chain is not available");
        return;
    };
    assert_eq!(
        out.lines().collect::<Vec<_>>(),
        vec!["300.000000", "10.000000", "40.000000", "2.000000"]
    );
}

#[test]
fn differentiates_an_if() {
    let dir = tmp_dir();
    let w1 = dir.join("w1.bin");
    let w2 = dir.join("w2.bin");
    std::fs::write(&w1, [1.0f32, 5.0].map(|v| v.to_le_bytes()).concat()).unwrap();
    std::fs::write(&w2, [5.0f32, 1.0].map(|v| v.to_le_bytes()).concat()).unwrap();
    let Some(out) = run(&format!(
        r#"
fn loss(w: Tensor<2xf32>) -> f32 {{
    let a = tensor.extract(w, [0])
    let b = tensor.extract(w, [1])
    let l = if a > b {{
        yield a * 2.0
    }} else {{
        yield b * 3.0
    }}
    yield l
}}

fn main() {{
    let w1 = mlr.load<Tensor<2xf32>>("{}")
    let w2 = mlr.load<Tensor<2xf32>>("{}")
    let (l1, dw1) = value_and_grad(loss, w1)
    let (l2, dw2) = value_and_grad(loss, w2)
    print(l1)
    print(tensor.extract(dw1, [0]))
    print(tensor.extract(dw1, [1]))
    print(l2)
    print(tensor.extract(dw2, [0]))
    print(tensor.extract(dw2, [1]))
}}
"#,
        w1.display(),
        w2.display()
    )) else {
        eprintln!("skipping: the LLVM tool chain is not available");
        return;
    };
    // w1 = [1, 5]: the else branch (b * 3); w2 = [5, 1]: the then (a * 2).
    assert_eq!(
        out.lines().collect::<Vec<_>>(),
        vec!["15.000000", "0.000000", "3.000000", "10.000000", "2.000000", "0.000000"]
    );
}

#[test]
fn rejects_a_loop_that_uses_its_carried_value_nonlinearly() {
    let err = compile_err(
        r#"
fn loss(w: Tensor<4xf32>) -> f32 {
    let zero: f32 = 0.0
    let total = for i in 0..4 step 1 iter(acc = zero) {
        yield acc * tensor.extract(w, [i])
    }
    yield total
}

fn main() {
    let w = tensor.zeros<Tensor<4xf32>>()
    let (l, dw) = value_and_grad(loss, w)
    print(l)
}
"#,
    );
    assert!(err.contains("cannot be differentiated"), "{}", err);
    assert!(err.contains("multiplied by a constant"), "{}", err);
}

#[test]
fn pulls_in_another_file_with_use() {
    let Some(out) = run_script("scripts/run.sh", "examples/modules/use_struct.mlr") else {
        eprintln!("skipping: the LLVM tool chain is not available");
        return;
    };
    // dot([1, 2], [3, 4]) = 11.
    assert_eq!(out.trim(), "11.000000");
}

#[test]
fn instantiates_a_generic_function() {
    let Some(out) = run_script("scripts/run.sh", "examples/modules/generic.mlr") else {
        eprintln!("skipping: the LLVM tool chain is not available");
        return;
    };
    // filled<4>(2.0)[3] = 2; sum([2,2,2,2] * [0.5,..]) = 4; sum(filled<3>(10)) = 30.
    assert_eq!(out.lines().collect::<Vec<_>>(), vec!["2.000000", "4.000000", "30.000000"]);
}

#[test]
fn reports_a_use_of_a_missing_file_and_a_circular_use() {
    let err = compile_err(
        r#"
use "no/such/file.mlr"

fn main() {
    print(1)
}
"#,
    );
    assert!(err.contains("cannot read"), "{}", err);
}

#[test]
fn rejects_a_generic_call_without_shape_arguments() {
    let err = compile_err(
        r#"
fn filled<N: i64>(v: f32) -> Tensor<Nxf32> {
    let t = tensor.zeros<Tensor<Nxf32>>()
    let out = for i in 0..N step 1 iter(t = t) {
        yield tensor.insert(t, [i], v)
    }
    yield out
}

fn main() {
    let a = filled(2.0)
    print(tensor.extract(a, [0]))
}
"#,
    );
    assert!(err.contains("shape argument"), "{}", err);
}

#[test]
fn reduces_and_reorders_along_an_axis() {
    // A 3x3 matrix; every printed value follows from the rows
    // [3,1,2], [9,7,8] and [5,6,4].
    let dir = tmp_dir();
    let mat = dir.join("mat.bin");
    std::fs::write(
        &mat,
        [3.0f32, 1.0, 2.0, 9.0, 7.0, 8.0, 5.0, 6.0, 4.0]
            .map(|v| v.to_le_bytes())
            .concat(),
    )
    .unwrap();
    let Some(out) = run(&format!(
        r#"
fn main() {{
    let m = mlr.load<Tensor<3x3xf32>>("{}")
    let s = sum_axis(m, 1)
    print(tensor.extract(s, [0]))
    print(tensor.extract(s, [1]))
    let mx = tmax_axis(m, 0)
    print(tensor.extract(mx, [0]))
    let am = argmin_axis(m, 1)
    print(tensor.extract(am, [0]))
    let srt = sort(m)
    print(tensor.extract(srt, [0, 0]))
    print(tensor.extract(srt, [2, 0]))
    let a = argsort(m)
    print(tensor.extract(a, [0, 0]))
    let c = cumsum(m)
    print(tensor.extract(c, [0, 2]))
    let me = mean_axis(m, 1)
    print(tensor.extract(me, [0]))
}}
"#,
        mat.display()
    )) else {
        eprintln!("skipping: the LLVM tool chain is not available");
        return;
    };
    assert_eq!(
        out.lines().collect::<Vec<_>>(),
        vec![
            "6.000000", "24.000000", "9.000000", "1", "1.000000", "4.000000", "1", "6.000000",
            "2.000000"
        ]
    );
}

#[test]
fn loads_rows_of_a_raw_file_at_runtime() {
    // A raw tensor (magic, rank, dims, values) with 4 rows of 5; the batch
    // reads the last two of them.
    let dir = tmp_dir();
    let batch = dir.join("batch.bin");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"MLRRWT01");
    bytes.extend_from_slice(&2i64.to_le_bytes());
    bytes.extend_from_slice(&4i64.to_le_bytes());
    bytes.extend_from_slice(&5i64.to_le_bytes());
    for v in 1..=20 {
        bytes.extend_from_slice(&(v as f32).to_le_bytes());
    }
    std::fs::write(&batch, &bytes).unwrap();
    let Some(out) = run(&format!(
        r#"
fn main() {{
    let b = mlr.load_batch<Tensor<2x5xf32>>("{}", 2)
    print(tensor.extract(b, [0, 0]))
    print(tensor.extract(b, [1, 4]))
    let r = mlr.load_runtime<Tensor<4x5xf32>>("{}")
    print(tensor.extract(r, [0, 0]))
}}
"#,
        batch.display(),
        batch.display()
    )) else {
        eprintln!("skipping: the LLVM tool chain is not available");
        return;
    };
    assert_eq!(out.lines().collect::<Vec<_>>(), vec!["11.000000", "20.000000", "1.000000"]);
}

#[test]
fn uses_the_random_functions_of_the_library() {
    let Some(out) = run(
        r#"
fn main() {
    let r1 = rand_uniform<Tensor<3xf32>>(7, 0.0, 1.0)
    let r2 = rand_uniform<Tensor<3xf32>>(7, 0.0, 1.0)
    let d = map(r1, r2) |p, q| {
        yield p - q
    }
    print(sum(d))
    let g = rand_normal<Tensor<3xf32>>(3, 0.0, 1.0)
    print(tmax(g) > tmin(g))
    let perm = rand_permutation(6, 11)
    print(tensor.extract(perm, [0]) != tensor.extract(perm, [1]))
}
"#,
    ) else {
        eprintln!("skipping: the LLVM tool chain is not available");
        return;
    };
    let lines: Vec<&str> = out.lines().collect();
    // The same seed gives the same values: the difference is zero.
    assert_eq!(lines[0], "0.000000", "{}", out);
    assert_eq!(lines[1], "true", "{}", out);
    assert_eq!(lines[2], "true", "{}", out);
}

#[test]
fn loads_a_tensor_at_runtime() {
    // A raw tensor file (the format of `tools/to_raw.py`), loaded when the
    // program runs; the same data loaded at compile time must agree, and
    // the runtime tensor must be writable.
    let Some(out) = run(
        r#"
fn main() {
    let w = mlr.load_runtime<Tensor<784x32xf32>>("examples/train/aw1_rt.bin")
    let c = mlr.load<Tensor<784x32xf32>>("examples/train/aw1.bin")
    let d = map(w, c) |a, b| {
        yield a - b
    }
    print(sum(d))
    let w2 = map(w) |a| {
        yield a + 1.0
    }
    print(tensor.extract(w2, [0, 0]) - tensor.extract(w, [0, 0]))
}
"#,
    ) else {
        eprintln!("skipping: the LLVM tool chain is not available");
        return;
    };
    let lines: Vec<f32> = out.lines().map(|l| l.parse().unwrap()).collect();
    // The runtime load and the compile-time load agree.
    assert_eq!(lines[0], 0.0, "{}", out);
    // The runtime tensor is writable.
    assert_eq!(lines[1], 1.0, "{}", out);
}

#[test]
fn reads_a_tensor_from_the_shell() {
    let Some(out) = run(
        r#"
fn main() {
    let w = exec_tensor<Tensor<2x3xf32>>("python3 -c 'print(1, 2, 3); print(4, 5, 6)'")
    print(tensor.extract(w, [1, 2]))
}
"#,
    ) else {
        eprintln!("skipping: the LLVM tool chain is not available");
        return;
    };
    assert_eq!(out.trim(), "6.000000", "{}", out);
}

#[test]
fn runs_the_mini_batch_training() {
    let Some(out) = run_script("scripts/run.sh", "examples/train/train_mnist_full.mlr") else {
        eprintln!("skipping: the LLVM tool chain is not available");
        return;
    };
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 6, "{}", out);
    let first: f32 = lines[0].parse().unwrap();
    assert!((first - 2.30).abs() < 0.01, "{}", out);
    assert_eq!(lines[5], "1.000000", "{}", out);
}

#[test]
fn differentiates_a_function() {
    let mlir = compile(
        r#"
fn loss(w: Tensor<3x4xf32>, b: Tensor<4xf32>, x: Tensor<2x3xf32>, y: Tensor<2xi32>) -> f32 {
    let logits = add_bias(matmul(x, w), b)
    yield cross_entropy(logits, y)
}

fn main() {
    let w = tensor.zeros<Tensor<3x4xf32>>()
    let b = tensor.zeros<Tensor<4xf32>>()
    let x = tensor.zeros<Tensor<2x3xf32>>()
    let y = tensor.zeros<Tensor<2xi32>>()
    let (l, dw, db, _, _) = value_and_grad(loss, w, b, x, y)
    print(l)
    print(tensor.extract(dw, [0, 0]))
}
"#,
    );
    assert!(mlir.contains("func.call @mlr_cross_entropy"), "{}", mlir);
    assert!(mlir.contains("mlr_matmul_grad"), "{}", mlir);
    assert!(mlir.contains("mlr_add_bias_grad"), "{}", mlir);
    assert!(mlir.contains("mlr_cross_entropy_grad"), "{}", mlir);
    // The zero gradient of the labels, lowered to a memset call.
    assert!(mlir.contains("mlr_zero"), "{}", mlir);
    verify_mlir(&mlir);
}

#[test]
fn rejects_an_undifferentiable_function() {
    let err = compile_err(
        r#"
fn loss(w: Tensor<3x4xf32>, x: Tensor<2x3xf32>, y: Tensor<2xi32>) -> f32 {
    let logits = matmul(x, w)
    let picked = map(logits) |v| {
        yield v * v
    }
    yield tensor.extract(picked, [0, 0])
}

fn main() {
    let w = tensor.zeros<Tensor<3x4xf32>>()
    let x = tensor.zeros<Tensor<2x3xf32>>()
    let y = tensor.zeros<Tensor<2xi32>>()
    let (l, dw, _, _) = value_and_grad(loss, w, x, y)
    print(l)
}
"#,
    );
    assert!(
        err.contains("cannot be differentiated"),
        "{}",
        err
    );
}

#[test]
fn runs_the_training_of_the_cnn() {
    let Some(out) = run_script("scripts/run.sh", "examples/train/train_cnn.mlr") else {
        eprintln!("skipping: the LLVM tool chain is not available");
        return;
    };
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 7, "{}", out);
    let first: f32 = lines[0].parse().unwrap();
    assert!((first - 2.27).abs() < 0.05, "{}", out);
    // Every image is classified correctly after the training.
    assert_eq!(lines[6], "100", "{}", out);
}

#[test]
fn runs_the_training_of_the_mlp() {
    let Some(out) = run_script("scripts/run.sh", "examples/train/train_mlp.mlr") else {
        eprintln!("skipping: the LLVM tool chain is not available");
        return;
    };
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 7, "{}", out);
    // The loss starts at the cross entropy of a random model and falls.
    let first: f32 = lines[0].parse().unwrap();
    let last: f32 = lines[5].parse().unwrap();
    assert!((first - 2.30).abs() < 0.01, "{}", out);
    assert!(last < 0.05, "{}", out);
    // Every image is classified correctly after the training.
    assert_eq!(lines[6], "100", "{}", out);
}

#[test]
fn runs_the_mnist_example_on_the_cpu() {
    let Some(out) = run_script("scripts/run.sh", "examples/mnist/mnist.mlr") else {
        eprintln!("skipping: the LLVM tool chain is not available");
        return;
    };
    assert_eq!(out.lines().collect::<Vec<_>>(), vec!["100", "1.000000"]);
}

#[test]
fn runs_the_mnist_example_on_the_gpu() {
    if !cfg!(target_os = "macos") {
        eprintln!("skipping: the Metal Performance Shaders are macOS only");
        return;
    }
    let Some(out) = run_script("scripts/run_mps.sh", "examples/mnist/mnist.mlr") else {
        eprintln!("skipping: the Metal Performance Shaders are not available");
        return;
    };
    assert_eq!(out.lines().collect::<Vec<_>>(), vec!["100", "1.000000"]);
}
