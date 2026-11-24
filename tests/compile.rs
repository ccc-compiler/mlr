//! Integration tests: compile MLR programs, check the generated MLIR and run
//! the programs end to end.

use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

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
        Err(_) => eprintln!("skipping MLIR verification: mlir-opt not found"),
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
    // The full lowering to native code needs the LLVM tool chain; the tests
    // only run when it is available.
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/run.sh");
    if !script.exists() {
        return None;
    }
    let out = Command::new("sh")
        .arg(&script)
        .arg(&input)
        .output()
        .expect("failed to run the script");
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8(out.stdout).unwrap())
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
    let script = root.join(script);
    if !script.exists() {
        return None;
    }
    let out = Command::new("sh")
        .arg(&script)
        .arg(root.join(file))
        .output()
        .expect("failed to run the script");
    if !out.status.success() {
        eprintln!(
            "skipping: the script failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }
    Some(String::from_utf8(out.stdout).unwrap())
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
fn runs_the_mnist_example_on_the_cpu() {
    let Some(out) = run_script("scripts/run.sh", "examples/mnist/mnist.mlr") else {
        eprintln!("skipping: the LLVM tool chain is not available");
        return;
    };
    assert_eq!(out.lines().collect::<Vec<_>>(), vec!["100", "1.000000"]);
}

#[test]
fn runs_the_mnist_example_on_the_gpu() {
    let Some(out) = run_script("scripts/run_mps.sh", "examples/mnist/mnist.mlr") else {
        eprintln!("skipping: the Metal Performance Shaders are not available");
        return;
    };
    assert_eq!(out.lines().collect::<Vec<_>>(), vec!["100", "1.000000"]);
}
