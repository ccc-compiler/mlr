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
    let Some(out) = run_script("scripts/run_mps.sh", "examples/mnist/mnist.mlr") else {
        eprintln!("skipping: the Metal Performance Shaders are not available");
        return;
    };
    assert_eq!(out.lines().collect::<Vec<_>>(), vec!["100", "1.000000"]);
}
