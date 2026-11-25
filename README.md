# MLR

MLR is a high level programming language that compiles to
[MLIR](https://mlir.llvm.org/).

The language is built so that its semantics are *isomorphic* to MLIR's IR
model: every compound construct is a region that ends in a `yield`, every
binding is an SSA value, tensor shapes are part of the types, and the compiler
lowers the program through a staged pipeline that keeps the high level
structure intact as long as possible.

```mlr
fn sum_to(n: i32) -> i32 {
    let total = for i in 0..n step 1 iter(acc = 0) {
        yield acc + i
    }
    yield total
}

fn scale(x: Tensor<64x128xf32>, s: f32) -> Tensor<64x128xf32> {
    yield map(x) {unroll = 4} |v| {
        yield v * s
    }
}
```

becomes

```mlir
func.func @sum_to(%arg0: i32) -> i32 {
  %0 = arith.constant 0 : index
  %1 = arith.index_cast %arg0 : i32 to index
  %2 = arith.constant 1 : index
  %3 = arith.constant 0 : i32
  %4 = scf.for %5 = %0 to %1 step %2 iter_args(%6 = %3) -> (i32) {
    %7 = arith.index_cast %5 : index to i32
    %8 = arith.addi %6, %7 : i32
    scf.yield %8 : i32
  }
  func.return %4 : i32
}
```

## Build

```
cargo build --release
```

The compiler is written in Rust and has **no external dependencies**.

## Usage

```
mlr [options] <file.mlr>

  -o, --output <file>   write the generated MLIR to <file> (default: stdout)
      --stage <1-5>     stop after a lowering stage (default: 5)
      --check           only parse and type check the program
      --ast             print the typed tree instead of MLIR
      --verify          verify the generated MLIR with `mlir-opt`
  -h, --help            show this help
```

The lowering stages are

| Stage | Result                                                        |
| ----- | ------------------------------------------------------------- |
| 1     | high level IR: `func`, `scf`, `arith`, `tensor`, `mlr.*`       |
| 2     | canonicalized IR (constant folding, dead code elimination)    |
| 3     | domain lowering: `mlr.map` → `linalg.generic`, `mlr.reduce` → `scf.for` |
| 4     | bufferization: `tensor.*` → `memref.*`                          |
| 5     | runtime lowering: `mlr.*` → `func.call @mlr_*`                |

## Run a program

`scripts/run.sh` compiles an MLR program, lowers the MLIR to LLVM IR and runs
it (needs `mlir-opt`, `mlir-translate` and `clang`):

```
$ scripts/run.sh examples/scalar.mlr
3
120
45
100
```

## MNIST: one program, two execution paths

`examples/mnist/mnist.mlr` classifies handwritten digits with a small
convolutional network (a 5x5 convolution, max pooling and two dense layers).
The weights are trained with `tools/train_mnist.py` (plain numpy) and embedded
into the program at compile time with `mlr.load`.

```
$ scripts/run.sh examples/mnist/mnist.mlr        # the CPU runtime
100
1.000000

$ scripts/run_mps.sh examples/mnist/mnist.mlr    # Metal Performance Shaders
mlr: Metal Performance Shaders on Apple M3
100
1.000000
```

Both scripts compile the *same* program: the domain operations (`matmul`,
`conv2d`, ...) are lowered to calls into the runtime library, and the runtime
library has two implementations with the same interface:

* `runtime/mlr_runtime.c` computes everything with plain loops on the CPU,
* `runtime/mlr_runtime_mps.mm` computes the matrix multiplication and the
  convolution with `MPSMatrixMultiplication` on the GPU (the convolution
  becomes an im2col transformation followed by a matrix multiplication).

The program itself does not know which runtime it is linked against.

## Training inside MLR

MLR trains too: `value_and_grad(loss, w, b, x, y)` differentiates a function
with reverse mode automatic differentiation, and the training loop is a loop
that carries the weights:

```
let trained = for step in 0..300 step 1 iter(state = start) {
    let (l, dw, db, _, _) = value_and_grad(loss, state.w1, state.b1, x, y)
    yield state with { w1: map(state.w1, dw) |a, g| { yield a - 0.5 * g } }
}
mlr.save_pt("trained.pt", trained.w1, "w1")
```

`scripts/run.sh examples/train/train_mlp.mlr` trains an MLP on a hundred
MNIST images (the loss falls from 2.30 to 0.001 and every image is then
classified correctly); the gradients agree with PyTorch to f32 precision
(`tools/check_grad.py`).

## Real checkpoints

`mlr.load` embeds a tensor into the program at compile time. Besides a raw
`.bin` file it reads PyTorch checkpoints (`.pt`) and `safetensors` files, and a
`hf://<repo>/<file>` path downloads the file from the Hugging Face hub first:

```
let w = mlr.load<Tensor<50257x768xf32>>("hf://openai-community/gpt2/model.safetensors", "wte.weight")
```

At run time `mlr.save_pt("out.pt", w, "w")` writes tensors back into a
checkpoint that `torch.load(path, weights_only=True)` reads as a dictionary.
See [doc/mlr.md](doc/mlr.md) for the details.

## Documentation

* [doc/mlr.md](doc/mlr.md) — the language specification
* [doc/architecture.md](doc/architecture.md) — the compiler and its lowering
  pipeline
* [doc/struct.md](doc/struct.md) — the design of the struct types
* [doc/train.md](doc/train.md) — the design of the training (backpropagation)

## Layout

| Path                     | Contents                                    |
| ------------------------ | ------------------------------------------- |
| `src/lexer.rs`           | tokenizer                                   |
| `src/parser.rs`          | recursive descent parser (regions and yields) |
| `src/ast.rs`             | syntax tree                                 |
| `src/check.rs`           | type and shape checker                      |
| `src/ir.rs`              | MLIR shaped IR, verifier and printer         |
| `src/build.rs`           | Stage 1: typed tree → high level IR          |
| `src/passes.rs`          | Stage 2 and 3: canonicalization, domain lowering |
| `src/passes/bufferize.rs`| Stage 4: bufferization                      |
| `src/passes/runtime_ops.rs` | Stage 5: lowering of the domain operations |
| `runtime/`               | the CPU runtime and the Metal Performance Shaders runtime |
| `tools/train_mnist.py`   | trains the network of the MNIST example     |
| `examples/`              | example programs                            |
| `tests/`                 | integration tests                           |
