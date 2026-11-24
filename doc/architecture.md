# MLR — architecture

MLR is a language whose *semantics are isomorphic to MLIR's IR model*. The
compiler is written in Rust and has no external dependencies; it produces MLIR
text that can be verified with `mlir-opt` and lowered to LLVM with the
standard MLIR passes.

```
MLR source
   │
   ▼  lexer.rs / parser.rs
syntax tree (regions, yields, types with shapes)
   │
   ▼  check.rs
typed tree (types unified, shapes checked)
   │
   ▼  build.rs                     ── Stage 1
high level IR: func / scf / arith / tensor / mlr.*
   │
   ▼  passes.rs (canonicalize)     ── Stage 2
   │
   ▼  passes.rs (lower_domain)     ── Stage 3
   │                                  mlr.map    → linalg.generic
   │                                  mlr.reduce → scf.for
   ▼  passes/bufferize.rs           ── Stage 4
memref based IR: memref.alloc / memref.load / memref.store + scf.for
   │
   ▼  passes/runtime_ops.rs         ── Stage 5
calls into the runtime library: func.call @mlr_matmul, @mlr_conv2d, ...
   │
   ▼  ir.rs (printer)
MLIR text
```

## The IR (`src/ir.rs`)

The compiler does not use MLIR as a library; it has its own IR that mirrors
MLIR's model:

| IR concept        | MLIR counterpart                          |
| ----------------- | ----------------------------------------- |
| `Value`           | `mlir::Value` (SSA value with a type)     |
| `Op`              | `mlir::Operation`                         |
| `Region` / `Block`| `mlir::Region` / `mlir::Block`            |
| `Type`            | `mlir::Type`                              |
| `Attr`            | `mlir::Attribute`                         |
| `Loc`             | `mlir::Location`                          |

Because the IR has the same shape as MLIR's, lowering is a *structural
mapping* instead of a reconstruction: the region of a source `if` becomes the
region of an `scf.if`, the `yield` becomes the region's terminator, and the
loop-carried values become the block arguments of `scf.for`.

The IR comes with

* a **verifier** (`ir::verify`) that checks that terminators are present, that
  every operand is defined before use and that no SSA value is defined twice,
* a **printer** (`ir::print_module`) that emits MLIR's textual format,
* a **renumbering** step (`ir::renumber`) that makes the printed names dense
  and readable (`%0`, `%1`, ...).

## Stage 1 — building the IR (`src/build.rs`)

| MLR                     | IR operation                              |
| ----------------------- | ----------------------------------------- |
| `fn`                    | `func.func` with a single region          |
| `if`                    | `scf.if` with a then and an else region   |
| `for ... iter(..)`      | `scf.for` with `iter_args`                |
| `map`                   | `mlr.map` (generic form) with a region    |
| `reduce`                | `mlr.reduce` (generic form) with a region |
| `let`                   | an SSA value                              |
| `+ - * / %`             | `arith.addi` / `addf` / ...               |
| `== != < <= > >=`       | `arith.cmpi` / `arith.cmpf`               |
| `&& \|\| !`             | `arith.andi` / `ori` / `xori`             |
| `print`                 | `func.call @mlr_print_*`                  |
| `tensor.empty`          | `tensor.empty`                            |
| `tensor.extract`        | `tensor.extract`                          |
| `matmul` and friends    | `mlr.matmul` (generic form)               |
| `mlr.load<T>(path)`     | `arith.constant` with a dense attribute   |
| `mlr.save_pt(...)`      | `mlr.save_pt` (generic form)              |

Every operation inherits the source location of the statement it belongs to,
so the generated MLIR can be mapped back to the source.

## Stage 2 — canonicalization (`src/passes.rs`)

* constant folding of `arith` operations whose operands are constants,
* dead code elimination of pure operations whose results are unused.

## Stage 3 — domain lowering (`src/passes.rs`)

The `mlr.*` operations are the language's *domain* level: they keep the
iteration space and the shape of the computation intact instead of expanding
it into scalar loops.

* `mlr.map` becomes a `linalg.generic` with an identity affine map and one
  `parallel` iterator per dimension. The body of the region is preserved
  verbatim, only the terminator changes from `mlr.yield` to `linalg.yield`.
* `mlr.reduce` becomes an `scf.for` that accumulates over the reduced
  dimension. The element is read with `tensor.extract` and the accumulator is
  the loop-carried value.

## Stage 4 — bufferization (`src/passes/bufferize.rs`)

The tensor semantics of the high level IR are turned into explicit memory
operations:

| Tensor operation         | Memory operation                     |
| ------------------------ | ------------------------------------ |
| `tensor.empty`           | `memref.alloc`                       |
| `tensor.zeros`           | `memref.alloc` + a fill loop         |
| `tensor.extract`         | `memref.load`                        |
| `tensor.insert`          | `memref.store`                       |
| `linalg.generic`         | nested `scf.for` loops over memrefs  |
| tensor arguments         | memref arguments                     |

The pass is *semantics guided*: a buffer is reused **in place** when the value
that owns it is not read again after the operation (last use) and the
operation is element wise. In that case no new buffer is allocated and the
input buffer becomes the output buffer.

## Stage 5 — the runtime interface (`src/passes/runtime_ops.rs`)

The domain operations (`mlr.matmul`, `mlr.conv2d`, `mlr.maxpool2d`,
`mlr.relu`, `mlr.flatten`, `mlr.row`, `mlr.add_bias` and `mlr.argmax`) are
lowered to calls into the runtime library:

1. the operands are buffers, their addresses are taken with
   `memref.extract_aligned_pointer_as_index`,
2. the shape of the operation becomes a few `index` constants,
3. a `func.call` to `mlr_matmul`, `mlr_conv2d`, ... is emitted and the
   matching `func.func private` declaration is added to the module.

The runtime interface is deliberately tiny: every operation receives raw
pointers and the sizes of the tensors. That is what makes it possible to
implement it twice — once with plain loops on the CPU
(`runtime/mlr_runtime.c`) and once with the Metal Performance Shaders on the
GPU (`runtime/mlr_runtime_mps.mm`) — without touching the compiler or the
program.

The GPU runtime computes the matrix multiplication with
`MPSMatrixMultiplication` and turns the convolution into an im2col
transformation followed by a matrix multiplication. The remaining operations
stay on the CPU.

## Effects

Every function is `pure` unless its body performs an effect (`print`). The
contract is recorded on the MLIR function:

```mlir
func.func @main() attributes {effects = "io"} { ... }
```

This is the hook for MLIR's `MemoryEffectOpInterface`: a consumer can query
the contract instead of guessing.

## Tooling

```
mlr program.mlr -o program.mlir     # compile (all stages)
mlr program.mlr --stage 3           # stop after the domain lowering
mlr program.mlr --check             # only parse and type check
mlr program.mlr --ast               # dump the typed tree
mlr program.mlr --verify            # verify the result with mlir-opt
MLR_DUMP_IR=1 mlr program.mlr       # dump the IR of every function
scripts/run.sh program.mlr          # compile, lower to LLVM and run
scripts/run_mps.sh program.mlr      # the same, but with the GPU runtime
```

## Testing

* unit tests next to every module (lexer, parser, type checker, IR),
* integration tests (`tests/compile.rs`) that compile snippets, check the
  generated MLIR, verify it with `mlir-opt` and run the programs end to end,
* an end to end test of the MNIST example on the CPU and on the GPU.
