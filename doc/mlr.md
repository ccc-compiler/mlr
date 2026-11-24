# MLR — the language

MLR is a high level language whose semantics are *isomorphic* to MLIR's IR
model: every compound construct is a **region** that ends in a **yield**, every
binding is an **SSA value**, and every type carries the shape information that
the corresponding MLIR type needs.

```mlr
fn sum_to(n: i32) -> i32 {
    let total = for i in 0..n step 1 iter(acc = 0) {
        yield acc + i
    }
    yield total
}
```

The program above lowers to a `scf.for` whose loop-carried value is the block
argument of the loop body:

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

## Design rules

1. **Regions and yields are first class.** `if`, `for ... iter(..)`, `map` and
   `reduce` are *expressions* with a region. The region of an expression is
   exactly the region of the corresponding MLIR operation.
2. **Loop state is explicit.** The values that a loop carries are written in
   `iter(name = initial)`, which maps onto `scf.for`'s `iter_args`.
3. **Shapes live in the types.** `Tensor<64x128xf32>` becomes
   `tensor<64x128xf32>`; a dynamic dimension is written `?`.
4. **Values are immutable.** There is no `var`: a binding is an SSA value and
   cannot be reassigned. Mutation only exists where the semantics ask for it
   (tensor values, buffers).
5. **Effects are declared.** A function is `pure` unless it performs an effect
   such as `print`; the contract is recorded on the MLIR function.
6. **Locations travel with the code.** Every operation carries the source
   location of the statement it came from.

## Types

| MLR                   | MLIR                |
| --------------------- | ------------------- |
| `i1` / `bool`         | `i1`                |
| `i8` `i16` `i32` `i64`| `i8` `i16` `i32` `i64` |
| `f16` `bf16` `f32` `f64` | `f16` `bf16` `f32` `f64` |
| `index`               | `index`             |
| `Tensor<64x128xf32>`  | `tensor<64x128xf32>` |
| `Tensor<?xf32>`       | `tensor<?xf32>`     |

Integer literals default to `i32`, floating point literals to `f64`; a literal
is converted to the type of the other operand when the two meet. `index` and
fixed width integers mix freely: the `index` operand is converted. An integer
that meets a float is converted to the float (`arith.sitofp`).

Tensor dimensions are unified when they meet: a dynamic dimension `?` that is
unified with a static one becomes static, and two static dimensions must be
equal.

## Declarations

```
fn name(a: i32, b: f32) -> i32 {
    yield a
}
```

The result type may be omitted; the function then returns nothing.

## Statements and expressions

```
let x = 1                 // binding (a statement)
let y: i64 = 2            // binding with an explicit type
print(x)                  // effect statement
yield x + 1               // the value of the region
```

A block is a sequence of statements followed by a value. The value can be
written with an explicit `yield` or as the last expression of the block.

### Conditionals

```
let r = if cond {
    yield 1
} else {
    yield 2
}
```

Both branches must yield the same type. `else if` is supported.

### Loops

```
let total = for i in 0..n step 1 iter(acc = 0) {
    yield acc + i
}
```

* The induction variable `i` has the type `index`.
* `iter(name = initial)` declares the loop-carried values; the body must yield
  exactly one value per declaration.
* `step` is optional and defaults to `1`.
* The value of the expression is the final value of the loop-carried values.

### Tensor operations

```
let scaled = map(x) {unroll = 4} |v| {
    yield v * 2.0
}

let total = reduce(0.0, x) |acc, v| {
    yield acc + v
}
```

* `map` applies its body to every element of the input tensor(s); the result
  has the shape of the input. The attributes in `{...}` are scheduling hints
  and are carried into the IR.
* `reduce` folds a rank-1 tensor into a scalar, starting from the given
  accumulator.
* `tensor.empty<Tensor<4xf32>>()` creates a tensor, `tensor.zeros<...>()`
  creates a zero filled one, `tensor.extract(t, [i])` reads an element.

### Domain operations

The operations below are the *domain* level of the language: they describe a
whole tensor computation instead of a scalar loop. They are lowered to calls
into the runtime library (see stage 5), which has a CPU implementation and a
Metal Performance Shaders implementation.

| Operation                   | Types                                            | Meaning                          |
| --------------------------- | ------------------------------------------------ | -------------------------------- |
| `matmul(a, b)`              | `Tensor<MxKxf32>`, `Tensor<KxNxf32>` → `Tensor<MxNxf32>` | matrix product          |
| `matmul(v, b)`              | `Tensor<Kxf32>`, `Tensor<KxNxf32>` → `Tensor<Nxf32>` | a vector times a matrix      |
| `add_bias(m, b)`            | `Tensor<MxNxf32>`, `Tensor<Nxf32>` → `Tensor<MxNxf32>` | adds a bias to every row     |
| `conv2d(x, w, b)`           | `Tensor<HxWxf32>`, `Tensor<Ox1xKxKxf32>`, `Tensor<Ox f32>` → `Tensor<OxHxWxf32>` | convolution, padding `K/2` |
| `maxpool2d(x, k)`           | `Tensor<OxHxWxf32>` → `Tensor<Ox(H/k)x(W/k)xf32>` | max pooling with a `k x k` window |
| `relu(x)`                   | any tensor → the same shape                     | rectified linear unit           |
| `flatten(x)`                | any tensor → `Tensor<Nxf32>`                    | a flat copy of the tensor       |
| `row(t, i)`                 | `Tensor<MxNxf32>` → `Tensor<Nxf32>`             | the `i`-th row                   |
| `argmax(x)`                 | `Tensor<Nxf32>` → `i32`                         | the index of the largest element |
| `embed(t, id)`              | `Tensor<VxNxf32>`, `i32` → `Tensor<Nxf32>`      | a row of a table                 |
| `embed_seq(t, ids)`         | `Tensor<VxNxf32>`, `Tensor<Mxi32>` → `Tensor<MxNxf32>` | the rows of a table       |
| `transpose(x)`              | `Tensor<MxNxf32>` → `Tensor<NxMxf32>`           | the transpose of a matrix        |
| `layernorm(x, g, b)`        | `Tensor<MxNxf32>`, `Tensor<Nxf32>`, `Tensor<Nxf32>` → `Tensor<MxNxf32>` | normalizes every row |
| `softmax_causal(x, n)`      | `Tensor<MxNxf32>`, `i32` → `Tensor<MxNxf32>`    | the softmax of every row, columns beyond `n` masked out |
| `sample(x, t, seed)`        | `Tensor<Nxf32>`, `f32`, `i32` → `i32`           | a token drawn from the softmax of the logits at temperature `t` |

A vector counts as a matrix with a single row, and an image may be written
with or without its (single) channel dimension.

### Tensor data

```
let weights = mlr.load<Tensor<4x1x5x5xf32>>("weights.bin")
let w = mlr.load<Tensor<768x768xf32>>("model.pt", "wte.weight")
let e = mlr.load<Tensor<50257x768xf32>>("hf://openai-community/gpt2/model.safetensors", "wte.weight")
```

`mlr.load` reads a tensor and embeds it into the program at compile time. Three
kinds of files are understood:

* **A raw file** (`weights.bin`) of little endian `f32` (or `i32`) values. The
  file must contain exactly as many elements as the type has.
* **A PyTorch checkpoint** (`.pt`). The second argument names the tensor in the
  checkpoint; it may be omitted when the checkpoint holds a single tensor.
* **A `safetensors` file** (`.safetensors`), with the name of the tensor as the
  second argument.

A path that starts with `hf://<repo>/<file>` downloads the file from the
Hugging Face hub into `~/.cache/mlr/huggingface/` and then reads it, so

```
let w = mlr.load<Tensor<50257x768xf32>>("hf://openai-community/gpt2/model.safetensors", "wte.weight")
```

loads a tensor of the GPT-2 embedding matrix. The data becomes a global
constant buffer (`memref.global`), so the program needs no file access when it
runs.

### Writing a checkpoint

```
mlr.save_pt("out.pt", w, "w", b, "b")
```

`mlr.save_pt` writes tensors into a PyTorch checkpoint at run time. Every pair
of a tensor (`f32` or `i32`) and a name becomes an entry that
`torch.load(path, weights_only=True)` returns as a dictionary. The tensors are
collected until the statement ends and the `.pt` archive is then written.

### Builtins

`print(x)` prints a scalar, `print_byte(x)` writes one byte and `mlr.save_pt`
writes a checkpoint; these are the only impure operations. The scalar
functions are `exp`, `ln`, `sqrt`, `tanh`, `sigmoid` and `gelu`.

The examples end with `examples/gpt/gpt.mlr`, a causal transformer (two
layers, two heads, gelu, layer normalization) that generates text byte by
byte; its weights were trained with `tools/train_torch_models.py`.

## Lowering

| Stage | MLR                       | MLIR                                             |
| ----- | ------------------------- | ------------------------------------------------ |
| 1     | typed AST                 | `func`, `scf.if`, `scf.for`, `arith`, `tensor`, `mlr.*` |
| 2     | high level IR             | constant folding, dead code elimination          |
| 3     | `mlr.map`                 | `linalg.generic` (element wise, parallel)        |
| 3     | `mlr.reduce`              | `scf.for` accumulation                           |
| 4     | `tensor.*`                | `memref.alloc` / `memref.load` / `memref.store`  |
| 4     | `linalg.generic`          | nested `scf.for` loops over `memref`s            |
| 4     | a constant tensor         | `memref.global` + `memref.get_global`            |
| 5     | `mlr.matmul` and friends  | `func.call @mlr_matmul` and friends             |
| 5     | `mlr.save_pt`             | string globals + `func.call @mlr_save_pt_*`     |

See [architecture.md](architecture.md) for the details of every stage.

## Not implemented (yet)

* symbolic dimensions with equality constraints (the current implementation
  unifies `?` with static dimensions but does not track symbolic identities
  across a function),
* `View`/`MutView` types and their memory space attributes,
* one-shot bufferization with full alias equivalence classes (the current
  implementation reuses a buffer in place when the value is not read again),
* runtime shape guards,
* a `mirth`-style custom dialect written in ODS (the `mlr.*` operations are
  emitted in the generic form),
* GPU lowering of the *whole* program: only the domain operations run on the
  GPU (through the Metal Performance Shaders runtime), the rest of the
  program is compiled for the CPU,
* asynchronous copies and synchronisation.
