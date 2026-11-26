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
| a struct type         | the values of its fields |

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

## Structs

```
struct Attention {
    wq: Tensor<32x16xf32>,
    wk: Tensor<32x16xf32>,
    wv: Tensor<32x16xf32>
}

struct Block {
    ln_g: Tensor<32xf32>,
    attn: Attention,
    fc1: Tensor<32x128xf32>,
    fc2: Tensor<128x32xf32>
}

impl Block {
    fn forward(self, x: Tensor<16x32xf32>) -> Tensor<16x32xf32> {
        let h = layernorm(x, self.ln_g, self.ln_g)
        let m = self.attn.forward(h)
        ...
    }
}

fn main() {
    let block = Block { ln_g: g, attn: attn, fc1: f1, fc2: f2 }
    let y = block.forward(x)
    let other = block with { fc2: new_fc2 }
}
```

* A struct bundles values under a name; a field is a scalar, a tensor or
  another struct (but never the struct itself).
* The literal `Name { field: expr, ... }` names every field exactly once.
* `recv.field` reads a field, through any nesting depth.
* `recv with { field: expr }` is a functional update: it produces a new value
  and leaves `recv` untouched.
* A method is declared in an `impl` block; its first parameter is `self`.
  `recv.method(args)` calls it, which is the same as a function call
  `Struct.method(recv, args)`. Since values cannot be assigned to, a method
  never writes into `self`: it yields a new value instead.
* A struct may flow through `let`, function calls, `if` and loop-carried
  `iter` values; the compiler flattens it into the values of its fields at
  the boundaries, so the runtime interface does not change.

See [struct.md](struct.md) for the design notes.

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
* The value of the expression is the final value of the first loop-carried
  value; `let (a, b) = for ... iter(a = .., b = ..)` unpacks into **all** of
  them.

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
| `softmax(x)`                | `Tensor<MxNxf32>` → `Tensor<MxNxf32>`           | the softmax of every row |
| `cross_entropy(logits, y)`  | `Tensor<MxVxf32>`, `Tensor<Mxi32>` → `f32`      | the mean cross entropy of the rows |
| `sum(x)`                    | any tensor → `f32`                              | the sum of the elements |
| `add(a, b)`                 | two tensors of the same shape                   | the element wise sum |
| `softmax_tri(x)`            | `Tensor<NxNxf32>` → `Tensor<NxNxf32>`           | the softmax of the rows with a causal mask (decoder attention) |
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

`print("text", x, ...)` writes its pieces in order — a string literal as is, a
scalar formatted by its type — and ends with a newline; a lone `print(x)` still
prints one value per line. `print_byte(x)` writes one byte and `mlr.save_pt`
writes a checkpoint; these are the only impure operations. The scalar
functions are `exp`, `ln`, `sqrt`, `tanh`, `sigmoid` and `gelu`.

The scalar math of the library: `abs`, `min`, `max`, `clamp`, `pow`,
`floor`, `ceil`, `round`, `sign`, `is_nan`, `is_finite`, `pi`, `e` — besides
`exp`, `ln`, `sqrt`, `tanh`, `sigmoid` and `gelu`.

The examples end with `examples/gpt/gpt.mlr`, a causal transformer (two
layers, two heads, gelu, layer normalization) that generates text byte by
byte; its weights were trained with `tools/train_torch_models.py`.

## Runtime loading

`mlr.load` embeds the tensor into the program at compile time; for large
models that makes the compilation slow. `mlr.load_runtime` reads the file
**when the program runs** — the compiler only embeds the path, and the
result is a writable tensor:

```
let w = mlr.load_runtime<Tensor<784x32xf32>>("model_w1.bin")
```

The file carries the shape in a small header (magic `MLRRWT01`, the rank,
the dimensions, then the little endian f32/i32 values);
`tools/to_raw.py` converts a `.safetensors` or `.npy` file. A `hf://` path
is resolved at compile time (the file is downloaded into the cache once),
so the run needs no network.

## Standard library

The tensor functions of the runtime (see [lib.md](lib.md) for the full list):

| Function | Types | Meaning |
| -------- | ----- | ------- |
| `tmin(a, b)` `tmax(a, b)` | two tensors of one shape, or one tensor → scalar | element wise or reduction |
| `tclamp(x, lo, hi)` | tensor, f32, f32 | clamps every element |
| `tpow(x, p)` | tensor, f32 | element wise power |
| `texp(x)` `tln(x)` | tensor | element wise exponential/logarithm |
| `mean(x)` `var(x)` `std(x)` `norm1(x)` `norm2(x)` `dot(a, b)` | tensor(s) → f32 | reductions |
| `argmin(x)` | tensor → i32 | the index of the smallest element |
| `reshape<Tensor<...>>(x)` | tensor | the same elements in a new shape |
| `concat(a, b)` | two tensors | along the first axis |
| `rows(t, start, count)` | tensor, i32, i32 | a slice of the rows |
| `reverse(x)` | tensor | the rows in the reverse order |
| `iota(n)` | i32 → `Tensor<nxi32>` | the integers 0..n-1 |
| `linspace(a, b, n)` | f32, f32, i32 | n evenly spaced values |
| `onehot(ids, vocab)` | `Tensor<Mxi32>`, i32 | a one-hot matrix |
| `rand_uniform<Tensor<...>>(seed, lo, hi)` | i32, f32, f32 | uniform random values (reproducible: the seed is explicit) |
| `rand_normal<Tensor<...>>(seed, mean, std)` | i32, f32, f32 | normal random values (Box–Muller) |
| `rand_permutation(n, seed)` | i32, i32 → `Tensor<nxi32>` | a random permutation (Fisher–Yates) |
| `gather(t, idx)` | tensor, `Tensor<Kxi32>` | the selected rows (any element type) |
| `dropout(x, seed, rate)` | tensor, i32, f32 | training-time dropout (the mask is recomputed in the gradient) |
| `accuracy(logits, labels)` | `Tensor<MxVxf32>`, `Tensor<Mxi32>` → f32 | the share of the rows whose argmax is the label |

The system functions call the shell (the command is a string literal, the
arguments are numbers that fill the `{0}`, `{1}`… placeholders):

```
let code = exec("ls -la")                     // the exit code
let n = exec_i("wc -l < data.txt")            // the first integer of the output
let lr = exec_f("python3 -c 'print(0.001)'")  // the first float
let w = exec_tensor<Tensor<100x784xf32>>(
    "python3 -c 'import numpy as np, sys; np.savetxt(sys.stdout, np.random.rand(100, 784))'"
)
assert(n > 0, "the data set is empty")
let t0 = now_ms()
exit(1)                                       // ends the program
```

All of them are impure, and none of them can appear in a function that is
differentiated with `value_and_grad`.

## Training

```
fn loss(w: Tensor<3x4xf32>, b: Tensor<4xf32>, x: Tensor<2x3xf32>, y: Tensor<2xi32>) -> f32 {
    let logits = add_bias(matmul(x, w), b)
    yield cross_entropy(logits, y)
}

let (l, dw, db, _, _) = value_and_grad(loss, w, b, x, y)
```

`value_and_grad(f, args...)` differentiates `f` with reverse mode automatic
differentiation: it returns the loss of `f` and one gradient per **tensor**
parameter (a zero tensor for the ones that do not reach the loss, so the
number of the values is fixed; integer tensors get no gradient). The
compiler derives the backward pass from the IR of `f` — no tape, no runtime
overhead beyond the gradient calls themselves.

A training loop is a loop that carries the weights and yields new ones; a
struct bundles them:

```
let trained = for step in 0..300 step 1 iter(state = start) {
    let (l, dw, db, _, _) = value_and_grad(loss, state.w, state.b, x, y)
    if step % 100 == 0 {
        print(l)
    }
    yield state with {
        w: map(state.w, dw) |a, g| { yield a - 0.1 * g },
        b: map(state.b, db) |a, g| { yield a - 0.1 * g }
    }
}
mlr.save_pt("trained.pt", trained.w, "w", trained.b, "b")
```

A tensor and a scalar (or two tensors of the same shape) combine element
wise with `+ - * /`, which is what `w - 0.1 * dw` does. `let (a, b) = expr`
unpacks several values at once; `_` skips one.

The function that is differentiated must be a straight line program of
operations with a known gradient (`matmul`, `add_bias`, `relu`, `gelu`,
`cross_entropy`, `conv2d`, `maxpool2d`, `flatten`, `embed`, `embed_seq`,
`sum`); a `for`, `if` or `map` inside it is a compile error. The examples
train an MLP and a CNN (`examples/train`), and
`tools/check_grad.py` compares the gradients with PyTorch. See
[train.md](train.md) for the design and the next milestones.

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
