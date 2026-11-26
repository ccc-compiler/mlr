# MLR

[English](README.md) · [繁體中文](README.tw.md)

MLR 是一種高階程式語言，會編譯成 [MLIR](https://mlir.llvm.org/)。

這門語言的設計目標，是讓**語意與 MLIR 的 IR 模型「同構」（isomorphic）**：每個
複合結構都是一個以 `yield` 收尾的 region，每個綁定都是一個 SSA 值，張量形狀是
型別的一部分；編譯器以分階段的流程降低程式，並盡可能保留高階結構。

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

會變成

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

## 設計

MLR 最具辨識度的特性，是**原始語言的語意與 MLIR 的 IR 模型同構**，因此降低
（lowering）是「結構對結構」的映射，而不是重新建構：

* 每個複合結構（`if`、`for`、`map`、`reduce`）都是一個以 `yield` 收尾的
  region，會變成某個 `scf.*` 運算的 region；
* 每個綁定都是不可變的 SSA 值——沒有 `var`，也不能重新賦值；
* 張量的形狀是型別的一部分（`Tensor<64x128xf32>` ↔ `tensor<64x128xf32>`）。

值是不可變的：`tensor.insert` 會產生一個**新的**張量值。後文看到的「就地
（in-place）」寫入，其實是第 4 階段 bufferization 的最佳化——當舊值之後不再被
讀取時，就重用它的 buffer。mutation 是實作細節，不是語言特性。記憶體也是同樣
的道理：SSA 的「最後一次使用」會告訴編譯器何時可以重用或釋放 buffer，因此既
不需要手動記憶體管理，也不需要 borrow checker。

**字串**是唯一刻意的例外。語言沒有字串**型別**，只有字串**字面量**，用於檔名、
shell 命令與訊息。`print` 可同時接受字面量與純量並加以格式化：

```mlr
print("n = ", n, ", twice = ", n * 2)
```

任何需要真正字串值的需求——讀回文字、串接、在執行期組出路徑——都交給 shell
橋接（`exec*`），它只回傳數字與張量。加入字串型別意味著動態長度與堆積生命
週期，會與「靜態形狀、與 MLIR 同構」的設計衝突，因此目前刻意不支援（見
[doc/lib.md](doc/lib.md) §0 與 [doc/roadmap.md](doc/roadmap.md)）。

其他實用特性：

* 編譯器是**零依賴**的 Rust 程式，擁有自己一套鏡射 MLIR 的 IR（含 verifier
  與 printer），輸出 MLIR **文字**；
* 內建訓練：`value_and_grad` 以反向模式自動微分一個函式；
* 真實的 checkpoint（`.pt`、safetensors、`hf://`）會在編譯期嵌入
  （`mlr.load`），並在執行期寫回（`mlr.save_pt`）；
* 同一份程式可連結到 CPU runtime 或 Metal Performance Shaders runtime，兩者
  共用同一個介面。

## 建置

```
cargo build --release
```

編譯器以 Rust 撰寫，且**沒有任何外部依賴**。

## 用法

```
mlr [options] <file.mlr>

  -o, --output <file>   write the generated MLIR to <file> (default: stdout)
      --stage <1-5>     stop after a lowering stage (default: 5)
      --check           only parse and type check the program
      --ast             print the typed tree instead of MLIR
      --verify          verify the generated MLIR with `mlir-opt`
  -h, --help            show this help
```

降低階段如下

| 階段 | 結果                                                          |
| ---- | ------------------------------------------------------------- |
| 1    | 高階 IR：`func`、`scf`、`arith`、`tensor`、`mlr.*`             |
| 2    | 正規化後的 IR（常數摺疊、死碼消除）                            |
| 3    | 領域降低：`mlr.map` → `linalg.generic`、`mlr.reduce` → `scf.for` |
| 4    | bufferization：`tensor.*` → `memref.*`                         |
| 5    | runtime 降低：`mlr.*` → `func.call @mlr_*`                    |

## 執行程式

`scripts/run.sh` 會編譯 MLR 程式、把 MLIR 降低成 LLVM IR 並執行
（需要 `mlir-opt`、`mlir-translate` 與 `clang`）：

```
$ scripts/run.sh examples/scalar.mlr
3
120
45
100
```

## MNIST：一支程式，兩條執行路徑

`examples/mnist/mnist.mlr` 用一個小型卷積網路（5x5 卷積、max pooling 與兩層
dense）分類手寫數字。權重以 `tools/train_mnist.py`（純 numpy）訓練，並透過
`mlr.load` 在編譯期嵌入程式。

```
$ scripts/run.sh examples/mnist/mnist.mlr        # the CPU runtime
100
1.000000

$ scripts/run_mps.sh examples/mnist/mnist.mlr    # Metal Performance Shaders
mlr: Metal Performance Shaders on Apple M3
100
1.000000
```

兩個腳本編譯的是**同一支**程式：領域運算（`matmul`、`conv2d`……）會降低成對
runtime 函式庫的呼叫，而 runtime 函式庫有兩個介面相同的實作：

* `runtime/mlr_runtime.c` 以純迴圈在 CPU 上計算全部運算，
* `runtime/mlr_runtime_mps.mm` 以 `MPSMatrixMultiplication` 在 GPU 上計算矩陣
  乘法（前向與反向）與卷積（卷積會先轉成 im2col，再做一次矩陣乘法）。

程式本身並不知道它連結的是哪一個 runtime。

## 在 MLR 裡訓練

MLR 也能訓練：`value_and_grad(loss, w, b, x, y)` 以反向模式自動微分一個函式，
而訓練迴圈就是一個會攜帶權重的迴圈：

```
let trained = for step in 0..300 step 1 iter(state = start) {
    let (l, dw, db, _, _) = value_and_grad(loss, state.w1, state.b1, x, y)
    yield state with { w1: map(state.w1, dw) |a, g| { yield a - 0.5 * g } }
}
mlr.save_pt("trained.pt", trained.w1, "w1")
```

`scripts/run.sh examples/train/train_mlp.mlr` 會用一百張 MNIST 圖片訓練一個
MLP（loss 從 2.30 降到 0.001，之後每張圖片都分類正確）；梯度與 PyTorch 在 f32
精度內一致（`tools/check_grad.py`）。

## 真實的 checkpoint

`mlr.load` 會在編譯期把張量嵌入程式。除了原始 `.bin` 檔，它也能讀取 PyTorch
checkpoint（`.pt`）與 `safetensors` 檔，而 `hf://<repo>/<file>` 路徑會先從
Hugging Face hub 下載檔案：

```
let w = mlr.load<Tensor<50257x768xf32>>("hf://openai-community/gpt2/model.safetensors", "wte.weight")
```

執行期 `mlr.save_pt("out.pt", w, "w")` 會把張量寫回 checkpoint，讓
`torch.load(path, weights_only=True)` 讀到一個字典。詳見
[doc/mlr.md](doc/mlr.md)。

## 文件

* [doc/mlr.md](doc/mlr.md) — 語言規格
* [doc/architecture.md](doc/architecture.md) — 編譯器與其降低流程
* [doc/struct.md](doc/struct.md) — struct 型別的設計
* [doc/train.md](doc/train.md) — 訓練（反向傳播）的設計
* [doc/lib.md](doc/lib.md) — 標準函式庫的設計（數學、張量、隨機、shell 橋接）
* [doc/v1.x.md](doc/v1.x.md) — v1.x 路線圖

## 目錄結構

| 路徑                      | 內容                                        |
| ------------------------- | ------------------------------------------- |
| `src/lexer.rs`            | 分詞器                                      |
| `src/parser.rs`           | 遞迴下降剖析器（region 與 yield）           |
| `src/ast.rs`              | 語法樹                                      |
| `src/check.rs`            | 型別與形狀檢查器                            |
| `src/ir.rs`               | 鏡射 MLIR 的 IR、verifier 與 printer        |
| `src/build.rs`            | 第 1 階段：型別化語法樹 → 高階 IR            |
| `src/passes.rs`           | 第 2、3 階段：正規化、領域降低              |
| `src/passes/bufferize.rs` | 第 4 階段：bufferization                    |
| `src/passes/runtime_ops.rs` | 第 5 階段：領域運算的降低                 |
| `runtime/`                | CPU runtime 與 Metal Performance Shaders runtime |
| `tools/train_mnist.py`    | 訓練 MNIST 範例所用的網路                   |
| `examples/`               | 範例程式                                    |
| `tests/`                  | 整合測試                                    |
