# 設計提案：MLR 的改良路線（v0.3）

> 狀態：**步驟 1–3 已完成**（v0.2）；§4 執行期載入與 §5 平行化待做。
>
> 完成記錄：
> * §1 NaN bug——確認已在 bufferize 的 const/params 排除中修復，補了兩個
>   回歸測試；`train_adam.mlr` 恢復真 bias correction。
> * §2 `let (a, b) = for ...`——`for` 的 iter 值全部可解包（匿名 struct
>   機制，與 `value_and_grad` 同）；`train_adam.mlr`/`train_mnist_full.mlr`
>   改寫為解包版，struct 樣板消失。
> * §3 verifier——運算元型別一致性、`func.call` 與宣告一致、回傳與簽名
>   一致；三個負向測試（`tests/ir.rs`，新增 lib target 供測試直達 IR）。
>
> 來源：標準庫與訓練（M1–M3）實作過程中發現的問題與缺口，每一項都附
> 「為什麼值得」與「怎麼做」。

---

## 1. 修掉已知的 NaN bug（正確性）

### 問題

`for` 同時攜帶 struct 與純量、body 用 `with` 更新 struct 時，數值會壞
（NaN；見 `train.md` 的已知問題）。`examples/train/train_adam.mlr` 現在
繞過它（狀態全放 struct），但根因還在。

### 根因

`build_for` 把每個 iter binding 攤平成 leaves（struct）或單值（純量），
`build_block` 的 tail 再攤平一次。當 with 的未更新欄位沿用 iter arg 的
buffer、而更新的欄位是新的 buffer 時，bufferize 的 scf.for 分支對
「混合新舊 buffer 的 yield」對映錯位——某個 leaf 的結果接到別的 leaf 的
buffer 上。

### 做法

1. 寫一個最小重現（struct + 純量 iter + with，列印每步的欄位值）。
2. 在 `build_for`/`build_block` 的 leaf 對映加斷言：leaf 的型別與順序
   必須與 iter 宣告一致——先讓錯誤變成編譯期錯誤而非 NaN。
3. 修正對映；斷言移除或保留為 verifier 規則。
4. `train_adam.mlr` 改回真正的 bias correction（`c1`/`c2` 純量 iter），
   與 PyTorch 的 Adam 對照 loss 曲線。

### 驗收

重現程式輸出正確；`cargo test` 加 2 個測試（重現 + Adam bias
correction）；全部訓練範例數值不變。

---

## 2. `for` 回傳所有 iter 值（人體工學）

### 問題

`for` 的值是**第一個** iter binding（`doc/mlr.md` 的 Loops 一節）。
每個訓練迴圈都被迫把狀態打包成 struct（`train_adam.mlr`、
`train_mnist_full.mlr`、`train_gpt.mlr` 都是），而且想拿「最終權重 +
最後 loss」兩樣東西時拿不到。

### 做法（兩段，可只做第一段）

**第一段（小）**：`let (a, b, ...) = for ...`——`LetMulti` 的解包已存在，
讓 `for` 表達式在解包語境下回傳**全部** iter 終值：

```
let (w, b, last) = for step in 0..300 step 1 iter(w = w0, b = b0, last = 0.0) {
    ...
    yield w1, b1, l
}
```

實作：checker 在 `LetMulti` 旁記錄「此 for 需要多值」；build 的
`build_for` 已經有 `result_leaves`（每個 iter binding 的 leaves），
回傳一個匿名 struct 即可——與 `value_and_grad` 同機制，零新 IR。

**第二段（可選）**：`for` 的值改成「匿名 struct 永遠包含全部 iter
值」，單 iter 時自動解包——語意更一致，但會動到所有現有迴圈，放最後。

### 驗收

`train_adam.mlr` 與 `train_mnist_full.mlr` 改寫成解包版（struct 消失或
僅剩語意需要的）；輸出與現版逐 byte 相同。

---

## 3. 加強 IR verifier（正確性）

### 問題

`ir::verify` 只查 terminator、SSA 定義與基本形狀。這次多個 bug
（`row` 的 i32/f32 配錯、`func.call` 多回傳列印、ABI 順序）都是
mlir-opt 才抓到——在 CI 沒有 mlir-opt 的機器上就漏掉了。

### 做法

在 `ir::verify` 加三類檢查：

1. **運算元的型別一致**：同一 value 的所有 uses 與其 def 同型別
   （這次 `%2` 同時是 i32 與 memref 的 bug 會被直接攔下）。
2. **`func.call` 與宣告一致**：callee 的參數/回傳數量與型別。
3. **`func.return` 與函式簽名一致**（數量與型別）。

### 驗收

三個負向單元測試（手工構造壞 IR，呼叫 `ir::verify` 期待 Err）；
`cargo test` 全綠。

---

## 4. 執行期張量載入（語言能力，大模型場景的關鍵）

### 問題

`mlr.load` 是**編譯期**嵌入：載 GPT-2（548 MB）會讓 MLIR 的
`memref.global` 有 5.5 億個位元組的 dense attribute——列印、驗證、
編譯全會爆炸。現在唯一的方法是 `exec_tensor`（經過文字 stdout，
慢且只適合小矩陣）。

### 語法

```
// 型別參數給形狀；路徑仍是字面量；回傳「可寫」的張量。
let w = mlr.load_runtime<Tensor<50257x768xf32>>("model.bin")
let w = mlr.load_runtime<Tensor<50257x768xf32>>("hf://openai-community/gpt2/model.safetensors", "wte.weight")
```

### 語意

* 編譯期只嵌入**路徑**（字串 global，機制同 `exec`）；執行期才開檔。
* 回傳的張量**可寫**（`memref.alloc` + 讀檔填充）——訓練的初始權重
  直接用，`save_pt` 存回。
* 支援與 `mlr.load` 相同的三種格式：raw、`.pt`、`.safetensors`、
  `hf://`（`pt.rs` 的讀取器搬到 runtime 可用的形式——**這是主要工作**：
  `pt.rs` 是 Rust，runtime 是 C。兩個選項：
  (a) 用 C 重寫一個最小的 reader（raw + safetensors，約 150 行；
      `.pt` 的 pickle 丟給 `exec` 呼叫 Python 轉成 raw），
  (b) 編譯器在編譯期把模型**轉成 raw `.bin`**（含 shape 檔），
      runtime 只讀 raw——簡單很多，且轉換一次之後複用。
  **建議 (b)**：`mlr.load_runtime` 只支援 raw（+ shape 內嵌在檔頭），
  其他格式由 `mlr.load`（編譯期）或 `exec` 前處理。）

### 檔案格式（建議）

```
[8 bytes: magic "MLRRWT01"]
[8 bytes: rank]
[rank * 8 bytes: dims]
[元素 * 4 bytes: little endian f32 或 i32]
```

`mlr.load_runtime` 依型別參數檢查 rank 與 dims。

### 實作

| 階段 | 工作 |
| ---- | ---- |
| 檢查 | `mlr.load_runtime` 規則（型別參數必須靜態、f32/i32） |
| Stage 1 | `mlr.load_runtime` op：路徑與（可選的）key 為 attrs，結果為張量 |
| Stage 4 | bufferize：結果是**可寫 alloc**（不是 get_global） |
| Stage 5 | `func.call @mlr_load_runtime(path_ptr, out_ptr, count)`；路徑字串 global |
| Runtime | 開檔、驗 magic/dims、讀入；hf:// 用 `exec` 先下載（或讀快取） |
| 測試 | 載入 → 訓練 → `save_pt`；與 `mlr.load` 的數值一致 |

### 驗收

`examples/gpt2/gpt2.mlr`：`hf://` 下載（或預先轉檔）→
`load_runtime` → 推論一個 token 的 logits——編譯秒級、binary 幾 KB。

---

## 5. 平行化（效能）

### 問題

CPU runtime 全單執行緒；MPS 只接了前向的 matmul/conv2d。訓練的
反向（GEMM）與所有元素級運算都在單執行緒的 C 迴圈裡。

### 做法（分三步，各自獨立可交）

1. **編譯旗標**：`scripts/run.sh` 加 `-O2 -ffast-math`（數值容許）與
   `-march=native`——零風險，先量一次 baseline。
2. **OpenMP**：給重的 kernel（`mlr_matmul`、`mlr_matmul_grad`、
   `mlr_conv2d*`）加 `#pragma omp parallel for`；run.sh 加
   `-Xclang -fopenmp`（Apple clang 內建）。元素級與縮減不動（成本低）。
3. **MPS 反向**：`mlr_matmul_grad` 的兩個 GEMM 換成
   `MPSMatrixMultiplication`（`mlr_runtime_mps.mm` 的 rename 機制已支援，
   與前向同一套）。

### 驗收

`examples/train/train_mnist_full.mlr` 的 200 步耗時（`now_ms()` 量）
基線 vs 各步；數值與單執行緒版一致（ 相對誤差 < 1e-5，fast-math 除外
——先用基準測試確認誤差再決定要不要 ffast-math）。

---

## 6. `runtime_ops.rs` 表驅動重構（工程衛生）

### 問題

17+ 個 `lower_*` 函式 90% 是同一個樣板：alloc 結果、取運算元指標、
常數、`func.call`。加一個 op 要寫 40 行重複程式。

### 做法

宣告表取代手寫函式：

```rust
struct LowerSpec {
    name: &'static str,          // domain op
    kernel: &'static str,        // runtime 函式
    // 每個參數：Tensor（取指標）、Scalar（直接）、Count（元素數）、
    // Alloc（配置結果）、Attr(n)（屬性 p0…）
    params: &'static [Param],
}
```

`lower_domain_op` 依表產生 IR。RUNTIME 宣告表與 `LowerSpec` 合併
（型別順序本來就要一致——這次多個 bug 正是兩張表不同步）。

### 驗收

行數減半（量測）；`cargo test` 與 `run_all` 全綠；之後加 op 只寫表。

---

## 7. 其餘記錄（不排程）

| 項目 | 原因 / 時機 |
| ---- | ---- |
| 字串型別 | 大工程；`assert` 訊息 + `exec` 字面量已覆蓋，等真實需求 |
| 多檔編譯 / `use` | 程式還不大；等 examples 開始互相複製貼上時做 |
| 函式泛型 | struct 已覆蓋大部分；等 `adam_step` 類邏輯真的重複到痛 |
| `tmax` 一名兩義 | 改名 `tmax_value` 會破壞現有範例；記錄為 v1.0 前的清理 |
| CI | 值得，但本地 `scripts/test.sh` + `run_all.sh` 已可用 |
| 錯誤訊息（span 標註、多錯誤累積） | checker 已有 span；等使用者變多 |

---

## 8. 建議的執行順序

| 步 | 內容 | 預估 |
| -- | ---- | ---- |
| 1 | §1 NaN bug + §3 verifier | **完成** |
| 2 | §2 for 解包（`train_adam`/`train_mnist_full` 改寫為解包版） | **完成** |
| 3 | §4 執行期載入（方案 b：raw + 檔頭） | 一天 |
| 4 | §5 平行化（旗標 → OpenMP → MPS 反向） | 半天起 |
| 5 | §6 表驅動重構 | 半天 |
