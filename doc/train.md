# 設計提案：MLR 的訓練（反向傳遞）

> 狀態：**M1–M3 已實作**（v0.2）。與本文件的差異：`value_and_grad` 的多值
> 回傳用匿名的 struct 打包（`let (a, b) = ...` 解包）；梯度函式在
> build 階段（stage 1 之後）產生，不是 stage 3.5 的 pass；廣播
> （`w - 0.01 * dw`）與 `let` 解包是隨訓練一起加入的語言功能。
>
> 目標：讓 MLR 自己訓練模型——現在一條鏈是
> 「Python 訓練 → 匯出 `.bin`/`.pt` → `mlr.load` → MLR 推理」；
> 做完這份設計後變成「MLR 載入初始權重 → 訓練迴圈 → `mlr.save_pt` 存回」。
>
> 已具備的基礎：`for ... iter` 迴圈攜帶值（權重更新就是它）、
> `mlr.save_pt`（存回權重）、staged lowering（AD 可以做成一個 pass）、
> CPU/MPS 雙 runtime 的同介面呼叫。

---

## 1. 範圍

### 做（M1，第一個里程碑）

| 功能 | 說明 |
| ---- | ---- |
| `value_and_grad(f, args...)` | 對函式 `f` 做反向傳遞：回傳 `(loss, d arg_1, d arg_2, ...)`，每個**張量參數**一個梯度；純量參數沒有梯度。**已實作** |
| AD pass | 編譯期對 stage 1 IR 做 reverse-mode 自動微分（source transformation，不是執行期 tape）。**已實作**：`src/passes/autodiff.rs` |
| 新 domain op | `cross_entropy(logits, labels) -> f32`（平均、數值穩定）、`softmax(x)`（推論用）。**已實作** |
| 梯度 runtime | `mlr_matmul_grad`、`mlr_add_bias_grad`、`mlr_relu_grad`、`mlr_cross_entropy_grad`、`mlr_gelu_grad`（CPU；MPS 的反向也走同一批 C 函式）。**已實作** |
| SGD | 語言層直接寫：`w - lr * dw`，不需要新語法 |
| 端到端 | MNIST MLP 在 MLR 內訓練：**已實作**——`examples/train/train_mlp.mlr`，loss 2.30 → 0.0013，訓練後 100/100；梯度與 numpy/PyTorch 逐值一致（`examples/train/grad_check.mlr` vs `tools/check_grad.py`） |

### 之後的里程碑

| 里程碑 | 內容 |
| ------ | ---- |
| M2 | `conv2d`、`maxpool2d`、`embed`/`embed_seq` 的梯度（CNN 與語言模型的訓練）。**已實作**，另含 `sum`、`flatten` 的梯度——`examples/train/train_cnn.mlr` 在 MLR 內把 MNIST CNN 訓練到 100/100 |
| M3 | `layernorm`、`gelu`、`transpose`、`softmax` 的梯度 → GPT 微調；Adam（狀態用 iter 攜帶）；MPS 梯度核心。**已實作**：`examples/train/train_gpt.mlr` 在 MLR 內微調 GPT 的語言模型頭（loss 5.5 → 0.014）；`examples/train/train_adam.mlr` 用語言本身寫的 Adam 訓練 softmax 回歸（2.29 → 0.002）。梯度核心由 CPU/MPS 共用的 C runtime 提供（matmul 的 MPS 版只加速前向，反向走同一批 C 函式） |

### 明確不做

| 排除 | 原因 |
| ---- | ---- |
| 執行期 tape / 動態圖 | IR 是靜態形狀的，靜態 source transformation 更符合 staged lowering 的設計 |
| `sample`、`argmax` 的梯度 | 離散、不可微 |
| 浮點以外的梯度 | 只有 f32 |
| 多 loss、每參數關閉梯度（requires_grad） | v1 對**所有**張量參數求梯度，呼叫端忽略不要的即可；之後有需求再加 argnums |
| 資料載入／shuffle | v1 用 `mlr.load` 一次載入、固定順序切片；shuffle 之後用現成的隨機數 |

---

## 2. 語法

```mlr
// 要微分的函式：多個張量參數、回傳一個純量 loss。
fn loss(w: Tensor<784x128xf32>, b: Tensor<128xf32>,
        x: Tensor<32x784xf32>, y: Tensor<32xi32>) -> f32 {
    let logits = add_bias(matmul(x, w), b)
    yield cross_entropy(logits, y)
}

fn main() {
    let w0 = mlr.load<Tensor<784x128xf32>>("mlp.pt", "w")
    let b0 = mlr.load<Tensor<128xf32>>("mlp.pt", "b")
    let images = mlr.load<Tensor<100x784xf32>>("examples/mlp/images.bin")
    let labels = mlr.load<Tensor<100xi32>>("examples/mlp/labels.bin")

    // 訓練迴圈：權重是 loop-carried 值，每一步產生新的。
    let (w, b) = for step in 0..40 step 1 iter(w = w0, b = b0) {
        let x = ...   // 一個 batch（v1: 固定切片）
        let y = ...
        // loss 與每個張量參數的梯度，一次前向＋一次反向。
        let (l, dw, db) = value_and_grad(loss, w, b, x, y)
        if step % 10 == 0 {
            print(l)
        }
        yield w - 0.01 * dw, b - 0.01 * db
    }
    mlr.save_pt("trained.pt", w, "w", b, "b")
}
```

### 隨訓練加入的語言功能（已實作）

* `let (a, b, c) = expr`：一次解包多個值；`_` 跳過。
* 張量與純量（或同形狀張量）的 `+ - * /` 直接寫，編譯成元素級 `map`：
  `w - 0.01 * dw`。
* 權重用 struct 打包放進 `iter` 迴圈攜帶（`for` 的值是第一個 iter 值，
  所以把狀態放第一位）。

### 語法決定點（請確認）

1. **`value_and_grad(f, args...)`**：`f` 必須是**函式名稱**（不是值；MLR 的
   函式不是一等公民）。它不是普通函式：checker 特殊處理，回傳型別為
   `(f32, f 的每個張量參數的梯度...)`，梯度順序與 `f` 的張量參數順序一致。
   純量參數（如 `y: Tensor<32xi32>` 的標籤？不，標籤是張量但整數——
   **整數張量不求梯度**，只有 f32 張量有）。
2. **不放 `requires_grad`**：對整數張量（標籤）自動不求梯度，浮點張量全求。
   呼叫端用 `let (_, dw, db) = ...` 忽略不要的。
3. **優化器不做成語法**：SGD 就是 `w - lr * dw`；Adam 等有狀態的優化器用
   iter 多帶幾個狀態值。之後若樣板太吵，再考慮 `adam(w, dw, m, v, step)`。
4. **`if` 裡可以 `print`**（副作用已支援），但 loss 函式本體**必須是直線
   程式**（見 §4.4）。

---

## 3. 語意

### 3.1 梯度定義

`value_and_grad(f, a_1..a_n)` 回傳：

```
l = f(a_1..a_n)
d a_i = ∂l / ∂a_i        （只對 f32 張量參數；形狀與 a_i 相同）
```

梯度對**批次平均**（`cross_entropy` 內部對 rows 取平均），與
PyTorch 的 `loss = F.cross_entropy(logits, labels)` 預設行為一致，方便對照。

### 3.2 正確性標準

- **有限差分**：`(f(a + h·e) - f(a - h·e)) / 2h` 對照，每個 op 一個單元測試。
- **對照 PyTorch**：同一組權重、同一個 batch，`grad` 與
  `loss.backward()` 的結果在 `1e-5` 相對誤差內（f32）。

### 3.3 與不可變語意的關係

權重不是常數 global——訓練迴圈裡權重是 `iter` 攜帶值，每步 `yield` 新的。
bufferize 的 in-place 規則照舊（最後一次使用可原地寫），不需要新的記憶體語意。

---

## 4. 實作

### 4.1 AD pass 的位置

新 pass 放在 **stage 3.5**（domain lowering 之後、bufferize 之前）：

```
stage 1  typed IR（mlr.* ops，tensor 型別）
stage 2  canonicalize
stage 3  domain lowering（map -> linalg、reduce -> scf.for）
stage 3.5  autodiff（新）：對被 value_and_grad 呼叫的函式產生反向程式
stage 4  bufferize
stage 5  runtime calls
```

在 stage 1 之後也可以（對 `mlr.*` 的 adjoint 最直接），放 3.5 是因為
`map`/`reduce` 已降級成 `linalg`/`scf.for`，反向的元素級運算可以直接產生
新的 `linalg.generic`（elementwise 的 adjoint 還是 elementwise）。

### 4.2 產生什麼 IR

對每個被 `value_and_grad` 呼叫的函式 `f`，產生一個新函式
`f.value_and_grad`（名字 mangle），簽名 = `f` 的參數，回傳
`(f32, f 的每個 f32 張量參數一個張量)`：

```
前向（保留啟動值）：
    %y1 = mlr.matmul(x, w)            // 留下 %y1 給反向
    %y2 = mlr.add_bias(%y1, b)
    %l  = mlr.cross_entropy(%y2, y)
反向（從 loss 倒著走）：
    %d2 = mlr.cross_entropy_grad(%y2, y, 1.0)     // d logits
    %d1 = mlr.add_bias_grad(%d2, ...)             // -> (dx1, db)
    %dx = mlr.matmul_grad(%y1, w, %d1)            // -> (dx, dw)
    回傳 (l, 0 張量(不需的), ..., dw, db, 0 張量(x 的梯度), 0(標籤))
```

不需要的梯度（輸入 x、整數標籤）回傳**零張量**，維持「每個張量參數一個
回傳值」的固定形狀——checker 已知哪些要、哪些不要，零張量由
`tensor.zeros` 產生，DCE 若發現沒人用會自動刪掉。

### 4.3 每個 op 的 adjoint（M1）

| op | 前向 | 反向（新 runtime 函式） |
| -- | ---- | ---- |
| `matmul(x, w)` | `y = x·w` | `dx = dy·wᵀ`，`dw = xᵀ·dy` → `mlr_matmul_grad(x, w, dy) -> (dx, dw)` |
| `add_bias(m, b)` | `y = m + b` | `dm = dy`，`db = 每行總和` → `mlr_add_bias_grad(dy) -> (dm, db)` |
| `map(h) \|v\| ...`（元素級） | `y = φ(h)` | `dh = dy · φ'(h)`；對已知元素級函式（relu、gelu、sigmoid、tanh、exp、ln、sqrt）內建 adjoint，任意使用者閉包 **M1 不支援** |
| `relu(x)` | `y = max(x, 0)` | `dx = dy·(x>0)` → `mlr_relu_grad(x, dy)` |
| `cross_entropy(logits, labels)` | `l = −mean log p[label]` | `dlogits = (softmax(logits) − onehot(labels)) / rows` → `mlr_cross_entropy_grad(logits, labels, scale)` |
| `gelu(x)` | `y = ½x(1+erf(x/√2))` | `dx = dy·(Φ(x) + x·φ(x))` → `mlr_gelu_grad(x, dy)` |
| `layernorm(x, g, b)` | 見 M3 | 標準公式，需要保留 mean 與 rstd |
| `conv2d` / `maxpool2d` / `embed*` | **已實作** | conv：`mlr_conv2d_grad` 直接算相關性（scatter 形式）；pool：反向重算 argmax（與前向同一個 tie breaking）；embed：零表 + scatter-add（單執行緒） |
| `sum(t)` / `flatten(x)` | **已實作** | sum 的反向把梯度填滿每個元素；flatten 的反向是重排（拷貝），使 CNN 的訓練得以串接 |
| `layernorm` / `transpose` / `softmax` / `softmax_tri` | **已實作（M3）** | layernorm 的反向重算 μ/σ（公式：`dx = (g∘dy − mean(g∘dy) − x̂·mean(g∘dy∘x̂))/σ`）；transpose 的反向就是 transpose；softmax 的反向 `y·(dy−⟨dy,y⟩)` 用前向輸出 |
| `softmax_tri(x)`（新 op） | **已實作（M3）** | 因果遮罩的 softmax（解碼器注意力），遮罩在 kernel 裡——GPT 的訓練因此不需要 `for` 迴圈建 mask |
| 元素級 `map` | **已實作（M3）** | 兩種模式：`map(t) \|v\| gelu(v)`（非線性，kernel 梯度）與線性 body（`v * c`、`v + c`、`c − v`、`v / c`——廣播運算的形狀，反向是同樣的線性映射） |
| 函式呼叫 | **已實作（M3）** | 被微分函式裡對純前向函式的呼叫會被內聯（參數代換、值重編號），loss 可以拆成輔助函式 |

### 4.4 控制流程限制（M1）

被微分的函式必須是**直線程式**：

* 允許：`let`、domain ops、元素級 `map`、`tensor.extract`（常數索引）。
* 不允許：`for`、`if`（資料相關的控制流）、`reduce`、`tensor.insert`。
  checker 在 `value_and_grad` 檢查時報錯——理由：靜態反向沒有分支的
  累積語意；之後有需求再加（線性控制流 + 迴圈攜帶累積是標準做法）。

### 4.5 記憶體

反向需要前向的中間值（`x`、`y1`、softmax 的機率…）。它們是函式內的
tensor 值，bufferize 照常配置；小模型的記憶體完全沒問題。之後若要省，
可以做「重算換記憶體」（checkpointing），不影響語意。

### 4.6 各階段工作清單

| 工作 | 檔案 |
| ---- | ---- |
| `cross_entropy` / `softmax` 的 checker、build、runtime 前向 | `check.rs`、`build.rs`、`runtime/mlr_runtime.c` |
| `value_and_grad` 的 checker 規則（直線程式檢查）與 build（呼叫 mangle 函式） | `check.rs`、`build.rs` |
| AD pass：對 `f` 產生 `f.value_and_grad`（§4.2 的正反向） | `src/passes/autodiff.rs`（新） |
| 梯度 runtime（M1 四個函式） | `runtime/mlr_runtime.c`；MPS 之後 |
| 測試：有限差分、PyTorch 對照、端到端訓練 | `tests/`、`tools/train_check.py` |
| 文件 | `doc/mlr.md`、本文件標記已實作 |

### 4.7 不動的東西

* 語言核心（語法、型別、SSA、region/yield）——**零改動**。
* bufferize、canonicalize、runtime_ops 的既有邏輯——AD pass 產生的是
  一般的 `mlr.*` 呼叫與 `linalg.generic`，走現有降級路徑。
* MPS runtime——M1 的訓練在 CPU runtime 上跑；梯度函式之後再加 mm 版。

---

## 5. 驗收標準

1. **梯度正確**：每個 M1 op 的有限差分測試 + 與 PyTorch `backward()` 對照
   （相對誤差 < 1e-5）。
2. **端到端**：MLR 內訓練 MLP（784→128→64→10）40 步，loss 從 ~2.3 降到
   < 1.0，`mlr.save_pt` 存回後**用現有的推論程式**（`examples/mlp`）驗證
   準確率明顯高於隨機（≥ 60%）。
3. **不破壞推論**：`scripts/run_all.sh` 14 項全綠；`cargo test` 全綠。
4. 文件更新；本文件標記「已實作（M1）」。

---

## 6. 之後（不在這次做）

* `conv2d`/`maxpool2d`/`embed*` 的梯度（M2）。
* Adam、learning rate schedule、梯度裁剪（M3；裁剪是純元素級運算，語言可直接寫）。
* 資料 shuffle、mini-batch 的隨機取樣。
* 線性控制流的 AD（`for` 的反向 = 反向 `for` + 累積）。

### 已知問題

* **純量 loop-carried 值與 struct `with` 混用**：`for` 同時攜帶 struct 與
  純量、body 用 `with` 更新 struct 時，數值會壞（NaN）。訓練範例把狀態全
  放進 struct（或全用獨立 iter）即可繞過；根因在 build/bufferize 的
  leaf 對映，待修。
* Adam 的 bias correction 用常數近似（1−βᵗ ≈ 1），只影響最初幾步。
