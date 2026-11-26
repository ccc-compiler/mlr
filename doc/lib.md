# 設計提案：MLR 的標準函式庫

> 狀態：**M1 與 M2 已實作**（v0.2）；§10 的清單記錄不做的事。
>
> 目標：把 MLR 從「能跑模型」補成「能寫完整程式」的標準庫。分成七個
> 模組：**數學、張量運算、隨機、建立與形狀、指標與評估、系統（shell）、
> 執行環境**。大多數函式是「一個 C kernel + 一個 domain op」的組合，
> 走現有的 stage 5 降級路徑，CPU/MPS 自動共用。

---

## 0. 設計原則

1. **靜態形狀、f32 為主**：所有張量函式的形狀在編譯期已知；精度跟著
   runtime（f32），與現有 domain ops 一致。
2. **沒有字串值**：凡是需要字串的（命令、檔名、訊息）一律用**字面量**；
   這是語言的邊界，函式庫不繞過它。
3. **每個函式 = 一個 kernel**：不作弊在編譯期展開成 `map`（那會讓 IR
   膨脹），而是 runtime 呼叫——訓練迴圈裡的反覆呼叫才便宜。
4. **與 `value_and_grad` 的關係**：數學/張量函式只要可微就會進 AD 白名單
   （M2 起）；隨機、系統、IO 一律不可微（checker 擋）。
5. **命名**：純量函式小寫（`abs`、`clamp`）；張量操作用語意的名字
   （`reshape`、`concat`、`onehot`）；系統類前綴 `exec`。

---

## 1. 模組總覽

| 模組 | 內容 | 里程碑 |
| ---- | ---- | ---- |
| §2 數學（純量） | abs min max clamp pow floor ceil round sign is_nan is_finite pi e | **已實作** |
| §3 張量算術 | tmin tmax tclamp tpow texp tln（元素級，與 broadcast 互補） | **已實作** |
| §4 縮減 | sum(有) mean tmax→純量 tmin→純量 argmin norm1 norm2 var std dot | **已實作** |
| §5 形狀 | reshape concat rows(切片) reverse | **已實作** |
| §6 建立 | iota linspace onehot rand_uniform rand_normal rand_permutation | **已實作** |
| §7 索引與批次 | gather（shuffle = gather + rand_permutation） dropout accuracy | **已實作** |
| §8 系統 | exec exec_i exec_f exec_tensor | **已實作** |
| §9 執行環境 | now_ms exit assert | **已實作** |

實作筆記（M1 的偏差）：`assert` 的訊息以字串 global 傳給
`mlr_assert(cond, msg)`（與 exec 的命令同機制）；`exec` 的子程序輸出重導
到 stderr（`1>&2`），避免 SIGPIPE 且保持可見；`exec_i`/`exec_f` 讀取第一
行、`strtol`/`strtod` 解析、drain 之後 `pclose`。
| §10 之後 | cumsum sort argsort 資料載入器 檔案 IO | cumsum/sort/argsort 已於 v1.5 實作；沿軸縮減（sum_axis 等）同；批次讀取 = `mlr.load_batch`；檔案 IO 仍記錄不做 |

---

## 2. 數學（純量，f32/f64 跟著運算元）

現有：`exp ln sqrt tanh sigmoid gelu`。

新增（全是 runtime 一行、checker 一個名字）：

| 函式 | 說明 |
| ---- | ---- |
| `abs(x)` | 絕對值；整數也支援 |
| `min(a, b)` / `max(a, b)` | 兩數取小/大（型別經過 unify） |
| `clamp(x, lo, hi)` | 夾在區間內 |
| `pow(a, b)` | 冪（`fmin` 到 f32/f64 跟著運算元） |
| `floor(x)` `ceil(x)` `round(x)` | 取整；回傳浮點（要整數再轉） |
| `sign(x)` | −1 / 0 / 1（浮點） |
| `is_nan(x)` `is_finite(x)` | 回 bool；訓練除錯用 |
| `pi()` `e()` | 常數函式（沒有常數語法，函式最省事） |

### 語法決定點 A

`min`/`max` 是純量版；張量版叫 `tmin`/`tmax`（§3）還是張量版直接佔用
`min`、純量叫 `smin`？**建議：純量 `min`/`max`，張量 `tmin`/`tmax`**
——訓練程式裡純量版比較常寫。

---

## 3. 張量算術（元素級，形狀不變）

broadcast（`+ - * /` 與純量）已涵蓋大多數；這裡補 **兩個張量之間** 與
**非線性** 的元素級操作：

| 函式 | 說明 | 反向（AD） |
| ---- | ---- | ---- |
| `tmin(a, b)` / `tmax(a, b)` | 元素級取小/大 | 梯度給勝者（a<b 判斷） |
| `tclamp(x, lo, hi)` | lo/hi 是 f32 純量 | 區間內 = dy，否則 0 |
| `tpow(x, p)` | p 為 f32 純量 | `dy · p·x^(p−1)` |
| `texp(x)` `tln(x)` | 元素級 exp/ln（避免包 map） | 標準 |

用途：ReLU 的變體、数值穩定（logit clamp）、自訂激活。

---

## 4. 縮減（張量 → 純量）

現有：`sum`、`argmax`（rank 1）。

| 函式 | 說明 | 反向 |
| ---- | ---- | ---- |
| `mean(x)` | 平均 = sum/n | sum 的 1/n |
| `tmax(x)` / `tmin(x)` | 最大/最小值（純量） | 最大元素處 = dy |
| `argmin(x)` | 最小值索引（rank 1，i32） | 不可微 |
| `norm1(x)` / `norm2(x)` | L1/L2 範數 | L2: `x/norm2`；L1: `sign(x)` |
| `var(x)` / `std(x)` | 變異數/標準差 | 標準公式 |
| `dot(a, b)` | 兩個 rank-1 張量的內積 = `sum(a*b)` 的快捷 | 各自乘對方 |

`sum` 已存在；`mean` 是訓練 loss 的常用寫法（`sum(map(...))/n` 的快捷）。

---

## 5. 形狀操作（M1）

全部是**拷貝**語意（buffer 模型一致）。

| 函式 | 說明 | 反向 |
| ---- | ---- | ---- |
| `reshape<Tensor<...>>(x)` | 型別參數給新形狀；元素數必須相同；列主序重排（其實只是重標籤 + 拷貝） | reshape 回去 |
| `concat(a, b)` | 沿第 0 軸串接（兩者 rank 相同、其他維相同） | 各自切回 |
| `rows(t, start, count)` | 連續列切片：`Tensor<Mx...>` → `Tensor<count x ...>` | 寫回原位（scatter） |
| `reverse(x)` | 沿第 0 軸反轉 | 反轉回來 |

用途：`concat` 疊特徵、`rows` 切 batch、`reshape` 接 CNN→全連接
（搭配 `flatten`）。

---

## 6. 建立與隨機（M2）

| 函式 | 說明 |
| ---- | ---- |
| `iota(n)` | `Tensor<nxi32>` = 0..n−1 |
| `linspace(a, b, n)` | f32 均勻格點 |
| `onehot(ids, vocab)` | `Tensor<Mxi32>` → `Tensor<MxVxf32>` |
| `rand_uniform<Tensor<...>>(seed, lo, hi)` | 均勻隨機（seed 是 i32；同 seed 同結果，可重現） |
| `rand_normal<Tensor<...>>(seed, mean, std)` | 常態隨機（Box–Muller） |
| `rand_permutation(n, seed)` | `Tensor<nxi32>` 的隨機排列（Fisher–Yates） |

隨機函式的 seed 是**顯式參數**——沒有全域隨機狀態，程式可重現；
PRNG 沿用 `mlr_sample` 的 xorshift，不同的呼叫點用 `seed + 常數` 區分
（文件寫清楚這個慣例）。

### 語法決定點 B

隨機形狀用**型別參數**（`rand_uniform<Tensor<100x10xf32>>(seed, 0, 1)`，
與 `tensor.zeros<Tensor<...>>()` 一致）——確認。

---

## 7. 索引與批次（M2）

| 函式 | 說明 | 反向 |
| ---- | ---- | ---- |
| `gather(t, idx)` | `Tensor<MxNxf32>` + `Tensor<Kxi32>` → `Tensor<KxNxf32>`（列 gather；embed_seq 的泛化） | scatter-add 回表 |
| `shuffle(t, perm)` | `gather(t, rand_permutation(n, seed))` 的快捷 | gather 的反向 |
| `dropout(x, seed, rate)` | 訓練期隨機歸零 + 1/(1−rate) 縮放（rand_uniform + 比較 + 乘的 kernel 快捷） | 前向記 mask |
| `accuracy(logits, labels)` | `Tensor<MxVxf32>` + `Tensor<Mxi32>` → f32（0..1）；評估迴圈的快捷 | 不可微 |

`gather` 讓「每步抽一個 batch」變成自然語言：
`let bx = gather(images, rows_of_batch)`。

---

## 8. 系統（shell 橋接，M1；`exec_tensor` M2）

| 函式 | 說明 |
| ---- | ---- |
| `exec(cmd, args...) -> i32` | `/bin/sh -c` 執行；回退出碼；stdout 繼承 |
| `exec_i(cmd, args...) -> i32` | 擷取 stdout，解析第一個整數；失敗即終止 |
| `exec_f(cmd, args...) -> f32` | 同上，浮點 |
| `exec_tensor<Tensor<MxNxf32>>(cmd, args...)`（M2） | stdout 讀空白分隔矩陣直接成張量；上限 16 MiB |

* 命令必須是**字面量**；參數是數值，以 `{0}`、`{1}` 插入；`{{`/`}}` 逸出。
* 參數格式化：整數十進位、浮點 `%.17g`（無損）。
* 都是不純（io）；AD 白名單不含 → 被微分函式裡用會編譯錯誤。
* 實作：命令樣板走 `mlr.save_pt` 同款的字串 global；參數裝進
  `memref<Nxf64>` 傳給 runtime 的 `mlr_exec(cmd, kind, n, args_ptr)`
  （`popen` + `WEXITSTATUS`；`strtof` 解析）。
* 與 `mlr.load` 的分工：HF 權重直接 `hf://`；Python 算整批資料寫
  `.bin` 再載；算一個數用 `exec_i`/`exec_f`；算一個張量用 `exec_tensor`。

### 語法決定點 C

`exec` 失敗（退出碼非 0）時 `exec_i`/`exec_f`/`exec_tensor` **直接終止**
（回傳型別無法表達失敗；軟失敗需要 result 型別，不做）——確認。

---

## 9. 執行環境（M1）

| 函式 | 說明 |
| ---- | ---- |
| `now_ms()` | 牆鐘毫秒（i64）——訓練計時：`let t0 = now_ms() ... print(now_ms() - t0)` |
| `exit(code)` | 結束程式（i32） |
| `assert(cond, "訊息")` | cond 為假時印訊息並以退出碼 1 結束；訊息是字面量 |

`assert` 是訓練腳本的護欄（檢查載入的形狀/數值範圍）；訊息字面量讓
runtime 不需要字串型別。

---

## 10. 記錄不做（與原因）

| 不做 | 原因 |
| ---- | ---- |
| 字串型別與串接 | 語言層大工程；字面量 + assert 訊息已覆蓋需求 |
| `cumsum`/`sort`/`argsort` | scan 需要新的 IR 形狀（region 或特殊 kernel），先記錄；`argsort` 可用 Python 算 |
| 檔案 IO（寫 `.bin`/讀文字） | `mlr.save_pt`/`mlr.load` + shell 重導向已覆蓋 |
| `getenv` | 需要字串回傳 |
| 多維軸的縮減（沿指定軸） | M1 只做全縮減；沿軸縮減等 real use case 出現（reduce 語法已有 region 版可手寫） |
| 複數、f16 儲存運算 | 沒有場景 |

---

## 11. 實作形狀

每個函式 = checker 規則（一行型別檢查）+ domain op（`DomainOp` 列舉 +
`name()`）+ stage 5 lowering（指標/常數 → `func.call`）+ C kernel。
有反向的函式再加 AD adjoint（一個 `mlr.<op>_grad` op + kernel）。
隨機/系統/環境類不進白名單。

工作量估計（不含 exec 系列，見 §8 的獨立估算）：
* M1：約 20 個 kernel、每個 5–30 行 C；checker/lowering 各是表格行。
* M2：隨機 3 個 + gather/shuffle/dropout/accuracy 4 個 + exec_tensor。

## 12. 驗收

1. 每個可微函式的梯度過 PyTorch 對照（納入 `tools/check_grad.py`）。
2. `examples/lib/lib.mlr`：一個程式演示每個模組（輸出可預期、寫進測試）。
3. `examples/train/train_mnist_full.mlr`：**已實作**——`rand_permutation`
   + `rows` + `gather` 抽 mini-batch，`value_and_grad` 訓練，`accuracy`
   評估（200 步後 1.000000）。
4. `scripts/run_all.sh` **26 項全綠**（13 範例 × 2 runtime）；
   `cargo test` **22 + 41 全綠**（含每模組的測試）。

## 13. 之後

* 字串型別（讀回 shell 輸出、路徑運算）。
* ~~沿軸縮減、`cumsum`/`sort`~~（**v1.5 已實作**：`sum_axis`/`mean_axis`/
  `tmin_axis`/`tmax_axis`/`argmin_axis`/`argmax_axis`、`sort`/`argsort`/
  `cumsum`；批次讀取 = `mlr.load_batch`，shuffle 用現成的
  `rand_permutation` + `gather`）。
* 嵌入 Python 的反向方向：MLR 當 library 被 Python 呼叫（C API）。
