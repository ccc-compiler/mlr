# 設計：MLR 的 struct 與 impl

> 狀態：**已實作**（v0.2）。本文件與實作的差異：struct 在編譯期就攤平成
> 欄位值的打包（SOA），不經過 LLVM struct；函式邊界在 stage 1（build）
> 而非 stage 5 拆解。
>
> 目標：讓 MLR 能把「模型的一部分」打包成具名型別（權重＋偏壓＋作用在它們
> 上的函式），例如一個 transformer block、一層 GRU。這是現在
> `examples/gpt/gpt.mlr` 最大的痛點：`block(...)` 要傳 15 個參數。

---

## 1. 範圍（第一版做什麼、不做什麼）

### 做

| 功能 | 說明 |
| ---- | ---- |
| `struct` 定義 | 具名欄位，欄位型別為純量、tensor 或**其他 struct（by value）** |
| 建構 | `Name { field: expr, ... }`，欄位可省略者必須有 default（第一版：全部必填） |
| 欄位讀取 | `s.w`（點號存取） |
| 欄位更新（功能性） | `s with { w: new_w }` 產生新值（SSA 不可變，語意如同整份拷貝） |
| 方法 | `impl Name { fn f(self, ...) -> T { ... } }`，呼叫 `s.f(args)` |
| 嵌套 | struct 的欄位可以是另一個 struct；`block.attn.wq` 串接存取 |
| 多回傳 | `yield` 可以回傳 struct |

### 不做（第一版明確排除，寫進錯誤訊息）

| 排除 | 原因 |
| ---- | ---- |
| 字串欄位 | 語言目前沒有字串型別值（只有字面量，用於路徑/key） |
| 泛型 / 型別參數 | tensor 維度在型別裡已經是「參數」；泛型會讓 checker 大改 |
| 繼承 / 動態分派 / trait | MLIR 靜態編譯、無 vtable 概念；多型用「enum + switch」未來再議 |
| heap / 所有權 / borrow | 一切 by value；memref 的 in-place 行為沿用現有 bufferize 規則 |
| `static` / 類別欄位、可見性（pub/private） | 全部 module 內可見即可 |
| closure / 把方法當值傳遞 | 函式尚非一等公民 |

> **名稱**：對外叫 `struct` + `impl`（Rust 風格），不叫 `class`——沒有
> 繼承與動態分派，叫 class 會誤導。

---

## 2. 語法

```mlr
// ── 定義：放在函式之前，module 頂層 ──
struct Vec3 {
    x: f32,
    y: f32,
    z: f32
}

struct Attention {
    wq: Tensor<32x16xf32>,
    wk: Tensor<32x16xf32>,
    wv: Tensor<32x16xf32>
}

// 欄位可以是另一個 struct
struct Block {
    attn: Attention,
    fc1: Tensor<32x128xf32>,
    fc2: Tensor<128x32xf32>
}

// ── 方法：impl 區塊；第一個參數固定是 self ──
impl Vec3 {
    fn dot(self, other: Vec3) -> f32 {
        yield self.x * other.x + self.y * other.y + self.z * other.z
    }
    fn scaled(self, s: f32) -> Vec3 {
        yield Vec3 { x: self.x * s, y: self.y * s, z: self.z * s }
    }
}

impl Block {
    // 內部可以再用 let 綁定中間的 struct
    fn attention(self, x: Tensor<16x32xf32>) -> Tensor<16x32xf32> {
        let h = layernorm(x, self.ln1_g, self.ln1_b)   // ← 嵌套 + 方法內用 self
        ...
    }
}

// ── 使用 ──
fn main() {
    let wq = mlr.load<Tensor<32x16xf32>>("examples/gpt/layer0_wq0.bin")
    ...
    let attn = Attention { wq: wq, wk: wk, wv: wv }
    let block = Block { attn: attn, fc1: fc1, fc2: fc2 }

    let q = matmul(x, block.attn.wq)          // 串接欄位存取
    let d = block.attn.wq.row_dot(block.attn.wk)   // 方法呼叫（若有的話）
    let block2 = block with { fc1: new_fc1 }  // 功能性更新
}
```

### 語法細節（決定點，請確認）

1. **建構子**：一律 `Name { field: expr, ... }`，欄位順序不限、全部必填。
   不做位置式建構（`Name(a, b, c)`）——它和函式呼叫無法區分。
2. **更新**：`expr with { field: expr2, ... }` 只寫出指定的欄位，其餘拷貝。
   （替代方案是 `let mut` + `s.x = ...`，但 MLR 的 `let` 不可變是核心設計，
   不想為 struct 開特例。）
3. **方法呼叫**：`recv.f(a, b)` 脫糖為 `f(recv, a, b)`。若同名自由函式與
   方法並存，點號呼叫優先找 method table。
4. **self**：隱含參數，型別固定為所屬 struct；顯式寫在參數列第一位
   （像 Rust）。
5. **`self` 的欄位寫入**：不允許（`self.w = ...` 是編譯錯誤）；方法要改
   struct 就 `yield` 新的回去，呼叫端 `let s2 = s.f(...)`。

---

## 3. 語意

### 3.1 Value semantics

- struct 是**值的 tuple**：指定、傳參、回傳都是淺拷貝（實際上編譯期只是
  多個 SSA 值的打包，沒有真正的 copy）。
- tensor 欄位照舊走現有 bufferize 規則：最後一次使用可 in-place，否則配置
  新 buffer。struct 不改變這條規則，只是把多個 memref 綁在一起。

### 3.2 型別檢查

- `CType` 新增 `CType::Struct(DefId)`，`DefId` 指向 checker 的 struct 表：
  `Vec<StructDef { name, fields: Vec<(String, CType)>, methods: Vec<MethodSig> }>`。
- 規則：
  - 欄位型別必須是純量 / tensor / 已定義的 struct（禁止遞迴、禁止字串）；
  - 兩個 struct 就算欄位完全相同也不互通（名義型別，nominal）；
  - `with` 之後產生同型別的新值；
  - 方法名稱在 impl 表內唯一；自由函式與方法不同命名空間（點號才查方法）。

### 3.3 遞迴與大小

- 禁止 struct 直接或間接包含自己（無指標，一定無窮大）——checker 擋掉。
- struct 欄位數上限（建議 64），錯誤訊息明確。

---

## 4. 實作（lowering）方案

### 4.1 IR 表示

`src/ir.rs` 新增：

```rust
pub enum Type {
    ...
    /// A struct by value: the fields in order.
    Struct(Vec<Type>),          // 匿名、按欄位順序展開
}
```

- struct 的**名義性**在 checker 層處理，進到 IR 後就只是型別的 tuple
  （方法名稱也不進 IR）。
- 新運算（stage 1 產生，generic 形式，如 `mlr.*` 一樣）：

```
%0 = "mlr.struct"(%a, %b, %c) : (f32, tensor<32x16xf32>, ...) -> (!mlr.struct<...>)
%f = "mlr.field"(%0) {index = 1} : (!mlr.struct<...>) -> tensor<32x16xf32>
```

### 4.2 降級路徑（stage 5）

兩個候選，**建議 A**：

**方案 A：直接展開成 LLVM struct 運算**

stage 5 把 `mlr.struct` / `mlr.field` 換成 LLVM dialect：

```
%0 = llvm.mlir.undef : !llvm.struct<(f32, tensor...)>   // 欄位逐一 insertvalue
%1 = llvm.insertvalue %a, %0[0] : ...
%f = llvm.extractvalue %1[1] : ...
```

- `mlir-translate --mlir-to-llvmir` 原生支援，`clang` 也原生處理 C struct
  傳參／回傳，函式簽名不會爆炸。
- 風險：tensor 欄位進 LLVM struct 變成 array-in-struct，**按值傳遞大 tensor
  欄位會整份拷貝**——所以規則訂為：**進出函式邊界時 struct 會被拆解**。
  詳見 4.3。

**方案 B：SOA（struct of arrays）**——把 struct 一律拆成多個 SSA 值，
函式回傳多值、呼叫端再打包。
- 優點：完全不用碰 LLVM struct。
- 缺點：`yield` struct、嵌套 struct 時引數數量組合爆炸；IR 難讀。

### 4.3 函式邊界規則（重要）

tensor 欄位按值傳遞會拷貝整塊記憶體。因此：

- **函式參數與回傳值中的 struct**：stage 5 拆解成多個參數
  （`fn f(s: Block)` → `f(s.attn.wq, s.attn.wk, ..., s.fc2)` 的形狀），
  內部再重新打包。這讓 ABI 永遠是「指標 + index」的組合，MPS runtime
  介面不變。
- **方法 = 普通函式**：`impl Block { fn forward(...) }` 降級成
  `@Block_forward(self拆解後的欄位..., ...)`，與現有 runtime call 機制完全
  相容。
- 若之後測出 LLVM struct 直接傳也夠快（欄位小、數量少），再把這條規則放寬。

### 4.4 各階段工作清單

| 階段 | 工作 | 檔案 |
| ---- | ---- | ---- |
| 語法 | `struct` / `impl` 頂層項目、`Name { .. }`、`expr with { .. }`、`a.b`、`recv.f(..)` | `lexer.rs`（`with` 關鍵字）、`parser.rs`、`ast.rs` |
| 檢查 | `StructDef` 表、`CType::Struct`、欄位/方法規則、遞迴禁令 | `check.rs` |
| Stage 1 | 建構 → `mlr.struct`、存取 → `mlr.field`、方法 → 普通呼叫（mangle 名字） | `build.rs` |
| Stage 2–4 | `mlr.struct` / `mlr.field` 對 canonicalize / bufferize 是不透明純運算，原則上不用改；bufferize 要讓 field 的 memref 正確映射 | `passes.rs`、`passes/bufferize.rs` |
| Stage 5 | `mlr.struct`/`mlr.field` → `llvm.insertvalue`/`extractvalue`；函式邊界拆解 | `passes/runtime_ops.rs`、`ir.rs`（列印） |
| Runtime | 不需要新函式 | — |
| 測試 | 語法/型別錯誤、嵌套、方法、with、round-trip 對照 numpy | `tests/compile.rs` |

### 4.5 需要動到的既有規則（影響評估）

- `ir.rs` 的 printer/verifier：新增 `Type::Struct` 與兩個 generic op 的列印。
- `ir.rs` `Type::scalar()`/`name()`：`Struct` 不是 scalar，函式宣告表
  （`FuncDecl`）不會出現 struct（邊界已拆解），影響小。
- `check.rs` 的 `check_call`：方法呼叫走新分支。
- 現有範例不受影響（純加法）。

---

## 5. 驗收標準

1. `examples/gpt/gpt.mlr` 改寫成 struct 版（`Attention`、`Block`、`Gpt`），
   行為與現版逐 byte 相同（greedy 對照 numpy）。
2. `scripts/run_all.sh` 全綠；`cargo test` 新增 ≥ 8 個測試：
   - struct 定義/建構/存取/with 的正例 4 個
   - 錯誤例：未知欄位、缺欄位、型別不合、遞迴 struct、方法內寫 self
3. `doc/mlr.md` 補章節；本文件標記為「已實作」。

---

## 6. 之後（不在這次做）

- 泛型 struct（`struct Layer<T> { w: Tensor<WxNxf32>, act: T }`）— 需要型別
  參數化，等 tensor 維度 unification 穩定後再評估。
- enum + pattern matching（多型儒 want 的方向）。
- 把方法當值（函式指標）→ 才可能有 callback 風格的訓練迴圈。
