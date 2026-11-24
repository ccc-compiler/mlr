//! Reading and writing the checkpoint formats of PyTorch.
//!
//! Two formats are supported:
//!
//! * `.pt` — the ZIP archive that `torch.save` writes. The archive holds a
//!   pickle stream with the metadata of the tensors and the raw data of the
//!   storages. The pickle is interpreted by a small interpreter that knows
//!   the operations that PyTorch uses.
//! * `.safetensors` — the format of the HuggingFace models: a JSON header
//!   with the metadata of the tensors followed by the raw data.
//!
//! The tensors are read as `f32` (or `i32`) values, whatever their type in
//! the file was, so that they can be embedded into an MLR program.

// Some of the helpers (the writer and the metadata accessors) are only used
// by the tests, but they belong to the public surface of the module.
#![allow(dead_code)]

use std::collections::BTreeMap;

// ───────── the tensors ─────────

/// The type of the elements of a tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    Bool,
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F16 | DType::I16 => 2,
            DType::F32 | DType::I32 => 4,
            DType::F64 | DType::I64 => 8,
            DType::I8 | DType::U8 | DType::Bool => 1,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::F16 => "f16",
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::I8 => "i8",
            DType::I16 => "i16",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::U8 => "u8",
            DType::Bool => "bool",
        }
    }
}

/// A tensor of a checkpoint.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub dtype: DType,
    pub shape: Vec<i64>,
    /// The elements, in the order of the shape (row major).
    pub data: Vec<f64>,
    /// The elements that are integers, when the tensor is an integer tensor.
    pub ints: Vec<i64>,
}

impl Tensor {
    fn new(dtype: DType, shape: Vec<i64>) -> Tensor {
        let count: usize = shape.iter().map(|d| *d as usize).product();
        match dtype {
            DType::F16 | DType::F32 | DType::F64 => Tensor {
                dtype,
                shape,
                data: vec![0.0; count],
                ints: vec![],
            },
            _ => Tensor {
                dtype,
                shape,
                data: vec![],
                ints: vec![0; count],
            },
        }
    }

    pub fn count(&self) -> usize {
        self.data.len().max(self.ints.len())
    }

    /// The values as `f32`.
    pub fn as_f32(&self) -> Vec<f32> {
        match self.dtype {
            DType::F16 | DType::F32 | DType::F64 => self.data.iter().map(|v| *v as f32).collect(),
            _ => self.ints.iter().map(|v| *v as f32).collect(),
        }
    }

    /// The values as `i32`.
    pub fn as_i32(&self) -> Vec<i32> {
        match self.dtype {
            DType::F16 | DType::F32 | DType::F64 => self.data.iter().map(|v| *v as i32).collect(),
            _ => self.ints.iter().map(|v| *v as i32).collect(),
        }
    }
}

// ───────── ZIP ─────────

/// The entries of a ZIP archive that is stored without compression.
fn zip_entries(data: &[u8]) -> Result<BTreeMap<String, Vec<u8>>, String> {
    // The end of the central directory.
    let mut end = None;
    let start = data.len().saturating_sub(66_000);
    let mut i = data.len().saturating_sub(22);
    while i >= start {
        if data[i..].starts_with(&[0x50, 0x4b, 0x05, 0x06]) {
            end = Some(i);
            break;
        }
        if i == 0 {
            break;
        }
        i -= 1;
    }
    let end = end.ok_or_else(|| "not a ZIP archive".to_string())?;
    let count = u16::from_le_bytes([data[end + 10], data[end + 11]]) as usize;
    let mut offset = u32::from_le_bytes([
        data[end + 16],
        data[end + 17],
        data[end + 18],
        data[end + 19],
    ]) as usize;

    let mut entries = BTreeMap::new();
    for _ in 0..count {
        if data.len() < offset + 46 || !data[offset..].starts_with(&[0x50, 0x4b, 0x01, 0x02]) {
            break;
        }
        let method = u16::from_le_bytes([data[offset + 10], data[offset + 11]]);
        let size = u32::from_le_bytes([
            data[offset + 24],
            data[offset + 25],
            data[offset + 26],
            data[offset + 27],
        ]) as usize;
        let name_len = u16::from_le_bytes([data[offset + 28], data[offset + 29]]) as usize;
        let extra_len = u16::from_le_bytes([data[offset + 30], data[offset + 31]]) as usize;
        let comment_len = u16::from_le_bytes([data[offset + 32], data[offset + 33]]) as usize;
        let local = u32::from_le_bytes([
            data[offset + 42],
            data[offset + 43],
            data[offset + 44],
            data[offset + 45],
        ]) as usize;
        let name = String::from_utf8_lossy(&data[offset + 46..offset + 46 + name_len]).to_string();
        offset += 46 + name_len + extra_len + comment_len;

        if method != 0 {
            return Err(format!(
                "`{}` is compressed, only archives without compression are supported",
                name
            ));
        }
        // The local header repeats the name and the extra field.
        if data.len() < local + 30 || !data[local..].starts_with(&[0x50, 0x4b, 0x03, 0x04]) {
            return Err(format!("`{}` has a broken local header", name));
        }
        let local_name = u16::from_le_bytes([data[local + 26], data[local + 27]]) as usize;
        let local_extra = u16::from_le_bytes([data[local + 28], data[local + 29]]) as usize;
        let start = local + 30 + local_name + local_extra;
        if data.len() < start + size {
            return Err(format!("`{}` is truncated", name));
        }
        entries.insert(name, data[start..start + size].to_vec());
    }
    Ok(entries)
}

// ───────── the pickle ─────────

/// A value of the pickle stream.
#[derive(Clone, Debug)]
enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    /// A global name: a module and a name.
    Global(String, String),
    /// A persistent id: the pieces of the tuple.
    Persistent(Vec<Value>),
    /// A tensor that was rebuilt by `torch._utils._rebuild_tensor_v2`.
    Tensor {
        storage: String,
        dtype: DType,
        offset: i64,
        size: Vec<i64>,
        stride: Vec<i64>,
    },
    Opaque,
}

impl Value {
    fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(v) => Some(*v),
            Value::Bool(b) => Some(*b as i64),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    fn as_tuple(&self) -> Option<&[Value]> {
        match self {
            Value::Tuple(v) => Some(v),
            _ => None,
        }
    }
}

/// The storage classes of PyTorch and their element type.
fn storage_dtype(module: &str, name: &str) -> Option<DType> {
    if module != "torch" {
        return None;
    }
    Some(match name {
        "HalfStorage" | "Float16Storage" => DType::F16,
        "FloatStorage" | "Float32Storage" => DType::F32,
        "DoubleStorage" | "Float64Storage" => DType::F64,
        "CharStorage" | "Int8Storage" => DType::I8,
        "ShortStorage" | "Int16Storage" => DType::I16,
        "IntStorage" | "Int32Storage" => DType::I32,
        "LongStorage" | "Int64Storage" => DType::I64,
        "ByteStorage" | "UInt8Storage" => DType::U8,
        "BoolStorage" => DType::Bool,
        _ => return None,
    })
}

/// Interprets a pickle stream and returns the value it holds.
fn pickle_load(data: &[u8]) -> Result<Value, String> {
    let mut stack: Vec<Value> = Vec::new();
    let mut marks: Vec<usize> = Vec::new();
    let mut memo: BTreeMap<usize, Value> = BTreeMap::new();
    let mut pos = 0usize;

    macro_rules! fail {
        ($($arg:tt)*) => {
            return Err(format!($($arg)*))
        };
    }

    while pos < data.len() {
        let op = data[pos];
        pos += 1;
        match op {
            0x80 => {
                // PROTO
                pos += 1;
            }
            0x95 => {
                // FRAME
                pos += 8;
            }
            0x94 => {
                // MEMOIZE
                let value = stack.last().cloned().unwrap_or(Value::None);
                memo.insert(memo.len(), value);
            }
            0x28 => marks.push(stack.len()), // MARK
            0x7d => stack.push(Value::Dict(Vec::new())), // EMPTY_DICT
            0x5d => stack.push(Value::Dict(Vec::new())), // EMPTY_DICT (the other one)
            0x29 => stack.push(Value::Tuple(Vec::new())), // EMPTY_TUPLE
            0x5b => stack.push(Value::List(Vec::new())), // EMPTY_LIST
            0x88 => stack.push(Value::Bool(true)),  // NEWTRUE
            0x89 => stack.push(Value::Bool(false)), // NEWFALSE
            0x4e => stack.push(Value::None),        // NONE
            0x71 | 0x72 => {
                // BINPUT / LONG_BINPUT
                let index = if op == 0x71 {
                    let v = *data.get(pos).ok_or("truncated pickle")? as usize;
                    pos += 1;
                    v
                } else {
                    let v = u32::from_le_bytes([
                        *data.get(pos).ok_or("truncated pickle")?,
                        *data.get(pos + 1).ok_or("truncated pickle")?,
                        *data.get(pos + 2).ok_or("truncated pickle")?,
                        *data.get(pos + 3).ok_or("truncated pickle")?,
                    ]) as usize;
                    pos += 4;
                    v
                };
                let value = stack.last().cloned().unwrap_or(Value::None);
                memo.insert(index, value);
            }
            0x68 | 0x6a => {
                // BINGET / LONG_BINGET
                let index = if op == 0x68 {
                    let v = *data.get(pos).ok_or("truncated pickle")? as usize;
                    pos += 1;
                    v
                } else {
                    let v = u32::from_le_bytes([
                        *data.get(pos).ok_or("truncated pickle")?,
                        *data.get(pos + 1).ok_or("truncated pickle")?,
                        *data.get(pos + 2).ok_or("truncated pickle")?,
                        *data.get(pos + 3).ok_or("truncated pickle")?,
                    ]) as usize;
                    pos += 4;
                    v
                };
                let value = memo.get(&index).cloned().unwrap_or(Value::None);
                stack.push(value);
            }
            0x4a => {
                // BININT
                let v = i32::from_le_bytes([
                    *data.get(pos).ok_or("truncated pickle")?,
                    *data.get(pos + 1).ok_or("truncated pickle")?,
                    *data.get(pos + 2).ok_or("truncated pickle")?,
                    *data.get(pos + 3).ok_or("truncated pickle")?,
                ]);
                pos += 4;
                stack.push(Value::Int(v as i64));
            }
            0x4b => {
                // BININT1
                let v = *data.get(pos).ok_or("truncated pickle")? as i64;
                pos += 1;
                stack.push(Value::Int(v));
            }
            0x4d => {
                // BININT2
                let v = u16::from_le_bytes([
                    *data.get(pos).ok_or("truncated pickle")?,
                    *data.get(pos + 1).ok_or("truncated pickle")?,
                ]) as i64;
                pos += 2;
                stack.push(Value::Int(v));
            }
            0x8a => {
                // LONG1
                let n = *data.get(pos).ok_or("truncated pickle")? as usize;
                pos += 1;
                let mut value: i64 = 0;
                for i in 0..n.min(8) {
                    value |= (*data.get(pos + i).ok_or("truncated pickle")? as i64) << (8 * i);
                }
                pos += n;
                stack.push(Value::Int(value));
            }
            0x47 => {
                // BINFLOAT
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(data.get(pos..pos + 8).ok_or("truncated pickle")?);
                pos += 8;
                stack.push(Value::Float(f64::from_be_bytes(bytes)));
            }
            0x8c => {
                // SHORT_BINUNICODE
                let n = *data.get(pos).ok_or("truncated pickle")? as usize;
                pos += 1;
                let text = data.get(pos..pos + n).ok_or("truncated pickle")?;
                pos += n;
                stack.push(Value::Str(String::from_utf8_lossy(text).to_string()));
            }
            0x58 => {
                // BINUNICODE
                let n = u32::from_le_bytes([
                    *data.get(pos).ok_or("truncated pickle")?,
                    *data.get(pos + 1).ok_or("truncated pickle")?,
                    *data.get(pos + 2).ok_or("truncated pickle")?,
                    *data.get(pos + 3).ok_or("truncated pickle")?,
                ]) as usize;
                pos += 4;
                let text = data.get(pos..pos + n).ok_or("truncated pickle")?;
                pos += n;
                stack.push(Value::Str(String::from_utf8_lossy(text).to_string()));
            }
            0x8d => {
                // BINUNICODE8
                let n = u64::from_le_bytes([
                    *data.get(pos).ok_or("truncated pickle")?,
                    *data.get(pos + 1).ok_or("truncated pickle")?,
                    *data.get(pos + 2).ok_or("truncated pickle")?,
                    *data.get(pos + 3).ok_or("truncated pickle")?,
                    *data.get(pos + 4).ok_or("truncated pickle")?,
                    *data.get(pos + 5).ok_or("truncated pickle")?,
                    *data.get(pos + 6).ok_or("truncated pickle")?,
                    *data.get(pos + 7).ok_or("truncated pickle")?,
                ]) as usize;
                pos += 8;
                let text = data.get(pos..pos + n).ok_or("truncated pickle")?;
                pos += n;
                stack.push(Value::Str(String::from_utf8_lossy(text).to_string()));
            }
            0x43 => {
                // SHORT_BINBYTES
                let n = *data.get(pos).ok_or("truncated pickle")? as usize;
                pos += 1;
                let bytes = data.get(pos..pos + n).ok_or("truncated pickle")?;
                pos += n;
                stack.push(Value::Bytes(bytes.to_vec()));
            }
            0x42 => {
                // BINBYTES
                let n = u32::from_le_bytes([
                    *data.get(pos).ok_or("truncated pickle")?,
                    *data.get(pos + 1).ok_or("truncated pickle")?,
                    *data.get(pos + 2).ok_or("truncated pickle")?,
                    *data.get(pos + 3).ok_or("truncated pickle")?,
                ]) as usize;
                pos += 4;
                let bytes = data.get(pos..pos + n).ok_or("truncated pickle")?;
                pos += n;
                stack.push(Value::Bytes(bytes.to_vec()));
            }
            0x85 | 0x86 | 0x87 => {
                // TUPLE1, TUPLE2, TUPLE3
                let n = match op {
                    0x85 => 1,
                    0x86 => 2,
                    _ => 3,
                };
                if stack.len() < n {
                    fail!("truncated pickle");
                }
                let items = stack.split_off(stack.len() - n);
                stack.push(Value::Tuple(items));
            }
            0x74 => {
                // TUPLE
                let mark = marks.pop().unwrap_or(0);
                let items = stack.split_off(mark);
                stack.push(Value::Tuple(items));
            }
            0x6c => {
                // LIST
                let mark = marks.pop().unwrap_or(0);
                let items = stack.split_off(mark);
                stack.push(Value::List(items));
            }
            0x61 => {
                // APPEND
                let value = stack.pop().unwrap_or(Value::None);
                if let Some(Value::List(items)) = stack.last_mut() {
                    items.push(value);
                }
            }
            0x65 => {
                // APPENDS
                let mark = marks.pop().unwrap_or(0);
                let items = stack.split_off(mark);
                if let Some(Value::List(list)) = stack.last_mut() {
                    list.extend(items);
                }
            }
            0x73 => {
                // SETITEM
                let value = stack.pop().unwrap_or(Value::None);
                let key = stack.pop().unwrap_or(Value::None);
                if let Some(Value::Dict(entries)) = stack.last_mut() {
                    entries.push((key, value));
                }
            }
            0x75 => {
                // SETITEMS
                let mark = marks.pop().unwrap_or(0);
                let items = stack.split_off(mark);
                if let Some(Value::Dict(entries)) = stack.last_mut() {
                    let mut i = 0;
                    while i + 1 < items.len() {
                        entries.push((items[i].clone(), items[i + 1].clone()));
                        i += 2;
                    }
                }
            }
            0x63 => {
                // GLOBAL
                let module = read_line(data, &mut pos)?;
                let name = read_line(data, &mut pos)?;
                stack.push(Value::Global(module, name));
            }
            0x93 => {
                // STACK_GLOBAL
                if stack.len() < 2 {
                    fail!("truncated pickle");
                }
                let name = stack.pop().unwrap_or(Value::None);
                let module = stack.pop().unwrap_or(Value::None);
                stack.push(Value::Global(
                    module.as_str().unwrap_or("").to_string(),
                    name.as_str().unwrap_or("").to_string(),
                ));
            }
            0x51 => {
                // BINPERSID
                let value = stack.pop().unwrap_or(Value::None);
                stack.push(Value::Persistent(match value {
                    Value::Tuple(items) => items,
                    other => vec![other],
                }));
            }
            0x52 => {
                // REDUCE
                let args = stack.pop().unwrap_or(Value::Tuple(Vec::new()));
                let callable = stack.pop().unwrap_or(Value::Opaque);
                let items = args.as_tuple().unwrap_or(&[]).to_vec();
                let reduced = match &callable {
                    Value::Global(module, name) => match (module.as_str(), name.as_str()) {
                        ("torch._utils", "_rebuild_tensor_v2")
                        | ("torch._utils", "_rebuild_tensor") => {
                            rebuild_tensor(&items)
                        }
                        ("torch._utils", "_rebuild_parameter")
                        | ("torch._utils", "_rebuild_parameter_with_state") => {
                            rebuild_tensor(&items)
                        }
                        _ => Value::Opaque,
                    },
                    _ => Value::Opaque,
                };
                stack.push(reduced);
            }
            0x62 => {
                // BUILD: the state of an object, ignored.
                stack.pop();
                stack.pop();
            }
            0x2e => {
                // STOP
                break;
            }
            other => fail!("unsupported pickle operation 0x{:02x}", other),
        }
    }
    stack.pop().ok_or_else(|| "empty pickle".to_string())
}

/// Reads a line of a pickle stream (the lines end with a newline).
fn read_line(data: &[u8], pos: &mut usize) -> Result<String, String> {
    let start = *pos;
    while *pos < data.len() && data[*pos] != b'\n' {
        *pos += 1;
    }
    let text = String::from_utf8_lossy(&data[start..*pos]).to_string();
    *pos += 1;
    Ok(text)
}

/// The arguments of `torch._utils._rebuild_tensor_v2`.
fn rebuild_tensor(items: &[Value]) -> Value {
    if items.len() < 4 {
        return Value::Opaque;
    }
    let storage = match &items[0] {
        Value::Persistent(parts) => {
            // ('storage', <the storage class>, <the key>, <the location>, <the count>)
            if parts.len() < 3 {
                return Value::Opaque;
            }
            let key = parts[2].as_str().unwrap_or("").to_string();
            let dtype = match &parts[1] {
                Value::Global(module, name) => storage_dtype(module, name),
                _ => None,
            };
            match dtype {
                Some(dtype) => (key, dtype),
                None => return Value::Opaque,
            }
        }
        _ => return Value::Opaque,
    };
    let offset = items[1].as_int().unwrap_or(0);
    let size: Vec<i64> = items[2]
        .as_tuple()
        .unwrap_or(&[])
        .iter()
        .map(|v| v.as_int().unwrap_or(0))
        .collect();
    let stride: Vec<i64> = items[3]
        .as_tuple()
        .unwrap_or(&[])
        .iter()
        .map(|v| v.as_int().unwrap_or(0))
        .collect();
    Value::Tensor {
        storage: storage.0,
        dtype: storage.1,
        offset,
        size,
        stride,
    }
}

// ───────── the checkpoints ─────────

/// Reads the tensors of a `.pt` file.
pub fn read_pt(data: &[u8]) -> Result<BTreeMap<String, Tensor>, String> {
    let entries = zip_entries(data)?;
    // The pickle is the entry that ends with `data.pkl`.
    let pickle = entries
        .iter()
        .find(|(name, _)| name.ends_with("data.pkl"))
        .map(|(_, data)| data.clone())
        .ok_or_else(|| "the archive has no `data.pkl`".to_string())?;
    let value = pickle_load(&pickle)?;

    // The storages of the archive.
    let mut storages: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for (name, bytes) in &entries {
        if let Some(rest) = name.split_once("/data/") {
            storages.insert(rest.1.to_string(), bytes.clone());
        }
    }

    // Collect the tensors and the storages they refer to.
    let mut found: Vec<(String, Value)> = Vec::new();
    collect_values(&value, "", &mut found);
    let mut out = BTreeMap::new();
    for (path, item) in found {
        let Value::Tensor {
            storage,
            dtype,
            offset,
            size,
            stride,
        } = item
        else {
            continue;
        };
        let bytes = storages
            .get(&storage)
            .ok_or_else(|| format!("the archive has no storage `{}`", storage))?;
        let tensor = read_tensor(dtype, &size, &stride, offset, bytes)?;
        out.insert(path, tensor);
    }
    Ok(out)
}

/// Collects the tensors of a value with their paths.
fn collect_values(value: &Value, prefix: &str, out: &mut Vec<(String, Value)>) {
    match value {
        Value::Dict(entries) => {
            for (key, item) in entries {
                let Some(name) = key.as_str() else { continue };
                let path = if prefix.is_empty() {
                    name.to_string()
                } else {
                    format!("{}.{}", prefix, name)
                };
                collect_values(item, &path, out);
            }
        }
        Value::List(items) => {
            for (i, item) in items.iter().enumerate() {
                let path = if prefix.is_empty() {
                    i.to_string()
                } else {
                    format!("{}.{}", prefix, i)
                };
                collect_values(item, &path, out);
            }
        }
        Value::Tuple(items) => {
            for (i, item) in items.iter().enumerate() {
                let path = if prefix.is_empty() {
                    i.to_string()
                } else {
                    format!("{}.{}", prefix, i)
                };
                collect_values(item, &path, out);
            }
        }
        Value::Tensor { .. } => out.push((prefix.to_string(), value.clone())),
        _ => {}
    }
}

/// Reads the elements of a tensor out of the raw data of its storage.
fn read_tensor(
    dtype: DType,
    size: &[i64],
    stride: &[i64],
    offset: i64,
    bytes: &[u8],
) -> Result<Tensor, String> {
    let mut tensor = Tensor::new(dtype, size.to_vec());
    let count = tensor.count();
    let element = dtype.size();
    let contiguous = stride.is_empty()
        || (stride.len() == size.len()
            && stride
                .iter()
                .zip(size.iter())
                .enumerate()
                .all(|(i, (s, _d))| *s == row_major_stride(size, i)));
    if contiguous {
        let start = offset as usize * element;
        if bytes.len() < start + count * element {
            return Err("the storage is too small".to_string());
        }
        for i in 0..count {
            let at = start + i * element;
            let value = read_element(dtype, &bytes[at..at + element]);
            match dtype {
                DType::F16 | DType::F32 | DType::F64 => tensor.data[i] = value,
                _ => tensor.ints[i] = value as i64,
            }
        }
    } else {
        // A view with a stride: walk over the indices.
        let mut index = vec![0i64; size.len()];
        for i in 0..count {
            let mut at = offset as usize;
            for (s, value) in stride.iter().zip(index.iter()) {
                at += ((*s) * value) as usize;
            }
            let at = at * element;
            if at + element > bytes.len() {
                return Err("the storage is too small".to_string());
            }
            let value = read_element(dtype, &bytes[at..at + element]);
            match dtype {
                DType::F16 | DType::F32 | DType::F64 => tensor.data[i] = value,
                _ => tensor.ints[i] = value as i64,
            }
            // The next index in row major order.
            let mut d = size.len();
            while d > 0 {
                d -= 1;
                index[d] += 1;
                if index[d] < size[d] {
                    break;
                }
                index[d] = 0;
            }
        }
    }
    Ok(tensor)
}

/// The stride of a row major tensor.
fn row_major_stride(size: &[i64], dim: usize) -> i64 {
    let mut stride = 1i64;
    for d in size.iter().skip(dim + 1) {
        stride *= d;
    }
    stride
}

/// Reads one element of the raw data of a storage.
fn read_element(dtype: DType, bytes: &[u8]) -> f64 {
    match dtype {
        DType::F16 => {
            let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
            f16_to_f64(bits)
        }
        DType::F32 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64,
        DType::F64 => f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
        DType::I8 => bytes[0] as i8 as f64,
        DType::U8 => bytes[0] as f64,
        DType::Bool => (bytes[0] != 0) as i64 as f64,
        DType::I16 => i16::from_le_bytes([bytes[0], bytes[1]]) as f64,
        DType::I32 => i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64,
        DType::I64 => i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]) as f64,
    }
}

/// Converts a half precision float to a double.
fn f16_to_f64(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let mantissa = (bits & 0x03ff) as f64;
    let value = if exponent == 0 {
        mantissa * 2f64.powi(-24)
    } else if exponent == 0x1f {
        f64::INFINITY
    } else {
        (1.0 + mantissa / 1024.0) * 2f64.powi(exponent - 15)
    };
    sign * value
}

// ───────── safetensors ─────────

/// A minimal JSON reader, enough for the header of a safetensors file.
struct Json<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Json<'a> {
    fn skip_space(&mut self) {
        while self.pos < self.data.len() && (self.data[self.pos] as char).is_whitespace() {
            self.pos += 1;
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.skip_space();
        if self.data.get(self.pos) != Some(&b'"') {
            return Err("expected a string in the JSON header".to_string());
        }
        self.pos += 1;
        let mut out = String::new();
        loop {
            let byte = *self.data.get(self.pos).ok_or("truncated JSON")?;
            self.pos += 1;
            match byte {
                b'"' => break,
                b'\\' => {
                    let escape = *self.data.get(self.pos).ok_or("truncated JSON")?;
                    self.pos += 1;
                    match escape {
                        b'n' => out.push('\n'),
                        b't' => out.push('\t'),
                        b'r' => out.push('\r'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'u' => {
                            let hex = self
                                .data
                                .get(self.pos..self.pos + 4)
                                .ok_or("truncated JSON")?;
                            self.pos += 4;
                            let code = u32::from_str_radix(&String::from_utf8_lossy(hex), 16)
                                .map_err(|_| "bad escape in the JSON header")?;
                            out.push(char::from_u32(code).unwrap_or('?'));
                        }
                        other => out.push(other as char),
                    }
                }
                other => {
                    // A UTF-8 character.
                    let start = self.pos - 1;
                    let len = utf8_len(other);
                    self.pos = start + len;
                    let text = self
                        .data
                        .get(start..self.pos)
                        .ok_or("truncated JSON")?;
                    out.push_str(&String::from_utf8_lossy(text));
                }
            }
        }
        Ok(out)
    }

    fn value(&mut self) -> Result<JsonValue, String> {
        self.skip_space();
        match self.data.get(self.pos) {
            Some(b'{') => {
                self.pos += 1;
                let mut entries = Vec::new();
                loop {
                    self.skip_space();
                    if self.data.get(self.pos) == Some(&b'}') {
                        self.pos += 1;
                        break;
                    }
                    let key = self.string()?;
                    self.skip_space();
                    if self.data.get(self.pos) != Some(&b':') {
                        return Err("expected `:` in the JSON header".to_string());
                    }
                    self.pos += 1;
                    let value = self.value()?;
                    entries.push((key, value));
                    self.skip_space();
                    if self.data.get(self.pos) == Some(&b',') {
                        self.pos += 1;
                    }
                }
                Ok(JsonValue::Object(entries))
            }
            Some(b'[') => {
                self.pos += 1;
                let mut items = Vec::new();
                loop {
                    self.skip_space();
                    if self.data.get(self.pos) == Some(&b']') {
                        self.pos += 1;
                        break;
                    }
                    items.push(self.value()?);
                    self.skip_space();
                    if self.data.get(self.pos) == Some(&b',') {
                        self.pos += 1;
                    }
                }
                Ok(JsonValue::Array(items))
            }
            Some(b'"') => Ok(JsonValue::String(self.string()?)),
            Some(b't') => {
                self.pos += 4;
                Ok(JsonValue::Bool(true))
            }
            Some(b'f') => {
                self.pos += 5;
                Ok(JsonValue::Bool(false))
            }
            Some(_) => {
                let start = self.pos;
                while self.pos < self.data.len()
                    && matches!(self.data[self.pos], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
                {
                    self.pos += 1;
                }
                let text = String::from_utf8_lossy(&self.data[start..self.pos]).to_string();
                text.parse::<f64>()
                    .map(JsonValue::Number)
                    .map_err(|_| format!("`{}` is not a number", text))
            }
            None => Err("truncated JSON".to_string()),
        }
    }
}

/// The length of a UTF-8 character.
fn utf8_len(byte: u8) -> usize {
    if byte < 0x80 {
        1
    } else if byte >> 5 == 0b110 {
        2
    } else if byte >> 4 == 0b1110 {
        3
    } else {
        4
    }
}

/// A value of the JSON header.
enum JsonValue {
    Object(Vec<(String, JsonValue)>),
    Array(Vec<JsonValue>),
    String(String),
    Number(f64),
    Bool(bool),
}

impl JsonValue {
    fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(entries) => entries
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s),
            _ => None,
        }
    }

    fn as_number(&self) -> Option<f64> {
        match self {
            JsonValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(items) => Some(items),
            _ => None,
        }
    }
}

/// The type of a safetensors tensor.
fn safetensors_dtype(name: &str) -> Option<DType> {
    Some(match name {
        "F16" => DType::F16,
        "F32" => DType::F32,
        "F64" => DType::F64,
        "I8" => DType::I8,
        "I16" => DType::I16,
        "I32" => DType::I32,
        "I64" => DType::I64,
        "U8" => DType::U8,
        "BOOL" => DType::Bool,
        _ => return None,
    })
}

/// Reads the tensors of a safetensors file.
pub fn read_safetensors(data: &[u8]) -> Result<BTreeMap<String, Tensor>, String> {
    // The header is a JSON object at the beginning, its length is given by
    // the first eight bytes.
    if data.len() < 8 {
        return Err("the file is too short".to_string());
    }
    let length = u64::from_le_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ]) as usize;
    if data.len() < 8 + length {
        return Err("the header is truncated".to_string());
    }
    let header = &data[8..8 + length];
    let mut json = Json {
        data: header,
        pos: 0,
    };
    let value = json.value()?;
    let entries = match &value {
        JsonValue::Object(entries) => entries,
        _ => return Err("the header is not an object".to_string()),
    };

    let mut out = BTreeMap::new();
    for (name, item) in entries {
        if name == "__metadata__" {
            continue;
        }
        let dtype = item
            .get("dtype")
            .and_then(|v| v.as_str())
            .and_then(safetensors_dtype)
            .ok_or_else(|| format!("`{}` has an unknown type", name))?;
        let shape: Vec<i64> = item
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|items| items.iter().filter_map(|v| v.as_number()).map(|n| n as i64).collect())
            .unwrap_or_default();
        let begin: Vec<usize> = item
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .map(|items| items.iter().filter_map(|v| v.as_number()).map(|n| n as usize).collect())
            .unwrap_or_default();
        let base = 8 + length;
        let (from, to) = match begin.as_slice() {
            [a, b] => (*a, *b),
            _ => return Err(format!("`{}` has no data", name)),
        };
        if base + to > data.len() {
            return Err(format!("`{}` is truncated", name));
        }
        let bytes = &data[base + from..base + to];
        let tensor = read_tensor(dtype, &shape, &[], 0, bytes)?;
        out.insert(name.clone(), tensor);
    }
    Ok(out)
}

/// The file of a name that can be downloaded.
///
/// A name that starts with `hf://` is a file of a repository of
/// HuggingFace: `hf://gpt2/model.safetensors`. The file is downloaded with
/// `curl` and kept in the cache of the user.
pub fn resolve(name: &str) -> Result<String, String> {
    let rest = match name.strip_prefix("hf://") {
        Some(rest) => rest,
        None => return Ok(name.to_string()),
    };
    // The repository is everything but the last part of the name.
    let (repo, file) = match rest.rsplit_once('/') {
        Some((repo, file)) => (repo, file),
        None => {
            return Err(format!(
                "`{}` is not a file of a repository, it must be `hf://<repository>/<file>`",
                name
            ))
        }
    };
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo, file
    );
    // The cache of the user.
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let cache = format!("{}/.cache/mlr/huggingface", home);
    let file = format!("{}/{}", cache, file.replace('/', "_"));
    std::fs::create_dir_all(&cache)
        .map_err(|e| format!("cannot create the cache `{}`: {}", cache, e))?;
    // The size of the file, when the server tells it.
    let size = remote_size(&url);
    // The file is downloaded in pieces: the download of a big file can be
    // interrupted, and the next try continues where it stopped.
    let mut tries = 0;
    loop {
        tries += 1;
        let done = match size {
            Some(size) => std::fs::metadata(&file)
                .map(|m| m.len() == size)
                .unwrap_or(false),
            None => std::path::Path::new(&file).exists(),
        };
        if done {
            return Ok(file);
        }
        if tries > 20 {
            return Err(format!("cannot download `{}`", url));
        }
        let _ = std::process::Command::new("curl")
            .args(["-fL", "-sS", "-C", "-", "--retry", "3", "-o", &file, &url])
            .status()
            .map_err(|e| format!("cannot run `curl`: {}", e))?;
    }
}

/// The size of a file that is on the network, when the server tells it.
fn remote_size(url: &str) -> Option<u64> {
    let out = std::process::Command::new("curl")
        .args(["-sIL", url])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    // The last answer of the server has the size.
    let mut size = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(value) = line
            .strip_prefix("content-length:")
            .or_else(|| line.strip_prefix("Content-Length:"))
        {
            if let Ok(value) = value.trim().parse::<u64>() {
                size = Some(value);
            }
        }
    }
    size
}

/// Reads the tensors of a checkpoint file, whatever its format is.
pub fn read_checkpoint(path: &str) -> Result<BTreeMap<String, Tensor>, String> {
    let data = std::fs::read(path).map_err(|e| format!("cannot read `{}`: {}", path, e))?;
    if path.ends_with(".safetensors") {
        read_safetensors(&data)
    } else {
        read_pt(&data)
    }
}

/// The names of the tensors of a checkpoint.
pub fn names(tensors: &BTreeMap<String, Tensor>) -> Vec<String> {
    tensors.keys().cloned().collect()
}

// ───────── the writer ─────────

/// The CRC of a chunk of data (the ZIP format needs it).
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for byte in data {
        crc ^= *byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

/// Writes the tensors of a checkpoint in the `.pt` format.
pub fn write_pt(path: &str, tensors: &BTreeMap<String, Tensor>) -> Result<(), String> {
    let mut pickle: Vec<u8> = Vec::new();
    let mut storages: Vec<(String, Vec<u8>)> = Vec::new();

    pickle.push(0x80); // PROTO 2
    pickle.push(0x02);
    pickle.push(0x7d); // EMPTY_DICT
    pickle.push(0x28); // MARK
    for (index, (name, tensor)) in tensors.iter().enumerate() {
        put_string(&mut pickle, name);
        // The storage of the tensor.
        let storage = index.to_string();
        let raw = raw_bytes(tensor);
        storages.push((storage.clone(), raw.clone()));
        // The function that rebuilds the tensor.
        put_global(&mut pickle, "torch._utils", "_rebuild_tensor_v2");
        // The arguments of the function.
        pickle.push(0x28); // MARK
        // The persistent id of the storage:
        // ('storage', torch.FloatStorage, '<key>', 'cpu', <count>)
        pickle.push(0x28); // MARK
        put_string(&mut pickle, "storage");
        put_global(&mut pickle, "torch", storage_class(tensor.dtype));
        put_string(&mut pickle, &storage);
        put_string(&mut pickle, "cpu");
        put_int(&mut pickle, tensor.count() as i64);
        pickle.push(0x74); // TUPLE
        pickle.push(0x51); // BINPERSID
        put_int(&mut pickle, 0); // the storage offset
        put_tuple(&mut pickle, &tensor.shape);
        put_tuple(&mut pickle, &contiguous_stride(&tensor.shape));
        pickle.push(0x89); // NEWFALSE
        // The empty ordered dictionary of the hooks.
        put_global(&mut pickle, "collections", "OrderedDict");
        pickle.push(0x28); // MARK
        pickle.push(0x7d); // EMPTY_DICT
        pickle.push(0x74); // TUPLE
        pickle.push(0x52); // REDUCE
        pickle.push(0x74); // TUPLE (the arguments)
        pickle.push(0x52); // REDUCE
    }
    pickle.push(0x75); // SETITEMS
    pickle.push(0x2e); // STOP

    // The archive.
    let mut out: Vec<u8> = Vec::new();
    let mut entries: Vec<(String, Vec<u8>)> = vec![("archive/data.pkl".to_string(), pickle)];
    for (name, bytes) in storages {
        entries.push((format!("archive/data/{}", name), bytes));
    }
    entries.push(("archive/version".to_string(), b"3\n".to_vec()));
    entries.push(("archive/byteorder".to_string(), b"little".to_vec()));

    let mut offsets = Vec::new();
    for (name, bytes) in &entries {
        let offset = out.len();
        let crc = crc32(bytes);
        out.extend_from_slice(&[0x50, 0x4b, 0x03, 0x04]);
        out.extend_from_slice(&20u16.to_le_bytes()); // the version
        out.extend_from_slice(&0u16.to_le_bytes()); // the flags
        out.extend_from_slice(&0u16.to_le_bytes()); // the method: stored
        out.extend_from_slice(&0u16.to_le_bytes()); // the time
        out.extend_from_slice(&0u16.to_le_bytes()); // the date
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // the extra field
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(bytes);
        offsets.push((name.clone(), crc, bytes.len(), offset));
    }
    let directory = out.len();
    for (name, crc, size, offset) in &offsets {
        out.extend_from_slice(&[0x50, 0x4b, 0x01, 0x02]);
        out.extend_from_slice(&20u16.to_le_bytes()); // the version
        out.extend_from_slice(&20u16.to_le_bytes()); // the version
        out.extend_from_slice(&0u16.to_le_bytes()); // the flags
        out.extend_from_slice(&0u16.to_le_bytes()); // the method
        out.extend_from_slice(&0u16.to_le_bytes()); // the time
        out.extend_from_slice(&0u16.to_le_bytes()); // the date
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(*size as u32).to_le_bytes());
        out.extend_from_slice(&(*size as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // the extra field
        out.extend_from_slice(&0u16.to_le_bytes()); // the comment
        out.extend_from_slice(&0u16.to_le_bytes()); // the disk
        out.extend_from_slice(&0u16.to_le_bytes()); // the internal attributes
        out.extend_from_slice(&0u32.to_le_bytes()); // the external attributes
        out.extend_from_slice(&(*offset as u32).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
    }
    let size = out.len() - directory;
    out.extend_from_slice(&[0x50, 0x4b, 0x05, 0x06]);
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(offsets.len() as u16).to_le_bytes());
    out.extend_from_slice(&(offsets.len() as u16).to_le_bytes());
    out.extend_from_slice(&(size as u32).to_le_bytes());
    out.extend_from_slice(&(directory as u32).to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());

    std::fs::write(path, &out).map_err(|e| format!("cannot write `{}`: {}", path, e))
}

/// The name of the storage class of a type.
fn storage_class(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => "HalfStorage",
        DType::F32 => "FloatStorage",
        DType::F64 => "DoubleStorage",
        DType::I8 => "CharStorage",
        DType::I16 => "ShortStorage",
        DType::I32 => "IntStorage",
        DType::I64 => "LongStorage",
        DType::U8 => "ByteStorage",
        DType::Bool => "BoolStorage",
    }
}

/// The raw bytes of a tensor.
fn raw_bytes(tensor: &Tensor) -> Vec<u8> {
    let mut out = Vec::with_capacity(tensor.count() * tensor.dtype.size());
    match tensor.dtype {
        DType::F16 => {
            for value in &tensor.data {
                out.extend_from_slice(&f64_to_f16(*value).to_le_bytes());
            }
        }
        DType::F32 => {
            for value in &tensor.data {
                out.extend_from_slice(&(*value as f32).to_le_bytes());
            }
        }
        DType::F64 => {
            for value in &tensor.data {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        DType::I8 | DType::U8 | DType::Bool => {
            for value in &tensor.ints {
                out.push(*value as u8);
            }
        }
        DType::I16 => {
            for value in &tensor.ints {
                out.extend_from_slice(&(*value as i16).to_le_bytes());
            }
        }
        DType::I32 => {
            for value in &tensor.ints {
                out.extend_from_slice(&(*value as i32).to_le_bytes());
            }
        }
        DType::I64 => {
            for value in &tensor.ints {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    out
}

/// Converts a double to a half precision float.
fn f64_to_f16(value: f64) -> u16 {
    let bits = value as f32;
    let sign = if bits < 0.0 { 0x8000u16 } else { 0 };
    let magnitude = bits.abs();
    if magnitude.is_nan() {
        return sign | 0x7e00;
    }
    if magnitude.is_infinite() {
        return sign | 0x7c00;
    }
    if magnitude >= 65520.0 {
        return sign | 0x7bff;
    }
    if magnitude < 2.0f32.powi(-24) {
        return sign;
    }
    let exponent = magnitude.log2().floor() as i32;
    let mantissa = ((magnitude / 2f32.powi(exponent) - 1.0) * 1024.0).round() as i32;
    let mut half = ((exponent + 15) as u16) << 10 | mantissa as u16;
    if mantissa == 1024 {
        half = (((exponent + 16) as u16) << 10) | 0;
    }
    sign | half
}

/// The stride of a row major tensor.
fn contiguous_stride(shape: &[i64]) -> Vec<i64> {
    let mut stride = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    stride
}

/// Writes a string in the pickle format.
fn put_string(out: &mut Vec<u8>, text: &str) {
    // PyTorch writes the strings with `BINUNICODE`, and the reader of the
    // weights of PyTorch only knows that one.
    let bytes = text.as_bytes();
    out.push(0x58); // BINUNICODE
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

/// Writes a global name in the pickle format.
fn put_global(out: &mut Vec<u8>, module: &str, name: &str) {
    out.push(0x63); // GLOBAL
    out.extend_from_slice(module.as_bytes());
    out.push(b'\n');
    out.extend_from_slice(name.as_bytes());
    out.push(b'\n');
}

/// Writes an integer in the pickle format.
fn put_int(out: &mut Vec<u8>, value: i64) {
    if (0..256).contains(&value) {
        out.push(0x4b); // BININT1
        out.push(value as u8);
    } else {
        out.push(0x4a); // BININT
        out.extend_from_slice(&(value as i32).to_le_bytes());
    }
}

/// Writes a tuple of integers in the pickle format.
fn put_tuple(out: &mut Vec<u8>, values: &[i64]) {
    if values.len() < 256 {
        out.push(0x28); // MARK
        for value in values {
            put_int(out, *value);
        }
        out.push(0x74); // TUPLE
    } else {
        out.push(0x28);
        for value in values {
            put_int(out, *value);
        }
        out.push(0x74);
    }
}

// ───────── the tests ─────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A tensor of floats.
    fn floats(shape: Vec<i64>, data: Vec<f64>) -> Tensor {
        Tensor {
            dtype: DType::F32,
            shape,
            data,
            ints: vec![],
        }
    }

    /// A tensor of integers.
    fn integers(shape: Vec<i64>, data: Vec<i64>) -> Tensor {
        Tensor {
            dtype: DType::I64,
            shape,
            data: vec![],
            ints: data,
        }
    }

    #[test]
    fn reads_and_writes_a_checkpoint() {
        let dir = std::env::temp_dir().join(format!("mlr-pt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.pt");
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "w".to_string(),
            floats(vec![2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        );
        tensors.insert("b".to_string(), integers(vec![2], vec![7, 9]));
        write_pt(path.to_str().unwrap(), &tensors).unwrap();

        let back = read_checkpoint(path.to_str().unwrap()).unwrap();
        assert_eq!(back.len(), 2);
        let w = back.get("w").unwrap();
        assert_eq!(w.shape, vec![2, 3]);
        assert_eq!(w.as_f32(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = back.get("b").unwrap();
        assert_eq!(b.as_i32(), vec![7, 9]);
    }

    #[test]
    fn reads_a_safetensors_file() {
        // A file with one tensor of four floats.
        let mut header = String::from(
            "{\"w\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}",
        );
        // The header is padded with spaces to a multiple of eight.
        while header.len() % 8 != 0 {
            header.push(' ');
        }
        let mut data = Vec::new();
        data.extend_from_slice(&(header.len() as u64).to_le_bytes());
        data.extend_from_slice(header.as_bytes());
        for value in [1.0f32, 2.0, 3.0, 4.0] {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let tensors = read_safetensors(&data).unwrap();
        let w = tensors.get("w").unwrap();
        assert_eq!(w.shape, vec![2, 2]);
        assert_eq!(w.as_f32(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn reads_the_names_of_the_tensors() {
        let dir = std::env::temp_dir().join(format!("mlr-pt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("one.pt");
        let mut tensors = BTreeMap::new();
        tensors.insert("only".to_string(), floats(vec![2], vec![1.0, 2.0]));
        write_pt(path.to_str().unwrap(), &tensors).unwrap();
        let back = read_checkpoint(path.to_str().unwrap()).unwrap();
        assert_eq!(names(&back), vec!["only".to_string()]);
    }

    #[test]
    fn a_name_that_is_not_a_file_is_kept() {
        assert_eq!(resolve("model.bin").unwrap(), "model.bin");
    }
}
