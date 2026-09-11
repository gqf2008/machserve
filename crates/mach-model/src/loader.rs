//! Safetensors weight loading.
//!
//! Implements the safetensors binary format (8-byte little-endian header
//! length, JSON header, raw tensor data) and maps Llama/Qwen-style tensor
//! names onto the slice [`Weights`] layout. F32/F16/BF16 tensors are loaded
//! and converted to f32.

use crate::fp8::Fp8Tensor;
use crate::q4::Q4Tensor;
use crate::vision::{
    VisionCheckpointLayout, VisionConfig, VisionLayerWeights, VisionLinear, VisionWeights,
};
use crate::weights::{LayerWeightsFp8, LayerWeightsQ4, WeightsFp8, WeightsQ4};
use crate::{Config, Error, LayerWeights, Weights};
#[cfg(windows)]
use memmap2::Mmap;
use serde::de::DeserializeSeed;
use serde::de::{self, IgnoredAny, MapAccess, Visitor};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Safetensors headers contain tensor names/shapes/offsets only; 64 MiB is
/// already orders of magnitude larger than real checkpoints. The cap prevents
/// a corrupt length prefix from driving a multi-GB allocation before JSON
/// parsing can reject it.
const MAX_HEADER_BYTES: u64 = 64 << 20;

/// Summary returned by [`validate_checkpoint`] after reading only safetensors
/// headers. `payload_bytes` is the sum of tensor data spans, excluding file
/// headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointLayout {
    pub shards: usize,
    pub tensors: usize,
    pub payload_bytes: u64,
}

/// A decoded tensor from a safetensors file.
struct RawTensor {
    dtype: String,
    shape: Vec<usize>,
    /// Data offsets relative to the start of the data section.
    start: usize,
    end: usize,
}

/// Qwen3.5 VLM checkpoints namespace the text stack under
/// `model.language_model.*` (`Qwen3_5ForConditionalGeneration`); the loaders
/// and validation share this remap to the `model.*` layout.
fn normalize_tensor_name(name: &str) -> String {
    match name.strip_prefix("model.language_model.") {
        Some(rest) => format!("model.{rest}"),
        None => name.to_string(),
    }
}

#[derive(serde::Deserialize)]
struct RawTensorJson {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

struct SafetensorsHeaderVisitor {
    data_len: usize,
}

impl<'de> DeserializeSeed<'de> for SafetensorsHeaderVisitor {
    type Value = HashMap<String, RawTensor>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for SafetensorsHeaderVisitor {
    type Value = HashMap<String, RawTensor>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a safetensors header object")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut tensors = HashMap::new();
        while let Some(name) = map.next_key::<String>()? {
            if name == "__metadata__" {
                let _: IgnoredAny = map.next_value()?;
                continue;
            }
            let v: RawTensorJson = map.next_value()?;
            if v.data_offsets[0] > v.data_offsets[1] || v.data_offsets[1] > self.data_len as u64 {
                return Err(de::Error::custom(format!(
                    "tensor {name}: data_offsets {:?} out of bounds (data len {})",
                    v.data_offsets, self.data_len
                )));
            }
            let shape = v
                .shape
                .into_iter()
                .map(|d| {
                    usize::try_from(d)
                        .map_err(|_| de::Error::custom(format!("tensor {name}: shape too large")))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let name = normalize_tensor_name(&name);
            if tensors.contains_key(&name) {
                return Err(de::Error::custom(format!(
                    "duplicate tensor {name} after namespace normalization"
                )));
            }
            tensors.insert(
                name.clone(),
                RawTensor {
                    dtype: v.dtype,
                    shape,
                    start: usize::try_from(v.data_offsets[0]).map_err(|_| {
                        de::Error::custom(format!("tensor {name}: offset too large"))
                    })?,
                    end: usize::try_from(v.data_offsets[1]).map_err(|_| {
                        de::Error::custom(format!("tensor {name}: offset too large"))
                    })?,
                },
            );
        }
        Ok(tensors)
    }
}

/// Parse a safetensors header and validate every tensor range against the
/// shard's data-section length. No tensor payload is read. The visitor is
/// intentionally not `serde_json::Value`: duplicate JSON keys must be
/// rejected rather than silently overwritten by the map implementation.
fn parse_safetensors_header_bytes(
    header: &[u8],
    data_len: usize,
) -> Result<HashMap<String, RawTensor>, Error> {
    let mut de = serde_json::Deserializer::from_slice(header);
    let tensors = SafetensorsHeaderVisitor { data_len }
        .deserialize(&mut de)
        .map_err(|e| Error::Model(format!("bad JSON header: {e}")))?;
    de.end()
        .map_err(|e| Error::Model(format!("bad JSON header trailing data: {e}")))?;
    Ok(tensors)
}

/// Reads and parses only the 8-byte length prefix + JSON header of one shard.
fn read_safetensors_header(path: &Path) -> Result<(HashMap<String, RawTensor>, u64), Error> {
    let mut file = File::open(path).map_err(|e| Error::Model(format!("open {path:?}: {e}")))?;
    let file_len = file
        .metadata()
        .map_err(|e| Error::Model(format!("metadata {path:?}: {e}")))?
        .len();
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|e| Error::Model(format!("read {path:?} header prefix: {e}")))?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len > MAX_HEADER_BYTES {
        return Err(Error::Model(format!(
            "{path:?}: header length {header_len} exceeds the {MAX_HEADER_BYTES}-byte cap"
        )));
    }
    let data_start = 8u64
        .checked_add(header_len)
        .ok_or_else(|| Error::Model(format!("{path:?}: header length overflow")))?;
    if data_start > file_len {
        return Err(Error::Model(format!(
            "{path:?}: header length {header_len} exceeds file size {file_len}"
        )));
    }
    let header_len = usize::try_from(header_len)
        .map_err(|_| Error::Model(format!("{path:?}: header too large")))?;
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|e| Error::Model(format!("read {path:?} header: {e}")))?;
    let data_bytes = file_len - data_start;
    let data_len = usize::try_from(data_bytes)
        .map_err(|_| Error::Model(format!("{path:?}: data section too large")))?;
    let tensors = parse_safetensors_header_bytes(&header_bytes, data_len)?;
    Ok((tensors, data_bytes))
}

/// Open a checkpoint file without granting concurrent writers on Windows.
/// `FILE_SHARE_READ` denies write/delete sharing for this handle, matching the
/// immutability precondition of the read-only memory map.
#[cfg(windows)]
fn open_safetensors_readonly(path: &Path) -> Result<File, Error> {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_SHARE_READ: u32 = 0x0000_0001;
    std::fs::OpenOptions::new()
        .read(true)
        .share_mode(FILE_SHARE_READ)
        .open(path)
        .map_err(|e| Error::Model(format!("open {path:?} without write sharing: {e}")))
}

/// Parsed safetensors shard storage. On Windows payload is memory-mapped and
/// faulted on demand; on other platforms it falls back to an owned `Vec<u8>`
/// because safe, mandatory no-writer file locking is not portable.
#[cfg(windows)]
type ParsedStorage = Mmap;
#[cfg(not(windows))]
type ParsedStorage = Vec<u8>;

struct ParsedSafetensors {
    tensors: HashMap<String, RawTensor>,
    storage: ParsedStorage,
    data_start: usize,
}

impl ParsedSafetensors {
    fn data(&self) -> &[u8] {
        &self.storage[self.data_start..]
    }
}

/// Open and parse a safetensors file. Windows uses a read-only mmap with
/// write/delete sharing denied; other platforms fall back to an owned buffer.
/// Callers only borrow tensor ranges while encoding/quantizing weights.
fn parse_safetensors(path: &Path) -> Result<ParsedSafetensors, Error> {
    #[cfg(windows)]
    {
        let file = open_safetensors_readonly(path)?;
        let file_len = file
            .metadata()
            .map_err(|e| Error::Model(format!("metadata {path:?}: {e}")))?
            .len();
        if file_len < 8 {
            return Err(Error::Model(format!("{path:?}: file too short")));
        }
        // SAFETY: the Windows handle denies write/delete sharing for the
        // mapping lifetime; the mapping itself is read-only and never exposed
        // mutably.
        let map =
            unsafe { Mmap::map(&file) }.map_err(|e| Error::Model(format!("mmap {path:?}: {e}")))?;
        let map_len = map.len();
        if map_len < 8 {
            return Err(Error::Model(format!(
                "{path:?}: file shrank below the safetensors prefix while mapping"
            )));
        }
        let header_len = u64::from_le_bytes(map[0..8].try_into().unwrap());
        if header_len > MAX_HEADER_BYTES {
            return Err(Error::Model(format!(
                "{path:?}: header length {header_len} exceeds the {MAX_HEADER_BYTES}-byte cap"
            )));
        }
        let data_start = 8u64
            .checked_add(header_len)
            .ok_or_else(|| Error::Model(format!("{path:?}: header length overflow")))?;
        if data_start > map_len as u64 {
            return Err(Error::Model(format!(
                "{path:?}: header length out of range for mapped length {map_len}"
            )));
        }
        let data_start = usize::try_from(data_start)
            .map_err(|_| Error::Model(format!("{path:?}: header too large")))?;
        let data_len = map_len
            .checked_sub(data_start)
            .ok_or_else(|| Error::Model(format!("{path:?}: mapped data length underflow")))?;
        let tensors = parse_safetensors_header_bytes(&map[8..data_start], data_len)?;
        Ok(ParsedSafetensors {
            tensors,
            storage: map,
            data_start,
        })
    }
    #[cfg(not(windows))]
    {
        let (tensors, data) = parse_safetensors_owned(path)?;
        Ok(ParsedSafetensors {
            tensors,
            storage: data,
            data_start: 0,
        })
    }
}

/// Owned-buffer parser used by the f32/f16 multi-shard concatenation path and
/// as the non-Windows fallback for the single-file/Q4/FP8 parser. On Windows
/// those non-sharded paths use a read-only mmap instead.
fn parse_safetensors_owned(path: &Path) -> Result<(HashMap<String, RawTensor>, Vec<u8>), Error> {
    let mut bytes = std::fs::read(path).map_err(|e| Error::Model(format!("read {path:?}: {e}")))?;
    if bytes.len() < 8 {
        return Err(Error::Model(format!("{path:?}: file too short")));
    }
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    if header_len > MAX_HEADER_BYTES {
        return Err(Error::Model(format!(
            "{path:?}: header length {header_len} exceeds the {MAX_HEADER_BYTES}-byte cap"
        )));
    }
    let data_start = 8u64
        .checked_add(header_len)
        .ok_or_else(|| Error::Model(format!("{path:?}: header length overflow")))?;
    if data_start > bytes.len() as u64 {
        return Err(Error::Model(format!(
            "{path:?}: header length out of range"
        )));
    }
    let header_len = usize::try_from(header_len)
        .map_err(|_| Error::Model(format!("{path:?}: header too large")))?;
    let tensors =
        parse_safetensors_header_bytes(&bytes[8..8 + header_len], bytes.len() - 8 - header_len)?;
    let data = bytes.split_off(8 + header_len);
    Ok((tensors, data))
}

/// Parsed HF shard index used by [`validate_checkpoint`].
struct CheckpointIndex {
    tensors: HashMap<String, String>,
    total_size: Option<u64>,
}

/// Read a Hugging Face `model.safetensors.index.json`, if present.
///
/// Returns `(tensor -> basename, metadata.total_size?)`. Paths in the index are
/// deliberately restricted to a single sibling filename: a malformed index
/// must not make validation escape the checkpoint directory.
fn read_checkpoint_index(dir: &Path) -> Result<Option<CheckpointIndex>, Error> {
    let path = dir.join("model.safetensors.index.json");
    if !path.is_file() {
        return Ok(None);
    }
    let text =
        std::fs::read_to_string(&path).map_err(|e| Error::Model(format!("read {path:?}: {e}")))?;
    let value: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| Error::Model(format!("parse {path:?}: {e}")))?;
    let map = value["weight_map"]
        .as_object()
        .ok_or_else(|| Error::Model(format!("{path:?}: weight_map missing or not an object")))?;
    let mut out = HashMap::with_capacity(map.len());
    for (name, file) in map {
        let file = file
            .as_str()
            .ok_or_else(|| Error::Model(format!("{path:?}: weight_map[{name}] not a string")))?;
        let rel = Path::new(file);
        if rel.is_absolute() || rel.components().count() != 1 {
            return Err(Error::Model(format!(
                "{path:?}: weight_map[{name}] must name a sibling shard, got {file:?}"
            )));
        }
        if rel.extension().is_none_or(|x| x != "safetensors") {
            return Err(Error::Model(format!(
                "{path:?}: weight_map[{name}] is not a .safetensors file: {file:?}"
            )));
        }
        let shard = dir.join(rel);
        if !shard.is_file() {
            return Err(Error::Model(format!(
                "{path:?}: indexed shard {file:?} is missing"
            )));
        }
        let name = normalize_tensor_name(name);
        if out.insert(name.clone(), file.to_string()).is_some() {
            return Err(Error::Model(format!(
                "{path:?}: duplicate tensor {name} after namespace normalization"
            )));
        }
    }
    let total = value["metadata"]["total_size"]
        .as_f64()
        .map(|n| {
            if n.fract() == 0.0 && n >= 0.0 {
                Ok(n as u64)
            } else {
                Err(Error::Model(format!(
                    "{path:?}: metadata.total_size is not a non-negative integer"
                )))
            }
        })
        .transpose()?;
    Ok(Some(CheckpointIndex {
        tensors: out,
        total_size: total,
    }))
}

/// Validate a checkpoint using only safetensors headers and the optional HF
/// shard index. This catches missing shards, truncated payloads, out-of-range
/// tensor offsets, duplicate tensors, stale index entries and index/tensor
/// placement mismatches before any weight allocation or GPU upload.
pub fn validate_checkpoint(path: &Path) -> Result<CheckpointLayout, Error> {
    let dir = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    };
    // A standalone file can share a models root with an unrelated or even
    // malformed HF index. The index only governs directory checkpoints; file
    // inputs are validated on their own.
    let index = if path.is_dir() {
        read_checkpoint_index(&dir)?
    } else {
        None
    };
    if let Some(index) = &index {
        let expected: HashSet<&str> = index.tensors.values().map(String::as_str).collect();
        let actual: HashSet<String> = std::fs::read_dir(&dir)
            .map_err(|e| Error::Model(format!("read dir {dir:?}: {e}")))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .filter_map(|p| p.file_name().and_then(|x| x.to_str()).map(str::to_string))
            .collect();
        let mut extra: Vec<&str> = actual
            .iter()
            .map(String::as_str)
            .filter(|name| !expected.contains(name))
            .collect();
        if !extra.is_empty() {
            extra.sort_unstable();
            return Err(Error::Model(format!(
                "{dir:?}: unindexed .safetensors files present: {}",
                extra.join(", ")
            )));
        }
    }
    let files: Vec<PathBuf> = if let Some(index) = &index {
        let mut names: Vec<&str> = index.tensors.values().map(String::as_str).collect();
        names.sort_unstable();
        names.dedup();
        names.into_iter().map(|name| dir.join(name)).collect()
    } else if path.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(path)
            .map_err(|e| Error::Model(format!("read dir {path:?}: {e}")))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        files
    } else {
        vec![path.to_path_buf()]
    };
    if files.is_empty() {
        return Err(Error::Model(format!("no .safetensors files in {path:?}")));
    }

    let expected = index.as_ref().map(|index| &index.tensors);
    let mut seen: HashMap<String, String> = HashMap::new();
    let mut tensors = 0usize;
    let mut payload_bytes = 0u64;
    for file in &files {
        let (shard_tensors, _) = read_safetensors_header(file)?;
        let file_name = file
            .file_name()
            .and_then(|x| x.to_str())
            .ok_or_else(|| Error::Model(format!("{file:?}: non-Unicode shard filename")))?;
        for (name, tensor) in shard_tensors {
            if let Some(map) = expected {
                let want = map.get(&name).ok_or_else(|| {
                    Error::Model(format!("{file:?}: tensor {name} is absent from the index"))
                })?;
                if want != file_name {
                    return Err(Error::Model(format!(
                        "{file:?}: tensor {name} is indexed in {want:?}"
                    )));
                }
            }
            if let Some(previous) = seen.insert(name.clone(), file_name.to_string()) {
                return Err(Error::Model(format!(
                    "tensor {name} appears in both {previous} and {file_name}"
                )));
            }
            tensors = tensors
                .checked_add(1)
                .ok_or_else(|| Error::Model("tensor count overflow".into()))?;
            payload_bytes = payload_bytes
                .checked_add((tensor.end - tensor.start) as u64)
                .ok_or_else(|| Error::Model("payload byte count overflow".into()))?;
        }
    }
    if let Some(index) = index {
        if let Some(missing) = index.tensors.keys().find(|name| !seen.contains_key(*name)) {
            return Err(Error::Model(format!(
                "indexed tensor {missing} was not found in any shard"
            )));
        }
        if let Some(total) = index.total_size
            && payload_bytes != total
        {
            return Err(Error::Model(format!(
                "index metadata.total_size={total}, but shard headers describe {payload_bytes} bytes"
            )));
        }
    }
    Ok(CheckpointLayout {
        shards: files.len(),
        tensors,
        payload_bytes,
    })
}

/// Validate a Qwen3.5/Qwen3.8 vision tower from safetensors headers only.
///
/// Every expected `model.visual.*` tensor must be present exactly once with
/// the configured shape and a payload length matching its dtype. Text and
/// auxiliary (`mtp.*`) tensors are ignored; unexpected visual tensors fail
/// loudly so a renamed/misclassified tower cannot silently run with missing
/// weights.
pub fn validate_vision_checkpoint(
    path: &Path,
    cfg: &VisionConfig,
) -> Result<VisionCheckpointLayout, Error> {
    cfg.validate()?;
    let checkpoint = validate_checkpoint(path)?;
    let expected: HashMap<String, Vec<usize>> = cfg.expected_tensors().into_iter().collect();
    let files: Vec<PathBuf> = if path.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(path)
            .map_err(|e| Error::Model(format!("read dir {path:?}: {e}")))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        files
    } else {
        vec![path.to_path_buf()]
    };
    if files.is_empty() {
        return Err(Error::Model(format!("no .safetensors files in {path:?}")));
    }

    let mut seen: HashMap<String, Vec<usize>> = HashMap::new();
    let mut payload_bytes = 0u64;
    for file in &files {
        let (tensors, _) = read_safetensors_header(file)?;
        for (name, tensor) in tensors {
            if !name.starts_with("model.visual.") {
                continue;
            }
            if let Some(previous) = seen.insert(name.clone(), tensor.shape.clone()) {
                return Err(Error::Model(format!(
                    "{file:?}: visual tensor {name} appears more than once (previous shape {previous:?})"
                )));
            }
            let want = expected.get(&name).ok_or_else(|| {
                Error::Model(format!("{file:?}: unexpected visual tensor {name}"))
            })?;
            if &tensor.shape != want {
                return Err(Error::Model(format!(
                    "{file:?}: visual tensor {name} has shape {:?}, expected {want:?}",
                    tensor.shape
                )));
            }
            let elem_bytes = match tensor.dtype.as_str() {
                "F32" => 4usize,
                "F16" | "BF16" => 2usize,
                other => {
                    return Err(Error::Model(format!(
                        "{file:?}: visual tensor {name} has unsupported dtype {other}"
                    )));
                }
            };
            let elems = tensor
                .shape
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| Error::Model(format!("{name}: element count overflow")))?;
            let want_span = elems
                .checked_mul(elem_bytes)
                .ok_or_else(|| Error::Model(format!("{name}: byte span overflow")))?;
            let got_span = tensor.end - tensor.start;
            if got_span != want_span {
                return Err(Error::Model(format!(
                    "{file:?}: visual tensor {name} has {got_span} payload bytes, expected {want_span}"
                )));
            }
            payload_bytes = payload_bytes
                .checked_add(got_span as u64)
                .ok_or_else(|| Error::Model("vision payload byte count overflow".into()))?;
        }
    }

    let mut missing: Vec<&str> = expected
        .keys()
        .filter(|name| !seen.contains_key(*name))
        .map(String::as_str)
        .collect();
    if !missing.is_empty() {
        missing.sort_unstable();
        return Err(Error::Model(format!(
            "{} vision tensor(s) missing; first: {}",
            missing.len(),
            missing[0]
        )));
    }
    Ok(VisionCheckpointLayout {
        shards: checkpoint.shards,
        tensors: seen.len(),
        payload_bytes,
    })
}
/// Load the Qwen3.5/Qwen3.8 vision tower into host f32 weights.
///
/// The checkpoint is streamed shard by shard; only the selected visual
/// tensors are converted, so the text/LLM tensors and `mtp.*` stack are never
/// materialized by this path.
pub fn load_vision_weights(path: &Path, cfg: &VisionConfig) -> Result<VisionWeights, Error> {
    cfg.validate()?;
    let expected: HashMap<String, Vec<usize>> = cfg.expected_tensors().into_iter().collect();
    let mut files: Vec<PathBuf> = if path.is_dir() {
        std::fs::read_dir(path)
            .map_err(|e| Error::Model(format!("read dir {path:?}: {e}")))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect()
    } else {
        vec![path.to_path_buf()]
    };
    files.sort();
    if files.is_empty() {
        return Err(Error::Model(format!("no .safetensors files in {path:?}")));
    }

    let mut tensors: HashMap<String, Vec<f32>> = HashMap::with_capacity(expected.len());
    for file in &files {
        let parsed = parse_safetensors(file)?;
        for (name, tensor) in &parsed.tensors {
            if !name.starts_with("model.visual.") {
                continue;
            }
            let shape = expected.get(name).ok_or_else(|| {
                Error::Model(format!("{file:?}: unexpected visual tensor {name}"))
            })?;
            if &tensor.shape != shape {
                return Err(Error::Model(format!(
                    "{file:?}: visual tensor {name} has shape {:?}, expected {shape:?}",
                    tensor.shape
                )));
            }
            let elems = shape
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| Error::Model(format!("{name}: element count overflow")))?;
            let values = tensor_f32(parsed.data(), tensor, elems, name)?;
            if tensors.insert(name.clone(), values).is_some() {
                return Err(Error::Model(format!(
                    "{file:?}: visual tensor {name} appears more than once"
                )));
            }
        }
    }

    let take = |map: &mut HashMap<String, Vec<f32>>, suffix: &str| -> Result<Vec<f32>, Error> {
        map.remove(suffix)
            .ok_or_else(|| Error::Model(format!("vision tensor {suffix} is missing")))
    };
    let take_linear =
        |map: &mut HashMap<String, Vec<f32>>, prefix: &str| -> Result<VisionLinear, Error> {
            Ok(VisionLinear {
                weight: take(map, &format!("{prefix}.weight"))?,
                bias: take(map, &format!("{prefix}.bias"))?,
            })
        };

    let patch_embed_weight = take(&mut tensors, "model.visual.patch_embed.proj.weight")?;
    let patch_embed_bias = take(&mut tensors, "model.visual.patch_embed.proj.bias")?;
    let pos_embed_weight = take(&mut tensors, "model.visual.pos_embed.weight")?;
    let mut layers = Vec::with_capacity(cfg.depth);
    for li in 0..cfg.depth {
        let p = |suffix: &str| format!("model.visual.blocks.{li}.{suffix}");
        layers.push(VisionLayerWeights {
            norm1_weight: take(&mut tensors, &p("norm1.weight"))?,
            norm1_bias: take(&mut tensors, &p("norm1.bias"))?,
            qkv: take_linear(&mut tensors, &p("attn.qkv"))?,
            attn_proj: take_linear(&mut tensors, &p("attn.proj"))?,
            norm2_weight: take(&mut tensors, &p("norm2.weight"))?,
            norm2_bias: take(&mut tensors, &p("norm2.bias"))?,
            mlp_fc1: take_linear(&mut tensors, &p("mlp.linear_fc1"))?,
            mlp_fc2: take_linear(&mut tensors, &p("mlp.linear_fc2"))?,
        });
    }
    let merger_norm_weight = take(&mut tensors, "model.visual.merger.norm.weight")?;
    let merger_norm_bias = take(&mut tensors, "model.visual.merger.norm.bias")?;
    let merger_fc1 = take_linear(&mut tensors, "model.visual.merger.linear_fc1")?;
    let merger_fc2 = take_linear(&mut tensors, "model.visual.merger.linear_fc2")?;
    if let Some(name) = tensors.keys().next() {
        return Err(Error::Model(format!("unconsumed visual tensor {name}")));
    }

    Ok(VisionWeights {
        patch_embed_weight,
        patch_embed_bias,
        pos_embed_weight,
        layers,
        merger_norm_weight,
        merger_norm_bias,
        merger_fc1,
        merger_fc2,
    })
}
/// Loads a tensor and converts it to f32 `[out, in]` row-major.
fn tensor_f32(data: &[u8], t: &RawTensor, expected: usize, name: &str) -> Result<Vec<f32>, Error> {
    let n: usize = t.shape.iter().product();
    if n != expected {
        return Err(Error::Model(format!(
            "{name}: shape {:?} has {n} elems, expected {expected}",
            t.shape
        )));
    }
    let span = t.end - t.start;
    let bytes = data
        .get(t.start..t.end)
        .ok_or_else(|| Error::Model(format!("{name}: data range out of bounds")))?;
    let mut out = Vec::with_capacity(n);
    match t.dtype.as_str() {
        "F32" => {
            if span != n * 4 {
                return Err(Error::Model(format!("{name}: F32 size mismatch")));
            }
            for i in 0..n {
                out.push(f32::from_le_bytes(
                    bytes[i * 4..i * 4 + 4].try_into().unwrap(),
                ));
            }
        }
        "F16" => {
            if span != n * 2 {
                return Err(Error::Model(format!("{name}: F16 size mismatch")));
            }
            for i in 0..n {
                let u = u16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap());
                out.push(f16_to_f32(u));
            }
        }
        "BF16" => {
            if span != n * 2 {
                return Err(Error::Model(format!("{name}: BF16 size mismatch")));
            }
            for i in 0..n {
                let u = u16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap());
                out.push(bf16_to_f32(u));
            }
        }
        other => return Err(Error::Model(format!("{name}: unsupported dtype {other}"))),
    }
    Ok(out)
}

/// Parallel f32 conversion for large tensors: splits the element range across
/// threads (each converts its byte slice to f32), concatenating in order.
/// Bit-equal to [`tensor_f32`]. Falls back to the scalar path below ~1M
/// elements (thread spawn overhead is not worth it for small tensors).
fn tensor_f32_par(
    data: &[u8],
    t: &RawTensor,
    expected: usize,
    name: &str,
) -> Result<Vec<f32>, Error> {
    let n: usize = t.shape.iter().product();
    if n != expected {
        return Err(Error::Model(format!(
            "{name}: shape {:?} has {n} elems, expected {expected}",
            t.shape
        )));
    }
    if n < (1 << 20) {
        return tensor_f32(data, t, expected, name);
    }
    let elem = match t.dtype.as_str() {
        "F32" => 4usize,
        "F16" | "BF16" => 2usize,
        other => return Err(Error::Model(format!("{name}: unsupported dtype {other}"))),
    };
    let span = t.end - t.start;
    if span != n * elem {
        return Err(Error::Model(format!("{name}: size mismatch")));
    }
    let bytes = data
        .get(t.start..t.end)
        .ok_or_else(|| Error::Model(format!("{name}: data range out of bounds")))?;
    let n_threads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(4)
        .min(16);
    let chunk = n.div_ceil(n_threads);
    let dtype = t.dtype.as_str();
    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_threads);
        for th in 0..n_threads {
            let e0 = th * chunk;
            let e1 = (e0 + chunk).min(n);
            if e0 >= e1 {
                continue;
            }
            handles.push(s.spawn(move || {
                let mut local = Vec::with_capacity(e1 - e0);
                match dtype {
                    "F32" => {
                        for i in e0..e1 {
                            let off = i * 4;
                            local.push(f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
                        }
                    }
                    "F16" => {
                        for i in e0..e1 {
                            let off = i * 2;
                            let u = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap());
                            local.push(f16_to_f32(u));
                        }
                    }
                    "BF16" => {
                        for i in e0..e1 {
                            let off = i * 2;
                            let u = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap());
                            local.push(bf16_to_f32(u));
                        }
                    }
                    other => {
                        return Err(Error::Model(format!("{name}: unsupported dtype {other}")));
                    }
                }
                Ok(local)
            }));
        }
        let mut merged = Vec::with_capacity(n);
        for h in handles {
            merged.extend_from_slice(&h.join().unwrap()?);
        }
        Ok(merged)
    })
}

/// Half-precision float to f32.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign << 31
        } else {
            // subnormal
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | ((e as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (man << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// BF16 to f32 (zero-extend).
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Loads a single safetensors checkpoint into [`Weights`].
pub fn load_safetensors(path: &Path, cfg: &Config, tie_embeddings: bool) -> Result<Weights, Error> {
    let parsed = parse_safetensors(path)?;
    build_weights(&parsed.tensors, parsed.data(), cfg, tie_embeddings)
}

/// Loads every `*.safetensors` shard in `path` and merges them into one
/// [`Weights`] (Qwen-8B+ checkpoints ship as 5..65 shards; per-file tensor
/// offsets are rebased onto the concatenated data section).
pub fn load_safetensors_dir(
    path: &Path,
    cfg: &Config,
    tie_embeddings: bool,
) -> Result<Weights, Error> {
    // Accept either a single .safetensors file or a directory of shards.
    let mut files: Vec<std::path::PathBuf> = if path.is_dir() {
        std::fs::read_dir(path)
            .map_err(|e| Error::Model(format!("read dir {path:?}: {e}")))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect()
    } else {
        vec![path.to_path_buf()]
    };
    files.sort();
    if files.is_empty() {
        return Err(Error::Model(format!("no .safetensors files in {path:?}")));
    }
    let mut tensors = HashMap::new();
    let mut data = Vec::new();
    for f in &files {
        let (mut shard_tensors, shard_data) = parse_safetensors_owned(f)?;
        let base = data.len();
        for rt in shard_tensors.values_mut() {
            rt.start += base;
            rt.end += base;
        }
        tensors.extend(shard_tensors);
        data.extend(shard_data);
    }
    build_weights(&tensors, &data, cfg, tie_embeddings)
}

/// Qwen3 QK-norm weight: real checkpoints store ONE `[head_dim]` vector
/// shared across all heads; older test checkpoints may store per-head
/// `[n_heads * head_dim]`. Accept both, tiling the shared form to per-head.
fn load_qk_norm(
    tensors: &HashMap<String, RawTensor>,
    data: &[u8],
    name: &str,
    per_head: usize,
    head_dim: usize,
) -> Result<Vec<f32>, Error> {
    let Some(t) = tensors.get(name) else {
        return Ok(Vec::new());
    };
    let n: usize = t.shape.iter().product();
    if n == per_head {
        return tensor_f32(data, t, per_head, name);
    }
    if n == head_dim {
        let shared = tensor_f32(data, t, head_dim, name)?;
        let mut v = Vec::with_capacity(per_head);
        for _ in 0..(per_head / head_dim) {
            v.extend_from_slice(&shared);
        }
        return Ok(v);
    }
    Err(Error::Model(format!("{name}: unexpected QK-norm size {n}")))
}

/// Qwen3 ships q_norm/k_norm as one SHARED `[head_dim]` vector, while the
/// QK-norm kernel indexes per head (`[n_heads, head_dim]`). The f32 loader
/// broadcasts via `load_qk_norm`; the Q4/FP8 loaders call this after the
/// layer is built so both storage paths match the kernel contract.
fn broadcast_qk_norm(q_norm: &mut Vec<f32>, k_norm: &mut Vec<f32>, cfg: &Config) {
    let hd = cfg.head_dim;
    if q_norm.len() == hd && cfg.n_heads > 1 {
        let shared = q_norm.clone();
        for _ in 1..cfg.n_heads {
            q_norm.extend_from_slice(&shared);
        }
    }
    if k_norm.len() == hd && cfg.n_kv_heads > 1 {
        let shared = k_norm.clone();
        for _ in 1..cfg.n_kv_heads {
            k_norm.extend_from_slice(&shared);
        }
    }
}

/// Splits a fused MLA q projection `[n_heads * (nope + rope), kk]` into the
/// per-head non-RoPE and RoPE row blocks.
///
/// Real DeepSeek checkpoints ship ONE fused matrix per layer and split it per
/// head *after* the projection: `q_proj [heads*(nope+rope), d]` when
/// `q_lora_rank == 0` (DeepSeek-V2-Lite) and `q_b_proj [heads*(nope+rope),
/// q_lora_rank]` when the low-rank q path is used (DeepSeek-V2 236B). The
/// runtime keeps the halves separate so RoPE touches only the second half.
fn split_mla_q(
    fused: &[f32],
    heads: usize,
    nope: usize,
    rope: usize,
    kk: usize,
) -> (Vec<f32>, Vec<f32>) {
    let per_head = nope + rope;
    assert_eq!(
        fused.len(),
        heads * per_head * kk,
        "fused MLA q: expected [{}x{}], got {}",
        heads * per_head,
        kk,
        fused.len()
    );
    let mut nope_w = Vec::with_capacity(heads * nope * kk);
    let mut rope_w = Vec::with_capacity(heads * rope * kk);
    for h in 0..heads {
        let base = h * per_head * kk;
        nope_w.extend_from_slice(&fused[base..base + nope * kk]);
        rope_w.extend_from_slice(&fused[base + nope * kk..base + per_head * kk]);
    }
    (nope_w, rope_w)
}

/// Builds [`Weights`] from a merged tensor map (single file or shards).
fn build_weights(
    tensors: &HashMap<String, RawTensor>,
    data: &[u8],
    cfg: &Config,
    tie_embeddings: bool,
) -> Result<Weights, Error> {
    let d = cfg.d_model;
    let nq = cfg.n_heads * cfg.head_dim;
    // Qwen3.5 `attn_output_gate` doubles the q_proj width (per-head gate in
    // the second half of each head's block).
    let nq_load = if cfg.attn_output_gate { nq * 2 } else { nq };
    let nkv = cfg.n_kv_heads * cfg.head_dim;

    // Like `get`, but returns an empty vector when the tensor is absent
    // (optional biases).
    let get_opt = |name: &str, expected: usize| -> Result<Vec<f32>, Error> {
        if !tensors.contains_key(name) {
            return Ok(Vec::new());
        }
        tensor_f32(data, &tensors[name], expected, name)
    };
    let get = |name: &str, expected: usize| -> Result<Vec<f32>, Error> {
        let t = tensors
            .get(name)
            .ok_or_else(|| Error::Model(format!("missing tensor {name}")))?;
        tensor_f32(data, t, expected, name)
    };

    let tok_emb = get("model.embed_tokens.weight", cfg.vocab_size * d)?;
    let mut rms_final = get("model.norm.weight", d)?;
    let lm_head = match tensors.get("lm_head.weight") {
        Some(_) => get("lm_head.weight", cfg.vocab_size * d)?,
        None if tie_embeddings => tok_emb.clone(),
        None => {
            return Err(Error::Model(
                "lm_head.weight missing and tie_embeddings=false".into(),
            ));
        }
    };

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let p = |suffix: &str| format!("model.layers.{i}.{suffix}");
        // Per-layer MoE detection: Qwen-MoE checkpoints mix dense layers
        // (`mlp_only_layers`, e.g. Qwen3-MoE) with routed-expert layers. A layer
        // is MoE iff it carries a router tensor (`mlp.gate.weight`).
        let is_moe = cfg.num_experts > 0 && tensors.contains_key(&p("mlp.gate.weight"));
        let einter = cfg.expert_size();
        let (moe_router, moe_wg, moe_wu, moe_wd) = if is_moe {
            let ne = cfg.num_experts;
            let router = get(&p("mlp.gate.weight"), ne * d)?;
            let mut wg = Vec::with_capacity(ne * einter * d);
            let mut wu = Vec::with_capacity(ne * einter * d);
            let mut wd = Vec::with_capacity(ne * d * einter);
            for e in 0..ne {
                let ep = |s: &str| format!("model.layers.{i}.mlp.experts.{e}.{s}");
                wg.extend(get(&ep("gate_proj.weight"), einter * d)?);
                wu.extend(get(&ep("up_proj.weight"), einter * d)?);
                wd.extend(get(&ep("down_proj.weight"), d * einter)?);
            }
            (router, wg, wu, wd)
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };
        // MLA (DeepSeek-V2 style): compressed KV + low-rank Q replace the
        // standard q/k/v/o projections when kv_lora_rank > 0.
        let mla = cfg.kv_lora_rank > 0;
        // Qwen3.5 hybrid: linear-attention layers carry `linear_attn.*`
        // tensors instead of `self_attn.*`. Which layers are linear is a pure
        // config function (`layer_is_full_attn`); tensor emptiness doubles as
        // the per-layer dispatch flag downstream.
        let gdn = cfg.gdn_enabled() && !cfg.layer_is_full_attn(i);
        let gdn_kd = cfg.gdn_key_dim();
        let gdn_vd = cfg.gdn_value_dim();
        let gdn_conv_dim = 2 * gdn_kd + gdn_vd;
        let (
            gdn_in_qkv,
            gdn_in_z,
            gdn_in_a,
            gdn_in_b,
            gdn_conv_w,
            gdn_a_log,
            gdn_dt_bias,
            gdn_norm,
            gdn_out,
        ) = if gdn {
            (
                get(&p("linear_attn.in_proj_qkv.weight"), gdn_conv_dim * d)?,
                get(&p("linear_attn.in_proj_z.weight"), gdn_vd * d)?,
                get(&p("linear_attn.in_proj_a.weight"), cfg.gdn_v_heads * d)?,
                get(&p("linear_attn.in_proj_b.weight"), cfg.gdn_v_heads * d)?,
                get(
                    &p("linear_attn.conv1d.weight"),
                    gdn_conv_dim * cfg.gdn_conv_kernel,
                )?,
                get(&p("linear_attn.A_log"), cfg.gdn_v_heads)?,
                get(&p("linear_attn.dt_bias"), cfg.gdn_v_heads)?,
                get(&p("linear_attn.norm.weight"), cfg.gdn_head_dim)?,
                get(&p("linear_attn.out_proj.weight"), d * gdn_vd)?,
            )
        } else {
            (
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        };
        // Shared experts (DeepSeek-V2): a dense SwiGLU MLP of width
        // `n_shared_experts * expert_size` on routed layers, added to the
        // routed experts' output. Dense layers never have one.
        let shinter = cfg.shared_size();
        let (shared_wg, shared_wu, shared_wd) = if is_moe && shinter > 0 {
            let sp = |proj: &str| format!("model.layers.{i}.mlp.shared_experts.{proj}");
            (
                get(&sp("gate_proj.weight"), shinter * d)?,
                get(&sp("up_proj.weight"), shinter * d)?,
                get(&sp("down_proj.weight"), d * shinter)?,
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        let (mla_q_a, mla_q_a_norm, mla_q_b, mla_q_rope, mla_kv_a, mla_kv_a_norm, mla_kv_b, mla_o) =
            if mla {
                // One fused q projection, split per head into the non-RoPE and
                // RoPE halves (see [`split_mla_q`]). `q_lora_rank > 0`
                // (DeepSeek-V2 236B) reads `q_b_proj` and contracts over the
                // normalized q_lora; `q_lora_rank == 0` (V2-Lite, whose config
                // ships `q_lora_rank: null`) reads `q_proj` and contracts over
                // the layer input.
                let kk = if cfg.q_lora_rank > 0 {
                    cfg.q_lora_rank
                } else {
                    d
                };
                let fused_n = cfg.n_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) * kk;
                let fused = if cfg.q_lora_rank > 0 {
                    get(&p("self_attn.q_b_proj.weight"), fused_n)?
                } else {
                    get(&p("self_attn.q_proj.weight"), fused_n)?
                };
                let (q_nope_w, q_rope_w) = split_mla_q(
                    &fused,
                    cfg.n_heads,
                    cfg.qk_nope_head_dim,
                    cfg.qk_rope_head_dim,
                    kk,
                );
                // The low-rank q path (`q_a` + `q_a_layernorm`) exists only
                // when `q_lora_rank > 0`.
                let (q_a, q_a_norm) = if cfg.q_lora_rank > 0 {
                    (
                        get(&p("self_attn.q_a_proj.weight"), cfg.q_lora_rank * d)?,
                        get(&p("self_attn.q_a_layernorm.weight"), cfg.q_lora_rank)?,
                    )
                } else {
                    (Vec::new(), Vec::new())
                };
                (
                    q_a,
                    q_a_norm,
                    q_nope_w,
                    q_rope_w,
                    get(
                        &p("self_attn.kv_a_proj_with_mqa.weight"),
                        (cfg.kv_lora_rank + cfg.qk_rope_head_dim) * d,
                    )?,
                    get(&p("self_attn.kv_a_layernorm.weight"), cfg.kv_lora_rank)?,
                    get(
                        &p("self_attn.kv_b_proj.weight"),
                        cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim) * cfg.kv_lora_rank,
                    )?,
                    get(
                        &p("self_attn.o_proj.weight"),
                        d * cfg.n_heads * cfg.v_head_dim,
                    )?,
                )
            } else {
                (
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                )
            };
        let lw = LayerWeights {
            wq: if mla || gdn {
                Vec::new()
            } else {
                get(&p("self_attn.q_proj.weight"), d * nq_load)?
            },
            wk: if mla || gdn {
                Vec::new()
            } else {
                get(&p("self_attn.k_proj.weight"), d * nkv)?
            },
            wv: if mla || gdn {
                Vec::new()
            } else {
                get(&p("self_attn.v_proj.weight"), d * nkv)?
            },
            wo: if mla || gdn {
                Vec::new()
            } else {
                get(&p("self_attn.o_proj.weight"), nq * d)?
            },
            rms_attn: get(&p("input_layernorm.weight"), d)?,
            wg: if is_moe {
                Vec::new()
            } else {
                get(&p("mlp.gate_proj.weight"), cfg.intermediate_size * d)?
            },
            wu: if is_moe {
                Vec::new()
            } else {
                get(&p("mlp.up_proj.weight"), cfg.intermediate_size * d)?
            },
            wd: if is_moe {
                Vec::new()
            } else {
                get(&p("mlp.down_proj.weight"), d * cfg.intermediate_size)?
            },
            rms_mlp: get(&p("post_attention_layernorm.weight"), d)?,
            // Qwen2 checkpoints ship q/k/v biases (even with
            // `attention_bias: false`); default to empty (no bias) when absent.
            bq: get_opt(&p("self_attn.q_proj.bias"), nq)?,
            bk: get_opt(&p("self_attn.k_proj.bias"), nkv)?,
            bv: get_opt(&p("self_attn.v_proj.bias"), nkv)?,
            // Qwen3 QK-norm: per-head RMSNorm on q/k after projection.
            q_norm: load_qk_norm(
                tensors,
                data,
                &p("self_attn.q_norm.weight"),
                cfg.n_heads * cfg.head_dim,
                cfg.head_dim,
            )?,
            k_norm: load_qk_norm(
                tensors,
                data,
                &p("self_attn.k_norm.weight"),
                cfg.n_kv_heads * cfg.head_dim,
                cfg.head_dim,
            )?,
            mla_q_a,
            mla_q_a_norm,
            mla_q_b,
            mla_q_rope,
            mla_kv_a,
            mla_kv_a_norm,
            mla_kv_b,
            mla_o,
            moe_router,
            moe_wg,
            moe_wu,
            moe_wd,
            shared_wg,
            shared_wu,
            shared_wd,
            gdn_in_qkv,
            gdn_in_z,
            gdn_in_a,
            gdn_in_b,
            gdn_conv_w,
            gdn_a_log,
            gdn_dt_bias,
            gdn_norm,
            gdn_out,
        };
        layers.push(lw);
    }

    // Qwen3.5 zero-centered norms: the checkpoint stores `w` with forward
    // `x * (1 + w)` (`Qwen3_5RMSNorm`); shift so the runtime's plain `x * w`
    // matches. The GDN gated norm multiplies plainly and is NOT shifted.
    if cfg.zero_centered_norm {
        for l in &mut layers {
            for v in &mut l.rms_attn {
                *v += 1.0;
            }
            for v in &mut l.rms_mlp {
                *v += 1.0;
            }
            for v in &mut l.q_norm {
                *v += 1.0;
            }
            for v in &mut l.k_norm {
                *v += 1.0;
            }
        }
        for v in &mut rms_final {
            *v += 1.0;
        }
    }

    Ok(Weights {
        tok_emb,
        rms_final,
        lm_head,
        layers,
    })
}

/// Loads a checkpoint into storage-Q4 form, streaming shards one at a time.
/// On Windows shard payload is mmap-backed (no private whole-shard copy); on
/// other platforms it retains one owned raw-shard buffer at a time. Host
/// private memory is ~= packed Q4 weights plus that platform-dependent scratch
/// (8B model: ~5GB instead of ~48GB for the f32 path).
///
/// Every GEMM weight is quantized to int4 as it is read: Windows uses a
/// read-only mmap; other platforms use the owned-buffer fallback. Norms and
/// biases stay f32.
pub fn load_safetensors_q4(
    path: &Path,
    cfg: &Config,
    tie_embeddings: bool,
) -> Result<WeightsQ4, Error> {
    let d = cfg.d_model;
    let nq = cfg.n_heads * cfg.head_dim;
    let nkv = cfg.n_kv_heads * cfg.head_dim;
    let inter = cfg.intermediate_size;
    let einter = cfg.expert_size();
    let mla = cfg.kv_lora_rank > 0;
    let ne = cfg.num_experts;

    // GEMM tensors are quantized; small tensors (norms/biases) stay f32.
    let mut big: HashMap<String, Q4Tensor> = HashMap::new();
    let mut small: HashMap<String, Vec<f32>> = HashMap::new();

    // Accept either a single .safetensors file or a directory of shards.
    let mut files: Vec<std::path::PathBuf> = if path.is_dir() {
        std::fs::read_dir(path)
            .map_err(|e| Error::Model(format!("read dir {path:?}: {e}")))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect()
    } else {
        vec![path.to_path_buf()]
    };
    files.sort();
    if files.is_empty() {
        return Err(Error::Model(format!("no .safetensors files in {path:?}")));
    }

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(16);

    for file in &files {
        let parsed = parse_safetensors(file)?;
        let tensors = &parsed.tensors;
        let data = parsed.data();

        // Classify this shard's tensors: GEMM weights are quantized (parallel),
        // norms/biases stay f32 (inline). Big-vs-small is decided by whether the
        // name maps to a known GEMM weight.
        let mut big_work: Vec<(String, usize)> = Vec::new();
        for (name, t) in tensors {
            // VLM auxiliary stacks (Qwen3.5 VLM): HF's own text-only class
            // ignores `^mtp.*` / `^model.visual.*` on load. Skipping here
            // also keeps `mtp.layers.*.self_attn.q_proj.weight` out of the
            // classifiers' `contains()` patterns below.
            if name.starts_with("mtp.") || name.starts_with("model.visual.") {
                continue;
            }
            let n: usize = t.shape.iter().product();
            let expected = if name == "model.embed_tokens.weight" || name == "lm_head.weight" {
                Some(cfg.vocab_size * d)
            } else if let Some(e) = expected_q4_size(name, cfg, d, nq, nkv, inter, einter, mla) {
                Some(e)
            } else if let Some(e) = expected_small_size(name, cfg, d, nq, nkv, mla, ne) {
                // q_norm/k_norm: accept both shared [head_dim] and per-head
                // [n_heads*head_dim] / [n_kv_heads*head_dim] forms.
                let e = if (name.contains("q_norm.weight") || name.contains("k_norm.weight"))
                    && n != e
                    && (n == cfg.n_heads * cfg.head_dim || n == cfg.n_kv_heads * cfg.head_dim)
                {
                    n
                } else {
                    e
                };
                small.insert(
                    name.clone(),
                    load_small_f32(data, t, e, name, cfg.zero_centered_norm)?,
                );
                None
            } else {
                // Unknown auxiliary tensors (e.g. shared_expert.*) are skipped,
                // matching the f32 loader's behavior.
                eprintln!("q4 loader: skipping unknown tensor {name}");
                None
            };
            if let Some(e) = expected {
                big_work.push((name.clone(), e));
            }
        }

        // Decode + quantize the GEMM tensors in parallel: per-element CPU work
        // dominates Q4 load, so the shard's tensors are split across threads
        // (peak host RAM is one shard's decoded f32 tensors, then the shard's
        // mmap window or owned shard buffer is dropped before the next shard).
        let next = std::sync::atomic::AtomicUsize::new(0);
        let results: std::sync::Mutex<Vec<(String, Q4Tensor)>> =
            std::sync::Mutex::new(Vec::with_capacity(big_work.len()));
        let err: std::sync::Mutex<Option<Error>> = std::sync::Mutex::new(None);
        std::thread::scope(|s| {
            for _ in 0..n_threads {
                s.spawn(|| {
                    loop {
                        if err.lock().unwrap().is_some() {
                            break;
                        }
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some((name, expected)) = big_work.get(i) else {
                            break;
                        };
                        match tensor_f32_par(data, &tensors[name], *expected, name) {
                            Ok(f) => {
                                let q = Q4Tensor::quantize_par(&f);
                                results.lock().unwrap().push((name.clone(), q));
                            }
                            Err(e) => {
                                let mut g = err.lock().unwrap();
                                if g.is_none() {
                                    *g = Some(e);
                                }
                                break;
                            }
                        }
                    }
                });
            }
        });
        if let Some(e) = err.into_inner().unwrap() {
            return Err(e);
        }
        for (name, q) in results.into_inner().unwrap() {
            big.insert(name, q);
        }
        // `parsed` (and its mmap) is dropped before the next shard.
    }

    // Assemble per-layer Q4 weights.
    // Fused MLA q projection → per-head non-RoPE / RoPE halves. The row blocks
    // are group-aligned (`kk` is `d_model` or `q_lora_rank`), so the Q4 split
    // slices packed bytes + scales without re-quantizing.
    let mla_fused_q = |fused: Q4Tensor| -> (Q4Tensor, Q4Tensor) {
        let kk = if cfg.q_lora_rank > 0 {
            cfg.q_lora_rank
        } else {
            d
        };
        let per_head = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
        let mut nope_blocks = Vec::with_capacity(cfg.n_heads);
        let mut rope_blocks = Vec::with_capacity(cfg.n_heads);
        for h in 0..cfg.n_heads {
            nope_blocks.push((h * per_head, cfg.qk_nope_head_dim));
            rope_blocks.push((h * per_head + cfg.qk_nope_head_dim, cfg.qk_rope_head_dim));
        }
        let nope = Q4Tensor::concat_many(&fused.split_row_blocks(kk, &nope_blocks));
        let rope = Q4Tensor::concat_many(&fused.split_row_blocks(kk, &rope_blocks));
        (nope, rope)
    };
    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let p = |suffix: &str| format!("model.layers.{i}.{suffix}");
        let is_moe = ne > 0 && small.contains_key(&p("mlp.gate.weight"));
        // Qwen3.5 hybrid: linear-attention layers carry `linear_attn.*`.
        let gdn = cfg.gdn_enabled() && !cfg.layer_is_full_attn(i);
        // Shared experts (DeepSeek-V2): dense SwiGLU MLP of width
        // `n_shared_experts * expert_size`, added to the routed experts' sum.
        let shinter = cfg.shared_size();
        let (shared_wg, shared_wu, shared_wd) = if is_moe && shinter > 0 {
            let sp = |proj: &str| format!("model.layers.{i}.mlp.shared_experts.{proj}");
            (
                big.remove(&sp("gate_proj.weight")).expect("shared gate"),
                big.remove(&sp("up_proj.weight")).expect("shared up"),
                big.remove(&sp("down_proj.weight")).expect("shared down"),
            )
        } else {
            (
                Q4Tensor::default(),
                Q4Tensor::default(),
                Q4Tensor::default(),
            )
        };
        let mut lw = LayerWeightsQ4 {
            wq: if mla || gdn {
                Q4Tensor::default()
            } else {
                big.remove(&p("self_attn.q_proj.weight")).expect("q_proj")
            },
            wk: if mla || gdn {
                Q4Tensor::default()
            } else {
                big.remove(&p("self_attn.k_proj.weight")).expect("k_proj")
            },
            wv: if mla || gdn {
                Q4Tensor::default()
            } else {
                big.remove(&p("self_attn.v_proj.weight")).expect("v_proj")
            },
            wo: if mla || gdn {
                Q4Tensor::default()
            } else {
                big.remove(&p("self_attn.o_proj.weight")).expect("o_proj")
            },
            rms_attn: small
                .remove(&p("input_layernorm.weight"))
                .expect("input_layernorm"),
            wg: if is_moe {
                Q4Tensor::default()
            } else {
                big.remove(&p("mlp.gate_proj.weight")).expect("gate_proj")
            },
            wu: if is_moe {
                Q4Tensor::default()
            } else {
                big.remove(&p("mlp.up_proj.weight")).expect("up_proj")
            },
            wd: if is_moe {
                Q4Tensor::default()
            } else {
                big.remove(&p("mlp.down_proj.weight")).expect("down_proj")
            },
            rms_mlp: small
                .remove(&p("post_attention_layernorm.weight"))
                .expect("post_attention_layernorm"),
            bq: small
                .remove(&p("self_attn.q_proj.bias"))
                .unwrap_or_default(),
            bk: small
                .remove(&p("self_attn.k_proj.bias"))
                .unwrap_or_default(),
            bv: small
                .remove(&p("self_attn.v_proj.bias"))
                .unwrap_or_default(),
            q_norm: small
                .remove(&p("self_attn.q_norm.weight"))
                .unwrap_or_default(),
            k_norm: small
                .remove(&p("self_attn.k_norm.weight"))
                .unwrap_or_default(),
            mla_q_a: if mla && cfg.q_lora_rank > 0 {
                big.remove(&p("self_attn.q_a_proj.weight"))
                    .expect("mla q_a")
            } else {
                Q4Tensor::default()
            },
            mla_q_a_norm: if mla && cfg.q_lora_rank > 0 {
                small
                    .remove(&p("self_attn.q_a_layernorm.weight"))
                    .expect("mla q_a_norm")
            } else {
                Vec::new()
            },
            // Placeholders: overwritten below from the fused projection, which
            // is loaded once and split per head into the two halves.
            mla_q_b: Q4Tensor::default(),
            mla_q_rope: Q4Tensor::default(),
            mla_kv_a: if mla {
                big.remove(&p("self_attn.kv_a_proj_with_mqa.weight"))
                    .expect("mla kv_a")
            } else {
                Q4Tensor::default()
            },
            mla_kv_a_norm: if mla {
                small
                    .remove(&p("self_attn.kv_a_layernorm.weight"))
                    .expect("mla kv_a_norm")
            } else {
                Vec::new()
            },
            mla_kv_b: if mla {
                big.remove(&p("self_attn.kv_b_proj.weight"))
                    .expect("mla kv_b")
            } else {
                Q4Tensor::default()
            },
            mla_o: if mla {
                big.remove(&p("self_attn.o_proj.weight")).expect("mla o")
            } else {
                Q4Tensor::default()
            },
            moe_router: small.remove(&p("mlp.gate.weight")).unwrap_or_default(),
            moe_wg: Q4Tensor::default(),
            moe_wu: Q4Tensor::default(),
            moe_wd: Q4Tensor::default(),
            shared_wg,
            shared_wu,
            shared_wd,
            gdn_in_qkv: if gdn {
                big.remove(&p("linear_attn.in_proj_qkv.weight"))
                    .expect("gdn in_proj_qkv")
            } else {
                Q4Tensor::default()
            },
            gdn_in_z: if gdn {
                big.remove(&p("linear_attn.in_proj_z.weight"))
                    .expect("gdn in_proj_z")
            } else {
                Q4Tensor::default()
            },
            gdn_in_a: if gdn {
                small
                    .remove(&p("linear_attn.in_proj_a.weight"))
                    .expect("gdn in_proj_a")
            } else {
                Vec::new()
            },
            gdn_in_b: if gdn {
                small
                    .remove(&p("linear_attn.in_proj_b.weight"))
                    .expect("gdn in_proj_b")
            } else {
                Vec::new()
            },
            gdn_conv_w: if gdn {
                small
                    .remove(&p("linear_attn.conv1d.weight"))
                    .expect("gdn conv1d")
            } else {
                Vec::new()
            },
            gdn_a_log: if gdn {
                small.remove(&p("linear_attn.A_log")).expect("gdn A_log")
            } else {
                Vec::new()
            },
            gdn_dt_bias: if gdn {
                small
                    .remove(&p("linear_attn.dt_bias"))
                    .expect("gdn dt_bias")
            } else {
                Vec::new()
            },
            gdn_norm: if gdn {
                small
                    .remove(&p("linear_attn.norm.weight"))
                    .expect("gdn norm")
            } else {
                Vec::new()
            },
            gdn_out: if gdn {
                big.remove(&p("linear_attn.out_proj.weight"))
                    .expect("gdn out_proj")
            } else {
                Q4Tensor::default()
            },
        };
        if mla {
            let fused = if cfg.q_lora_rank > 0 {
                big.remove(&p("self_attn.q_b_proj.weight"))
                    .expect("mla q_b")
            } else {
                big.remove(&p("self_attn.q_proj.weight"))
                    .expect("mla q_proj")
            };
            let (nope, rope) = mla_fused_q(fused);
            lw.mla_q_b = nope;
            lw.mla_q_rope = rope;
        }
        if is_moe {
            let ne_i = ne;
            // Single-pass per-tensor concat: the sequential fold re-clones the
            // growing prefix per expert (O(n²) byte traffic — minutes on
            // 256-expert checkpoints); concat_many appends in one O(total)
            // pass (byte-identical for the group-aligned expert tensors).
            let mut concat = |name: &str| -> Q4Tensor {
                let parts: Vec<Q4Tensor> = (0..ne_i)
                    .map(|e| {
                        big.remove(&format!("model.layers.{i}.mlp.experts.{e}.{name}"))
                            .expect("exp tensor")
                    })
                    .collect();
                Q4Tensor::concat_many(&parts)
            };
            lw.moe_wg = concat("gate_proj.weight");
            lw.moe_wu = concat("up_proj.weight");
            lw.moe_wd = concat("down_proj.weight");
        }
        broadcast_qk_norm(&mut lw.q_norm, &mut lw.k_norm, cfg);
        layers.push(lw);
    }

    let tok_emb = big.remove("model.embed_tokens.weight").expect("tok_emb");
    let lm_head = match big.remove("lm_head.weight") {
        Some(t) => t,
        None if tie_embeddings => tok_emb.clone(),
        None => {
            return Err(Error::Model(
                "lm_head.weight missing and tie_embeddings=false".into(),
            ));
        }
    };
    Ok(WeightsQ4 {
        tok_emb,
        rms_final: small.remove("model.norm.weight").expect("norm"),
        lm_head,
        layers,
    })
}

/// Loads a checkpoint into storage-FP8 form, streaming shards one at a time.
/// On Windows shard payload is mmap-backed (no private whole-shard copy); on
/// other platforms it retains one owned raw-shard buffer at a time. Host
/// private memory is ~= packed FP8 weights plus that platform-dependent
/// scratch (8B model: ~8GB instead of ~48GB f32 / ~16GB f16).
///
/// Every GEMM weight is quantized to E4M3 (one byte/element + one f32 scale
/// per tensor; MoE experts keep per-expert scales and concatenate by appending
/// packed bytes + scales) as it is read. Windows uses a read-only mmap; other
/// platforms retain the previous owned-shard fallback. Norms and biases stay f32.
pub fn load_safetensors_fp8(
    path: &Path,
    cfg: &Config,
    tie_embeddings: bool,
) -> Result<WeightsFp8, Error> {
    let d = cfg.d_model;
    let nq = cfg.n_heads * cfg.head_dim;
    let nkv = cfg.n_kv_heads * cfg.head_dim;
    let inter = cfg.intermediate_size;
    let einter = cfg.expert_size();
    let mla = cfg.kv_lora_rank > 0;
    let ne = cfg.num_experts;

    // GEMM tensors are quantized; small tensors (norms/biases) stay f32.
    let mut big: HashMap<String, Fp8Tensor> = HashMap::new();
    let mut small: HashMap<String, Vec<f32>> = HashMap::new();

    // Accept either a single .safetensors file or a directory of shards.
    let mut files: Vec<std::path::PathBuf> = if path.is_dir() {
        std::fs::read_dir(path)
            .map_err(|e| Error::Model(format!("read dir {path:?}: {e}")))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect()
    } else {
        vec![path.to_path_buf()]
    };
    files.sort();
    if files.is_empty() {
        return Err(Error::Model(format!("no .safetensors files in {path:?}")));
    }

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(16);

    for file in &files {
        let parsed = parse_safetensors(file)?;
        let tensors = &parsed.tensors;
        let data = parsed.data();

        // Classify this shard's tensors: GEMM weights are quantized (parallel),
        // norms/biases stay f32 (inline). Big-vs-small is decided by whether the
        // name maps to a known GEMM weight (same shape matcher as the Q4
        // loader; only the quantization differs).
        let mut big_work: Vec<(String, usize)> = Vec::new();
        for (name, t) in tensors {
            // VLM auxiliary stacks (Qwen3.5 VLM): same skip as the Q4 loader.
            if name.starts_with("mtp.") || name.starts_with("model.visual.") {
                continue;
            }
            let n: usize = t.shape.iter().product();
            let expected = if name == "model.embed_tokens.weight" || name == "lm_head.weight" {
                Some(cfg.vocab_size * d)
            } else if let Some(e) = expected_q4_size(name, cfg, d, nq, nkv, inter, einter, mla) {
                Some(e)
            } else if let Some(e) = expected_small_size(name, cfg, d, nq, nkv, mla, ne) {
                // q_norm/k_norm: accept both shared [head_dim] and per-head
                // [n_heads*head_dim] / [n_kv_heads*head_dim] forms.
                let e = if (name.contains("q_norm.weight") || name.contains("k_norm.weight"))
                    && n != e
                    && (n == cfg.n_heads * cfg.head_dim || n == cfg.n_kv_heads * cfg.head_dim)
                {
                    n
                } else {
                    e
                };
                small.insert(
                    name.clone(),
                    load_small_f32(data, t, e, name, cfg.zero_centered_norm)?,
                );
                None
            } else {
                // Unknown auxiliary tensors (e.g. shared_expert.*) are skipped,
                // matching the f32 loader's behavior.
                eprintln!("fp8 loader: skipping unknown tensor {name}");
                None
            };
            if let Some(e) = expected {
                big_work.push((name.clone(), e));
            }
        }

        // Decode + quantize the GEMM tensors in parallel: per-element CPU work
        // dominates FP8 load, so the shard's tensors are split across threads
        // (peak host RAM is one shard's decoded f32 tensors, then the shard's
        // mmap window or owned shard buffer is dropped before the next shard).
        let next = std::sync::atomic::AtomicUsize::new(0);
        let results: std::sync::Mutex<Vec<(String, Fp8Tensor)>> =
            std::sync::Mutex::new(Vec::with_capacity(big_work.len()));
        let err: std::sync::Mutex<Option<Error>> = std::sync::Mutex::new(None);
        std::thread::scope(|s| {
            for _ in 0..n_threads {
                s.spawn(|| {
                    loop {
                        if err.lock().unwrap().is_some() {
                            break;
                        }
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some((name, expected)) = big_work.get(i) else {
                            break;
                        };
                        match tensor_f32(data, &tensors[name], *expected, name) {
                            Ok(f) => {
                                let q = Fp8Tensor::quantize_par(&f);
                                results.lock().unwrap().push((name.clone(), q));
                            }
                            Err(e) => {
                                let mut g = err.lock().unwrap();
                                if g.is_none() {
                                    *g = Some(e);
                                }
                                break;
                            }
                        }
                    }
                });
            }
        });
        if let Some(e) = err.into_inner().unwrap() {
            return Err(e);
        }
        for (name, q) in results.into_inner().unwrap() {
            big.insert(name, q);
        }
        // `parsed` (and its mmap) is dropped before the next shard.
    }

    // Assemble per-layer FP8 weights.
    // Fused MLA q projection → per-head non-RoPE / RoPE halves. FP8 keeps a
    // per-tensor scale, so the halves are re-quantized from the sliced values
    // (byte-identical to loading them as separate tensors).
    let mla_fused_q = |fused: Fp8Tensor| -> (Fp8Tensor, Fp8Tensor) {
        let kk = if cfg.q_lora_rank > 0 {
            cfg.q_lora_rank
        } else {
            d
        };
        let per_head = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
        let mut nope_blocks = Vec::with_capacity(cfg.n_heads);
        let mut rope_blocks = Vec::with_capacity(cfg.n_heads);
        for h in 0..cfg.n_heads {
            nope_blocks.push((h * per_head, cfg.qk_nope_head_dim));
            rope_blocks.push((h * per_head + cfg.qk_nope_head_dim, cfg.qk_rope_head_dim));
        }
        let nope = Fp8Tensor::concat_many(&fused.split_row_blocks(kk, &nope_blocks));
        let rope = Fp8Tensor::concat_many(&fused.split_row_blocks(kk, &rope_blocks));
        (nope, rope)
    };
    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let p = |suffix: &str| format!("model.layers.{i}.{suffix}");
        let is_moe = ne > 0 && small.contains_key(&p("mlp.gate.weight"));
        let gdn = cfg.gdn_enabled() && !cfg.layer_is_full_attn(i);
        // Shared experts (DeepSeek-V2): dense SwiGLU MLP of width
        // `n_shared_experts * expert_size`, added to the routed experts' sum.
        let shinter = cfg.shared_size();
        let (shared_wg, shared_wu, shared_wd) = if is_moe && shinter > 0 {
            let sp = |proj: &str| format!("model.layers.{i}.mlp.shared_experts.{proj}");
            (
                big.remove(&sp("gate_proj.weight")).expect("shared gate"),
                big.remove(&sp("up_proj.weight")).expect("shared up"),
                big.remove(&sp("down_proj.weight")).expect("shared down"),
            )
        } else {
            (
                Fp8Tensor::default(),
                Fp8Tensor::default(),
                Fp8Tensor::default(),
            )
        };
        let mut lw = LayerWeightsFp8 {
            wq: if mla || gdn {
                Fp8Tensor::default()
            } else {
                big.remove(&p("self_attn.q_proj.weight")).expect("q_proj")
            },
            wk: if mla || gdn {
                Fp8Tensor::default()
            } else {
                big.remove(&p("self_attn.k_proj.weight")).expect("k_proj")
            },
            wv: if mla || gdn {
                Fp8Tensor::default()
            } else {
                big.remove(&p("self_attn.v_proj.weight")).expect("v_proj")
            },
            wo: if mla || gdn {
                Fp8Tensor::default()
            } else {
                big.remove(&p("self_attn.o_proj.weight")).expect("o_proj")
            },
            rms_attn: small
                .remove(&p("input_layernorm.weight"))
                .expect("input_layernorm"),
            wg: if is_moe {
                Fp8Tensor::default()
            } else {
                big.remove(&p("mlp.gate_proj.weight")).expect("gate_proj")
            },
            wu: if is_moe {
                Fp8Tensor::default()
            } else {
                big.remove(&p("mlp.up_proj.weight")).expect("up_proj")
            },
            wd: if is_moe {
                Fp8Tensor::default()
            } else {
                big.remove(&p("mlp.down_proj.weight")).expect("down_proj")
            },
            rms_mlp: small
                .remove(&p("post_attention_layernorm.weight"))
                .expect("post_attention_layernorm"),
            bq: small
                .remove(&p("self_attn.q_proj.bias"))
                .unwrap_or_default(),
            bk: small
                .remove(&p("self_attn.k_proj.bias"))
                .unwrap_or_default(),
            bv: small
                .remove(&p("self_attn.v_proj.bias"))
                .unwrap_or_default(),
            q_norm: small
                .remove(&p("self_attn.q_norm.weight"))
                .unwrap_or_default(),
            k_norm: small
                .remove(&p("self_attn.k_norm.weight"))
                .unwrap_or_default(),
            mla_q_a: if mla && cfg.q_lora_rank > 0 {
                big.remove(&p("self_attn.q_a_proj.weight"))
                    .expect("mla q_a")
            } else {
                Fp8Tensor::default()
            },
            mla_q_a_norm: if mla && cfg.q_lora_rank > 0 {
                small
                    .remove(&p("self_attn.q_a_layernorm.weight"))
                    .expect("mla q_a_norm")
            } else {
                Vec::new()
            },
            // Placeholders: overwritten below from the fused projection, which
            // is loaded once and split per head into the two halves.
            mla_q_b: Fp8Tensor::default(),
            mla_q_rope: Fp8Tensor::default(),
            mla_kv_a: if mla {
                big.remove(&p("self_attn.kv_a_proj_with_mqa.weight"))
                    .expect("mla kv_a")
            } else {
                Fp8Tensor::default()
            },
            mla_kv_a_norm: if mla {
                small
                    .remove(&p("self_attn.kv_a_layernorm.weight"))
                    .expect("mla kv_a_norm")
            } else {
                Vec::new()
            },
            mla_kv_b: if mla {
                big.remove(&p("self_attn.kv_b_proj.weight"))
                    .expect("mla kv_b")
            } else {
                Fp8Tensor::default()
            },
            mla_o: if mla {
                big.remove(&p("self_attn.o_proj.weight")).expect("mla o")
            } else {
                Fp8Tensor::default()
            },
            moe_router: small.remove(&p("mlp.gate.weight")).unwrap_or_default(),
            moe_wg: Fp8Tensor::default(),
            moe_wu: Fp8Tensor::default(),
            moe_wd: Fp8Tensor::default(),
            shared_wg,
            shared_wu,
            shared_wd,
            gdn_in_qkv: if gdn {
                big.remove(&p("linear_attn.in_proj_qkv.weight"))
                    .expect("gdn in_proj_qkv")
            } else {
                Fp8Tensor::default()
            },
            gdn_in_z: if gdn {
                big.remove(&p("linear_attn.in_proj_z.weight"))
                    .expect("gdn in_proj_z")
            } else {
                Fp8Tensor::default()
            },
            gdn_in_a: if gdn {
                small
                    .remove(&p("linear_attn.in_proj_a.weight"))
                    .expect("gdn in_proj_a")
            } else {
                Vec::new()
            },
            gdn_in_b: if gdn {
                small
                    .remove(&p("linear_attn.in_proj_b.weight"))
                    .expect("gdn in_proj_b")
            } else {
                Vec::new()
            },
            gdn_conv_w: if gdn {
                small
                    .remove(&p("linear_attn.conv1d.weight"))
                    .expect("gdn conv1d")
            } else {
                Vec::new()
            },
            gdn_a_log: if gdn {
                small.remove(&p("linear_attn.A_log")).expect("gdn A_log")
            } else {
                Vec::new()
            },
            gdn_dt_bias: if gdn {
                small
                    .remove(&p("linear_attn.dt_bias"))
                    .expect("gdn dt_bias")
            } else {
                Vec::new()
            },
            gdn_norm: if gdn {
                small
                    .remove(&p("linear_attn.norm.weight"))
                    .expect("gdn norm")
            } else {
                Vec::new()
            },
            gdn_out: if gdn {
                big.remove(&p("linear_attn.out_proj.weight"))
                    .expect("gdn out_proj")
            } else {
                Fp8Tensor::default()
            },
        };
        if mla {
            let fused = if cfg.q_lora_rank > 0 {
                big.remove(&p("self_attn.q_b_proj.weight"))
                    .expect("mla q_b")
            } else {
                big.remove(&p("self_attn.q_proj.weight"))
                    .expect("mla q_proj")
            };
            let (nope, rope) = mla_fused_q(fused);
            lw.mla_q_b = nope;
            lw.mla_q_rope = rope;
        }
        if is_moe {
            let ne_i = ne;
            // Single-pass per-tensor concat (the sequential fold re-clones the
            // growing prefix per expert — O(n²) byte traffic).
            let mut concat = |name: &str| -> Fp8Tensor {
                let parts: Vec<Fp8Tensor> = (0..ne_i)
                    .map(|e| {
                        big.remove(&format!("model.layers.{i}.mlp.experts.{e}.{name}"))
                            .expect("exp tensor")
                    })
                    .collect();
                Fp8Tensor::concat_many(&parts)
            };
            lw.moe_wg = concat("gate_proj.weight");
            lw.moe_wu = concat("up_proj.weight");
            lw.moe_wd = concat("down_proj.weight");
        }
        broadcast_qk_norm(&mut lw.q_norm, &mut lw.k_norm, cfg);
        layers.push(lw);
    }

    let tok_emb = big.remove("model.embed_tokens.weight").expect("tok_emb");
    let lm_head = match big.remove("lm_head.weight") {
        Some(t) => t,
        None if tie_embeddings => tok_emb.clone(),
        None => {
            return Err(Error::Model(
                "lm_head.weight missing and tie_embeddings=false".into(),
            ));
        }
    };
    Ok(WeightsFp8 {
        tok_emb,
        rms_final: small.remove("model.norm.weight").expect("norm"),
        lm_head,
        layers,
    })
}

/// Loads a small (f32) tensor, applying the Qwen3.5 zero-centered-norm shift
/// (`x * (1 + w)` checkpoints store `w`; the runtime multiplies plainly, so
/// the loader adds 1 to the layer/final/q/k norm tensors). The GDN gated
/// norm is ones-init and passes through untouched.
fn load_small_f32(
    data: &[u8],
    t: &RawTensor,
    expected: usize,
    name: &str,
    zero_centered: bool,
) -> Result<Vec<f32>, Error> {
    let mut v = tensor_f32(data, t, expected, name)?;
    if zero_centered
        && (name == "model.norm.weight"
            || name.contains("input_layernorm.weight")
            || name.contains("post_attention_layernorm.weight")
            || name.contains("q_norm.weight")
            || name.contains("k_norm.weight"))
    {
        for x in &mut v {
            *x += 1.0;
        }
    }
    Ok(v)
}

/// Expected element count for a quantized (GEMM) weight tensor by name.
#[allow(clippy::too_many_arguments)]
fn expected_q4_size(
    name: &str,
    cfg: &Config,
    d: usize,
    nq: usize,
    nkv: usize,
    inter: usize,
    einter: usize,
    mla: bool,
) -> Option<usize> {
    // Match against the same names build_weights uses. MLA's `o_proj` is a
    // different shape, so it must be matched before the dense `o_proj`.
    if mla && name.contains("self_attn.o_proj.weight") {
        Some(d * cfg.n_heads * cfg.v_head_dim)
    } else if mla && cfg.q_lora_rank > 0 && name.contains("self_attn.q_a_proj.weight") {
        Some(cfg.q_lora_rank * d)
    } else if mla && cfg.q_lora_rank > 0 && name.contains("self_attn.q_b_proj.weight") {
        // Fused: the RoPE and non-RoPE rows are interleaved per head and split
        // after loading (see [`split_mla_q`]).
        Some(cfg.n_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) * cfg.q_lora_rank)
    } else if mla && cfg.q_lora_rank == 0 && name.contains("self_attn.q_proj.weight") {
        // Fused `q_proj` (DeepSeek-V2-Lite ships `q_lora_rank: null`): one
        // `[heads*(nope+rope), d]` matrix holding both halves per head.
        Some(cfg.n_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) * d)
    } else if mla && name.contains("self_attn.kv_a_proj_with_mqa.weight") {
        Some((cfg.kv_lora_rank + cfg.qk_rope_head_dim) * d)
    } else if mla && name.contains("self_attn.kv_b_proj.weight") {
        Some(cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim) * cfg.kv_lora_rank)
    } else if name.contains("mlp.shared_experts.")
        && (name.ends_with("gate_proj.weight")
            || name.ends_with("up_proj.weight")
            || name.ends_with("down_proj.weight"))
    {
        // DeepSeek-V2 shared experts: one dense SwiGLU MLP of width
        // `n_shared_experts * expert_size` (not per-expert tensors).
        Some(cfg.shared_size() * d)
    } else if name.contains("linear_attn.in_proj_qkv.weight") {
        // Qwen3.5 gated DeltaNet: fused q/k/v projection.
        Some((2 * cfg.gdn_key_dim() + cfg.gdn_value_dim()) * d)
    } else if name.contains("linear_attn.in_proj_z.weight")
        || name.contains("linear_attn.out_proj.weight")
    {
        // Qwen3.5 gated DeltaNet: z projection and output projection both
        // span the value dim ([v, d] and [d, v]).
        Some(cfg.gdn_value_dim() * d)
    } else if name.contains("self_attn.q_proj.weight") {
        // Qwen3.5 `attn_output_gate` doubles the width (per-head sigmoid gate
        // rides along in q_proj).
        Some(d * nq * if cfg.attn_output_gate { 2 } else { 1 })
    } else if name.contains("self_attn.k_proj.weight") || name.contains("self_attn.v_proj.weight") {
        Some(d * nkv)
    } else if name.contains("self_attn.o_proj.weight") {
        Some(nq * d)
    } else if name.contains("mlp.gate_proj.weight")
        || name.contains("mlp.up_proj.weight")
        || name.contains("mlp.down_proj.weight")
    {
        Some(inter * d)
    } else if name.contains("mlp.experts.")
        && (name.ends_with("gate_proj.weight")
            || name.ends_with("up_proj.weight")
            || name.ends_with("down_proj.weight"))
    {
        Some(einter * d)
    } else {
        None
    }
}

/// Expected element count for a small (f32) tensor by name.
fn expected_small_size(
    name: &str,
    cfg: &Config,
    d: usize,
    nq: usize,
    nkv: usize,
    mla: bool,
    ne: usize,
) -> Option<usize> {
    if name == "model.norm.weight"
        || name.contains("input_layernorm.weight")
        || name.contains("post_attention_layernorm.weight")
    {
        Some(d)
    } else if name.ends_with("self_attn.q_proj.bias") {
        Some(nq)
    } else if name.ends_with("self_attn.k_proj.bias") || name.ends_with("self_attn.v_proj.bias") {
        Some(nkv)
    } else if name.contains("mlp.gate.weight") {
        Some(ne * d)
    } else if name.contains("self_attn.q_norm.weight") || name.contains("self_attn.k_norm.weight") {
        Some(cfg.head_dim)
    } else if mla && name.contains("self_attn.q_a_layernorm.weight") {
        Some(cfg.q_lora_rank)
    } else if mla && name.contains("self_attn.kv_a_layernorm.weight") {
        Some(cfg.kv_lora_rank)
    } else if name.contains("linear_attn.in_proj_a.weight")
        || name.contains("linear_attn.in_proj_b.weight")
    {
        // GDN gate inputs: tiny `[gdn_v_heads, d]` GEMMs kept f32 (they feed
        // softplus/sigmoid, where quantization error shifts a whole head's
        // decay).
        Some(cfg.gdn_v_heads * d)
    } else if name.contains("linear_attn.conv1d.weight") {
        Some((2 * cfg.gdn_key_dim() + cfg.gdn_value_dim()) * cfg.gdn_conv_kernel)
    } else if name.contains("linear_attn.A_log") || name.contains("linear_attn.dt_bias") {
        // Bare nn.Parameter names (no `.weight` suffix) — direct parameters,
        // not submodules.
        Some(cfg.gdn_v_heads)
    } else if name.contains("linear_attn.norm.weight") {
        Some(cfg.gdn_head_dim)
    } else {
        None
    }
}

/// Concatenates two quantized tensors (per-expert MoE tensors). Delegates to
/// `Q4Tensor::concat`: group-aligned tensors append packed int4 bytes + scales
/// directly (O(1) per expert, fixing the old O(ne^2) dequant+requant that made
/// Q4 loading of MoE checkpoints ~30x slower than f32).
#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(dtype: &str, n: usize, seed: u64) -> (RawTensor, Vec<u8>) {
        let elem = match dtype {
            "F32" => 4usize,
            _ => 2usize,
        };
        let mut data = Vec::with_capacity(n * elem);
        let mut r = seed;
        for i in 0..n {
            r = r
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = (((r >> 33) as f64) / ((1u64 << 31) as f64)) as f32 * 3.0 - 1.5;
            match dtype {
                "F32" => data.extend_from_slice(&v.to_le_bytes()),
                "F16" => data.extend_from_slice(&crate::fp16::f32_to_f16(v).to_le_bytes()),
                "BF16" => data.extend_from_slice(&(((v.to_bits() >> 16) as u16).to_le_bytes())),
                _ => unreachable!(),
            }
            let _ = i;
        }
        let t = RawTensor {
            dtype: dtype.to_string(),
            shape: vec![n],
            start: 0,
            end: n * elem,
        };
        (t, data)
    }

    #[cfg(windows)]
    #[test]
    fn windows_mmap_denies_concurrent_writer() {
        let path = std::env::temp_dir().join(format!(
            "machserve-mmap-share-{}.safetensors",
            std::process::id()
        ));
        let header = r#"{"x":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 16]);
        std::fs::write(&path, bytes).unwrap();

        let parsed = parse_safetensors(&path).unwrap();
        assert!(
            std::fs::OpenOptions::new().write(true).open(&path).is_err(),
            "a writer must not open the file while the mmap is live"
        );
        drop(parsed);
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("writer becomes available after the mmap is dropped");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn tensor_f32_par_matches_scalar_bitwise() {
        for dtype in ["F32", "F16", "BF16"] {
            // Large (parallel path) and small (fallback path).
            for n in [2usize << 20, 1usize << 21, 1000usize] {
                let (t, data) = synthetic(dtype, n, 42 + n as u64);
                let a = tensor_f32(&data, &t, n, "w").unwrap();
                let b = tensor_f32_par(&data, &t, n, "w").unwrap();
                assert_eq!(a, b, "dtype={dtype} n={n}");
            }
        }
    }
}
