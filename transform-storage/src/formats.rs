//! Binary format specifications for GMAT tensor storage.
//!
//! Defines header structures, metadata, and constants for the GMAT format
//! used by GraphMatrix serialization.

use std::collections::HashMap;
use std::io::{self, Read, Write};

/// GMAT format magic bytes
pub const GMAT_MAGIC: [u8; 4] = *b"GMAT";

/// GMAT format version 1 (no metadata)
pub const GMAT_VERSION_V1: u16 = 1;

/// GMAT format version 2 (with metadata support)
pub const GMAT_VERSION_V2: u16 = 2;

/// GMAT format default version
pub const GMAT_VERSION: u16 = GMAT_VERSION_V2;

/// GMAT file header for GraphMatrix serialization.
///
/// V1 format (23 bytes, no metadata):
/// - magic: 4 bytes (b"GMAT")
/// - version: 2 bytes (u16=1, little-endian)
/// - format: 1 byte (0-7 for block formats)
/// - rows: 8 bytes (u64, little-endian)
/// - cols: 8 bytes (u64, little-endian)
///
/// V2 format (31+ bytes, with metadata):
/// - magic: 4 bytes (b"GMAT")
/// - version: 2 bytes (u16=2, little-endian)
/// - format: 1 byte (0-7 for block formats)
/// - rows: 8 bytes (u64, little-endian)
/// - cols: 8 bytes (u64, little-endian)
/// - metadata_len: 4 bytes (u32, little-endian)
/// - metadata: metadata_len bytes (JSON)
#[derive(Debug, Clone, PartialEq)]
pub struct GmatHeader {
    pub version: u16,
    pub format: u8,
    pub rows: u64,
    pub cols: u64,
    pub metadata: Option<GmatMetadata>,
}

impl GmatHeader {
    /// V1 header size in bytes (no metadata)
    pub const V1_SIZE: usize = 23;

    /// Create a new V1 header (no metadata)
    pub fn new(format: u8, rows: u64, cols: u64) -> Self {
        Self {
            version: GMAT_VERSION_V1,
            format,
            rows,
            cols,
            metadata: None,
        }
    }

    /// Create a new V2 header with metadata
    pub fn new_with_metadata(format: u8, rows: u64, cols: u64, metadata: GmatMetadata) -> Self {
        Self {
            version: GMAT_VERSION_V2,
            format,
            rows,
            cols,
            metadata: Some(metadata),
        }
    }

    /// Get the metadata, if present
    pub fn metadata(&self) -> Option<&GmatMetadata> {
        self.metadata.as_ref()
    }

    /// Get mutable metadata reference, if present
    pub fn metadata_mut(&mut self) -> Option<&mut GmatMetadata> {
        self.metadata.as_mut()
    }

    /// Set metadata (converts to V2)
    pub fn with_metadata(mut self, metadata: GmatMetadata) -> Self {
        self.metadata = Some(metadata);
        self.version = GMAT_VERSION_V2;
        self
    }

    /// Write header to writer
    pub fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        // Determine version based on metadata presence
        let version = if self.metadata.is_some() {
            GMAT_VERSION_V2
        } else {
            GMAT_VERSION_V1
        };

        // Write common header fields
        w.write_all(&GMAT_MAGIC)?;
        w.write_all(&version.to_le_bytes())?;
        w.write_all(&[self.format])?;
        w.write_all(&self.rows.to_le_bytes())?;
        w.write_all(&self.cols.to_le_bytes())?;

        // Write metadata section for V2
        if let Some(ref metadata) = self.metadata {
            let json_bytes = metadata.to_json_bytes()?;
            let metadata_len = json_bytes.len() as u32;
            w.write_all(&metadata_len.to_le_bytes())?;
            w.write_all(&json_bytes)?;
        }

        Ok(())
    }

    /// Read header from reader
    pub fn read_from<R: Read>(r: &mut R) -> io::Result<Self> {
        // Read magic
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != GMAT_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid GMAT magic: expected {:?}, got {:?}", GMAT_MAGIC, magic),
            ));
        }

        // Read version
        let mut version_bytes = [0u8; 2];
        r.read_exact(&mut version_bytes)?;
        let version = u16::from_le_bytes(version_bytes);

        // Validate version
        if version != GMAT_VERSION_V1 && version != GMAT_VERSION_V2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported GMAT version: {version} (expected 1 or 2)"),
            ));
        }

        // Read format
        let mut format_byte = [0u8; 1];
        r.read_exact(&mut format_byte)?;
        let format = format_byte[0];
        
        // Validate format is one of the known values (0-7 for single and dual-row formats)
        if format > 7 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid GMAT format: {format} (expected 0-7)"),
            ));
        }

        // Read dimensions
        let mut rows_bytes = [0u8; 8];
        r.read_exact(&mut rows_bytes)?;
        let rows = u64::from_le_bytes(rows_bytes);

        let mut cols_bytes = [0u8; 8];
        r.read_exact(&mut cols_bytes)?;
        let cols = u64::from_le_bytes(cols_bytes);

        // Read metadata for V2
        let metadata = if version == GMAT_VERSION_V2 {
            let mut metadata_len_bytes = [0u8; 4];
            r.read_exact(&mut metadata_len_bytes)?;
            let metadata_len = u32::from_le_bytes(metadata_len_bytes) as usize;

            let mut json_bytes = vec![0u8; metadata_len];
            r.read_exact(&mut json_bytes)?;
            Some(GmatMetadata::from_json_bytes(&json_bytes)?)
        } else {
            None
        };

        Ok(Self {
            version,
            format,
            rows,
            cols,
            metadata,
        })
    }

    /// Get shape as (rows, cols) tuple
    pub fn shape(&self) -> (usize, usize) {
        (self.rows as usize, self.cols as usize)
    }
}

/// Standard metadata key constants
pub mod metadata_keys {
    /// Original data type before conversion (e.g., "BF16", "F32", "F16")
    pub const ORIGINAL_DTYPE: &str = "gmat.original_dtype";
    /// Block format used (e.g., "B8x4", "B16x8")
    pub const BLOCK_FORMAT: &str = "gmat.block_format";
    /// Creation timestamp (ISO 8601 format)
    pub const CREATED_AT: &str = "gmat.created_at";
    /// Source format (e.g., "safetensors", "pytorch")
    pub const SOURCE_FORMAT: &str = "gmat.source_format";
    /// Conversion information
    pub const CONVERSION_INFO: &str = "gmat.conversion_info";
}

/// Metadata container for GMAT format.
///
/// Stores arbitrary key-value pairs with typed JSON values. Supports integers,
/// floats, strings, booleans, arrays, and objects. Standard keys are defined
/// in the `metadata_keys` module. Serializes to/from JSON.
#[derive(Debug, Clone, PartialEq)]
pub struct GmatMetadata {
    inner: HashMap<String, serde_json::Value>,
}

impl GmatMetadata {
    /// Create a new empty metadata container
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    /// Get raw JSON value by key
    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.inner.get(key)
    }

    /// Get a string value by key
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.inner.get(key).and_then(|v| v.as_str())
    }

    /// Get an i64 value by key
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.inner.get(key).and_then(|v| v.as_i64())
    }

    /// Get a u64 value by key
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.inner.get(key).and_then(|v| v.as_u64())
    }

    /// Get an f64 value by key
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.inner.get(key).and_then(|v| v.as_f64())
    }

    /// Get a bool value by key
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.inner.get(key).and_then(|v| v.as_bool())
    }

    /// Set a string value
    pub fn set_str(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.inner.insert(key.into(), serde_json::Value::String(value.into()));
    }

    /// Set an integer value
    pub fn set_i64(&mut self, key: impl Into<String>, value: i64) {
        self.inner.insert(key.into(), serde_json::Value::Number(value.into()));
    }

    /// Set an unsigned integer value
    pub fn set_u64(&mut self, key: impl Into<String>, value: u64) {
        self.inner.insert(key.into(), serde_json::json!(value));
    }

    /// Set a float value
    pub fn set_f64(&mut self, key: impl Into<String>, value: f64) {
        if let Some(n) = serde_json::Number::from_f64(value) {
            self.inner.insert(key.into(), serde_json::Value::Number(n));
        }
    }

    /// Set a bool value
    pub fn set_bool(&mut self, key: impl Into<String>, value: bool) {
        self.inner.insert(key.into(), serde_json::Value::Bool(value));
    }

    /// Set a raw JSON value
    pub fn set(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.inner.insert(key.into(), value);
    }

    /// Remove a metadata key
    pub fn remove(&mut self, key: &str) -> Option<serde_json::Value> {
        self.inner.remove(key)
    }

    /// Check if a key exists
    pub fn contains_key(&self, key: &str) -> bool {
        self.inner.contains_key(key)
    }

    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Iterate over all key-value pairs
    pub fn iter(&self) -> impl Iterator<Item = (&str, &serde_json::Value)> {
        self.inner.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Get mutable access to inner map
    pub fn as_map_mut(&mut self) -> &mut HashMap<String, serde_json::Value> {
        &mut self.inner
    }

    /// Serialize metadata to JSON bytes
    pub fn to_json_bytes(&self) -> io::Result<Vec<u8>> {
        serde_json::to_vec(&self.inner)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Deserialize metadata from JSON bytes
    pub fn from_json_bytes(bytes: &[u8]) -> io::Result<Self> {
        let inner = serde_json::from_slice(bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(Self { inner })
    }

    /// Create from a HashMap of string key-value pairs (for compatibility)
    pub fn from_string_map(map: HashMap<String, String>) -> Self {
        let inner = map
            .into_iter()
            .map(|(k, v)| (k, serde_json::Value::String(v)))
            .collect();
        Self { inner }
    }
}

impl Default for GmatMetadata {
    fn default() -> Self {
        Self::new()
    }
}
