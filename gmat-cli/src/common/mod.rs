//! Common utilities shared between import and export modules.

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use std::fs;
use std::path::{Path, PathBuf};

/// Discover safetensor files from a path (file or directory).
pub fn discover_safetensor_files(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        match path.extension().and_then(|s| s.to_str()) {
            Some("safetensors") => Ok(vec![path.to_path_buf()]),
            _ => anyhow::bail!("Expected .safetensors file, got: {}", path.display()),
        }
    } else if path.is_dir() {
        let mut files: Vec<PathBuf> = fs::read_dir(path)
            .with_context(|| format!("Failed to read directory: {}", path.display()))?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .collect();

        if files.is_empty() {
            anyhow::bail!("No .safetensors files found in: {}", path.display());
        }
        files.sort_unstable();
        Ok(files)
    } else {
        anyhow::bail!("Path does not exist: {}", path.display())
    }
}

/// Load and validate a GMAT model directory, returning the directory path and parsed metadata.
pub fn load_gmat_model(model_path: &str) -> Result<(PathBuf, serde_json::Value)> {
    let model_dir = Path::new(model_path);
    if !model_dir.is_dir() {
        anyhow::bail!("Model path must be a directory: {}", model_path);
    }

    let metadata_path = model_dir.join("metadata.json");
    let metadata_json = fs::read_to_string(&metadata_path)
        .with_context(|| format!("Failed to read metadata.json in {}", model_path))?;

    let metadata: serde_json::Value = serde_json::from_str(&metadata_json)
        .context("Failed to parse metadata.json")?;

    Ok((model_dir.to_path_buf(), metadata))
}

/// Load a JSON config file, requiring it to exist.
pub fn load_config<T: DeserializeOwned>(config_path: Option<&str>, config_name: &str) -> Result<T> {
    let path = config_path.ok_or_else(|| {
        anyhow::anyhow!("--config is required. Use --generate-config to create {}", config_name)
    })?;

    let json = fs::read_to_string(path)
        .with_context(|| format!("Failed to read config: {}", path))?;

    serde_json::from_str(&json)
        .with_context(|| format!("Failed to parse config: {}", path))
}

/// Convert raw bytes to f32 based on dtype.
pub fn bytes_to_f32(data: &[u8], dtype: safetensors::Dtype, count: usize) -> Result<Vec<f32>> {
    use safetensors::Dtype;

    match dtype {
        Dtype::F32 => {
            if data.len() != count * 4 {
                anyhow::bail!("F32 size mismatch: expected {} bytes, got {}", count * 4, data.len());
            }
            Ok(data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        }
        Dtype::F16 => convert_f16_to_f32(data, count),
        Dtype::BF16 => convert_bf16_to_f32(data, count),
        _ => anyhow::bail!("Unsupported dtype: {:?}. Only F32, F16, BF16 supported.", dtype),
    }
}

fn convert_f16_to_f32(data: &[u8], count: usize) -> Result<Vec<f32>> {
    if data.len() != count * 2 {
        anyhow::bail!("F16 size mismatch: expected {} bytes, got {}", count * 2, data.len());
    }
    // Safety: We verified the length, and f16 is 2 bytes
    let slice = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const half::f16, count)
    };
    Ok(slice.iter().map(|v| v.to_f32()).collect())
}

fn convert_bf16_to_f32(data: &[u8], count: usize) -> Result<Vec<f32>> {
    if data.len() != count * 2 {
        anyhow::bail!("BF16 size mismatch: expected {} bytes, got {}", count * 2, data.len());
    }
    // Safety: We verified the length, and bf16 is 2 bytes
    let slice = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const half::bf16, count)
    };
    Ok(slice.iter().map(|v| v.to_f32()).collect())
}
