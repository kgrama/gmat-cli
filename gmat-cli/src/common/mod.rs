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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ==================== discover_safetensor_files tests ====================

    #[test]
    fn test_discover_single_safetensor_file() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("model.safetensors");
        fs::File::create(&file_path).unwrap();

        let result = discover_safetensor_files(&file_path).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], file_path);
    }

    #[test]
    fn test_discover_rejects_non_safetensor_file() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("model.bin");
        fs::File::create(&file_path).unwrap();

        let result = discover_safetensor_files(&file_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Expected .safetensors file"));
    }

    #[test]
    fn test_discover_directory_with_safetensors() {
        let dir = TempDir::new().unwrap();
        fs::File::create(dir.path().join("model-00001.safetensors")).unwrap();
        fs::File::create(dir.path().join("model-00002.safetensors")).unwrap();
        fs::File::create(dir.path().join("config.json")).unwrap(); // should be ignored

        let result = discover_safetensor_files(dir.path()).unwrap();
        assert_eq!(result.len(), 2);
        // Should be sorted
        assert!(result[0].to_string_lossy().contains("00001"));
        assert!(result[1].to_string_lossy().contains("00002"));
    }

    #[test]
    fn test_discover_empty_directory_fails() {
        let dir = TempDir::new().unwrap();

        let result = discover_safetensor_files(dir.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No .safetensors files found"));
    }

    #[test]
    fn test_discover_nonexistent_path_fails() {
        let result = discover_safetensor_files(Path::new("/nonexistent/path"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Path does not exist"));
    }

    // ==================== load_gmat_model tests ====================

    #[test]
    fn test_load_gmat_model_success() {
        let dir = TempDir::new().unwrap();
        let metadata = r#"{"architecture": "llama", "layers": 32}"#;
        fs::write(dir.path().join("metadata.json"), metadata).unwrap();

        let (path, parsed) = load_gmat_model(dir.path().to_str().unwrap()).unwrap();
        assert_eq!(path, dir.path());
        assert_eq!(parsed["architecture"], "llama");
        assert_eq!(parsed["layers"], 32);
    }

    #[test]
    fn test_load_gmat_model_not_directory() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("model.gmat");
        fs::File::create(&file_path).unwrap();

        let result = load_gmat_model(file_path.to_str().unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be a directory"));
    }

    #[test]
    fn test_load_gmat_model_missing_metadata() {
        let dir = TempDir::new().unwrap();

        let result = load_gmat_model(dir.path().to_str().unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("metadata.json"));
    }

    #[test]
    fn test_load_gmat_model_invalid_json() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("metadata.json"), "not valid json").unwrap();

        let result = load_gmat_model(dir.path().to_str().unwrap());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("parse"));
    }

    // ==================== load_config tests ====================

    #[test]
    fn test_load_config_success() {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("config.json");
        fs::write(&config_path, r#"{"name": "test", "value": 42}"#).unwrap();

        #[derive(serde::Deserialize)]
        struct TestConfig {
            name: String,
            value: i32,
        }

        let config: TestConfig = load_config(Some(config_path.to_str().unwrap()), "config.json").unwrap();
        assert_eq!(config.name, "test");
        assert_eq!(config.value, 42);
    }

    #[test]
    fn test_load_config_none_path_fails() {
        #[derive(Debug, serde::Deserialize)]
        struct TestConfig {}

        let result: Result<TestConfig> = load_config(None, "my_config.json");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("--config is required"));
    }

    #[test]
    fn test_load_config_missing_file() {
        #[derive(serde::Deserialize)]
        struct TestConfig {}

        let result: Result<TestConfig> = load_config(Some("/nonexistent/config.json"), "config.json");
        assert!(result.is_err());
    }

    // ==================== bytes_to_f32 tests ====================

    #[test]
    fn test_bytes_to_f32_f32() {
        use safetensors::Dtype;

        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let result = bytes_to_f32(&bytes, Dtype::F32, 4).unwrap();
        assert_eq!(result, values);
    }

    #[test]
    fn test_bytes_to_f32_f32_size_mismatch() {
        use safetensors::Dtype;

        let bytes = vec![0u8; 12]; // 3 floats worth of bytes
        let result = bytes_to_f32(&bytes, Dtype::F32, 4); // but we say 4
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("size mismatch"));
    }

    #[test]
    fn test_bytes_to_f32_f16() {
        use safetensors::Dtype;

        let f16_values: Vec<half::f16> = vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
        ];
        let bytes: Vec<u8> = f16_values.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let result = bytes_to_f32(&bytes, Dtype::F16, 2).unwrap();
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_bytes_to_f32_bf16() {
        use safetensors::Dtype;

        let bf16_values: Vec<half::bf16> = vec![
            half::bf16::from_f32(1.5),
            half::bf16::from_f32(3.0),
        ];
        let bytes: Vec<u8> = bf16_values.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let result = bytes_to_f32(&bytes, Dtype::BF16, 2).unwrap();
        assert!((result[0] - 1.5).abs() < 0.01);
        assert!((result[1] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_bytes_to_f32_unsupported_dtype() {
        use safetensors::Dtype;

        let bytes = vec![0u8; 8];
        let result = bytes_to_f32(&bytes, Dtype::I64, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unsupported dtype"));
    }

    #[test]
    fn test_bytes_to_f32_f16_size_mismatch() {
        use safetensors::Dtype;

        let bytes = vec![0u8; 4]; // 2 f16 values
        let result = bytes_to_f32(&bytes, Dtype::F16, 3); // but we say 3
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("size mismatch"));
    }

    #[test]
    fn test_bytes_to_f32_bf16_size_mismatch() {
        use safetensors::Dtype;

        let bytes = vec![0u8; 4]; // 2 bf16 values
        let result = bytes_to_f32(&bytes, Dtype::BF16, 3); // but we say 3
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("size mismatch"));
    }
}
