//! GGUF validation via llama.cpp.
//!
//! This module provides validation of exported GGUF files by attempting
//! to load them with llama.cpp. Requires the `validate` feature.

use anyhow::Result;

/// Validate a GGUF file by loading it with llama.cpp.
///
/// This function attempts to load the GGUF file to verify:
/// - File structure is valid
/// - Metadata is parseable
/// - Tensor data is accessible
///
/// # Errors
///
/// Returns an error if:
/// - The `validate` feature is not enabled
/// - The file cannot be loaded by llama.cpp
#[cfg(feature = "validate")]
pub fn validate_gguf(path: &str) -> Result<()> {
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::model::LlamaModel;
    use llama_cpp_2::model::params::LlamaModelParams;
    use std::path::Path;

    println!("\nValidating GGUF: {}", path);

    let path = Path::new(path);
    if !path.exists() {
        anyhow::bail!("GGUF file not found: {}", path.display());
    }

    // Initialize llama.cpp backend
    let backend = LlamaBackend::init().map_err(|e| anyhow::anyhow!("Failed to init llama backend: {:?}", e))?;

    let params = LlamaModelParams::default();

    match LlamaModel::load_from_file(&backend, path, &params) {
        Ok(_model) => {
            println!("  GGUF validation passed!");
            println!("  Model loaded successfully by llama.cpp");
            Ok(())
        }
        Err(e) => {
            anyhow::bail!("GGUF validation failed: {:?}", e);
        }
    }
}

/// Stub implementation when validate feature is not enabled.
#[cfg(not(feature = "validate"))]
pub fn validate_gguf(_path: &str) -> Result<()> {
    anyhow::bail!(
        "GGUF validation requires the 'validate' feature.\n\
         Rebuild with: cargo build --features validate\n\n\
         Note: This requires a C++ toolchain and libclang to build llama.cpp."
    )
}
