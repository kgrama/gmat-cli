//! Conversion utilities for GraphMatrix to various inference formats.
//!
//! This module provides conversions to:
//! - Quantized tensors (i4-i16, f4-f16) with log-aware scale optimization
//! - Sparse tensor formats (CSR, COO)
//! - Log-sparse representations
//! - GGUF quantization formats (Q4_K, Q8_0, etc.) for llama.cpp

mod export_helpers;
mod log_sparse;
mod safetensors_import;
mod sparse;

pub mod gguf_quant;
mod quant;

// Sparse formats
pub use log_sparse::{LogSparseCsrMatrix, LogSparseMatrix};
pub use sparse::{CooMatrix, CsrMatrix};

// Unified quantization API
pub use quant::{
    quantize, quantize_with_activations, quantize_with_center, quantize_with_saliency,
    ActivationStats, PackFormat, QuantDType, QuantParams, QuantizedTensors, StaticSaliency,
};

// SafeTensor import
pub use safetensors_import::{extract_metadata, extract_metadata_from_bytes};

// Export helpers
pub use export_helpers::{prepare_export_metadata, prepare_export_metadata_with_dtype};
