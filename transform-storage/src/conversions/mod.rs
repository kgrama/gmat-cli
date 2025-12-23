//! Conversion utilities for GraphMatrix to various inference formats.
//!
//! This module provides conversions to:
//! - Quantized tensors (i4-i16, f4-f16) with log-aware scale optimization
//! - Sparse tensor formats (CSR, COO)
//! - Log-sparse representations
//! - GGUF quantization formats (Q4_K, Q8_0, etc.) for llama.cpp

mod sparse;
mod log_sparse;
mod safetensors_import;
mod export_helpers;

mod quant;
pub mod gguf_quant;

// Sparse formats
pub use sparse::{CsrMatrix, CooMatrix};
pub use log_sparse::{LogSparseMatrix, LogSparseCsrMatrix};

// Unified quantization API
pub use quant::{
    QuantDType, QuantParams, QuantizedTensors, PackFormat,
    ActivationStats, StaticSaliency,
    quantize, quantize_with_center, quantize_with_activations, quantize_with_saliency,
};

// SafeTensor import
pub use safetensors_import::{extract_metadata, extract_metadata_from_bytes};

// Export helpers
pub use export_helpers::{prepare_export_metadata, prepare_export_metadata_with_dtype};
