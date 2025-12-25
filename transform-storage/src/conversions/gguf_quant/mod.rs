//! GGUF Quantization Module
//!
//! Export GMAT log-domain blocks to standard GGUF format for llama.cpp inference.
//! All output is 100% GGUF-compliant - trellis optimization only affects scale selection.

mod iquant;
mod kquant;
mod legacy;
mod quantize;
mod trellis;
mod types;
mod utils;

pub use quantize::{compute_tensor_importance, quantize_to_gguf};
pub use types::*;

#[cfg(test)]
mod tests;
