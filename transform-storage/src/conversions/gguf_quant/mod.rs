//! GGUF Quantization Module
//!
//! Export GMAT log-domain blocks to standard GGUF format for llama.cpp inference.
//! All output is 100% GGUF-compliant - trellis optimization only affects scale selection.

mod types;
mod legacy;
mod kquant;
mod trellis;
mod utils;
mod iquant;
mod quantize;

pub use types::*;
pub use quantize::{quantize_to_gguf, compute_tensor_importance};

#[cfg(test)]
mod tests;
