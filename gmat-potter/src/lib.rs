//! gmat-potter - Transform, tracing, and signal processing utilities for GMAT.
//!
//! This crate provides tools for analyzing and transforming model weights:
//! - Trellis quantization optimization
//! - Sparse-to-sparse path tracing
//! - DCT/DST frequency analysis
//! - Wavelet transforms
//! - Weight visualization

pub mod tracing;
pub mod transforms;
pub mod trellis;
