//! gmat-machete - Pruning, sparsity, and model compression utilities for GMAT.
//!
//! This crate provides tools for reducing model size and complexity through:
//! - Weight pruning (magnitude, gradient, structured)
//! - Sparsity analysis and enforcement
//! - Expert pruning for MoE models
//! - Knowledge distillation support
//! - Low-rank approximation

pub mod pruning;
pub mod sparsity;
