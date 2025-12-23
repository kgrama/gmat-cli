//! K-Quant formats (Q2_K through Q6_K)
//!
//! K-quants use 256-element super-blocks with groups of elements.
//! Each group has per-group scales enabling finer quantization control.

mod config;
mod scales;
mod encode;

pub use config::*;
pub use scales::*;
pub use encode::*;
