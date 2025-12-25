//! K-Quant formats (Q2_K through Q6_K)
//!
//! K-quants use 256-element super-blocks with groups of elements.
//! Each group has per-group scales enabling finer quantization control.

mod config;
mod encode;
mod scales;

pub use config::*;
pub use encode::*;
pub use scales::*;
