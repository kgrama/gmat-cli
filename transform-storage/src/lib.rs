//! Transform Storage - Block-based sparse coefficient storage.
//!
//! Storage format for sparse coefficients using block-based representation.
//! Domain independent - works for Walsh, DCT, or any sparse tensor.
//!
//! # Block Formats
//! - `Block16x8`: 16 elements, 8-bit magnitudes (e1m7), 22 bytes non-empty, 4 octave range
//! - `Block16x4`: 16 elements, 4-bit magnitudes (e0m4), 14 bytes non-empty, 1 octave range
//! - `Block8x8`: 8 elements, 8-bit magnitudes (e1m7), 12 bytes non-empty, 4 octave range
//! - `Block8x4`: 8 elements, 4-bit magnitudes (e0m4), 8 bytes non-empty, 1 octave range
//! - Empty blocks: only 2 bytes (scale_log sentinel)
//!
//! # Usage
//! ```ignore
//! use transform_storage::{GraphMatrix, Block16x8, StorageConfig, BlockFormat};
//!
//! // From dense data
//! let data = vec![1.0, 0.0, 2.0, 0.0, /* ... */];
//! let matrix: GraphMatrix<Block16x8> = GraphMatrix::from_dense(&data, (4, 4));
//!
//! // With config and type erasure
//! let config = StorageConfig::new()
//!     .format(BlockFormat::Block16x4);
//! let any_matrix = AnyGraphMatrix::from_dense(&data, (4, 4), &config);
//! ```

pub mod block;
pub mod blocks;
pub mod config;
pub mod conversions;
pub mod formats;
pub mod graph_matrix;

#[cfg(test)]
mod graph_matrix_tests;

// Re-exports
pub use block::Block;
pub use blocks::{Block16x8, Block16x4, Block8x8, Block8x4, BlockFormat};
pub use config::{AnyGraphMatrix, StorageConfig};
pub use conversions::{CooMatrix, CsrMatrix, LogSparseMatrix, LogSparseCsrMatrix, QuantDType, QuantParams, QuantizedTensors, PackFormat, quantize, quantize_with_center, extract_metadata, extract_metadata_from_bytes, prepare_export_metadata, prepare_export_metadata_with_dtype};
pub use formats::{GmatHeader, GmatMetadata, metadata_keys, GMAT_MAGIC, GMAT_VERSION, GMAT_VERSION_V1, GMAT_VERSION_V2};
pub use graph_matrix::GraphMatrix;


