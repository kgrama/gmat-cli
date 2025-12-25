//! Tests for GraphMatrix

use super::blocks::{AnyBlock, BlockFormat};
use super::graph_matrix::GraphMatrix;
use candle_core::{Device, Tensor};
use tempfile::NamedTempFile;

// ========================================================================
// from_dense tests
// ========================================================================

#[test]
fn test_from_dense_basic() {
    let data = vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0];
    let matrix = GraphMatrix::from_dense(&data, (2, 4), BlockFormat::B16x8);

    assert_eq!(matrix.shape(), (2, 4));
    assert_eq!(matrix.nnz(), 4);
}

#[test]
fn test_from_dense_all_zeros() {
    let data = vec![0.0f32; 64];
    let matrix = GraphMatrix::from_dense(&data, (4, 16), BlockFormat::B16x8);

    assert_eq!(matrix.shape(), (4, 16));
    assert_eq!(matrix.nnz(), 0);
    assert_eq!(matrix.density(), 0.0);
}

#[test]
fn test_from_dense_all_nonzero() {
    let data: Vec<f32> = (1..=32).map(|x| x as f32).collect();
    let matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);

    assert_eq!(matrix.shape(), (2, 16));
    assert_eq!(matrix.nnz(), 32);
    assert_eq!(matrix.density(), 1.0);
}

#[test]
fn test_from_dense_partial_blocks() {
    // 20 columns = 2 full blocks (16) + partial block (4)
    let data = vec![1.0f32; 20];
    let matrix = GraphMatrix::from_dense(&data, (1, 20), BlockFormat::B16x8);

    assert_eq!(matrix.shape(), (1, 20));
    assert_eq!(matrix.nnz(), 20);
}

#[test]
#[should_panic(expected = "Data length")]
fn test_from_dense_mismatched_length() {
    let data = vec![1.0f32; 10];
    let _matrix = GraphMatrix::from_dense(&data, (2, 8), BlockFormat::B16x8);
}

// ========================================================================
// from_blocks tests
// ========================================================================

#[test]
fn test_from_blocks_valid() {
    let blocks = vec![
        AnyBlock::encode_16x8(&[1.0; 16]),
        AnyBlock::encode_16x8(&[2.0; 16]),
    ];
    let matrix = GraphMatrix::from_blocks(blocks, (2, 16), BlockFormat::B16x8);

    assert_eq!(matrix.shape(), (2, 16));
}

#[test]
#[should_panic(expected = "Block count mismatch")]
fn test_from_blocks_wrong_count() {
    let blocks = vec![AnyBlock::encode_16x8(&[1.0; 16])];
    let _matrix = GraphMatrix::from_blocks(blocks, (2, 16), BlockFormat::B16x8);
    // expects 2 blocks
}

// ========================================================================
// nnz and density tests
// ========================================================================

#[test]
fn test_nnz_sparse() {
    let mut data = vec![0.0f32; 32];
    data[0] = 1.0;
    data[15] = 2.0;
    data[16] = 3.0;
    data[31] = 4.0;

    let matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);
    assert_eq!(matrix.nnz(), 4);
}

#[test]
fn test_density_half() {
    let mut data = vec![0.0f32; 16];
    for i in 0..8 {
        data[i] = (i + 1) as f32;
    }

    let matrix = GraphMatrix::from_dense(&data, (1, 16), BlockFormat::B16x8);
    assert_eq!(matrix.density(), 0.5);
}

#[test]
fn test_density_empty_matrix() {
    let matrix = GraphMatrix::from_dense(&[], (0, 0), BlockFormat::B16x8);
    assert_eq!(matrix.density(), 0.0);
}

// ========================================================================
// row_iter tests
// ========================================================================

#[test]
fn test_row_iter_basic() {
    let mut data = vec![0.0f32; 32];
    data[0] = 1.0; // row 0, col 0
    data[5] = 2.0; // row 0, col 5
    data[16] = 3.0; // row 1, col 0
    data[20] = 4.0; // row 1, col 4

    let matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);

    // Check row 0
    let row0: Vec<(usize, f32)> = matrix.row_iter(0).collect();
    assert_eq!(row0.len(), 2);
    assert_eq!(row0[0].0, 0); // column index
    assert_eq!(row0[1].0, 5); // column index

    // Check row 1
    let row1: Vec<(usize, f32)> = matrix.row_iter(1).collect();
    assert_eq!(row1.len(), 2);
    assert_eq!(row1[0].0, 0);
    assert_eq!(row1[1].0, 4);
}

#[test]
fn test_row_iter_empty_row() {
    let mut data = vec![0.0f32; 32];
    data[0] = 1.0; // only row 0 has data

    let matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);

    let row1: Vec<_> = matrix.row_iter(1).collect();
    assert!(row1.is_empty());
}

#[test]
fn test_row_iter_partial_block() {
    // 20 columns = 2 blocks, but second block only has 4 valid elements
    let mut data = vec![0.0f32; 20];
    data[18] = 5.0; // col 18 in partial block

    let matrix = GraphMatrix::from_dense(&data, (1, 20), BlockFormat::B16x8);

    let row: Vec<_> = matrix.row_iter(0).collect();
    assert_eq!(row.len(), 1);
    assert_eq!(row[0].0, 18);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn test_row_iter_out_of_bounds() {
    let data = vec![1.0f32; 16];
    let matrix = GraphMatrix::from_dense(&data, (1, 16), BlockFormat::B16x8);
    let _: Vec<_> = matrix.row_iter(1).collect(); // row 1 doesn't exist
}

// ========================================================================
// block_iter tests
// ========================================================================

#[test]
fn test_block_iter() {
    let data = vec![1.0f32; 48]; // 3 rows x 16 cols = 3 blocks
    let matrix = GraphMatrix::from_dense(&data, (3, 16), BlockFormat::B16x8);

    let blocks: Vec<_> = matrix.block_iter().collect();
    assert_eq!(blocks.len(), 3);
}

// ========================================================================
// col_iter and col_index tests
// ========================================================================

#[test]
fn test_col_index_not_built() {
    let data = vec![1.0f32; 16];
    let matrix = GraphMatrix::from_dense(&data, (1, 16), BlockFormat::B16x8);

    assert!(!matrix.has_col_index());
}

#[test]
fn test_build_col_index() {
    let data = vec![1.0f32; 32];
    let mut matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);

    assert!(!matrix.has_col_index());
    matrix.build_col_index();
    assert!(matrix.has_col_index());
}

#[test]
fn test_col_iter_basic() {
    let mut data = vec![0.0f32; 32];
    data[0] = 1.0; // row 0, col 0
    data[16] = 2.0; // row 1, col 0
    data[5] = 3.0; // row 0, col 5

    let mut matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);
    matrix.build_col_index();

    // Check col 0
    let col0: Vec<(usize, f32)> = matrix.col_iter(0).collect();
    assert_eq!(col0.len(), 2);

    // Check col 5
    let col5: Vec<(usize, f32)> = matrix.col_iter(5).collect();
    assert_eq!(col5.len(), 1);
    assert_eq!(col5[0].0, 0); // row index
}

#[test]
fn test_drop_col_index() {
    let data = vec![1.0f32; 32];
    let mut matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);

    matrix.build_col_index();
    assert!(matrix.has_col_index());

    matrix.drop_col_index();
    assert!(!matrix.has_col_index());
}

#[test]
#[should_panic(expected = "Column index not built")]
fn test_col_iter_without_index() {
    let data = vec![1.0f32; 16];
    let matrix = GraphMatrix::from_dense(&data, (1, 16), BlockFormat::B16x8);
    let _: Vec<_> = matrix.col_iter(0).collect();
}

// ========================================================================
// memory_bytes tests
// ========================================================================

#[test]
fn test_memory_bytes_empty_blocks() {
    let data = vec![0.0f32; 32];
    let matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);

    // 2 empty blocks = 2 * 2 bytes = 4 bytes + vec overhead
    let mem = matrix.memory_bytes();
    assert!(mem >= 4);
}

#[test]
fn test_memory_bytes_nonempty_blocks() {
    // Use varied values to avoid scale_log = 0 (which equals EMPTY_SCALE)
    let data: Vec<f32> = (1..=32).map(|x| x as f32).collect();
    let matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);

    // 2 non-empty Block16x8 = 2 * 22 bytes = 44 bytes + vec overhead
    let mem = matrix.memory_bytes();
    assert!(mem >= 44, "Expected >= 44 bytes, got {}", mem);
}

#[test]
fn test_memory_bytes_with_col_index() {
    let data = vec![1.0f32; 32];
    let mut matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);

    let mem_before = matrix.memory_bytes();
    matrix.build_col_index();
    let mem_after = matrix.memory_bytes();

    assert!(mem_after > mem_before);
}

// ========================================================================
// save/load tests
// ========================================================================

#[test]
fn test_save_load_roundtrip_block16x8() {
    let data: Vec<f32> = (1..=32).map(|x| x as f32).collect();
    let matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);

    let temp_file = NamedTempFile::new().unwrap();
    matrix.save(temp_file.path()).unwrap();

    let loaded = GraphMatrix::load(temp_file.path()).unwrap();

    assert_eq!(matrix.shape(), loaded.shape());
    assert_eq!(matrix.nnz(), loaded.nnz());
}

#[test]
fn test_save_load_roundtrip_block16x4() {
    let data: Vec<f32> = (1..=32).map(|x| x as f32).collect();
    let matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x4);

    let temp_file = NamedTempFile::new().unwrap();
    matrix.save(temp_file.path()).unwrap();

    let loaded = GraphMatrix::load(temp_file.path()).unwrap();

    assert_eq!(matrix.shape(), loaded.shape());
    assert_eq!(matrix.nnz(), loaded.nnz());
}

#[test]
fn test_save_load_roundtrip_block8x4() {
    let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let matrix = GraphMatrix::from_dense(&data, (2, 8), BlockFormat::B8x4);

    let temp_file = NamedTempFile::new().unwrap();
    matrix.save(temp_file.path()).unwrap();

    let loaded = GraphMatrix::load(temp_file.path()).unwrap();

    assert_eq!(matrix.shape(), loaded.shape());
    assert_eq!(matrix.nnz(), loaded.nnz());
}

#[test]
fn test_save_load_empty_blocks() {
    let data = vec![0.0f32; 32];
    let matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);

    let temp_file = NamedTempFile::new().unwrap();
    matrix.save(temp_file.path()).unwrap();

    let loaded = GraphMatrix::load(temp_file.path()).unwrap();

    assert_eq!(loaded.nnz(), 0);
}

#[test]
fn test_save_load_sparse_data() {
    let mut data = vec![0.0f32; 64];
    data[0] = 1.0;
    data[15] = 2.0;
    data[32] = 3.0;
    data[63] = 4.0;

    let matrix = GraphMatrix::from_dense(&data, (4, 16), BlockFormat::B16x8);

    let temp_file = NamedTempFile::new().unwrap();
    matrix.save(temp_file.path()).unwrap();

    let loaded = GraphMatrix::load(temp_file.path()).unwrap();

    assert_eq!(matrix.nnz(), loaded.nnz());
    assert_eq!(loaded.nnz(), 4);
}

// ========================================================================
// Tensor conversion tests
// ========================================================================

#[test]
fn test_tensor_roundtrip() {
    let device = Device::Cpu;
    // e1m7 has 4 octave range, can handle simple linear values
    let original_data: Vec<f32> = (1..=32).map(|x| x as f32).collect();

    let original = Tensor::from_vec(original_data.clone(), (4, 8), &device).unwrap();

    let matrix = GraphMatrix::from_tensor(&original, BlockFormat::B8x8).unwrap();
    let restored = matrix.to_tensor(&device).unwrap();

    assert_eq!(restored.dims(), &[4, 8]);

    // Values should be approximately equal (quantization error)
    let restored_data: Vec<f32> = restored.flatten_all().unwrap().to_vec1().unwrap();
    for (orig, rest) in original_data.iter().zip(restored_data.iter()) {
        if *orig == 0.0 {
            assert_eq!(*rest, 0.0);
        } else {
            let ratio = rest / orig;
            // e1m7 has ~4 octave range, expect ratio within 0.9-1.1
            assert!(
                (0.9..1.1).contains(&ratio),
                "orig={}, restored={}",
                orig,
                rest
            );
        }
    }
}

#[test]
fn test_from_tensor_shape() {
    let device = Device::Cpu;
    let tensor = Tensor::zeros((4, 20), candle_core::DType::F32, &device).unwrap();

    let matrix = GraphMatrix::from_tensor(&tensor, BlockFormat::B16x8).unwrap();
    assert_eq!(matrix.shape(), (4, 20));
}

// ========================================================================
// Different block type tests
// ========================================================================

#[test]
fn test_block8x4_matrix() {
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let matrix = GraphMatrix::from_dense(&data, (3, 8), BlockFormat::B8x4);

    assert_eq!(matrix.shape(), (3, 8));
    assert_eq!(matrix.nnz(), 24);
}

#[test]
fn test_block16x4_matrix() {
    let data: Vec<f32> = (1..=48).map(|x| x as f32).collect();
    let matrix = GraphMatrix::from_dense(&data, (3, 16), BlockFormat::B16x4);

    assert_eq!(matrix.shape(), (3, 16));
    assert_eq!(matrix.nnz(), 48);
}

// ========================================================================
// Log-sparse reconstruction error tests
// ========================================================================

/// Compute relative reconstruction error: |original - reconstructed| / |original|
fn relative_error(original: f32, reconstructed: f32) -> f32 {
    if original == 0.0 {
        if reconstructed == 0.0 {
            0.0
        } else {
            f32::INFINITY
        }
    } else {
        (original - reconstructed).abs() / original.abs()
    }
}

#[test]
fn test_log_sparse_reconstruction_error_block8x8() {
    // Use two Block8x8 blocks for 16 values - each block handles narrower range
    // Values 2-16 span too many octaves for single block's offset range from median
    let data: [f32; 16] = [
        2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0, -14.0, 16.0, 0.0, 0.0, 0.0,
    ];

    // Split into two matrices with Block8x8
    let matrix1 = GraphMatrix::from_dense(&data[0..8], (1, 8), BlockFormat::B8x8);
    let matrix2 = GraphMatrix::from_dense(&data[8..16], (1, 8), BlockFormat::B8x8);

    let log_sparse1 = matrix1.to_log_sparse();
    let log_sparse2 = matrix2.to_log_sparse();
    let linear1 = log_sparse1.to_linear();
    let linear2 = log_sparse2.to_linear();

    let mut max_error: f32 = 0.0;
    let mut sum_error: f32 = 0.0;
    let mut count = 0;

    // Check first block
    for (&orig, &recon) in data[0..8].iter().zip(linear1.values.iter()) {
        if orig != 0.0 {
            assert_eq!(
                orig.signum(),
                recon.signum(),
                "Sign mismatch: original={}, reconstructed={}",
                orig,
                recon
            );
            let err = relative_error(orig, recon);
            max_error = max_error.max(err);
            sum_error += err;
            count += 1;
        }
    }

    // Check second block
    for (&orig, &recon) in data[8..16].iter().zip(linear2.values.iter()) {
        if orig != 0.0 {
            assert_eq!(
                orig.signum(),
                recon.signum(),
                "Sign mismatch: original={}, reconstructed={}",
                orig,
                recon
            );
            let err = relative_error(orig, recon);
            max_error = max_error.max(err);
            sum_error += err;
            count += 1;
        }
    }

    let avg_error = sum_error / count as f32;

    // e1m7 (8-bit) should have < 2% average error for values within each block's range
    println!(
        "Block8x8 (e1m7): avg_error={:.4}%, max_error={:.4}%",
        avg_error * 100.0,
        max_error * 100.0
    );
    assert!(
        avg_error < 0.03,
        "Average error {:.4}% exceeds 3%",
        avg_error * 100.0
    );
    assert!(
        max_error < 0.08,
        "Max error {:.4}% exceeds 8%",
        max_error * 100.0
    );
}

#[test]
fn test_log_sparse_reconstruction_error_block16x4() {
    // e0m4 encodes offsets in [0.0, 1.0) from median, so ~1 octave range
    // Using values from 8.0 to 12.0 (factor 1.5 = 0.58 octaves)
    let data: [f32; 16] = [
        8.0, -8.5, 9.0, -9.5, 10.0, -10.5, 11.0, -11.5, 12.0, -8.2, 9.2, -10.2, 11.2, 0.0, 0.0, 0.0,
    ];

    let matrix = GraphMatrix::from_dense(&data, (1, 16), BlockFormat::B16x4);
    let log_sparse = matrix.to_log_sparse();
    let linear = log_sparse.to_linear();

    let mut max_error: f32 = 0.0;
    let mut sum_error: f32 = 0.0;
    let mut count = 0;

    for (&orig, &recon) in data.iter().zip(linear.values.iter()) {
        if orig != 0.0 {
            assert_eq!(
                orig.signum(),
                recon.signum(),
                "Sign mismatch: original={}, reconstructed={}",
                orig,
                recon
            );

            let err = relative_error(orig, recon);
            max_error = max_error.max(err);
            sum_error += err;
            count += 1;
        }
    }

    let avg_error = sum_error / count as f32;

    // e0m4 (4-bit) has lower precision, expect < 10% average error
    println!(
        "Block16x4 (e0m4): avg_error={:.4}%, max_error={:.4}%",
        avg_error * 100.0,
        max_error * 100.0
    );
    assert!(
        avg_error < 0.15,
        "Average error {:.4}% exceeds 15%",
        avg_error * 100.0
    );
    assert!(
        max_error < 0.30,
        "Max error {:.4}% exceeds 30%",
        max_error * 100.0
    );
}

#[test]
fn test_log_sparse_reconstruction_error_block8x4() {
    // e0m4 encodes offsets in [0.0, 1.0) from median, so ~1 octave range
    // Using values from 8.0 to 11.0 (factor 1.375 = 0.46 octaves)
    let data: [f32; 8] = [8.0, -8.5, 9.0, -9.5, 10.0, -10.5, 11.0, -8.2];

    let matrix = GraphMatrix::from_dense(&data, (1, 8), BlockFormat::B8x4);
    let log_sparse = matrix.to_log_sparse();
    let linear = log_sparse.to_linear();

    let mut max_error: f32 = 0.0;
    let mut sum_error: f32 = 0.0;
    let mut count = 0;

    for (&orig, &recon) in data.iter().zip(linear.values.iter()) {
        if orig != 0.0 {
            assert_eq!(
                orig.signum(),
                recon.signum(),
                "Sign mismatch: original={}, reconstructed={}",
                orig,
                recon
            );

            let err = relative_error(orig, recon);
            max_error = max_error.max(err);
            sum_error += err;
            count += 1;
        }
    }

    let avg_error = sum_error / count as f32;

    // e0m4 (4-bit) has lower precision
    println!(
        "Block8x4 (e0m4): avg_error={:.4}%, max_error={:.4}%",
        avg_error * 100.0,
        max_error * 100.0
    );
    assert!(
        avg_error < 0.15,
        "Average error {:.4}% exceeds 15%",
        avg_error * 100.0
    );
    assert!(
        max_error < 0.30,
        "Max error {:.4}% exceeds 30%",
        max_error * 100.0
    );
}

#[test]
fn test_log_sparse_csr_matches_coo() {
    // Verify CSR and COO produce same results
    let data: [f32; 16] = [
        4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0, -13.0, 14.0, -15.0, 16.0, 0.0, 0.0, 0.0,
    ];

    let matrix = GraphMatrix::from_dense(&data, (1, 16), BlockFormat::B16x8);

    let log_coo = matrix.to_log_sparse();
    let log_csr = matrix.to_log_sparse_csr();

    let linear_coo = log_coo.to_linear();
    let linear_csr = log_csr.to_linear();

    assert_eq!(linear_coo.values.len(), linear_csr.values.len());
    for (coo_val, csr_val) in linear_coo.values.iter().zip(linear_csr.values.iter()) {
        assert!(
            (coo_val - csr_val).abs() < 1e-6,
            "COO={} vs CSR={}",
            coo_val,
            csr_val
        );
    }
}
