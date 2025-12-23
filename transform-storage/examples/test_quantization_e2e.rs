//! End-to-end quantization tests for all block formats → I6 output.
//!
//! Tests the full pipeline: f32 data → GraphMatrix (block encoded) → quantize I6 → verify

use candle_core::Device;
use transform_storage::{GraphMatrix, BlockFormat, QuantDType, PackFormat, quantize};

fn test_block_format(name: &str, format: BlockFormat, data: &[f32], shape: (usize, usize)) {
    let matrix = GraphMatrix::from_dense(data, shape, format);
    let device = Device::Cpu;

    // Get block-decoded values (what the matrix actually stores after block encoding)
    let block_decoded: Vec<f32> = matrix.to_tensor(&device).unwrap()
        .flatten_all().unwrap().to_vec1().unwrap();

    // Quantize to I6 with simple packing
    let result = quantize(&matrix, QuantDType::I6, PackFormat::None, 0.0, &device).unwrap();

    let quantized: Vec<i64> = result.weights.flatten_all().unwrap().to_vec1().unwrap();
    let scale = result.params.scale.unwrap_or(1.0);

    // Compute errors relative to block-decoded values (not original)
    let mut max_block_error = 0.0f32;
    let mut max_quant_error = 0.0f32;
    let mut total_quant_error = 0.0f32;
    let mut count = 0;

    for i in 0..data.len() {
        let orig = data[i];
        let block_val = block_decoded[i];
        let dequant = quantized[i] as f32 * scale;

        if orig.abs() > 1e-10 {
            // Block encoding error (original → block)
            let block_err = (orig - block_val).abs() / orig.abs();
            max_block_error = max_block_error.max(block_err);

            // Quantization error (block → dequantized)
            if block_val.abs() > 1e-10 {
                let quant_err = (block_val - dequant).abs() / block_val.abs();
                max_quant_error = max_quant_error.max(quant_err);
                total_quant_error += quant_err;
                count += 1;
            }
        }
    }

    let avg_quant_error = if count > 0 { total_quant_error / count as f32 } else { 0.0 };

    // Check I6 range [-32, 31]
    let in_range = quantized.iter().all(|&q| q >= -32 && q <= 31);

    println!("{:20} | shape {:?} | block_err: {:5.2}% | quant_err: {:5.2}% (avg {:5.2}%) | I6 range: {}",
             name, shape,
             max_block_error * 100.0,
             max_quant_error * 100.0,
             avg_quant_error * 100.0,
             if in_range { "OK" } else { "FAIL" });

    assert!(in_range, "{} has values outside I6 range", name);
}

fn main() {
    println!("=== End-to-End Quantization Tests (→ I6) ===\n");

    // Test data: values within ~3 octave range (fits well in all block formats)
    // This simulates realistic LLM weight distributions
    let test_data_16: Vec<f32> = vec![
        0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5,
        -0.6, -0.8, -1.1, -1.3, -1.6, -1.9, -2.2, -2.8,
    ];

    let test_data_32: Vec<f32> = vec![
        0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5,
        -0.6, -0.8, -1.1, -1.3, -1.6, -1.9, -2.2, -2.8,
        0.4, 0.6, 0.9, 1.1, 1.4, 1.7, 2.1, 2.4,
        -0.5, -0.7, -1.0, -1.2, -1.5, -1.8, -2.0, -2.6,
    ];

    let test_data_64: Vec<f32> = (0..64)
        .map(|i| {
            let base = 0.5 + (i % 16) as f32 * 0.15;
            if i % 2 == 0 { base } else { -base }
        })
        .collect();

    println!("Single-row block formats:");
    println!("{:-<90}", "");

    // B16x8: 16 elements, 8-bit magnitude (e1m7), ~4 octave range
    test_block_format("B16x8", BlockFormat::B16x8, &test_data_16, (1, 16));

    // B16x4: 16 elements, 4-bit magnitude (e0m4), ~2 octave range
    test_block_format("B16x4", BlockFormat::B16x4, &test_data_16, (1, 16));

    // B8x8: 8 elements, 8-bit magnitude (e1m7), ~4 octave range
    test_block_format("B8x8", BlockFormat::B8x8, &test_data_16, (2, 8));

    // B8x4: 8 elements, 4-bit magnitude (e0m4), ~2 octave range
    test_block_format("B8x4", BlockFormat::B8x4, &test_data_16, (2, 8));

    println!("\nDual-row block formats:");
    println!("{:-<90}", "");

    // DualRow16x8: 2 rows × 16 elements, shared scale
    test_block_format("DualRow16x8", BlockFormat::DualRow16x8, &test_data_32, (2, 16));

    // DualRow16x4: 2 rows × 16 elements, shared scale
    test_block_format("DualRow16x4", BlockFormat::DualRow16x4, &test_data_32, (2, 16));

    // DualRow8x8: 2 rows × 8 elements, shared scale
    test_block_format("DualRow8x8", BlockFormat::DualRow8x8, &test_data_32, (4, 8));

    // DualRow8x4: 2 rows × 8 elements, shared scale
    test_block_format("DualRow8x4", BlockFormat::DualRow8x4, &test_data_32, (4, 8));

    println!("\nLarger matrices (multiple blocks):");
    println!("{:-<90}", "");

    // Multiple blocks - 4 rows × 16 cols = 4 B16x8 blocks
    test_block_format("B16x8 (4×16)", BlockFormat::B16x8, &test_data_64, (4, 16));

    // Multiple dual-row blocks - 4 rows × 16 cols = 2 DualRow16x8 blocks
    test_block_format("DualRow16x8 (4×16)", BlockFormat::DualRow16x8, &test_data_64, (4, 16));

    // 8 rows × 8 cols = 8 B8x8 blocks
    test_block_format("B8x8 (8×8)", BlockFormat::B8x8, &test_data_64, (8, 8));

    // 8 rows × 8 cols = 4 DualRow8x8 blocks
    test_block_format("DualRow8x8 (8×8)", BlockFormat::DualRow8x8, &test_data_64, (8, 8));

    println!("\n=== All tests passed ===");
}
