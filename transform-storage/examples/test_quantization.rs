//! End-to-end quantization test.
//!
//! Tests the full pipeline: GraphMatrix → quantize → packed tensors → dequantize → verify error

use candle_core::Device;
use transform_storage::{GraphMatrix, BlockFormat, QuantDType, PackFormat, quantize};

fn main() {
    println!("=== Quantization End-to-End Test ===\n");

    // Test data spanning multiple orders of magnitude
    let test_values: Vec<f32> = vec![
        // Group 1: small values
        0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2,
        // Group 2: medium values
        0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0,
        // Repeat with negatives for 256 total (AWQ/GPTQ need cols divisible by 8)
        -0.001, -0.002, -0.005, -0.01, -0.02, -0.05, -0.1, -0.2,
        -0.5, -1.0, -2.0, -5.0, -10.0, -20.0, -50.0, -100.0,
    ];

    // Pad to 128 columns (minimum for AWQ/GPTQ group)
    let mut data = test_values.clone();
    while data.len() < 128 {
        data.push(data[data.len() % test_values.len()]);
    }

    let matrix = GraphMatrix::from_dense(&data, (1, 128), BlockFormat::B16x8);
    let device = Device::Cpu;

    println!("Input: {} values, range [{:.4}, {:.4}]",
             data.len(),
             data.iter().cloned().fold(f32::INFINITY, f32::min),
             data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    println!();

    // Test each format
    for (dtype, pack) in [
        (QuantDType::I4, PackFormat::None),
        (QuantDType::I4, PackFormat::Packed),
        (QuantDType::I4, PackFormat::Awq),
        (QuantDType::I4, PackFormat::Gptq),
        (QuantDType::I8, PackFormat::None),
        (QuantDType::I8, PackFormat::Packed),
        (QuantDType::F16, PackFormat::None),
    ] {
        let result = quantize(&matrix, dtype, pack, 0.0, &device).unwrap();

        let weights_shape = result.weights.dims();
        let weights_dtype = result.weights.dtype();
        let has_scales = result.scales.is_some();
        let has_zeros = result.zeros.is_some();

        println!("{:?} + {:?}:", dtype, pack);
        println!("  weights: {:?} {:?}", weights_shape, weights_dtype);
        if has_scales {
            println!("  scales:  {:?}", result.scales.as_ref().unwrap().dims());
        }
        if has_zeros {
            println!("  zeros:   {:?}", result.zeros.as_ref().unwrap().dims());
        }
        println!("  log2_range: {:.1} bits", result.params.log2_range);
        println!("  log2_center: {:.2}", result.params.log2_center);
        println!("  nnz: {}", result.params.nnz);

        // For non-grouped formats, compute reconstruction error
        if let Some(scale) = result.params.scale {
            if dtype == QuantDType::I8 && pack == PackFormat::Packed {
                // Dequantize and check error
                let packed: Vec<u8> = result.weights.flatten_all().unwrap().to_vec1().unwrap();
                let mut max_error = 0.0f32;
                let mut total_error = 0.0f32;
                let mut count = 0;

                for (i, &p) in packed.iter().enumerate() {
                    let q = p as i8 as f32; // reinterpret as signed
                    let reconstructed = q * scale;
                    let original = data[i];

                    if original != 0.0 {
                        let rel_error = (original - reconstructed).abs() / original.abs();
                        max_error = max_error.max(rel_error);
                        total_error += rel_error;
                        count += 1;
                    }
                }

                if count > 0 {
                    println!("  avg_error: {:.2}%, max_error: {:.2}%",
                             100.0 * total_error / count as f32,
                             100.0 * max_error);
                }
            }
        }
        println!();
    }

    println!("=== AWQ Tensor Inspection ===\n");

    let awq = quantize(&matrix, QuantDType::I4, PackFormat::Awq, 0.0, &device).unwrap();
    let packed: Vec<u32> = awq.weights.flatten_all().unwrap().to_vec1().unwrap();

    // Unpack first u32 (8 i4 values)
    println!("First packed u32: 0x{:08x}", packed[0]);
    print!("Unpacked values: ");
    for i in 0..8 {
        let nibble = ((packed[0] >> (i * 4)) & 0x0F) as i8;
        // Sign extend from 4 bits
        let signed = if nibble & 0x08 != 0 { nibble | !0x0F } else { nibble };
        print!("{:3} ", signed);
    }
    println!();

    let scales: Vec<half::f16> = awq.scales.unwrap().flatten_all().unwrap().to_vec1().unwrap();
    println!("Group scales: {:?}", scales.iter().map(|s| s.to_f32()).collect::<Vec<_>>());

    println!("\n=== GPTQ Tensor Inspection ===\n");

    let gptq = quantize(&matrix, QuantDType::I4, PackFormat::Gptq, 0.0, &device).unwrap();
    let packed: Vec<u32> = gptq.weights.flatten_all().unwrap().to_vec1().unwrap();

    println!("First packed u32: 0x{:08x}", packed[0]);
    print!("Unpacked values (unsigned): ");
    for i in 0..8 {
        let nibble = (packed[0] >> (i * 4)) & 0x0F;
        print!("{:3} ", nibble);
    }
    println!();

    let scales: Vec<half::f16> = gptq.scales.unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let zeros: Vec<u32> = gptq.zeros.unwrap().flatten_all().unwrap().to_vec1().unwrap();

    println!("Group scales: {:?}", scales.iter().map(|s| s.to_f32()).collect::<Vec<_>>());
    println!("Packed zeros: 0x{:08x}", zeros[0]);

    println!("\n=== Test Complete ===");
}
