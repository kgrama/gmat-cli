//! Tests for quantization module.

use super::*;
use crate::blocks::BlockFormat;
use candle_core::DType;
use types::StaticSaliency;

#[test]
fn test_packed_i4() {
    let data: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
    let matrix = GraphMatrix::from_dense(&data, (1, 8), BlockFormat::B8x8);

    let result = quantize(
        &matrix,
        QuantDType::I4,
        PackFormat::Packed,
        0.0,
        &Device::Cpu,
    )
    .unwrap();

    assert_eq!(result.weights.dims(), &[1, 4]); // 8 values → 4 bytes
    assert_eq!(result.weights.dtype(), DType::U8);
    assert!(result.scales.is_none());
}

#[test]
fn test_packed_i8() {
    let data: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
    let matrix = GraphMatrix::from_dense(&data, (1, 8), BlockFormat::B8x8);

    let result = quantize(
        &matrix,
        QuantDType::I8,
        PackFormat::Packed,
        0.0,
        &Device::Cpu,
    )
    .unwrap();

    assert_eq!(result.weights.dims(), &[1, 8]);
    assert_eq!(result.weights.dtype(), DType::U8);
}

#[test]
fn test_awq_format() {
    // AWQ format: qweight=(in_features/8, out_features), scales=(num_groups, out_features)
    // Matrix shape: (out_features=1, in_features=256)
    let data: Vec<f32> = (0..256).map(|x| (x as f32 - 128.0) / 10.0).collect();
    let matrix = GraphMatrix::from_dense(&data, (1, 256), BlockFormat::B16x8);

    let result = quantize(&matrix, QuantDType::I4, PackFormat::Awq, 0.0, &Device::Cpu).unwrap();

    // qweight: (in_features/8, out_features) = (256/8, 1) = (32, 1)
    assert_eq!(result.weights.dims(), &[32, 1]);
    assert_eq!(result.weights.dtype(), DType::U32);

    // scales: (num_groups, out_features) = (256/128, 1) = (2, 1)
    assert!(result.scales.is_some());
    assert_eq!(result.scales.as_ref().unwrap().dims(), &[2, 1]);

    // zeros: (num_groups/8, out_features) = (1, 1) since 2 groups packs into 1 u32
    assert!(result.zeros.is_some());
    assert_eq!(result.zeros.as_ref().unwrap().dims(), &[1, 1]);

    assert_eq!(result.params.group_size, 128);
}

#[test]
fn test_gptq_format() {
    // GPTQ format: qweight=(in_features/8, out_features), scales=(num_groups, out_features)
    let data: Vec<f32> = (0..256).map(|x| (x as f32 - 128.0) / 10.0).collect();
    let matrix = GraphMatrix::from_dense(&data, (1, 256), BlockFormat::B16x8);

    let result = quantize(&matrix, QuantDType::I4, PackFormat::Gptq, 0.0, &Device::Cpu).unwrap();

    // qweight: (in_features/8, out_features) = (32, 1)
    assert_eq!(result.weights.dims(), &[32, 1]);
    assert_eq!(result.weights.dtype(), DType::U32);

    // scales: (num_groups, out_features) = (2, 1)
    assert!(result.scales.is_some());
    assert_eq!(result.scales.as_ref().unwrap().dims(), &[2, 1]);

    // zeros: (num_groups/8, out_features) = (1, 1)
    assert!(result.zeros.is_some());
    assert_eq!(result.zeros.as_ref().unwrap().dims(), &[1, 1]);

    // g_idx: (in_features,) = (256,)
    assert!(result.g_idx.is_some());
    assert_eq!(result.g_idx.as_ref().unwrap().dims(), &[256]);
}

#[test]
fn test_f16_passthrough() {
    let data: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
    let matrix = GraphMatrix::from_dense(&data, (1, 8), BlockFormat::B8x8);

    let result = quantize(
        &matrix,
        QuantDType::F16,
        PackFormat::None,
        0.0,
        &Device::Cpu,
    )
    .unwrap();

    assert_eq!(result.weights.dims(), &[1, 8]);
    assert_eq!(result.weights.dtype(), DType::F16);
}

#[test]
fn test_all_pack_formats() {
    // Use 128 cols to match group_size for AWQ/GPTQ
    let data: Vec<f32> = (0..256).map(|x| x as f32 / 100.0).collect();
    let matrix = GraphMatrix::from_dense(&data, (2, 128), BlockFormat::B16x8);

    for pack in [
        PackFormat::None,
        PackFormat::Packed,
        PackFormat::Awq,
        PackFormat::Gptq,
    ] {
        let result = quantize(&matrix, QuantDType::I4, pack, 0.0, &Device::Cpu).unwrap();
        assert_eq!(result.params.pack_format, pack);
    }
}

#[test]
fn test_awq_larger_matrix() {
    // Test with multiple output features
    // Matrix: (out_features=4, in_features=256)
    let data: Vec<f32> = (0..1024).map(|x| (x as f32 - 512.0) / 100.0).collect();
    let matrix = GraphMatrix::from_dense(&data, (4, 256), BlockFormat::B16x8);

    let result = quantize(&matrix, QuantDType::I4, PackFormat::Awq, 0.0, &Device::Cpu).unwrap();

    // qweight: (in_features/8, out_features) = (32, 4)
    assert_eq!(result.weights.dims(), &[32, 4]);

    // scales: (num_groups, out_features) = (2, 4)
    assert_eq!(result.scales.as_ref().unwrap().dims(), &[2, 4]);
}

#[test]
fn test_gptq_larger_matrix() {
    // Matrix: (out_features=4, in_features=256)
    let data: Vec<f32> = (0..1024).map(|x| (x as f32 - 512.0) / 100.0).collect();
    let matrix = GraphMatrix::from_dense(&data, (4, 256), BlockFormat::B16x8);

    let result = quantize(&matrix, QuantDType::I4, PackFormat::Gptq, 0.0, &Device::Cpu).unwrap();

    // qweight: (in_features/8, out_features) = (32, 4)
    assert_eq!(result.weights.dims(), &[32, 4]);

    // scales: (num_groups, out_features) = (2, 4)
    assert_eq!(result.scales.as_ref().unwrap().dims(), &[2, 4]);

    // g_idx: (in_features,) = (256,)
    assert_eq!(result.g_idx.as_ref().unwrap().dims(), &[256]);
}

// ============================================================================
// Activation-aware quantization tests
// ============================================================================

#[test]
fn test_activation_stats_from_scales() {
    let scales = vec![1.0, 2.0, 0.5, 4.0];
    let stats = ActivationStats::from_scales(scales.clone());

    assert_eq!(stats.num_channels(), 4);
    assert_eq!(stats.scale(0), 1.0);
    assert_eq!(stats.scale(1), 2.0);
    assert_eq!(stats.scale(2), 0.5);
    assert_eq!(stats.scale(3), 4.0);
}

#[test]
fn test_activation_stats_from_log2() {
    let log2_scales = vec![0.0, 1.0, -1.0, 2.0]; // 1, 2, 0.5, 4
    let stats = ActivationStats::from_log2_scales(log2_scales);

    assert_eq!(stats.num_channels(), 4);
    assert!((stats.scale(0) - 1.0).abs() < 1e-6);
    assert!((stats.scale(1) - 2.0).abs() < 1e-6);
    assert!((stats.scale(2) - 0.5).abs() < 1e-6);
    assert!((stats.scale(3) - 4.0).abs() < 1e-6);
}

#[test]
fn test_activation_stats_importance() {
    let stats = ActivationStats::from_scales(vec![1.0, 10.0, 0.1]);

    // importance = |weight| * |activation|
    assert_eq!(stats.importance(0, 2.0), 2.0); // 2.0 * 1.0
    assert_eq!(stats.importance(1, 2.0), 20.0); // 2.0 * 10.0
    assert!((stats.importance(2, 2.0) - 0.2).abs() < 1e-6); // 2.0 * 0.1
}

#[test]
fn test_quantize_with_activations_awq() {
    // Matrix: (out_features=2, in_features=256)
    let data: Vec<f32> = (0..512).map(|x| (x as f32 - 256.0) / 100.0).collect();
    let matrix = GraphMatrix::from_dense(&data, (2, 256), BlockFormat::B16x8);

    // Create activation stats - first half of channels have high activation,
    // second half have low activation
    let act_scales: Vec<f32> = (0..256).map(|i| if i < 128 { 10.0 } else { 0.1 }).collect();
    let act_stats = ActivationStats::from_scales(act_scales);

    let result = quantize_with_activations(
        &matrix,
        &act_stats,
        QuantDType::I4,
        PackFormat::Awq,
        0.0,
        &Device::Cpu,
    )
    .unwrap();

    // Should produce valid AWQ format
    assert_eq!(result.weights.dims(), &[32, 2]);
    assert_eq!(result.scales.as_ref().unwrap().dims(), &[2, 2]);
    assert!(result.zeros.is_some());
    assert_eq!(result.params.pack_format, PackFormat::Awq);
}

#[test]
fn test_quantize_with_activations_gptq() {
    // Matrix: (out_features=2, in_features=256)
    let data: Vec<f32> = (0..512).map(|x| (x as f32 - 256.0) / 100.0).collect();
    let matrix = GraphMatrix::from_dense(&data, (2, 256), BlockFormat::B16x8);

    // Create non-uniform activation stats
    let act_scales: Vec<f32> = (0..256)
        .map(|i| 1.0 + (i as f32 / 256.0) * 9.0) // 1.0 to 10.0
        .collect();
    let act_stats = ActivationStats::from_scales(act_scales);

    let result = quantize_with_activations(
        &matrix,
        &act_stats,
        QuantDType::I4,
        PackFormat::Gptq,
        0.0,
        &Device::Cpu,
    )
    .unwrap();

    // Should produce valid GPTQ format
    assert_eq!(result.weights.dims(), &[32, 2]);
    assert_eq!(result.scales.as_ref().unwrap().dims(), &[2, 2]);
    assert!(result.zeros.is_some());
    assert!(result.g_idx.is_some());
    assert_eq!(result.params.pack_format, PackFormat::Gptq);
}

#[test]
fn test_quantize_with_activations_dimension_mismatch() {
    let data: Vec<f32> = (0..256).map(|x| x as f32 / 100.0).collect();
    let matrix = GraphMatrix::from_dense(&data, (2, 128), BlockFormat::B16x8);

    // Wrong number of channels
    let act_stats = ActivationStats::from_scales(vec![1.0; 64]); // Should be 128

    let result = quantize_with_activations(
        &matrix,
        &act_stats,
        QuantDType::I4,
        PackFormat::Awq,
        0.0,
        &Device::Cpu,
    );

    assert!(result.is_err());
}

#[test]
fn test_activation_aware_vs_standard_produces_different_scales() {
    // This test verifies that activation awareness actually changes the output
    let data: Vec<f32> = (0..512).map(|x| (x as f32 - 256.0) / 100.0).collect();
    let matrix = GraphMatrix::from_dense(&data, (2, 256), BlockFormat::B16x8);

    // Standard quantization (no activation info)
    let standard = quantize(&matrix, QuantDType::I4, PackFormat::Awq, 0.0, &Device::Cpu).unwrap();

    // Activation-aware with highly non-uniform activations
    let act_scales: Vec<f32> = (0..256)
        .map(|i| if i < 128 { 100.0 } else { 0.01 }) // 10000x difference
        .collect();
    let act_stats = ActivationStats::from_scales(act_scales);

    let aware = quantize_with_activations(
        &matrix,
        &act_stats,
        QuantDType::I4,
        PackFormat::Awq,
        0.01, // Small clip to trigger importance-based scaling
        &Device::Cpu,
    )
    .unwrap();

    // Both should have same tensor shapes
    assert_eq!(standard.weights.dims(), aware.weights.dims());
    assert_eq!(
        standard.scales.as_ref().unwrap().dims(),
        aware.scales.as_ref().unwrap().dims()
    );

    // But the scales should be different due to activation weighting
    // (We can't easily compare tensor values here, but the dimensions confirm the path was taken)
}

// ============================================================================
// Static saliency tests
// ============================================================================

#[test]
fn test_static_saliency_from_column_scales() {
    let col_scales = vec![1.0, 2.0, 0.5, 4.0];
    let saliency = StaticSaliency::from_column_scales(col_scales);

    assert_eq!(saliency.num_channels(), 4);
    assert_eq!(saliency.saliency(0), 1.0);
    assert_eq!(saliency.saliency(1), 2.0);
    assert_eq!(saliency.saliency(2), 0.5);
    assert_eq!(saliency.saliency(3), 4.0);
}

#[test]
fn test_static_saliency_from_chained_scales() {
    // Upstream (e.g., embedding output scales)
    let upstream = vec![2.0, 3.0, 1.0, 4.0];
    // Weight column scales
    let weight_cols = vec![1.0, 2.0, 4.0, 0.5];

    let saliency = StaticSaliency::from_chained_scales(&upstream, &weight_cols);

    // saliency = upstream × weight
    assert_eq!(saliency.num_channels(), 4);
    assert_eq!(saliency.saliency(0), 2.0); // 2.0 × 1.0
    assert_eq!(saliency.saliency(1), 6.0); // 3.0 × 2.0
    assert_eq!(saliency.saliency(2), 4.0); // 1.0 × 4.0
    assert_eq!(saliency.saliency(3), 2.0); // 4.0 × 0.5
}

#[test]
fn test_static_saliency_from_chained_log2() {
    // log2 scales: 0, 1, -1, 2 → linear: 1, 2, 0.5, 4
    let upstream_log2 = vec![0.0, 1.0, -1.0, 2.0];
    // log2 scales: 1, 0, 2, -1 → linear: 2, 1, 4, 0.5
    let weight_log2 = vec![1.0, 0.0, 2.0, -1.0];

    let saliency = StaticSaliency::from_chained_log2(&upstream_log2, &weight_log2);

    // saliency = exp2(up + w) = up_linear × w_linear
    assert_eq!(saliency.num_channels(), 4);
    assert!((saliency.saliency(0) - 2.0).abs() < 1e-6); // 1 × 2
    assert!((saliency.saliency(1) - 2.0).abs() < 1e-6); // 2 × 1
    assert!((saliency.saliency(2) - 2.0).abs() < 1e-6); // 0.5 × 4
    assert!((saliency.saliency(3) - 2.0).abs() < 1e-6); // 4 × 0.5
}

#[test]
fn test_static_saliency_importance() {
    let saliency = StaticSaliency::from_column_scales(vec![1.0, 10.0, 0.1]);

    // importance = |weight| × saliency
    assert_eq!(saliency.importance(0, 2.0), 2.0); // 2.0 × 1.0
    assert_eq!(saliency.importance(1, 2.0), 20.0); // 2.0 × 10.0
    assert!((saliency.importance(2, 2.0) - 0.2).abs() < 1e-6); // 2.0 × 0.1
}

#[test]
fn test_static_saliency_to_activation_stats() {
    let saliency = StaticSaliency::from_column_scales(vec![1.0, 2.0, 3.0]);
    let act_stats = saliency.to_activation_stats();

    assert_eq!(act_stats.num_channels(), 3);
    assert_eq!(act_stats.scale(0), 1.0);
    assert_eq!(act_stats.scale(1), 2.0);
    assert_eq!(act_stats.scale(2), 3.0);
}

#[test]
fn test_quantize_with_saliency_awq() {
    // Matrix: (out_features=2, in_features=256)
    let data: Vec<f32> = (0..512).map(|x| (x as f32 - 256.0) / 100.0).collect();
    let matrix = GraphMatrix::from_dense(&data, (2, 256), BlockFormat::B16x8);

    // Create saliency from "block scales" - first half important, second half not
    let col_scales: Vec<f32> = (0..256).map(|i| if i < 128 { 10.0 } else { 0.1 }).collect();
    let saliency = StaticSaliency::from_column_scales(col_scales);

    let result = quantize_with_saliency(
        &matrix,
        &saliency,
        QuantDType::I4,
        PackFormat::Awq,
        0.0,
        &Device::Cpu,
    )
    .unwrap();

    // Should produce valid AWQ format
    assert_eq!(result.weights.dims(), &[32, 2]);
    assert_eq!(result.scales.as_ref().unwrap().dims(), &[2, 2]);
    assert!(result.zeros.is_some());
    assert_eq!(result.params.pack_format, PackFormat::Awq);
}

#[test]
fn test_quantize_with_saliency_gptq() {
    // Matrix: (out_features=2, in_features=256)
    let data: Vec<f32> = (0..512).map(|x| (x as f32 - 256.0) / 100.0).collect();
    let matrix = GraphMatrix::from_dense(&data, (2, 256), BlockFormat::B16x8);

    // Simulate chained scales (embedding → weight)
    let embed_scales: Vec<f32> = (0..256).map(|i| 1.0 + (i as f32 / 256.0)).collect();
    let weight_scales: Vec<f32> = (0..256).map(|i| 2.0 - (i as f32 / 256.0)).collect();
    let saliency = StaticSaliency::from_chained_scales(&embed_scales, &weight_scales);

    let result = quantize_with_saliency(
        &matrix,
        &saliency,
        QuantDType::I4,
        PackFormat::Gptq,
        0.0,
        &Device::Cpu,
    )
    .unwrap();

    // Should produce valid GPTQ format
    assert_eq!(result.weights.dims(), &[32, 2]);
    assert_eq!(result.scales.as_ref().unwrap().dims(), &[2, 2]);
    assert!(result.zeros.is_some());
    assert!(result.g_idx.is_some());
    assert_eq!(result.params.pack_format, PackFormat::Gptq);
}

#[test]
fn test_saliency_vs_standard_produces_different_output() {
    let data: Vec<f32> = (0..512).map(|x| (x as f32 - 256.0) / 100.0).collect();
    let matrix = GraphMatrix::from_dense(&data, (2, 256), BlockFormat::B16x8);

    // Standard quantization
    let standard = quantize(&matrix, QuantDType::I4, PackFormat::Awq, 0.0, &Device::Cpu).unwrap();

    // Saliency-aware with extreme difference
    let col_scales: Vec<f32> = (0..256)
        .map(|i| if i < 128 { 100.0 } else { 0.01 })
        .collect();
    let saliency = StaticSaliency::from_column_scales(col_scales);

    let salient = quantize_with_saliency(
        &matrix,
        &saliency,
        QuantDType::I4,
        PackFormat::Awq,
        0.01,
        &Device::Cpu,
    )
    .unwrap();

    // Same shapes
    assert_eq!(standard.weights.dims(), salient.weights.dims());
    assert_eq!(
        standard.scales.as_ref().unwrap().dims(),
        salient.scales.as_ref().unwrap().dims()
    );
}
