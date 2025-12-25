//! Integration tests for GGUF quantization

use super::utils::select_quant_with_fallback;
use super::*;

#[test]
fn test_dimension_fallback() {
    use GgufQuantType::*;

    // K-quant with proper alignment
    assert_eq!(select_quant_with_fallback(256, Q4_K_M), Q4_K_M);
    assert_eq!(select_quant_with_fallback(512, Q4_K_M), Q4_K_M);

    // K-quant falls back to Q8_0 for 32-aligned but not 256-aligned
    assert_eq!(select_quant_with_fallback(128, Q4_K_M), Q8_0);
    assert_eq!(select_quant_with_fallback(32, Q4_K_M), Q8_0);
    assert_eq!(select_quant_with_fallback(64, Q5_K_S), Q8_0);

    // Legacy formats work with 32-alignment
    assert_eq!(select_quant_with_fallback(32, Q8_0), Q8_0);
    assert_eq!(select_quant_with_fallback(64, Q4_0), Q4_0);
    assert_eq!(select_quant_with_fallback(128, Q5_0), Q5_0);
}

#[test]
#[should_panic(expected = "must be multiple of 32")]
fn test_dimension_fallback_invalid() {
    select_quant_with_fallback(100, GgufQuantType::Q4_K_M);
}

#[test]
#[should_panic(expected = "must be multiple of 32")]
fn test_dimension_fallback_invalid_legacy() {
    select_quant_with_fallback(17, GgufQuantType::Q8_0);
}

#[test]
fn test_quant_type_properties() {
    use GgufQuantType::*;

    // Block sizes
    assert_eq!(Q4_0.block_size(), 32);
    assert_eq!(Q8_0.block_size(), 32);
    assert_eq!(Q4_K_M.block_size(), 256);
    assert_eq!(IQ4_NL.block_size(), 256);

    // Alignment checks
    assert!(Q4_0.is_aligned(32));
    assert!(Q4_0.is_aligned(64));
    assert!(!Q4_0.is_aligned(31));

    assert!(Q4_K_M.is_aligned(256));
    assert!(Q4_K_M.is_aligned(512));
    assert!(!Q4_K_M.is_aligned(128));

    // Format categories
    assert!(Q4_K_M.is_kquant());
    assert!(!Q4_K_M.is_iquant());
    assert!(IQ4_NL.is_iquant());
    assert!(!IQ4_NL.is_kquant());
    assert!(!Q8_0.is_kquant());
    assert!(!Q8_0.is_iquant());
}

// TODO: Add integration tests once GraphMatrix test helpers are available
//
// #[test]
// fn test_quantize_to_gguf_q8_0() {
//     // Create test matrix with known values
//     // Quantize to Q8_0
//     // Verify output size and format
// }
//
// #[test]
// fn test_quantize_to_gguf_q4_k() {
//     // Create test matrix with 256-column width
//     // Quantize to Q4_K_M with Standard and Trellis
//     // Compare quality metrics
// }
//
// #[test]
// fn test_trellis_improves_quality() {
//     // Create matrix with smooth gradient (trellis should help)
//     // Quantize with Standard and Trellis
//     // Verify trellis has equal or lower error
// }
//
// #[test]
// fn test_gguf_format_valid() {
//     // Quantize to various formats
//     // Verify header bytes are reasonable f16 values
//     // Verify output size matches expected block layout
// }
