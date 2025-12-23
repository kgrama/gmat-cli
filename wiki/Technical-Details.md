# Technical Details

This page explains the internals of GMAT storage: block formats, encoding schemes, and the tradeoffs between accuracy, compression, and sparsity handling.

## GMAT Storage Architecture

GMAT stores tensors as **block-sparse matrices**. Instead of storing every element, it:

1. Divides the tensor into fixed-size blocks (8 or 16 elements)
2. Stores only non-zero blocks
3. Uses logarithmic encoding for efficient magnitude representation

```
Dense Tensor (4096 x 4096)
    ↓
Block Decomposition (8 or 16 elements per block)
    ↓
Sparse Block Storage (empty blocks = 2 bytes)
    ↓
Log-domain Magnitude Encoding (4-bit or 8-bit)
```

## Block Formats

GMAT supports 8 block format variants:

| Format | Elements | Magnitude Bits | Non-Empty Size | Empty Size | Octave Range |
|--------|----------|----------------|----------------|------------|--------------|
| `B8x4` | 8 | 4-bit (e0m4) | 9 bytes | 1 byte | 1 octave |
| `B8x8` | 8 | 8-bit (e1m7) | 13 bytes | 1 byte | 4 octaves |
| `B16x4` | 16 | 4-bit (e0m4) | 16 bytes | 2 bytes | 1 octave |
| `B16x8` | 16 | 8-bit (e1m7) | 24 bytes | 2 bytes | 4 octaves |
| `DualRow8x4` | 8 | 4-bit | 9 bytes | 1 byte | 1 octave |
| `DualRow8x8` | 8 | 8-bit | 13 bytes | 1 byte | 4 octaves |
| `DualRow16x4` | 16 | 4-bit | 16 bytes | 2 bytes | 1 octave |
| `DualRow16x8` | 16 | 8-bit | 24 bytes | 2 bytes | 4 octaves |

**DualRow variants** process two adjacent rows together, which can improve cache locality for certain access patterns.

## Block Structure

Each non-empty block contains:

```
┌─────────────────────────────────────────────────────┐
│ zero_map (1-2 bytes)   - Bitmask of non-zero elements │
│ scale_log (2 bytes)    - f16 base scale (log2 domain) │
│ signs (1-2 bytes)      - Bitmask of negative elements │
│ shift_map (1-2 bytes)  - Bitmask for octave extension │
│ magnitudes (4-16 bytes) - Encoded magnitude offsets   │
└─────────────────────────────────────────────────────┘
```

**Empty blocks** store only `zero_map = 0`, taking just 1-2 bytes.

## Logarithmic Encoding

GMAT uses log-domain representation for efficient magnitude storage:

```
value = sign × 2^(scale_log + offset)
```

Where:
- `scale_log`: f16 base scale (shared across block)
- `offset`: Per-element magnitude offset (4 or 8 bits)
- `sign`: Per-element sign bit

### e0m4 (4-bit) Encoding

- **Range**: [0, 1) in log2 units = 1 octave
- **Resolution**: 16 steps per octave
- **With shift bit**: Extends to 2 octaves

```
offset = nibble / 16.0
// nibble ∈ [0, 15] → offset ∈ [0, 0.9375]
```

### e1m7 (8-bit) Encoding

- **Range**: [0, 2) in log2 units = 4 octaves
- **Resolution**: 128 steps per octave
- **Format**: 1-bit exponent + 7-bit mantissa

```
e = raw >> 7           // exponent: 0 or 1
m = (raw & 0x7F) / 128 // mantissa: [0, 0.992]
offset = e + m         // total: [0, 1.992]
```

With shift bit, extends to 4 octaves total.

## Compression vs Accuracy Tradeoffs

### Block Size (8 vs 16 elements)

| Factor | 8-element | 16-element |
|--------|-----------|------------|
| Sparsity granularity | Finer | Coarser |
| Overhead per block | Lower | Higher |
| Scale sharing | 8 elements share scale | 16 elements share scale |
| Best for | Highly sparse | Moderately sparse |

**Tradeoff**: Larger blocks amortize overhead but force more elements to share a single scale, potentially reducing accuracy.

### Magnitude Bits (4 vs 8)

| Factor | 4-bit (e0m4) | 8-bit (e1m7) |
|--------|--------------|--------------|
| Storage | 0.5 bytes/element | 1 byte/element |
| Dynamic range | 1-2 octaves | 4 octaves |
| Quantization error | ~6% per step | ~0.8% per step |
| Best for | Uniform magnitudes | Varied magnitudes |

**Tradeoff**: 4-bit saves space but introduces more quantization error, especially when element magnitudes within a block vary significantly.

## Sparsity Handling

GMAT exploits sparsity at the block level:

### Dense Tensor
```
Storage = rows × cols × 4 bytes (f32)
Example: 4096 × 4096 = 64 MB
```

### GMAT with 50% Sparsity (B8x8)
```
Non-empty blocks ≈ (rows × cols) / 8 / 2
Storage ≈ (blocks × 13 bytes) + (empty_blocks × 1 byte)
Example: ~4.2 MB (15x compression)
```

### GMAT with 90% Sparsity (B8x8)
```
Non-empty blocks ≈ (rows × cols) / 8 / 10
Storage ≈ (blocks × 13 bytes) + (empty_blocks × 1 byte)
Example: ~0.9 MB (70x compression)
```

The key insight: **empty blocks cost almost nothing** (1-2 bytes vs 8-16 bytes for non-empty).

## Format Selection Guide

| Scenario | Recommended Format | Reasoning |
|----------|-------------------|-----------|
| Pruned model (>50% sparse) | `B8x8` | Fine granularity, good accuracy |
| Dense model, size priority | `B16x4` | Maximum compression |
| Dense model, accuracy priority | `B8x8` or `B16x8` | Better dynamic range |
| Unknown sparsity | `B8x8` | Good default balance |

## Relationship to GGUF Quantization

GMAT's block encoding is **independent** of GGUF quantization:

```
SafeTensors (f32/f16/bf16)
    ↓
GMAT (block-sparse, log-encoded)  ← Block format chosen here
    ↓
GGUF (quantized: Q4, Q6, Q8, etc.) ← Quantization chosen here
```

- **GMAT block format**: Controls intermediate storage efficiency
- **GGUF quantization**: Controls final deployment size/quality

You can use aggressive GMAT compression (B16x4) with high-quality GGUF (Q8), or vice versa. They're orthogonal choices.

## Memory Layout

GMAT files use a simple layout:

```
┌────────────────────────────────┐
│ Header (magic, version, meta) │
├────────────────────────────────┤
│ Block 0                        │
│ Block 1                        │
│ ...                            │
│ Block N                        │
└────────────────────────────────┘
```

Blocks are stored in row-major order. Empty blocks are stored inline (not skipped) to preserve indexing, but take minimal space.

## Performance Considerations

### Encoding (Import)
- O(n) scan for min magnitude (scale calculation)
- O(n) offset computation
- Parallelized across tensors via Rayon

### Decoding (Export)
- O(1) per element: `value = sign × 2^(scale + offset)`
- Cache-friendly sequential access
- Parallelized across tensors

### Memory Efficiency
- Streaming pipeline: bounded buffer of tensors in flight
- Individual tensor files: only load what you need
- Empty blocks: minimal memory footprint

## See Also

- [[Import-Command]] - Block format selection during import
- [[Export-Command]] - Quantization during export
- [[Configuration-Files]] - Full configuration reference
