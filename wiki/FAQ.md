# FAQ

Frequently asked questions about GMAT defaults, optimization options, and design decisions.

## Why These Defaults?

### Default Quantization: Q4_K_M

**Q: Why is Q4_K_M the default quantization type?**

Q4_K_M (4-bit K-quant, medium) offers the best balance for most deployments:

| Format | Bits | Quality | Size | Use Case |
|--------|------|---------|------|----------|
| Q8_0 | 8 | Excellent | Large | Quality-critical, plenty of VRAM |
| Q6_K | 6 | Very good | Medium | High-quality production |
| **Q4_K_M** | 4 | Good | Small | **General production (default)** |
| Q4_0 | 4 | Acceptable | Smallest | Edge/mobile, maximum compression |

Q4_K_M specifically uses:
- 4-bit quantization with K-quant's improved scale handling
- 256-element superblocks with 16-element sub-groups
- Per-group scales and minimums for better dynamic range

For most LLM workloads, Q4_K_M achieves 90-95% of full-precision quality at 4x size reduction.

### Default Block Format: B8x8

**Q: Why is B8x8 the default GMAT block format?**

B8x8 (8 elements, 8-bit magnitudes) provides the best accuracy-to-size tradeoff:

| Format | Elements | Bits | Octave Range | Best For |
|--------|----------|------|--------------|----------|
| B8x4 | 8 | 4 | 1-2 | Maximum compression |
| **B8x8** | 8 | 8 | 4 | **General use (default)** |
| B16x4 | 16 | 4 | 1-2 | Large sparse models |
| B16x8 | 16 | 8 | 4 | High accuracy, large scale |

Why 8 elements over 16?
- Finer sparsity granularity (skip smaller empty regions)
- Each block shares one scale, so fewer elements = less scale quantization error

Why 8-bit over 4-bit?
- 4 octaves of dynamic range (vs 1-2 for 4-bit)
- 128 steps per octave (vs 16 for 4-bit)
- Critical for layers with high weight variance

### Default Scale Optimization: Trellis

**Q: Why is trellis the default scale optimization?**

Trellis optimization uses dynamic programming to find scale assignments that minimize total quantization error while encouraging smoothness between adjacent scales. This typically improves quality by 0.5-2% perplexity with minimal overhead.

```
Standard:  Each group picks its own optimal scale independently
Trellis:   DP finds globally-optimal scales with smoothness penalty
```

The `trellis_lambda` parameter (default: 0.3) controls the smoothness penalty:
- `0.0`: No smoothing (equivalent to standard)
- `0.3`: Moderate smoothing (default)
- `1.0`: Strong smoothing (may over-smooth)

---

## Trellis Optimization

### What is Trellis Quantization?

**Q: What does "trellis" mean in GMAT?**

Trellis quantization is a dynamic programming technique borrowed from audio/video codecs. Instead of quantizing each block independently, it considers sequences of blocks and finds the globally optimal quantization that minimizes total error.

For GGUF K-quants (Q4_K_M, Q5_K_M, Q6_K), each 256-element superblock contains 16 sub-groups of 16 elements. Trellis optimizes the scale assignment across these sub-groups:

```
Standard approach:
  Group 1 → pick best scale for group 1
  Group 2 → pick best scale for group 2
  ...
  (each decision is independent)

Trellis approach:
  Consider all groups together
  Find scale sequence that minimizes: Σ(quantization_error) + λ×Σ(scale_jumps)
  (decisions are jointly optimal)
```

### Why "Without Hessian"?

**Q: What's the difference between GMAT's trellis and GPTQ/AWQ?**

Traditional methods like GPTQ and AWQ use the **Hessian matrix** (second derivatives of loss) computed from calibration data to weight quantization errors. This requires:
1. A calibration dataset
2. Forward passes through the model
3. O(n²) storage for the Hessian

GMAT's trellis operates **without Hessian** information:
- Uses only the weight values themselves
- No calibration data required
- O(n) time and space complexity

The tradeoff:

| Method | Calibration | Accuracy | Speed |
|--------|-------------|----------|-------|
| GPTQ | Required | Best | Slow |
| AWQ | Required | Very good | Medium |
| **GMAT Trellis** | None | Good | Fast |

For production workflows where you're quantizing many models or fine-tunes, GMAT's approach is often preferable: slightly lower accuracy but much faster iteration.

---

## Static Saliency

### What is Static Saliency?

**Q: GMAT mentions "saliency without calibration" - how does that work?**

Saliency measures how important each weight is to the model's output. Traditional methods (AWQ, SmoothQuant) compute this from activation patterns during calibration runs.

GMAT's **static saliency** approximates importance using only weight statistics:

```
Traditional (with calibration):
  importance[i] = |weight[i]| × mean(|activation[i]|)
  ↑ requires forward passes

Static saliency (no calibration):
  importance[i] = |weight[i]| × upstream_scale[i]
  ↑ derived from weight distributions
```

The insight: embedding layer scales and layer-to-layer weight scales correlate with activation magnitudes. By chaining these scales through the network, GMAT approximates saliency without running the model.

This provides ~80-90% of the benefit of full calibration for most LLMs, with zero inference cost.

### When to Use Calibration Instead?

Static saliency works well when:
- Processing many models/fine-tunes (speed matters)
- Models follow standard architectures (LLaMA, GPT, etc.)
- Moderate quantization (Q4_K_M to Q8_0)

Consider calibration-based tools when:
- Maximum accuracy is critical
- Aggressive quantization (Q2, Q3)
- Non-standard architectures
- Quantizing for specific tasks/domains

---

## Quantization Type Selection

### Per-Tensor Overrides

**Q: Why would I set different quantization per tensor?**

Not all tensors are equally sensitive to quantization:

| Tensor Type | Sensitivity | Recommendation |
|-------------|-------------|----------------|
| `token_embd.weight` | **High** | Q6_K or Q8_0 |
| `output.weight` | **High** | Q6_K or Q8_0 |
| `attn_q/k/v.weight` | Medium | Q4_K_M to Q6_K |
| `ffn_*.weight` | Lower | Q4_K_M or Q4_0 |

Embeddings and output layers handle the token vocabulary directly - quantization errors here affect every token. FFN layers are more tolerant because their errors average out across many parameters.

Example config:
```json
{
  "quantization": {
    "default_type": "q4_k_m",
    "per_tensor": {
      "<embedding-uuid>": "q8_0",
      "<output-uuid>": "q8_0"
    }
  }
}
```

### K-Quants vs Legacy

**Q: What's the difference between Q4_0 and Q4_K_M?**

Legacy formats (Q4_0, Q5_0, Q8_0):
- Simple: one f16 scale per 32 elements
- Symmetric quantization
- Faster to encode/decode
- Less accurate for varied distributions

K-quant formats (Q4_K_M, Q5_K_M, Q6_K):
- Complex: 256-element superblocks with sub-group scales
- Asymmetric with minimums (better dynamic range)
- ~20-30% better perplexity at same bit rate
- Slightly slower encode/decode

**Use K-quants** (Q4_K_M, Q6_K) unless you need:
- Maximum inference speed (use Q4_0, Q8_0)
- Compatibility with older llama.cpp versions

---

## Common Workflows

### Multi-Tier Deployment

**Q: How do I create multiple quantized versions for different tiers?**

Store one GMAT, create multiple export configs:

```
model.gmat/                        # Full precision source
├── export/
│   ├── export-economy.json        # Q4_0 default, all tensors
│   ├── export-balanced.json       # Q4_K_M default, Q6_K critical
│   └── export-premium.json        # Q8_0 default, Q8_0 critical
└── gguf/
    ├── model-economy.gguf         # Smallest, fastest
    ├── model-balanced.gguf        # Production default
    └── model-premium.gguf         # Highest quality
```

All three GGUFs come from the same source tensors - quantization is just configuration.

### Fine-Tune Factory

**Q: I have 50 fine-tunes. How does GMAT help?**

Each fine-tune shares most tensors with the base model:

1. **Consistent workflow**: Same import/export process for all
2. **Tensor-level organization**: UUIDs identify shared vs changed tensors
3. **Reusable configs**: Same export profiles work across fine-tunes

Future roadmap includes deduplication - storing only the tensors that differ from base.

---

## Troubleshooting

### Scale Optimization Disabled

**Q: When should I use `"standard"` instead of `"trellis"`?**

Use standard scale optimization when:
- Debugging quantization issues (simpler to reason about)
- Comparing against other tools (match their behavior)
- Maximum encode speed (trellis adds ~10-20% overhead)

For production, trellis is almost always better.

### Dimension Alignment

**Q: Why does quantization fail for some tensors?**

K-quant formats require 256-element alignment. Legacy formats require 32-element alignment. If a tensor doesn't meet these requirements, GMAT falls back:

```
Requested Q4_K_M for 128-element tensor → Falls back to Q8_0 (32-aligned)
Requested Q8_0 for 17-element tensor → Error (not 32-aligned)
```

Small tensors (biases, layer norms) typically aren't quantized.

---

## See Also

- [[Technical-Details]] - Block formats, encoding, compression
- [[Configuration-Files]] - Full config reference
- [[Export-Command]] - Export options and examples
