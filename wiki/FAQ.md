# FAQ

Frequently asked questions about GMAT-CLI, organized by topic for easy navigation.

---

## 1. General Questions

### What is GMAT-CLI?

GMAT-CLI is a CPU-only model management tool that converts SafeTensors models to GGUF format with quantization. It provides an intermediate GMAT storage format that enables flexible, configuration-driven quantization without re-importing models.

**Key workflow:**
```
SafeTensors → gmat import → GMAT Storage → gmat export → GGUF (quantized)
```

The GMAT intermediate format uses tensor-addressed storage with UUID-based individual tensor files, making quantization a configuration choice rather than a one-time conversion.

### Why use GMAT-CLI instead of llama.cpp directly?

GMAT-CLI offers several advantages:

1. **CPU-only processing**: No GPU required for conversion or quantization
2. **Large model support**: Stream 70B+ models with bounded memory usage
3. **Flexible quantization**: Store once, export multiple quantization profiles
4. **Per-tensor control**: Override quantization for specific tensors (e.g., Q8_0 embeddings, Q4_K_M FFN layers)
5. **Production workflow**: Optimized for converting many models/fine-tunes with consistent configs

If you're managing multiple quantization profiles or processing models on CPU-only infrastructure, GMAT-CLI is designed for your workflow.

### Is GPU required?

**No.** GMAT-CLI runs entirely on CPU. There are no CUDA, ROCm, or GPU dependencies. This means you can:
- Convert models on any server or laptop
- Run in CI/CD pipelines without GPU runners
- Process 70B+ models with just CPU and sufficient RAM/disk

The streaming pipeline uses bounded memory, so even large models work without excessive RAM.

### What platforms are supported?

- **Linux**: Full support (primary platform)
- **macOS**: Full support
- **Windows**: WSL2 recommended (native Windows not officially tested)

NVMe storage is recommended for best performance, especially with 70B+ models.

---

## 2. Installation & Setup

### What Rust version is required?

**Rust 1.70+** is required. Install from [rustup.rs](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update
```

Verify installation:
```bash
rustc --version  # Should show 1.70 or higher
```

### How do I fix build errors?

Common build issues:

**Issue: "error: linker 'cc' not found"**
```bash
# Ubuntu/Debian
sudo apt install build-essential

# macOS (install Xcode Command Line Tools)
xcode-select --install
```

**Issue: "failed to fetch ..."**
```bash
# Update cargo registry
cargo clean
rm -rf ~/.cargo/registry/index/*
cargo build --release
```

**Issue: Out of memory during build**
```bash
# Limit parallel jobs
cargo build --release -j 2
```

### How do I configure PATH after installation?

After `cargo install gmat-cli`, ensure `~/.cargo/bin` is in your PATH:

```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH="$HOME/.cargo/bin:$PATH"

# Reload shell or source config
source ~/.bashrc

# Verify
gmat --version
```

---

## 3. Import Questions

### What input formats are supported?

**Primary format: SafeTensors**
- Single-file `.safetensors` models
- Multi-file sharded SafeTensors (e.g., `model-00001-of-00003.safetensors`)

**NOT supported:**
- PyTorch `.pth` or `.bin` files (convert to SafeTensors first)
- GGUF imports (planned for future versions)

To convert PyTorch to SafeTensors, use:
```python
from safetensors.torch import save_file
save_file(model.state_dict(), "model.safetensors")
```

### How do I handle large models (70B+)?

For models over 32GB:

1. **Ensure sufficient RAM**: 32GB+ recommended for 70B models
2. **Use NVMe storage**: Faster I/O reduces processing time
3. **Monitor disk space**: Need 2-3x model size for GMAT storage + temp files
4. **Import with default block format**: B8x8 balances size and accuracy

Example for 70B model:
```bash
gmat import model.safetensors output.gmat
# Uses streaming pipeline with bounded memory
```

The streaming producer-consumer pipeline processes tensors incrementally, avoiding full model load.

### Does GMAT support sharded models?

**Yes.** Multi-file SafeTensors are automatically detected and merged:

```bash
gmat import model-00001-of-00003.safetensors output.gmat
# Automatically finds and processes all shards
```

All shards must be in the same directory with consistent naming:
- `model-00001-of-00003.safetensors`
- `model-00002-of-00003.safetensors`
- `model-00003-of-00003.safetensors`

### How do I generate a config file?

Use `--generate-config` during import:

```bash
gmat import model.safetensors output.gmat --generate-config
# Creates: output.gmat/import_config.json
```

This generates a template you can modify for custom block formats or per-tensor settings. See [Configuration Files](Configuration-Files.md) for full schema.

---

## 4. Export & Quantization Questions

### Which quantization type should I choose?

| Format | Bits | Quality | Size | Best For |
|--------|------|---------|------|----------|
| **Q8_0** | 8 | Excellent | Large | Quality-critical, plenty of VRAM |
| **Q6_K** | 6 | Very good | Medium | High-quality production |
| **Q4_K_M** | 4 | Good | Small | General production **(default)** |
| **Q4_0** | 4 | Acceptable | Smallest | Edge/mobile, maximum compression |
| **Q2_K** | 2-3 | Experimental | Tiny | Research/extreme compression |

**Recommendation:**
- Start with **Q4_K_M** (default) for balanced quality/size
- Use **Q6_K** or **Q8_0** for critical tensors (embeddings, output layer)
- Use **Q4_0** for maximum compression on resource-constrained devices

### What are the quality vs size tradeoffs?

Measured on LLaMA-7B (approximate perplexity increase):

| Format | Size (GB) | Perplexity Δ | vs Q8_0 Quality |
|--------|-----------|--------------|-----------------|
| Q8_0 | 7.0 | +0.00 | 100% (baseline) |
| Q6_K | 5.4 | +0.05 | ~98% |
| Q4_K_M | 3.9 | +0.15 | ~95% |
| Q4_0 | 3.7 | +0.30 | ~90% |
| Q2_K | 2.8 | +0.80 | ~75% |

**Rule of thumb:** Each halving of bits costs 5-10% perplexity for well-optimized quantization.

### What's the difference between IQ4_XS and IQ4_NL?

Both are 4-bit I-quant formats with improved accuracy over Q4_K_M:

| Format | Approach | Quality | Size | Speed |
|--------|----------|---------|------|-------|
| **Q4_K_M** | K-quant with superblocks | Baseline | Baseline | Fast |
| **IQ4_NL** | Non-linear quantization | +2-3% perplexity | +5% size | Medium |
| **IQ4_XS** | Extra-small codebook | +1-2% perplexity | Same as Q4_K_M | Slower |

**When to use:**
- **IQ4_NL**: Best quality at 4-bit, slightly larger files
- **IQ4_XS**: Same size as Q4_K_M, better quality, slower inference
- **Q4_K_M**: Balanced default, widest compatibility

**Compatibility note:** I-quant formats require llama.cpp commit `b1696` or later.

### What is Trellis optimization?

Trellis optimization uses dynamic programming to find globally optimal scale assignments across quantization blocks, minimizing total error while encouraging smoothness.

```
Standard:  Each block picks its own optimal scale independently
Trellis:   DP finds globally-optimal scales with smoothness penalty
```

**Benefits:**
- 0.5-2% perplexity improvement
- Minimal encoding overhead (~10-20% slower)
- No calibration data required

**Configuration:**
```json
{
  "quantization": {
    "scale_optimization": "trellis",
    "trellis_lambda": 0.3
  }
}
```

**Lambda values:**
- `0.0`: No smoothing (equivalent to standard)
- `0.3`: Moderate smoothing **(default)**
- `1.0`: Strong smoothing (may over-smooth)

**When to disable:** Use `"standard"` when debugging quantization issues or matching other tools' behavior.

### How do per-tensor overrides work?

Override quantization for specific tensors using their UUIDs:

```json
{
  "quantization": {
    "default_type": "q4_k_m",
    "per_tensor": {
      "<embedding-uuid>": "q8_0",
      "<output-uuid>": "q8_0",
      "<attn-q-uuid>": "q6_k"
    }
  }
}
```

**Why use overrides?**

| Tensor Type | Sensitivity | Recommendation |
|-------------|-------------|----------------|
| `token_embd.weight` | **High** | Q6_K or Q8_0 |
| `output.weight` | **High** | Q6_K or Q8_0 |
| `attn_q/k/v.weight` | Medium | Q4_K_M to Q6_K |
| `ffn_*.weight` | Lower | Q4_K_M or Q4_0 |

Embeddings and output layers handle token vocabulary directly - quantization errors affect every token. FFN layers are more tolerant.

---

## 5. Performance Questions

### How much memory do I need?

**Minimum requirements by model size:**

| Model Size | RAM Required | Storage Required | CPU Cores |
|------------|-------------|------------------|-----------|
| 7B | 8GB | 20GB | 4+ |
| 13B | 16GB | 40GB | 4+ |
| 30B | 24GB | 90GB | 8+ |
| 70B+ | 32GB+ | 200GB+ | 8+ |

**Storage:** Need 2-3x model size for GMAT intermediate storage + temp files. NVMe recommended.

**RAM:** Bounded by streaming pipeline, but larger models need more memory for parallel processing buffers.

### How many CPU cores are recommended?

- **Minimum:** 4 cores (will work but slower)
- **Recommended:** 8+ cores for optimal parallelization
- **Ideal:** 16+ cores for 70B+ models

GMAT uses Rayon for parallel tensor processing. More cores = faster import/export, especially for large models.

### How can I optimize processing speed?

**Hardware:**
1. Use NVMe storage (10x faster than HDD)
2. More CPU cores (linear scaling up to ~16 cores)
3. Faster RAM (3200MHz+ helps with large models)

**Software:**
1. Close unnecessary applications (free up RAM)
2. Use default B8x8 block format (faster than B16x8)
3. Disable trellis for faster encoding: `"scale_optimization": "standard"`

**Example:** 70B model on 16-core CPU with NVMe:
- Import: ~15-25 minutes
- Export (Q4_K_M): ~20-35 minutes

---

## 6. Troubleshooting

### "Error: Out of memory" during import/export

**Causes:**
1. Insufficient RAM for model size
2. Too many parallel workers for available memory
3. Memory leak or fragmentation

**Solutions:**
```bash
# 1. Limit parallel workers (reduces memory usage)
export RAYON_NUM_THREADS=4
gmat import model.safetensors output.gmat

# 2. Free up memory
# Close browsers, IDEs, other applications

# 3. Check available memory
free -h  # Linux
vm_stat  # macOS

# 4. For 70B+ models, ensure 32GB+ RAM
```

### "Error: Unsupported tensor shape" during quantization

**Cause:** Tensor doesn't meet alignment requirements:
- K-quant formats (Q4_K_M, Q6_K) require 256-element alignment
- Legacy formats (Q4_0, Q8_0) require 32-element alignment

**Solution:** GMAT auto-falls back to Q8_0 for misaligned tensors. If this fails:
```json
{
  "quantization": {
    "per_tensor": {
      "<problematic-uuid>": "f16"
    }
  }
}
```

Small tensors (biases, layer norms) are typically kept at full precision.

### Processing is very slow

**Check these factors:**

1. **Storage speed:**
   ```bash
   # Test disk speed
   dd if=/dev/zero of=testfile bs=1G count=1 oflag=direct
   # Should see 500MB/s+ for NVMe, 100MB/s+ for SSD
   ```

2. **CPU usage:**
   ```bash
   htop  # or top
   # Should see near 100% CPU on all cores during processing
   ```

3. **Memory pressure:**
   ```bash
   free -h
   # If swap is heavily used, you need more RAM
   ```

4. **Block format:** B16x8 is slower than B8x8 (default)

### "Error: Failed to parse SafeTensors header"

**Causes:**
1. Corrupted download
2. Not a SafeTensors file (PyTorch .bin, etc.)
3. Incomplete file (download interrupted)

**Solutions:**
```bash
# 1. Verify file integrity
ls -lh model.safetensors
# Check size matches expected

# 2. Verify SafeTensors format
head -c 100 model.safetensors
# Should start with JSON header

# 3. Re-download or convert from PyTorch
```

### Exported GGUF fails to load in llama.cpp

**Check compatibility:**

1. **I-quant formats:** Require llama.cpp commit `b1696+`
   ```bash
   cd llama.cpp
   git log --oneline | head -1
   # Should be b1696 or later for IQ4_XS/IQ4_NL
   ```

2. **Architecture support:** Verify model type is supported
3. **File corruption:** Re-export GGUF with `--verbose` for debugging

---

## 7. Compatibility

### What llama.cpp versions are compatible?

| GMAT Feature | llama.cpp Requirement |
|--------------|----------------------|
| Q4_K_M, Q6_K | Any recent version |
| Q8_0, Q4_0 | Any version |
| IQ4_XS, IQ4_NL | Commit `b1696+` |
| Sharded GGUF | Recent versions (2024+) |

**Recommendation:** Use latest llama.cpp release for best compatibility.

### What model architectures are supported?

**Text models:**
- LLaMA (1, 2, 3)
- Qwen, Qwen2
- Phi (1, 2, 3)
- Gemma
- DeepSeek
- Mistral, Mixtral (MoE)

**Vision-language models:**
- LLaVA
- Qwen-VL
- Kimi-VL
- InternVL

**Encoder-decoder models:**
- T5
- BART
- Whisper

See [Technical Details](Technical-Details.md) for architecture-specific details.

### Will GGUF import be supported?

**Planned for future versions.** Current workflow is SafeTensors → GMAT → GGUF.

For now, to modify existing GGUF quantization:
1. Obtain original SafeTensors model
2. Import to GMAT
3. Export with desired quantization

### Will other export formats be supported?

Current focus is GGUF as the primary inference format. Future considerations:
- ONNX export
- TensorRT-LLM export
- Custom quantization formats

Submit feature requests on GitHub if you need specific formats.

---

## See Also

- [Home](Home.md) - Overview and navigation
- [Installation](Installation.md) - Setup and prerequisites
- [Import Command](Import-Command.md) - Import usage and options
- [Export Command](Export-Command.md) - Export and quantization reference
- [Configuration Files](Configuration-Files.md) - Config schema and examples
- [Technical Details](Technical-Details.md) - Deep dive into block formats and algorithms
