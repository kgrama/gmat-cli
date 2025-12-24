# GMAT CLI

**Model management infrastructure between training and inference. No GPU required.**

GMAT-CLI converts HuggingFace models (SafeTensors) to tensor-addressed GMAT format, then exports to production-ready GGUF with fine-grained quantization control. All processing runs on CPU with bounded memory usage—convert 70B+ models on any machine without specialized hardware.

## Why GMAT?

### The Problem

Managing LLMs in production is painful:

| Challenge | Reality |
|-----------|---------|
| **Storage explosion** | Base model + 20 fine-tunes = 20 near-duplicate 70B copies |
| **Quantization chaos** | Ad-hoc decisions, no reproducibility |
| **No audit trail** | "How was that production model quantized?" |
| **One-size-fits-all** | Same quantization for embeddings and FFN layers? |

### The GMAT Approach

```
SafeTensors → GMAT → GGUF
   (source)   (managed)  (deployed)
```

**1. Tensor-Addressed Storage**

GMAT stores each tensor as an individual UUID-addressed file:
```
model.gmat/
├── metadata.json
└── tensors/
    ├── 3b6d6f4a-9cf5-4d97-9b08-ce07ddc0435d.gmat  # output.weight
    ├── e268e395-888f-49ed-9f59-24ea9d10be77.gmat  # token_embd.weight
    └── ...
```

This enables:
- Tensor-level versioning and diffing
- Foundation for deduplication across fine-tunes
- Individual tensor replacement/updates

**2. Quantization as Configuration**

Store full-precision GMAT once. Create multiple export configs:

```
model.gmat/
    ├── export-q4-cost.json      → model-q4.gguf (cost tier)
    ├── export-q8-premium.json   → model-q8.gguf (quality tier)
    └── export-q6-balanced.json  → model-q6.gguf (balanced)
```

Same source tensors, different deployment profiles. Version-control your quantization decisions.

**3. Per-Tensor Precision Control**

Not all tensors are equal:

| Tensor Type | Impact | Recommendation |
|-------------|--------|----------------|
| Embeddings (`token_embd`) | High | Q6-Q8 |
| Output layer (`lm_head`) | High | Q6-Q8 |
| Attention Q/K | Medium | Q4-Q5 |
| FFN gate/up | Lower | Q4 |

GMAT auto-recommends based on tensor analysis, with full override capability.

**4. Reproducible, Auditable Builds**

JSON configs capture every decision:
- Which tensors included
- Per-tensor quantization types
- Scale optimization settings
- Model metadata

Put configs in git. Know exactly how any deployment was built.

## Use Cases

### Fine-Tune Factory

You have 50 fine-tunes of Llama 70B. Each shares 99% of tensors with base model.

GMAT's tensor-addressed storage:
- Consistent workflow across all fine-tunes
- Tensor-level organization enables future deduplication
- Same export configs work across fine-tunes

### Multi-Tier Deployment

Different SLAs need different quality/cost tradeoffs:

| Tier | Quantization | Use Case |
|------|--------------|----------|
| Premium | Q8 default, embeddings Q8 | Quality-critical |
| Standard | Q6 default, embeddings Q8 | Balanced |
| Economy | Q4 default, embeddings Q6 | Cost-sensitive |

One GMAT source → three export configs → three deployment targets.

### Compliance & Audit

Regulated industries need reproducibility:
- JSON configs in version control
- Full trail of quantization decisions
- Rebuild any historical deployment exactly

### Large Model Processing

70B+ models don't fit in RAM. GMAT handles this on CPU:
- **No GPU required**—runs on any server or laptop
- Streaming producer-consumer pipeline with bounded memory
- Memory-mapped I/O avoids loading full files
- Parallel processing via Rayon (scales with CPU cores)
- Sharded output for deployment flexibility

## Quick Start

```bash
# Install
make install

# Import: SafeTensors → GMAT
gmat import --model ./model.safetensors --generate-config
# Edit import_config.json if needed
gmat import --model ./model.safetensors --config import_config.json -o ./output

# Export: GMAT → GGUF
gmat export --model ./output/model.gmat --generate-config
# Edit export_config.json for your quantization profile
gmat export --model ./output/model.gmat --config export_config.json -o model.gguf
```

## Documentation

- [Installation](Installation.md) - Build and install
- [Import Command](Import-Command.md) - SafeTensors to GMAT conversion
- [Export Command](Export-Command.md) - GMAT to GGUF with quantization
- [Configuration Files](Configuration-Files.md) - Config file reference
- [Technical Details](Technical-Details.md) - Block formats, encoding, compression tradeoffs
- [FAQ](FAQ.md) - Defaults, trellis optimization, static saliency

## Example

See `example/tiny_llm/` for a complete multi-profile workflow:

```
tiny_llm.safetensors  →  model.gmat/  →  tiny_llm-economy.gguf   (Q4, smallest)
                              ↓
                         export configs  →  tiny_llm-balanced.gguf  (Q4/Q6 mix)
                              ↓
                              →  tiny_llm-premium.gguf  (Q8, highest quality)
```

Three export profiles from one GMAT source - illustrating quantization as configuration.
