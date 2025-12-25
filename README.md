# GMAT-CLI

**Model management infrastructure between training and inference. No GPU required.**

Convert HuggingFace models to compact high fidelity storage format, then export to GGUF with fine-grained quantization control. Read and store once, deploy many ways. Runs entirely on CPU with streaming processing—convert 70B+ models on any machine (just add time).

## The Problem

You're managing LLMs:
- Base model (70B) + 20 fine-tunes = 20 near-duplicate copies
- Different uses need different quantization
- No audit trail for "how was that model quantized?"
- Ad-hoc scripts that don't scale

## The Solution

```
SafeTensors → GMAT (tensor-addressed storage) → GGUF (multiple profiles)
```

**GMAT format** stores each tensor as an individual UUID-addressed file. This enables:
- **Quantization as configuration**: Same source → multiple deployment profiles
- **Per-tensor precision**: Embeddings at Q8, FFN at Q4
- **Reproducible builds**: Version-control your quantization configs
- **Future deduplication**: Tensor-level addressing for fine-tune management

## Quick Start

```bash
# Install
make install

# Import: SafeTensors → GMAT
gmat import --model ./model.safetensors --generate-config
gmat import --model ./model.safetensors --config import_config.json -o ./output

# Export: GMAT → GGUF (create different profiles from same source)
gmat export --model ./output/model.gmat --generate-config
gmat export --model ./output/model.gmat --config export_config.json -o model-q4.gguf
```

## Key Features

| Feature | Benefit |
|---------|---------|
| **CPU-only** | No GPU needed—runs on any server or laptop |
| Per-tensor quantization | Critical tensors stay high-precision |
| Streaming processing | Handle models larger than RAM |
| Auto-config generation | Extracts metadata from HuggingFace config.json |
| Quantization profiles | One source → custom configs |
| Sharded output | Split large models for deployment |

## Use Cases

**Fine-tune factory**: 50 fine-tunes of Llama 70B? Tensor-addressed storage enables consistent management and future deduplication.

**Reproducable**: JSON configs in git = full trail of quantization decisions.

**Large models**: Streaming pipeline processes 70B+ models without memory pressure.

## Documentation

See the [wiki](wiki/Home.md) for detailed documentation:
- [Installation](wiki/Installation.md)
- [Import Command](wiki/Import-Command.md)
- [Export Command](wiki/Export-Command.md)
- [Configuration Files](wiki/Configuration-Files.md)

## Examples

### Tiny LLM (Quick Start)
The `example/tiny_llm` directory contains a complete workflow with a small model:
```
tiny_llm.safetensors  → import_config.json → model.gmat/
                                              ↓
                        export_config.json → tiny_llm.gguf
```

### Kimi2-VL (Production Scale)
The `example/kimi2` directory demonstrates processing a 16B MoE vision-language model:

| Property | Value |
|----------|-------|
| Model | Kimi2-VL (moonshotai/Kimi-VL-A3B-Instruct) |
| Architecture | Vision-Language MoE (64 experts) |
| Tensors | 9,197 |
| Source | ~31 GB (7 shards) |
| GMAT | ~25 GB |

See [example/kimi2/README.md](example/kimi2/README.md) for full details.

## License

See [LICENSE](LICENSE).
