# Kimi2-VL Example

This example demonstrates GMAT processing of **Kimi2-VL**, a 16B MoE (Mixture of Experts) vision-language model from Moonshot AI.

## Model Overview

| Property | Value |
|----------|-------|
| Model | Kimi2-VL (moonshotai/Kimi-VL-A3B-Instruct) |
| Architecture | kimi_vl (vision-language MoE) |
| Parameters | ~16B total |
| Vocab Size | 163,840 |
| Experts | 64 experts per layer |
| Tensors | 9,197 |
| Source Size | ~31 GB (7 shards) |
| GMAT Size | ~25 GB |

## Directory Structure

```
kimi2/
├── import_config.json      # Import settings (abbreviated, showing key tensor patterns)
├── model.json              # Model metadata from GMAT storage
├── export_config.json      # Export settings with per-tensor quantization
└── README.md               # This file
```

## Tensor Patterns

The model includes several tensor categories:

### Language Model (MoE)
```
language_model.model.layers.{0-11}.mlp.experts.{0-63}.{gate,up,down}_proj.weight
language_model.model.layers.{0-11}.mlp.shared_experts.{gate,up,down}_proj.weight
language_model.model.layers.{0-11}.self_attn.{q,kv_a,kv_b,o}_proj.weight
language_model.model.layers.{0-11}.{input,post_attention}_layernorm.weight
```

### Vision Tower
```
vision_tower.encoder.blocks.{0-26}.{wqkv,wo}.{weight,bias}
vision_tower.encoder.blocks.{0-26}.mlp.{fc0,fc1}.{weight,bias}
vision_tower.encoder.blocks.{0-26}.norm{0,1}.{weight,bias}
vision_tower.patch_embed.proj.{weight,bias}
vision_tower.encoder.final_layernorm.{weight,bias}
```

### Multi-Modal Projector
```
multi_modal_projector.linear_{1,2}.{weight,bias}
multi_modal_projector.pre_norm.weight
```

## Quantization Strategy

The export config uses mixed precision:

| Tensor Type | Quantization | Reason |
|-------------|--------------|--------|
| Default | Q4_K_M | Balance of size/quality |
| Embeddings | Q6_K | Critical for vocab accuracy |
| LayerNorm | Q8_0 | Small tensors, keep precision |
| Vision biases | Q8_0 | Small tensors |
| LM Head | Q6_K | Output quality |

> **Note**: MoE-aware quantization (per-expert settings) is not yet supported. All expert weights use the default quantization type.

## Workflow

### Import
```bash
# Download model from HuggingFace
hf download moonshotai/Kimi-VL-A3B-Instruct --local-dir ./kimi2-hf

# Generate initial config (scans directory for all shards)
gmat import --model ./kimi2-hf/ --generate-config

# Import all shards
gmat import --model ./kimi2-hf/ --config import_config.json -o ./kimi2-gmat
```

### Export to GGUF
```bash
# Export with Q4_K_M default, trellis optimization
gmat export \
  --model ./kimi2-gmat/model.gmat \
  --config export_config.json \
  -o kimi2-q4km.gguf
```

## Key Features Demonstrated

1. **Multi-Shard Import**: 7 SafeTensor shards → single GMAT store
2. **Vision-Language**: Separate vision encoder + projector + language model
3. **Large Scale**: 9,197 tensors managed with UUID-based addressing
4. **Per-Tensor Quantization**: Critical tensors (embeddings, norms) kept at higher precision
5. **Trellis Optimization**: `scale_optimization: "trellis"` with lambda=0.3 for quality quantization

## Limitations

- **MoE metadata**: Architecture fields (hidden_size, num_layers, etc.) not fully extracted from kimi_vl config
- **MoE quantization**: No per-expert quantization support yet; all experts use default quant type
