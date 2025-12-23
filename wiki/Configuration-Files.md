# Configuration Files

GMAT CLI uses JSON configuration files to control import and export operations.

## Import Configuration

### File: `import_config.json`

```json
{
  "source_format": "safetensors",
  "block_format": "B8x8",
  "tensor_map": [
    {
      "source": "model.embed_tokens.weight",
      "target": "e268e395-888f-49ed-9f59-24ea9d10be77",
      "include": true
    },
    {
      "source": "model.layers.0.self_attn.q_proj.weight",
      "target": "df11f72a-85ce-41e1-b354-1253749bd43d",
      "include": true
    }
  ],
  "metadata": {
    "architecture": "llama",
    "vocab_size": 32000,
    "hidden_size": 192,
    "num_layers": 1,
    "num_attention_heads": 3,
    "intermediate_size": 1024,
    "max_position_embeddings": 1024
  }
}
```

### Import Fields

| Field | Type | Description |
|-------|------|-------------|
| `source_format` | string | Source format: `"safetensors"` or `"gguf"` |
| `block_format` | string | Block format for storage (see [[Import-Command#block-formats]]) |
| `tensor_map` | array | List of tensor mappings |
| `metadata` | object | Model metadata to preserve |

### Tensor Mapping (Import)

| Field | Type | Description |
|-------|------|-------------|
| `source` | string | Source tensor name from the model |
| `target` | string | UUID for GMAT storage |
| `include` | boolean | Whether to include this tensor |

### Model Metadata

| Field | Type | Description |
|-------|------|-------------|
| `architecture` | string | Model architecture (e.g., `"llama"`, `"mistral"`) |
| `vocab_size` | integer | Vocabulary size |
| `hidden_size` | integer | Hidden dimension size |
| `num_layers` | integer | Number of transformer layers |
| `num_attention_heads` | integer | Number of attention heads |
| `intermediate_size` | integer | FFN intermediate size |
| `max_position_embeddings` | integer | Maximum sequence length |

---

## Export Configuration

### File: `export_config.json`

```json
{
  "target_format": "gguf",
  "quantization": {
    "default_type": "q4_k_m",
    "scale_optimization": "trellis",
    "trellis_lambda": 0.3,
    "per_tensor": {
      "e268e395-888f-49ed-9f59-24ea9d10be77": "q6_k",
      "3b6d6f4a-9cf5-4d97-9b08-ce07ddc0435d": "q6_k",
      "df11f72a-85ce-41e1-b354-1253749bd43d": "q8_0"
    }
  },
  "tensor_map": [
    {
      "source": "3b6d6f4a-9cf5-4d97-9b08-ce07ddc0435d",
      "target": "output.weight"
    },
    {
      "source": "e268e395-888f-49ed-9f59-24ea9d10be77",
      "target": "token_embd.weight"
    }
  ],
  "shard_size": null
}
```

### Export Fields

| Field | Type | Description |
|-------|------|-------------|
| `target_format` | string | Target format: `"gguf"` (currently only GGUF is supported) |
| `quantization` | object | Quantization configuration |
| `tensor_map` | array | List of tensor export mappings |
| `shard_size` | integer/null | Shard size in bytes (null for single file) |

### Quantization Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_type` | string | `"q8_0"` | Default quantization type for all tensors |
| `scale_optimization` | string | `"trellis"` | Scale optimization: `"standard"` or `"trellis"` |
| `trellis_lambda` | float | `0.3` | Smoothness penalty for trellis optimization |
| `per_tensor` | object | `{}` | Per-tensor quantization overrides (UUID -> type) |

### Tensor Mapping (Export)

| Field | Type | Description |
|-------|------|-------------|
| `source` | string | GMAT tensor UUID |
| `target` | string | Target tensor name in GGUF format |

---

## Example from tiny_llm

### Import Config (`example/tiny_llm/import_config.json`)

This configuration imports a tiny LLaMA-style model:

```json
{
  "source_format": "safetensors",
  "block_format": "B8x8",
  "tensor_map": [
    {
      "source": "model.layers.0.post_attention_layernorm.weight",
      "target": "9852d94a-bbd1-4a03-b523-d1d6b84e4c8e",
      "include": true
    },
    {
      "source": "lm_head.weight",
      "target": "3b6d6f4a-9cf5-4d97-9b08-ce07ddc0435d",
      "include": true
    }
  ],
  "metadata": {
    "architecture": "llama",
    "vocab_size": 32000,
    "hidden_size": 192,
    "num_layers": 1,
    "num_attention_heads": 3,
    "intermediate_size": 1024,
    "max_position_embeddings": 1024
  }
}
```

### Export Config (`example/tiny_llm/export_config.json`)

This configuration exports with mixed quantization:

```json
{
  "target_format": "gguf",
  "quantization": {
    "default_type": "q4_k_m",
    "per_tensor": {
      "df11f72a-85ce-41e1-b354-1253749bd43d": "q8_0",
      "8bc97abd-918e-4b33-9bb2-cc57761bfbc6": "q8_0",
      "e268e395-888f-49ed-9f59-24ea9d10be77": "q6_k",
      "3b6d6f4a-9cf5-4d97-9b08-ce07ddc0435d": "q6_k"
    }
  },
  "tensor_map": [
    {
      "source": "3b6d6f4a-9cf5-4d97-9b08-ce07ddc0435d",
      "target": "output.weight"
    },
    {
      "source": "e268e395-888f-49ed-9f59-24ea9d10be77",
      "target": "token_embd.weight"
    },
    {
      "source": "99ff4033-4060-4b3a-b2ad-ec48d5abd1c0",
      "target": "blk.0.ffn_down.weight"
    }
  ]
}
```

---

## GMAT Model Metadata

After import, the GMAT model contains `metadata.json`:

```json
{
  "config": {
    "block_format": "B8x8",
    "metadata": {
      "architecture": "llama",
      "hidden_size": 192,
      "intermediate_size": 1024,
      "max_position_embeddings": 1024,
      "num_attention_heads": 3,
      "num_layers": 1,
      "vocab_size": 32000
    },
    "source_format": "safetensors",
    "tensor_map": [...]
  },
  "tensor_name_map": {
    "lm_head.weight": "3b6d6f4a-9cf5-4d97-9b08-ce07ddc0435d",
    "model.embed_tokens.weight": "e268e395-888f-49ed-9f59-24ea9d10be77"
  },
  "total_tensors": 9
}
```

This metadata is used when generating export configurations and provides a mapping between original tensor names and their GMAT UUIDs.

---

## Tips

1. **Generate configs first**: Always use `--generate-config` to create a template, then customize.

2. **Exclude unnecessary tensors**: Set `include: false` for layer norms or other tensors you don't need.

3. **Use per-tensor quantization**: Important tensors (embeddings, output) benefit from higher quality quantization.

4. **Shard large models**: For models >5GB, use `shard_size` to split output files.

5. **Preserve metadata**: Fill in the metadata section for better model documentation.
