# Export Command

The `export` command converts GMAT models to GGUF format with quantization.

## Usage

```bash
gmat export [OPTIONS]
```

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--model <PATH>` | `-m` | Path to the GMAT model directory (required) |
| `--config <PATH>` | `-c` | Path to export config JSON (optional) |
| `--output <PATH>` | `-o` | Output path for exported model (default: `model.gguf`) |
| `--shard-size <MB>` | | Shard size in MB (e.g., 5000 for 5GB). Overrides config file. |
| `--generate-config` | | Generate a template config instead of exporting |

## Workflow

### Step 1: Generate Configuration

Generate an export configuration from your GMAT model:

```bash
gmat export --model ./model.gmat --generate-config
```

This analyzes all tensors and creates `export_config.json` with:
- Tensor UUID to GGUF name mappings
- Recommended quantization types based on tensor importance
- Default quantization settings

### Step 2: Edit Configuration (Optional)

Customize `export_config.json`:
- Adjust per-tensor quantization types
- Change default quantization
- Configure sharding for large models
- Modify scale optimization settings

### Step 3: Run Export

```bash
gmat export --model ./model.gmat --config export_config.json --output model.gguf
```

## Quantization Types

### Legacy Quantization

| Type | Bits | Description |
|------|------|-------------|
| `q8_0` | 8 | High quality, larger size |
| `q5_0` | 5 | Balanced |
| `q5_1` | 5 | Balanced with different encoding |
| `q4_0` | 4 | Smaller size |
| `q4_1` | 4 | Smaller with different encoding |

### K-Quantization (Recommended)

| Type | Description |
|------|-------------|
| `q2_k` | Smallest, lowest quality |
| `q3_k_s` | 3-bit small variant |
| `q3_k_m` | 3-bit medium variant |
| `q3_k_l` | 3-bit large variant |
| `q4_k_s` | 4-bit small variant |
| `q4_k_m` | 4-bit medium variant (default) |
| `q5_k_s` | 5-bit small variant |
| `q5_k_m` | 5-bit medium variant |
| `q6_k` | 6-bit, high quality |

### I-Quantization

| Type | Description |
|------|-------------|
| `iq4_xs` | Importance-weighted 4-bit extra-small |
| `iq4_nl` | Importance-weighted 4-bit non-linear |

## Automatic Quantization Recommendations

When generating a config, the tool recommends quantization based on:

| Tensor Type | Recommendation |
|-------------|----------------|
| Token embeddings (`embed_tokens`) | `q8_0` or `q6_k` |
| LM head (`lm_head`) | `q8_0` or `q6_k` |
| Attention output (`o_proj`) | `q6_k` or `q5_k_m` |
| Attention value (`v_proj`) | `q6_k` or `q5_k_m` |
| FFN down projection | `q6_k` or `q5_k_m` |
| Attention Q/K projections | `q4_k_m` |
| FFN gate/up projections | `q4_k_m` |
| Small tensors (<1M params) | `q8_0` |

## Scale Optimization

Two optimization methods are available:

| Method | Description |
|--------|-------------|
| `standard` | Standard scale optimization |
| `trellis` | Trellis quantization with smoothness penalty (default) |

Configure trellis lambda (smoothness penalty) with `trellis_lambda` (default: 0.3).

## Sharding

For large models, enable sharding to split output into multiple files:

```bash
# Via command line (overrides config)
gmat export --model ./model.gmat --output model.gguf --shard-size 5000

# Via config file
{
  "shard_size": 5000000000
}
```

Output files will be named: `model-00001-of-00003.gguf`, etc.

## Example

Using the example from `example/tiny_llm/`:

```bash
cd example/tiny_llm

gmat export \
  --model model.gmat \
  --config export_config.json \
  --output tiny_llm.gguf
```

Output:

```
Exporting to GGUF: tiny_llm.gguf
Scale optimization: Trellis { lambda: 0.3 }

Processing 9 tensors...
[1/9] output.weight [32000x192] -> Q6_K
[2/9] token_embd.weight [32000x192] -> Q6_K
[3/9] blk.0.ffn_down.weight [192x1024] -> Q8_0
...
Done! 9 tensors, 12582912 bytes
```

## Tensor Name Mapping

SafeTensor names are automatically mapped to GGUF conventions:

| SafeTensor Name | GGUF Name |
|----------------|-----------|
| `model.embed_tokens.weight` | `token_embd.weight` |
| `lm_head.weight` | `output.weight` |
| `model.norm.weight` | `output_norm.weight` |
| `model.layers.N.self_attn.q_proj.weight` | `blk.N.attn_q.weight` |
| `model.layers.N.self_attn.k_proj.weight` | `blk.N.attn_k.weight` |
| `model.layers.N.self_attn.v_proj.weight` | `blk.N.attn_v.weight` |
| `model.layers.N.self_attn.o_proj.weight` | `blk.N.attn_output.weight` |
| `model.layers.N.mlp.gate_proj.weight` | `blk.N.ffn_gate.weight` |
| `model.layers.N.mlp.up_proj.weight` | `blk.N.ffn_up.weight` |
| `model.layers.N.mlp.down_proj.weight` | `blk.N.ffn_down.weight` |
| `model.layers.N.input_layernorm.weight` | `blk.N.attn_norm.weight` |
| `model.layers.N.post_attention_layernorm.weight` | `blk.N.ffn_norm.weight` |

## Memory Efficiency

Like import, export uses streaming to minimize memory:
- Tensors are quantized in parallel using Rayon
- A bounded channel limits in-flight tensors
- Results are written to GGUF incrementally

See also: [Configuration Files](Configuration-Files.md#export-configuration)
