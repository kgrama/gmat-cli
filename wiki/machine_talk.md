# GMAT-CLI LLM Reference

Minimal token reference for AI assistants.

## What is GMAT-CLI

GMAT-CLI converts HuggingFace model files (SafeTensors) to llama.cpp inference files (GGUF) with quantization. Runs on CPU only - no GPU needed. Handles models larger than RAM (70B+) via streaming.

### Two-Phase Workflow

```
SafeTensors (.safetensors) → [gmat import] → GMAT Storage → [gmat export] → GGUF (.gguf)
```

**Phase 1 - Import**: Reads SafeTensors weights, stores each tensor as separate UUID-named file in a directory. This intermediate format enables:
- One import, multiple exports with different quantization
- Per-tensor analysis before quantization
- Resume/modify without re-downloading

**Phase 2 - Export**: Reads GMAT storage, applies quantization (Q4_K_M, Q8_0, etc.), writes GGUF file for llama.cpp/ollama inference.

### Supported Models

Text: Llama, Qwen, DeepSeek, Mixtral, Mistral, Gemma, Phi, MiniMax-M2
Vision: LLaVA, Qwen-VL, Kimi-VL, InternVL
Encoder-decoder: T5, BART, Whisper

## Commands

```bash
# Import: SafeTensors → GMAT
gmat import --model <path> --generate-config  # auto-generate config
gmat import --model <path> --config import_config.json -o <output>

# Export: GMAT → GGUF
gmat export --model <gmat-dir> --generate-config  # auto-generate config
gmat export --model <gmat-dir> --config export_config.json -o model.gguf
```

## Config: import_config.json

```json
{
  "source_format": "safetensors",
  "block_format": "B16x8",
  "tensor_map": [{"source_name": "model.embed_tokens.weight", "uuid": "..."}],
  "metadata": {"architecture": "llama", "vocab_size": 128256}
}
```

## Config: export_config.json

```json
{
  "target_format": "gguf",
  "quantization": {
    "default_type": "q4_k_m",
    "scale_optimization": "trellis",
    "trellis_lambda": 0.3,
    "per_tensor": {"<uuid>": "q8_0"}
  },
  "tensor_map": [{"source": "<uuid>", "target": "blk.0.attn_q.weight"}],
  "shard_size": 5000000000,
  "special_token_keys": {
    "bos": "tokenizer.ggml.bos_token_id",
    "eos": "tokenizer.ggml.eos_token_id"
  }
}
```

## Tensor Name Mapping (SafeTensors → GGUF)

| SafeTensors | GGUF |
|-------------|------|
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

### MoE Tensors

| SafeTensors | GGUF |
|-------------|------|
| `model.layers.N.mlp.gate.weight` | `blk.N.ffn_gate_inp.weight` |
| `model.layers.N.mlp.experts.E.gate_proj.weight` | `blk.N.ffn_gate.E.weight` |
| `model.layers.N.mlp.experts.E.up_proj.weight` | `blk.N.ffn_up.E.weight` |
| `model.layers.N.mlp.experts.E.down_proj.weight` | `blk.N.ffn_down.E.weight` |
| `model.layers.N.mlp.shared_experts.gate_proj.weight` | `blk.N.ffn_gate_shexp.weight` |
| `model.layers.N.block_sparse_moe.experts.E.w1.weight` | `blk.N.ffn_gate.E.weight` |
| `model.layers.N.block_sparse_moe.experts.E.w2.weight` | `blk.N.ffn_down.E.weight` |
| `model.layers.N.block_sparse_moe.experts.E.w3.weight` | `blk.N.ffn_up.E.weight` |

## Quantization Types

| Type | Bits | Use |
|------|------|-----|
| `f16` | 16 | Unaligned tensors, biases |
| `q8_0` | 8 | Small tensors, embeddings |
| `q6_k` | 6 | High-quality layers |
| `q5_k_m` | 5 | Balanced |
| `q4_k_m` | 4 | Default bulk |
| `q3_k_m` | 3 | Aggressive compression |
| `q2_k` | 2 | Maximum compression |
| `iq4_xs` | 4 | Improved 4-bit |

## Special Token Keys

Default mappings in export config:

| Token Type | GGUF Key |
|------------|----------|
| `bos` | `tokenizer.ggml.bos_token_id` |
| `eos` | `tokenizer.ggml.eos_token_id` |
| `unk` | `tokenizer.ggml.unknown_token_id` |
| `pad` | `tokenizer.ggml.padding_token_id` |
| `sep` | `tokenizer.ggml.seperator_token_id` |
| `cls` | `tokenizer.ggml.cls_token_id` |
| `mask` | `tokenizer.ggml.mask_token_id` |
| `eot` | `tokenizer.ggml.eot_token_id` |

Override with `special_token_keys` in export_config.json.

## Supported Models

Llama, Qwen, DeepSeek, Mixtral, Mistral, Gemma, Phi, MiniMax-M2. Vision: LLaVA, Qwen-VL, Kimi-VL. Encoder-decoder: T5, BART, Whisper.

## Workflow

1. `gmat import --generate-config` → creates import_config.json
2. `gmat import --config import_config.json` → creates GMAT storage + tokens.bin
3. `gmat export --generate-config` → creates export_config.json with quant recommendations
4. `gmat export --config export_config.json` → creates GGUF
