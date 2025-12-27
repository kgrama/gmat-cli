# Model Configuration Files

Model-specific configuration for GMAT export. Each file defines tensor name mappings, quantization recommendations, metadata key aliases, and special token mappings.

## Usage

Pass the model config file to `--model-config` when generating an export config:

```bash
gmat export --model /path/to/gmat-model --generate-config --model-config export-overrides/models/llama.json
```

The generated `export_config.json` will contain pre-resolved values from the model config.

## File Structure

Each JSON file can contain:

| Field | Description |
|-------|-------------|
| `gguf_architecture` | GGUF architecture name (e.g., "llama", "phi3", "gemma2") |
| `special_token_keys` | Maps token types ("bos", "eos", etc.) to GGUF metadata keys |
| `metadata_key_aliases` | Maps canonical keys to HuggingFace config alternatives |
| `quant_recommendations` | Quantization type recommendations by tensor name pattern |
| `tensor_map_patterns` | SafeTensors to GGUF tensor name mappings with `{N}` and `{E}` placeholders |

## Files

| File | Models | Notes |
|------|--------|-------|
| `llama.json` | Llama, Llama 2/3, Mistral, Mixtral | Base patterns for most models |
| `qwen2.json` | Qwen, Qwen2, Qwen2-VL | Standard Llama-style naming |
| `deepseek.json` | DeepSeek V2/V3 | MoE with shared experts |
| `gemma.json` | Gemma 2 | Extra pre/post feedforward layernorms |
| `phi.json` | Phi-3 | Fused gate_up_proj, fused qkv_proj |
| `glm.json` | GLM-4, ChatGLM | Fused gate_up_proj, attention biases |
| `kimi.json` | Kimi-VL | VLM prefix, MLA attention, MoE experts |

## Pattern Syntax

Tensor name patterns use `{N}` for layer numbers and `{E}` for expert indices:

```json
{
  "tensor_map_patterns": {
    "model.layers.{N}.self_attn.q_proj.weight": "blk.{N}.attn_q.weight",
    "model.layers.{N}.mlp.experts.{E}.gate_proj.weight": "blk.{N}.ffn_gate.{E}.weight"
  }
}
```
