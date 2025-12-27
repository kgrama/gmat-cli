# Tiny LLM Example

This example demonstrates GMAT's core workflow: **one source model, multiple deployment profiles**.

## Directory Structure

```
tiny_llm/
├── hf-model/                   # Source: HuggingFace model
│   ├── tiny_llm.safetensors
│   ├── config.json             # Model config (auto-read for metadata)
│   ├── tokenizer.json          # Tokenizer vocabulary
│   └── tokenizer_config.json   # Special token definitions
│
├── import_config.json          # Import settings: tensor mappings, block format
│
├── model.gmat/                 # GMAT format: tensor-addressed storage
│   ├── metadata.json           # Model metadata + tensor UUID mappings
│   ├── tokens.bin              # Tokenizer vocabulary (binary format)
│   └── tensors/                # Individual tensor files (UUID-named)
│       ├── 3b6d6f4a-....gmat
│       └── ...
│
└── export/                     # Export profiles and outputs
    ├── export-balanced.json    # Profile: Q4 default, critical tensors Q6-Q8
    └── gguf/                   # Output: production-ready GGUF files
        └── tiny_llm-balanced.gguf
```

## The Workflow

### Step 1: Import to GMAT

Convert SafeTensors to tensor-addressed GMAT format:

```bash
cd example/tiny_llm

# Generate config (auto-populates metadata from hf-model/config.json)
gmat import --model hf-model/tiny_llm.safetensors --generate-config

# Import
gmat import --model hf-model/tiny_llm.safetensors --config import_config.json -o .
```

Result: `model.gmat/` with individual tensor files.

### Step 2: Create Export Profiles

Generate an export config with automatic importance analysis:

```bash
gmat export --model model.gmat --generate-config
mv export_config.json export/export-balanced.json
```

### Step 3: Export to GGUF

Generate deployment artifacts with tokenizer metadata embedded:

```bash
gmat export --model model.gmat --config export/export-balanced.json -o export/gguf/tiny_llm-balanced.gguf
```

The export process:
- Loads tensors from GMAT storage
- Loads tokenizer from `tokens.bin`
- Quantizes tensors according to config
- Writes GGUF with tokenizer metadata (tokens, scores, special token IDs)

## Key Concepts Illustrated

### 1. Tensor-Addressed Storage

Each tensor has a stable UUID. The `model.gmat/tensors/` directory contains:
```
3b6d6f4a-9cf5-4d97-9b08-ce07ddc0435d.gmat  → output.weight (lm_head)
e268e395-888f-49ed-9f59-24ea9d10be77.gmat  → token_embd.weight
...
```

These UUIDs are referenced in export configs, enabling:
- Consistent tensor identification across exports
- Per-tensor quantization overrides
- Future: deduplication across fine-tunes

### 2. Quantization as Configuration

Export configs are small JSON files that define:
- Default quantization type
- Per-tensor overrides (critical tensors get higher precision)
- Scale optimization settings

**Version control these configs** - they capture your quantization decisions.

### 3. Per-Tensor Precision

Export config allows per-tensor quantization overrides:

```json
{
  "quantization": {
    "default_type": "q4_k_m",
    "per_tensor": {
      "<embed-uuid>": "q8_0",  // embeddings: higher precision
      "<output-uuid>": "q8_0"  // output: higher precision
    }
  }
}
```

The `--generate-config` command analyzes tensor importance and automatically assigns quantization types.

### 4. Metadata Auto-Population

The `hf-model/config.json` file (HuggingFace format) is automatically read during import:
```json
{
  "model_type": "llama",
  "vocab_size": 32000,
  "hidden_size": 192,
  ...
}
```

This metadata flows into `import_config.json` and `model.gmat/metadata.json`.

### 5. Tokenizer Flow

Tokenizer files are automatically parsed during import:
- `tokenizer_config.json` → special token mappings (bos, eos, unk, pad)
- `tokenizer.json` → vocabulary and tokenizer type (BPE, Unigram, etc.)

The tokenizer data is:
1. Stored as `model.gmat/tokens.bin` during import
2. Loaded and embedded into GGUF metadata during export
3. Configurable via `special_token_keys` in export config for custom mappings

## Try It

```bash
cd example/tiny_llm

# Clean start
rm -rf model.gmat export/gguf/*.gguf import_config.json

# Full workflow
gmat import --model hf-model/tiny_llm.safetensors --generate-config
gmat import --model hf-model/tiny_llm.safetensors --config import_config.json -o .

# Generate export config
gmat export --model model.gmat --generate-config
mv export_config.json export/export-balanced.json

# Export to GGUF
gmat export --model model.gmat --config export/export-balanced.json -o export/gguf/tiny_llm-balanced.gguf

# Check output
ls -la export/gguf/*.gguf
```
