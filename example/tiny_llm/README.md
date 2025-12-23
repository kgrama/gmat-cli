# Tiny LLM Example

This example demonstrates GMAT's core workflow: **one source model, multiple deployment profiles**.

## Directory Structure

```
tiny_llm/
├── hf-model/                   # Source: HuggingFace model
│   ├── tiny_llm.safetensors
│   └── config.json             # Model config (auto-read for metadata)
│
├── import_config.json          # Import settings: tensor mappings, block format
│
├── model.gmat/                 # GMAT format: tensor-addressed storage
│   ├── metadata.json           # Model metadata + tensor UUID mappings
│   └── tensors/                # Individual tensor files (UUID-named)
│       ├── 3b6d6f4a-....gmat
│       └── ...
│
└── export/                     # Export profiles and outputs
    ├── export-economy.json     # Profile: aggressive Q4, smallest size
    ├── export-balanced.json    # Profile: Q4 default, critical tensors Q6-Q8
    ├── export-premium.json     # Profile: Q8 throughout, highest quality
    └── gguf/                   # Output: production-ready GGUF files
        ├── tiny_llm-economy.gguf
        ├── tiny_llm-balanced.gguf
        └── tiny_llm-premium.gguf
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

The same GMAT source supports multiple quantization profiles:

| Profile | Default | Embeddings | Use Case |
|---------|---------|------------|----------|
| `export-economy.json` | Q4_0 | Q4_K_M | Cost-sensitive, edge devices |
| `export-balanced.json` | Q4_K_M | Q6_K | Production default |
| `export-premium.json` | Q8_0 | Q8_0 | Quality-critical applications |

### Step 3: Export to GGUF

Generate deployment artifacts from your chosen profile:

```bash
# Economy tier
gmat export --model model.gmat --config export/export-economy.json -o export/gguf/tiny_llm-economy.gguf

# Balanced tier
gmat export --model model.gmat --config export/export-balanced.json -o export/gguf/tiny_llm-balanced.gguf

# Premium tier
gmat export --model model.gmat --config export/export-premium.json -o export/gguf/tiny_llm-premium.gguf
```

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

Compare the profiles:

```json
// export/export-economy.json - aggressive compression
"default_type": "q4_0",
"per_tensor": {
  "e268e395-...": "q4_k_m",  // embeddings: slightly better than default
  "3b6d6f4a-...": "q4_k_m"   // output: slightly better than default
}

// export/export-premium.json - maximum quality
"default_type": "q8_0",
"per_tensor": {
  "e268e395-...": "q8_0",    // embeddings: full precision
  "3b6d6f4a-...": "q8_0"     // output: full precision
}
```

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

## Try It

```bash
cd example/tiny_llm

# Clean start
rm -rf model.gmat export/gguf/*.gguf import_config.json

# Full workflow
gmat import --model hf-model/tiny_llm.safetensors --generate-config
gmat import --model hf-model/tiny_llm.safetensors --config import_config.json -o .

# Generate all three deployment tiers
gmat export --model model.gmat --config export/export-economy.json -o export/gguf/tiny_llm-economy.gguf
gmat export --model model.gmat --config export/export-balanced.json -o export/gguf/tiny_llm-balanced.gguf
gmat export --model model.gmat --config export/export-premium.json -o export/gguf/tiny_llm-premium.gguf

# Compare sizes
ls -la export/gguf/*.gguf
```
