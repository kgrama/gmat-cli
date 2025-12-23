# Import Command

The `import` command converts SafeTensors models to GMAT format.

## Usage

```bash
gmat import [OPTIONS]
```

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--model <PATH>` | `-m` | Path to the model file or folder (required) |
| `--config <PATH>` | `-c` | Path to import config JSON (optional) |
| `--output <PATH>` | `-o` | Output path for GMAT model (default: `output/`) |
| `--generate-config` | | Generate a template config instead of importing |

## Workflow

### Step 1: Generate Configuration

First, generate a template configuration from your model:

```bash
gmat import --model ./tiny_llm.safetensors --generate-config
```

This scans the model and creates `import_config.json` with:
- All tensor names discovered in the model
- UUID mappings for each tensor
- Default block format (`B8x8`)
- Auto-populated model metadata (from `config.json` or safetensor header)

### Step 2: Edit Configuration (Optional)

Review and customize `import_config.json` if needed:
- Set `include: false` for tensors you want to skip
- Verify auto-populated model metadata
- Choose a different block format if needed

## Metadata Auto-Population

The `--generate-config` command automatically extracts model metadata from:

1. **config.json** (HuggingFace format) - Looks for `config.json` in the same directory as the model
2. **SafeTensor header** - Falls back to `__metadata__` embedded in the safetensor file

Supported metadata fields:
- `architecture` / `model_type`
- `vocab_size`
- `hidden_size`
- `num_layers` / `num_hidden_layers`
- `num_attention_heads`
- `intermediate_size`
- `max_position_embeddings`

### Step 3: Run Import

```bash
gmat import --model ./tiny_llm.safetensors --config import_config.json --output ./output
```

## Output Structure

The import creates a `.gmat` directory structure:

```
output/model.gmat/
├── metadata.json       # Model metadata and tensor mappings
└── tensors/
    ├── uuid1.gmat      # Individual tensor files
    ├── uuid2.gmat
    └── ...
```

## Block Formats

Choose a block format based on your use case:

| Format | Description |
|--------|-------------|
| `B8x4` | 8 rows x 4 columns per block |
| `B8x8` | 8 rows x 8 columns per block (default) |
| `B16x4` | 16 rows x 4 columns per block |
| `B16x8` | 16 rows x 8 columns per block |
| `DualRow8x4` | Dual-row variant, 8x4 |
| `DualRow8x8` | Dual-row variant, 8x8 |
| `DualRow16x4` | Dual-row variant, 16x4 |
| `DualRow16x8` | Dual-row variant, 16x8 |

## Example

Using the example from `example/tiny_llm/`:

```bash
# Navigate to example directory
cd example/tiny_llm

# Import the tiny LLM model
gmat import \
  --model tiny_llm.safetensors \
  --config import_config.json \
  --output .
```

Output:

```
Output: ./model.gmat
Block format: B8x8
Found 1 safetensors file(s)
Processing 12 tensors...
[1/12] model.embed_tokens.weight [32000x192] -> e268e395-888f-49ed-9f59-24ea9d10be77.gmat (nnz=6144000, sparsity=0.0%)
[2/12] model.layers.0.self_attn.q_proj.weight [192x192] -> df11f72a-85ce-41e1-b354-1253749bd43d.gmat (nnz=36864, sparsity=0.0%)
...
=== Import Complete: 12 tensors ===
```

## Memory Efficiency

The import process uses streaming to minimize memory usage:
- Tensors are processed one at a time through a producer-consumer pattern
- Only a bounded buffer of tensors (2 x CPU cores) are held in memory
- Each tensor is immediately written to disk after conversion

## Supported Source Formats

- **SafeTensors** (`.safetensors`) - Primary format, fully supported
- Multi-file sharded SafeTensors models (automatically discovered)

See also: [[Configuration-Files#import-configuration]]
