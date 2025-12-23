# GMAT CLI

A command-line tool for importing and exporting GMAT (Graph Matrix) model files. GMAT is a storage format designed for LLM tensor weights that supports efficient sparse storage and various block formats.

## Overview

GMAT CLI provides two main operations:

- **Import**: Convert SafeTensors models to GMAT format
- **Export**: Convert GMAT models to GGUF format with quantization support

## Features

- Streaming tensor processing for minimal memory usage
- Parallel processing using Rayon for fast conversions
- Multiple block format options for storage optimization
- Comprehensive quantization support (Q4, Q5, Q6, Q8 variants)
- Per-tensor quantization configuration
- Sharded output for large models
- Automatic config generation from model files

## Quick Start

```bash
# Install
make install

# Generate import config from a SafeTensors model
gmat import --model ./model.safetensors --generate-config

# Import to GMAT format
gmat import --model ./model.safetensors --config import_config.json --output ./output

# Generate export config from a GMAT model
gmat export --model ./output/model.gmat --generate-config

# Export to GGUF format
gmat export --model ./output/model.gmat --config export_config.json --output model.gguf
```

## Documentation

- [[Installation]]
- [[Import-Command]]
- [[Export-Command]]
- [[Configuration-Files]]

## Example Workflow

See the `example/tiny_llm` directory for a complete example with:
- Sample SafeTensors model (`tiny_llm.safetensors`)
- Import configuration (`import_config.json`)
- Converted GMAT model (`model.gmat/`)
- Export configuration (`export_config.json`)
- Exported GGUF model (`tiny_llm.gguf`)
