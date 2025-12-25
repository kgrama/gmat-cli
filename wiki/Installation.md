# Installation

GMAT-CLI is a pure CPU tool—no GPU, CUDA, or special drivers required. It runs on any machine with a Rust toolchain.

## System Requirements

### Minimum

| Component | Requirement |
|-----------|-------------|
| **CPU** | 4 cores (x86_64 or ARM64) |
| **RAM** | 8 GB |
| **Storage** | 2× model size (source + GMAT output) |
| **OS** | Linux, macOS, Windows (WSL2) |

### Recommended (for 70B+ models)

| Component | Requirement |
|-----------|-------------|
| **CPU** | 8+ cores |
| **RAM** | 32 GB |
| **Storage** | NVMe SSD, 3× model size |
| **OS** | Linux (best I/O performance) |

### Notes

- **Memory usage is bounded**: GMAT uses streaming pipelines, so RAM requirements don't scale linearly with model size
- **CPU cores matter**: Parallel processing via Rayon scales with available cores
- **Storage I/O**: NVMe recommended for large models; HDD works but slower
- **No GPU required**: All operations run on CPU

## Prerequisites

- Rust toolchain (1.70 or later recommended)
- Cargo package manager
- No GPU required

## Building from Source

### Debug Build (faster compilation)

```bash
make build
```

### Release Build (optimized)

```bash
make release
```

## Installing the Binary

### Default Installation (to ~/.cargo/bin)

```bash
make install
```

### Custom Installation Directory

```bash
make install INSTALL_DIR=/usr/local/bin
```

## Verifying Installation

```bash
gmat --help
```

Expected output:

```
CLI tool for importing and exporting GMAT model files

Usage: gmat <COMMAND>

Commands:
  import  Import a SafeTensors/GGUF model to GMAT format
  export  Export a GMAT model to GGUF format
  help    Print this message or the help of the given subcommand(s)

Options:
  -h, --help  Print help
```

## Uninstalling

```bash
make uninstall
```

## Development Commands

| Command | Description |
|---------|-------------|
| `make build` | Build debug binary |
| `make release` | Build optimized release binary |
| `make test` | Run all tests |
| `make lint` | Run clippy lints |
| `make fmt` | Format code |
| `make fmt-check` | Check code formatting |
| `make clean` | Remove build artifacts |
| `make doc` | Build and open documentation |
| `make check` | Check compilation without building |

## Running Without Installing

```bash
make run ARGS="--help"
make run ARGS="import --model ./model.safetensors --generate-config"
```
