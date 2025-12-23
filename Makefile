# Makefile for gmat-cli
# A CLI tool for importing and exporting GMAT model files

# Configuration
CARGO := cargo
BINARY_NAME := gmat
PROJECT_DIR := gmat-cli
TARGET_DIR := $(PROJECT_DIR)/target
INSTALL_DIR ?= $(HOME)/.cargo/bin

# Default target
.PHONY: all
all: build

# Build in debug mode (faster compilation)
.PHONY: build
build:
	$(CARGO) build --manifest-path $(PROJECT_DIR)/Cargo.toml

# Build in release mode (optimized)
.PHONY: release
release:
	$(CARGO) build --manifest-path $(PROJECT_DIR)/Cargo.toml --release

# Run the CLI (debug build)
.PHONY: run
run:
	$(CARGO) run --manifest-path $(PROJECT_DIR)/Cargo.toml -- $(ARGS)

# Run tests for all workspace members
.PHONY: test
test:
	$(CARGO) test --manifest-path $(PROJECT_DIR)/Cargo.toml
	$(CARGO) test --manifest-path transform-storage/Cargo.toml

# Run clippy lints
.PHONY: lint
lint:
	$(CARGO) clippy --manifest-path $(PROJECT_DIR)/Cargo.toml -- -D warnings
	$(CARGO) clippy --manifest-path transform-storage/Cargo.toml -- -D warnings

# Format code
.PHONY: fmt
fmt:
	$(CARGO) fmt --manifest-path $(PROJECT_DIR)/Cargo.toml
	$(CARGO) fmt --manifest-path transform-storage/Cargo.toml

# Check formatting without making changes
.PHONY: fmt-check
fmt-check:
	$(CARGO) fmt --manifest-path $(PROJECT_DIR)/Cargo.toml -- --check
	$(CARGO) fmt --manifest-path transform-storage/Cargo.toml -- --check

# Clean build artifacts
.PHONY: clean
clean:
	$(CARGO) clean --manifest-path $(PROJECT_DIR)/Cargo.toml
	$(CARGO) clean --manifest-path transform-storage/Cargo.toml

# Install the binary (default: ~/.cargo/bin, override with INSTALL_DIR)
.PHONY: install
install: release
	@mkdir -p $(INSTALL_DIR)
	cp $(TARGET_DIR)/release/$(BINARY_NAME) $(INSTALL_DIR)/

# Uninstall the binary
.PHONY: uninstall
uninstall:
	rm -f $(INSTALL_DIR)/$(BINARY_NAME)

# Check compilation without producing binaries
.PHONY: check
check:
	$(CARGO) check --manifest-path $(PROJECT_DIR)/Cargo.toml

# Build documentation
.PHONY: doc
doc:
	$(CARGO) doc --manifest-path $(PROJECT_DIR)/Cargo.toml --no-deps --open

# Show help
.PHONY: help
help:
	@echo "gmat-cli Makefile targets:"
	@echo ""
	@echo "  build     - Build debug binary (default)"
	@echo "  release   - Build optimized release binary"
	@echo "  run       - Run the CLI (use ARGS=\"...\" for arguments)"
	@echo "  test      - Run all tests"
	@echo "  lint      - Run clippy lints"
	@echo "  fmt       - Format code"
	@echo "  fmt-check - Check code formatting"
	@echo "  clean     - Remove build artifacts"
	@echo "  install   - Install binary (use INSTALL_DIR=... to override)"
	@echo "  uninstall - Remove installed binary"
	@echo "  check     - Check compilation without building"
	@echo "  doc       - Build and open documentation"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make                              # Build debug binary"
	@echo "  make release                      # Build release binary"
	@echo "  make run ARGS=\"--help\"            # Run with arguments"
	@echo "  make install                      # Install to ~/.cargo/bin"
	@echo "  make install INSTALL_DIR=/usr/local/bin  # Install to custom dir"
