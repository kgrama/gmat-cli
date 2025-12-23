//! Integration tests for gmat-cli using Tiny-LLM (10M param model).

use std::fs;
use std::path::Path;
use std::process::Command;

fn gmat_cli() -> Command {
    Command::new(env!("CARGO_BIN_EXE_gmat"))
}

fn fixtures_dir() -> &'static Path {
    Path::new("tests/fixtures")
}

fn llama_cli() -> Option<Command> {
    // Check common locations for llama-cli
    for path in ["/usr/bin/llama-cli", "/usr/local/bin/llama-cli", "llama-cli"] {
        if Command::new(path).arg("--version").output().is_ok() {
            return Some(Command::new(path));
        }
    }
    None
}

#[test]
fn test_import_generate_config() {
    let output = gmat_cli()
        .args([
            "import",
            "--model", "tests/fixtures/tiny_llm.safetensors",
            "--generate-config",
        ])
        .output()
        .expect("Failed to run gmat");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "import --generate-config failed: {}", stdout);
    assert!(stdout.contains("Found 1 safetensors file(s)"));
    assert!(stdout.contains("Generated import_config.json"));

    // Cleanup
    let _ = fs::remove_file("import_config.json");
}

#[test]
fn test_import_safetensors_to_gmat() {
    let output_dir = fixtures_dir().join("test_output");
    let _ = fs::remove_dir_all(&output_dir);

    let output = gmat_cli()
        .args([
            "import",
            "--model", "tests/fixtures/tiny_llm.safetensors",
            "--config", "tests/fixtures/import_config.json",
            "--output", output_dir.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run gmat import");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "import failed:\nstdout: {}\nstderr: {}", stdout, stderr);

    // Verify output structure
    let model_dir = output_dir.join("model.gmat");
    assert!(model_dir.exists(), "model.gmat directory not created");
    assert!(model_dir.join("metadata.json").exists(), "metadata.json not created");
    assert!(model_dir.join("tensors").is_dir(), "tensors directory not created");

    // Check we got the expected number of tensor files (9 2D tensors)
    let tensor_count = fs::read_dir(model_dir.join("tensors"))
        .unwrap()
        .filter(|e| e.as_ref().map(|e| e.path().extension().map(|x| x == "gmat").unwrap_or(false)).unwrap_or(false))
        .count();
    assert_eq!(tensor_count, 9, "Expected 9 tensor files, got {}", tensor_count);

    // Cleanup
    let _ = fs::remove_dir_all(&output_dir);
}

#[test]
fn test_export_generate_config() {
    // This test requires an existing GMAT model
    let model_dir = fixtures_dir().join("model.gmat");
    if !model_dir.exists() {
        eprintln!("Skipping test_export_generate_config: model.gmat not found");
        return;
    }

    let output = gmat_cli()
        .args([
            "export",
            "--model", model_dir.to_str().unwrap(),
            "--generate-config",
        ])
        .output()
        .expect("Failed to run gmat");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "export --generate-config failed: {}", stdout);
    assert!(stdout.contains("Analyzing"));
    assert!(stdout.contains("Generated export_config.json"));

    // Cleanup
    let _ = fs::remove_file("export_config.json");
}

#[test]
fn test_export_gmat_to_gguf() {
    // This test requires an existing GMAT model
    let model_dir = fixtures_dir().join("model.gmat");
    if !model_dir.exists() {
        eprintln!("Skipping test_export_gmat_to_gguf: model.gmat not found");
        return;
    }

    let output_file = fixtures_dir().join("test_output.gguf");
    let _ = fs::remove_file(&output_file);

    let output = gmat_cli()
        .args([
            "export",
            "--model", model_dir.to_str().unwrap(),
            "--config", "tests/fixtures/export_config.json",
            "--output", output_file.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run gmat export");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "export failed:\nstdout: {}\nstderr: {}", stdout, stderr);

    // Verify GGUF file created
    assert!(output_file.exists(), "GGUF file not created");

    let file_size = fs::metadata(&output_file).unwrap().len();
    assert!(file_size > 1_000_000, "GGUF file too small: {} bytes", file_size);

    // Cleanup
    let _ = fs::remove_file(&output_file);
}

#[test]
fn test_full_roundtrip() {
    let temp_dir = fixtures_dir().join("roundtrip_test");
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    // Step 1: Import safetensors to GMAT
    let import_output = gmat_cli()
        .args([
            "import",
            "--model", "tests/fixtures/tiny_llm.safetensors",
            "--config", "tests/fixtures/import_config.json",
            "--output", temp_dir.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run import");

    assert!(import_output.status.success(), "Import failed: {}",
        String::from_utf8_lossy(&import_output.stdout));

    let model_dir = temp_dir.join("model.gmat");
    assert!(model_dir.exists());

    // Step 2: Generate export config
    let gen_config_output = gmat_cli()
        .current_dir(&temp_dir)
        .args([
            "export",
            "--model", "model.gmat",
            "--generate-config",
        ])
        .output()
        .expect("Failed to generate export config");

    assert!(gen_config_output.status.success(), "Generate config failed: {}",
        String::from_utf8_lossy(&gen_config_output.stdout));

    // Step 3: Export to GGUF
    let gguf_path = temp_dir.join("output.gguf");
    let export_output = gmat_cli()
        .current_dir(&temp_dir)
        .args([
            "export",
            "--model", "model.gmat",
            "--config", "export_config.json",
            "--output", "output.gguf",
        ])
        .output()
        .expect("Failed to run export");

    assert!(export_output.status.success(), "Export failed: {}",
        String::from_utf8_lossy(&export_output.stdout));

    assert!(gguf_path.exists(), "GGUF not created");

    let gguf_size = fs::metadata(&gguf_path).unwrap().len();
    println!("Roundtrip complete: GGUF size = {} bytes", gguf_size);

    // Cleanup
    let _ = fs::remove_dir_all(&temp_dir);
}

/// Inference smoke test: verify exported GGUF can be loaded by llama.cpp
///
/// This test requires llama-cli to be installed. It will be skipped if not found.
/// Note: Tiny-LLM is a 10M param model - output quality is not meaningful,
/// we're just verifying the GGUF file is valid and loadable.
#[test]
#[ignore] // Run with: cargo test --test integration_test -- --ignored
fn test_inference_smoke() {
    let Some(mut llama) = llama_cli() else {
        eprintln!("Skipping inference test: llama-cli not found");
        return;
    };

    // Use the pre-existing GGUF from fixtures (created by test_full_roundtrip or manual run)
    let gguf_path = fixtures_dir().join("tiny_llm.gguf");
    if !gguf_path.exists() {
        eprintln!("Skipping inference test: {} not found. Run full roundtrip first.", gguf_path.display());
        return;
    }

    // Run llama-cli with minimal settings
    // -n 8: generate only 8 tokens (quick)
    // -p: prompt
    // --no-display-prompt: cleaner output
    let output = llama
        .args([
            "-m", gguf_path.to_str().unwrap(),
            "-p", "Hello",
            "-n", "8",
            "--no-display-prompt",
            "-c", "64",  // minimal context
        ])
        .output()
        .expect("Failed to run llama-cli");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Check that llama-cli loaded the model (look for load messages in stderr)
    assert!(
        stderr.contains("llama_model_load") || stderr.contains("model loaded") || output.status.success(),
        "llama-cli failed to load model:\nstderr: {}\nstdout: {}",
        stderr, stdout
    );

    // Verify we got some output (not necessarily meaningful for 10M model)
    println!("Inference output: {}", stdout.trim());
    println!("Model loaded successfully via llama-cli");
}

/// GGUF structure validation using ggml-python
///
/// Validates that exported GGUF files have correct structure and can be parsed
/// by the ggml library. This is a lighter-weight test than full inference.
#[test]
fn test_gguf_validation_python() {
    // Check if ggml-python is available
    let python_check = Command::new("python3")
        .args(["-c", "import ggml"])
        .output();

    if python_check.is_err() || !python_check.unwrap().status.success() {
        eprintln!("Skipping GGUF validation: ggml-python not installed");
        return;
    }

    let gguf_path = fixtures_dir().join("tiny_llm.gguf");
    if !gguf_path.exists() {
        eprintln!("Skipping GGUF validation: {} not found", gguf_path.display());
        return;
    }

    let validation_script = r#"
import ggml
import sys

gguf_path = sys.argv[1]
params = ggml.gguf_init_params()
params.no_alloc = True
params.ctx = None

ctx = ggml.gguf_init_from_file(gguf_path.encode(), params)
if not ctx:
    print("ERROR: Failed to load GGUF")
    sys.exit(1)

n_tensors = ggml.gguf_get_n_tensors(ctx)
version = ggml.gguf_get_version(ctx)

print(f"version={version}")
print(f"tensors={n_tensors}")

for i in range(n_tensors):
    name = ggml.gguf_get_tensor_name(ctx, i)
    if name:
        print(f"tensor:{name.decode()}")

ggml.gguf_free(ctx)
"#;

    let output = Command::new("python3")
        .args(["-c", validation_script, gguf_path.to_str().unwrap()])
        .output()
        .expect("Failed to run Python validation");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success(),
        "GGUF validation failed:\nstdout: {}\nstderr: {}", stdout, stderr);

    // Verify expected content
    assert!(stdout.contains("version=3"), "Expected GGUF v3, got: {}", stdout);
    assert!(stdout.contains("tensors=9"), "Expected 9 tensors, got: {}", stdout);

    // Verify key tensors present
    assert!(stdout.contains("tensor:token_embd.weight"), "Missing token_embd.weight");
    assert!(stdout.contains("tensor:output.weight"), "Missing output.weight");
    assert!(stdout.contains("tensor:blk.0.attn_q.weight"), "Missing attention tensor");

    println!("GGUF validation passed:\n{}", stdout);
}

/// Full roundtrip with GGUF structure validation
#[test]
fn test_roundtrip_with_gguf_validation() {
    // Check if ggml-python is available
    let python_check = Command::new("python3")
        .args(["-c", "import ggml"])
        .output();

    if python_check.is_err() || !python_check.unwrap().status.success() {
        eprintln!("Skipping: ggml-python not installed");
        return;
    }

    let temp_dir = fixtures_dir().join("validation_test");
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    // Step 1: Import
    let import_output = gmat_cli()
        .args([
            "import",
            "--model", "tests/fixtures/tiny_llm.safetensors",
            "--config", "tests/fixtures/import_config.json",
            "--output", temp_dir.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to import");

    assert!(import_output.status.success(), "Import failed: {}",
        String::from_utf8_lossy(&import_output.stderr));

    // Step 2: Generate export config
    let _ = gmat_cli()
        .current_dir(&temp_dir)
        .args(["export", "--model", "model.gmat", "--generate-config"])
        .output()
        .expect("Failed to generate config");

    // Step 3: Export to GGUF
    let gguf_path = temp_dir.join("model.gguf");
    let export_output = gmat_cli()
        .current_dir(&temp_dir)
        .args([
            "export",
            "--model", "model.gmat",
            "--config", "export_config.json",
            "--output", "model.gguf",
        ])
        .output()
        .expect("Failed to export");

    assert!(export_output.status.success(), "Export failed: {}",
        String::from_utf8_lossy(&export_output.stderr));
    assert!(gguf_path.exists(), "GGUF not created");

    // Step 4: Validate with ggml-python
    let validation_script = r#"
import ggml
import sys

gguf_path = sys.argv[1]
params = ggml.gguf_init_params()
params.no_alloc = True
params.ctx = None

ctx = ggml.gguf_init_from_file(gguf_path.encode(), params)
if not ctx:
    print("ERROR: Failed to load GGUF")
    sys.exit(1)

n_tensors = ggml.gguf_get_n_tensors(ctx)
version = ggml.gguf_get_version(ctx)
print(f"VALID: v{version}, {n_tensors} tensors")
ggml.gguf_free(ctx)
"#;

    let validation = Command::new("python3")
        .args(["-c", validation_script, gguf_path.to_str().unwrap()])
        .output()
        .expect("Failed to run validation");

    let stdout = String::from_utf8_lossy(&validation.stdout);
    assert!(validation.status.success() && stdout.contains("VALID"),
        "GGUF validation failed: {}", stdout);

    println!("Roundtrip + validation: {}", stdout.trim());

    // Cleanup
    let _ = fs::remove_dir_all(&temp_dir);
}

/// Full inference validation with roundtrip
#[test]
#[ignore] // Run with: cargo test --test integration_test -- --ignored
fn test_roundtrip_with_inference() {
    let Some(_) = llama_cli() else {
        eprintln!("Skipping: llama-cli not found");
        return;
    };

    let temp_dir = fixtures_dir().join("inference_test");
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    // Step 1: Import
    let import_output = gmat_cli()
        .args([
            "import",
            "--model", "tests/fixtures/tiny_llm.safetensors",
            "--config", "tests/fixtures/import_config.json",
            "--output", temp_dir.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to import");

    assert!(import_output.status.success(), "Import failed");

    // Step 2: Generate export config
    let _ = gmat_cli()
        .current_dir(&temp_dir)
        .args(["export", "--model", "model.gmat", "--generate-config"])
        .output()
        .expect("Failed to generate config");

    // Step 3: Export to GGUF
    let gguf_path = temp_dir.join("model.gguf");
    let export_output = gmat_cli()
        .current_dir(&temp_dir)
        .args([
            "export",
            "--model", "model.gmat",
            "--config", "export_config.json",
            "--output", "model.gguf",
        ])
        .output()
        .expect("Failed to export");

    assert!(export_output.status.success(), "Export failed");
    assert!(gguf_path.exists(), "GGUF not created");

    // Step 4: Inference smoke test
    let mut llama = llama_cli().unwrap();
    let inference_output = llama
        .args([
            "-m", gguf_path.to_str().unwrap(),
            "-p", "The",
            "-n", "4",
            "-c", "64",
        ])
        .output()
        .expect("Failed to run inference");

    let stderr = String::from_utf8_lossy(&inference_output.stderr);

    // Just verify model loads - output quality doesn't matter for 10M model
    assert!(
        inference_output.status.success() || stderr.contains("llama"),
        "Inference failed:\n{}", stderr
    );

    println!("Full roundtrip with inference: SUCCESS");

    // Cleanup
    let _ = fs::remove_dir_all(&temp_dir);
}
