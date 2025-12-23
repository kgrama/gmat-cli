//! GMAT CLI - Import and export GMAT model files.

use clap::{Parser, Subcommand};

mod common;
mod config;
mod export;
mod import;

#[derive(Parser)]
#[command(name = "gmat")]
#[command(about = "CLI tool for importing and exporting GMAT model files")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Import a SafeTensors/GGUF model to GMAT format
    Import {
        /// Path to the model file or folder
        #[arg(short, long)]
        model: String,

        /// Path to import config JSON (optional)
        #[arg(short, long)]
        config: Option<String>,

        /// Output path for GMAT model
        #[arg(short, long)]
        output: Option<String>,

        /// Generate a template config instead of importing
        #[arg(long)]
        generate_config: bool,
    },

    /// Export a GMAT model to GGUF format
    Export {
        /// Path to the GMAT model
        #[arg(short, long)]
        model: String,

        /// Path to export config JSON (optional)
        #[arg(short, long)]
        config: Option<String>,

        /// Output path for exported model
        #[arg(short, long)]
        output: Option<String>,

        /// Shard size in MB (e.g., 5000 for 5GB). Overrides config file.
        #[arg(long)]
        shard_size: Option<u64>,

        /// Generate a template config instead of exporting
        #[arg(long)]
        generate_config: bool,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Import {
            model,
            config,
            output,
            generate_config,
        } => {
            if generate_config {
                import::generate_config_template(&model)?;
            } else {
                import::run(&model, config.as_deref(), output.as_deref())?;
            }
        }
        Commands::Export {
            model,
            config,
            output,
            shard_size,
            generate_config,
        } => {
            if generate_config {
                export::generate_config_template(&model)?;
            } else {
                // Convert MB to bytes if provided
                let shard_bytes = shard_size.map(|mb| mb * 1_000_000);
                export::run(&model, config.as_deref(), output.as_deref(), shard_bytes)?;
            }
        }
    }

    Ok(())
}
