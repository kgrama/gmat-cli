//! Async binary IO for TokenIdTree persistence.
//!
//! Uses a compact binary format for fast loading of large vocabularies.
//! Format: [magic:4][version:1][header_len:4][header_json][tree_bincode]

use anyhow::{Context, Result};
use std::path::Path;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};

use super::TokenIdTree;

/// Magic bytes for token tree files.
const MAGIC: &[u8; 4] = b"GTTK";
/// Current format version.
const VERSION: u8 = 1;

/// Header stored as JSON for human inspection.
#[derive(serde::Serialize, serde::Deserialize)]
struct Header {
    version: u8,
    vocab_size: usize,
    source_format: String,
    tokenizer_type: String,
}

/// Save a TokenIdTree to a binary file asynchronously.
///
/// File is saved to `{path}/tokens.bin` within the model directory.
pub async fn save_token_tree(tree: &TokenIdTree, model_dir: &Path) -> Result<()> {
    let path = model_dir.join("tokens.bin");
    let file = File::create(&path)
        .await
        .with_context(|| format!("Failed to create tokens.bin: {}", path.display()))?;
    let mut writer = BufWriter::new(file);

    // Write magic and version
    writer.write_all(MAGIC).await?;
    writer.write_all(&[VERSION]).await?;

    // Prepare header
    let header = Header {
        version: VERSION,
        vocab_size: tree.vocab_size(),
        source_format: format!("{:?}", tree.source_format),
        tokenizer_type: format!("{:?}", tree.tokenizer_type),
    };
    let header_json = serde_json::to_vec(&header)?;

    // Write header length and header
    writer
        .write_all(&(header_json.len() as u32).to_le_bytes())
        .await?;
    writer.write_all(&header_json).await?;

    // Serialize tree with bincode (sync, but fast)
    let tree_bytes =
        bincode::serialize(tree).context("Failed to serialize token tree with bincode")?;
    writer.write_all(&tree_bytes).await?;

    writer.flush().await?;
    Ok(())
}

/// Load a TokenIdTree from a binary file asynchronously.
///
/// Loads from `{path}/tokens.bin` within the model directory.
pub async fn load_token_tree(model_dir: &Path) -> Result<TokenIdTree> {
    let path = model_dir.join("tokens.bin");
    let file = File::open(&path)
        .await
        .with_context(|| format!("Failed to open tokens.bin: {}", path.display()))?;
    let mut reader = BufReader::new(file);

    // Read and verify magic
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).await?;
    if &magic != MAGIC {
        anyhow::bail!(
            "Invalid token tree file: expected magic {:?}, got {:?}",
            MAGIC,
            magic
        );
    }

    // Read and verify version
    let mut version = [0u8; 1];
    reader.read_exact(&mut version).await?;
    if version[0] != VERSION {
        anyhow::bail!(
            "Unsupported token tree version: expected {}, got {}",
            VERSION,
            version[0]
        );
    }

    // Read header length and header
    let mut header_len_bytes = [0u8; 4];
    reader.read_exact(&mut header_len_bytes).await?;
    let header_len = u32::from_le_bytes(header_len_bytes) as usize;

    let mut header_bytes = vec![0u8; header_len];
    reader.read_exact(&mut header_bytes).await?;
    let _header: Header =
        serde_json::from_slice(&header_bytes).context("Failed to parse token tree header")?;

    // Read remaining bytes for tree
    let mut tree_bytes = Vec::new();
    reader.read_to_end(&mut tree_bytes).await?;

    let tree: TokenIdTree =
        bincode::deserialize(&tree_bytes).context("Failed to deserialize token tree")?;

    Ok(tree)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::{SourceFormat, TokenEntry, TokenizerType};
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_save_and_load_token_tree() {
        let dir = TempDir::new().unwrap();

        // Build a simple tree using right field directly
        let world = TokenEntry::new(2u32, "world".to_string(), 2);
        let mut hello = TokenEntry::new(1u32, "hello".to_string(), 1);
        hello.right = Some(Box::new(world));
        let mut bos = TokenEntry::special(0u32, "<s>".to_string(), 0, "bos");
        bos.right = Some(Box::new(hello));

        let tree = TokenIdTree::new(SourceFormat::HuggingFace, TokenizerType::Bpe, bos);

        // Save
        save_token_tree(&tree, dir.path()).await.unwrap();

        // Verify file exists
        assert!(dir.path().join("tokens.bin").exists());

        // Load
        let loaded = load_token_tree(dir.path()).await.unwrap();

        assert_eq!(loaded.vocab_size(), 3);
        assert_eq!(loaded.source_format, SourceFormat::HuggingFace);
        assert_eq!(loaded.tokenizer_type, TokenizerType::Bpe);
    }

    #[tokio::test]
    async fn test_invalid_magic() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("tokens.bin");

        // Write invalid file
        tokio::fs::write(&path, b"XXXX").await.unwrap();

        let result = load_token_tree(dir.path()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("magic"));
    }
}
