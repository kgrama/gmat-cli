//! Token vocabulary tree for format-agnostic tokenizer representation.
//!
//! This module provides an intermediate representation for tokenizer vocabularies
//! that can be converted between different formats (GGUF, ONNX, SafeTensors, etc.)
//! while maintaining the correct token-to-embedding-row mapping.

mod io;
mod special_config;

use serde::{Deserialize, Serialize};

pub use io::{load_token_tree, save_token_tree};
pub use special_config::SpecialTokenMapping;

/// Source format of the tokenizer vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceFormat {
    Gguf,
    Onnx,
    SafeTensors,
    HuggingFace,
    Tiktoken,
}

/// Tokenizer model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TokenizerType {
    Bpe,
    Unigram,
    WordPiece,
    WordLevel,
    Tiktoken,
}

/// A single token entry in the vocabulary tree.
///
/// For BPE tokenizers: `left` and `right` represent merge children.
/// e.g., "hello" = merge("hel", "lo") → left="hel", right="lo"
///
/// For flat tokenizers (WordPiece, WordLevel, Unigram):
/// `left` = None, `right` = next token in sequence (linked list).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEntry {
    /// Token identifier (can be string or numeric depending on format).
    pub id: TokenId,
    /// The actual token string.
    pub token: String,
    /// Row index in the embedding matrix.
    pub embedding_row: u32,
    /// Whether this is a special token (BOS, EOS, PAD, etc.).
    pub special: bool,
    /// The actual special token type (e.g., "bos", "eos", "pad").
    pub special_type: String,
    /// Score for Unigram models (log probability).
    pub score: Option<f32>,
    /// Left child (BPE: left merge component, flat: always None).
    pub left: Option<Box<TokenEntry>>,
    /// Right child (BPE: right merge component, flat: next token in list).
    pub right: Option<Box<TokenEntry>>,
}

/// Token identifier that can be either a string or numeric ID.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TokenId {
    String(String),
    Numeric(u32),
}

impl From<u32> for TokenId {
    fn from(id: u32) -> Self {
        TokenId::Numeric(id)
    }
}

impl From<String> for TokenId {
    fn from(id: String) -> Self {
        TokenId::String(id)
    }
}

impl From<&str> for TokenId {
    fn from(id: &str) -> Self {
        TokenId::String(id.to_string())
    }
}

impl TokenId {
    /// Get numeric value for ordering (String ids hash to u32).
    pub fn as_u32(&self) -> u32 {
        match self {
            TokenId::Numeric(n) => *n,
            TokenId::String(s) => {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                s.hash(&mut hasher);
                hasher.finish() as u32
            }
        }
    }
}

/// Token ID tree for format-agnostic vocabulary representation.
///
/// This structure serves as an intermediate representation when converting
/// between tokenizer formats. The tree structure preserves BPE merge hierarchy
/// which is needed for correct tokenization behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenIdTree {
    /// Original format this vocabulary was loaded from.
    pub source_format: SourceFormat,
    /// Type of tokenizer model.
    pub tokenizer_type: TokenizerType,
    /// Root of the token tree (or flat list for non-hierarchical tokenizers).
    pub root: TokenEntry,
}

impl TokenIdTree {
    /// Create a new token tree with a root entry.
    pub fn new(source_format: SourceFormat, tokenizer_type: TokenizerType, root: TokenEntry) -> Self {
        Self {
            source_format,
            tokenizer_type,
            root,
        }
    }

    /// Count all tokens in the tree.
    pub fn vocab_size(&self) -> usize {
        Self::count_entries(&self.root)
    }

    fn count_entries(entry: &TokenEntry) -> usize {
        let mut count = 1;
        if let Some(ref left) = entry.left {
            count += Self::count_entries(left);
        }
        if let Some(ref right) = entry.right {
            count += Self::count_entries(right);
        }
        count
    }

    /// Collect all entries into a Vec (for iteration).
    pub fn collect_entries(&self) -> Vec<&TokenEntry> {
        let mut entries = Vec::new();
        Self::collect_from_entry(&self.root, &mut entries);
        entries
    }

    fn collect_from_entry<'a>(entry: &'a TokenEntry, out: &mut Vec<&'a TokenEntry>) {
        out.push(entry);
        if let Some(ref left) = entry.left {
            Self::collect_from_entry(left, out);
        }
        if let Some(ref right) = entry.right {
            Self::collect_from_entry(right, out);
        }
    }

    /// Get all special tokens.
    pub fn special_tokens(&self) -> Vec<&TokenEntry> {
        self.collect_entries().into_iter().filter(|e| e.special).collect()
    }
}

impl TokenEntry {
    /// Create a new token entry.
    pub fn new(id: impl Into<TokenId>, token: String, embedding_row: u32) -> Self {
        Self {
            id: id.into(),
            token,
            embedding_row,
            special: false,
            special_type: String::new(),
            score: None,
            left: None,
            right: None,
        }
    }

    /// Create a special token entry.
    pub fn special(id: impl Into<TokenId>, token: String, embedding_row: u32, special_type: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            token,
            embedding_row,
            special: true,
            special_type: special_type.into(),
            score: None,
            left: None,
            right: None,
        }
    }

    /// Set the score (for Unigram models).
    pub fn with_score(mut self, score: f32) -> Self {
        self.score = Some(score);
        self
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_token_entry() {
        let entry = TokenEntry::new(0u32, "hello".to_string(), 0);
        assert_eq!(entry.token, "hello");
        assert_eq!(entry.embedding_row, 0);
        assert!(!entry.special);
        assert!(entry.score.is_none());
    }

    #[test]
    fn test_create_special_token() {
        let entry = TokenEntry::special(0u32, "<s>".to_string(), 0, "bos");
        assert!(entry.special);
        assert_eq!(entry.special_type, "bos");
    }

    #[test]
    fn test_token_with_score() {
        let entry = TokenEntry::new(0u32, "hello".to_string(), 0).with_score(-2.5);
        assert_eq!(entry.score, Some(-2.5));
    }

    #[test]
    fn test_token_id_variants() {
        let numeric: TokenId = 42u32.into();
        let string: TokenId = "tok_42".into();

        assert_eq!(numeric, TokenId::Numeric(42));
        assert_eq!(string, TokenId::String("tok_42".to_string()));
    }
}
