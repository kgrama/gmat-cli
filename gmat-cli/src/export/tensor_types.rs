//! Tensor entry types for export pipeline.
//!
//! Handles parsing tensor metadata from JSON and resolving paths for quantization.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Represents a tensor entry from metadata - either simple 2D or N-D with planes.
#[derive(Debug, Clone)]
pub enum TensorEntry {
    /// Simple 2D tensor: single UUID
    Simple { uuid: String },
    /// N-D tensor split into planes
    NdPlanes {
        original_shape: Vec<usize>,
        matrix_shape: (usize, usize),
        plane_uuids: Vec<String>,
    },
}

impl TensorEntry {
    /// Parse a tensor entry from metadata JSON value.
    pub fn from_json(value: &serde_json::Value) -> Option<Self> {
        if let Some(uuid) = value.as_str() {
            // Simple 2D tensor
            Some(TensorEntry::Simple {
                uuid: uuid.to_string(),
            })
        } else if let Some(obj) = value.as_object() {
            // N-D tensor with planes
            let original_shape: Vec<usize> = obj
                .get("original_shape")?
                .as_array()?
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect();

            let matrix_arr = obj.get("matrix_shape")?.as_array()?;
            let matrix_shape = (
                matrix_arr.first()?.as_u64()? as usize,
                matrix_arr.get(1)?.as_u64()? as usize,
            );

            let plane_uuids: Vec<String> = obj
                .get("plane_uuids")?
                .as_array()?
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();

            Some(TensorEntry::NdPlanes {
                original_shape,
                matrix_shape,
                plane_uuids,
            })
        } else {
            None
        }
    }
}

/// Entry type for quantization job - resolved paths ready for processing.
#[derive(Debug, Clone)]
pub enum QuantizeEntry {
    /// Simple 2D tensor
    Simple { tensor_path: PathBuf },
    /// N-D tensor with multiple planes to reassemble
    NdPlanes {
        original_shape: Vec<usize>,
        matrix_shape: (usize, usize),
        plane_paths: Vec<PathBuf>,
    },
}

/// Check if a tensor file exists and log warning if not.
fn validate_tensor_path(path: &Path, context: &str) -> bool {
    if path.exists() {
        true
    } else {
        eprintln!("Warning: missing {}: {}", context, path.display());
        false
    }
}

/// Find a tensor entry by UUID in the name_to_entry map.
/// The config uses UUID as source, but metadata maps name -> entry.
/// Returns a QuantizeEntry with validated paths.
pub fn find_tensor_entry(
    name_to_entry: &HashMap<String, TensorEntry>,
    source_uuid: &str,
    tensors_dir: &Path,
) -> Option<QuantizeEntry> {
    // First check if source_uuid directly matches a simple tensor's UUID
    for entry in name_to_entry.values() {
        match entry {
            TensorEntry::Simple { uuid } if uuid == source_uuid => {
                let tensor_path = tensors_dir.join(format!("{}.gmat", uuid));
                if !validate_tensor_path(&tensor_path, "tensor") {
                    return None;
                }
                return Some(QuantizeEntry::Simple { tensor_path });
            }
            TensorEntry::NdPlanes {
                plane_uuids,
                original_shape,
                matrix_shape,
            } => {
                // Check if source_uuid matches any plane UUID (for per-plane export)
                // or the base UUID pattern (first plane UUID without suffix)
                let base_uuid = plane_uuids.first()?.strip_suffix("_0")?;
                if source_uuid == base_uuid || plane_uuids.contains(&source_uuid.to_string()) {
                    // Build paths for all planes
                    let plane_paths: Vec<PathBuf> = plane_uuids
                        .iter()
                        .map(|uuid| tensors_dir.join(format!("{}.gmat", uuid)))
                        .collect();

                    // Validate all paths exist
                    for path in &plane_paths {
                        if !validate_tensor_path(path, "tensor plane") {
                            return None;
                        }
                    }

                    return Some(QuantizeEntry::NdPlanes {
                        original_shape: original_shape.clone(),
                        matrix_shape: *matrix_shape,
                        plane_paths,
                    });
                }
            }
            _ => {}
        }
    }

    // Fallback: try as a simple tensor path directly
    let tensor_path = tensors_dir.join(format!("{}.gmat", source_uuid));
    if tensor_path.exists() {
        return Some(QuantizeEntry::Simple { tensor_path });
    }

    eprintln!("Warning: tensor not found: {}", source_uuid);
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    // ==================== TensorEntry::from_json tests ====================

    #[test]
    fn test_tensor_entry_from_json_simple() {
        let json = json!("550e8400-e29b-41d4-a716-446655440000");
        let entry = TensorEntry::from_json(&json).unwrap();

        match entry {
            TensorEntry::Simple { uuid } => {
                assert_eq!(uuid, "550e8400-e29b-41d4-a716-446655440000");
            }
            _ => panic!("Expected Simple variant"),
        }
    }

    #[test]
    fn test_tensor_entry_from_json_1d_tensor() {
        // 1D tensor [64] becomes matrix_shape [1, 64]
        let json = json!({
            "original_shape": [64],
            "matrix_shape": [1, 64],
            "num_planes": 1,
            "plane_uuids": ["uuid_0"]
        });
        let entry = TensorEntry::from_json(&json).unwrap();

        match entry {
            TensorEntry::NdPlanes {
                original_shape,
                matrix_shape,
                plane_uuids,
            } => {
                assert_eq!(original_shape, vec![64]);
                assert_eq!(matrix_shape, (1, 64));
                assert_eq!(plane_uuids.len(), 1);
            }
            _ => panic!("Expected NdPlanes variant"),
        }
    }

    #[test]
    fn test_tensor_entry_from_json_3d_planes() {
        let json = json!({
            "original_shape": [4, 32, 64],
            "matrix_shape": [32, 64],
            "num_planes": 4,
            "plane_uuids": ["uuid_0", "uuid_1", "uuid_2", "uuid_3"]
        });
        let entry = TensorEntry::from_json(&json).unwrap();

        match entry {
            TensorEntry::NdPlanes {
                original_shape,
                matrix_shape,
                plane_uuids,
            } => {
                assert_eq!(original_shape, vec![4, 32, 64]);
                assert_eq!(matrix_shape, (32, 64));
                assert_eq!(plane_uuids.len(), 4);
                assert_eq!(plane_uuids[0], "uuid_0");
            }
            _ => panic!("Expected NdPlanes variant"),
        }
    }

    #[test]
    fn test_tensor_entry_from_json_4d_tensor() {
        let json = json!({
            "original_shape": [2, 4, 32, 64],
            "matrix_shape": [32, 64],
            "num_planes": 8,
            "plane_uuids": ["u_0", "u_1", "u_2", "u_3", "u_4", "u_5", "u_6", "u_7"]
        });
        let entry = TensorEntry::from_json(&json).unwrap();

        match entry {
            TensorEntry::NdPlanes {
                original_shape,
                matrix_shape,
                plane_uuids,
            } => {
                assert_eq!(original_shape, vec![2, 4, 32, 64]);
                assert_eq!(matrix_shape, (32, 64));
                assert_eq!(plane_uuids.len(), 8);
            }
            _ => panic!("Expected NdPlanes variant"),
        }
    }

    #[test]
    fn test_tensor_entry_from_json_invalid() {
        let json = json!(123); // number, not string or object
        assert!(TensorEntry::from_json(&json).is_none());
    }

    #[test]
    fn test_tensor_entry_from_json_missing_fields() {
        let json = json!({
            "original_shape": [4, 32, 64],
            // missing matrix_shape and plane_uuids
        });
        assert!(TensorEntry::from_json(&json).is_none());
    }

    // ==================== find_tensor_entry tests ====================

    #[test]
    fn test_find_tensor_entry_simple() {
        let dir = TempDir::new().unwrap();
        let uuid = "550e8400-e29b-41d4-a716-446655440000";

        // Create a dummy tensor file
        std::fs::write(dir.path().join(format!("{}.gmat", uuid)), b"dummy").unwrap();

        let mut name_to_entry = HashMap::new();
        name_to_entry.insert(
            "model.weight".to_string(),
            TensorEntry::Simple {
                uuid: uuid.to_string(),
            },
        );

        let result = find_tensor_entry(&name_to_entry, uuid, dir.path());
        assert!(result.is_some());

        match result.unwrap() {
            QuantizeEntry::Simple { tensor_path } => {
                assert!(tensor_path.exists());
            }
            _ => panic!("Expected Simple variant"),
        }
    }

    #[test]
    fn test_find_tensor_entry_nd_planes() {
        let dir = TempDir::new().unwrap();
        let base_uuid = "550e8400-e29b-41d4-a716-446655440000";

        // Create plane files
        for i in 0..4 {
            std::fs::write(
                dir.path().join(format!("{}_{}.gmat", base_uuid, i)),
                b"dummy",
            )
            .unwrap();
        }

        let plane_uuids: Vec<String> = (0..4).map(|i| format!("{}_{}", base_uuid, i)).collect();

        let mut name_to_entry = HashMap::new();
        name_to_entry.insert(
            "conv.weight".to_string(),
            TensorEntry::NdPlanes {
                original_shape: vec![4, 32, 64],
                matrix_shape: (32, 64),
                plane_uuids,
            },
        );

        // Find by base UUID
        let result = find_tensor_entry(&name_to_entry, base_uuid, dir.path());
        assert!(result.is_some());

        match result.unwrap() {
            QuantizeEntry::NdPlanes {
                original_shape,
                matrix_shape,
                plane_paths,
            } => {
                assert_eq!(original_shape, vec![4, 32, 64]);
                assert_eq!(matrix_shape, (32, 64));
                assert_eq!(plane_paths.len(), 4);
                for path in &plane_paths {
                    assert!(path.exists());
                }
            }
            _ => panic!("Expected NdPlanes variant"),
        }
    }

    #[test]
    fn test_find_tensor_entry_missing_file() {
        let dir = TempDir::new().unwrap();
        let uuid = "nonexistent-uuid";

        let mut name_to_entry = HashMap::new();
        name_to_entry.insert(
            "model.weight".to_string(),
            TensorEntry::Simple {
                uuid: uuid.to_string(),
            },
        );

        let result = find_tensor_entry(&name_to_entry, uuid, dir.path());
        assert!(result.is_none());
    }

    #[test]
    fn test_find_tensor_entry_fallback_direct_path() {
        let dir = TempDir::new().unwrap();
        let uuid = "direct-uuid";

        // Create file but don't add to name_to_entry
        std::fs::write(dir.path().join(format!("{}.gmat", uuid)), b"dummy").unwrap();

        let name_to_entry: HashMap<String, TensorEntry> = HashMap::new();

        // Should find via fallback
        let result = find_tensor_entry(&name_to_entry, uuid, dir.path());
        assert!(result.is_some());
    }

    #[test]
    fn test_find_tensor_entry_missing_plane() {
        let dir = TempDir::new().unwrap();
        let base_uuid = "550e8400-e29b-41d4-a716-446655440000";

        // Create only some plane files (missing plane 2)
        std::fs::write(dir.path().join(format!("{}_0.gmat", base_uuid)), b"dummy").unwrap();
        std::fs::write(dir.path().join(format!("{}_1.gmat", base_uuid)), b"dummy").unwrap();
        // plane 2 missing
        std::fs::write(dir.path().join(format!("{}_3.gmat", base_uuid)), b"dummy").unwrap();

        let plane_uuids: Vec<String> = (0..4).map(|i| format!("{}_{}", base_uuid, i)).collect();

        let mut name_to_entry = HashMap::new();
        name_to_entry.insert(
            "conv.weight".to_string(),
            TensorEntry::NdPlanes {
                original_shape: vec![4, 32, 64],
                matrix_shape: (32, 64),
                plane_uuids,
            },
        );

        // Should fail because plane 2 is missing
        let result = find_tensor_entry(&name_to_entry, base_uuid, dir.path());
        assert!(result.is_none());
    }
}
