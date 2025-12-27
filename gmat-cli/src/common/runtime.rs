//! Runtime utilities for CPU and parallelism detection.

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Thread-safe progress tracker for parallel operations.
pub struct ProgressTracker {
    processed: AtomicUsize,
    total: usize,
    label: String,
}

impl ProgressTracker {
    pub fn new(total: usize, label: impl Into<String>) -> Self {
        Self {
            processed: AtomicUsize::new(0),
            total,
            label: label.into(),
        }
    }

    /// Increment with conditional display (every N items or at end).
    pub fn increment_every(&self, n: usize) {
        let current = self.processed.fetch_add(1, Ordering::Relaxed) + 1;
        if current.is_multiple_of(n) || current == self.total {
            eprint!("\r{}: {}/{}", self.label, current, self.total);
            let _ = std::io::stderr().flush();
        }
    }

    /// Increment and display with additional context.
    pub fn increment_with_extra(&self, extra: &str) {
        let current = self.processed.fetch_add(1, Ordering::Relaxed) + 1;
        eprint!("\r{}: {}/{} {}", self.label, current, self.total, extra);
        let _ = std::io::stderr().flush();
    }

    /// Print newline after progress is complete.
    pub fn finish(&self) {
        eprintln!();
    }
}

/// Run an async function in a new tokio runtime.
/// Consolidates the repeated pattern of Runtime::new()?.block_on(f).
pub fn run_blocking<F, T>(f: F) -> anyhow::Result<T>
where
    F: std::future::Future<Output = anyhow::Result<T>>,
{
    tokio::runtime::Runtime::new()?.block_on(f)
}

/// Get number of available CPU cores.
///
/// Returns the number of available parallelism units (typically CPU cores),
/// falling back to 4 if detection fails.
pub fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_cpus_returns_positive() {
        assert!(num_cpus() > 0);
    }
}
