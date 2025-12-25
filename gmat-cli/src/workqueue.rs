//! Async work queue using tokio channels and tokio-rayon for CPU-bound work.
//!
//! Architecture:
//! - Producer: reads/deserializes items, sends to work channel
//! - Workers: pull from channel, process on rayon pool, send to output channel
//! - Writer: pulls from output channel, persists to disk

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};

/// Shared pipeline state for coordination and progress tracking.
pub struct PipelineState {
    /// Set to false to signal all threads to stop.
    pub running: AtomicBool,
    /// Number of items sent by producer.
    pub produced: AtomicU64,
    /// Number of items completed by writer.
    pub completed: AtomicU64,
}

impl PipelineState {
    pub fn new() -> Self {
        Self {
            running: AtomicBool::new(true),
            produced: AtomicU64::new(0),
            completed: AtomicU64::new(0),
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    pub fn inc_produced(&self) -> u64 {
        self.produced.fetch_add(1, Ordering::Relaxed) + 1
    }

    pub fn inc_completed(&self) -> u64 {
        self.completed.fetch_add(1, Ordering::Relaxed) + 1
    }
}

impl Default for PipelineState {
    fn default() -> Self {
        Self::new()
    }
}

/// Run a pipeline: producer -> workers (rayon) -> writer
///
/// - `input_buffer`: capacity of producer -> worker channel
/// - `output_buffer`: capacity of worker -> writer channel
/// - `producer`: async fn that sends items to the work channel
/// - `process`: sync fn (runs on rayon) that transforms I -> O
/// - `writer`: async fn that receives processed items and writes them
pub async fn run_pipeline<I, O, E, P, W, WR, PFut, WRFut>(
    input_buffer: usize,
    output_buffer: usize,
    producer: P,
    process: W,
    writer: WR,
) -> Result<(), E>
where
    I: Send + 'static,
    O: Send + 'static,
    E: Send + From<anyhow::Error> + 'static,
    P: FnOnce(mpsc::Sender<I>, Arc<PipelineState>) -> PFut + Send + 'static,
    PFut: std::future::Future<Output = Result<(), E>> + Send + 'static,
    W: Fn(I) -> Result<O, E> + Send + Sync + 'static,
    WR: FnOnce(mpsc::Receiver<Result<O, E>>, Arc<PipelineState>) -> WRFut + Send + 'static,
    WRFut: std::future::Future<Output = Result<(), E>> + Send + 'static,
{
    let state = Arc::new(PipelineState::new());

    let (work_tx, mut work_rx) = mpsc::channel::<I>(input_buffer);
    let (out_tx, out_rx) = mpsc::channel::<Result<O, E>>(output_buffer);

    let process = Arc::new(process);

    // Spawn producer
    let producer_state = Arc::clone(&state);
    let producer_handle = tokio::spawn(async move {
        producer(work_tx, producer_state).await
    });

    // Spawn writer
    let writer_state = Arc::clone(&state);
    let writer_handle = tokio::spawn(async move {
        writer(out_rx, writer_state).await
    });

    // Limit concurrent workers to rayon's thread pool size
    let max_workers = rayon::current_num_threads();
    let semaphore = Arc::new(Semaphore::new(max_workers));

    // Process items: pull from work channel, run on rayon, send to output
    while let Some(item) = work_rx.recv().await {
        if !state.is_running() {
            break;
        }

        let process = Arc::clone(&process);
        let out_tx = out_tx.clone();
        let state = Arc::clone(&state);
        let permit = Arc::clone(&semaphore)
            .acquire_owned()
            .await
            .expect("semaphore closed");

        tokio::spawn(async move {
            let result = tokio_rayon::spawn(move || process(item)).await;

            if state.is_running() {
                let _ = out_tx.send(result).await;
            }
            drop(permit); // Release semaphore slot
        });
    }

    // Drop our sender so writer knows we're done
    drop(out_tx);

    // Wait for producer and writer
    producer_handle.await.map_err(|e| anyhow::anyhow!("producer panicked: {}", e))??;
    writer_handle.await.map_err(|e| anyhow::anyhow!("writer panicked: {}", e))??;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pipeline_simple() {
        let result: Result<(), anyhow::Error> = run_pipeline(
            4,
            4,
            |tx, state| async move {
                for i in 0..10 {
                    state.inc_produced();
                    tx.send(i).await.map_err(|e| anyhow::anyhow!("{}", e))?;
                }
                Ok(())
            },
            |x: i32| Ok(x * 2),
            |mut rx, state| async move {
                let mut results = Vec::new();
                while let Some(result) = rx.recv().await {
                    results.push(result?);
                    state.inc_completed();
                }
                assert_eq!(results.len(), 10);
                Ok(())
            },
        )
        .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_pipeline_state_tracking() {
        let state_clone = Arc::new(std::sync::Mutex::new(None));
        let state_capture = Arc::clone(&state_clone);

        let _: Result<(), anyhow::Error> = run_pipeline(
            2,
            2,
            |tx, state| async move {
                for i in 0..5 {
                    state.inc_produced();
                    tx.send(i).await.map_err(|e| anyhow::anyhow!("{}", e))?;
                }
                Ok(())
            },
            |x: i32| Ok(x + 1),
            |mut rx, state| async move {
                while let Some(result) = rx.recv().await {
                    let _ = result?;
                    state.inc_completed();
                }
                *state_capture.lock().unwrap() = Some((
                    state.produced.load(Ordering::Relaxed),
                    state.completed.load(Ordering::Relaxed),
                ));
                Ok(())
            },
        )
        .await;

        let (produced, completed) = state_clone.lock().unwrap().unwrap();
        assert_eq!(produced, 5);
        assert_eq!(completed, 5);
    }

}
