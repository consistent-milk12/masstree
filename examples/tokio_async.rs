//! Tokio async integration example for [`MassTree15`].
//!
//! [`MassTree15`] is synchronous but thread-safe, making it ideal for use with
//! Tokio's multi-threaded runtime. This example demonstrates patterns for
//! integrating [`MassTree`] with async code.
//!
//! Key patterns:
//! - Shared state across async tasks
//! - `spawn_blocking` for CPU-intensive operations
//! - Background maintenance tasks
//! - Rate-limited concurrent access
//!
//! Run with: `cargo run --example tokio_async --release`

use masstree::MassTree15;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio::time::Instant;

/// Shared application state containing the [`MassTree`]
struct AppState {
    tree: MassTree15<String>,
}

impl AppState {
    fn new() -> Self {
        Self {
            tree: MassTree15::new(),
        }
    }
}

#[tokio::main]
#[expect(clippy::too_many_lines, clippy::unwrap_used, clippy::similar_names)]
async fn main() {
    println!("=== MassTree + Tokio Async Integration ===\n");

    // =========================================================================
    // Example 1: Shared State Across Async Tasks
    // =========================================================================
    println!("--- Example 1: Shared State Across Async Tasks ---\n");

    let state = Arc::new(AppState::new());

    // Spawn multiple async tasks that access the same tree
    let mut handles = vec![];

    for task_id in 0..4 {
        let state = Arc::clone(&state);
        let handle = tokio::spawn(async move {
            let guard = state.tree.guard();

            // Each task inserts its own entries
            for i in 0..1000 {
                let key = format!("task{task_id}/entry{i}");
                let value = format!("value_from_task_{task_id}");
                let _ = state.tree.insert_with_guard(key.as_bytes(), value, &guard);
            }

            // Count entries for this task
            let prefix = format!("task{task_id}/");
            let mut count = 0;
            state.tree.scan_prefix(
                prefix.as_bytes(),
                |_, _| {
                    count += 1;
                    true
                },
                &guard,
            );

            (task_id, count)
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        let (task_id, count) = handle.await.unwrap();
        println!("Task {task_id} inserted {count} entries");
    }
    println!("Total tree size: {}\n", state.tree.len());

    // =========================================================================
    // Example 2: spawn_blocking for CPU-intensive operations
    // =========================================================================
    println!("--- Example 2: spawn_blocking for Heavy Operations ---\n");

    let state = Arc::new(AppState::new());

    // Pre-populate with data
    {
        let guard = state.tree.guard();
        for i in 0..100_000 {
            let key = format!("item/{i:06}");
            let value = format!("value_{i}");
            let _ = state.tree.insert_with_guard(key.as_bytes(), value, &guard);
        }
    }
    println!("Pre-populated with 100,000 entries");

    let start = Instant::now();

    // For CPU-intensive operations, use spawn_blocking to avoid blocking
    // the async runtime's worker threads
    let state_clone = Arc::clone(&state);
    let result = tokio::task::spawn_blocking(move || {
        let guard = state_clone.tree.guard();
        let mut count = 0;
        let mut total_len = 0usize;

        // CPU-intensive: scan and process all entries
        for entry in state_clone.tree.iter(&guard) {
            count += 1;
            // entry.value() returns &ValuePtr<String>
            total_len += entry.value().len();
        }

        (count, total_len)
    })
    .await
    .unwrap();

    let elapsed = start.elapsed();
    println!(
        "Scanned {} entries, total value length: {} bytes",
        result.0, result.1
    );
    println!("Completed in {elapsed:?}\n");

    // =========================================================================
    // Example 3: Background Maintenance Task
    // =========================================================================
    println!("--- Example 3: Background Maintenance Task ---\n");

    let state = Arc::new(AppState::new());

    // Spawn a background task for periodic maintenance
    let maintenance_state = Arc::clone(&state);
    let maintenance_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(100));
        let mut iterations = 0;

        loop {
            interval.tick().await;
            iterations += 1;

            // Process pending coalesce operations (cleanup empty leaves)
            let guard = maintenance_state.tree.guard();
            let processed = maintenance_state.tree.process_coalesce(&guard);

            if processed > 0 {
                println!("Maintenance: processed {processed} empty leaves");
            }

            // Run for 5 iterations then stop
            if iterations >= 5 {
                println!("Maintenance task completed after {iterations} iterations");
                break;
            }
        }
    });

    // Simulate workload with inserts and removes
    let workload_state = Arc::clone(&state);
    let workload_handle = tokio::spawn(async move {
        for round in 0..3 {
            // Insert batch - guard must not be held across await!
            {
                let guard = workload_state.tree.guard();
                for i in 0..1000 {
                    let key = format!("temp/round{round}/item{i}");
                    let _ = workload_state.tree.insert_with_guard(
                        key.as_bytes(),
                        format!("value_{i}"),
                        &guard,
                    );
                }
                println!(
                    "Round {round}: Inserted 1000 entries, size = {}",
                    workload_state.tree.len()
                );
            } // guard dropped here before await

            // Small delay - no guard held across this await
            tokio::time::sleep(Duration::from_millis(50)).await;

            // Remove batch
            {
                let guard = workload_state.tree.guard();
                for i in 0..1000 {
                    let key = format!("temp/round{round}/item{i}");
                    let _ = workload_state
                        .tree
                        .remove_with_guard(key.as_bytes(), &guard);
                }
                println!(
                    "Round {round}: Removed 1000 entries, size = {}",
                    workload_state.tree.len()
                );
            } // guard dropped here
        }
    });

    // Wait for both to complete
    let _ = tokio::join!(maintenance_handle, workload_handle);
    println!("Final tree size: {}\n", state.tree.len());

    // =========================================================================
    // Example 4: Rate-Limited Concurrent Access
    // =========================================================================
    println!("--- Example 4: Rate-Limited Concurrent Access ---\n");

    let state = Arc::new(AppState::new());

    // Pre-populate
    {
        let guard = state.tree.guard();
        for i in 0..10_000 {
            let key = format!("rate/{i:05}");
            let _ = state
                .tree
                .insert_with_guard(key.as_bytes(), format!("v{i}"), &guard);
        }
    }

    // Limit concurrent tree access with a semaphore
    let semaphore = Arc::new(Semaphore::new(8)); // Max 8 concurrent readers

    let mut handles = vec![];
    let start = Instant::now();

    for i in 0..100 {
        let state = Arc::clone(&state);
        let sem = Arc::clone(&semaphore);

        let handle = tokio::spawn(async move {
            // Acquire permit before accessing tree
            let _permit = sem.acquire().await.unwrap();

            // Perform tree operation - guard must not be held across await!
            let value = {
                let key = format!("rate/{:05}", i % 10_000);
                let guard = state.tree.guard();
                state.tree.get_ref(key.as_bytes(), &guard).cloned()
            }; // guard dropped here

            // Simulate some async work - no guard held here
            tokio::time::sleep(Duration::from_micros(100)).await;

            value
        });
        handles.push(handle);
    }

    let mut found = 0;
    for handle in handles {
        if handle.await.unwrap().is_some() {
            found += 1;
        }
    }

    let elapsed = start.elapsed();
    println!("100 rate-limited lookups completed in {elapsed:?}");
    println!("Found: {found} entries\n");

    // =========================================================================
    // Example 5: Async Producer-Consumer Pattern
    // =========================================================================
    println!("--- Example 5: Async Producer-Consumer Pattern ---\n");

    let state = Arc::new(AppState::new());
    let (tx, mut rx) = tokio::sync::mpsc::channel::<(String, String)>(100);

    // Producer task - generates key-value pairs
    let producer = tokio::spawn(async move {
        for i in 0..1000 {
            let key = format!("stream/item{i:04}");
            let value = format!("streamed_value_{i}");
            if tx.send((key, value)).await.is_err() {
                break;
            }
            // Simulate async data source
            if i % 100 == 0 {
                tokio::time::sleep(Duration::from_micros(10)).await;
            }
        }
    });

    // Consumer task - writes to tree
    // Note: We create a new guard for each insert because LocalGuard is not Send.
    // In high-throughput scenarios, consider using spawn_blocking for batched inserts.
    let consumer_state = Arc::clone(&state);
    let consumer = tokio::spawn(async move {
        let mut count = 0;

        while let Some((key, value)) = rx.recv().await {
            // Create guard inside the loop - it cannot be held across await
            let guard = consumer_state.tree.guard();
            let _ = consumer_state
                .tree
                .insert_with_guard(key.as_bytes(), value, &guard);
            count += 1;
        }

        count
    });

    let (_, consumed) = tokio::join!(producer, consumer);
    println!("Consumed and inserted {} entries", consumed.unwrap());
    println!("Tree size: {}\n", state.tree.len());

    // =========================================================================
    // Example 6: Async-friendly Batch Operations
    // =========================================================================
    println!("--- Example 6: Async-friendly Batch Operations ---\n");

    let state = Arc::new(AppState::new());

    // Batch insert from async context using spawn_blocking
    let entries: Vec<(String, String)> = (0..50_000)
        .map(|i| (format!("batch/{i:06}"), format!("batch_value_{i}")))
        .collect();

    let start = Instant::now();

    let state_clone = Arc::clone(&state);
    tokio::task::spawn_blocking(move || {
        let guard = state_clone.tree.guard();
        for (key, value) in entries {
            let _ = state_clone
                .tree
                .insert_with_guard(key.as_bytes(), value, &guard);
        }
    })
    .await
    .unwrap();

    let elapsed = start.elapsed();
    println!("Batch inserted 50,000 entries in {elapsed:?}");
    println!(
        "Rate: {:.2} M ops/sec",
        50_000.0 / elapsed.as_secs_f64() / 1_000_000.0
    );
    println!("Tree size: {}\n", state.tree.len());

    // =========================================================================
    // Example 7: Multiple Concurrent Readers with Single Writer
    // =========================================================================
    println!("--- Example 7: Concurrent Readers + Single Writer ---\n");

    let state = Arc::new(AppState::new());

    // Pre-populate
    {
        let guard = state.tree.guard();
        for i in 0..10_000 {
            let key = format!("rw/{i:05}");
            let _ = state
                .tree
                .insert_with_guard(key.as_bytes(), format!("v{i}"), &guard);
        }
    }

    let start = Instant::now();

    // Single writer task
    let writer_state = Arc::clone(&state);
    let writer = tokio::spawn(async move {
        let mut writes = 0;
        for round in 0..100 {
            // Guard must be scoped - cannot be held across await
            {
                let guard = writer_state.tree.guard();
                for i in 0..100 {
                    let key = format!("rw/{:05}", (round * 100 + i) % 10_000);
                    let _ = writer_state.tree.insert_with_guard(
                        key.as_bytes(),
                        format!("updated_{round}_{i}"),
                        &guard,
                    );
                    writes += 1;
                }
            } // guard dropped here
            // Yield to readers periodically - no guard held
            tokio::task::yield_now().await;
        }
        writes
    });

    // Multiple reader tasks
    let mut readers = vec![];
    for reader_id in 0..4 {
        let state = Arc::clone(&state);
        let handle = tokio::spawn(async move {
            let mut reads = 0;
            for round in 0..1000 {
                // Guard must be scoped - cannot be held across await
                {
                    let guard = state.tree.guard();
                    let key = format!("rw/{:05}", (reader_id * 1000 + round) % 10_000);
                    if state.tree.get_ref(key.as_bytes(), &guard).is_some() {
                        reads += 1;
                    }
                } // guard dropped here
                // Yield occasionally - no guard held
                if round % 100 == 0 {
                    tokio::task::yield_now().await;
                }
            }
            reads
        });
        readers.push(handle);
    }

    let write_count = writer.await.unwrap();
    let mut total_reads = 0;
    for reader in readers {
        total_reads += reader.await.unwrap();
    }

    let elapsed = start.elapsed();
    println!("Writer: {write_count} writes");
    println!("Readers: {total_reads} total reads across 4 tasks");
    println!("Completed in {elapsed:?}\n");

    // =========================================================================
    // Example 8: Graceful Shutdown with Cleanup
    // =========================================================================
    println!("--- Example 8: Graceful Shutdown Pattern ---\n");

    let state = Arc::new(AppState::new());
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel::<()>();

    // Worker task that respects shutdown signal
    let worker_state = Arc::clone(&state);
    let worker = tokio::spawn(async move {
        let mut ops = 0;
        loop {
            // Check for shutdown signal
            if shutdown_rx.try_recv().is_ok() {
                println!("Worker received shutdown signal");
                break;
            }

            // Do some work - guard must be scoped
            {
                let guard = worker_state.tree.guard();
                let key = format!("worker/op{ops}");
                let _ =
                    worker_state
                        .tree
                        .insert_with_guard(key.as_bytes(), format!("v{ops}"), &guard);
                ops += 1;
            } // guard dropped here

            // Don't spin too fast - no guard held across await
            tokio::time::sleep(Duration::from_micros(100)).await;
        }
        ops
    });

    // Let worker run for a bit
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Signal shutdown
    let _ = shutdown_tx.send(());
    let ops_completed = worker.await.unwrap();

    // Final cleanup - process all pending coalesce
    let guard = state.tree.guard();
    let cleaned = state.tree.process_coalesce(&guard);

    println!("Worker completed {ops_completed} operations");
    println!("Cleanup processed {cleaned} empty leaves");
    println!("Final tree size: {}", state.tree.len());

    println!("\n=== Tokio Examples Complete ===");
}
