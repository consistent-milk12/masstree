//! Common test utilities with comprehensive tracing setup.
//!
//! # Usage
//!
//! ```rust,ignore
//! mod common;
//!
//! #[test]
//! fn my_test() {
//!     common::init_tracing();
//!     // ... test code with tracing::info!, tracing::debug!, etc.
//! }
//! ```
//!
//! # Configuration
//!
//! Environment variables:
//! - `RUST_LOG`: Filter directives (e.g., `masstree=debug,masstree::tree::locked=trace`)
//! - `MASSTREE_LOG_DIR`: Log directory (default: `logs/`)
//! - `MASSTREE_LOG_CONSOLE`: Set to "0" to disable console output
//!
//! # Log Files
//!
//! Logs are written to `logs/masstree.json` as a formatted JSON array.
//! Each entry is pretty-printed for readability.
//!
//! ```bash
//! # View logs directly (already formatted)
//! cat logs/masstree.json | jq .
//!
//! # Find events for a specific key
//! just log-key 16
//!
//! # Show only errors
//! just log-errors
//! ```

#![allow(dead_code)]

use std::env;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, Once};

use tracing::Level;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer, Registry};

/// Ensures tracing is only initialized once across all tests.
static INIT: Once = Once::new();

/// Initialize the tracing subscriber with file and console logging.
///
/// Safe to call multiple times - only the first call takes effect.
/// Logs are written to `logs/masstree.json` in JSON format.
pub fn init_tracing() {
    INIT.call_once(|| {
        setup_tracing();
    });
}

/// Configuration for tracing setup.
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Directory for log files.
    pub log_dir: PathBuf,
    /// Log file name.
    pub log_file: String,
    /// Enable console output.
    pub console_enabled: bool,
    /// Default log level if RUST_LOG is not set.
    pub default_level: Level,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            log_dir: PathBuf::from("logs"),
            log_file: "masstree.json".to_string(),
            console_enabled: true,
            default_level: Level::INFO,
        }
    }
}

impl TracingConfig {
    /// Create config from environment variables.
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(dir) = env::var("MASSTREE_LOG_DIR") {
            config.log_dir = PathBuf::from(dir);
        }

        if env::var("MASSTREE_LOG_CONSOLE").is_ok_and(|v| v == "0") {
            config.console_enabled = false;
        }

        config
    }
}

/// Create an EnvFilter from RUST_LOG or use default level.
fn make_filter(default_level: Level) -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(format!("{default_level}")))
}

/// Thread-safe file writer that outputs formatted JSON arrays.
///
/// Instead of NDJSON (one compact JSON object per line), this writer:
/// 1. Writes `[\n` on first entry
/// 2. Pretty-prints each JSON object with 2-space indentation
/// 3. Adds commas between entries
/// 4. Writes `\n]` on drop to close the array
struct FormattedJsonWriter {
    file: Mutex<File>,
    first_entry: AtomicBool,
}

impl FormattedJsonWriter {
    fn new(path: PathBuf) -> std::io::Result<Self> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        // Write opening bracket
        file.write_all(b"[\n")?;
        Ok(Self {
            file: Mutex::new(file),
            first_entry: AtomicBool::new(true),
        })
    }

    /// Parse compact JSON, pretty-print it, and write to file.
    fn write_formatted(&self, compact_json: &[u8]) -> std::io::Result<usize> {
        // Skip empty lines
        let trimmed = compact_json
            .iter()
            .copied()
            .filter(|&b| b != b'\n' && b != b'\r')
            .collect::<Vec<_>>();
        if trimmed.is_empty() {
            return Ok(compact_json.len());
        }

        // Parse the compact JSON
        let value: serde_json::Value = match serde_json::from_slice(&trimmed) {
            Ok(v) => v,
            Err(_) => {
                // If parsing fails, write as-is (shouldn't happen with tracing)
                let mut file = self.file.lock().unwrap();
                return file.write(compact_json);
            }
        };

        // Pretty-print with 2-space indentation
        let pretty = serde_json::to_string_pretty(&value)
            .unwrap_or_else(|_| String::from_utf8_lossy(&trimmed).to_string());

        // Indent each line by 2 spaces (for array nesting)
        let indented: String = pretty
            .lines()
            .map(|line| format!("  {line}"))
            .collect::<Vec<_>>()
            .join("\n");

        let mut file = self.file.lock().unwrap();

        // Add comma before entry (except for first)
        if self.first_entry.swap(false, Ordering::SeqCst) {
            // First entry - no comma
            write!(file, "{indented}")?;
        } else {
            // Subsequent entries - comma before
            write!(file, ",\n{indented}")?;
        }

        drop(file);

        Ok(compact_json.len())
    }
}

impl Write for &FormattedJsonWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.write_formatted(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let mut file = self.file.lock().unwrap();
        file.flush()
    }
}

impl Drop for FormattedJsonWriter {
    fn drop(&mut self) {
        // Close the JSON array
        if let Ok(mut file) = self.file.lock() {
            let _ = file.write_all(b"\n]\n");
            let _ = file.flush();
        }
    }
}

#[expect(clippy::expect_used)]
fn setup_tracing() {
    let config = TracingConfig::from_env();

    // Create log directory
    std::fs::create_dir_all(&config.log_dir).expect("Failed to create log directory");

    let log_path = config.log_dir.join(&config.log_file);

    // Create file writer (leaked to have 'static lifetime)
    // Uses FormattedJsonWriter to output pretty-printed JSON arrays
    let file_writer: &'static FormattedJsonWriter = Box::leak(Box::new(
        FormattedJsonWriter::new(log_path).expect("Failed to create log file"),
    ));

    // === Console Layer ===
    let console_layer = if config.console_enabled {
        Some(
            tracing_subscriber::fmt::layer()
                .with_thread_ids(true)
                .with_thread_names(true)
                .with_target(true)
                .with_file(true)
                .with_line_number(true)
                .with_span_events(FmtSpan::CLOSE)
                .with_ansi(true)
                .compact()
                .with_filter(make_filter(config.default_level)),
        )
    } else {
        None
    };

    // === File Layer (JSON format) ===
    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(move || file_writer)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_target(true)
        .with_file(true)
        .with_line_number(true)
        .with_span_events(FmtSpan::CLOSE)
        .json()
        .with_filter(make_filter(config.default_level));

    // Compose and install subscriber (use try_init to avoid panic if lib already set one)
    let _ = Registry::default()
        .with(console_layer)
        .with(file_layer)
        .try_init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracing_init() {
        init_tracing();
        tracing::info!("Tracing initialized successfully");
        tracing::debug!(key = "test_key", value = 42, "Debug event");
        tracing::trace!(thread = ?std::thread::current().id(), "Trace event");
    }
}
