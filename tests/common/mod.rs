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
//! Logs are written to `logs/masstree.jsonl` as newline-delimited JSON (NDJSON).
//! Use `jq` for pretty-printing and filtering:
//!
//! ```bash
//! # Pretty-print all logs
//! cat logs/masstree.jsonl | jq .
//!
//! # Convert to JSON array
//! cat logs/masstree.jsonl | jq -s .
//!
//! # Find events for a specific key
//! cat logs/masstree.jsonl | jq 'select(.fields.ikey == 16)'
//!
//! # Show only errors
//! cat logs/masstree.jsonl | jq 'select(.level == "ERROR")'
//! ```

#![allow(dead_code)]

use std::env;
use std::fs::OpenOptions;
use std::path::PathBuf;
use std::sync::Once;

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
            log_file: "masstree.jsonl".to_string(),
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

#[expect(clippy::expect_used)]
fn setup_tracing() {
    let config = TracingConfig::from_env();

    // Create log directory
    std::fs::create_dir_all(&config.log_dir).expect("Failed to create log directory");

    let log_path = config.log_dir.join(&config.log_file);

    // Open file in append mode (nextest runs tests in separate processes)
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .expect("Failed to open log file");

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

    // === File Layer (NDJSON format) ===
    // Writes one JSON object per line. Use `jq` for pretty-printing.
    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::sync::Mutex::new(file))
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
