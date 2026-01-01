//! Aggregate Criterion benchmark results into summary CSV/table.
//!
//! Walks `target/criterion/` and extracts statistics from `estimates.json` files.
//!
//! ## Usage
//!
//! ```bash
//! # Table output (default)
//! cargo run --bin criterion_summary
//!
//! # CSV output
//! cargo run --bin criterion_summary -- --csv
//!
//! # Custom criterion directory
//! cargo run --bin criterion_summary -- --dir target/criterion
//!
//! # Filter by pattern
//! cargo run --bin criterion_summary -- --filter "read_scaling"
//!
//! # Filter by benchmark file (extracts group names from source)
//! cargo run --bin criterion_summary -- --bench range_masstree24_crit
//!
//! # Compare new vs base (shows regression/improvement)
//! cargo run --bin criterion_summary -- --compare
//! ```

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::fmt::Write;
use std::fs as StdFs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use comfy_table::{Cell, Color, Table, presets::UTF8_FULL_CONDENSED};
use regex::Regex;
use serde::Deserialize;

/// Aggregate Criterion benchmark results into a summary.
#[derive(Parser, Debug)]
#[command(name = "criterion_summary")]
#[command(about = "Aggregate Criterion benchmark results into summary CSV/table")]
struct Args {
    /// Criterion output directory
    #[arg(short, long, default_value = "target/criterion")]
    dir: PathBuf,

    /// Output as CSV instead of table
    #[arg(long)]
    csv: bool,

    /// Filter benchmarks by name pattern (regex)
    #[arg(short, long)]
    filter: Option<String>,

    /// Filter by benchmark file name (extracts group names from benches/<name>.rs)
    #[arg(short, long)]
    bench: Option<String>,

    /// Compare new vs base results (show delta %)
    #[arg(long)]
    compare: bool,

    /// Sort by column: name, mean, median (default: name)
    #[arg(short, long, default_value = "name")]
    sort: SortColumn,

    /// Output file (prints to stdout if not specified)
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, Default)]
enum SortColumn {
    #[default]
    Name,
    Mean,
    Median,
}

/// Criterion's estimates.json structure
#[derive(Debug, Deserialize)]
struct Estimates {
    mean: Estimate,
    median: Estimate,
    std_dev: Estimate,
    #[serde(default)]
    slope: Option<Estimate>,
}

#[derive(Debug, Deserialize)]
#[expect(clippy::struct_field_names)]
struct Estimate {
    point_estimate: f64,

    #[allow(dead_code)]
    standard_error: f64,

    #[allow(dead_code)]
    confidence_interval: ConfidenceInterval,
}

#[derive(Debug, Deserialize)]
struct ConfidenceInterval {
    #[allow(dead_code)]
    confidence_level: f64,

    #[allow(dead_code)]
    lower_bound: f64,

    #[allow(dead_code)]
    upper_bound: f64,
}

/// Aggregated benchmark result
#[derive(Debug, Clone)]
struct BenchmarkResult {
    name: String,
    mean_ns: f64,
    median_ns: f64,
    std_dev_ns: f64,

    #[allow(dead_code)] // Available for future use (throughput calculations)
    slope_ns: Option<f64>,

    // For comparison mode
    #[allow(dead_code)] // Kept for symmetry, may be used for mean comparison
    base_mean_ns: Option<f64>,
    base_median_ns: Option<f64>,
}

impl BenchmarkResult {
    fn mean_formatted(&self) -> String {
        format_duration(self.mean_ns)
    }

    fn median_formatted(&self) -> String {
        format_duration(self.median_ns)
    }

    fn std_dev_formatted(&self) -> String {
        format_duration(self.std_dev_ns)
    }

    fn delta_percent(&self) -> Option<f64> {
        self.base_median_ns.map(|base| {
            if base == 0.0 {
                0.0
            } else {
                (self.median_ns - base) / base * 100.0
            }
        })
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Build filter: either from --filter regex or --bench file extraction
    let filter_re = match (&args.filter, &args.bench) {
        (Some(_), Some(_)) => {
            bail!("Cannot use both --filter and --bench simultaneously");
        }

        (Some(f), None) => Some(Regex::new(f).context("Invalid filter regex")?),

        (None, Some(bench_name)) => {
            let groups = extract_benchmark_groups(bench_name)?;
            if groups.is_empty() {
                bail!("No benchmark groups found in benches/{bench_name}.rs");
            }

            eprintln!(
                "Found {} groups in {}: {}",
                groups.len(),
                bench_name,
                groups
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
                    + if groups.len() > 3 { "..." } else { "" }
            );
            // Build regex that matches any of the group names at the start
            let pattern = format!(
                "^({})",
                groups
                    .into_iter()
                    .map(|g| regex::escape(&g))
                    .collect::<Vec<_>>()
                    .join("|")
            );
            Some(
                Regex::new(&pattern)
                    .context("Failed to build filter regex from benchmark groups")?,
            )
        }
        (None, None) => None,
    };

    let mut results = collect_results(&args.dir, filter_re.as_ref(), args.compare)?;

    // Sort results
    match args.sort {
        SortColumn::Name => results.sort_by(|a, b| a.name.cmp(&b.name)),

        SortColumn::Mean => {
            results.sort_by(|a, b| a.mean_ns.partial_cmp(&b.mean_ns).unwrap_or(Ordering::Equal));
        }

        SortColumn::Median => {
            results.sort_by(|a, b| {
                a.median_ns
                    .partial_cmp(&b.median_ns)
                    .unwrap_or(Ordering::Equal)
            });
        }
    }

    let output = if args.csv {
        format_csv(&results, args.compare)
    } else {
        format_table(&results, args.compare)
    };

    if let Some(path) = args.output {
        StdFs::write(&path, &output).context("Failed to write output file")?;
        eprintln!("Written to {}", path.display());
    } else {
        print!("{output}");
    }

    Ok(())
}

/// Extract benchmark group names from a benchmark source file.
///
/// Looks for patterns like:
/// - `benchmark_group("group_name")`
/// - `c.benchmark_group("group_name")`
/// - `group.bench_function(BenchmarkId::new("func", ...)` (extracts parent group)
fn extract_benchmark_groups(bench_name: &str) -> Result<HashSet<String>> {
    let bench_path = PathBuf::from("benches").join(format!("{bench_name}.rs"));

    if !bench_path.exists() {
        bail!("Benchmark file not found: {}", bench_path.display());
    }

    let content = StdFs::read_to_string(&bench_path)
        .with_context(|| format!("Failed to read {}", bench_path.display()))?;

    let mut groups = HashSet::new();

    // Pattern 1: benchmark_group("name") or c.benchmark_group("name")
    let group_re = Regex::new(r#"\.?benchmark_group\s*\(\s*"([^"]+)"\s*\)"#).expect("valid regex");
    for cap in group_re.captures_iter(&content) {
        if let Some(name) = cap.get(1) {
            groups.insert(name.as_str().to_string());
        }
    }

    // Pattern 2: c.bench_function("name", ...) - standalone benchmarks
    let bench_fn_re = Regex::new(r#"\.bench_function\s*\(\s*"([^"]+)"\s*,"#).expect("valid regex");

    for cap in bench_fn_re.captures_iter(&content) {
        if let Some(name) = cap.get(1) {
            groups.insert(name.as_str().to_string());
        }
    }

    Ok(groups)
}

/// Recursively collect benchmark results from estimates.json files
#[expect(clippy::type_complexity)]
fn collect_results(
    dir: &Path,
    filter: Option<&Regex>,
    include_base: bool,
) -> Result<Vec<BenchmarkResult>> {
    let mut new_results: BTreeMap<String, (f64, f64, f64, Option<f64>)> = BTreeMap::new();
    let mut base_results: BTreeMap<String, (f64, f64)> = BTreeMap::new();

    // Walk the criterion directory - collect new and base separately
    walk_criterion_dir(dir, &mut |bench_name, estimates_path, is_new| {
        // Apply filter
        if let Some(re) = filter
            && !re.is_match(&bench_name)
        {
            return Ok(());
        }

        // Parse estimates.json
        let content = StdFs::read_to_string(estimates_path)
            .with_context(|| format!("Failed to read {}", estimates_path.display()))?;
        let estimates: Estimates = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse {}", estimates_path.display()))?;

        if is_new {
            new_results.insert(
                bench_name,
                (
                    estimates.mean.point_estimate,
                    estimates.median.point_estimate,
                    estimates.std_dev.point_estimate,
                    estimates.slope.map(|s| s.point_estimate),
                ),
            );
        } else if include_base {
            base_results.insert(
                bench_name,
                (
                    estimates.mean.point_estimate,
                    estimates.median.point_estimate,
                ),
            );
        }

        Ok(())
    })?;

    // Merge new and base results
    let results = new_results
        .into_iter()
        .map(|(name, (mean_ns, median_ns, std_dev_ns, slope_ns))| {
            let (base_mean_ns, base_median_ns) = if include_base {
                base_results
                    .get(&name)
                    .map_or((None, None), |&(m, md)| (Some(m), Some(md)))
            } else {
                (None, None)
            };

            BenchmarkResult {
                name,
                mean_ns,
                median_ns,
                std_dev_ns,
                slope_ns,
                base_mean_ns,
                base_median_ns,
            }
        })
        .collect();

    Ok(results)
}

/// Walk criterion directory structure and call handler for each estimates.json
fn walk_criterion_dir<F>(dir: &Path, handler: &mut F) -> Result<()>
where
    F: FnMut(String, &Path, bool) -> Result<()>,
{
    if !dir.exists() {
        anyhow::bail!("Criterion directory not found: {}", dir.display());
    }

    // Pattern: dir/<group>/<function>/<param>?/(new|base)/estimates.json
    // or:      dir/<group_function>/(new|base)/estimates.json
    walk_dir_recursive(dir, &[], handler)
}

fn walk_dir_recursive<F>(dir: &Path, path_parts: &[&str], handler: &mut F) -> Result<()>
where
    F: FnMut(String, &Path, bool) -> Result<()>,
{
    let entries =
        StdFs::read_dir(dir).with_context(|| format!("Failed to read {}", dir.display()))?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();

        if path.is_dir() {
            // Check if this is a "new" or "base" directory with estimates.json
            if name == "new" || name == "base" {
                let estimates_path = path.join("estimates.json");
                if estimates_path.exists() {
                    let bench_name = path_parts.join("/");
                    let is_new = name == "new";
                    handler(bench_name, &estimates_path, is_new)?;
                }
            } else if name != "report" {
                // Skip "report" directories, recurse into others
                let mut new_parts = path_parts.to_vec();
                new_parts.push(&name);
                walk_dir_recursive(&path, &new_parts, handler)?;
            }
        }
    }

    Ok(())
}

/// Format duration in appropriate units
fn format_duration(ns: f64) -> String {
    if ns >= 1_000_000_000.0 {
        format!("{:.2} s", ns / 1_000_000_000.0)
    } else if ns >= 1_000_000.0 {
        format!("{:.2} ms", ns / 1_000_000.0)
    } else if ns >= 1_000.0 {
        format!("{:.2} µs", ns / 1_000.0)
    } else {
        format!("{ns:.2} ns")
    }
}

/// Format results as a table
fn format_table(results: &[BenchmarkResult], show_comparison: bool) -> String {
    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED);

    // Header
    let mut header = vec![
        Cell::new("Benchmark").fg(Color::Cyan),
        Cell::new("Mean").fg(Color::Cyan),
        Cell::new("Median").fg(Color::Cyan),
        Cell::new("Std Dev").fg(Color::Cyan),
    ];
    if show_comparison {
        header.push(Cell::new("Base Median").fg(Color::Cyan));
        header.push(Cell::new("Δ%").fg(Color::Yellow));
    }
    table.set_header(header);

    // Rows
    for result in results {
        let mut row = vec![
            Cell::new(&result.name),
            Cell::new(result.mean_formatted()),
            Cell::new(result.median_formatted()),
            Cell::new(result.std_dev_formatted()),
        ];

        if show_comparison {
            let base_str = result
                .base_median_ns
                .map_or_else(|| "-".to_string(), format_duration);

            row.push(Cell::new(base_str));

            let delta_cell = result.delta_percent().map_or_else(
                || Cell::new("-"),
                |delta| {
                    let text = format!("{delta:+.1}%");

                    if delta < -2.0 {
                        Cell::new(text).fg(Color::Green) // Faster
                    } else if delta > 2.0 {
                        Cell::new(text).fg(Color::Red) // Slower
                    } else {
                        Cell::new(text) // Within noise
                    }
                },
            );
            row.push(delta_cell);
        }

        table.add_row(row);
    }

    format!("{table}\n")
}

/// Format results as CSV
fn format_csv(results: &[BenchmarkResult], show_comparison: bool) -> String {
    let mut output = String::new();

    // Header
    if show_comparison {
        output.push_str("benchmark,mean_ns,median_ns,std_dev_ns,base_median_ns,delta_percent\n");
    } else {
        output.push_str("benchmark,mean_ns,median_ns,std_dev_ns\n");
    }

    // Rows
    for result in results {
        // Escape benchmark name if it contains commas
        let name = if result.name.contains(',') {
            format!("\"{}\"", result.name)
        } else {
            result.name.clone()
        };

        if show_comparison {
            let base = result
                .base_median_ns
                .map_or_else(String::new, |v| format!("{v:.2}"));
            let delta = result
                .delta_percent()
                .map_or_else(String::new, |v| format!("{v:.2}"));
            writeln!(
                output,
                "{},{:.2},{:.2},{:.2},{},{}",
                name, result.mean_ns, result.median_ns, result.std_dev_ns, base, delta
            )
            .unwrap();
        } else {
            writeln!(
                output,
                "{},{:.2},{:.2},{:.2}",
                name, result.mean_ns, result.median_ns, result.std_dev_ns
            )
            .unwrap();
        }
    }

    output
}
