//! Parser for divan benchmark output.
//!
//! Usage:
//!   `cargo run --features bins --bin divan_parser -- -f results.txt -c median`
//!   `cargo run --features bins --bin divan_parser -- -f run1.txt --compare run2.txt -c median`
//!   `cargo run --features bins --bin divan_parser -- -f run1.txt --compare run2.txt -c mean --throughput`

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;
use std::sync::LazyLock;

use clap::{Parser, ValueEnum};
use comfy_table::{presets::UTF8_FULL_CONDENSED, Cell, Color, Table};
use regex::Regex;
use serde::Serialize;

// ============================================================================
// Regex Patterns
// ============================================================================

static TIMING_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(\d+\.?\d*)\s*(ms|µs|us|ns|s)\b").unwrap());

static THROUGHPUT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(\d+\.?\d*)\s*([KMG]?item/s)\b").unwrap());

static NORMALIZE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"/masstree\d+[^/]*/").unwrap());

// Match a benchmark data line: extracts indent level and content
// Groups: 1=indent chars, 2=branch marker, 3=rest of line
static LINE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^([│ ]*)(├─|╰─)\s*(.*)$").unwrap());

// ============================================================================
// CLI
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "divan_parser", about = "Parse and compare divan benchmark output")]
struct Args {
    /// Input file (stdin if not provided)
    #[arg(short, long)]
    file: Option<PathBuf>,

    /// Compare against second file
    #[arg(long)]
    compare: Option<PathBuf>,

    /// Columns to display
    #[arg(short, long, value_delimiter = ',', default_value = "median")]
    columns: Vec<Column>,

    /// Output JSON
    #[arg(long)]
    json: bool,

    /// Filter by regex
    #[arg(short = 'F', long)]
    filter: Option<String>,

    /// Fuzzy match names (strip /masstreeN/ segments)
    #[arg(long)]
    fuzzy: bool,

    /// Show throughput instead of timing
    #[arg(long)]
    throughput: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Column {
    Fastest,
    Slowest,
    Median,
    Mean,
    Samples,
    Iters,
}

impl Column {
    const fn name(self) -> &'static str {
        match self {
            Self::Fastest => "fastest",
            Self::Slowest => "slowest",
            Self::Median => "median",
            Self::Mean => "mean",
            Self::Samples => "samples",
            Self::Iters => "iters",
        }
    }
}

// ============================================================================
// Data Types
// ============================================================================

#[derive(Debug, Default, Clone, Serialize)]
struct BenchResult {
    name: String,
    // Timing values
    fastest: Option<String>,
    slowest: Option<String>,
    median: Option<String>,
    mean: Option<String>,
    samples: Option<String>,
    iters: Option<String>,
    // Throughput values
    fastest_tput: Option<String>,
    slowest_tput: Option<String>,
    median_tput: Option<String>,
    mean_tput: Option<String>,
}

impl BenchResult {
    fn get(&self, col: Column, throughput: bool) -> Option<&str> {
        if throughput {
            match col {
                Column::Fastest => self.fastest_tput.as_deref(),
                Column::Slowest => self.slowest_tput.as_deref(),
                Column::Median => self.median_tput.as_deref(),
                Column::Mean => self.mean_tput.as_deref(),
                Column::Samples => self.samples.as_deref(),
                Column::Iters => self.iters.as_deref(),
            }
        } else {
            match col {
                Column::Fastest => self.fastest.as_deref(),
                Column::Slowest => self.slowest.as_deref(),
                Column::Median => self.median.as_deref(),
                Column::Mean => self.mean.as_deref(),
                Column::Samples => self.samples.as_deref(),
                Column::Iters => self.iters.as_deref(),
            }
        }
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse();

    let input = read_input(&args.file)?;
    let results = parse_divan(&input, args.filter.as_deref())?;

    if let Some(ref cmp_path) = args.compare {
        let input2 = fs::read_to_string(cmp_path)
            .map_err(|e| format!("Failed to read {}: {e}", cmp_path.display()))?;
        let results2 = parse_divan(&input2, args.filter.as_deref())?;

        if args.json {
            print_comparison_json(&results, &results2, &args.columns, args.fuzzy, args.throughput)?;
        } else {
            print_comparison_table(&results, &results2, &args.columns, args.fuzzy, args.throughput);
        }
    } else if args.json {
        print_json(&results, &args.columns, args.throughput)?;
    } else {
        print_table(&results, &args.columns, args.throughput);
    }

    Ok(())
}

fn read_input(path: &Option<PathBuf>) -> Result<String, String> {
    match path {
        Some(p) => fs::read_to_string(p).map_err(|e| format!("Failed to read {}: {e}", p.display())),
        None => {
            let mut buf = String::new();
            io::stdin().read_to_string(&mut buf).map_err(|e| format!("stdin: {e}"))?;
            Ok(buf)
        }
    }
}

// ============================================================================
// Parser
// ============================================================================

fn parse_divan(input: &str, filter: Option<&str>) -> Result<Vec<BenchResult>, String> {
    let filter_re = filter.map(Regex::new).transpose().map_err(|e| format!("Bad filter: {e}"))?;

    // Strip markdown code blocks
    let input = input.trim();
    let input = if input.starts_with("```") {
        let start = input.find('\n').map_or(0, |i| i + 1);
        let end = input.rfind("```").unwrap_or(input.len());
        &input[start..end]
    } else {
        input
    };

    let mut results: Vec<BenchResult> = Vec::new();
    let mut path: Vec<String> = Vec::new();
    let mut last_bench_name: Option<String> = None;

    for line in input.lines() {
        // Skip header/empty lines
        if line.trim().is_empty() || line.contains("Timer precision") ||
           (line.contains("fastest") && line.contains("slowest") && line.contains("median")) {
            continue;
        }

        // Try to match a benchmark line with branch marker
        if let Some(caps) = LINE_RE.captures(line) {
            let indent = caps.get(1).map_or("", |m| m.as_str());
            let content = caps.get(3).map_or("", |m| m.as_str());

            // Calculate depth: count │ or use space-based (3 chars per level)
            let depth = {
                let pipes = indent.chars().filter(|&c| c == '│').count();
                if pipes > 0 { pipes } else { indent.chars().filter(|&c| c == ' ').count() / 3 }
            };

            // Split content by │ to separate name from data columns
            let parts: Vec<&str> = content.split('│').collect();
            let first_part = parts.first().map_or("", |s| s.trim());

            // Extract name (remove any trailing timing)
            let name = TIMING_RE.find(first_part)
                .map_or(first_part, |m| first_part[..m.start()].trim());

            if name.is_empty() {
                continue;
            }

            // Update path stack
            path.truncate(depth);
            path.push(name.to_string());
            let full_name = path.join("/");

            // Check if this line has data (columns with values)
            let columns: Vec<&str> = parts.iter().skip(1).map(|s| s.trim()).collect();
            let has_data = columns.iter().any(|c| !c.is_empty());

            if !has_data {
                continue;
            }

            // Apply filter
            if let Some(ref re) = filter_re {
                if !re.is_match(&full_name) {
                    continue;
                }
            }

            // Extract timing from first part (fastest is inline with name)
            let fastest = TIMING_RE.find(first_part).map(|m| m.as_str().to_string());

            let result = BenchResult {
                name: full_name.clone(),
                fastest,
                slowest: columns.first().filter(|s| !s.is_empty()).map(|s| s.to_string()),
                median: columns.get(1).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                mean: columns.get(2).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                samples: columns.get(3).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                iters: columns.get(4).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                ..Default::default()
            };

            last_bench_name = Some(full_name);
            results.push(result);
        }
        // Check for throughput continuation line (no branch marker, has item/s)
        else if THROUGHPUT_RE.is_match(line) {
            if let Some(ref name) = last_bench_name {
                if let Some(result) = results.iter_mut().find(|r| &r.name == name) {
                    let tputs: Vec<Option<String>> = line
                        .split('│')
                        .map(|p| THROUGHPUT_RE.find(p.trim()).map(|m| m.as_str().to_string()))
                        .collect();

                    if let Some(Some(t)) = tputs.first() { result.fastest_tput = Some(t.clone()); }
                    if let Some(Some(t)) = tputs.get(1) { result.slowest_tput = Some(t.clone()); }
                    if let Some(Some(t)) = tputs.get(2) { result.median_tput = Some(t.clone()); }
                    if let Some(Some(t)) = tputs.get(3) { result.mean_tput = Some(t.clone()); }
                }
            }
        }
    }

    Ok(results)
}

// ============================================================================
// Output - Single File
// ============================================================================

fn print_table(results: &[BenchResult], columns: &[Column], throughput: bool) {
    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED);

    let mut header = vec![Cell::new("Benchmark").fg(Color::Cyan)];
    for col in columns {
        let name = if throughput { format!("{} (tput)", col.name()) } else { col.name().to_string() };
        header.push(Cell::new(name).fg(Color::Cyan));
    }
    table.set_header(header);

    for r in results {
        let mut row = vec![Cell::new(&r.name)];
        for col in columns {
            row.push(Cell::new(r.get(*col, throughput).unwrap_or("-")));
        }
        table.add_row(row);
    }

    println!("{table}");
}

fn print_json(results: &[BenchResult], columns: &[Column], throughput: bool) -> Result<(), String> {
    let data: Vec<_> = results.iter().map(|r| {
        let mut m = HashMap::new();
        m.insert("name".to_string(), r.name.clone());
        for col in columns {
            if let Some(v) = r.get(*col, throughput) {
                m.insert(col.name().to_string(), v.to_string());
            }
        }
        m
    }).collect();

    println!("{}", serde_json::to_string_pretty(&data).map_err(|e| e.to_string())?);
    Ok(())
}

// ============================================================================
// Output - Comparison
// ============================================================================

fn normalize_name(name: &str) -> String {
    NORMALIZE_RE.replace_all(name, "/").to_string()
}

fn parse_value(s: &str, is_throughput: bool) -> Option<f64> {
    let s = s.trim();
    let (num, unit) = s.split_once(' ')?;
    let n: f64 = num.parse().ok()?;

    if is_throughput {
        Some(match unit {
            "item/s" => n,
            "Kitem/s" => n * 1e3,
            "Mitem/s" => n * 1e6,
            "Gitem/s" => n * 1e9,
            _ => return None,
        })
    } else {
        Some(match unit {
            "s" => n * 1e6,
            "ms" => n * 1e3,
            "µs" | "us" => n,
            "ns" => n / 1e3,
            _ => return None,
        })
    }
}

/// Calculate delta percentage. For timing: +% = regression. For throughput: +% = regression (lower).
fn calc_delta(old: &str, new: &str, is_throughput: bool) -> Option<f64> {
    let old_v = parse_value(old, is_throughput)?;
    let new_v = parse_value(new, is_throughput)?;
    if old_v == 0.0 { return None; }

    let delta = (new_v - old_v) / old_v * 100.0;
    // For throughput, higher is better, so negate to make +% = regression
    Some(if is_throughput { -delta } else { delta })
}

fn print_comparison_table(
    results_a: &[BenchResult],
    results_b: &[BenchResult],
    columns: &[Column],
    fuzzy: bool,
    throughput: bool,
) {
    let b_map: HashMap<String, &BenchResult> = results_b.iter()
        .map(|r| (if fuzzy { normalize_name(&r.name) } else { r.name.clone() }, r))
        .collect();

    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED);

    let mut header = vec![Cell::new("Benchmark").fg(Color::Cyan)];
    for col in columns {
        header.push(Cell::new(format!("{}(A)", col.name())).fg(Color::Blue));
        header.push(Cell::new(format!("{}(B)", col.name())).fg(Color::Magenta));
        header.push(Cell::new("Δ(B-A)%").fg(Color::Yellow));
    }
    table.set_header(header);

    for ra in results_a {
        let key = if fuzzy { normalize_name(&ra.name) } else { ra.name.clone() };
        let Some(rb) = b_map.get(&key) else { continue };

        // Skip if no matching data
        let has_match = columns.iter().any(|c| ra.get(*c, throughput).is_some() && rb.get(*c, throughput).is_some());
        if !has_match { continue; }

        let mut row = vec![Cell::new(&ra.name)];
        for col in columns {
            let va = ra.get(*col, throughput);
            let vb = rb.get(*col, throughput);

            row.push(Cell::new(va.unwrap_or("-")));
            row.push(Cell::new(vb.unwrap_or("-")));

            let delta_cell = match (va, vb) {
                (Some(a), Some(b)) => {
                    let is_tput = throughput && matches!(col, Column::Fastest | Column::Slowest | Column::Median | Column::Mean);
                    match calc_delta(a, b, is_tput) {
                        Some(d) if d < -2.0 => Cell::new(format!("{d:+.1}%")).fg(Color::Green),
                        Some(d) if d > 2.0 => Cell::new(format!("{d:+.1}%")).fg(Color::Red),
                        Some(d) => Cell::new(format!("{d:+.1}%")),
                        None => Cell::new("-"),
                    }
                }
                _ => Cell::new("-"),
            };
            row.push(delta_cell);
        }

        table.add_row(row);
    }

    println!("{table}");
}

fn print_comparison_json(
    results_a: &[BenchResult],
    results_b: &[BenchResult],
    columns: &[Column],
    fuzzy: bool,
    throughput: bool,
) -> Result<(), String> {
    let b_map: HashMap<String, &BenchResult> = results_b.iter()
        .map(|r| (if fuzzy { normalize_name(&r.name) } else { r.name.clone() }, r))
        .collect();

    let mut data = Vec::new();
    for ra in results_a {
        let key = if fuzzy { normalize_name(&ra.name) } else { ra.name.clone() };
        let Some(rb) = b_map.get(&key) else { continue };

        let has_match = columns.iter().any(|c| ra.get(*c, throughput).is_some() && rb.get(*c, throughput).is_some());
        if !has_match { continue; }

        let mut entry = serde_json::Map::new();
        entry.insert("name".to_string(), serde_json::Value::String(ra.name.clone()));

        for col in columns {
            let va = ra.get(*col, throughput);
            let vb = rb.get(*col, throughput);

            entry.insert(format!("{}_a", col.name()), serde_json::Value::String(va.unwrap_or("-").to_string()));
            entry.insert(format!("{}_b", col.name()), serde_json::Value::String(vb.unwrap_or("-").to_string()));

            if let (Some(a), Some(b)) = (va, vb) {
                let is_tput = throughput && matches!(col, Column::Fastest | Column::Slowest | Column::Median | Column::Mean);
                if let Some(d) = calc_delta(a, b, is_tput) {
                    entry.insert(format!("{}_delta", col.name()), serde_json::json!(d));
                }
            }
        }

        data.push(serde_json::Value::Object(entry));
    }

    println!("{}", serde_json::to_string_pretty(&data).map_err(|e| e.to_string())?);
    Ok(())
}
