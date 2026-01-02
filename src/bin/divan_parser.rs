//! Parser for divan benchmark output.
//!
//! Usage:
//!   `cargo bench 2>&1 | cargo run --bin divan_parser -- --columns median`
//!   `cargo run --bin divan_parser -- -f results.txt -c median,mean`
//!   `cargo run --bin divan_parser -- -f results.txt -c median --json`
//!   `cargo run --bin divan_parser -- -f run1.md --compare run2.md -c median  # Side-by-side comparison`

#![expect(clippy::indexing_slicing, reason = "fail fast")]

use std::collections::HashMap;
use std::fs as StdFs;
use std::io::{self, Read};
use std::path::PathBuf;
use std::string::ToString;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use comfy_table::{Cell, Color, Table, presets::UTF8_FULL_CONDENSED};
use nom::{
    IResult, Parser as NomParser,
    branch::alt,
    bytes::complete::tag,
    character::complete::{char, space0},
    multi::many0,
};
use regex::Regex;
use serde::Serialize;
use serde_json as SJSON;

/// Parse divan benchmark output and display selected columns.
#[derive(Parser, Debug)]
#[command(name = "divan_parser")]
#[command(about = "Parse divan benchmark output and display selected columns")]
struct Args {
    /// Input file (reads from stdin if not provided)
    #[arg(short, long)]
    file: Option<PathBuf>,

    /// Second file for side-by-side comparison
    #[arg(long)]
    compare: Option<PathBuf>,

    /// Columns to display (comma-separated)
    #[arg(short, long, value_delimiter = ',', default_value = "median")]
    columns: Vec<Column>,

    /// Output as JSON instead of table
    #[arg(long)]
    json: bool,

    /// Filter benchmarks by name pattern
    #[arg(short = 'F', long)]
    filter: Option<String>,

    /// Fuzzy match benchmark names (strips /masstree24/ segments for comparison)
    #[arg(long)]
    fuzzy: bool,
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
    const fn header(self) -> &'static str {
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

#[derive(Debug, Serialize)]
struct BenchResult {
    name: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    fastest: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    slowest: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    median: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    mean: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    samples: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    iters: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let input = if let Some(path) = &args.file {
        StdFs::read_to_string(path).context("Failed to read input file")?
    } else {
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .context("Failed to read stdin")?;
        buf
    };

    // Strip markdown code block if present (.md files)
    let input = strip_markdown_code_block(&input);
    let results = parse_divan_output(&input, args.filter.as_deref())?;

    // Check for comparison mode
    if let Some(compare_path) = &args.compare {
        let input2 = StdFs::read_to_string(compare_path).context("Failed to read compare file")?;
        let input2 = strip_markdown_code_block(&input2);
        let results2 = parse_divan_output(&input2, args.filter.as_deref())?;

        if args.json {
            output_comparison_json(&results, &results2, &args.columns, args.fuzzy)?;
        } else {
            output_comparison_table(&results, &results2, &args.columns, args.fuzzy);
        }
    } else if args.json {
        output_json(&results, &args.columns)?;
    } else {
        output_table(&results, &args.columns);
    }

    Ok(())
}

/// Strip markdown code block wrapper if present (```bash or ```text).
fn strip_markdown_code_block(input: &str) -> String {
    let input = input.trim();

    if input.starts_with("```bash") || input.starts_with("```text") || input.starts_with("```") {
        let start = input.find('\n').map_or(0, |i| i + 1);
        let end = input.rfind("```").unwrap_or(input.len());
        input[start..end].to_string()
    } else {
        input.to_string()
    }
}

// ============================================================================
// Nom Parsers
// ============================================================================

/// Parse a tree indent character (│ followed by spaces)
fn tree_indent(input: &str) -> IResult<&str, ()> {
    let (input, _) = char('│')(input)?;
    let (input, _) = space0(input)?;
    Ok((input, ()))
}

/// Parse multiple tree indents, returning the depth
fn tree_depth(input: &str) -> IResult<&str, usize> {
    let (input, indents) = many0(tree_indent).parse(input)?;
    Ok((input, indents.len()))
}

/// Parse a branch marker (├─ or ╰─)
fn branch_marker(input: &str) -> IResult<&str, &str> {
    alt((tag("├─ "), tag("╰─ "), tag("├─"), tag("╰─"))).parse(input)
}

/// Parsed line data
#[derive(Debug)]
struct ParsedLine<'a> {
    depth: usize,
    name: &'a str,
    columns: Vec<&'a str>,
}

/// Parse a complete benchmark line
fn parse_line(input: &str) -> IResult<&str, ParsedLine<'_>> {
    // Parse tree depth (count of │ before branch marker)
    let (input, depth) = tree_depth(input)?;

    // Parse branch marker
    let (input, _) = branch_marker(input)?;

    // Now we need to find where the name ends and columns begin
    // Split by │ to get columns
    let parts: Vec<&str> = input.split('│').collect();

    if parts.is_empty() {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::SeparatedList,
        )));
    }

    // First part contains: name + optional fastest timing
    let first_part = parts[0].trim();

    // Extract name by removing trailing timing pattern
    let name = extract_name_simple(first_part);

    // Collect remaining columns
    let columns: Vec<&str> = parts[1..].iter().map(|s| s.trim()).collect();

    Ok((
        "",
        ParsedLine {
            depth,
            name,
            columns,
        },
    ))
}

/// Simple name extraction - remove trailing timing pattern
fn extract_name_simple(input: &str) -> &str {
    // Find where the name ends by looking for timing pattern from the end
    // Pattern: spaces + digits + optional decimal + spaces + unit
    let bytes = input.as_bytes();
    let mut i = bytes.len();

    // Skip trailing whitespace
    while i > 0 && bytes[i - 1].is_ascii_whitespace() {
        i -= 1;
    }

    // Check if we have a unit (letters at the end)
    let unit_end = i;
    while i > 0
        && (bytes[i - 1].is_ascii_alphabetic() || bytes[i - 1] == b'/' || input[..i].ends_with('µ'))
    {
        i -= 1;
        // Handle µ which is multi-byte
        if i > 0 && bytes[i - 1] == 0xC2 {
            i -= 1;
        }
    }

    if i == unit_end {
        // No unit found, return as-is
        return input.trim();
    }

    // Skip whitespace before unit
    while i > 0 && bytes[i - 1].is_ascii_whitespace() {
        i -= 1;
    }

    // Check for number (digits and optional dot)
    let mut found_digit = false;
    while i > 0 && (bytes[i - 1].is_ascii_digit() || bytes[i - 1] == b'.') {
        found_digit = true;
        i -= 1;
    }

    if !found_digit {
        // No number found, return as-is
        return input.trim();
    }

    // Return everything before the timing
    input[..i].trim()
}

// ============================================================================
// Main Parsing Logic
// ============================================================================

fn parse_divan_output(input: &str, filter: Option<&str>) -> Result<Vec<BenchResult>> {
    let mut results = Vec::new();
    let mut path_stack: Vec<String> = Vec::new();

    let filter_re = filter.map(Regex::new).transpose()?;

    for line in input.lines() {
        // Skip empty, header, and throughput-only lines
        if line.trim().is_empty()
            || line.contains("Timer precision")
            || (line.contains("fastest") && line.contains("median"))
        {
            continue;
        }

        // Must have a branch marker
        if !line.contains('├') && !line.contains('╰') {
            continue;
        }

        // Parse the line
        let Ok((_, parsed)) = parse_line(line) else {
            continue;
        };

        if parsed.name.is_empty() {
            continue;
        }

        // Update path stack
        path_stack.truncate(parsed.depth);
        path_stack.push(parsed.name.to_string());

        // Check if line has timing data (non-empty columns)
        let has_data = parsed.columns.iter().any(|c| !c.is_empty());
        if !has_data {
            continue;
        }

        let full_name = path_stack.join("/");

        // Apply filter
        if let Some(ref re) = filter_re
            && !re.is_match(&full_name)
        {
            continue;
        }

        // Build result
        // Columns: [slowest, median, mean, samples, iters]
        // Fastest is in the part containing the branch marker (├ or ╰)
        let fastest = {
            let parts: Vec<&str> = line.split('│').collect();
            // Find the part containing the benchmark name (has ├ or ╰)
            parts
                .iter()
                .find(|p| p.contains('├') || p.contains('╰'))
                .and_then(|p| extract_timing(p))
        };

        let result = BenchResult {
            name: full_name,
            fastest,
            slowest: parsed
                .columns
                .first()
                .filter(|s| !s.is_empty())
                .map(ToString::to_string),
            median: parsed
                .columns
                .get(1)
                .filter(|s| !s.is_empty())
                .map(ToString::to_string),
            mean: parsed
                .columns
                .get(2)
                .filter(|s| !s.is_empty())
                .map(ToString::to_string),
            samples: parsed
                .columns
                .get(3)
                .filter(|s| !s.is_empty())
                .map(ToString::to_string),
            iters: parsed
                .columns
                .get(4)
                .filter(|s| !s.is_empty())
                .map(ToString::to_string),
        };

        results.push(result);
    }

    Ok(results)
}

/// Extract timing value from a string (e.g., "7.406 ms" from "├─ 1    7.406 ms")
fn extract_timing(input: &str) -> Option<String> {
    // Match timing patterns: "7.406 ms", "973.7 µs", "17.21 ns"
    let re = Regex::new(r"(\d+\.?\d*)\s*(ms|µs|us|ns|s)").ok()?;
    re.find(input).map(|m| m.as_str().to_string())
}

// ============================================================================
// Output Functions
// ============================================================================

fn output_table(results: &[BenchResult], columns: &[Column]) {
    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED);

    // Build header
    let mut header = vec![Cell::new("Benchmark").fg(Color::Cyan)];
    for col in columns {
        header.push(Cell::new(col.header()).fg(Color::Cyan));
    }
    table.set_header(header);

    // Add rows
    for result in results {
        let mut row = vec![Cell::new(&result.name)];

        for col in columns {
            let value = match col {
                Column::Fastest => result.fastest.as_deref(),

                Column::Slowest => result.slowest.as_deref(),

                Column::Median => result.median.as_deref(),

                Column::Mean => result.mean.as_deref(),

                Column::Samples => result.samples.as_deref(),

                Column::Iters => result.iters.as_deref(),
            };

            row.push(Cell::new(value.unwrap_or("-")));
        }

        table.add_row(row);
    }

    println!("{table}");
}

fn output_json(results: &[BenchResult], columns: &[Column]) -> Result<()> {
    #[derive(Serialize)]
    struct FilteredResult {
        name: String,

        #[serde(flatten)]
        values: HashMap<String, String>,
    }

    let filtered: Vec<FilteredResult> = results
        .iter()
        .map(|r| {
            let mut values = HashMap::new();

            for col in columns {
                let value = match col {
                    Column::Fastest => r.fastest.as_deref(),

                    Column::Slowest => r.slowest.as_deref(),

                    Column::Median => r.median.as_deref(),

                    Column::Mean => r.mean.as_deref(),

                    Column::Samples => r.samples.as_deref(),

                    Column::Iters => r.iters.as_deref(),
                };

                if let Some(v) = value {
                    values.insert(col.header().to_string(), v.to_string());
                }
            }
            FilteredResult {
                name: r.name.clone(),
                values,
            }
        })
        .collect();

    println!("{}", SJSON::to_string_pretty(&filtered)?);
    Ok(())
}

// ============================================================================
// Comparison Output Functions
// ============================================================================

/// Normalize benchmark name for fuzzy matching.
/// Strips intermediate segments like `/masstree24/` or `/masstree24_inline/`.
#[expect(clippy::unwrap_used, reason = "Valid Regex Pattern: I checked")]
fn normalize_name(name: &str) -> String {
    // Pattern: remove /masstree24/ or /masstree24_something/ from middle of path
    let re = Regex::new(r"/masstree24[^/]*/").unwrap();
    re.replace_all(name, "/").to_string()
}

fn get_column_value(result: &BenchResult, col: Column) -> Option<&str> {
    match col {
        Column::Fastest => result.fastest.as_deref(),

        Column::Slowest => result.slowest.as_deref(),

        Column::Median => result.median.as_deref(),

        Column::Mean => result.mean.as_deref(),

        Column::Samples => result.samples.as_deref(),

        Column::Iters => result.iters.as_deref(),
    }
}

/// Parse timing string to microseconds for comparison
fn parse_timing_us(s: &str) -> Option<f64> {
    let s = s.trim();
    let (num_str, unit) = s.split_once(' ')?;
    let num: f64 = num_str.parse().ok()?;

    Some(match unit {
        "s" => num * 1_000_000.0,

        "ms" => num * 1_000.0,

        "µs" | "us" => num,

        "ns" => num / 1_000.0,

        _ => return None,
    })
}

/// Calculate percentage change: (new - old) / old * 100
fn calc_delta(old: &str, new: &str) -> Option<f64> {
    let old_us = parse_timing_us(old)?;
    let new_us = parse_timing_us(new)?;

    if old_us == 0.0 {
        return None;
    }

    Some((new_us - old_us) / old_us * 100.0)
}

#[expect(clippy::similar_names)]
fn output_comparison_table(
    results_a: &[BenchResult],
    results_b: &[BenchResult],
    columns: &[Column],
    fuzzy: bool,
) {
    // Build lookup map for results_b
    let b_map: HashMap<String, &BenchResult> = results_b
        .iter()
        .map(|r| {
            let key = if fuzzy {
                normalize_name(&r.name)
            } else {
                r.name.clone()
            };
            (key, r)
        })
        .collect();

    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED);

    // Build header: Benchmark | col(A) | col(B) | Δ% | ...
    let mut header = vec![Cell::new("Benchmark").fg(Color::Cyan)];
    for col in columns {
        header.push(Cell::new(format!("{}(A)", col.header())).fg(Color::Blue));
        header.push(Cell::new(format!("{}(B)", col.header())).fg(Color::Magenta));
        header.push(Cell::new("Δ%").fg(Color::Yellow));
    }
    table.set_header(header);

    // Add rows - only include benchmarks present in both files
    for result_a in results_a {
        let lookup_key = if fuzzy {
            normalize_name(&result_a.name)
        } else {
            result_a.name.clone()
        };
        let Some(result_b) = b_map.get(&lookup_key) else {
            continue; // Skip if not in B
        };

        // Check that at least one column has data in both
        let has_data = columns.iter().any(|col| {
            get_column_value(result_a, *col).is_some() && get_column_value(result_b, *col).is_some()
        });
        if !has_data {
            continue;
        }

        let mut row = vec![Cell::new(&result_a.name)];

        for col in columns {
            let val_a = get_column_value(result_a, *col);
            let val_b = get_column_value(result_b, *col);

            row.push(Cell::new(val_a.unwrap_or("-")));
            row.push(Cell::new(val_b.unwrap_or("-")));

            // Calculate delta
            let delta_cell = match (val_a, val_b) {
                (Some(a), Some(b)) => {
                    calc_delta(a, b).map_or_else(
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
                    )
                }

                _ => Cell::new("-"),
            };

            row.push(delta_cell);
        }

        table.add_row(row);
    }

    println!("{table}");
}

fn output_comparison_json(
    results_a: &[BenchResult],
    results_b: &[BenchResult],
    columns: &[Column],
    fuzzy: bool,
) -> Result<()> {
    use std::collections::HashMap;

    #[derive(Serialize)]
    struct ComparisonResult {
        name: String,

        #[serde(flatten)]
        values: HashMap<String, SJSON::Value>,
    }

    let b_map: HashMap<String, &BenchResult> = results_b
        .iter()
        .map(|r| {
            let key = if fuzzy {
                normalize_name(&r.name)
            } else {
                r.name.clone()
            };
            (key, r)
        })
        .collect();

    let compared: Vec<ComparisonResult> = results_a
        .iter()
        .filter_map(|r_a| {
            let lookup_key = if fuzzy {
                normalize_name(&r_a.name)
            } else {
                r_a.name.clone()
            };
            let r_b = b_map.get(&lookup_key)?;

            // Check that at least one column has data in both
            let has_data = columns.iter().any(|col| {
                get_column_value(r_a, *col).is_some() && get_column_value(r_b, *col).is_some()
            });
            if !has_data {
                return None;
            }

            let mut values = HashMap::new();

            for col in columns {
                let val_a = get_column_value(r_a, *col);
                let val_b = get_column_value(r_b, *col);

                values.insert(
                    format!("{}_a", col.header()),
                    SJSON::Value::String(val_a.unwrap_or("-").to_string()),
                );

                values.insert(
                    format!("{}_b", col.header()),
                    SJSON::Value::String(val_b.unwrap_or("-").to_string()),
                );

                if let (Some(a), Some(b)) = (val_a, val_b)
                    && let Some(delta) = calc_delta(a, b)
                {
                    values.insert(
                        format!("{}_delta", col.header()),
                        SJSON::Value::Number(
                            SJSON::Number::from_f64(delta).unwrap_or_else(|| 0.into()),
                        ),
                    );
                }
            }

            Some(ComparisonResult {
                name: r_a.name.clone(),
                values,
            })
        })
        .collect();

    println!("{}", SJSON::to_string_pretty(&compared)?);
    Ok(())
}
