//! Analyze stress test run logs to find patterns in failures.
//!
//! Usage: `cargo run --release --bin analyze_runs [logs/runs]`

use rayon::prelude::*;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as FmtWrite;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let logs_dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("logs/runs"));

    if !logs_dir.exists() {
        eprintln!("Directory not found: {}", logs_dir.display());
        eprintln!("Usage: cargo run --release --bin analyze_runs [logs/runs]");
        std::process::exit(1);
    }

    let mut entries: Vec<_> = fs::read_dir(&logs_dir)
        .expect("Failed to read directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .collect();
    entries.sort_by_key(|e| e.path());

    if entries.is_empty() {
        eprintln!("No .json files found in {}", logs_dir.display());
        std::process::exit(1);
    }

    println!(
        "Analyzing {} log files from {}\n",
        entries.len(),
        logs_dir.display()
    );

    // Parallel analysis of all runs
    let all_runs: Vec<RunAnalysis> = entries
        .par_iter()
        .map(|entry| analyze_run(&entry.path()))
        .collect();

    // Generate markdown report
    let markdown = generate_markdown_report(&all_runs);

    // Write to file
    let output_path = logs_dir.join("AnalyzerSummary.md");
    match File::create(&output_path) {
        Ok(mut file) => {
            if let Err(e) = file.write_all(markdown.as_bytes()) {
                eprintln!("Failed to write report: {}", e);
            } else {
                println!("Report written to: {}", output_path.display());
            }
        }
        Err(e) => eprintln!("Failed to create report file: {}", e),
    }

    // Also print summary to stdout
    print_summary(&all_runs);
}

/// Log entry structure matching our tracing format
#[derive(Debug, Deserialize)]
struct LogEntry {
    #[serde(default)]
    fields: Fields,
    #[serde(rename = "threadId", default)]
    thread_id: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct Fields {
    message: Option<String>,
    ikey: Option<String>,
    retry: Option<u64>,
    leaf_bound: Option<String>,
    leaf_ptr: Option<String>,
    expected_upper: Option<String>,
    expected_lower: Option<String>,
    coverage_gap: Option<u64>,
    target_gap: Option<u64>,
    gap: Option<u64>,
    advance_count: Option<u64>,
    k0: Option<String>,
    k_last: Option<String>,
    child_idx: Option<u64>,
    inode_ptr: Option<String>,
    // SLOW_SPLIT fields
    split_ikey: Option<String>,
    propagate_elapsed_us: Option<u64>,
    is_layer_root: Option<bool>,
    total_elapsed_us: Option<u64>,
    // INTERNODE_SPLIT fields
    sibling_ptr: Option<String>,
    sibling_is_split_locked: Option<bool>,
    popup_key: Option<String>,
    sibling_nkeys: Option<u64>,
    updated_children: Option<u64>,
    updated_internode_children: Option<u64>,
    grandparent_ptr: Option<String>,
    insert_pos: Option<u64>,
    gp_nkeys: Option<u64>,
    sibling_lower: Option<String>,
    sibling_upper: Option<String>,
    new_parent: Option<String>,
    result_ok: Option<bool>,
    grandparent_was_root: Option<bool>,
    grandparent_nkeys: Option<u64>,
    height: Option<u64>,
}

#[derive(Debug, Default)]
struct RunAnalysis {
    name: String,
    total_lines: usize,
    coverage_collapse: EventStats,
    far_landing: EventStats,
    far_landing_insert: EventStats, // FAR_LANDING in insert path (capped at 20 logs)
    blink_limit: EventStats,
    insert_abort: EventStats, // INSERT_ABORT: exceeded max anomaly retries
    internode_descent: EventStats,
    slow_events: SlowEventStats,                // All SLOW_* events
    split_stats: SplitStats,                    // SLOW_SPLIT propagation analysis
    internode_split_stats: InternodeSplitStats, // INTERNODE_SPLIT* events
    // Per-key analysis
    key_events: HashMap<String, KeyStats>,
    // Per-thread analysis
    thread_events: HashMap<String, usize>,
    // Internode routing analysis
    internode_routes: HashMap<String, InternodeStats>,
    // Stuck leaf analysis
    stuck_leaf: StuckLeafAnalysis,
}

#[derive(Debug, Default)]
struct SlowEventStats {
    leaf_lock: usize,
    lock: usize,
    lock_yield: usize,
    parent_lock: usize,
    propagate: usize,
    split: usize,
    stable: usize,
}

/// Analysis of SLOW_SPLIT propagation patterns
#[derive(Debug, Default)]
struct SplitStats {
    total: usize,
    zero_propagate: usize,    // propagate_elapsed_us == 0
    nonzero_propagate: usize, // propagate_elapsed_us > 0
    max_propagate_us: u64,
    splits_near_stuck: Vec<SplitEvent>, // Splits near the stuck leaf bound
}

#[derive(Debug, Clone)]
struct SplitEvent {
    split_ikey: String,
    split_val: u64,
    propagate_us: u64,
    is_layer_root: bool,
}

/// Analysis of INTERNODE_SPLIT* events (Help-Along Protocol tracking)
#[derive(Debug, Default)]
struct InternodeSplitStats {
    /// Total internode sibling creations
    sibling_created: usize,
    /// Siblings created with split-lock (should equal sibling_created after fix)
    sibling_split_locked: usize,
    /// Leaf children parent pointer updates
    leaf_children_updated: usize,
    /// Internode children parent pointer updates (in split_into)
    internode_children_updated: usize,
    /// Sibling installations into grandparent
    grandparent_installs: usize,
    /// New root creations (main tree)
    new_root_created: usize,
    /// Layer root creations
    layer_root_created: usize,
    /// Recursive splits (grandparent full)
    recursive_splits: usize,
    /// Sibling unlocks
    sibling_unlocks: usize,
    /// Unlock failures (result_ok = false)
    unlock_failures: usize,
    /// Sample events for debugging
    samples: Vec<InternodeSplitSample>,
}

#[derive(Debug, Clone)]
#[expect(dead_code, reason = "Fields used for debug output via Debug derive")]
struct InternodeSplitSample {
    event_type: String,
    sibling_ptr: Option<String>,
    popup_key: Option<String>,
    grandparent_ptr: Option<String>,
    result_ok: Option<bool>,
}

/// Analysis of stuck leaf bound concentration
#[derive(Debug, Default)]
struct StuckLeafAnalysis {
    leaf_bound_counts: HashMap<String, usize>,
    max_leaf_bound: Option<String>,
    max_count: usize,
    stuck_ratio: f64,
    severity: Severity,
    onset_line: Option<usize>, // Line where stuck behavior starts
    stuck_ikey_range: Option<(u64, u64)>, // (min, max) ikey values stuck
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
enum Severity {
    #[default]
    Normal,
    Elevated, // >30% stuck ratio
    Critical, // >50% stuck ratio
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Normal => write!(f, "🟢 NORMAL"),
            Severity::Elevated => write!(f, "🟡 ELEVATED"),
            Severity::Critical => write!(f, "🔴 CRITICAL"),
        }
    }
}

#[derive(Debug, Default, Clone)]
struct EventStats {
    count: usize,
    retry_distribution: BTreeMap<u64, usize>,
    samples: Vec<Fields>,
}

#[derive(Debug, Default)]
struct KeyStats {
    coverage_collapse: usize,
    far_landing: usize,
    blink_limit: usize,
    insert_abort: usize,
    max_retry: u64,
    // Retry distribution for this key (retry_level -> count)
    retry_distribution: HashMap<u64, usize>,
    leaf_bounds: HashMap<String, usize>,
    threads: HashMap<String, usize>,
}

impl KeyStats {
    /// Calculate number of distinct operations from retry distribution.
    /// If we see retries 1,2,3,4,5 with counts 100,100,100,100,100, that's 100 operations.
    fn operation_count(&self) -> usize {
        // Operations = events at retry level 1 (or max of any level if no retry 1)
        self.retry_distribution
            .get(&1)
            .copied()
            .unwrap_or_else(|| self.retry_distribution.values().max().copied().unwrap_or(0))
    }

    /// Check if this is a "Sticky Leaf" pattern (>90% of events hit same leaf).
    fn sticky_leaf(&self) -> Option<(&String, f64)> {
        let total: usize = self.leaf_bounds.values().sum();
        if total < 10 {
            return None;
        }
        self.leaf_bounds
            .iter()
            .max_by_key(|(_, count)| *count)
            .filter(|(_, count)| (**count as f64 / total as f64) > 0.9)
            .map(|(lb, count)| (lb, *count as f64 / total as f64))
    }
}

#[derive(Debug, Default)]
struct InternodeStats {
    visits: usize,
    child_indices: HashMap<u64, usize>,
    k0: Option<String>,
    k_last: Option<String>,
}

fn analyze_run(path: &PathBuf) -> RunAnalysis {
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to open {}: {}", path.display(), e);
            return RunAnalysis {
                name,
                ..Default::default()
            };
        }
    };

    let reader = BufReader::with_capacity(1024 * 1024, file); // 1MB buffer
    let mut analysis = RunAnalysis {
        name,
        ..Default::default()
    };

    for line in reader.lines() {
        let Ok(line) = line else { continue };

        analysis.total_lines += 1;

        let entry: LogEntry = match serde_json::from_str(&line) {
            Ok(e) => e,
            Err(_) => continue,
        };

        let msg = match &entry.fields.message {
            Some(m) => m.as_str(),
            None => continue,
        };

        let thread = entry.thread_id.unwrap_or_else(|| "unknown".to_string());
        *analysis.thread_events.entry(thread.clone()).or_insert(0) += 1;

        match msg {
            m if m.contains("COVERAGE_COLLAPSE") => {
                process_event(&mut analysis.coverage_collapse, &entry.fields);
                if let Some(ikey) = &entry.fields.ikey {
                    let ks = analysis.key_events.entry(ikey.clone()).or_default();
                    ks.coverage_collapse += 1;
                    if let Some(r) = entry.fields.retry {
                        ks.max_retry = ks.max_retry.max(r);
                        *ks.retry_distribution.entry(r).or_insert(0) += 1;
                    }
                    if let Some(lb) = &entry.fields.leaf_bound {
                        *ks.leaf_bounds.entry(lb.clone()).or_insert(0) += 1;
                    }
                    *ks.threads.entry(thread).or_insert(0) += 1;
                }
            }
            // FAR_LANDING_RETRY must come before FAR_LANDING (substring match)
            m if m.contains("FAR_LANDING_RETRY") => {
                process_event(&mut analysis.far_landing, &entry.fields);
                if let Some(ikey) = &entry.fields.ikey {
                    let ks = analysis.key_events.entry(ikey.clone()).or_default();
                    ks.far_landing += 1;
                    // FIX: Update max_retry for FAR_LANDING_RETRY too
                    if let Some(r) = entry.fields.retry {
                        ks.max_retry = ks.max_retry.max(r);
                        *ks.retry_distribution.entry(r).or_insert(0) += 1;
                    }
                    if let Some(lb) = &entry.fields.leaf_bound {
                        *ks.leaf_bounds.entry(lb.clone()).or_insert(0) += 1;
                    }
                    *ks.threads.entry(thread).or_insert(0) += 1;
                }
            }
            // FAR_LANDING (insert path, capped at 20 logs in source)
            m if m.contains("FAR_LANDING") => {
                process_event(&mut analysis.far_landing_insert, &entry.fields);
                if let Some(ikey) = &entry.fields.ikey {
                    let ks = analysis.key_events.entry(ikey.clone()).or_default();
                    ks.far_landing += 1;
                    if let Some(lb) = &entry.fields.leaf_bound {
                        *ks.leaf_bounds.entry(lb.clone()).or_insert(0) += 1;
                    }
                    *ks.threads.entry(thread).or_insert(0) += 1;
                }
            }
            m if m.contains("BLINK_LIMIT") => {
                process_event(&mut analysis.blink_limit, &entry.fields);
                if let Some(ikey) = &entry.fields.ikey {
                    let ks = analysis.key_events.entry(ikey.clone()).or_default();
                    ks.blink_limit += 1;
                }
            }
            m if m.contains("INSERT_ABORT") => {
                process_event(&mut analysis.insert_abort, &entry.fields);
                if let Some(ikey) = &entry.fields.ikey {
                    let ks = analysis.key_events.entry(ikey.clone()).or_default();
                    ks.insert_abort += 1;
                }
            }
            m if m.contains("INTERNODE_DESCENT") => {
                analysis.internode_descent.count += 1;
                if let Some(inode_ptr) = &entry.fields.inode_ptr {
                    let is = analysis
                        .internode_routes
                        .entry(inode_ptr.clone())
                        .or_default();
                    is.visits += 1;
                    if let Some(idx) = entry.fields.child_idx {
                        *is.child_indices.entry(idx).or_insert(0) += 1;
                    }
                    if is.k0.is_none() {
                        is.k0.clone_from(&entry.fields.k0);
                        is.k_last.clone_from(&entry.fields.k_last);
                    }
                }
            }
            // SLOW_* events for lock contention analysis
            m if m.contains("SLOW_LEAF_LOCK") => analysis.slow_events.leaf_lock += 1,
            m if m.contains("SLOW_LOCK_YIELD") => analysis.slow_events.lock_yield += 1,
            m if m.contains("SLOW_LOCK") => analysis.slow_events.lock += 1,
            m if m.contains("SLOW_PARENT_LOCK") => analysis.slow_events.parent_lock += 1,
            m if m.contains("SLOW_PROPAGATE") => analysis.slow_events.propagate += 1,
            m if m.contains("SLOW_SPLIT") => {
                analysis.slow_events.split += 1;
                analysis.split_stats.total += 1;

                // Track propagation patterns
                let propagate_us = entry.fields.propagate_elapsed_us.unwrap_or(0);
                if propagate_us == 0 {
                    analysis.split_stats.zero_propagate += 1;
                } else {
                    analysis.split_stats.nonzero_propagate += 1;
                    analysis.split_stats.max_propagate_us =
                        analysis.split_stats.max_propagate_us.max(propagate_us);
                }

                // Track split for later near-stuck analysis
                if let Some(split_ikey) = &entry.fields.split_ikey {
                    if let Ok(val) = u64::from_str_radix(split_ikey, 16) {
                        if analysis.split_stats.splits_near_stuck.len() < 200 {
                            analysis.split_stats.splits_near_stuck.push(SplitEvent {
                                split_ikey: split_ikey.clone(),
                                split_val: val,
                                propagate_us,
                                is_layer_root: entry.fields.is_layer_root.unwrap_or(false),
                            });
                        }
                    }
                }
            }
            m if m.contains("SLOW_STABLE") => analysis.slow_events.stable += 1,
            // INTERNODE_SPLIT* events for Help-Along Protocol tracking
            m if m.contains("INTERNODE_SPLIT: created sibling") => {
                analysis.internode_split_stats.sibling_created += 1;
                if entry.fields.sibling_is_split_locked == Some(true) {
                    analysis.internode_split_stats.sibling_split_locked += 1;
                }
                if analysis.internode_split_stats.samples.len() < 10 {
                    analysis
                        .internode_split_stats
                        .samples
                        .push(InternodeSplitSample {
                            event_type: "created".to_string(),
                            sibling_ptr: entry.fields.sibling_ptr.clone(),
                            popup_key: None,
                            grandparent_ptr: None,
                            result_ok: None,
                        });
                }
            }
            m if m.contains("INTERNODE_SPLIT: updated leaf children") => {
                analysis.internode_split_stats.leaf_children_updated += 1;
            }
            m if m.contains("INTERNODE_SPLIT_INTO: updated internode children") => {
                analysis.internode_split_stats.internode_children_updated += 1;
            }
            m if m.contains("INTERNODE_SPLIT_INSTALL") => {
                analysis.internode_split_stats.grandparent_installs += 1;
                if analysis.internode_split_stats.samples.len() < 10 {
                    analysis
                        .internode_split_stats
                        .samples
                        .push(InternodeSplitSample {
                            event_type: "install".to_string(),
                            sibling_ptr: entry.fields.sibling_ptr.clone(),
                            popup_key: entry.fields.popup_key.clone(),
                            grandparent_ptr: entry.fields.grandparent_ptr.clone(),
                            result_ok: None,
                        });
                }
            }
            m if m.contains("INTERNODE_SPLIT_RECURSIVE") => {
                analysis.internode_split_stats.recursive_splits += 1;
            }
            m if m.contains("INTERNODE_SPLIT_UNLOCK") => {
                analysis.internode_split_stats.sibling_unlocks += 1;
                if entry.fields.result_ok == Some(false) {
                    analysis.internode_split_stats.unlock_failures += 1;
                }
                // Track new_parent type
                if let Some(ref np) = entry.fields.new_parent {
                    if np == "NEW_ROOT" {
                        analysis.internode_split_stats.new_root_created += 1;
                    } else if np == "LAYER_ROOT" {
                        analysis.internode_split_stats.layer_root_created += 1;
                    }
                }
                if analysis.internode_split_stats.samples.len() < 10 {
                    analysis
                        .internode_split_stats
                        .samples
                        .push(InternodeSplitSample {
                            event_type: "unlock".to_string(),
                            sibling_ptr: entry.fields.sibling_ptr.clone(),
                            popup_key: entry.fields.popup_key.clone(),
                            grandparent_ptr: entry.fields.grandparent_ptr.clone(),
                            result_ok: entry.fields.result_ok,
                        });
                }
            }
            _ => {}
        }

        // Track stuck leaf analysis
        if let Some(lb) = &entry.fields.leaf_bound {
            *analysis
                .stuck_leaf
                .leaf_bound_counts
                .entry(lb.clone())
                .or_insert(0) += 1;

            // Track onset line for the eventual max leaf bound
            let count = *analysis.stuck_leaf.leaf_bound_counts.get(lb).unwrap();
            if count == 1 && analysis.stuck_leaf.onset_line.is_none() {
                // We'll refine this after we know which leaf is max
            }

            // Track ikey range for stuck keys
            if let Some(ikey) = &entry.fields.ikey {
                if let Ok(ikey_val) = u64::from_str_radix(ikey, 16) {
                    match &mut analysis.stuck_leaf.stuck_ikey_range {
                        Some((min, max)) => {
                            *min = (*min).min(ikey_val);
                            *max = (*max).max(ikey_val);
                        }
                        None => {
                            analysis.stuck_leaf.stuck_ikey_range = Some((ikey_val, ikey_val));
                        }
                    }
                }
            }
        }
    }

    // Finalize stuck leaf analysis
    finalize_stuck_leaf_analysis(&mut analysis);

    analysis
}

/// Post-process stuck leaf analysis to compute max leaf, ratio, and severity
fn finalize_stuck_leaf_analysis(analysis: &mut RunAnalysis) {
    // Find max stuck leaf bound
    let mut max_lb = None;
    let mut max_count = 0usize;
    for (lb, count) in &analysis.stuck_leaf.leaf_bound_counts {
        if *count > max_count {
            max_count = *count;
            max_lb = Some(lb.clone());
        }
    }

    analysis.stuck_leaf.max_leaf_bound = max_lb.clone();
    analysis.stuck_leaf.max_count = max_count;

    // Calculate ratio
    let total_lb_events: usize = analysis.stuck_leaf.leaf_bound_counts.values().sum();
    analysis.stuck_leaf.stuck_ratio = if total_lb_events > 0 {
        max_count as f64 / total_lb_events as f64
    } else {
        0.0
    };

    // Determine severity
    analysis.stuck_leaf.severity = if analysis.stuck_leaf.stuck_ratio > 0.5 {
        Severity::Critical
    } else if analysis.stuck_leaf.stuck_ratio > 0.3 {
        Severity::Elevated
    } else {
        Severity::Normal
    };

    // Filter splits near stuck leaf for analysis
    if let Some(ref max_lb) = max_lb {
        if let Ok(stuck_val) = u64::from_str_radix(max_lb, 16) {
            // Keep only splits within 200k of the stuck value
            analysis.split_stats.splits_near_stuck.retain(|s| {
                let diff = if s.split_val > stuck_val {
                    s.split_val - stuck_val
                } else {
                    stuck_val - s.split_val
                };
                diff < 200_000
            });
        }
    }
}

fn process_event(stats: &mut EventStats, fields: &Fields) {
    stats.count += 1;
    if let Some(retry) = fields.retry {
        *stats.retry_distribution.entry(retry).or_insert(0) += 1;
    }
    if stats.samples.len() < 3 {
        stats.samples.push(fields.clone());
    }
}

impl Clone for Fields {
    fn clone(&self) -> Self {
        Self {
            message: self.message.clone(),
            ikey: self.ikey.clone(),
            retry: self.retry,
            leaf_bound: self.leaf_bound.clone(),
            leaf_ptr: self.leaf_ptr.clone(),
            expected_upper: self.expected_upper.clone(),
            expected_lower: self.expected_lower.clone(),
            coverage_gap: self.coverage_gap,
            target_gap: self.target_gap,
            gap: self.gap,
            advance_count: self.advance_count,
            k0: self.k0.clone(),
            k_last: self.k_last.clone(),
            child_idx: self.child_idx,
            inode_ptr: self.inode_ptr.clone(),
            split_ikey: self.split_ikey.clone(),
            propagate_elapsed_us: self.propagate_elapsed_us,
            is_layer_root: self.is_layer_root,
            total_elapsed_us: self.total_elapsed_us,
            // INTERNODE_SPLIT fields
            sibling_ptr: self.sibling_ptr.clone(),
            sibling_is_split_locked: self.sibling_is_split_locked,
            popup_key: self.popup_key.clone(),
            sibling_nkeys: self.sibling_nkeys,
            updated_children: self.updated_children,
            updated_internode_children: self.updated_internode_children,
            grandparent_ptr: self.grandparent_ptr.clone(),
            insert_pos: self.insert_pos,
            gp_nkeys: self.gp_nkeys,
            sibling_lower: self.sibling_lower.clone(),
            sibling_upper: self.sibling_upper.clone(),
            new_parent: self.new_parent.clone(),
            result_ok: self.result_ok,
            grandparent_was_root: self.grandparent_was_root,
            grandparent_nkeys: self.grandparent_nkeys,
            height: self.height,
        }
    }
}

/// Generate a structured Markdown report
fn generate_markdown_report(all_runs: &[RunAnalysis]) -> String {
    let mut md = String::with_capacity(64 * 1024);

    // Header
    let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
    writeln!(md, "# Stress Test Analysis Report").unwrap();
    writeln!(md, "\n**Generated:** {}\n", timestamp).unwrap();

    // Executive Summary
    writeln!(md, "## Executive Summary\n").unwrap();

    let total_runs = all_runs.len();
    let critical_runs: Vec<_> = all_runs
        .iter()
        .filter(|r| r.stuck_leaf.severity == Severity::Critical)
        .collect();
    let elevated_runs: Vec<_> = all_runs
        .iter()
        .filter(|r| r.stuck_leaf.severity == Severity::Elevated)
        .collect();
    let aborted_runs: Vec<_> = all_runs
        .iter()
        .filter(|r| r.insert_abort.count > 0)
        .collect();

    let total_cc: usize = all_runs.iter().map(|r| r.coverage_collapse.count).sum();
    let total_fl: usize = all_runs.iter().map(|r| r.far_landing.count).sum();
    let total_aborts: usize = all_runs.iter().map(|r| r.insert_abort.count).sum();

    writeln!(md, "| Metric | Value |").unwrap();
    writeln!(md, "|--------|-------|").unwrap();
    writeln!(md, "| Total Runs | {} |", total_runs).unwrap();
    writeln!(
        md,
        "| Critical Runs | {} ({:.1}%) |",
        critical_runs.len(),
        100.0 * critical_runs.len() as f64 / total_runs as f64
    )
    .unwrap();
    writeln!(
        md,
        "| Elevated Runs | {} ({:.1}%) |",
        elevated_runs.len(),
        100.0 * elevated_runs.len() as f64 / total_runs as f64
    )
    .unwrap();
    writeln!(md, "| Runs with Aborts | {} |", aborted_runs.len()).unwrap();
    writeln!(md, "| Total Coverage Collapse | {} |", total_cc).unwrap();
    writeln!(md, "| Total Far Landing | {} |", total_fl).unwrap();
    writeln!(md, "| Total Insert Aborts | {} |", total_aborts).unwrap();
    writeln!(md).unwrap();

    // Status indicator
    if critical_runs.is_empty() && aborted_runs.is_empty() {
        writeln!(
            md,
            "> ✅ **Status:** All runs completed without critical failures\n"
        )
        .unwrap();
    } else {
        writeln!(
            md,
            "> ⚠️ **Status:** {} critical failures detected\n",
            critical_runs.len() + aborted_runs.len()
        )
        .unwrap();
    }

    // Per-Run Summary Table
    writeln!(md, "## Per-Run Summary\n").unwrap();
    writeln!(
        md,
        "| Run | Lines | CoverageCollapse | FarLanding | Aborts | MaxStuckLeaf | Ratio | Severity |"
    )
    .unwrap();
    writeln!(
        md,
        "|-----|------:|----------------:|-----------:|-------:|-------------:|------:|----------|"
    )
    .unwrap();

    for run in all_runs {
        let max_lb = run.stuck_leaf.max_leaf_bound.as_deref().unwrap_or("N/A");
        let max_lb_short = if max_lb.len() > 12 {
            &max_lb[4..16]
        } else {
            max_lb
        };
        let severity_icon = match run.stuck_leaf.severity {
            Severity::Critical => "🔴",
            Severity::Elevated => "🟡",
            Severity::Normal => "🟢",
        };

        writeln!(
            md,
            "| {} | {} | {} | {} | {} | `{}` | {:.1}% | {} |",
            run.name,
            run.total_lines,
            run.coverage_collapse.count,
            run.far_landing.count,
            run.insert_abort.count,
            max_lb_short,
            run.stuck_leaf.stuck_ratio * 100.0,
            severity_icon
        )
        .unwrap();
    }
    writeln!(md).unwrap();

    // Stuck Leaf Bound Analysis
    writeln!(md, "## Stuck Leaf Bound Analysis\n").unwrap();
    writeln!(
        md,
        "| Run | MaxStuckLeaf | Count | Ratio | StuckIkeyMin | StuckIkeyMax | GapFromLeaf |"
    )
    .unwrap();
    writeln!(
        md,
        "|-----|-------------:|------:|------:|-------------:|-------------:|------------:|"
    )
    .unwrap();

    for run in all_runs {
        let max_lb = run.stuck_leaf.max_leaf_bound.as_deref().unwrap_or("N/A");
        let (min_ikey, max_ikey) = run.stuck_leaf.stuck_ikey_range.unwrap_or((0, 0));
        let leaf_val = u64::from_str_radix(max_lb, 16).unwrap_or(0);
        let gap = if max_ikey > leaf_val {
            max_ikey - leaf_val
        } else {
            0
        };

        writeln!(
            md,
            "| {} | `{:016x}` | {} | {:.1}% | `{:016x}` | `{:016x}` | {} |",
            run.name,
            leaf_val,
            run.stuck_leaf.max_count,
            run.stuck_leaf.stuck_ratio * 100.0,
            min_ikey,
            max_ikey,
            gap
        )
        .unwrap();
    }
    writeln!(md).unwrap();

    // Critical/Elevated Run Details
    let problem_runs: Vec<_> = all_runs
        .iter()
        .filter(|r| r.stuck_leaf.severity != Severity::Normal || r.insert_abort.count > 0)
        .collect();

    if !problem_runs.is_empty() {
        writeln!(md, "## Problem Run Details\n").unwrap();

        for run in problem_runs {
            writeln!(md, "### {} {}\n", run.name, run.stuck_leaf.severity).unwrap();

            writeln!(md, "**Overview:**").unwrap();
            writeln!(md, "- Total log lines: {}", run.total_lines).unwrap();
            writeln!(md, "- Coverage Collapse: {}", run.coverage_collapse.count).unwrap();
            writeln!(md, "- Far Landing: {}", run.far_landing.count).unwrap();
            writeln!(md, "- Insert Aborts: {}", run.insert_abort.count).unwrap();
            writeln!(md).unwrap();

            // Stuck leaf details
            if run.stuck_leaf.severity != Severity::Normal {
                writeln!(md, "**Stuck Leaf Analysis:**").unwrap();
                if let Some(ref lb) = run.stuck_leaf.max_leaf_bound {
                    let leaf_val = u64::from_str_radix(lb, 16).unwrap_or(0);
                    writeln!(md, "- Stuck leaf bound: `{}` ({})", lb, leaf_val).unwrap();
                    writeln!(
                        md,
                        "- Events on this leaf: {} ({:.1}%)",
                        run.stuck_leaf.max_count,
                        run.stuck_leaf.stuck_ratio * 100.0
                    )
                    .unwrap();

                    if let Some((min_ikey, max_ikey)) = run.stuck_leaf.stuck_ikey_range {
                        let min_gap = min_ikey.saturating_sub(leaf_val);
                        let max_gap = max_ikey.saturating_sub(leaf_val);
                        writeln!(md, "- Stuck ikey range: {} to {}", min_ikey, max_ikey).unwrap();
                        writeln!(
                            md,
                            "- Gap from leaf: {} to {} (keys being misrouted)",
                            min_gap, max_gap
                        )
                        .unwrap();
                    }
                }
                writeln!(md).unwrap();
            }

            // Split propagation
            if run.split_stats.total > 0 {
                writeln!(md, "**Split Propagation:**").unwrap();
                writeln!(md, "- Total splits: {}", run.split_stats.total).unwrap();
                writeln!(
                    md,
                    "- Zero propagation: {} ({:.1}%)",
                    run.split_stats.zero_propagate,
                    100.0 * run.split_stats.zero_propagate as f64 / run.split_stats.total as f64
                )
                .unwrap();
                writeln!(
                    md,
                    "- Non-zero propagation: {}",
                    run.split_stats.nonzero_propagate
                )
                .unwrap();
                if run.split_stats.max_propagate_us > 0 {
                    writeln!(
                        md,
                        "- Max propagation time: {}µs",
                        run.split_stats.max_propagate_us
                    )
                    .unwrap();
                }

                if !run.split_stats.splits_near_stuck.is_empty() {
                    writeln!(md, "\n**Splits near stuck leaf:**").unwrap();
                    writeln!(md, "| Split IKey | Value | Propagate µs | Layer Root |").unwrap();
                    writeln!(md, "|------------|------:|-------------:|:----------:|").unwrap();
                    for split in run.split_stats.splits_near_stuck.iter().take(10) {
                        let prop_icon = if split.propagate_us > 0 { "✓" } else { "✗" };
                        writeln!(
                            md,
                            "| `{}` | {} | {} {} | {} |",
                            split.split_ikey,
                            split.split_val,
                            split.propagate_us,
                            prop_icon,
                            if split.is_layer_root { "Yes" } else { "No" }
                        )
                        .unwrap();
                    }
                }
                writeln!(md).unwrap();
            }

            // Top stuck keys
            let mut stuck_keys: Vec<_> = run
                .key_events
                .iter()
                .filter(|(_, v)| v.coverage_collapse > 100 || v.far_landing > 100)
                .collect();
            stuck_keys.sort_by(|a, b| {
                (b.1.coverage_collapse + b.1.far_landing)
                    .cmp(&(a.1.coverage_collapse + a.1.far_landing))
            });

            if !stuck_keys.is_empty() {
                writeln!(md, "**Top Stuck Keys (>100 events):**").unwrap();
                writeln!(md, "| IKey | CC | FL | Aborts | MaxRetry | Pattern |").unwrap();
                writeln!(md, "|------|---:|---:|-------:|--------:|---------|").unwrap();
                for (ikey, stats) in stuck_keys.iter().take(10) {
                    let pattern = if stats.sticky_leaf().is_some() {
                        "🔴 Sticky"
                    } else {
                        "🟡 Churn"
                    };
                    writeln!(
                        md,
                        "| `{}` | {} | {} | {} | {} | {} |",
                        ikey,
                        stats.coverage_collapse,
                        stats.far_landing,
                        stats.insert_abort,
                        stats.max_retry,
                        pattern
                    )
                    .unwrap();
                }
                writeln!(md).unwrap();
            }
        }
    }

    // Cross-Run Analysis
    writeln!(md, "## Cross-Run Analysis\n").unwrap();

    let mut key_appearances: HashMap<String, Vec<String>> = HashMap::new();
    for run in all_runs {
        for (ikey, stats) in &run.key_events {
            if stats.coverage_collapse > 10 || stats.far_landing > 10 {
                key_appearances
                    .entry(ikey.clone())
                    .or_default()
                    .push(run.name.clone());
            }
        }
    }

    let mut multi_run_keys: Vec<_> = key_appearances
        .iter()
        .filter(|(_, runs)| runs.len() > 1)
        .collect();
    multi_run_keys.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    if multi_run_keys.is_empty() {
        writeln!(
            md,
            "No keys stuck in multiple runs (failures are independent).\n"
        )
        .unwrap();
    } else {
        writeln!(md, "**Keys stuck in multiple runs:**\n").unwrap();
        writeln!(md, "| Key | Runs | Run Names |").unwrap();
        writeln!(md, "|-----|-----:|-----------|").unwrap();
        for (key, runs) in multi_run_keys.iter().take(15) {
            writeln!(md, "| `{}` | {} | {} |", key, runs.len(), runs.join(", ")).unwrap();
        }
        writeln!(md).unwrap();
    }

    // Diagnosis
    writeln!(md, "## Diagnosis\n").unwrap();

    if !critical_runs.is_empty() {
        writeln!(md, "### 🔴 Critical: Permanent Routing Inconsistency\n").unwrap();
        writeln!(
            md,
            "**{} run(s)** have >50% of events stuck on a single leaf bound.\n",
            critical_runs.len()
        )
        .unwrap();
        writeln!(
            md,
            "**Root Cause:** Internode keys don't cover child subtree bounds after a split."
        )
        .unwrap();
        writeln!(
            md,
            "This means the parent internode's routing keys weren't updated correctly"
        )
        .unwrap();
        writeln!(
            md,
            "when a leaf split occurred, causing all keys above a certain value to be"
        )
        .unwrap();
        writeln!(md, "misrouted to the wrong leaf.\n").unwrap();
    }

    if !elevated_runs.is_empty() {
        writeln!(md, "### 🟡 Elevated: Significant Routing Issues\n").unwrap();
        writeln!(
            md,
            "**{} run(s)** have 30-50% of events stuck on a single leaf.\n",
            elevated_runs.len()
        )
        .unwrap();
    }

    if total_aborts > 0 {
        writeln!(md, "### ⚠️ Insert Aborts\n").unwrap();
        writeln!(
            md,
            "**{} insert(s)** exceeded maximum retry count and were aborted.\n",
            total_aborts
        )
        .unwrap();
    }

    if critical_runs.is_empty() && elevated_runs.is_empty() && total_aborts == 0 {
        writeln!(
            md,
            "✅ No critical issues detected. Normal transient routing events observed.\n"
        )
        .unwrap();
    }

    md
}

fn print_summary(all_runs: &[RunAnalysis]) {
    println!("{}", "=".repeat(140));
    println!("PER-RUN SUMMARY");
    println!("{}", "=".repeat(140));
    println!(
        "{:<12} {:>10} {:>12} {:>12} {:>10} {:>10} {:>18} {:>7} {:>14}",
        "Run",
        "Lines",
        "Coverage",
        "FarLanding",
        "Aborts",
        "SlowEvts",
        "MaxStuckLeaf",
        "Ratio",
        "Severity"
    );
    println!("{}", "-".repeat(140));

    for run in all_runs {
        let slow_total = run.slow_events.leaf_lock
            + run.slow_events.lock
            + run.slow_events.lock_yield
            + run.slow_events.parent_lock
            + run.slow_events.propagate
            + run.slow_events.split
            + run.slow_events.stable;

        let max_lb = run.stuck_leaf.max_leaf_bound.as_deref().unwrap_or("N/A");
        let max_lb_short = if max_lb.len() > 16 {
            &max_lb[..16]
        } else {
            max_lb
        };

        println!(
            "{:<12} {:>10} {:>12} {:>12} {:>10} {:>10} {:>18} {:>6.1}% {:>14}",
            run.name,
            run.total_lines,
            run.coverage_collapse.count,
            run.far_landing.count,
            run.insert_abort.count,
            slow_total,
            max_lb_short,
            run.stuck_leaf.stuck_ratio * 100.0,
            run.stuck_leaf.severity
        );
    }

    // Print stuck leaf summary table
    println!("\n{}", "=".repeat(140));
    println!("STUCK LEAF BOUND ANALYSIS");
    println!("{}", "=".repeat(140));
    println!(
        "{:<12} {:>18} {:>14} {:>7} {:>20} {:>20} {:>14}",
        "Run", "MaxStuckLeaf", "Count", "Ratio", "StuckIkeyMin", "StuckIkeyMax", "GapFromLeaf"
    );
    println!("{}", "-".repeat(140));

    for run in all_runs {
        let max_lb = run.stuck_leaf.max_leaf_bound.as_deref().unwrap_or("N/A");
        let max_lb_short = if max_lb.len() > 16 {
            &max_lb[..16]
        } else {
            max_lb
        };

        let (min_ikey, max_ikey) = run.stuck_leaf.stuck_ikey_range.unwrap_or((0, 0));
        let leaf_val = u64::from_str_radix(max_lb, 16).unwrap_or(0);
        let gap = if max_ikey > leaf_val {
            max_ikey - leaf_val
        } else {
            0
        };

        println!(
            "{:<12} {:>18} {:>14} {:>6.1}% {:>20} {:>20} {:>14}",
            run.name,
            max_lb_short,
            run.stuck_leaf.max_count,
            run.stuck_leaf.stuck_ratio * 100.0,
            format!("{:>16x}", min_ikey),
            format!("{:>16x}", max_ikey),
            format!("{:>12}", gap)
        );
    }

    // Problem runs detail
    // Include CRITICAL/ELEVATED severity runs
    let problem_runs: Vec<_> = all_runs
        .iter()
        .filter(|r| {
            r.blink_limit.count > 0
                || r.coverage_collapse.count > 1000
                || r.far_landing.count > 10000
                || r.insert_abort.count > 0
                || r.stuck_leaf.severity != Severity::Normal
        })
        .collect();

    if !problem_runs.is_empty() {
        println!("\n{}", "=".repeat(100));
        println!(
            "PROBLEM RUNS DETAIL ({} of {} runs)",
            problem_runs.len(),
            all_runs.len()
        );
        println!("{}", "=".repeat(100));

        for run in problem_runs {
            println!("\n>>> {} <<<", run.name);
            println!("  Total log lines: {}", run.total_lines);

            // Coverage collapse stats
            if run.coverage_collapse.count > 0 {
                println!("\n  COVERAGE_COLLAPSE: {}", run.coverage_collapse.count);
                println!("    Retry distribution:");
                for (retry, count) in &run.coverage_collapse.retry_distribution {
                    println!("      retry {retry}: {count}");
                }
                if let Some(sample) = run.coverage_collapse.samples.first() {
                    println!("    Sample event:");
                    println!("      ikey: {:?}", sample.ikey);
                    println!("      leaf_bound: {:?}", sample.leaf_bound);
                    println!("      expected_upper: {:?}", sample.expected_upper);
                    println!("      coverage_gap: {:?}", sample.coverage_gap);
                    println!("      target_gap: {:?}", sample.target_gap);
                }
            }

            // Far landing stats
            if run.far_landing.count > 0 {
                println!("\n  FAR_LANDING_RETRY: {}", run.far_landing.count);
                println!("    Retry distribution:");
                for (retry, count) in &run.far_landing.retry_distribution {
                    println!("      retry {retry}: {count}");
                }
            }

            // Insert aborts
            if run.insert_abort.count > 0 {
                println!(
                    "\n  INSERT_ABORT: {} (operations that exceeded max retries)",
                    run.insert_abort.count
                );
            }

            // Stuck leaf analysis (CRITICAL/ELEVATED runs)
            if run.stuck_leaf.severity != Severity::Normal {
                println!("\n  📍 STUCK LEAF ANALYSIS: {}", run.stuck_leaf.severity);
                if let Some(ref lb) = run.stuck_leaf.max_leaf_bound {
                    let leaf_val = u64::from_str_radix(lb, 16).unwrap_or(0);
                    println!("    Stuck leaf bound: {} ({} decimal)", lb, leaf_val);
                    println!(
                        "    Events on this leaf: {} ({:.1}% of all events)",
                        run.stuck_leaf.max_count,
                        run.stuck_leaf.stuck_ratio * 100.0
                    );

                    if let Some((min_ikey, max_ikey)) = run.stuck_leaf.stuck_ikey_range {
                        let min_gap = min_ikey.saturating_sub(leaf_val);
                        let max_gap = max_ikey.saturating_sub(leaf_val);
                        println!("    Stuck ikey range: {} to {}", min_ikey, max_ikey);
                        println!(
                            "    Gap from leaf: {} to {} (keys being misrouted)",
                            min_gap, max_gap
                        );
                    }
                }
            }

            // Split propagation analysis
            if run.split_stats.total > 0 {
                println!("\n  🔧 SPLIT PROPAGATION:");
                println!(
                    "    Total splits: {} (zero_prop: {}, nonzero_prop: {})",
                    run.split_stats.total,
                    run.split_stats.zero_propagate,
                    run.split_stats.nonzero_propagate
                );
                if run.split_stats.max_propagate_us > 0 {
                    println!(
                        "    Max propagation time: {}µs",
                        run.split_stats.max_propagate_us
                    );
                }

                // Show splits near stuck leaf if any
                if !run.split_stats.splits_near_stuck.is_empty() {
                    println!(
                        "    Splits near stuck leaf ({} found):",
                        run.split_stats.splits_near_stuck.len()
                    );
                    for split in run.split_stats.splits_near_stuck.iter().take(10) {
                        let prop_marker = if split.propagate_us > 0 { "✓" } else { "✗" };
                        println!(
                            "      {} split_ikey={} ({}) prop={}µs layer_root={}",
                            prop_marker,
                            split.split_ikey,
                            split.split_val,
                            split.propagate_us,
                            split.is_layer_root
                        );
                    }
                }
            }

            // Internode split stats (Help-Along Protocol)
            let is = &run.internode_split_stats;
            if is.sibling_created > 0 {
                println!("\n  🔗 INTERNODE SPLIT (Help-Along Protocol):");
                println!(
                    "    Siblings created: {} (split-locked: {})",
                    is.sibling_created, is.sibling_split_locked
                );
                if is.sibling_created != is.sibling_split_locked {
                    println!(
                        "    ⚠️  WARNING: {} siblings NOT split-locked (BUG!)",
                        is.sibling_created - is.sibling_split_locked
                    );
                }
                println!(
                    "    Parent updates: leaf_children={}, internode_children={}",
                    is.leaf_children_updated, is.internode_children_updated
                );
                println!(
                    "    Installations: grandparent={}, new_root={}, layer_root={}",
                    is.grandparent_installs, is.new_root_created, is.layer_root_created
                );
                println!("    Recursive splits: {}", is.recursive_splits);
                println!(
                    "    Unlocks: {} (failures: {})",
                    is.sibling_unlocks, is.unlock_failures
                );
                if is.unlock_failures > 0 {
                    println!(
                        "    ⚠️  WARNING: {} unlock failures detected!",
                        is.unlock_failures
                    );
                }
            }

            // Slow events summary
            let se = &run.slow_events;
            let slow_total = se.leaf_lock
                + se.lock
                + se.lock_yield
                + se.parent_lock
                + se.propagate
                + se.split
                + se.stable;
            if slow_total > 0 {
                println!("\n  SLOW EVENTS: {slow_total} total");
                if se.leaf_lock > 0 {
                    println!("    SLOW_LEAF_LOCK: {}", se.leaf_lock);
                }
                if se.lock > 0 {
                    println!("    SLOW_LOCK: {}", se.lock);
                }
                if se.lock_yield > 0 {
                    println!("    SLOW_LOCK_YIELD: {}", se.lock_yield);
                }
                if se.parent_lock > 0 {
                    println!("    SLOW_PARENT_LOCK: {}", se.parent_lock);
                }
                if se.propagate > 0 {
                    println!("    SLOW_PROPAGATE: {}", se.propagate);
                }
                if se.split > 0 {
                    println!("    SLOW_SPLIT: {}", se.split);
                }
                if se.stable > 0 {
                    println!("    SLOW_STABLE: {}", se.stable);
                }
            }

            // Top stuck keys - separate Sticky Leaf from Churn patterns
            let mut stuck_keys: Vec<_> = run
                .key_events
                .iter()
                .filter(|(_, v)| v.coverage_collapse > 10 || v.far_landing > 10)
                .collect();
            stuck_keys.sort_by(|a, b| {
                (b.1.coverage_collapse + b.1.far_landing)
                    .cmp(&(a.1.coverage_collapse + a.1.far_landing))
            });

            // Separate sticky leaf keys (routing permanently broken) from churn keys
            let (sticky_keys, churn_keys): (Vec<_>, Vec<_>) = stuck_keys
                .iter()
                .partition(|(_, stats)| stats.sticky_leaf().is_some());

            if !sticky_keys.is_empty() {
                println!("\n  🔴 STICKY LEAF KEYS (routing permanently broken, top 5):");
                for (ikey, stats) in sticky_keys.iter().take(5) {
                    let total = stats.coverage_collapse + stats.far_landing;
                    let ops = stats.operation_count();
                    let (leaf, pct) = stats.sticky_leaf().unwrap();
                    println!(
                        "    {} -> {} events, {} ops (CC:{}, FL:{}, aborts:{}, max_retry:{})",
                        ikey,
                        total,
                        ops,
                        stats.coverage_collapse,
                        stats.far_landing,
                        stats.insert_abort,
                        stats.max_retry
                    );
                    println!(
                        "      STUCK on leaf_bound {} ({:.1}% of events)",
                        leaf,
                        pct * 100.0
                    );
                }
            }

            if !churn_keys.is_empty() {
                println!("\n  🟡 CHURN KEYS (split race / transient, top 5):");
                for (ikey, stats) in churn_keys.iter().take(5) {
                    let total = stats.coverage_collapse + stats.far_landing;
                    let ops = stats.operation_count();
                    println!(
                        "    {} -> {} events, {} ops (CC:{}, FL:{}, aborts:{}, max_retry:{})",
                        ikey,
                        total,
                        ops,
                        stats.coverage_collapse,
                        stats.far_landing,
                        stats.insert_abort,
                        stats.max_retry
                    );
                    println!(
                        "      {} different leaf_bounds (churn pattern)",
                        stats.leaf_bounds.len()
                    );
                }
            }

            // Thread distribution for stuck keys
            let mut thread_totals: HashMap<String, usize> = HashMap::new();
            for stats in run.key_events.values() {
                for (thread, count) in &stats.threads {
                    *thread_totals.entry(thread.clone()).or_insert(0) += count;
                }
            }
            let mut thread_vec: Vec<_> = thread_totals.into_iter().collect();
            thread_vec.sort_by(|a, b| b.1.cmp(&a.1));
            if thread_vec.len() > 1 {
                println!("\n  THREAD DISTRIBUTION (top 5):");
                for (thread, count) in thread_vec.iter().take(5) {
                    println!("    {thread}: {count} events");
                }
            }

            // Internode routing hotspots
            let mut hot_inodes: Vec<_> = run
                .internode_routes
                .iter()
                .filter(|(_, v)| v.visits > 100)
                .collect();
            hot_inodes.sort_by(|a, b| b.1.visits.cmp(&a.1.visits));
            if !hot_inodes.is_empty() {
                println!("\n  HOT INTERNODES (>100 visits, top 5):");
                for (ptr, stats) in hot_inodes.iter().take(5) {
                    println!(
                        "    {} -> {} visits, k0={:?}, k_last={:?}",
                        ptr, stats.visits, stats.k0, stats.k_last
                    );
                    let mut indices: Vec<_> = stats.child_indices.iter().collect();
                    indices.sort_by(|a, b| b.1.cmp(a.1));
                    print!("      child_idx: ");
                    for (idx, cnt) in indices.iter().take(5) {
                        print!("[{idx}]={cnt} ");
                    }
                    println!();
                }
            }
        }
    }

    // Cross-run analysis
    println!("\n{}", "=".repeat(100));
    println!("CROSS-RUN ANALYSIS");
    println!("{}", "=".repeat(100));

    let mut key_appearances: HashMap<String, Vec<String>> = HashMap::new();
    for run in all_runs {
        for (ikey, stats) in &run.key_events {
            if stats.coverage_collapse > 10 || stats.far_landing > 10 {
                key_appearances
                    .entry(ikey.clone())
                    .or_default()
                    .push(run.name.clone());
            }
        }
    }

    let mut multi_run_keys: Vec<_> = key_appearances
        .iter()
        .filter(|(_, runs)| runs.len() > 1)
        .collect();
    multi_run_keys.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    if multi_run_keys.is_empty() {
        println!("  No keys stuck in multiple runs (good - failures are independent)");
    } else {
        println!("  Keys stuck in multiple runs:");
        for (key, runs) in multi_run_keys.iter().take(10) {
            println!("    {} -> {} runs: {:?}", key, runs.len(), runs);
        }
    }

    // Summary
    println!("\n{}", "=".repeat(100));
    println!("FINAL SUMMARY");
    println!("{}", "=".repeat(100));

    let total_cc: usize = all_runs.iter().map(|r| r.coverage_collapse.count).sum();
    let total_fl: usize = all_runs.iter().map(|r| r.far_landing.count).sum();
    let total_bl: usize = all_runs.iter().map(|r| r.blink_limit.count).sum();
    let total_aborts: usize = all_runs.iter().map(|r| r.insert_abort.count).sum();
    let total_sticky: usize = all_runs
        .iter()
        .map(|r| {
            r.key_events
                .values()
                .filter(|k| k.sticky_leaf().is_some())
                .count()
        })
        .sum();

    // FIX: Clean run must also have low FAR_LANDING and no aborts
    let clean_runs = all_runs
        .iter()
        .filter(|r| {
            r.coverage_collapse.count < 100
                && r.far_landing.count < 1000
                && r.blink_limit.count == 0
                && r.insert_abort.count == 0
        })
        .count();

    println!("  Total runs: {}", all_runs.len());
    println!(
        "  Clean runs: {} ({:.1}%)",
        clean_runs,
        100.0 * clean_runs as f64 / all_runs.len() as f64
    );
    println!(
        "  Problem runs: {} ({:.1}%)",
        all_runs.len() - clean_runs,
        100.0 * (all_runs.len() - clean_runs) as f64 / all_runs.len() as f64
    );
    println!("  Total coverage_collapse: {total_cc}");
    println!("  Total far_landing_retry: {total_fl}");
    println!("  Total blink_limit: {total_bl}");
    println!("  Total insert_abort: {total_aborts}");
    println!("  Total sticky_leaf keys: {total_sticky}");

    // Diagnosis summary
    if total_sticky > 0 {
        println!("\n  ⚠️  DIAGNOSIS: {total_sticky} sticky leaf keys detected across runs.");
        println!("     This indicates PERMANENT routing inconsistency in the tree.");
        println!("     Root cause: internode keys don't cover child subtree bounds.");
    }
    if total_aborts > 0 {
        println!("\n  ⚠️  DIAGNOSIS: {total_aborts} insert operations aborted after max retries.");
    }
}
