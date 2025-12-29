//! Profile binary for layer explosion workload (64-byte keys, 8 layers).
//!
//! Usage:
//!   cargo build --release --bin layer_explosion --features mimalloc
//!   perf record -g --call-graph dwarf -F 999 -- ./target/release/layer_explosion -j6
//!   perf report --stdio --no-children -g none --percent-limit 1

use std::sync::Arc;
use std::thread;

use masstree::MassTree;

const OPS_PER_THREAD: usize = 25_000;

fn make_long_key(val: u64) -> [u8; 64] {
    let mut key = [0u8; 64];
    // Spread the value across multiple 8-byte chunks
    // This forces layer creation at each chunk boundary
    let bytes = val.to_be_bytes();
    key[0..8].copy_from_slice(&bytes);
    key[8..16].copy_from_slice(&bytes);
    key[16..24].copy_from_slice(&val.wrapping_mul(0x9e3779b97f4a7c15).to_be_bytes());
    key[24..32].copy_from_slice(&val.wrapping_mul(0x517cc1b727220a95).to_be_bytes());
    // Rest stays zero - creates shared prefix patterns
    key
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut threads = 6;
    let mut iterations = 10;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-j" => {
                i += 1;
                threads = args[i].parse().expect("invalid thread count");
            }
            "-n" => {
                i += 1;
                iterations = args[i].parse().expect("invalid iteration count");
            }
            arg if arg.starts_with("-j") => {
                threads = arg[2..].parse().expect("invalid thread count");
            }
            arg if arg.starts_with("-n") => {
                iterations = arg[2..].parse().expect("invalid iteration count");
            }
            _ => {
                eprintln!("Usage: layer_explosion [-j<threads>] [-n<iterations>]");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    eprintln!(
        "Layer explosion: {} threads, {} iterations, {} ops/thread",
        threads, iterations, OPS_PER_THREAD
    );

    for iter in 0..iterations {
        let tree: Arc<MassTree<u64>> = Arc::new(MassTree::new());

        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let tree = Arc::clone(&tree);
                thread::spawn(move || {
                    let start = t * OPS_PER_THREAD;
                    let end = start + OPS_PER_THREAD;
                    for i in start..end {
                        let key = make_long_key(i as u64);
                        tree.insert(&key, i as u64);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        if iter == 0 {
            eprintln!("Iteration {}: tree populated", iter);
        }
    }

    eprintln!("Done");
}
