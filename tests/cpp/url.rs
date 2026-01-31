//! URL: URL-like key patterns.
//!
//! Port of `kvtest_url` from C++ `kvtest.hh`.
//!
//! Pattern:
//! - Keys that look like URLs with shared prefixes
//! - Tests trie behavior with hierarchical keys

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use masstree::MassTree15Inline;
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::sync::Arc;
use std::thread;

const SEED: u64 = 31949;
const N: usize = 20_000;

const DOMAINS: [&str; 4] = [
    "http://example.com/",
    "http://test.org/",
    "https://api.service.io/",
    "http://localhost:8080/",
];

const PATHS: [&str; 6] = ["users/", "api/v1/", "static/", "admin/", "auth/", "data/"];

fn make_url(rng: &mut StdRng) -> String {
    let domain = DOMAINS[rng.random_range(0..DOMAINS.len())];
    let path = PATHS[rng.random_range(0..PATHS.len())];
    let id: u32 = rng.random();
    format!("{domain}{path}{id}")
}

#[test]
fn url_single_thread() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let guard = tree.guard();
    let mut rng = StdRng::seed_from_u64(SEED);

    // Put phase
    let mut urls: Vec<String> = Vec::with_capacity(N);
    for i in 0..N {
        let url = make_url(&mut rng);
        tree.insert_with_guard(url.as_bytes(), i as u64, &guard);
        urls.push(url);
    }

    // Shuffle and get
    urls.shuffle(&mut rng);
    for (i, url) in urls.iter().enumerate() {
        let val = tree.get_with_guard(url.as_bytes(), &guard);
        assert!(val.is_some(), "URL {i} not found: {url}");
    }
}

#[test]
fn url_concurrent() {
    let tree = Arc::new(MassTree15Inline::<u64>::new());
    let num_threads = 4;
    let per_thread = N / num_threads;

    let handles: Vec<_> = (0..num_threads)
        .map(|tid| {
            let tree = Arc::clone(&tree);
            thread::spawn(move || {
                let guard = tree.guard();
                let mut rng = StdRng::seed_from_u64(SEED + tid as u64);

                let mut urls: Vec<String> = Vec::with_capacity(per_thread);

                for i in 0..per_thread {
                    let url = make_url(&mut rng);
                    let _ = tree.insert_with_guard(
                        url.as_bytes(),
                        (tid * per_thread + i) as u64,
                        &guard,
                    );

                    urls.push(url);
                }

                urls.shuffle(&mut rng);

                for url in &urls {
                    let val = tree.get_with_guard(url.as_bytes(), &guard);
                    assert!(val.is_some(), "URL not found: {url}");
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn url_shared_prefix_stress() {
    let tree: MassTree15Inline<u64> = MassTree15Inline::new();
    let guard = tree.guard();

    // All URLs share same domain - stress trie layer sharing
    let domain = "http://example.com/api/v1/users/";

    for i in 0..N {
        let url = format!("{domain}{i}");
        tree.insert_with_guard(url.as_bytes(), i as u64, &guard);
    }

    // Verify all present
    for i in 0..N {
        let url = format!("{domain}{i}");
        let val = tree.get_with_guard(url.as_bytes(), &guard);
        assert_eq!(val, Some(i as u64));
    }
}
