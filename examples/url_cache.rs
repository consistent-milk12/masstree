//! Real-world example: URL-based cache with [`MassTree15`].
//!
//! [`MassTree`] excels at keys with shared prefixes, making it ideal for:
//! - URL routing tables
//! - API endpoint caches
//! - File path indexing
//! - DNS caches
//!
//! This example demonstrates a thread-safe URL cache with:
//! - TTL-based expiration
//! - Prefix-based invalidation (e.g., invalidate all "/api/v1/*")
//! - Statistics tracking
//! - Background cleanup
//!
//! Run with: `cargo run --example url_cache --release`

#![expect(
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::indexing_slicing
)]

use masstree::MassTree15;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// A cached response with metadata
#[allow(dead_code)]
#[derive(Clone, Debug)]
struct CachedResponse {
    /// The cached response body
    body: String,

    /// HTTP status code
    status: u16,

    /// Unix timestamp when this entry expires
    expires_at: u64,

    /// When this entry was created
    created_at: u64,
}

impl CachedResponse {
    fn new(body: String, status: u16, ttl_secs: u64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            body,
            status,
            expires_at: now + ttl_secs,
            created_at: now,
        }
    }

    fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        now >= self.expires_at
    }
}

/// Cache statistics
struct CacheStats {
    hits: AtomicU64,
    misses: AtomicU64,
    inserts: AtomicU64,
    evictions: AtomicU64,
}

impl CacheStats {
    const fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            inserts: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    fn hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }
    fn miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }
    fn insert(&self) {
        self.inserts.fetch_add(1, Ordering::Relaxed);
    }
    fn evict(&self, count: u64) {
        self.evictions.fetch_add(count, Ordering::Relaxed);
    }

    fn report(&self) -> String {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total > 0 {
            hits as f64 / total as f64 * 100.0
        } else {
            0.0
        };

        format!(
            "Hits: {}, Misses: {}, Hit Rate: {:.1}%, Inserts: {}, Evictions: {}",
            hits,
            misses,
            hit_rate,
            self.inserts.load(Ordering::Relaxed),
            self.evictions.load(Ordering::Relaxed)
        )
    }
}

/// Thread-safe URL cache backed by `MassTree`
struct UrlCache {
    tree: MassTree15<CachedResponse>,
    stats: CacheStats,
    default_ttl: u64,
}

#[allow(dead_code)]
impl UrlCache {
    fn new(default_ttl_secs: u64) -> Self {
        Self {
            tree: MassTree15::new(),
            stats: CacheStats::new(),
            default_ttl: default_ttl_secs,
        }
    }

    /// Get a cached response, returning None if not found or expired
    ///
    /// If an expired entry is found, it is removed lazily (on-access cleanup).
    fn get(&self, url: &str) -> Option<CachedResponse> {
        let guard = self.tree.guard();

        match self.tree.get_with_guard(url.as_bytes(), &guard) {
            Some(entry) if !entry.is_expired() => {
                self.stats.hit();
                Some((*entry).clone())
            }
            Some(_) => {
                // Entry exists but expired - remove it lazily and count as miss
                let _ = self.tree.remove_with_guard(url.as_bytes(), &guard);
                self.stats.evict(1);
                self.stats.miss();
                None
            }
            None => {
                self.stats.miss();
                None
            }
        }
    }

    /// Cache a response with the default TTL
    fn set(&self, url: &str, body: String, status: u16) {
        self.set_with_ttl(url, body, status, self.default_ttl);
    }

    /// Cache a response with a custom TTL
    fn set_with_ttl(&self, url: &str, body: String, status: u16, ttl_secs: u64) {
        let entry = CachedResponse::new(body, status, ttl_secs);
        let guard = self.tree.guard();
        let _ = self.tree.insert_with_guard(url.as_bytes(), entry, &guard);
        self.stats.insert();
    }

    /// Invalidate all cached entries matching a prefix
    ///
    /// Example: `invalidate_prefix("/api/v1/")` removes all v1 API cache entries
    fn invalidate_prefix(&self, prefix: &str) -> usize {
        let guard = self.tree.guard();
        let mut to_remove = Vec::new();

        // Collect keys to remove (can't remove while iterating)
        self.tree.scan_prefix(
            prefix.as_bytes(),
            |key, _| {
                to_remove.push(key.to_vec());
                true
            },
            &guard,
        );

        // Remove collected keys
        let count = to_remove.len();
        for key in to_remove {
            let _ = self.tree.remove_with_guard(&key, &guard);
        }

        self.stats.evict(count as u64);
        count
    }

    /// Remove a specific cached entry
    fn invalidate(&self, url: &str) -> bool {
        let guard = self.tree.guard();
        match self.tree.remove_with_guard(url.as_bytes(), &guard) {
            Ok(Some(_)) => {
                self.stats.evict(1);
                true
            }
            _ => false,
        }
    }

    /// Clean up expired entries
    fn cleanup_expired(&self) -> usize {
        let guard = self.tree.guard();
        let mut to_remove = Vec::new();

        // Find all expired entries
        for entry in self.tree.iter(&guard) {
            // entry.value() returns &ValuePtr<CachedResponse>
            let cached = entry.value();
            if cached.is_expired() {
                to_remove.push(entry.key().to_vec());
            }
        }

        // Remove them
        let count = to_remove.len();
        for key in to_remove {
            let _ = self.tree.remove_with_guard(&key, &guard);
        }

        // Also process any pending coalesce operations
        self.tree.process_coalesce(&guard);

        self.stats.evict(count as u64);
        count
    }

    /// Get all cached URLs matching a prefix
    fn list_prefix(&self, prefix: &str) -> Vec<String> {
        let guard = self.tree.guard();
        let mut urls = Vec::new();

        self.tree.scan_prefix(
            prefix.as_bytes(),
            |key, _| {
                if let Ok(url) = String::from_utf8(key.to_vec()) {
                    urls.push(url);
                }
                true
            },
            &guard,
        );

        urls
    }

    /// Get cache size
    fn len(&self) -> usize {
        self.tree.len()
    }

    /// Get statistics report
    fn stats(&self) -> String {
        self.stats.report()
    }
}

fn fetch_from_database(user_id: u32) -> String {
    // Simulate database fetch
    thread::sleep(Duration::from_micros(100));
    format!(r#"{{"id":{user_id},"name":"User{user_id}","source":"database"}}"#)
}

fn get_user(cache: &UrlCache, user_id: u32) -> String {
    let url = format!("/api/users/{user_id}");

    // Try cache first
    if let Some(cached) = cache.get(&url) {
        return cached.body;
    }

    // Cache miss - fetch from "database"
    let response = fetch_from_database(user_id);

    // Store in cache for next time
    cache.set(&url, response.clone(), 200);

    response
}

fn main() {
    println!("=== URL Cache with MassTree ===\n");

    let cache = Arc::new(UrlCache::new(300)); // 5 minute default TTL

    // =========================================================================
    // Populate cache with sample data
    // =========================================================================
    println!("--- Populating Cache ---\n");

    // Simulate caching API responses
    let api_endpoints = vec![
        ("/api/v1/users/1", r#"{"id":1,"name":"Alice"}"#, 200),
        ("/api/v1/users/2", r#"{"id":2,"name":"Bob"}"#, 200),
        ("/api/v1/users/3", r#"{"id":3,"name":"Charlie"}"#, 200),
        ("/api/v1/posts/1", r#"{"id":1,"title":"Hello World"}"#, 200),
        (
            "/api/v1/posts/2",
            r#"{"id":2,"title":"MassTree is fast"}"#,
            200,
        ),
        (
            "/api/v2/users/1",
            r#"{"id":1,"name":"Alice","v2":true}"#,
            200,
        ),
        ("/api/v2/users/2", r#"{"id":2,"name":"Bob","v2":true}"#, 200),
        ("/static/js/app.js", "console.log('hello');", 200),
        ("/static/css/style.css", "body { margin: 0; }", 200),
        ("/images/logo.png", "[binary data]", 200),
    ];

    for (url, body, status) in &api_endpoints {
        cache.set(url, body.to_string(), *status);
    }

    println!("Cached {} endpoints", api_endpoints.len());
    println!("Cache size: {}\n", cache.len());

    // =========================================================================
    // Cache lookups
    // =========================================================================
    println!("--- Cache Lookups ---\n");

    // Successful lookups
    if let Some(response) = cache.get("/api/v1/users/1") {
        println!(
            "GET /api/v1/users/1 -> {} (status {})",
            response.body, response.status
        );
    }

    if let Some(response) = cache.get("/api/v2/users/1") {
        println!(
            "GET /api/v2/users/1 -> {} (status {})",
            response.body, response.status
        );
    }

    // Cache miss
    if cache.get("/api/v1/users/999").is_none() {
        println!("GET /api/v1/users/999 -> MISS");
    }

    println!("\nStats: {}\n", cache.stats());

    // =========================================================================
    // Prefix-based operations
    // =========================================================================
    println!("--- Prefix-based Operations ---\n");

    // List all v1 API endpoints
    let v1_endpoints = cache.list_prefix("/api/v1/");
    println!("All /api/v1/* endpoints ({}):", v1_endpoints.len());
    for url in &v1_endpoints {
        println!("  {url}");
    }

    // List all static assets
    let static_assets = cache.list_prefix("/static/");
    println!("\nAll /static/* assets ({}):", static_assets.len());
    for url in &static_assets {
        println!("  {url}");
    }

    println!();

    // =========================================================================
    // Prefix invalidation
    // =========================================================================
    println!("--- Prefix Invalidation ---\n");

    println!("Cache size before: {}", cache.len());

    // Invalidate all v1 API cache (e.g., after a deployment)
    let invalidated = cache.invalidate_prefix("/api/v1/");
    println!("Invalidated {invalidated} entries matching /api/v1/*");
    println!("Cache size after: {}", cache.len());

    // Verify v1 endpoints are gone
    if cache.get("/api/v1/users/1").is_none() {
        println!("/api/v1/users/1 is now a cache MISS (as expected)");
    }

    // v2 endpoints should still be cached
    if cache.get("/api/v2/users/1").is_some() {
        println!("/api/v2/users/1 is still a HIT");
    }

    println!("\nStats: {}\n", cache.stats());

    // =========================================================================
    // Concurrent access simulation
    // =========================================================================
    println!("--- Concurrent Access Simulation ---\n");

    // Re-populate v1 endpoints
    for (url, body, status) in api_endpoints
        .iter()
        .filter(|(u, _, _)| u.contains("/api/v1/"))
    {
        cache.set(url, body.to_string(), *status);
    }

    let cache_clone = Arc::clone(&cache);
    let start = Instant::now();

    // Spawn multiple reader threads
    let mut handles = vec![];
    for thread_id in 0..4 {
        let cache = Arc::clone(&cache_clone);
        let handle = thread::spawn(move || {
            let urls = [
                "/api/v1/users/1",
                "/api/v1/users/2",
                "/api/v2/users/1",
                "/static/js/app.js",
                "/nonexistent/path",
            ];

            let mut hits = 0;
            let mut misses = 0;

            for _ in 0..10_000 {
                let url = urls[thread_id % urls.len()];
                if cache.get(url).is_some() {
                    hits += 1;
                } else {
                    misses += 1;
                }
            }

            (hits, misses)
        });
        handles.push(handle);
    }

    // Spawn a writer thread
    let cache = Arc::clone(&cache_clone);
    let writer = thread::spawn(move || {
        for i in 0..1000 {
            let url = format!("/api/v1/temp/{i}");
            cache.set(&url, format!("temp_response_{i}"), 200);
        }
    });

    // Wait for all threads
    let mut total_hits = 0;
    let mut total_misses = 0;
    for handle in handles {
        let (hits, misses) = handle.join().unwrap();
        total_hits += hits;
        total_misses += misses;
    }
    writer.join().unwrap();

    let elapsed = start.elapsed();
    println!("4 readers (10k ops each) + 1 writer (1k inserts)");
    println!("Completed in {elapsed:?}");
    println!("Total hits: {total_hits}, Total misses: {total_misses}");
    println!("Cache size: {}\n", cache_clone.len());

    // =========================================================================
    // TTL and expiration
    // =========================================================================
    println!("--- TTL and Expiration ---\n");

    let cache = Arc::new(UrlCache::new(1)); // 1 second TTL for demo

    // Add entries with short TTL
    cache.set_with_ttl("/short/1", "expires soon".to_string(), 200, 1);
    cache.set_with_ttl("/short/2", "expires soon".to_string(), 200, 1);
    cache.set_with_ttl("/long/1", "expires later".to_string(), 200, 60);

    println!("Added 2 short-TTL and 1 long-TTL entries");
    println!("Cache size: {}", cache.len());

    // Immediate lookup should hit
    if cache.get("/short/1").is_some() {
        println!("/short/1 is a HIT (not yet expired)");
    }

    // Wait for expiration
    println!("Waiting 1.5 seconds for expiration...");
    thread::sleep(Duration::from_millis(1500));

    // Now short-TTL entries should be "expired" (counted as miss)
    if cache.get("/short/1").is_none() {
        println!("/short/1 is now a MISS (expired)");
    }

    // Long-TTL entry should still hit
    if cache.get("/long/1").is_some() {
        println!("/long/1 is still a HIT");
    }

    // Cleanup expired entries
    let cleaned = cache.cleanup_expired();
    println!("Cleaned up {cleaned} expired entries");
    println!("Cache size after cleanup: {}", cache.len());

    println!("\nFinal stats: {}", cache.stats());

    // =========================================================================
    // Real-world pattern: Cache-aside with fallback
    // =========================================================================
    println!("\n--- Cache-Aside Pattern ---\n");

    let cache = UrlCache::new(300);

    // First call - cache miss, fetches from "database"
    let start = Instant::now();
    let user1 = get_user(&cache, 42);
    let first_fetch = start.elapsed();
    println!("First fetch (cache miss): {user1}");
    println!("Time: {first_fetch:?}");

    // Second call - cache hit, no database fetch
    let start = Instant::now();
    let user1_cached = get_user(&cache, 42);
    let second_fetch = start.elapsed();
    println!("Second fetch (cache hit): {user1_cached}");
    println!("Time: {second_fetch:?}");

    println!(
        "\nCache-aside pattern: {:.0}x speedup on hit",
        first_fetch.as_nanos() as f64 / second_fetch.as_nanos().max(1) as f64
    );

    println!("\n=== URL Cache Example Complete ===");
}
