//! Real-world example: Concurrent session store with [`MassTree15`].
//!
//! A thread-safe session store suitable for web applications. Demonstrates:
//! - Session creation and validation
//! - Session data storage and retrieval
//! - User-based session lookup (prefix scans)
//! - Session expiration and cleanup
//! - Concurrent access patterns
//!
//! This pattern is common in web frameworks where you need:
//! - Fast session lookups by session ID
//! - Ability to invalidate all sessions for a user
//! - Background cleanup of expired sessions
//!
//! **Design Note**: This example uses efficient dual-indexing:
//! - Primary index: `session:{session_id}` -> `Session` (full session data)
//! - User index: `user:{user_id}:{session_id}` -> `()` (empty marker, just for lookup)
//!
//! This avoids storing duplicate session data in both indexes.
//!
//! Run with: `cargo run --example session_store --release`

#![expect(clippy::unwrap_used)]

use masstree::{MassTree15, MassTree15Inline};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Unique session identifier
type SessionId = String;
type UserId = u64;

/// Session data stored in the session store
#[allow(dead_code)]
#[derive(Clone, Debug)]
struct Session {
    /// Unique session identifier
    id: SessionId,

    /// User who owns this session
    user_id: UserId,

    /// When the session was created
    created_at: u64,

    /// When the session expires
    expires_at: u64,

    /// When the session was last accessed
    last_accessed: u64,

    /// IP address of the client
    ip_address: String,

    /// User agent string
    user_agent: String,

    /// Custom session data (e.g., preferences, cart items)
    data: HashMap<String, String>,
}

impl Session {
    fn new(id: SessionId, user_id: UserId, ttl_secs: u64, ip: &str, user_agent: &str) -> Self {
        let now = current_timestamp();
        Self {
            id,
            user_id,
            created_at: now,
            expires_at: now + ttl_secs,
            last_accessed: now,
            ip_address: ip.to_string(),
            user_agent: user_agent.to_string(),
            data: HashMap::new(),
        }
    }

    fn is_expired(&self) -> bool {
        current_timestamp() >= self.expires_at
    }

    fn touch(&mut self) {
        self.last_accessed = current_timestamp();
    }

    fn set_data(&mut self, key: &str, value: &str) {
        self.data.insert(key.to_string(), value.to_string());
    }

    fn get_data(&self, key: &str) -> Option<&String> {
        self.data.get(key)
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Generates a unique session ID
#[expect(clippy::cast_possible_truncation)]
fn generate_session_id() -> SessionId {
    use std::sync::atomic::AtomicU64;
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let timestamp = current_timestamp();
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let random: u32 = (timestamp as u32) ^ (counter as u32);

    format!("sess_{timestamp:x}_{counter:x}_{random:08x}")
}

/// Thread-safe session store backed by [`MassTree`]
///
/// Key format:
/// - Primary index: `session:{session_id}` -> `Session` (full session data)
/// - User index: `user:{user_id}:{session_id}` -> `()` (empty marker for user lookups)
///
/// This design avoids storing duplicate session data. The user index only stores
/// markers to enable efficient prefix scans for "all sessions for user X".
struct SessionStore {
    /// Main session storage: session:{session_id} -> Session
    sessions: MassTree15<Session>,

    /// User index: user:{user_id}:{session_id} -> () (marker only)
    /// Using MassTree15Inline<()> since we don't need to store any data
    user_index: MassTree15Inline<()>,

    /// Statistics
    active_sessions: AtomicU64,
    total_created: AtomicU64,
    total_expired: AtomicU64,

    /// Default session TTL in seconds
    default_ttl: u64,
}

impl SessionStore {
    fn new(default_ttl_secs: u64) -> Self {
        Self {
            sessions: MassTree15::new(),
            user_index: MassTree15Inline::new(),
            active_sessions: AtomicU64::new(0),
            total_created: AtomicU64::new(0),
            total_expired: AtomicU64::new(0),
            default_ttl: default_ttl_secs,
        }
    }

    /// Create a new session for a user
    fn create_session(&self, user_id: UserId, ip: &str, user_agent: &str) -> SessionId {
        let session_id = generate_session_id();
        let session = Session::new(
            session_id.clone(),
            user_id,
            self.default_ttl,
            ip,
            user_agent,
        );

        // Use separate guards for each tree (they have different collectors)
        let session_guard = self.sessions.guard();
        let user_guard = self.user_index.guard();

        // Store full session in primary index
        let primary_key = format!("session:{session_id}");
        let _ = self
            .sessions
            .insert_with_guard(primary_key.as_bytes(), session, &session_guard);

        // Store only a marker in user index (no duplicate session data!)
        let user_key = format!("user:{user_id}:{session_id}");
        let _ = self
            .user_index
            .insert_with_guard(user_key.as_bytes(), (), &user_guard);

        self.active_sessions.fetch_add(1, Ordering::Relaxed);
        self.total_created.fetch_add(1, Ordering::Relaxed);

        session_id
    }

    /// Get a session by ID, returning None if not found or expired
    fn get_session(&self, session_id: &str) -> Option<Arc<Session>> {
        let guard = self.sessions.guard();
        let key = format!("session:{session_id}");

        match self.sessions.get_with_guard(key.as_bytes(), &guard) {
            Some(session) if !session.is_expired() => Some(session),
            Some(_) => {
                // Expired session - clean it up lazily
                drop(guard); // Release guard before calling invalidate
                self.invalidate_session(session_id);
                None
            }
            None => None,
        }
    }

    /// Validate a session exists and is not expired
    fn is_valid(&self, session_id: &str) -> bool {
        self.get_session(session_id).is_some()
    }

    /// Update session data
    ///
    /// Note: This clones the session to update it. For high-frequency updates,
    /// consider using interior mutability (e.g., RwLock inside Session).
    fn update_session_data(&self, session_id: &str, key: &str, value: &str) -> bool {
        self.get_session(session_id).is_some_and(|session| {
            let mut updated = (*session).clone();
            updated.touch();
            updated.set_data(key, value);

            let guard = self.sessions.guard();

            // Only update primary index - user index just has markers
            let primary_key = format!("session:{session_id}");
            let _ = self
                .sessions
                .insert_with_guard(primary_key.as_bytes(), updated, &guard);

            true
        })
    }

    /// Get session data
    fn get_session_data(&self, session_id: &str, key: &str) -> Option<String> {
        self.get_session(session_id)
            .and_then(|s| s.get_data(key).cloned())
    }

    /// Invalidate (delete) a specific session
    fn invalidate_session(&self, session_id: &str) -> bool {
        let session_guard = self.sessions.guard();
        let primary_key = format!("session:{session_id}");

        // First get the session to find the user_id
        self.sessions
            .get_with_guard(primary_key.as_bytes(), &session_guard)
            .is_some_and(|session| {
                let user_id = session.user_id;
                let user_key = format!("user:{user_id}:{session_id}");

                // Remove from primary index
                let _ = self
                    .sessions
                    .remove_with_guard(primary_key.as_bytes(), &session_guard);

                // Remove from user index (separate guard)
                let user_guard = self.user_index.guard();
                let _ = self
                    .user_index
                    .remove_with_guard(user_key.as_bytes(), &user_guard);

                self.active_sessions.fetch_sub(1, Ordering::Relaxed);
                true
            })
    }

    /// Invalidate all sessions for a user (e.g., on password change or logout-all)
    fn invalidate_user_sessions(&self, user_id: UserId) -> usize {
        let user_guard = self.user_index.guard();
        let prefix = format!("user:{user_id}:");

        // Collect all session IDs for this user from the user index
        // The key format is "user:{user_id}:{session_id}", so we extract session_id from the key
        let mut session_ids = Vec::new();
        self.user_index.scan_prefix(
            prefix.as_bytes(),
            |key, _| {
                // Extract session_id from key: "user:{user_id}:{session_id}"
                if let Ok(key_str) = std::str::from_utf8(key)
                    && let Some(session_id) = key_str.strip_prefix(&prefix)
                {
                    session_ids.push(session_id.to_string());
                }
                true
            },
            &user_guard,
        );
        drop(user_guard); // Release before calling invalidate_session

        // Invalidate each session
        let count = session_ids.len();
        for session_id in session_ids {
            self.invalidate_session(&session_id);
        }

        count
    }

    /// Get all active sessions for a user
    fn get_user_sessions(&self, user_id: UserId) -> Vec<Arc<Session>> {
        let user_guard = self.user_index.guard();
        let session_guard = self.sessions.guard();
        let prefix = format!("user:{user_id}:");

        let mut sessions = Vec::new();

        // Scan user index to get session IDs, then look up full sessions
        self.user_index.scan_prefix(
            prefix.as_bytes(),
            |key, _| {
                // Extract session_id from key, then look up full session
                if let Ok(key_str) = std::str::from_utf8(key)
                    && let Some(session_id) = key_str.strip_prefix(&prefix)
                {
                    let session_key = format!("session:{session_id}");
                    if let Some(session) = self
                        .sessions
                        .get_with_guard(session_key.as_bytes(), &session_guard)
                        && !session.is_expired()
                    {
                        sessions.push(session);
                    }
                }
                true
            },
            &user_guard,
        );

        sessions
    }

    /// Clean up expired sessions
    fn cleanup_expired(&self) -> usize {
        let guard = self.sessions.guard();
        let mut expired_sessions = Vec::new();

        // Scan all sessions and find expired ones
        self.sessions.scan_prefix(
            b"session:",
            |_, session| {
                if session.is_expired() {
                    expired_sessions.push(session.id.clone());
                }
                true
            },
            &guard,
        );
        drop(guard); // Release before calling invalidate_session

        // Invalidate all expired sessions
        let count = expired_sessions.len();
        for session_id in expired_sessions {
            self.invalidate_session(&session_id);
        }

        // Process coalesce queues for both trees
        let session_guard = self.sessions.guard();
        let user_guard = self.user_index.guard();
        self.sessions.process_coalesce(&session_guard);
        self.user_index.process_coalesce(&user_guard);

        self.total_expired
            .fetch_add(count as u64, Ordering::Relaxed);
        count
    }

    /// Get store statistics
    fn stats(&self) -> String {
        format!(
            "Active: {}, Total Created: {}, Total Expired: {}, Sessions: {}, User Index: {}",
            self.active_sessions.load(Ordering::Relaxed),
            self.total_created.load(Ordering::Relaxed),
            self.total_expired.load(Ordering::Relaxed),
            self.sessions.len(),
            self.user_index.len(),
        )
    }
}

#[expect(
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::cast_precision_loss
)]
fn main() {
    println!("=== Concurrent Session Store with MassTree ===\n");

    // =========================================================================
    // Basic session operations
    // =========================================================================
    println!("--- Basic Session Operations ---\n");

    let store = Arc::new(SessionStore::new(3600)); // 1 hour sessions

    // Create sessions for different users
    let alice_session = store.create_session(1, "192.168.1.100", "Mozilla/5.0");
    let bob_session = store.create_session(2, "192.168.1.101", "Chrome/120.0");
    let charlie_session = store.create_session(3, "10.0.0.50", "Safari/17.0");

    println!("Created sessions:");
    println!("  Alice (user 1): {alice_session}");
    println!("  Bob (user 2): {bob_session}");
    println!("  Charlie (user 3): {charlie_session}");
    println!("\nStats: {}\n", store.stats());

    // Validate sessions
    println!("Session validation:");
    println!(
        "  Alice's session valid: {}",
        store.is_valid(&alice_session)
    );
    println!(
        "  Invalid session valid: {}",
        store.is_valid("fake_session_id")
    );

    println!();

    // =========================================================================
    // Session data storage
    // =========================================================================
    println!("--- Session Data Storage ---\n");

    // Store data in Alice's session
    store.update_session_data(&alice_session, "theme", "dark");
    store.update_session_data(&alice_session, "language", "en");
    store.update_session_data(&alice_session, "cart_items", "3");

    // Retrieve data
    if let Some(theme) = store.get_session_data(&alice_session, "theme") {
        println!("Alice's theme: {theme}");
    }
    if let Some(lang) = store.get_session_data(&alice_session, "language") {
        println!("Alice's language: {lang}");
    }
    if let Some(cart) = store.get_session_data(&alice_session, "cart_items") {
        println!("Alice's cart items: {cart}");
    }

    // Try to get data from non-existent key
    if store
        .get_session_data(&alice_session, "nonexistent")
        .is_none()
    {
        println!("Non-existent key returns None (as expected)");
    }

    println!();

    // =========================================================================
    // Multi-session user
    // =========================================================================
    println!("--- Multi-Session User ---\n");

    // Alice logs in from multiple devices
    let alice_mobile = store.create_session(1, "10.0.0.1", "MobileApp/2.0");
    let _alice_tablet = store.create_session(1, "10.0.0.2", "TabletApp/1.5");

    println!("Alice now has 3 sessions (desktop, mobile, tablet)");

    // List all of Alice's sessions
    let alice_sessions = store.get_user_sessions(1);
    println!("Alice's active sessions ({}):", alice_sessions.len());
    for session in &alice_sessions {
        println!(
            "  {} from {} ({})",
            session.id, session.ip_address, session.user_agent
        );
    }

    println!("\nStats: {}\n", store.stats());

    // =========================================================================
    // Session invalidation
    // =========================================================================
    println!("--- Session Invalidation ---\n");

    // Single session logout
    println!("Logging out Alice's mobile session...");
    let invalidated = store.invalidate_session(&alice_mobile);
    println!("Session invalidated: {invalidated}");

    // Verify mobile session is gone
    println!("Mobile session valid: {}", store.is_valid(&alice_mobile));
    println!("Desktop session valid: {}", store.is_valid(&alice_session));

    // Alice changes her password - invalidate ALL her sessions
    println!("\nAlice changed password - invalidating all sessions...");
    let count = store.invalidate_user_sessions(1);
    println!("Invalidated {count} sessions for user 1");

    // Verify all Alice's sessions are gone
    let alice_sessions = store.get_user_sessions(1);
    println!("Alice's remaining sessions: {}", alice_sessions.len());

    println!("\nStats: {}\n", store.stats());

    // =========================================================================
    // Session expiration
    // =========================================================================
    println!("--- Session Expiration ---\n");

    let store = Arc::new(SessionStore::new(1)); // 1 second sessions for demo

    // Create sessions that will expire quickly
    let _ = store.create_session(100, "1.2.3.4", "TestClient");
    let _ = store.create_session(101, "5.6.7.8", "TestClient");
    let _ = store.create_session(102, "9.10.11.12", "TestClient");

    println!("Created 3 short-lived sessions");
    println!("Stats before expiry: {}", store.stats());

    // Wait for expiration
    println!("Waiting 1.5 seconds for expiration...");
    thread::sleep(Duration::from_millis(1500));

    // Cleanup expired sessions
    let cleaned = store.cleanup_expired();
    println!("Cleaned up {cleaned} expired sessions");
    println!("Stats after cleanup: {}\n", store.stats());

    // =========================================================================
    // Concurrent access
    // =========================================================================
    println!("--- Concurrent Access Simulation ---\n");

    let store = Arc::new(SessionStore::new(3600));

    // Pre-create some sessions
    let mut session_ids: Vec<String> = Vec::new();
    for user_id in 0..100 {
        session_ids.push(store.create_session(user_id, "127.0.0.1", "LoadTest"));
    }

    println!("Created 100 sessions for load test");
    let start = std::time::Instant::now();

    // Spawn reader threads
    let mut handles = vec![];
    for _ in 0..4 {
        let store = Arc::clone(&store);
        let sessions = session_ids.clone();
        let handle = thread::spawn(move || {
            let mut valid = 0;
            for _ in 0..10_000 {
                let idx = (valid as usize) % sessions.len();
                if store.is_valid(&sessions[idx]) {
                    valid += 1;
                }
            }
            valid
        });
        handles.push(handle);
    }

    // Spawn writer threads (updating session data)
    for _ in 0..2 {
        let store = Arc::clone(&store);
        let sessions = session_ids.clone();
        let handle = thread::spawn(move || {
            let mut updates = 0;
            for i in 0..5_000 {
                let idx = i % sessions.len();
                if store.update_session_data(&sessions[idx], "counter", &i.to_string()) {
                    updates += 1;
                }
            }
            updates
        });
        handles.push(handle);
    }

    // Wait and collect results
    let mut total_reads = 0;
    let mut total_writes = 0;
    for (i, handle) in handles.into_iter().enumerate() {
        let count = handle.join().unwrap();
        if i < 4 {
            total_reads += count;
        } else {
            total_writes += count;
        }
    }

    let elapsed = start.elapsed();
    println!("4 reader threads (10k validations each) + 2 writer threads (5k updates each)");
    println!("Total reads: {total_reads}, Total writes: {total_writes}");
    println!("Completed in {elapsed:?}");
    println!(
        "Throughput: {:.2} ops/ms\n",
        f64::from(total_reads + total_writes) / elapsed.as_millis() as f64
    );

    // =========================================================================
    // Background cleanup task simulation
    // =========================================================================
    println!("--- Background Cleanup Task ---\n");

    let store = Arc::new(SessionStore::new(1)); // Short TTL for demo

    // Simulate continuous session creation and cleanup
    let producer_store = Arc::clone(&store);
    let producer = thread::spawn(move || {
        for i in 0..500 {
            producer_store.create_session(i % 50, "127.0.0.1", "Producer");
            if i % 100 == 0 {
                thread::sleep(Duration::from_millis(10));
            }
        }
    });

    let cleaner_store = Arc::clone(&store);
    let cleaner = thread::spawn(move || {
        let mut total_cleaned = 0;
        for _ in 0..10 {
            thread::sleep(Duration::from_millis(150));
            let cleaned = cleaner_store.cleanup_expired();
            if cleaned > 0 {
                total_cleaned += cleaned;
            }
        }
        total_cleaned
    });

    producer.join().unwrap();
    let total_cleaned = cleaner.join().unwrap();

    println!("Producer created 500 sessions");
    println!("Cleaner removed {total_cleaned} expired sessions");
    println!("Final stats: {}", store.stats());

    // =========================================================================
    // Security pattern: Rate limiting by session
    // =========================================================================
    println!("\n--- Security: Per-Session Rate Tracking ---\n");

    let store = SessionStore::new(3600);
    let session = store.create_session(1, "192.168.1.1", "Client");

    // Track request count in session
    for i in 1..=5 {
        let current = store
            .get_session_data(&session, "request_count")
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0);

        let new_count = current + 1;
        store.update_session_data(&session, "request_count", &new_count.to_string());

        let allowed = new_count <= 3;
        println!("Request {i}: count={new_count}, allowed={allowed}");
    }

    println!("\nRate limit demonstration: requests 4 and 5 exceeded threshold");

    println!("\n=== Session Store Example Complete ===");
}
