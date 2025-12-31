//! Debug test with flush
use anyhow::{Result as AnyResult, anyhow as ANYHOW};
use masstree::MassTree24;
use std::io::{self, Write};

fn main() -> AnyResult<()> {
    let prefix = b"test";

    let nk = 5000u64;
    let tree: MassTree24<u64> = MassTree24::new();
    let guard = tree.guard();

    for i in 0..nk {
        let key = make_key(prefix, i);
        tree.insert_with_guard(&key, i, &guard)
            .map_err(|e| ANYHOW!("{e:?}"))?;
    }

    println!("Inserted {nk} keys");
    io::stdout().flush().map_err(|e| ANYHOW!("{e:?}"))?;

    for i in 0..nk {
        let key = make_key(prefix, i);
        let _ = tree.remove_with_guard(&key, &guard);
    }

    println!("Removed all keys");
    io::stdout().flush().map_err(|e| ANYHOW!("{e:?}"))?;

    println!("Starting verification loop...");
    io::stdout().flush().map_err(|e| ANYHOW!("Error: {e:?}"))?;

    for i in 0..nk {
        eprint!("\rGetting key {i}...");
        io::stderr().flush().map_err(|e| ANYHOW!("Error: {e:?}"))?;

        let key = make_key(prefix, i);
        let _val = tree.get_with_guard(&key, &guard);

        eprintln!(" done");
    }

    println!("\nDone!");

    Ok(())
}

fn make_key(prefix: &[u8], val: u64) -> Vec<u8> {
    let mut key = prefix.to_vec();
    key.extend_from_slice(&val.to_be_bytes());
    key
}
