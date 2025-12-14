use clap::{Parser, Subcommand};
use secp256k1::rand::{self, RngCore};
use secp256k1::{Secp256k1, SecretKey};
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::time::Instant;

// Import common functions from lib.rs
use mocnpub_main::gpu::{
    TripleBufferMiner, calculate_optimal_batch_size, generate_pubkeys_with_prefix_match,
    get_max_keys_per_thread, get_sm_count, init_gpu,
};
use mocnpub_main::{add_u64x4_scalar, adjust_privkey_for_endomorphism, prefixes_to_bits};
use mocnpub_main::{bytes_to_u64x4, pubkey_bytes_to_npub, u64x4_to_bytes};
use mocnpub_main::{pubkey_to_npub, seckey_to_nsec, validate_prefix};

/// Nostr npub mining tool 🔑
#[derive(Parser, Debug)]
#[command(name = "mocnpub")]
#[command(about = "Nostr npub mining tool 🔑", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Mine npub with specified prefix (GPU accelerated 🚀)
    Mine {
        /// Prefix to mine (bech32 string following npub1)
        ///
        /// Single prefix: "abc", "test", "satoshi"
        /// Multiple prefixes (OR): "m0ctane0,m0ctane2,m0ctane3" (comma-separated)
        /// Full npub example: specify "abc" part of npub1abc...
        #[arg(short, long)]
        prefix: String,

        /// Output file (optional, defaults to stdout)
        #[arg(short, long)]
        output: Option<String>,

        /// Number of keys to find (0 = unlimited, default: 1)
        #[arg(short, long, default_value = "1")]
        limit: usize,

        /// Batch size (default: 4000000)
        #[arg(long, default_value = "4000000")]
        batch_size: usize,

        /// GPU threads per block (default: 128, optimal for RTX 5070 Ti)
        #[arg(long, default_value = "128")]
        threads_per_block: u32,
    },

    /// Verify GPU calculations against CPU (fuzzing-like testing)
    Verify {
        /// Number of iterations (0 = unlimited, run until Ctrl+C)
        #[arg(short, long, default_value = "0")]
        iterations: u64,
    },
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Mine {
            prefix,
            output,
            limit,
            batch_size,
            threads_per_block,
        } => run_mine(prefix, output, limit, batch_size, threads_per_block),
        Commands::Verify { iterations } => run_verify(iterations),
    }
}

/// Run the mine subcommand
fn run_mine(
    prefix: String,
    output: Option<String>,
    limit: usize,
    batch_size: usize,
    threads_per_block: u32,
) -> io::Result<()> {
    // Split prefix by comma and convert to Vec
    let prefixes: Vec<String> = prefix.split(',').map(|s| s.trim().to_string()).collect();

    // Validate each prefix
    for p in &prefixes {
        if let Err(err_msg) = validate_prefix(p) {
            eprintln!("❌ Error: {}", err_msg);
            std::process::exit(1);
        }
    }

    let keys_per_thread = get_max_keys_per_thread();

    println!("🔥 mocnpub - Nostr npub mining 🔥");
    if prefixes.len() == 1 {
        println!("Prefix: '{}'", prefixes[0]);
    } else {
        println!("Prefixes (OR): {}", prefixes.join(", "));
    }
    println!("Batch size: {}", batch_size);
    println!(
        "Threads/block: {}, Keys/thread: {} (build-time)",
        threads_per_block, keys_per_thread
    );
    println!(
        "Limit: {}\n",
        if limit == 0 {
            "unlimited".to_string()
        } else {
            limit.to_string()
        }
    );

    mining_loop(
        &prefixes,
        limit,
        batch_size,
        threads_per_block,
        keys_per_thread,
        output.as_deref(),
    )
}

/// Main mining loop (GPU-side prefix matching)
fn mining_loop(
    prefixes: &[String],
    limit: usize,
    batch_size: usize,
    threads_per_block: u32,
    keys_per_thread: u32,
    output_path: Option<&str>,
) -> io::Result<()> {
    // Initialize GPU
    let ctx = match init_gpu() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("❌ GPU initialization failed: {}", e);
            std::process::exit(1);
        }
    };
    println!("✅ GPU initialized successfully!");

    // Get SM count and optimize batch_size (Tail Effect mitigation)
    let sm_count = match get_sm_count(&ctx) {
        Ok(count) => count,
        Err(e) => {
            eprintln!("⚠️ Could not get SM count: {}, using default batch_size", e);
            0
        }
    };

    let batch_size = if sm_count > 0 {
        match calculate_optimal_batch_size(&ctx, batch_size, threads_per_block) {
            Ok(adjusted) => {
                if adjusted != batch_size {
                    println!(
                        "📐 SM count: {}, adjusted batch_size: {} → {} (Tail Effect mitigation)",
                        sm_count, batch_size, adjusted
                    );
                } else {
                    println!(
                        "📐 SM count: {}, batch_size: {} (already optimal)",
                        sm_count, batch_size
                    );
                }
                adjusted
            }
            Err(_) => batch_size,
        }
    } else {
        batch_size
    };

    // Convert prefixes to bit patterns (pre-computed)
    let prefix_bits = prefixes_to_bits(prefixes);
    println!(
        "📊 Prefix patterns prepared: {} pattern(s)\n",
        prefix_bits.len()
    );

    let start = Instant::now();
    let mut total_count: u64 = 0;
    let mut found_count: usize = 0;
    let mut rng = rand::thread_rng();

    // Prepare file output (append mode)
    let mut output_file = if let Some(path) = output_path {
        Some(OpenOptions::new().create(true).append(true).open(path)?)
    } else {
        None
    };

    // Parameter settings
    let max_matches: u32 = 1000; // generous buffer

    // Create TripleBufferMiner (PTX, streams, buffers are initialized once)
    let mut miner = match TripleBufferMiner::new(
        &ctx,
        &prefix_bits,
        max_matches,
        threads_per_block,
        batch_size,
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("❌ Failed to create TripleBufferMiner: {}", e);
            std::process::exit(1);
        }
    };
    println!("✅ TripleBufferMiner initialized (PTX cached, 3 buffers pre-allocated)");

    // Secret key buffers (base keys) - 3 sets for triple buffering
    // Always 2 buffers in GPU, 1 being processed by CPU (RNG or result processing)
    let mut host_bytes: [Vec<[u8; 32]>; 3] = [
        vec![[0u8; 32]; batch_size],
        vec![[0u8; 32]; batch_size],
        vec![[0u8; 32]; batch_size],
    ];
    let mut host_u64: [Vec<[u64; 4]>; 3] = [
        vec![[0u64; 4]; batch_size],
        vec![[0u64; 4]; batch_size],
        vec![[0u64; 4]; batch_size],
    ];

    // Generate and launch ALL 3 initial batches
    for buf_idx in 0..3 {
        for i in 0..batch_size {
            rng.fill_bytes(&mut host_bytes[buf_idx][i]);
            host_u64[buf_idx][i] = bytes_to_u64x4(&host_bytes[buf_idx][i]);
        }
        if let Err(e) = miner.launch_single(buf_idx, &host_u64[buf_idx]) {
            eprintln!("❌ GPU kernel launch error: {}", e);
            std::process::exit(1);
        }
    }
    println!("🎯 Triple buffering enabled (always 2 kernels in GPU queue)");

    // Rotation index: which buffer to collect next
    // Pattern: collect(N) → launch(N) → RNG(N) → rotate
    let mut collect_idx = 0usize;

    // Main loop (triple buffer rotation)
    loop {
        // 1. Collect results from buffer[collect_idx] (blocking)
        // Note: While collecting, buffer[(collect_idx+1)%3] is still processing in GPU!
        let matches = match miner.collect_single(collect_idx) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("❌ GPU kernel error: {}", e);
                std::process::exit(1);
            }
        };

        // Update attempt count (endomorphism checks 3 X-coordinates per key)
        total_count += (batch_size as u64) * (keys_per_thread as u64) * 3;

        // 2. Process matched results
        // Now we use m.base_key directly (returned from GPU), no need for host-side buffer lookup
        for m in matches {
            found_count += 1;

            // Recover secret key: (base_key + offset) * λ^endo_type mod n
            // endo_type: 0 = original, 1 = λ*k, 2 = λ²*k (for endomorphism)
            let key_with_offset = add_u64x4_scalar(&m.base_key, m.offset);
            let actual_key_u64 = adjust_privkey_for_endomorphism(&key_with_offset, m.endo_type);
            let actual_key_bytes = u64x4_to_bytes(&actual_key_u64);

            // Generate nsec from secret key
            let sk = SecretKey::from_slice(&actual_key_bytes).expect("Invalid secret key");
            let nsec = seckey_to_nsec(&sk);

            // Get public key (recomputed from secret key - this is the correct value)
            let secp = Secp256k1::new();
            let pk = sk.public_key(&secp);
            let pk_hex = pk.to_string();
            let pk_x_only = &pk_hex[2..];

            // Compute npub (using public key recomputed from secret key)
            let npub = pubkey_to_npub(&pk);
            let npub_body = &npub[5..];

            // Verify consistency with pubkey_x returned from GPU
            let gpu_npub = pubkey_bytes_to_npub(&u64x4_to_bytes(&m.pubkey_x));
            if npub != gpu_npub {
                eprintln!(
                    "⚠️  Warning: GPU pubkey_x mismatch! endo_type={}",
                    m.endo_type
                );
                eprintln!("    GPU npub:    {}", gpu_npub);
                eprintln!("    Actual npub: {}", npub);
            }

            // Identify matched prefix (verified against actual npub)
            let matched_prefix = prefixes
                .iter()
                .find(|p| npub_body.starts_with(p.as_str()))
                .map(|p| p.as_str())
                .unwrap_or("?");

            let elapsed = start.elapsed();
            let elapsed_secs = elapsed.as_secs_f64();
            let keys_per_sec = total_count as f64 / elapsed_secs;

            // Format result
            let output_text = format!(
                "✅ Found #{}! ({} attempts, GPU prefix match)\n\
                 Matched prefix: '{}'\n\n\
                 Elapsed: {:.2} sec\n\
                 Performance: {:.2} keys/sec\n\n\
                 Secret key (hex): {}\n\
                 Secret key (nsec): {}\n\
                 Public key (compressed): {}\n\
                 Public key (x-coord): {}\n\
                 Public key (npub): {}\n\
{}\n",
                found_count,
                total_count,
                matched_prefix,
                elapsed_secs,
                keys_per_sec,
                sk.display_secret(),
                nsec,
                pk,
                pk_x_only,
                npub,
                "=".repeat(80)
            );

            // Output
            if let Some(ref mut file) = output_file {
                file.write_all(output_text.as_bytes())?;
                file.flush()?;
            }
            print!("{}", output_text);
            io::stdout().flush()?;

            // Exit if limit reached
            if limit > 0 && found_count >= limit {
                let final_elapsed = start.elapsed();
                let final_elapsed_secs = final_elapsed.as_secs_f64();
                println!("\n🎉 GPU mining complete!");
                println!("Keys found: {}", found_count);
                println!("Total attempts: {}", total_count);
                println!("Elapsed: {:.2} sec", final_elapsed_secs);
                println!(
                    "Performance: {:.2} keys/sec",
                    total_count as f64 / final_elapsed_secs
                );
                if let Some(path) = output_path {
                    println!("Results saved to: {}", path);
                }
                return Ok(());
            }
        }

        // Progress display (every 10 batches)
        let batch_keys = (batch_size as u64) * (keys_per_thread as u64);
        if total_count.is_multiple_of(batch_keys * 10) {
            let elapsed_secs = start.elapsed().as_secs_f64();
            let keys_per_sec = total_count as f64 / elapsed_secs;
            let mins = (elapsed_secs / 60.0) as u64;
            let secs = (elapsed_secs % 60.0) as u64;
            println!(
                "{} attempts... ({}:{:02}, {:.2} keys/sec, found: {})",
                total_count, mins, secs, keys_per_sec, found_count
            );
        }

        // 3. Launch next batch for this stream (it's now free after collect)
        // Use the data that was generated 2 rotations ago (ready to use)
        let ready_data_idx = (collect_idx + 2) % 3;
        if let Err(e) = miner.launch_single(collect_idx, &host_u64[ready_data_idx]) {
            eprintln!("❌ GPU kernel launch error: {}", e);
            std::process::exit(1);
        }

        // 5. Generate new RNG data into the buffer we just processed
        // This buffer is now safe to overwrite (results already extracted)
        // It will be used 2 rotations later
        for i in 0..batch_size {
            rng.fill_bytes(&mut host_bytes[collect_idx][i]);
            host_u64[collect_idx][i] = bytes_to_u64x4(&host_bytes[collect_idx][i]);
        }

        // 6. Rotate to next buffer
        collect_idx = (collect_idx + 1) % 3;
    }
}

/// Run the verify subcommand (GPU vs CPU verification using production kernel)
///
/// This verifies the full production pipeline:
/// - Montgomery's Trick (batch inverse)
/// - Sequential key strategy (base_key + offset)
/// - Endomorphism (β, β² for 3x X-coordinate checks)
/// - GPU-side prefix matching
fn run_verify(iterations: u64) -> io::Result<()> {
    println!("🔬 mocnpub - GPU Verification Mode (Production Kernel) 🔬");
    println!(
        "Iterations: {}\n",
        if iterations == 0 {
            "unlimited (Ctrl+C to stop)".to_string()
        } else {
            iterations.to_string()
        }
    );

    // Initialize GPU
    let ctx = match init_gpu() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("❌ GPU initialization failed: {}", e);
            std::process::exit(1);
        }
    };
    println!("✅ GPU initialized successfully!");

    let keys_per_thread = get_max_keys_per_thread();
    println!("Keys per thread: {} (build-time)", keys_per_thread);

    let secp = Secp256k1::new();
    let mut rng = rand::thread_rng();
    let start = Instant::now();
    let mut batch_count: u64 = 0;
    let mut total_keys_checked: u64 = 0;
    let mut total_matches_verified: u64 = 0;
    let mut errors: u64 = 0;

    // Use ALL 32 bech32 characters as prefixes for 100% match rate
    // This verifies EVERY key generated by the GPU!
    // bech32 charset: 023456789acdefghjklmnpqrstuvwxyz (excludes 1, b, i, o)
    let bech32_chars = "023456789acdefghjklmnpqrstuvwxyz";
    let all_prefixes: Vec<String> = bech32_chars.chars().map(|c| c.to_string()).collect();
    let prefix_bits = prefixes_to_bits(&all_prefixes);
    println!(
        "Verify prefixes: all {} bech32 chars (100% match rate = full verification)",
        all_prefixes.len()
    );

    // Batch size for verification (smaller than production for faster feedback)
    let batch_size: usize = 1024;
    let threads_per_block: u32 = 128;
    // With 32 prefixes (all bech32 chars), EVERY key matches!
    // So max_matches needs to be at least batch_size * keys_per_thread * 3 (endo)
    let max_matches: u32 = (batch_size as u32) * keys_per_thread * 3 + 1000;

    println!(
        "Batch size: {} threads × {} keys/thread × 3 (endo) = {} keys/batch\n",
        batch_size,
        keys_per_thread,
        batch_size as u64 * keys_per_thread as u64 * 3
    );
    println!("Starting verification...\n");

    // Buffers for base keys
    let mut privkey_bytes: Vec<[u8; 32]> = vec![[0u8; 32]; batch_size];
    let mut privkeys_u64: Vec<[u64; 4]> = vec![[0u64; 4]; batch_size];

    loop {
        // Check iteration limit (count batches, not individual keys)
        if iterations > 0 && batch_count >= iterations {
            break;
        }

        // Generate random base keys
        for i in 0..batch_size {
            rng.fill_bytes(&mut privkey_bytes[i]);
            privkeys_u64[i] = bytes_to_u64x4(&privkey_bytes[i]);
        }

        // Call production kernel
        let matches = match generate_pubkeys_with_prefix_match(
            &ctx,
            &privkeys_u64,
            &prefix_bits,
            max_matches,
            threads_per_block,
        ) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("❌ GPU kernel error: {}", e);
                errors += 1;
                batch_count += 1;
                continue;
            }
        };

        // Verify each match
        for m in &matches {
            // Reconstruct the actual secret key
            // actual_key = (base_key + offset) * λ^endo_type mod n
            let base_key = &privkeys_u64[m.base_idx as usize];
            let key_with_offset = add_u64x4_scalar(base_key, m.offset);
            let actual_key_u64 = adjust_privkey_for_endomorphism(&key_with_offset, m.endo_type);
            let actual_key_bytes = u64x4_to_bytes(&actual_key_u64);

            // Create secret key (may fail if >= n, which is rare but possible)
            let sk = match SecretKey::from_slice(&actual_key_bytes) {
                Ok(sk) => sk,
                Err(_) => {
                    // This can happen rarely when the adjusted key >= n
                    // Not an error, just skip this match
                    continue;
                }
            };

            // Compute public key using CPU
            let cpu_pk = sk.public_key(&secp);
            let cpu_pk_bytes = cpu_pk.serialize_uncompressed();
            let cpu_px_bytes: [u8; 32] = cpu_pk_bytes[1..33].try_into().unwrap();
            let cpu_px = bytes_to_u64x4(&cpu_px_bytes);

            // Compare GPU pubkey_x with CPU result
            if m.pubkey_x != cpu_px {
                eprintln!("❌ PUBKEY MISMATCH!");
                eprintln!(
                    "  Batch: {}, base_idx: {}, offset: {}, endo_type: {}",
                    batch_count, m.base_idx, m.offset, m.endo_type
                );
                eprintln!(
                    "  Base key:   {:016x}{:016x}{:016x}{:016x}",
                    base_key[3], base_key[2], base_key[1], base_key[0]
                );
                eprintln!(
                    "  Actual key: {:016x}{:016x}{:016x}{:016x}",
                    actual_key_u64[3], actual_key_u64[2], actual_key_u64[1], actual_key_u64[0]
                );
                eprintln!(
                    "  GPU Px:     {:016x}{:016x}{:016x}{:016x}",
                    m.pubkey_x[3], m.pubkey_x[2], m.pubkey_x[1], m.pubkey_x[0]
                );
                eprintln!(
                    "  CPU Px:     {:016x}{:016x}{:016x}{:016x}",
                    cpu_px[3], cpu_px[2], cpu_px[1], cpu_px[0]
                );
                errors += 1;
                continue;
            }

            // Verify npub - with 32 prefixes, every valid npub should match one
            let npub = pubkey_to_npub(&cpu_pk);
            let npub_body = &npub[5..]; // Remove "npub1"

            // Verify the first character is a valid bech32 char (sanity check)
            let first_char = npub_body.chars().next().unwrap_or('?');
            if !bech32_chars.contains(first_char) {
                eprintln!("❌ INVALID BECH32 CHAR!");
                eprintln!(
                    "  Batch: {}, base_idx: {}, offset: {}, endo_type: {}",
                    batch_count, m.base_idx, m.offset, m.endo_type
                );
                eprintln!("  npub: {}", npub);
                eprintln!("  First char '{}' not in bech32 charset", first_char);
                errors += 1;
                continue;
            }

            total_matches_verified += 1;
        }

        batch_count += 1;
        total_keys_checked += batch_size as u64 * keys_per_thread as u64 * 3;

        // Progress display (every 100 batches)
        if batch_count.is_multiple_of(100) {
            let elapsed_secs = start.elapsed().as_secs_f64();
            let keys_per_sec = total_keys_checked as f64 / elapsed_secs;
            println!(
                "✅ {} batches, {} keys checked, {} matches verified ({:.2}K keys/sec, {} errors)",
                batch_count,
                total_keys_checked,
                total_matches_verified,
                keys_per_sec / 1_000.0,
                errors
            );
        }
    }

    // Final summary
    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let keys_per_sec = total_keys_checked as f64 / elapsed_secs;

    println!("\n🎉 Verification complete!");
    println!("Total batches: {}", batch_count);
    println!("Total keys checked: {}", total_keys_checked);
    println!("Total matches verified: {}", total_matches_verified);
    println!("Errors: {}", errors);
    println!("Elapsed: {:.2} sec", elapsed_secs);
    println!("Rate: {:.2}K keys/sec", keys_per_sec / 1_000.0);

    if errors > 0 {
        eprintln!("\n⚠️  {} errors detected!", errors);
        std::process::exit(1);
    } else {
        println!(
            "\n✅ All {} matches verified correctly!",
            total_matches_verified
        );
    }

    Ok(())
}
