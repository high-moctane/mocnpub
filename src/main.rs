use clap::{Parser, Subcommand};
use secp256k1::rand::{self, RngCore};
use secp256k1::{Secp256k1, SecretKey};
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Instant;

// Import common functions from lib.rs
use mocnpub_main::gpu::{
    SequentialTripleBufferMiner, calculate_optimal_batch_size, get_max_keys_per_thread,
    get_sm_count, init_gpu,
};
use mocnpub_main::{adjust_privkey_for_endomorphism, prefixes_to_bits};
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

        /// Number of parallel miners (each with 3 streams = triple buffering)
        /// More miners may improve GPU utilization, like make -j
        #[arg(long, default_value = "2")]
        miners: usize,
    },

    /// Verify GPU calculations against CPU (fuzzing-like testing)
    Verify {
        /// Number of iterations (0 = unlimited, run until Ctrl+C)
        #[arg(short, long, default_value = "0")]
        iterations: u64,

        /// Batch size in threads (default: 256 = 2 blocks for blockIdx verification)
        #[arg(long, default_value = "256")]
        batch_size: usize,

        /// Number of miners (default: CPU logical core count)
        #[arg(long)]
        miners: Option<usize>,
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
            miners,
        } => run_mine(prefix, output, limit, batch_size, threads_per_block, miners),
        Commands::Verify {
            iterations,
            batch_size,
            miners,
        } => run_verify(iterations, batch_size, miners),
    }
}

/// Run the mine subcommand
fn run_mine(
    prefix: String,
    output: Option<String>,
    limit: usize,
    batch_size: usize,
    threads_per_block: u32,
    miners: usize,
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
    println!("Parallel miners: {} (× 3 streams each)", miners);
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
        miners,
    )
}

/// Match result with miner ID for multi-miner setup
struct MinerMatch {
    miner_id: usize,
    gpu_match: mocnpub_main::gpu::GpuMatch,
}

/// Main mining loop (GPU-side prefix matching with parallel miners)
fn mining_loop(
    prefixes: &[String],
    limit: usize,
    batch_size: usize,
    threads_per_block: u32,
    keys_per_thread: u32,
    output_path: Option<&str>,
    num_miners: usize,
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

    // Shared state for parallel miners
    let total_count = Arc::new(AtomicU64::new(0));
    let found_count = Arc::new(AtomicUsize::new(0));
    let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Channel for results from miners
    let (tx, rx) = mpsc::channel::<MinerMatch>();

    // Prepare file output (append mode)
    let mut output_file = if let Some(path) = output_path {
        Some(OpenOptions::new().create(true).append(true).open(path)?)
    } else {
        None
    };

    // Parameter settings
    let max_matches: u32 = 1000; // generous buffer
    // * 3 for endomorphism: each key generates 3 X-coordinates (P, β*P, β²*P)
    let batch_keys = (batch_size as u64) * (keys_per_thread as u64) * 3;

    // Create and launch parallel miners
    let mut handles = Vec::new();

    for miner_id in 0..num_miners {
        let ctx = ctx.clone();
        let prefix_bits = prefix_bits.clone();
        let tx = tx.clone();
        let total_count = total_count.clone();
        let stop_flag = stop_flag.clone();

        let handle = std::thread::spawn(move || {
            // Create miner in this thread
            let mut miner = match SequentialTripleBufferMiner::new(
                &ctx,
                &prefix_bits,
                max_matches,
                threads_per_block,
                batch_size,
            ) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("❌ Miner {} failed to initialize: {}", miner_id, e);
                    return;
                }
            };

            let mut rng = rand::thread_rng();
            let mut host_u64: [[u64; 4]; 3] = [[0u64; 4]; 3];

            // Launch initial batches
            #[allow(clippy::needless_range_loop)]
            for buf_idx in 0..3 {
                let mut key_bytes = [0u8; 32];
                rng.fill_bytes(&mut key_bytes);
                host_u64[buf_idx] = bytes_to_u64x4(&key_bytes);
                if let Err(e) = miner.launch_single(buf_idx, &host_u64[buf_idx]) {
                    eprintln!("❌ Miner {} launch error: {}", miner_id, e);
                    return;
                }
            }

            let mut collect_idx = 0usize;

            // Mining loop
            while !stop_flag.load(Ordering::Relaxed) {
                // Collect results
                let matches = match miner.collect_single(collect_idx) {
                    Ok(result) => result,
                    Err(e) => {
                        eprintln!("❌ Miner {} collect error: {}", miner_id, e);
                        break;
                    }
                };

                // Update total count
                total_count.fetch_add(batch_keys, Ordering::Relaxed);

                // Send matches to main thread
                for m in matches {
                    if tx
                        .send(MinerMatch {
                            miner_id,
                            gpu_match: m,
                        })
                        .is_err()
                    {
                        // Receiver dropped, stop
                        return;
                    }
                }

                // Generate new key and launch
                let mut key_bytes = [0u8; 32];
                rng.fill_bytes(&mut key_bytes);
                host_u64[collect_idx] = bytes_to_u64x4(&key_bytes);
                if let Err(e) = miner.launch_single(collect_idx, &host_u64[collect_idx]) {
                    eprintln!("❌ Miner {} launch error: {}", miner_id, e);
                    break;
                }

                collect_idx = (collect_idx + 1) % 3;
            }
        });

        handles.push(handle);
    }

    // Drop our copy of tx so rx.iter() will end when all miners stop
    drop(tx);

    println!(
        "✅ {} parallel miner(s) initialized ({}×3 = {} streams total)",
        num_miners,
        num_miners,
        num_miners * 3
    );
    println!("🎯 Mining started with parallel triple buffering");

    // Progress tracking
    let mut last_progress_count: u64 = 0;

    // Main thread: process results
    loop {
        // Check for matches with timeout
        match rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(miner_match) => {
                let m = miner_match.gpu_match;
                let current_found = found_count.fetch_add(1, Ordering::Relaxed) + 1;

                // Adjust for endomorphism
                let actual_key_u64 = adjust_privkey_for_endomorphism(&m.base_key, m.endo_type);
                let actual_key_bytes = u64x4_to_bytes(&actual_key_u64);

                let sk = SecretKey::from_slice(&actual_key_bytes).expect("Invalid secret key");
                let nsec = seckey_to_nsec(&sk);

                let secp = Secp256k1::new();
                let pk = sk.public_key(&secp);
                let pk_hex = pk.to_string();
                // Skip first 2 chars ("02" or "03" prefix) to get X-coordinate only
                let pk_x_only = &pk_hex[2..];

                let npub = pubkey_to_npub(&pk);
                let npub_body = &npub[5..];

                // Verify GPU result
                let gpu_npub = pubkey_bytes_to_npub(&u64x4_to_bytes(&m.pubkey_x));
                if npub != gpu_npub {
                    eprintln!(
                        "⚠️  Warning: GPU pubkey_x mismatch! miner={}, endo_type={}",
                        miner_match.miner_id, m.endo_type
                    );
                    eprintln!("    GPU npub:    {}", gpu_npub);
                    eprintln!("    Actual npub: {}", npub);
                }

                let matched_prefix = prefixes
                    .iter()
                    .find(|p| npub_body.starts_with(p.as_str()))
                    .map(|p| p.as_str())
                    .unwrap_or("?");

                let current_total = total_count.load(Ordering::Relaxed);
                let elapsed = start.elapsed();
                let elapsed_secs = elapsed.as_secs_f64();
                let keys_per_sec = current_total as f64 / elapsed_secs;

                let output_text = format!(
                    "✅ Found #{}! ({} attempts, miner #{})\n\
                     Matched prefix: '{}'\n\n\
                     Elapsed: {:.2} sec\n\
                     Performance: {:.2} keys/sec\n\n\
                     Secret key (hex): {}\n\
                     Secret key (nsec): {}\n\
                     Public key (compressed): {}\n\
                     Public key (x-coord): {}\n\
                     Public key (npub): {}\n\
{}\n",
                    current_found,
                    current_total,
                    miner_match.miner_id,
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

                if let Some(ref mut file) = output_file {
                    file.write_all(output_text.as_bytes())?;
                    file.flush()?;
                }
                print!("{}", output_text);
                io::stdout().flush()?;

                // Check limit
                if limit > 0 && current_found >= limit {
                    stop_flag.store(true, Ordering::Relaxed);
                    break;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // No match, check progress
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // All miners stopped
                break;
            }
        }

        // Progress display (every ~10 batches worth of keys, regardless of num_miners)
        let current_total = total_count.load(Ordering::Relaxed);
        if current_total >= last_progress_count + batch_keys * 10 {
            last_progress_count = current_total;
            let elapsed_secs = start.elapsed().as_secs_f64();
            let keys_per_sec = current_total as f64 / elapsed_secs;
            let mins = (elapsed_secs / 60.0) as u64;
            let secs = (elapsed_secs % 60.0) as u64;
            let current_found = found_count.load(Ordering::Relaxed);
            println!(
                "{} attempts... ({}:{:02}, {:.2} keys/sec, found: {}, {} miners)",
                current_total, mins, secs, keys_per_sec, current_found, num_miners
            );
        }
    }

    // Wait for all miner threads
    for handle in handles {
        let _ = handle.join();
    }

    // Final summary
    let final_elapsed = start.elapsed();
    let final_elapsed_secs = final_elapsed.as_secs_f64();
    let final_total = total_count.load(Ordering::Relaxed);
    let final_found = found_count.load(Ordering::Relaxed);

    println!("\n🎉 GPU mining complete!");
    println!("Keys found: {}", final_found);
    println!("Total attempts: {}", final_total);
    println!("Elapsed: {:.2} sec", final_elapsed_secs);
    println!(
        "Performance: {:.2} keys/sec",
        final_total as f64 / final_elapsed_secs
    );
    if let Some(path) = output_path {
        println!("Results saved to: {}", path);
    }

    Ok(())
}

/// Run the verify subcommand (GPU vs CPU verification using production kernel)
///
/// This verifies the full production pipeline:
/// - Montgomery's Trick (batch inverse)
/// - Sequential key strategy (base_key + offset)
/// - Endomorphism (β, β² for 3x X-coordinate checks)
/// - GPU-side prefix matching
///
/// Uses SequentialTripleBufferMiner with small batch_size and multiple miners
/// to reduce atomic contention while keeping GPU saturated.
fn run_verify(iterations: u64, batch_size: usize, miners_opt: Option<usize>) -> io::Result<()> {
    println!("🔬 mocnpub - GPU Verification Mode (Production Kernel) 🔬");
    println!(
        "Iterations: {}\n",
        if iterations == 0 {
            "unlimited (Ctrl+C to stop)".to_string()
        } else {
            iterations.to_string()
        }
    );

    // Initialize GPU (init_gpu already returns Arc<CudaContext>)
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

    // Small batch size to reduce atomic contention (2 blocks minimum for blockIdx verification)
    let threads_per_block: u32 = 128;
    // Default miners to CPU logical core count
    let num_miners: usize = miners_opt.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(16)
    });
    // max_matches per miner: batch_size * keys_per_thread * 3 (endo) + buffer
    let max_matches: u32 = (batch_size as u32) * keys_per_thread * 3 + 1000;

    println!(
        "Batch size: {} threads × {} keys/thread × 3 (endo) = {} keys/batch",
        batch_size,
        keys_per_thread,
        batch_size as u64 * keys_per_thread as u64 * 3
    );
    println!("Miners: {} (each with triple buffering)", num_miners);
    println!("Starting verification...\n");

    // Shared counters across all threads
    let total_batches = Arc::new(AtomicU64::new(0));
    let total_keys_checked = Arc::new(AtomicU64::new(0));
    let total_matches_verified = Arc::new(AtomicU64::new(0));
    let total_errors = Arc::new(AtomicU64::new(0));
    let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let start = Instant::now();

    // Spawn multiple miner threads
    std::thread::scope(|s| {
        for miner_id in 0..num_miners {
            let ctx = Arc::clone(&ctx);
            let prefix_bits = prefix_bits.clone();
            let total_batches = Arc::clone(&total_batches);
            let total_keys_checked = Arc::clone(&total_keys_checked);
            let total_matches_verified = Arc::clone(&total_matches_verified);
            let total_errors = Arc::clone(&total_errors);
            let stop_flag = Arc::clone(&stop_flag);

            s.spawn(move || {
                // Create miner for this thread
                let mut miner = match SequentialTripleBufferMiner::new(
                    &ctx,
                    &prefix_bits,
                    max_matches,
                    threads_per_block,
                    batch_size,
                ) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("❌ Miner {} creation failed: {}", miner_id, e);
                        return;
                    }
                };

                let secp = Secp256k1::new();
                let mut rng = rand::thread_rng();

                // Base keys for triple buffering (one per buffer)
                let mut base_keys: [[u64; 4]; 3] = [[0u64; 4]; 3];

                // Initial launches for all 3 buffers
                #[allow(clippy::needless_range_loop)]
                for buf_idx in 0..3 {
                    let mut key_bytes = [0u8; 32];
                    rng.fill_bytes(&mut key_bytes);
                    base_keys[buf_idx] = bytes_to_u64x4(&key_bytes);

                    if let Err(e) = miner.launch_single(buf_idx, &base_keys[buf_idx]) {
                        eprintln!("❌ Miner {} launch failed: {}", miner_id, e);
                        return;
                    }
                }

                let mut current_buf = 0usize;

                loop {
                    // Check stop conditions
                    if stop_flag.load(Ordering::Relaxed) {
                        break;
                    }
                    if iterations > 0 && total_batches.load(Ordering::Relaxed) >= iterations {
                        stop_flag.store(true, Ordering::Relaxed);
                        break;
                    }

                    // Collect results from current buffer
                    let matches = match miner.collect_single(current_buf) {
                        Ok(m) => m,
                        Err(e) => {
                            eprintln!("❌ Miner {} collect failed: {}", miner_id, e);
                            total_errors.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                    };

                    // Launch next batch immediately (before CPU verification)
                    // Skip re-launch if we're stopping
                    if !stop_flag.load(Ordering::Relaxed) {
                        let mut new_key_bytes = [0u8; 32];
                        rng.fill_bytes(&mut new_key_bytes);
                        base_keys[current_buf] = bytes_to_u64x4(&new_key_bytes);

                        if let Err(e) = miner.launch_single(current_buf, &base_keys[current_buf]) {
                            eprintln!("❌ Miner {} re-launch failed: {}", miner_id, e);
                            total_errors.fetch_add(1, Ordering::Relaxed);
                        }
                    }

                    // CPU verification of collected matches
                    let mut local_verified = 0u64;
                    let mut local_errors = 0u64;

                    for m in &matches {
                        // For SequentialTripleBufferMiner, base_key is stored in m.base_key
                        // and offset is the key index within the thread's range
                        let actual_key_u64 =
                            adjust_privkey_for_endomorphism(&m.base_key, m.endo_type);
                        let actual_key_bytes = u64x4_to_bytes(&actual_key_u64);

                        // Create secret key (may fail if >= n)
                        let sk = match SecretKey::from_slice(&actual_key_bytes) {
                            Ok(sk) => sk,
                            Err(_) => continue,
                        };

                        // Compute public key using CPU
                        let cpu_pk = sk.public_key(&secp);
                        let cpu_pk_bytes = cpu_pk.serialize_uncompressed();
                        let cpu_px_bytes: [u8; 32] = cpu_pk_bytes[1..33].try_into().unwrap();
                        let cpu_px = bytes_to_u64x4(&cpu_px_bytes);

                        // Compare GPU pubkey_x with CPU result
                        if m.pubkey_x != cpu_px {
                            eprintln!("❌ PUBKEY MISMATCH! (miner {})", miner_id);
                            eprintln!(
                                "  base_idx: {}, offset: {}, endo_type: {}",
                                m.base_idx, m.offset, m.endo_type
                            );
                            eprintln!(
                                "  GPU Px: {:016x}{:016x}{:016x}{:016x}",
                                m.pubkey_x[3], m.pubkey_x[2], m.pubkey_x[1], m.pubkey_x[0]
                            );
                            eprintln!(
                                "  CPU Px: {:016x}{:016x}{:016x}{:016x}",
                                cpu_px[3], cpu_px[2], cpu_px[1], cpu_px[0]
                            );
                            local_errors += 1;
                            continue;
                        }

                        // Verify npub first char is valid bech32
                        let npub = pubkey_to_npub(&cpu_pk);
                        let npub_body = &npub[5..];
                        let first_char = npub_body.chars().next().unwrap_or('?');
                        if !bech32_chars.contains(first_char) {
                            eprintln!("❌ INVALID BECH32 CHAR! (miner {})", miner_id);
                            eprintln!("  npub: {}", npub);
                            local_errors += 1;
                            continue;
                        }

                        local_verified += 1;
                    }

                    // Update global counters
                    total_batches.fetch_add(1, Ordering::Relaxed);
                    total_keys_checked.fetch_add(
                        batch_size as u64 * keys_per_thread as u64 * 3,
                        Ordering::Relaxed,
                    );
                    total_matches_verified.fetch_add(local_verified, Ordering::Relaxed);
                    total_errors.fetch_add(local_errors, Ordering::Relaxed);

                    // Move to next buffer
                    current_buf = (current_buf + 1) % 3;
                }
            });
        }

        // Progress display in main thread
        loop {
            std::thread::sleep(std::time::Duration::from_secs(2));

            let batches = total_batches.load(Ordering::Relaxed);
            let keys = total_keys_checked.load(Ordering::Relaxed);
            let verified = total_matches_verified.load(Ordering::Relaxed);
            let errors = total_errors.load(Ordering::Relaxed);

            if stop_flag.load(Ordering::Relaxed) {
                break;
            }
            if iterations > 0 && batches >= iterations {
                stop_flag.store(true, Ordering::Relaxed);
                break;
            }

            let elapsed_secs = start.elapsed().as_secs_f64();
            let keys_per_sec = keys as f64 / elapsed_secs;
            println!(
                "✅ {} batches, {} keys checked, {} matches verified ({:.2}M keys/sec, {} errors)",
                batches,
                keys,
                verified,
                keys_per_sec / 1_000_000.0,
                errors
            );
        }
    });

    // Final summary
    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let keys = total_keys_checked.load(Ordering::Relaxed);
    let verified = total_matches_verified.load(Ordering::Relaxed);
    let errors = total_errors.load(Ordering::Relaxed);
    let batches = total_batches.load(Ordering::Relaxed);
    let keys_per_sec = keys as f64 / elapsed_secs;

    println!("\n🎉 Verification complete!");
    println!("Total batches: {}", batches);
    println!("Total keys checked: {}", keys);
    println!("Total matches verified: {}", verified);
    println!("Errors: {}", errors);
    println!("Elapsed: {:.2} sec", elapsed_secs);
    println!("Rate: {:.2}M keys/sec", keys_per_sec / 1_000_000.0);

    if errors > 0 {
        eprintln!("\n⚠️  {} errors detected!", errors);
        std::process::exit(1);
    } else {
        println!("\n✅ All {} matches verified correctly!", verified);
    }

    Ok(())
}
