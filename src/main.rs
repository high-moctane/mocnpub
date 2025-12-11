use clap::{Parser, Subcommand};
use secp256k1::rand::{self, RngCore};
use secp256k1::{PublicKey, Secp256k1, SecretKey};
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Instant;

// Import common functions from lib.rs
use mocnpub_main::gpu::{
    calculate_optimal_batch_size, generate_pubkeys_with_prefix_match, get_sm_count, init_gpu,
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
    /// Mine npub with specified prefix
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

        /// Number of threads (default: auto-detect CPU cores)
        #[arg(short, long)]
        threads: Option<usize>,

        /// Number of keys to find (0 = unlimited, default: 1)
        #[arg(short, long, default_value = "1")]
        limit: usize,

        /// Enable GPU mode (use CUDA for fast mining)
        #[arg(long)]
        gpu: bool,

        /// GPU batch size (default: 3584000, 400 waves)
        #[arg(long, default_value = "3584000")]
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

/// Get keys_per_thread value determined at build time
/// Can be specified via MAX_KEYS_PER_THREAD env var (default: 1408)
fn get_max_keys_per_thread() -> u32 {
    env!("MAX_KEYS_PER_THREAD")
        .parse()
        .expect("MAX_KEYS_PER_THREAD must be a valid u32")
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Mine {
            prefix,
            output,
            threads,
            limit,
            gpu,
            batch_size,
            threads_per_block,
        } => run_mine(
            prefix,
            output,
            threads,
            limit,
            gpu,
            batch_size,
            threads_per_block,
        ),
        Commands::Verify { iterations } => run_verify(iterations),
    }
}

/// Run the mine subcommand
fn run_mine(
    prefix: String,
    output: Option<String>,
    threads: Option<usize>,
    limit: usize,
    gpu: bool,
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

    // Determine thread count (from args or auto-detect CPU cores)
    let num_threads = threads.unwrap_or_else(num_cpus::get);

    println!("🔥 mocnpub - Nostr npub mining 🔥");
    if prefixes.len() == 1 {
        println!("Prefix: '{}'", prefixes[0]);
    } else {
        println!("Prefixes (OR): {}", prefixes.join(", "));
    }

    // Branch based on GPU or CPU mode
    if gpu {
        let keys_per_thread = get_max_keys_per_thread();
        println!("Mode: GPU (CUDA) 🚀");
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
        return run_gpu_mining(
            &prefixes,
            limit,
            batch_size,
            threads_per_block,
            keys_per_thread,
            output.as_deref(),
        );
    }

    println!("Mode: CPU");
    println!("Threads: {}", num_threads);
    println!(
        "Limit: {}\n",
        if limit == 0 {
            "unlimited".to_string()
        } else {
            limit.to_string()
        }
    );

    // Shared counters across all threads
    let total_count = Arc::new(AtomicU64::new(0));
    let found_count = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    // Share prefixes via Arc
    let prefixes = Arc::new(prefixes);

    // Create channel (worker threads → main thread)
    // (SecretKey, PublicKey, npub, matched_prefix, attempt_count)
    let (sender, receiver) = mpsc::channel::<(SecretKey, PublicKey, String, String, u64)>();

    // Spawn threads
    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let prefixes = Arc::clone(&prefixes);
            let total_count = Arc::clone(&total_count);
            let found_count = Arc::clone(&found_count);
            let sender = sender.clone();
            let limit = limit;

            std::thread::spawn(move || {
                let secp = Secp256k1::new();
                let mut local_count = 0u64;

                loop {
                    // Exit loop if limit reached (0 = unlimited, never exit)
                    if limit > 0 && found_count.load(Ordering::Relaxed) >= limit {
                        break;
                    }

                    let (sk, pk) = secp.generate_keypair(&mut rand::thread_rng());
                    local_count += 1;

                    // Convert to bech32 format
                    let npub = pubkey_to_npub(&pk);
                    // Remove "npub1" to get bech32 body only
                    let npub_body = &npub[5..]; // "npub1" is 5 chars

                    // Check if any prefix matches (OR logic)
                    if let Some(matched_prefix) =
                        prefixes.iter().find(|p| npub_body.starts_with(p.as_str()))
                    {
                        // Increment found count
                        let count = found_count.fetch_add(1, Ordering::Relaxed) + 1;

                        // Get current attempt count
                        let current_total = total_count.load(Ordering::Relaxed) + local_count;

                        // Send result via channel (including matched_prefix)
                        if sender
                            .send((sk, pk, npub.clone(), matched_prefix.clone(), current_total))
                            .is_err()
                        {
                            // Main thread has terminated
                            break;
                        }

                        // Exit loop if limit reached (0 = unlimited, never exit)
                        if limit > 0 && count >= limit {
                            break;
                        }
                    }

                    // Update global counter periodically (every 100 iterations)
                    if local_count.is_multiple_of(100) {
                        total_count.fetch_add(100, Ordering::Relaxed);
                    }
                }

                // Add remaining count at the end
                let remainder = local_count % 100;
                if remainder > 0 {
                    total_count.fetch_add(remainder, Ordering::Relaxed);
                }
            })
        })
        .collect();

    // Drop sender (receiver returns None when all worker threads finish)
    drop(sender);

    // Progress display thread
    let total_count_progress = Arc::clone(&total_count);
    let found_count_progress = Arc::clone(&found_count);
    let limit_progress = limit;
    let progress_handle = std::thread::spawn(move || {
        loop {
            // Exit if limit reached (0 = unlimited, never exit)
            if limit_progress > 0 && found_count_progress.load(Ordering::Relaxed) >= limit_progress
            {
                break;
            }
            std::thread::sleep(std::time::Duration::from_secs(1));
            let count = total_count_progress.load(Ordering::Relaxed);
            let found = found_count_progress.load(Ordering::Relaxed);
            if count > 0 {
                println!("{} attempts... (found: {})", count, found);
            }
        }
    });

    // Prepare file output (append mode)
    let mut output_file = if let Some(ref output_path) = output {
        Some(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(output_path)?,
        )
    } else {
        None
    };

    // Receive and output results on main thread
    let mut result_count = 0;
    while let Ok((sk, pk, npub, matched_prefix, current_total)) = receiver.recv() {
        result_count += 1;
        let elapsed = start.elapsed();
        let elapsed_secs = elapsed.as_secs_f64();
        let keys_per_sec = current_total as f64 / elapsed_secs;

        let nsec = seckey_to_nsec(&sk);
        let pk_hex = pk.to_string();
        let pk_x_only = &pk_hex[2..]; // x-coord only (remove first 2 chars of compressed format)

        // Format result
        let output_text = format!(
            "✅ Found #{}! ({} attempts, {} threads)\n\
             Matched prefix: '{}'\n\n\
             Elapsed: {:.2} sec\n\
             Performance: {:.2} keys/sec\n\n\
             Secret key (hex): {}\n\
             Secret key (nsec): {}\n\
             Public key (compressed): {}\n\
             Public key (x-coord): {}\n\
             Public key (npub): {}\n\
{}\n",
            result_count,
            current_total,
            num_threads,
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

        // Output to appropriate destination
        if let Some(ref mut file) = output_file {
            // Append to file
            file.write_all(output_text.as_bytes())?;
            file.flush()?;
        }
        // Also output to stdout (regardless of file output)
        print!("{}", output_text);
        io::stdout().flush()?;
    }

    // Wait for all threads to finish
    for handle in handles {
        handle.join().unwrap();
    }
    progress_handle.join().unwrap();

    // Display final results
    let final_count = total_count.load(Ordering::Relaxed);
    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    println!("\n🎉 Mining complete!");
    println!("Keys found: {}", result_count);
    println!("Total attempts: {}", final_count);
    println!("Elapsed: {:.2} sec", elapsed_secs);
    if let Some(ref output_path) = output {
        println!("Results saved to: {}", output_path);
    }

    Ok(())
}

/// GPU mining mode (GPU-side prefix matching)
fn run_gpu_mining(
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

    // Secret key buffers (base keys)
    let mut privkey_bytes: Vec<[u8; 32]> = vec![[0u8; 32]; batch_size];
    let mut privkeys_u64: Vec<[u64; 4]> = vec![[0u64; 4]; batch_size];

    // Main loop
    loop {
        // 1. Generate random base keys (CPU)
        for i in 0..batch_size {
            rng.fill_bytes(&mut privkey_bytes[i]);
            privkeys_u64[i] = bytes_to_u64x4(&privkey_bytes[i]);
        }

        // 2. GPU public key generation + prefix matching
        // Note: keys_per_thread is fixed to MAX_KEYS_PER_THREAD at compile time
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
                std::process::exit(1);
            }
        };

        // Update attempt count (endomorphism checks 3 X-coordinates)
        total_count += (batch_size as u64) * (keys_per_thread as u64) * 3;

        // 3. Process matched results
        for m in matches {
            found_count += 1;

            // Recover secret key: (base_key + offset) * λ^endo_type mod n
            // endo_type: 0 = original, 1 = λ*k, 2 = λ²*k (for endomorphism)
            let base_key = &privkeys_u64[m.base_idx as usize];
            let key_with_offset = add_u64x4_scalar(base_key, m.offset);
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
        if batch_count % 100 == 0 {
            let elapsed_secs = start.elapsed().as_secs_f64();
            let keys_per_sec = total_keys_checked as f64 / elapsed_secs;
            println!(
                "✅ {} batches, {} keys checked, {} matches verified ({:.2}M keys/sec, {} errors)",
                batch_count,
                total_keys_checked,
                total_matches_verified,
                keys_per_sec / 1_000_000.0,
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
    println!("Rate: {:.2}M keys/sec", keys_per_sec / 1_000_000.0);

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
