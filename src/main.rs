use clap::Parser;
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
///
/// Mining tool to find npub (Nostr public key) with specified prefix.
#[derive(Parser, Debug)]
#[command(name = "mocnpub")]
#[command(about = "Nostr npub mining tool 🔑", long_about = None)]
struct Args {
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
}

/// Get keys_per_thread value determined at build time
/// Can be specified via MAX_KEYS_PER_THREAD env var (default: 1408)
fn get_max_keys_per_thread() -> u32 {
    env!("MAX_KEYS_PER_THREAD")
        .parse()
        .expect("MAX_KEYS_PER_THREAD must be a valid u32")
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    // Split prefix by comma and convert to Vec
    let prefixes: Vec<String> = args
        .prefix
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    // Validate each prefix
    for prefix in &prefixes {
        if let Err(err_msg) = validate_prefix(prefix) {
            eprintln!("❌ Error: {}", err_msg);
            std::process::exit(1);
        }
    }

    // Determine thread count (from args or auto-detect CPU cores)
    let num_threads = args.threads.unwrap_or_else(num_cpus::get);

    println!("🔥 mocnpub - Nostr npub mining 🔥");
    if prefixes.len() == 1 {
        println!("Prefix: '{}'", prefixes[0]);
    } else {
        println!("Prefixes (OR): {}", prefixes.join(", "));
    }

    // Branch based on GPU or CPU mode
    if args.gpu {
        let keys_per_thread = get_max_keys_per_thread();
        println!("Mode: GPU (CUDA) 🚀");
        println!("Batch size: {}", args.batch_size);
        println!(
            "Threads/block: {}, Keys/thread: {} (build-time)",
            args.threads_per_block, keys_per_thread
        );
        println!(
            "Limit: {}\n",
            if args.limit == 0 {
                "unlimited".to_string()
            } else {
                args.limit.to_string()
            }
        );
        return run_gpu_mining(
            &prefixes,
            args.limit,
            args.batch_size,
            args.threads_per_block,
            keys_per_thread,
            args.output.as_deref(),
        );
    }

    println!("Mode: CPU");
    println!("Threads: {}", num_threads);
    println!(
        "Limit: {}\n",
        if args.limit == 0 {
            "unlimited".to_string()
        } else {
            args.limit.to_string()
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
            let limit = args.limit;

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
    let limit_progress = args.limit;
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
    let mut output_file = if let Some(ref output_path) = args.output {
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
    if let Some(ref output_path) = args.output {
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
