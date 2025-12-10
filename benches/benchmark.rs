use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mocnpub_main::gpu::{
    generate_pubkeys_batch, generate_pubkeys_sequential_batch,
    generate_pubkeys_sequential_montgomery_batch, init_gpu, test_mod_mult_gpu, test_mod_square_gpu,
};
use mocnpub_main::{pubkey_to_npub, seckey_to_nsec, validate_prefix};
use secp256k1::{Secp256k1, rand};
use std::hint::black_box;

/// Benchmark: Key generation performance
///
/// Measures secp256k1 keypair generation speed
fn bench_keypair_generation(c: &mut Criterion) {
    let secp = Secp256k1::new();

    c.bench_function("keypair_generation", |b| {
        b.iter(|| {
            let (sk, pk) = secp.generate_keypair(&mut rand::thread_rng());
            black_box((sk, pk))
        })
    });
}

/// Benchmark: Public key to npub conversion
///
/// Measures bech32 encoding performance
fn bench_pubkey_to_npub(c: &mut Criterion) {
    let secp = Secp256k1::new();
    let (_sk, pk) = secp.generate_keypair(&mut rand::thread_rng());

    c.bench_function("pubkey_to_npub", |b| {
        b.iter(|| {
            let npub = pubkey_to_npub(black_box(&pk));
            black_box(npub)
        })
    });
}

/// Benchmark: Secret key to nsec conversion
///
/// Measures bech32 encoding performance
fn bench_seckey_to_nsec(c: &mut Criterion) {
    let secp = Secp256k1::new();
    let (sk, _pk) = secp.generate_keypair(&mut rand::thread_rng());

    c.bench_function("seckey_to_nsec", |b| {
        b.iter(|| {
            let nsec = seckey_to_nsec(black_box(&sk));
            black_box(nsec)
        })
    });
}

/// Benchmark: Prefix matching
///
/// Measures npub prefix matching performance
fn bench_prefix_matching(c: &mut Criterion) {
    let secp = Secp256k1::new();
    let (_sk, pk) = secp.generate_keypair(&mut rand::thread_rng());
    let npub = pubkey_to_npub(&pk);
    let npub_body = &npub[5..]; // Remove "npub1"
    let prefix = "test";

    c.bench_function("prefix_matching", |b| {
        b.iter(|| {
            let matches = black_box(npub_body).starts_with(black_box(prefix));
            black_box(matches)
        })
    });
}

/// Benchmark: Prefix validation
///
/// Measures validate_prefix() performance
fn bench_validate_prefix(c: &mut Criterion) {
    let prefix = "m0ctane";

    c.bench_function("validate_prefix", |b| {
        b.iter(|| {
            let result = validate_prefix(black_box(prefix));
            black_box(result)
        })
    });
}

/// Benchmark: Complete mining cycle
///
/// Measures the full flow: key generation → npub conversion → prefix matching
/// (Similar to the actual mining loop)
fn bench_mining_cycle(c: &mut Criterion) {
    let secp = Secp256k1::new();
    let prefix = "test";

    c.bench_function("mining_cycle", |b| {
        b.iter(|| {
            let (_sk, pk) = secp.generate_keypair(&mut rand::thread_rng());
            let npub = pubkey_to_npub(&pk);
            let npub_body = &npub[5..];
            let matches = npub_body.starts_with(prefix);
            black_box(matches)
        })
    });
}

// ============================================================================
// GPU Benchmarks
// ============================================================================

/// GPU Benchmark: Batch vs Sequential vs Montgomery
///
/// Compares three methods under the same conditions
fn bench_gpu_methods(c: &mut Criterion) {
    let ctx = init_gpu().expect("Failed to initialize GPU");

    let mut group = c.benchmark_group("gpu_methods");

    // Test configurations: (num_threads, keys_per_thread)
    let configs = [
        // Small scale (warm-up)
        (256, 64),  // 16,384 keys
        (256, 256), // 65,536 keys
        // Medium scale
        (1024, 64),   // 65,536 keys
        (1024, 256),  // 262,144 keys
        (1024, 1024), // 1,048,576 keys (1M keys!)
        // Large scale (10000 consecutive keys!)
        (1024, 4096),  // 4,194,304 keys (4M keys!)
        (1024, 10000), // 10,240,000 keys (10M keys!) 🔥
        // Increase thread count
        (2048, 256),  // 524,288 keys
        (2048, 1024), // 2,097,152 keys (2M keys!)
        (4096, 256),  // 1,048,576 keys (1M keys!)
    ];

    for (num_threads, keys_per_thread) in configs {
        let total_keys = num_threads * keys_per_thread;

        // Prepare keys for Batch
        let batch_keys: Vec<[u64; 4]> =
            (2..(2 + total_keys as u64)).map(|k| [k, 0, 0, 0]).collect();

        // Prepare base keys for Sequential/Montgomery
        let base_keys: Vec<[u64; 4]> = (0..num_threads)
            .map(|i| [2 + (i * keys_per_thread) as u64, 0, 0, 0])
            .collect();

        let config_name = format!("{}t_{}k", num_threads, keys_per_thread);

        // Batch (Full PointMult per key)
        group.bench_with_input(
            BenchmarkId::new("batch", &config_name),
            &(&batch_keys,),
            |b, (keys,)| b.iter(|| generate_pubkeys_batch(&ctx, black_box(keys)).unwrap()),
        );

        // Sequential (Phase 1: PointAddMixed + JacobianToAffine per key)
        group.bench_with_input(
            BenchmarkId::new("sequential", &config_name),
            &(&base_keys, keys_per_thread as u32),
            |b, (keys, kpt)| {
                b.iter(|| generate_pubkeys_sequential_batch(&ctx, black_box(keys), *kpt).unwrap())
            },
        );

        // Montgomery (Phase 2: PointAddMixed + batch inverse)
        group.bench_with_input(
            BenchmarkId::new("montgomery", &config_name),
            &(&base_keys, keys_per_thread as u32),
            |b, (keys, kpt)| {
                b.iter(|| {
                    generate_pubkeys_sequential_montgomery_batch(&ctx, black_box(keys), *kpt)
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// GPU Benchmark: Investigate keys_per_thread impact
///
/// How Montgomery's Trick effectiveness changes with keys_per_thread
fn bench_gpu_keys_per_thread(c: &mut Criterion) {
    let ctx = init_gpu().expect("Failed to initialize GPU");

    let mut group = c.benchmark_group("gpu_keys_per_thread");

    let num_threads = 1024; // More practical thread count
    // 10K → 100K consecutive keys! How far can it scale?
    let keys_per_thread_options = [10000, 20000, 50000, 100000];

    for keys_per_thread in keys_per_thread_options {
        let base_keys: Vec<[u64; 4]> = (0..num_threads)
            .map(|i| [2 + (i * keys_per_thread) as u64, 0, 0, 0])
            .collect();

        let config_name = format!("{}", keys_per_thread);

        // Montgomery only (Phase 2)
        group.bench_with_input(
            BenchmarkId::new("montgomery", &config_name),
            &(&base_keys, keys_per_thread as u32),
            |b, (keys, kpt)| {
                b.iter(|| {
                    generate_pubkeys_sequential_montgomery_batch(&ctx, black_box(keys), *kpt)
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// GPU Benchmark: _ModSquare vs _ModMult
///
/// Compare squaring (a²) vs multiplication (a*a) speed
fn bench_mod_square_vs_mult(c: &mut Criterion) {
    let ctx = init_gpu().expect("Failed to initialize GPU");

    let mut group = c.benchmark_group("mod_square_vs_mult");

    // Test value: moderately large number
    let a = [
        0x123456789ABCDEFu64,
        0xFEDCBA9876543210u64,
        0x1111111111111111u64,
        0x2222222222222222u64,
    ];

    // _ModSquare: a²
    group.bench_function("mod_square", |b| {
        b.iter(|| test_mod_square_gpu(&ctx, black_box(&a)).unwrap())
    });

    // _ModMult: a * a（現在の _ModSquare の内部実装と同等）
    group.bench_function("mod_mult_self", |b| {
        b.iter(|| test_mod_mult_gpu(&ctx, black_box(&a), black_box(&a)).unwrap())
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_keypair_generation,
    bench_pubkey_to_npub,
    bench_seckey_to_nsec,
    bench_prefix_matching,
    bench_validate_prefix,
    bench_mining_cycle
);

criterion_group!(
    gpu_benches,
    bench_gpu_methods,
    bench_gpu_keys_per_thread,
    bench_mod_square_vs_mult
);

criterion_main!(benches, gpu_benches);
