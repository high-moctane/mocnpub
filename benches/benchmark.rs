use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::hint::black_box;
use secp256k1::{rand, Secp256k1};
use mocnpub_main::{pubkey_to_npub, seckey_to_nsec, validate_prefix};
use mocnpub_main::gpu::{
    init_gpu, generate_pubkeys_batch,
    generate_pubkeys_sequential_batch,
    generate_pubkeys_sequential_montgomery_batch,
};

/// ベンチマーク: 鍵生成のパフォーマンス
///
/// secp256k1 の鍵生成がどれくらい速いかを測定
fn bench_keypair_generation(c: &mut Criterion) {
    let secp = Secp256k1::new();

    c.bench_function("keypair_generation", |b| {
        b.iter(|| {
            let (sk, pk) = secp.generate_keypair(&mut rand::thread_rng());
            black_box((sk, pk))
        })
    });
}

/// ベンチマーク: 公開鍵を npub に変換
///
/// bech32 エンコードのパフォーマンスを測定
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

/// ベンチマーク: 秘密鍵を nsec に変換
///
/// bech32 エンコードのパフォーマンスを測定
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

/// ベンチマーク: prefix マッチング
///
/// npub の prefix マッチングのパフォーマンスを測定
fn bench_prefix_matching(c: &mut Criterion) {
    let secp = Secp256k1::new();
    let (_sk, pk) = secp.generate_keypair(&mut rand::thread_rng());
    let npub = pubkey_to_npub(&pk);
    let npub_body = &npub[5..]; // "npub1" を除去
    let prefix = "test";

    c.bench_function("prefix_matching", |b| {
        b.iter(|| {
            let matches = black_box(npub_body).starts_with(black_box(prefix));
            black_box(matches)
        })
    });
}

/// ベンチマーク: prefix 検証
///
/// validate_prefix() のパフォーマンスを測定
fn bench_validate_prefix(c: &mut Criterion) {
    let prefix = "m0ctane";

    c.bench_function("validate_prefix", |b| {
        b.iter(|| {
            let result = validate_prefix(black_box(prefix));
            black_box(result)
        })
    });
}

/// ベンチマーク: 完全なマイニングサイクル
///
/// 鍵生成 → npub 変換 → prefix マッチングの一連の流れを測定
/// （実際のマイニングループに近い）
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

/// GPU ベンチマーク: Batch vs Sequential vs Montgomery
///
/// 3つの方式を同じ条件で比較
fn bench_gpu_methods(c: &mut Criterion) {
    let ctx = init_gpu().expect("Failed to initialize GPU");

    let mut group = c.benchmark_group("gpu_methods");

    // テスト設定: (num_threads, keys_per_thread)
    let configs = [
        (256, 64),   // 16,384 keys
        (256, 256),  // 65,536 keys
        (1024, 64),  // 65,536 keys
        (1024, 256), // 262,144 keys
    ];

    for (num_threads, keys_per_thread) in configs {
        let total_keys = num_threads * keys_per_thread;

        // Batch 用のキー準備
        let batch_keys: Vec<[u64; 4]> = (2..(2 + total_keys as u64))
            .map(|k| [k, 0, 0, 0])
            .collect();

        // Sequential/Montgomery 用のベースキー準備
        let base_keys: Vec<[u64; 4]> = (0..num_threads)
            .map(|i| [2 + (i * keys_per_thread) as u64, 0, 0, 0])
            .collect();

        let config_name = format!("{}t_{}k", num_threads, keys_per_thread);

        // Batch (Full PointMult per key)
        group.bench_with_input(
            BenchmarkId::new("batch", &config_name),
            &(&batch_keys,),
            |b, (keys,)| {
                b.iter(|| {
                    generate_pubkeys_batch(&ctx, black_box(keys)).unwrap()
                })
            },
        );

        // Sequential (Phase 1: PointAddMixed + JacobianToAffine per key)
        group.bench_with_input(
            BenchmarkId::new("sequential", &config_name),
            &(&base_keys, keys_per_thread as u32),
            |b, (keys, kpt)| {
                b.iter(|| {
                    generate_pubkeys_sequential_batch(&ctx, black_box(keys), *kpt).unwrap()
                })
            },
        );

        // Montgomery (Phase 2: PointAddMixed + batch inverse)
        group.bench_with_input(
            BenchmarkId::new("montgomery", &config_name),
            &(&base_keys, keys_per_thread as u32),
            |b, (keys, kpt)| {
                b.iter(|| {
                    generate_pubkeys_sequential_montgomery_batch(&ctx, black_box(keys), *kpt).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// GPU ベンチマーク: keys_per_thread の影響を調べる
///
/// Montgomery's Trick の効果が keys_per_thread でどう変わるか
fn bench_gpu_keys_per_thread(c: &mut Criterion) {
    let ctx = init_gpu().expect("Failed to initialize GPU");

    let mut group = c.benchmark_group("gpu_keys_per_thread");

    let num_threads = 256;
    let keys_per_thread_options = [16, 32, 64, 128, 256];

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
                    generate_pubkeys_sequential_montgomery_batch(&ctx, black_box(keys), *kpt).unwrap()
                })
            },
        );
    }

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
    bench_gpu_keys_per_thread
);

criterion_main!(benches, gpu_benches);
