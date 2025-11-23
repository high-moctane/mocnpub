use criterion::{black_box, criterion_group, criterion_main, Criterion};
use secp256k1::{rand, Secp256k1};
use mocnpub_main::{pubkey_to_npub, seckey_to_nsec, validate_prefix};

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

criterion_group!(
    benches,
    bench_keypair_generation,
    bench_pubkey_to_npub,
    bench_seckey_to_nsec,
    bench_prefix_matching,
    bench_validate_prefix,
    bench_mining_cycle
);
criterion_main!(benches);
