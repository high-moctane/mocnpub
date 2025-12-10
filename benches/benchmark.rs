use criterion::{Criterion, criterion_group, criterion_main};
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
