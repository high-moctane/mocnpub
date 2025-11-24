#![no_main]

use libfuzzer_sys::fuzz_target;
use mocnpub_main::gpu::{init_gpu, test_mod_inv_gpu, test_mod_mult_gpu};

// secp256k1 prime p = 2^256 - 2^32 - 977
const P: [u64; 4] = [
    0xFFFFFFFEFFFFFC2F,
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
];

/// Check if a == 0
fn is_zero(a: &[u64; 4]) -> bool {
    a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0
}

/// Check if a >= p
fn is_greater_or_equal(a: &[u64; 4], p: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] > p[i] {
            return true;
        }
        if a[i] < p[i] {
            return false;
        }
    }
    // a == p
    true
}

fuzz_target!(|data: &[u8]| {
    // Need at least 32 bytes for 4x u64
    if data.len() < 32 {
        return;
    }

    // Convert bytes to u64 array (little-endian)
    let mut a = [0u64; 4];
    for i in 0..4 {
        a[i] = u64::from_le_bytes([
            data[i * 8],
            data[i * 8 + 1],
            data[i * 8 + 2],
            data[i * 8 + 3],
            data[i * 8 + 4],
            data[i * 8 + 5],
            data[i * 8 + 6],
            data[i * 8 + 7],
        ]);
    }

    // Skip if a == 0 or a >= p
    if is_zero(&a) || is_greater_or_equal(&a, &P) {
        return;
    }

    // Initialize GPU (cache the context for performance)
    // We use type inference to avoid importing cudarc in the fuzz target
    use std::sync::{Arc, OnceLock};
    use cudarc::driver::CudaContext;

    static GPU_CONTEXT: OnceLock<Arc<CudaContext>> = OnceLock::new();
    let ctx = GPU_CONTEXT
        .get_or_init(|| init_gpu().expect("GPU initialization failed"))
        .clone();

    // Compute inv(a) on GPU
    let inv_a = match test_mod_inv_gpu(&ctx, &a) {
        Ok(inv) => inv,
        Err(_) => {
            // GPU error - skip this input
            return;
        }
    };

    // Compute a * inv(a) mod p on GPU
    let result = match test_mod_mult_gpu(&ctx, &a, &inv_a) {
        Ok(res) => res,
        Err(_) => {
            // GPU error - skip this input
            return;
        }
    };

    // Invariant: a * inv(a) ≡ 1 (mod p)
    assert_eq!(
        result,
        [1, 0, 0, 0],
        "Modular inverse failed: a = {:?}, inv(a) = {:?}, a * inv(a) = {:?}",
        a,
        inv_a,
        result
    );
});
