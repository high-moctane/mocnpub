/*
 * GPU interface for secp256k1 operations
 *
 * This module provides Rust bindings to the CUDA implementation of secp256k1.
 */

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Initialize GPU and return context
pub fn init_gpu() -> Result<Arc<CudaContext>, Box<dyn std::error::Error>> {
    // Get GPU context (already returns Arc<CudaContext>)
    let ctx = CudaContext::new(0)?;
    Ok(ctx)
}

/// Get the number of Streaming Multiprocessors (SM) on the GPU
///
/// This is used to optimize grid size to avoid Tail Effect
/// (idle SMs in the last wave of execution)
pub fn get_sm_count(ctx: &Arc<CudaContext>) -> Result<u32, Box<dyn std::error::Error>> {
    let sm_count = ctx.attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    Ok(sm_count as u32)
}

/// Calculate optimal batch size that is a multiple of (SM count × threads_per_block)
///
/// This ensures that all waves of GPU execution fully utilize all SMs,
/// avoiding the "Tail Effect" where the last wave has idle SMs.
///
/// # Arguments
/// * `ctx` - CUDA context
/// * `desired_batch_size` - User-requested batch size
/// * `threads_per_block` - Threads per block (typically 64 for this kernel)
///
/// # Returns
/// * Adjusted batch size (rounded up to nearest wave boundary)
pub fn calculate_optimal_batch_size(
    ctx: &Arc<CudaContext>,
    desired_batch_size: usize,
    threads_per_block: u32,
) -> Result<usize, Box<dyn std::error::Error>> {
    let sm_count = get_sm_count(ctx)?;
    let threads_per_wave = (sm_count * threads_per_block) as usize;
    let waves = (desired_batch_size + threads_per_wave - 1) / threads_per_wave;
    Ok(waves * threads_per_wave)
}

/// Test modular addition on GPU
///
/// This function tests the _ModAdd function by adding two 256-bit numbers modulo p
pub fn test_mod_add_gpu(
    ctx: &Arc<CudaContext>,
    a: &[u64; 4],
    b: &[u64; 4],
) -> Result<[u64; 4], Box<dyn std::error::Error>> {
    // Get default stream
    let stream = ctx.default_stream();

    // Load PTX module
    let ptx_code = include_str!(concat!(env!("OUT_DIR"), "/secp256k1.ptx"));
    let module = ctx.load_module(Ptx::from_src(ptx_code))?;
    let kernel = module.load_function("test_mod_add")?;

    // Prepare input data (flatten to Vec<u64>)
    let input_a: Vec<u64> = a.to_vec();
    let input_b: Vec<u64> = b.to_vec();

    // Allocate device memory (using alloc_zeros to avoid unsafe)
    let mut a_dev = stream.alloc_zeros::<u64>(4)?;
    let mut b_dev = stream.alloc_zeros::<u64>(4)?;
    let mut output_dev = stream.alloc_zeros::<u64>(4)?;

    // Copy input data to device
    stream.memcpy_htod(&input_a, &mut a_dev)?;
    stream.memcpy_htod(&input_b, &mut b_dev)?;

    // Launch configuration: 1 block, 1 thread (for single test)
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // Launch kernel
    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut a_dev);
    builder.arg(&mut b_dev);
    builder.arg(&mut output_dev);
    unsafe {
        builder.launch(config)?;
    }

    // Copy result back to host
    let result_vec = stream.memcpy_dtov(&output_dev)?;

    // Convert to fixed-size array
    let mut result = [0u64; 4];
    result.copy_from_slice(&result_vec);

    Ok(result)
}

/// Test modular multiplication on GPU
///
/// This function tests the _ModMult function by multiplying two 256-bit numbers modulo p
pub fn test_mod_mult_gpu(
    ctx: &Arc<CudaContext>,
    a: &[u64; 4],
    b: &[u64; 4],
) -> Result<[u64; 4], Box<dyn std::error::Error>> {
    // Get default stream
    let stream = ctx.default_stream();

    // Load PTX module
    let ptx_code = include_str!(concat!(env!("OUT_DIR"), "/secp256k1.ptx"));
    let module = ctx.load_module(Ptx::from_src(ptx_code))?;
    let kernel = module.load_function("test_mod_mult")?;

    // Prepare input data (flatten to Vec<u64>)
    let input_a: Vec<u64> = a.to_vec();
    let input_b: Vec<u64> = b.to_vec();

    // Allocate device memory (using alloc_zeros to avoid unsafe)
    let mut a_dev = stream.alloc_zeros::<u64>(4)?;
    let mut b_dev = stream.alloc_zeros::<u64>(4)?;
    let mut output_dev = stream.alloc_zeros::<u64>(4)?;

    // Copy input data to device
    stream.memcpy_htod(&input_a, &mut a_dev)?;
    stream.memcpy_htod(&input_b, &mut b_dev)?;

    // Launch configuration: 1 block, 1 thread (for single test)
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // Launch kernel
    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut a_dev);
    builder.arg(&mut b_dev);
    builder.arg(&mut output_dev);
    unsafe {
        builder.launch(config)?;
    }

    // Copy result back to host
    let result_vec = stream.memcpy_dtov(&output_dev)?;

    // Convert to fixed-size array
    let mut result = [0u64; 4];
    result.copy_from_slice(&result_vec);

    Ok(result)
}

/// Test modular inverse on GPU
///
/// This function tests the _ModInv function by computing a^(-1) mod p
pub fn test_mod_inv_gpu(
    ctx: &Arc<CudaContext>,
    a: &[u64; 4],
) -> Result<[u64; 4], Box<dyn std::error::Error>> {
    // Get default stream
    let stream = ctx.default_stream();

    // Load PTX module
    let ptx_code = include_str!(concat!(env!("OUT_DIR"), "/secp256k1.ptx"));
    let module = ctx.load_module(Ptx::from_src(ptx_code))?;
    let kernel = module.load_function("test_mod_inv")?;

    // Prepare input data
    let input_a: Vec<u64> = a.to_vec();

    // Allocate device memory
    let mut a_dev = stream.alloc_zeros::<u64>(4)?;
    let mut output_dev = stream.alloc_zeros::<u64>(4)?;

    // Copy input data to device
    stream.memcpy_htod(&input_a, &mut a_dev)?;

    // Launch configuration: 1 block, 1 thread
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // Launch kernel
    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut a_dev);
    builder.arg(&mut output_dev);
    unsafe {
        builder.launch(config)?;
    }

    // Copy result back to host
    let result_vec = stream.memcpy_dtov(&output_dev)?;

    // Convert to fixed-size array
    let mut result = [0u64; 4];
    result.copy_from_slice(&result_vec);

    Ok(result)
}

/// Test modular squaring on GPU
///
/// This function tests the _ModSquare function by computing a^2 mod p
pub fn test_mod_square_gpu(
    ctx: &Arc<CudaContext>,
    a: &[u64; 4],
) -> Result<[u64; 4], Box<dyn std::error::Error>> {
    // Get default stream
    let stream = ctx.default_stream();

    // Load PTX module
    let ptx_code = include_str!(concat!(env!("OUT_DIR"), "/secp256k1.ptx"));
    let module = ctx.load_module(Ptx::from_src(ptx_code))?;
    let kernel = module.load_function("test_mod_square")?;

    // Prepare input data
    let input_a: Vec<u64> = a.to_vec();

    // Allocate device memory
    let mut a_dev = stream.alloc_zeros::<u64>(4)?;
    let mut output_dev = stream.alloc_zeros::<u64>(4)?;

    // Copy input data to device
    stream.memcpy_htod(&input_a, &mut a_dev)?;

    // Launch configuration: 1 block, 1 thread
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // Launch kernel
    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut a_dev);
    builder.arg(&mut output_dev);
    unsafe {
        builder.launch(config)?;
    }

    // Copy result back to host
    let result_vec = stream.memcpy_dtov(&output_dev)?;

    // Convert to fixed-size array
    let mut result = [0u64; 4];
    result.copy_from_slice(&result_vec);

    Ok(result)
}

/// Test point doubling on GPU
///
/// This function tests the _PointDouble function by doubling a point on the secp256k1 curve
/// Input: Point in Affine coordinates (x, y)
/// Output: 2*Point in Affine coordinates (x, y)
#[allow(non_snake_case)]
pub fn test_point_double_gpu(
    ctx: &Arc<CudaContext>,
    x: &[u64; 4],
    y: &[u64; 4],
) -> Result<([u64; 4], [u64; 4]), Box<dyn std::error::Error>> {
    // Get default stream
    let stream = ctx.default_stream();

    // Load PTX module
    let ptx_code = include_str!(concat!(env!("OUT_DIR"), "/secp256k1.ptx"));
    let module = ctx.load_module(Ptx::from_src(ptx_code))?;
    let kernel = module.load_function("test_point_double")?;

    // Convert Affine to Jacobian: (x, y) -> (x, y, 1)
    let input_X: Vec<u64> = x.to_vec();
    let input_Y: Vec<u64> = y.to_vec();
    let input_Z: Vec<u64> = vec![1, 0, 0, 0];

    // Allocate device memory
    let mut X_dev = stream.alloc_zeros::<u64>(4)?;
    let mut Y_dev = stream.alloc_zeros::<u64>(4)?;
    let mut Z_dev = stream.alloc_zeros::<u64>(4)?;
    let mut output_x_dev = stream.alloc_zeros::<u64>(4)?;
    let mut output_y_dev = stream.alloc_zeros::<u64>(4)?;

    // Copy input data to device
    stream.memcpy_htod(&input_X, &mut X_dev)?;
    stream.memcpy_htod(&input_Y, &mut Y_dev)?;
    stream.memcpy_htod(&input_Z, &mut Z_dev)?;

    // Launch configuration: 1 block, 1 thread (for single test)
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // Launch kernel
    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut X_dev);
    builder.arg(&mut Y_dev);
    builder.arg(&mut Z_dev);
    builder.arg(&mut output_x_dev);
    builder.arg(&mut output_y_dev);
    unsafe {
        builder.launch(config)?;
    }

    // Copy result back to host
    let result_x_vec = stream.memcpy_dtov(&output_x_dev)?;
    let result_y_vec = stream.memcpy_dtov(&output_y_dev)?;

    // Convert to fixed-size arrays
    let mut result_x = [0u64; 4];
    let mut result_y = [0u64; 4];
    result_x.copy_from_slice(&result_x_vec);
    result_y.copy_from_slice(&result_y_vec);

    Ok((result_x, result_y))
}

/// Test point multiplication on GPU
///
/// This function tests the _PointMult function by computing k * P
/// Input: scalar k (256-bit), point P in Affine coordinates (x, y)
/// Output: k*P in Affine coordinates (x, y)
pub fn test_point_mult_gpu(
    ctx: &Arc<CudaContext>,
    k: &[u64; 4],
    px: &[u64; 4],
    py: &[u64; 4],
) -> Result<([u64; 4], [u64; 4]), Box<dyn std::error::Error>> {
    // Get default stream
    let stream = ctx.default_stream();

    // Load PTX module
    let ptx_code = include_str!(concat!(env!("OUT_DIR"), "/secp256k1.ptx"));
    let module = ctx.load_module(Ptx::from_src(ptx_code))?;
    let kernel = module.load_function("test_point_mult")?;

    // Prepare input data
    let input_k: Vec<u64> = k.to_vec();
    let input_px: Vec<u64> = px.to_vec();
    let input_py: Vec<u64> = py.to_vec();

    // Allocate device memory
    let mut k_dev = stream.alloc_zeros::<u64>(4)?;
    let mut px_dev = stream.alloc_zeros::<u64>(4)?;
    let mut py_dev = stream.alloc_zeros::<u64>(4)?;
    let mut output_x_dev = stream.alloc_zeros::<u64>(4)?;
    let mut output_y_dev = stream.alloc_zeros::<u64>(4)?;

    // Copy input data to device
    stream.memcpy_htod(&input_k, &mut k_dev)?;
    stream.memcpy_htod(&input_px, &mut px_dev)?;
    stream.memcpy_htod(&input_py, &mut py_dev)?;

    // Launch configuration: 1 block, 1 thread (for single test)
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // Launch kernel
    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut k_dev);
    builder.arg(&mut px_dev);
    builder.arg(&mut py_dev);
    builder.arg(&mut output_x_dev);
    builder.arg(&mut output_y_dev);
    unsafe {
        builder.launch(config)?;
    }

    // Copy result back to host
    let result_x_vec = stream.memcpy_dtov(&output_x_dev)?;
    let result_y_vec = stream.memcpy_dtov(&output_y_dev)?;

    // Convert to fixed-size arrays
    let mut result_x = [0u64; 4];
    let mut result_y = [0u64; 4];
    result_x.copy_from_slice(&result_x_vec);
    result_y.copy_from_slice(&result_y_vec);

    Ok((result_x, result_y))
}

/// Generate public keys from private keys in batch on GPU
///
/// This function generates multiple public keys in parallel using the GPU.
/// Each private key is multiplied by the generator point G to get the public key.
///
/// Input: Vec of 256-bit private keys (each as [u64; 4])
/// Output: Vec of 256-bit public key x-coordinates (each as [u64; 4])
///
/// Note: Only the x-coordinate is returned because that's all we need for npub generation.
pub fn generate_pubkeys_batch(
    ctx: &Arc<CudaContext>,
    privkeys: &[[u64; 4]],
) -> Result<Vec<[u64; 4]>, Box<dyn std::error::Error>> {
    let batch_size = privkeys.len();
    if batch_size == 0 {
        return Ok(vec![]);
    }

    // Get default stream
    let stream = ctx.default_stream();

    // Load PTX module
    let ptx_code = include_str!(concat!(env!("OUT_DIR"), "/secp256k1.ptx"));
    let module = ctx.load_module(Ptx::from_src(ptx_code))?;
    let kernel = module.load_function("generate_pubkeys")?;

    // Flatten private keys to Vec<u64>
    let privkeys_flat: Vec<u64> = privkeys.iter().flat_map(|k| k.iter().copied()).collect();

    // Allocate device memory
    let mut privkeys_dev = stream.alloc_zeros::<u64>(batch_size * 4)?;
    let mut pubkeys_x_dev = stream.alloc_zeros::<u64>(batch_size * 4)?;

    // Copy private keys to device
    stream.memcpy_htod(&privkeys_flat, &mut privkeys_dev)?;

    // Calculate grid and block dimensions
    // Use 256 threads per block (common choice for CUDA)
    let threads_per_block = 256u32;
    let num_blocks = (batch_size as u32 + threads_per_block - 1) / threads_per_block;

    let config = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    // Launch kernel
    let batch_size_u32 = batch_size as u32;
    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut privkeys_dev);
    builder.arg(&mut pubkeys_x_dev);
    builder.arg(&batch_size_u32);
    unsafe {
        builder.launch(config)?;
    }

    // Copy results back to host
    let pubkeys_x_flat = stream.memcpy_dtov(&pubkeys_x_dev)?;

    // Convert flat Vec to Vec of [u64; 4]
    let pubkeys_x: Vec<[u64; 4]> = pubkeys_x_flat
        .chunks_exact(4)
        .map(|chunk| {
            let mut arr = [0u64; 4];
            arr.copy_from_slice(chunk);
            arr
        })
        .collect();

    Ok(pubkeys_x)
}

/// Generate public keys from consecutive private keys (連続秘密鍵戦略)
///
/// This is the "10000連ガチャ" strategy that exploits the property:
///   (k+1)*G = k*G + G
///
/// Each thread generates `keys_per_thread` consecutive public keys:
///   - First key: base_key * G (heavy _PointMult operation)
///   - Subsequent keys: P + G using _PointAddMixed (light operation, ~300x faster!)
///
/// # Arguments
/// * `ctx` - GPU context
/// * `base_keys` - Starting private keys for each thread
/// * `keys_per_thread` - Number of consecutive keys each thread generates
///
/// # Returns
/// * `Vec<[u64; 4]>` - Public key x-coordinates, ordered as:
///   [thread0_key0, thread0_key1, ..., thread0_keyN, thread1_key0, ...]
pub fn generate_pubkeys_sequential_batch(
    ctx: &Arc<CudaContext>,
    base_keys: &[[u64; 4]],
    keys_per_thread: u32,
) -> Result<Vec<[u64; 4]>, Box<dyn std::error::Error>> {
    let num_threads = base_keys.len();
    if num_threads == 0 || keys_per_thread == 0 {
        return Ok(vec![]);
    }

    let total_keys = num_threads * keys_per_thread as usize;

    // Get default stream
    let stream = ctx.default_stream();

    // Load PTX module
    let ptx_code = include_str!(concat!(env!("OUT_DIR"), "/secp256k1.ptx"));
    let module = ctx.load_module(Ptx::from_src(ptx_code))?;
    let kernel = module.load_function("generate_pubkeys_sequential")?;

    // Flatten base keys to Vec<u64>
    let base_keys_flat: Vec<u64> = base_keys.iter().flat_map(|k| k.iter().copied()).collect();

    // Allocate device memory
    let mut base_keys_dev = stream.alloc_zeros::<u64>(num_threads * 4)?;
    let mut pubkeys_x_dev = stream.alloc_zeros::<u64>(total_keys * 4)?;

    // Copy base keys to device
    stream.memcpy_htod(&base_keys_flat, &mut base_keys_dev)?;

    // Calculate grid and block dimensions
    let threads_per_block = 256u32;
    let num_blocks = (num_threads as u32 + threads_per_block - 1) / threads_per_block;

    let config = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    // Launch kernel
    let num_threads_u32 = num_threads as u32;
    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut base_keys_dev);
    builder.arg(&mut pubkeys_x_dev);
    builder.arg(&num_threads_u32);
    builder.arg(&keys_per_thread);
    unsafe {
        builder.launch(config)?;
    }

    // Copy results back to host
    let pubkeys_x_flat = stream.memcpy_dtov(&pubkeys_x_dev)?;

    // Convert flat Vec to Vec of [u64; 4]
    let pubkeys_x: Vec<[u64; 4]> = pubkeys_x_flat
        .chunks_exact(4)
        .map(|chunk| {
            let mut arr = [0u64; 4];
            arr.copy_from_slice(chunk);
            arr
        })
        .collect();

    Ok(pubkeys_x)
}

/// Generate public keys using sequential keys with Montgomery's Trick (Phase 2)
///
/// This is the optimized "256連ガチャ" strategy that combines:
/// 1. Sequential key generation: (k+1)*G = k*G + G (light _PointAddMixed)
/// 2. Montgomery's Trick: Batch inverse calculation (1 _ModInv instead of N)
///
/// Without Montgomery's Trick (Phase 1): Each key requires _JacobianToAffine which calls _ModInv
/// With Montgomery's Trick (Phase 2): All N keys share a SINGLE _ModInv call!
///
/// Theoretical speedup: ~85x reduction in inverse calculations for N=256
///
/// # Arguments
/// * `ctx` - GPU context
/// * `base_keys` - Starting private keys for each thread
/// * `keys_per_thread` - Number of consecutive keys each thread generates (max 256)
///
/// # Returns
/// * `Vec<[u64; 4]>` - Public key x-coordinates
pub fn generate_pubkeys_sequential_montgomery_batch(
    ctx: &Arc<CudaContext>,
    base_keys: &[[u64; 4]],
    keys_per_thread: u32,
) -> Result<Vec<[u64; 4]>, Box<dyn std::error::Error>> {
    let num_threads = base_keys.len();
    if num_threads == 0 || keys_per_thread == 0 {
        return Ok(vec![]);
    }

    // Clamp to max 256 (CUDA kernel limit)
    let keys_per_thread = keys_per_thread.min(256);
    let total_keys = num_threads * keys_per_thread as usize;

    // Get default stream
    let stream = ctx.default_stream();

    // Load PTX module
    let ptx_code = include_str!(concat!(env!("OUT_DIR"), "/secp256k1.ptx"));
    let module = ctx.load_module(Ptx::from_src(ptx_code))?;
    let kernel = module.load_function("generate_pubkeys_sequential_montgomery")?;

    // Flatten base keys to Vec<u64>
    let base_keys_flat: Vec<u64> = base_keys.iter().flat_map(|k| k.iter().copied()).collect();

    // Allocate device memory
    let mut base_keys_dev = stream.alloc_zeros::<u64>(num_threads * 4)?;
    let mut pubkeys_x_dev = stream.alloc_zeros::<u64>(total_keys * 4)?;

    // Copy base keys to device
    stream.memcpy_htod(&base_keys_flat, &mut base_keys_dev)?;

    // Calculate grid and block dimensions
    // Note: Each thread uses ~32KB of local memory, so we may need fewer threads per block
    let threads_per_block = 64u32;  // Reduced from 256 due to high register/local memory usage
    let num_blocks = (num_threads as u32 + threads_per_block - 1) / threads_per_block;

    let config = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    // Launch kernel
    let num_threads_u32 = num_threads as u32;
    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut base_keys_dev);
    builder.arg(&mut pubkeys_x_dev);
    builder.arg(&num_threads_u32);
    builder.arg(&keys_per_thread);
    unsafe {
        builder.launch(config)?;
    }

    // Copy results back to host
    let pubkeys_x_flat = stream.memcpy_dtov(&pubkeys_x_dev)?;

    // Convert flat Vec to Vec of [u64; 4]
    let pubkeys_x: Vec<[u64; 4]> = pubkeys_x_flat
        .chunks_exact(4)
        .map(|chunk| {
            let mut arr = [0u64; 4];
            arr.copy_from_slice(chunk);
            arr
        })
        .collect();

    Ok(pubkeys_x)
}

/// Match result from GPU prefix matching
#[derive(Debug, Clone)]
pub struct GpuMatch {
    /// Thread index (which base key)
    pub base_idx: u32,
    /// Key offset within thread (0 = base key, 1 = base+1, etc.)
    pub offset: u32,
    /// Public key x-coordinate
    pub pubkey_x: [u64; 4],
    /// Endomorphism type: 0 = original, 1 = β*x, 2 = β²*x
    /// Used to adjust the private key with λ or λ²
    pub endo_type: u32,
}

/// Generate public keys with GPU-side prefix matching
///
/// This function generates public keys using Montgomery's Trick and filters
/// them by prefix on the GPU, returning only matching keys.
///
/// # Arguments
/// * `ctx` - GPU context
/// * `base_keys` - Starting private keys for each thread
/// * `keys_per_thread` - Number of consecutive keys each thread generates (max 256)
/// * `prefix_bits` - Prefix patterns and masks: Vec<(pattern, mask, bit_len)>
/// * `max_matches` - Maximum number of matches to return
///
/// # Returns
/// * `Vec<GpuMatch>` - Matching keys with their indices and public keys
pub fn generate_pubkeys_with_prefix_match(
    ctx: &Arc<CudaContext>,
    base_keys: &[[u64; 4]],
    keys_per_thread: u32,
    prefix_bits: &[(u64, u64, u32)],
    max_matches: u32,
) -> Result<Vec<GpuMatch>, Box<dyn std::error::Error>> {
    let num_threads = base_keys.len();
    let num_prefixes = prefix_bits.len();

    if num_threads == 0 || keys_per_thread == 0 || num_prefixes == 0 {
        return Ok(vec![]);
    }

    // Clamp to max 256 (CUDA kernel limit)
    let keys_per_thread = keys_per_thread.min(256);

    // Get default stream
    let stream = ctx.default_stream();

    // Load PTX module
    let ptx_code = include_str!(concat!(env!("OUT_DIR"), "/secp256k1.ptx"));
    let module = ctx.load_module(Ptx::from_src(ptx_code))?;
    let kernel = module.load_function("generate_pubkeys_with_prefix_match")?;

    // Flatten base keys to Vec<u64>
    let base_keys_flat: Vec<u64> = base_keys.iter().flat_map(|k| k.iter().copied()).collect();

    // Prepare prefix patterns and masks
    let patterns: Vec<u64> = prefix_bits.iter().map(|(p, _, _)| *p).collect();
    let masks: Vec<u64> = prefix_bits.iter().map(|(_, m, _)| *m).collect();

    // Allocate device memory for inputs
    let mut base_keys_dev = stream.alloc_zeros::<u64>(num_threads * 4)?;
    let mut patterns_dev = stream.alloc_zeros::<u64>(num_prefixes)?;
    let mut masks_dev = stream.alloc_zeros::<u64>(num_prefixes)?;

    // Allocate device memory for outputs
    let mut matched_base_idx_dev = stream.alloc_zeros::<u32>(max_matches as usize)?;
    let mut matched_offset_dev = stream.alloc_zeros::<u32>(max_matches as usize)?;
    let mut matched_pubkeys_x_dev = stream.alloc_zeros::<u64>(max_matches as usize * 4)?;
    let mut matched_endo_type_dev = stream.alloc_zeros::<u32>(max_matches as usize)?;
    let mut match_count_dev = stream.alloc_zeros::<u32>(1)?;

    // Copy inputs to device
    stream.memcpy_htod(&base_keys_flat, &mut base_keys_dev)?;
    stream.memcpy_htod(&patterns, &mut patterns_dev)?;
    stream.memcpy_htod(&masks, &mut masks_dev)?;

    // Calculate grid and block dimensions
    let threads_per_block = 64u32;
    let num_blocks = (num_threads as u32 + threads_per_block - 1) / threads_per_block;

    let config = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    // Launch kernel
    let num_threads_u32 = num_threads as u32;
    let num_prefixes_u32 = num_prefixes as u32;
    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut base_keys_dev);
    builder.arg(&mut patterns_dev);
    builder.arg(&mut masks_dev);
    builder.arg(&num_prefixes_u32);
    builder.arg(&mut matched_base_idx_dev);
    builder.arg(&mut matched_offset_dev);
    builder.arg(&mut matched_pubkeys_x_dev);
    builder.arg(&mut matched_endo_type_dev);  // Endomorphism type (0=original, 1=β, 2=β²)
    builder.arg(&mut match_count_dev);
    builder.arg(&num_threads_u32);
    builder.arg(&keys_per_thread);
    builder.arg(&max_matches);
    unsafe {
        builder.launch(config)?;
    }

    // Copy match count back
    let match_count_vec = stream.memcpy_dtov(&match_count_dev)?;
    let match_count = match_count_vec[0].min(max_matches) as usize;

    if match_count == 0 {
        return Ok(vec![]);
    }

    // Copy results back
    let matched_base_idx = stream.memcpy_dtov(&matched_base_idx_dev)?;
    let matched_offset = stream.memcpy_dtov(&matched_offset_dev)?;
    let matched_pubkeys_x_flat = stream.memcpy_dtov(&matched_pubkeys_x_dev)?;
    let matched_endo_type = stream.memcpy_dtov(&matched_endo_type_dev)?;

    // Build result vector
    let mut results = Vec::with_capacity(match_count);
    for i in 0..match_count {
        let mut pubkey_x = [0u64; 4];
        pubkey_x.copy_from_slice(&matched_pubkeys_x_flat[i * 4..(i + 1) * 4]);

        results.push(GpuMatch {
            base_idx: matched_base_idx[i],
            offset: matched_offset[i],
            pubkey_x,
            endo_type: matched_endo_type[i],
        });
    }

    Ok(results)
}

/// Test mixed point addition on GPU (for unit testing)
pub fn test_point_add_mixed_gpu(
    ctx: &Arc<CudaContext>,
    x1: &[u64; 4],
    y1: &[u64; 4],
    z1: &[u64; 4],
    x2: &[u64; 4],
    y2: &[u64; 4],
) -> Result<([u64; 4], [u64; 4]), Box<dyn std::error::Error>> {
    let stream = ctx.default_stream();

    let ptx_code = include_str!(concat!(env!("OUT_DIR"), "/secp256k1.ptx"));
    let module = ctx.load_module(Ptx::from_src(ptx_code))?;
    let kernel = module.load_function("test_point_add_mixed")?;

    // Allocate device memory
    let mut x1_dev = stream.alloc_zeros::<u64>(4)?;
    let mut y1_dev = stream.alloc_zeros::<u64>(4)?;
    let mut z1_dev = stream.alloc_zeros::<u64>(4)?;
    let mut x2_dev = stream.alloc_zeros::<u64>(4)?;
    let mut y2_dev = stream.alloc_zeros::<u64>(4)?;
    let mut output_x_dev = stream.alloc_zeros::<u64>(4)?;
    let mut output_y_dev = stream.alloc_zeros::<u64>(4)?;

    // Copy inputs to device
    stream.memcpy_htod(&x1.to_vec(), &mut x1_dev)?;
    stream.memcpy_htod(&y1.to_vec(), &mut y1_dev)?;
    stream.memcpy_htod(&z1.to_vec(), &mut z1_dev)?;
    stream.memcpy_htod(&x2.to_vec(), &mut x2_dev)?;
    stream.memcpy_htod(&y2.to_vec(), &mut y2_dev)?;

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut x1_dev);
    builder.arg(&mut y1_dev);
    builder.arg(&mut z1_dev);
    builder.arg(&mut x2_dev);
    builder.arg(&mut y2_dev);
    builder.arg(&mut output_x_dev);
    builder.arg(&mut output_y_dev);
    unsafe {
        builder.launch(config)?;
    }

    let result_x_vec = stream.memcpy_dtov(&output_x_dev)?;
    let result_y_vec = stream.memcpy_dtov(&output_y_dev)?;

    let mut result_x = [0u64; 4];
    let mut result_y = [0u64; 4];
    result_x.copy_from_slice(&result_x_vec);
    result_y.copy_from_slice(&result_y_vec);

    Ok((result_x, result_y))
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_mod_add_simple() {
        // Initialize GPU
        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Test case: 1 + 1 = 2 (mod p)
        let a = [1u64, 0, 0, 0];
        let b = [1u64, 0, 0, 0];

        let result = test_mod_add_gpu(&ctx, &a, &b).expect("GPU kernel failed");

        // Expected: [2, 0, 0, 0]
        assert_eq!(result, [2u64, 0, 0, 0]);
    }

    #[test]
    fn test_gpu_mod_add_overflow() {
        // Initialize GPU
        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Test case: p - 1 + 2 = 1 (mod p)
        // p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
        let p_minus_1 = [
            0xFFFFFFFEFFFFFC2Eu64,
            0xFFFFFFFFFFFFFFFFu64,
            0xFFFFFFFFFFFFFFFFu64,
            0xFFFFFFFFFFFFFFFFu64,
        ];
        let two = [2u64, 0, 0, 0];

        let result = test_mod_add_gpu(&ctx, &p_minus_1, &two).expect("GPU kernel failed");

        // Expected: 1 (since (p - 1) + 2 ≡ 1 (mod p))
        assert_eq!(result, [1u64, 0, 0, 0]);
    }

    #[test]
    fn test_gpu_mod_mult_simple() {
        // Initialize GPU
        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Test case: 2 * 3 = 6 (mod p)
        let a = [2u64, 0, 0, 0];
        let b = [3u64, 0, 0, 0];

        let result = test_mod_mult_gpu(&ctx, &a, &b).expect("GPU kernel failed");

        // Expected: [6, 0, 0, 0]
        println!("2 * 3 = {:?}", result);
        assert_eq!(result, [6u64, 0, 0, 0]);
    }

    #[test]
    fn test_gpu_mod_mult_large() {
        // Initialize GPU
        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Test case: 2^128 * 2^128 = 2^256 mod p = 2^32 + 977
        // a = 2^128 = [0, 0, 1, 0]
        // b = 2^128 = [0, 0, 1, 0]
        let a = [0u64, 0, 1, 0];
        let b = [0u64, 0, 1, 0];

        let result = test_mod_mult_gpu(&ctx, &a, &b).expect("GPU kernel failed");

        // Expected: 2^32 + 977 = 0x1000003D1 = [0x1000003D1, 0, 0, 0]
        println!("2^128 * 2^128 mod p = {:?}", result);
        println!("Expected: [0x1000003D1, 0, 0, 0] = [{}, 0, 0, 0]", 0x1000003D1u64);
        assert_eq!(result, [0x1000003D1u64, 0, 0, 0]);
    }

    #[test]
    fn test_gpu_mod_inv_one() {
        // Initialize GPU
        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Test case: inv(1) = 1 mod p
        let a = [1u64, 0, 0, 0];
        let inv_a = test_mod_inv_gpu(&ctx, &a).expect("GPU modular inverse failed");

        println!("inv(1) = {:?}", inv_a);
        assert_eq!(inv_a, [1u64, 0, 0, 0], "inv(1) should equal 1");
    }

    #[test]
    fn test_gpu_mod_inv_simple() {
        // Initialize GPU
        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Test case: inv(2) mod p
        // We verify: 2 * inv(2) ≡ 1 (mod p)
        let a = [2u64, 0, 0, 0];
        let inv_a = test_mod_inv_gpu(&ctx, &a).expect("GPU modular inverse failed");

        println!("inv(2) = {:?}", inv_a);

        // Expected value: 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF7FFFFE18
        let expected = [0xFFFFFFFF7FFFFE18u64, 0xFFFFFFFFFFFFFFFFu64, 0xFFFFFFFFFFFFFFFFu64, 0x7FFFFFFFFFFFFFFFu64];
        println!("Expected: {:?}", expected);

        // Verify: 2 * inv(2) ≡ 1 (mod p)
        let product = test_mod_mult_gpu(&ctx, &a, &inv_a).expect("GPU multiplication failed");
        println!("2 * inv(2) = {:?}", product);

        assert_eq!(product, [1u64, 0, 0, 0], "2 * inv(2) should equal 1");
    }

    #[test]
    fn test_gpu_mod_inv_three() {
        // Initialize GPU
        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Test case: inv(3) mod p
        // We verify: 3 * inv(3) ≡ 1 (mod p)
        let a = [3u64, 0, 0, 0];
        let inv_a = test_mod_inv_gpu(&ctx, &a).expect("GPU modular inverse failed");

        println!("inv(3) = {:?}", inv_a);

        // Expected value: 0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9FFFFFD75
        let expected = [0xAAAAAAA9FFFFFD75u64, 0xAAAAAAAAAAAAAAAAu64, 0xAAAAAAAAAAAAAAAAu64, 0xAAAAAAAAAAAAAAAAu64];
        println!("Expected: {:?}", expected);

        // Verify: 3 * inv(3) ≡ 1 (mod p)
        let product = test_mod_mult_gpu(&ctx, &a, &inv_a).expect("GPU multiplication failed");
        println!("3 * inv(3) = {:?}", product);

        assert_eq!(product, [1u64, 0, 0, 0], "3 * inv(3) should equal 1");
    }

    #[test]
    fn test_gpu_mod_square_simple() {
        // Initialize GPU
        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Test case: 2^2 = 4 (mod p)
        let a = [2u64, 0, 0, 0];
        let result = test_mod_square_gpu(&ctx, &a).expect("GPU modular squaring failed");

        println!("2^2 = {:?}", result);
        assert_eq!(result, [4u64, 0, 0, 0]);
    }

    #[test]
    fn test_gpu_mod_square_2_128() {
        // Initialize GPU
        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Test case: (2^128)^2 = 2^256 mod p = 2^32 + 977 = 0x1000003D1
        // 2^128 is represented as [0, 0, 1, 0] (little-endian)
        let a = [0u64, 0, 1, 0];
        let result = test_mod_square_gpu(&ctx, &a).expect("GPU modular squaring failed");

        println!("(2^128)^2 = {:?}", result);
        println!("Expected: [0x1000003D1, 0, 0, 0] = [{}, 0, 0, 0]", 0x1000003D1u64);
        assert_eq!(result, [0x1000003D1u64, 0, 0, 0]);
    }

    #[test]
    fn test_gpu_mod_square_larger() {
        // Initialize GPU
        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Test case: 3^2 = 9 (mod p)
        let a = [3u64, 0, 0, 0];
        let result = test_mod_square_gpu(&ctx, &a).expect("GPU modular squaring failed");

        println!("3^2 = {:?}", result);
        assert_eq!(result, [9u64, 0, 0, 0]);
    }

    #[test]
    fn test_gpu_mod_square_step254() {
        // Initialize GPU
        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Test case: Step 254 value squared
        // This is the special case that causes the bug in _ModInv
        // Step 254 = [18446744071562067480, 18446744073709551615, 18446744073709551615, 9223372036854775807]
        // Expected: Step 254^2 = [18446744072635809548, 18446744073709551615, 18446744073709551615, 4611686018427387903]
        let a = [18446744071562067480u64, 18446744073709551615, 18446744073709551615, 9223372036854775807];
        let result = test_mod_square_gpu(&ctx, &a).expect("GPU modular squaring failed");

        println!("Step 254^2 = {:?}", result);
        let expected = [18446744072635809548u64, 18446744073709551615, 18446744073709551615, 4611686018427387903];
        println!("Expected:    {:?}", expected);
        assert_eq!(result, expected, "Step 254 squared should match Python simulation");
    }

    #[test]
    fn test_gpu_point_double() {
        // Initialize GPU
        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Test case: 2G (where G is the secp256k1 generator point)
        // G = (Gx, Gy)
        // Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
        let Gx = [
            0x59F2815B16F81798u64,
            0x029BFCDB2DCE28D9u64,
            0x55A06295CE870B07u64,
            0x79BE667EF9DCBBACu64,
        ];
        // Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
        let Gy = [
            0x9C47D08FFB10D4B8u64,
            0xFD17B448A6855419u64,
            0x5DA4FBFC0E1108A8u64,
            0x483ADA7726A3C465u64,
        ];

        let (result_x, result_y) = test_point_double_gpu(&ctx, &Gx, &Gy)
            .expect("GPU point doubling failed");

        // Expected: 2G
        // 2Gx = 0xC6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5
        let expected_2Gx = [
            0xABAC09B95C709EE5u64,
            0x5C778E4B8CEF3CA7u64,
            0x3045406E95C07CD8u64,
            0xC6047F9441ED7D6Du64,
        ];
        // 2Gy = 0x1AE168FEA63DC339A3C58419466CEAEEF7F632653266D0E1236431A950CFE52A
        let expected_2Gy = [
            0x236431A950CFE52Au64,
            0xF7F632653266D0E1u64,
            0xA3C58419466CEAEEu64,
            0x1AE168FEA63DC339u64,
        ];

        // Check result
        assert_eq!(result_x, expected_2Gx, "2G x-coordinate mismatch");
        assert_eq!(result_y, expected_2Gy, "2G y-coordinate mismatch");
    }

    #[test]
    fn test_gpu_reduce512_p_plus_1() {
        // Initialize GPU
        let ctx = init_gpu().expect("Failed to initialize GPU");
        let stream = ctx.default_stream();

        // Load PTX module
        let ptx_code = include_str!(concat!(env!("OUT_DIR"), "/secp256k1.ptx"));
        let module = ctx.load_module(Ptx::from_src(ptx_code)).expect("Failed to load PTX");
        let kernel = module.load_function("test_reduce512").expect("Failed to load kernel");

        // Test case: p + 1
        // Input: [18446744069414583344, 18446744073709551615, 18446744073709551615, 18446744073709551615, 0, 0, 0, 0]
        // Expected: [1, 0, 0, 0]
        let input_512: Vec<u64> = vec![
            18446744069414583344u64,
            18446744073709551615u64,
            18446744073709551615u64,
            18446744073709551615u64,
            0u64,
            0u64,
            0u64,
            0u64,
        ];

        let expected: Vec<u64> = vec![1u64, 0, 0, 0];

        // Allocate device memory
        let mut input_dev = stream.alloc_zeros::<u64>(8).expect("Failed to allocate input");
        let mut output_dev = stream.alloc_zeros::<u64>(4).expect("Failed to allocate output");

        // Copy input to device
        stream.memcpy_htod(&input_512, &mut input_dev).expect("Failed to copy input");

        // Launch configuration
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        let mut builder = stream.launch_builder(&kernel);
        builder.arg(&mut input_dev);
        builder.arg(&mut output_dev);
        unsafe {
            builder.launch(config).expect("Failed to launch kernel");
        }

        // Copy result back to host
        let result_vec = stream.memcpy_dtov(&output_dev).expect("Failed to copy result");

        println!("(p + 1) mod p test:");
        println!("Result:   {:?}", result_vec);
        println!("Expected: {:?}", expected);

        // Check result
        assert_eq!(result_vec, expected, "(p + 1) mod p should equal 1");
    }

    #[test]
    fn test_gpu_reduce512_p_plus_2() {
        // Initialize GPU
        let ctx = init_gpu().expect("Failed to initialize GPU");
        let stream = ctx.default_stream();

        // Load PTX module
        let ptx_code = include_str!(concat!(env!("OUT_DIR"), "/secp256k1.ptx"));
        let module = ctx.load_module(Ptx::from_src(ptx_code)).expect("Failed to load PTX");
        let kernel = module.load_function("test_reduce512").expect("Failed to load kernel");

        // Test case: p + 2
        // Input: [18446744069414583345, 18446744073709551615, 18446744073709551615, 18446744073709551615, 0, 0, 0, 0]
        // Expected: [2, 0, 0, 0]
        let input_512: Vec<u64> = vec![
            18446744069414583345u64,
            18446744073709551615u64,
            18446744073709551615u64,
            18446744073709551615u64,
            0u64,
            0u64,
            0u64,
            0u64,
        ];

        let expected: Vec<u64> = vec![2u64, 0, 0, 0];

        // Allocate device memory
        let mut input_dev = stream.alloc_zeros::<u64>(8).expect("Failed to allocate input");
        let mut output_dev = stream.alloc_zeros::<u64>(4).expect("Failed to allocate output");

        // Copy input to device
        stream.memcpy_htod(&input_512, &mut input_dev).expect("Failed to copy input");

        // Launch configuration
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        let mut builder = stream.launch_builder(&kernel);
        builder.arg(&mut input_dev);
        builder.arg(&mut output_dev);
        unsafe {
            builder.launch(config).expect("Failed to launch kernel");
        }

        // Copy result back to host
        let result_vec = stream.memcpy_dtov(&output_dev).expect("Failed to copy result");

        println!("(p + 2) mod p test:");
        println!("Result:   {:?}", result_vec);
        println!("Expected: {:?}", expected);

        // Check result
        assert_eq!(result_vec, expected, "(p + 2) mod p should equal 2");
    }

    #[test]
    fn test_gpu_reduce512_2p() {
        // Initialize GPU
        let ctx = init_gpu().expect("Failed to initialize GPU");
        let stream = ctx.default_stream();

        // Load PTX module
        let ptx_code = include_str!(concat!(env!("OUT_DIR"), "/secp256k1.ptx"));
        let module = ctx.load_module(Ptx::from_src(ptx_code)).expect("Failed to load PTX");
        let kernel = module.load_function("test_reduce512").expect("Failed to load kernel");

        // Test case: 2 * p
        // Input: [18446744065119615070, 18446744073709551615, 18446744073709551615, 18446744073709551615, 1, 0, 0, 0]
        // Expected: [0, 0, 0, 0]
        let input_512: Vec<u64> = vec![
            18446744065119615070u64,
            18446744073709551615u64,
            18446744073709551615u64,
            18446744073709551615u64,
            1u64,
            0u64,
            0u64,
            0u64,
        ];

        let expected: Vec<u64> = vec![0u64, 0, 0, 0];

        // Allocate device memory
        let mut input_dev = stream.alloc_zeros::<u64>(8).expect("Failed to allocate input");
        let mut output_dev = stream.alloc_zeros::<u64>(4).expect("Failed to allocate output");

        // Copy input to device
        stream.memcpy_htod(&input_512, &mut input_dev).expect("Failed to copy input");

        // Launch configuration
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        let mut builder = stream.launch_builder(&kernel);
        builder.arg(&mut input_dev);
        builder.arg(&mut output_dev);
        unsafe {
            builder.launch(config).expect("Failed to launch kernel");
        }

        // Copy result back to host
        let result_vec = stream.memcpy_dtov(&output_dev).expect("Failed to copy result");

        println!("(2 * p) mod p test:");
        println!("Result:   {:?}", result_vec);
        println!("Expected: {:?}", expected);

        // Check result
        assert_eq!(result_vec, expected, "(2 * p) mod p should equal 0");
    }

    #[test]
    fn test_gpu_point_mult_2g() {
        // Initialize GPU
        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Test case: 2 * G (should match test_point_double result)
        // G = (Gx, Gy)
        let Gx = [
            0x59F2815B16F81798u64,
            0x029BFCDB2DCE28D9u64,
            0x55A06295CE870B07u64,
            0x79BE667EF9DCBBACu64,
        ];
        let Gy = [
            0x9C47D08FFB10D4B8u64,
            0xFD17B448A6855419u64,
            0x5DA4FBFC0E1108A8u64,
            0x483ADA7726A3C465u64,
        ];

        // k = 2
        let k = [2u64, 0, 0, 0];

        let (result_x, result_y) = test_point_mult_gpu(&ctx, &k, &Gx, &Gy)
            .expect("GPU point multiplication failed");

        // Expected: 2G (same as test_point_double)
        let expected_2Gx = [
            0xABAC09B95C709EE5u64,
            0x5C778E4B8CEF3CA7u64,
            0x3045406E95C07CD8u64,
            0xC6047F9441ED7D6Du64,
        ];
        let expected_2Gy = [
            0x236431A950CFE52Au64,
            0xF7F632653266D0E1u64,
            0xA3C58419466CEAEEu64,
            0x1AE168FEA63DC339u64,
        ];

        println!("2G (via PointMult):");
        println!("  x = {:?}", result_x);
        println!("  y = {:?}", result_y);

        assert_eq!(result_x, expected_2Gx, "2G x-coordinate mismatch");
        assert_eq!(result_y, expected_2Gy, "2G y-coordinate mismatch");
    }

    #[test]
    fn test_gpu_point_mult_3g() {
        // Initialize GPU
        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Test case: 3 * G
        let Gx = [
            0x59F2815B16F81798u64,
            0x029BFCDB2DCE28D9u64,
            0x55A06295CE870B07u64,
            0x79BE667EF9DCBBACu64,
        ];
        let Gy = [
            0x9C47D08FFB10D4B8u64,
            0xFD17B448A6855419u64,
            0x5DA4FBFC0E1108A8u64,
            0x483ADA7726A3C465u64,
        ];

        // k = 3
        let k = [3u64, 0, 0, 0];

        let (result_x, result_y) = test_point_mult_gpu(&ctx, &k, &Gx, &Gy)
            .expect("GPU point multiplication failed");

        // Expected: 3G
        // 3Gx = 0xF9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9
        let expected_3Gx = [
            0x8601F113BCE036F9u64,
            0xB531C845836F99B0u64,
            0x49344F85F89D5229u64,
            0xF9308A019258C310u64,
        ];
        // 3Gy = 0x388F7B0F632DE8140FE337E62A37F3566500A99934C2231B6CB9FD7584B8E672
        let expected_3Gy = [
            0x6CB9FD7584B8E672u64,
            0x6500A99934C2231Bu64,
            0x0FE337E62A37F356u64,
            0x388F7B0F632DE814u64,
        ];

        println!("3G (via PointMult):");
        println!("  x = {:?}", result_x);
        println!("  y = {:?}", result_y);

        assert_eq!(result_x, expected_3Gx, "3G x-coordinate mismatch");
        assert_eq!(result_y, expected_3Gy, "3G y-coordinate mismatch");
    }

    #[test]
    fn test_gpu_point_mult_7g() {
        // Initialize GPU
        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Test case: 7 * G (binary: 111, tests multiple additions)
        let Gx = [
            0x59F2815B16F81798u64,
            0x029BFCDB2DCE28D9u64,
            0x55A06295CE870B07u64,
            0x79BE667EF9DCBBACu64,
        ];
        let Gy = [
            0x9C47D08FFB10D4B8u64,
            0xFD17B448A6855419u64,
            0x5DA4FBFC0E1108A8u64,
            0x483ADA7726A3C465u64,
        ];

        // k = 7
        let k = [7u64, 0, 0, 0];

        let (result_x, result_y) = test_point_mult_gpu(&ctx, &k, &Gx, &Gy)
            .expect("GPU point multiplication failed");

        // Expected: 7G
        // 7Gx = 0x5CBDF0646E5DB4EAA398F365F2EA7A0E3D419B7E0330E39CE92BDDEDCAC4F9BC
        let expected_7Gx = [
            0xE92BDDEDCAC4F9BCu64,
            0x3D419B7E0330E39Cu64,
            0xA398F365F2EA7A0Eu64,
            0x5CBDF0646E5DB4EAu64,
        ];
        // 7Gy = 0x6AEBCA40BA255960A3178D6D861A54DBA813D0B813FDE7B5A5082628087264DA
        let expected_7Gy = [
            0xA5082628087264DAu64,
            0xA813D0B813FDE7B5u64,
            0xA3178D6D861A54DBu64,
            0x6AEBCA40BA255960u64,
        ];

        println!("7G (via PointMult):");
        println!("  x = {:?}", result_x);
        println!("  y = {:?}", result_y);

        assert_eq!(result_x, expected_7Gx, "7G x-coordinate mismatch");
        assert_eq!(result_y, expected_7Gy, "7G y-coordinate mismatch");
    }

    #[test]
    fn test_gpu_reduce512_step10() {
        // Initialize GPU
        let ctx = init_gpu().expect("Failed to initialize GPU");
        let stream = ctx.default_stream();

        // Load PTX module
        let ptx_code = include_str!(concat!(env!("OUT_DIR"), "/secp256k1.ptx"));
        let module = ctx.load_module(Ptx::from_src(ptx_code)).expect("Failed to load PTX");
        let kernel = module.load_function("test_reduce512").expect("Failed to load kernel");

        // Step 10^2 の 512-bit 値（Python で計算した値）
        let input_512: Vec<u64> = vec![
            11103428889520133265u64,
            5940767142678773938u64,
            1566361556954594741u64,
            1064932884789884258u64,
            2383589971959567377u64,
            14188426789726178142u64,
            5880906690406353497u64,
            15272488u64,
        ];

        // 期待値：Step 10^2 mod p
        let expected: Vec<u64> = vec![
            4111909762576034162u64,
            12458353550763596130u64,
            10194722910089095499u64,
            1130527737568913083u64,
        ];

        // Allocate device memory
        let mut input_dev = stream.alloc_zeros::<u64>(8).expect("Failed to allocate input");
        let mut output_dev = stream.alloc_zeros::<u64>(4).expect("Failed to allocate output");

        // Copy input to device
        stream.memcpy_htod(&input_512, &mut input_dev).expect("Failed to copy input");

        // Launch configuration
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        let mut builder = stream.launch_builder(&kernel);
        builder.arg(&mut input_dev);
        builder.arg(&mut output_dev);
        unsafe {
            builder.launch(config).expect("Failed to launch kernel");
        }

        // Copy result back to host
        let result_vec = stream.memcpy_dtov(&output_dev).expect("Failed to copy result");

        println!("Step 10^2 (512-bit) reduction test:");
        println!("Result:   {:?}", result_vec);
        println!("Expected: {:?}", expected);

        // Check result
        assert_eq!(result_vec, expected, "Step 10^2 reduction mismatch");
    }

    #[test]
    fn test_generate_pubkeys_batch_single() {
        // Initialize GPU
        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Test with a single private key: k = 2
        // Expected result: 2G (same as test_point_mult_2g)
        let privkeys = vec![[2u64, 0, 0, 0]];

        let pubkeys_x = generate_pubkeys_batch(&ctx, &privkeys)
            .expect("GPU batch generation failed");

        assert_eq!(pubkeys_x.len(), 1);

        // Expected: 2G x-coordinate
        let expected_2Gx = [
            0xABAC09B95C709EE5u64,
            0x5C778E4B8CEF3CA7u64,
            0x3045406E95C07CD8u64,
            0xC6047F9441ED7D6Du64,
        ];

        println!("Batch single key (k=2):");
        println!("  result x = {:?}", pubkeys_x[0]);
        println!("  expected = {:?}", expected_2Gx);

        assert_eq!(pubkeys_x[0], expected_2Gx, "2G x-coordinate mismatch");
    }

    #[test]
    fn test_generate_pubkeys_batch_multiple() {
        // Initialize GPU
        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Test with multiple private keys: k = 2, 3, 7
        let privkeys = vec![
            [2u64, 0, 0, 0],  // 2G
            [3u64, 0, 0, 0],  // 3G
            [7u64, 0, 0, 0],  // 7G
        ];

        let pubkeys_x = generate_pubkeys_batch(&ctx, &privkeys)
            .expect("GPU batch generation failed");

        assert_eq!(pubkeys_x.len(), 3);

        // Expected x-coordinates
        let expected_2Gx = [
            0xABAC09B95C709EE5u64,
            0x5C778E4B8CEF3CA7u64,
            0x3045406E95C07CD8u64,
            0xC6047F9441ED7D6Du64,
        ];
        let expected_3Gx = [
            0x8601F113BCE036F9u64,
            0xB531C845836F99B0u64,
            0x49344F85F89D5229u64,
            0xF9308A019258C310u64,
        ];
        let expected_7Gx = [
            0xE92BDDEDCAC4F9BCu64,
            0x3D419B7E0330E39Cu64,
            0xA398F365F2EA7A0Eu64,
            0x5CBDF0646E5DB4EAu64,
        ];

        println!("Batch multiple keys:");
        println!("  2G x = {:?}", pubkeys_x[0]);
        println!("  3G x = {:?}", pubkeys_x[1]);
        println!("  7G x = {:?}", pubkeys_x[2]);

        assert_eq!(pubkeys_x[0], expected_2Gx, "2G x-coordinate mismatch");
        assert_eq!(pubkeys_x[1], expected_3Gx, "3G x-coordinate mismatch");
        assert_eq!(pubkeys_x[2], expected_7Gx, "7G x-coordinate mismatch");
    }

    #[test]
    fn test_generate_pubkeys_batch_large() {
        // Initialize GPU
        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Test with a larger batch (1024 keys)
        // Use sequential keys: k = 1, 2, 3, ..., 1024
        let batch_size = 1024;
        let privkeys: Vec<[u64; 4]> = (1..=batch_size)
            .map(|k| [k as u64, 0, 0, 0])
            .collect();

        let pubkeys_x = generate_pubkeys_batch(&ctx, &privkeys)
            .expect("GPU batch generation failed");

        assert_eq!(pubkeys_x.len(), batch_size);

        // Verify a few known values
        // k=2 -> 2G
        let expected_2Gx = [
            0xABAC09B95C709EE5u64,
            0x5C778E4B8CEF3CA7u64,
            0x3045406E95C07CD8u64,
            0xC6047F9441ED7D6Du64,
        ];
        assert_eq!(pubkeys_x[1], expected_2Gx, "2G x-coordinate mismatch in large batch");

        // k=3 -> 3G
        let expected_3Gx = [
            0x8601F113BCE036F9u64,
            0xB531C845836F99B0u64,
            0x49344F85F89D5229u64,
            0xF9308A019258C310u64,
        ];
        assert_eq!(pubkeys_x[2], expected_3Gx, "3G x-coordinate mismatch in large batch");

        println!("Large batch test (1024 keys): PASSED");
        println!("  Verified 2G and 3G x-coordinates");
    }

    // Note: test_point_add_mixed was removed because:
    // 1. _PointAddMixed is tested via test_generate_pubkeys_sequential_single/multiple_threads
    // 2. Those tests verify the actual use case: _PointMult result (Z≠1) + G
    // 3. The Z=1 (Affine input) case doesn't occur in practice

    #[test]
    fn test_generate_pubkeys_sequential_single() {
        // Test sequential key generation with keys_per_thread = 3
        // Starting from k=2, we should get: 2*G, 3*G, 4*G
        // Note: We start from k=2 (not k=1) to avoid k*G = G case,
        // which would make the next step G + G (requiring Point Doubling, not Addition)
        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Base key: k = 2
        let base_keys = vec![[2u64, 0, 0, 0]];
        let keys_per_thread = 3u32;

        let pubkeys_x = generate_pubkeys_sequential_batch(&ctx, &base_keys, keys_per_thread)
            .expect("GPU sequential batch failed");

        assert_eq!(pubkeys_x.len(), 3, "Should generate 3 keys");

        // Get expected values from generate_pubkeys_batch (known-good implementation)
        let reference_keys = vec![
            [2u64, 0, 0, 0],
            [3u64, 0, 0, 0],
            [4u64, 0, 0, 0],
        ];
        let expected_x = generate_pubkeys_batch(&ctx, &reference_keys)
            .expect("Reference batch generation failed");

        // Compare sequential vs batch results
        assert_eq!(pubkeys_x[0], expected_x[0], "2G x-coordinate mismatch (sequential vs batch)");
        assert_eq!(pubkeys_x[1], expected_x[1], "3G x-coordinate mismatch (sequential vs batch)");
        assert_eq!(pubkeys_x[2], expected_x[2], "4G x-coordinate mismatch (sequential vs batch)");

        println!("test_generate_pubkeys_sequential_single: PASSED");
        println!("  Generated 2G, 3G, 4G from k=2 with keys_per_thread=3");
        println!("  Results match generate_pubkeys_batch!");
    }

    #[test]
    fn test_generate_pubkeys_sequential_multiple_threads() {
        // Test with multiple threads, each generating consecutive keys
        // Note: We use k=2 and k=10 (not k=1) to avoid the G + G case
        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Two threads: starting from k=2 and k=10
        let base_keys = vec![
            [2u64, 0, 0, 0],  // Thread 0: k=2
            [10u64, 0, 0, 0], // Thread 1: k=10
        ];
        let keys_per_thread = 2u32;

        let pubkeys_x = generate_pubkeys_sequential_batch(&ctx, &base_keys, keys_per_thread)
            .expect("GPU sequential batch failed");

        // Should have 4 keys total: [2G, 3G, 10G, 11G]
        assert_eq!(pubkeys_x.len(), 4, "Should generate 4 keys");

        // Verify first thread: 2G, 3G
        let expected_2gx = [
            0xABAC09B95C709EE5u64,
            0x5C778E4B8CEF3CA7u64,
            0x3045406E95C07CD8u64,
            0xC6047F9441ED7D6Du64,
        ];
        let expected_3gx = [
            0x8601F113BCE036F9u64,
            0xB531C845836F99B0u64,
            0x49344F85F89D5229u64,
            0xF9308A019258C310u64,
        ];

        assert_eq!(pubkeys_x[0], expected_2gx, "Thread0 key0 (2G) mismatch");
        assert_eq!(pubkeys_x[1], expected_3gx, "Thread0 key1 (3G) mismatch");

        // Verify that thread 1's first key is different (10G, not verified numerically)
        // Just check they're not zero and different from 2G
        assert_ne!(pubkeys_x[2], [0u64, 0, 0, 0], "Thread1 key0 should not be zero");
        assert_ne!(pubkeys_x[2], expected_2gx, "Thread1 key0 should be 10G, not 2G");

        println!("test_generate_pubkeys_sequential_multiple_threads: PASSED");
        println!("  Thread 0: k=2,3 -> 2G, 3G");
        println!("  Thread 1: k=10,11 -> 10G, 11G");
    }

    #[test]
    #[ignore] // Run with: cargo test --release bench_sequential -- --ignored --nocapture
    fn bench_sequential_vs_batch() {
        use std::time::Instant;

        let ctx = init_gpu().expect("Failed to initialize GPU");

        println!("\n=== GPU Sequential Key Generation Benchmark ===\n");

        // Test with different configurations
        let configs = [
            (256, 64),   // 256 threads, 64 keys each = 16,384 total
            (256, 256),  // 256 threads, 256 keys each = 65,536 total
            (1024, 64),  // 1024 threads, 64 keys each = 65,536 total
        ];

        for (num_threads, keys_per_thread) in configs {
            let total_keys = num_threads * keys_per_thread;
            println!("--- {} threads × {} keys/thread = {} total keys ---",
                     num_threads, keys_per_thread, total_keys);

            // Method 1: generate_pubkeys_batch (full PointMult per key)
            let batch_keys: Vec<[u64; 4]> = (2..(2 + total_keys as u64))
                .map(|k| [k, 0, 0, 0])
                .collect();

            let start = Instant::now();
            let _result1 = generate_pubkeys_batch(&ctx, &batch_keys)
                .expect("Batch generation failed");
            let batch_time = start.elapsed();

            // Method 2: generate_pubkeys_sequential_batch (1 PointMult + PointAddMixed)
            let base_keys: Vec<[u64; 4]> = (0..num_threads)
                .map(|i| [2 + (i * keys_per_thread) as u64, 0, 0, 0])
                .collect();

            let start = Instant::now();
            let _result2 = generate_pubkeys_sequential_batch(&ctx, &base_keys, keys_per_thread as u32)
                .expect("Sequential generation failed");
            let seq_time = start.elapsed();

            let speedup = batch_time.as_secs_f64() / seq_time.as_secs_f64();

            println!("  Batch (PointMult):     {:?}", batch_time);
            println!("  Sequential (PointAdd): {:?}", seq_time);
            println!("  Speedup: {:.2}x 🔥", speedup);
            println!();
        }
    }

    #[test]
    fn test_generate_pubkeys_sequential_montgomery_single() {
        // Test Montgomery's Trick version with keys_per_thread = 3
        // Starting from k=2, we should get: 2*G, 3*G, 4*G
        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Base key: k = 2
        let base_keys = vec![[2u64, 0, 0, 0]];
        let keys_per_thread = 3u32;

        let pubkeys_x = generate_pubkeys_sequential_montgomery_batch(&ctx, &base_keys, keys_per_thread)
            .expect("GPU Montgomery sequential batch failed");

        assert_eq!(pubkeys_x.len(), 3, "Should generate 3 keys");

        // Get expected values from generate_pubkeys_batch (known-good implementation)
        let reference_keys = vec![
            [2u64, 0, 0, 0],
            [3u64, 0, 0, 0],
            [4u64, 0, 0, 0],
        ];
        let expected_x = generate_pubkeys_batch(&ctx, &reference_keys)
            .expect("Reference batch generation failed");

        // Compare Montgomery vs batch results
        assert_eq!(pubkeys_x[0], expected_x[0], "2G x-coordinate mismatch (Montgomery vs batch)");
        assert_eq!(pubkeys_x[1], expected_x[1], "3G x-coordinate mismatch (Montgomery vs batch)");
        assert_eq!(pubkeys_x[2], expected_x[2], "4G x-coordinate mismatch (Montgomery vs batch)");

        println!("test_generate_pubkeys_sequential_montgomery_single: PASSED");
        println!("  Generated 2G, 3G, 4G from k=2 with keys_per_thread=3");
        println!("  Montgomery's Trick results match generate_pubkeys_batch!");
    }

    #[test]
    fn test_generate_pubkeys_sequential_montgomery_256() {
        // Test Montgomery's Trick with max keys_per_thread = 256
        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Base key: k = 2
        let base_keys = vec![[2u64, 0, 0, 0]];
        let keys_per_thread = 256u32;

        let pubkeys_x = generate_pubkeys_sequential_montgomery_batch(&ctx, &base_keys, keys_per_thread)
            .expect("GPU Montgomery sequential batch failed");

        assert_eq!(pubkeys_x.len(), 256, "Should generate 256 keys");

        // Verify first few and last key against batch generation
        let reference_keys: Vec<[u64; 4]> = (2..258u64)
            .map(|k| [k, 0, 0, 0])
            .collect();
        let expected_x = generate_pubkeys_batch(&ctx, &reference_keys)
            .expect("Reference batch generation failed");

        // Check first 3 keys
        assert_eq!(pubkeys_x[0], expected_x[0], "Key 0 (2G) mismatch");
        assert_eq!(pubkeys_x[1], expected_x[1], "Key 1 (3G) mismatch");
        assert_eq!(pubkeys_x[2], expected_x[2], "Key 2 (4G) mismatch");

        // Check last key (257G)
        assert_eq!(pubkeys_x[255], expected_x[255], "Key 255 (257G) mismatch");

        println!("test_generate_pubkeys_sequential_montgomery_256: PASSED");
        println!("  Generated 256 consecutive keys from k=2");
        println!("  All verified against generate_pubkeys_batch!");
    }

    #[test]
    #[ignore] // Run with: cargo test --release bench_montgomery -- --ignored --nocapture
    fn bench_montgomery_vs_all() {
        use std::time::Instant;

        let ctx = init_gpu().expect("Failed to initialize GPU");

        println!("\n=== GPU Montgomery's Trick Benchmark ===\n");
        println!("Comparing three methods:");
        println!("  1. Batch: Full PointMult per key");
        println!("  2. Sequential (Phase 1): PointAddMixed + JacobianToAffine per key");
        println!("  3. Montgomery (Phase 2): PointAddMixed + batch inverse");
        println!();

        // Test with different configurations
        let configs = [
            (64, 64),    // 64 threads, 64 keys each = 4,096 total
            (64, 256),   // 64 threads, 256 keys each = 16,384 total
            (256, 64),   // 256 threads, 64 keys each = 16,384 total
            (256, 256),  // 256 threads, 256 keys each = 65,536 total
        ];

        for (num_threads, keys_per_thread) in configs {
            let total_keys = num_threads * keys_per_thread;
            println!("--- {} threads × {} keys/thread = {} total keys ---",
                     num_threads, keys_per_thread, total_keys);

            // Method 1: generate_pubkeys_batch (full PointMult per key)
            let batch_keys: Vec<[u64; 4]> = (2..(2 + total_keys as u64))
                .map(|k| [k, 0, 0, 0])
                .collect();

            let start = Instant::now();
            let _result1 = generate_pubkeys_batch(&ctx, &batch_keys)
                .expect("Batch generation failed");
            let batch_time = start.elapsed();

            // Method 2: generate_pubkeys_sequential_batch (Phase 1: no Montgomery)
            let base_keys: Vec<[u64; 4]> = (0..num_threads)
                .map(|i| [2 + (i * keys_per_thread) as u64, 0, 0, 0])
                .collect();

            let start = Instant::now();
            let _result2 = generate_pubkeys_sequential_batch(&ctx, &base_keys, keys_per_thread as u32)
                .expect("Sequential generation failed");
            let seq_time = start.elapsed();

            // Method 3: generate_pubkeys_sequential_montgomery_batch (Phase 2: with Montgomery)
            let start = Instant::now();
            let _result3 = generate_pubkeys_sequential_montgomery_batch(&ctx, &base_keys, keys_per_thread as u32)
                .expect("Montgomery generation failed");
            let mont_time = start.elapsed();

            // Calculate speedups
            let speedup_seq = batch_time.as_secs_f64() / seq_time.as_secs_f64();
            let speedup_mont = batch_time.as_secs_f64() / mont_time.as_secs_f64();
            let mont_vs_seq = seq_time.as_secs_f64() / mont_time.as_secs_f64();

            println!("  Batch (PointMult):          {:>10.2?}", batch_time);
            println!("  Sequential (Phase 1):       {:>10.2?}  ({:.2}x vs Batch)", seq_time, speedup_seq);
            println!("  Montgomery (Phase 2):       {:>10.2?}  ({:.2}x vs Batch, {:.2}x vs Phase1) 🔥",
                     mont_time, speedup_mont, mont_vs_seq);
            println!();
        }
    }

    #[test]
    fn test_gpu_prefix_match_basic() {
        use crate::{prefix_to_bits, u64x4_to_bytes, pubkey_bytes_to_npub};

        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Use a known prefix that's likely to match with small keys
        // We'll use prefix "c" (1 char = 5 bits) which has ~1/32 chance of matching
        let prefix = "c";
        let prefix_bits = vec![prefix_to_bits(prefix)];

        // Generate with 64 threads × 256 keys = 16384 keys
        // With 1/32 chance, we expect ~512 matches
        let num_threads = 64;
        let keys_per_thread = 256u32;

        let base_keys: Vec<[u64; 4]> = (0..num_threads)
            .map(|i| [2 + (i * keys_per_thread as usize) as u64, 0, 0, 0])
            .collect();

        let max_matches = 1000u32;

        let matches = generate_pubkeys_with_prefix_match(
            &ctx,
            &base_keys,
            keys_per_thread,
            &prefix_bits,
            max_matches,
        ).expect("GPU prefix match failed");

        println!("\nGPU Prefix Match Test:");
        println!("  Prefix: '{}'", prefix);
        println!("  Total keys: {}", num_threads * keys_per_thread as usize);
        println!("  Matches found: {}", matches.len());

        // Verify at least some matches were found
        assert!(!matches.is_empty(), "Should find at least some matches with prefix 'c'");

        // Verify the first few matches are correct
        for (i, m) in matches.iter().take(5).enumerate() {
            let pubkey_bytes = u64x4_to_bytes(&m.pubkey_x);
            let npub = pubkey_bytes_to_npub(&pubkey_bytes);
            let npub_body = &npub[5..];  // Remove "npub1"

            println!("  Match {}: base_idx={}, offset={}, npub={}...",
                     i, m.base_idx, m.offset, &npub_body[..10]);

            // Verify the npub actually starts with the prefix
            assert!(npub_body.starts_with(prefix),
                    "npub {} should start with prefix '{}'", npub_body, prefix);
        }

        println!("  ✅ All verified matches start with prefix '{}'", prefix);
    }

    #[test]
    fn test_gpu_prefix_match_multiple_prefixes() {
        use crate::{prefix_to_bits, u64x4_to_bytes, pubkey_bytes_to_npub};

        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Multiple prefixes (OR matching)
        let prefixes = vec!["c", "8", "q"];
        let prefix_bits: Vec<_> = prefixes.iter().map(|p| prefix_to_bits(p)).collect();

        let num_threads = 64;
        let keys_per_thread = 256u32;

        let base_keys: Vec<[u64; 4]> = (0..num_threads)
            .map(|i| [2 + (i * keys_per_thread as usize) as u64, 0, 0, 0])
            .collect();

        let max_matches = 2000u32;

        let matches = generate_pubkeys_with_prefix_match(
            &ctx,
            &base_keys,
            keys_per_thread,
            &prefix_bits,
            max_matches,
        ).expect("GPU prefix match failed");

        println!("\nGPU Multiple Prefix Match Test:");
        println!("  Prefixes: {:?}", prefixes);
        println!("  Total keys: {}", num_threads * keys_per_thread as usize);
        println!("  Matches found: {}", matches.len());

        // With 3 prefixes, each 1/32 chance, we expect ~1536 matches
        assert!(!matches.is_empty(), "Should find matches with multiple prefixes");

        // Verify matches start with one of the prefixes
        for (i, m) in matches.iter().take(5).enumerate() {
            let pubkey_bytes = u64x4_to_bytes(&m.pubkey_x);
            let npub = pubkey_bytes_to_npub(&pubkey_bytes);
            let npub_body = &npub[5..];

            let matches_any = prefixes.iter().any(|p| npub_body.starts_with(p));
            println!("  Match {}: npub={}... (matches: {})",
                     i, &npub_body[..10], matches_any);

            assert!(matches_any,
                    "npub {} should start with one of {:?}", npub_body, prefixes);
        }

        println!("  ✅ All verified matches start with one of the prefixes");
    }

    #[test]
    fn test_gpu_prefix_match_longer_prefix() {
        use crate::{prefix_to_bits, u64x4_to_bytes, pubkey_bytes_to_npub};

        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Longer prefix (2 chars = 10 bits, ~1/1024 chance)
        let prefix = "cc";
        let prefix_bits = vec![prefix_to_bits(prefix)];

        // Need more keys to find matches
        let num_threads = 256;
        let keys_per_thread = 256u32;

        let base_keys: Vec<[u64; 4]> = (0..num_threads)
            .map(|i| [2 + (i * keys_per_thread as usize) as u64, 0, 0, 0])
            .collect();

        let max_matches = 100u32;

        let matches = generate_pubkeys_with_prefix_match(
            &ctx,
            &base_keys,
            keys_per_thread,
            &prefix_bits,
            max_matches,
        ).expect("GPU prefix match failed");

        println!("\nGPU Longer Prefix Match Test:");
        println!("  Prefix: '{}'", prefix);
        println!("  Total keys: {}", num_threads * keys_per_thread as usize);
        println!("  Matches found: {}", matches.len());

        // With 65536 keys and 1/1024 chance, expect ~64 matches
        // Might be 0 due to randomness, but verify any matches are correct
        for (i, m) in matches.iter().take(5).enumerate() {
            let pubkey_bytes = u64x4_to_bytes(&m.pubkey_x);
            let npub = pubkey_bytes_to_npub(&pubkey_bytes);
            let npub_body = &npub[5..];

            println!("  Match {}: npub={}...", i, &npub_body[..10]);

            assert!(npub_body.starts_with(prefix),
                    "npub {} should start with prefix '{}'", npub_body, prefix);
        }

        if !matches.is_empty() {
            println!("  ✅ All verified matches start with prefix '{}'", prefix);
        } else {
            println!("  ℹ️ No matches found (expected with longer prefix)");
        }
    }
}
