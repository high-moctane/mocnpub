/*
 * GPU interface for secp256k1 operations
 *
 * This module provides Rust bindings to the CUDA implementation of secp256k1.
 */

use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Get MAX_KEYS_PER_THREAD (compile-time constant)
/// This matches the value used in the CUDA kernel.
pub fn get_max_keys_per_thread() -> u32 {
    env!("MAX_KEYS_PER_THREAD")
        .parse()
        .expect("MAX_KEYS_PER_THREAD must be a valid u32")
}

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
    let waves = desired_batch_size.div_ceil(threads_per_wave);
    Ok(waves * threads_per_wave)
}

/// Test modular addition on GPU
///
/// This function tests the _ModAdd function by adding two 256-bit numbers modulo p
#[cfg(test)]
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
    let result_vec = stream.clone_dtoh(&output_dev)?;

    // Convert to fixed-size array
    let mut result = [0u64; 4];
    result.copy_from_slice(&result_vec);

    Ok(result)
}

/// Test modular multiplication on GPU
///
/// This function tests the _ModMult function by multiplying two 256-bit numbers modulo p
#[cfg(test)]
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
    let result_vec = stream.clone_dtoh(&output_dev)?;

    // Convert to fixed-size array
    let mut result = [0u64; 4];
    result.copy_from_slice(&result_vec);

    Ok(result)
}

/// Test modular inverse on GPU
///
/// This function tests the _ModInv function by computing a^(-1) mod p
#[cfg(test)]
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
    let result_vec = stream.clone_dtoh(&output_dev)?;

    // Convert to fixed-size array
    let mut result = [0u64; 4];
    result.copy_from_slice(&result_vec);

    Ok(result)
}

/// Test modular squaring on GPU
///
/// This function tests the _ModSquare function by computing a^2 mod p
#[cfg(test)]
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
    let result_vec = stream.clone_dtoh(&output_dev)?;

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
#[cfg(test)]
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
    let result_x_vec = stream.clone_dtoh(&output_x_dev)?;
    let result_y_vec = stream.clone_dtoh(&output_y_dev)?;

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
#[cfg(test)]
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
    let result_x_vec = stream.clone_dtoh(&output_x_dev)?;
    let result_y_vec = stream.clone_dtoh(&output_y_dev)?;

    // Convert to fixed-size arrays
    let mut result_x = [0u64; 4];
    let mut result_y = [0u64; 4];
    result_x.copy_from_slice(&result_x_vec);
    result_y.copy_from_slice(&result_y_vec);

    Ok((result_x, result_y))
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
    /// Base secret key (copied from GPU at match time)
    pub base_key: [u64; 4],
    /// Endomorphism type: 0 = original, 1 = β*x, 2 = β²*x
    /// Used to adjust the private key with λ or λ²
    pub endo_type: u32,
}

/// Data for a single stream buffer (sequential key strategy)
struct SequentialStreamBuffer {
    base_key_dev: cudarc::driver::CudaSlice<u64>, // Only 4 u64! (32 bytes)
    matched_base_idx_dev: cudarc::driver::CudaSlice<u32>,
    matched_offset_dev: cudarc::driver::CudaSlice<u32>,
    matched_pubkeys_x_dev: cudarc::driver::CudaSlice<u64>,
    matched_secret_keys_dev: cudarc::driver::CudaSlice<u64>,
    matched_endo_type_dev: cudarc::driver::CudaSlice<u32>,
    match_count_dev: cudarc::driver::CudaSlice<u32>,
    num_threads: usize,
}

/// Triple-buffer miner with sequential key strategy (VRAM-efficient)
///
/// Uses a single base_key per batch instead of batch_size keys:
/// - VRAM reduction: batch_size * 32 bytes -> 32 bytes per buffer
/// - Branch divergence reduction: sequential keys have similar upper bits
///
/// Uses 3 buffers for gap-free GPU utilization (same as TripleBufferMiner)
pub struct SequentialTripleBufferMiner {
    streams: Vec<std::sync::Arc<cudarc::driver::CudaStream>>,
    _module: std::sync::Arc<CudaModule>,
    kernel: CudaFunction,
    patterns_dev: cudarc::driver::CudaSlice<u64>,
    masks_dev: cudarc::driver::CudaSlice<u64>,
    num_prefixes: usize,
    max_matches: u32,
    threads_per_block: u32,
    batch_size: usize,
    bufs: Vec<SequentialStreamBuffer>,
}

impl SequentialTripleBufferMiner {
    /// Create a new sequential triple-buffer miner
    pub fn new(
        ctx: &std::sync::Arc<CudaContext>,
        prefix_bits: &[(u64, u64, u32)],
        max_matches: u32,
        threads_per_block: u32,
        batch_size: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let num_prefixes = prefix_bits.len();

        // Create three streams using fork
        let stream_0 = ctx.default_stream();
        let stream_1 = stream_0.fork()?;
        let stream_2 = stream_1.fork()?;
        let streams = vec![stream_0.clone(), stream_1, stream_2];

        // Load PTX module
        let ptx_code = include_str!(concat!(env!("OUT_DIR"), "/secp256k1.ptx"));
        let module = ctx.load_module(Ptx::from_src(ptx_code))?;
        let kernel = module.load_function("generate_pubkeys_sequential")?;

        // Prepare prefix patterns and masks
        let patterns: Vec<u64> = prefix_bits.iter().map(|(p, _, _)| *p).collect();
        let masks: Vec<u64> = prefix_bits.iter().map(|(_, m, _)| *m).collect();

        let mut patterns_dev = stream_0.alloc_zeros::<u64>(num_prefixes.max(1))?;
        let mut masks_dev = stream_0.alloc_zeros::<u64>(num_prefixes.max(1))?;
        if num_prefixes > 0 {
            stream_0.memcpy_htod(&patterns, &mut patterns_dev)?;
            stream_0.memcpy_htod(&masks, &mut masks_dev)?;
        }

        // Pre-allocate buffers for all 3 streams
        // Note: base_key_dev is only 4 u64 (32 bytes) instead of batch_size * 4!
        let mut bufs = Vec::with_capacity(3);
        for stream in &streams {
            let buf = SequentialStreamBuffer {
                base_key_dev: stream.alloc_zeros::<u64>(4)?, // Only 32 bytes!
                matched_base_idx_dev: stream.alloc_zeros::<u32>(max_matches as usize)?,
                matched_offset_dev: stream.alloc_zeros::<u32>(max_matches as usize)?,
                matched_pubkeys_x_dev: stream.alloc_zeros::<u64>(max_matches as usize * 4)?,
                matched_secret_keys_dev: stream.alloc_zeros::<u64>(max_matches as usize * 4)?,
                matched_endo_type_dev: stream.alloc_zeros::<u32>(max_matches as usize)?,
                match_count_dev: stream.alloc_zeros::<u32>(1)?,
                num_threads: batch_size,
            };
            bufs.push(buf);
        }

        // Sync to ensure all allocations are complete
        for stream in &streams {
            stream.synchronize()?;
        }

        Ok(Self {
            streams,
            _module: module,
            kernel,
            patterns_dev,
            masks_dev,
            num_prefixes,
            max_matches,
            threads_per_block,
            batch_size,
            bufs,
        })
    }

    /// Launch a single buffer's kernel asynchronously (non-blocking)
    ///
    /// # Arguments
    /// * `idx` - Buffer index (0, 1, or 2)
    /// * `base_key` - Single base key for this batch (all threads compute from this)
    pub fn launch_single(
        &mut self,
        idx: usize,
        base_key: &[u64; 4],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.num_prefixes == 0 {
            return Ok(());
        }

        let stream = &self.streams[idx];
        let buf = &mut self.bufs[idx];

        // Reset match_count to 0
        stream.memcpy_htod(&[0u32], &mut buf.match_count_dev)?;

        // Transfer single base key to device (only 32 bytes!)
        stream.memcpy_htod(base_key.as_slice(), &mut buf.base_key_dev)?;

        // Launch kernel
        let num_blocks = (buf.num_threads as u32).div_ceil(self.threads_per_block);
        let config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (self.threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        let num_threads_u32 = buf.num_threads as u32;
        let num_prefixes_u32 = self.num_prefixes as u32;

        let mut builder = stream.launch_builder(&self.kernel);
        builder.arg(&mut buf.base_key_dev);
        builder.arg(&mut self.patterns_dev);
        builder.arg(&mut self.masks_dev);
        builder.arg(&num_prefixes_u32);
        builder.arg(&mut buf.matched_base_idx_dev);
        builder.arg(&mut buf.matched_offset_dev);
        builder.arg(&mut buf.matched_pubkeys_x_dev);
        builder.arg(&mut buf.matched_secret_keys_dev);
        builder.arg(&mut buf.matched_endo_type_dev);
        builder.arg(&mut buf.match_count_dev);
        builder.arg(&num_threads_u32);
        builder.arg(&self.max_matches);
        unsafe {
            builder.launch(config)?;
        }

        Ok(())
    }

    /// Collect results from a single buffer (blocking)
    ///
    /// # Arguments
    /// * `idx` - Buffer index (0, 1, or 2)
    pub fn collect_single(&self, idx: usize) -> Result<Vec<GpuMatch>, Box<dyn std::error::Error>> {
        if self.num_prefixes == 0 {
            return Ok(vec![]);
        }

        let stream = &self.streams[idx];
        let buf = &self.bufs[idx];

        stream.synchronize()?;

        let match_count_vec = stream.clone_dtoh(&buf.match_count_dev)?;
        let match_count = match_count_vec[0].min(self.max_matches) as usize;

        if match_count == 0 {
            return Ok(vec![]);
        }

        let matched_base_idx = stream.clone_dtoh(&buf.matched_base_idx_dev)?;
        let matched_offset = stream.clone_dtoh(&buf.matched_offset_dev)?;
        let matched_pubkeys_x_flat = stream.clone_dtoh(&buf.matched_pubkeys_x_dev)?;
        let matched_secret_keys_flat = stream.clone_dtoh(&buf.matched_secret_keys_dev)?;
        let matched_endo_type = stream.clone_dtoh(&buf.matched_endo_type_dev)?;

        let mut results = Vec::with_capacity(match_count);
        for i in 0..match_count {
            let mut pubkey_x = [0u64; 4];
            pubkey_x.copy_from_slice(&matched_pubkeys_x_flat[i * 4..(i + 1) * 4]);
            let mut secret_key = [0u64; 4];
            secret_key.copy_from_slice(&matched_secret_keys_flat[i * 4..(i + 1) * 4]);

            results.push(GpuMatch {
                base_idx: matched_base_idx[i],
                offset: matched_offset[i],
                pubkey_x,
                base_key: secret_key, // This is the actual secret key
                endo_type: matched_endo_type[i],
            });
        }

        Ok(results)
    }

    /// Get the batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

/// Generate public keys with GPU-side prefix matching
///
/// This function generates public keys using Montgomery's Trick and filters
/// them by prefix on the GPU, returning only matching keys.
///
/// # Arguments
/// * `ctx` - GPU context
/// * `base_keys` - Starting private keys for each thread
/// * `prefix_bits` - Prefix patterns and masks: Vec<(pattern, mask, bit_len)>
/// * `max_matches` - Maximum number of matches to return
/// * `threads_per_block` - Number of threads per block (typically 32, 64, 128, or 256)
///
/// Note: keys_per_thread is fixed to MAX_KEYS_PER_THREAD at compile time.
///
/// # Returns
/// * `Vec<GpuMatch>` - Matching keys with their indices and public keys
pub fn generate_pubkeys_with_prefix_match(
    ctx: &Arc<CudaContext>,
    base_keys: &[[u64; 4]],
    prefix_bits: &[(u64, u64, u32)],
    max_matches: u32,
    threads_per_block: u32,
) -> Result<Vec<GpuMatch>, Box<dyn std::error::Error>> {
    let num_threads = base_keys.len();
    let num_prefixes = prefix_bits.len();

    if num_threads == 0 || num_prefixes == 0 {
        return Ok(vec![]);
    }

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
    let mut matched_base_keys_dev = stream.alloc_zeros::<u64>(max_matches as usize * 4)?;
    let mut matched_endo_type_dev = stream.alloc_zeros::<u32>(max_matches as usize)?;
    let mut match_count_dev = stream.alloc_zeros::<u32>(1)?;

    // Copy inputs to device
    stream.memcpy_htod(&base_keys_flat, &mut base_keys_dev)?;
    stream.memcpy_htod(&patterns, &mut patterns_dev)?;
    stream.memcpy_htod(&masks, &mut masks_dev)?;

    // Calculate grid and block dimensions (threads_per_block is now a parameter)
    let num_blocks = (num_threads as u32).div_ceil(threads_per_block);

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
    builder.arg(&mut matched_base_keys_dev);
    builder.arg(&mut matched_endo_type_dev); // Endomorphism type (0=original, 1=β, 2=β²)
    builder.arg(&mut match_count_dev);
    builder.arg(&num_threads_u32);
    // Note: keys_per_thread is now fixed to MAX_KEYS_PER_THREAD in the kernel
    builder.arg(&max_matches);
    unsafe {
        builder.launch(config)?;
    }

    // Copy match count back
    let match_count_vec = stream.clone_dtoh(&match_count_dev)?;
    let match_count = match_count_vec[0].min(max_matches) as usize;

    if match_count == 0 {
        return Ok(vec![]);
    }

    // Copy results back
    let matched_base_idx = stream.clone_dtoh(&matched_base_idx_dev)?;
    let matched_offset = stream.clone_dtoh(&matched_offset_dev)?;
    let matched_pubkeys_x_flat = stream.clone_dtoh(&matched_pubkeys_x_dev)?;
    let matched_base_keys_flat = stream.clone_dtoh(&matched_base_keys_dev)?;
    let matched_endo_type = stream.clone_dtoh(&matched_endo_type_dev)?;

    // Build result vector
    let mut results = Vec::with_capacity(match_count);
    for i in 0..match_count {
        let mut pubkey_x = [0u64; 4];
        pubkey_x.copy_from_slice(&matched_pubkeys_x_flat[i * 4..(i + 1) * 4]);
        let mut base_key = [0u64; 4];
        base_key.copy_from_slice(&matched_base_keys_flat[i * 4..(i + 1) * 4]);

        results.push(GpuMatch {
            base_idx: matched_base_idx[i],
            offset: matched_offset[i],
            pubkey_x,
            base_key,
            endo_type: matched_endo_type[i],
        });
    }

    Ok(results)
}

/// Generate public keys with sequential key strategy (VRAM-efficient)
///
/// This function uses a single base_key and computes sequential secret keys on GPU:
///   thread i computes keys: base_key + i * MAX_KEYS_PER_THREAD + 0, 1, 2, ...
///
/// Benefits:
///   1. VRAM reduction: batch_size * 32 bytes -> 32 bytes
///   2. Branch divergence reduction: sequential keys have similar upper bits
///   3. Enables larger batch_size and MAX_KEYS_PER_THREAD
///
/// # Arguments
/// * `ctx` - GPU context
/// * `base_key` - Single starting private key (all threads compute from this)
/// * `num_threads` - Number of threads (batch_size)
/// * `prefix_bits` - Prefix patterns and masks: Vec<(pattern, mask, bit_len)>
/// * `max_matches` - Maximum number of matches to return
/// * `threads_per_block` - Number of threads per block
///
/// # Returns
/// * `Vec<GpuMatch>` - Matching keys with their indices and actual secret keys
pub fn generate_pubkeys_sequential(
    ctx: &Arc<CudaContext>,
    base_key: &[u64; 4],
    num_threads: usize,
    prefix_bits: &[(u64, u64, u32)],
    max_matches: u32,
    threads_per_block: u32,
) -> Result<Vec<GpuMatch>, Box<dyn std::error::Error>> {
    let num_prefixes = prefix_bits.len();

    if num_threads == 0 || num_prefixes == 0 {
        return Ok(vec![]);
    }

    // Get default stream
    let stream = ctx.default_stream();

    // Load PTX module
    let ptx_code = include_str!(concat!(env!("OUT_DIR"), "/secp256k1.ptx"));
    let module = ctx.load_module(Ptx::from_src(ptx_code))?;
    let kernel = module.load_function("generate_pubkeys_sequential")?;

    // Prepare prefix patterns and masks
    let patterns: Vec<u64> = prefix_bits.iter().map(|(p, _, _)| *p).collect();
    let masks: Vec<u64> = prefix_bits.iter().map(|(_, m, _)| *m).collect();

    // Allocate device memory for inputs (only 4 u64 for base_key!)
    let mut base_key_dev = stream.alloc_zeros::<u64>(4)?;
    let mut patterns_dev = stream.alloc_zeros::<u64>(num_prefixes)?;
    let mut masks_dev = stream.alloc_zeros::<u64>(num_prefixes)?;

    // Allocate device memory for outputs
    let mut matched_base_idx_dev = stream.alloc_zeros::<u32>(max_matches as usize)?;
    let mut matched_offset_dev = stream.alloc_zeros::<u32>(max_matches as usize)?;
    let mut matched_pubkeys_x_dev = stream.alloc_zeros::<u64>(max_matches as usize * 4)?;
    let mut matched_secret_keys_dev = stream.alloc_zeros::<u64>(max_matches as usize * 4)?;
    let mut matched_endo_type_dev = stream.alloc_zeros::<u32>(max_matches as usize)?;
    let mut match_count_dev = stream.alloc_zeros::<u32>(1)?;

    // Copy inputs to device
    stream.memcpy_htod(base_key.as_slice(), &mut base_key_dev)?;
    stream.memcpy_htod(&patterns, &mut patterns_dev)?;
    stream.memcpy_htod(&masks, &mut masks_dev)?;

    // Calculate grid and block dimensions
    let num_blocks = (num_threads as u32).div_ceil(threads_per_block);

    let config = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    // Launch kernel
    let num_threads_u32 = num_threads as u32;
    let num_prefixes_u32 = num_prefixes as u32;
    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut base_key_dev);
    builder.arg(&mut patterns_dev);
    builder.arg(&mut masks_dev);
    builder.arg(&num_prefixes_u32);
    builder.arg(&mut matched_base_idx_dev);
    builder.arg(&mut matched_offset_dev);
    builder.arg(&mut matched_pubkeys_x_dev);
    builder.arg(&mut matched_secret_keys_dev);
    builder.arg(&mut matched_endo_type_dev);
    builder.arg(&mut match_count_dev);
    builder.arg(&num_threads_u32);
    builder.arg(&max_matches);
    unsafe {
        builder.launch(config)?;
    }

    // Copy match count back
    let match_count_vec = stream.clone_dtoh(&match_count_dev)?;
    let match_count = match_count_vec[0].min(max_matches) as usize;

    if match_count == 0 {
        return Ok(vec![]);
    }

    // Copy results back
    let matched_base_idx = stream.clone_dtoh(&matched_base_idx_dev)?;
    let matched_offset = stream.clone_dtoh(&matched_offset_dev)?;
    let matched_pubkeys_x_flat = stream.clone_dtoh(&matched_pubkeys_x_dev)?;
    let matched_secret_keys_flat = stream.clone_dtoh(&matched_secret_keys_dev)?;
    let matched_endo_type = stream.clone_dtoh(&matched_endo_type_dev)?;

    // Build result vector
    let mut results = Vec::with_capacity(match_count);
    for i in 0..match_count {
        let mut pubkey_x = [0u64; 4];
        pubkey_x.copy_from_slice(&matched_pubkeys_x_flat[i * 4..(i + 1) * 4]);
        let mut secret_key = [0u64; 4];
        secret_key.copy_from_slice(&matched_secret_keys_flat[i * 4..(i + 1) * 4]);

        results.push(GpuMatch {
            base_idx: matched_base_idx[i],
            offset: matched_offset[i],
            pubkey_x,
            base_key: secret_key, // This is the actual secret key now
            endo_type: matched_endo_type[i],
        });
    }

    Ok(results)
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
        println!(
            "Expected: [0x1000003D1, 0, 0, 0] = [{}, 0, 0, 0]",
            0x1000003D1u64
        );
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
        let expected = [
            0xFFFFFFFF7FFFFE18u64,
            0xFFFFFFFFFFFFFFFFu64,
            0xFFFFFFFFFFFFFFFFu64,
            0x7FFFFFFFFFFFFFFFu64,
        ];
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
        let expected = [
            0xAAAAAAA9FFFFFD75u64,
            0xAAAAAAAAAAAAAAAAu64,
            0xAAAAAAAAAAAAAAAAu64,
            0xAAAAAAAAAAAAAAAAu64,
        ];
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
        println!(
            "Expected: [0x1000003D1, 0, 0, 0] = [{}, 0, 0, 0]",
            0x1000003D1u64
        );
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
        let a = [
            18446744071562067480u64,
            18446744073709551615,
            18446744073709551615,
            9223372036854775807,
        ];
        let result = test_mod_square_gpu(&ctx, &a).expect("GPU modular squaring failed");

        println!("Step 254^2 = {:?}", result);
        let expected = [
            18446744072635809548u64,
            18446744073709551615,
            18446744073709551615,
            4611686018427387903,
        ];
        println!("Expected:    {:?}", expected);
        assert_eq!(
            result, expected,
            "Step 254 squared should match Python simulation"
        );
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

        let (result_x, result_y) =
            test_point_double_gpu(&ctx, &Gx, &Gy).expect("GPU point doubling failed");

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
        let module = ctx
            .load_module(Ptx::from_src(ptx_code))
            .expect("Failed to load PTX");
        let kernel = module
            .load_function("test_reduce512")
            .expect("Failed to load kernel");

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
        let mut input_dev = stream
            .alloc_zeros::<u64>(8)
            .expect("Failed to allocate input");
        let mut output_dev = stream
            .alloc_zeros::<u64>(4)
            .expect("Failed to allocate output");

        // Copy input to device
        stream
            .memcpy_htod(&input_512, &mut input_dev)
            .expect("Failed to copy input");

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
        let result_vec = stream
            .clone_dtoh(&output_dev)
            .expect("Failed to copy result");

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
        let module = ctx
            .load_module(Ptx::from_src(ptx_code))
            .expect("Failed to load PTX");
        let kernel = module
            .load_function("test_reduce512")
            .expect("Failed to load kernel");

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
        let mut input_dev = stream
            .alloc_zeros::<u64>(8)
            .expect("Failed to allocate input");
        let mut output_dev = stream
            .alloc_zeros::<u64>(4)
            .expect("Failed to allocate output");

        // Copy input to device
        stream
            .memcpy_htod(&input_512, &mut input_dev)
            .expect("Failed to copy input");

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
        let result_vec = stream
            .clone_dtoh(&output_dev)
            .expect("Failed to copy result");

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
        let module = ctx
            .load_module(Ptx::from_src(ptx_code))
            .expect("Failed to load PTX");
        let kernel = module
            .load_function("test_reduce512")
            .expect("Failed to load kernel");

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
        let mut input_dev = stream
            .alloc_zeros::<u64>(8)
            .expect("Failed to allocate input");
        let mut output_dev = stream
            .alloc_zeros::<u64>(4)
            .expect("Failed to allocate output");

        // Copy input to device
        stream
            .memcpy_htod(&input_512, &mut input_dev)
            .expect("Failed to copy input");

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
        let result_vec = stream
            .clone_dtoh(&output_dev)
            .expect("Failed to copy result");

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

        let (result_x, result_y) =
            test_point_mult_gpu(&ctx, &k, &Gx, &Gy).expect("GPU point multiplication failed");

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

        let (result_x, result_y) =
            test_point_mult_gpu(&ctx, &k, &Gx, &Gy).expect("GPU point multiplication failed");

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

        let (result_x, result_y) =
            test_point_mult_gpu(&ctx, &k, &Gx, &Gy).expect("GPU point multiplication failed");

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
        let module = ctx
            .load_module(Ptx::from_src(ptx_code))
            .expect("Failed to load PTX");
        let kernel = module
            .load_function("test_reduce512")
            .expect("Failed to load kernel");

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
        let mut input_dev = stream
            .alloc_zeros::<u64>(8)
            .expect("Failed to allocate input");
        let mut output_dev = stream
            .alloc_zeros::<u64>(4)
            .expect("Failed to allocate output");

        // Copy input to device
        stream
            .memcpy_htod(&input_512, &mut input_dev)
            .expect("Failed to copy input");

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
        let result_vec = stream
            .clone_dtoh(&output_dev)
            .expect("Failed to copy result");

        println!("Step 10^2 (512-bit) reduction test:");
        println!("Result:   {:?}", result_vec);
        println!("Expected: {:?}", expected);

        // Check result
        assert_eq!(result_vec, expected, "Step 10^2 reduction mismatch");
    }

    #[test]
    fn test_gpu_prefix_match_basic() {
        use crate::{prefix_to_bits, pubkey_bytes_to_npub, u64x4_to_bytes};

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

        let threads_per_block = 64u32;
        let matches = generate_pubkeys_with_prefix_match(
            &ctx,
            &base_keys,
            &prefix_bits,
            max_matches,
            threads_per_block,
        )
        .expect("GPU prefix match failed");

        let keys_per_thread = get_max_keys_per_thread();
        println!("\nGPU Prefix Match Test:");
        println!("  Prefix: '{}'", prefix);
        println!("  Total keys: {}", num_threads * keys_per_thread as usize);
        println!("  Matches found: {}", matches.len());

        // Verify at least some matches were found
        assert!(
            !matches.is_empty(),
            "Should find at least some matches with prefix 'c'"
        );

        // Verify the first few matches are correct
        for (i, m) in matches.iter().take(5).enumerate() {
            let pubkey_bytes = u64x4_to_bytes(&m.pubkey_x);
            let npub = pubkey_bytes_to_npub(&pubkey_bytes);
            let npub_body = &npub[5..]; // Remove "npub1"

            println!(
                "  Match {}: base_idx={}, offset={}, npub={}...",
                i,
                m.base_idx,
                m.offset,
                &npub_body[..10]
            );

            // Verify the npub actually starts with the prefix
            assert!(
                npub_body.starts_with(prefix),
                "npub {} should start with prefix '{}'",
                npub_body,
                prefix
            );
        }

        println!("  ✅ All verified matches start with prefix '{}'", prefix);
    }

    #[test]
    fn test_gpu_prefix_match_multiple_prefixes() {
        use crate::{prefix_to_bits, pubkey_bytes_to_npub, u64x4_to_bytes};

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

        let threads_per_block = 64u32;
        let matches = generate_pubkeys_with_prefix_match(
            &ctx,
            &base_keys,
            &prefix_bits,
            max_matches,
            threads_per_block,
        )
        .expect("GPU prefix match failed");

        let keys_per_thread = get_max_keys_per_thread();
        println!("\nGPU Multiple Prefix Match Test:");
        println!("  Prefixes: {:?}", prefixes);
        println!("  Total keys: {}", num_threads * keys_per_thread as usize);
        println!("  Matches found: {}", matches.len());

        // With 3 prefixes, each 1/32 chance, we expect ~1536 matches
        assert!(
            !matches.is_empty(),
            "Should find matches with multiple prefixes"
        );

        // Verify matches start with one of the prefixes
        for (i, m) in matches.iter().take(5).enumerate() {
            let pubkey_bytes = u64x4_to_bytes(&m.pubkey_x);
            let npub = pubkey_bytes_to_npub(&pubkey_bytes);
            let npub_body = &npub[5..];

            let matches_any = prefixes.iter().any(|p| npub_body.starts_with(p));
            println!(
                "  Match {}: npub={}... (matches: {})",
                i,
                &npub_body[..10],
                matches_any
            );

            assert!(
                matches_any,
                "npub {} should start with one of {:?}",
                npub_body, prefixes
            );
        }

        println!("  ✅ All verified matches start with one of the prefixes");
    }

    #[test]
    fn test_gpu_prefix_match_longer_prefix() {
        use crate::{prefix_to_bits, pubkey_bytes_to_npub, u64x4_to_bytes};

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

        let threads_per_block = 64u32;
        let matches = generate_pubkeys_with_prefix_match(
            &ctx,
            &base_keys,
            &prefix_bits,
            max_matches,
            threads_per_block,
        )
        .expect("GPU prefix match failed");

        let keys_per_thread = get_max_keys_per_thread();
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

            assert!(
                npub_body.starts_with(prefix),
                "npub {} should start with prefix '{}'",
                npub_body,
                prefix
            );
        }

        if !matches.is_empty() {
            println!("  ✅ All verified matches start with prefix '{}'", prefix);
        } else {
            println!("  ℹ️ No matches found (expected with longer prefix)");
        }
    }

    #[test]
    fn test_gpu_sequential_basic() {
        use crate::{prefix_to_bits, pubkey_bytes_to_npub, u64x4_to_bytes};

        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Use prefix "c" (1 char = 5 bits) which has ~1/32 chance of matching
        let prefix = "c";
        let prefix_bits = vec![prefix_to_bits(prefix)];

        // Single base key - all threads compute sequential keys from this
        let base_key: [u64; 4] = [2, 0, 0, 0];
        let num_threads = 64;
        let max_matches = 1000u32;
        let threads_per_block = 64u32;

        let matches = generate_pubkeys_sequential(
            &ctx,
            &base_key,
            num_threads,
            &prefix_bits,
            max_matches,
            threads_per_block,
        )
        .expect("GPU sequential prefix match failed");

        let keys_per_thread = get_max_keys_per_thread();
        println!("\nGPU Sequential Key Test:");
        println!("  Prefix: '{}'", prefix);
        println!("  Base key: {:?}", base_key);
        println!("  Num threads: {}", num_threads);
        println!("  Keys per thread: {}", keys_per_thread);
        println!("  Total keys: {}", num_threads * keys_per_thread as usize);
        println!("  Matches found: {}", matches.len());

        // Verify at least some matches were found
        assert!(
            !matches.is_empty(),
            "Should find at least some matches with prefix 'c'"
        );

        // Verify the first few matches are correct
        for (i, m) in matches.iter().take(5).enumerate() {
            let pubkey_bytes = u64x4_to_bytes(&m.pubkey_x);
            let npub = pubkey_bytes_to_npub(&pubkey_bytes);
            let npub_body = &npub[5..]; // Remove "npub1"

            // Compute expected secret key for verification
            let expected_offset =
                (m.base_idx as u64) * (keys_per_thread as u64) + (m.offset as u64);
            let expected_key: [u64; 4] = [
                base_key[0] + expected_offset,
                base_key[1],
                base_key[2],
                base_key[3],
            ];

            println!(
                "  Match {}: base_idx={}, offset={}, endo={}, npub={}...",
                i,
                m.base_idx,
                m.offset,
                m.endo_type,
                &npub_body[..10]
            );
            println!("    Secret key returned: {:?}", m.base_key);
            println!("    Expected key: {:?}", expected_key);

            // Verify the npub actually starts with the prefix
            assert!(
                npub_body.starts_with(prefix),
                "npub {} should start with prefix '{}'",
                npub_body,
                prefix
            );

            // Verify the secret key is correctly computed
            assert_eq!(
                m.base_key, expected_key,
                "Secret key should match expected: {:?} vs {:?}",
                m.base_key, expected_key
            );
        }

        println!(
            "  ✅ All verified matches start with prefix '{}' and have correct secret keys",
            prefix
        );
    }

    #[test]
    fn test_gpu_sequential_verify_secret_key() {
        use crate::{prefix_to_bits, pubkey_bytes_to_npub, seckey_to_nsec, u64x4_to_bytes};
        use secp256k1::{PublicKey, Secp256k1, SecretKey};

        let ctx = init_gpu().expect("Failed to initialize GPU");

        // Use an easy prefix to find matches quickly
        let prefix = "0";
        let prefix_bits = vec![prefix_to_bits(prefix)];

        let base_key: [u64; 4] = [1, 0, 0, 0];
        let num_threads = 128;
        let max_matches = 100u32;
        let threads_per_block = 128u32;

        let matches = generate_pubkeys_sequential(
            &ctx,
            &base_key,
            num_threads,
            &prefix_bits,
            max_matches,
            threads_per_block,
        )
        .expect("GPU sequential prefix match failed");

        println!("\nGPU Sequential Key Verification Test:");
        println!("  Matches found: {}", matches.len());

        assert!(!matches.is_empty(), "Should find at least one match");

        // Verify the first match by computing pubkey from secret key
        let m = &matches[0];

        // Convert GPU secret key to bytes (little-endian limbs to big-endian bytes)
        let secret_bytes = u64x4_to_bytes(&m.base_key);

        // Apply endomorphism adjustment if needed
        let secp = Secp256k1::new();
        let secret_key = SecretKey::from_slice(&secret_bytes).expect("Invalid secret key");
        let pubkey = PublicKey::from_secret_key(&secp, &secret_key);
        let pubkey_bytes: [u8; 32] = pubkey.x_only_public_key().0.serialize();

        // Convert to npub
        let npub = pubkey_bytes_to_npub(&pubkey_bytes);
        let nsec = seckey_to_nsec(&secret_key);

        println!("  First match:");
        println!(
            "    base_idx={}, offset={}, endo_type={}",
            m.base_idx, m.offset, m.endo_type
        );
        println!("    nsec: {}", nsec);
        println!("    npub: {}", npub);

        // For endo_type=0, the npub from our secret key should match
        if m.endo_type == 0 {
            let npub_body = &npub[5..]; // Remove "npub1"
            assert!(
                npub_body.starts_with(prefix),
                "npub {} should start with prefix '{}' for endo_type=0",
                npub_body,
                prefix
            );
            println!("  ✅ Secret key verification passed for endo_type=0");
        } else {
            // For endo_type 1 or 2, we need to adjust the secret key with λ
            // This is a known limitation - for now just verify the match was found
            println!(
                "  ℹ️ endo_type={}, skipping direct verification (needs λ adjustment)",
                m.endo_type
            );
        }
    }
}
