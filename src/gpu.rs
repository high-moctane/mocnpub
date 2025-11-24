/*
 * GPU interface for secp256k1 operations
 *
 * This module provides Rust bindings to the CUDA implementation of secp256k1.
 */

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Initialize GPU and return context
pub fn init_gpu() -> Result<Arc<CudaContext>, Box<dyn std::error::Error>> {
    // Get GPU context (already returns Arc<CudaContext>)
    let ctx = CudaContext::new(0)?;
    Ok(ctx)
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
    let ptx_code = include_str!("../cuda/secp256k1.ptx");
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
    let ptx_code = include_str!("../cuda/secp256k1.ptx");
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
    let ptx_code = include_str!("../cuda/secp256k1.ptx");
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
    let ptx_code = include_str!("../cuda/secp256k1.ptx");
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
pub fn test_point_double_gpu(
    ctx: &Arc<CudaContext>,
    x: &[u64; 4],
    y: &[u64; 4],
) -> Result<([u64; 4], [u64; 4]), Box<dyn std::error::Error>> {
    // Get default stream
    let stream = ctx.default_stream();

    // Load PTX module
    let ptx_code = include_str!("../cuda/secp256k1.ptx");
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

#[cfg(test)]
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
        let ptx_code = include_str!("../cuda/secp256k1.ptx");
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
        let ptx_code = include_str!("../cuda/secp256k1.ptx");
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
        let ptx_code = include_str!("../cuda/secp256k1.ptx");
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
    fn test_gpu_reduce512_step10() {
        // Initialize GPU
        let ctx = init_gpu().expect("Failed to initialize GPU");
        let stream = ctx.default_stream();

        // Load PTX module
        let ptx_code = include_str!("../cuda/secp256k1.ptx");
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
}
