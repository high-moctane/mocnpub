/*
 * secp256k1 GPU implementation for mocnpub
 *
 * License: MIT
 */

#include <stdint.h>

// ============================================================================
// secp256k1 Constants
// ============================================================================

// Prime p = 2^256 - 2^32 - 977
// 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
__constant__ uint64_t _P[4] = {
    0xFFFFFFFEFFFFFC2FULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL
};

// Generator point G (x coordinate)
// 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
__constant__ uint64_t _Gx[4] = {
    0x59F2815B16F81798ULL,
    0x029BFCDB2DCE28D9ULL,
    0x55A06295CE870B07ULL,
    0x79BE667EF9DCBBACULL
};

// Generator point G (y coordinate)
// 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
__constant__ uint64_t _Gy[4] = {
    0x9C47D08FFB10D4B8ULL,
    0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL,
    0x483ADA7726A3C465ULL
};

// ============================================================================
// 256-bit Arithmetic Helper Functions (Device Functions)
// ============================================================================

/**
 * Add two 256-bit numbers (a + b)
 * Returns the result and carry
 */
__device__ void _Add256(const uint64_t a[4], const uint64_t b[4], uint64_t result[4], uint64_t* carry)
{
    uint64_t c = 0;

    // Add with carry
    for (int i = 0; i < 4; i++) {
        uint64_t sum = a[i] + c;
        c = (sum < c) ? 1 : 0;  // Detect overflow

        uint64_t final_sum = sum + b[i];
        c += (final_sum < b[i]) ? 1 : 0;  // Detect overflow

        result[i] = final_sum;
    }

    *carry = c;
}

/**
 * Subtract two 256-bit numbers (a - b)
 * Assumes a >= b
 */
__device__ void _Sub256(const uint64_t a[4], const uint64_t b[4], uint64_t result[4])
{
    uint64_t borrow = 0;

    for (int i = 0; i < 4; i++) {
        uint64_t diff = a[i] - b[i] - borrow;
        borrow = (a[i] < b[i] + borrow) ? 1 : 0;
        result[i] = diff;
    }
}

/**
 * Compare two 256-bit numbers
 * Returns: 1 if a > b, 0 if a == b, -1 if a < b
 */
__device__ int _Compare256(const uint64_t a[4], const uint64_t b[4])
{
    for (int i = 3; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

/**
 * Modular addition: (a + b) mod p
 */
__device__ void _ModAdd(const uint64_t a[4], const uint64_t b[4], uint64_t result[4])
{
    uint64_t sum[4];
    uint64_t carry;

    _Add256(a, b, sum, &carry);

    // If carry or sum >= p, subtract p
    if (carry || _Compare256(sum, _P) >= 0) {
        _Sub256(sum, _P, result);
    } else {
        for (int i = 0; i < 4; i++) {
            result[i] = sum[i];
        }
    }
}

/**
 * Modular subtraction: (a - b) mod p
 * If a < b, add p first
 */
__device__ void _ModSub(const uint64_t a[4], const uint64_t b[4], uint64_t result[4])
{
    if (_Compare256(a, b) >= 0) {
        _Sub256(a, b, result);
    } else {
        // a < b, so compute (a + p) - b
        uint64_t temp[4];
        uint64_t carry;
        _Add256(a, _P, temp, &carry);
        _Sub256(temp, b, result);
    }
}

/**
 * Multiply two 64-bit numbers, returning low and high parts
 */
__device__ void _Mult64(uint64_t a, uint64_t b, uint64_t* low, uint64_t* high)
{
    // Use CUDA's built-in 64-bit multiply
    *low = a * b;
    *high = __umul64hi(a, b);
}

/**
 * Reduce a 512-bit number modulo p
 * Uses secp256k1-specific reduction: p = 2^256 - 2^32 - 977
 *
 * Key insight: 2^256 mod p = 2^32 + 977
 * So: high * 2^256 mod p = high * (2^32 + 977) mod p
 */
__device__ void _Reduce512(const uint64_t in[8], uint64_t result[4])
{
    // Split into low (256-bit) and high (256-bit)
    uint64_t low[4], high[4];
    for (int i = 0; i < 4; i++) {
        low[i] = in[i];
        high[i] = in[i + 4];
    }

    // Compute high * (2^32 + 977)
    // = (high << 32) + high * 977

    // Part 1: high << 32 (shift left by 32 bits = shift by 0.5 words)
    uint64_t shifted[5] = {0};
    shifted[0] = high[0] << 32;
    shifted[1] = (high[0] >> 32) | (high[1] << 32);
    shifted[2] = (high[1] >> 32) | (high[2] << 32);
    shifted[3] = (high[2] >> 32) | (high[3] << 32);
    shifted[4] = high[3] >> 32;

    // Part 2: high * 977
    uint64_t mult977[5] = {0};
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t prod = high[i] * 977 + carry;
        mult977[i] = prod;
        carry = (high[i] >> 32) * 977 + (prod >> 32) - ((prod & 0xFFFFFFFFULL) >> 32);
    }
    mult977[4] = carry;

    // Add: shifted + mult977
    uint64_t sum[5] = {0};
    carry = 0;
    for (int i = 0; i < 5; i++) {
        uint64_t s = shifted[i] + mult977[i] + carry;
        carry = (s < shifted[i]) ? 1 : 0;
        if (mult977[i] + carry > s) carry = 1;
        sum[i] = s;
    }

    // Add to low
    uint64_t temp[5];
    temp[0] = low[0];
    temp[1] = low[1];
    temp[2] = low[2];
    temp[3] = low[3];
    temp[4] = 0;

    carry = 0;
    for (int i = 0; i < 5; i++) {
        uint64_t s = temp[i] + sum[i] + carry;
        carry = (s < temp[i]) ? 1 : 0;
        temp[i] = s;
    }

    // Now reduce: while temp >= p, subtract p
    // At most 2-3 iterations needed
    for (int iter = 0; iter < 3; iter++) {
        // Check if temp >= p
        bool ge = false;
        if (temp[4] > 0) {
            ge = true;
        } else {
            // Compare temp[0..3] with p
            ge = _Compare256(temp, _P) >= 0;
        }

        if (ge) {
            // Subtract p
            uint64_t diff[4];
            _Sub256(temp, _P, diff);
            for (int i = 0; i < 4; i++) {
                temp[i] = diff[i];
            }
            temp[4] = 0;
        } else {
            break;
        }
    }

    // Copy result
    for (int i = 0; i < 4; i++) {
        result[i] = temp[i];
    }
}

/**
 * Modular multiplication: (a * b) mod p
 * Uses secp256k1-specific reduction
 */
__device__ void _ModMult(const uint64_t a[4], const uint64_t b[4], uint64_t result[4])
{
    uint64_t temp[8] = {0};  // 512-bit result

    // Multiply a * b (256-bit × 256-bit = 512-bit)
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t low, high;
            _Mult64(a[i], b[j], &low, &high);

            // Add to temp[i+j]
            uint64_t sum = temp[i + j] + low + carry;
            carry = (sum < temp[i + j]) ? 1 : 0;
            carry += high;

            temp[i + j] = sum;
        }
        temp[i + 4] += carry;
    }

    // Reduce modulo p
    _Reduce512(temp, result);
}

/**
 * Modular squaring: (a * a) mod p
 * Optimized version of _ModMult(a, a, result)
 */
__device__ void _ModSquare(const uint64_t a[4], uint64_t result[4])
{
    // For simplicity, use _ModMult
    // TODO: Implement optimized squaring (can skip half of the multiplications)
    _ModMult(a, a, result);
}

/**
 * Modular inverse: a^(-1) mod p
 * Uses Fermat's Little Theorem: a^(p-2) mod p = a^(-1) mod p
 * Implemented using binary exponentiation (square-and-multiply, MSB to LSB)
 */
__device__ void _ModInv(const uint64_t a[4], uint64_t result[4])
{
    // p - 2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
    const uint64_t exp[4] = {
        0xFFFFFFFEFFFFFC2DULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL
    };

    // Binary exponentiation: compute a^exp mod p
    // Process from MSB to LSB
    uint64_t res[4] = {1, 0, 0, 0};  // Start with 1
    bool started = false;  // Track if we've seen the first 1 bit

    // Iterate through all bits of exp (from MSB to LSB)
    for (int i = 3; i >= 0; i--) {
        uint64_t e = exp[i];
        for (int j = 63; j >= 0; j--) {
            uint64_t bit = (e >> j) & 1;

            if (!started) {
                // Skip leading zeros until we find the first 1
                if (bit == 1) {
                    started = true;
                    // First bit: res = a (no squaring yet)
                    for (int k = 0; k < 4; k++) {
                        res[k] = a[k];
                    }
                }
            } else {
                // Square res
                uint64_t temp[4];
                _ModMult(res, res, temp);
                for (int k = 0; k < 4; k++) {
                    res[k] = temp[k];
                }

                // If bit is set, multiply by a
                if (bit == 1) {
                    _ModMult(res, a, temp);
                    for (int k = 0; k < 4; k++) {
                        res[k] = temp[k];
                    }
                }
            }
        }
    }

    for (int i = 0; i < 4; i++) {
        result[i] = res[i];
    }
}

// ============================================================================
// Elliptic Curve Point Operations (Jacobian Coordinates)
// ============================================================================

/**
 * Point Doubling in Jacobian coordinates: (X1, Y1, Z1) -> 2*(X1, Y1, Z1)
 *
 * Algorithm (secp256k1 has a=0, which simplifies M calculation):
 *   S = 4 * X1 * Y1^2
 *   M = 3 * X1^2
 *   X3 = M^2 - 2*S
 *   Y3 = M * (S - X3) - 8 * Y1^4
 *   Z3 = 2 * Y1 * Z1
 *
 * Cost: 8M + 3S (8 multiplications, 3 squarings)
 */
__device__ void _PointDouble(
    const uint64_t X1[4], const uint64_t Y1[4], const uint64_t Z1[4],
    uint64_t X3[4], uint64_t Y3[4], uint64_t Z3[4]
)
{
    uint64_t Y1_squared[4], Y1_fourth[4];
    uint64_t S[4], M[4], M_squared[4];
    uint64_t temp[4], temp2[4];

    // Y1_squared = Y1^2
    _ModSquare(Y1, Y1_squared);

    // Y1_fourth = Y1^4 = (Y1^2)^2
    _ModSquare(Y1_squared, Y1_fourth);

    // S = 4 * X1 * Y1^2
    _ModMult(X1, Y1_squared, temp);      // temp = X1 * Y1^2
    _ModAdd(temp, temp, temp2);          // temp2 = 2 * X1 * Y1^2
    _ModAdd(temp2, temp2, S);            // S = 4 * X1 * Y1^2

    // M = 3 * X1^2 (secp256k1 has a=0)
    _ModSquare(X1, temp);                // temp = X1^2
    _ModAdd(temp, temp, temp2);          // temp2 = 2 * X1^2
    _ModAdd(temp2, temp, M);             // M = 3 * X1^2

    // X3 = M^2 - 2*S
    _ModSquare(M, M_squared);            // M_squared = M^2
    _ModAdd(S, S, temp);                 // temp = 2*S
    _ModSub(M_squared, temp, X3);        // X3 = M^2 - 2*S

    // Y3 = M * (S - X3) - 8 * Y1^4
    _ModSub(S, X3, temp);                // temp = S - X3
    _ModMult(M, temp, temp2);            // temp2 = M * (S - X3)
    // 8 * Y1^4 = 2 * (2 * (2 * Y1^4))
    _ModAdd(Y1_fourth, Y1_fourth, temp); // temp = 2 * Y1^4
    _ModAdd(temp, temp, temp);           // temp = 4 * Y1^4
    _ModAdd(temp, temp, temp);           // temp = 8 * Y1^4
    _ModSub(temp2, temp, Y3);            // Y3 = M * (S - X3) - 8 * Y1^4

    // Z3 = 2 * Y1 * Z1
    _ModMult(Y1, Z1, temp);              // temp = Y1 * Z1
    _ModAdd(temp, temp, Z3);             // Z3 = 2 * Y1 * Z1
}

/**
 * Point Addition in Jacobian coordinates: (X1, Y1, Z1) + (X2, Y2, Z2)
 *
 * Algorithm:
 *   U1 = X1 * Z2^2
 *   U2 = X2 * Z1^2
 *   S1 = Y1 * Z2^3
 *   S2 = Y2 * Z1^3
 *   H = U2 - U1
 *   R = S2 - S1
 *   X3 = R^2 - H^3 - 2*U1*H^2
 *   Y3 = R * (U1*H^2 - X3) - S1*H^3
 *   Z3 = Z1 * Z2 * H
 *
 * Cost: 12M + 4S (12 multiplications, 4 squarings)
 */
__device__ void _PointAdd(
    const uint64_t X1[4], const uint64_t Y1[4], const uint64_t Z1[4],
    const uint64_t X2[4], const uint64_t Y2[4], const uint64_t Z2[4],
    uint64_t X3[4], uint64_t Y3[4], uint64_t Z3[4]
)
{
    uint64_t Z1_squared[4], Z1_cubed[4];
    uint64_t Z2_squared[4], Z2_cubed[4];
    uint64_t U1[4], U2[4], S1[4], S2[4];
    uint64_t H[4], H_squared[4], H_cubed[4];
    uint64_t R[4], R_squared[4];
    uint64_t temp[4], temp2[4];

    // Z1^2 and Z1^3
    _ModSquare(Z1, Z1_squared);
    _ModMult(Z1_squared, Z1, Z1_cubed);

    // Z2^2 and Z2^3
    _ModSquare(Z2, Z2_squared);
    _ModMult(Z2_squared, Z2, Z2_cubed);

    // U1 = X1 * Z2^2
    _ModMult(X1, Z2_squared, U1);

    // U2 = X2 * Z1^2
    _ModMult(X2, Z1_squared, U2);

    // S1 = Y1 * Z2^3
    _ModMult(Y1, Z2_cubed, S1);

    // S2 = Y2 * Z1^3
    _ModMult(Y2, Z1_cubed, S2);

    // H = U2 - U1
    _ModSub(U2, U1, H);

    // R = S2 - S1
    _ModSub(S2, S1, R);

    // H^2 and H^3
    _ModSquare(H, H_squared);
    _ModMult(H_squared, H, H_cubed);

    // R^2
    _ModSquare(R, R_squared);

    // X3 = R^2 - H^3 - 2*U1*H^2
    _ModMult(U1, H_squared, temp);       // temp = U1 * H^2
    _ModAdd(temp, temp, temp2);          // temp2 = 2 * U1 * H^2
    _ModSub(R_squared, H_cubed, temp);   // temp = R^2 - H^3
    _ModSub(temp, temp2, X3);            // X3 = R^2 - H^3 - 2*U1*H^2

    // Y3 = R * (U1*H^2 - X3) - S1*H^3
    _ModMult(U1, H_squared, temp);       // temp = U1 * H^2
    _ModSub(temp, X3, temp2);            // temp2 = U1*H^2 - X3
    _ModMult(R, temp2, temp);            // temp = R * (U1*H^2 - X3)
    _ModMult(S1, H_cubed, temp2);        // temp2 = S1 * H^3
    _ModSub(temp, temp2, Y3);            // Y3 = R * (U1*H^2 - X3) - S1*H^3

    // Z3 = Z1 * Z2 * H
    _ModMult(Z1, Z2, temp);              // temp = Z1 * Z2
    _ModMult(temp, H, Z3);               // Z3 = Z1 * Z2 * H
}

/**
 * Convert Jacobian coordinates to Affine coordinates
 * (X, Y, Z) -> (x, y) where x = X/Z^2, y = Y/Z^3
 */
__device__ void _JacobianToAffine(
    const uint64_t X[4], const uint64_t Y[4], const uint64_t Z[4],
    uint64_t x[4], uint64_t y[4]
)
{
    uint64_t Z_inv[4], Z_inv_squared[4], Z_inv_cubed[4];

    // Z_inv = Z^(-1)
    _ModInv(Z, Z_inv);

    // Z_inv^2
    _ModSquare(Z_inv, Z_inv_squared);

    // Z_inv^3 = Z_inv^2 * Z_inv
    _ModMult(Z_inv_squared, Z_inv, Z_inv_cubed);

    // x = X * Z_inv^2
    _ModMult(X, Z_inv_squared, x);

    // y = Y * Z_inv^3
    _ModMult(Y, Z_inv_cubed, y);
}

// ============================================================================
// Test Kernels
// ============================================================================

/**
 * Test kernel: Modular addition
 */
extern "C" __global__ void test_mod_add(
    const uint64_t* input_a,   // [batch_size * 4]
    const uint64_t* input_b,   // [batch_size * 4]
    uint64_t* output           // [batch_size * 4]
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t a[4], b[4], result[4];

    // Load inputs
    for (int i = 0; i < 4; i++) {
        a[i] = input_a[idx * 4 + i];
        b[i] = input_b[idx * 4 + i];
    }

    // Perform modular addition
    _ModAdd(a, b, result);

    // Store result
    for (int i = 0; i < 4; i++) {
        output[idx * 4 + i] = result[i];
    }
}

/**
 * Test kernel: Modular multiplication
 */
extern "C" __global__ void test_mod_mult(
    const uint64_t* input_a,   // [batch_size * 4]
    const uint64_t* input_b,   // [batch_size * 4]
    uint64_t* output           // [batch_size * 4]
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t a[4], b[4], result[4];

    // Load inputs
    for (int i = 0; i < 4; i++) {
        a[i] = input_a[idx * 4 + i];
        b[i] = input_b[idx * 4 + i];
    }

    // Perform modular multiplication
    _ModMult(a, b, result);

    // Store result
    for (int i = 0; i < 4; i++) {
        output[idx * 4 + i] = result[i];
    }
}

/**
 * Test kernel: Modular inverse
 */
extern "C" __global__ void test_mod_inv(
    const uint64_t* input_a,   // [batch_size * 4]
    uint64_t* output           // [batch_size * 4]
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t a[4], result[4];

    // Load input
    for (int i = 0; i < 4; i++) {
        a[i] = input_a[idx * 4 + i];
    }

    // Perform modular inverse
    _ModInv(a, result);

    // Store result
    for (int i = 0; i < 4; i++) {
        output[idx * 4 + i] = result[i];
    }
}

/**
 * Test kernel: Modular squaring
 */
extern "C" __global__ void test_mod_square(
    const uint64_t* input_a,   // [batch_size * 4]
    uint64_t* output           // [batch_size * 4]
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t a[4], result[4];

    // Load input
    for (int i = 0; i < 4; i++) {
        a[i] = input_a[idx * 4 + i];
    }

    // Perform modular squaring
    _ModSquare(a, result);

    // Store result
    for (int i = 0; i < 4; i++) {
        output[idx * 4 + i] = result[i];
    }
}

/**
 * Test kernel: Point Doubling
 * Input: Point in Jacobian coordinates (X, Y, Z)
 * Output: 2*Point in Affine coordinates (x, y)
 */
extern "C" __global__ void test_point_double(
    const uint64_t* input_X,   // [4] Jacobian X coordinate
    const uint64_t* input_Y,   // [4] Jacobian Y coordinate
    const uint64_t* input_Z,   // [4] Jacobian Z coordinate
    uint64_t* output_x,        // [4] Affine x coordinate
    uint64_t* output_y         // [4] Affine y coordinate
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only process first thread (simple test)
    if (idx != 0) return;

    uint64_t X1[4], Y1[4], Z1[4];
    uint64_t X3[4], Y3[4], Z3[4];
    uint64_t x[4], y[4];

    // Load input point
    for (int i = 0; i < 4; i++) {
        X1[i] = input_X[i];
        Y1[i] = input_Y[i];
        Z1[i] = input_Z[i];
    }

    // Perform point doubling (Jacobian)
    _PointDouble(X1, Y1, Z1, X3, Y3, Z3);

    // Convert to Affine coordinates
    _JacobianToAffine(X3, Y3, Z3, x, y);

    // Store result
    for (int i = 0; i < 4; i++) {
        output_x[i] = x[i];
        output_y[i] = y[i];
    }
}
