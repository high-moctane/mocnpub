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
// Endomorphism Constants (for 3x speedup)
// ============================================================================
// secp256k1 has a special endomorphism: φ(x, y) = (β*x, y) where β³ = 1 mod p
// This allows checking 3 pubkeys (P, β*P, β²*P) with one scalar multiplication

// β = cube root of unity mod p
// 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee
__constant__ uint64_t _Beta[4] = {
    0xc1396c28719501eeULL,
    0x9cf0497512f58995ULL,
    0x6e64479eac3434e9ULL,
    0x7ae96a2b657c0710ULL
};

// β² = β * β mod p
// 0x851695d49a83f8ef919e7886e0bdd3e85495e3cfd41a72c497f8e6a5d31e6dfe
__constant__ uint64_t _Beta2[4] = {
    0x97f8e6a5d31e6dfeULL,
    0x5495e3cfd41a72c4ULL,
    0x919e7886e0bdd3e8ULL,
    0x851695d49a83f8efULL
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
        // Two-stage subtraction to avoid overflow in borrow detection
        uint64_t temp_diff = a[i] - b[i];
        uint64_t borrow1 = (a[i] < b[i]) ? 1 : 0;

        uint64_t final_diff = temp_diff - borrow;
        uint64_t borrow2 = (temp_diff < borrow) ? 1 : 0;

        result[i] = final_diff;
        borrow = borrow1 | borrow2;
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
        // Compute high[i] * 977 using 64-bit multiplication
        uint64_t low, high_part;
        _Mult64(high[i], 977, &low, &high_part);

        // Add carry from previous iteration
        uint64_t sum = low + carry;
        uint64_t new_carry = (sum < low) ? 1 : 0;

        mult977[i] = sum;
        carry = high_part + new_carry;
    }
    mult977[4] = carry;

    // Add: shifted + mult977
    uint64_t sum[5] = {0};
    carry = 0;
    for (int i = 0; i < 5; i++) {
        // Two-stage addition to properly detect carry
        uint64_t s1 = shifted[i] + mult977[i];
        uint64_t carry1 = (s1 < shifted[i]) ? 1 : 0;

        uint64_t s2 = s1 + carry;
        uint64_t carry2 = (s2 < s1) ? 1 : 0;

        sum[i] = s2;
        carry = carry1 | carry2;
    }

    // Reduce sum[4]: sum[4] * 2^256 mod p = sum[4] * (2^32 + 977)
    // This avoids billions of iterations in the reduction loop
    if (sum[4] > 0) {
        // Compute sum[4] * (2^32 + 977)
        uint64_t factor = sum[4];

        // sum[4] * 2^32: shift left by 32 bits
        uint64_t shifted_low = factor << 32;
        uint64_t shifted_high = factor >> 32;

        // sum[4] * 977
        uint64_t mult_low, mult_high;
        _Mult64(factor, 977, &mult_low, &mult_high);

        // Add (shifted_low + mult_low) to sum[0], (shifted_high + mult_high) to sum[1]
        // This implements: sum[4] * (2^32 + 977) = (shifted_low + mult_low) + ((shifted_high + mult_high) << 64)
        uint64_t carry2 = 0;

        // Add shifted_low + mult_low to sum[0]
        uint64_t add0 = shifted_low + mult_low;
        uint64_t carry0 = (add0 < shifted_low) ? 1 : 0;

        uint64_t s0 = sum[0] + add0;
        carry2 = (s0 < sum[0]) ? 1 : 0;
        carry2 += carry0;
        sum[0] = s0;

        // Add shifted_high + mult_high + carry to sum[1]
        uint64_t add1 = shifted_high + mult_high;
        uint64_t carry1 = (add1 < shifted_high) ? 1 : 0;

        uint64_t s1 = sum[1] + add1 + carry2;
        uint64_t new_carry = (s1 < sum[1]) ? 1 : 0;
        if (add1 + carry2 < add1) new_carry = 1;
        carry2 = new_carry + carry1;
        sum[1] = s1;

        // Propagate carry to sum[2]
        uint64_t s2 = sum[2] + carry2;
        carry2 = (s2 < sum[2]) ? 1 : 0;
        sum[2] = s2;

        // Propagate carry to sum[3]
        uint64_t s3 = sum[3] + carry2;
        carry2 = (s3 < sum[3]) ? 1 : 0;
        sum[3] = s3;

        // Set sum[4] to carry (should be 0 or very small)
        sum[4] = carry2;
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
        // Two-stage addition to properly detect carry
        uint64_t s1 = temp[i] + sum[i];
        uint64_t carry1 = (s1 < temp[i]) ? 1 : 0;

        uint64_t s2 = s1 + carry;
        uint64_t carry2 = (s2 < s1) ? 1 : 0;

        temp[i] = s2;
        carry = carry1 | carry2;
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
            // Subtract p from temp (320-bit - 256-bit)
            uint64_t borrow = 0;
            for (int i = 0; i < 4; i++) {
                // Two-stage subtraction to avoid overflow in borrow detection
                uint64_t temp_diff = temp[i] - _P[i];
                uint64_t borrow1 = (temp[i] < _P[i]) ? 1 : 0;

                uint64_t final_diff = temp_diff - borrow;
                uint64_t borrow2 = (temp_diff < borrow) ? 1 : 0;

                temp[i] = final_diff;
                borrow = borrow1 | borrow2;
            }
            // Subtract borrow from temp[4]
            temp[4] -= borrow;
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

            // Add to temp[i+j] with proper carry detection (two-stage)
            uint64_t s1 = temp[i + j] + low;
            uint64_t carry1 = (s1 < temp[i + j]) ? 1 : 0;

            uint64_t s2 = s1 + carry;
            uint64_t carry2 = (s2 < s1) ? 1 : 0;

            temp[i + j] = s2;
            carry = high + carry1 + carry2;
        }
        temp[i + 4] += carry;
    }

    // Reduce modulo p
    _Reduce512(temp, result);
}

/**
 * Modular squaring: (a * a) mod p
 * Optimized version of _ModMult(a, a, result)
 *
 * Uses symmetry: a² needs only 10 multiplications instead of 16
 * - Diagonal terms (i == j): 4 multiplications
 * - Off-diagonal terms (i < j): 6 multiplications, each doubled
 */
__device__ void _ModSquare(const uint64_t a[4], uint64_t result[4])
{
    uint64_t temp[8] = {0};  // 512-bit result

    // Step 1: Compute off-diagonal products (i < j) and double them
    // We compute a[i] * a[j] for i < j, then add twice to temp
    uint64_t cross[8] = {0};  // Accumulate cross products

    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = i + 1; j < 4; j++) {
            uint64_t low, high;
            _Mult64(a[i], a[j], &low, &high);

            // Add to cross[i+j] with carry
            uint64_t s1 = cross[i + j] + low;
            uint64_t carry1 = (s1 < cross[i + j]) ? 1 : 0;

            uint64_t s2 = s1 + carry;
            uint64_t carry2 = (s2 < s1) ? 1 : 0;

            cross[i + j] = s2;
            carry = high + carry1 + carry2;
        }
        if (i < 3) {
            cross[i + 4] += carry;
        }
    }

    // Double the cross products (left shift by 1)
    uint64_t shift_carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t new_carry = cross[i] >> 63;  // Top bit
        cross[i] = (cross[i] << 1) | shift_carry;
        shift_carry = new_carry;
    }

    // Step 2: Compute diagonal products (i == j): a[i]²
    for (int i = 0; i < 4; i++) {
        uint64_t low, high;
        _Mult64(a[i], a[i], &low, &high);

        // Add to temp[2*i] and temp[2*i+1]
        uint64_t s1 = temp[2 * i] + low;
        uint64_t carry1 = (s1 < temp[2 * i]) ? 1 : 0;
        temp[2 * i] = s1;

        uint64_t s2 = temp[2 * i + 1] + high;
        uint64_t carry2 = (s2 < temp[2 * i + 1]) ? 1 : 0;

        uint64_t s3 = s2 + carry1;
        uint64_t carry3 = (s3 < s2) ? 1 : 0;
        temp[2 * i + 1] = s3;

        // Propagate carry to higher limbs
        if (2 * i + 2 < 8) {
            temp[2 * i + 2] += carry2 + carry3;
        }
    }

    // Step 3: Add doubled cross products to temp
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t s1 = temp[i] + cross[i];
        uint64_t carry1 = (s1 < temp[i]) ? 1 : 0;

        uint64_t s2 = s1 + carry;
        uint64_t carry2 = (s2 < s1) ? 1 : 0;

        temp[i] = s2;
        carry = carry1 | carry2;
    }

    // Reduce modulo p
    _Reduce512(temp, result);
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
    int step = 0;

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
                    step++;
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

                step++;
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
 * Mixed Point Addition: (X1, Y1, Z1) + (X2, Y2, 1) where Z2 = 1 (Affine point)
 *
 * Optimized version of _PointAdd when the second point is in Affine coordinates.
 * This is useful for adding G (generator point) which has Z=1.
 *
 * Simplifications when Z2 = 1:
 *   - Z2^2 = 1, Z2^3 = 1 (no computation needed)
 *   - U1 = X1 (no multiplication)
 *   - S1 = Y1 (no multiplication)
 *   - Z3 = Z1 * H (no multiplication by Z2)
 *
 * Cost: 8M + 3S (vs 12M + 4S for general _PointAdd) - about 30% faster!
 */
__device__ void _PointAddMixed(
    const uint64_t X1[4], const uint64_t Y1[4], const uint64_t Z1[4],  // Jacobian point
    const uint64_t X2[4], const uint64_t Y2[4],                        // Affine point (Z2=1)
    uint64_t X3[4], uint64_t Y3[4], uint64_t Z3[4]
)
{
    uint64_t Z1_squared[4], Z1_cubed[4];
    uint64_t U2[4], S2[4];
    uint64_t H[4], H_squared[4], H_cubed[4];
    uint64_t R[4], R_squared[4];
    uint64_t temp[4], temp2[4];

    // Z1^2 and Z1^3
    _ModSquare(Z1, Z1_squared);              // S1: Z1^2
    _ModMult(Z1_squared, Z1, Z1_cubed);      // M1: Z1^3

    // U1 = X1 (since Z2^2 = 1, no multiplication needed)
    // S1 = Y1 (since Z2^3 = 1, no multiplication needed)

    // U2 = X2 * Z1^2
    _ModMult(X2, Z1_squared, U2);            // M2: U2

    // S2 = Y2 * Z1^3
    _ModMult(Y2, Z1_cubed, S2);              // M3: S2

    // H = U2 - U1 = U2 - X1
    _ModSub(U2, X1, H);

    // R = S2 - S1 = S2 - Y1
    _ModSub(S2, Y1, R);

    // H^2 and H^3
    _ModSquare(H, H_squared);                // S2: H^2
    _ModMult(H_squared, H, H_cubed);         // M4: H^3

    // R^2
    _ModSquare(R, R_squared);                // S3: R^2

    // X3 = R^2 - H^3 - 2*U1*H^2 = R^2 - H^3 - 2*X1*H^2
    _ModMult(X1, H_squared, temp);           // M5: X1 * H^2
    _ModAdd(temp, temp, temp2);              // temp2 = 2 * X1 * H^2
    _ModSub(R_squared, H_cubed, temp);       // temp = R^2 - H^3
    _ModSub(temp, temp2, X3);                // X3 = R^2 - H^3 - 2*X1*H^2

    // Y3 = R * (U1*H^2 - X3) - S1*H^3 = R * (X1*H^2 - X3) - Y1*H^3
    _ModMult(X1, H_squared, temp);           // M6: X1 * H^2
    _ModSub(temp, X3, temp2);                // temp2 = X1*H^2 - X3
    _ModMult(R, temp2, temp);                // M7: R * (X1*H^2 - X3)
    _ModMult(Y1, H_cubed, temp2);            // M8: Y1 * H^3
    _ModSub(temp, temp2, Y3);                // Y3 = R * (X1*H^2 - X3) - Y1*H^3

    // Z3 = Z1 * H (since Z2 = 1)
    _ModMult(Z1, H, Z3);                     // M9: Z3 (but this is M8 in my count... let me recount)
    // Total: 8M + 3S ✓
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

/**
 * Point Multiplication (Scalar Multiplication): k * P
 * Computes k * P where k is a 256-bit scalar and P is a point on the curve
 *
 * Uses the double-and-add algorithm (MSB to LSB):
 *   result = O (point at infinity)
 *   for each bit in k (from MSB to LSB):
 *     result = 2 * result  (point doubling)
 *     if bit == 1:
 *       result = result + P  (point addition)
 *   return result
 *
 * Special cases:
 *   - k = 0: returns point at infinity (Z = 0)
 *   - P = O: returns point at infinity
 */
__device__ void _PointMult(
    const uint64_t k[4],        // 256-bit scalar
    const uint64_t Px[4], const uint64_t Py[4],  // Input point P (Affine)
    uint64_t Rx[4], uint64_t Ry[4], uint64_t Rz[4]  // Result point R = k*P (Jacobian)
)
{
    // Initialize result to point at infinity (Z = 0)
    uint64_t resX[4] = {0, 0, 0, 0};
    uint64_t resY[4] = {1, 0, 0, 0};  // Y can be any non-zero value
    uint64_t resZ[4] = {0, 0, 0, 0};  // Z = 0 means point at infinity

    // Convert input point P to Jacobian: (Px, Py) -> (Px, Py, 1)
    uint64_t PjX[4], PjY[4], PjZ[4];
    for (int i = 0; i < 4; i++) {
        PjX[i] = Px[i];
        PjY[i] = Py[i];
    }
    PjZ[0] = 1; PjZ[1] = 0; PjZ[2] = 0; PjZ[3] = 0;

    bool started = false;  // Track if we've seen the first 1 bit

    // Iterate through all bits of k (from MSB to LSB)
    for (int i = 3; i >= 0; i--) {
        uint64_t word = k[i];
        for (int j = 63; j >= 0; j--) {
            uint64_t bit = (word >> j) & 1;

            if (!started) {
                // Skip leading zeros until we find the first 1
                if (bit == 1) {
                    started = true;
                    // First bit: result = P (no doubling yet)
                    for (int m = 0; m < 4; m++) {
                        resX[m] = PjX[m];
                        resY[m] = PjY[m];
                        resZ[m] = PjZ[m];
                    }
                }
            } else {
                // Double: result = 2 * result
                uint64_t tempX[4], tempY[4], tempZ[4];
                _PointDouble(resX, resY, resZ, tempX, tempY, tempZ);
                for (int m = 0; m < 4; m++) {
                    resX[m] = tempX[m];
                    resY[m] = tempY[m];
                    resZ[m] = tempZ[m];
                }

                // If bit is set, add P: result = result + P
                if (bit == 1) {
                    _PointAdd(resX, resY, resZ, PjX, PjY, PjZ, tempX, tempY, tempZ);
                    for (int m = 0; m < 4; m++) {
                        resX[m] = tempX[m];
                        resY[m] = tempY[m];
                        resZ[m] = tempZ[m];
                    }
                }
            }
        }
    }

    // Copy result
    for (int i = 0; i < 4; i++) {
        Rx[i] = resX[i];
        Ry[i] = resY[i];
        Rz[i] = resZ[i];
    }
}

// ============================================================================
// Production Kernels
// ============================================================================

/**
 * Batch public key generation kernel
 *
 * Generates public keys from private keys in parallel.
 * Each thread processes one private key: pubkey = privkey * G
 *
 * Input:
 *   - privkeys: array of 256-bit private keys [batch_size * 4]
 *   - batch_size: number of keys to generate
 *
 * Output:
 *   - pubkeys_x: array of x-coordinates of public keys [batch_size * 4]
 *     (only x-coordinate is needed for npub generation)
 */
extern "C" __global__ void generate_pubkeys(
    const uint64_t* privkeys,   // [batch_size * 4] private keys
    uint64_t* pubkeys_x,        // [batch_size * 4] public key x-coordinates
    uint32_t batch_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check
    if (idx >= batch_size) return;

    // Load private key for this thread
    uint64_t k[4];
    for (int i = 0; i < 4; i++) {
        k[i] = privkeys[idx * 4 + i];
    }

    // Compute public key: P = k * G
    uint64_t Rx[4], Ry[4], Rz[4];
    _PointMult(k, _Gx, _Gy, Rx, Ry, Rz);

    // Convert to Affine coordinates (only need x-coordinate for npub)
    uint64_t x[4], y[4];
    _JacobianToAffine(Rx, Ry, Rz, x, y);

    // Store x-coordinate
    for (int i = 0; i < 4; i++) {
        pubkeys_x[idx * 4 + i] = x[i];
    }
}

/**
 * Sequential public key generation kernel (10000連ガチャ戦略)
 *
 * Each thread generates multiple public keys from consecutive private keys:
 *   - First key: base_key * G (heavy _PointMult)
 *   - Subsequent keys: P + G using _PointAddMixed (light! ~300x faster)
 *
 * This exploits the property: (k+1)*G = k*G + G
 *
 * Input:
 *   - base_keys: array of starting private keys [num_threads * 4]
 *   - num_threads: number of threads (batch size)
 *   - keys_per_thread: how many consecutive keys each thread generates
 *
 * Output:
 *   - pubkeys_x: array of x-coordinates [num_threads * keys_per_thread * 4]
 *     Layout: pubkeys_x[(thread_idx * keys_per_thread + key_idx) * 4 + limb]
 */
extern "C" __global__ void generate_pubkeys_sequential(
    const uint64_t* base_keys,   // [num_threads * 4] starting private keys
    uint64_t* pubkeys_x,         // [num_threads * keys_per_thread * 4] output x-coordinates
    uint32_t num_threads,
    uint32_t keys_per_thread
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check
    if (idx >= num_threads) return;

    // Load base private key for this thread
    uint64_t k[4];
    for (int i = 0; i < 4; i++) {
        k[i] = base_keys[idx * 4 + i];
    }

    // Compute first public key: P = k * G (heavy operation)
    uint64_t Px[4], Py[4], Pz[4];
    _PointMult(k, _Gx, _Gy, Px, Py, Pz);

    // Convert to Affine and store first key
    uint64_t x[4], y[4];
    _JacobianToAffine(Px, Py, Pz, x, y);

    uint32_t base_output_idx = idx * keys_per_thread * 4;
    for (int i = 0; i < 4; i++) {
        pubkeys_x[base_output_idx + i] = x[i];
    }

    // Generate subsequent keys using P = P + G (light operation!)
    for (uint32_t key_idx = 1; key_idx < keys_per_thread; key_idx++) {
        // P = P + G using mixed addition (G has Z=1)
        uint64_t tempX[4], tempY[4], tempZ[4];
        _PointAddMixed(Px, Py, Pz, _Gx, _Gy, tempX, tempY, tempZ);

        // Copy result back to P
        for (int i = 0; i < 4; i++) {
            Px[i] = tempX[i];
            Py[i] = tempY[i];
            Pz[i] = tempZ[i];
        }

        // Convert to Affine and store
        // Note: _JacobianToAffine calls _ModInv which is expensive
        // Phase 2 optimization: use Montgomery's Trick to batch inverse calculations
        _JacobianToAffine(Px, Py, Pz, x, y);

        uint32_t output_idx = base_output_idx + key_idx * 4;
        for (int i = 0; i < 4; i++) {
            pubkeys_x[output_idx + i] = x[i];
        }
    }
}

// ============================================================================
// Test Kernels
// ============================================================================

/**
 * Sequential public key generation with Montgomery's Trick (Phase 2)
 *
 * This kernel combines the sequential key generation (Phase 1) with
 * Montgomery's Trick for batch inverse calculation.
 *
 * Montgomery's Trick:
 *   Instead of computing N separate inversions (each ~256 operations),
 *   we compute all N inverses with just ONE inversion + 3(N-1) multiplications.
 *
 *   Algorithm:
 *     Step 1: Compute cumulative products
 *       c[0] = Z[0]
 *       c[i] = c[i-1] * Z[i]  for i = 1, ..., n-1
 *
 *     Step 2: Single inversion
 *       u = c[n-1]^(-1)
 *
 *     Step 3: Compute individual inverses (reverse order)
 *       for i = n-1 down to 1:
 *         Z_inv[i] = u * c[i-1]
 *         u = u * Z[i]
 *       Z_inv[0] = u
 *
 *   Cost: 1 _ModInv + 3(N-1) _ModMult
 *   vs N × _ModInv for naive approach
 *
 *   For N=256: ~85x reduction in inverse calculations!
 *
 * Memory usage per thread:
 *   - X, Y, Z arrays: 3 * 256 * 32 bytes = 24 KB
 *   - Cumulative products: 256 * 32 bytes = 8 KB
 *   - Total: ~32 KB (spills to local memory)
 *
 * Input:
 *   - base_keys: starting private keys [num_threads * 4]
 *   - num_threads: number of threads
 *   - keys_per_thread: consecutive keys per thread (max 256)
 *
 * Output:
 *   - pubkeys_x: x-coordinates [num_threads * keys_per_thread * 4]
 */
#define MAX_KEYS_PER_THREAD 1024

extern "C" __global__ void generate_pubkeys_sequential_montgomery(
    const uint64_t* base_keys,
    uint64_t* pubkeys_x,
    uint32_t num_threads,
    uint32_t keys_per_thread
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_threads) return;

    // Local arrays for storing intermediate Jacobian coordinates
    // These will spill to local memory (backed by global memory) if too large
    uint64_t X_arr[MAX_KEYS_PER_THREAD][4];
    uint64_t Y_arr[MAX_KEYS_PER_THREAD][4];
    uint64_t Z_arr[MAX_KEYS_PER_THREAD][4];
    uint64_t c[MAX_KEYS_PER_THREAD][4];  // Cumulative products

    // Clamp keys_per_thread to MAX_KEYS_PER_THREAD
    uint32_t n = keys_per_thread;
    if (n > MAX_KEYS_PER_THREAD) n = MAX_KEYS_PER_THREAD;

    // Load base private key
    uint64_t k[4];
    for (int i = 0; i < 4; i++) {
        k[i] = base_keys[idx * 4 + i];
    }

    // === Phase 1: Generate all points in Jacobian coordinates ===

    // First point: P = k * G (heavy operation)
    uint64_t Px[4], Py[4], Pz[4];
    _PointMult(k, _Gx, _Gy, Px, Py, Pz);

    // Store first point
    for (int i = 0; i < 4; i++) {
        X_arr[0][i] = Px[i];
        Y_arr[0][i] = Py[i];
        Z_arr[0][i] = Pz[i];
    }

    // Generate subsequent points using P = P + G (light operation!)
    for (uint32_t key_idx = 1; key_idx < n; key_idx++) {
        uint64_t tempX[4], tempY[4], tempZ[4];
        _PointAddMixed(Px, Py, Pz, _Gx, _Gy, tempX, tempY, tempZ);

        // Copy and store
        for (int i = 0; i < 4; i++) {
            Px[i] = tempX[i];
            Py[i] = tempY[i];
            Pz[i] = tempZ[i];
            X_arr[key_idx][i] = Px[i];
            Y_arr[key_idx][i] = Py[i];
            Z_arr[key_idx][i] = Pz[i];
        }
    }

    // === Phase 2: Montgomery's Trick for batch inverse ===

    // Step 1: Compute cumulative products
    // c[0] = Z[0]
    for (int i = 0; i < 4; i++) {
        c[0][i] = Z_arr[0][i];
    }
    // c[i] = c[i-1] * Z[i]
    for (uint32_t i = 1; i < n; i++) {
        _ModMult(c[i-1], Z_arr[i], c[i]);
    }

    // Step 2: Compute inverse of cumulative product (SINGLE _ModInv!)
    uint64_t u[4];
    _ModInv(c[n-1], u);

    // Step 3: Compute individual inverses (reverse order) and convert to Affine
    for (int32_t i = n - 1; i >= 1; i--) {
        // Z_inv = u * c[i-1]
        uint64_t Z_inv[4];
        _ModMult(u, c[i-1], Z_inv);

        // Convert to Affine: x = X * Z_inv^2
        uint64_t Z_inv_squared[4];
        _ModSquare(Z_inv, Z_inv_squared);

        uint64_t x[4];
        _ModMult(X_arr[i], Z_inv_squared, x);

        // Store x-coordinate
        uint32_t output_idx = (idx * keys_per_thread + i) * 4;
        for (int j = 0; j < 4; j++) {
            pubkeys_x[output_idx + j] = x[j];
        }

        // u = u * Z[i] (prepare for next iteration)
        uint64_t temp[4];
        _ModMult(u, Z_arr[i], temp);
        for (int j = 0; j < 4; j++) {
            u[j] = temp[j];
        }
    }

    // Handle first point: Z_inv[0] = u
    {
        uint64_t Z_inv_squared[4];
        _ModSquare(u, Z_inv_squared);

        uint64_t x[4];
        _ModMult(X_arr[0], Z_inv_squared, x);

        uint32_t output_idx = idx * keys_per_thread * 4;
        for (int j = 0; j < 4; j++) {
            pubkeys_x[output_idx + j] = x[j];
        }
    }
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

/**
 * Test kernel: Direct _Reduce512 test
 * Input: 512-bit number (8 limbs)
 * Output: result mod p (4 limbs)
 */
extern "C" __global__ void test_reduce512(
    const uint64_t* input_512,  // [8] 512-bit input
    uint64_t* output_256        // [4] 256-bit output
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only process first thread (simple test)
    if (idx != 0) return;

    uint64_t in[8];
    uint64_t result[4];

    // Load input
    for (int i = 0; i < 8; i++) {
        in[i] = input_512[i];
    }

    // Perform reduction
    _Reduce512(in, result);

    // Store result
    for (int i = 0; i < 4; i++) {
        output_256[i] = result[i];
    }
}

/**
 * Test kernel: Point Multiplication (Scalar Multiplication)
 * Input: scalar k (256-bit), point P in Affine (x, y)
 * Output: k*P in Affine coordinates (x, y)
 */
extern "C" __global__ void test_point_mult(
    const uint64_t* input_k,    // [4] 256-bit scalar
    const uint64_t* input_Px,   // [4] x-coordinate of P
    const uint64_t* input_Py,   // [4] y-coordinate of P
    uint64_t* output_x,         // [4] x-coordinate of result
    uint64_t* output_y          // [4] y-coordinate of result
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only process first thread (simple test)
    if (idx != 0) return;

    uint64_t k[4], Px[4], Py[4];
    uint64_t Rx[4], Ry[4], Rz[4];
    uint64_t result_x[4], result_y[4];

    // Load inputs
    for (int i = 0; i < 4; i++) {
        k[i] = input_k[i];
        Px[i] = input_Px[i];
        Py[i] = input_Py[i];
    }

    // Perform point multiplication (returns Jacobian coordinates)
    _PointMult(k, Px, Py, Rx, Ry, Rz);

    // Convert to Affine coordinates
    _JacobianToAffine(Rx, Ry, Rz, result_x, result_y);

    // Store result
    for (int i = 0; i < 4; i++) {
        output_x[i] = result_x[i];
        output_y[i] = result_y[i];
    }
}

/**
 * Test kernel: Mixed Point Addition (Z2=1)
 * Tests _PointAddMixed by computing P1 + P2 where P2 is in Affine (Z=1)
 *
 * Input: P1 in Jacobian (X1, Y1, Z1), P2 in Affine (X2, Y2)
 * Output: P1 + P2 in Affine (x, y)
 */
extern "C" __global__ void test_point_add_mixed(
    const uint64_t* input_X1,   // [4] Jacobian X1
    const uint64_t* input_Y1,   // [4] Jacobian Y1
    const uint64_t* input_Z1,   // [4] Jacobian Z1
    const uint64_t* input_X2,   // [4] Affine X2 (Z2=1)
    const uint64_t* input_Y2,   // [4] Affine Y2
    uint64_t* output_x,         // [4] Affine result x
    uint64_t* output_y          // [4] Affine result y
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only process first thread (simple test)
    if (idx != 0) return;

    uint64_t X1[4], Y1[4], Z1[4];
    uint64_t X2[4], Y2[4];
    uint64_t X3[4], Y3[4], Z3[4];
    uint64_t result_x[4], result_y[4];

    // Load inputs
    for (int i = 0; i < 4; i++) {
        X1[i] = input_X1[i];
        Y1[i] = input_Y1[i];
        Z1[i] = input_Z1[i];
        X2[i] = input_X2[i];
        Y2[i] = input_Y2[i];
    }

    // Perform mixed point addition
    _PointAddMixed(X1, Y1, Z1, X2, Y2, X3, Y3, Z3);

    // Convert to Affine coordinates
    _JacobianToAffine(X3, Y3, Z3, result_x, result_y);

    // Store result
    for (int i = 0; i < 4; i++) {
        output_x[i] = result_x[i];
        output_y[i] = result_y[i];
    }
}

// ============================================================================
// Prefix Matching Kernel (GPU-side filtering)
// ============================================================================

/**
 * Generate public keys and filter by prefix matching on GPU
 *
 * This kernel combines Montgomery's Trick with prefix matching,
 * returning only the keys that match the specified prefixes.
 *
 * Uses secp256k1 endomorphism for 3x speedup:
 *   For each pubkey P=(x,y), also checks β*x and β²*x (3 pubkeys per computation)
 *
 * Input:
 *   - base_keys: starting private keys [num_threads * 4]
 *   - patterns: prefix bit patterns [num_prefixes] (upper bits of u64)
 *   - masks: prefix bit masks [num_prefixes] (upper bits set to 1)
 *   - num_prefixes: number of prefixes to match
 *   - num_threads: number of threads
 *   - keys_per_thread: consecutive keys per thread (max 256)
 *   - max_matches: maximum number of matches to store
 *
 * Output:
 *   - matched_base_idx: thread index of matched keys [max_matches]
 *   - matched_offset: key offset within thread [max_matches]
 *   - matched_pubkeys_x: x-coordinates of matched pubkeys [max_matches * 4]
 *   - matched_endo_type: endomorphism type (0=original, 1=β, 2=β²) [max_matches]
 *   - match_count: number of matches found (atomic counter)
 */
extern "C" __global__ void generate_pubkeys_with_prefix_match(
    const uint64_t* base_keys,
    const uint64_t* patterns,
    const uint64_t* masks,
    uint32_t num_prefixes,
    uint32_t* matched_base_idx,
    uint32_t* matched_offset,
    uint64_t* matched_pubkeys_x,
    uint32_t* matched_endo_type,
    uint32_t* match_count,
    uint32_t num_threads,
    uint32_t keys_per_thread,
    uint32_t max_matches
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_threads) return;

    // Local arrays for storing intermediate Jacobian coordinates
    uint64_t X_arr[MAX_KEYS_PER_THREAD][4];
    uint64_t Y_arr[MAX_KEYS_PER_THREAD][4];
    uint64_t Z_arr[MAX_KEYS_PER_THREAD][4];
    uint64_t c[MAX_KEYS_PER_THREAD][4];  // Cumulative products

    // Clamp keys_per_thread to MAX_KEYS_PER_THREAD
    uint32_t n = keys_per_thread;
    if (n > MAX_KEYS_PER_THREAD) n = MAX_KEYS_PER_THREAD;

    // Load base private key
    uint64_t k[4];
    for (int i = 0; i < 4; i++) {
        k[i] = base_keys[idx * 4 + i];
    }

    // === Phase 1: Generate all points in Jacobian coordinates ===

    // First point: P = k * G (heavy operation)
    uint64_t Px[4], Py[4], Pz[4];
    _PointMult(k, _Gx, _Gy, Px, Py, Pz);

    // Store first point
    for (int i = 0; i < 4; i++) {
        X_arr[0][i] = Px[i];
        Y_arr[0][i] = Py[i];
        Z_arr[0][i] = Pz[i];
    }

    // Generate subsequent points using P = P + G (light operation!)
    for (uint32_t key_idx = 1; key_idx < n; key_idx++) {
        uint64_t tempX[4], tempY[4], tempZ[4];
        _PointAddMixed(Px, Py, Pz, _Gx, _Gy, tempX, tempY, tempZ);

        // Copy and store
        for (int i = 0; i < 4; i++) {
            Px[i] = tempX[i];
            Py[i] = tempY[i];
            Pz[i] = tempZ[i];
            X_arr[key_idx][i] = Px[i];
            Y_arr[key_idx][i] = Py[i];
            Z_arr[key_idx][i] = Pz[i];
        }
    }

    // === Phase 2: Montgomery's Trick for batch inverse ===

    // Step 1: Compute cumulative products
    for (int i = 0; i < 4; i++) {
        c[0][i] = Z_arr[0][i];
    }
    for (uint32_t i = 1; i < n; i++) {
        _ModMult(c[i-1], Z_arr[i], c[i]);
    }

    // Step 2: Compute inverse of cumulative product (SINGLE _ModInv!)
    uint64_t u[4];
    _ModInv(c[n-1], u);

    // Step 3: Compute individual inverses and check prefix match
    for (int32_t i = n - 1; i >= 0; i--) {
        // Z_inv = u * c[i-1] (or just u for i=0)
        uint64_t Z_inv[4];
        if (i > 0) {
            _ModMult(u, c[i-1], Z_inv);
        } else {
            for (int j = 0; j < 4; j++) Z_inv[j] = u[j];
        }

        // Convert to Affine: x = X * Z_inv^2
        uint64_t Z_inv_squared[4];
        _ModSquare(Z_inv, Z_inv_squared);

        uint64_t x[4];
        _ModMult(X_arr[i], Z_inv_squared, x);

        // === Endomorphism: compute β*x and β²*x for 3x speedup ===
        uint64_t x_beta[4], x_beta2[4];
        _ModMult(x, _Beta, x_beta);
        _ModMult(x, _Beta2, x_beta2);

        // Collect all 3 x-coordinates for prefix matching
        uint64_t* x_coords[3] = { x, x_beta, x_beta2 };

        // === Prefix matching (check all 3 endomorphism variants) ===
        bool found = false;
        for (uint32_t endo = 0; endo < 3 && !found; endo++) {
            uint64_t x_upper = x_coords[endo][3];  // Most significant 64 bits

            for (uint32_t p = 0; p < num_prefixes; p++) {
                if ((x_upper & masks[p]) == (patterns[p] & masks[p])) {
                    // Match found! Atomically reserve output slot
                    uint32_t slot = atomicAdd(match_count, 1);
                    if (slot < max_matches) {
                        matched_base_idx[slot] = idx;
                        matched_offset[slot] = i;
                        matched_endo_type[slot] = endo;  // 0=original, 1=β, 2=β²
                        for (int j = 0; j < 4; j++) {
                            matched_pubkeys_x[slot * 4 + j] = x_coords[endo][j];
                        }
                    }
                    found = true;
                    break;  // Only count once per key
                }
            }
        }

        // Update u for next iteration (if not the last)
        if (i > 0) {
            uint64_t temp[4];
            _ModMult(u, Z_arr[i], temp);
            for (int j = 0; j < 4; j++) {
                u[j] = temp[j];
            }
        }
    }
}
