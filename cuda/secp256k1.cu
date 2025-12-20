/*
 * secp256k1 GPU implementation for mocnpub
 *
 * License: MIT
 */

#include <stdint.h>

// ============================================================================
// secp256k1 Constants (compile-time #define for PTX immediate embedding)
// ============================================================================

// Prime p = 2^256 - 2^32 - 977
// 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
// Note: P1=P2=P3 are all 0xFFFFFFFFFFFFFFFF (special form used for optimization)
#define P0 0xFFFFFFFEFFFFFC2FULL
#define P1 0xFFFFFFFFFFFFFFFFULL
#define P2 0xFFFFFFFFFFFFFFFFULL
#define P3 0xFFFFFFFFFFFFFFFFULL
#define P123 0xFFFFFFFFFFFFFFFFULL  // P1 = P2 = P3 (max uint64, special form)

// Generator point G (x coordinate)
// 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
#define GX0 0x59F2815B16F81798ULL
#define GX1 0x029BFCDB2DCE28D9ULL
#define GX2 0x55A06295CE870B07ULL
#define GX3 0x79BE667EF9DCBBACULL

// Generator point G (y coordinate)
// 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
#define GY0 0x9C47D08FFB10D4B8ULL
#define GY1 0xFD17B448A6855419ULL
#define GY2 0x5DA4FBFC0E1108A8ULL
#define GY3 0x483ADA7726A3C465ULL

// ============================================================================
// Endomorphism Constants (for 3x speedup)
// ============================================================================
// secp256k1 has a special endomorphism: φ(x, y) = (β*x, y) where β³ = 1 mod p
// This allows checking 3 pubkeys (P, β*P, β²*P) with one scalar multiplication

// β = cube root of unity mod p
// 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee
#define BETA0 0xc1396c28719501eeULL
#define BETA1 0x9cf0497512f58995ULL
#define BETA2 0x6e64479eac3434e9ULL
#define BETA3 0x7ae96a2b657c0710ULL

// β² = β * β mod p
// 0x851695d49a83f8ef919bb86153cbcb16630fb68aed0a766a3ec693d68e6afa40
#define BETA2_0 0x3ec693d68e6afa40ULL
#define BETA2_1 0x630fb68aed0a766aULL
#define BETA2_2 0x919bb86153cbcb16ULL
#define BETA2_3 0x851695d49a83f8efULL

// ============================================================================
// dG Table for Sequential Key Strategy (precomputed at runtime)
// ============================================================================
// dG = MAX_KEYS_PER_THREAD * G
// Table: [dG, 2*dG, 4*dG, ..., 2^23*dG] (24 entries)
// Each entry is (X[4], Y[4]) = 8 uint64_t values
// Total: 24 * 8 = 192 uint64_t = 1536 bytes
// This is initialized from host via cuMemcpyHtoD to the symbol address
__constant__ uint64_t _dG_table[192];

// ============================================================================
// Prefix Matching Constants (set at runtime via cuMemcpyHtoD)
// ============================================================================
// patterns[i] = upper 32 bits of target prefix (after bech32 encoding)
// masks[i] = bitmask for comparison (1s for significant bits)
// Using constant memory for broadcast optimization (all threads read same values)
__constant__ uint32_t _patterns[256];
__constant__ uint32_t _masks[256];
__constant__ uint32_t _num_prefixes;
__constant__ uint32_t _num_threads;
__constant__ uint32_t _max_matches;

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
 * Returns the result and borrow (1 if a < b, 0 otherwise)
 */
__device__ void _Sub256(const uint64_t a[4], const uint64_t b[4], uint64_t result[4], uint64_t* borrow_out)
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

    *borrow_out = borrow;
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
 * Branchless implementation to avoid warp divergence
 */
__device__ void _ModAdd(const uint64_t a[4], const uint64_t b[4], uint64_t result[4])
{
    // P as compile-time constants
    const uint64_t p[4] = {P0, P1, P2, P3};

    // Always compute sum = a + b
    uint64_t sum[4];
    uint64_t carry;
    _Add256(a, b, sum, &carry);

    // Always compute diff = sum - p (may underflow)
    uint64_t diff[4];
    uint64_t borrow;
    _Sub256(sum, p, diff, &borrow);

    // Use diff if: carry || !borrow (i.e., sum >= p or overflow)
    // use_diff = carry | (1 - borrow)
    uint64_t use_diff = carry | (1 - borrow);
    uint64_t mask = -use_diff;  // use_diff ? 0xFFFF... : 0

    // result = (diff & mask) | (sum & ~mask)
    for (int i = 0; i < 4; i++) {
        result[i] = (diff[i] & mask) | (sum[i] & ~mask);
    }
}

/**
 * Modular subtraction: (a - b) mod p
 * Branchless implementation to avoid warp divergence
 */
__device__ void _ModSub(const uint64_t a[4], const uint64_t b[4], uint64_t result[4])
{
    // Always compute a - b (may underflow)
    uint64_t diff[4];
    uint64_t borrow;
    _Sub256(a, b, diff, &borrow);

    // If borrow, add p back (branchless)
    // mask = borrow ? 0xFFFFFFFFFFFFFFFF : 0
    uint64_t mask = -borrow;

    // diff + (p & mask) - unrolled with #define constants
    uint64_t p_masked[4];
    p_masked[0] = P0 & mask;
    p_masked[1] = P1 & mask;
    p_masked[2] = P2 & mask;
    p_masked[3] = P3 & mask;

    uint64_t carry;
    _Add256(diff, p_masked, result, &carry);
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
 * Reduce the 5th limb of a 320-bit number: sum[4] * 2^256 mod p = sum[4] * (2^32 + 977)
 * Adds the result back to sum[0..4], avoiding billions of iterations in a naive reduction loop.
 */
__device__ void _ReduceOverflow(uint64_t sum[5])
{
    if (sum[4] == 0) return;

    uint64_t factor = sum[4];

    // factor * 2^32: shift left by 32 bits
    uint64_t shifted_low = factor << 32;
    uint64_t shifted_high = factor >> 32;

    // factor * 977
    uint64_t mult_low, mult_high;
    _Mult64(factor, 977, &mult_low, &mult_high);

    // Add (shifted_low + mult_low) to sum[0], (shifted_high + mult_high) to sum[1]
    // This implements: factor * (2^32 + 977) = (shifted_low + mult_low) + ((shifted_high + mult_high) << 64)
    uint64_t carry = 0;

    // Add shifted_low + mult_low to sum[0]
    uint64_t add0 = shifted_low + mult_low;
    uint64_t carry0 = (add0 < shifted_low) ? 1 : 0;

    uint64_t s0 = sum[0] + add0;
    carry = (s0 < sum[0]) ? 1 : 0;
    carry += carry0;
    sum[0] = s0;

    // Add shifted_high + mult_high + carry to sum[1]
    // Using 2-stage addition to properly detect carry
    uint64_t add1 = shifted_high + mult_high;
    uint64_t carry1 = (add1 < shifted_high) ? 1 : 0;

    // Stage 1: sum[1] + add1
    uint64_t s1a = sum[1] + add1;
    uint64_t carry_a = (s1a < sum[1]) ? 1 : 0;

    // Stage 2: s1a + carry
    uint64_t s1 = s1a + carry;
    uint64_t carry_b = (s1 < s1a) ? 1 : 0;

    // Total carry: carry1 (from add1) + carry_a (from stage1) + carry_b (from stage2)
    // Note: This can be 0, 1, 2, or 3, which is fine for propagation
    carry = carry1 + carry_a + carry_b;
    sum[1] = s1;

    // Propagate carry to sum[2]
    uint64_t s2 = sum[2] + carry;
    carry = (s2 < sum[2]) ? 1 : 0;
    sum[2] = s2;

    // Propagate carry to sum[3]
    uint64_t s3 = sum[3] + carry;
    carry = (s3 < sum[3]) ? 1 : 0;
    sum[3] = s3;

    // Set sum[4] to carry (should be 0 or very small)
    sum[4] = carry;
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
    _ReduceOverflow(sum);

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
    // Using #define constants for P (P1=P2=P3=P123=max_uint64)
    for (int iter = 0; iter < 3; iter++) {
        // Check if temp >= p
        // Since P1=P2=P3=P123=max_uint64, comparison is simplified:
        // - If any of temp[3,2,1] < max_uint64, then temp < p
        // - Otherwise check temp[0] >= P0
        bool ge = false;
        if (temp[4] > 0) {
            ge = true;
        } else if (temp[3] < P123) {
            ge = false;
        } else if (temp[2] < P123) {
            ge = false;
        } else if (temp[1] < P123) {
            ge = false;
        } else {
            // temp[3]=temp[2]=temp[1]=P123 (max_uint64), check temp[0] vs P0
            ge = temp[0] >= P0;
        }

        if (ge) {
            // Subtract p from temp (320-bit - 256-bit) - unrolled
            uint64_t borrow = 0;

            // i = 0
            uint64_t temp_diff0 = temp[0] - P0;
            uint64_t borrow1_0 = (temp[0] < P0) ? 1 : 0;
            uint64_t final_diff0 = temp_diff0 - borrow;
            uint64_t borrow2_0 = (temp_diff0 < borrow) ? 1 : 0;
            temp[0] = final_diff0;
            borrow = borrow1_0 | borrow2_0;

            // i = 1, 2, 3: P1 = P2 = P3 = P123 (max_uint64)
            // temp[i] - P123 always underflows unless temp[i] == P123
            // So borrow1 = (temp[i] < P123) ? 1 : 0 = (temp[i] != P123) ? 1 : 0
            uint64_t temp_diff1 = temp[1] - P123;
            uint64_t borrow1_1 = (temp[1] < P123) ? 1 : 0;
            uint64_t final_diff1 = temp_diff1 - borrow;
            uint64_t borrow2_1 = (temp_diff1 < borrow) ? 1 : 0;
            temp[1] = final_diff1;
            borrow = borrow1_1 | borrow2_1;

            // i = 2
            uint64_t temp_diff2 = temp[2] - P123;
            uint64_t borrow1_2 = (temp[2] < P123) ? 1 : 0;
            uint64_t final_diff2 = temp_diff2 - borrow;
            uint64_t borrow2_2 = (temp_diff2 < borrow) ? 1 : 0;
            temp[2] = final_diff2;
            borrow = borrow1_2 | borrow2_2;

            // i = 3
            uint64_t temp_diff3 = temp[3] - P123;
            uint64_t borrow1_3 = (temp[3] < P123) ? 1 : 0;
            uint64_t final_diff3 = temp_diff3 - borrow;
            uint64_t borrow2_3 = (temp_diff3 < borrow) ? 1 : 0;
            temp[3] = final_diff3;
            borrow = borrow1_3 | borrow2_3;

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
 * Modular multiplication by Beta: (a * β) mod p
 * Uses compile-time #define constants for β (BETA0-BETA3)
 */
__device__ void _ModMultByBeta(const uint64_t a[4], uint64_t result[4])
{
    const uint64_t beta[4] = {BETA0, BETA1, BETA2, BETA3};
    uint64_t temp[8] = {0};

    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t low, high;
            _Mult64(a[i], beta[j], &low, &high);

            uint64_t s1 = temp[i + j] + low;
            uint64_t carry1 = (s1 < temp[i + j]) ? 1 : 0;

            uint64_t s2 = s1 + carry;
            uint64_t carry2 = (s2 < s1) ? 1 : 0;

            temp[i + j] = s2;
            carry = high + carry1 + carry2;
        }
        temp[i + 4] += carry;
    }

    _Reduce512(temp, result);
}

/**
 * Modular multiplication by Beta²: (a * β²) mod p
 * Uses compile-time #define constants for β² (BETA2_0-BETA2_3)
 */
__device__ void _ModMultByBeta2(const uint64_t a[4], uint64_t result[4])
{
    const uint64_t beta2[4] = {BETA2_0, BETA2_1, BETA2_2, BETA2_3};
    uint64_t temp[8] = {0};

    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t low, high;
            _Mult64(a[i], beta2[j], &low, &high);

            uint64_t s1 = temp[i + j] + low;
            uint64_t carry1 = (s1 < temp[i + j]) ? 1 : 0;

            uint64_t s2 = s1 + carry;
            uint64_t carry2 = (s2 < s1) ? 1 : 0;

            temp[i + j] = s2;
            carry = high + carry1 + carry2;
        }
        temp[i + 4] += carry;
    }

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
 *
 * Optimized using Addition Chain (based on RustCrypto k256 / Peter Dettman's work):
 * - Standard binary exponentiation: 256 squares + ~128 multiplications
 * - Addition Chain: 255 squares + 14 multiplications (114 fewer multiplications!)
 *
 * The chain exploits the structure of p-2:
 *   p-2 = 0xFFFFFFFEFFFFFC2D has block lengths {1, 2, 22, 223}
 *   Chain: [1], [2], 3, 6, 9, 11, [22], 44, 88, 176, 220, [223]
 */
__device__ void _ModInv(const uint64_t a[4], uint64_t result[4])
{
    uint64_t x2[4], x3[4], x6[4], x9[4], x11[4];
    uint64_t x22[4], x44[4], x88[4], x176[4], x220[4], x223[4];
    uint64_t t[4];

    // x2 = a^3 (binary: 11)
    _ModSquare(a, t);           // a^2
    _ModMult(t, a, x2);         // a^3

    // x3 = a^7 (binary: 111)
    _ModSquare(x2, t);          // a^6
    _ModMult(t, a, x3);         // a^7

    // x6 = a^63 (binary: 111111) = a^(2^6 - 1)
    _ModSquare(x3, t);          // 1
    _ModSquare(t, t);           // 2
    _ModSquare(t, t);           // 3
    _ModMult(t, x3, x6);        // a^63

    // x9 = a^511 (binary: 111111111) = a^(2^9 - 1)
    _ModSquare(x6, t);          // 1
    _ModSquare(t, t);           // 2
    _ModSquare(t, t);           // 3
    _ModMult(t, x3, x9);        // a^511

    // x11 = a^2047 (binary: 11111111111) = a^(2^11 - 1)
    _ModSquare(x9, t);          // 1
    _ModSquare(t, t);           // 2
    _ModMult(t, x2, x11);       // a^2047

    // x22 = a^(2^22 - 1)
    for (int i = 0; i < 11; i++) {
        _ModSquare(i == 0 ? x11 : t, t);
    }
    _ModMult(t, x11, x22);

    // x44 = a^(2^44 - 1)
    for (int i = 0; i < 22; i++) {
        _ModSquare(i == 0 ? x22 : t, t);
    }
    _ModMult(t, x22, x44);

    // x88 = a^(2^88 - 1)
    for (int i = 0; i < 44; i++) {
        _ModSquare(i == 0 ? x44 : t, t);
    }
    _ModMult(t, x44, x88);

    // x176 = a^(2^176 - 1)
    for (int i = 0; i < 88; i++) {
        _ModSquare(i == 0 ? x88 : t, t);
    }
    _ModMult(t, x88, x176);

    // x220 = a^(2^220 - 1)
    for (int i = 0; i < 44; i++) {
        _ModSquare(i == 0 ? x176 : t, t);
    }
    _ModMult(t, x44, x220);

    // x223 = a^(2^223 - 1)
    _ModSquare(x220, t);        // 1
    _ModSquare(t, t);           // 2
    _ModSquare(t, t);           // 3
    _ModMult(t, x3, x223);

    // Final assembly: compute a^(p-2) from x223, x22, x2, and a
    // p-2 = 2^256 - 2^32 - 979
    //     = (2^223 - 1) * 2^23 + (2^22 - 1) * 2^(23-22) + ...

    // t = x223 * 2^23
    for (int i = 0; i < 23; i++) {
        _ModSquare(i == 0 ? x223 : t, t);
    }
    _ModMult(t, x22, t);        // + x22

    // t = t * 2^5
    for (int i = 0; i < 5; i++) {
        _ModSquare(t, t);
    }
    _ModMult(t, a, t);          // + a

    // t = t * 2^3
    _ModSquare(t, t);           // 1
    _ModSquare(t, t);           // 2
    _ModSquare(t, t);           // 3
    _ModMult(t, x2, t);         // + x2 (= a^3)

    // t = t * 2^2
    _ModSquare(t, t);           // 1
    _ModSquare(t, t);           // 2
    _ModMult(t, a, result);     // + a -> final result
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
 * Cost: 3M + 4S (3 multiplications, 4 squarings)
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
 * Mixed Point Addition: (X1, Y1, Z1) + (X2, Y2, 1) where Z2 = 1 (Affine point)
 *
 * Optimized for adding an Affine point (Z=1) to a Jacobian point.
 * This is useful for adding G (generator point) which has Z=1.
 *
 * Simplifications when Z2 = 1:
 *   - Z2^2 = 1, Z2^3 = 1 (no computation needed)
 *   - U1 = X1 (no multiplication)
 *   - S1 = Y1 (no multiplication)
 *   - Z3 = Z1 * H (no multiplication by Z2)
 *
 * Cost: 8M + 3S (vs 12M + 4S for general point addition) - about 33% faster!
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
    uint64_t X1_H2[4];  // X1 * H^2 (reused to avoid duplicate computation)
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
    _ModMult(X1, H_squared, X1_H2);          // M5: X1 * H^2 (save for reuse)
    _ModAdd(X1_H2, X1_H2, temp);             // temp = 2 * X1 * H^2
    _ModSub(R_squared, H_cubed, temp2);      // temp2 = R^2 - H^3
    _ModSub(temp2, temp, X3);                // X3 = R^2 - H^3 - 2*X1*H^2

    // Y3 = R * (U1*H^2 - X3) - S1*H^3 = R * (X1*H^2 - X3) - Y1*H^3
    // Reuse X1_H2 instead of recomputing X1 * H^2
    _ModSub(X1_H2, X3, temp);                // temp = X1*H^2 - X3
    _ModMult(R, temp, temp2);                // M6: R * (X1*H^2 - X3)
    _ModMult(Y1, H_cubed, temp);             // M7: Y1 * H^3
    _ModSub(temp2, temp, Y3);                // Y3 = R * (X1*H^2 - X3) - Y1*H^3

    // Z3 = Z1 * H (since Z2 = 1)
    _ModMult(Z1, H, Z3);                     // M8: Z3
    // Total: 8M + 3S (optimized by reusing X1*H^2)
}

/**
 * Mixed Point Addition with Generator G: (X1, Y1, Z1) + G
 *
 * Specialized version of _PointAddMixed that uses the generator point G
 * with compile-time #define constants (GX0-GX3, GY0-GY3).
 *
 * Cost: 8M + 3S (same as _PointAddMixed)
 */
__device__ void _PointAddMixedG(
    const uint64_t X1[4], const uint64_t Y1[4], const uint64_t Z1[4],
    uint64_t X3[4], uint64_t Y3[4], uint64_t Z3[4]
)
{
    // Generator point G as compile-time constants
    const uint64_t Gx[4] = {GX0, GX1, GX2, GX3};
    const uint64_t Gy[4] = {GY0, GY1, GY2, GY3};

    uint64_t Z1_squared[4], Z1_cubed[4];
    uint64_t U2[4], S2[4];
    uint64_t H[4], H_squared[4], H_cubed[4];
    uint64_t R[4], R_squared[4];
    uint64_t X1_H2[4];
    uint64_t temp[4], temp2[4];

    // Z1^2 and Z1^3
    _ModSquare(Z1, Z1_squared);
    _ModMult(Z1_squared, Z1, Z1_cubed);

    // U2 = Gx * Z1^2
    _ModMult(Gx, Z1_squared, U2);

    // S2 = Gy * Z1^3
    _ModMult(Gy, Z1_cubed, S2);

    // H = U2 - X1
    _ModSub(U2, X1, H);

    // R = S2 - Y1
    _ModSub(S2, Y1, R);

    // H^2 and H^3
    _ModSquare(H, H_squared);
    _ModMult(H_squared, H, H_cubed);

    // R^2
    _ModSquare(R, R_squared);

    // X3 = R^2 - H^3 - 2*X1*H^2
    _ModMult(X1, H_squared, X1_H2);
    _ModAdd(X1_H2, X1_H2, temp);
    _ModSub(R_squared, H_cubed, temp2);
    _ModSub(temp2, temp, X3);

    // Y3 = R * (X1*H^2 - X3) - Y1*H^3
    _ModSub(X1_H2, X3, temp);
    _ModMult(R, temp, temp2);
    _ModMult(Y1, H_cubed, temp);
    _ModSub(temp2, temp, Y3);

    // Z3 = Z1 * H
    _ModMult(Z1, H, Z3);
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
 * Compute base_pubkey + idx * dG using precomputed dG table
 *
 * Instead of _PointMult(k, G) with 256 double-and-add operations,
 * we use a precomputed table of dG, 2*dG, 4*dG, ..., 2^23*dG
 * and perform at most 24 point additions based on the bits of idx.
 *
 * This reduces ~384 operations to ~12 operations (on average).
 *
 * @param idx             Thread index (0 to batch_size/MAX_KEYS_PER_THREAD - 1)
 * @param dG_table        Precomputed table: [dG, 2*dG, 4*dG, ..., 2^23*dG]
 *                        Layout: 24 entries, each entry is (X[4], Y[4]) = 8 uint64_t
 *                        Total: 24 * 8 = 192 uint64_t values
 * @param base_pubkey_x   X coordinate of base_pubkey (Affine)
 * @param base_pubkey_y   Y coordinate of base_pubkey (Affine)
 * @param Rx, Ry, Rz      Output: base_pubkey + idx * dG (Jacobian)
 */
__device__ void _PointMultByIndex(
    uint32_t idx,
    const uint64_t base_pubkey_x[4],
    const uint64_t base_pubkey_y[4],
    uint64_t Rx[4], uint64_t Ry[4], uint64_t Rz[4]
)
{
    // Initialize result to base_pubkey (Jacobian with Z = 1)
    for (int i = 0; i < 4; i++) {
        Rx[i] = base_pubkey_x[i];
        Ry[i] = base_pubkey_y[i];
    }
    Rz[0] = 1; Rz[1] = 0; Rz[2] = 0; Rz[3] = 0;

    // If idx == 0, no additions needed, just return base_pubkey
    if (idx == 0) {
        return;
    }

    // Add dG_table[bit] for each set bit in idx
    for (int bit = 0; bit < 24; bit++) {
        if ((idx >> bit) & 1) {
            // Load _dG_table[bit] from constant memory (Affine coordinates)
            uint64_t dG_x[4], dG_y[4];
            for (int i = 0; i < 4; i++) {
                dG_x[i] = _dG_table[bit * 8 + i];
                dG_y[i] = _dG_table[bit * 8 + 4 + i];
            }

            // Add: R = R + dG_table[bit]
            uint64_t tempX[4], tempY[4], tempZ[4];
            _PointAddMixed(Rx, Ry, Rz, dG_x, dG_y, tempX, tempY, tempZ);
            for (int i = 0; i < 4; i++) {
                Rx[i] = tempX[i];
                Ry[i] = tempY[i];
                Rz[i] = tempZ[i];
            }
        }
    }
}

// ============================================================================
// Production Kernels
// ============================================================================

// MAX_KEYS_PER_THREAD: Can be specified via -D option at build time (default: 1500)
#ifndef MAX_KEYS_PER_THREAD
#define MAX_KEYS_PER_THREAD 1500
#endif

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

// ============================================================================
// Sequential Key Strategy Kernel (VRAM-efficient version)
// ============================================================================

/**
 * Generate public keys using sequential key strategy
 *
 * This kernel uses a single base_key and computes sequential secret keys:
 *   thread i computes keys: base_key + i * MAX_KEYS_PER_THREAD + 0, 1, 2, ...
 *
 * Benefits:
 *   1. VRAM reduction: batch_size * 32 bytes -> 32 bytes (single key)
 *   2. Branch divergence reduction: sequential keys have similar upper bits
 *   3. Enables larger batch_size and MAX_KEYS_PER_THREAD
 *
 * Input:
 *   - base_key: single starting private key [4 limbs]
 *   - base_pubkey_x/y: precomputed base_key * G [4 limbs each]
 *   - (constant memory) _patterns: prefix bit patterns
 *   - (constant memory) _masks: prefix bit masks
 *   - (constant memory) _num_prefixes: number of prefixes to match
 *   - (constant memory) _num_threads: number of threads
 *   - (constant memory) _max_matches: maximum number of matches to store
 *
 * Output:
 *   - matched_base_idx: thread index of matched keys [max_matches]
 *   - matched_offset: key offset within thread [max_matches]
 *   - matched_pubkeys_x: x-coordinates of matched pubkeys [max_matches * 4]
 *   - matched_secret_keys: actual secret keys (base_key + global_offset + i) [max_matches * 4]
 *   - matched_endo_type: endomorphism type (0=original, 1=β, 2=β²) [max_matches]
 *   - match_count: number of matches found (atomic counter)
 */
extern "C" __global__ void __launch_bounds__(128, 5) generate_pubkeys_sequential(
    const uint64_t* base_key,       // Single base key [4 limbs]
    const uint64_t* base_pubkey_x,  // base_key * G, X coordinate [4 limbs]
    const uint64_t* base_pubkey_y,  // base_key * G, Y coordinate [4 limbs]
    // dG_table, patterns, masks, num_prefixes, num_threads, max_matches are now in constant memory
    uint32_t* matched_base_idx,
    uint32_t* matched_offset,
    uint64_t* matched_pubkeys_x,
    uint64_t* matched_secret_keys,  // Actual secret keys (not base_key)
    uint32_t* matched_endo_type,
    uint32_t* match_count
)
{
    // All runtime constants are now in constant memory:
    // _patterns, _masks, _num_prefixes, _num_threads, _max_matches

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= _num_threads) return;

    // Local arrays for storing intermediate Jacobian coordinates
    // Note: Y_arr is not needed! Prefix matching only uses x-coordinate (BIP-340)
    uint64_t X_arr[MAX_KEYS_PER_THREAD][4];
    uint64_t Z_arr[MAX_KEYS_PER_THREAD][4];
    uint64_t c[MAX_KEYS_PER_THREAD][4];

    const uint32_t n = MAX_KEYS_PER_THREAD;

    // === Compute starting secret key: base_key + idx * MAX_KEYS_PER_THREAD ===
    uint64_t global_offset = (uint64_t)idx * MAX_KEYS_PER_THREAD;
    uint64_t k[4];

    // Add global_offset to base_key (with carry propagation)
    uint64_t sum = base_key[0] + global_offset;
    uint64_t carry = (sum < base_key[0]) ? 1 : 0;
    k[0] = sum;

    for (int i = 1; i < 4; i++) {
        sum = base_key[i] + carry;
        carry = (sum < carry) ? 1 : 0;
        k[i] = sum;
    }

    // Save starting key for this thread (needed for match reporting)
    uint64_t thread_base_key[4];
    for (int i = 0; i < 4; i++) {
        thread_base_key[i] = k[i];
    }

    // === Phase 1: Generate all points in Jacobian coordinates ===

    // First point: P = base_pubkey + idx * dG (fast table lookup!)
    // Instead of _PointMult(k, G) with 256 double-and-add operations,
    // we use precomputed dG table and perform at most 24 point additions.
    uint64_t Px[4], Py[4], Pz[4];
    _PointMultByIndex(idx, base_pubkey_x, base_pubkey_y, Px, Py, Pz);

    for (int i = 0; i < 4; i++) {
        X_arr[0][i] = Px[i];
        Z_arr[0][i] = Pz[i];
    }

    // Generate subsequent points using P = P + G
    for (uint32_t key_idx = 1; key_idx < n; key_idx++) {
        uint64_t tempX[4], tempY[4], tempZ[4];
        _PointAddMixedG(Px, Py, Pz, tempX, tempY, tempZ);

        for (int i = 0; i < 4; i++) {
            Px[i] = tempX[i];
            Py[i] = tempY[i];
            Pz[i] = tempZ[i];
            X_arr[key_idx][i] = Px[i];
            Z_arr[key_idx][i] = Pz[i];
        }
    }

    // === Phase 2: Montgomery's Trick for batch inverse ===

    for (int i = 0; i < 4; i++) {
        c[0][i] = Z_arr[0][i];
    }
    for (uint32_t i = 1; i < n; i++) {
        _ModMult(c[i-1], Z_arr[i], c[i]);
    }

    uint64_t u[4];
    _ModInv(c[n-1], u);

    // === Phase 3: Compute individual inverses and check prefix match ===
    for (int32_t i = n - 1; i >= 0; i--) {
        uint64_t Z_inv[4];
        if (i > 0) {
            _ModMult(u, c[i-1], Z_inv);
        } else {
            for (int j = 0; j < 4; j++) Z_inv[j] = u[j];
        }

        uint64_t Z_inv_squared[4];
        _ModSquare(Z_inv, Z_inv_squared);

        uint64_t x[4];
        _ModMult(X_arr[i], Z_inv_squared, x);

        // Endomorphism: compute β*x and β²*x
        uint64_t x_beta[4], x_beta2[4];
        _ModMultByBeta(x, x_beta);
        _ModMultByBeta2(x, x_beta2);

        uint64_t* x_coords[3] = { x, x_beta, x_beta2 };

        // Prefix matching (32-bit for speed, CPU re-verifies with 64-bit)
        for (uint32_t endo = 0; endo < 3; endo++) {
            // Extract upper 32 bits of x coordinate for fast matching
            uint32_t x_upper32 = (uint32_t)(x_coords[endo][3] >> 32);

            // Optimized path for single prefix (most common case)
            bool matched = false;
            if (_num_prefixes == 1) {
                matched = ((x_upper32 & _masks[0]) == _patterns[0]);
            } else {
                // 32bit × 2 concatenation: check 2 prefixes with 1 64-bit operation
                // This halves the loop iterations for multiple prefixes
                uint64_t x_doubled = ((uint64_t)x_upper32 << 32) | x_upper32;

                uint32_t pair_count = _num_prefixes / 2;
                for (uint32_t p = 0; p < pair_count; p++) {
                    uint32_t idx = p * 2;
                    // Concatenate two 32-bit patterns/masks into 64-bit
                    uint64_t combined_pattern = ((uint64_t)_patterns[idx + 1] << 32) | _patterns[idx];
                    uint64_t combined_mask = ((uint64_t)_masks[idx + 1] << 32) | _masks[idx];

                    // XOR and mask: if lower 32 bits are 0, patterns[idx] matches
                    //               if upper 32 bits are 0, patterns[idx+1] matches
                    uint64_t diff = (x_doubled ^ combined_pattern) & combined_mask;
                    if ((diff & 0xFFFFFFFFULL) == 0 || (diff >> 32) == 0) {
                        matched = true;
                        break;
                    }
                }

                // Handle odd number of prefixes: check the last one separately
                if (!matched && (_num_prefixes & 1)) {
                    uint32_t p = _num_prefixes - 1;
                    if ((x_upper32 & _masks[p]) == _patterns[p]) {
                        matched = true;
                    }
                }
            }

            if (matched) {
                uint32_t slot = atomicAdd(match_count, 1);
                if (slot < _max_matches) {
                    matched_base_idx[slot] = idx;
                    matched_offset[slot] = i;
                    matched_endo_type[slot] = endo;

                    for (int j = 0; j < 4; j++) {
                        matched_pubkeys_x[slot * 4 + j] = x_coords[endo][j];
                    }

                    // Compute actual secret key: thread_base_key + i
                    // (with carry propagation)
                    uint64_t actual_key[4];
                    uint64_t key_sum = thread_base_key[0] + (uint64_t)i;
                    uint64_t key_carry = (key_sum < thread_base_key[0]) ? 1 : 0;
                    actual_key[0] = key_sum;

                    for (int j = 1; j < 4; j++) {
                        key_sum = thread_base_key[j] + key_carry;
                        key_carry = (key_sum < key_carry) ? 1 : 0;
                        actual_key[j] = key_sum;
                    }

                    for (int j = 0; j < 4; j++) {
                        matched_secret_keys[slot * 4 + j] = actual_key[j];
                    }
                }
            }
        }

        // Update u for next iteration
        if (i > 0) {
            uint64_t temp[4];
            _ModMult(u, Z_arr[i], temp);
            for (int j = 0; j < 4; j++) {
                u[j] = temp[j];
            }
        }
    }
}
