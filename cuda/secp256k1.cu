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
#define P0 0xFFFFFFFEFFFFFC2FULL
#define P1 0xFFFFFFFFFFFFFFFFULL
#define P2 0xFFFFFFFFFFFFFFFFULL
#define P3 0xFFFFFFFFFFFFFFFFULL

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
// 64-bit Arithmetic Primitives (PTX carry chain)
// ============================================================================

/**
 * Add two 64-bit numbers (a + b)
 * Returns carry (0 or 1)
 *
 * Uses PTX 32-bit carry chain for efficiency.
 */
__device__ uint32_t _Add64(uint64_t a, uint64_t b, uint64_t* sum)
{
    uint32_t a0 = (uint32_t)a;
    uint32_t a1 = (uint32_t)(a >> 32);
    uint32_t b0 = (uint32_t)b;
    uint32_t b1 = (uint32_t)(b >> 32);
    uint32_t r0, r1, carry;

    asm volatile (
        "add.cc.u32  %0, %3, %5;\n\t"   // r0 = a0 + b0, carry out
        "addc.cc.u32 %1, %4, %6;\n\t"   // r1 = a1 + b1 + carry, carry out
        "addc.u32    %2, 0, 0;\n\t"     // carry = 0 + 0 + carry
        : "=r"(r0), "=r"(r1), "=r"(carry)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1)
    );

    *sum = ((uint64_t)r1 << 32) | r0;
    return carry;
}

/**
 * Add two 320-bit numbers (5 limbs each)
 * Returns carry-out (0 or 1)
 *
 * Uses PTX 32-bit carry chain for efficiency.
 * Replaces _Add64 + _Addc64 x 4 pattern (27 PTX instructions → 11 instructions).
 */
__device__ uint32_t _Add320(const uint64_t a[5], const uint64_t b[5], uint64_t sum[5])
{
    uint32_t a0 = (uint32_t)a[0];
    uint32_t a1 = (uint32_t)(a[0] >> 32);
    uint32_t a2 = (uint32_t)a[1];
    uint32_t a3 = (uint32_t)(a[1] >> 32);
    uint32_t a4 = (uint32_t)a[2];
    uint32_t a5 = (uint32_t)(a[2] >> 32);
    uint32_t a6 = (uint32_t)a[3];
    uint32_t a7 = (uint32_t)(a[3] >> 32);
    uint32_t a8 = (uint32_t)a[4];
    uint32_t a9 = (uint32_t)(a[4] >> 32);

    uint32_t b0 = (uint32_t)b[0];
    uint32_t b1 = (uint32_t)(b[0] >> 32);
    uint32_t b2 = (uint32_t)b[1];
    uint32_t b3 = (uint32_t)(b[1] >> 32);
    uint32_t b4 = (uint32_t)b[2];
    uint32_t b5 = (uint32_t)(b[2] >> 32);
    uint32_t b6 = (uint32_t)b[3];
    uint32_t b7 = (uint32_t)(b[3] >> 32);
    uint32_t b8 = (uint32_t)b[4];
    uint32_t b9 = (uint32_t)(b[4] >> 32);

    uint32_t r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, carry;

    asm volatile (
        "add.cc.u32   %0, %11, %21;\n\t"   // r0 = a0 + b0
        "addc.cc.u32  %1, %12, %22;\n\t"   // r1 = a1 + b1 + carry
        "addc.cc.u32  %2, %13, %23;\n\t"   // r2 = a2 + b2 + carry
        "addc.cc.u32  %3, %14, %24;\n\t"   // r3 = a3 + b3 + carry
        "addc.cc.u32  %4, %15, %25;\n\t"   // r4 = a4 + b4 + carry
        "addc.cc.u32  %5, %16, %26;\n\t"   // r5 = a5 + b5 + carry
        "addc.cc.u32  %6, %17, %27;\n\t"   // r6 = a6 + b6 + carry
        "addc.cc.u32  %7, %18, %28;\n\t"   // r7 = a7 + b7 + carry
        "addc.cc.u32  %8, %19, %29;\n\t"   // r8 = a8 + b8 + carry
        "addc.cc.u32  %9, %20, %30;\n\t"   // r9 = a9 + b9 + carry
        "addc.u32     %10, 0, 0;\n\t"      // carry = 0 + 0 + carry
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4),
          "=r"(r5), "=r"(r6), "=r"(r7), "=r"(r8), "=r"(r9), "=r"(carry)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4),
          "r"(a5), "r"(a6), "r"(a7), "r"(a8), "r"(a9),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(b4),
          "r"(b5), "r"(b6), "r"(b7), "r"(b8), "r"(b9)
    );

    sum[0] = ((uint64_t)r1 << 32) | r0;
    sum[1] = ((uint64_t)r3 << 32) | r2;
    sum[2] = ((uint64_t)r5 << 32) | r4;
    sum[3] = ((uint64_t)r7 << 32) | r6;
    sum[4] = ((uint64_t)r9 << 32) | r8;

    return carry;
}

/**
 * Add 128-bit + carry to 256-bit number in-place
 * a[0..3] += (b0, b1, carry_in, 0)
 * Returns carry-out (0 or 1)
 *
 * Uses PTX 32-bit carry chain for efficiency.
 * Replaces _Add64 + _Addc64 x 3 pattern (21 PTX instructions → 9 instructions).
 */
__device__ uint32_t _Add256Plus128(uint64_t a[4], uint64_t b0, uint64_t b1, uint32_t carry_in)
{
    uint32_t a0 = (uint32_t)a[0];
    uint32_t a1 = (uint32_t)(a[0] >> 32);
    uint32_t a2 = (uint32_t)a[1];
    uint32_t a3 = (uint32_t)(a[1] >> 32);
    uint32_t a4 = (uint32_t)a[2];
    uint32_t a5 = (uint32_t)(a[2] >> 32);
    uint32_t a6 = (uint32_t)a[3];
    uint32_t a7 = (uint32_t)(a[3] >> 32);

    uint32_t b0_lo = (uint32_t)b0;
    uint32_t b0_hi = (uint32_t)(b0 >> 32);
    uint32_t b1_lo = (uint32_t)b1;
    uint32_t b1_hi = (uint32_t)(b1 >> 32);

    uint32_t r0, r1, r2, r3, r4, r5, r6, r7, carry;

    asm volatile (
        "add.cc.u32   %0, %9, %17;\n\t"    // r0 = a0 + b0_lo
        "addc.cc.u32  %1, %10, %18;\n\t"   // r1 = a1 + b0_hi + carry
        "addc.cc.u32  %2, %11, %19;\n\t"   // r2 = a2 + b1_lo + carry
        "addc.cc.u32  %3, %12, %20;\n\t"   // r3 = a3 + b1_hi + carry
        "addc.cc.u32  %4, %13, %21;\n\t"   // r4 = a4 + carry_in + carry
        "addc.cc.u32  %5, %14, 0;\n\t"     // r5 = a5 + 0 + carry
        "addc.cc.u32  %6, %15, 0;\n\t"     // r6 = a6 + 0 + carry
        "addc.cc.u32  %7, %16, 0;\n\t"     // r7 = a7 + 0 + carry
        "addc.u32     %8, 0, 0;\n\t"       // carry = 0 + 0 + carry
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3),
          "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7), "=r"(carry)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(a4), "r"(a5), "r"(a6), "r"(a7),
          "r"(b0_lo), "r"(b0_hi), "r"(b1_lo), "r"(b1_hi), "r"(carry_in)
    );

    a[0] = ((uint64_t)r1 << 32) | r0;
    a[1] = ((uint64_t)r3 << 32) | r2;
    a[2] = ((uint64_t)r5 << 32) | r4;
    a[3] = ((uint64_t)r7 << 32) | r6;

    return carry;
}

/**
 * Add 128-bit number to array in-place
 * a[0..1] += (b0, b1)
 * Returns carry-out (0 or 1)
 *
 * Uses PTX 32-bit carry chain for efficiency.
 * Replaces _Add64 + _Add64x3 pattern (9 PTX instructions → 5 instructions).
 */
__device__ uint32_t _Add128(uint64_t a[2], uint64_t b0, uint64_t b1)
{
    uint32_t a0 = (uint32_t)a[0];
    uint32_t a1 = (uint32_t)(a[0] >> 32);
    uint32_t a2 = (uint32_t)a[1];
    uint32_t a3 = (uint32_t)(a[1] >> 32);

    uint32_t b0_lo = (uint32_t)b0;
    uint32_t b0_hi = (uint32_t)(b0 >> 32);
    uint32_t b1_lo = (uint32_t)b1;
    uint32_t b1_hi = (uint32_t)(b1 >> 32);

    uint32_t r0, r1, r2, r3, carry;

    asm volatile (
        "add.cc.u32   %0, %5, %9;\n\t"    // r0 = a0 + b0_lo
        "addc.cc.u32  %1, %6, %10;\n\t"   // r1 = a1 + b0_hi + carry
        "addc.cc.u32  %2, %7, %11;\n\t"   // r2 = a2 + b1_lo + carry
        "addc.cc.u32  %3, %8, %12;\n\t"   // r3 = a3 + b1_hi + carry
        "addc.u32     %4, 0, 0;\n\t"      // carry = 0 + 0 + carry
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(carry)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0_lo), "r"(b0_hi), "r"(b1_lo), "r"(b1_hi)
    );

    a[0] = ((uint64_t)r1 << 32) | r0;
    a[1] = ((uint64_t)r3 << 32) | r2;

    return carry;
}

/**
 * Add two 128-bit numbers, store result in separate output
 * (a0, a1) + (b0, b1) → (result[0], result[1])
 * Returns carry-out (0 or 1)
 *
 * Uses PTX 32-bit carry chain for efficiency.
 * Used in _ReduceOverflow for (shifted + mult) calculation.
 */
__device__ uint32_t _Add128To(uint64_t a0, uint64_t a1, uint64_t b0, uint64_t b1, uint64_t result[2])
{
    uint32_t a0_lo = (uint32_t)a0;
    uint32_t a0_hi = (uint32_t)(a0 >> 32);
    uint32_t a1_lo = (uint32_t)a1;
    uint32_t a1_hi = (uint32_t)(a1 >> 32);

    uint32_t b0_lo = (uint32_t)b0;
    uint32_t b0_hi = (uint32_t)(b0 >> 32);
    uint32_t b1_lo = (uint32_t)b1;
    uint32_t b1_hi = (uint32_t)(b1 >> 32);

    uint32_t r0, r1, r2, r3, carry;

    asm volatile (
        "add.cc.u32   %0, %5, %9;\n\t"    // r0 = a0_lo + b0_lo
        "addc.cc.u32  %1, %6, %10;\n\t"   // r1 = a0_hi + b0_hi + carry
        "addc.cc.u32  %2, %7, %11;\n\t"   // r2 = a1_lo + b1_lo + carry
        "addc.cc.u32  %3, %8, %12;\n\t"   // r3 = a1_hi + b1_hi + carry
        "addc.u32     %4, 0, 0;\n\t"      // carry = 0 + 0 + carry
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(carry)
        : "r"(a0_lo), "r"(a0_hi), "r"(a1_lo), "r"(a1_hi),
          "r"(b0_lo), "r"(b0_hi), "r"(b1_lo), "r"(b1_hi)
    );

    result[0] = ((uint64_t)r1 << 32) | r0;
    result[1] = ((uint64_t)r3 << 32) | r2;

    return carry;
}

/**
 * Add uint64 scalar to uint256, propagate carry through all limbs
 * result[0..3] = base[0..3] + scalar
 *
 * Uses PTX 32-bit carry chain for efficiency.
 * Used for secret key calculation: base_key + offset.
 */
__device__ void _PropagateCarry256(const uint64_t base[4], uint64_t scalar, uint64_t result[4])
{
    uint32_t b0 = (uint32_t)base[0];
    uint32_t b1 = (uint32_t)(base[0] >> 32);
    uint32_t b2 = (uint32_t)base[1];
    uint32_t b3 = (uint32_t)(base[1] >> 32);
    uint32_t b4 = (uint32_t)base[2];
    uint32_t b5 = (uint32_t)(base[2] >> 32);
    uint32_t b6 = (uint32_t)base[3];
    uint32_t b7 = (uint32_t)(base[3] >> 32);

    uint32_t s0 = (uint32_t)scalar;
    uint32_t s1 = (uint32_t)(scalar >> 32);

    uint32_t r0, r1, r2, r3, r4, r5, r6, r7;

    asm volatile (
        "add.cc.u32   %0, %8, %16;\n\t"   // r0 = b0 + s0
        "addc.cc.u32  %1, %9, %17;\n\t"   // r1 = b1 + s1 + carry
        "addc.cc.u32  %2, %10, 0;\n\t"    // r2 = b2 + 0 + carry
        "addc.cc.u32  %3, %11, 0;\n\t"    // r3 = b3 + 0 + carry
        "addc.cc.u32  %4, %12, 0;\n\t"    // r4 = b4 + 0 + carry
        "addc.cc.u32  %5, %13, 0;\n\t"    // r5 = b5 + 0 + carry
        "addc.cc.u32  %6, %14, 0;\n\t"    // r6 = b6 + 0 + carry
        "addc.u32     %7, %15, 0;\n\t"    // r7 = b7 + 0 + carry
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7)
        : "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(b4), "r"(b5), "r"(b6), "r"(b7),
          "r"(s0), "r"(s1)
    );

    result[0] = ((uint64_t)r1 << 32) | r0;
    result[1] = ((uint64_t)r3 << 32) | r2;
    result[2] = ((uint64_t)r5 << 32) | r4;
    result[3] = ((uint64_t)r7 << 32) | r6;
}

/**
 * Add 512-bit numbers in-place
 * a[0..7] += b[0..7]
 *
 * Uses PTX 32-bit carry chain for efficiency.
 * Replaces _Add64x3 loop (48 PTX instructions → 17 instructions).
 */
__device__ void _Add512(uint64_t a[8], const uint64_t b[8])
{
    uint32_t a0  = (uint32_t)a[0];  uint32_t a1  = (uint32_t)(a[0] >> 32);
    uint32_t a2  = (uint32_t)a[1];  uint32_t a3  = (uint32_t)(a[1] >> 32);
    uint32_t a4  = (uint32_t)a[2];  uint32_t a5  = (uint32_t)(a[2] >> 32);
    uint32_t a6  = (uint32_t)a[3];  uint32_t a7  = (uint32_t)(a[3] >> 32);
    uint32_t a8  = (uint32_t)a[4];  uint32_t a9  = (uint32_t)(a[4] >> 32);
    uint32_t a10 = (uint32_t)a[5];  uint32_t a11 = (uint32_t)(a[5] >> 32);
    uint32_t a12 = (uint32_t)a[6];  uint32_t a13 = (uint32_t)(a[6] >> 32);
    uint32_t a14 = (uint32_t)a[7];  uint32_t a15 = (uint32_t)(a[7] >> 32);

    uint32_t b0  = (uint32_t)b[0];  uint32_t b1  = (uint32_t)(b[0] >> 32);
    uint32_t b2  = (uint32_t)b[1];  uint32_t b3  = (uint32_t)(b[1] >> 32);
    uint32_t b4  = (uint32_t)b[2];  uint32_t b5  = (uint32_t)(b[2] >> 32);
    uint32_t b6  = (uint32_t)b[3];  uint32_t b7  = (uint32_t)(b[3] >> 32);
    uint32_t b8  = (uint32_t)b[4];  uint32_t b9  = (uint32_t)(b[4] >> 32);
    uint32_t b10 = (uint32_t)b[5];  uint32_t b11 = (uint32_t)(b[5] >> 32);
    uint32_t b12 = (uint32_t)b[6];  uint32_t b13 = (uint32_t)(b[6] >> 32);
    uint32_t b14 = (uint32_t)b[7];  uint32_t b15 = (uint32_t)(b[7] >> 32);

    uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
    uint32_t r8, r9, r10, r11, r12, r13, r14, r15;

    asm volatile (
        "add.cc.u32   %0,  %16, %32;\n\t"
        "addc.cc.u32  %1,  %17, %33;\n\t"
        "addc.cc.u32  %2,  %18, %34;\n\t"
        "addc.cc.u32  %3,  %19, %35;\n\t"
        "addc.cc.u32  %4,  %20, %36;\n\t"
        "addc.cc.u32  %5,  %21, %37;\n\t"
        "addc.cc.u32  %6,  %22, %38;\n\t"
        "addc.cc.u32  %7,  %23, %39;\n\t"
        "addc.cc.u32  %8,  %24, %40;\n\t"
        "addc.cc.u32  %9,  %25, %41;\n\t"
        "addc.cc.u32  %10, %26, %42;\n\t"
        "addc.cc.u32  %11, %27, %43;\n\t"
        "addc.cc.u32  %12, %28, %44;\n\t"
        "addc.cc.u32  %13, %29, %45;\n\t"
        "addc.cc.u32  %14, %30, %46;\n\t"
        "addc.u32     %15, %31, %47;\n\t"
        : "=r"(r0),  "=r"(r1),  "=r"(r2),  "=r"(r3),
          "=r"(r4),  "=r"(r5),  "=r"(r6),  "=r"(r7),
          "=r"(r8),  "=r"(r9),  "=r"(r10), "=r"(r11),
          "=r"(r12), "=r"(r13), "=r"(r14), "=r"(r15)
        : "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
          "r"(a4),  "r"(a5),  "r"(a6),  "r"(a7),
          "r"(a8),  "r"(a9),  "r"(a10), "r"(a11),
          "r"(a12), "r"(a13), "r"(a14), "r"(a15),
          "r"(b0),  "r"(b1),  "r"(b2),  "r"(b3),
          "r"(b4),  "r"(b5),  "r"(b6),  "r"(b7),
          "r"(b8),  "r"(b9),  "r"(b10), "r"(b11),
          "r"(b12), "r"(b13), "r"(b14), "r"(b15)
    );

    a[0] = ((uint64_t)r1  << 32) | r0;
    a[1] = ((uint64_t)r3  << 32) | r2;
    a[2] = ((uint64_t)r5  << 32) | r4;
    a[3] = ((uint64_t)r7  << 32) | r6;
    a[4] = ((uint64_t)r9  << 32) | r8;
    a[5] = ((uint64_t)r11 << 32) | r10;
    a[6] = ((uint64_t)r13 << 32) | r12;
    a[7] = ((uint64_t)r15 << 32) | r14;
}

/**
 * Add three 64-bit numbers (a + b + c)
 * Returns carry-out (0, 1, or 2)
 *
 * Uses PTX 32-bit carry chain for efficiency.
 * Useful for accumulating products in multiplication loops.
 */
__device__ uint32_t _Add64x3(uint64_t a, uint64_t b, uint64_t c, uint64_t* sum)
{
    uint32_t a0 = (uint32_t)a;
    uint32_t a1 = (uint32_t)(a >> 32);
    uint32_t b0 = (uint32_t)b;
    uint32_t b1 = (uint32_t)(b >> 32);
    uint32_t c0 = (uint32_t)c;
    uint32_t c1 = (uint32_t)(c >> 32);
    uint32_t s0, s1, carry_out;

    asm volatile (
        // Step 1: a + b
        "add.cc.u32  %0, %3, %5;\n\t"    // t0 = a0 + b0, carry
        "addc.cc.u32 %1, %4, %6;\n\t"    // t1 = a1 + b1 + carry
        "addc.u32    %2, 0, 0;\n\t"      // c_ab = carry (0 or 1)
        // Step 2: (a + b) + c
        "add.cc.u32  %0, %0, %7;\n\t"    // s0 = t0 + c0, carry
        "addc.cc.u32 %1, %1, %8;\n\t"    // s1 = t1 + c1 + carry
        "addc.u32    %2, %2, 0;\n\t"     // carry_out = c_ab + carry (0, 1, or 2)
        : "=r"(s0), "=r"(s1), "=r"(carry_out)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(c0), "r"(c1)
    );

    *sum = ((uint64_t)s1 << 32) | s0;
    return carry_out;  // 0, 1, or 2
}

// ============================================================================
// 256-bit Arithmetic Helper Functions (Device Functions)
// ============================================================================

/**
 * Add two 256-bit numbers (a + b)
 * Returns the result and carry
 *
 * Uses PTX 32-bit carry chain (add.cc/addc.cc) for efficiency.
 */
__device__ void _Add256(const uint64_t a[4], const uint64_t b[4], uint64_t result[4], uint64_t* carry)
{
    // Extract 32-bit limbs from 64-bit values (little-endian)
    uint32_t a0 = (uint32_t)a[0];
    uint32_t a1 = (uint32_t)(a[0] >> 32);
    uint32_t a2 = (uint32_t)a[1];
    uint32_t a3 = (uint32_t)(a[1] >> 32);
    uint32_t a4 = (uint32_t)a[2];
    uint32_t a5 = (uint32_t)(a[2] >> 32);
    uint32_t a6 = (uint32_t)a[3];
    uint32_t a7 = (uint32_t)(a[3] >> 32);

    uint32_t b0 = (uint32_t)b[0];
    uint32_t b1 = (uint32_t)(b[0] >> 32);
    uint32_t b2 = (uint32_t)b[1];
    uint32_t b3 = (uint32_t)(b[1] >> 32);
    uint32_t b4 = (uint32_t)b[2];
    uint32_t b5 = (uint32_t)(b[2] >> 32);
    uint32_t b6 = (uint32_t)b[3];
    uint32_t b7 = (uint32_t)(b[3] >> 32);

    uint32_t r0, r1, r2, r3, r4, r5, r6, r7, c;

    // 256-bit addition using PTX carry chain
    // add.cc  = add with carry-out
    // addc.cc = add with carry-in and carry-out
    // addc    = add with carry-in (final)
    asm volatile (
        "add.cc.u32   %0, %9, %17;\n\t"    // r0 = a0 + b0, carry out
        "addc.cc.u32  %1, %10, %18;\n\t"   // r1 = a1 + b1 + carry, carry out
        "addc.cc.u32  %2, %11, %19;\n\t"   // r2 = a2 + b2 + carry, carry out
        "addc.cc.u32  %3, %12, %20;\n\t"   // r3 = a3 + b3 + carry, carry out
        "addc.cc.u32  %4, %13, %21;\n\t"   // r4 = a4 + b4 + carry, carry out
        "addc.cc.u32  %5, %14, %22;\n\t"   // r5 = a5 + b5 + carry, carry out
        "addc.cc.u32  %6, %15, %23;\n\t"   // r6 = a6 + b6 + carry, carry out
        "addc.cc.u32  %7, %16, %24;\n\t"   // r7 = a7 + b7 + carry, carry out
        "addc.u32     %8, 0, 0;\n\t"       // c = 0 + 0 + carry (extract final carry)
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3),
          "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7), "=r"(c)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(a4), "r"(a5), "r"(a6), "r"(a7),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "r"(b4), "r"(b5), "r"(b6), "r"(b7)
    );

    // Combine 32-bit limbs back to 64-bit values
    result[0] = ((uint64_t)r1 << 32) | r0;
    result[1] = ((uint64_t)r3 << 32) | r2;
    result[2] = ((uint64_t)r5 << 32) | r4;
    result[3] = ((uint64_t)r7 << 32) | r6;
    *carry = c;
}

/**
 * Subtract two 256-bit numbers (a - b)
 * Returns the result and borrow (1 if a < b, 0 otherwise)
 *
 * Uses PTX 32-bit borrow chain (sub.cc/subc.cc) for efficiency.
 */
__device__ void _Sub256(const uint64_t a[4], const uint64_t b[4], uint64_t result[4], uint64_t* borrow_out)
{
    // Extract 32-bit limbs from 64-bit values (little-endian)
    uint32_t a0 = (uint32_t)a[0];
    uint32_t a1 = (uint32_t)(a[0] >> 32);
    uint32_t a2 = (uint32_t)a[1];
    uint32_t a3 = (uint32_t)(a[1] >> 32);
    uint32_t a4 = (uint32_t)a[2];
    uint32_t a5 = (uint32_t)(a[2] >> 32);
    uint32_t a6 = (uint32_t)a[3];
    uint32_t a7 = (uint32_t)(a[3] >> 32);

    uint32_t b0 = (uint32_t)b[0];
    uint32_t b1 = (uint32_t)(b[0] >> 32);
    uint32_t b2 = (uint32_t)b[1];
    uint32_t b3 = (uint32_t)(b[1] >> 32);
    uint32_t b4 = (uint32_t)b[2];
    uint32_t b5 = (uint32_t)(b[2] >> 32);
    uint32_t b6 = (uint32_t)b[3];
    uint32_t b7 = (uint32_t)(b[3] >> 32);

    uint32_t r0, r1, r2, r3, r4, r5, r6, r7, borrow;

    // 256-bit subtraction using PTX borrow chain
    // sub.cc  = subtract with borrow-out
    // subc.cc = subtract with borrow-in and borrow-out
    // subc    = subtract with borrow-in (final)
    asm volatile (
        "sub.cc.u32   %0, %9, %17;\n\t"    // r0 = a0 - b0, borrow out
        "subc.cc.u32  %1, %10, %18;\n\t"   // r1 = a1 - b1 - borrow, borrow out
        "subc.cc.u32  %2, %11, %19;\n\t"   // r2 = a2 - b2 - borrow, borrow out
        "subc.cc.u32  %3, %12, %20;\n\t"   // r3 = a3 - b3 - borrow, borrow out
        "subc.cc.u32  %4, %13, %21;\n\t"   // r4 = a4 - b4 - borrow, borrow out
        "subc.cc.u32  %5, %14, %22;\n\t"   // r5 = a5 - b5 - borrow, borrow out
        "subc.cc.u32  %6, %15, %23;\n\t"   // r6 = a6 - b6 - borrow, borrow out
        "subc.cc.u32  %7, %16, %24;\n\t"   // r7 = a7 - b7 - borrow, borrow out
        "subc.u32     %8, 0, 0;\n\t"       // borrow = 0 - 0 - borrow (extract final borrow)
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3),
          "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7), "=r"(borrow)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(a4), "r"(a5), "r"(a6), "r"(a7),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "r"(b4), "r"(b5), "r"(b6), "r"(b7)
    );

    // Combine 32-bit limbs back to 64-bit values
    result[0] = ((uint64_t)r1 << 32) | r0;
    result[1] = ((uint64_t)r3 << 32) | r2;
    result[2] = ((uint64_t)r5 << 32) | r4;
    result[3] = ((uint64_t)r7 << 32) | r6;
    // subc.u32 0, 0 with borrow gives 0xFFFFFFFF if borrow, 0 otherwise
    *borrow_out = borrow & 1;  // Convert 0xFFFFFFFF to 1
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
        uint64_t sum;
        uint32_t new_carry = _Add64(low, carry, &sum);

        mult977[i] = sum;
        carry = high_part + new_carry;
    }
    mult977[4] = carry;

    // Add: shifted + mult977 (using PTX carry chain, 11 instructions)
    uint64_t sum[5];
    _Add320(shifted, mult977, sum);

    // Reduce sum[4]: sum[4] * 2^256 mod p = sum[4] * (2^32 + 977)
    // Inlined: sum[4] == 0 is rare after 8-limb to 5-limb conversion
    {
        uint64_t factor = sum[4];
        uint64_t shifted_low = factor << 32;
        uint64_t shifted_high = factor >> 32;
        uint64_t mult_low, mult_high;
        _Mult64(factor, 977, &mult_low, &mult_high);
        uint64_t add[2];
        uint32_t c = _Add128To(shifted_low, shifted_high, mult_low, mult_high, add);
        sum[4] = _Add256Plus128(sum, add[0], add[1], c);
    }

    // Add to low
    uint64_t temp[5];
    temp[0] = low[0];
    temp[1] = low[1];
    temp[2] = low[2];
    temp[3] = low[3];
    temp[4] = 0;

    // Add: temp + sum (using PTX carry chain, 11 instructions)
    _Add320(temp, sum, temp);

    // Now reduce: while temp >= p, subtract p
    // At most 2-3 iterations needed
    // temp >= p iff: temp[4] > 0, OR (temp[3..1] all == max_uint64 AND temp[0] >= P0)
    const uint64_t p[4] = {P0, P1, P2, P3};
    while ((temp[4] > 0) ||
           (temp[3] == P3 && temp[2] == P2 && temp[1] == P1 && temp[0] >= P0)) {
        // Subtract p from temp using PTX borrow chain (single _Sub256 call)
        uint64_t borrow64;
        _Sub256(temp, p, temp, &borrow64);
        temp[4] -= borrow64;
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

            // Add temp[i+j] + low + carry using PTX carry chain
            uint32_t c = _Add64x3(temp[i + j], low, carry, &temp[i + j]);
            carry = high + c;
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

            // Add cross[i+j] + low + carry using PTX carry chain
            uint32_t c = _Add64x3(cross[i + j], low, carry, &cross[i + j]);
            carry = high + c;
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

        // Add to temp[2*i..2*i+1] using single PTX call (5 instructions)
        uint32_t c2 = _Add128(&temp[2 * i], low, high);

        // Propagate carry to higher limbs
        if (2 * i + 2 < 8) {
            temp[2 * i + 2] += c2;
        }
    }

    // Step 3: Add doubled cross products to temp using single PTX call (16 instructions)
    _Add512(temp, cross);

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
    _ModSquare(x11, t);
    for (int i = 1; i < 11; i++) {
        _ModSquare(t, t);
    }
    _ModMult(t, x11, x22);

    // x44 = a^(2^44 - 1)
    _ModSquare(x22, t);
    for (int i = 1; i < 22; i++) {
        _ModSquare(t, t);
    }
    _ModMult(t, x22, x44);

    // x88 = a^(2^88 - 1)
    _ModSquare(x44, t);
    for (int i = 1; i < 44; i++) {
        _ModSquare(t, t);
    }
    _ModMult(t, x44, x88);

    // x176 = a^(2^176 - 1)
    _ModSquare(x88, t);
    for (int i = 1; i < 88; i++) {
        _ModSquare(t, t);
    }
    _ModMult(t, x88, x176);

    // x220 = a^(2^220 - 1)
    _ModSquare(x176, t);
    for (int i = 1; i < 44; i++) {
        _ModSquare(t, t);
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
    _ModSquare(x223, t);
    for (int i = 1; i < 23; i++) {
        _ModSquare(t, t);
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
 *   - Z3^2 = Z1^2 * H^2 (computed for Montgomery's Trick optimization)
 *
 * Cost: 9M + 2S (vs 12M + 4S for general point addition)
 *   - 8M + 2S for core point addition
 *   - +1M for computing Z3^2 (saves 1S per call in chain)
 */
// In-place point addition: P1 = P1 + P2 (Mixed coordinates)
// P1: Jacobian (X, Y, Z, Z_squared), P2: Affine (X2, Y2, Z2=1)
__device__ void _PointAddMixed(
    uint64_t X[4], uint64_t Y[4], uint64_t Z[4],  // Jacobian point (in/out)
    uint64_t Z_squared[4],                         // Z^2 (in/out)
    const uint64_t X2[4], const uint64_t Y2[4]     // Affine point (Z2=1)
)
{
    uint64_t Z_cubed[4];
    uint64_t U2[4], S2[4];
    uint64_t H[4], H_squared[4], H_cubed[4];
    uint64_t R[4], R_squared[4];
    uint64_t X_H2[4];  // X * H^2 (reused to avoid duplicate computation)
    uint64_t temp[4], temp2[4];
    uint64_t newX[4], newY[4], newZ[4], newZ_squared[4];

    // Z^3 (Z^2 is passed as input)
    _ModMult(Z_squared, Z, Z_cubed);          // M1: Z^3

    // U1 = X (since Z2^2 = 1, no multiplication needed)
    // S1 = Y (since Z2^3 = 1, no multiplication needed)

    // U2 = X2 * Z^2
    _ModMult(X2, Z_squared, U2);              // M2: U2

    // S2 = Y2 * Z^3
    _ModMult(Y2, Z_cubed, S2);                // M3: S2

    // H = U2 - U1 = U2 - X
    _ModSub(U2, X, H);

    // R = S2 - S1 = S2 - Y
    _ModSub(S2, Y, R);

    // H^2 and H^3
    _ModSquare(H, H_squared);                 // S2: H^2
    _ModMult(H_squared, H, H_cubed);          // M4: H^3

    // R^2
    _ModSquare(R, R_squared);                 // S3: R^2

    // newX = R^2 - H^3 - 2*U1*H^2 = R^2 - H^3 - 2*X*H^2
    _ModMult(X, H_squared, X_H2);             // M5: X * H^2 (save for reuse)
    _ModAdd(X_H2, X_H2, temp);                // temp = 2 * X * H^2
    _ModSub(R_squared, H_cubed, temp2);       // temp2 = R^2 - H^3
    _ModSub(temp2, temp, newX);               // newX = R^2 - H^3 - 2*X*H^2

    // newY = R * (U1*H^2 - newX) - S1*H^3 = R * (X*H^2 - newX) - Y*H^3
    // Reuse X_H2 instead of recomputing X * H^2
    _ModSub(X_H2, newX, temp);                // temp = X*H^2 - newX
    _ModMult(R, temp, temp2);                 // M6: R * (X*H^2 - newX)
    _ModMult(Y, H_cubed, temp);               // M7: Y * H^3
    _ModSub(temp2, temp, newY);               // newY = R * (X*H^2 - newX) - Y*H^3

    // newZ = Z * H (since Z2 = 1)
    _ModMult(Z, H, newZ);                     // M8: newZ

    // newZ^2 = Z^2 * H^2 (for Montgomery's Trick optimization)
    _ModMult(Z_squared, H_squared, newZ_squared);  // M9: newZ^2

    // Write results back
    for (int i = 0; i < 4; i++) {
        X[i] = newX[i];
        Y[i] = newY[i];
        Z[i] = newZ[i];
        Z_squared[i] = newZ_squared[i];
    }
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
 * @param Rz_squared      Output: Rz^2 (for Montgomery's Trick optimization)
 */
__device__ void _PointMultByIndex(
    uint32_t idx,
    const uint64_t base_pubkey_x[4],
    const uint64_t base_pubkey_y[4],
    uint64_t Rx[4], uint64_t Ry[4], uint64_t Rz[4],
    uint64_t Rz_squared[4]
)
{
    // Initialize result to base_pubkey (Jacobian with Z = 1)
    for (int i = 0; i < 4; i++) {
        Rx[i] = base_pubkey_x[i];
        Ry[i] = base_pubkey_y[i];
    }
    Rz[0] = 1; Rz[1] = 0; Rz[2] = 0; Rz[3] = 0;
    // Z^2 = 1 (since Z = 1)
    Rz_squared[0] = 1; Rz_squared[1] = 0; Rz_squared[2] = 0; Rz_squared[3] = 0;

    // Add dG_table[bit] for each set bit in idx
    for (int bit = 0; bit < 24; bit++) {
        if ((idx >> bit) & 1) {
            // Load _dG_table[bit] from constant memory (Affine coordinates)
            uint64_t dG_x[4], dG_y[4];
            for (int i = 0; i < 4; i++) {
                dG_x[i] = _dG_table[bit * 8 + i];
                dG_y[i] = _dG_table[bit * 8 + 4 + i];
            }

            // Add: R = R + dG_table[bit] (in-place)
            _PointAddMixed(Rx, Ry, Rz, Rz_squared, dG_x, dG_y);
        }
    }
}

// ============================================================================
// Production Kernels
// ============================================================================

// MAX_KEYS_PER_THREAD: Can be specified via -D option at build time (default: 1600)
#ifndef MAX_KEYS_PER_THREAD
#define MAX_KEYS_PER_THREAD 1600
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
 * Generate public keys using sequential key strategy with dG table optimization
 *
 * Each thread computes MAX_KEYS_PER_THREAD sequential keys:
 *   secret_key[i] = base_key + thread_idx * MAX_KEYS_PER_THREAD + i
 *   pubkey[i] = base_pubkey + thread_idx * dG + i * G
 *
 * Key optimizations:
 *   1. dG table: Precomputed [dG, 2*dG, 4*dG, ..., 2^23*dG] in constant memory
 *      - Initial pubkey via table lookup (~12 PointAdd) instead of PointMult (~384 ops)
 *   2. Sequential keys: Only pass base_key (32 bytes) instead of all keys
 *   3. Montgomery's Trick: Batch inverse for Jacobian -> Affine conversion
 *   4. Endomorphism: Check 3 prefixes (P, β*P, β²*P) per pubkey computation
 *
 * Constant memory inputs:
 *   - _dG_table[192]: dG table (24 entries × 8 uint64_t)
 *   - _patterns[256], _masks[256]: prefix matching patterns
 *   - _num_prefixes, _num_threads, _max_matches: runtime parameters
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

    // Local arrays for storing intermediate Jacobian coordinates (SoA layout)
    // Note: Y_arr is not needed! Prefix matching only uses x-coordinate (BIP-340)
    // SoA (Structure of Arrays) for better cache locality when iterating by key_idx
    uint64_t X_arr_0[MAX_KEYS_PER_THREAD], X_arr_1[MAX_KEYS_PER_THREAD];
    uint64_t X_arr_2[MAX_KEYS_PER_THREAD], X_arr_3[MAX_KEYS_PER_THREAD];
    uint64_t Z_arr_0[MAX_KEYS_PER_THREAD], Z_arr_1[MAX_KEYS_PER_THREAD];
    uint64_t Z_arr_2[MAX_KEYS_PER_THREAD], Z_arr_3[MAX_KEYS_PER_THREAD];
    uint64_t c_0[MAX_KEYS_PER_THREAD], c_1[MAX_KEYS_PER_THREAD];
    uint64_t c_2[MAX_KEYS_PER_THREAD], c_3[MAX_KEYS_PER_THREAD];

    const uint32_t n = MAX_KEYS_PER_THREAD;

    // === Compute starting secret key: base_key + idx * MAX_KEYS_PER_THREAD ===
    uint64_t global_offset = (uint64_t)idx * MAX_KEYS_PER_THREAD;
    uint64_t k[4];

    // Add global_offset to base_key: k = base_key + global_offset
    _PropagateCarry256(base_key, global_offset, k);

    // Save starting key for this thread (needed for match reporting)
    uint64_t thread_base_key[4];
    for (int i = 0; i < 4; i++) {
        thread_base_key[i] = k[i];
    }

    // === Phase 1: Generate all points in Jacobian coordinates ===

    // First point: P = base_pubkey + idx * dG (fast table lookup!)
    // Instead of _PointMult(k, G) with 256 double-and-add operations,
    // we use precomputed dG table and perform at most 24 point additions.
    uint64_t Px[4], Py[4], Pz[4], Pz_squared[4];
    _PointMultByIndex(idx, base_pubkey_x, base_pubkey_y, Px, Py, Pz, Pz_squared);

    X_arr_0[0] = Px[0]; X_arr_1[0] = Px[1]; X_arr_2[0] = Px[2]; X_arr_3[0] = Px[3];
    // Store Z^2 instead of Z (for Montgomery's Trick optimization)
    Z_arr_0[0] = Pz_squared[0]; Z_arr_1[0] = Pz_squared[1];
    Z_arr_2[0] = Pz_squared[2]; Z_arr_3[0] = Pz_squared[3];

    // Generate subsequent points using P = P + G
    // Also compute cumulative products c[i] = c[i-1] * Z[i]^2 for Montgomery's Trick
    const uint64_t Gx[4] = {GX0, GX1, GX2, GX3};
    const uint64_t Gy[4] = {GY0, GY1, GY2, GY3};

    // Initialize c[0] = Z^2[0] (cumulative product of Z^2 values)
    c_0[0] = Z_arr_0[0]; c_1[0] = Z_arr_1[0]; c_2[0] = Z_arr_2[0]; c_3[0] = Z_arr_3[0];

    for (uint32_t key_idx = 1; key_idx < n; key_idx++) {
        // P = P + G (in-place)
        _PointAddMixed(Px, Py, Pz, Pz_squared, Gx, Gy);

        X_arr_0[key_idx] = Px[0]; X_arr_1[key_idx] = Px[1];
        X_arr_2[key_idx] = Px[2]; X_arr_3[key_idx] = Px[3];
        // Store Z^2 instead of Z
        Z_arr_0[key_idx] = Pz_squared[0]; Z_arr_1[key_idx] = Pz_squared[1];
        Z_arr_2[key_idx] = Pz_squared[2]; Z_arr_3[key_idx] = Pz_squared[3];

        // Loop fusion: compute c[key_idx] = c[key_idx-1] * Z^2[key_idx]
        uint64_t c_prev[4] = {c_0[key_idx-1], c_1[key_idx-1], c_2[key_idx-1], c_3[key_idx-1]};
        uint64_t Z_sq_cur[4] = {Z_arr_0[key_idx], Z_arr_1[key_idx], Z_arr_2[key_idx], Z_arr_3[key_idx]};
        uint64_t c_result[4];
        _ModMult(c_prev, Z_sq_cur, c_result);
        c_0[key_idx] = c_result[0]; c_1[key_idx] = c_result[1];
        c_2[key_idx] = c_result[2]; c_3[key_idx] = c_result[3];
    }

    // === Phase 2: Montgomery's Trick for batch inverse ===
    // (cumulative products c[i] already computed above)

    uint64_t u[4];
    {
        uint64_t c_last[4] = {c_0[n-1], c_1[n-1], c_2[n-1], c_3[n-1]};
        _ModInv(c_last, u);
    }

    // === Phase 3: Compute individual inverses and check prefix match ===
    // Since c[i] = Z[0]^2 * Z[1]^2 * ... * Z[i]^2 (cumulative product of Z^2),
    // we get (Z[i]^-1)^2 directly without needing to square!
    for (int32_t i = n - 1; i >= 0; i--) {
        uint64_t Z_inv_squared[4];
        if (i > 0) {
            uint64_t c_prev[4] = {c_0[i-1], c_1[i-1], c_2[i-1], c_3[i-1]};
            _ModMult(u, c_prev, Z_inv_squared);  // Direct (Z^-1)^2 !
        } else {
            for (int j = 0; j < 4; j++) Z_inv_squared[j] = u[j];
        }

        // No _ModSquare needed! Z_inv_squared is already (Z^-1)^2

        uint64_t x[4];
        uint64_t X_cur[4] = {X_arr_0[i], X_arr_1[i], X_arr_2[i], X_arr_3[i]};
        _ModMult(X_cur, Z_inv_squared, x);

        // Endomorphism: compute β*x and β²*x
        const uint64_t beta[4] = {BETA0, BETA1, BETA2, BETA3};
        const uint64_t beta2[4] = {BETA2_0, BETA2_1, BETA2_2, BETA2_3};
        uint64_t x_beta[4], x_beta2[4];
        _ModMult(x, beta, x_beta);
        _ModMult(x, beta2, x_beta2);

        uint64_t* x_coords[3] = { x, x_beta, x_beta2 };

        // Prefix matching (32-bit for speed, CPU re-verifies with 64-bit)
        for (uint32_t endo = 0; endo < 3; endo++) {
            // Extract upper 32 bits of x coordinate for fast matching
            uint32_t x_upper32 = (uint32_t)(x_coords[endo][3] >> 32);

            // Optimized path for single prefix (most common case)
            // Simple 32-bit matching: first prefix inline, rest in loop
            // When _num_prefixes == 1, the loop condition (1 < 1) is immediately false,
            // allowing branch prediction to work effectively.
            bool matched = ((x_upper32 & _masks[0]) == _patterns[0]);
            for (uint32_t p = 1; p < _num_prefixes; p++) {
                matched |= ((x_upper32 & _masks[p]) == _patterns[p]);
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

                    // Compute actual secret key: actual_key = thread_base_key + i
                    uint64_t actual_key[4];
                    _PropagateCarry256(thread_base_key, (uint64_t)i, actual_key);

                    for (int j = 0; j < 4; j++) {
                        matched_secret_keys[slot * 4 + j] = actual_key[j];
                    }
                }
            }
        }

        // Update u for next iteration: u = u * Z[i]^2
        // (Z_arr stores Z^2 values, not Z)
        if (i > 0) {
            uint64_t Z_sq_cur[4] = {Z_arr_0[i], Z_arr_1[i], Z_arr_2[i], Z_arr_3[i]};
            uint64_t temp[4];
            _ModMult(u, Z_sq_cur, temp);
            for (int j = 0; j < 4; j++) {
                u[j] = temp[j];
            }
        }
    }
}
