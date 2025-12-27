// Allow C-style loops for 256-bit arithmetic (0..4 is clearer for limb operations)
#![allow(clippy::needless_range_loop)]

use bech32::{Bech32, Hrp, encode};
use secp256k1::{PublicKey, SecretKey};

// GPU module
pub mod gpu;

// =============================================================================
// Byte array ↔ [u64; 4] conversion functions (for GPU integration)
// =============================================================================

/// Convert byte array (32 bytes, big-endian) to [u64; 4] (little-endian limbs)
///
/// Used when passing private keys to GPU
/// - Input: big-endian byte array (`byte[0]` is the most significant byte)
/// - Output: little-endian limbs (`limb[0]` is the least significant 64 bits)
pub fn bytes_to_u64x4(bytes: &[u8; 32]) -> [u64; 4] {
    let mut result = [0u64; 4];
    // byte[24..32] → limb[0] (least significant)
    // byte[16..24] → limb[1]
    // byte[8..16]  → limb[2]
    // byte[0..8]   → limb[3] (most significant)
    for i in 0..4 {
        let offset = (3 - i) * 8; // reverse order
        result[i] = u64::from_be_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
    }
    result
}

/// Convert [u64; 4] (little-endian limbs) to byte array (32 bytes, big-endian)
///
/// Used when converting public keys returned from GPU to npub
/// - Input: little-endian limbs (`limb[0]` is the least significant 64 bits)
/// - Output: big-endian byte array (`byte[0]` is the most significant byte)
pub fn u64x4_to_bytes(value: &[u64; 4]) -> [u8; 32] {
    let mut result = [0u8; 32];
    // limb[3] (most significant) → byte[0..8]
    // limb[2]                    → byte[8..16]
    // limb[1]                    → byte[16..24]
    // limb[0] (least significant) → byte[24..32]
    for i in 0..4 {
        let offset = (3 - i) * 8; // reverse order
        let bytes = value[i].to_be_bytes();
        result[offset..offset + 8].copy_from_slice(&bytes);
    }
    result
}

/// Convert public key bytes (x-coordinate only, 32 bytes) to npub
///
/// Used to directly convert public keys returned from GPU to npub
/// The existing `pubkey_to_npub` goes through secp256k1::PublicKey,
/// but this function converts directly from byte array
pub fn pubkey_bytes_to_npub(pubkey_bytes: &[u8; 32]) -> String {
    let hrp = Hrp::parse("npub").expect("valid hrp");
    encode::<Bech32>(hrp, pubkey_bytes).expect("failed to encode npub")
}

/// Convert public key (x-coordinate only, 32 bytes) to npub
pub fn pubkey_to_npub(pubkey: &PublicKey) -> String {
    // Get hex string of public key (compressed format)
    let pk_hex = pubkey.to_string();
    // Extract x-coordinate only (remove first 2 characters)
    let pk_x_only = &pk_hex[2..];

    // Convert hex string to 32-byte array
    let mut bytes = [0u8; 32];
    hex::decode_to_slice(pk_x_only, &mut bytes).expect("Invalid hex string");

    // bech32 encode
    let hrp = Hrp::parse("npub").expect("valid hrp");
    encode::<Bech32>(hrp, &bytes).expect("failed to encode npub")
}

/// Convert secret key (32 bytes) to nsec
pub fn seckey_to_nsec(seckey: &SecretKey) -> String {
    // Get byte array of secret key
    let bytes = seckey.secret_bytes();

    // bech32 encode
    let hrp = Hrp::parse("nsec").expect("valid hrp");
    encode::<Bech32>(hrp, &bytes).expect("failed to encode nsec")
}

// =============================================================================
// Prefix → Bit array conversion (for fast GPU matching)
// =============================================================================

/// bech32 character set (order matters! each character's position corresponds to its 5-bit value)
const BECH32_CHARSET: &str = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";

/// Convert prefix to bit array (for fast GPU matching)
///
/// Convert each bech32 character to 5-bit value and concatenate into upper bits of u64
///
/// # Arguments
/// * `prefix` - Prefix to convert (max 12 characters = 60 bits)
///
/// # Returns
/// * `(pattern, mask, bit_len)` - Pattern, mask, and bit length
///   - pattern: Prefix concatenated as 5 bits each (placed in upper bits of u64)
///   - mask: Mask with valid bits set to 1 (upper bit_len bits are 1)
///   - bit_len: Number of valid bits (prefix_len * 5)
///
/// # Example
/// ```
/// use mocnpub_main::prefix_to_bits;
/// let (pattern, mask, bit_len) = prefix_to_bits("m0");
/// // 'm' = 27 (11011), '0' = 15 (01111)
/// // pattern = 0b11011_01111_000...0 (upper 10 bits)
/// // mask    = 0b11111_11111_000...0 (upper 10 bits are 1)
/// // bit_len = 10
/// ```
pub fn prefix_to_bits(prefix: &str) -> (u64, u64, u32) {
    let mut pattern: u64 = 0;
    let mut bit_pos: u32 = 64; // Place from upper bits

    for ch in prefix.chars() {
        // Get position in bech32 character set (0-31)
        let value = BECH32_CHARSET.find(ch).expect("invalid bech32 char") as u64;

        // Shift by 5 bits and place
        bit_pos -= 5;
        pattern |= value << bit_pos;
    }

    let bit_len = (prefix.len() as u32) * 5;
    let mask = if bit_len >= 64 {
        u64::MAX
    } else {
        !((1u64 << (64 - bit_len)) - 1) // Upper bit_len bits are 1
    };

    (pattern, mask, bit_len)
}

/// Convert multiple prefixes to bit arrays
///
/// # Returns
/// * `Vec<(pattern, mask, bit_len)>` - Pattern, mask, and bit length for each prefix
pub fn prefixes_to_bits(prefixes: &[String]) -> Vec<(u64, u64, u32)> {
    prefixes.iter().map(|p| prefix_to_bits(p)).collect()
}

// =============================================================================
// 256-bit arithmetic (for consecutive secret key strategy)
// =============================================================================

/// Add offset to 256-bit value ([u64; 4])
///
/// Used in consecutive secret key strategy to compute base_key + offset.
/// Since offset is u32, only addition to least significant limb and carry propagation needed.
///
/// # Arguments
/// * `base` - 256-bit value (little-endian limbs: `base[0]` is least significant)
/// * `offset` - Value to add (max u32::MAX)
///
/// # Returns
/// * `[u64; 4]` - Result of base + offset
///
/// # Example
/// ```
/// use mocnpub_main::add_u64x4_scalar;
/// let base = [0xFFFFFFFF_FFFFFFFFu64, 0, 0, 0];
/// let result = add_u64x4_scalar(&base, 1);
/// assert_eq!(result, [0, 1, 0, 0]); // carry occurred
/// ```
pub fn add_u64x4_scalar(base: &[u64; 4], offset: u32) -> [u64; 4] {
    let mut result = *base;

    // Add offset to least significant limb
    let (sum, carry) = result[0].overflowing_add(offset as u64);
    result[0] = sum;

    // Propagate carry
    if carry {
        let (sum, carry) = result[1].overflowing_add(1);
        result[1] = sum;
        if carry {
            let (sum, carry) = result[2].overflowing_add(1);
            result[2] = sum;
            if carry {
                result[3] = result[3].wrapping_add(1);
            }
        }
    }

    result
}

// =============================================================================
// Endomorphism Support (for 3x speedup)
// =============================================================================

/// secp256k1 group order n
/// n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
const N: [u64; 4] = [
    0xBFD25E8CD0364141,
    0xBAAEDCE6AF48A03B,
    0xFFFFFFFFFFFFFFFE,
    0xFFFFFFFFFFFFFFFF,
];

/// λ = cube root of unity mod n (for endomorphism)
/// λ = 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72
const LAMBDA: [u64; 4] = [
    0xdf02967c1b23bd72,
    0x122e22ea20816678,
    0xa5261c028812645a,
    0x5363ad4cc05c30e0,
];

/// λ² = λ * λ mod n
/// λ² = 0xac9c52b33fa3cf1f5ad9e3fd77ed9ba4a880b9fc8ec739c2e0cfc810b51283ce
const LAMBDA_SQ: [u64; 4] = [
    0xe0cfc810b51283ce,
    0xa880b9fc8ec739c2,
    0x5ad9e3fd77ed9ba4,
    0xac9c52b33fa3cf1f,
];

/// Adjust private key for endomorphism
///
/// When using endomorphism, we check 3 public keys: P, β*P, β²*P
/// The corresponding private keys are: k, λ*k, λ²*k (mod n)
///
/// # Arguments
/// * `privkey` - Original private key (base + offset)
/// * `endo_type` - 0 = original, 1 = λ*k, 2 = λ²*k
///
/// # Returns
/// Adjusted private key as [u64; 4]
pub fn adjust_privkey_for_endomorphism(privkey: &[u64; 4], endo_type: u32) -> [u64; 4] {
    match endo_type {
        0 => *privkey,
        1 => mod_n_mult(privkey, &LAMBDA),
        2 => mod_n_mult(privkey, &LAMBDA_SQ),
        _ => *privkey, // Should never happen
    }
}

/// 256-bit multiplication modulo n (secp256k1 group order)
///
/// Computes (a * b) mod n using schoolbook multiplication followed by Barrett-like reduction
fn mod_n_mult(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    // Step 1: 256x256 -> 512-bit multiplication
    let mut product = [0u64; 8];

    for i in 0..4 {
        let mut carry = 0u64;
        for j in 0..4 {
            let pos = i + j;
            let (lo, hi) = mul_u64(a[i], b[j]);

            // Add low part
            let (sum, c1) = product[pos].overflowing_add(lo);
            let (sum, c2) = sum.overflowing_add(carry);
            product[pos] = sum;
            carry = hi + (c1 as u64) + (c2 as u64);
        }
        product[i + 4] = product[i + 4].wrapping_add(carry);
    }

    // Step 2: Reduce mod n
    reduce_mod_n(&product)
}

/// Multiply two u64 values, returning (low, high) parts of the 128-bit result
#[inline]
fn mul_u64(a: u64, b: u64) -> (u64, u64) {
    let result = (a as u128) * (b as u128);
    (result as u64, (result >> 64) as u64)
}

/// Reduce a 512-bit number modulo n
///
/// Uses iterative reduction with the constant k = 2^256 - n
fn reduce_mod_n(val: &[u64; 8]) -> [u64; 4] {
    let high = [val[4], val[5], val[6], val[7]];

    // If high part is non-zero, we need to reduce
    if high != [0, 0, 0, 0] {
        return reduce_512_mod_n_simple(val);
    }

    // High part is zero, just check if low >= n
    let mut result = [val[0], val[1], val[2], val[3]];

    // If result >= n, subtract n
    while cmp_u64x4(&result, &N) >= 0 {
        result = sub_u64x4(&result, &N);
    }

    result
}

/// Simple 512-bit to 256-bit reduction modulo n
fn reduce_512_mod_n_simple(val: &[u64; 8]) -> [u64; 4] {
    // 2^256 - n as [u64; 3] (fits in 129 bits)
    // = 0x014551231950B75FC4402DA1732FC9BEBF
    let k: [u64; 3] = [0x402DA1732FC9BEBF, 0x4551231950B75FC4, 0x0000000000000001];

    // Extract low and high 256-bit parts
    let low = [val[0], val[1], val[2], val[3]];
    let high = [val[4], val[5], val[6], val[7]];

    // Compute high * k (256-bit * 129-bit = up to 385 bits)
    // We'll do this carefully
    let mut product = [0u64; 8];
    for i in 0..4 {
        if high[i] == 0 {
            continue;
        }
        let mut carry = 0u128;
        for j in 0..3 {
            let pos = i + j;
            let p = (high[i] as u128) * (k[j] as u128) + (product[pos] as u128) + carry;
            product[pos] = p as u64;
            carry = p >> 64;
        }
        // Propagate remaining carry
        let mut pos = i + 3;
        while carry != 0 && pos < 8 {
            let sum = (product[pos] as u128) + carry;
            product[pos] = sum as u64;
            carry = sum >> 64;
            pos += 1;
        }
    }

    // Add low to product
    let mut carry = 0u128;
    for i in 0..4 {
        let sum = (low[i] as u128) + (product[i] as u128) + carry;
        product[i] = sum as u64;
        carry = sum >> 64;
    }
    // Propagate carry
    for i in 4..8 {
        if carry == 0 {
            break;
        }
        let sum = (product[i] as u128) + carry;
        product[i] = sum as u64;
        carry = sum >> 64;
    }

    // Check if product still has high bits (recursive reduction)
    let new_high = [product[4], product[5], product[6], product[7]];
    if new_high != [0, 0, 0, 0] {
        // Recursively reduce
        return reduce_512_mod_n_simple(&product);
    }

    // Final result
    let mut result = [product[0], product[1], product[2], product[3]];

    // Final reduction: while result >= n, subtract n
    while cmp_u64x4(&result, &N) >= 0 {
        result = sub_u64x4(&result, &N);
    }

    result
}

/// Compare two [u64; 4] values: returns -1, 0, or 1
fn cmp_u64x4(a: &[u64; 4], b: &[u64; 4]) -> i32 {
    for i in (0..4).rev() {
        if a[i] > b[i] {
            return 1;
        }
        if a[i] < b[i] {
            return -1;
        }
    }
    0
}

/// Subtract two [u64; 4] values: a - b (assumes a >= b)
fn sub_u64x4(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut result = [0u64; 4];
    let mut borrow = 0u64;

    for i in 0..4 {
        let (diff, b1) = a[i].overflowing_sub(b[i]);
        let (diff, b2) = diff.overflowing_sub(borrow);
        result[i] = diff;
        borrow = (b1 as u64) + (b2 as u64);
    }

    result
}

// =============================================================================
// Prefix validation
// =============================================================================

/// Validate prefix (only allow valid bech32 characters)
///
/// Valid bech32 characters: 023456789acdefghjklmnpqrstuvwxyz (32 characters)
/// Invalid characters: 1, b, i, o (excluded to avoid confusion)
///
/// # Returns
/// - Ok(()) : prefix is valid
/// - Err(String) : error message
pub fn validate_prefix(prefix: &str) -> Result<(), String> {
    // Use BECH32_CHARSET for validation (same 32 characters, different order)
    // BECH32_CHARSET order matters for encoding, but not for contains() check

    // Empty string check
    if prefix.is_empty() {
        return Err("Prefix cannot be empty".to_string());
    }

    // Check each character
    for (i, ch) in prefix.chars().enumerate() {
        // Check for uppercase
        if ch.is_uppercase() {
            return Err(format!(
                "Invalid prefix '{}': bech32 does not allow uppercase letters (found '{}' at position {})\n\
                 Hint: Use lowercase instead",
                prefix, ch, i
            ));
        }

        // Check for invalid bech32 characters
        if !BECH32_CHARSET.contains(ch) {
            // Add detailed explanation for commonly confused characters
            let hint = match ch {
                '1' => "Character '1' is not allowed (reserved as separator in bech32)",
                'b' | 'i' | 'o' => {
                    "Character is excluded to avoid confusion with similar-looking characters"
                }
                _ => "Character is not in the bech32 character set",
            };

            return Err(format!(
                "Invalid prefix '{}': bech32 does not allow '{}'\n\
                 {}\n\
                 Valid characters: {}",
                prefix, ch, hint, BECH32_CHARSET
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use secp256k1::SecretKey;

    #[test]
    fn test_validate_prefix_valid() {
        // Test valid prefixes
        assert!(validate_prefix("test").is_ok());
        assert!(validate_prefix("0").is_ok());
        assert!(validate_prefix("00").is_ok());
        assert!(validate_prefix("ac").is_ok());
        assert!(validate_prefix("m0ctane").is_ok());
    }

    #[test]
    fn test_validate_prefix_invalid_chars() {
        // Prefix containing invalid characters (1, b, i, o)
        assert!(validate_prefix("abc").is_err()); // 'b' is invalid
        assert!(validate_prefix("test1").is_err()); // '1' is invalid
        assert!(validate_prefix("testi").is_err()); // 'i' is invalid
        assert!(validate_prefix("testo").is_err()); // 'o' is invalid
    }

    #[test]
    fn test_validate_prefix_uppercase() {
        // Prefix containing uppercase
        assert!(validate_prefix("Test").is_err());
        assert!(validate_prefix("TEST").is_err());
        assert!(validate_prefix("TeSt").is_err());
    }

    #[test]
    fn test_validate_prefix_empty() {
        // Empty string
        assert!(validate_prefix("").is_err());
    }

    #[test]
    fn test_seckey_to_nsec() {
        // Test secret key (hex)
        let sk_hex = "3bf0c63fcb93463407af97a5e5ee64fa883d107ef9e558472c4eb9aaaefa459d";
        let sk = SecretKey::from_slice(&hex::decode(sk_hex).unwrap()).unwrap();
        let nsec = seckey_to_nsec(&sk);

        // Correct nsec (generated from implementation)
        assert_eq!(
            nsec,
            "nsec180cvv07tjdrrgpa0j7j7tmnyl2yr6yr7l8j4s3evf6u64th6gkwsgyumg0"
        );

        // Verify nsec format is correct
        assert!(nsec.starts_with("nsec1"));
        assert_eq!(nsec.len(), 63); // nsec1 + 58 chars
    }

    #[test]
    fn test_pubkey_to_npub() {
        // Generate public key from test secret key
        let sk_hex = "3bf0c63fcb93463407af97a5e5ee64fa883d107ef9e558472c4eb9aaaefa459d";
        let sk = SecretKey::from_slice(&hex::decode(sk_hex).unwrap()).unwrap();
        let secp = secp256k1::Secp256k1::new();
        let pk = sk.public_key(&secp);

        let npub = pubkey_to_npub(&pk);

        // Correct npub (generated from implementation)
        assert_eq!(
            npub,
            "npub1wxxh2mmqeaghnme4kwwudkel7k8sfsrnf7qld4zppu9sglwljq5shd0y24"
        );

        // Verify npub format is correct
        assert!(npub.starts_with("npub1"));
        assert_eq!(npub.len(), 63); // npub1 + 58 chars
    }

    #[test]
    fn test_validate_prefix_error_messages() {
        // Verify error message content
        let err = validate_prefix("abc").unwrap_err();
        assert!(err.contains("bech32 does not allow 'b'"));
        assert!(err.contains("excluded to avoid confusion"));

        let err = validate_prefix("test1").unwrap_err();
        assert!(err.contains("bech32 does not allow '1'"));
        assert!(err.contains("reserved as separator"));

        let err = validate_prefix("Test").unwrap_err();
        assert!(err.contains("uppercase letters"));
        assert!(err.contains("Use lowercase instead"));

        let err = validate_prefix("").unwrap_err();
        assert!(err.contains("cannot be empty"));
    }

    #[test]
    fn test_bytes_u64x4_roundtrip() {
        // Roundtrip test: bytes → u64x4 → bytes
        let original_bytes: [u8; 32] = [
            0xC6, 0x04, 0x7F, 0x94, 0x41, 0xED, 0x7D, 0x6D, // byte[0..8]
            0x30, 0x45, 0x40, 0x6E, 0x95, 0xC0, 0x7C, 0xD8, // byte[8..16]
            0x5C, 0x77, 0x8E, 0x4B, 0x8C, 0xEF, 0x3C, 0xA7, // byte[16..24]
            0xAB, 0xAC, 0x09, 0xB9, 0x5C, 0x70, 0x9E, 0xE5, // byte[24..32]
        ];

        let u64x4 = bytes_to_u64x4(&original_bytes);
        let roundtrip_bytes = u64x4_to_bytes(&u64x4);

        assert_eq!(
            original_bytes, roundtrip_bytes,
            "roundtrip should preserve bytes"
        );
    }

    #[test]
    fn test_u64x4_to_bytes_2g() {
        // Test using 2G x-coordinate
        // GPU result: [0xABAC09B95C709EE5, 0x5C778E4B8CEF3CA7, 0x3045406E95C07CD8, 0xC6047F9441ED7D6D]
        // Expected (big-endian bytes): C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5
        let gpu_result: [u64; 4] = [
            0xABAC09B95C709EE5u64,
            0x5C778E4B8CEF3CA7u64,
            0x3045406E95C07CD8u64,
            0xC6047F9441ED7D6Du64,
        ];

        let bytes = u64x4_to_bytes(&gpu_result);
        let hex_str = hex::encode(bytes);

        assert_eq!(
            hex_str, "c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5",
            "2G x-coordinate should match"
        );
    }

    #[test]
    fn test_pubkey_bytes_to_npub_2g() {
        // Convert 2G x-coordinate to npub
        let pubkey_bytes: [u8; 32] = [
            0xC6, 0x04, 0x7F, 0x94, 0x41, 0xED, 0x7D, 0x6D, 0x30, 0x45, 0x40, 0x6E, 0x95, 0xC0,
            0x7C, 0xD8, 0x5C, 0x77, 0x8E, 0x4B, 0x8C, 0xEF, 0x3C, 0xA7, 0xAB, 0xAC, 0x09, 0xB9,
            0x5C, 0x70, 0x9E, 0xE5,
        ];

        let npub = pubkey_bytes_to_npub(&pubkey_bytes);

        // Verify npub format is correct
        assert!(npub.starts_with("npub1"), "npub should start with 'npub1'");
        assert_eq!(npub.len(), 63, "npub should be 63 characters");

        println!("2G npub: {}", npub);
    }

    #[test]
    fn test_prefix_to_bits_single_char() {
        // Single character test: 'q' = 0, 'm' = 27, 'l' = 31
        let (pattern, mask, bit_len) = prefix_to_bits("q");
        assert_eq!(bit_len, 5);
        assert_eq!(pattern, 0b00000_u64 << 59); // 'q' = 0
        assert_eq!(mask, 0b11111_u64 << 59);

        let (pattern, _, _) = prefix_to_bits("m");
        assert_eq!(pattern, 0b11011_u64 << 59); // 'm' = 27

        let (pattern, _, _) = prefix_to_bits("l");
        assert_eq!(pattern, 0b11111_u64 << 59); // 'l' = 31
    }

    #[test]
    fn test_prefix_to_bits_m0() {
        // 'm0' = 27, 15 = 11011_01111
        let (pattern, mask, bit_len) = prefix_to_bits("m0");
        assert_eq!(bit_len, 10);

        // 'm' = 27 (11011), '0' = 15 (01111)
        // Placed in upper 10 bits: 11011_01111_00...0
        let expected_pattern = (0b11011_01111_u64) << 54;
        assert_eq!(pattern, expected_pattern);

        // Mask: upper 10 bits are 1
        let expected_mask = 0b11111_11111_u64 << 54;
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_prefix_to_bits_m0ctane() {
        // 'm0ctane' (7 characters = 35 bits)
        // 'm'=27, '0'=15, 'c'=24, 't'=11, 'a'=29, 'n'=19, 'e'=25
        let (pattern, mask, bit_len) = prefix_to_bits("m0ctane");
        assert_eq!(bit_len, 35);

        // Concatenate 5-bit value of each character
        let m = 27u64; // 11011
        let zero = 15u64; // 01111
        let c = 24u64; // 11000
        let t = 11u64; // 01011
        let a = 29u64; // 11101
        let n = 19u64; // 10011
        let e = 25u64; // 11001

        let expected_pattern =
            (m << 30 | zero << 25 | c << 20 | t << 15 | a << 10 | n << 5 | e) << (64 - 35);
        assert_eq!(pattern, expected_pattern);

        // Mask: upper 35 bits are 1
        let expected_mask = !((1u64 << (64 - 35)) - 1);
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_prefix_to_bits_matches_npub() {
        // Consistency test with actual npub
        // Generate 2G npub and verify prefix matches
        let pubkey_bytes: [u8; 32] = [
            0xC6, 0x04, 0x7F, 0x94, 0x41, 0xED, 0x7D, 0x6D, 0x30, 0x45, 0x40, 0x6E, 0x95, 0xC0,
            0x7C, 0xD8, 0x5C, 0x77, 0x8E, 0x4B, 0x8C, 0xEF, 0x3C, 0xA7, 0xAB, 0xAC, 0x09, 0xB9,
            0x5C, 0x70, 0x9E, 0xE5,
        ];

        let npub = pubkey_bytes_to_npub(&pubkey_bytes);
        let npub_body = &npub[5..]; // Remove "npub1"
        println!("2G npub body: {}", npub_body);

        // Create prefix from first few characters of npub body and test bit matching
        let prefix = &npub_body[..4]; // First 4 characters
        println!("Testing prefix: {}", prefix);

        let (pattern, mask, bit_len) = prefix_to_bits(prefix);
        println!("pattern: {:064b}", pattern);
        println!("mask:    {:064b}", mask);
        println!("bit_len: {}", bit_len);

        // Get upper 64 bits of pubkey_bytes
        let pubkey_upper = u64::from_be_bytes([
            pubkey_bytes[0],
            pubkey_bytes[1],
            pubkey_bytes[2],
            pubkey_bytes[3],
            pubkey_bytes[4],
            pubkey_bytes[5],
            pubkey_bytes[6],
            pubkey_bytes[7],
        ]);
        println!("pubkey upper: {:064b}", pubkey_upper);

        // Should match!
        assert_eq!(pubkey_upper & mask, pattern & mask, "prefix should match");
    }
}

#[cfg(test)]
mod endomorphism_tests {
    use super::*;

    #[test]
    fn test_mod_n_mult_lambda_sq() {
        // Correct value calculated with Python
        // k = 0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
        let k: [u64; 4] = [
            0x1234567890abcdef,
            0x1234567890abcdef,
            0x1234567890abcdef,
            0x1234567890abcdef,
        ];

        // Correct λ²*k mod n = 0x30f00e02e8cdf3ecd8166c5214a47c18e7402da72d337fed8281d3ae181c72ae
        let expected: [u64; 4] = [
            0x8281d3ae181c72ae,
            0xe7402da72d337fed,
            0xd8166c5214a47c18,
            0x30f00e02e8cdf3ec,
        ];

        let result = mod_n_mult(&k, &LAMBDA_SQ);

        println!("k = {:016x}{:016x}{:016x}{:016x}", k[3], k[2], k[1], k[0]);
        println!(
            "Expected λ²*k = {:016x}{:016x}{:016x}{:016x}",
            expected[3], expected[2], expected[1], expected[0]
        );
        println!(
            "Actual λ²*k   = {:016x}{:016x}{:016x}{:016x}",
            result[3], result[2], result[1], result[0]
        );

        assert_eq!(
            result, expected,
            "mod_n_mult(k, LAMBDA_SQ) should match Python calculation"
        );
    }
}
