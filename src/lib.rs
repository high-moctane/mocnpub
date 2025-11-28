use secp256k1::{PublicKey, SecretKey};
use bech32::{encode, Bech32, Hrp};
use hex;

// GPU module
pub mod gpu;

// =============================================================================
// バイト列 ↔ [u64; 4] 変換関数（GPU との連携用）
// =============================================================================

/// バイト列（32バイト、big-endian）を [u64; 4]（little-endian limbs）に変換
///
/// 秘密鍵を GPU に渡すときに使用
/// - 入力: big-endian のバイト列（byte[0] が最上位バイト）
/// - 出力: little-endian limbs（limb[0] が最下位 64 bit）
pub fn bytes_to_u64x4(bytes: &[u8; 32]) -> [u64; 4] {
    let mut result = [0u64; 4];
    // byte[24..32] → limb[0]（最下位）
    // byte[16..24] → limb[1]
    // byte[8..16]  → limb[2]
    // byte[0..8]   → limb[3]（最上位）
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

/// [u64; 4]（little-endian limbs）をバイト列（32バイト、big-endian）に変換
///
/// GPU から返ってきた公開鍵を npub に変換するときに使用
/// - 入力: little-endian limbs（limb[0] が最下位 64 bit）
/// - 出力: big-endian のバイト列（byte[0] が最上位バイト）
pub fn u64x4_to_bytes(value: &[u64; 4]) -> [u8; 32] {
    let mut result = [0u8; 32];
    // limb[3]（最上位）→ byte[0..8]
    // limb[2]          → byte[8..16]
    // limb[1]          → byte[16..24]
    // limb[0]（最下位）→ byte[24..32]
    for i in 0..4 {
        let offset = (3 - i) * 8; // reverse order
        let bytes = value[i].to_be_bytes();
        result[offset..offset + 8].copy_from_slice(&bytes);
    }
    result
}

/// 公開鍵のバイト列（x座標のみ、32バイト）を npub に変換
///
/// GPU から返ってきた公開鍵を直接 npub に変換するために使用
/// 既存の `pubkey_to_npub` は secp256k1::PublicKey を経由するが、
/// この関数はバイト列から直接変換する
pub fn pubkey_bytes_to_npub(pubkey_bytes: &[u8; 32]) -> String {
    let hrp = Hrp::parse("npub").expect("valid hrp");
    encode::<Bech32>(hrp, pubkey_bytes).expect("failed to encode npub")
}

/// 公開鍵（x座標のみ32バイト）を npub に変換
pub fn pubkey_to_npub(pubkey: &PublicKey) -> String {
    // 公開鍵の hex 文字列を取得（圧縮形式）
    let pk_hex = pubkey.to_string();
    // x座標のみを抽出（先頭2文字を除去）
    let pk_x_only = &pk_hex[2..];

    // hex 文字列を 32 バイトのバイト列に変換
    let mut bytes = [0u8; 32];
    hex::decode_to_slice(pk_x_only, &mut bytes).expect("Invalid hex string");

    // bech32 エンコード
    let hrp = Hrp::parse("npub").expect("valid hrp");
    encode::<Bech32>(hrp, &bytes).expect("failed to encode npub")
}

/// 秘密鍵（32バイト）を nsec に変換
pub fn seckey_to_nsec(seckey: &SecretKey) -> String {
    // 秘密鍵のバイト列を取得
    let bytes = seckey.secret_bytes();

    // bech32 エンコード
    let hrp = Hrp::parse("nsec").expect("valid hrp");
    encode::<Bech32>(hrp, &bytes).expect("failed to encode nsec")
}

// =============================================================================
// Prefix → Bit列変換（GPU での高速マッチング用）
// =============================================================================

/// bech32 の文字セット（順番が重要！各文字の位置が 5 bit 値に対応）
const BECH32_CHARSET: &str = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";

/// prefix を bit 列に変換（GPU での高速マッチング用）
///
/// bech32 の各文字を 5 bit 値に変換し、連結して u64 の上位ビットに配置
///
/// # Arguments
/// * `prefix` - 変換する prefix（最大 12 文字 = 60 bit）
///
/// # Returns
/// * `(pattern, mask, bit_len)` - パターン、マスク、ビット長
///   - pattern: prefix を 5 bit ずつ連結した値（u64 の上位ビットに配置）
///   - mask: 有効ビットが 1 のマスク（上位 bit_len ビットが 1）
///   - bit_len: 有効ビット数（prefix_len * 5）
///
/// # Example
/// ```
/// let (pattern, mask, bit_len) = prefix_to_bits("m0");
/// // 'm' = 27 (11011), '0' = 15 (01111)
/// // pattern = 0b11011_01111_000...0 (上位 10 bit)
/// // mask    = 0b11111_11111_000...0 (上位 10 bit が 1)
/// // bit_len = 10
/// ```
pub fn prefix_to_bits(prefix: &str) -> (u64, u64, u32) {
    let mut pattern: u64 = 0;
    let mut bit_pos: u32 = 64;  // 上位から配置

    for ch in prefix.chars() {
        // bech32 文字セットでの位置を取得（0-31）
        let value = BECH32_CHARSET.find(ch).expect("invalid bech32 char") as u64;

        // 5 bit 分シフトして配置
        bit_pos -= 5;
        pattern |= value << bit_pos;
    }

    let bit_len = (prefix.len() as u32) * 5;
    let mask = if bit_len >= 64 {
        u64::MAX
    } else {
        !((1u64 << (64 - bit_len)) - 1)  // 上位 bit_len ビットが 1
    };

    (pattern, mask, bit_len)
}

/// 複数の prefix を bit 列に変換
///
/// # Returns
/// * `Vec<(pattern, mask, bit_len)>` - 各 prefix のパターン、マスク、ビット長
pub fn prefixes_to_bits(prefixes: &[String]) -> Vec<(u64, u64, u32)> {
    prefixes.iter().map(|p| prefix_to_bits(p)).collect()
}

// =============================================================================
// 256-bit 演算（連続秘密鍵戦略用）
// =============================================================================

/// 256-bit 値（[u64; 4]）に offset を加算
///
/// 連続秘密鍵戦略で、base_key + offset を計算するために使用。
/// offset は u32 なので、最下位 limb への加算と carry 伝播のみ。
///
/// # Arguments
/// * `base` - 256-bit 値（little-endian limbs: base[0] が最下位）
/// * `offset` - 加算する値（最大 u32::MAX）
///
/// # Returns
/// * `[u64; 4]` - base + offset の結果
///
/// # Example
/// ```
/// use mocnpub_main::add_u64x4_scalar;
/// let base = [0xFFFFFFFF_FFFFFFFFu64, 0, 0, 0];
/// let result = add_u64x4_scalar(&base, 1);
/// assert_eq!(result, [0, 1, 0, 0]); // carry が発生
/// ```
pub fn add_u64x4_scalar(base: &[u64; 4], offset: u32) -> [u64; 4] {
    let mut result = *base;

    // offset を最下位 limb に加算
    let (sum, carry) = result[0].overflowing_add(offset as u64);
    result[0] = sum;

    // carry を伝播
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
// Prefix 検証
// =============================================================================

/// prefix の妥当性を検証（bech32 の有効文字のみを許可）
///
/// bech32 で使用可能な文字: 023456789acdefghjklmnpqrstuvwxyz (32文字)
/// 使用不可な文字: 1, b, i, o（混同を避けるため除外されている）
///
/// # Returns
/// - Ok(()) : prefix が有効
/// - Err(String) : エラーメッセージ
pub fn validate_prefix(prefix: &str) -> Result<(), String> {
    // bech32 の有効な文字セット（32文字）
    const VALID_CHARS: &str = "023456789acdefghjklmnpqrstuvwxyz";

    // 空文字チェック
    if prefix.is_empty() {
        return Err("Prefix cannot be empty".to_string());
    }

    // 各文字をチェック
    for (i, ch) in prefix.chars().enumerate() {
        // 大文字をチェック
        if ch.is_uppercase() {
            return Err(format!(
                "Invalid prefix '{}': bech32 does not allow uppercase letters (found '{}' at position {})\n\
                 Hint: Use lowercase instead",
                prefix, ch, i
            ));
        }

        // bech32 で無効な文字をチェック
        if !VALID_CHARS.contains(ch) {
            // 特に混同しやすい文字には詳しい説明を追加
            let hint = match ch {
                '1' => "Character '1' is not allowed (reserved as separator in bech32)",
                'b' | 'i' | 'o' => "Character is excluded to avoid confusion with similar-looking characters",
                _ => "Character is not in the bech32 character set",
            };

            return Err(format!(
                "Invalid prefix '{}': bech32 does not allow '{}'\n\
                 {}\n\
                 Valid characters: {}",
                prefix, ch, hint, VALID_CHARS
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
        // 有効な prefix のテスト
        assert!(validate_prefix("test").is_ok());
        assert!(validate_prefix("0").is_ok());
        assert!(validate_prefix("00").is_ok());
        assert!(validate_prefix("ac").is_ok());
        assert!(validate_prefix("m0ctane").is_ok());
    }

    #[test]
    fn test_validate_prefix_invalid_chars() {
        // 無効な文字（1, b, i, o）を含む prefix
        assert!(validate_prefix("abc").is_err()); // 'b' が無効
        assert!(validate_prefix("test1").is_err()); // '1' が無効
        assert!(validate_prefix("testi").is_err()); // 'i' が無効
        assert!(validate_prefix("testo").is_err()); // 'o' が無効
    }

    #[test]
    fn test_validate_prefix_uppercase() {
        // 大文字を含む prefix
        assert!(validate_prefix("Test").is_err());
        assert!(validate_prefix("TEST").is_err());
        assert!(validate_prefix("TeSt").is_err());
    }

    #[test]
    fn test_validate_prefix_empty() {
        // 空文字
        assert!(validate_prefix("").is_err());
    }

    #[test]
    fn test_seckey_to_nsec() {
        // テスト用の秘密鍵（hex）
        let sk_hex = "3bf0c63fcb93463407af97a5e5ee64fa883d107ef9e558472c4eb9aaaefa459d";
        let sk = SecretKey::from_slice(&hex::decode(sk_hex).unwrap()).unwrap();
        let nsec = seckey_to_nsec(&sk);

        // 正しい nsec（実装から生成された値）
        assert_eq!(nsec, "nsec180cvv07tjdrrgpa0j7j7tmnyl2yr6yr7l8j4s3evf6u64th6gkwsgyumg0");

        // nsec の形式が正しいことを確認
        assert!(nsec.starts_with("nsec1"));
        assert_eq!(nsec.len(), 63); // nsec1 + 58文字
    }

    #[test]
    fn test_pubkey_to_npub() {
        // テスト用の秘密鍵から公開鍵を生成
        let sk_hex = "3bf0c63fcb93463407af97a5e5ee64fa883d107ef9e558472c4eb9aaaefa459d";
        let sk = SecretKey::from_slice(&hex::decode(sk_hex).unwrap()).unwrap();
        let secp = secp256k1::Secp256k1::new();
        let pk = sk.public_key(&secp);

        let npub = pubkey_to_npub(&pk);

        // 正しい npub（実装から生成された値）
        assert_eq!(npub, "npub1wxxh2mmqeaghnme4kwwudkel7k8sfsrnf7qld4zppu9sglwljq5shd0y24");

        // npub の形式が正しいことを確認
        assert!(npub.starts_with("npub1"));
        assert_eq!(npub.len(), 63); // npub1 + 58文字
    }

    #[test]
    fn test_validate_prefix_error_messages() {
        // エラーメッセージの内容を確認
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
        // ラウンドトリップテスト：bytes → u64x4 → bytes
        let original_bytes: [u8; 32] = [
            0xC6, 0x04, 0x7F, 0x94, 0x41, 0xED, 0x7D, 0x6D,  // byte[0..8]
            0x30, 0x45, 0x40, 0x6E, 0x95, 0xC0, 0x7C, 0xD8,  // byte[8..16]
            0x5C, 0x77, 0x8E, 0x4B, 0x8C, 0xEF, 0x3C, 0xA7,  // byte[16..24]
            0xAB, 0xAC, 0x09, 0xB9, 0x5C, 0x70, 0x9E, 0xE5,  // byte[24..32]
        ];

        let u64x4 = bytes_to_u64x4(&original_bytes);
        let roundtrip_bytes = u64x4_to_bytes(&u64x4);

        assert_eq!(original_bytes, roundtrip_bytes, "roundtrip should preserve bytes");
    }

    #[test]
    fn test_u64x4_to_bytes_2g() {
        // 2G の x 座標を使ったテスト
        // GPU の結果: [0xABAC09B95C709EE5, 0x5C778E4B8CEF3CA7, 0x3045406E95C07CD8, 0xC6047F9441ED7D6D]
        // 期待値 (big-endian bytes): C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5
        let gpu_result: [u64; 4] = [
            0xABAC09B95C709EE5u64,
            0x5C778E4B8CEF3CA7u64,
            0x3045406E95C07CD8u64,
            0xC6047F9441ED7D6Du64,
        ];

        let bytes = u64x4_to_bytes(&gpu_result);
        let hex_str = hex::encode(&bytes);

        assert_eq!(
            hex_str,
            "c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5",
            "2G x-coordinate should match"
        );
    }

    #[test]
    fn test_pubkey_bytes_to_npub_2g() {
        // 2G の x 座標を npub に変換
        let pubkey_bytes: [u8; 32] = [
            0xC6, 0x04, 0x7F, 0x94, 0x41, 0xED, 0x7D, 0x6D,
            0x30, 0x45, 0x40, 0x6E, 0x95, 0xC0, 0x7C, 0xD8,
            0x5C, 0x77, 0x8E, 0x4B, 0x8C, 0xEF, 0x3C, 0xA7,
            0xAB, 0xAC, 0x09, 0xB9, 0x5C, 0x70, 0x9E, 0xE5,
        ];

        let npub = pubkey_bytes_to_npub(&pubkey_bytes);

        // npub の形式が正しいことを確認
        assert!(npub.starts_with("npub1"), "npub should start with 'npub1'");
        assert_eq!(npub.len(), 63, "npub should be 63 characters");

        println!("2G npub: {}", npub);
    }

    #[test]
    fn test_prefix_to_bits_single_char() {
        // 1文字のテスト: 'q' = 0, 'm' = 27, 'l' = 31
        let (pattern, mask, bit_len) = prefix_to_bits("q");
        assert_eq!(bit_len, 5);
        assert_eq!(pattern, 0b00000_u64 << 59);  // 'q' = 0
        assert_eq!(mask, 0b11111_u64 << 59);

        let (pattern, _, _) = prefix_to_bits("m");
        assert_eq!(pattern, 0b11011_u64 << 59);  // 'm' = 27

        let (pattern, _, _) = prefix_to_bits("l");
        assert_eq!(pattern, 0b11111_u64 << 59);  // 'l' = 31
    }

    #[test]
    fn test_prefix_to_bits_m0() {
        // 'm0' = 27, 15 = 11011_01111
        let (pattern, mask, bit_len) = prefix_to_bits("m0");
        assert_eq!(bit_len, 10);

        // 'm' = 27 (11011), '0' = 15 (01111)
        // 上位 10 bit に配置: 11011_01111_00...0
        let expected_pattern = (0b11011_01111_u64) << 54;
        assert_eq!(pattern, expected_pattern);

        // マスク: 上位 10 bit が 1
        let expected_mask = 0b11111_11111_u64 << 54;
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_prefix_to_bits_m0ctane() {
        // 'm0ctane' (7文字 = 35 bit)
        // 'm'=27, '0'=15, 'c'=24, 't'=11, 'a'=29, 'n'=19, 'e'=25
        let (pattern, mask, bit_len) = prefix_to_bits("m0ctane");
        assert_eq!(bit_len, 35);

        // 各文字の 5 bit 値を連結
        let m = 27u64;  // 11011
        let zero = 15u64;  // 01111
        let c = 24u64;  // 11000
        let t = 11u64;  // 01011
        let a = 29u64;  // 11101
        let n = 19u64;  // 10011
        let e = 25u64;  // 11001

        let expected_pattern = (m << 30 | zero << 25 | c << 20 | t << 15 | a << 10 | n << 5 | e) << (64 - 35);
        assert_eq!(pattern, expected_pattern);

        // マスク: 上位 35 bit が 1
        let expected_mask = !((1u64 << (64 - 35)) - 1);
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_prefix_to_bits_matches_npub() {
        // 実際の npub との整合性テスト
        // 2G の npub を生成して、prefix がマッチすることを確認
        let pubkey_bytes: [u8; 32] = [
            0xC6, 0x04, 0x7F, 0x94, 0x41, 0xED, 0x7D, 0x6D,
            0x30, 0x45, 0x40, 0x6E, 0x95, 0xC0, 0x7C, 0xD8,
            0x5C, 0x77, 0x8E, 0x4B, 0x8C, 0xEF, 0x3C, 0xA7,
            0xAB, 0xAC, 0x09, 0xB9, 0x5C, 0x70, 0x9E, 0xE5,
        ];

        let npub = pubkey_bytes_to_npub(&pubkey_bytes);
        let npub_body = &npub[5..];  // "npub1" を除去
        println!("2G npub body: {}", npub_body);

        // npub body の最初の数文字で prefix を作って、ビットマッチングをテスト
        let prefix = &npub_body[..4];  // 最初の 4 文字
        println!("Testing prefix: {}", prefix);

        let (pattern, mask, bit_len) = prefix_to_bits(prefix);
        println!("pattern: {:064b}", pattern);
        println!("mask:    {:064b}", mask);
        println!("bit_len: {}", bit_len);

        // pubkey_bytes の上位 64 bit を取得
        let pubkey_upper = u64::from_be_bytes([
            pubkey_bytes[0], pubkey_bytes[1], pubkey_bytes[2], pubkey_bytes[3],
            pubkey_bytes[4], pubkey_bytes[5], pubkey_bytes[6], pubkey_bytes[7],
        ]);
        println!("pubkey upper: {:064b}", pubkey_upper);

        // マッチするはず！
        assert_eq!(pubkey_upper & mask, pattern & mask, "prefix should match");
    }
}
