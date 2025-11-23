use secp256k1::{PublicKey, SecretKey};
use bech32::{encode, Bech32, Hrp};
use hex;

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
}
