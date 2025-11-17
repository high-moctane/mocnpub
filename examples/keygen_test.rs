use secp256k1::rand;
use secp256k1::{Secp256k1, PublicKey, SecretKey};
use bech32::{Hrp, encode, Bech32};
use hex;

/// 公開鍵（x座標のみ32バイト）を npub に変換
fn pubkey_to_npub(pubkey: &PublicKey) -> String {
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
fn seckey_to_nsec(seckey: &SecretKey) -> String {
    // 秘密鍵のバイト列を取得
    let bytes = seckey.secret_bytes();

    // bech32 エンコード
    let hrp = Hrp::parse("nsec").expect("valid hrp");
    encode::<Bech32>(hrp, &bytes).expect("failed to encode nsec")
}

fn main() {
    println!("🔑 Nostr 鍵生成テスト 🔑\n");

    let secp = Secp256k1::new();

    // 1つ目の鍵ペア生成
    let (secret_key, public_key) = secp.generate_keypair(&mut rand::thread_rng());

    let pk_hex = public_key.to_string();
    let pk_x_only = &pk_hex[2..]; // x座標のみ（圧縮形式の先頭2文字を除去）

    // bech32 形式に変換
    let npub = pubkey_to_npub(&public_key);
    let nsec = seckey_to_nsec(&secret_key);

    println!("秘密鍵（hex）: {}", secret_key.display_secret());
    println!("秘密鍵（nsec）: {}", nsec);
    println!("公開鍵（圧縮形式）: {}", public_key);
    println!("公開鍵（x座標のみ）: {}", pk_x_only);
    println!("公開鍵（npub）: {}", npub);

    // 簡単なマイニングのデモ（prefix マッチング）
    println!("\n🔥 prefix マイニングのデモ（prefix: '00'） 🔥\n");

    let mut count = 0;
    loop {
        let (sk, pk) = secp.generate_keypair(&mut rand::thread_rng());
        count += 1;

        let pk_hex = pk.to_string();
        let pk_x_only = &pk_hex[2..]; // x座標のみ（圧縮形式の先頭2文字を除去）

        if pk_x_only.starts_with("00") {
            // bech32 形式に変換
            let npub = pubkey_to_npub(&pk);
            let nsec = seckey_to_nsec(&sk);

            println!("✅ 見つかりました！（{}回試行）", count);
            println!("秘密鍵（hex）: {}", sk.display_secret());
            println!("秘密鍵（nsec）: {}", nsec);
            println!("公開鍵（圧縮形式）: {}", pk);
            println!("公開鍵（x座標のみ）: {}", pk_x_only);
            println!("公開鍵（npub）: {}", npub);
            break;
        }

        if count % 100 == 0 {
            println!("{}回試行中...", count);
        }
    }
}
