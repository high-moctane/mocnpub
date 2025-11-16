use secp256k1::rand;
use secp256k1::Secp256k1;

fn main() {
    println!("🔑 Nostr 鍵生成テスト 🔑\n");

    let secp = Secp256k1::new();

    // 1つ目の鍵ペア生成
    let (secret_key, public_key) = secp.generate_keypair(&mut rand::thread_rng());

    let pk_hex = public_key.to_string();
    let pk_x_only = &pk_hex[2..]; // x座標のみ（圧縮形式の先頭2文字を除去）

    println!("秘密鍵（hex）: {}", secret_key.display_secret());
    println!("公開鍵（圧縮形式）: {}", public_key);
    println!("公開鍵（x座標のみ）: {}", pk_x_only);

    // 簡単なマイニングのデモ（prefix マッチング）
    println!("\n🔥 prefix マイニングのデモ（prefix: '00'） 🔥\n");

    let mut count = 0;
    loop {
        let (sk, pk) = secp.generate_keypair(&mut rand::thread_rng());
        count += 1;

        let pk_hex = pk.to_string();
        let pk_x_only = &pk_hex[2..]; // x座標のみ（圧縮形式の先頭2文字を除去）

        if pk_x_only.starts_with("00") {
            println!("✅ 見つかりました！（{}回試行）", count);
            println!("秘密鍵（hex）: {}", sk.display_secret());
            println!("公開鍵（圧縮形式）: {}", pk);
            println!("公開鍵（x座標のみ）: {}", pk_x_only);
            break;
        }

        if count % 100 == 0 {
            println!("{}回試行中...", count);
        }
    }
}
