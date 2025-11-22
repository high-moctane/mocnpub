use clap::Parser;
use secp256k1::rand;
use secp256k1::{PublicKey, Secp256k1, SecretKey};
use bech32::{encode, Bech32, Hrp};
use hex;
use std::fs::File;
use std::io::{self, Write};
use std::time::Instant;

/// Nostr npub マイニングツール 🔑
///
/// 指定した prefix を持つ npub（Nostr 公開鍵）を見つけるマイニングツール。
/// CPU 版の実装で、GPU 版は Step 3 で実装予定。
#[derive(Parser, Debug)]
#[command(name = "mocnpub")]
#[command(about = "Nostr npub マイニングツール 🔑", long_about = None)]
struct Args {
    /// マイニングする prefix（npub1 に続く bech32 文字列）
    ///
    /// 例: "abc", "test", "satoshi"
    /// 完全な npub 例: npub1abc... の "abc" 部分を指定
    #[arg(short, long)]
    prefix: String,

    /// 結果を出力するファイル（オプション、デフォルトは stdout）
    #[arg(short, long)]
    output: Option<String>,
}

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

fn main() -> io::Result<()> {
    let args = Args::parse();

    println!("🔥 mocnpub - Nostr npub マイニング 🔥");
    println!("Prefix: '{}'\n", args.prefix);

    let secp = Secp256k1::new();
    let mut count = 0;
    let start = Instant::now();

    loop {
        let (sk, pk) = secp.generate_keypair(&mut rand::thread_rng());
        count += 1;

        // bech32 形式に変換
        let npub = pubkey_to_npub(&pk);
        // "npub1" を除去して、bech32 文字列の部分だけを取り出す
        let npub_body = &npub[5..]; // "npub1" は5文字

        // prefix マッチング判定（npub の bech32 部分で比較）
        if npub_body.starts_with(&args.prefix) {
            let elapsed = start.elapsed();
            let elapsed_secs = elapsed.as_secs_f64();
            let keys_per_sec = count as f64 / elapsed_secs;

            let nsec = seckey_to_nsec(&sk);
            let pk_hex = pk.to_string();
            let pk_x_only = &pk_hex[2..]; // x座標のみ（圧縮形式の先頭2文字を除去）

            // 結果を整形
            let output_text = format!(
                "✅ 見つかりました！（{}回試行）\n\n\
                 経過時間: {:.2}秒\n\
                 パフォーマンス: {:.2} keys/sec\n\n\
                 秘密鍵（hex）: {}\n\
                 秘密鍵（nsec）: {}\n\
                 公開鍵（圧縮形式）: {}\n\
                 公開鍵（x座標のみ）: {}\n\
                 公開鍵（npub）: {}\n",
                count,
                elapsed_secs,
                keys_per_sec,
                sk.display_secret(),
                nsec,
                pk,
                pk_x_only,
                npub
            );

            // 出力先に応じて出力
            if let Some(output_file) = &args.output {
                // ファイルに出力
                let mut file = File::create(output_file)?;
                file.write_all(output_text.as_bytes())?;
                println!("{}", output_text);
                println!("結果をファイルに保存しました: {}", output_file);
            } else {
                // stdout に出力
                println!("{}", output_text);
            }

            break;
        }

        // 進捗表示（100回ごと）
        if count % 100 == 0 {
            println!("{}回試行中...", count);
        }
    }

    Ok(())
}
