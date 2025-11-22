use clap::Parser;
use secp256k1::rand;
use secp256k1::{PublicKey, Secp256k1, SecretKey};
use bech32::{encode, Bech32, Hrp};
use hex;
use std::fs::File;
use std::io::{self, Write};
use std::time::Instant;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;

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

    /// スレッド数（デフォルト: CPU コア数を自動検出）
    #[arg(short, long)]
    threads: Option<usize>,
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

/// prefix の妥当性を検証（bech32 の有効文字のみを許可）
///
/// bech32 で使用可能な文字: 023456789acdefghjklmnpqrstuvwxyz (32文字)
/// 使用不可な文字: 1, b, i, o（混同を避けるため除外されている）
///
/// # Returns
/// - Ok(()) : prefix が有効
/// - Err(String) : エラーメッセージ
fn validate_prefix(prefix: &str) -> Result<(), String> {
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

fn main() -> io::Result<()> {
    let args = Args::parse();

    // prefix の妥当性を検証
    if let Err(err_msg) = validate_prefix(&args.prefix) {
        eprintln!("❌ Error: {}", err_msg);
        std::process::exit(1);
    }

    // スレッド数を決定（引数指定 or CPU コア数）
    let num_threads = args.threads.unwrap_or_else(num_cpus::get);

    println!("🔥 mocnpub - Nostr npub マイニング 🔥");
    println!("Prefix: '{}'", args.prefix);
    println!("Threads: {}\n", num_threads);

    // 全スレッド共有のカウンタとフラグ
    let total_count = Arc::new(AtomicU64::new(0));
    let found = Arc::new(AtomicBool::new(false));
    let start = Instant::now();

    // 結果を保存する（Option<(SecretKey, PublicKey, String)>）
    let result: Arc<std::sync::Mutex<Option<(SecretKey, PublicKey, String)>>> = Arc::new(std::sync::Mutex::new(None));

    // スレッドを起動
    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let prefix = args.prefix.clone();
            let total_count = Arc::clone(&total_count);
            let found = Arc::clone(&found);
            let result = Arc::clone(&result);

            std::thread::spawn(move || {
                let secp = Secp256k1::new();
                let mut local_count = 0u64;

                loop {
                    // 他のスレッドが見つけたらループを抜ける
                    if found.load(Ordering::Relaxed) {
                        break;
                    }

                    let (sk, pk) = secp.generate_keypair(&mut rand::thread_rng());
                    local_count += 1;

                    // bech32 形式に変換
                    let npub = pubkey_to_npub(&pk);
                    // "npub1" を除去して、bech32 文字列の部分だけを取り出す
                    let npub_body = &npub[5..]; // "npub1" は5文字

                    // prefix マッチング判定（npub の bech32 部分で比較）
                    if npub_body.starts_with(&prefix) {
                        // 見つかったことを通知
                        found.store(true, Ordering::Relaxed);

                        // 結果を保存
                        let mut result_lock = result.lock().unwrap();
                        *result_lock = Some((sk, pk, npub));
                        break;
                    }

                    // 定期的に全体カウンタを更新（100回ごと）
                    if local_count % 100 == 0 {
                        total_count.fetch_add(100, Ordering::Relaxed);
                    }
                }

                // 最後に残りのカウントを加算
                let remainder = local_count % 100;
                if remainder > 0 {
                    total_count.fetch_add(remainder, Ordering::Relaxed);
                }
            })
        })
        .collect();

    // 進捗表示スレッド
    let total_count_progress = Arc::clone(&total_count);
    let found_progress = Arc::clone(&found);
    let progress_handle = std::thread::spawn(move || {
        loop {
            if found_progress.load(Ordering::Relaxed) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_secs(1));
            let count = total_count_progress.load(Ordering::Relaxed);
            if count > 0 {
                println!("{}回試行中...", count);
            }
        }
    });

    // 全スレッドの終了を待つ
    for handle in handles {
        handle.join().unwrap();
    }
    progress_handle.join().unwrap();

    // 結果を取得
    let result_lock = result.lock().unwrap();
    if let Some((sk, pk, npub)) = &*result_lock {
        let elapsed = start.elapsed();
        let elapsed_secs = elapsed.as_secs_f64();
        let count = total_count.load(Ordering::Relaxed);
        let keys_per_sec = count as f64 / elapsed_secs;

        let nsec = seckey_to_nsec(&sk);
        let pk_hex = pk.to_string();
        let pk_x_only = &pk_hex[2..]; // x座標のみ（圧縮形式の先頭2文字を除去）

        // 結果を整形
        let output_text = format!(
            "✅ 見つかりました！（{}回試行、{}スレッド）\n\n\
             経過時間: {:.2}秒\n\
             パフォーマンス: {:.2} keys/sec\n\n\
             秘密鍵（hex）: {}\n\
             秘密鍵（nsec）: {}\n\
             公開鍵（圧縮形式）: {}\n\
             公開鍵（x座標のみ）: {}\n\
             公開鍵（npub）: {}\n",
            count,
            num_threads,
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
    } else {
        println!("見つかりませんでした（予期しないエラー）");
    }

    Ok(())
}
