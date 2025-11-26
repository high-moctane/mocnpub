use clap::Parser;
use secp256k1::rand::{self, RngCore};
use secp256k1::{PublicKey, Secp256k1, SecretKey};
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::time::Instant;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};

// lib.rs から共通関数を import
use mocnpub_main::{pubkey_to_npub, seckey_to_nsec, validate_prefix};
use mocnpub_main::{bytes_to_u64x4, u64x4_to_bytes, pubkey_bytes_to_npub};
use mocnpub_main::gpu::{init_gpu, generate_pubkeys_batch};

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
    /// 単一 prefix: "abc", "test", "satoshi"
    /// 複数 prefix（OR 指定）: "m0ctane0,m0ctane2,m0ctane3"（カンマ区切り）
    /// 完全な npub 例: npub1abc... の "abc" 部分を指定
    #[arg(short, long)]
    prefix: String,

    /// 結果を出力するファイル（オプション、デフォルトは stdout）
    #[arg(short, long)]
    output: Option<String>,

    /// スレッド数（デフォルト: CPU コア数を自動検出）
    #[arg(short, long)]
    threads: Option<usize>,

    /// 見つける鍵の個数（0 = 無限、デフォルト: 1）
    #[arg(short, long, default_value = "1")]
    limit: usize,

    /// GPU モードを有効化（CUDA を使用して高速マイニング）
    #[arg(long)]
    gpu: bool,

    /// GPU バッチサイズ（デフォルト: 65536）
    #[arg(long, default_value = "65536")]
    batch_size: usize,
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    // prefix をカンマ区切りで split して Vec に変換
    let prefixes: Vec<String> = args.prefix
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    // 各 prefix の妥当性を検証
    for prefix in &prefixes {
        if let Err(err_msg) = validate_prefix(prefix) {
            eprintln!("❌ Error: {}", err_msg);
            std::process::exit(1);
        }
    }

    // スレッド数を決定（引数指定 or CPU コア数）
    let num_threads = args.threads.unwrap_or_else(num_cpus::get);

    println!("🔥 mocnpub - Nostr npub マイニング 🔥");
    if prefixes.len() == 1 {
        println!("Prefix: '{}'", prefixes[0]);
    } else {
        println!("Prefixes (OR): {}", prefixes.join(", "));
    }

    // GPU モードか CPU モードかで分岐
    if args.gpu {
        println!("Mode: GPU (CUDA) 🚀");
        println!("Batch size: {}", args.batch_size);
        println!("Limit: {}\n", if args.limit == 0 { "無限".to_string() } else { args.limit.to_string() });
        return run_gpu_mining(&prefixes, args.limit, args.batch_size, args.output.as_deref());
    }

    println!("Mode: CPU");
    println!("Threads: {}", num_threads);
    println!("Limit: {}\n", if args.limit == 0 { "無限".to_string() } else { args.limit.to_string() });

    // 全スレッド共有のカウンタ
    let total_count = Arc::new(AtomicU64::new(0));
    let found_count = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    // prefixes を Arc で共有
    let prefixes = Arc::new(prefixes);

    // channel を作成（ワーカースレッド → メインスレッド）
    // (SecretKey, PublicKey, npub, matched_prefix, 試行回数)
    let (sender, receiver) = mpsc::channel::<(SecretKey, PublicKey, String, String, u64)>();

    // スレッドを起動
    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let prefixes = Arc::clone(&prefixes);
            let total_count = Arc::clone(&total_count);
            let found_count = Arc::clone(&found_count);
            let sender = sender.clone();
            let limit = args.limit;

            std::thread::spawn(move || {
                let secp = Secp256k1::new();
                let mut local_count = 0u64;

                loop {
                    // limit 個見つかったらループを抜ける（0 = 無限の場合は抜けない）
                    if limit > 0 && found_count.load(Ordering::Relaxed) >= limit {
                        break;
                    }

                    let (sk, pk) = secp.generate_keypair(&mut rand::thread_rng());
                    local_count += 1;

                    // bech32 形式に変換
                    let npub = pubkey_to_npub(&pk);
                    // "npub1" を除去して、bech32 文字列の部分だけを取り出す
                    let npub_body = &npub[5..]; // "npub1" は5文字

                    // 複数 prefix のマッチング判定（どれか1つにマッチすれば OK）
                    if let Some(matched_prefix) = prefixes.iter().find(|p| npub_body.starts_with(p.as_str())) {
                        // 見つかった個数をインクリメント
                        let count = found_count.fetch_add(1, Ordering::Relaxed) + 1;

                        // 現在の試行回数を取得
                        let current_total = total_count.load(Ordering::Relaxed) + local_count;

                        // 結果を channel 経由で送信（matched_prefix も含める）
                        if sender.send((sk, pk, npub.clone(), matched_prefix.clone(), current_total)).is_err() {
                            // メインスレッドが終了している場合
                            break;
                        }

                        // limit 個見つかったらループを抜ける（0 = 無限の場合は抜けない）
                        if limit > 0 && count >= limit {
                            break;
                        }
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

    // sender を drop（全ワーカースレッドが終了したら receiver が None を返すようにする）
    drop(sender);

    // 進捗表示スレッド
    let total_count_progress = Arc::clone(&total_count);
    let found_count_progress = Arc::clone(&found_count);
    let limit_progress = args.limit;
    let progress_handle = std::thread::spawn(move || {
        loop {
            // limit 個見つかったら終了（0 = 無限の場合は終了しない）
            if limit_progress > 0 && found_count_progress.load(Ordering::Relaxed) >= limit_progress {
                break;
            }
            std::thread::sleep(std::time::Duration::from_secs(1));
            let count = total_count_progress.load(Ordering::Relaxed);
            let found = found_count_progress.load(Ordering::Relaxed);
            if count > 0 {
                println!("{}回試行中... (見つかった: {}個)", count, found);
            }
        }
    });

    // ファイル出力の準備（append モード）
    let mut output_file = if let Some(ref output_path) = args.output {
        Some(OpenOptions::new()
            .create(true)
            .append(true)
            .open(output_path)?)
    } else {
        None
    };

    // メインスレッドで結果を受信・出力
    let mut result_count = 0;
    while let Ok((sk, pk, npub, matched_prefix, current_total)) = receiver.recv() {
        result_count += 1;
        let elapsed = start.elapsed();
        let elapsed_secs = elapsed.as_secs_f64();
        let keys_per_sec = current_total as f64 / elapsed_secs;

        let nsec = seckey_to_nsec(&sk);
        let pk_hex = pk.to_string();
        let pk_x_only = &pk_hex[2..]; // x座標のみ（圧縮形式の先頭2文字を除去）

        // 結果を整形
        let output_text = format!(
            "✅ {}個目が見つかりました！（{}回試行、{}スレッド）\n\
             マッチした prefix: '{}'\n\n\
             経過時間: {:.2}秒\n\
             パフォーマンス: {:.2} keys/sec\n\n\
             秘密鍵（hex）: {}\n\
             秘密鍵（nsec）: {}\n\
             公開鍵（圧縮形式）: {}\n\
             公開鍵（x座標のみ）: {}\n\
             公開鍵（npub）: {}\n\
{}\n",
            result_count,
            current_total,
            num_threads,
            matched_prefix,
            elapsed_secs,
            keys_per_sec,
            sk.display_secret(),
            nsec,
            pk,
            pk_x_only,
            npub,
            "=".repeat(80)
        );

        // 出力先に応じて出力
        if let Some(ref mut file) = output_file {
            // ファイルに append
            file.write_all(output_text.as_bytes())?;
            file.flush()?;
        }
        // stdout にも出力（ファイル出力の有無に関わらず）
        print!("{}", output_text);
        io::stdout().flush()?;
    }

    // 全スレッドの終了を待つ
    for handle in handles {
        handle.join().unwrap();
    }
    progress_handle.join().unwrap();

    // 最終結果を表示
    let final_count = total_count.load(Ordering::Relaxed);
    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    println!("\n🎉 マイニング完了！");
    println!("見つかった鍵: {}個", result_count);
    println!("総試行回数: {}回", final_count);
    println!("経過時間: {:.2}秒", elapsed_secs);
    if let Some(ref output_path) = args.output {
        println!("結果をファイルに保存しました: {}", output_path);
    }

    Ok(())
}

/// GPU マイニングモード
fn run_gpu_mining(
    prefixes: &[String],
    limit: usize,
    batch_size: usize,
    output_path: Option<&str>,
) -> io::Result<()> {
    // GPU 初期化
    let ctx = match init_gpu() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("❌ GPU initialization failed: {}", e);
            std::process::exit(1);
        }
    };
    println!("✅ GPU initialized successfully!\n");

    let start = Instant::now();
    let mut total_count: u64 = 0;
    let mut found_count: usize = 0;
    let mut rng = rand::thread_rng();

    // ファイル出力の準備（append モード）
    let mut output_file = if let Some(path) = output_path {
        Some(OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?)
    } else {
        None
    };

    // 秘密鍵のバッファ（バイト列として保持、結果出力時に nsec を生成するため）
    let mut privkey_bytes: Vec<[u8; 32]> = vec![[0u8; 32]; batch_size];
    let mut privkeys_u64: Vec<[u64; 4]> = vec![[0u64; 4]; batch_size];

    // メインループ
    loop {
        // 1. ランダムな秘密鍵をバッチで生成（CPU）
        for i in 0..batch_size {
            rng.fill_bytes(&mut privkey_bytes[i]);
            privkeys_u64[i] = bytes_to_u64x4(&privkey_bytes[i]);
        }

        // 2. GPU で公開鍵を生成
        let pubkeys_x = match generate_pubkeys_batch(&ctx, &privkeys_u64) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("❌ GPU kernel error: {}", e);
                std::process::exit(1);
            }
        };

        // 3. CPU で npub に変換＆prefix マッチング
        for i in 0..batch_size {
            total_count += 1;

            // [u64; 4] → [u8; 32] → npub
            let pubkey_bytes = u64x4_to_bytes(&pubkeys_x[i]);
            let npub = pubkey_bytes_to_npub(&pubkey_bytes);
            let npub_body = &npub[5..]; // "npub1" は5文字

            // prefix マッチング
            if let Some(matched_prefix) = prefixes.iter().find(|p| npub_body.starts_with(p.as_str())) {
                found_count += 1;

                let elapsed = start.elapsed();
                let elapsed_secs = elapsed.as_secs_f64();
                let keys_per_sec = total_count as f64 / elapsed_secs;

                // 秘密鍵から nsec を生成
                let sk = SecretKey::from_slice(&privkey_bytes[i])
                    .expect("Invalid secret key");
                let nsec = seckey_to_nsec(&sk);

                // 公開鍵を取得（表示用）
                let secp = Secp256k1::new();
                let pk = sk.public_key(&secp);
                let pk_hex = pk.to_string();
                let pk_x_only = &pk_hex[2..];

                // 結果を整形
                let output_text = format!(
                    "✅ {}個目が見つかりました！（{}回試行、GPU）\n\
                     マッチした prefix: '{}'\n\n\
                     経過時間: {:.2}秒\n\
                     パフォーマンス: {:.2} keys/sec\n\n\
                     秘密鍵（hex）: {}\n\
                     秘密鍵（nsec）: {}\n\
                     公開鍵（圧縮形式）: {}\n\
                     公開鍵（x座標のみ）: {}\n\
                     公開鍵（npub）: {}\n\
{}\n",
                    found_count,
                    total_count,
                    matched_prefix,
                    elapsed_secs,
                    keys_per_sec,
                    sk.display_secret(),
                    nsec,
                    pk,
                    pk_x_only,
                    npub,
                    "=".repeat(80)
                );

                // 出力
                if let Some(ref mut file) = output_file {
                    file.write_all(output_text.as_bytes())?;
                    file.flush()?;
                }
                print!("{}", output_text);
                io::stdout().flush()?;

                // limit 個見つかったら終了
                if limit > 0 && found_count >= limit {
                    // 最終結果を表示
                    let final_elapsed = start.elapsed();
                    let final_elapsed_secs = final_elapsed.as_secs_f64();
                    println!("\n🎉 GPU マイニング完了！");
                    println!("見つかった鍵: {}個", found_count);
                    println!("総試行回数: {}回", total_count);
                    println!("経過時間: {:.2}秒", final_elapsed_secs);
                    println!("パフォーマンス: {:.2} keys/sec", total_count as f64 / final_elapsed_secs);
                    if let Some(path) = output_path {
                        println!("結果をファイルに保存しました: {}", path);
                    }
                    return Ok(());
                }
            }
        }

        // 進捗表示（バッチごと）
        if total_count % (batch_size as u64 * 10) == 0 {
            let elapsed_secs = start.elapsed().as_secs_f64();
            let keys_per_sec = total_count as f64 / elapsed_secs;
            println!("{}回試行中... ({:.2} keys/sec, 見つかった: {}個)",
                     total_count, keys_per_sec, found_count);
        }
    }
}
