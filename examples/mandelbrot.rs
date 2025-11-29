use num_complex::Complex64;
use image::{ImageBuffer, Rgb};
use std::time::Instant;
use std::sync::Arc;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

// 画像サイズ
const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

// 複素数平面の範囲
const X_MIN: f32 = -2.5;
const X_MAX: f32 = 1.0;
const Y_MIN: f32 = -1.0;
const Y_MAX: f32 = 1.0;

// 最大反復回数
const MAX_ITER: u32 = 1000;

// PTX コード（build.rs で自動生成）
const PTX_CODE: &str = include_str!(concat!(env!("OUT_DIR"), "/mandelbrot.ptx"));

/// ピクセル座標を複素数平面の座標に変換
fn pixel_to_complex(px: u32, py: u32) -> Complex64 {
    let x = X_MIN as f64 + (px as f64 / WIDTH as f64) * (X_MAX - X_MIN) as f64;
    let y = Y_MIN as f64 + (py as f64 / HEIGHT as f64) * (Y_MAX - Y_MIN) as f64;
    Complex64::new(x, y)
}

/// マンデルブロ集合の計算（発散するまでの反復回数を返す）
fn mandelbrot(c: Complex64) -> u32 {
    let mut z = Complex64::new(0.0, 0.0);
    for n in 0..MAX_ITER {
        if z.norm() >= 2.0 {
            return n;
        }
        z = z * z + c;
    }
    MAX_ITER
}

/// 反復回数を色に変換（グラデーション）
fn color_map(iter: u32) -> Rgb<u8> {
    if iter == MAX_ITER {
        // マンデルブロ集合に属する → 黒
        Rgb([0, 0, 0])
    } else {
        // 発散した → 反復回数に応じて色付け
        // シンプルなグラデーション（青 → 緑 → 赤）
        let t = iter as f64 / MAX_ITER as f64;
        let r = (9.0 * (1.0 - t) * t * t * t * 255.0) as u8;
        let g = (15.0 * (1.0 - t) * (1.0 - t) * t * t * 255.0) as u8;
        let b = (8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t * 255.0) as u8;
        Rgb([r, g, b])
    }
}

/// 反復回数の配列から画像を生成
fn create_image_from_iters(iters: &[u32]) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut img = ImageBuffer::new(WIDTH, HEIGHT);

    for py in 0..HEIGHT {
        for px in 0..WIDTH {
            let idx = (py * WIDTH + px) as usize;
            let iter = iters[idx];
            let color = color_map(iter);
            img.put_pixel(px, py, color);
        }
    }

    img
}

/// CPU 版マンデルブロ集合の画像を生成
fn generate_mandelbrot_cpu() -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut img = ImageBuffer::new(WIDTH, HEIGHT);

    for py in 0..HEIGHT {
        for px in 0..WIDTH {
            let c = pixel_to_complex(px, py);
            let iter = mandelbrot(c);
            let color = color_map(iter);
            img.put_pixel(px, py, color);
        }
    }

    img
}

/// GPU 版マンデルブロ集合の画像を生成
fn generate_mandelbrot_gpu(ctx: &Arc<CudaContext>) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn std::error::Error>> {
    let stream = ctx.default_stream();

    // PTX コードをロード
    let module = ctx.load_module(Ptx::from_src(PTX_CODE))?;
    let kernel = module.load_function("mandelbrot_kernel")?;

    // デバイスメモリを確保（結果を格納する配列）
    let total_pixels = (WIDTH * HEIGHT) as usize;
    let mut result_dev = stream.alloc_zeros::<u32>(total_pixels)?;

    // カーネル起動設定
    // ブロックサイズ: 16x16 = 256 スレッド
    // グリッドサイズ: (WIDTH/16) x (HEIGHT/16)
    let block_dim = (16, 16, 1);
    let grid_dim = ((WIDTH + 15) / 16, (HEIGHT + 15) / 16, 1);
    let cfg = LaunchConfig {
        grid_dim,
        block_dim,
        shared_mem_bytes: 0,
    };

    // カーネルを起動（ビルダーパターン）
    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut result_dev);           // result
    builder.arg(&(WIDTH as i32));          // width
    builder.arg(&(HEIGHT as i32));         // height
    builder.arg(&X_MIN);                   // x_min
    builder.arg(&X_MAX);                   // x_max
    builder.arg(&Y_MIN);                   // y_min
    builder.arg(&Y_MAX);                   // y_max
    builder.arg(&(MAX_ITER as i32));      // max_iter
    unsafe { builder.launch(cfg)? };

    // 結果をホストにコピー
    let result_host = stream.memcpy_dtov(&result_dev)?;

    // 画像を生成
    Ok(create_image_from_iters(&result_host))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌀 mocnpub - マンデルブロ集合（CPU vs GPU） 🌀\n");

    // GPU コンテキストを作成
    println!("GPU デバイス 0 に接続中...");
    let ctx = CudaContext::new(0)?;
    println!("✅ GPU デバイス 0 に接続成功！\n");

    // ===== CPU 版 =====
    println!("🖥️  CPU 版マンデルブロ集合を生成中...");
    let start_cpu = Instant::now();
    let img_cpu = generate_mandelbrot_cpu();
    let duration_cpu = start_cpu.elapsed();
    println!("✅ CPU 版完了！所要時間: {:.2}秒", duration_cpu.as_secs_f64());

    let filename_cpu = "mandelbrot_cpu.png";
    img_cpu.save(filename_cpu)?;
    println!("✅ 画像を保存しました: {}\n", filename_cpu);

    // ===== GPU 版 =====
    println!("🚀 GPU 版マンデルブロ集合を生成中...");
    let start_gpu = Instant::now();
    let img_gpu = generate_mandelbrot_gpu(&ctx)?;
    let duration_gpu = start_gpu.elapsed();
    println!("✅ GPU 版完了！所要時間: {:.4}秒", duration_gpu.as_secs_f64());

    let filename_gpu = "mandelbrot_gpu.png";
    img_gpu.save(filename_gpu)?;
    println!("✅ 画像を保存しました: {}\n", filename_gpu);

    // ===== パフォーマンス比較 =====
    println!("📊 パフォーマンス比較:");
    println!("  CPU: {:.2}秒", duration_cpu.as_secs_f64());
    println!("  GPU: {:.4}秒", duration_gpu.as_secs_f64());
    let speedup = duration_cpu.as_secs_f64() / duration_gpu.as_secs_f64();
    println!("  🔥 高速化: {:.1}倍", speedup);

    Ok(())
}
