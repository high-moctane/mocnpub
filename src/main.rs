use num_complex::Complex64;
use image::{ImageBuffer, Rgb};
use std::time::Instant;

// 画像サイズ
const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

// 複素数平面の範囲
const X_MIN: f64 = -2.5;
const X_MAX: f64 = 1.0;
const Y_MIN: f64 = -1.0;
const Y_MAX: f64 = 1.0;

// 最大反復回数
const MAX_ITER: u32 = 1000;

/// ピクセル座標を複素数平面の座標に変換
fn pixel_to_complex(px: u32, py: u32) -> Complex64 {
    let x = X_MIN + (px as f64 / WIDTH as f64) * (X_MAX - X_MIN);
    let y = Y_MIN + (py as f64 / HEIGHT as f64) * (Y_MAX - Y_MIN);
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌀 mocnpub - マンデルブロ集合（CPU 版） 🌀\n");

    // CPU 版マンデルブロを生成
    println!("CPU 版マンデルブロ集合を生成中...");
    let start = Instant::now();
    let img = generate_mandelbrot_cpu();
    let duration = start.elapsed();

    println!("✅ 生成完了！所要時間: {:.2}秒", duration.as_secs_f64());

    // 画像を保存
    let filename = "mandelbrot_cpu.png";
    img.save(filename)?;
    println!("✅ 画像を保存しました: {}", filename);

    Ok(())
}
