// build.rs - CUDA カーネルを自動的に PTX にコンパイル
//
// cargo build 時に自動的に nvcc を実行して、.cu ファイルから .ptx を生成します。
// Windows と WSL/Linux の両方に対応しています。

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cuda_dir = manifest_dir.join("cuda");

    // Read MAX_KEYS_PER_THREAD from environment variable (default: 1408)
    // Rebuild when this env var changes
    println!("cargo:rerun-if-env-changed=MAX_KEYS_PER_THREAD");
    let max_keys_per_thread = env::var("MAX_KEYS_PER_THREAD")
        .unwrap_or_else(|_| "1408".to_string());

    // Pass to Rust code via cargo:rustc-env
    println!("cargo:rustc-env=MAX_KEYS_PER_THREAD={}", max_keys_per_thread);
    println!("cargo:warning=MAX_KEYS_PER_THREAD={}", max_keys_per_thread);

    // Find nvcc
    let nvcc = find_nvcc().expect(
        "Could not find nvcc. Please ensure CUDA Toolkit is installed and either:\n\
         1. Set CUDA_PATH environment variable, or\n\
         2. Add nvcc to your PATH",
    );

    println!("cargo:warning=Using nvcc: {}", nvcc.display());

    // Compile secp256k1.cu to PTX
    compile_cu_to_ptx(&nvcc, &cuda_dir.join("secp256k1.cu"), &out_dir, &max_keys_per_thread);

    // Compile mandelbrot.cu to PTX (in root directory)
    compile_cu_to_ptx(&nvcc, &manifest_dir.join("mandelbrot.cu"), &out_dir, &max_keys_per_thread);
}

fn compile_cu_to_ptx(nvcc: &PathBuf, cu_file: &PathBuf, out_dir: &PathBuf, max_keys_per_thread: &str) {
    let file_stem = cu_file.file_stem().unwrap().to_str().unwrap();
    let ptx_file = out_dir.join(format!("{}.ptx", file_stem));

    println!("cargo:rerun-if-changed={}", cu_file.display());

    // Check if .cu file exists
    if !cu_file.exists() {
        println!(
            "cargo:warning=Skipping {}: file not found",
            cu_file.display()
        );
        return;
    }

    println!(
        "cargo:warning=Compiling {} -> {}",
        cu_file.display(),
        ptx_file.display()
    );

    // TODO: 将来的には GPU アーキテクチャを動的に検出したい
    // 現在は RTX 5070 Ti (Blackwell, sm_120) を想定
    // PTX は前方互換性があるので、古いアーキテクチャで生成しても
    // JIT コンパイル時に適切なアーキテクチャにコンパイルされる
    // 注意: CUDA 13.0 では sm_75 (Turing) が最小サポートアーキテクチャ
    let arch = "sm_120"; // Blackwell（RTX 50 シリーズ）

    // Windows の場合、cl.exe に UTF-8 として解釈させるオプションを追加
    let mut args = vec![
        "-ptx".to_string(),
        "-o".to_string(),
        ptx_file.to_str().unwrap().to_string(),
        cu_file.to_str().unwrap().to_string(),
        format!("-arch={}", arch),
        // MAX_KEYS_PER_THREAD を定義（環境変数から取得した値）
        format!("-D MAX_KEYS_PER_THREAD={}", max_keys_per_thread),
        // Visual Studio 2026 など新しいバージョンでもコンパイルできるように
        "-allow-unsupported-compiler".to_string(),
        // ソースレベルのプロファイリング用（ncu-ui で行番号を表示）
        "-lineinfo".to_string(),
    ];

    // Windows では cl.exe に UTF-8 オプションを渡す
    if cfg!(target_os = "windows") {
        args.push("-Xcompiler".to_string());
        args.push("/utf-8".to_string());
    }

    let output = Command::new(nvcc)
        .args(&args)
        .output()
        .expect("Failed to execute nvcc");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!(
            "nvcc failed to compile {}\n\nstdout:\n{}\n\nstderr:\n{}",
            cu_file.display(),
            stdout,
            stderr
        );
    }
}

fn find_nvcc() -> Option<PathBuf> {
    // 1. Check CUDA_PATH environment variable
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let nvcc = PathBuf::from(&cuda_path)
            .join("bin")
            .join(nvcc_executable());
        if nvcc.exists() {
            return Some(nvcc);
        }
    }

    // 2. Try common paths based on OS
    let common_paths = if cfg!(target_os = "windows") {
        vec![
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        ]
    } else {
        // Linux / WSL
        vec![
            "/usr/local/cuda-13.0/bin",
            "/usr/local/cuda-12.8/bin",
            "/usr/local/cuda/bin",
        ]
    };

    for path in common_paths {
        let nvcc = PathBuf::from(path).join(nvcc_executable());
        if nvcc.exists() {
            return Some(nvcc);
        }
    }

    // 3. Try to find nvcc in PATH using `which` (Unix) or `where` (Windows)
    let which_cmd = if cfg!(target_os = "windows") {
        "where"
    } else {
        "which"
    };

    if let Ok(output) = Command::new(which_cmd).arg("nvcc").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout);
            let path = path.trim();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }

    None
}

fn nvcc_executable() -> &'static str {
    if cfg!(target_os = "windows") {
        "nvcc.exe"
    } else {
        "nvcc"
    }
}
