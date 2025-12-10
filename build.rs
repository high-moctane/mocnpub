// build.rs - Automatically compile CUDA kernels to PTX
//
// Automatically runs nvcc during cargo build to generate .ptx from .cu files.
// Supports both Windows and WSL/Linux.

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
    let max_keys_per_thread =
        env::var("MAX_KEYS_PER_THREAD").unwrap_or_else(|_| "1408".to_string());

    // Pass to Rust code via cargo:rustc-env
    println!(
        "cargo:rustc-env=MAX_KEYS_PER_THREAD={}",
        max_keys_per_thread
    );
    println!("cargo:warning=MAX_KEYS_PER_THREAD={}", max_keys_per_thread);

    // Detect GPU architecture
    // Rebuild when CUDA_ARCH env var changes
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    let arch = detect_gpu_arch();
    println!("cargo:warning=Using GPU architecture: {}", arch);

    // Find nvcc
    let nvcc = find_nvcc().expect(
        "Could not find nvcc. Please ensure CUDA Toolkit is installed and either:\n\
         1. Set CUDA_PATH environment variable, or\n\
         2. Add nvcc to your PATH",
    );

    println!("cargo:warning=Using nvcc: {}", nvcc.display());

    // Compile secp256k1.cu to PTX
    compile_cu_to_ptx(
        &nvcc,
        &cuda_dir.join("secp256k1.cu"),
        &out_dir,
        &max_keys_per_thread,
        &arch,
    );
}

fn compile_cu_to_ptx(
    nvcc: &PathBuf,
    cu_file: &PathBuf,
    out_dir: &PathBuf,
    max_keys_per_thread: &str,
    arch: &str,
) {
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
        "cargo:warning=Compiling {} -> {} (arch={})",
        cu_file.display(),
        ptx_file.display(),
        arch
    );

    // Add UTF-8 option for cl.exe on Windows
    let mut args = vec![
        "-ptx".to_string(),
        "-o".to_string(),
        ptx_file.to_str().unwrap().to_string(),
        cu_file.to_str().unwrap().to_string(),
        format!("-arch={}", arch),
        // Define MAX_KEYS_PER_THREAD (value from environment variable)
        format!("-D MAX_KEYS_PER_THREAD={}", max_keys_per_thread),
        // Allow newer Visual Studio versions (e.g., VS 2026)
        "-allow-unsupported-compiler".to_string(),
        // Source-level profiling (show line numbers in ncu-ui)
        "-lineinfo".to_string(),
    ];

    // Pass UTF-8 option to cl.exe on Windows
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

/// Detect GPU architecture for PTX compilation
///
/// Priority:
/// 1. CUDA_ARCH environment variable (e.g., "sm_86", "sm_120")
/// 2. Auto-detect via nvidia-smi
/// 3. Fallback to sm_75 (Turing, minimum supported by CUDA 13.0)
fn detect_gpu_arch() -> String {
    // 1. Check CUDA_ARCH environment variable
    if let Ok(arch) = env::var("CUDA_ARCH") {
        println!("cargo:warning=GPU arch from CUDA_ARCH env: {}", arch);
        return arch;
    }

    // 2. Try to detect via nvidia-smi
    if let Some(arch) = detect_gpu_arch_nvidia_smi() {
        println!(
            "cargo:warning=GPU arch auto-detected via nvidia-smi: {}",
            arch
        );
        return arch;
    }

    // 3. Fallback to sm_75 (Turing)
    // PTX is forward-compatible, so this will work on newer GPUs via JIT
    println!("cargo:warning=GPU arch fallback to sm_75 (Turing)");
    "sm_75".to_string()
}

/// Detect GPU compute capability via nvidia-smi
///
/// nvidia-smi --query-gpu=compute_cap --format=csv,noheader
/// Returns e.g., "8.6" for RTX 3080, "12.0" for RTX 5070 Ti
fn detect_gpu_arch_nvidia_smi() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let compute_cap = stdout.lines().next()?.trim();

    // Convert "8.6" -> "sm_86", "12.0" -> "sm_120"
    let arch = compute_cap_to_sm(compute_cap)?;
    Some(arch)
}

/// Convert compute capability string to sm_XX format
/// e.g., "8.6" -> "sm_86", "12.0" -> "sm_120"
fn compute_cap_to_sm(compute_cap: &str) -> Option<String> {
    let parts: Vec<&str> = compute_cap.split('.').collect();
    if parts.len() != 2 {
        return None;
    }

    let major: u32 = parts[0].parse().ok()?;
    let minor: u32 = parts[1].parse().ok()?;

    // sm_XY where X is major, Y is minor
    Some(format!("sm_{}{}", major, minor))
}
