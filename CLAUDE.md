# mocnpub - Nostr npub Mining with CUDA 🔥

**Last Updated**: 2025-12-27

This file provides project context for Claude Code to assist with development.

---

## 🎯 Project Overview

**mocnpub** is a high-performance Nostr npub vanity address miner.

**Goal**:
- Find nsec (private key) that produces npub with desired prefix
- Achieve maximum performance using GPGPU (CUDA) 🚀

**Final Results**:
- **5.9B keys/sec** (**84,935x** faster than CPU) 🔥
- 8-character prefix found in ~4 minutes

---

## 🛠️ Technology Stack

### Language: Rust 🦀

- Cross-platform (WSL + Windows)
- Rich crypto libraries (`secp256k1` Rust bindings)
- Long-term stability (static linking, no runtime dependencies)
- CUDA integration via `cudarc` crate

### GPGPU: CUDA 🔥

- Optimized for NVIDIA GPUs (RTX 5070 Ti)
- Best performance (NVIDIA-specific optimizations)
- Well-documented (easier to learn)

---

## 🖥️ Development Environment

### Building

```bash
cargo build --release
```

PTX is auto-compiled by `build.rs`.

### WSL + Windows Workflow

- Develop, commit, push in WSL
- `git pull` and run on Windows
- Windows native execution maximizes performance

### NixOS Setup

NixOS splits CUDA Toolkit into separate store paths (`cuda_nvcc`, `cuda_cudart`, etc.) and
merges them via `cuda-merged`. However, `nvcc` resolves its symlink to the real `cuda_nvcc`
binary and fails to find `cuda_runtime.h` in the separate `cuda-merged` include directory.

**Required configuration**:
- `cudaPackages_13_1.cudatoolkit` in `environment.systemPackages`
- `environment.variables.CUDA_PATH = "${pkgs.cudaPackages_13_1.cudatoolkit}"` — `build.rs` reads this to pass `-I$CUDA_PATH/include` to nvcc
- NixOS WSL: `LD_LIBRARY_PATH=/usr/lib/wsl/lib` for `libcuda.so` (Windows driver passthrough)

**glibc 2.42+ workaround**: Already handled in `build.rs` with `-U_GNU_SOURCE` (glibc declares `rsqrt`/`rsqrtf` in C23 mode, conflicting with CUDA's `math_functions.h`)

---

## 🚀 Optimization Journey (6 weeks)

| Step | Content | Result |
|------|---------|--------|
| Step 0-1 | Environment setup, Mandelbrot | GPU verified |
| Step 2-2.5 | CPU miner | 70K keys/sec |
| Step 3 | GPU port | 1.16M keys/sec (16x) |
| Step 4 | Consecutive keys + Montgomery | 391M keys/sec (5,586x) |
| Step 5-13 | Parameter tuning | 4.15B keys/sec (59,286x) |
| Step 14-35 | PTX optimization | **5.94B keys/sec (84,935x)** |

### Key Optimizations

**Algorithmic**:
- Consecutive secret keys + PointAdd (~300x lighter than ScalarMult)
- Montgomery's Trick (~85x reduction in inversions)
- Endomorphism (2.9x coverage)
- dG table precompute (+12.7%)
- Addition Chain (128→14 multiplications for inversion)
- Z² cumulative product strategy (1600 fewer ModSquare ops)

**GPU**:
- Triple Buffering (100% GPU utilization)
- Constant Memory (dG table, patterns/masks)
- Branchless arithmetic (_ModSub/_ModAdd)
- `__launch_bounds__(128, 5)`

**PTX**:
- Inline PTX assembly (carry/borrow chains)
- Specialized functions: _Add64x3, _Add320, _Sub256
- Pipeline stall reduction
- Loop fusion

### Detailed Documentation

See `docs/` for deep dives:
- `docs/JOURNEY.md` — Development story
- `docs/OPTIMIZATION.md` — Technical details
- `docs/LEARNING.md` — Learning path

---

## 📁 Project Structure

```
src/
├── main.rs      # CLI entry point (clap, Mine subcommand)
├── lib.rs       # Core utilities (byte conversion, prefix matching)
└── gpu.rs       # CUDA integration (cudarc, triple buffering)

cuda/
└── secp256k1.cu # CUDA kernel (secp256k1, Montgomery, PTX)

learning/
├── mandelbrot.rs/.cu  # GPU learning experiments
└── keygen_test.rs     # Key generation tests

benches/
└── benchmark.rs # Criterion benchmarks

build.rs         # PTX auto-compilation
```

---

## 📚 Development Philosophy

- Learn by doing (CUDA, Rust, secp256k1 all new)
- Take time to understand deeply
- Keep learning files in git (project growth matters) 🌱

---

*Built with Claude Code* 🌸
