# mocnpub

Nostr npub vanity address miner with GPU acceleration (CUDA).

**3.24 billion keys/sec** on RTX 5070 Ti - find an 8-character prefix in ~6 minutes!

## Features

- GPU-accelerated mining using CUDA
- CPU fallback mode with multi-threading
- Multiple prefix search (OR matching)
- Optimized secp256k1 implementation with endomorphism
- Automatic GPU parameter tuning (Tail Effect prevention)

## Requirements

### GPU Mode (Recommended)

- NVIDIA GPU (Compute Capability 7.5+)
- CUDA Toolkit 12.x or 13.x
- Windows or Linux (WSL supported for building)

### CPU Mode

- Any x86_64 system
- Multi-core CPU recommended

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/high-moctane/mocnpub.git
cd mocnpub

# Build (requires CUDA Toolkit for GPU support)
cargo build --release
```

### Build Options

You can customize the build with environment variables:

```bash
# Custom keys per thread (default: 1408)
MAX_KEYS_PER_THREAD=2048 cargo build --release
```

## Usage

### Basic Usage

```bash
# GPU mode (recommended)
./target/release/mocnpub-main --gpu --prefix m0ctane

# CPU mode
./target/release/mocnpub-main --prefix m0ctane
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--prefix <PREFIX>` | Prefix to search for (required) | - |
| `--gpu` | Enable GPU mode | false |
| `--batch-size <N>` | GPU batch size | 3584000 |
| `--threads-per-block <N>` | GPU threads per block | 128 |
| `--limit <N>` | Number of keys to find | 1 |

### Multiple Prefixes

Search for multiple prefixes at once (OR matching):

```bash
./target/release/mocnpub-main --gpu --prefix m0ctane,sakura,n0str
```

### Examples

```bash
# Find a 4-character prefix (fast, ~5 seconds)
./target/release/mocnpub-main --gpu --prefix 0000

# Find an 8-character prefix (~6 minutes)
./target/release/mocnpub-main --gpu --prefix m0ctane0

# Find multiple keys
./target/release/mocnpub-main --gpu --prefix moc --limit 5
```

## Performance

Benchmarked on RTX 5070 Ti (16GB VRAM):

| Mode | Performance | vs CPU |
|------|-------------|--------|
| CPU (16 threads) | ~70K keys/sec | 1x |
| GPU (CUDA) | **3.24B keys/sec** | **46,000x** |

### Expected Search Time

| Prefix Length | Combinations | Expected Time (GPU) |
|---------------|--------------|---------------------|
| 4 chars | ~1M | < 1 sec |
| 6 chars | ~1B | ~0.3 sec |
| 8 chars | ~1T | ~6 min |
| 10 chars | ~1P | ~4 days |

Note: bech32 uses 32 characters (excluding 1, b, i, o), so each character adds ~5 bits of entropy.

## Technical Details

### Optimizations

- **Endomorphism**: 3x speedup using secp256k1's special properties
- **Montgomery's Trick**: Batch modular inverse computation
- **Branchless arithmetic**: Reduced warp divergence
- **Optimal batch sizing**: 400 waves for maximum GPU utilization

### Architecture

- Pure Rust with inline CUDA (PTX)
- No external secp256k1 library dependency on GPU
- Custom 256-bit modular arithmetic

## Troubleshooting

### CUDA not found

Ensure CUDA Toolkit is installed and `nvcc` is in your PATH:

```bash
# Check CUDA installation
nvcc --version

# Set CUDA_PATH if needed
export CUDA_PATH=/usr/local/cuda
```

### Out of memory

Reduce batch size:

```bash
./target/release/mocnpub-main --gpu --prefix m0ctane --batch-size 1000000
```

### WSL Performance

For best performance, run the compiled binary on Windows native, not in WSL.
Build in WSL, then copy to Windows or use `git pull` on Windows.

## License

MIT
