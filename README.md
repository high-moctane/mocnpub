# mocnpub 🌸

Nostr npub vanity address miner with GPU acceleration (CUDA).

**5.8 billion keys/sec** on RTX 5070 Ti - find an 8-character prefix in ~3.5 minutes!

## Features

- GPU-accelerated mining using CUDA
- Multiple prefix search (OR matching)
- Optimized secp256k1 implementation with endomorphism
- PTX-level optimizations for maximum performance
- Triple buffering for 100% GPU utilization

## Requirements

- NVIDIA GPU (Compute Capability 7.5+)
- CUDA Toolkit 12.x or 13.x
- Windows or Linux (WSL supported for building)

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
# Custom keys per thread (default: 1600)
MAX_KEYS_PER_THREAD=2048 cargo build --release
```

## Usage

### Basic Usage

```bash
./target/release/mocnpub-main mine --prefix m0ctane
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--prefix <PREFIX>` | Prefix to search for (required) | - |
| `--limit <N>` | Number of keys to find (0 = unlimited) | 1 |
| `--output <FILE>` | Output file (optional) | stdout |
| `--batch-size <N>` | GPU batch size | 4000000 |
| `--threads-per-block <N>` | GPU threads per block | 128 |
| `--miners <N>` | Number of parallel miners | 2 |

### Multiple Prefixes

Search for multiple prefixes at once (OR matching):

```bash
./target/release/mocnpub-main mine --prefix m0ctane,sakura,n0str
```

### Examples

```bash
# Find a 4-character prefix (fast, < 1 second)
./target/release/mocnpub-main mine --prefix 0000

# Find an 8-character prefix (~3.5 minutes)
./target/release/mocnpub-main mine --prefix m0ctane0

# Find multiple keys
./target/release/mocnpub-main mine --prefix m0c --limit 5

# Save to file
./target/release/mocnpub-main mine --prefix test --output keys.txt
```

## Performance

Benchmarked on RTX 5070 Ti (16GB VRAM):

**5.8 billion keys/sec** (82,857x faster than CPU baseline)

### Expected Search Time

| Prefix Length | Combinations | Expected Time |
|---------------|--------------|---------------|
| 4 chars | ~1M | < 1 sec |
| 6 chars | ~1B | < 1 sec |
| 8 chars | ~1T | ~3.5 min |
| 10 chars | ~1P | ~2.5 days |

Note: bech32 uses 32 characters (excluding 1, b, i, o), so each character adds ~5 bits of entropy.

## Technical Details

### Optimizations

- **Endomorphism**: 2.9x coverage using secp256k1's special properties
- **Montgomery's Trick**: ~85x reduction in modular inversions
- **Sequential key strategy**: PointAdd instead of full scalar multiplication
- **Addition Chain**: 114 multiplications eliminated in ModInv
- **PTX inline assembly**: Hand-tuned carry/borrow chains
- **Triple buffering**: 100% GPU utilization

### Architecture

- Pure Rust with inline CUDA (PTX)
- No external secp256k1 library dependency on GPU
- Custom 256-bit modular arithmetic with PTX optimizations

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

Rebuild with smaller `MAX_KEYS_PER_THREAD`:

```bash
MAX_KEYS_PER_THREAD=800 cargo build --release
```

### WSL Performance

For best performance, run the compiled binary on Windows native, not in WSL.
Build in WSL, then copy to Windows or use `git pull` on Windows.

## Documentation

For detailed information about the development journey:

- [JOURNEY.md](docs/JOURNEY.md) - The complete story of building mocnpub
- [OPTIMIZATION.md](docs/OPTIMIZATION.md) - Technical deep-dive into all 35 optimization steps
- [LEARNING.md](docs/LEARNING.md) - Learning path from beginner to PTX mastery
- [CODE_REVIEW.md](docs/CODE_REVIEW.md) - Code review by Claude (Web) 🌸

## Built With

This project was developed through pair programming with [Claude Code](https://claude.com/claude-code) 🌸

## License

MIT
