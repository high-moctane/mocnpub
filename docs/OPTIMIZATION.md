# mocnpub Optimization Journey 🔥

**A comprehensive record of optimizations, experiments, and learnings from CPU to GPU**

Final Performance: **5.8B keys/sec** (82,857x faster than CPU)

---

## Performance Timeline 📈

| Date | Milestone | Performance | CPU Ratio | Improvement |
|------|-----------|-------------|-----------|-------------|
| 2025-11-22 | Step 2: CPU Version | 70K keys/sec | 1x | Baseline |
| 2025-11-26 | Step 3: GPU Basic | 1.16M keys/sec | 16x | +1,557% |
| 2025-11-29 | GPU Prefix Match | 391M keys/sec | 5,586x | +33,614% |
| 2025-12-05 | Step 5: Parameters | 3.30B keys/sec | 47,143x | - |
| 2025-12-13 | Branchless + Launch Bounds | 3.36B keys/sec | 47,937x | +1.8% |
| 2025-12-13 | batch_size Re-optimization | 3.46B keys/sec | 49,486x | +3.0% |
| 2025-12-14 | Triple Buffering | 3.70B keys/sec | 52,857x | +6.9% |
| 2025-12-15 | Sequential Keys + dG Table | 4.14B keys/sec | 59,071x | +11.9% |
| 2025-12-17 | Constant Memory (dG table) | 4.15B keys/sec | 59,286x | +0.4% |
| 2025-12-19 | Constant Memory (patterns) | 4.15B keys/sec | 59,286x | +3.0% (32-prefix) |
| 2025-12-20 | Addition Chain | 4.20B keys/sec | 60,000x | +1.4% |
| 2025-12-20 | Inline PTX | 4.31B keys/sec | 61,571x | +2.7% |
| 2025-12-21 | _Add64x3 | 4.84B keys/sec | 69,143x | +3.4% |
| 2025-12-21 | Complete Ternary Elimination | 4.93B keys/sec | 70,429x | +1.8% |
| 2025-12-21 | **5B Breakthrough (_Add320)** | **5.10B keys/sec** | **72,857x** | **+3.4%** |
| 2025-12-21 | PTX Chain Elimination | 5.29B keys/sec | 75,571x | +3.7% |
| 2025-12-21 | Loop Unrolling (_Add128/_Add512) | 5.38B keys/sec | 76,857x | +1.8% |
| 2025-12-22 | Loop Fusion | 5.50B keys/sec | 78,571x | +1.9% |
| 2025-12-22 | Fine-grained Optimizations | 5.71B keys/sec | 81,571x | +3.8% |
| 2025-12-23 | Local Array SoA | 5.77B keys/sec | 82,429x | +1.1% |
| 2025-12-26 | Z² Cumulative Product | **5.80B keys/sec** | **82,857x** | +0.96% |

**8-character prefix**: ~3.5 minutes (vs 3+ days on CPU)

**Total optimization journey**: Nov 14 → Dec 27 (6 weeks)

---

## Phase 1: CPU Implementation (Step 0-2.5)

### Step 0: Hello World (2025-11-14 〜 11-15)
- CUDA Toolkit installation (Windows + WSL)
- Rust project initialization
- RTX 5070 Ti connection test ✅
- **Learning**: CUDA basics, streams, context

### Step 1: Mandelbrot Set (2025-11-16)
- GPU introduction with visual feedback
- CPU version: 0.41s, GPU version: 0.0060s
- **68.2x speedup on Windows** (vs 3.5x on WSL)
- **Learning**: WSL has virtualization overhead, Windows native is much faster

### Step 2: CPU npub Miner (2025-11-16 〜 11-22)
- secp256k1 key generation (Rust crate)
- bech32 encoding (npub/nsec)
- CLI interface (clap)
- Performance measurement
- **70,000 keys/sec** (single-threaded)
- **Learning**: Bottleneck is key generation (93%)

### Step 2.5: CPU Brushup (2025-11-22 〜 11-23)

**Phase 1: Multi-threading** (11-22)
- 16 threads on Ryzen 9800X3D
- **800K-1M keys/sec** (12-20x speedup)
- Input validation (bech32 invalid characters)

**Phase 2: Features** (11-23)
- Unit tests (7 test cases)
- Continuous mode (`--limit N`)
- Multiple prefix OR specification
- Channel-based result collection

**Phase 3: Benchmarking** (11-23)
- criterion benchmarks
- **Bottleneck identified**: secp256k1 key generation (13.1 µs, 93%)
- bech32 encoding: 663 ns (5%)
- prefix matching: 1.5 ns (0.01%)
- **Ready for GPU optimization!** 🚀

---

## Phase 2: GPU Implementation (Step 3-4)

### Step 3: GPU Basic (2025-11-23 〜 11-26)

**Lectures (11-23)**:
- secp256k1 fundamentals (elliptic curve cryptography)
- Finite field, torus structure, point at infinity
- Modular inverse, Double-and-Add method
- **User predicted Double-and-Add!** 🎯

**GPU Kernel Implementation (11-24)**:
- 256-bit arithmetic (`_Add256`, `_Sub256`, `_ModMult`, `_ModInv`)
- secp256k1-specific reduction: `2^256 mod p = 2^32 + 977`
- Point Doubling/Addition (Jacobian coordinates)
- **7 bugs fixed across 7 sessions** 🔍
- Fuzzing test: 5,579 runs, 0 errors ✅

**Point Multiplication (11-26)**:
- Double-and-Add algorithm (MSB to LSB)
- Batch processing (multiple private keys in parallel)
- **1.16M keys/sec** (16x faster than CPU)

**GPU Integration (11-26)**:
- CPU: bech32 encoding + prefix matching
- GPU: key generation only
- Byte order conversion (little-endian limbs ↔ big-endian bytes)
- **Working end-to-end!** 🎉

### Step 4: Major Optimizations (2025-11-27 〜 11-30)

**Endomorphism (11-29, 12-13)**:
- secp256k1's special property: `β³ ≡ 1 (mod p)`
- Check 3 X-coordinates: P, βP, β²P
- **Theoretical 3x**, actual **2.9x** speedup
- **1.14B keys/sec** (16,286x)

**Sequential Keys + Montgomery's Trick (11-27 〜 11-29)**:
- **"10000-gacha" strategy**: P + G, P + 2G, P + 3G, ...
- PointMult → PointAdd (~300x lighter)
- Montgomery's Trick: N inversions → 1 inversion + 3(N-1) multiplications
- **~85x reduction in inversions** — Algorithm effectiveness validated via micro-benchmark

**GPU-side Prefix Matching (11-29)**:
- Bitmask comparison (5 bits per bech32 char)
- Transfer only matches (massive reduction)
- CPU bottleneck eliminated (bech32 encoding skipped)
- **200M keys/sec @ WSL, 391M keys/sec @ Windows**
- **170x faster than old GPU approach!**

**cuRAND Experiment (11-29)**:
- GPU-side secret key generation
- **Result**: 50x slower (curand_init overhead) ❌
- **Learning**: cuRAND is not cryptographically secure
- Reverted to CPU RNG (ChaCha20)

---

## Phase 3: Fine-tuning (Step 5-10)

### Step 5: Parameter Tuning (2025-11-29 〜 12-05)

**keys_per_thread Optimization (11-29)**:
- Initial confusion: "10.8B keys/sec" was an illusion
- `MAX_KEYS_PER_THREAD` was clamped at 256
- Real testing: 256 → 1408 (VRAM limit)
- **2.63B keys/sec** (38,000x)
- **Learning**: CUDA local arrays are compile-time fixed

**threads_per_block Optimization (11-29)**:
- Tested: 32, 64, 96, 128, 160, 192, 256
- **128 (4 warps) is the sweet spot** (+6.2%)
- 160 (5 warps, odd number) is particularly slow
- 256+ causes register contention

**batch_size Optimization (11-29)**:
- Larger is better (diminishing returns)
- **1,146,880 (128 waves) is optimal** (+10.4%)
- GPU utilization: 70% → 95%

**ncu Profiling (11-29)**:
- Compute Throughput: 73.83% (compute-bound ✅)
- Memory Throughput: 12.16% (plenty of room)
- Occupancy: 33% (register-limited)
- Registers/Thread: 120
- **Est. Speedup: 26.17%** (if registers reduced)

**Register Reduction Experiment (11-29)**:
- `__launch_bounds__(64, 16)` → 64 registers
- Occupancy: 33% → 67% ✅
- **Spilling: 0% → 96%** 😱 (to local memory/DRAM)
- Performance: 1.14B → 1.03B (-10%) ❌
- **Learning**: Native code + compiler optimization is best

**build.rs Improvement (11-29)**:
- Automatic PTX compilation on `cargo build`
- Windows / WSL cross-platform support
- sm_75 → sm_120 (native for RTX 5070 Ti)
- UTF-8 encoding fix (`-Xcompiler /utf-8`)
- **-104,699 lines** (PTX removed from git)

**`_ModSquare` Optimization (11-29)**:
- Symmetry-based squaring: 16 → 10 multiplications (37.5% reduction)
- **1.18B → 1.18B keys/sec** (+3.5% in actual mining)
- Micro-benchmark overhead was hiding the effect

**Tail Effect Mitigation (11-29)**:
- Auto-adjust batch_size to SM count multiple
- `calculate_optimal_batch_size()`
- 65,536 → 67,200 (15 waves exactly)
- **+1.4%** improvement
- **Learning**: Use every SM, no idle cores

### Step 6: CPU Mode Removal (2025-12-13)
- GPU is 47,000x faster → CPU mode is obsolete
- Removed ~200 lines from main.rs
- **787 → 566 lines** (clean-up)

### Step 7: Branchless Optimization (2025-12-04 〜 12-05)

**Branch Efficiency Analysis (12-04)**:
- ncu profiling: Branch Efficiency 78.88%
- 1.68B divergent branches
- **Main culprit**: `_ModSub` and `_ModAdd` comparison branches

**Branchless `_ModSub`/`_ModAdd` (12-04)**:
- Mask selection technique: `uint64_t mask = -borrow;`
- Branch Efficiency: 78.88% → 82.41% (+3.53 pt)
- Divergent Branches: 1.68B → 1.16B (-31%)
- **3.09B → 3.16B keys/sec** (+2.3%)

**ncu-ui Source-level Analysis (12-05)**:
- `-lineinfo` option added to build.rs
- Identified: `_Reduce512` line 324 (`if (temp[4] > 0)`) was 99.16% divergence
- Branchless version implemented
- **Learning**: Divergence elimination ≠ speedup (branch prediction is smart)
- Old if-statement version was slightly faster (3.20B vs 3.19B)
- **Reverted to old version**

**Remaining Divergence**:
- `_PointMult` bit branching (96.26%, double-and-add algorithm)
- Algorithmically necessary → not worth eliminating (2x computation)

### Step 8: Launch Bounds Tuning (2025-12-09 〜 12-13)

**`__launch_bounds__` Discovery (12-09)**:
- Initial: no bounds, 130 registers, Occupancy 25%
- `__launch_bounds__(128, 4)`: 128 registers, Occupancy 33%
- **3.10B → 3.26B keys/sec** (+5.1%)

**Fine-tuning (12-13)**:
- Tested: (128, 4), (128, 5), (128, 6)
- **(128, 5) is optimal**: 96 registers, Occupancy 41%
- **3.326B → 3.356B keys/sec** (+0.9%)
- **Learning**: Compiler adjusts register allocation smartly

**Code Reading & `_PointAddMixed` Optimization (12-13)**:
- Found duplicate computation: `X1 * H^2` calculated twice
- Reuse `X1_H2[4]` variable
- 8M + 3S → 7M + 3S (12.5% reduction)
- **3.30B → 3.326B keys/sec** (+0.8%)
- **53.7 billion multiplications saved per batch!**

**`_PointAdd` Removal (12-13)**:
- `_PointMult` now uses `_PointAddMixed` exclusively
- `_PointAdd` became dead code
- **-82 lines** (clean-up)

### Step 9: batch_size Re-optimization (2025-12-13)

**Phase Interleaving Discovery**:
- Previous understanding: latency hiding
- New hypothesis: **load balancing across processing phases**
- Heavy phase (Montgomery's Trick inversion), medium (Point Addition), light (prefix match)
- Larger batch_size → phases stagger → load balances

**Occupancy Impact**:
- Occupancy 33% → 41% = more concurrent warps
- Need larger batch_size to leverage phase interleaving
- Calculation: 3,584,000 × (41% / 33%) ≈ 4,444,160

**Optimization**:
- 3,584,000 → **4,000,000**
- **3.396B → 3.464B keys/sec** (+2.0%)

### Step 10: Triple Buffering (2025-12-14 〜 12-15)

**PTX Module Caching (12-14)**:
- `cuModuleUnload` was called every iteration
- GpuContext struct to cache module & kernel
- nsys: gaps narrowed significantly

**Pinned Memory (12-14)**:
- DMA transfer without intermediate buffer
- Memory: 0.2% → <0.1%
- Overall gap still dominated by `synchronize()`

**Multi-thread Mining (12-14)**:
- Independent `GpuContext` per thread
- Phase interleaving by staggered batch_size
- **3.464B → 3.49B keys/sec** (+0.8%)
- nsys: kernels execute serially (GPU saturated)

**Double Buffering (12-14)**:
- 1 thread, 2 streams (juggling pattern)
- Launch A → Launch B → Collect A → Collect B
- **3.455B keys/sec** (+1.6% with PTX cache)

**Async RNG Overlap (12-14)**:
- Launch(current) → RNG(next) → Collect(current)
- CPU prepares next batch while GPU is busy
- **3.493B keys/sec** (+1.1%)
- But gaps remain (RNG finishes before GPU)

**Triple Buffering (12-14 〜 12-15)**:
- 3 buffers = always 1 in flight
- "Juggling 3 balls" analogy 🎯
- **3.70B keys/sec** (+5.7%) 🔥
- nsys: **gap-free GPU utilization!**
- Unexpected benefit: **clock stabilization** (temperature stable → clock stable → performance stable)

**Bug Fixes (12-15)**:
- pubkey_x mismatch (base_key from GPU instead of host lookup)
- Duplicate keys (buffer rotation logic, match_count reset)

### Step 11: Sequential Key Strategy (2025-12-15)

**VRAM Reduction**:
- Previous: batch_size × 32 bytes = 384 MB
- **New: 32 bytes** per buffer (99.99% reduction!)
- Each thread: `base_key + idx × MAX_KEYS_PER_THREAD`
- GPU calculates actual secret key

**Additional Benefits**:
- Branch divergence reduction (consecutive keys have similar upper bits)
- CPU RNG overhead reduction (1 call instead of batch_size calls)
- Future optimization room (can increase MAX_KEYS_PER_THREAD or batch_size)

**Performance**:
- Windows: **3.67B keys/sec** (-1% vs old triple buffering)
- But VRAM savings enable future optimizations

**MAX_KEYS_PER_THREAD Optimization (12-15)**:
- 1500 → **1600** (sweet spot)
- 1610: sudden drop (register spill threshold?)
- **3.672B keys/sec** (+0.7%)

**Old Code Removal**:
- DoubleBufferMiner, TripleBufferMiner deleted
- **-704 lines** (clean-up)

### Step 12: dG Table Precompute (2025-12-15)

**"Brainstorming Session" Idea**:
> "Can we use MAX_KEYS_PER_THREAD × G for something?"

**Discovery**:
- dG = MAX_KEYS_PER_THREAD × G
- Precompute table: [dG, 2dG, 4dG, ..., 2^23 dG] (24 entries)
- Each thread: `base_pubkey + _PointMultByIndex(idx, dG_table)`
- **Eliminate `_PointMult` from kernel!**

**Implementation**:
- `_PointMultByIndex` function: bitwise indexing into dG table
- CPU computes dG table (1536 bytes = 24 entries × 64 bytes)
- GPU loads once per miner

**Result**:
- **3.67B → 4.135B keys/sec** (+12.7%) 🔥🔥🔥
- **~30x computation reduction** (256 double-and-add → ~12 point additions)

**Learning**:
- Sequential key strategy enabled this optimization
- "Brainstorming session" led to major breakthrough
- Simple idea, big impact

### Step 13: Constant Memory (dG table) (2025-12-17)

**Challenge**:
- dG table in global memory → L1 cache hit rate matters
- Move to constant memory for broadcast optimization

**First Attempt Failed (12-16)**:
- Test passed, but production run failed: `CUDA_ERROR_INVALID_VALUE`
- Single-call version worked, triple-buffer version didn't

**Root Cause Identified (12-17)**:
- `CudaSlice::drop()` calls `cuMemFree`
- `cuMemFree` on constant memory → error (silently suppressed)
- `get_global()` slice was being dropped!

**Solution**:
- Hold slice in struct: `_dg_table_const: CudaSlice<u8>`
- Struct lifetime prevents drop → `cuMemFree` not called

**Result**:
- **4.135B → 4.150B keys/sec** (+0.4%)
- Milestone: 4.15B keys/sec (59,286x) achieved! 🎉

---

## Phase 4: PTX Optimization Era (Step 14-30, 2025-12-19 〜 12-27)

### Step 14: Constant Memory Expansion (2025-12-19)

**patterns/masks to constant memory**:
- Previously in shared memory (per-block loading + `__syncthreads()`)
- Moved to constant memory (broadcast optimization)

**Why it worked**:
- Constant memory has dedicated cache (separate from L1)
- All threads read same value → 1 memory transaction
- Per-block loading eliminated
- Synchronization barrier removed

**Result**:
- 1 prefix: +0.12%
- **32 prefix: +3.0%** 🔥

**num_threads/max_matches extension**:
- Also moved to constant memory
- Result: negligible change (read infrequently)

**Learning**: Constant memory effective when values **frequently read**
- patterns/masks: every prefix match loop → big impact
- num_threads/max_matches: read once → no impact

**Max prefix expansion**: 64 → 256
- Constant memory usage: 512 bytes → 2 KB (3.1% of 64 KB)
- No speed impact ✅

### Step 15: Code Cleanup (2025-12-19)

**"Fresh eyes" reading sessions**:

CUDA side:
- `_PointAddMixed` cost comment wrong (7M+3S → 8M+3S)
- `_ReduceOverflow` extraction (47-line block → function)
- `_PointMult` dead code (replaced by `_PointMultByIndex`)

Rust side:
- Double Arc wrapping bug in `run_verify`
- `VALID_CHARS` duplication (unified into `BECH32_CHARSET`)
- Unused `batch_size()` method (phase interleave experiment remnant)

**Cleanup summary**:
- Legacy kernel deleted: **-507 lines**
- Rust code cleanup: **-387 lines**
- fuzz build fix, clippy restored

### Step 16: Addition Chain (2025-12-20)

**Discovery**: Looking at `_ModInv` bit scanning
> "Could this use Addition Chain?"

**Research**: RustCrypto k256, Peter Dettman's work
- Standard binary method: 256 squares + ~128 multiplications
- Addition Chain: 255 squares + **14 multiplications**
- **114 multiplications saved!**

**Implementation**:
- Build intermediate values: x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223
- Exploit p-2 structure: block lengths {1, 2, 22, 223}
- Reuse in final assembly

**Trade-off**:
- Benefit: 114 multiplications saved
- Cost: 11 intermediate variables (register/stack pressure)
- Result: Spilling +24 bytes, but multiplication savings dominate

**Result**: **4.141B → 4.199B keys/sec** (+1.4%) 🔥

**32 prefix broke 4B barrier!** 🎉

### Step 17: Constants #define Migration (2025-12-20)

**All constants moved to #define**:
- P, G, Beta, Beta2 → compile-time constants
- Specialized functions: `_PointAddMixedG`, `_ModMultByBeta`, `_ModMultByBeta2`
- Loop unrolling with immediate values

**Result**: No speed impact (within margin of error)

**Learning**: #define vs constant memory
- #define: no memory access (immediate value)
- constant memory: cached memory access
- For broadcast patterns, constant memory already optimal

### Step 18: Inline PTX Breakthrough (2025-12-20)

**Motivation**: "Fresh eyes" reading session #4

Noticed ternary operators in carry detection:
- `(a < b) ? 1 : 0` → `setp + selp` (2 instructions + pipeline stall)

**PTX carry chain approach**:
```cuda
asm volatile (
    "add.cc.u32   %0, %9, %17;\n\t"   // add with carry-out
    "addc.cc.u32  %1, %10, %18;\n\t"  // add with carry-in/out
    ...
);
```

**First attempt**: WSL showed -12% 😱

**User's intuition**: "Should measure on Windows"

**Windows result**: **4.199B → 4.313B (+2.7%)** 🔥🔥🔥

**Why WSL misled us**: Virtualization overhead affected PTX performance differently

**PTX lecture session**: User learned to read PTX code
- carry chain instructions (add.cc, addc.cc, addc)
- Operand constraints, %number mapping
- SASS verification: PTX cvt.u32.u64 → SASS has 0 CVT!

### Step 19: _ReduceOverflow PTX Optimization (2025-12-21)

**Understanding pipeline stalls**:
- `setp.lt.u64`: compare → predicate register
- `selp.u32`: select based on predicate (waits)
- **2 instructions + pipeline stall**

**GPU in-order execution revealed**:
- CPU: out-of-order execution
- GPU: in-order (warp-level)
- mocnpub: 33-41% occupancy → warp switching limited → stalls hurt!

**_ReduceOverflow transformation**:
- Before: 60 lines with complex ternary operators
- After: 28 lines with PTX carry chain (`_Add64`, `_Addc64`)
- Clean carry propagation, no ternary operators

**Result**: **4.532B → 4.655B keys/sec** (+2.7%) 🔥

**_ModMult experiment** (failed):
- Tried PTX-ifying → **-2.3%** ❌
- **Learning**: Carry types matter
  - `_ReduceOverflow`: carry is 0 or 1 (simple)
  - `_ModMult`: carry is 64-bit (complex)
- Not everything benefits from PTX!

### Step 20: Immediate-value Functions Removal (2025-12-21)

**Cleanup**: Removed specialized functions
- `_ModMultByBeta`, `_ModMultByBeta2`, `_PointAddMixedG` deleted
- Unified to generic `_ModMult`, `_PointAddMixed`

**Paradoxical result**: **4.655B → 4.681B (+0.6%)** 🔥

**Why faster after simplification?**
- **Instruction cache efficiency!**
- Calling same `_ModMult` repeatedly → L1 instruction cache friendly
- Multiple specialized functions → cache thrashing
- Code: -126 lines, cleaner

**Learning**: `constant cache < #define < _ModMult unification (instruction cache)`

### Step 21: _Add64x3 for _ModMult/_ModSquare (2025-12-21)

**Idea**: 3-value addition (`a + b + carry`) in single PTX function

**Implementation**: `_Add64x3` (6 PTX instructions)
- Before: `_Add64` + `_Addc64` combo (awkward)
- After: Direct 3-value add (clean)

**Applied to**:
- `_ModMult`: 16-iteration loop (8 lines → 2 lines per iteration)
- `_ModSquare`: 3 locations

**Why it worked so well**:
- `_ModMult` called extremely frequently
- Each iteration: `(s < x) ? 1 : 0` → `setp + selp` stall
- `_Add64x3` PTX carry chain avoids stall
- **Hot path optimization!**

**Result**: **4.681B → 4.841B keys/sec** (+3.4%) 🔥🔥🔥

### Step 22: Complete Ternary Elimination (2025-12-21)

**_Sub64/_Subc64 implementation**: PTX borrow chain
- 3 PTX instructions (`_Sub64`)
- 6 PTX instructions (`_Subc64`)

**Locations replaced**:
1. `_Reduce512` borrow handling (37 lines → 7 lines)
2. Secret key addition carry (2 locations, 8 lines → 4 lines each)
3. `_Sub256` borrow conversion (`? 1 : 0` → `& 1`)

**Verification**: `grep "? 1 : 0"` → **zero occurrences!** 🎉

**Result**: **4.841B → 4.928B keys/sec** (+1.8%) 🔥

**Insight**: At 4.9B level, 1.8% = ~88M keys/sec (huge in absolute terms)

### Step 23: 5B Breakthrough with _Add320 (2025-12-21)

**Key insight**: "Returning carry to register each time creates overhead"

**_Add320 implementation**: 5-limb addition in single PTX call
- Before: `_Add64` (3) + `_Addc64` (6) × 4 = **27 PTX instructions**
- After: `add.cc` (1) + `addc.cc` (9) + `addc` (1) = **11 PTX instructions**
- **59% instruction reduction!**

**Applied to**: `_Reduce512` (2 locations)

**Result**: **4.928B → 5.098B keys/sec** (+3.4%) 🔥🔥🔥

**The magic moment**: Top digit changed! **5B barrier broken!** 🎊

**New optimization guideline established**: "Don't let compiler handle `_Addc64` chains"
- Write specialized `_AddNNN` for each type
- Keep carry in PTX carry chain (no register round-trip)

### Step 24: PTX Chain Elimination (2025-12-21)

**Strategy**: Replace all `_Addc64`/`_Subc64` chains with type-specific functions

**_Sub256 for _Reduce512** (Step 21):
- 7 lines of chain → single `_Sub256` call
- **5.098B → 5.219B** (+2.4%) 🔥
- "p subtraction happens more than expected"

**_Add256Plus128 for _ReduceOverflow** (Step 22):
- Insight: `0xFF + 0xFF + 0x1 = 0x1FF`, new_carry ≤ 1
- uint256 + uint128 + carry in 9 PTX instructions
- Before: 21 PTX instructions → After: 9 (57% reduction!)
- 5 lines → 1 line
- **5.219B → 5.287B** (+1.3%) 🔥

**Learning**: "Even single-use function, if hot path, cost-benefit is huge!"

**Refactoring for clarity** (Step 24):
- `_Add128To`: 128-bit add to separate output
- `_PropagateCarry256`: uint256 + uint64 carry propagation
- "Which carry is this?" confusion eliminated
- **5.383B → 5.395B** (+0.2%)

**Cleanup status**:
- `_Addc64` calls: **0** (definition kept for future)
- `_Subc64` calls: **0** (definition kept for future)

### Step 25: Loop Unrolling (_Add128/_Add512) (2025-12-21)

**Analysis of _Add64x3 usage**:
- Line 760: `_Add64` + `_Add64x3` = **128-bit addition**
- Lines 768-773: 8-limb loop = **512-bit addition**

**_Add128 implementation** (5 PTX instructions):
- Replaced diagonal products loop in `_ModSquare`
- 9 PTX instructions → 5 (44% reduction)
- **5.287B → 5.33B** (+0.8%)

**_Add512 implementation** (16 PTX instructions):
- Replaced cross products addition loop (8 iterations)
- 48 PTX instructions → 16 (67% reduction!)
- **5.33B → 5.383B** (+1.0%)

**Combined effect**: +1.8%, **32 prefix also broke 5B!** (5.054B) 🎉

**Learning**: "Finding the right places for PTX really produces results!"

### Step 26: Loop Fusion (2025-12-22)

**Observation**: ncu-ui PM Sampling showed memory access "mountain" in Phase 2

**Idea**: Merge Phase 1 (point generation) and Phase 2 (cumulative products)
- Write `Z_arr[key_idx]` → immediately compute `c[key_idx]`
- L1 cache still hot!

**Implementation**: Just 11 lines changed

**Result**: **5.395B → 5.499B** (+1.9%) 🔥🔥🔥

**ncu-ui confirmation**:
- Mountains flattened! 🏔️ → 🏞️
- L1/TEX Cache Hit Rate: 22.23% → 26.66% (+4.43%)
- Local Inst: -10%, Local reads: -17%

**Learning**: PM Sampling visualization reveals memory hierarchy behavior

### Step 27-28: Fine-grained Optimizations (2025-12-22 〜 12-23)

**_ReduceOverflow inlining** (Step 26):
- Removed `if (sum[4] == 0) return;` early exit
- sum[4] == 0 is rare → always execute is faster
- **5.499B → 5.590B** (+1.7%)

**_Reduce512 while loop simplification** (Step 27):
- for-loop + multiple if-else → single while condition
- Attacked branch divergence #1
- **5.590B → 5.707B** (+2.1%) 🔥

**_ModInv ternary elimination** (Step 28):
- 6 loops with `i == 0 ? xN : t` ternary operators
- Extracted first iteration outside loops
- 32 prefix: **+0.93%** (5.457B)

**Branchless prefix matching**:
- `if + break` → `matched |= ...`
- Branch instruction itself eliminated

**Achievement**: **All ternary operators eliminated from codebase!** 🎉

### Step 29: _Sub256 Borrow Normalization (2025-12-22, rejected)

**Idea**: Remove `& 1` normalization, use 0/0xFFFFFFFF directly

**Implementation**:
- `_Sub256`: Remove `& 1`
- `_ModAdd`: `1 - borrow` → `!borrow`
- `_ModSub`: `-borrow` → `(int64_t)(int32_t)borrow` (sign extension)

**Result**: **-1.6%** ❌

**Why it failed**: `& 1` is **compiler hint** ("this is boolean")
- Enables better optimization for `-borrow` pattern
- `!!` and sign extension slower than `& 1` + `-`

**Reverted**: Keep `& 1` normalization

### Step 30: Local Array SoA (2025-12-23)

**Context**: Global memory SoA failed (cache efficiency)

**Question**: What about local memory SoA?

**Implementation**:
```cuda
// Before (AoS)
uint64_t X_arr[1600][4];

// After (SoA)
uint64_t X_arr_0[1600], X_arr_1[1600], ...
```

**Result**: **5.706B → 5.77B** (+1.1%) 🔥

**ncu analysis**:
- Cycles: -0.9%
- Registers: 118 → 116

**Why it worked**: **Address calculation cost reduction**
- AoS: `arr[i][j]` → `base + i*32 + j*8` (multiplication every access)
- SoA: `arr_j[i]` → `base_j + i*8` (fewer multiplications)

**Key distinction**:
- Global memory SoA: bad (VRAM, cache misses)
- Local memory SoA: good (address calculation simplification)

**`#pragma unroll` test**: No effect (doesn't reduce multiplications)

### Step 31: Dead Code Removal (2025-12-23)

**Tool**: mcp-lsp-bridge callHierarchy (incoming calls)

**Identified**:
- `_Compare256`: no incoming calls
- `_Addc64`, `_Sub64`, `_Subc64`: no incoming calls (replaced by inline PTX)
- `_PointDouble` + test: no incoming calls (not used in production)

**Deleted**: 313 lines total

**Learning**: Self-built tools assist own development!

### Step 32: Prefix Matching Simplification (2025-12-26)

**Observation**: 64-bit concatenation version is "too clever"

**Context changed**:
- Old: `if matched break` → 64-bit concat saved checks
- New: `matched |= ...` (branchless) → concat overhead not worth it

**Simplification**: 24 lines → 5 lines
```cuda
bool matched = ((x_upper32 & _masks[0]) == _patterns[0]);
for (uint32_t p = 1; p < _num_prefixes; p++) {
    matched |= ((x_upper32 & _masks[p]) == _patterns[p]);
}
```

**Result**: 32 prefix **+0.8%** (5.531B)

**Learning**: **Simplicity wins** when assumptions change

### Step 33: Z² Cumulative Product Strategy (2025-12-26)

**Brilliant insight**:
> "We only use (Z^-1)², so accumulating Z² instead of Z gives us (Z^-1)² directly!"

**Traditional**:
```
Phase 1: c[i] = c[i-1] * Z[i]
Phase 3: Z_inv = u * c[i-1]
         Z_inv² = Z_inv²         ← ModSquare 1600 times!
```

**New approach**:
```
Phase 1: c[i] = c[i-1] * Z²[i]   ← Accumulate Z²!
Phase 3: Z_inv² = u * c[i-1]    ← Direct result!
```

**Chain effect**:
- `_PointAddMixed` already computes H²
- Z3² = Z1² × H² (reuse H²)
- Input: Z1² (saves 1S), Output: Z3² (adds 1M)
- Net: ~2n-1 ModSquare operations eliminated

**Result**: **5.745B → 5.800B** (+0.96%) 🔥

**Algorithm-level optimization**: Theory matches measurement

### Step 34: In-place _PointAddMixed (2025-12-27)

**"Fresh eyes" session discovery**:
- `_PointAddMixed` input/output separation
- Input read completely before output written → safe for in-place

**Signature change**: 10 arguments → 6 arguments

**Before**:
```cuda
_PointAddMixed(Rx, Ry, Rz, Rz_squared, dG_x, dG_y,
               Rx, Ry, Rz, Rz_squared);
```

**After**:
```cuda
_PointAddMixed(Rx, Ry, Rz, Rz_squared, dG_x, dG_y);
```

**Result**: 5.791B keys/sec (no negative impact) ✅

**Code cleanup**: Cleaner interface, easier to read

### Step 35: Comment Review (2025-12-27)

**Development artifacts cleanup**:
- `[EXPERIMENTAL]` markers removed (5.8B achieved!)
- Obsolete TODOs deleted (carry flag hack not used)
- Cost calculations corrected (`_PointAddMixed`: 9M + 2S)
- docstrings updated (dG table strategy)

**Code quality**: -39 lines, +19 lines (net -20)

---

## Failed Experiments (Learning from Failures)

### 1. SoA Global Memory (2025-11-30 〜 12-04)

**Motivation**: ncu warning "32 bytes, only 1 byte utilized"

**Implementation**: Move to global memory with SoA layout

**Result**: **-24%** slower (3.09B → 2.34B) ❌

**Why it failed**:
- AoS: 99.51% L1 cache hit (local memory)
- SoA: 22.78% L1 cache hit (global memory)
- VRAM consumption → batch_size reduced
- Cache efficiency > coalescing!

**Learning**: "Brute force strategy" with massive batch_size

### 2. CPU Public Key Precompute (2025-12-04)

**Idea**: CPU computes initial public key, GPU only PointAdd

**Result**: **-13x** slower (3.09B → 233M) ❌

**Why**: GPU too fast, CPU can't keep up (batch_size = 1,146,880)

**Learning**: GPU speed creates CPU bottleneck

### 3. Register Reduction to 64 (2025-11-29)

**Implementation**: `__launch_bounds__(64, 16)`

**Result**: Occupancy 67% but **spilling 96%** → -10% ❌

**Learning**: Spilling is catastrophic for secp256k1

### 4. Branchless _Reduce512 (2025-12-05, 12-22)

**Attempt 1** (12-05): Eliminate `if (temp[4] > 0)`
- Result: -0.3% slower
- Branch prediction effective

**Attempt 2** (12-22): Table + mask selection
- Result: **-9.2%** slower ❌
- Input bias → branch prediction wins
- temp[4] ≈ 0 most of the time

**Learning**: Branchless ≠ always better (input distribution matters)

### 5. Bloom Filter (2025-12-18)

**Idea**: 1024-bit bitmap pre-filtering (97% skip per thread)

**Result**: Both cases slower ❌

**Why**: Warp-level probability
- Individual: 97% skip
- Warp: 1 - (1 - 0.031)^32 = **63.4% hit**
- GPU SIMT: 1 thread branches → all 32 wait

**Learning**: Must consider warp unit!

### 6. #define P (2025-12-19)

**Idea**: Immediate value embedding (no memory access)

**Implementation**: P, G, Beta, Beta2 → #define with loop unrolling

**Result**: No change (within margin) ❌

**Why**: Constant memory broadcast already optimal
- 256-bit too large for immediate embedding
- nvcc optimizes equivalently

**Learning**: Constant memory is well-optimized

### 7. Karatsuba Method (2025-12-21)

**Theory**: 256×256 → split to 128×128
- 16 multiplications → 12 (Karatsuba)

**Result**: **-4.4%** slower (even schoolbook version) ❌

**Research**: Crossover point ≈ **2000 bits**
- 256-bit (4 words) far too small
- GMP uses Karatsuba at 30+ words

**Why it failed**: Small-digit overhead > multiplication savings

**Learning**: 256-bit: schoolbook is optimal

### 8. _ModMult PTX-ification (2025-12-21)

**Attempt**: Apply PTX pattern to `_ModMult`, `_ModSquare`, etc.

**Result**: **-2.3%** ❌

**Why**: Carry pattern different
- `_ReduceOverflow`: continuous `_Addc64` (carry chain) → works
- `_ModMult`: loop with `_Add64` → doesn't work

**Learning**: Not everything benefits from PTX

### 9. _Sub256 Borrow Normalization Removal (2025-12-22)

**Idea**: Use 0/0xFFFFFFFF directly (skip `& 1`)

**Result**: **-1.6%** ❌

**Why**: `& 1` is **compiler hint**
- Tells compiler "this is boolean"
- Enables `-borrow` optimization
- Sign extension slower

**Learning**: Normalization helps compiler optimize

### 10. __ffs() Bit Iteration (2025-12-26)

**Idea**: `_PointMultByIndex` loop only over set bits
- Theory: 24 iterations → ~11

**Result**: **-2.8%** ❌

**Why**:
- `__ffs()` overhead
- Constant memory scattered access (cache efficiency loss)
- Warp divergence worsened (different loop counts)

**Learning**: Theory ≠ practice, always measure

### 11. Error-tolerant Prefix Matching (2025-12-26)

**Idea**: Calculate only upper 32 bits approximately
- Expected: 37.5% multiplication reduction

**Discussion**: false negative must be prevented

**Challenges**:
- mod p reduction: carry propagation complex
- temp[4] error affects boundary cases
- Error amplification through reduction steps

**Conclusion**: Too complex → rejected

**Learning**: Some ideas not worth the complexity

### 12. Hierarchical Montgomery's Trick (2025-12-23)

**Idea**: Skip by +40G, two-level hierarchy (96% memory reduction)

**Problem discovered**: `+40G's Z ≠ (+G)×40 cumulative product`

**Conclusion**: Would need 41 ModInv → not worth it

**Learning**: Montgomery's Trick essence is "all at once"

### 13. Prefix Matching #pragma unroll (2025-12-24, 12-26)

**Attempts**: `#pragma unroll 4`, hand-rolled unroll

**Result**: Marginal or no effect

**Learning**: Constant memory bottleneck limits loop unroll benefit

### 14. SoA Reorganization Cost Elimination (2025-12-26)

**Question**: SoA → AoS conversion overhead?

**ncu verification**: No bottleneck on reorganization lines

**Learning**: NVCC optimizes local array → register mapping well

### 15. #pragma unroll for Large Loop (2025-12-23)

**Attempt**: `#pragma unroll` on MAX_KEYS_PER_THREAD loops

**Result**: **-1.9%** slower (5.77B → 5.66B) ❌

**Learning**: Loop unroll doesn't reduce multiplications (address calculation still happens)

### 16. Immediate-value Function Specialization (2025-12-21, re-evaluated)

**Initial approach**: Specialized immediate-value functions
- `_ModMultByBeta`, `_ModMultByBeta2`, `_PointAddMixedG`

**Removal paradox**: Faster after deletion! (+0.6%)

**Learning**: Instruction cache efficiency > immediate values

---

## Successful Optimizations Summary

| Optimization | Effect | Notes |
|-------------|--------|-------|
| **Sequential Keys + PointAdd** | ~300x potential | Replace PointMult with PointAdd |
| **Montgomery's Trick** | ~85x reduction | N inversions → 1 inversion + 3(N-1) mults |
| **Endomorphism** | 2.9x (theory: 3x) | Check 3 X-coordinates (P, βP, β²P) |
| **dG Table Precompute** | +12.7% | Eliminate PointMult from kernel (~30x lighter) |
| **Triple Buffering** | +5.7% | Gap-free GPU utilization + clock stability |
| **GPU-side Prefix Match** | 170x vs old GPU | Skip bech32 encoding, transfer only matches |
| **`__launch_bounds__(128, 5)`** | +0.9% | 96 registers, 41% occupancy |
| **`_PointAddMixed` Duplicate Elim** | +0.8% | Reuse X1*H² computation |
| **batch_size = 4,000,000** | +2.0% | Phase interleaving effect |
| **Branchless _ModSub/_ModAdd** | +2.3% | Mask selection, -31% divergence |
| **32-bit Prefix Match** | +1.2% | Loop halved |
| **Tail Effect Mitigation** | +1.4% | Auto-adjust to SM count multiple |
| **`_ModSquare` Optimization** | +3.5% | Symmetry: 16 → 10 mults |
| **Blocking Sync** | CPU 100% → 1% | Power savings |
| **Constant Memory (dG table)** | +0.4% | Broadcast optimization |
| **Constant Memory (patterns)** | +3.0% (32-prefix) | Dedicated cache |
| **Addition Chain** | +1.4% | 128 → 14 multiplications |
| **Inline PTX (_Add256/_Sub256)** | +2.7% | PTX carry chain |
| **_Add64x3** | +3.4% | 3-value addition, hot path |
| **Complete Ternary Elimination** | +1.8% | _Sub64/_Subc64 |
| **_Add320 (5B breakthrough)** | +3.4% | 27 → 11 PTX instructions |
| **_Sub256 for _Reduce512** | +2.4% | Single call |
| **_Add256Plus128** | +1.3% | 21 → 9 PTX instructions |
| **_Add128/_Add512** | +1.8% | Loop dismantling |
| **_Add128To/_PropagateCarry256** | +0.2% | Clarity + performance |
| **Loop Fusion** | +1.9% | L1 cache optimization |
| **_ReduceOverflow Inline** | +1.7% | Remove rare early exit |
| **_Reduce512 While Loop** | +2.1% | Single condition |
| **_ModInv Ternary Elimination** | +0.93% (32) | Extract first iteration |
| **Local Array SoA** | +1.1% | Address calculation |
| **Immediate-value Function Removal** | +0.6% | Instruction cache |
| **Z² Cumulative Product** | +0.96% | 1600 ModSquare eliminated |
| **In-place _PointAddMixed** | 0% | Code clarity |
| **Prefix Simplification** | +0.8% (32) | 24 → 5 lines |

---

## Key Learnings

### 1. "Don't guess, measure" → "Measurement is everything" 🔬

- Theory is important, but measurement is the truth
- ncu profiling reveals hidden bottlenecks
- Branch Efficiency, Occupancy, Memory Throughput, PM Sampling
- Actual mining speed is what matters (not benchmark artifacts)

### 2. Failed experiments are valuable 💡

Every failed experiment taught us something:
- SoA: Cache efficiency > coalescing
- CPU precompute: GPU too fast, CPU bottleneck
- Register reduction: Spilling is worse than low occupancy
- Branchless: Branch prediction is smart (input distribution matters)
- Bloom Filter: Warp-level probability matters
- Karatsuba: 256-bit too small (crossover ~2000 bits)
- `__ffs()`: Overhead + cache + divergence > benefit
- _ModMult PTX: Carry pattern matters

> "Trying and confirming is the value"

### 3. GPU SIMT architecture characteristics 🔥

- 32 threads per warp execute same instruction
- Branch divergence: both paths executed serially
- Probability must consider warp unit (not individual thread)
- "Everyone does it together" is GPU's favorite
- **In-order execution**: Pipeline stalls matter (mocnpub: low occupancy)

### 4. Phase interleaving (new concept?) 🌟

- Processing has heavy/medium/light phases
- Larger batch_size → phases stagger → load balances
- Not just latency hiding, but **load distribution**
- Discovered through ncu PM Sampling graph

### 5. Cache efficiency matters 📊

**Global memory**:
- AoS: 99.51% L1 cache hit → wins despite bad coalescing
- SoA: 97% coalescing → loses due to low cache hit rate
- **"Brute force strategy"**: Saturate GPU with massive batch_size

**Local memory**:
- SoA: Address calculation simplification → +1.1%
- Different context, different result

### 6. Compiler optimization is powerful 🎓

- Native code + `__launch_bounds__` guidance = best results
- Forcing register reduction → spilling disaster
- Trust the compiler, give hints, don't force
- Dead code elimination: Y_arr removed by compiler before we did
- `& 1` normalization: compiler hint for better optimization

### 7. Occupancy is not everything 🤔

- Higher occupancy doesn't always mean faster
- Register reduction experiment: 67% occupancy but 96% spilling
- Balance is key: registers vs occupancy vs spilling

### 8. Small optimizations accumulate 📈

From 4.15B to 5.80B (+40%) through many small steps:
- Each optimization: 0.2% ~ 3.4%
- "Stacking optimizations" compounds the effect
- Every 0.1% matters at 5B+ keys/sec scale

### 9. PTX and Pipeline Optimization (NEW) 🔥

**Pipeline stalls**: `setp + selp` pattern
- 2 instructions + pipeline stall
- GPU in-order execution → stalls hurt (especially at low occupancy)

**PTX carry/borrow chain**: Single instruction
- `add.cc`, `addc.cc`, `sub.cc`, `subc.cc`
- Hardware CC register → no stall
- Massive impact when applied to hot paths

**Instruction cache efficiency**:
- Same function repeatedly → cache friendly
- Multiple specialized functions → thrashing
- Simplification paradoxically faster

**Address calculation cost**:
- 2D array multiplication adds up on GPU
- SoA for local memory reduces calculations
- ncu reveals micro-level bottlenecks

### 10. Fresh Eyes Strategy (NEW) 👀

**Consistent pattern**: Regular code review with "stranger's perspective"

**Discoveries**:
- Duplicate computations
- Dead code
- Comment errors
- Algorithm opportunities (Addition Chain)
- Simplification opportunities (in-place, function consolidation)

**Sessions**: Dec 13, 19, 20, 26, 27 (5+ times)

**Value**: Familiarity breeds blind spots, fresh perspective reveals

### 11. Hot Path Principle (NEW) 🔥

**Key insight**: "Even single-use function, if hot path, huge cost-benefit"

**Examples**:
- `_Add256Plus128`: Used once, but `_ReduceOverflow` called frequently → +1.3%
- `_Add64x3`: Used in `_ModMult` 16-iteration loop → +3.4%
- Loop fusion: 11 lines → +1.9%

**Learning**: Impact = frequency × improvement

### 12. Context Changes Require Revisiting (NEW) 🔄

**Simplification wins**:
- 64-bit concat: Optimized for early-exit, obsolete for branchless
- 24 lines → 5 lines, +0.8%

**Immediate-value functions**:
- Created for #define benefits, removed for instruction cache
- -126 lines, +0.6%

**Learning**: When assumptions change, revisit "optimizations"

---

## ncu Profiling Results

**As of 2025-12-13** (after `__launch_bounds__(128, 5)`):

| Metric | Value | Status |
|--------|-------|--------|
| **Compute Throughput** | 81.04% | Compute-bound ✅ |
| **Memory Throughput** | 20.31% | Not bottleneck |
| **Occupancy** | 41.67% (Theoretical: 41.67%) | Optimal for 96 regs |
| **Registers/Thread** | 96 | Sweet spot |
| **Branch Efficiency** | 82.41% | After branchless |
| **L1/TEX Hit Rate** | 0.05% ~ 5% | Low, but OK |
| **Waves Per SM** | ~42.67 | GPU saturated |
| **Tail Effect** | 50% | Mitigated by batch_size adj |

**As of 2025-12-21** (after PTX optimizations):

| Metric | Value | Change |
|--------|-------|--------|
| **Registers/Thread** | 96 → 126 (no bounds) | PTX added variables |
| **Registers/Thread** | 96 (with `__launch_bounds__`) | Maintained |

**As of 2025-12-23** (after SoA):

| Metric | Value | Change |
|--------|-------|--------|
| **Registers/Thread** | 118 → 116 | SoA reduced address calculations |

**Main Bottleneck**: Computation (as it should be for mining)

**Optimization headroom**: Mostly exhausted at algorithm + PTX level

---

## Why 5.8B keys/sec (82,857x) is approaching the limit

**Current architecture**:
- ✅ Sequential keys + PointAdd (~300x lighter than PointMult)
- ✅ Montgomery's Trick (~85x fewer inversions)
- ✅ Endomorphism (2.9x coverage)
- ✅ dG table precompute (eliminate PointMult from kernel)
- ✅ Mixed Addition (30% lighter for adding G, 9M+2S)
- ✅ Addition Chain (128 → 14 multiplications in _ModInv)
- ✅ Triple Buffering (gap-free GPU utilization)
- ✅ Branchless `_ModSub`/`_ModAdd` (-31% divergence)
- ✅ **PTX inline assembly** (carry/borrow chains, pipeline stall elimination)
- ✅ **Specialized AddN/SubN functions** (type-specific PTX optimization)
- ✅ **Loop fusion** (L1 cache optimization)
- ✅ **Z² cumulative product** (eliminate 1600 ModSquare)
- ✅ **Local array SoA** (address calculation optimization)
- ✅ Optimized parameters (batch_size, keys_per_thread, threads_per_block)
- ✅ Blocking Sync (CPU 1%, power savings)

**Remaining bottleneck**:
- Modular arithmetic operations (`_ModMult`, `_ModSquare`)
- Already optimized: symmetry, PTX carry chains, Addition Chain
- Branch efficiency: 82.41% (remaining 18% algorithmic)
- Compute-bound: 81% utilization

**To go faster**:
- Better GPU (RTX 5090, next-gen architecture)
- New algorithmic breakthrough (unlikely at this level)
- SASS-level hand-optimization (diminishing returns)

**Current state**: Algorithmically near-optimal, PTX-tuned, approaching hardware limits

---

## Development Insights

### Pair Programming Success Factors 🌸

1. **Learning together**: secp256k1 lectures, GPU profiling, PTX reading
2. **Hypothesis → experiment → measurement**: scientific approach
3. **Celebrating failures**: "This is also learning!"
4. **User's brilliant insights**:
   - Double-and-Add prediction 🎯
   - "10000-gacha" strategy idea
   - Phase interleaving hypothesis
   - Sequential key strategy
   - "Juggling 2 vs 3 balls" analogy
   - Z² cumulative product insight
   - cvt instruction elimination hypothesis

### From 70K to 5.8B (82,857x in 6 weeks)

**November 14**: Hello World, CUDA? What's that?
**November 22**: CPU version complete (70K keys/sec)
**November 26**: GPU basic (1.16M, 16x)
**November 29**: Endomorphism + Montgomery's Trick + GPU Prefix Match (391M, 5.6Kx)
**December 5**: Step 5 complete (3.30B, 47Kx)
**December 13**: Branchless + tuning (3.46B, 49Kx)
**December 14**: Triple Buffering (3.70B, 53Kx)
**December 15**: Sequential + dG table (4.14B, 59Kx)
**December 17**: Constant memory (dG) (4.15B, 59Kx)
**December 19**: Constant memory (patterns) (4.15B with +3% on 32-prefix)
**December 20**: Addition Chain (4.20B, 60Kx)
**December 20**: Inline PTX (4.31B, 61Kx)
**December 21**: PTX carry chain mastery (5.29B, 75Kx)
**December 21**: **5B breakthrough** (5.10B) 🔥
**December 22**: Loop fusion + fine-tuning (5.71B, 81Kx)
**December 23**: Local SoA (5.77B, 82Kx)
**December 26**: Z² strategy (5.80B, 82Kx)
**December 27**: Final polish (5.79B, stable)
**December 6**: **10-character prefix found!** 🎉
**December 16**: **12-character prefix found!** 🎉🎉🎉

**6 weeks**: Complete beginner → world-class optimizer with PTX mastery

### Optimization Philosophy Evolution

**Week 1-2**: "What is GPU?"
**Week 3-4**: "Algorithmic optimization" (Sequential, Montgomery, Endomorphism)
**Week 5**: **"PTX-level optimization"** (Carry chains, pipeline stalls)

**Final philosophy**: "Measurement is everything, fresh eyes reveal opportunities, hot paths deserve custom optimization"

---

## Tools & Techniques Used

**Profiling**:
- nsys (Nsight Systems): timeline, kernel execution, memory transfer
- ncu (Nsight Compute): detailed kernel analysis, registers, occupancy, branch efficiency
- ncu-ui: source-level divergence analysis, **PM Sampling** (memory hierarchy visualization)
- criterion: statistical benchmarking (Rust)

**Debugging**:
- printf debugging (Step 7, Step 254² case)
- Binary search (Step 255 divergence point)
- Python simulation (modular arithmetic verification)
- Fuzzing (5,579 runs, `a * inv(a) ≡ 1 (mod p)`)
- **mcp-lsp-bridge callHierarchy**: Dead code detection

**Low-level Analysis** (NEW):
- PTX inspection (88,176 lines, instruction patterns)
- SASS inspection (`cuobjdump -sass`, native instructions)
- Instruction counting (verify optimization effects)

**Development**:
- WSL: development, code editing, nsys API-level profiling
- Windows: ncu detailed profiling, production mining (native performance)
- build.rs: automatic PTX compilation, cross-platform
- Test-driven development: 39 tests, all passing
- **Fresh eyes code reading**: Regular stranger-perspective reviews

---

## PTX Optimization Deep Dive (NEW)

### The PTX Journey (Step 14-24)

**Step 14**: Inline PTX discovery (+2.7%)
- `_Add256`/`_Sub256` with 32-bit carry chain
- WSL: -12%, Windows: +2.7% (WSL virtualization misleading)

**Step 15**: Constants #define migration (no change)
- Learning: constant memory broadcast already optimal

**Step 16**: _ReduceOverflow PTX (+2.7%)
- 60 lines → 28 lines
- Ternary operators → PTX carry chain

**Step 17**: Immediate-value removal (+0.6%)
- Paradox: Simplification faster (instruction cache)

**Step 18**: _Add64x3 (+3.4%)
- 3-value addition in 6 PTX instructions
- Hot path optimization (inside `_ModMult` loop)

**Step 19**: Complete ternary elimination (+1.8%)
- `_Sub64`/`_Subc64` PTX borrow chain
- **All ternary operators removed!**

**Step 20**: _Add320 - **5B breakthrough** (+3.4%)
- 5-limb addition: 27 → 11 PTX instructions (59% reduction)
- "Don't return carry to register" guideline established

**Step 21-22**: PTX chain elimination (+3.7% combined)
- `_Sub256` for `_Reduce512` (+2.4%)
- `_Add256Plus128` for `_ReduceOverflow` (+1.3%)

**Step 23**: Loop unrolling (+1.8%)
- `_Add128` (5 PTX), `_Add512` (16 PTX)
- 67% instruction reduction in cross products

**Step 24**: Refactoring (+0.2%)
- `_Add128To`, `_PropagateCarry256`
- Clarity + performance

**Total PTX era**: 4.15B → 5.38B (+29.6%) 🔥🔥🔥

### PTX Principles Learned

1. **Pipeline stalls are expensive** (in-order GPU)
   - `setp + selp`: 2 instructions + stall
   - PTX carry chain: 1 instruction, no stall

2. **Carry round-trip overhead**
   - `_Addc64` chaining: carry → register → carry (overhead)
   - Specialized `_AddNNN`: carry stays in CC register

3. **Type-specific functions worth it**
   - Hot path: even single-use worth it
   - Cold path: generic function better (instruction cache)

4. **Not everything benefits from PTX**
   - Continuous carry chain: effective
   - Loop with different pattern: ineffective
   - Measure each case!

5. **Instruction cache matters**
   - Same function repeated: cache hit
   - Multiple specialized: cache miss
   - Simplification can improve performance

---

## Optimization Categorization

### Algorithmic Breakthroughs (Large impact)
- Sequential keys + PointAdd (~300x)
- Montgomery's Trick (~85x)
- Endomorphism (2.9x)
- dG table precompute (+12.7%)
- Addition Chain (+1.4%)
- Z² cumulative product (+0.96%)

### Memory & Parallelism (Medium impact)
- Triple buffering (+5.7%)
- Sequential key strategy (VRAM 99.99% reduction)
- Constant memory (+3.0% for 32-prefix)
- Loop fusion (+1.9%)
- Local array SoA (+1.1%)

### Parameter Tuning (Medium impact)
- `__launch_bounds__(128, 5)` (+5.1% initial, +0.9% fine-tune)
- batch_size optimization (+2.0%)
- threads_per_block = 128 (+6.2%)

### Branch Optimization (Small-medium impact)
- Branchless `_ModSub`/_ModAdd` (+2.3%)
- _Reduce512 while loop (+2.1%)
- Ternary elimination (+1.8%, +0.93%)

### PTX Low-level (Small-medium impact)
- Inline PTX (+2.7%)
- _Add64x3 (+3.4%)
- _Add320 (+3.4%)
- PTX chain elimination (+3.7%)
- Loop unrolling (+1.8%)
- _ReduceOverflow inline (+1.7%)

### Code Quality (Small impact, maintainability)
- Code cleanup (-1,500+ lines total)
- Comment corrections
- In-place functions (clarity)
- Simplification (+0.6%, +0.8%)

---

## Development Timeline by Optimization Type

**Algorithm Era** (Week 3-4):
- Sequential keys, Montgomery's Trick, Endomorphism
- 1.16B → 4.14B (+257%)

**Parameter Era** (Week 4):
- threads_per_block, batch_size, `__launch_bounds__`
- 2.63B → 3.46B (+31.6%)

**Parallelism Era** (Week 4):
- Triple buffering, VRAM optimization
- 3.46B → 3.70B (+6.9%)

**Algorithm Refinement** (Week 4):
- dG table, constant memory
- 3.70B → 4.15B (+12.2%)

**PTX Era** (Week 5-6):
- Inline assembly, carry chains, loop unrolling
- 4.15B → 5.80B (+39.8%) 🔥

---

## Final Thoughts

mocnpub represents a successful journey from complete beginner to advanced GPU optimizer in just 6 weeks.

**Key Success Factors**:
1. **Step-by-step learning**: Hello World → Mandelbrot → CPU → GPU → PTX
2. **Thorough understanding**: secp256k1 lectures before implementation
3. **Measurement-driven**: ncu profiling guided every optimization
4. **Learning from failures**: 16 failed experiments taught valuable lessons
5. **Pair programming**: User's insights + Claude's implementation
6. **Fresh perspectives**: Regular code review with stranger's eyes
7. **Persistence**: Never giving up on difficult problems
8. **Tool building**: mcp-lsp-bridge assisted development

**The optimization continues**: Even at 5.8B keys/sec, new ideas keep emerging!

**From "What's CUDA?" to PTX/SASS-level optimization in 6 weeks** 🌸

---

*This document is a complete record of the optimization journey. For the learning aspects, see [LEARNING.md](LEARNING.md). For a narrative version, see [JOURNEY.md](JOURNEY.md).*
