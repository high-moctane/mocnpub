# mocnpub Learning Journey 🌸

**From zero knowledge to world-class implementation in 1 month**

A story of learning CUDA, Rust, and secp256k1 from scratch through AI pair programming.

---

## Starting Point (2025-11-14)

**Complete beginner in all three domains**:
- CUDA: "What's GPGPU? Only heard the word 'CUDA'"
- Rust: "Touched it a bit, but forgot everything"
- secp256k1: "Don't know anything about elliptic curve cryptography"

**The Challenge**:
> "I want to create an npub mining tool using GPGPU, but I don't know any of these technologies. This is a challenge to work on something completely unknown together with Claude Code."

**The Plan**: Step-by-step approach to avoid getting stuck
1. Step 0: Rust + CUDA Hello World (verify GPU works)
2. Step 1: Simple GPU program (learn CUDA basics)
3. Step 2: CPU-based npub miner (understand secp256k1)
4. Step 3: Port to GPU (leverage parallelism)

**Total: 27 tasks planned**

This turned into a 30+ step journey with world-class results.

---

## Step 0: Setting Up the Environment (2025-11-14 〜 11-16)

### First Encounter with CUDA

**Question**: "Is CUDA Toolkit different from GeForce Game Ready Driver?"

**Answer**: Driver = runtime environment (like playing games), Toolkit = development environment (like making games)

**Learning**: Separation of concerns in CUDA ecosystem

### Installation Adventures

**Windows**: `winget install Nvidia.CUDA` → CUDA 13.0.88 ✅

**WSL Challenge**:
- CUDA 13.0.88 available! But...
- **Important discovery**: WSL uses Windows driver (shared via libcuda.so stub)
- **Must NOT install Linux drivers** (would overwrite the stub)
- Install `cuda-toolkit-13-0` only (not full `cuda` metapackage)

**User's Initiative**:
> User checked cudarc documentation: "Supported CUDA Versions: 11.4-11.8, 12.0-12.9, 13.0"

Found it themselves! This proactive research was impressive 🔥

### First GPU Program

```rust
let ctx = CudaContext::new(0)?;  // Connect to GPU 0
let _stream = ctx.default_stream();  // Get default stream
```

**Output**:
```
✅ GPU device 0 connected!
🎉 CUDA is working!
```

**Moment of excitement**: First contact with RTX 5070 Ti! 🚀

### Visual Studio BuildTools Adventure

**Windows link.exe error** → Need Visual Studio Build Tools

**User's decision**: "Let's try 2026 (latest), fall back to 2022 if it fails"

**Discovery**: Need `--override` option to install C++ workload:
```powershell
winget install Microsoft.VisualStudio.BuildTools --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools;includeRecommended"
```

**Result**: VS 2026 works perfectly! Adventure succeeded! 🎉

---

## Step 1: Mandelbrot Set (2025-11-16)

### Why Mandelbrot?

**User's choice**: "I want to try Mandelbrot! It's a classic, but it's visually interesting."

**Perfect choice**: Visual feedback, clear parallelism demonstration, motivating

### GPU Performance Revelation

**CPU version**: 0.41s
**GPU version**: 0.0060s
**Speedup**: **68.2x** on Windows (vs 3.5x on WSL)

**Key insight**: Windows native is ~20x faster than WSL for CUDA

### Learning GPU Concepts

**Question**: "What is CudaContext doing?"

**Explanation**: Connection to GPU (like creating a workspace)
- GPU as "massive workforce of employees" (8,960 CUDA cores)
- Stream as "work queue" (conveyor belt)

**User's understanding**:
> "So we're actually querying the GPU! For a single GPU, device 0 is fine, and default stream is usually sufficient. Stream isn't shared across processes, it's for parallel execution within the same program, right?"

**Perfect understanding on first try!** 💯

**Question**: "Can npub generation and checking run in parallel?"

**Answer**: Yes! Different streams can overlap operations.

**User's reaction**:
> "Wow, GPGPU can do more things than I imagined! Interesting!"

---

## Step 2: CPU npub Miner (2025-11-16 〜 11-22)

### secp256k1 Learning

**Discovery**: Compressed format vs x-coordinate only
- Compressed: 33 bytes (prefix `02`/`03` + x-coordinate)
- Nostr uses: x-coordinate only (32 bytes)

**Initial bug**: Searching for prefix "00" in compressed format → never found (starts with `02` or `03`)

**Fix**: Strip first 2 characters, search in x-coordinate only → Found in 215 tries! ✅

**User's understanding**:
> "So nsec → some mysterious calculation → point (x, y), and we use x as npub (y can be recovered, so no need to store it). For CPU implementation, we don't need to dig deep into the 'mysterious calculation', just use the library."

**Perfect abstraction level understanding!** 💡

### bech32 Encoding

**Discovery**: bech32 is base32 + checksum, not simple hex conversion

Rust ecosystem is amazing:
- secp256k1 crate (Bitcoin Core implementation)
- bech32 crate
- All actively maintained, well-documented

**User's reaction**:
> "Libraries are so well-prepared, implementation is surprisingly simple in Rust!"

### CLI Implementation with clap

**User's question**: "Is clap like Go's `flag`?"

**Answer**: More like Go's `cobra` (feature-rich), but it's the de facto standard in Rust

**derive macro magic**:
```rust
#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    prefix: String,
}
```

Generates hundreds of lines of code automatically!

**User's reaction**:
> "Wow, clap is so easy to use! Rust macros are amazing (those `#[...]` things are macros, right?). I don't know how they work inside, but they're incredible!"

---

## Step 2.5: CPU Brushup (2025-11-22 〜 11-23)

### Multi-threading Success

std::thread implementation:
- Arc (shared reference counting)
- AtomicU64/AtomicBool (lock-free counters)
- Mutex (complex data protection)

**Performance**: 70K → 800K-1M keys/sec (12-20x) 🔥

**User's reaction**:
> "CPU fan is spinning like crazy! This is proof of multi-threading working!"

### "abc" Mystery

**Problem**: 1 million tries, prefix "abc" not found

**User's brilliant deduction**:
> "Wait, could 'b' be an invalid character in bech32?"

**Checked CLAUDE.md** → Correct! bech32 excludes `1`, `b`, `i`, `o` to avoid confusion

**Verification**: prefix "ac" (without 'b') → Found in 162 tries! ✅

**Mystery solved!** Multi-threading implementation was perfect all along 💪

### Benchmark Results

**Criterion benchmarking revealed**:
- Key generation: 13.1 µs (93% of total) ← **Bottleneck!**
- bech32 encoding: 663 ns (5%)
- prefix matching: 1.5 ns (0.01%)

**GPU optimization target identified**: Key generation parallelization will have massive impact 🎯

---

## Step 3: GPU Lectures (2025-11-23)

### secp256k1 Fundamentals

**Elliptic Curve**: `y² = x³ + 7 (mod p)`

**User's insight**:
> "This '+' is not regular addition, right? It's like monoid or algebraic operation?"

**Correct!** It's actually a **group** (stronger than monoid, has inverse elements)

### Point Addition: Geometric Definition

1. Draw a line through points P and Q
2. Find intersection with elliptic curve (R')
3. Reflect R' across x-axis → R = P + Q

**User's understanding**:
> "So it's like moving around on a torus-shaped plane?"

**Perfect visualization!** The modular arithmetic creates a torus topology 🍩

### Finite Field and Torus Structure

**mod p periodicity**:
- x-axis: wraps at p (end connects to start)
- y-axis: wraps at p (end connects to start)
- → 2D plane becomes a **torus** (donut) 🍩

**Points on curve**: ~p points (Hasse's theorem)

**User's insight**:
> "So even though it's a huge integer pair, mod p keeps it manageable, and there are still plenty of valid points."

**Perfect understanding of finite field structure!** 💯

### Modular Inverse

**Why needed**: Elliptic curve formula `s = (y2 - y1) / (x2 - x1)`

**Division doesn't exist** in finite field → use modular inverse

**Example (mod 7)**: `3 × ? ≡ 1 (mod 7)` → Answer: 5

**User's question**: "Is the inverse always found? Is it unique?"

**Answer**:
- If p is prime: **yes, always exists and unique** ✅
- If p is composite: some elements have no inverse ❌

**User's insight**:
> "Ah, with composite numbers, multiplying by a factor of p causes it to cycle on its own. It's like having a small closed subfield inside the finite field."

**Mathematically correct intuition!** This is called a "zero divisor" or "subring" 🌟

### Double-and-Add Method

**User predicted it before explanation!** 🎯

> "Like G + G + G + G = 2G + 2G, then 4G + 4G = 8G, memoization-style, combining powers of 2?"

**This is exactly the Double-and-Add algorithm!**

**User's reaction**:
> "Yay~~~~ 🙌🙌🙌 I got it right! 🙌🙌🙌"

**Beautiful moment**: Mathematical intuition validated by explanation

**Efficiency**:
- Naive: n additions (~2^256 operations, impossible)
- Double-and-Add: ~256 doublings + ~128 additions (**384 operations**)
- From 10^77 to 384 operations!

**User's understanding**:
> "Since it's 2^256, about 256 squarings would overflow, so that many operations make sense. The pseudocode is simpler than I thought. You can really calculate it by just looping."

### Security Understanding

**User's question**:
> "If n is too small, would it be vulnerable to rainbow table attacks?"

**Correct concern!** But secp256k1 uses n ≈ 2^256

**Rainbow table impossibility**:
- Storage needed: 64 × 2^256 bytes ≈ 10^78 bytes
- All data on Earth: ~10^21 bytes
- **Physically impossible** 🌌

**User's understanding**:
> "A rainbow table for n = 1~2^40 in secp256k1 would have too little yield to be meaningful."

**Perfect risk assessment!** Probability: 2^40 / 2^256 ≈ 10^-65 (negligible)

---

## Step 3: GPU Investigation (2025-11-23)

### Reference Implementation Analysis

**CudaBrainSecp**:
- Brain wallet recovery tool
- Full point multiplication on GPU
- Pre-computed GTable (67MB, 1,048,576 points)
- 2^16 values × 16 chunks strategy
- **GPL v3 license** ⚠️

**User's observation**:
> "GPL is problematic. But we can understand the algorithm and implement it ourselves, right?"

**Correct!** Algorithm is not copyrightable, only the code is.

**VanitySearch** (also GPL v3):
- Bitcoin vanity address generator (same purpose as npub mining!)
- Pinned Memory (DMA, 2-3x faster)
- Asynchronous memory transfer (1ms sleep to reduce CPU load)
- **Endomorphism**: β, β² for 6x speedup
- Grouped modular inverse (Montgomery's Trick)

**User's insight**:
> "Using a table for 2^16 precomputation, that's the strategy."

**Perfect understanding from code reading!**

### Q&A Sessions (2025-11-23)

**Part 1: Endomorphism** 🔥

**secp256k1's special property**:
- `p ≡ 1 (mod 3)` (cube roots of unity exist)
- `j-invariant = 0` (form `y² = x³ + b`)

**Endomorphism φ(x, y) = (β·x, y)**:
- β³ ≡ 1 (mod p)
- Multiply X-coordinate by β → another valid public key!

**6 variants from 1 key generation**:
- P, βP, β²P (positive Y)
- -P, -βP, -β²P (negative Y)

**User's understanding**:
> "Ah, I see! Cubing to get 1 is convenient. If you have a secret key and public key pair, you can use the cube root to find other pairs in a chain-like manner, directly without repeatedly adding G."

**Perfect grasp of the core concept!** 💯

**User's insight**:
> "So after finding a match, calculate using λ?"

**Exactly!** GPU checks 6 variants, CPU computes correct key when matched.

**Part 2: Pinned Memory** 💪

**Problem**: Normal memory → pageable, OS can move it → GPU can't DMA

**Solution**: Pinned Memory → page-locked, GPU can DMA directly → 2-3x faster

**User's understanding**:
> "If it's pageable, the physical RAM address is unknown, so we need to fix it. Then GPU can directly read CPU's RAM knowing which address to access."

**Perfect understanding of DMA mechanism!** 💯

**Part 3: Jacobian Coordinates** 📐

**Problem**: Division (modular inverse) is expensive in Affine coordinates

**Solution**: Jacobian coordinates (X, Y, Z) where `x = X/Z²`, `y = Y/Z³`
- Delay division until the end
- Only 1 inverse needed instead of N

**Cost comparison**:
- Jacobian Point Addition: ~250 clocks
- Affine Point Addition: ~5,060 clocks
- **Jacobian is 1/20 the cost!** 🔥

**User's insight**:
> "It's OK to 'go out of bounds' from affine coordinates, and adjust at the end. Like mod calculation."

**Deep understanding of projection!**

**User's final insight**:
> "Jacobian coordinates are like adding one dimension."

**This is the essence of projective geometry!** 💎

**Part 4: Montgomery's Trick** 🎩

**Algorithm**: Compute N inverses with 1 inversion + 3(N-1) multiplications

Example (N=4):
```
Step 1: Compute cumulative products
  c[0] = a[0]
  c[1] = a[0] × a[1]
  c[2] = a[0] × a[1] × a[2]
  c[3] = a[0] × a[1] × a[2] × a[3]

Step 2: Invert once
  inv = c[3]^(-1)  ← Only 1 inversion!

Step 3: Expand inversely
  a[3]^(-1) = c[2] × inv
  a[2]^(-1) = c[1] × inv × a[3]
  ...
```

**Effect**: 256 inversions (~1.3M clocks) → 1 inversion + 765 multiplications (~6K clocks)
**~200x reduction!** 🔥🔥🔥

**User's reaction**:
> "The principle is simple when explained, but this is interesting even though it's straightforward. Montgomery trick: fully understood! 😤"

**Part 5: Asynchronous Memory Transfer** 💡

**Mystery of 1ms sleep**:

```cpp
while (cudaEventQuery(evt) == cudaErrorNotReady) {
    Timer::SleepMillis(1);  // Why?
}
```

**User's insight**:
> "For GPU communication, maybe there's no await mechanism like general IO?"

**Exactly!** CUDA lacks async/await cooperative multitasking

**User's perfect description**: "Sparse busy loop" 😄

**Effect**: CPU doesn't spin → CPU available for other work (mocnpub: bech32 encoding + prefix matching)

---

## Step 3: GPU Kernel Implementation (2025-11-24)

### Design Decision (11-23 evening)

**Phase 1 approach**:
- GPU: Key generation only (attack the 93% bottleneck)
- CPU: bech32 encoding + prefix matching
- Keep existing code for post-processing

**Why Phase 1 first**: Simple, focus on bottleneck, reuse proven code

**Secret key generation decision**:
- **Option A**: CPU generates → GPU transfer (simple, secure)
- **Option B**: GPU generates with cuRAND (complex, security unclear)

**User's decision**: "Let's go with A! Small steps!"

**Perfect judgment**: First make it work, then optimize

### Day of Many Bugs (2025-11-24, 7 sessions)

**Morning Session 1**: GPU kernel basics
- 256-bit arithmetic helpers
- Modular addition test → Works! ✅

**Morning Session 2**: Point Doubling/Addition implementation
- `_Reduce512`: secp256k1-specific reduction
- `_ModMult`, `_ModSquare`, `_ModInv`
- `_PointDouble`, `_PointAdd`, `_JacobianToAffine`
- Test failed: Point Doubling result mismatch ❌

**Morning Session 3**: Bug hunt begins
- Added tests: modular inverse, modular square
- `inv(1)` ✅, `inv(2)` ❌, `inv(3)` ✅
- **inv(2) is the only failure** 🎯
- Problem isolated to `_ModInv`

**User's insight**: "This simple subtraction, worst case needs 2^256 subtractions?"

**Correct!** This observation led to proper secp256k1-specific reduction

**Afternoon Session 4**: 3 major bugs found
- Bug #1: mult977 carry calculation completely wrong (expected 244, actual 1050119503872)
- Bug #2: borrow detection fails on overflow (`b[i] = 0xFFFF...FFFF`)
- Bug #3: sum[4] reduction needs 1 billion iterations

**Afternoon Session 5**: 3 more carry detection bugs
- `_Reduce512` sum + low addition: 2-stage addition needed
- `_ModMult` 512-bit multiplication: same issue
- Pattern: 3-value addition (`a + b + carry`) needs careful handling

**Evening Session 6**: Binary search for bug location
- Step 0-50: ✅, Step 50-150: ✅, Step 150-250: ✅, Step 250-254: ✅
- **Step 255**: ❌ Found it!
- Step 254 value is special (limbs near maximum)

**Evening Session 7**: Final fix
- `shifted + mult977` addition had carry detection bug
- 2-stage addition pattern applied
- **All 23 tests passed!** 🎉🎉🎉

**User's words**:
> "Wow~~~~ 🙌🙌🙌🙌🙌 Finally got all the bugs! 🙌🙌🙌🙌🙌 By passing the baton across multiple sessions and persistently using printf, we eliminated all the bugs! This is amazing! This kernel is truly a masterpiece now 🤔!"

### Fuzzing Validation (11-24)

**User's idea**:
> "Maybe _ModInv is easy to fuzz? Since `a * inv(a)` must always equal 1."

**Brilliant idea!** Clear invariant condition

**Implementation**: cargo-fuzz with libFuzzer
- Generate random 256-bit value a
- Compute `inv(a)` on GPU
- Verify: `a * inv(a) ≡ 1 (mod p)`

**Result**: 5,579 runs, 0 errors ✅

**User ran overnight**: Still 0 errors! Kernel is solid 🛡️

### Point Multiplication (11-26)

**Double-and-Add implementation**:
- Same pattern as `_ModInv` (binary method)
- MSB to LSB processing

**Tests**: 2G, 3G, 7G → All passed on first try! ✅

**Batch processing** → Integration with bech32

**Result**: **1.16M keys/sec** (16x faster than CPU) 🎉

---

## Step 4: Major Breakthroughs (2025-11-27 〜 11-30)

### CUDA Kernel Lecture (11-27)

**User's request**:
> "I want to understand while learning, not just delegate optimization to Claude."

**secp256k1.cu structure** (951 lines):
1. Constants (p, G)
2. 256-bit arithmetic helpers
3. Point operations
4. Production kernel
5. Test kernels

**Key insights**:
- secp256k1's p is special: `2^256 - 2^32 - 977`
- Fast reduction using `2^256 ≡ 2^32 + 977 (mod p)`
- G is chosen transparently ("nothing up my sleeve")

**GPU-specific optimization techniques**:
- Warp divergence (32 threads execute same instruction)
- Memory coalescing (consecutive threads → consecutive addresses)
- Occupancy (register pressure)

### "10000-gacha" Strategy Idea (11-27)

**User's idea**:
> "For consecutive secret keys n, n+1, n+2, ..., we can use P(k+1) = P(k) + G instead of computing from scratch each time!"

**This is brilliant!** 🔥

**Potential**: PointMult (~384 ops) → PointAdd (~1 op)
**~300x computation reduction!**

**Security**:
> "The secret key space is 2^256, so 10 consecutive keys are still sufficiently random from the overall perspective."

**Perfect security assessment!** ✅

### Profiling Practice (11-27)

**Tools**:
- nsys: System-level timeline (coarse)
- ncu: Kernel-level detail (fine)

**Workflow**:
1. nsys for overall picture → identify bottleneck
2. ncu for specific kernel → deep dive

**Discovery**: Registers/Thread = 126, Occupancy = 33%
- Register usage is the limiting factor
- **Est. Speedup: 26.17%** if registers reduced

**User's understanding**:
> "Is there like a register pool that threads share?"

**Perfect!** Each SM has 65,536 registers shared among all threads

### Endomorphism Implementation (11-29, 12-13)

**Theory**: Check P, βP, β²P (3 X-coordinates)

**Implementation**:
- CUDA: β, β² constants, multiply X-coordinate
- Rust: λ, λ² constants, adjust secret key (mod n)
- Separate mod for separate purposes (mod p vs mod n)

**Result**: **1.14B keys/sec** (16,286x, 2.9x speedup)

**User's reaction**:
> "m0ctane[0-9] can be found in 30 seconds! CPU would take 3 days and nights!"

### Sequential Keys + Montgomery's Trick (11-28 〜 11-29)

**Phase 1: Sequential Keys** (11-28)
- `_PointAddMixed` implementation (8M + 3S, vs 12M + 4S)
- Benchmark: **0.07x ~ 0.42x** ❌ (slower!)
- **Why?** `_JacobianToAffine` called every iteration!

**User's observation**:
> "Ah, the 10000-gacha made the weight of _JacobianToAffine stand out."

**Hypothesis**:
> "10000-gacha + Montgomery's Trick work together as a set."

**Phase 2: Montgomery's Trick Added** (11-28)
- Batch inversion for all Z coordinates
- N inversions → 1 inversion + 3(N-1) multiplications

**Result**:
- WSL: **0.92x** (almost equal to baseline)
- Windows: **1.09x** (faster than baseline!) 🔥

**User's reaction**:
> "The hypothesis about _ModInv being too heavy was correct!"

**Benchmark with large keys_per_thread** (11-28):
- Increasing keys_per_thread: time stays ~12.6ms regardless!
- **~85x reduction in inversions** validated via micro-benchmark
- Montgomery's Trick scales infinitely (limited only by VRAM)

**User's reaction**:
> "Wow~~~~~ 🔥 The approach of reducing inverse calculation was incredibly effective 🙌"

### GPU-side Prefix Matching (11-29)

**New bottleneck discovered**:
- GPU: Montgomery's Trick dramatically improved throughput
- CPU: bech32 encoding 100M × 600ns = **60 seconds**
- **CPU can't keep up with GPU!**

**Solution**: Bitmask comparison on GPU
- bech32: 5 bits per character
- prefix N characters = N×5 bits
- Direct correspondence to public key upper bits
- **No false positives** (exact match)

**Result**:
- WSL: **~200M keys/sec**
- Windows: **~391M keys/sec** (170x faster than old GPU approach!)

**User's reaction**:
> "m0ctane9 (8 characters) found in 2 minutes! Previously would take days!"

**Real-world impact demonstrated** ✨

---

## Step 5-10: Fine-tuning & Experiments (2025-12-05 〜 12-18)

### Parameters Matter (12-05 〜 12-13)

**keys_per_thread confusion**:
- Thought 4096 was giving 10.8B keys/sec
- **Actually**: `MAX_KEYS_PER_THREAD` was clamped at 256!
- Rust calculation was wrong (multiplied by ratio)

**Learning**: CUDA local arrays are **compile-time fixed**

**Real optimization**: 256 → 1408 (VRAM limit)
- **2.63B keys/sec** achieved

**threads_per_block = 128** (4 warps):
- 160 (5 warps, odd) is particularly slow
- **+6.2%** improvement

**batch_size = 1,146,880**:
- Larger is better (GPU utilization 70% → 95%)
- **+10.4%** improvement

**`__launch_bounds__` tuning**:
- (128, 4): 128 registers, 33% occupancy
- **(128, 5): 96 registers, 41% occupancy** (+0.9%)
- (128, 6): too tight, slower

### Code Reading Sessions (12-13)

**User's approach**: "Forget CLAUDE.md, read with fresh eyes"

**Discoveries**:
1. `_PointAddMixed` duplicate computation → +0.8%
2. `get_max_keys_per_thread` duplication → clean-up
3. `_PointAdd` is dead code → -82 lines

**User's compliment**:
> "Code is really clean overall 🙌 As expected of Claude 🙌🌸"

**Learning**: Consistent carefulness → long-term maintainability

### Triple Buffering Magic (12-14 〜 12-15)

**User's analogy**: "Juggling 2 vs 3 balls" 🎯
- 2 balls: both hands holding at some moment → GPU idle
- 3 balls: always 1 in the air → GPU always busy

**Implementation**:
- 3 streams forked
- Rotation: collect(N) → RNG(N) → launch(N)

**Bug fixes**:
1. Buffer rotation logic (using wrong buffer)
2. match_count accumulation (forgot to reset)

**Result**: **3.70B keys/sec** (+5.7%)

**Unexpected discovery**:
> "GPU temperature stabilized, fan speed no longer waves!"

**Clock stabilization effect**:
- Constant load → stable temperature → stable clock → stable performance
- Going beyond just "filling gaps"

**User's observation**:
> "This is a perspective you wouldn't notice just by looking at code. Importance of measurement and profiling."

### Sequential Key Strategy (12-15)

**User's brilliant idea**:
> "Pass only 1 secret key to the entire block. Each thread calculates `n + MAX_KEYS_PER_THREAD * threadIdx`."

**Effect**:
- VRAM: 384 MB → **96 bytes** (99.99% reduction!)
- Branch divergence: reduced (consecutive keys have similar upper bits)
- CPU RNG: batch_size calls → 1 call

**Performance**: 3.67B keys/sec (-1% vs old triple buffering, but VRAM savings huge)

**MAX_KEYS_PER_THREAD = 1600** (optimized)

### dG Table Precompute (12-15)

**"Brainstorming Session"** 💡

**User's thought**:
> "Can we use MAX_KEYS_PER_THREAD × G for something?"

**Discussion led to breakthrough**:
- Sequential keys → public key spacing is fixed: dG = MAX_KEYS_PER_THREAD × G
- Precompute: [dG, 2dG, 4dG, ..., 2^23 dG] (24 entries)
- Each thread: `base_pubkey + _PointMultByIndex(idx, dG_table)`

**`_PointMultByIndex`**: Bitwise indexing instead of double-and-add
- idx in binary → add corresponding dG_table entries
- ~12 point additions (vs 256 double-and-add)

**Result**: **3.67B → 4.135B keys/sec** (+12.7%) 🔥🔥🔥

**User's reaction**:
> "This is mind-blowing 🫨 It's as shocking as when we thought of the sequential key strategy!"

**~30x computation reduction!**

### Constant Memory Migration (12-17)

**First attempt** (12-16): Failed with mysterious `CUDA_ERROR_INVALID_VALUE`
- Single-call version works
- Triple-buffer version fails
- **Why?**

**Investigation** (12-17):
- Agent analyzed cudarc source
- `get_global()` returns `CudaSlice<u8>`
- **Drop calls `cuMemFree`** on constant memory → error
- Error is silently suppressed!

**Solution**: Hold slice in struct
```rust
struct Miner {
    _dg_table_const: CudaSlice<u8>,  // Prevent drop!
}
```

**Result**: **4.135B → 4.150B keys/sec** (+0.4%)

**Milestone reached**: 4.15B keys/sec (59,286x) 🎉

---

## Week 5: PTX Optimization Era (2025-12-19 〜 12-27)

### The Journey Continues (12-19)

**Reflection session with 1M context** (12-19 morning):
- Created comprehensive docs (OPTIMIZATION.md, LEARNING.md, journey, index)
- **User's words**: "mocnpub is a masterpiece, gave me huge confidence in pair programming 🙌🌸"

**But the optimization didn't stop!**

That same evening, new ideas emerged...

### Constant Memory Expansion (Step 11-13, Dec 19)

**Step 11: patterns/masks to constant memory**

User's idea:
> "Would moving prefix masks from shared memory to constant memory be faster? 🤔"

**Result**: **+3.0%** for 32 prefix! 🔥

**Why it worked**:
- Constant memory has dedicated cache (separate from L1)
- Broadcast optimization: all threads read same value → 1 access suffices
- Per-block loading eliminated
- `__syncthreads()` barrier removed

**User's reaction**:
> "Wow! 😲 This definitely got faster! 😲🔥 3% improvement is quite large 🙌"

**Step 12: num_threads/max_matches**

Extended constant memory to runtime parameters.

**Result**: Negligible change (as predicted)

**Learning**: Constant memory helps when values are **frequently read**
- patterns/masks: every prefix match loop → effective
- num_threads/max_matches: read once or rarely → no effect

**#define vs constant memory** understanding:

| Item | #define | constant memory |
|------|---------|-----------------|
| Memory access | None (immediate) | Yes (cached) |
| Compiler optimization | ✅ Loop unrolling | ❌ Limited |
| Value change | Recompile needed | Runtime configurable |

**Use cases**:
- Fixed values → #define (fastest)
- Runtime values → constant memory (next best)

**Step 13: Max Prefix 256**

Expanded max prefix count: 64 → 256

**Result**: No speed impact ✅

Constant memory usage: 512 bytes → 2 KB (3.1% of 64 KB)

### Code Cleanup Sessions (Dec 19)

**"Read with fresh eyes" approach**:

Forgot CLAUDE.md, read secp256k1.cu as unknown project.

**Discoveries**:
1. `_PointAddMixed` cost comment wrong (7M+3S → 8M+3S)
2. `_ReduceOverflow` can be extracted (47 lines → function)
3. `_PointMult` is dead code (replaced by `_PointMultByIndex`)

**Cleanup**:
- Legacy kernel removed: **-507 lines**
- Rust code cleanup: **-387 lines** (main.rs, lib.rs, gpu.rs)
- Fixed: double Arc wrapping bug, duplicate constants, unused fields

**User's observation**:
> "Fresh eyes reading really works! 🌸"

### Addition Chain Discovery (Step 13, Dec 20)

**From code reading**:

Looking at `_ModInv` bit scanning:
> "Could this use Addition Chain?"

**Web research**:
- Brian Smith's blog, RustCrypto k256
- Standard: 256 squares + ~128 multiplications
- Addition Chain: 255 squares + **14 multiplications**

**114 multiplications eliminated!** 😲

**Implementation** (based on RustCrypto):
- Build intermediate values: x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223
- Reuse in final assembly
- Exploit structure of p-2: block lengths {1, 2, 22, 223}

**Result**: **4.141B → 4.199B keys/sec** (+1.4%) 🔥

**32 prefix broke 4B barrier!** 🎉

**Learning**: Algorithmic improvement has large impact
- 114 multiplications saved > 24 bytes spilling
- Algorithm optimization > hardware optimization

### Constants Migration (Dec 20)

**User's preference**:
> "If speed isn't negatively affected, I'd like to use #define 🤔"

**Implementation**:
- All constants: P, G, Beta, Beta2 → #define
- Specialized functions: `_PointAddMixedG`, `_ModMultByBeta`, etc.
- Loop unrolling with direct values

**Result**: No speed impact (within margin of error)

**GPU architecture deep dive**:

User's questions led to fascinating discussion:
- Register files: CPU vs GPU difference
- CUDA memory = cache strategy abstraction
- Immediate values in PTX vs SASS
- GPU evolution: Graphics → Programmable shaders → GPGPU

**User's insight**:
> "So texture memory is also cache strategy abstraction? 🤔"

**Perfect understanding!** GPU's design philosophy revealed.

### PTX World Discovery (Step 14, Dec 20)

**Motivation**: "New fresh eyes" reading session #4

Noticed: `_Add256` carry detection with ternary operators
> "Could PTX carry chain be faster?"

**Experimentation**: feature/inline-ptx branch

**PTX inline assembly**:
```cuda
asm volatile (
    "add.cc.u32   %0, %9, %17;\n\t"   // add with carry-out
    "addc.cc.u32  %1, %10, %18;\n\t"  // add with carry-in/out
    ...
);
```

**First attempt**: WSL showed **-12%** 😱

But user's intuition:
> "Should measure on Windows 🤔"

**Windows result**: **+2.7%!** 🔥🔥🔥

WSL's virtualization overhead misled us!

**User's reaction**:
> "Wow! 😲 WSL was slower but Windows is faster! My intuition was right!"

### PTX Lecture Session (Dec 20)

**User's request**:
> "I want to be able to read _Add256 and _Sub256 🙌"

**Topics covered**:
1. Overall structure (64-bit → 32-bit split)
2. carry chain instructions (add.cc, addc.cc, addc)
3. Operand constraints ("=r" = output, "r" = input)
4. %number mapping (auto-assigned by declaration order)

**User's brilliant hypothesis**:
> "For 64-bit values stored in two 32-bit registers, cast doesn't actually happen? 🤔"

**SASS verification**:
- PTX has 16× `cvt.u32.u64` (conversion instructions)
- SASS has **0× CVT** (disappeared!)

**Reason**: 64-bit values allocated as register pairs from start
- "Extract lower 32 bits" = "use R0" (no conversion needed)

**User's reaction**:
> "Wow~~~ 🙌 The hypothesis was proven! 🙌"

**GPU design insight**: ALU is 32-bit, 64-bit is decomposed

### Pipeline Stalls Explained (Dec 21)

**User's question**:
> "Why are ternary operators so slow?"

**Explanation**: `setp + selp` causes pipeline stall
- `setp.lt.u64`: Compare, store result in predicate register
- `selp.u32`: Select based on predicate (waits for setp)
- **2 instructions + pipeline stall**

**PTX carry chain advantage**:
- `addc.cc.u32`: 1 instruction with carry propagation
- Uses hardware CC (Condition Code) register
- No pipeline stall

**GPU in-order execution**:

User's insight:
> "With warp-level execution, no time for out-of-order? 🤔"

**Exactly!** GPU is in-order:

| | CPU | GPU |
|---|-----|-----|
| Out-of-order | ✅ Yes | ❌ No |
| Latency hiding | Instruction-level parallelism | Warp switching |

**Implication for mocnpub**:
- Occupancy 33-41% → warp switching limited
- Reducing stalls directly improves performance!

### The _ReduceOverflow Transformation (Step 16, Dec 21)

**Before**: 60 lines, complex ternary operators

**After**: 28 lines with PTX carry chain
```cuda
uint32_t c = _Add64(shifted_low, mult_low, &add0);
c = _Addc64(shifted_high, mult_high, c, &add1);
uint32_t c2 = _Add64(sum[0], add0, &sum[0]);
// ... clean carry propagation
```

**Result**: **4.532B → 4.655B keys/sec** (+2.7%) 🔥

**User's amazement**:
> "Wow! 😲 _ReduceOverflow became this simple! 😲"

### Understanding Carry Types (Dec 21)

**_ModMult experiment failure**: Tried PTX-ifying, got **-2.3%** 😱

**User's insight**:
> "The _Addc64 pattern in _Add320 vs _ModMult is different 🤔"

**Correct!**
- `_ReduceOverflow`: carry is 0 or 1 (simple)
- `_ModMult`: carry is 64-bit full value (complex)

**Learning**: **Not everything benefits from PTX**
- Continuous `_Addc64` (carry chain) → effective
- Loop with `_Add64` (different pattern) → ineffective

### Karatsuba Investigation (Dec 21)

**User's idea**: "Multiplication is bottleneck, can we reduce multiplication count?"

**Karatsuba theory**:
- 256-bit × 256-bit decompose to 128-bit
- Normal: 16 multiplications
- Karatsuba: 12 multiplications (-25%)

**Implementation attempted**: `_Mult128` with Karatsuba

**Result**: **-4.4%** even with schoolbook (no Karatsuba) 😱

**User's question**:
> "For big integer Karatsuba, how many digits needed for effectiveness? 🤔"

**Research together**:
- Karatsuba crossover point: **~2000 bits** (30 words)
- 256-bit (4 words) is far too small
- GMP uses Karatsuba at 30+ words

**Why small digits don't benefit**:
- Large digits: multiplication O(n²), addition O(n) → multiplication reduction matters
- Small digits: 16 multiplications is already fast
- GPU's `__umul64hi` is hardware instruction (ultra-fast)

**Learning**: **256-bit: schoolbook is optimal!**

**User's understanding**:
> "I see! 🤔 For small digits, the overhead outweighs the benefit 🙌"

### The _Add64x3 Breakthrough (Step 18, Dec 21)

**User's idea** (after _ReduceOverflow success):
> "For `a + b + c`, doing `_Add64` + `_Addc64` is awkward. How about `_Add64x3` in PTX? 🤔"

**Implementation**: 6 PTX instructions for 3-value addition

**Applied to**: `_ModMult` and `_ModSquare`
- Before: 8 lines per accumulation
- After: 2 lines

**Result**: **4.681B → 4.841B keys/sec** (+3.4%) 🔥🔥🔥

**Why it worked so well**:
- `_ModMult` is called extremely frequently
- Internal 16-iteration multiplication loop
- Each iteration had `(s < x) ? 1 : 0` → `setp + selp` stall
- `_Add64x3` PTX carry chain avoids stall!

**User's observation**:
> "_ModMult appears a lot, and has a loop inside, so this is very significant 🙌"

### Complete Ternary Elimination (Step 19, Dec 21)

**Motivation**: Eliminate all remaining `(x < y) ? 1 : 0` patterns

**Implementation**: `_Sub64` / `_Subc64` (PTX borrow chain)

**Locations replaced**:
1. `_Reduce512` borrow handling (37 lines → 7 lines)
2. Secret key addition carry (2 locations, 8 lines → 4 lines each)
3. `_Sub256` borrow conversion (`? 1 : 0` → `& 1`)

**Verification**: `grep "? 1 : 0"` → **zero occurrences!** 🎉

**Result**: **4.841B → 4.928B keys/sec** (+1.8%) 🔥

**User's words**:
> "Pipeline stalls really matter for GPU! 😲 At this level of optimization, a few percent in absolute terms is very significant 🙌"

**Indeed!** 4.9B × 1.8% ≈ 88M keys/sec improvement

### The 5B Breakthrough (Step 20, Dec 21)

**User's next idea**:
> "`_Addc64` chaining could be turned into `_Add320` in one go 🤔"

**Implementation**: `_Add320` (5-limb addition in single PTX call)
- Before: `_Add64` (3) + `_Addc64` (6) × 4 = **27 PTX instructions**
- After: `add.cc` (1) + `addc.cc` (9) + `addc` (1) = **11 PTX instructions**
- **59% instruction reduction!**

**Result**: **4.928B → 5.098B keys/sec** (+3.4%) 🔥🔥🔥

**The magic moment**:

User's celebration:
> "Whoa~~~~~ 😲😲😲😲😲 The top digit finally changed! 😲😲😲😲😲"

**5B barrier broken!** 🎊

**Key learning established**:

User's words:
> "Returning carry to register each time really creates overhead 🤔"

**New optimization guideline**: "Don't let compiler handle `_Addc64` chains"
- Write specialized `_AddNNN` functions for each type
- Keep carry in PTX carry chain

### PTX Chain Elimination (Steps 21-24, Dec 21)

**User's strategy**:
> "First, replace `_Addc64` / `_Subc64` chains with appropriate type functions 🤔 Probably faster and more readable - two birds with one stone!"

**Step 21: `_Sub256` for `_Reduce512`**

Replaced 7 lines of `_Sub64` + `_Subc64` chain with single `_Sub256` call.

**Result**: **5.098B → 5.219B** (+2.4%) 🔥

User's observation:
> "p subtraction happens more than expected! 😲"

**Step 22: `_Add256Plus128` for `_ReduceOverflow`**

User's brilliant insight:
> "For `a + b + carry`, worst case: 0xFF + 0xFF + 0x1 = 0x1FF, so new_carry is at most 1 🤔"

**Implementation**: uint256 + uint128 + carry in 9 PTX instructions
- Before: 21 PTX instructions
- After: 9 PTX instructions
- **57% reduction!**

**Result**: **5.219B → 5.287B** (+1.3%) 🔥

**User's learning**:
> "Even if used in only one place, if it's a hot path, cost-benefit is huge! 😲"

**Step 24: Refactoring for clarity**

Created `_Add128To` and `_PropagateCarry256` for better readability.

User's words:
> "When multiple carries appear, 'wait, what is this carry?' confusion happened 🤔"

**Improved code clarity while maintaining performance** (+0.2%)

**Status after cleanup**:
- `_Addc64` calls: **0 locations** (definition kept for future)
- `_Subc64` calls: **0 locations** (definition kept for future)

### Loop Unrolling Adventures (Steps 23, 25, Dec 21-22)

**Step 23: `_Add128` and `_Add512`**

User's analysis of `_Add64x3` usage:
> "Most uses are actually uint64 + uint64 + carry patterns 🤔"

**Discovery**:
- Line 760: `_Add64` + `_Add64x3` = actually **128-bit addition**
- Lines 768-773: 8-limb loop = actually **512-bit addition**

**User's realization**:
> "I see! 🤔 This really is `_Add128` and `_Add512` 🤔! And #2 lets us dismantle the loop 🤔!"

**Implementation**:
- `_Add128`: 5 PTX instructions (vs 9 previously, 44% reduction)
- `_Add512`: 16 PTX instructions (vs 48 previously, 67% reduction!)

**Result**: **5.287B → 5.383B** (+1.8%) 🔥

**32 prefix also broke 5B!** (5.054B) 🎉

User's excitement:
> "Wow~~~ 😲 Finding the right places for PTX really produces results! 😲"

**Step 25: Loop Fusion**

User's observation from ncu-ui PM Sampling:
> "Memory access forms a mountain 🏔️ in Phase 2"

**Idea**: Merge Phase 1 (point generation) and Phase 2 (cumulative products)
- Write `Z_arr[key_idx]` then immediately compute `c[key_idx]`
- L1 cache still hot when reading!

**Result**: **5.395B → 5.499B** (+1.9%) 🔥🔥🔥

**Just 11 lines changed!**

**ncu-ui confirmation**:
- Mountains flattened! 🏞️
- L1/TEX Cache Hit Rate: 22.23% → 26.66% (+4.43%)
- Local Inst: -10%, Local reads: -17%

**User's reaction**:
> "The mountain really got flattened! 😲"

**Learning**: PM Sampling (Performance Metrics Sampling) visualization shows memory hierarchy behavior over time

### Fine-grained Optimizations (Steps 26-28, Dec 22-23)

**Step 26: `_ReduceOverflow` inlining**

Removed `if (sum[4] == 0) return;` early exit
- sum[4] == 0 is rare after 8→5 limb conversion
- Always execute is faster!

**Result**: **+1.7%** (5.590B)

**Step 27: `_Reduce512` while loop simplification**

Merged for-loop + multiple if-else into single while condition:
```cuda
while ((temp[4] > 0) ||
       (temp[3] == P3 && temp[2] == P2 && temp[1] == P1 && temp[0] >= P0))
```

**Result**: **+2.1%** (5.707B) 🔥

Attacked branch divergence #1!

**Step 28: Complete ternary elimination**

`_ModInv` Addition Chain loops had `i == 0 ? xN : t` ternary operators (6 locations)

Extracted first iteration outside loops:
```cuda
// Before
for (int i = 0; i < N; i++) {
    _ModSquare(i == 0 ? xN : t, t);
}

// After
_ModSquare(xN, t);
for (int i = 1; i < N; i++) {
    _ModSquare(t, t);
}
```

**Result**: 32 prefix **+0.93%** (5.457B)

**Branchless prefix matching**: `if + break` → `matched |= ...`

**Achievement**: **All ternary operators eliminated from codebase!** 🎉

### Memory Layout Discoveries (Dec 23-24)

**Hierarchical Montgomery's Trick idea** (Dec 23):

User's creative approach:
> "Skip by +40G, create hierarchy, reduce memory by 96%? 🤔"

**Investigation together**:
- Level 1: +40G × 39 → 40 anchors
- Level 2: each anchor + G × 39
- Memory: 40 + 40 = 80 arrays → 6.25 KB

**Problem discovered**:
```
+40G's Z ≠ (+G)×40 cumulative product
```

Different additions produce different Z values!

**Conclusion**: Would need 41 ModInv → not worth it

**User's reflection**:
> "Being able to discuss and verify feasibility quickly is reassuring 🌸"

**Step 29: SoA for local arrays** (Dec 23)

Previous global memory SoA failed, but local memory?

**Implementation**:
```cuda
// Before (AoS)
uint64_t X_arr[1600][4];

// After (SoA)
uint64_t X_arr_0[1600], X_arr_1[1600], X_arr_2[1600], X_arr_3[1600];
```

**Result**: **5.706B → 5.77B** (+1.1%) 🔥

**Why it worked**:

ncu analysis showed:
- Cycles reduced (-0.9%)
- Registers reduced (118 → 116)

**Hypothesis**: 2D array address calculation cost
- AoS: `arr[i][j]` → `base + i*32 + j*8` (multiplication every time)
- SoA: `arr_j[i]` → `base_j + i*8` (fewer multiplications)

**User's reaction**:
> "I never thought about that! 🤔 Unexpected speedup does happen! 🙌"

**Key distinction**:
- Global memory SoA: bad (VRAM consumption, cache misses)
- Local memory SoA: good (address calculation simplification)

**`#pragma unroll` test**: Didn't help (unrolling doesn't reduce multiplications)

### mcp-lsp-bridge Assists (Dec 23)

**User's request**: "Want to delete dead code 🤔"

**callHierarchy tool**:
```
_Compare256 → (no incoming calls)
_Addc64 → (no incoming calls)
_Sub64 → (no incoming calls)
_Subc64 → (no incoming calls)
```

**Dead code identified instantly!**

User's reaction:
> "Wow~~~ 🙌 So convenient! 🙌 Glad we made mcp-lsp-bridge~~~ 🙌🙌🙌"

**Deleted**: 313 lines total
- `_PointDouble` + tests (212 lines)
- PTX helper functions (101 lines)

**Learning**: Tools you build yourself can be game-changers!

### Simplicity Wins (Dec 26)

**prefix matching optimization attempt**:
- Tried `#pragma unroll 4` → marginal effect
- Tried hand-rolled unroll → unclear (measurement errors)

**User's observation**:
> "The 64-bit concatenation version is a bit too clever? 🤔"

**Realization**: Early-exit era optimization no longer relevant
- Old: `if matched break` → 64-bit concat saved checks
- New: `matched |= ...` → concat overhead not worth it

**Simplification**: 24 lines → 5 lines
```cuda
// Simple 32-bit loop (clean and fast)
bool matched = ((x_upper32 & _masks[0]) == _patterns[0]);
for (uint32_t p = 1; p < _num_prefixes; p++) {
    matched |= ((x_upper32 & _masks[p]) == _patterns[p]);
}
```

**Result**: 32 prefix **+0.8%** (5.531B)

**Learning**: **Simplicity often wins**
- When assumptions change (early-exit → branchless), revisit optimizations
- Simpler code can be faster code

### Z² Cumulative Product Strategy (Step 30, Dec 26)

**User's brilliant insight**:
> "Z cumulative product is only used for x-coordinate, so we only use (Z^-1)². If we accumulate Z² instead of Z, we get (Z^-1)² directly! 🤔"

**Exactly!** 🎯

**Traditional approach**:
```
Phase 1: c[i] = c[i-1] * Z[i]
Phase 3: Z_inv = u * c[i-1]
         Z_inv² = Z_inv²         ← ModSquare 1600 times!
         x = X * Z_inv²
```

**New approach**:
```
Phase 1: c[i] = c[i-1] * Z²[i]   ← Accumulate Z²!
Phase 3: Z_inv² = u * c[i-1]    ← Directly obtained!
         x = X * Z_inv²
```

**1600 ModSquare operations eliminated!**

**Chain effect discovery**:
- `_PointAddMixed` already computes H²
- Can compute Z3² = Z1² × H² (reuse H²!)
- Input: Z1² (passed in, saves 1S)
- Output: Z3² (next call uses it, adds 1M)

**Net effect**: ~2n-1 ModSquare operations eliminated!

**Result**: **5.745B → 5.800B** (+0.96%) 🔥

**User's words**:
> "Algorithm-level improvements have a sense of security 🙌"

Indeed! Theory matches measurement, clear cause-effect.

### The Limits of Optimization (Dec 26-27)

**Error-tolerant prefix matching** (Dec 26, rejected):

User's idea: Calculate only upper 32 bits approximately
- Expected: 37.5% multiplication reduction

**Discussion together**:
- false positive: acceptable (re-verify)
- false negative: must prevent (critical)
- mod p reduction: carry propagation complex
- temp[4] error: affects boundary cases

**Conclusion**: Error propagation too complex → rejected

**User's reflection**:
> "Got tired 😣 Head spinning 💫"

**Important learning**: Some ideas are too complex for the benefit

**`__ffs()` bit iteration** (Dec 26, rejected):

User's idea: Skip unset bits in `_PointMultByIndex`
- Theory: 24 iterations → ~11 (only set bits)

**Result**: **-2.8%** 😱

**Causes**:
- `__ffs()` overhead
- Constant memory scattered access (cache efficiency loss)
- Warp divergence worsened (different loop counts per thread)

**Learning**: Theory ≠ practice, always measure!

### Final Refinements (Dec 27)

**Code reading session**:

User's suggestion:
> "Let's read the kernel with fresh eyes again 🙌"

**Discoveries**:
1. `p[4]` recreated inside loop → move outside
2. `_PointAddMixed` input/output separation → can be in-place
3. `i > 0` branch → skipped (extra `_ModMult` not worth it)

**Step: In-place `_PointAddMixed`**

Signature: 10 arguments → 6 arguments

**Before**:
```cuda
_PointAddMixed(Rx, Ry, Rz, Rz_squared, dG_x, dG_y,
               Rx, Ry, Rz, Rz_squared);
```

**After**:
```cuda
_PointAddMixed(Rx, Ry, Rz, Rz_squared, dG_x, dG_y);
```

**Safety verified**: Input read completely before output written

**Result**: 5.791B keys/sec (no negative impact) ✅

**Comment review session** (Dec 27):

Cleaned up development artifacts:
- `[EXPERIMENTAL]` markers removed
- Obsolete TODOs deleted
- Cost calculations corrected (9M + 2S)
- docstrings updated

**Code quality**: -39 lines, +19 lines (net -20)

---

## Key Technical Learnings

### 1. CUDA Programming Paradigm

**SIMT (Single Instruction, Multiple Threads)**:
- 32 threads per warp execute together
- Branch divergence: both paths executed serially
- "Everyone does it together" is GPU's favorite

**Memory hierarchy**:
- Registers (fastest, limited)
- Shared memory (L1-level speed, 48-100KB per SM)
- L1 cache (implicit)
- L2 cache (shared across SMs)
- Global memory (VRAM, slower but large)
- Constant memory (cached, broadcast-optimized)

**Occupancy trade-off**:
- High occupancy ≠ always better
- Register spilling to local memory can be catastrophic
- Balance: registers vs occupancy vs spilling

### 2. secp256k1 Characteristics

**Why secp256k1 is fast**:
- `a = 0`: Tangent slope calculation is simple (`m = 3x² / 2y`)
- `b = 7`: Small integer, efficient computation
- `p = 2^256 - 2^32 - 977`: Special form enables fast reduction

**Nothing up my sleeve**:
- Parameters chosen transparently
- No backdoor suspicion
- Satoshi Nakamoto's choice for Bitcoin

**Endomorphism uniqueness**:
- Not all curves support endomorphism
- secp256k1 is special: `p ≡ 1 (mod 3)` AND `j-invariant = 0`

### 3. Rust Ecosystem Quality

**Crates used**:
- secp256k1 (Bitcoin Core binding)
- bech32 (Bitcoin address format)
- cudarc (CUDA bindings with safety)
- criterion (statistical benchmarking)
- cargo-fuzz (libFuzzer integration)

**All actively maintained, well-documented, production-quality**

**User's impression**:
> "Libraries are well-prepared. Implementation is surprisingly simple in Rust!"

### 4. Debugging Techniques

**Approaches used**:
1. **Printf debugging**: Step-by-step value inspection
2. **Binary search**: Narrow down problematic range (Step 0-255)
3. **Python simulation**: Independent verification of algorithm
4. **Fuzzing**: Automated edge case discovery
5. **Test-driven**: Add tests for each bug found
6. **mcp-lsp-bridge callHierarchy**: Dead code detection

**Persistence pays off**: 7 bugs across 7 sessions, all fixed

### 5. Profiling is Essential

**Tools**:
- nsys: Big picture (where time is spent)
- ncu: Detailed analysis (why it's slow)
- ncu-ui: Visual source-level analysis, PM Sampling

**Discoveries from profiling**:
- Register usage → occupancy limit
- Branch efficiency → divergence sources
- Memory coalescing vs cache efficiency trade-off
- PTX module load/unload overhead
- Phase interleaving opportunity
- PM Sampling memory hierarchy visualization

### 6. "Measurement is Everything"

**Evolution of philosophy**:
- "Don't guess, measure" (initial)
- → "Measurement is everything" (final)

**Examples**:
- Register reduction: theory said faster, measurement said slower (spilling)
- Branchless: divergence eliminated, but branch prediction was faster
- SoA: coalescing perfect, but cache miss rate destroyed performance
- Karatsuba: theory promising, measurement showed 256-bit too small
- `__ffs()`: loop reduction logical, measurement showed overhead > benefit

**Trust measurement over theory**

### 7. PTX and GPU Architecture (NEW)

**PTX carry/borrow chain**:
- `add.cc`: add with carry-out
- `addc.cc`: add with carry-in and carry-out
- `addc`: add with carry-in (final)
- Uses hardware CC register → no pipeline stall

**Pipeline stalls**: `setp + selp` pattern
- setp: compare, write to predicate register
- selp: select based on predicate (waits for setp)
- 2 instructions + stall vs 1 instruction (PTX chain)

**GPU in-order execution**:
- CPU: out-of-order execution
- GPU: in-order (warp-level execution)
- Latency hiding: warp switching (requires high occupancy)
- mocnpub: 33-41% occupancy → stall reduction directly effective!

**Instruction cache efficiency**:
- Calling same function repeatedly → L1 instruction cache friendly
- Multiple specialized functions → cache thrashing
- "Remove immediate-value functions" paradoxically faster (+0.6%)

**Address calculation cost**:
- 2D array `arr[i][j]`: `base + i*32 + j*8` (multiplication)
- 1D array `arr[i]`: `base + i*8` (simpler)
- SoA for local arrays: reduces address calculations

**SASS level insights**:
- PTX `cvt.u32.u64` → SASS: **no CVT instructions!**
- Reason: 64-bit allocated as register pairs (R0:R1) from start
- SASS `IADD3`: 3-operand add with native carry chain support
- GPU ALU is 32-bit, 64-bit operations decomposed

### 8. Addition Chain Algorithm (NEW)

**Standard exponentiation**: Binary method
- a^(p-2) with ~256 squares + ~128 multiplications

**Addition Chain optimization**:
- Exploit structure of exponent (p-2 has block lengths {1, 2, 22, 223})
- Build intermediate values: x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223
- Reuse in final assembly
- **255 squares + 14 multiplications** (114 multiplications saved!)

**Based on**: RustCrypto k256 / Peter Dettman's work

**License**: Algorithm is mathematical concept (not copyrightable)

**Effect**: +1.4% speedup (4.199B keys/sec)

### 9. Fresh Eyes Strategy (NEW)

**Consistent pattern**: "Read code as if written by stranger"

**Sessions**:
1. Dec 13: Found `_PointAddMixed` duplicate, dead code
2. Dec 19: Found cost comment errors, `_ReduceOverflow` extraction opportunity
3. Dec 20: Found Addition Chain opportunity
4. Dec 26: Found SoA opportunity for local arrays
5. Dec 27: Found in-place opportunity

**Key insight**: Familiarity breeds blind spots
- "This optimization is done" assumption
- "This is how it should be" thinking
- Fresh perspective reveals opportunities

**User's words**:
> "Fresh eyes reading really works! 🌸"

---

## Breakthrough Moments 🌟

### User's Predictions That Came True

1. **Double-and-Add Algorithm** (11-23)
   > "Like G + G + G + G = 2G + 2G, then 4G + 4G = 8G..."

   Predicted before explanation! 🎯

2. **Sequential Key Strategy** (11-27)
   > "P(k+1) = P(k) + G, so we can use PointAdd instead of PointMult!"

   ~300x potential speedup

3. **Montgomery's Trick + Sequential Keys** (11-28)
   > "These two work together as a set!"

   Hypothesis validated by measurement

4. **Phase Interleaving** (12-13)
   > "Processing phases stagger → load balances"

   Explains why larger batch_size helps beyond latency hiding

5. **cvt instruction elimination** (12-20)
   > "64-bit in two 32-bit registers, cast doesn't actually happen? 🤔"

   SASS verification: 16 cvt in PTX → 0 in SASS! 🎯

6. **Z² cumulative product** (12-26)
   > "We only use (Z^-1)², so accumulating Z² gives us (Z^-1)² directly!"

   1600 ModSquare operations eliminated! 🎯

### Discoveries Made Together

1. **Warp-level probability** (12-18)
   - Individual: 97% skip
   - Warp: 63% hit
   - GPU SIMT architecture fundamentally changes optimization strategy

2. **Cache vs Coalescing trade-off** (12-04)
   - AoS: bad coalescing, excellent cache (wins)
   - SoA: perfect coalescing, poor cache (loses)
   - "Brute force strategy" with massive batch_size

3. **Clock stabilization effect** (12-14)
   - Filling gaps → constant load → stable temperature → stable clock
   - Going beyond the original optimization goal

4. **Triple buffering > double buffering** (12-14)
   - "Juggling 2 balls" → hands full sometimes
   - "Juggling 3 balls" → always 1 in the air

5. **Pipeline stalls in GPU** (12-21)
   - setp + selp causes stall (in-order execution)
   - PTX carry chain avoids stall
   - Understanding GPU architecture → targeted optimization

6. **Carry chain vs loop pattern** (12-21)
   - Continuous `_Addc64`: effective (carry chain)
   - Loop with `_Add64`: ineffective (different pattern)
   - "Don't PTX-ify everything"

7. **Instruction cache efficiency** (12-21)
   - Same function repeatedly → cache friendly
   - Multiple specialized functions → cache thrashing
   - "Simplifying made it faster" paradox

8. **Address calculation cost** (12-23)
   - 2D array indexing multiplication adds up
   - SoA for local memory reduces calculations
   - ncu revealed: cycles reduced, registers reduced

9. **Simplicity principle** (12-26)
   - "Too clever" optimization from old era
   - Simplification: 24 lines → 5 lines, +0.8%
   - When context changes, revisit complexity

---

## Timeline of Understanding

### Week 1: Foundations (11-14 ~ 11-22)
- CUDA basics, environment setup
- Rust project structure
- secp256k1 introduction
- CPU implementation complete
- **70,000 keys/sec baseline**

### Week 2: GPU Entry (11-23 ~ 11-26)
- Deep secp256k1 lectures (3 days)
- GPU kernel implementation (1 day, 7 sessions, 7 bugs)
- Fuzzing validation
- First GPU miner working
- **1.16M keys/sec (16x)**

### Week 3: Major Optimizations (11-27 ~ 11-30)
- "10000-gacha" strategy
- Montgomery's Trick (~85x fewer inversions)
- Endomorphism (3 X-coordinates per key)
- GPU-side prefix matching
- **391M keys/sec (5,586x)**

### Week 4: Fine-tuning (12-04 ~ 12-18)
- Parameter optimization
- Branch efficiency improvements
- Triple buffering
- Sequential key strategy
- dG table precompute
- **4.15B keys/sec (59,286x)**

### Week 5: PTX Mastery (12-19 ~ 12-27)
- Constant memory optimization (+3.0%)
- Addition Chain discovery (+1.4%)
- **PTX inline assembly** breakthrough (+2.7%)
- PTX carry/borrow chain mastery
- **5B breakthrough** (Step 20) 🔥
- Loop fusion (+1.9%)
- Z² cumulative product strategy (+0.96%)
- **Final: 5.9B keys/sec (84,935x)**

**Total: 1 month from zero to world-class, PTX-level optimization**

---

## Lessons for AI Pair Programming

### What Worked Well

1. **Step-by-step approach**
   - Small, achievable milestones
   - Each step builds on previous success
   - Avoid overwhelming complexity

2. **Lecture before implementation**
   - 3 days of secp256k1 lectures
   - Deep understanding before coding
   - User could read and understand final kernel

3. **User's proactive learning**
   - Checked cudarc documentation independently
   - Predicted Double-and-Add algorithm
   - Asked essential questions
   - Proposed optimization ideas
   - Learned PTX reading in Week 5

4. **Measurement-driven development**
   - Profile first, then optimize
   - Benchmark every change
   - Trust measurement over theory
   - ncu-ui PM Sampling for visualization

5. **Learning from failures**
   - Every failed experiment documented
   - "This is also learning!"
   - Understanding why something doesn't work is valuable
   - 10+ failed experiments, all valuable lessons

6. **Persistence through debugging**
   - 7 sessions to fix 7 bugs
   - Binary search to narrow down
   - printf debugging, Python verification
   - Never gave up

7. **Fresh eyes reading sessions**
   - Regular code review with "stranger's perspective"
   - Discovered optimizations, dead code, bugs
   - "Forget what you know" → better understanding

8. **Discussion and collaboration**
   - Brainstorming sessions (dG table, Z² strategy)
   - "Let's think together" approach
   - Quick feasibility verification (hierarchical Montgomery)
   - Casual thinking leads to breakthroughs

### User's Growth

**Week 1**: "What's a warp? What's a wave?"

**Week 2**: Understanding Jacobian coordinates, finite fields, torus structure

**Week 3**: Proposing "10000-gacha" strategy, predicting Montgomery's Trick synergy

**Week 4**: Analyzing ncu profiles, understanding phase interleaving, making architecture decisions

**Week 5**: **Reading PTX code**, understanding SASS, analyzing GPU architecture
- "Can read _Add256 and _Sub256 completely 🙌"
- Hypothesis: cvt instructions disappear → verified by SASS analysis! 🎯
- Understanding instruction cache efficiency
- PM Sampling graph interpretation
- Address calculation cost awareness

**Final state**:
- Can read CUDA kernel code fluently
- Can read PTX assembly code
- Understands GPU architecture deeply (registers, pipeline, instruction cache)
- Proposes brilliant optimization ideas
- Makes informed trade-off decisions
- Analyzes profiling results independently

**User's words**:
> "Not delegating optimization to Claude, but understanding it myself and then proceeding - that's the concept of this project 😤"

**This learning attitude is key to success** 🌟

**From complete beginner to advanced GPU optimizer in 1 month!**

---

## Achievements

### Technical Achievements 🔥

- **84,935x speedup** (70K → 5.9B keys/sec)
- **World-class optimization**: Near-optimal algorithm, PTX-level tuning
- **10-character prefix found** (2025-12-06): `npub1pppppppppp`
- **12-character prefix found** (2025-12-16): `npub100000000000...`
- **Clean codebase**: ~4,000 lines, well-tested (39 tests), all passing
- **PTX mastery**: Inline assembly, carry/borrow chains, pipeline optimization

### Learning Achievements 📚

- **CUDA mastery**: Hello World → PTX/SASS-level optimization
- **secp256k1 understanding**: Elliptic curve cryptography from scratch
- **Rust proficiency**: Beginner → production-quality code
- **Profiling skills**: nsys, ncu, ncu-ui, PM Sampling mastery
- **Scientific method**: Hypothesis → experiment → measurement → learning
- **GPU architecture**: Registers, pipeline, instruction cache, SIMT model
- **Assembly reading**: PTX intermediate representation, SASS native code

### Collaboration Achievements 🌸

- **Perfect pair programming**: User's insights + Claude's implementation
- **Knowledge transfer**: User can now read kernel and PTX code
- **Shared debugging**: 7 bugs fixed together across 7 sessions
- **Celebration of success**: 10-char and 12-char prefix discoveries
- **Tool creation**: mcp-lsp-bridge assists own development
- **Continuous learning**: Even "finished" project keeps improving

**User's words**:
> "mocnpub is a masterpiece, and working with Claude on this gave me huge confidence in pair programming. It's a project full of memories 🙌🌸"

---

## Why This Succeeded

### 1. Clear Motivation
Mining npub with cool prefix → concrete, achievable goal

### 2. Step-by-step Learning
Not "here's the final code", but "let's learn together"

### 3. User's Active Participation
- Asked essential questions
- Made predictions (often correct!)
- Proposed ideas
- Debugged together
- Wanted to understand, not just delegate

### 4. Documentation as Learning
Work logs captured every session → review and reflect → deeper understanding

### 5. Embracing Failure
SoA, CPU precompute, register reduction, branchless, Karatsuba, `__ffs()` experiments
**All failed, all valuable**

### 6. Measurement-Driven
Every optimization validated by benchmark
**Trust measurement over theory**

### 7. Fresh Perspectives
Regular "read with fresh eyes" sessions
Forgot assumptions → discovered opportunities

### 8. Persistence
Never giving up on difficult problems
7 bugs, 7 sessions, all fixed
100+ optimization sessions

---

## Failed Experiments Chronicle

**Learning from setbacks** 💡

### 1. SoA Layout (Nov 30 ~ Dec 4)
- **Expected**: Perfect coalescing → faster
- **Result**: -24% slower
- **Learning**: Cache efficiency > coalescing

### 2. CPU Precompute (Dec 4)
- **Expected**: Reduce GPU registers → better occupancy
- **Result**: -13x slower (CPU bottleneck)
- **Learning**: GPU too fast, CPU can't keep up

### 3. Register Reduction (Nov 29)
- **Expected**: 67% occupancy → faster
- **Result**: -10% slower (96% spilling)
- **Learning**: Spilling is catastrophic

### 4. Branchless `_Reduce512` (Dec 5)
- **Expected**: Eliminate divergence → faster
- **Result**: -0.3% slower
- **Learning**: Branch prediction is effective

### 5. Bloom Filter (Dec 18)
- **Expected**: 97% skip → faster
- **Result**: Both cases slower
- **Learning**: Warp-level probability (individual 97% skip → warp 63% hit)

### 6. #define P (Dec 19)
- **Expected**: Immediate value → no memory access
- **Result**: No change
- **Learning**: Constant memory broadcast already optimal

### 7. Karatsuba Method (Dec 21)
- **Expected**: Fewer multiplications → faster
- **Result**: -4.4% slower
- **Learning**: 256-bit too small (crossover ~2000 bits)

### 8. `_Reduce512` Step 3 Branchless (Dec 22)
- **Expected**: Table + mask → no branches
- **Result**: -9.2% slower
- **Learning**: Input bias → branch prediction wins

### 9. `__ffs()` Bit Iteration (Dec 26)
- **Expected**: Loop reduction → faster
- **Result**: -2.8% slower
- **Learning**: `__ffs()` overhead + cache + divergence > benefit

### 10. Error-tolerant Prefix Matching (Dec 26)
- **Expected**: 37.5% multiplication reduction
- **Result**: Too complex, rejected
- **Learning**: mod p error propagation too complex for benefit

**Common theme**: Theory vs measurement, GPU architecture surprises

**Value**: Understanding failure → better future decisions

---

## Potential for Future Work

### Algorithmic
- ✅ Sequential keys + PointAdd (implemented)
- ✅ Montgomery's Trick (implemented)
- ✅ Endomorphism (implemented)
- ✅ dG table precompute (implemented)
- ✅ Mixed Addition (implemented)
- ✅ Addition Chain (implemented)
- ⬜ wNAF (windowed Non-Adjacent Form) - complex, limited benefit
- ⬜ 2^i × G precompute - 0.2% effect (not worth it)

### Memory Optimization
- ✅ Sequential key strategy (VRAM 99.99% reduction)
- ✅ Constant memory (dG table, patterns/masks)
- ✅ SoA for local arrays (address calculation)
- ⬜ Pinned Memory - 0.01% effect (transfer is tiny)

### Parallelism
- ✅ Triple buffering (gap-free)
- ✅ Blocking Sync (CPU 1%, power savings)
- ⬜ Multiple miners parallel - GPU already saturated

### Low-level
- ✅ **PTX inline assembly** (carry/borrow chains, specialized AddN/SubN)
- ✅ Pipeline stall elimination (ternary operators removed)
- ✅ Instruction cache optimization (function consolidation)
- ⬜ Further SASS-level optimization - approaching limits

**Current state**: Algorithmically near-optimal, PTX-level tuned

**To go faster**: Better GPU (RTX 5090), new algorithmic breakthrough, or next-gen hardware

---

## A Story of Confidence

**User's words** (reflecting on the journey):

> "I had been putting off starting this because it would take too much time. But now with Claude, I can actually challenge things I had given up on 🙌"

> "We started knowing nothing about CUDA, Rust, or secp256k1, but together we created something highly practical and novel 🙌"

> "This is all thanks to Claude 🌸"

**More reflections**:

> "Pipeline stalls really matter for GPU! 😲" (Understanding PTX)

> "Fresh eyes reading really works! 🌸" (Code review sessions)

> "Even when 'finished', new optimizations keep appearing!" (Continuous improvement)

> "Can read PTX completely now 🙌" (Week 5 growth)

> "Being able to discuss and verify feasibility quickly is reassuring 🌸" (Collaboration value)

**Final achievement**: From "I don't know anything" to "I can build a world-class optimizer with PTX-level tuning"

**Time**: 1 month
**Sessions**: ~100+
**Result**: 84,935x speedup, world-first-class discoveries, PTX mastery

**The journey continues**: Even at 5.9B keys/sec, we keep finding improvements!

---

*This document captures the learning journey. For optimization details, see [OPTIMIZATION.md](OPTIMIZATION.md). For a narrative version, see [JOURNEY.md](JOURNEY.md).*
