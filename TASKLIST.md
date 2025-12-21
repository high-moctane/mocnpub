# mocnpub タスクリスト 📋

**作成日**: 2025-11-14
**最終更新**: 2025-12-21
**進捗**: Step 0〜23 完了！🎉 5.383B keys/sec 達成！🔥🔥🔥

---

## 📊 全体サマリー

| Step | 概要 | 状態 |
|------|------|------|
| Step 0 | Rust + CUDA の Hello World | ✅ 完了 |
| Step 1 | GPU で簡単なプログラム（マンデルブロ集合）| ✅ 完了 |
| Step 2 | CPU 版 npub マイニング | ✅ 完了 |
| Step 2.5 | CPU 版のブラッシュアップ | ✅ 完了 |
| Step 3 | GPU 版に移行（16倍高速化）| ✅ 完了 |
| Step 4 | GPU カーネル高速化（45,000倍！）| ✅ 完了 |
| Step 5 | 最終チューニング & 完成度向上 | ✅ 完了 |
| Step 6 | コードクリーンアップ（CPU モード削除）| ✅ 完了 |
| Step 7 | Triple Buffering + Sequential Key Strategy（3.67B、VRAM 99.99%削減）| ✅ 完了 🔥 |
| Step 8 | dG テーブルプリコンピュート（_PointMult 削減、+12.7%）| ✅ 完了 🔥🔥🔥 |
| Step 9 | Blocking Sync（CPU 使用率 100% → 1%、電力消費削減）| ✅ 完了 🔥🔥🔥 |
| Step 10 | 32-bit Prefix Matching（複数 prefix 時 +1.2%）| ✅ 完了 🔥 |
| Step 11 | Constant Memory（patterns/masks、複数 prefix 時 +3.0%）| ✅ 完了 🔥 |
| Step 12 | Max Prefix 256（64 → 256 に拡張、速度影響なし）| ✅ 完了 🔥 |
| Step 13 | Addition Chain（_ModInv 乗算 128→14、+1.4%）| ✅ 完了 🔥🔥🔥 |
| Step 14 | インライン PTX（_Add256/_Sub256、+2.7%）| ✅ 完了 🔥🔥🔥 |
| Step 15 | _Add64/_Addc64 で _Reduce512 最適化（+5.1%）| ✅ 完了 🔥🔥🔥 |
| Step 16 | _ReduceOverflow も PTX 化（+2.7%）| ✅ 完了 🔥🔥🔥 |
| Step 17 | 即値化関数削除（-119行、+0.6%）| ✅ 完了 🔥 |
| Step 18 | _Add64x3 で _ModMult/_ModSquare 最適化（+3.4%）| ✅ 完了 🔥🔥🔥 |
| Step 19 | _Sub64/_Subc64 で全三項演算子削除（+1.8%）| ✅ 完了 🔥🔥🔥 |
| Step 20 | _Add320 で _Reduce512 最適化（+3.4%、5B突破！）| ✅ 完了 🔥🔥🔥 |
| Step 21 | _Sub256 で _Reduce512 減算最適化（+2.4%）| ✅ 完了 🔥🔥🔥 |
| Step 22 | _Add256Plus128 で _ReduceOverflow 最適化（+1.3%）| ✅ 完了 🔥 |
| Step 23 | _Add128/_Add512 で _ModSquare 最適化（+1.8%、32 prefix も 5B 突破！）| ✅ 完了 🔥🔥🔥 |

---

## 🏆 現在のパフォーマンス

| 段階 | スループット | CPU比 |
|------|-------------|-------|
| CPU（16スレッド） | ~70,000 keys/sec | 1x |
| GPU Montgomery（エンドモルフィズム前） | ~391M keys/sec | 5,586x |
| GPU + エンドモルフィズム | 1.14B keys/sec | 16,286x |
| GPU + _ModSquare 最適化 | 1.18B keys/sec | 16,857x |
| GPU + Tail Effect 対策 | 1.196B keys/sec | 17,089x |
| GPU + keys_per_thread 最適化（1408） | 2.63B keys/sec | 38,000x |
| GPU + threads_per_block 最適化（128） | 2.80B keys/sec | 40,000x |
| GPU + batch_size 最適化（1146880） | 3.09B keys/sec | 44,000x |
| GPU + ブランチレス化（_ModSub/_ModAdd） | 3.16B keys/sec | 45,143x |
| GPU + batch_size 再最適化（3584000） | 3.24B keys/sec | 46,286x |
| GPU + `__launch_bounds__(128, 4)` | 3.26B keys/sec | 46,571x |
| GPU + keys_per_thread 再最適化（1500） | 3.30B keys/sec | 47,143x |
| GPU + `_PointAddMixed` 最適化（7M+3S） | 3.326B keys/sec | 47,514x |
| GPU + `__launch_bounds__(128, 5)` | 3.356B keys/sec | 47,937x |
| GPU + `_ModInv`/`_PointMult` 最適化 | 3.396B keys/sec | 48,512x |
| GPU + batch_size 再々最適化（4000000） | 3.464B keys/sec | 49,486x |
| GPU + Multi-thread mining（threads=2, batch=2M） | 3.49B keys/sec | 49,857x |
| GPU + Multi-thread mining（2:1 batch ratio） | 3.50B keys/sec | 50,000x |
| GPU + Triple Buffering（3 streams） | 3.70B keys/sec | 52,857x |
| GPU + Sequential Key Strategy（VRAM 99.99%削減、1600 keys/thread） | 3.67B keys/sec | 52,429x |
| GPU + dG テーブルプリコンピュート（_PointMult 削減） | 4.135B keys/sec | 59,071x |
| GPU + Blocking Sync（CPU 使用率 1%） | 4.136B keys/sec | 59,086x |
| GPU + Constant Memory（patterns/masks） | 4.141B keys/sec | 59,157x |
| GPU + Addition Chain（_ModInv 乗算 128→14） | 4.199B keys/sec | 59,991x |
| GPU + インライン PTX（_Add256/_Sub256 carry chain） | 4.313B keys/sec | 61,614x |
| GPU + _Add64/_Addc64 で _Reduce512 最適化 | 4.532B keys/sec | 64,743x |
| GPU + _ReduceOverflow も PTX 化 | 4.655B keys/sec | 66,500x |
| GPU + 即値化関数削除（命令キャッシュ効率向上） | 4.681B keys/sec | 66,871x |
| GPU + _Add64x3 で _ModMult/_ModSquare 最適化 | 4.841B keys/sec | 69,157x |
| GPU + _Sub64/_Subc64 で全三項演算子削除 | 4.928B keys/sec | 70,400x |
| GPU + _Add320 で _Reduce512 最適化 | 5.098B keys/sec | 72,835x |
| GPU + _Sub256 で _Reduce512 減算最適化 | 5.219B keys/sec | 74,553x |
| GPU + _Add256Plus128 で _ReduceOverflow 最適化 | 5.287B keys/sec | 75,529x |
| **GPU + _Add128/_Add512 で _ModSquare 最適化** | **5.383B keys/sec** | **76,903x** 🔥🔥🔥 |

**8文字 prefix が約 4 分で見つかる！** 🎉
**CPU 使用率が 100% → 1% に削減！電力消費大幅削減！** 💡
**32 prefix 時：5.054B keys/sec（5B 突破！）** 💪

---

## ✅ 完了した Step

### Step 0: Rust + CUDA の Hello World 🌸
- [x] CUDA Toolkit 13.0 インストール（Windows + WSL）
- [x] Rust + cudarc 0.17.8 でセットアップ
- [x] RTX 5070 Ti への接続確認

### Step 1: GPU で簡単なプログラム 🔥
- [x] マンデルブロ集合を CPU/GPU で実装
- [x] GPU 版で 3.5倍高速化を達成
- [x] CUDA カーネル、PTX、cudarc の基本を習得

### Step 2: CPU 版 npub マイニング 💪
- [x] secp256k1 と Nostr の鍵生成を理解
- [x] bech32 エンコーディング（npub/nsec）を実装
- [x] CLI インターフェース（clap）を実装
- [x] パフォーマンス測定（~70,000 keys/sec）

### Step 2.5: CPU 版のブラッシュアップ 🔧
- [x] マルチスレッド対応（16スレッド、12〜20倍高速化）
- [x] 入力検証（bech32 無効文字チェック）
- [x] テストコード（7つのテストケース）
- [x] 継続モード（`--limit` オプション）
- [x] 複数 prefix の OR 指定
- [x] ベンチマーク（criterion 0.6）

### Step 3: GPU 版に移行 🚀
- [x] 参考実装の調査（VanitySearch, CudaBrainSecp）
- [x] 独自の secp256k1 CUDA カーネルを実装
- [x] Point Multiplication、バッチ処理を実装
- [x] GPU 版で 16倍高速化を達成

### Step 4: GPU カーネル高速化 🔥🔥🔥
- [x] 連続秘密鍵戦略（1億連ガチャ）
- [x] Montgomery's Trick（逆元のバッチ計算）
- [x] Mixed Addition（G の Z=1 を活用）
- [x] GPU 側 prefix マッチング（bech32 スキップ）
- [x] **エンドモルフィズム（β, β²）**：2.9倍高速化 🔥

---

## ❌ SoA 最適化の実験（2025-11-30〜12-04、見送り）

### 実装内容

SoA（Structure of Arrays）最適化を実装：
- Montgomery's Trick のワークバッファをローカルメモリ → グローバルメモリ（SoA レイアウト）に移行
- `MAX_KEYS_PER_THREAD` のコンパイル時制限を撤廃
- keys_per_thread を任意の値に設定可能に

### バグ修正（2025-12-04）

**keys_per_thread 制限バグを発見・修正**：
- `gpu.rs` で `keys_per_thread.min(4096)` により制限されていた
- これにより keys/sec が異常に大きく表示されていた

### ncu プロファイリング比較（2025-12-04、2025-12-09 `__launch_bounds__` 適用後）

| 指標 | AoS 版 | SoA 版 |
|------|--------|--------|
| **Compute Throughput** | **81〜83%** | 77.41% |
| **Waves Per SM** | **42.67** | 0.75 |
| L1/TEX Hit Rate | ~5% | ~5% |
| Global Coalescing | 8/32 (25%) | **31.12/32 (97%)** |
| batch_size | 8960 blocks | 210 blocks |
| Occupancy | **33%** | - |
| Registers/Thread | **128** | - |
| keys/sec | **3.26B** | 2.70B |

### 結論 ❌

**SoA 版は見送り。AoS 版を本筋として最適化する。**

**なぜ AoS 版が速いのか**：
1. **batch_size を大きくできる** — VRAM 消費が少ない → GPU をフル活用（42.67 waves）
2. **並列度でメモリレイテンシを隠蔽** — L1 ヒット率は低いが、大量の block で隠蔽
3. **Compute heavy** — secp256k1 の演算は分岐が少なく、メモリ待ちの間も計算で埋められる

**なぜ SoA 版が遅いのか**：
1. **VRAM を大量消費** — 4 つのワークバッファが必要 → batch_size を小さくせざるを得ない
2. **GPU を活用しきれない** — Waves Per SM: 0.75（AoS は 42.67）
3. **コアレッシング 97% でも遅い** — 並列度の方が重要だった

### 学び 💡

- **並列度が正義** — batch_size を大量にして、メモリレイテンシを隠蔽
- **Compute heavy なカーネルは強い** — 分岐が少なく、メモリ待ちの間も計算で埋められる
- **コアレッシングだけが正義ではない** — SoA はコアレッシング 97% でも遅かった
- **VRAM 効率も重要** — メモリを食いすぎると並列度が下がる

---

## ❌ CPU 公開鍵プリコンピュート戦法（2025-12-04、見送り）

### アイデア 💡

**現状**：GPU カーネル内で `_PointMult(k, G)` を呼んで初期公開鍵を計算している

**新戦法**：CPU 側で初期公開鍵 `(Px, Py)` を計算して GPU に渡す

### 実験結果 📊

**実装完了、動作確認 OK**（正しい npub が見つかった）

| 版 | keys/sec | Registers | Occupancy |
|---|---|---|---|
| 元の AoS 版（GPU で `_PointMult`）| **3.09B** | 130 | 33% |
| CPU プリコンピュート（シングルスレッド）| 216M | - | - |
| CPU プリコンピュート（rayon 並列化）| 844M | **98** | **33%** |

### 結論 ❌

1. **レジスタは減った**（130 → 98、約25%減）
2. **しかし Occupancy は変わらなかった**（33% のまま）
3. **CPU がボトルネック**：batch_size = 1,146,880 回の公開鍵計算が GPU より遅い
4. **結果的に 3.7 倍遅くなった**（3.09B → 844M）

### 学び 💡

- `_PointMult` を削除してもレジスタ 98 では Occupancy 改善のしきい値に届かない
- GPU の計算が速すぎて、CPU での鍵生成が追いつかない
- 「CPU は暇してる」という前提が間違いだった

---

## 📋 Step 5: 最終チューニング & 完成度向上 🎯

**現在のフェーズ**：アプリケーション完成度向上

| # | タスク | 状態 |
|---|--------|------|
| 1 | keys_per_thread 固定化（ビルド時確定） | ✅ 完了 |
| 2 | MAX_KEYS_PER_THREAD を環境変数で指定 | ✅ 完了 |
| 3 | 不要な引数の削除（`--keys-per-thread` 等） | ✅ 完了 |
| 4 | batch_size 最終調整（4000000、+2.0%） | ✅ 完了 |
| 5 | README 整備 | ✅ 完了 |

---

### ✅ 完了した最適化

- **Multi-thread mining（threads=2, batch=2M）**（2025-12-13）🔥
  - 独立した mining loop を複数スレッドで実行
  - 各スレッドで微妙に異なる batch_size を使用（phase interleaving 強化）
  - 理論上は 0.1% の改善のはずが、実測で +0.8% の効果
  - **Phase interleaving の効果が実証された！**
  - 結果：3.464B → **3.49B keys/sec**（+0.8%）🔥

- **batch_size 再々最適化（4000000）**（2025-12-13）🔥
  - `__launch_bounds__(128, 5)` で Occupancy が 33% → 41% に向上した影響
  - より多くの並列実行が可能になり、最適な batch_size が増加
  - **Phase interleaving** 効果：処理フェーズのずれによる負荷分散
  - 結果：3.396B → **3.464B keys/sec**（+2.0%）🔥

- **`_ModInv`/`_PointMult` 最適化**（2025-12-13）🔥
  - `_ModInv` で `_ModMult(res, res, temp)` → `_ModSquare(res, temp)` に変更（255回の二乗で効果）
  - `_PointMult` で `_PointAdd` → `_PointAddMixed` に変更（P は Z=1 なので 12M+4S → 7M+3S）
  - 未使用 `step` 変数を削除
  - 結果：3.356B → **3.396B keys/sec**（+1.2%）🔥

- **`__launch_bounds__(128, 5)` チューニング**（2025-12-13）🔥
  - minBlocksPerMultiprocessor を 4 → 5 に変更
  - コンパイラがレジスタを 128 → 96 に削減（スピルなし）
  - Theoretical Occupancy: 33% → 41%
  - 結果：3.326B → **3.356B keys/sec**（+0.9%）🔥

- **`_PointAddMixed` 最適化**（2025-12-13）🔥
  - `X1 * H^2` の重複計算を削除（730行目と737行目で同じ計算をしていた）
  - `X1_H2` 変数に保存して再利用
  - 乗算回数：8M + 3S → 7M + 3S（12.5%削減）
  - `__launch_bounds__(128, 4)` のおかげでレジスタ数 128 維持（外すと 130）
  - 結果：3.30B → **3.326B keys/sec**（+0.8%）🔥

- **`__launch_bounds__(128, 4)` で Occupancy 改善**（2025-12-09）🔥🔥
  - レジスタ数を 130 → 128 に制限（minBlocksPerMultiprocessor = 4）
  - Theoretical Occupancy: 25% → 33%（+33%）
  - 結果：3.10B → **3.26B keys/sec**（+5%）🔥
  - 128 = 2^7 で GPU アーキテクチャと相性が良い

- **不要コード削除**（2025-12-09）🧹
  - 古い CUDA カーネル 3 つ削除（`generate_pubkeys`, `generate_pubkeys_sequential`, `generate_pubkeys_sequential_montgomery`）
  - 未使用 Rust 関数 3 つ削除（`wide_mul_u128`, `u64x4_to_bytes_for_scalar`, `generate_pubkeys_batch`）
  - 結果：**約 930 行削減**、警告ゼロ、テスト 36 件全通過 ✅

- **patterns/masks 共有メモリ化**（2025-12-08）🔥
  - 複数 prefix 指定時のパフォーマンス低下（3.16B→2.79B、-11.7%）を改善
  - グローバルメモリから共有メモリにロード、ブロック内で協調してアクセス
  - 結果：32 prefix 時 2.79B → **2.88B keys/sec**（+3.4%）、速度低下 11.7%→9.1%

- **`_ModSub`/`_ModAdd` ブランチレス化**（2025-12-04）🔥
  - Branch Efficiency 78.88% の原因を特定（256-bit 比較分岐がランダム → 約 50% ダイバージェンス）
  - `_Sub256` を borrow を返すように変更
  - `_ModSub`/`_ModAdd` をマスク選択でブランチレス化
  - 結果：3.09B → **3.16B keys/sec**（+2.3%）🔥

- **batch_size 最適化**（2025-11-29）
  - デフォルトを 65536 → 1146880（128 waves）に拡張
  - カーネル起動オーバーヘッドを削減、GPU 使用率 70% → 95%
  - 結果：2.80B → **3.09B keys/sec**（+10.4%）🔥

- **threads_per_block 最適化**（2025-11-29）
  - デフォルトを 64 → 128 に変更
  - 128 = 4 warps がスイートスポット（256 以上はレジスタ競合で遅くなる）
  - 結果：2.63B → **2.80B keys/sec**（+6.2%）🔥

- **keys_per_thread 最適化**（2025-11-29）
  - MAX_KEYS_PER_THREAD を 256 → 1408 に拡張
  - VRAM 16GB の限界を探索、最適値を発見
  - 結果：1.196B → **2.63B keys/sec**（+120%）🔥🔥🔥

- **Tail Effect 対策**（2025-11-29）
  - SM 数を動的取得、batch_size を SM × threads_per_block の整数倍に自動調整
  - 65536 → 67200（15 waves ぴったり、アイドル SM ゼロ）
  - 結果：1.18B → **1.196B keys/sec**（+1.4%）🔥

- **`_ModSquare` 最適化**（2025-11-29）
  - 対称性を利用：16回 → 10回の乗算（37.5% 削減）
  - ベンチマーク：約 3.5% 改善
  - 結果：1.14B → **1.18B keys/sec** 🔥

### ✅ 完了したプロファイリング

- **nsys プロファイリング**（2025-11-29）
  - カーネル実行: 99.9%（ほぼ全時間が GPU 計算）
  - メモリ転送: 0.1%（オーバーヘッド極小）
  - GPU 使用率: 95%（batch_size 最適化後）

- **ncu プロファイリング**（2025-11-29, 2025-11-30）
  - Compute Throughput: 45〜74%（Compute bound）
  - Memory Throughput: 12.16%
  - Occupancy: 33%（レジスタ 130/thread が制限要因）
  - Local Memory Spilling: 0%（スピルなし ✅）
  - **メモリコアレッシング問題を検出**：
    - local loads: 32バイト中1バイトしか活用できていない（Est. Speedup ~10%）
    - local stores: 32バイト中1バイトしか活用できていない（Est. Speedup ~12%）
    - → SoA 最適化で ~20% 改善の余地あり

### 見送った最適化

| 最適化 | 理由 |
|--------|------|
| 2^i × G プリコンピュート | 効果 0.2%（`_PointMult` が全体の 0.4%）|
| PTX carry/borrow | NVCC が最適化済み |
| CUDA Streams | Triple Buffering で実装済み（+5.7%）|
| Pinned Memory | 連続秘密鍵戦略で転送が 100 bytes/バッチに削減 → 効果 0.01% 以下 |
| **レジスタ削減** | スピル 96% 発生で逆効果（1.14B → 1.03B）|
| **CPU 公開鍵プリコンピュート** | CPU がボトルネック、Occupancy 改善なし（3.09B → 844M）|
| **`_PointMult` ブランチレス化** | アルゴリズム的に必要な分岐（ダブル＆アッド法のビット分岐）、計算量 2 倍になるため見送り |
| **`_Reduce512` ブランチレス化** | ダイバージェンス 99.16% を解消したが、実測で旧実装のほうが高速（3.20B vs 3.19B）→ 分岐予測が効いていた可能性 |
| **`_PointAdd` 重複計算最適化** | `_PointMult` 内でのみ使用（全体の約 8%）、レジスタ 128/128 でスピルリスク高、効果 0.1% 未満 |
| **Bloom Filter プレフィルタリング** | GPU の SIMT と相性が悪い。warp 単位では 63% がヒット → オーバーヘッド増（-1%）。10,000 bit 以上必要 |
| ~~_P を #define 化~~ | **実装済み（2025-12-20）**: P, G, Beta, Beta2 を全て #define 化。速度影響なし（4.146B → 4.146B）。コードがクリーンに |
| **Karatsuba 法（`_Mult128`）** | **逆効果（2025-12-21）**: 256-bit は crossover point（約 2000 bit）より小さい。分割しただけで -4.4%（4.681B → 4.474B）。加算オーバーヘッド > 乗算削減 |

---

## 📋 Step 6: コードクリーンアップ（CPU モード削除）🧹

**背景**：GPU が 47,143 倍の高速化を達成した今、CPU モードの存在意義がなくなった。
CPU（Ryzen 9800X3D、8コア16スレッド）で連続秘密鍵戦法を実装しても、GPU には遠く及ばない。
思い切って削除し、「最初から GPU でやるつもりだった」かのようにリファクタする。

**削除対象**：
- `main.rs` の CPU モード関連コード（約200行）
- `--gpu` フラグ（常に GPU モードに）
- `--threads` オプション

**lib.rs は変更なし**（すべて GPU モードでも使用している）

### タスク一覧

| # | タスク | 内容 | 状態 |
|---|--------|------|------|
| 1 | **CPU モードのコード削除** | `run_mine` 内の CPU 分岐を削除 | ✅ 完了 |
| 2 | **`--gpu` フラグ削除** | 常に GPU モードに（フラグ不要に）| ✅ 完了 |
| 3 | **`--threads` オプション削除** | CPU スレッド数指定は不要に | ✅ 完了 |
| 4 | **不要な import 削除** | `mpsc`, `AtomicU64`, `AtomicUsize` 等を整理 | ✅ 完了 |
| 5 | **ビルド & テスト** | `cargo build --release` & `cargo test` | ✅ 完了 |
| 6 | **関数構造の整理** | `run_gpu_mining` → `mining_loop` にリネーム | ✅ 完了 |
| 7 | **CLI 説明文の更新** | `Mine` サブコマンドの説明を GPU 前提に | ✅ 完了 |
| 8 | **最終確認** | 動作確認、コミット | ✅ 完了 |

### 実績（2025-12-13）🎉

- **221行のコード削減**（787行 → 566行）
- **main.rs がシンプルに**（CPU/GPU の分岐がなくなった）
- **メンテナンスコスト減**
- **ユーザーが誤って遅い CPU モードを使うリスクがなくなる**

---

## ✅ Step 7: Triple Buffering + Sequential Key Strategy（2025-12-15 完了）🔥

### 実装内容

**Phase 1: トリプルバッファリング**：3 つの Stream で GPU を常時 100% 稼働

- `TripleBufferMiner` 構造体を追加（`gpu.rs`）
- 3 つの Stream を `stream.fork()` で作成
- 3 つのホストバッファをローテーション
- `launch_single(idx)` / `collect_single(idx)` で個別操作

**Phase 2: 連続秘密鍵戦略**：VRAM 消費を 99.99% 削減

- `SequentialTripleBufferMiner` 構造体を追加
- バッチ全体で 1 つの base_key だけを渡す
- 各スレッドは `base_key + global_idx * MAX_KEYS_PER_THREAD` から計算開始
- VRAM: 384 MB → 96 bytes（99.99% 削減！）
- MAX_KEYS_PER_THREAD のデフォルト: 1500 → 1600 に最適化

### 結果 🎉

| 指標 | Triple Buffering | + Sequential Key |
|------|------------------|------------------|
| **スループット** | 3.70B keys/sec | **3.67B keys/sec** |
| **CPU 比** | 52,857x | **52,429x** |
| **VRAM（base_keys）** | 384 MB | **96 bytes** 🔥 |

### nsys 確認結果

- カーネルが **隙間なくびっちり** 詰まっている！
- 3 つの Stream（Default stream 7, Stream 13, Stream 14）が交互に動作
- GPU 使用率が安定して 100% 近くに

### 発見 💡

**隙間を埋めることで、隙間を埋める以上の効果があった！**

1. **GPU 常時 100% 稼働** → 温度が安定
2. **温度が安定** → クロックも安定（サーマルスロットリングなし）
3. **クロックが安定** → パフォーマンスも安定

**以前**：アイドル → 温度下がる → クロック上がる → 負荷 → 温度上がる → ファンが波打つ
**今**：常に負荷 → 温度一定 → クロック一定 → パフォーマンス一定！

### 既知のバグ 🐛

- エンドモルフィズム（endo_type=1, 2）で pubkey_x の不一致が発生
- 次のセッションで調査予定

**詳細は CLAUDE.md を参照**

---

## 🚀 Step 8: dG テーブルプリコンピュート（_PointMult 削減）

**アイデア**：連続秘密鍵戦略で、各スレッドの秘密鍵の間隔が `MAX_KEYS_PER_THREAD` に固定されていることを活用！

### 現状の問題

各スレッドが `_PointMult(base_key + global_idx * MAX_KEYS_PER_THREAD, G)` を呼んで最初の公開鍵を計算
→ 256 回のダブル＆アッド（重い！）

### 新しいアプローチ

1. **CPU 側で事前計算**：
   - `dG = MAX_KEYS_PER_THREAD * G`（定数）
   - `dG_table[i] = 2^i * dG`（24 エントリ、最大 1600 万スレッド対応）
   - `base_key` と `base_pubkey = base_key * G`

2. **GPU 側で高速計算**：
   - 秘密鍵: `base_key + global_idx * MAX_KEYS_PER_THREAD`（スカラー加算）
   - 公開鍵: `base_pubkey + global_idx * dG`
   - `global_idx * dG` は dG_table からルックアップ + PointAdd（最大 24 回）

### 期待効果

| 方式 | ダブル | アッド | 合計 |
|------|--------|--------|------|
| 従来の `_PointMult(256-bit)` | 256回 | ~128回 | ~384回 |
| **新方式（dG_table）** | **0回** | **~12回** | **~12回** |

**約 30 倍の削減！** 🔥

### タスク一覧

| # | タスク | 状態 |
|---|--------|------|
| 1 | CUDA: `_PointMultByIndex` 関数を作成（テーブルルックアップ + PointAdd） | ✅ |
| 2 | CUDA: `generate_pubkeys_sequential` を修正して `_PointMult` を削除 | ✅ |
| 3 | Rust: dG テーブル（24 エントリ）を計算する関数を作成 | ✅ |
| 4 | Rust: base_pubkey を計算してカーネルに渡す | ✅ |
| 5 | テスト: 正しい npub が生成されることを確認（39 テスト全通過）| ✅ |
| 6 | ベンチマーク: **4.135B keys/sec 達成！（+12.7%）** | ✅ 🔥🔥🔥 |
| 7 | 最適化: constant memory | 💡 不要（グローバルメモリで十分高速）|

### 結果

**+12.7% の高速化を達成！** 🎉

- 3.67B → **4.135B keys/sec**
- CPU 比 **59,071 倍**

`_PointMult`（256回のダブル＆アッド）を `_PointMultByIndex`（最大24回のPointAdd）に置き換えたことで、初期公開鍵計算のコストを大幅に削減。

---

## ✅ Step 9: Blocking Sync（2025-12-18 完了）🔥

### 背景

- `synchronize()` がスピンウェイト（busy wait）で CPU を 100% 使っていた
- 電力消費削減のために blocking sync に変更したい
- CUDA では `cuDevicePrimaryCtxSetFlags_v2` で `CU_CTX_SCHED_BLOCKING_SYNC` を設定

### 実装内容

**わずか 18 行の追加で実現！**

```rust
// import 追加
use cudarc::driver::sys::{
    cuDevicePrimaryCtxSetFlags_v2, CUctx_flags_enum::CU_CTX_SCHED_BLOCKING_SYNC,
};

// init_gpu() に追加
result::init()?;
let cu_device = result::device::get(0)?;
unsafe {
    cuDevicePrimaryCtxSetFlags_v2(cu_device, CU_CTX_SCHED_BLOCKING_SYNC as u32).result()?;
}
```

### 結果 🎉

| 指標 | Before | After |
|------|--------|-------|
| **CPU 使用率** | ~100% | **~1%** 🔥🔥🔥 |
| **スループット** | 4.135B | **4.136B** ✅ |
| **電力消費** | 高い | **大幅削減** 💡 |

### 学び 💡

- cudarc 0.18.2 では `cuDevicePrimaryCtxSetFlags_v2` が sys レイヤーで公開されている
- Safe API ラッパー（`primary_ctx::set_flags()`）は存在しない
- `CudaContext::new()` の **前** に `result::init()` と flags 設定が必要

---

## ✅ Step 10: 32-bit Prefix Matching（2025-12-18 完了）🔥

### 背景

- ncu プロファイリングで prefix matching ループが **17%** の Warp Stall を占めていた
- 64-bit 比較を 32-bit に変更して高速化を狙う
- 1 prefix のときに速度低下がないことを維持（State of the Art）

### 実装内容

**Phase 1: 32-bit 化 + 1 prefix 特殊化**

- GPU の patterns/masks を `uint64_t` → `uint32_t` に変更
- 共有メモリ使用量: 1024 bytes → 512 bytes（半減）
- 1 prefix 時はループをスキップ（最速パス維持）
- CPU 側で 64-bit 再検証（false positive フィルタリング）

**Phase 2: 32-bit × 2 連結**

- 2 つの 32-bit pattern を 64-bit に連結して同時チェック
- ループ回数が半減（32 prefix → 16 iterations）
- XOR + mask で効率的に 2 prefix を同時判定

### 結果 🎉

| ケース | 64-bit 比較 | 32-bit 最適化 | 変化 |
|--------|-------------|---------------|------|
| **1 prefix** | 4.135B | **4.131B** | 変化なし ✅ |
| **32 prefix** | 3.793B | **3.839B** | **+1.21%** 🔥 |

### 学び 💡

- 32-bit 比較は 64-bit より高速（特に複数 prefix 時）
- 1 prefix 特殊化で最速パスを維持できる
- 32-bit × 2 連結でループ回数を半減できる
- CPU 再検証のオーバーヘッドは無視できる（false positive は超レア）

### 今後の可能性

- **並列リダクション**：match の有無だけ先にチェック → match したときだけ詳細処理
- さらに数 % の改善が期待できる

---

## ✅ Step 11: Constant Memory（2025-12-19 完了）🔥

### 背景

- Step 10 で patterns/masks を shared memory にロードしていた
- shared memory は L1 キャッシュと共有されるため、使いすぎると L1 が減る
- constant memory は専用キャッシュを持ち、ブロードキャスト最適化がある

### 実装内容

**CUDA 側**：
- `__constant__ uint32_t _patterns[64], _masks[64], _num_prefixes` を追加
- shared memory のロードと `__syncthreads()` を削除
- カーネル引数から patterns, masks, num_prefixes を削除（3 引数減）

**Rust 側**：
- `module.get_global()` で constant memory を取得
- `memcpy_htod()` で mining 開始前に値を設定

### 結果 🎉

| ケース | Before (shared) | After (constant) | 変化 |
|--------|-----------------|------------------|------|
| **1 prefix** | 4.136B | **4.141B** | **+0.12%** |
| **32 prefix** | 3.839B | **3.954B** | **+3.0%** 🔥 |

### 学び 💡

- **constant memory は専用キャッシュ**：L1 とは別枠、使っても他のメモリアクセスに影響しない
- **ブロードキャスト最適化**：全スレッドが同じアドレスを読む → 1回のアクセスで済む
- **per-block ロード不要**：shared memory は各ブロックでロードが必要だったが、constant memory は mining 開始時に1回だけ
- **`__syncthreads()` 削減**：同期バリアのオーバーヘッドがなくなった
- **全 GPU 共通 64 KB**：Compute Capability 1.x から変わらない仕様

---

## ✅ Step 12: Max Prefix 256（2025-12-19 完了）🔥

### 背景

- 複数 prefix を指定すると、どれかに当たる確率が上がって効率的
- 従来は最大 64 prefix までしか指定できなかった
- constant memory の余裕を活かして 256 に拡張

### 実装内容

**CUDA 側**：
- `__constant__ uint32_t _patterns[64]` → `[256]`
- `__constant__ uint32_t _masks[64]` → `[256]`

**Rust 側**：
- `patterns_padded.resize(64, 0)` → `resize(256, 0)`
- `masks_padded.resize(64, 0)` → `resize(256, 0)`

**constant memory 使用量**：
- Before: 512 bytes
- After: 2048 bytes（2 KB / 64 KB = 3.1%）

### 結果 🎉

| ケース | Before | After | 変化 |
|--------|--------|-------|------|
| **1 prefix** | 4.141B | **4.146B** | 変化なし ✅ |
| **32 prefix** | 3.954B | **3.957B** | 変化なし ✅ |
| **64 prefix** | N/A | **3.759B** | 動作確認 ✅ |

### 学び 💡

- **速度への悪影響なし**：constant memory を 4 倍に拡張しても、キャッシュ効率は維持
- **64 prefix → 256 prefix**：より多くの prefix を同時に探せるようになった
- **constant memory は余裕がある**：64 KB 中 2 KB しか使っていない

---

## ✅ Step 13: Addition Chain（2025-12-20 完了）🔥🔥🔥

### 背景

- `_ModInv` は Fermat の小定理で `a^(p-2) mod p` を計算
- 従来の二乗-乗算法：256 回の二乗 + ~128 回の乗算
- 乗算が warp stall のボトルネックになっていた

### Addition Chain とは

**メモ化による累乗計算の最適化**！

- 中間結果（x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223）をキャッシュ
- 後から再利用して乗算回数を削減
- p-2 の特殊な構造を活用：ブロック長 {1, 2, 22, 223}

### 実装内容

**RustCrypto k256 / Peter Dettman の Addition Chain を参考に実装**：

```
x2 = a^3, x3 = a^7, x6 = a^63, x9 = a^511, x11 = a^2047
x22 = a^(2^22-1), x44 = a^(2^44-1), x88 = a^(2^88-1)
x176 = a^(2^176-1), x220 = a^(2^220-1), x223 = a^(2^223-1)
```

最終組み立て：`x223 * 2^23 + x22 * 2^5 + a * 2^3 + x2 * 2^2 + a`

### 結果 🎉

| 指標 | Before | After | 変化 |
|------|--------|-------|------|
| **乗算回数** | ~128 | **14** | **-114 回！** |
| **二乗回数** | 256 | 255 | -1 |
| **1 prefix** | 4.141B | **4.199B** | **+1.4%** 🔥 |
| **32 prefix** | 3.954B | **4.011B** | **+1.4%** 🔥 |

### ptxas 比較（sm_89）

| 指標 | Before | After | 差分 |
|------|--------|-------|------|
| スタック | 144,032 | 144,096 | +64 bytes |
| スピル | 0 | 24 bytes | +24 bytes |
| レジスタ | 96 | 96 | 同じ |

### 学び 💡

- **アルゴリズムの改善は上がり幅が大きい**：乗算 114 回削減の効果がスピル増加を上回った
- **Addition Chain は数学的概念**：ライセンス制約なし、自由に実装可能
- **中間変数のメモ化は有効**：GPU でもレジスタ圧が許容範囲なら効果大
- **32 prefix でも 4B 突破！**：大台に乗った 🎉

---

## ✅ Step 14: インライン PTX（2025-12-20 完了）🔥🔥🔥

### 背景

- PTX の carry chain 命令（`add.cc`/`addc.cc`）は **32-bit 専用**
- NVCC は 64-bit carry を `setp.lt.u64` + `selp.u64` の 3 命令で実装
- インライン PTX で 32-bit carry chain を使えば、1 命令で carry を伝播できる

### 実装内容

**`_Add256` と `_Sub256` を PTX carry chain で書き換え**：

```cuda
asm volatile (
    "add.cc.u32   %0, %9, %17;\n\t"    // r0 = a0 + b0, carry out
    "addc.cc.u32  %1, %10, %18;\n\t"   // r1 = a1 + b1 + carry
    ...
    "addc.u32     %8, 0, 0;\n\t"       // c = 0 + 0 + carry（最終 carry）
    : "=r"(r0), "=r"(r1), ... "=r"(c)
    : "r"(a0), "r"(a1), ... "r"(b7)
);
```

**PTX carry chain 命令**：
| 命令 | 意味 |
|------|------|
| `add.cc.u32` | 加算 + carry 出力 |
| `addc.cc.u32` | 加算 + carry 入出力 |
| `addc.u32` | 加算 + carry 入力のみ |

### 結果 🎉

| ケース | Before | After | 変化 |
|--------|--------|-------|------|
| **1 prefix** | 4.199B | **4.313B** | **+2.7%** 🔥 |
| **32 prefix** | 4.011B | **4.105B** | **+2.3%** 🔥 |

### 学び 💡

- **GPU の ALU は 32-bit が基本単位**：64-bit 演算も内部では 32-bit × 2 に分解
- **PTX の cvt 命令は SASS で消える**：レジスタ割り当てで解決（変換コストなし）
- **SASS の `IADD3`/`IADD3.X` が carry chain をネイティブサポート**
- **WSL と Windows で結果が異なる**：WSL では -12%、Windows では +2.7%（JIT 最適化の違い？）
- **インライン PTX は有効な最適化手法**：256-bit 演算が多い secp256k1 に特に効果的

### 今後の展望 🚀

- `_ReduceOverflow` や他の関数も `_Add64`/`_Addc64` で置き換え
- さらに数 % の改善が期待できる

---

## ✅ Step 15: _Add64/_Addc64 で _Reduce512 最適化（2025-12-21 完了）🔥🔥🔥

### 背景

- Step 14 で `_Add256`/`_Sub256` を PTX 化して +2.7% 達成
- `_Reduce512` 内にも `(sum < a) ? 1 : 0` パターンが大量にあった
- 64-bit 単位の加算プリミティブを作って置き換えを試みる

### 実装内容

**新しい 64-bit プリミティブを追加**：

```cuda
// carry-in なし（3 PTX 命令）
uint32_t _Add64(uint64_t a, uint64_t b, uint64_t* sum);

// carry-in あり（6 PTX 命令）
uint32_t _Addc64(uint64_t a, uint64_t b, uint32_t carry_in, uint64_t* sum);
```

**`_Reduce512` の 3 箇所を置き換え**：

1. `high * 977` の carry 処理
2. `shifted + mult977` (5 limbs) の加算
3. `temp + sum` (5 limbs) の加算

**Before（各箇所 8 行）**：
```cuda
uint64_t s1 = shifted[i] + mult977[i];
uint64_t carry1 = (s1 < shifted[i]) ? 1 : 0;
uint64_t s2 = s1 + carry;
uint64_t carry2 = (s2 < s1) ? 1 : 0;
sum[i] = s2;
carry = carry1 | carry2;
```

**After（各箇所 3-4 行）**：
```cuda
c = _Add64(shifted[0], mult977[0], &sum[0]);
for (int i = 1; i < 5; i++) {
    c = _Addc64(shifted[i], mult977[i], c, &sum[i]);
}
```

### 結果 🎉

| ケース | Before | After | 変化 |
|--------|--------|-------|------|
| **1 prefix** | 4.313B | **4.532B** | **+5.1%** 🔥🔥🔥 |
| **32 prefix** | 4.105B | **4.308B** | **+4.9%** 🔥🔥🔥 |

**CPU 比 64,743 倍！** 🚀

### 学び 💡

- **キャリーをキャリーとして計算することが重要**：`(sum < a) ? 1 : 0` は PTX で `setp + selp` になるが、PTX carry chain は 1 命令で carry を伝播
- **`_Reduce512` は頻繁に呼ばれる**：`_ModMult` の中で使われるため、最適化効果が大きい
- **64-bit プリミティブの汎用性**：他の関数（`_ReduceOverflow` 等）にも適用可能

---

## ✅ Step 16: _ReduceOverflow も PTX 化（2025-12-21 完了）🔥🔥🔥

### 背景

- Step 15 で `_Reduce512` を最適化したが、その中から呼ばれる `_ReduceOverflow` にも同じパターンがあった
- 8 箇所の `(x < y) ? 1 : 0` を `_Add64`/`_Addc64` で置き換え
- 60 行 → 28 行に簡素化（53% 削減）

### 実装内容

**Before（60行、複雑な carry 検出）**：
```cuda
uint64_t s1 = sum[0] + add0;
carry = (s1 < sum[0]) ? 1 : 0;
carry += carry0;
// ... 大量の三項演算子
```

**After（28行、シンプルな PTX carry chain）**：
```cuda
uint32_t c = _Add64(shifted_low, mult_low, &add0);
c = _Addc64(shifted_high, mult_high, c, &add1);
uint32_t c2 = _Add64(sum[0], add0, &sum[0]);
c2 = _Addc64(sum[1], add1, c2, &sum[1]);
c2 = _Addc64(sum[2], c, c2, &sum[2]);
c2 = _Addc64(sum[3], 0, c2, &sum[3]);
sum[4] = c2;
```

### 結果 🎉

| ケース | Before | After | 変化 |
|--------|--------|-------|------|
| **1 prefix** | 4.532B | **4.655B** | **+2.7%** 🔥 |
| **32 prefix** | 4.308B | **4.412B** | **+2.4%** 🔥 |

**CPU 比 66,500 倍！** 🚀

### 学び 💡

- **`_ReduceOverflow` は `_Reduce512` から呼ばれる** → 最適化効果が大きい
- **シンプルなコードは読みやすい** → 60行が28行に
- **PTX carry chain は効果的** → `setp + selp` のパイプラインストールを回避

### 今後の展望 🔮

`_ModMult` 系（`_ModMult`, `_ModMultByBeta`, `_ModSquare`）への適用は**逆効果**だった：
- 1 prefix: 4.655B → 4.549B（-2.3%）
- ループ内での `_Add64` 連打がレジスタ圧を増加させた可能性

**次のアプローチ候補**：
1. `_Addcc64`（3 値加算用）：`a + b + c` を効率的に計算
2. `_Addc64` の命令数削減：6 命令 → 4 命令
3. `_Sub64`/`_Subc64` を作成して borrow 検出を最適化

---

## ✅ Step 17: 即値化関数削除（2025-12-21 完了）🔥

### 背景

- `_ModMultByBeta`, `_ModMultByBeta2`, `_PointAddMixedG` は即値（#define）を使った特殊化版
- 以前のベンチマークで、即値化しても速度変化なしだった
- Karatsuba 実験に向けて `_ModMult` を 1 つに統一したい

### 実装内容

**削除した関数**：
- `_ModMultByBeta`（29行）→ `_ModMult(x, beta, result)` に置換
- `_ModMultByBeta2`（29行）→ `_ModMult(x, beta2, result)` に置換
- `_PointAddMixedG`（62行）→ `_PointAddMixed(..., Gx, Gy, ...)` に置換

**コード削減**：-126行、+7行 = **-119行**

### 結果 🎉

| ケース | Before | After | 変化 |
|--------|--------|-------|------|
| **1 prefix** | 4.655B | **4.681B** | **+0.6%** 🔥 |
| **32 prefix** | 4.412B | **4.423B** | **+0.2%** 🔥 |

**CPU 比 66,871 倍！** 🚀

### 学び 💡

- **命令キャッシュ効率**：同じ `_ModMult` を連打するほうが L1 命令キャッシュに優しい
- **コード整理で速くなることもある**：予想に反して +0.6% 改善
- **キャッシュ階層の理解**：
  - constant cache < #define 即値 < _ModMult 一本化（命令キャッシュ効率）

### 今後の展望：Karatsuba 法 🔮

**目標**：`_ModMult` の乗算回数を削減（16回 → 9回）

**階層構造**：
```
_Mult64   ← GPU/コンパイラが既に最適化してるかも（内部 32-bit）
   ↓
_Mult128  ← ここからカラツバ化の効果が出そう
   ↓
_Mult256  ← _Mult128 を呼ぶ or _ModMult にべた書き
```

**段階的アプローチ**：
1. まず `_Mult128` を Karatsuba 化
2. 効果を測定
3. 良ければ `_Mult256` も

---

## 📝 備考

**作業履歴の詳細**：`~/sakura-chan/diary/` と `~/sakura-chan/work/` に記録

**技術的な詳細**：`CLAUDE.md` を参照
