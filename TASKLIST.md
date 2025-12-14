# mocnpub タスクリスト 📋

**作成日**: 2025-11-14
**最終更新**: 2025-12-14
**進捗**: Step 0〜6 完了！Step 7（CUDA Streams & Phase Interleave）実験中 🔬

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
| Step 7 | CUDA Streams & Phase Interleave | 🔬 実験中 |

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
| **GPU + Multi-thread mining（2:1 batch ratio）** | **3.50B keys/sec** | **50,000x** 🔥🔥🔥 |

**8文字 prefix が約 6 分で見つかる！** 🎉

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
| CUDA Streams | 転送がボトルネックではない |
| **レジスタ削減** | スピル 96% 発生で逆効果（1.14B → 1.03B）|
| **CPU 公開鍵プリコンピュート** | CPU がボトルネック、Occupancy 改善なし（3.09B → 844M）|
| **`_PointMult` ブランチレス化** | アルゴリズム的に必要な分岐（ダブル＆アッド法のビット分岐）、計算量 2 倍になるため見送り |
| **`_Reduce512` ブランチレス化** | ダイバージェンス 99.16% を解消したが、実測で旧実装のほうが高速（3.20B vs 3.19B）→ 分岐予測が効いていた可能性 |
| **`_PointAdd` 重複計算最適化** | `_PointMult` 内でのみ使用（全体の約 8%）、レジスタ 128/128 でスピルリスク高、効果 0.1% 未満 |

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

## 🔬 Step 7: CUDA Streams & Phase Interleave（実験中）

**作業ブランチ**: `feature/double-buffer`（参照用に残す）

### 概念整理

**2つの別々の目標**：

| 戦法 | 目的 | 状態 |
|------|------|------|
| **ダブルバッファリング** | カーネル間の隙間を埋める | 🔬 実験中（効果確認済み）|
| **Phase Interleave** | 負荷の山を分散 | 💡 仮説段階 |

### feature/double-buffer での実験結果（2025-12-14）

| 実験 | 結果 |
|------|------|
| PTX キャッシュ | `cuModuleUnload` 問題を解決 ✅ |
| Pinned Memory | Memory 0.2% → <0.1%（効果は小さい）|
| Multi-thread mining（独立 context） | 隙間は埋まったが **並列実行はできず** |
| 2:1 batch_size 比率 | **3.50B keys/sec** 達成 🔥 |

**重要な発見**：
- 独立した CUDA Context を使っても、カーネルは並列実行されなかった
- シリアル実行だが、隙間はほぼ埋まった → 隙間埋め効果で +0.3% 改善

### 今後のタスク

| # | タスク | 内容 | 状態 |
|---|--------|------|------|
| 1 | **1 thread 2 stream お手玉版** | 本来のダブルバッファリングを実装 | 📋 TODO |
| 2 | **複数ダブルバッファリングの並列実行** | Phase Interleave の検証 | 📋 TODO |

**詳細は CLAUDE.md を参照**

---

## 📝 備考

**作業履歴の詳細**：`~/sakura-chan/diary/` と `~/sakura-chan/work/` に記録

**技術的な詳細**：`CLAUDE.md` を参照
