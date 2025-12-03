# mocnpub タスクリスト 📋

**作成日**: 2025-11-14
**最終更新**: 2025-12-04
**進捗**: Step 0〜4 完了！🎉 CPU 公開鍵プリコンピュート戦法に挑戦中！

---

## 📊 全体サマリー

| Step | 概要 | 状態 |
|------|------|------|
| Step 0 | Rust + CUDA の Hello World | ✅ 完了 |
| Step 1 | GPU で簡単なプログラム（マンデルブロ集合）| ✅ 完了 |
| Step 2 | CPU 版 npub マイニング | ✅ 完了 |
| Step 2.5 | CPU 版のブラッシュアップ | ✅ 完了 |
| Step 3 | GPU 版に移行（16倍高速化）| ✅ 完了 |
| Step 4 | GPU カーネル高速化（116,000倍！）| ✅ 完了 |

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
| **GPU + batch_size 最適化（1146880）** | **3.09B keys/sec** | **44,000x** 🔥🔥🔥 |

**8文字 prefix が約 10 秒で見つかる！** 🎉

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

## 🔬 SoA 最適化の実験（2025-11-30）

### 実装内容

SoA（Structure of Arrays）最適化を実装：
- Montgomery's Trick のワークバッファをローカルメモリ → グローバルメモリ（SoA レイアウト）に移行
- `MAX_KEYS_PER_THREAD` のコンパイル時制限を撤廃
- keys_per_thread を任意の値に設定可能に

### 実験結果

**WSL での表示**（keys_per_thread を大きくすると劇的に速くなる...ように見える）：

| keys_per_thread | batch_size | 表示上の keys/sec |
|-----------------|------------|------------------|
| 1,408 (AoS) | 1,146,880 | 3.09B |
| 1,408 (SoA) | 71,680 | 974M |
| 32,768 | 2,240 | 6.87B |
| 65,536 | 1,120 | 13.6B |
| 524,288 | 140 | 111B |

**Windows での実測**：
- 表示: 192B keys/sec
- 体感: 約1分に1個（AoS 版と同じ）
- **結論**: keys/sec の計算にバグがある可能性 ⚠️

### 確認済み

✅ **生成される鍵は正しい**（nsec から event を作成して検証済み）

### 次のステップ（SoA 実験は一旦保留）

- [ ] ~~keys/sec の計算箇所を確認~~ → 一旦保留
- [ ] ~~AoS 版と SoA 版の実測比較~~ → 一旦保留

---

## 🚀 CPU 公開鍵プリコンピュート戦法（2025-12-04〜）

### アイデア 💡

**現状**：GPU カーネル内で `_PointMult(k, G)` を呼んで初期公開鍵を計算している

**新戦法**：CPU 側で初期公開鍵 `(Px, Py)` を計算して GPU に渡す

```
現状:
  CPU: k を生成 → GPU に転送
  GPU: _PointMult(k, G) で P を計算 ← 重い！
  GPU: +G を繰り返す

新戦法:
  CPU: k を生成
  CPU: secp256k1 ライブラリで P = k * G を計算 ← 高速！
  GPU に (k, Px, Py) を転送
  GPU: いきなり +G を開始！ ← _PointMult 不要！
```

### 期待効果

- **レジスタ削減** → Occupancy 改善の可能性
- **GPU カーネルの簡素化** → コンパイラ最適化が効きやすく？
- **CPU は暇してる** → CPU 側の計算増加は問題なし

### 実装タスク

- [ ] AoS 版に戻す（`git revert 347ac80`）
- [ ] CPU で `(Px, Py)` を計算する処理を追加（`src/main.rs`）
- [ ] GPU カーネルに `(k, Px, Py)` を渡すように変更（`src/gpu.rs`）
- [ ] GPU 側の `_PointMult` 呼び出しを削除（`cuda/secp256k1.cu`）
- [ ] 効果を測定（Windows で実測）

### 注意事項

- AoS 版（動作確認済み、3.09B keys/sec）をベースにする
- SoA 版のカウンタバグは後で調査

---

## 📋 将来の最適化計画

| # | 最適化 | 期待効果 | 優先度 |
|---|--------|----------|--------|
| 1 | **CPU 公開鍵プリコンピュート** | レジスタ削減 → Occupancy 改善 | **高**（実装中）|
| 2 | SoA カウンタバグ修正 | 正確な測定 | 中（一旦保留）|
| 3 | ダブルバッファリング | <1%（カーネル間隙間は 0.1%）| 低 |

### ✅ 完了した最適化

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

---

## 📝 備考

**作業履歴の詳細**：`~/sakura-chan/diary/` と `~/sakura-chan/work/` に記録

**技術的な詳細**：`CLAUDE.md` を参照
