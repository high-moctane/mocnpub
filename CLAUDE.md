# mocnpub - Nostr npub マイニングプロジェクト 🔥

**Last Updated**: 2025-12-27

このファイルには mocnpub プロジェクトの開発方針、技術選択、現在の状況が記載されています。

---

## 🎯 プロジェクト概要

**mocnpub** は、Nostr の npub マイニングプログラムです。

**目的**：
- かっこいい prefix を持つ npub（自分のほしい prefix）になる nsec を探す
- GPGPU（CUDA）を使って爆速マイニング 🚀

**最終成果**：
- **5.8B keys/sec**（CPU の **82,857 倍**）🔥
- 8文字 prefix が約 4 分で見つかる

---

## 🛠️ 技術選択

### 言語：Rust 🦀

- WSL でも Windows でもビルド可能
- 暗号ライブラリが豊富（`secp256k1` の Rust バインディング）
- 数年後も動く（静的リンク、ランタイム不要）
- CUDA との連携が可能（`cudarc` クレート）

### GPGPU：CUDA 🔥

- RTX 5070 Ti（NVIDIA）に最適
- パフォーマンス最高（NVIDIA 専用最適化）
- 資料が豊富（学習しやすい）

---

## 🖥️ 開発環境

### ビルド方法

```bash
cargo build --release
```

PTX は `build.rs` で自動コンパイルされます。

### Windows との同期

- WSL で開発・commit・push
- Windows で `git pull` して実行
- Windows native で実行するとパフォーマンス最大化

---

## 🚀 開発の旅（6週間）

### 概要

| Step | 内容 | 成果 |
|------|------|------|
| Step 0-1 | 環境セットアップ、Mandelbrot | GPU 動作確認 |
| Step 2-2.5 | CPU 版マイナー | 70K keys/sec |
| Step 3 | GPU 版に移植 | 1.16M keys/sec (16x) |
| Step 4 | 連続秘密鍵 + Montgomery | 391M keys/sec (5,586x) |
| Step 5-13 | パラメータチューニング | 4.15B keys/sec (59,286x) |
| Step 14-35 | PTX 最適化 | **5.80B keys/sec (82,857x)** |

### 主要な最適化

**アルゴリズム**：
- 連続秘密鍵 + PointAdd（〜300x 軽量化）
- Montgomery's Trick（〜85x 逆元削減）
- エンドモルフィズム（2.9x カバレッジ）
- dG テーブルプリコンピュート（+12.7%）
- Addition Chain（乗算 128→14 回）
- Z² 累積積戦略（ModSquare 1600回削減）

**GPU 最適化**：
- Triple Buffering（GPU 常時 100% 稼働）
- Constant Memory（dG, patterns/masks）
- ブランチレス演算（_ModSub/_ModAdd）
- `__launch_bounds__(128, 5)`

**PTX 最適化**：
- インライン PTX（carry/borrow chain）
- _Add64x3, _Add320, _Sub256 等の専用関数
- パイプラインストール削減
- ループ融合

### 詳細ドキュメント

最適化の詳細は `docs/` を参照：
- `docs/JOURNEY.md` — 開発の物語
- `docs/OPTIMIZATION.md` — 技術的詳細
- `docs/LEARNING.md` — 学習の軌跡

---

## 📚 開発方針

### 学びながら進める 🌸
- GPGPU は未知の分野からスタート
- Rust もほぼ初心者からスタート
- secp256k1 も初めて
- **焦らず、丁寧に、落穂拾いしながら進める** 💕

### ファイル管理方針 📂
- 学習用ファイルも git 管理に含める
- プロジェクトの成長過程を大切に 🌱

---

## 🔗 参考プロジェクト

### rana
- **URL**: https://github.com/grunch/rana
- **言語**: Rust + CUDA
- **参考価値**: 実装の参考に（完全にコピーするのではなく、学びながら自分で作る）

---

## 📋 タスク管理

### 現在の状況（2025-12-27）

**リポジトリ公開に向けた仕上げフェーズに入っています** 🚀

- 最適化は完了（5.8B keys/sec、82,857x）
- ドキュメント作成などの仕上げ作業中

### 進行中のタスクリスト

** @docs/TASKLIST.md ** を参照してください。

公開向けドキュメントの作成進捗を管理しています。

### 言語について

**ここから先の文書や commit message は英語で作成します。**

（CLAUDE.md 自体は後で公開向けに作り直すので、この追記は日本語で書いています）
