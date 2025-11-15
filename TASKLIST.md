# mocnpub タスクリスト 📋

**作成日**: 2025-11-14
**最終更新**: 2025-11-15
**進捗**: 6/27 完了 (22%)

---

## 📊 全体サマリー

| Step | 概要 | 進捗 |
|------|------|------|
| Step 0 | Rust + CUDA の Hello World 🌸 | ✅ 6/6 |
| Step 1 | GPU で簡単なプログラム 🔥 | 0/6 |
| Step 2 | CPU 版 npub マイニング 💪 | 0/8 |
| Step 3 | GPU 版に移行 🚀 | 0/7 |

---

## Step 0: Rust + CUDA の Hello World 🌸

**目的**：CUDA ツールキットと Rust の開発環境をセットアップし、最小限の CUDA プログラムで動作確認

### タスク (6/6) ✅

- [x] CUDA Toolkit のインストール確認（Windows）
- [x] Rust のインストール確認（WSL + Windows）
- [x] Rust プロジェクトの初期化（`cargo init`）
- [x] CUDA バインディングクレートの調査（`cudarc`, `cuda-sys` 等）
- [x] 最小限の CUDA プログラムを実装（Hello World）
- [x] WSL でビルド・実行確認

### 完了内容
- **Windows**: CUDA Toolkit 13.0.88（winget 経由）
- **WSL**: CUDA Toolkit 13.0.88（apt 経由、cuda-toolkit-13-0）
- **Rust**: WSL 1.90.0、Windows 1.91.1
- **cudarc**: 0.17.8（CUDA 13.0 サポート確認済み）
- **GPU 接続テスト**: RTX 5070 Ti に接続成功 ✅

### 備考
- WSL で開発・ビルド・実行する方針に決定
- Windows 側もインストール済みだが、WSL を主に使用
- PATH 設定（~/.bashrc に追加済み）

---

## Step 1: GPU で簡単なプログラム 🔥

**目的**：CUDA の基本（カーネル、スレッド、メモリ管理）を体感し、パフォーマンス測定を行う

### タスク (0/6)

- [ ] 簡単な題材を決定（マンデルブロ集合 or 配列の足し算 or その他）
- [ ] CPU 版を実装（ベンチマーク比較用）
- [ ] GPU カーネルを実装（最小限の並列化）
- [ ] メモリ転送を実装（Host ↔ Device）
- [ ] パフォーマンス測定（CPU vs GPU）
- [ ] 結果の可視化（画像出力 or ログ出力）

### 備考
- マンデルブロ集合は視覚的で面白い 🎨
- 配列の足し算はシンプルで学習に最適 📚
- パフォーマンス測定には `std::time::Instant` を使用

---

## Step 2: CPU 版 npub マイニング 💪

**目的**：secp256k1 と Nostr の鍵生成の仕組みを理解し、CPU 版マイニングを実装

### タスク (0/8)

- [ ] secp256k1 について学ぶ（楕円曲線暗号の基礎）
- [ ] Nostr の鍵生成仕様を調査（NIP-01）
- [ ] `secp256k1` クレートを調査・導入
- [ ] 秘密鍵（nsec）→ 公開鍵（npub）の変換を実装
- [ ] prefix マッチングロジックを実装（bech32 エンコーディング）
- [ ] CLI インターフェースを実装（`clap` クレート）
- [ ] パフォーマンス測定（秒あたりの試行回数）
- [ ] テスト実行（実際に prefix がマッチする nsec を見つける）

### 備考
- NIP-01: https://github.com/nostr-protocol/nips/blob/master/01.md
- bech32 エンコーディングは `bech32` クレートを使用
- CLI オプション：`--prefix <PREFIX>`, `--threads <N>` 等

---

## Step 3: GPU 版に移行 🚀

**目的**：CPU 版のロジックを GPU に移植し、爆速マイニングを実現

### タスク (0/7)

- [ ] CPU 版のロジックを分析（GPU 化できる部分を特定）
- [ ] secp256k1 の GPU 実装を調査（rana の実装を参考に）
- [ ] GPU カーネルで鍵生成を実装（大量並列化）
- [ ] メモリ転送を最適化（バッチ処理）
- [ ] prefix マッチング判定を GPU で実装
- [ ] パフォーマンス比較（CPU vs GPU、何倍速くなったか）
- [ ] 最終テスト（実際に prefix がマッチする nsec を見つける）

### 備考
- rana は CPU 版のマイニングツール（GPU 版ではない）
- GPU 実装は独自に学びながら実装する
- メモリ転送がボトルネックになる可能性があるので注意
- カーネル最適化（共有メモリ、レジスタ使用量）は後回しでOK

---

## 🎯 現在の状況

**現在のステップ**: Step 1（GPU で簡単なプログラム）
**次のタスク**: 簡単な題材を決定（マンデルブロ集合 or 配列の足し算）
**Step 0**: ✅ 完了！

---

## 📝 作業履歴

### セッション1（2025-11-14 00:00〜00:45）
- プロジェクト開始 🎉
- 技術選択（Rust + CUDA）を決定
- 段階的アプローチ（Step 0〜3）を設計
- CLAUDE.md と TASKLIST.md を作成

### セッション2（2025-11-15 22:38〜23:04）
- Step 0 を完全クリア！🎉
- Windows に CUDA Toolkit 13.0 をインストール（winget）
- Windows に Rust 1.91.1 をインストール（winget）
- WSL に CUDA Toolkit 13.0 をインストール（apt）
- cudarc 0.17.8 を選択（CUDA 13.0 サポート確認）
- GPU デバイステストを実装・実行成功 ✅
- RTX 5070 Ti への接続確認完了 🔥
- git commit x2（cargo init、cudarc 追加）

---

**次回セッション**: Step 1 の題材選定から開始 🔥
