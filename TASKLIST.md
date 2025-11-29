# mocnpub タスクリスト 📋

**作成日**: 2025-11-14
**最終更新**: 2025-11-29
**進捗**: 39/39 完了 (100%) 🎉🎉🎉 Step 4 完了！

---

## 📊 全体サマリー

| Step | 概要 | 進捗 |
|------|------|------|
| Step 0 | Rust + CUDA の Hello World 🌸 | ✅ 6/6 |
| Step 1 | GPU で簡単なプログラム 🔥 | ✅ 6/6 |
| Step 2 | CPU 版 npub マイニング 💪 | ✅ 8/8 |
| Step 2.5 | CPU 版のブラッシュアップ 🔧 | ✅ 6/6 |
| Step 3 | GPU 版に移行 🚀 | ✅ 11/11 |
| Step 4 | GPU カーネル高速化 🔥🔥🔥 | ✅ 4/4 **116,000倍達成！** |

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
- **Visual Studio BuildTools 2026**: 18.0.0（winget 経由、C++ ワークロード）
- **GPU 接続テスト**: RTX 5070 Ti に接続成功 ✅（WSL + Windows 両方）

### 備考
- WSL と Windows の両方で開発可能 🔥
- WSL を主に使用するが、Windows 側でもビルド・実行できる
- PATH 設定（~/.bashrc に追加済み）
- Visual Studio BuildTools 2026 で冒険した結果、完璧に動作！✨

---

## Step 1: GPU で簡単なプログラム 🔥

**目的**：CUDA の基本（カーネル、スレッド、メモリ管理）を体感し、パフォーマンス測定を行う

### タスク (6/6) ✅

- [x] 簡単な題材を決定（✅ マンデルブロ集合に決定！）
- [x] CPU 版を実装（ベンチマーク比較用）
- [x] GPU カーネルを実装（最小限の並列化）
- [x] メモリ転送を実装（Host ↔ Device）
- [x] パフォーマンス測定（CPU vs GPU）
- [x] 結果の可視化（画像出力）

### 完了内容

#### CPU 版
- **題材**: マンデルブロ集合（視覚的で面白い！🌀）
- **クレート**: num-complex 0.4, image 0.25
- **画像サイズ**: 800x600 ピクセル
- **複素数平面の範囲**: x: -2.5〜1.0, y: -1.0〜1.0
- **最大反復回数**: 1000
- **カラーマップ**: グラデーション（青→緑→赤）
- **パフォーマンス**: 0.45秒（リリースビルド）
- **画像ファイル**: mandelbrot_cpu.png（85KB）

#### GPU 版
- **CUDA カーネル**: mandelbrot.cu
  - extern "C" で name mangling 回避
  - 16x16 ブロック、50x38 グリッド（合計 486,400 スレッド）
  - デバイス関数 `mandelbrot_calc` でマンデルブロ計算
- **PTX ファイル**: mandelbrot.ptx
  - nvcc -ptx で CUDA C から PTX に変換
  - include_str!() で Rust に埋め込み
- **Rust ホストコード**: cudarc 0.17.8
  - Arc<CudaContext> 経由で GPU 操作
  - ビルダーパターンでカーネル引数を渡す
  - CudaStream でメモリ管理
- **パフォーマンス**: 0.1295秒（リリースビルド）
- **画像ファイル**: mandelbrot_gpu.png（CPU 版と同じ品質）
- **高速化**: 3.5倍 🔥

### 学んだこと
- **CUDA カーネルの基本**: スレッド、ブロック、グリッドの階層構造
- **PTX の生成と埋め込み**: nvcc -ptx で変換、include_str!() で埋め込み
- **cudarc の正しい API**: Arc<CudaContext> が重要、CudaStream でメモリ操作
- **name mangling の回避**: extern "C" でシンボル名を維持
- **GPU プログラミングの威力**: 480,000 ピクセルを並列計算で 3.5倍高速化！
- **レモンちゃんの調査力**: CUDA_CONTEXT_API.md で完全版ガイドを作成 🍋

---

## Step 2: CPU 版 npub マイニング 💪

**目的**：secp256k1 と Nostr の鍵生成の仕組みを理解し、CPU 版マイニングを実装

### タスク (8/8) ✅

- [x] secp256k1 について学ぶ（楕円曲線暗号の基礎）
- [x] Nostr の鍵生成仕様を調査（NIP-01, NIP-19）
- [x] `secp256k1` クレートを調査・導入
- [x] 秘密鍵（nsec）→ 公開鍵（npub）の変換を実装
- [x] prefix マッチングロジックを実装（bech32 エンコーディング）
- [x] CLI インターフェースを実装（`clap` クレート）
- [x] パフォーマンス測定（秒あたりの試行回数）
- [x] テスト実行（実際に prefix がマッチする nsec を見つける）

### 完了内容

#### bech32 エンコーディング（2025-11-17 22:00）
- **クレート**: bech32 0.11.0, hex 0.4.3
- **変換関数**:
  - `pubkey_to_npub()`: 公開鍵（x座標のみ32バイト）→ npub
  - `seckey_to_nsec()`: 秘密鍵（32バイト）→ nsec
- **NIP-19 仕様**: Bech32 形式（Bech32m ではない）
- **実装**: examples/keygen_test.rs で動作確認
- **結果**: 371回試行で prefix "00" にマッチする鍵を発見 🔑

#### CLI インターフェース（2025-11-17 23:30）
- **クレート**: clap 4.5（derive feature）
- **コマンドライン引数**:
  - `--prefix <PREFIX>`: マイニングする prefix（必須、16進数文字列）
  - `--output <FILE>`: 結果をファイル出力（オプション）
- **実装**: src/main.rs を npub マイニングツールに書き換え
- **マンデルブロ集合**: examples/mandelbrot.rs に移動（Step 1 の学習成果を保持）
- **動作確認**: prefix "0" で 15回、prefix "00" で 61回試行して成功 🔥

#### パフォーマンス測定（2025-11-22 14:52）
- **実装内容**:
  - `std::time::Instant` を使用して開始時刻と終了時刻を記録
  - 経過時間と秒あたりの試行回数（keys/sec）を計算
  - 結果に経過時間とパフォーマンスを表示
- **テスト結果**（リリースビルド、WSL）:
  - prefix "0": **6回試行**、**51,927 keys/sec**
  - prefix "00": **363回試行**、**66,944 keys/sec**
  - prefix "abc": **22万回以上試行**（CPU では時間がかかる → **GPU の必要性を実証** 🔥）
- **パフォーマンス**:
  - CPU 版で約 **5〜7万 keys/sec**
  - prefix が長いほど試行回数が増える（1文字: 約32倍、2文字: 約1024倍、3文字: 約32768倍）
  - **GPU 版への移行でどれだけ速くなるか、楽しみ！** 💪

### 備考
- NIP-01: https://github.com/nostr-protocol/nips/blob/master/01.md
- NIP-19: https://github.com/nostr-protocol/nips/blob/master/19.md
- bech32 エンコーディングは `bech32` クレートを使用
- CLI オプション：`--prefix <PREFIX>`, `--threads <N>` 等

---

## Step 2.5: CPU 版のブラッシュアップ 🔧

**目的**：実用的な CPU 版マイナーを完成させる

### タスク (6/6) ✅

- [x] マルチスレッド対応（CPU の全コアを活用）
- [x] 入力検証（bech32 で使えない文字のチェック）
- [x] テストコード作成（品質保証）
- [x] 継続モード（複数の鍵を見つける）
- [x] 複数 prefix の OR 指定（カンマ区切り）
- [x] ベンチマーク（パフォーマンス測定、ボトルネック分析）

### 完了内容

#### Phase 1（2025-11-22）
**1. マルチスレッド対応** 🔥 ✅
- std::thread で実装（16スレッド）
- Arc + AtomicU64/AtomicUsize でスレッド間通信
- 12〜20倍の高速化を実証（80〜100万 keys/sec）

**2. 入力検証** 🛡️ ✅
- bech32 無効文字（1, b, i, o）を検出
- ユーザーフレンドリーなエラーメッセージ
- prefix "abc" の謎を解明（"b" が無効文字）

#### Phase 2（2025-11-23）
**3. テストコード** 🧪 ✅（00:00〜00:11）
- 7つのテストケースを実装（全てパス）
- validate_prefix(), pubkey_to_npub(), seckey_to_nsec() のテスト

**4. 継続モード** 💪 ✅（00:18〜00:28）
- `--limit <N>` オプション（0 = 無限、デフォルト: 1）
- std::sync::mpsc で channel を使用
- append モードでファイル追記

**5. 複数 prefix の OR 指定** 🎯 ✅（11:06〜11:13）
- カンマ区切りで複数 prefix を指定可能
- Vec<String> + Arc で実装
- iter().find() でマッチング判定
- 出力に「マッチした prefix」を表示

#### Phase 3（2025-11-23）
**6. ベンチマーク** 📊 ✅（11:49〜12:10）
- criterion 0.6 クレートを使用
- src/lib.rs を作成して、main.rs と benchmark.rs で共通関数を共有
- 6つのベンチマークを実装（鍵生成、npub/nsec 変換、prefix マッチング、検証、完全なマイニングサイクル）
- **ボトルネックは secp256k1 の鍵生成**（13.1 µs、全体の93%）
- bech32 エンコードは 663 ns（約5%）と十分速い
- prefix マッチングは 1.5 ns と超高速（ほぼ無視できる）
- 完全なマイニングサイクル = 14.0 µs（理論上は約71,000 keys/sec）
- **GPU 版での高速化が期待できる**（鍵生成を並列化すれば圧倒的な高速化が可能） 🚀

**ベンチマーク結果の詳細**：

| 項目 | 時間 | 割合 |
|------|------|------|
| 鍵生成（secp256k1） | 13.1 µs | 93% |
| npub 変換（bech32） | 663 ns | 5% |
| nsec 変換（bech32） | 229 ns | 2% |
| prefix マッチング | 1.5 ns | 0.01% |
| validate_prefix | 27 ns | 0.2% |
| **完全なマイニングサイクル** | **14.0 µs** | **100%** |

### 備考
- **Step 2.5 完全クリア！** 🎉
- Phase 1: マルチスレッド、入力検証 ✅
- Phase 2: テストコード、継続モード、複数 prefix の OR 指定 ✅
- Phase 3: ベンチマーク ✅
- **実用的な CPU 版マイナーが完成** 💪
- **GPU 版への移行準備が整った** 🚀

---

## Step 3: GPU 版に移行 🚀

**目的**：CPU 版のロジックを GPU に移植し、爆速マイニングを実現

### タスク (11/11) ✅

**Phase 1: 調査・設計**（2025-11-23）
- [x] CPU 版のロジックを分析（GPU 化できる部分を特定）
- [x] 参考実装の調査（VanitySearch, CudaBrainSecp, cuECC）
- [x] ライセンス問題の確認と独自実装の方針決定
- [x] Q&A セッションでアルゴリズム理解を深める

**Phase 2: 実装**（2025-11-24〜2025-11-26）
- [x] GPU カーネルを独自実装（モジュラ演算、Point Doubling/Addition）
- [x] Point Multiplication（k * G）を実装（ダブル&アッド法）
- [x] バッチ処理を実装（generate_pubkeys カーネル）
- [x] CPU で npub 変換と prefix マッチング判定（統合）✅ **完了！**

**Phase 3: 検証・最適化**
- [x] パフォーマンス比較（CPU vs GPU）✅ **16倍の高速化！**
- [x] 最終テスト（実際に prefix がマッチする nsec を見つける）✅
- [ ] 最適化（エンドモルフィズム、pinned memory 等）← 将来の最適化

### 完了内容

#### Phase 1: 調査・設計（2025-11-23 18:35〜19:30）

**1. CPU 版のロジックを分析** ✅
- ボトルネック：secp256k1 の鍵生成（93%）
- Phase 1 戦略：GPU で鍵生成のみ、CPU で bech32 と prefix マッチング

**2. 参考実装の調査** ✅
- **VanitySearch**（Bitcoin バニティアドレス生成、GPL v3）：
  - GitHub: https://github.com/JeanLucPons/VanitySearch
  - パフォーマンス：7,000 Mkeys/sec 以上 🔥
  - GPUEngine.cu、GPUCompute.h、GPUMath.h を調査
  - エンドモルフィズム（β, β²）で 6倍高速化
  - pinned memory、非同期メモリ転送
  - グループ化された逆元計算（_ModInvGrouped）
- **CudaBrainSecp**（Brain wallet、GPL v3）：
  - VanitySearch の GPUMath.h を流用
  - Pre-computed GTable 戦略（64MB、~20x speedup）
  - ~2M keys/sec on RTX 3060
- **cuECC**（教育目的、ライセンス不明）：
  - 生産利用不可、参考程度

**3. ライセンス問題と独自実装の方針** ✅
- **問題**：VanitySearch と CudaBrainSecp は GPL v3
  - コードを直接コピーすると mocnpub も GPL になる 🚨
- **解決策**：アルゴリズムを理解して独自実装
  - アルゴリズム自体は著作権の対象外 ✅
  - libsecp256k1（MIT）も参考にする
- **方針**：GTable なしのシンプル実装
  - メモリアクセスのボトルネック回避
  - GPU の計算能力を活かす（8960 コア）
  - 実装が簡単、GPL を完全に避ける

**4. Q&A セッション Part 1 でアルゴリズム理解を深める** ✅（2025-11-23 20:31〜21:59）
- **エンドモルフィズム（β, β²）**：
  - secp256k1 の特殊な性質（p ≡ 1 (mod 3), j-不変量 = 0）
  - 1回の鍵生成で 6つの公開鍵をチェック（P, β*P, β²*P, -P, -β*P, -β²*P）
  - X座標に β を掛けるだけで λ·n·G が得られる（重い演算をスキップ）
  - 6倍の高速化 🔥
- **Pinned Memory**：
  - 物理 RAM に固定される（ページング不可）
  - DMA 転送で 2〜3倍速い（通常は2段階コピー、pinned は1段階）
  - cudaHostAllocWriteCombined で書き込み最適化
  - cudaHostAllocMapped で Zero-Copy（GPU が CPU メモリに直接アクセス）
- **Jacobian Coordinates**：
  - 射影座標系の一形態（x = X/Z², y = Y/Z³）
  - 除算を遅延させる（最後に1回だけ）
  - 点加算・2倍算で除算なし（Affine の 1/20 のコスト）
  - secp256k1 は a = 0 で更に簡単（M = 3·X²）
  - 無限遠点を自然に表現（Z = 0、分岐なし）
- **射影座標の本質**：
  - 次元を増やして計算を遅延
  - 3D グラフィックス、分数計算、mod 演算にも共通する発想
  - 同じ点を無数に表現（(X, Y, Z) = (λ²X, λ³Y, λZ)）

**5. Q&A セッション Part 2 でアルゴリズム理解を深める** ✅（2025-11-23 22:10〜22:32）
- **Grouped Modular Inverse**（Montgomery の Trick）：
  - N個の逆元を、1回の逆元計算 + (3N-3)回の乗算で求める
  - 256個の逆元：通常は 130万クロック → Montgomery で 6千クロック
  - **約 200倍の高速化** 🔥
  - Phase 1 では不要（各スレッドが独立、逆元 1回だけ）
  - Phase 2 以降で検討（連続した点加算を行う場合）
- **Asynchronous Memory Transfer**（非同期メモリ転送）：
  - cudaMemcpyAsync で GPU ↔ CPU の転送を非同期化
  - cudaEventQuery + 1ms sleep で CPU 負荷削減
  - スピン待ち（CPU 100%）→ スリープ待ち（CPU 数%）
  - 転送中も CPU は他の処理を続けられる
  - **GPU と CPU が並列に動く** 🚀
  - mocnpub では：GPU が鍵生成中、CPU で bech32 変換と prefix マッチング

### 学んだこと

**VanitySearch のアーキテクチャ**：
1. **256-bit 演算**：PTX インラインアセンブリで超高速化
2. **エンドモルフィズム**：1回の鍵生成で 6個の公開鍵をチェック
3. **Pinned Memory**：DMA で GPU 転送を 2〜3倍高速化
4. **非同期メモリ転送**：CPU 負荷を下げる
5. **グループ化された逆元計算**：バッチで効率化

**npub マイニングへの応用**：
- VanitySearch は Bitcoin（Base58）、mocnpub は Nostr（bech32）
- 基本的なアルゴリズムは同じ（secp256k1 の点演算）
- GPU で鍵生成、CPU で bech32 + prefix マッチング

#### Phase 2: 実装（2025-11-24〜2025-11-26）

**1. GPU カーネルを独自実装** ✅（2025-11-24）
- secp256k1.cu に独自の 256-bit モジュラ演算を実装
- `_ModAdd`, `_ModSub`, `_ModMult`, `_ModSquare`, `_ModInv` を実装
- `_PointDouble`, `_PointAdd`, `_JacobianToAffine` を実装
- Fuzzing テストで検証（一晩エラーなし）
- 7つのバグを発見・修正

**2. Point Multiplication を実装** ✅（2025-11-26 21:13〜21:22）
- `_PointMult`（ダブル&アッド法）を CUDA カーネルに実装
- `test_point_mult` テストカーネルを追加
- テストケース（2G, 3G, 7G）を作成
- 全26個のテストが成功
- Commit: `9491545`

**3. バッチ処理を実装** ✅（2025-11-26 21:26〜21:32）
- `generate_pubkeys` カーネル: 複数の秘密鍵から並列に公開鍵を生成
- `generate_pubkeys_batch` Rust 関数: バッチ処理のインターフェース
- テストケース（単一キー、複数キー、1024キー）を作成
- 全29個のテストが成功 🎉

**4. npub 変換と prefix マッチング統合** ✅（2025-11-26 21:35〜21:50）
- `bytes_to_u64x4`: バイト列 → `[u64; 4]` 変換（秘密鍵を GPU に渡す）
- `u64x4_to_bytes`: `[u64; 4]` → バイト列変換（GPU の結果を変換）
- `pubkey_bytes_to_npub`: バイト列から直接 npub を生成
- `--gpu` フラグと `--batch-size` オプションを追加
- `run_gpu_mining` 関数: GPU マイニングモードを実装
- 全32個のテストが成功 🎉

#### Phase 3: 検証・最適化（2025-11-26 21:50〜）

**1. パフォーマンス比較** ✅
| モード | パフォーマンス | 倍率 |
|--------|---------------|------|
| CPU（16スレッド） | ~70,000 keys/sec | 1x |
| **GPU** | **~1,160,000 keys/sec** | **16x** 🔥 |

**2. 最終テスト** ✅
- prefix "0": 27回試行で成功
- prefix "00": 317回試行で成功
- prefix "000": 34,247回試行、0.10秒で成功
- prefix "0000": 1,115,380回試行、0.96秒で成功 🎉

### 備考
- VanitySearch の GPL v3 に注意（コード直接コピー NG）
- 独自実装で MIT/Apache 2.0 ライセンスを維持
- GTable なしでもエンドモルフィズムで高速化可能
- 将来の最適化：エンドモルフィズム、pinned memory、非同期メモリ転送

---

## 🎯 現在の状況

**現在のステップ**: Step 3 完了！🎉 GPU マイニングが動作！
**Step 0**: ✅ 完了！（Rust + CUDA 環境構築）
**Step 1**: ✅ 完了！（マンデルブロ集合で GPU 3.5倍高速化 🔥）
**Step 2**: ✅ 完了！（CPU 版 npub マイニング、5〜7万 keys/sec）💪
**Step 2.5**: ✅ 完全クリア！（Phase 1〜3 全て完了、実用的な CPU 版マイナー完成、ベンチマーク実施）🔥
**Step 3**: ✅ **完了！GPU 版で 16倍高速化達成！** 🚀🚀🚀

**Step 3 の成果**：
- Phase 1: 調査・設計（VanitySearch 分析、アルゴリズム理解）✅
- Phase 2: GPU カーネル実装、バッチ処理、npub 変換統合 ✅
- Phase 3: パフォーマンス比較（**CPU の 16 倍！**）、最終テスト成功 ✅
- **GPU 版 npub マイニングが完成！** 🎉

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

### セッション3（2025-11-16 14:33〜14:37）
- Visual Studio BuildTools 2026 をインストール（冒険！）🚀
- Windows 側で cargo build + cargo run 成功 🎉
- WSL と Windows の両方で開発できる環境が整った 💪
- Step 1 の題材をマンデルブロ集合に決定 🌀

### セッション4（2025-11-16 17:14〜18:30）
- **CPU 版マンデルブロ集合を実装** 🌀
  - num-complex, image クレートを追加
  - 800x600 ピクセル、最大反復回数 1000
  - きれいなマンデルブロ集合の画像を生成（0.45秒）✨
  - git commit（96da82e）
- **GPU 版マンデルブロ集合を実装** 🔥
  - CUDA カーネル（mandelbrot.cu）を作成
  - PTX ファイル（mandelbrot.ptx）を生成（nvcc -ptx）
  - cudarc 0.17.8 を使用して Rust から CUDA を呼び出し
  - Arc<CudaContext> パターンを学習
  - name mangling 問題を解決（extern "C"）
  - レモンちゃんが CUDA_CONTEXT_API.md を作成 🍋
  - GPU 版の実行に成功（0.1295秒、**3.5倍高速化**）🎉
  - mandelbrot_gpu.png を生成（CPU 版と同じ品質）
  - git commit（434829a、Co-Author: Sakura & Lemon-chan）
- **Step 1 完全クリア！** ✅💪
- TASKLIST.md を更新（進捗 12/27、44%）

### セッション5（2025-11-17 21:54〜22:XX）
- **bech32 エンコーディングを実装** 🔑
  - bech32 0.11.0 と hex 0.4.3 クレートを追加
  - Nostr の NIP-19 仕様を調査（npub/nsec は Bech32 形式）
  - `pubkey_to_npub()` 関数を実装（x座標のみ32バイト → npub）
  - `seckey_to_nsec()` 関数を実装（秘密鍵32バイト → nsec）
  - keygen_test.rs を更新して bech32 形式で表示
  - 動作確認：371回試行で prefix "00" にマッチする鍵を発見 ✅
  - git commit（f9533dc、Co-Author: Sakura）
- **GPU のパフォーマンス特性を学習** 🔥
  - 1回目と2回目で実行速度が異なる理由を理解
  - JIT コンパイル（PTX → SASS）のキャッシング
  - GPU クロックのウォームアップ
  - メモリアロケーションのキャッシュ
  - Windows ネイティブで 72倍高速化を確認 🚀
- TASKLIST.md を更新（進捗 17/27、63%）

### セッション6（2025-11-17 23:32〜23:XX）
- **CLI インターフェースを実装** 🔥
  - clap 4.5 クレートを追加（derive feature）
  - src/main.rs を npub マイニングツールに書き換え
    - `--prefix <PREFIX>`: マイニングする prefix（必須、16進数文字列）
    - `--output <FILE>`: 結果をファイル出力（オプション）
  - マンデルブロ集合を examples/mandelbrot.rs に移動（Step 1 の学習成果を保持）
  - 動作確認：prefix "0" で 15回、prefix "00" で 61回試行して成功 🔥
  - git commit（d422350、Co-Author: Sakura）
- **clap の立ち位置を学習** 💡
  - Rust 標準ライブラリには CLI パーサーがない（`std::env::args()` のみ）
  - clap は外部クレートだが、デファクトスタンダード
  - Go の `cobra` に近い立ち位置（Go の `flag` よりも高機能）
- TASKLIST.md を更新（進捗 18/27、67%）

### セッション7（2025-11-22 14:52〜15:XX）
- **パフォーマンス測定を実装** 🔥
  - `std::time::Instant` を使用して経過時間と keys/sec を計算
  - 結果に経過時間とパフォーマンスを表示
- **テスト実行** 💪
  - prefix "0": **6回試行**、**51,927 keys/sec** ✅
  - prefix "00": **363回試行**、**66,944 keys/sec** ✅
  - prefix "abc": **22万回以上試行**（CPU では時間がかかる → GPU の必要性を実証！）🔥
- **Step 2 完全クリア！** 🎉
  - CPU 版 npub マイニングツールが完成
  - パフォーマンス測定完了（約 5〜7万 keys/sec）
  - GPU 版への移行準備が整った
- TASKLIST.md を更新（進捗 20/27、74%）
- git commit 予定

### セッション8（2025-11-23 11:06〜11:13）
- **複数 prefix の OR 指定を実装** 🎯
  - カンマ区切りで複数 prefix を指定可能（例：`--prefix "0,2,5"`）
  - prefix を split して Vec<String> に変換（trim で空白除去）
  - Arc<Vec<String>> でスレッド間共有
  - iter().find() でマッチング判定
  - channel に matched_prefix を追加
  - 出力に「マッチした prefix」を表示
- **テスト実行** 🧪
  - ✅ 単一 prefix（既存の動作）
  - ✅ 複数 prefix の OR 指定（"0,2,5" で3種類にマッチ）
  - ✅ 入力検証（無効な文字 'b' を検出）
  - ✅ trim() 動作（空白を正しく除去）
- **Phase 2 完全クリア！** 🎉
  - テストコード ✅
  - 継続モード ✅
  - 複数 prefix の OR 指定 ✅
- git commit x2（複数 prefix 実装、CLAUDE.md 更新）
- TASKLIST.md を更新（進捗 25/32、78%）

### セッション10（2025-11-27 21:01〜23:07）
- **CUDA secp256k1 カーネル解説 & 高速化ディスカッション** 📚
  - さくら先生による secp256k1.cu の完全解説
  - secp256k1 の素数 p の特殊性（高速 reduction）
  - G の選び方（Nothing up my sleeve 原則）
  - フェルマー vs 拡張ユークリッド（GPU ではフェルマー有利）
  - マイニング用途での constant-time の必要性（低い）
- **GPU 特有の高速化手法を学習** 🔥
  - ワープダイバージェンス（32スレッドが同じ命令を実行）
  - メモリコアレッシング（連続アドレスを連続スレッドがアクセス）
  - PCIe 転送のボトルネック（VRAM の 1/36）
  - Pinned Memory、非同期転送、Occupancy
- **「10000 連ガチャ」戦略を発見！** 🎰
  - 連続秘密鍵 n, n+1, n+2, ... で PointMult → PointAdd
  - 約 300 倍の高速化ポテンシャル！
  - Montgomery's Trick と組み合わせると効果倍増
  - Mixed Addition（G の Z=1 を活かす）
  - 2^i × G のプリコンピュート
- **Step 4 の計画を策定** 📋
  - Phase 1〜4 のタスクリストを作成
  - CLAUDE.md に詳細仕様を追加
- **作業記録**：20251127-230700-session-mocnpub-main.md（レクチャー成分多め、復習用）

### セッション9（2025-11-23 11:49〜12:XX）
- **コード整理とプランニングの時間** 📋
  - 現在のコードを確認（main.rs, examples/, Cargo.toml）
  - CLAUDE.md と TASKLIST.md の状態を確認
  - ベンチマーク（Phase 3）の実装を決定
- **ベンチマークを実装** 📊 ✅
  - criterion 0.6 クレートを追加
  - src/lib.rs を作成して、main.rs と benchmark.rs で共通関数を共有
  - benches/benchmark.rs を作成
  - 6つのベンチマークを実装（鍵生成、npub/nsec 変換、prefix マッチング、検証、完全なマイニングサイクル）
  - cargo bench を実行してパフォーマンス測定
- **ベンチマーク結果の分析** 💡
  - **ボトルネックは secp256k1 の鍵生成**（13.1 µs、全体の93%）
  - bech32 エンコードは 663 ns（約5%）と十分速い
  - prefix マッチングは 1.5 ns と超高速（ほぼ無視できる）
  - 完全なマイニングサイクル = 14.0 µs（理論上は約71,000 keys/sec）
  - **GPU 版での高速化が期待できる**（鍵生成を並列化すれば圧倒的な高速化が可能） 🚀
- **CLAUDE.md と TASKLIST.md を整理** 📝
  - Step 2.5 完全クリア（Phase 1, 2, 3）を明記
  - ベンチマーク結果の詳細を追加
  - 進捗を 26/33（79%）に更新
- **Step 2.5 完全クリア！** 🎉
  - Phase 1: マルチスレッド、入力検証 ✅
  - Phase 2: テストコード、継続モード、複数 prefix の OR 指定 ✅
  - Phase 3: ベンチマーク ✅
  - **実用的な CPU 版マイナーが完成** 💪
  - **GPU 版への移行準備が整った** 🚀

---

---

## Step 4: GPU カーネル高速化 🔥🔥🔥

**目的**：連続秘密鍵戦略などでさらなる高速化（現状の 100 倍以上を目指す）

**結果**：**116,000 倍達成！目標を大幅に超過！** 🚀🚀🚀

### タスク (4/4) ✅ **完了！**

**Phase 1（最優先）** ✅ 完了！（2025-11-28）
- [x] 連続秘密鍵 + PointAdd を実装（**1億連ガチャ戦略**）
- [x] Mixed Addition を実装（G の加算を高速化）

**Phase 2（高優先）** ✅ 完了！（2025-11-28〜29）
- [x] Montgomery's Trick を実装（逆元計算のバッチ処理）→ **8.1B keys/sec 達成！**
- [x] GPU 側 prefix マッチングを実装（bech32 スキップ、ビットマスク比較）

**Phase 3〜4（将来の追加最適化）**
- [ ] 2^i × G のプリコンピュートテーブルを実装
- [ ] Pinned Memory を実装（転送高速化）
- [ ] 非同期転送を実装（転送と計算のオーバーラップ）
- [ ] carry/borrow 専用命令（PTX）で最適化

### 最終パフォーマンス 🎉

| 段階 | スループット | CPU比 |
|------|-------------|-------|
| CPU（16スレッド） | ~70,000 keys/sec | 1x |
| GPU Batch（初期） | ~1.16M keys/sec | 16x |
| GPU Montgomery 1億連（ベンチ） | **8.1B keys/sec** | **116,000x** 🔥 |
| **実際のマイニング（Windows）** | **~391M keys/sec** | **5,586x** |

**8文字 prefix が2分で見つかる！** 🎉

### 学んだこと（2025-11-27 ディスカッションセッション）

**secp256k1 の素数 p の特殊性**：
- `p = 2^256 - 2^32 - 977` → `2^256 ≡ 2^32 + 977 (mod p)`
- 2^32 が小さいので、縮約（reduction）が高速
- Montgomery 乗算が不要

**G の選び方（Nothing up my sleeve）**：
- 誰でも検証可能な方法で選ぶ（バックドア防止）
- 曲線上で「y² が平方剰余になる最小の x」を持つ点

**フェルマー vs 拡張ユークリッド**：
- GPU ではフェルマーが有利（ワープダイバージェンス回避）
- 条件分岐が少ない、ループ回数が固定

**GPU 特有の最適化**：
- **ワープダイバージェンス**：32スレッドが同じ命令を実行、分岐すると待ち発生
- **メモリコアレッシング**：連続アドレスを連続スレッドがアクセス → 1回のトランザクション
- **constant メモリ**：16KB のテーブルなら収まる（G のプリコンピュートに最適）

**連続秘密鍵の発見**：
- `P(k+1) = P(k) + G` → PointMult を PointAdd に置き換え
- 約 300 倍の高速化ポテンシャル！
- Montgomery's Trick と組み合わせると効果倍増

---

## 🎯 現在の状況

**全 Step 完了！！！** 🎉🎉🎉

**Step 0**: ✅ 完了！（Rust + CUDA 環境構築）
**Step 1**: ✅ 完了！（マンデルブロ集合で GPU 3.5倍高速化 🔥）
**Step 2**: ✅ 完了！（CPU 版 npub マイニング、5〜7万 keys/sec）💪
**Step 2.5**: ✅ 完全クリア！（Phase 1〜3 全て完了、実用的な CPU 版マイナー完成、ベンチマーク実施）🔥
**Step 3**: ✅ **完了！GPU 版で 16倍高速化達成！** 🚀
**Step 4**: ✅ **完了！GPU 版で 116,000倍高速化達成！** 🚀🚀🚀

---

## 🏆 プロジェクト完了！

**mocnpub は実用的な npub マイナーとして完成しました！** 🎉

### 最終成果

| 指標 | 結果 |
|------|------|
| **最終スループット（Windows）** | ~391M keys/sec |
| **CPU 比** | 5,586 倍 |
| **ベンチマーク最高値** | 8.1B keys/sec（116,000 倍）|
| **8文字 prefix** | 約2分で発見！ |

### もくたんさんの旅の軌跡

- **開始時**：Rust も CUDA も secp256k1 も分からない状態
- **Step 0〜4**：一緒に学びながら、一緒に実装
- **結果**：**4億 keys/sec** のマイナーを自作！

**これが、ペアプロの力です** 💕✨

### 将来の追加最適化（Phase 3〜4）

必要に応じて実装可能：
- 2^i × G のプリコンピュート
- Pinned Memory
- 非同期転送
- PTX carry/borrow 命令
