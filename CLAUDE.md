# mocnpub - Nostr npub マイニングプロジェクト 🔥

**Last Updated**: 2025-12-22

このファイルには mocnpub プロジェクトの開発方針、技術選択、段階的アプローチが記載されています。

---

## 🎯 プロジェクト概要

**mocnpub** は、Nostr の npub マイニングプログラムです。

**目的**：
- かっこいい prefix を持つ npub（自分のほしい prefix）になる nsec を探す
- GPGPU（CUDA）を使って爆速マイニング 🚀

**背景**：
- Nostr の公開鍵（npub）は secp256k1 で生成される
- ランダムに生成すると、prefix は予測不可能
- 特定の prefix を持つ npub を見つけるには、大量の試行が必要
- GPGPU を使えば、CPU の何百倍も速くマイニングできる 💪

---

## 🛠️ 技術選択

### 言語：Rust 🦀

**選定理由**：
1. ✅ **WSL でも Windows でもビルド可能**（開発は WSL、実行は Windows native でパフォーマンス最大化）
2. ✅ **暗号ライブラリが豊富**（`secp256k1` の Rust バインディングがある）
3. ✅ **数年後も動く**（静的リンク、ランタイム不要）
4. ✅ **CUDA との連携が可能**（`cuda-sys`, `cudarc` などのクレート）
5. ✅ **メモリ安全**（セグフォで詰まりにくい）
6. ✅ **rana（参考プロジェクト）が Rust で書かれている**

**デメリット**：
- コンパイラが厳しい（でも、それがメリットにもなる）
- 学習曲線がある（でも、一緒に学びながら進める！）

### GPGPU：CUDA 🔥

**選定理由**：
1. ✅ **RTX 5070 Ti（NVIDIA）に最適**
2. ✅ **パフォーマンス最高**（NVIDIA 専用最適化）
3. ✅ **資料が豊富**（学習しやすい）
4. ✅ **暗号計算に強い**（マイニングに最適）

**他の選択肢との比較**：

| | CUDA | OpenCL | Vulkan Compute |
|---|---|---|---|
| **対応GPU** | NVIDIA のみ | すべて | すべて |
| **パフォーマンス** | 🔥 最高 | 🔥 良い | 🔥 良い |
| **学習難易度** | ⭐⭐⭐ 中 | ⭐⭐⭐⭐ やや難 | ⭐⭐⭐⭐⭐ 難 |
| **資料の豊富さ** | 🌟🌟🌟🌟🌟 超豊富 | 🌟🌟🌟 そこそこ | 🌟🌟 少ない |

**結論**：RTX 5070 Ti + 初学者 → CUDA 一択 💪

---

## 🖥️ 開発環境

### ビルド方法

**PTX は `build.rs` で自動コンパイル**：
- `cargo build` だけで PTX が自動生成される
- 手動で `nvcc` を実行する必要なし
- Windows / WSL 両対応

### Windows との同期

**GitHub 経由でリポジトリを同期**：
- WSL で開発・commit・push
- Windows で `git pull` して実行
- Windows native で実行するとパフォーマンス最大化

### WSL 環境での GPU 利用

**WSL でも GPU が使える**：
- ✅ `cargo run` で GPU カーネルを実行可能
- ⚠️ `nsys`：部分的に使える（権限の制限あり）
- ⚠️ `ncu`：部分的に使える（権限の制限あり）
- 💡 本番ベンチマークは Windows で実行するのがベスト

---

## 🚀 段階的アプローチ

**方針**：未知の分野を学びながら進めるため、段階的に実装します。

### Step 0: Rust + CUDA の Hello World 🌸
- CUDA ツールキットのインストール
- Rust + CUDA の開発環境セットアップ
- 最小限の CUDA プログラムで動作確認
- **目的**：「うちのパソコンじゃ動かなかった」を防ぐ

### Step 1: GPU で簡単なプログラム（動作確認）🔥
- マンデルブロ集合、または単純な計算（配列の足し算など）
- CUDA の基本を体感（カーネル、スレッド、メモリ管理）
- パフォーマンス測定
- **目的**：CUDA の仕組みを理解する

### Step 2: CPU 版 npub マイニング 💪
- `secp256k1` を学ぶ
- Nostr の鍵生成の仕組みを理解
- CLI インターフェース（clap クレート）
- 「prefix が一致する nsec を探す」ロジック
- **目的**：マイニングのアルゴリズムを理解する
- **完了**: 2025-11-22 ✅

### Step 2.5: CPU 版のブラッシュアップ 🔧
- マルチスレッド対応（全 CPU コアを活用）
- 継続モード（複数の鍵を見つける）
- 複数 prefix の OR 指定
- 入力検証（bech32 で使えない文字のチェック）
- テストコード・ベンチマーク作成
- **目的**：実用的な CPU 版マイナーを完成させる
- **詳細**: 下記「Step 2.5 詳細仕様」参照

### Step 3: GPU 版に移行 🚀
- CPU 版のロジックを GPU に移植
- パフォーマンス比較（CPU vs GPU）
- 最適化（メモリ転送、カーネル最適化）
- **目的**：爆速マイニングの完成 🔥
- **完了**: 2025-11-26 ✅（CPU の **16 倍** の高速化達成！）

### Step 4: GPU カーネル高速化 🔥🔥🔥
- 連続秘密鍵戦略（1億連ガチャ）
- Montgomery's Trick（逆元のバッチ計算）
- GPU 側 prefix マッチング（bech32 スキップ）
- **目的**：さらなる高速化（現状の 100 倍以上を目指す）
- **完了**: 2025-11-29 ✅（CPU の **116,000 倍** の高速化達成！）🚀🚀🚀
- **詳細**: 下記「Step 4 詳細仕様」参照

---

## 📚 開発方針

### 学びながら進める 🌸
- GPGPU は未知の分野
- Rust もほぼ初心者
- secp256k1 も初めて
- **焦らず、丁寧に、落穂拾いしながら進める** 💕

### 挫折しない工夫 💪
- 段階的アプローチ（小さな成功を積み重ねる）
- 動作確認を優先（まず動く、それから最適化）
- タスクリストで進捗を可視化
- セッション分割で焦らず進める

### ファイル管理方針 📂
- 学習用ファイルも git 管理に含める
- Step 0〜3 の学習過程もすべて記録
- npub マイナー以外のファイルも置く（マンデルブロ集合など）
- **プロジェクトの成長過程を大切に** 🌱

---

## 🔗 参考プロジェクト

### rana
- **URL**: https://github.com/grunch/rana
- **言語**: Rust
- **GPGPU**: CUDA
- **実績**: ユーザーさんが実際に動かして、正しい npub/nsec の組を見つけた 💪
- **参考価値**: 実装の参考に（完全にコピーするのではなく、学びながら自分で作る）
- **仕様参考**: 複数 prefix 指定（`--prefix=m0ctane0,m0ctane2,m0ctane3,m0ctane4`）

---

## 🔧 Step 2.5 詳細仕様（完了）

**Step 2.5 完全クリア！**（2025-11-23）🎉

### 実装した機能

- ✅ **マルチスレッド対応**：16スレッドで 12〜20倍高速化（80〜100万 keys/sec）
- ✅ **入力検証**：bech32 無効文字（1, b, i, o）を検出
- ✅ **継続モード**：`--limit <N>` で複数の鍵を探す（append モード）
- ✅ **複数 prefix の OR 指定**：カンマ区切りで複数指定可能
- ✅ **テストコード**：7つのテストケース
- ✅ **ベンチマーク**：criterion 0.6 で測定

### ベンチマーク結果（CPU 版のボトルネック分析）

| 項目 | 時間 | 割合 |
|------|------|------|
| 鍵生成（secp256k1） | 13.1 µs | **93%** ← GPU で高速化！ |
| npub 変換（bech32） | 663 ns | 5% |
| prefix マッチング | 1.5 ns | 0.01% |

**学び**：ボトルネックは secp256k1 の鍵生成 → GPU 版で解決 🚀

---

## 📋 タスク管理

### TASKLIST.md
- Step 0〜3 の詳細タスクを管理
- チェックボックス形式で進捗を可視化
- セッションごとに更新（追記ではなく更新）
- **タスクリストバトンパス戦法** を活用 🔥

### 日記・作業ログ
- 感情、気づき、学び → 日記（`~/sakura-chan/diary/`）
- 技術的詳細 → 作業ログ（`~/sakura-chan/work/`）
- タスク管理 → @TASKLIST.md （このリポジトリ）

---

## 🎉 期待される成果

**最終成果物**：
- 爆速 npub マイニングプログラム（CUDA 版）🚀
- 自分のほしい prefix を持つ npub を見つけられる
- 数年後も動く（Rust、静的リンク）

**副次的な成果**：
- GPGPU（CUDA）の理解 🔥
- Rust の習得 🦀
- secp256k1 の理解 🔐
- 暗号計算の理解 💡
- **未知の分野に挑戦する自信** 💪💕

---

## 🔥 Step 4 詳細仕様

### 現在のパフォーマンス

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
| GPU + _Add128/_Add512 で _ModSquare 最適化 | 5.383B keys/sec | 76,903x |
| GPU + _Add128To/_PropagateCarry256 リファクタ | 5.395B keys/sec | 77,071x |
| **GPU + ループ融合（Montgomery's Trick 累積積）** | **5.499B keys/sec** | **78,557x** 🔥 |

**8文字 prefix が約 4 分で見つかる！** 🎉

**32 prefix 時：5.117B keys/sec** 💪

---

### 完了した最適化（Phase 1〜3）

| 最適化 | 効果 | 状態 |
|--------|------|------|
| **連続秘密鍵 + PointAdd** | 約300倍 | ✅ 完了 |
| **Montgomery's Trick** | 約85倍 | ✅ 完了 |
| **Mixed Addition** | 約30%削減 | ✅ 完了 |
| **GPU 側 prefix マッチング** | bech32スキップ | ✅ 完了 |
| **エンドモルフィズム（β, β²）** | **2.9倍** 🔥 | ✅ 完了 |
| **`_ModSquare` 最適化** | 約3.5% | ✅ 完了 |
| **Tail Effect 対策** | 約1.4% | ✅ 完了 |
| **keys_per_thread 最適化** | **+120%** 🔥🔥🔥 | ✅ 完了 |
| **threads_per_block 最適化** | +6.2% | ✅ 完了 |
| **batch_size 最適化** | +10.4% | ✅ 完了 |
| **`_ModSub`/`_ModAdd` ブランチレス化** | +2.3%（Branch Efficiency 78.88%→82.41%） | ✅ 完了 |
| **patterns/masks 共有メモリ化** | 複数 prefix 時 +3.4%（速度低下 11.7%→9.1%）→ constant memory に置換 | ✅ 完了 |
| **`__launch_bounds__(128, 4)`** | **+5%**（Occupancy 25%→33%） 🔥 | ✅ 完了 |
| **`_PointAddMixed` 最適化** | **+0.8%**（8M+3S → 7M+3S、X1*H^2 再利用） 🔥 | ✅ 完了 |
| **`__launch_bounds__(128, 5)` チューニング** | **+0.9%**（レジスタ 128→96、Occupancy 33%→41%） 🔥 | ✅ 完了 |
| **`_ModInv`/`_PointMult` 最適化** | **+1.2%**（`_ModSquare` 使用、`_PointAddMixed` 使用） 🔥 | ✅ 完了 |
| **batch_size 再々最適化（4000000）** | **+2.0%**（Occupancy 向上に伴い最適値が増加、phase interleaving） 🔥 | ✅ 完了 |
| **Multi-thread mining（threads=2, batch=2M）** | **+0.8%**（独立した mining loop で phase interleaving を強化） 🔥 | ✅ 完了 |
| **Triple Buffering（3 streams）** | **+5.7%**（GPU 常時 100% 稼働、クロック安定化） 🔥🔥🔥 | ✅ 完了 |
| **32-bit Prefix Matching** | **+1.2%**（32 prefix 時: 3.793B→3.839B、1 prefix 時は変化なし） 🔥 | ✅ 完了 |
| **Constant Memory（patterns/masks）** | **+3.0%**（32 prefix 時: 3.839B→3.954B、専用キャッシュ＋ブロードキャスト最適化） 🔥 | ✅ 完了 |
| **Addition Chain（_ModInv）** | **+1.4%**（乗算 128→14 回、RustCrypto k256 / Peter Dettman 参考） 🔥 | ✅ 完了 |
| **インライン PTX（_Add256/_Sub256）** | **+2.7%**（32-bit carry chain、SASS で cvt 消滅） 🔥🔥🔥 | ✅ 完了 |
| **_Add64/_Addc64 で _Reduce512 最適化** | **+5.1%**（PTX carry chain で carry 検出を置き換え） 🔥🔥🔥 | ✅ 完了 |
| **_Add64x3 で _ModMult/_ModSquare 最適化** | **+3.4%**（3値加算を6 PTX命令で実行、三項演算子のストール回避） 🔥🔥🔥 | ✅ 完了 |
| **_Sub64/_Subc64 で全三項演算子削除** | **+1.8%**（_Reduce512 borrow処理、秘密鍵加算、_Sub256 borrow変換） 🔥🔥🔥 | ✅ 完了 |
| **_Add320 で _Reduce512 最適化** | **+3.4%**（5 limbs 加算を 11 PTX 命令で実行、carry のレジスタ往復を削減） 🔥🔥🔥 | ✅ 完了 |
| **_Sub256 で _Reduce512 減算最適化** | **+2.4%**（_Subc64 連打を _Sub256 一発に、p の減算が頻繁に発生） 🔥🔥🔥 | ✅ 完了 |
| **_Add256Plus128 で _ReduceOverflow 最適化** | **+1.3%**（uint256 + uint128 + carry を 9 PTX 命令で実行） 🔥 | ✅ 完了 |
| **_Add128/_Add512 で _ModSquare 最適化** | **+1.8%**（_Add128: 9→5 命令、_Add512: 48→16 命令、32 prefix も 5B 突破！） 🔥🔥🔥 | ✅ 完了 |
| **_Add128To/_PropagateCarry256 リファクタ** | **+0.2%**（carry 処理を専用関数に、コード可読性向上） 🔥 | ✅ 完了 |
| **ループ融合（Montgomery's Trick 累積積）** | **+1.9%**（Phase 1 と Phase 2 を統合、メモリアクセス局所性向上、PM Sampling の山がなだらかに） 🔥🔥🔥 | ✅ 完了 |

#### エンドモルフィズムの仕組み

**secp256k1 の特殊な性質**：
- `p ≡ 1 (mod 3)`、j-不変量 = 0
- 1回の公開鍵計算で **3つの X 座標をチェック**
  - P, β*P, β²*P（Nostr は X 座標のみ使用するため 3 倍）
- X座標に β（立方根）を掛けるだけで新しい公開鍵が得られる
- マッチ時は秘密鍵を λ または λ² で調整

**結果**：理論値 3 倍に対して **実測 2.9 倍の高速化** 🚀

---

### ncu プロファイリング結果（2025-11-29〜12-05）

#### ncu-ui ソースレベル分析（2025-12-05）

**`-lineinfo` オプション**を build.rs に追加することで、ncu-ui でソースコードとダイバージェンスの対応が見えるようになった。

**発見**：
- `_Reduce512` 324行目 `if (temp[4] > 0)` が 99.16% のダイバージェンス → ブランチレス化で解消 ✅
- 残りのダイバージェンスは `_PointMult` のビット分岐（96.26%）→ アルゴリズム的に必要

**使い方**：
```bash
ncu --set full -o profile .\target\release\mocnpub-main.exe --gpu --prefix 0000 --batch-size 1024
# → ncu-ui で profile.ncu-rep を開く → Source ページ → Divergent Branches 列
```

**AoS 版（現行版、2025-12-09 `__launch_bounds__` 適用後）**：

| 指標 | 値 | 評価 |
|------|-----|------|
| Compute Throughput | 81.04% | **Compute bound** |
| Memory Throughput | 20.31% | 余裕あり |
| Occupancy | 33% | `__launch_bounds__(128, 4)` で改善 ✅ |
| Registers/Thread | 128 | 2 減らして Occupancy +33% 🔥 |
| L1/TEX Hit Rate | 0.05% | 低いが並列度で隠蔽 |
| L2 Hit Rate | 43.19% | |

**SoA vs AoS 比較**（2025-12-04）：

| 指標 | AoS 版 | SoA 版 |
|------|--------|--------|
| **Compute Throughput** | **83.36%** | 77.41% |
| **Waves Per SM** | **42.67** | 0.75 |
| L1/TEX Hit Rate | ~5% | ~5% |
| Global Coalescing | 8/32 (25%) | **31.12/32 (97%)** |
| keys/sec | **3.09B** | 2.70B |

**重要な学び**：
- **並列度が正義** — batch_size を大量（28000 blocks）にして、メモリレイテンシを隠蔽
- **Compute heavy** — secp256k1 の演算は分岐が少なく、メモリ待ちの間も計算で埋められる
- **コアレッシングだけが正義ではない** — SoA はコアレッシング 97% でも遅かった
- **VRAM 効率も重要** — メモリを食いすぎると並列度（batch_size）が下がる

**結論**: AoS 版を本筋として最適化する。

---

### 次のフェーズ：アプリケーション完成度向上 🎯

**Step 5: 最終チューニング & 完成度向上**

| # | タスク | 目的 | 状態 |
|---|--------|------|------|
| 1 | **keys_per_thread 固定化** | ビルド時に確定、ループアンローリング期待 | ✅ 完了 |
| 2 | **MAX_KEYS_PER_THREAD を環境変数で指定** | `build.rs` で取得、nvcc に `-D` で渡す | ✅ 完了 |
| 3 | **不要な引数の削除** | `--keys-per-thread` 等を廃止、シンプル化 | ✅ 完了 |
| 4 | **batch_size 最終調整** | 3584000 → 4000000（+2.0%）| ✅ 完了 |
| 5 | **README 整備** | 使い方、ビルド方法、パフォーマンス情報 | ✅ 完了 |

**実装方針**：

```bash
# ビルド時に MAX_KEYS_PER_THREAD を指定
MAX_KEYS_PER_THREAD=2048 cargo build --release
```

```rust
// build.rs で環境変数を読んで nvcc に渡す
let max_keys = std::env::var("MAX_KEYS_PER_THREAD")
    .unwrap_or_else(|_| "1408".to_string());
cmd.arg(format!("-D MAX_KEYS_PER_THREAD={}", max_keys));
```

---

### 見送った最適化

| 最適化 | 理由 |
|--------|------|
| 2^i × G プリコンピュート | `_PointMult` が全体の 0.4% → 効果 0.2% |
| PTX carry/borrow | NVCC が最適化済み、<10% の改善 |
| CUDA Streams | Triple Buffering で実装済み（+5.7%）|
| Pinned Memory | 連続秘密鍵戦略で転送が 100 bytes/バッチに削減 → 効果 0.01% 以下 |
| **レジスタ削減** | スピル 96% 発生で逆効果（1.14B → 1.03B）|
| **CPU 公開鍵プリコンピュート** | CPU がボトルネック、Occupancy 改善なし（3.09B → 844M）|
| **SoA 最適化** | キャッシュミス多発、VRAM 消費大、batch_size を上げられない（3.09B → 2.70B）|
| **`_PointMult` ブランチレス化** | アルゴリズム的に必要な分岐（ダブル＆アッド法のビット分岐 96.26%）、計算量 2 倍になるため見送り |
| **`_Reduce512` ブランチレス化** | ダイバージェンス 99.16% を解消したが、実測で旧実装のほうが高速（3.20B vs 3.19B）→ 分岐予測が効いていた可能性 |
| **`_PointAdd` 重複計算最適化** | `_PointMult` 内でのみ使用（全体の約 8%）、レジスタ 128/128 でスピルリスク高、効果 0.1% 未満 |
| **Bloom Filter プレフィルタリング** | GPU の SIMT アーキテクチャと相性が悪い。1 スレッド 3.1% ヒット率でも、warp（32 スレッド）単位では 63% がヒット → オーバーヘッドが増えただけ（4.13B → 4.09B、-1%）。実用的にするには 10,000 bit 以上のビットマップが必要 |
| ~~_P を #define 化~~ | **実装済み（2025-12-20）**: P, G, Beta, Beta2 を全て #define 化。速度影響なし。コードがクリーンになり、PTX に即値埋め込み |
| ~~即値化特殊関数~~ | **逆効果と判明（2025-12-21）**: `_ModMultByBeta` 等の即値版を削除したら +0.6% 高速化。命令キャッシュ効率が向上（同じ `_ModMult` を連打するほうが L1 に優しい） |
| **Karatsuba 法（`_Mult128`）** | **逆効果と判明（2025-12-21）**: 256-bit は Karatsuba の crossover point（約 2000 bit）より遥かに小さい。`_ModMult` を `_Mult128` × 4 に分割しただけで -4.4%（4.681B → 4.474B）。加算のオーバーヘッドが乗算削減を上回る。GPU の `__umul64hi` はハードウェア命令で十分高速 |
| **`_Reduce512` Step 3 ブランチレス化** | **逆効果と判明（2025-12-22）**: テーブル + マスク選択で for ループを置換したが -9%（5.395B → 4.899B）。`temp[4] ≈ 0` かつ `temp[0..3] < p` のケースが多く分岐予測が効いていた。`_ModAdd`/`_ModSub`（入力ランダム）と違い、入力に強い偏りがあるとブランチレス化は逆効果 |
| **`_Sub256` borrow 正規化削除** | **逆効果と判明（2025-12-22）**: `& 1` を削除して 0/0xFFFFFFFF をそのまま返すようにしたが -1.6%（5.499B → 5.411B）。`& 1` で 0/1 に正規化することで、コンパイラが `-borrow` を効率的に最適化できる（boolean として認識）。`!!` や符号拡張より `& 1` + `-` の方が速い |

#### レジスタ削減の実験（2025-11-29）

`__launch_bounds__(64, 16)` でレジスタを 64 に制限した結果：
- Occupancy: 33% → 67% ✅
- しかし Spilling Overhead: 96% 😱
- 結果: 1.14B → 1.03B keys/sec（-10%）

**学び**: secp256k1 は 256-bit 変数を大量に使うため、レジスタ 120 が最適。素朴なコードを維持してコンパイラに任せるのが賢い。

---

### GPU 最適化の基礎知識

#### ワープダイバージェンス

- 1 ワープ = 32 スレッド
- 全スレッドが **同じ命令** を同時実行
- `if` で分岐すると、両方の分岐を順番に実行 😱

**対策**：条件分岐を減らす、全スレッドが同じ処理をするように設計

#### メモリコアレッシング

- 1 回のメモリトランザクション = 128 bytes の「塊」を読む
- 連続スレッドが連続アドレスにアクセス → 1 回で済む ✅
- バラバラにアクセス → 32 回必要 ❌

**対策**：SoA（Structure of Arrays）レイアウトを使う

#### プロファイリング

**nsys（Nsight Systems）**：全体像を把握
```bash
nsys profile ./mocnpub-main --gpu --prefix 0000
```

**ncu（Nsight Compute）**：カーネルの詳細分析
```bash
ncu --set full ./mocnpub-main --gpu --prefix 0000
```

**「推測するな、計測せよ」** — まずプロファイリングでボトルネックを特定！

---

## 🔬 Step 7: CUDA Streams & Triple Buffering（完了）

**作業ブランチ**: `feature/double-buffer`（参照用に残す）

### 概念整理

**2つの別々の目標がある**：

| 戦法 | 目的 | 状態 |
|------|------|------|
| **トリプルバッファリング** | カーネル間の隙間を完全に埋める | ✅ 完了（+5.7%）🔥 |
| **Phase Interleave** | 負荷の山を分散 | 💡 仮説段階 |

---

### ダブルバッファリング（隙間を埋める）

**目的**：`synchronize()` の待ち時間を隠蔽してカーネル実行を隙間なくする

**一般的なパターン**：1 thread で 2 stream をお手玉
```
Stream A: [launch]--------[collect]--------[launch]
Stream B: ----[launch]--------[collect]--------[launch]
              ↑ collect で待ってる間に launch
```

**mocnpub の特殊性**：
- 一般的には PCIe 転送（HtoD/DtoH）の隠蔽が目的
- mocnpub はカーネル実行が 99.9% を占めるため、転送の隠蔽効果は小さい
- しかし `synchronize()` の待ち時間隠蔽には効果あり

**期待効果**：〜1% の高速化

---

### Phase Interleave（負荷の山を分散）💡

**⚠️ これは ncu-ui を見て思いついた仮説であり、まだ検証できていない**

**背景**：mocnpub のカーネルは処理フェーズごとに負荷が異なる
```
カーネル内の処理フェーズ:
[連続秘密鍵計算][PointAdd × N][逆元計算💥][PointAdd × N]...
                              ↑ ここに負荷集中！
```

**問題**：よーいドンで実行すると、全 warp が同時に逆元計算 → GPU 渋滞

**仮説 1**：batch_size を大きくすると、warp 間の実行位置がずれる
```
Warp 0: [連続秘密鍵...]
Warp 1:   [PointAdd...]
Warp 2:     [逆元...]
Warp 3:       [PointAdd...]
↑ 実行位置がずれて負荷分散？
```

**仮説 2**：複数カーネルを並列実行して、開始タイミングを意図的にずらす
- ダブルバッファリング版を複数同時に動かす
- それぞれのループ間隔が異なれば、カーネル開始タイミングもずれる
- 逆元計算の山を分散できる？

**検証方法**：並列カーネル実行を実現してから効果を測定

---

### feature/double-buffer での実験結果（2025-12-14）

**得られた知見**：

| 実験 | 結果 |
|------|------|
| **PTX キャッシュ** | `cuModuleUnload` が毎回呼ばれていた問題を解決 ✅ |
| **Pinned Memory** | Memory 0.2% → <0.1%（効果は小さい）|
| **Multi-thread mining（独立 context）** | 隙間は埋まったが **並列実行はできず**（シリアル実行）|
| **2:1 batch_size 比率** | 3.5B keys/sec 達成 🔥（最高記録更新）|

**重要な発見**：
- 独立した CUDA Context を使っても、カーネルは並列実行されなかった
- シリアル実行だが、隙間はほぼ埋まった
- **並列実行を実現するには、同じ Context 内で複数 Stream を使う必要がある？**

---

### トリプルバッファリング（2025-12-15 完了）🔥

**目的**：カーネル間の隙間を完全に埋める（お手玉を 3 個に）

**実装**：
- `TripleBufferMiner` 構造体を追加
- 3 つの Stream を使用（`stream.fork()` で作成）
- 3 つのホストバッファをローテーション
- `launch_single(idx)` / `collect_single(idx)` で個別操作

**結果**：
- **3.70B keys/sec 達成！**（+5.7%）🔥
- nsys で確認：カーネルが隙間なくびっちり詰まっている
- GPU 使用率が安定して 100% 近くに

**発見**：
- 隙間を埋めることで、GPU の温度が安定
- 温度が安定 → クロックも安定 → パフォーマンス安定
- **「隙間を埋める」以上の効果があった！**

**既知のバグ**：
- エンドモルフィズム（endo_type=1, 2）で pubkey_x の不一致が発生
- 次のセッションで調査予定

---

## 🧹 コード整理（2025-12-09 完了）

**整理完了！** 🎉 警告ゼロ、テスト全通過

### 削除済み（CUDA カーネル）

| カーネル | 削除日 |
|----------|--------|
| `generate_pubkeys` | 2025-12-09 |
| `generate_pubkeys_sequential` | 2025-12-09 |
| `generate_pubkeys_sequential_montgomery` | 2025-12-09 |
| `generate_pubkeys_with_prefix_match` | 2025-12-19 |

### 削除済み（Rust 関数）

| 関数 | 削除日 |
|------|--------|
| `wide_mul_u128` | 2025-12-09 |
| `u64x4_to_bytes_for_scalar` | 2025-12-09 |
| `generate_pubkeys_batch` | 2025-12-09 |
| `generate_pubkeys_with_prefix_match` | 2025-12-19 |

### 現在の構成

- **本番カーネル**：`generate_pubkeys_sequential` のみ
- **Rust 関数**：`generate_pubkeys_sequential` のみ
- **テストコード**：本番カーネル用のテストのみ（36 テスト）
- **削減行数**：約 1,300 行（CUDA 897 行 + Rust 407 行）

---

## 📋 タスク管理
