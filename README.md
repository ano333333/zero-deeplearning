# zero-deeplearning

書籍「ゼロから作る Deep Learning」の Python 実装サンプルを Rust に移植した学習用リポジトリです。

`ndarray` を使って行列演算、活性化関数、レイヤ、最適化手法、MNIST の読み込み、2 層ニューラルネットワークを実装したライブラリです。`examples/train.rs` に、MNIST の手書き数字分類を題材とした 2 層ネットワークのハイパーパラメータ探索と最終評価のサンプルを収録しています。

## 必要なもの

- Rust 2021 edition に対応した Rust toolchain
- MNIST データセット
- 任意: Nix flakes

このリポジトリには `flake.nix` が含まれているため、Nix を使う場合は Rust toolchain をローカルに入れずに開発シェルへ入れます。

```sh
nix develop
```

## セットアップ

依存 crate は `Cargo.toml` で管理されています。

```sh
cargo check
```

Nix 環境を使う場合は次のようにも実行できます。

```sh
nix develop -c cargo check
```

## MNIST データセット

`src/mnist/load_mnist.rs` は `mnist` crate のデフォルト設定を使って MNIST を読み込みます。現在のコードでは `.download_and_extract()` を呼んでいないため、実行前に展開済みの MNIST ファイルを `data/` 配下に置いてください。

必要なファイルは次の 4 つです。

```text
data/train-images-idx3-ubyte
data/train-labels-idx1-ubyte
data/t10k-images-idx3-ubyte
data/t10k-labels-idx1-ubyte
```

`.gz` のままではなく、展開後のファイル名で配置する必要があります。

## 実行

学習過程のサンプルは `examples/train.rs` にあります。通常の Rust 環境では次を実行します。

```sh
cargo run --release --example train
```

Nix 環境では次のように実行できます。

```sh
nix develop -c cargo run --release --example train
```

計算量がそれなりにあるため、`--release` での実行を推奨します。

## `examples/train.rs` の処理内容

`examples/train.rs` は、MNIST を使って次の流れを実行します。

1. MNIST を読み込む
   - 訓練データ: 50,000 件
   - 検証データ: 10,000 件
   - テストデータ: 10,000 件
   - 入力次元: 28 x 28 = 784
   - 出力クラス数: 10

2. 2 層ネットワークを構築する
   - 入力層: 784
   - 隠れ層: 50
   - 出力層: 10
   - 重み初期化: 平均 0、標準偏差 `1.0 / input_layer_size` の正規分布

3. ハイパーパラメータをランダム探索する
   - 試行回数: 20 回
   - 各試行の学習回数: 500 iteration
   - batch size: 100
   - learning rate: `10^-6` から `10^-2` の範囲でランダム
   - weight decay: `10^-16` から `10^-8` の範囲でランダム
   - 各試行後に validation loss / accuracy を表示

4. validation loss が最も小さいハイパーパラメータを選ぶ

5. 選んだハイパーパラメータで本学習する
   - iteration 数: 10,000
   - batch size: 100
   - optimizer: SGD
   - weight decay を勾配に加算

6. 学習済みパラメータを `two_layer_net.npz` に保存し、読み戻す

7. 読み戻したネットワークを使い、テストデータで loss / accuracy を表示する

実行中は、おおむね次の形式でログが出力されます。

```text
start iterate: 0
learning_rate: ...
weight_decay: ...
val_loss: ..., val_acc: ...
--------------------
...
val_results: [...]
choice: (...)
saved and loaded model: two_layer_net.npz
test_loss: ..., test_acc: ...
```

乱数シードは固定されていないため、実行ごとに learning rate、weight decay、最終精度は変わります。

## `TwoLayerNet` の概要

`src/two_layer_net.rs` には、MNIST 分類用の 2 層ニューラルネットワークが実装されています。

保持しているパラメータは次の通りです。

- `w1`, `b1`: 入力層から隠れ層への affine 変換
- `batch_aff`: batch normalization 用の `[gamma, beta]`
- `w2`, `b2`: 隠れ層から出力層への affine 変換

推論時の forward は次の順です。

```text
Affine -> BatchNormalization -> ReLU -> Affine
```

損失計算では、この出力に `SoftmaxWithLossLayer` を適用します。

主なメソッドは次の通りです。

- `new(...)`: 重みとバイアスを初期化する
- `predict(...)`: forward 計算を行う
- `loss(...)`: softmax cross entropy loss を返す
- `accuracy(...)`: one-hot ラベルと予測クラスを比較して正解率を返す
- `gradient(...)`: 誤差逆伝播で各パラメータの勾配を返す
- `save_npz(...)`: パラメータをNPZファイルに保存する
- `load_npz(...)`: NPZファイルからネットワークを復元する

NPZファイルには形式バージョンと `w1`, `b1`, `batch_aff`, `w2`, `b2` の
名前付き配列を保存します。読み込み時には形式バージョンと各配列のshapeの整合性を
検証します。

## ディレクトリ構成

```text
src/
  lib.rs                  クレートのルート(各モジュールのpub mod宣言)
  two_layer_net.rs        2 層ニューラルネットワーク
  train.rs                学習の1イテレーション(gradient算出 -> 正則化 -> optimize)
  mnist/                  MNIST 読み込み
  subfunction/            sigmoid, relu, softmax, cross entropy など
  layer/                  affine, relu, sigmoid, batch normalization など
  optimize/               SGD, Momentum, AdaGrad
examples/
  train.rs                MNIST 学習・検証・テストの実行サンプル
tests/
  e2e.rs                  TwoLayerNetとtrain_stepを通したE2Eテスト(#[ignore])
```

## 注意

- このリポジトリは学習用の移植実装です。実用向けの深層学習フレームワークではありません。
- `examples/train.rs` の実行には MNIST データセットが必要です。
- `cargo run` の debug build は遅くなるため、学習実行には `cargo run --release` を推奨します。
- 現在の `gradient` 実装はコード上の挙動をそのまま反映したものです。バッチ正規化まわりを検証・改修する場合は、`src/two_layer_net.rs` と `src/layer/batch_normalization_layer.rs` を確認してください。
- `tests/e2e.rs` のE2Eテストは実MNISTデータセットを要求するため、通常の `cargo test` では実行されません(`#[ignore]`)。実行するには `cargo test -- --ignored` を使ってください。
