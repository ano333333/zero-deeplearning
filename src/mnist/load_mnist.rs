use mnist::*;
use ndarray::prelude::*;

/// MNISTデータセットを読み込む
///
/// MNISTデータセットを読み込み、訓練データ、訓練ラベル、検証データ、検証ラベルの4つの配列を返す。
///
/// # Arguments
///
/// * `training_size` - 訓練データのサイズ。省略された場合は50000。
/// * `validation_size` - 検証データのサイズ。省略された場合は10000。
///
/// # Returns
///
/// * `(train_data, trn_lbl, validation_data, val_lbl)` - 訓練データ、訓練ラベル、検証データ、検証ラベルの4つの配列。
///   * `train_data` - 訓練データ。形状は(訓練データのサイズ, 28*28)。
///   * `trn_lbl` - one-hot形式の訓練ラベル。形状は(訓練データのサイズ, 10)。
///   * `validation_data` - 検証データ。形状は(検証データのサイズ, 28*28)。
///   * `val_lbl` - one-hot形式の検証ラベル。形状は(検証データのサイズ, 10)。
///
/// # Examples
/// ```
///     let (train_data, trn_lbl, validation_data, val_lbl) = mnist::load_mnist::load_mnist(None, None);
/// ```
pub fn load_mnist(
    training_size: Option<u32>,
    validation_size: Option<u32>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let training_size = training_size.unwrap_or(50_000);
    let validation_size = validation_size.unwrap_or(10_000);
    let Mnist {
        trn_img,
        trn_lbl,
        val_img,
        val_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(training_size)
        .validation_set_length(validation_size)
        .test_set_length(10_000)
        .finalize();
    // trn_imgとval_imgを28*28要素づつに分け、全ての値を[0,256)から[0.0,1.0)に正規化
    let train_data = Array2::from_shape_vec((training_size as usize, 28 * 28), trn_img).unwrap();
    let train_data = train_data.mapv(|x| x as f64 / 256.0);
    let validation_data =
        Array2::from_shape_vec((validation_size as usize, 28 * 28), val_img).unwrap();
    let validation_data = validation_data.mapv(|x| x as f64 / 256.0);
    // trn_lblとval_lblをone-hot表現に変換
    let trn_lbl = Array2::from_shape_fn((training_size as usize, 10), |(i, j)| {
        if trn_lbl[i] == j as u8 {
            1.0
        } else {
            0.0
        }
    });
    let val_lbl = Array2::from_shape_fn((validation_size as usize, 10), |(i, j)| {
        if val_lbl[i] == j as u8 {
            1.0
        } else {
            0.0
        }
    });
    (train_data, trn_lbl, validation_data, val_lbl)
}
