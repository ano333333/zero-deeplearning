use mnist::*;
use ndarray::prelude::*;

/// MNISTデータセットを読み込む
///
/// MNISTデータセットを読み込み、訓練データ、訓練ラベル、検証データ、検証ラベルの4つの配列を返す。
///
/// # Arguments
///
/// * `base_path` - MNISTデータセットを配置したディレクトリ。省略された場合は`data/`。
/// * `training_size` - 訓練データのサイズ。省略された場合は50000。
/// * `validation_size` - 検証データのサイズ。省略された場合は500。
/// * `test_size` - テストデータのサイズ。省略された場合は10000。
///
/// # Returns
///
/// * `(train_data, trn_lbl, validation_data, val_lbl)` - 訓練データ、訓練ラベル、検証データ、検証ラベルの4つの配列。
///   * `train_data` - 訓練データ。形状は(訓練データのサイズ, 28*28)。
///   * `trn_lbl` - one-hot形式の訓練ラベル。形状は(訓練データのサイズ, 10)。
///   * `validation_data` - 検証データ。形状は(検証データのサイズ, 28*28)。
///   * `val_lbl` - one-hot形式の検証ラベル。形状は(検証データのサイズ, 10)。
///   * `test_data` - テストデータ。形状は(テストデータのサイズ, 28*28)。
///   * `test_lbl` - one-hot形式のテストラベル。形状は(テストデータのサイズ, 10)。
///
/// # Examples
/// ```
///     let (train_data, trn_lbl, validation_data, val_lbl, test_data, test_lbl) = mnist::load_mnist::load_mnist(None, None, None, None);
/// ```
pub fn load_mnist(
    base_path: Option<&str>,
    training_size: Option<u32>,
    validation_size: Option<u32>,
    test_size: Option<u32>,
) -> (
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
) {
    let base_path = base_path.unwrap_or("data/");
    let training_size = training_size.unwrap_or(50_000);
    let validation_size = validation_size.unwrap_or(500);
    let test_size = test_size.unwrap_or(10_000);
    let Mnist {
        trn_img,
        trn_lbl,
        val_img,
        val_lbl,
        tst_img,
        tst_lbl,
    } = MnistBuilder::new()
        .label_format_digit()
        .base_path(base_path)
        .training_set_length(training_size)
        .validation_set_length(validation_size)
        .test_set_length(test_size)
        .finalize();
    // trn_img,val_img,tst_imgを28*28要素づつに分け、全ての値を[0,256)から[0.0,1.0)に正規化
    let train_data = Array2::from_shape_vec((training_size as usize, 28 * 28), trn_img).unwrap();
    let train_data = train_data.mapv(|x| x as f64 / 256.0);
    let validation_data =
        Array2::from_shape_vec((validation_size as usize, 28 * 28), val_img).unwrap();
    let validation_data = validation_data.mapv(|x| x as f64 / 256.0);
    let test_data = Array2::from_shape_vec((test_size as usize, 28 * 28), tst_img).unwrap();
    let test_data = test_data.mapv(|x| x as f64 / 256.0);
    // trn_lbl,val_lbl,tst_lblをone-hot表現に変換
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
    let test_lbl = Array2::from_shape_fn((test_size as usize, 10), |(i, j)| {
        if tst_lbl[i] == j as u8 {
            1.0
        } else {
            0.0
        }
    });
    (
        train_data,
        trn_lbl,
        validation_data,
        val_lbl,
        test_data,
        test_lbl,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::{self, Write};
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    const TRAIN_HEADER_LENGTH: u32 = 60_000;
    const TEST_HEADER_LENGTH: u32 = 10_000;

    struct TestWorkspace {
        path: PathBuf,
    }

    impl TestWorkspace {
        fn new() -> Self {
            let id = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "zero-deeplearning-mnist-fixture-{}-{}",
                std::process::id(),
                id
            ));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        fn as_str(&self) -> &str {
            self.path.to_str().unwrap()
        }
    }

    impl Drop for TestWorkspace {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn loads_mnist_arrays_from_fixture_path() {
        let workspace = TestWorkspace::new();
        write_fixture(&workspace.path).unwrap();

        let (train_data, train_labels, validation_data, validation_labels, test_data, test_labels) =
            load_mnist(Some(workspace.as_str()), Some(2), Some(1), Some(2));

        assert_eq!(train_data.dim(), (2, 28 * 28));
        assert_eq!(train_labels.dim(), (2, 10));
        assert_eq!(validation_data.dim(), (1, 28 * 28));
        assert_eq!(validation_labels.dim(), (1, 10));
        assert_eq!(test_data.dim(), (2, 28 * 28));
        assert_eq!(test_labels.dim(), (2, 10));

        assert_eq!(train_data[[0, 0]], 0.0);
        assert_eq!(train_data[[1, 0]], 128.0 / 256.0);
        assert_eq!(validation_data[[0, 0]], 255.0 / 256.0);
        assert_eq!(test_data[[0, 0]], 64.0 / 256.0);
        assert_eq!(test_data[[1, 0]], 32.0 / 256.0);

        assert_eq!(train_labels[[0, 3]], 1.0);
        assert_eq!(train_labels[[1, 8]], 1.0);
        assert_eq!(validation_labels[[0, 1]], 1.0);
        assert_eq!(test_labels[[0, 4]], 1.0);
        assert_eq!(test_labels[[1, 9]], 1.0);
    }

    fn write_fixture(dir: &Path) -> io::Result<()> {
        write_images(
            &dir.join("train-images-idx3-ubyte"),
            TRAIN_HEADER_LENGTH,
            &[0, 128, 255],
        )?;
        write_labels(
            &dir.join("train-labels-idx1-ubyte"),
            TRAIN_HEADER_LENGTH,
            &[3, 8, 1],
        )?;
        write_images(
            &dir.join("t10k-images-idx3-ubyte"),
            TEST_HEADER_LENGTH,
            &[64, 32],
        )?;
        write_labels(
            &dir.join("t10k-labels-idx1-ubyte"),
            TEST_HEADER_LENGTH,
            &[4, 9],
        )?;
        Ok(())
    }

    fn write_labels(path: &Path, header_length: u32, labels: &[u8]) -> io::Result<()> {
        let mut file = File::create(path)?;
        file.write_all(&2049_u32.to_be_bytes())?;
        file.write_all(&header_length.to_be_bytes())?;
        file.write_all(labels)?;
        Ok(())
    }

    fn write_images(path: &Path, header_length: u32, first_pixels: &[u8]) -> io::Result<()> {
        let mut file = File::create(path)?;
        file.write_all(&2051_u32.to_be_bytes())?;
        file.write_all(&header_length.to_be_bytes())?;
        file.write_all(&28_u32.to_be_bytes())?;
        file.write_all(&28_u32.to_be_bytes())?;

        for &first_pixel in first_pixels {
            let mut image = [0_u8; 28 * 28];
            image[0] = first_pixel;
            file.write_all(&image)?;
        }

        Ok(())
    }
}
