use std::{error::Error, fmt, fs::File, io, path::Path};

use ndarray::prelude::*;
use ndarray_npy::{NpzReader, NpzWriter, ReadNpzError, WriteNpzError};
use ndarray_rand::{rand_distr::Distribution, RandomExt};

use crate::{
    layer::{
        affine_layer::AffineLayer, batch_normalization_layer::BatchNormalizationLayer,
        layer::Layer, relu_layer::ReluLayer, softmax_with_loss_layer::SoftmaxWithLossLayer,
    },
    subfunction::argmax::argmax,
};

#[derive(Clone)]
pub struct TwoLayerNet {
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    pub batch_aff: Array1<f64>,
    pub w2: Array2<f64>,
    pub b2: Array1<f64>,
}

pub struct TwoLayerNetGradient {
    pub dw1: Array2<f64>,
    pub db1: Array1<f64>,
    pub dbatch_aff: Array1<f64>,
    pub dw2: Array2<f64>,
    pub db2: Array1<f64>,
}

const MODEL_FORMAT_VERSION: u32 = 1;

/// `TwoLayerNet` のNPZファイルの読み書きで発生するエラー。
#[derive(Debug)]
pub enum TwoLayerNetIoError {
    Io(io::Error),
    ReadNpz(ReadNpzError),
    WriteNpz(WriteNpzError),
    UnsupportedFormatVersion(u32),
    InvalidParameters(String),
}

impl fmt::Display for TwoLayerNetIoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "failed to access model file: {error}"),
            Self::ReadNpz(error) => write!(f, "failed to read NPZ model: {error}"),
            Self::WriteNpz(error) => write!(f, "failed to write NPZ model: {error}"),
            Self::UnsupportedFormatVersion(version) => write!(
                f,
                "unsupported model format version {version} (supported: {MODEL_FORMAT_VERSION})"
            ),
            Self::InvalidParameters(message) => {
                write!(f, "invalid TwoLayerNet parameters: {message}")
            }
        }
    }
}

impl Error for TwoLayerNetIoError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::ReadNpz(error) => Some(error),
            Self::WriteNpz(error) => Some(error),
            Self::UnsupportedFormatVersion(_) | Self::InvalidParameters(_) => None,
        }
    }
}

impl From<io::Error> for TwoLayerNetIoError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<ReadNpzError> for TwoLayerNetIoError {
    fn from(error: ReadNpzError) -> Self {
        Self::ReadNpz(error)
    }
}

impl From<WriteNpzError> for TwoLayerNetIoError {
    fn from(error: WriteNpzError) -> Self {
        Self::WriteNpz(error)
    }
}

impl TwoLayerNet {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        dist: &impl Distribution<f64>,
    ) -> Self {
        let w1 = Array2::random((input_size, hidden_size), &dist);
        let b1 = Array1::random(hidden_size, &dist);
        let w2 = Array2::random((hidden_size, output_size), &dist);
        let b2 = Array1::random(output_size, &dist);
        TwoLayerNet {
            w1,
            b1,
            batch_aff: array![1.0, 0.0],
            w2,
            b2,
        }
    }

    /// ネットワークのパラメータを名前付き配列としてNPZファイルへ保存する。
    pub fn save_npz(&self, path: impl AsRef<Path>) -> Result<(), TwoLayerNetIoError> {
        self.validate_parameters()?;

        let file = File::create(path)?;
        let mut npz = NpzWriter::new(file);
        npz.add_array("format_version", &arr0(MODEL_FORMAT_VERSION))?;
        npz.add_array("w1", &self.w1)?;
        npz.add_array("b1", &self.b1)?;
        npz.add_array("batch_aff", &self.batch_aff)?;
        npz.add_array("w2", &self.w2)?;
        npz.add_array("b2", &self.b2)?;
        npz.finish()?;
        Ok(())
    }

    /// NPZファイルからパラメータを読み込み、ネットワークを復元する。
    pub fn load_npz(path: impl AsRef<Path>) -> Result<Self, TwoLayerNetIoError> {
        let file = File::open(path)?;
        let mut npz = NpzReader::new(file)?;

        let format_version: Array0<u32> = npz.by_name("format_version")?;
        if format_version[()] != MODEL_FORMAT_VERSION {
            return Err(TwoLayerNetIoError::UnsupportedFormatVersion(
                format_version[()],
            ));
        }

        let network = Self {
            w1: npz.by_name("w1")?,
            b1: npz.by_name("b1")?,
            batch_aff: npz.by_name("batch_aff")?,
            w2: npz.by_name("w2")?,
            b2: npz.by_name("b2")?,
        };
        network.validate_parameters()?;
        Ok(network)
    }

    fn validate_parameters(&self) -> Result<(), TwoLayerNetIoError> {
        if self.batch_aff.len() != 2 {
            return Err(TwoLayerNetIoError::InvalidParameters(format!(
                "batch_aff must have length 2, got {}",
                self.batch_aff.len()
            )));
        }
        if self.w1.ncols() != self.b1.len() {
            return Err(TwoLayerNetIoError::InvalidParameters(format!(
                "w1 has {} columns but b1 has length {}",
                self.w1.ncols(),
                self.b1.len()
            )));
        }
        if self.w1.ncols() != self.w2.nrows() {
            return Err(TwoLayerNetIoError::InvalidParameters(format!(
                "w1 has {} columns but w2 has {} rows",
                self.w1.ncols(),
                self.w2.nrows()
            )));
        }
        if self.w2.ncols() != self.b2.len() {
            return Err(TwoLayerNetIoError::InvalidParameters(format!(
                "w2 has {} columns but b2 has length {}",
                self.w2.ncols(),
                self.b2.len()
            )));
        }
        Ok(())
    }

    pub fn create_affine1(&self) -> AffineLayer<'_> {
        AffineLayer::new(&self.w1, &self.b1)
    }
    pub fn create_batch_normalization1(&self) -> BatchNormalizationLayer<'_> {
        BatchNormalizationLayer::new(self.w1.shape()[1], &self.batch_aff)
    }
    pub fn create_relu1(&self) -> ReluLayer<Ix2> {
        ReluLayer::new()
    }
    pub fn create_affine2(&self) -> AffineLayer<'_> {
        AffineLayer::new(&self.w2, &self.b2)
    }
    pub fn predict(&mut self, x: &Array2<f64>) -> Array2<f64> {
        let mut affine1 = self.create_affine1();
        let mut batch_normalization1 = self.create_batch_normalization1();
        let mut relu1 = self.create_relu1();
        let mut affine2 = self.create_affine2();
        let mut x = affine1.forward(x);
        x = batch_normalization1.forward(&x);
        x = relu1.forward(&x);
        x = affine2.forward(&x);
        x
    }
    pub fn loss(&mut self, x: &Array2<f64>, t: &Array2<f64>) -> f64 {
        let y = self.predict(x);
        let mut last_layer = SoftmaxWithLossLayer::new(t);
        last_layer.forward(&y)
    }
    pub fn accuracy(&mut self, x: &Array2<f64>, t: &Array2<f64>) -> f64 {
        let y = self.predict(x);
        let mut count = 0;
        for row in 0..y.shape()[0] {
            let y = y.index_axis(Axis(0), row);
            let t = t.index_axis(Axis(0), row);
            let y = argmax(y.view());
            let t = argmax(t.view());
            if y == t {
                count += 1;
            }
        }
        count as f64 / y.shape()[0] as f64
    }
    pub fn gradient(&mut self, x: &Array2<f64>, t: &Array2<f64>) -> TwoLayerNetGradient {
        let mut affine1 = self.create_affine1();
        let mut batch_normalization1 = self.create_batch_normalization1();
        let mut relu1 = self.create_relu1();
        let mut affine2 = self.create_affine2();

        let x = affine1.forward(x);
        let x = batch_normalization1.forward(&x);
        let x = relu1.forward(&x);
        let x = affine2.forward(&x);

        let mut last_layer = SoftmaxWithLossLayer::new(t);
        last_layer.forward(&x);

        let dout = 1.0;
        let dout = last_layer.backward(&dout);
        let dout = affine2.backward(&dout);
        let dout = relu1.backward(&dout);
        affine1.backward(&dout);

        TwoLayerNetGradient {
            dw1: affine1.dw.clone(),
            db1: affine1.db.clone(),
            dbatch_aff: batch_normalization1.daff.clone(),
            dw2: affine2.dw.clone(),
            db2: affine2.db.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use ndarray::array;

    use super::*;

    static NEXT_TEMP_FILE_ID: AtomicUsize = AtomicUsize::new(0);

    struct TempNpz(PathBuf);

    impl TempNpz {
        fn new() -> Self {
            let id = NEXT_TEMP_FILE_ID.fetch_add(1, Ordering::Relaxed);
            Self(
                std::env::temp_dir()
                    .join(format!("zero-deeplearning-{}-{id}.npz", std::process::id())),
            )
        }
    }

    impl Drop for TempNpz {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn network_with_known_parameters() -> TwoLayerNet {
        TwoLayerNet {
            w1: array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            b1: array![7.0, 8.0],
            batch_aff: array![1.5, -0.5],
            w2: array![[9.0, 10.0], [11.0, 12.0]],
            b2: array![13.0, 14.0],
        }
    }

    #[test]
    fn save_and_load_npz_round_trip() {
        let path = TempNpz::new();
        let expected = network_with_known_parameters();

        expected.save_npz(&path.0).unwrap();
        let actual = TwoLayerNet::load_npz(&path.0).unwrap();

        assert_eq!(actual.w1, expected.w1);
        assert_eq!(actual.b1, expected.b1);
        assert_eq!(actual.batch_aff, expected.batch_aff);
        assert_eq!(actual.w2, expected.w2);
        assert_eq!(actual.b2, expected.b2);
    }

    #[test]
    fn save_npz_rejects_inconsistent_parameter_shapes() {
        let path = TempNpz::new();
        let mut network = network_with_known_parameters();
        network.b1 = array![1.0];

        let error = network.save_npz(&path.0).unwrap_err();

        assert!(matches!(error, TwoLayerNetIoError::InvalidParameters(_)));
        assert!(!path.0.exists());
    }

    #[test]
    fn load_npz_rejects_unsupported_format_version() {
        let path = TempNpz::new();
        let file = File::create(&path.0).unwrap();
        let mut npz = NpzWriter::new(file);
        npz.add_array("format_version", &arr0(MODEL_FORMAT_VERSION + 1))
            .unwrap();
        npz.finish().unwrap();

        let error = match TwoLayerNet::load_npz(&path.0) {
            Ok(_) => panic!("loading an unsupported format version should fail"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            TwoLayerNetIoError::UnsupportedFormatVersion(version)
                if version == MODEL_FORMAT_VERSION + 1
        ));
    }
}
