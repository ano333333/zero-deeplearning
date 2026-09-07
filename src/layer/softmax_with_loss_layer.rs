use crate::layer::layer::Layer;
use crate::subfunction::{cross_entropy_error::cross_entropy_error, softmax_batch::softmax_batch};
use ndarray::prelude::Array2;

/// ソフトマックス関数と交差エントロピー誤差をまとめて計算する層。
///
/// バッチの各行をソフトマックスで確率分布に変換し、正解ラベル `t` (one-hot) との
/// 交差エントロピー誤差(バッチ平均)を出力する。
pub struct SoftmaxWithLossLayer {
    loss: f64,
    y: Array2<f64>,
    t: Array2<f64>,
}

impl SoftmaxWithLossLayer {
    /// 正解ラベル `t` (one-hot) を保持するSoftmaxWithLossLayerを生成する。
    pub fn new(t: &Array2<f64>) -> Self {
        SoftmaxWithLossLayer {
            loss: 0.0,
            y: Array2::zeros((0, 0)),
            t: t.clone(),
        }
    }
}

impl Layer<Array2<f64>, f64> for SoftmaxWithLossLayer {
    /// `y` の各行にソフトマックスを適用したうえで、`t` との交差エントロピー誤差(バッチ平均)を返す。
    ///
    /// `y` の形状が `t` の形状と異なる場合はpanicする。
    fn forward(&mut self, y: &Array2<f64>) -> f64 {
        assert_eq!(
            y.raw_dim(),
            self.t.raw_dim(),
            "y.raw_dim() must equal t.raw_dim()"
        );
        self.y = softmax_batch(y.view());
        self.loss = cross_entropy_error(self.y.view(), self.t.view());
        self.loss
    }
    fn backward(&mut self, _: &f64) -> Array2<f64> {
        let batch_size = self.t.shape()[0] as f64;
        (self.y.clone() - self.t.clone()) / batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const H: f64 = 1e-6;
    const EPSILON: f64 = 1e-6;

    #[test]
    fn forward_returns_non_negative_loss() {
        let t = array![[0.0, 1.0, 0.0]];
        let y = array![[1.0, 2.0, 3.0]];
        let loss = SoftmaxWithLossLayer::new(&t).forward(&y);

        assert!(loss >= 0.0, "expected non-negative loss, got {loss}");
    }

    #[test]
    fn forward_loss_decreases_when_correct_class_input_increases() {
        let t = array![[0.0, 1.0, 0.0]];
        let y_ten = array![[1.0, 10.0, 1.0]];
        let y_hundred = array![[1.0, 100.0, 1.0]];
        let loss_ten = SoftmaxWithLossLayer::new(&t).forward(&y_ten);
        let loss_hundred = SoftmaxWithLossLayer::new(&t).forward(&y_hundred);

        assert!(loss_ten > loss_hundred);
    }

    #[test]
    #[should_panic(expected = "y.raw_dim() must equal t.raw_dim()")]
    fn forward_rejects_input_with_different_shape_from_target() {
        let t = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let y = array![[1.0, 2.0, 3.0]];

        SoftmaxWithLossLayer::new(&t).forward(&y);
    }

    #[test]
    fn backward_matches_central_difference_for_multiple_batches() {
        let t = array![[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let y = array![[0.3, -0.2, 1.1], [-0.7, 0.8, 0.2]];
        let mut layer = SoftmaxWithLossLayer::new(&t);
        layer.forward(&y);
        let gradient = layer.backward(&1.0);

        for batch in 0..y.nrows() {
            for class in 0..y.ncols() {
                let mut y_plus = y.clone();
                let mut y_minus = y.clone();
                y_plus[[batch, class]] += H;
                y_minus[[batch, class]] -= H;

                let loss_plus = SoftmaxWithLossLayer::new(&t).forward(&y_plus);
                let loss_minus = SoftmaxWithLossLayer::new(&t).forward(&y_minus);
                let numerical_gradient = (loss_plus - loss_minus) / (2.0 * H);

                assert!(
                    (gradient[[batch, class]] - numerical_gradient).abs() < EPSILON,
                    "gradient mismatch at [{batch}, {class}]: backward={}, numerical={numerical_gradient}",
                    gradient[[batch, class]]
                );
            }
        }
    }
}
