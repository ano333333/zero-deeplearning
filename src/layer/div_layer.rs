use crate::layer::layer::Layer;
use ndarray::{prelude::Array, Dimension};

const MIN_ABSOLUTE_VALUE: f64 = 1e-12;

/// 入力の各要素の逆数 (`1 / x`) を計算する層。
pub struct DivLayer<Dim: Dimension> {
    x: Array<f64, Dim>,
}

impl<Dim: Dimension> DivLayer<Dim> {
    pub fn new() -> Self {
        DivLayer {
            x: Array::zeros(Dim::default()),
        }
    }
}

impl<Dim: Dimension> Layer<Array<f64, Dim>, Array<f64, Dim>> for DivLayer<Dim> {
    /// `1 / x` を要素ごとに計算して返す。
    ///
    /// `x` の各要素の絶対値が `MIN_ABSOLUTE_VALUE` 未満の場合はpanicする。
    fn forward(&mut self, x: &Array<f64, Dim>) -> Array<f64, Dim> {
        assert!(
            x.iter().all(|value| value.abs() >= MIN_ABSOLUTE_VALUE),
            "all elements of x must have an absolute value of at least 1e-12"
        );
        self.x = x.clone();
        1.0 / x
    }
    /// `dout` の形状が直前の `forward` に渡した `x` の形状と異なる場合はpanicする。
    fn backward(&mut self, dout: &Array<f64, Dim>) -> Array<f64, Dim> {
        assert_eq!(
            dout.shape(),
            self.x.shape(),
            "dout and x must have the same shape"
        );
        -dout / (&self.x * &self.x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const H: f64 = 1e-6;
    const EPSILON: f64 = 1e-6;

    #[test]
    fn forward_returns_reciprocal_of_each_element() {
        let mut layer = DivLayer::new();
        let x = array![[0.5, -1.0, 2.0], [4.0, -5.0, 10.0]];
        let expected = x.mapv(|value| 1.0 / value);

        let actual = layer.forward(&x);

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < EPSILON);
        }
    }

    #[test]
    #[should_panic(expected = "all elements of x must have an absolute value of at least 1e-12")]
    fn forward_rejects_elements_smaller_than_minimum_absolute_value() {
        let mut layer = DivLayer::new();
        let x = array![[1.0, 0.5e-12], [-1.0, -2.0]];

        layer.forward(&x);
    }

    #[test]
    fn forward_accepts_elements_at_minimum_absolute_value() {
        let mut layer = DivLayer::new();
        let x = array![[1e-12, -1e-12]];

        let actual = layer.forward(&x);

        assert_eq!(actual, array![[1e12, -1e12]]);
    }

    #[test]
    fn backward_matches_central_difference() {
        let mut layer = DivLayer::new();
        let x = array![[0.5, -1.0, 2.0], [4.0, -5.0, 10.0]];
        let dout = Array::ones(x.raw_dim());

        layer.forward(&x);
        let backward_gradient = layer.backward(&dout);
        let forward_plus_h = layer.forward(&(&x + H));
        let forward_minus_h = layer.forward(&(&x - H));
        let numerical_gradient = (forward_plus_h - forward_minus_h) / (2.0 * H);

        for (actual, expected) in backward_gradient.iter().zip(numerical_gradient.iter()) {
            assert!((actual - expected).abs() < EPSILON);
        }
    }

    #[test]
    #[should_panic(expected = "dout and x must have the same shape")]
    fn backward_rejects_gradient_with_different_shape_from_input() {
        let mut layer = DivLayer::new();
        let x = array![[0.5, -1.0, 2.0], [4.0, -5.0, 10.0]];
        let dout = array![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
        layer.forward(&x);

        layer.backward(&dout);
    }
}
