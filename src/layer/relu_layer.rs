use crate::layer::layer::Layer;
use ndarray::{prelude::Array, Dimension};

/// 入力の各要素にReLU (`max(0, x)`) を適用する層。
pub struct ReluLayer<Dim: Dimension> {
    mask: Array<f64, Dim>,
}

impl<Dim: Dimension> ReluLayer<Dim> {
    pub fn new() -> Self {
        ReluLayer {
            mask: Array::zeros(Dim::default()),
        }
    }
}

impl<Dim: Dimension> Layer<Array<f64, Dim>, Array<f64, Dim>> for ReluLayer<Dim> {
    /// `x` の各要素が正なら `x`、そうでなければ `0` を返す。
    fn forward(&mut self, x: &Array<f64, Dim>) -> Array<f64, Dim> {
        self.mask = x.map(|&x| if x > 0.0 { 1.0 } else { 0.0 });
        x.map(|&x| if x > 0.0 { x } else { 0.0 })
    }
    fn backward(&mut self, dout: &Array<f64, Dim>) -> Array<f64, Dim> {
        dout * &self.mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subfunction::relu::relu;
    use ndarray::array;

    const EPSILON: f64 = 1e-6;

    #[test]
    fn forward_keeps_positive_inputs() {
        let mut layer = ReluLayer::new();
        let x = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

        let out = layer.forward(&x);

        assert_eq!(out, x);
    }

    #[test]
    fn forward_zeroes_negative_inputs() {
        let mut layer = ReluLayer::new();
        let x = array![-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0];

        let out = layer.forward(&x);

        assert_eq!(out, Array::zeros(x.raw_dim()));
    }

    #[test]
    fn backward_matches_central_difference_for_positive_inputs() {
        let mut layer = ReluLayer::new();
        let x = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let dout = Array::ones(x.raw_dim());
        let numerical_gradient =
            (relu((&x + EPSILON).view()) - relu((&x - EPSILON).view())) / (2.0 * EPSILON);

        layer.forward(&x);
        let backward_gradient = layer.backward(&dout);

        for (actual, expected) in backward_gradient.iter().zip(numerical_gradient.iter()) {
            assert!((actual - expected).abs() < EPSILON);
        }
    }

    #[test]
    fn backward_matches_central_difference_for_negative_inputs() {
        let mut layer = ReluLayer::new();
        let x = array![-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0];
        let dout = Array::ones(x.raw_dim());
        let numerical_gradient =
            (relu((&x + EPSILON).view()) - relu((&x - EPSILON).view())) / (2.0 * EPSILON);

        layer.forward(&x);
        let backward_gradient = layer.backward(&dout);

        for (actual, expected) in backward_gradient.iter().zip(numerical_gradient.iter()) {
            assert!((actual - expected).abs() < EPSILON);
        }
    }
}
