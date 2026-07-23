use crate::layer::layer::Layer;
use ndarray::{prelude::Array, Dimension};

pub struct SigmoidLayer<Dim: Dimension> {
    out: Array<f64, Dim>,
}

impl<Dim: Dimension> SigmoidLayer<Dim> {
    pub fn new() -> Self {
        SigmoidLayer {
            out: Array::zeros(Dim::default()),
        }
    }
}

impl<Dim: Dimension> Layer<Array<f64, Dim>, Array<f64, Dim>> for SigmoidLayer<Dim> {
    fn forward(&mut self, x: &Array<f64, Dim>) -> Array<f64, Dim> {
        self.out = x.map(|&x| 1.0 / (1.0 + (-x).exp()));
        self.out.clone()
    }
    fn backward(&mut self, dout: &Array<f64, Dim>) -> Array<f64, Dim> {
        dout * &self.out * (1.0 - &self.out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subfunction::sigmoid::sigmoid;
    use ndarray::array;

    const EPSILON: f64 = 1e-6;

    #[test]
    fn forward_returns_half_for_zero() {
        let mut layer = SigmoidLayer::new();
        let x = array![0.0];

        let out = layer.forward(&x);

        assert!((out[0] - 0.5).abs() < EPSILON);
    }

    #[test]
    fn forward_is_monotonically_increasing() {
        let mut layer = SigmoidLayer::new();
        let x = array![-1.0, -0.5, 0.0, 0.5, 1.0];

        let out = layer.forward(&x);

        for (left, right) in out.iter().zip(out.iter().skip(1)) {
            assert!(left < right);
        }
    }

    #[test]
    fn forward_returns_values_between_zero_and_one() {
        let mut layer = SigmoidLayer::new();
        let x = array![-10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0];

        let out = layer.forward(&x);

        for value in out {
            assert!(0.0 < value);
            assert!(value < 1.0);
        }
    }

    #[test]
    fn backward_matches_central_difference() {
        let mut layer = SigmoidLayer::new();
        let x = array![
            -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8, 0.9, 1.0
        ];
        let dout = Array::ones(x.raw_dim());
        let numerical_gradient =
            (sigmoid((&x + EPSILON).view()) - sigmoid((&x - EPSILON).view())) / (2.0 * EPSILON);

        layer.forward(&x);
        let backward_gradient = layer.backward(&dout);

        for (actual, expected) in backward_gradient.iter().zip(numerical_gradient.iter()) {
            assert!((actual - expected).abs() < EPSILON);
        }
    }
}
