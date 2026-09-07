use crate::layer::layer::Layer;
use ndarray::{prelude::Array, Dimension};

pub struct ExpLayer<Dim: Dimension> {
    out: Array<f64, Dim>,
}

impl<Dim: Dimension> ExpLayer<Dim> {
    pub fn new() -> Self {
        ExpLayer {
            out: Array::zeros(Dim::default()),
        }
    }
}

impl<Dim: Dimension> Layer<Array<f64, Dim>, Array<f64, Dim>> for ExpLayer<Dim> {
    fn forward(&mut self, x: &Array<f64, Dim>) -> Array<f64, Dim> {
        self.out = x.map(|&x| x.exp());
        self.out.clone()
    }
    fn backward(&mut self, dout: &Array<f64, Dim>) -> Array<f64, Dim> {
        dout * &self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const H: f64 = 1e-6;
    const EPSILON: f64 = 1e-6;

    #[test]
    fn forward_returns_exp_of_each_element() {
        let mut layer = ExpLayer::new();
        let x: Array<f64, _> = array![[0.5, -1.0, 2.0], [4.0, -5.0, 10.0]];
        let expected = x.mapv(|value| value.exp());

        let actual = layer.forward(&x);

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < EPSILON);
        }
    }

    #[test]
    fn backward_matches_central_difference() {
        let mut layer = ExpLayer::new();
        let x = array![[0.5, -1.0, 2.0], [-2.0, -0.5, 1.0]];
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
}
