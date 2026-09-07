use crate::layer::layer::Layer;
use ndarray::{prelude::Array, Dimension};

/// 2つの入力を要素ごとに加算する層。
pub struct AddLayer<Dim: Dimension> {
    _dim: Dim,
}

impl<Dim: Dimension> AddLayer<Dim> {
    pub fn new() -> Self {
        AddLayer {
            _dim: Dim::default(),
        }
    }
}

impl<Dim: Dimension> Layer<(Array<f64, Dim>, Array<f64, Dim>), Array<f64, Dim>> for AddLayer<Dim> {
    /// `x + y` を要素ごとに計算して返す。
    fn forward(&mut self, (x, y): &(Array<f64, Dim>, Array<f64, Dim>)) -> Array<f64, Dim> {
        x + y
    }
    fn backward(&mut self, dout: &Array<f64, Dim>) -> (Array<f64, Dim>, Array<f64, Dim>) {
        (dout.clone(), dout.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn forward_adds_fixed_three_by_three_matrices_elementwise() {
        let mut layer = AddLayer::new();
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let y = array![[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]];

        let out = layer.forward(&(x, y));

        assert_eq!(
            out,
            array![[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
        );
    }

    #[test]
    fn backward_returns_upstream_gradient_for_both_inputs() {
        let mut layer = AddLayer::new();
        let dout = array![[1.0, 0.5, 2.0], [1.5, 1.0, 0.25], [0.0, 3.0, 4.0]];

        let (dx, dy) = layer.backward(&dout);

        assert_eq!(dx, dout);
        assert_eq!(dy, dout);
    }
}
