use crate::layer::layer::Layer;
use ndarray::{Array, Dimension};

/// 2つの入力を要素ごとに乗算する層。
pub struct MulLayer<Dim: Dimension> {
    x: Array<f64, Dim>,
    y: Array<f64, Dim>,
}

impl<Dim: Dimension> MulLayer<Dim> {
    pub fn new() -> Self {
        MulLayer {
            x: Array::zeros(Dim::default()),
            y: Array::zeros(Dim::default()),
        }
    }
}

impl<Dim: Dimension> Layer<(Array<f64, Dim>, Array<f64, Dim>), Array<f64, Dim>> for MulLayer<Dim> {
    /// `x * y` を要素ごとに計算して返す。
    fn forward(&mut self, (x, y): &(Array<f64, Dim>, Array<f64, Dim>)) -> Array<f64, Dim> {
        self.x = x.clone();
        self.y = y.clone();
        x * y
    }
    fn backward(&mut self, dout: &Array<f64, Dim>) -> (Array<f64, Dim>, Array<f64, Dim>) {
        (dout * &self.y, dout * &self.x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn forward_multiplies_fixed_three_by_three_matrices_elementwise() {
        let mut layer = MulLayer::new();
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let y = array![[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]];

        let out = layer.forward(&(x, y));

        assert_eq!(
            out,
            array![[9.0, 16.0, 21.0], [24.0, 25.0, 24.0], [21.0, 16.0, 9.0]]
        );
    }

    #[test]
    fn backward_uses_saved_three_by_three_inputs_for_gradients() {
        let mut layer = MulLayer::new();
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let y = array![[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]];
        let dout = array![[1.0, 0.5, 2.0], [1.5, 1.0, 0.25], [0.0, 3.0, 4.0]];

        layer.forward(&(x, y));
        let (dx, dy) = layer.backward(&dout);

        assert_eq!(
            dx,
            array![[9.0, 4.0, 14.0], [9.0, 5.0, 1.0], [0.0, 6.0, 4.0]]
        );
        assert_eq!(
            dy,
            array![[1.0, 1.0, 6.0], [6.0, 5.0, 1.5], [0.0, 24.0, 36.0]]
        );
    }
}
