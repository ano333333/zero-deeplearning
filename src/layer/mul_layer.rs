use crate::layer::layer::Layer;
use ndarray::{Array, Dimension};

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
    fn forward(&mut self, (x, y): &(Array<f64, Dim>, Array<f64, Dim>)) -> Array<f64, Dim> {
        self.x = x.clone();
        self.y = y.clone();
        x * y
    }
    fn backward(&mut self, dout: &Array<f64, Dim>) -> (Array<f64, Dim>, Array<f64, Dim>) {
        (dout * &self.y, dout * &self.x)
    }
}
