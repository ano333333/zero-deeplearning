use crate::layer::layer::Layer;
use ndarray::{prelude::Array, Dimension};

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
    fn forward(&mut self, x: &Array<f64, Dim>) -> Array<f64, Dim> {
        self.x = x.clone();
        1.0 / x
    }
    fn backward(&mut self, dout: &Array<f64, Dim>) -> Array<f64, Dim> {
        -dout / &self.x / &self.x
    }
}
