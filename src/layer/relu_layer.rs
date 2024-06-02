use crate::layer::layer::Layer;
use ndarray::{prelude::Array, Dimension};

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
    fn forward(&mut self, x: &Array<f64, Dim>) -> Array<f64, Dim> {
        self.mask = x.map(|&x| if x > 0.0 { 1.0 } else { 0.0 });
        x.map(|&x| if x > 0.0 { x } else { 0.0 })
    }
    fn backward(&mut self, dout: &Array<f64, Dim>) -> Array<f64, Dim> {
        dout * &self.mask
    }
}
