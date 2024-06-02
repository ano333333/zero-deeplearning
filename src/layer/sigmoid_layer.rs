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
