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
