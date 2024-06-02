use crate::layer::layer::Layer;
use ndarray::{prelude::Array, Dimension};

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
    fn forward(&mut self, (x, y): &(Array<f64, Dim>, Array<f64, Dim>)) -> Array<f64, Dim> {
        x + y
    }
    fn backward(&mut self, dout: &Array<f64, Dim>) -> (Array<f64, Dim>, Array<f64, Dim>) {
        (dout.clone(), dout.clone())
    }
}
