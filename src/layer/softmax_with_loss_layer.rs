use crate::layer::layer::Layer;
use crate::subfunction::{cross_entropy_error::cross_entropy_error, softmax_batch::softmax_batch};
use ndarray::prelude::Array2;

pub struct SoftmaxWithLossLayer<'a> {
    loss: f64,
    y: Array2<f64>,
    t: &'a Array2<f64>,
}

impl<'a> SoftmaxWithLossLayer<'a> {
    pub fn new(t: &'a Array2<f64>) -> Self {
        SoftmaxWithLossLayer {
            loss: 0.0,
            y: Array2::zeros((0, 0)),
            t,
        }
    }
}

impl<'a> Layer<Array2<f64>, f64> for SoftmaxWithLossLayer<'a> {
    fn forward(&mut self, y: &Array2<f64>) -> f64 {
        self.y = softmax_batch(y.view());
        self.loss = cross_entropy_error(self.y.view(), self.t.view());
        self.loss
    }
    fn backward(&mut self, _: &f64) -> Array2<f64> {
        let batch_size = self.t.shape()[0] as f64;
        (self.y - self.t) / batch_size
    }
}
