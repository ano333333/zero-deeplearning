use crate::layer::layer::Layer;
use crate::subfunction::{cross_entropy_error::cross_entropy_error, softmax_batch::softmax_batch};
use ndarray::prelude::Array2;

pub struct SoftmaxWithLossLayer {
    loss: f64,
    y: Array2<f64>,
    t: Array2<f64>,
}

impl SoftmaxWithLossLayer {
    pub fn new(t: &Array2<f64>) -> Self {
        SoftmaxWithLossLayer {
            loss: 0.0,
            y: Array2::zeros((0, 0)),
            t: t.clone(),
        }
    }
}

impl Layer<Array2<f64>, f64> for SoftmaxWithLossLayer {
    fn forward(&mut self, y: &Array2<f64>) -> f64 {
        self.y = softmax_batch(y.view());
        self.loss = cross_entropy_error(self.y.view(), self.t.view());
        self.loss
    }
    fn backward(&mut self, _: &f64) -> Array2<f64> {
        let batch_size = self.t.shape()[0] as f64;
        (self.y.clone() - self.t.clone()) / batch_size
    }
}
