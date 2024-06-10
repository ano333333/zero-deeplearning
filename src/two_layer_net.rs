use ndarray::prelude::*;
use ndarray_rand::{rand_distr::Distribution, RandomExt};

use crate::{
    layer::{
        affine_layer::AffineLayer, batch_normalization_layer::BatchNormalizationLayer,
        layer::Layer, relu_layer::ReluLayer, softmax_with_loss_layer::SoftmaxWithLossLayer,
    },
    subfunction::argmax::argmax,
};

#[derive(Clone)]
pub struct TwoLayerNet {
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    pub batch_aff: Array1<f64>,
    pub w2: Array2<f64>,
    pub b2: Array1<f64>,
}

pub struct TwoLayerNetGradient {
    pub dw1: Array2<f64>,
    pub db1: Array1<f64>,
    pub dbatch_aff: Array1<f64>,
    pub dw2: Array2<f64>,
    pub db2: Array1<f64>,
}

impl TwoLayerNet {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        dist: &impl Distribution<f64>,
    ) -> Self {
        let w1 = Array2::random((input_size, hidden_size), &dist);
        let b1 = Array1::random(hidden_size, &dist);
        let w2 = Array2::random((hidden_size, output_size), &dist);
        let b2 = Array1::random(output_size, &dist);
        TwoLayerNet {
            w1,
            b1,
            batch_aff: array![1.0, 0.0],
            w2,
            b2,
        }
    }
    pub fn create_affine1(&self) -> AffineLayer {
        AffineLayer::new(&self.w1, &self.b1)
    }
    pub fn create_batch_normalization1(&self) -> BatchNormalizationLayer {
        BatchNormalizationLayer::new(self.w1.shape()[1], &self.batch_aff)
    }
    pub fn create_relu1(&self) -> ReluLayer<Ix2> {
        ReluLayer::new()
    }
    pub fn create_affine2(&self) -> AffineLayer {
        AffineLayer::new(&self.w2, &self.b2)
    }
    pub fn predict(&mut self, x: &Array2<f64>) -> Array2<f64> {
        let mut affine1 = self.create_affine1();
        let mut batch_normalization1 = self.create_batch_normalization1();
        let mut relu1 = self.create_relu1();
        let mut affine2 = self.create_affine2();
        let mut x = affine1.forward(x);
        x = batch_normalization1.forward(&x);
        x = relu1.forward(&x);
        x = affine2.forward(&x);
        x
    }
    pub fn loss(&mut self, x: &Array2<f64>, t: &Array2<f64>) -> f64 {
        let y = self.predict(x);
        let mut last_layer = SoftmaxWithLossLayer::new(t);
        last_layer.forward(&y)
    }
    pub fn accuracy(&mut self, x: &Array2<f64>, t: &Array2<f64>) -> f64 {
        let y = self.predict(x);
        let mut count = 0;
        for row in 0..y.shape()[0] {
            let y = y.index_axis(Axis(0), row);
            let t = t.index_axis(Axis(0), row);
            let y = argmax(y.view());
            let t = argmax(t.view());
            if y == t {
                count += 1;
            }
        }
        count as f64 / y.shape()[0] as f64
    }
    pub fn gradient(&mut self, x: &Array2<f64>, t: &Array2<f64>) -> TwoLayerNetGradient {
        let mut affine1 = self.create_affine1();
        let mut batch_normalization1 = self.create_batch_normalization1();
        let mut relu1 = self.create_relu1();
        let mut affine2 = self.create_affine2();

        let x = affine1.forward(x);
        let x = batch_normalization1.forward(&x);
        let x = relu1.forward(&x);
        let x = affine2.forward(&x);

        let mut last_layer = SoftmaxWithLossLayer::new(t);
        last_layer.forward(&x);

        let dout = 1.0;
        let dout = last_layer.backward(&dout);
        let dout = affine2.backward(&dout);
        let dout = relu1.backward(&dout);
        affine1.backward(&dout);

        TwoLayerNetGradient {
            dw1: affine1.dw.clone(),
            db1: affine1.db.clone(),
            dbatch_aff: batch_normalization1.daff.clone(),
            dw2: affine2.dw.clone(),
            db2: affine2.db.clone(),
        }
    }
}
