mod layer;
use layer::affine_layer::AffineLayer;
use layer::layer::Layer;
use layer::relu_layer::ReluLayer;
use layer::softmax_with_loss_layer::SoftmaxWithLossLayer;
mod mnist;
mod optimize;
mod subfunction;

use ndarray::prelude::*;
use ndarray_rand::rand::seq::IteratorRandom;
use ndarray_rand::rand_distr::{Distribution, Normal};
use ndarray_rand::{rand, RandomExt};
use optimize::optimize::Optimize;
use subfunction::argmax::argmax;

use crate::layer::batch_normalization_layer::BatchNormalizationLayer;
use crate::optimize::sgd::SGD;
fn separator() -> String {
    (0..20).map(|_| "-").collect::<String>()
}

#[derive(Clone)]
struct TwoLayerNetWithBatchNormalizationLayer {
    w1: Array2<f64>,
    b1: Array1<f64>,
    batch_aff: Array1<f64>,
    w2: Array2<f64>,
    b2: Array1<f64>,
}

struct TwoLayerNetWithBatchNormalizationLayerGradient {
    dw1: Array2<f64>,
    db1: Array1<f64>,
    dbatch_aff: Array1<f64>,
    dw2: Array2<f64>,
    db2: Array1<f64>,
}

impl TwoLayerNetWithBatchNormalizationLayer {
    fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        dist: &impl Distribution<f64>,
    ) -> Self {
        let w1 = Array2::random((input_size, hidden_size), &dist);
        let b1 = Array1::random(hidden_size, &dist);
        let w2 = Array2::random((hidden_size, output_size), &dist);
        let b2 = Array1::random(output_size, &dist);
        TwoLayerNetWithBatchNormalizationLayer {
            w1,
            b1,
            batch_aff: array![1.0, 0.0],
            w2,
            b2,
        }
    }
    fn create_affine1(&self) -> AffineLayer {
        AffineLayer::new(&self.w1, &self.b1)
    }
    fn create_batch_normalization1(&self) -> BatchNormalizationLayer {
        BatchNormalizationLayer::new(self.w1.shape()[1], &self.batch_aff)
    }
    fn create_relu1(&self) -> ReluLayer<Ix2> {
        ReluLayer::new()
    }
    fn create_affine2(&self) -> AffineLayer {
        AffineLayer::new(&self.w2, &self.b2)
    }
    fn predict(&mut self, x: &Array2<f64>) -> Array2<f64> {
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
    fn loss(&mut self, x: &Array2<f64>, t: &Array2<f64>) -> f64 {
        let y = self.predict(x);
        let mut last_layer = SoftmaxWithLossLayer::new(t);
        last_layer.forward(&y)
    }
    fn accuracy(&mut self, x: &Array2<f64>, t: &Array2<f64>) -> f64 {
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
    fn gradient(
        &mut self,
        x: &Array2<f64>,
        t: &Array2<f64>,
    ) -> TwoLayerNetWithBatchNormalizationLayerGradient {
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

        TwoLayerNetWithBatchNormalizationLayerGradient {
            dw1: affine1.dw.clone(),
            db1: affine1.db.clone(),
            dbatch_aff: batch_normalization1.daff.clone(),
            dw2: affine2.dw.clone(),
            db2: affine2.db.clone(),
        }
    }
}

fn main() {
    let training_size = 50_000;
    let validation_size = 10_000;
    let input_layer_size = 28 * 28;
    let hidden_layer_size = 50;
    let output_layer_size = 10;
    let (x_train, t_train, x_test, t_test) =
        mnist::load_mnist::load_mnist(Some(training_size), Some(validation_size));

    let batch_size = 100;
    let iters_num = 10_000;
    // let iters_num = 500;
    let epoch_size = 100;
    let learning_rate = 0.1;

    let network = TwoLayerNetWithBatchNormalizationLayer::new(
        input_layer_size,
        hidden_layer_size,
        output_layer_size,
        // &Normal::new(0.0, 1.0 / (input_layer_size as f64)).unwrap(),
        &Normal::new(0.0, 1.0).unwrap(),
    );
    let mut network_sgd = network.clone();
    let mut sgd_w1 = SGD::<Ix2>::new(learning_rate);
    let mut sgd_b1 = SGD::<Ix1>::new(learning_rate);
    let mut sgd_w2 = SGD::<Ix2>::new(learning_rate);
    let mut sgd_b2 = SGD::<Ix1>::new(learning_rate);
    let mut sgd_batch_aff = SGD::<Ix1>::new(learning_rate);

    let all_indexes = (0..(training_size as usize)).collect::<Vec<usize>>();
    let mut rng = rand::thread_rng();
    // 各イテレートで用いる学習データのインデックスを固定化する
    let indexes = (0..iters_num)
        .map(|_| {
            all_indexes
                .iter()
                .choose_multiple(&mut rng, batch_size as usize)
                .iter()
                .map(|&i| *i)
                .collect::<Vec<usize>>()
        })
        .collect::<Vec<Vec<usize>>>();
    for i in 0..iters_num {
        // println!("count: {}", i);
        let batch_mask = &indexes[i];
        let x_batch = x_train.select(Axis(0), &batch_mask);
        let t_batch = t_train.select(Axis(0), &batch_mask);

        let grad = network_sgd.gradient(&x_batch, &t_batch);
        sgd_w1.update(&mut network_sgd.w1, &grad.dw1);
        sgd_b1.update(&mut network_sgd.b1, &grad.db1);
        sgd_batch_aff.update(&mut network_sgd.batch_aff, &grad.dbatch_aff);
        sgd_w2.update(&mut network_sgd.w2, &grad.dw2);
        sgd_b2.update(&mut network_sgd.b2, &grad.db2);

        if i % epoch_size == 0 {
            // 各ネットワークのlossとテストデータaccを、それぞれ少数第四位まで表示
            let train_loss_sgd = network_sgd.loss(&x_train, &t_train);
            let test_acc_sgd = network_sgd.accuracy(&x_test, &t_test);
            println!("epoch: {}", i / epoch_size);
            println!(
                "sgd      | train loss: {:?}, test acc: {:?}",
                train_loss_sgd, test_acc_sgd
            );
            println!("{}", separator());
        }
    }
    let train_loss_sgd = network_sgd.loss(&x_train, &t_train);
    let test_acc_sgd = network_sgd.accuracy(&x_test, &t_test);
    println!("epoch final");
    println!(
        "sgd      | train loss: {:?}, test acc: {:?}",
        train_loss_sgd, test_acc_sgd
    );
}
