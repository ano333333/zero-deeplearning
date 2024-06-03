mod layer;
use layer::affine_layer::AffineLayer;
use layer::layer::Layer;
use layer::relu_layer::ReluLayer;
use layer::softmax_with_loss_layer::SoftmaxWithLossLayer;
mod mnist;
mod subfunction;

use ndarray::prelude::*;
use ndarray_rand::rand::seq::IteratorRandom;
use ndarray_rand::rand_distr::{Normal, Uniform};
use ndarray_rand::{rand, RandomExt};
use subfunction::argmax::argmax;
use subfunction::cross_entropy_error::cross_entropy_error;
use subfunction::softmax_batch::softmax_batch;
fn separator() -> String {
    (0..20).map(|_| "-").collect::<String>()
}

struct TwoLayerNet {
    w1: Array2<f64>,
    b1: Array1<f64>,
    w2: Array2<f64>,
    b2: Array1<f64>,
}

struct TwoLayerNetGradient {
    dw1: Array2<f64>,
    db1: Array1<f64>,
    dw2: Array2<f64>,
    db2: Array1<f64>,
}

impl TwoLayerNet {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let w1 = Array2::random((input_size, hidden_size), Normal::new(0.0, 1.0e-4).unwrap());
        let b1 = Array1::random(hidden_size, Normal::new(0.0, 1.0e-4).unwrap());
        let w2 = Array2::random(
            (hidden_size, output_size),
            Normal::new(0.0, 1.0e-4).unwrap(),
        );
        let b2 = Array1::random(output_size, Normal::new(0.0, 1.0e-4).unwrap());
        TwoLayerNet { w1, b1, w2, b2 }
    }
    fn create_affine1(&self) -> AffineLayer {
        AffineLayer::new(&self.w1, &self.b1)
    }
    fn create_relu1(&self) -> ReluLayer<Ix2> {
        ReluLayer::new()
    }
    fn create_affine2(&self) -> AffineLayer {
        AffineLayer::new(&self.w2, &self.b2)
    }
    fn predict(&mut self, x: &Array2<f64>) -> Array2<f64> {
        let mut affine1 = self.create_affine1();
        let mut relu1 = self.create_relu1();
        let mut affine2 = self.create_affine2();
        let mut x = affine1.forward(x);
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
    fn gradient(&mut self, x: &Array2<f64>, t: &Array2<f64>) -> TwoLayerNetGradient {
        let mut affine1 = self.create_affine1();
        let mut relu1 = self.create_relu1();
        let mut affine2 = self.create_affine2();

        let x = affine1.forward(x);
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
            dw2: affine2.dw.clone(),
            db2: affine2.db.clone(),
        }
    }
    fn update(&mut self, grad: &TwoLayerNetGradient) {
        self.w1 -= &grad.dw1;
        self.b1 -= &grad.db1;
        self.w2 -= &grad.dw2;
        self.b2 -= &grad.db2;
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
    let epoch_size = training_size / batch_size as u32;
    let learning_rate = 0.1;

    let mut network = TwoLayerNet::new(input_layer_size, hidden_layer_size, output_layer_size);

    let indexes = (0..(training_size as usize)).collect::<Vec<usize>>();
    let mut rng = rand::thread_rng();
    for i in 0..iters_num {
        // println!("count: {}", i);
        let batch_mask = indexes
            .iter()
            .choose_multiple(&mut rng, batch_size as usize)
            .iter()
            .map(|&i| *i)
            .collect::<Vec<usize>>();
        let x_batch = x_train.select(Axis(0), &batch_mask);
        let t_batch = t_train.select(Axis(0), &batch_mask);

        // let loss = network.loss(&x_batch, &t_batch);
        // println!("loss(before): {}", loss);

        let mut grad = network.gradient(&x_batch, &t_batch);
        // println!("avg(dw1): {}", grad.dw1.sum() / grad.dw1.len() as f64);
        // println!("avg(db1): {}", grad.db1.sum() / grad.db1.len() as f64);
        // println!("avg(dw2): {}", grad.dw2.sum() / grad.dw2.len() as f64);
        // println!("avg(db2): {}", grad.db2.sum() / grad.db2.len() as f64);

        grad.dw1 *= learning_rate;
        grad.db1 *= learning_rate;
        grad.dw2 *= learning_rate;
        grad.db2 *= learning_rate;
        network.update(&grad);

        // let loss = network.loss(&x_batch, &t_batch);
        // println!("loss(after): {}", loss);
        if i % epoch_size == 0 {
            let train_acc = network.accuracy(&x_train, &t_train);
            let test_acc = network.accuracy(&x_test, &t_test);
            println!("train acc, test acc | {}, {}", train_acc, test_acc);
        }
        // println!("{}", separator());
    }
    println!("{}", separator());
    let train_acc = network.accuracy(&x_train, &t_train);
    let test_acc = network.accuracy(&x_test, &t_test);
    // x_test[0..5]がどのような値になっているかを確認
    let x = x_test.slice_move(s![0..5, ..]);
    let y = network.predict(&x);
    let y = softmax_batch(y.view());
    let t = t_test.slice_move(s![0..5, ..]);
    for i in 0..5 {
        let y = y.index_axis(Axis(0), i);
        let t = t.index_axis(Axis(0), i);
        println!("x[{}] -> {} (t={})", i, y, t);
    }
    println!("train acc, test acc | {}, {}", train_acc, test_acc);
}
