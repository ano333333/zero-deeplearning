mod layer;
use layer::add_layer::AddLayer;
use layer::div_layer::DivLayer;
use layer::exp_layer::ExpLayer;
use layer::relu_layer::ReluLayer;
use layer::sigmoid_layer::SigmoidLayer;
use layer::{layer::Layer, mul_layer::MulLayer};
mod mnist;

use ndarray::{prelude::*, NdIndex};
use ndarray_rand::{
    rand::{self, seq::IteratorRandom},
    rand_distr::{num_traits::SaturatingMul, Uniform},
    RandomExt,
};
use plotters::style::WHITE;

fn separator() -> String {
    (0..20).map(|_| "-").collect::<String>()
}

fn step_function<D: Dimension>(x: ArrayView<f64, D>) -> Array<f64, D> {
    x.mapv(|x| (x > 0.0) as i32 as f64)
}

fn sigmoid<D: Dimension>(x: ArrayView<f64, D>) -> Array<f64, D> {
    x.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}

fn relu<D: Dimension>(x: ArrayView<f64, D>) -> Array<f64, D> {
    x.mapv(|x| x.max(0.0))
}

fn identity_function<D: Dimension>(x: ArrayView<f64, D>) -> Array<f64, D> {
    x.to_owned()
}

fn softmax<D: Dimension>(x: ArrayView<f64, D>) -> Array<f64, D> {
    let max = x.fold(-1.0 / 0.0, |acc, &x| x.max(acc));
    let c = x.mapv(|x| (x - max).exp()).sum();
    x.mapv(|x| (x - max).exp() / c)
}

// 行ごとにsoftmaxを計算する
fn softmax_2(x: ArrayView2<f64>) -> Array2<f64> {
    let max = x.map_axis(Axis(1), |row| row.fold(-1.0 / 0.0, |acc, &x| x.max(acc)));
    let max = Array2::<f64>::from_shape_fn(x.raw_dim(), |(i, _)| max[i]);
    let x = x.to_owned() - &max;
    let x = x.mapv(|x| x.exp());
    let sum = x
        .axis_iter(Axis(0))
        .map(|row| row.sum())
        .collect::<Array1<f64>>();
    let sum = Array2::<f64>::from_shape_fn(x.raw_dim(), |(i, _)| sum[i]);
    x / &sum
}

struct Network {
    w1: Array2<f64>,
    b1: Array1<f64>,
    act1: fn(ArrayView2<f64>) -> Array2<f64>,
    w2: Array2<f64>,
    b2: Array1<f64>,
    act2: fn(ArrayView2<f64>) -> Array2<f64>,
    w3: Array2<f64>,
    b3: Array1<f64>,
    act3: fn(ArrayView2<f64>) -> Array2<f64>,
}

fn init_network() -> Network {
    Network {
        w1: array![[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]],
        b1: array![0.1, 0.2, 0.3],
        act1: sigmoid,
        w2: array![[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]],
        b2: array![0.1, 0.2],
        act2: sigmoid,
        w3: array![[0.1, 0.3], [0.2, 0.4]],
        b3: array![0.1, 0.2],
        act3: identity_function,
    }
}

fn forward(network: &Network, x: ArrayView2<f64>) -> Array2<f64> {
    let a1 = x.dot(&network.w1) + &network.b1;
    let z1 = (network.act1)(a1.view());
    let a2 = z1.dot(&network.w2) + &network.b2;
    let z2 = (network.act2)(a2.view());
    let a3 = z2.dot(&network.w3) + &network.b3;
    (network.act3)(a3.view())
}

fn cross_entropy_error(y: ArrayView2<f64>, t: ArrayView2<f64>) -> f64 {
    let delta = 1e-7;
    let batch_size = y.shape()[0] as f64;
    let log_y = y.mapv(|y| (y + delta).ln());
    -(log_y * &t).sum() / batch_size
}

fn numerical_gradient<D: Dimension>(
    f: &dyn Fn(ArrayView<f64, D>) -> f64,
    x: ArrayView<f64, D>,
) -> Array<f64, D>
where
    <D as ndarray::Dimension>::Pattern: NdIndex<D>,
{
    // let h = 1e-4;
    let h = 1.0;
    let mut grad = Array::<f64, D>::zeros(x.raw_dim());
    let mut x_mut = x.to_owned();
    for iter in x.indexed_iter() {
        println!("{:?}", iter);
        x_mut[iter.0.clone()] = iter.1 + h;
        let fxh1 = f(x_mut.view());
        x_mut[iter.0.clone()] = iter.1 - h;
        let fxh2 = f(x_mut.view());
        grad[iter.0.clone()] = (fxh1 - fxh2) / (2.0 * h);
        println!("fxh1: {}, fxh2: {}", fxh1, fxh2);
        println!("grad[iter.0.clone()]: {}", grad[iter.0.clone()]);
        x_mut[iter.0.clone()] = *iter.1;
    }
    grad
}

struct SimpleNet {
    w: Array2<f64>,
}

impl SimpleNet {
    fn new() -> Self {
        SimpleNet {
            w: Array::random((2, 3), Uniform::new(0.0, 1.0)),
        }
    }
    fn predict(&self, x: ArrayView2<f64>) -> Array2<f64> {
        x.dot(&self.w)
    }
    fn loss(&self, x: ArrayView2<f64>, t: ArrayView2<f64>) -> f64 {
        let z = self.predict(x);
        let y = softmax(z.view());
        cross_entropy_error(y.view(), t)
    }
}

fn argmax<D: Dimension>(x: ArrayView<f64, D>) -> <D as Dimension>::Pattern
where
    <D as ndarray::Dimension>::Pattern: NdIndex<D>,
{
    let mut max_index: Option<<D as Dimension>::Pattern> = None;
    for iter in x.indexed_iter() {
        match max_index {
            None => max_index = Some(iter.0),
            Some(ref index) => {
                if *iter.1 > x[index.clone()] {
                    max_index = Some(iter.0);
                }
            }
        }
    }
    max_index.unwrap()
}

#[derive(Clone)]
struct TwoLayerNet {
    w1: Array2<f64>,
    b1: Array1<f64>,
    z1: fn(ArrayView2<f64>) -> Array2<f64>,
    w2: Array2<f64>,
    b2: Array1<f64>,
    y: fn(ArrayView2<f64>) -> Array2<f64>,
}

struct TwoLayerNetGradient {
    dw1: Array2<f64>,
    db1: Array1<f64>,
    dw2: Array2<f64>,
    db2: Array1<f64>,
}

impl TwoLayerNet {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        TwoLayerNet {
            w1: Array::random((input_size, hidden_size), Uniform::new(-10.0, 10.0)),
            b1: Array::random(hidden_size, Uniform::new(-10.0, 10.0)),
            z1: sigmoid,
            w2: Array::random((hidden_size, output_size), Uniform::new(-10.0, 10.0)),
            b2: Array::random(output_size, Uniform::new(-10.0, 10.0)),
            y: softmax_2,
        }
    }
    fn predict(&self, x: ArrayView2<f64>) -> Array2<f64> {
        let a1 = x.dot(&self.w1) + &self.b1;
        let z1 = (self.z1)(a1.view());
        let a2 = z1.dot(&self.w2) + &self.b2;
        (self.y)(a2.view())
    }
    fn loss(&self, x: ArrayView2<f64>, t: ArrayView2<f64>) -> f64 {
        let y = self.predict(x);
        cross_entropy_error(y.view(), t)
    }
    fn accuracy(&self, x: ArrayView2<f64>, t: ArrayView2<f64>) -> f64 {
        let y = self.predict(x);
        let y = y.map_axis(Axis(1), argmax);
        let t = t.map_axis(Axis(1), argmax);
        let batch_size = x.shape()[0] as f64;
        let correct = y.iter().zip(t.iter()).filter(|(y, t)| y == t).count() as f64;
        correct / batch_size
    }
    fn numerical_gradient(&self, x: ArrayView2<f64>, t: ArrayView2<f64>) -> TwoLayerNetGradient {
        let h = 1.0e-4;
        let mut grad = TwoLayerNetGradient {
            dw1: Array::zeros(self.w1.raw_dim()),
            db1: Array::zeros(self.b1.raw_dim()),
            dw2: Array::zeros(self.w2.raw_dim()),
            db2: Array::zeros(self.b2.raw_dim()),
        };
        let mut copy = self.clone();
        for iter in self.w1.indexed_iter() {
            copy.w1[iter.0] += h;
            let loss1 = copy.loss(x, t);
            copy.w1[iter.0] -= 2.0 * h;
            let loss2 = copy.loss(x, t);
            copy.w1[iter.0] += h;
            grad.dw1[iter.0] = (loss1 - loss2) / (2.0 * h);
        }
        for iter in self.b1.indexed_iter() {
            copy.b1[iter.0] += h;
            let loss1 = copy.loss(x, t);
            copy.b1[iter.0] -= 2.0 * h;
            let loss2 = copy.loss(x, t);
            copy.b1[iter.0] += h;
            grad.db1[iter.0] = (loss1 - loss2) / (2.0 * h);
        }
        for iter in self.w2.indexed_iter() {
            copy.w2[iter.0] += h;
            let loss1 = copy.loss(x, t);
            copy.w2[iter.0] -= 2.0 * h;
            let loss2 = copy.loss(x, t);
            copy.w2[iter.0] += h;
            grad.dw2[iter.0] = (loss1 - loss2) / (2.0 * h);
        }
        for iter in self.b2.indexed_iter() {
            copy.b2[iter.0] += h;
            let loss1 = copy.loss(x, t);
            copy.b2[iter.0] -= 2.0 * h;
            let loss2 = copy.loss(x, t);
            copy.b2[iter.0] += h;
            grad.db2[iter.0] = (loss1 - loss2) / (2.0 * h);
        }
        grad
    }
}

fn main() {
    let x = array![[-3.0, -2.0, -1.0], [0.0, 1.0, 2.0]];
    let mut relu_layer = ReluLayer::new();
    let y = relu_layer.forward(&x);
    println!("y: {:?}", y);
    let dx = relu_layer.backward(&Array2::ones(x.raw_dim()));
    println!("dx: {:?}", dx);

    println!("{}", separator());

    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let mut sigmoid_layer = SigmoidLayer::new();
    let y = sigmoid_layer.forward(&x);
    let mut layer1 = MulLayer::new();
    let mut layer2 = ExpLayer::new();
    let mut layer3 = AddLayer::new();
    let mut layer4 = DivLayer::new();
    let y1 = layer1.forward(&(x.clone(), Array::from_elem((2, 2), -1.0)));
    let y2 = layer2.forward(&y1);
    let y3 = layer3.forward(&(y2, Array::from_elem((2, 2), 1.0)));
    let y4 = layer4.forward(&y3);
    println!("y: {:?}", y);
    println!("(check): {:?}", y4);
    let dx = sigmoid_layer.backward(&Array2::ones(x.raw_dim()));
    println!("dx: {:?}", dx);
    let dy3 = layer4.backward(&Array2::ones(x.raw_dim()));
    let dy2 = layer3.backward(&dy3).0;
    let dy1 = layer2.backward(&dy2);
    let dx = layer1.backward(&dy1).0;

    println!("(check): {:?}", dx);
}
