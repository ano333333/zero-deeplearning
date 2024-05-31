mod mnist;
use ndarray::{prelude::*, NdIndex};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

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
    let max = x.fold(0.0 / 0.0, |acc, &x| x.max(acc));
    let c = x.mapv(|x| (x - max).exp()).sum();
    x.mapv(|x| (x - max).exp() / c)
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
    f: fn(ArrayView<f64, D>) -> f64,
    x: ArrayView<f64, D>,
) -> Array<f64, D>
where
    <D as ndarray::Dimension>::Pattern: NdIndex<D>,
{
    let h = 1e-4;
    let mut grad = Array::<f64, D>::zeros(x.raw_dim());
    let mut x_mut = x.to_owned();
    for iter in x.indexed_iter() {
        x_mut[iter.0.clone()] = iter.1 + h;
        let fxh1 = f(x_mut.view());
        x_mut[iter.0.clone()] = iter.1 - h;
        let fxh2 = f(x_mut.view());
        grad[iter.0.clone()] = (fxh1 - fxh2) / (2.0 * h);
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

fn main() {
    let net = SimpleNet::new();
    println!("{:?}", net.w);
    println!("{}", separator());

    let x = array![[0.6, 0.9]];
    let p = net.predict(x.view());
    println!("{:?}", p);
    println!("{}", separator());

    let t = array![[0.0, 0.0, 1.0]];
    println!("{}", net.loss(x.view(), t.view()));
}
