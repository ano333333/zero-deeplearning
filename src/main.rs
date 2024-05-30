mod mnist;
use ndarray::{concatenate, prelude::*};

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

// fn forward(network: &Network, x: ArrayView1<f64>) -> Array1<f64> {
//     let a1 = x.dot(&network.w1) + &network.b1;
//     let z1 = sigmoid(a1.view());
//     let a2 = z1.dot(&network.w2) + &network.b2;
//     let z2 = sigmoid(a2.view());
//     let a3 = z2.dot(&network.w3) + &network.b3;
//     identity_function(a3.view())
// }

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

fn main() {
    let (train_data, trn_lbl, validation_data, val_lbl) = mnist::load_mnist::load_mnist(None, None);

    let t1 = array![[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
    let y1 = array![[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]];
    let error1 = cross_entropy_error(y1.view(), t1.view());
    println!("{}", error1);
    println!("{}", separator());

    let t2 = array![[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
    let y2 = array![[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0],];
    let error2 = cross_entropy_error(y2.view(), t2.view());
    println!("{}", error2);
    println!("{}", separator());

    let t = concatenate(Axis(0), &[t1.view(), t2.view()]).unwrap();
    let y = concatenate(Axis(0), &[y1.view(), y2.view()]).unwrap();
    let error = cross_entropy_error(y.view(), t.view());
    println!("{}", error);
    println!("{}", (error1 + error2) / 2.0);
}
