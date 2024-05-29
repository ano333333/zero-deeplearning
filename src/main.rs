use ndarray::{prelude::*, stack};
use plotters::prelude::*;

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

fn main() {
    let a = array![0.3, 2.9, 4.0];
    let y = softmax(a.view());
    println!("y: {:?}", y);
    println!("{}", separator());
    let a = array![1010.0, 1000.0, 990.0];
    let y = softmax(a.view());
    println!("y: {:?}", y);
    println!("{}", separator());
    let network = init_network();
    // let x = array![1.0, 0.5];
    // let y = forward(&network, x.view());
    // println!("y: {:?}", y);
    // println!("{}", separator());
    // let x = array![0.1, 0.2];
    // let y = forward(&network, x.view());
    // println!("y: {:?}", y);
    // println!("{}", separator());
    let x = array![[1.0, 0.5], [0.1, 0.2]];
    let y = forward(&network, x.view());
    println!("y: {:?}", y);
    println!("{}", separator());
}
