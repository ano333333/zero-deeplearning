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

struct Network {
    w1: Array2<f64>,
    b1: Array1<f64>,
    w2: Array2<f64>,
    b2: Array1<f64>,
    w3: Array2<f64>,
    b3: Array1<f64>,
}

fn init_network() -> Network {
    Network {
        w1: array![[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]],
        b1: array![0.1, 0.2, 0.3],
        w2: array![[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]],
        b2: array![0.1, 0.2],
        w3: array![[0.1, 0.3], [0.2, 0.4]],
        b3: array![0.1, 0.2],
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
    // let z1 = sigmoid(a1.view());
    let z1 = relu(a1.view());
    let a2 = z1.dot(&network.w2) + &network.b2;
    // let z2 = sigmoid(a2.view());
    let z2 = relu(a2.view());
    let a3 = z2.dot(&network.w3) + &network.b3;
    identity_function(a3.view())
}

fn main() {
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

    let root = BitMapBackend::new("images/forward.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption("Forward", ("sans-serif", 30).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-1.0..1.0, -1.0..1.0)
        .unwrap();
    chart.configure_mesh().draw().unwrap();

    // 直線x = -5,-4,...,4,5がforwardによりどのように移動するか?
    for x_coord in -5..=5 {
        // 行列xの各行は(x_coord,100.0)から(x_coord,-100.0)
        let x_col1 = Array::range(-100.0, 100.0, 0.1);
        let x_col0 = Array::from_elem(x_col1.len(), x_coord as f64);
        let x = stack![Axis(1), x_col0, x_col1];
        let y = forward(&network, x.view());
        println!("x: {:?}", x);
        println!("y: {:?}", y);
        // y[0],y[1],...,y[-1]を描画
        chart
            .draw_series(LineSeries::new(
                (0..y.shape()[0]).map(|i| (y[[i, 0]], y[[i, 1]])),
                &BLACK,
            ))
            .unwrap();
        println!("{}", separator());
    }
    root.present().unwrap();
}
