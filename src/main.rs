mod layer;
use layer::affine_layer::AffineLayer;
use layer::layer::Layer;
use layer::softmax_with_loss_layer::SoftmaxWithLossLayer;
mod mnist;
mod subfunction;

use ndarray::prelude::*;
use subfunction::cross_entropy_error::cross_entropy_error;
use subfunction::softmax_batch::softmax_batch;
fn separator() -> String {
    (0..20).map(|_| "-").collect::<String>()
}

fn main() {
    let x = array![[0.5, 0.45, 0.05], [0.3, 0.7, 0.0]];
    let w = array![[0.1, 0.3, 0.5], [0.2, 0.4, 0.6], [0.3, 0.5, 0.7]];
    let b = array![0.1, 0.2, 0.3];
    let mut affine_layer = AffineLayer::new(&w, &b);
    let y = affine_layer.forward(&x);
    println!("y: {:?}", y);
    let dx = affine_layer.backward(&Array2::from_elem((2, 3), 1.0));
    println!("dx: {:?}", dx);
    println!("dw: {:?}", affine_layer.dw);
    println!("db: {:?}", affine_layer.db);

    println!("{}", separator());

    let check_y = x.dot(&w) + &b;
    let check_dx = Array2::from_elem((2, 3), 1.0).dot(&w.t());
    let check_dw = x.t().dot(&Array2::from_elem((2, 3), 1.0));
    let check_db = Array1::from_elem(3, 2.0);
    println!("check_y: {:?}", check_y);
    println!("check_dx: {:?}", check_dx);
    println!("check_dw: {:?}", check_dw);
    println!("check_db: {:?}", check_db);

    println!("{}", separator());
    println!("{}", separator());

    let y = array![[0.5, 0.45, 0.05], [0.3, 0.7, 0.0]];
    let t = array![[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let mut softmax_with_loss_layer = SoftmaxWithLossLayer::new(&t);
    let loss = softmax_with_loss_layer.forward(&y);
    println!("loss: {}", loss);
    let dy = softmax_with_loss_layer.backward(&1.0);
    println!("dy: {:?}", dy);

    println!("{}", separator());

    let check_softmax = softmax_batch(y.view());
    let check_loss = cross_entropy_error(check_softmax.view(), t.view());
    println!("check_loss: {}", check_loss);
    let mut check_dy = Array2::zeros((2, 3));
    for i in 0..2 {
        for j in 0..3 {
            let mut y = y.clone();
            let org = y[[i, j]];
            y[[i, j]] = org + 1e-4;
            let softmax1 = softmax_batch(y.view());
            let loss1 = cross_entropy_error(softmax1.view(), t.view());
            y[[i, j]] = org - 1e-4;
            let softmax2 = softmax_batch(y.view());
            let loss2 = cross_entropy_error(softmax2.view(), t.view());
            y[[i, j]] = org;
            check_dy[[i, j]] = (loss1 - loss2) / (2.0 * 1e-4);
        }
    }
    println!("check_dy: {:?}", check_dy);
}
