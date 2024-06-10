mod layer;
use layer::affine_layer::AffineLayer;
use layer::layer::Layer;
use layer::relu_layer::ReluLayer;
use layer::softmax_with_loss_layer::SoftmaxWithLossLayer;
mod mnist;
mod optimize;
mod subfunction;
mod two_layer_net;
use ndarray::prelude::*;
use ndarray_rand::rand;
use ndarray_rand::rand::seq::IteratorRandom;
use ndarray_rand::rand::Rng;
use ndarray_rand::rand_distr::Normal;
use optimize::optimize::Optimize;
use two_layer_net::TwoLayerNet;
use two_layer_net::TwoLayerNetGradient;

use crate::layer::batch_normalization_layer::BatchNormalizationLayer;
use crate::optimize::sgd::SGD;
fn separator() -> String {
    (0..20).map(|_| "-").collect::<String>()
}

fn main() {
    let training_size = 50_000;
    let validation_size = 10_000;
    let input_layer_size = 28 * 28;
    let hidden_layer_size = 50;
    let output_layer_size = 10;
    let (x_train, t_train, x_val, t_val, x_test, t_test) =
        mnist::load_mnist::load_mnist(Some(training_size), Some(validation_size), Some(10_000));

    let batch_size = 100;
    let iters_num = 10_000;
    let iters_num_per_val = 500;

    let all_indexes = (0..(training_size as usize)).collect::<Vec<usize>>();
    let mut rng = rand::thread_rng();
    // 各イテレートで用いる学習データのインデックスを固定化する
    let indexes = (0..iters_num_per_val)
        .map(|_| {
            all_indexes
                .iter()
                .choose_multiple(&mut rng, batch_size as usize)
                .iter()
                .map(|&i| *i)
                .collect::<Vec<usize>>()
        })
        .collect::<Vec<Vec<usize>>>();
    // ハイパーパラメータ
    // - learning_rate
    // - weight_decay
    // をランダムに選択し学習データで学習。20回繰り返す
    // (バリデーションデータでのloss, learning_rate, weight_decay)のarray
    let mut val_results = Vec::<(f64, f64, f64)>::new();
    for i_val in 0..20 {
        let learning_rate = (10.0 as f64).powf(rng.gen_range(-6.0..-2.0));
        let weight_decay = (10.0 as f64).powf(rng.gen_range(-16.0..-8.0));
        println!(
            "start iterate: {}\nlearning_rate: {}\nweight_decay: {}",
            i_val, learning_rate, weight_decay
        );

        let mut network = TwoLayerNet::new(
            input_layer_size,
            hidden_layer_size,
            output_layer_size,
            &Normal::new(0.0, 1.0 / (input_layer_size as f64)).unwrap(),
        );
        let mut sgd_w1 = SGD::<Ix2>::new(learning_rate);
        let mut sgd_b1 = SGD::<Ix1>::new(learning_rate);
        let mut sgd_w2 = SGD::<Ix2>::new(learning_rate);
        let mut sgd_b2 = SGD::<Ix1>::new(learning_rate);
        let mut sgd_batch_aff = SGD::<Ix1>::new(learning_rate);

        // 学習
        for i in 0..iters_num_per_val {
            let batch_mask = &indexes[i];
            let x_batch = x_train.select(Axis(0), &batch_mask);
            let t_batch = t_train.select(Axis(0), &batch_mask);

            let mut grad = network.gradient(&x_batch, &t_batch);
            grad.dw1 = weight_decay * &network.w1 + &grad.dw1;
            grad.dw2 = weight_decay * &network.w2 + &grad.dw2;
            grad.db1 = weight_decay * &network.b1 + &grad.db1;
            grad.db2 = weight_decay * &network.b2 + &grad.db2;
            grad.dbatch_aff = weight_decay * &network.batch_aff + &grad.dbatch_aff;
            sgd_w1.update(&mut network.w1, &grad.dw1);
            sgd_b1.update(&mut network.b1, &grad.db1);
            sgd_batch_aff.update(&mut network.batch_aff, &grad.dbatch_aff);
            sgd_w2.update(&mut network.w2, &grad.dw2);
            sgd_b2.update(&mut network.b2, &grad.db2);
        }
        // 検証データで評価
        let val_loss = network.loss(&x_val, &t_val);
        let val_acc = network.accuracy(&x_val, &t_val);
        println!("val_loss: {:?}, val_acc: {:?}", val_loss, val_acc);
        println!("{}", separator());
        val_results.push((val_loss, learning_rate, weight_decay));
    }

    // val_resultsを、第一要素降順でソート
    val_results.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    println!("val_results: {:?}", val_results);
    println!("choice: {:?}", val_results[0]);

    // 一番良かった値を用いて本学習
    let learning_rate = val_results[0].1;
    let weight_decay = val_results[0].2;
    let mut network = TwoLayerNet::new(
        input_layer_size,
        hidden_layer_size,
        output_layer_size,
        &Normal::new(0.0, 1.0 / (input_layer_size as f64)).unwrap(),
    );
    let mut sgd_w1 = SGD::<Ix2>::new(learning_rate);
    let mut sgd_b1 = SGD::<Ix1>::new(learning_rate);
    let mut sgd_w2 = SGD::<Ix2>::new(learning_rate);
    let mut sgd_b2 = SGD::<Ix1>::new(learning_rate);
    let mut sgd_batch_aff = SGD::<Ix1>::new(learning_rate);

    for _ in 0..iters_num {
        let batch_mask = all_indexes
            .iter()
            .choose_multiple(&mut rng, batch_size as usize)
            .iter()
            .map(|&i| *i)
            .collect::<Vec<usize>>();
        let x_batch = x_train.select(Axis(0), &batch_mask);
        let t_batch = t_train.select(Axis(0), &batch_mask);

        let mut grad = network.gradient(&x_batch, &t_batch);
        grad.dw1 = weight_decay * &network.w1 + &grad.dw1;
        grad.dw2 = weight_decay * &network.w2 + &grad.dw2;
        grad.db1 = weight_decay * &network.b1 + &grad.db1;
        grad.db2 = weight_decay * &network.b2 + &grad.db2;
        grad.dbatch_aff = weight_decay * &network.batch_aff + &grad.dbatch_aff;
        sgd_w1.update(&mut network.w1, &grad.dw1);
        sgd_b1.update(&mut network.b1, &grad.db1);
        sgd_batch_aff.update(&mut network.batch_aff, &grad.dbatch_aff);
        sgd_w2.update(&mut network.w2, &grad.dw2);
        sgd_b2.update(&mut network.b2, &grad.db2);
    }
    // テストデータで評価
    let test_loss = network.loss(&x_test, &t_test);
    let test_acc = network.accuracy(&x_test, &t_test);
    println!("test_loss: {:?}, test_acc: {:?}", test_loss, test_acc);
}
