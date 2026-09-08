mod layer;
mod mnist;
mod optimize;
mod subfunction;
mod train;
mod two_layer_net;
use ndarray_rand::rand;
use ndarray_rand::rand::seq::IteratorRandom;
use ndarray_rand::rand::Rng;
use ndarray_rand::rand_distr::Normal;
use train::{create_optimizers, train_step};
use two_layer_net::TwoLayerNet;

use crate::optimize::sgd::SGDFactory;
fn separator() -> String {
    (0..20).map(|_| "-").collect::<String>()
}

fn main() {
    let training_size = 50_000;
    let validation_size = 10_000;
    let input_layer_size = 28 * 28;
    let hidden_layer_size = 50;
    let output_layer_size = 10;
    let mnist_data = mnist::load_mnist::load_mnist(
        None,
        Some(training_size),
        Some(validation_size),
        Some(10_000),
    );

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
        let sgd_factory = SGDFactory::new(learning_rate);
        let mut optimizers = create_optimizers(&network, &sgd_factory);

        // 学習
        for i in 0..iters_num_per_val {
            let batch_mask = &indexes[i];
            let (x_batch, t_batch) = mnist_data.train_batch(batch_mask);

            train_step(
                &mut network,
                &x_batch,
                &t_batch,
                weight_decay,
                &mut optimizers,
            );
        }
        // 検証データで評価
        let val_loss = network.loss(&mnist_data.validation_data, &mnist_data.validation_labels);
        let val_acc = network.accuracy(&mnist_data.validation_data, &mnist_data.validation_labels);
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
    let sgd_factory = SGDFactory::new(learning_rate);
    let mut optimizers = create_optimizers(&network, &sgd_factory);

    for _ in 0..iters_num {
        let batch_mask = all_indexes
            .iter()
            .choose_multiple(&mut rng, batch_size as usize)
            .iter()
            .map(|&i| *i)
            .collect::<Vec<usize>>();
        let (x_batch, t_batch) = mnist_data.train_batch(&batch_mask);

        train_step(
            &mut network,
            &x_batch,
            &t_batch,
            weight_decay,
            &mut optimizers,
        );
    }
    // テストデータで評価
    let test_loss = network.loss(&mnist_data.test_data, &mnist_data.test_labels);
    let test_acc = network.accuracy(&mnist_data.test_data, &mnist_data.test_labels);
    println!("test_loss: {:?}, test_acc: {:?}", test_loss, test_acc);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TwoLayerNetとtrain_stepを通したE2Eテスト。
    ///
    /// `data/` に配置された実MNISTデータセットを要求する。データが無い環境では
    /// 失敗するため、通常の `cargo test` では実行されないよう `#[ignore]` を
    /// 付けている。実行するには `cargo test -- --ignored` を使う。
    ///
    /// 小サイズの固定訓練データ(50件)でフルバッチ学習を合計300イテレート行い、
    /// 100イテレートごとにテストデータ(20件)でのlossを測定して、単調に
    /// 減少していることを確認する。あわせて、学習開始前と終了後のテストデータ
    /// でのaccuracyも測定し、学習によって上昇していることを確認する。
    #[test]
    #[ignore]
    fn e2e_loss_decreases_as_training_progresses() {
        let input_layer_size = 28 * 28;
        let hidden_layer_size = 50;
        let output_layer_size = 10;
        let learning_rate = 0.01;

        let mnist_data = mnist::load_mnist::load_mnist(None, Some(50), Some(1), Some(20));
        let train_indexes = (0..50).collect::<Vec<usize>>();
        let (x_train, t_train) = mnist_data.train_batch(&train_indexes);

        let mut network = TwoLayerNet::new(
            input_layer_size,
            hidden_layer_size,
            output_layer_size,
            &Normal::new(0.0, 1.0 / (input_layer_size as f64)).unwrap(),
        );
        let sgd_factory = SGDFactory::new(learning_rate);
        let mut optimizers = create_optimizers(&network, &sgd_factory);

        let initial_test_accuracy =
            network.accuracy(&mnist_data.test_data, &mnist_data.test_labels);

        let mut test_losses = Vec::new();
        for i in 1..=300 {
            train_step(&mut network, &x_train, &t_train, 0.0, &mut optimizers);

            if i % 100 == 0 {
                let test_loss = network.loss(&mnist_data.test_data, &mnist_data.test_labels);
                test_losses.push(test_loss);
            }
        }

        assert_eq!(test_losses.len(), 3);
        assert!(
            test_losses[0] > test_losses[1],
            "expected loss to decrease between iteration 100 and 200, got {:?}",
            test_losses
        );
        assert!(
            test_losses[1] > test_losses[2],
            "expected loss to decrease between iteration 200 and 300, got {:?}",
            test_losses
        );

        let final_test_accuracy =
            network.accuracy(&mnist_data.test_data, &mnist_data.test_labels);
        assert!(
            final_test_accuracy > initial_test_accuracy,
            "expected accuracy to improve after training, got {} -> {}",
            initial_test_accuracy,
            final_test_accuracy
        );
    }
}
