//! TwoLayerNetとtrain_stepを通したE2Eテスト。
//!
//! `data/` に配置された実MNISTデータセットを要求する。データが無い環境では
//! 失敗するため、通常の `cargo test` では実行されないよう `#[ignore]` を
//! 付けている。実行するには `cargo test -- --ignored` を使う。

use ndarray_rand::rand_distr::Normal;
use zero_deeplearning::optimize::sgd::SGDFactory;
use zero_deeplearning::train::{create_optimizers, train_step};
use zero_deeplearning::two_layer_net::TwoLayerNet;

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

    let mnist_data =
        zero_deeplearning::mnist::load_mnist::load_mnist(None, Some(50), Some(1), Some(20));
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

    let initial_test_accuracy = network.accuracy(&mnist_data.test_data, &mnist_data.test_labels);

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

    let final_test_accuracy = network.accuracy(&mnist_data.test_data, &mnist_data.test_labels);
    assert!(
        final_test_accuracy > initial_test_accuracy,
        "expected accuracy to improve after training, got {} -> {}",
        initial_test_accuracy,
        final_test_accuracy
    );
}
