use ndarray::prelude::*;

use crate::optimize::optimize::Optimize;
use crate::two_layer_net::{TwoLayerNet, TwoLayerNetGradient};

/// 勾配にL2正則化(weight decay)項を加算する。
///
/// `TwoLayerNet::gradient` が返す勾配は損失関数のみに基づく値であり、正則化項を含まない。
/// この関数は `weight_decay * パラメータ` を各勾配に加算した新しい勾配を返す。
pub fn apply_weight_decay(
    grad: TwoLayerNetGradient,
    network: &TwoLayerNet,
    weight_decay: f64,
) -> TwoLayerNetGradient {
    TwoLayerNetGradient {
        dw1: weight_decay * &network.w1 + &grad.dw1,
        db1: weight_decay * &network.b1 + &grad.db1,
        dbatch_aff: weight_decay * &network.batch_aff + &grad.dbatch_aff,
        dw2: weight_decay * &network.w2 + &grad.dw2,
        db2: weight_decay * &network.b2 + &grad.db2,
    }
}

/// 勾配を使って `network` の各パラメータをoptimizerで更新する。
pub fn optimize_step(
    network: &mut TwoLayerNet,
    grad: &TwoLayerNetGradient,
    optimize_w1: &mut impl Optimize<Ix2>,
    optimize_b1: &mut impl Optimize<Ix1>,
    optimize_batch_aff: &mut impl Optimize<Ix1>,
    optimize_w2: &mut impl Optimize<Ix2>,
    optimize_b2: &mut impl Optimize<Ix1>,
) {
    optimize_w1.update(&mut network.w1, &grad.dw1);
    optimize_b1.update(&mut network.b1, &grad.db1);
    optimize_batch_aff.update(&mut network.batch_aff, &grad.dbatch_aff);
    optimize_w2.update(&mut network.w2, &grad.dw2);
    optimize_b2.update(&mut network.b2, &grad.db2);
}

/// 「gradient算出 -> 正則化 -> optimize」を1イテレーションとして実行する。
pub fn train_step(
    network: &mut TwoLayerNet,
    x: &Array2<f64>,
    t: &Array2<f64>,
    weight_decay: f64,
    optimize_w1: &mut impl Optimize<Ix2>,
    optimize_b1: &mut impl Optimize<Ix1>,
    optimize_batch_aff: &mut impl Optimize<Ix1>,
    optimize_w2: &mut impl Optimize<Ix2>,
    optimize_b2: &mut impl Optimize<Ix1>,
) {
    let grad = network.gradient(x, t);
    let grad = apply_weight_decay(grad, network, weight_decay);
    optimize_step(
        network,
        &grad,
        optimize_w1,
        optimize_b1,
        optimize_batch_aff,
        optimize_w2,
        optimize_b2,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimize::sgd::SGD;
    use ndarray_rand::rand_distr::Normal;

    fn new_network() -> TwoLayerNet {
        TwoLayerNet::new(3, 4, 2, &Normal::new(0.0, 1.0).unwrap())
    }

    #[test]
    fn apply_weight_decay_adds_scaled_parameters_to_gradient() {
        let network = new_network();
        let grad = TwoLayerNetGradient {
            dw1: Array2::zeros(network.w1.dim()),
            db1: Array1::zeros(network.b1.dim()),
            dbatch_aff: Array1::zeros(network.batch_aff.dim()),
            dw2: Array2::zeros(network.w2.dim()),
            db2: Array1::zeros(network.b2.dim()),
        };

        let decayed = apply_weight_decay(grad, &network, 0.1);

        assert_eq!(decayed.dw1, 0.1 * &network.w1);
        assert_eq!(decayed.db1, 0.1 * &network.b1);
        assert_eq!(decayed.dbatch_aff, 0.1 * &network.batch_aff);
        assert_eq!(decayed.dw2, 0.1 * &network.w2);
        assert_eq!(decayed.db2, 0.1 * &network.b2);
    }

    #[test]
    fn apply_weight_decay_is_noop_when_weight_decay_is_zero() {
        let network = new_network();
        let grad = TwoLayerNetGradient {
            dw1: Array2::ones(network.w1.dim()),
            db1: Array1::ones(network.b1.dim()),
            dbatch_aff: Array1::ones(network.batch_aff.dim()),
            dw2: Array2::ones(network.w2.dim()),
            db2: Array1::ones(network.b2.dim()),
        };

        let decayed = apply_weight_decay(grad, &network, 0.0);

        assert_eq!(decayed.dw1, Array2::ones(network.w1.dim()));
        assert_eq!(decayed.db1, Array1::ones(network.b1.dim()));
        assert_eq!(decayed.dbatch_aff, Array1::ones(network.batch_aff.dim()));
        assert_eq!(decayed.dw2, Array2::ones(network.w2.dim()));
        assert_eq!(decayed.db2, Array1::ones(network.b2.dim()));
    }

    #[test]
    fn optimize_step_updates_all_parameters() {
        let mut network = new_network();
        let before = network.clone();
        let grad = TwoLayerNetGradient {
            dw1: Array2::ones(network.w1.dim()),
            db1: Array1::ones(network.b1.dim()),
            dbatch_aff: Array1::ones(network.batch_aff.dim()),
            dw2: Array2::ones(network.w2.dim()),
            db2: Array1::ones(network.b2.dim()),
        };
        let mut optimize_w1 = SGD::<Ix2>::new(0.1);
        let mut optimize_b1 = SGD::<Ix1>::new(0.1);
        let mut optimize_batch_aff = SGD::<Ix1>::new(0.1);
        let mut optimize_w2 = SGD::<Ix2>::new(0.1);
        let mut optimize_b2 = SGD::<Ix1>::new(0.1);

        optimize_step(
            &mut network,
            &grad,
            &mut optimize_w1,
            &mut optimize_b1,
            &mut optimize_batch_aff,
            &mut optimize_w2,
            &mut optimize_b2,
        );

        assert_eq!(network.w1, &before.w1 - 0.1);
        assert_eq!(network.b1, &before.b1 - 0.1);
        assert_eq!(network.batch_aff, &before.batch_aff - 0.1);
        assert_eq!(network.w2, &before.w2 - 0.1);
        assert_eq!(network.b2, &before.b2 - 0.1);
    }

    #[test]
    fn train_step_changes_parameters_using_computed_gradient() {
        let mut network = new_network();
        let before = network.clone();
        let x = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, -0.1, 0.4, 0.2]).unwrap();
        let t = array![[1.0, 0.0], [0.0, 1.0]];
        let mut optimize_w1 = SGD::<Ix2>::new(0.01);
        let mut optimize_b1 = SGD::<Ix1>::new(0.01);
        let mut optimize_batch_aff = SGD::<Ix1>::new(0.01);
        let mut optimize_w2 = SGD::<Ix2>::new(0.01);
        let mut optimize_b2 = SGD::<Ix1>::new(0.01);

        train_step(
            &mut network,
            &x,
            &t,
            0.0,
            &mut optimize_w1,
            &mut optimize_b1,
            &mut optimize_batch_aff,
            &mut optimize_w2,
            &mut optimize_b2,
        );

        assert_ne!(network.w1, before.w1);
        assert_ne!(network.w2, before.w2);
    }
}
