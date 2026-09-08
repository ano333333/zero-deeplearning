use ndarray::prelude::*;

use crate::optimize::optimize::{Optimize, OptimizeFactory};
use crate::two_layer_net::{TwoLayerNet, TwoLayerNetGradient};

/// `TwoLayerNet` の5パラメータ(w1, b1, batch_aff, w2, b2)分の `Optimize` の組。
///
/// `F: OptimizeFactory` が持つ `create` で、パラメータごとの次元(`Ix2` または
/// `Ix1`)に応じた `Optimize` を生成して保持する。
pub struct TwoLayerNetOptimizers<F: OptimizeFactory> {
    pub w1: F::Optimize<Ix2>,
    pub b1: F::Optimize<Ix1>,
    pub batch_aff: F::Optimize<Ix1>,
    pub w2: F::Optimize<Ix2>,
    pub b2: F::Optimize<Ix1>,
}

/// `factory` を使い、`network` の各パラメータの次元に合わせた
/// `TwoLayerNetOptimizers` を作成する。
///
/// # Examples
///
/// ```rust,ignore
/// use zero_deeplearning::optimize::sgd::SGDFactory;
/// use zero_deeplearning::train::create_optimizers;
///
/// let network = TwoLayerNet::new(784, 50, 10, &dist);
/// let factory = SGDFactory::new(0.1);
/// let optimizers = create_optimizers(&network, &factory);
/// ```
pub fn create_optimizers<F: OptimizeFactory>(
    network: &TwoLayerNet,
    factory: &F,
) -> TwoLayerNetOptimizers<F> {
    TwoLayerNetOptimizers {
        w1: factory.create(network.w1.raw_dim()),
        b1: factory.create(network.b1.raw_dim()),
        batch_aff: factory.create(network.batch_aff.raw_dim()),
        w2: factory.create(network.w2.raw_dim()),
        b2: factory.create(network.b2.raw_dim()),
    }
}

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

/// 勾配を使って `network` の各パラメータを `optimizers` で更新する。
pub fn optimize_step<F: OptimizeFactory>(
    network: &mut TwoLayerNet,
    grad: &TwoLayerNetGradient,
    optimizers: &mut TwoLayerNetOptimizers<F>,
) {
    optimizers.w1.update(&mut network.w1, &grad.dw1);
    optimizers.b1.update(&mut network.b1, &grad.db1);
    optimizers
        .batch_aff
        .update(&mut network.batch_aff, &grad.dbatch_aff);
    optimizers.w2.update(&mut network.w2, &grad.dw2);
    optimizers.b2.update(&mut network.b2, &grad.db2);
}

/// 「gradient算出 -> 正則化 -> optimize」を1イテレーションとして実行する。
pub fn train_step<F: OptimizeFactory>(
    network: &mut TwoLayerNet,
    x: &Array2<f64>,
    t: &Array2<f64>,
    weight_decay: f64,
    optimizers: &mut TwoLayerNetOptimizers<F>,
) {
    let grad = network.gradient(x, t);
    let grad = apply_weight_decay(grad, network, weight_decay);
    optimize_step(network, &grad, optimizers);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimize::sgd::SGDFactory;
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
    fn create_optimizers_builds_one_optimize_per_parameter() {
        let network = new_network();
        let factory = SGDFactory::new(0.1);

        let mut optimizers = create_optimizers(&network, &factory);
        let mut network_after = network.clone();
        let grad = TwoLayerNetGradient {
            dw1: Array2::ones(network.w1.dim()),
            db1: Array1::ones(network.b1.dim()),
            dbatch_aff: Array1::ones(network.batch_aff.dim()),
            dw2: Array2::ones(network.w2.dim()),
            db2: Array1::ones(network.b2.dim()),
        };

        optimize_step(&mut network_after, &grad, &mut optimizers);

        assert_eq!(network_after.w1, &network.w1 - 0.1);
        assert_eq!(network_after.b1, &network.b1 - 0.1);
        assert_eq!(network_after.batch_aff, &network.batch_aff - 0.1);
        assert_eq!(network_after.w2, &network.w2 - 0.1);
        assert_eq!(network_after.b2, &network.b2 - 0.1);
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
        let factory = SGDFactory::new(0.1);
        let mut optimizers = create_optimizers(&network, &factory);

        optimize_step(&mut network, &grad, &mut optimizers);

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
        let factory = SGDFactory::new(0.01);
        let mut optimizers = create_optimizers(&network, &factory);

        train_step(&mut network, &x, &t, 0.0, &mut optimizers);

        assert_ne!(network.w1, before.w1);
        assert_ne!(network.w2, before.w2);
    }
}
