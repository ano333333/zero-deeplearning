use ndarray::{Array, Dimension};

use super::optimize::{Optimize, OptimizeFactory};

pub struct Momentum<D: Dimension> {
    learning_rate: f64,
    momentum: f64,
    v: Array<f64, D>,
}

impl<D: Dimension> Momentum<D> {
    pub(crate) fn new(learning_rate: f64, momentum: f64, dim: D) -> Self {
        Self {
            learning_rate,
            momentum,
            v: Array::zeros(dim),
        }
    }
}

impl<D: Dimension> Optimize<D> for Momentum<D> {
    fn update(&mut self, w: &mut Array<f64, D>, grad: &Array<f64, D>) {
        let v = self.momentum * &self.v - self.learning_rate * grad;
        *w += &v;
        self.v = v;
    }
}

/// `Momentum` の生成器。母数として `learning_rate` と `momentum` を保持する。
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use zero_deeplearning::optimize::optimize::{Optimize, OptimizeFactory};
/// use zero_deeplearning::optimize::momentum::MomentumFactory;
///
/// let mut w = array![1.0, 2.0];
/// let grad = array![1.0, 1.0];
///
/// let factory = MomentumFactory::new(0.1, 0.9);
/// let mut momentum = factory.create(w.raw_dim());
/// momentum.update(&mut w, &grad);
///
/// assert_eq!(w, array![0.9, 1.9]);
/// ```
pub struct MomentumFactory {
    learning_rate: f64,
    momentum: f64,
}

impl MomentumFactory {
    /// 学習率 `learning_rate` と momentum係数 `momentum` を母数として持つ
    /// `MomentumFactory` を生成する。
    pub fn new(learning_rate: f64, momentum: f64) -> Self {
        Self {
            learning_rate,
            momentum,
        }
    }
}

impl OptimizeFactory for MomentumFactory {
    type Optimize<D: Dimension> = Momentum<D>;
    /// `dim` の形で速度 `v` をゼロ初期化した `Momentum` を生成する。
    fn create<D: Dimension>(&self, dim: D) -> Self::Optimize<D> {
        Momentum::new(self.learning_rate, self.momentum, dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use ndarray_rand::rand::{rngs::StdRng, Rng, SeedableRng};

    const TRIALS: usize = 10;
    const WINDOW: usize = 10;
    const THRESHOLD: f64 = 1e-8;
    // 収束したとみなす時点で原点にどれだけ近いかの許容誤差。
    const ORIGIN_TOLERANCE: f64 = 1e-4;

    /// `z = 1/20 x^2 + y^2` の点(x, y)における勾配。
    fn grad_at(w: &Array<f64, ndarray::Ix1>) -> Array<f64, ndarray::Ix1> {
        array![w[0] / 10.0, 2.0 * w[1]]
    }

    #[test]
    fn update_converges_to_origin_for_random_starting_points() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..TRIALS {
            let mut w = array![rng.random_range(-10.0..10.0), rng.random_range(-10.0..10.0)];
            let mut momentum = Momentum::new(0.1, 0.9, w.raw_dim());
            let mut recent_step_sizes: std::collections::VecDeque<f64> =
                std::collections::VecDeque::with_capacity(WINDOW);

            let mut converged = false;
            for _ in 0..100_000 {
                let grad = grad_at(&w);
                let before = w.clone();
                momentum.update(&mut w, &grad);
                let step_size = (&w - &before).mapv(|v| v * v).sum().sqrt();

                if recent_step_sizes.len() == WINDOW {
                    recent_step_sizes.pop_front();
                }
                recent_step_sizes.push_back(step_size);

                if recent_step_sizes.len() == WINDOW {
                    let average =
                        recent_step_sizes.iter().sum::<f64>() / recent_step_sizes.len() as f64;
                    if average <= THRESHOLD {
                        converged = true;
                        break;
                    }
                }
            }

            assert!(converged, "did not converge within iteration budget");
            let distance_from_origin = w.mapv(|v| v * v).sum().sqrt();
            assert!(
                distance_from_origin <= ORIGIN_TOLERANCE,
                "expected point near origin, got {w:?} (distance {distance_from_origin})"
            );
        }
    }

    #[test]
    fn factory_create_produces_optimize_with_same_parameters() {
        let mut w = array![1.0, 2.0];
        let grad = array![1.0, 1.0];
        let factory = MomentumFactory::new(0.1, 0.9);

        let mut momentum = factory.create(w.raw_dim());
        momentum.update(&mut w, &grad);

        assert_eq!(w, array![0.9, 1.9]);
    }
}
