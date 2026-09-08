use std::marker::PhantomData;

use ndarray::{Array, Dimension};

use super::optimize::{Optimize, OptimizeFactory};

pub struct SGD<D: Dimension> {
    learning_rate: f64,
    d: PhantomData<D>,
}

impl<D: Dimension> SGD<D> {
    pub(crate) fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            d: PhantomData,
        }
    }
}

impl<D: Dimension> Optimize<D> for SGD<D> {
    fn update(&mut self, w: &mut Array<f64, D>, grad: &Array<f64, D>) {
        *w -= &(grad * self.learning_rate);
    }
}

/// `SGD` の生成器。母数として `learning_rate` のみを保持する。
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use zero_deeplearning::optimize::optimize::{Optimize, OptimizeFactory};
/// use zero_deeplearning::optimize::sgd::SGDFactory;
///
/// let mut w = array![1.0, 2.0];
/// let grad = array![1.0, 1.0];
///
/// let factory = SGDFactory::new(0.1);
/// let mut sgd = factory.create(w.raw_dim());
/// sgd.update(&mut w, &grad);
///
/// assert_eq!(w, array![0.9, 1.9]);
/// ```
pub struct SGDFactory {
    learning_rate: f64,
}

impl SGDFactory {
    /// 学習率 `learning_rate` を母数として持つ `SGDFactory` を生成する。
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
}

impl OptimizeFactory for SGDFactory {
    type Optimize<D: Dimension> = SGD<D>;
    /// `dim` を無視して `SGD` を生成する。
    fn create<D: Dimension>(&self, _dim: D) -> Self::Optimize<D> {
        SGD::new(self.learning_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use ndarray_rand::rand::{rngs::StdRng, Rng, SeedableRng};

    const TRIALS: usize = 10;
    const WINDOW: usize = 10;
    const THRESHOLD: f64 = 1e-6;
    // 収束したとみなす時点で原点にどれだけ近いかの許容誤差。
    const ORIGIN_TOLERANCE: f64 = 1e-2;

    /// `z = 1/20 x^2 + y^2` の点(x, y)における勾配。
    fn grad_at(w: &Array<f64, ndarray::Ix1>) -> Array<f64, ndarray::Ix1> {
        array![w[0] / 10.0, 2.0 * w[1]]
    }

    #[test]
    fn update_converges_to_origin_for_random_starting_points() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..TRIALS {
            let mut w = array![rng.gen_range(-10.0..10.0), rng.gen_range(-10.0..10.0)];
            let mut sgd = SGD::new(0.1);
            let mut recent_step_sizes: std::collections::VecDeque<f64> =
                std::collections::VecDeque::with_capacity(WINDOW);

            let mut converged = false;
            for _ in 0..100_000 {
                let grad = grad_at(&w);
                let before = w.clone();
                sgd.update(&mut w, &grad);
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
    fn factory_create_produces_optimize_with_same_learning_rate() {
        let mut w = array![1.0, 2.0];
        let grad = array![1.0, 1.0];
        let factory = SGDFactory::new(0.1);

        let mut sgd = factory.create(w.raw_dim());
        sgd.update(&mut w, &grad);

        assert_eq!(w, array![0.9, 1.9]);
    }
}
