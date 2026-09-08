use ndarray::{Array, Dimension};

use super::optimize::Optimize;

pub struct AdaGrad<D: Dimension> {
    learning_rate: f64,
    h: Array<f64, D>,
}

impl<D: Dimension> AdaGrad<D> {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            h: Array::zeros(D::default()),
        }
    }
}

impl<D: Dimension> Optimize<D> for AdaGrad<D> {
    fn update(&mut self, w: &mut Array<f64, D>, grad: &Array<f64, D>) {
        // 初めての呼び出し時はhのdimが(0,0,...)なので、wの形に揃える
        if self.h.len() == 0 {
            self.h = Array::zeros(w.raw_dim());
        }
        let h = &self.h + &(grad * grad);
        let h_sqrt = h.map(|x| x.sqrt() + 1e-7);
        *w -= &(grad * self.learning_rate / h_sqrt);
        self.h = h;
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
            let mut w = array![rng.gen_range(-10.0..10.0), rng.gen_range(-10.0..10.0)];
            let mut ada_grad = AdaGrad::new(0.1);
            let mut recent_step_sizes: std::collections::VecDeque<f64> =
                std::collections::VecDeque::with_capacity(WINDOW);

            let mut converged = false;
            for _ in 0..100_000 {
                let grad = grad_at(&w);
                let before = w.clone();
                ada_grad.update(&mut w, &grad);
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
}
