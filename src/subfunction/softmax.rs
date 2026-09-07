use ndarray::{Array, ArrayView, Dimension};

pub fn softmax<D: Dimension>(x: ArrayView<f64, D>) -> Array<f64, D> {
    let max = x.fold(-1.0 / 0.0, |acc, &x| x.max(acc));
    let c = x.mapv(|x| (x - max).exp()).sum();
    x.mapv(|x| (x - max).exp() / c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const EPSILON: f64 = 1e-6;

    #[test]
    fn returns_values_that_sum_to_one() {
        let x = array![0.3, 2.9, 4.0];

        let out = softmax(x.view());

        assert!((out.sum() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn returns_uniform_distribution_for_equal_elements() {
        let x = array![1.0, 1.0, 1.0, 1.0];

        let out = softmax(x.view());

        for value in out {
            assert!((value - 0.25).abs() < EPSILON);
        }
    }

    #[test]
    fn preserves_order_of_input() {
        let x = array![0.3, 2.9, 4.0];

        let out = softmax(x.view());

        assert!(out[0] < out[1]);
        assert!(out[1] < out[2]);
    }

    #[test]
    fn does_not_overflow_for_large_elements() {
        let x = array![1000.0, 1000.0, 1000.0];

        let out = softmax(x.view());

        for value in out {
            assert!(value.is_finite());
        }
    }
}
