use ndarray::{Array, ArrayView, Dimension};

pub fn sigmoid<D: Dimension>(x: ArrayView<f64, D>) -> Array<f64, D> {
    x.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const EPSILON: f64 = 1e-6;

    #[test]
    fn returns_half_for_zero() {
        let x = array![0.0];

        let out = sigmoid(x.view());

        assert!((out[0] - 0.5).abs() < EPSILON);
    }

    #[test]
    fn is_monotonically_increasing() {
        let x = array![-1.0, -0.5, 0.0, 0.5, 1.0];

        let out = sigmoid(x.view());

        for (left, right) in out.iter().zip(out.iter().skip(1)) {
            assert!(left < right);
        }
    }

    #[test]
    fn returns_values_between_zero_and_one() {
        let x = array![-10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0];

        let out = sigmoid(x.view());

        for value in out {
            assert!(0.0 < value);
            assert!(value < 1.0);
        }
    }
}
