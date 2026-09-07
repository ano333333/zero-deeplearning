use ndarray::{Array, ArrayView, Dimension};

pub fn step_function<D: Dimension>(x: ArrayView<f64, D>) -> Array<f64, D> {
    x.mapv(|x| (x > 0.0) as i32 as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn returns_one_for_positive_and_zero_for_others() {
        let x = array![-2.0, -0.1, 0.0, 0.1, 2.0];

        let actual = step_function(x.view());

        assert_eq!(actual, array![0.0, 0.0, 0.0, 1.0, 1.0]);
    }
}
