use ndarray::{Array, ArrayView, Dimension};

pub fn relu<D: Dimension>(x: ArrayView<f64, D>) -> Array<f64, D> {
    x.mapv(|x| x.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn returns_x_for_positive_elements_and_zero_for_others() {
        let x = array![-2.0, -0.1, 0.0, 0.1, 2.0];

        let actual = relu(x.view());

        assert_eq!(actual, array![0.0, 0.0, 0.0, 0.1, 2.0]);
    }
}
