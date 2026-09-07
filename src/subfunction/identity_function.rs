use ndarray::{Array, ArrayView, Dimension};

/// `x` をそのまま複製して返す恒等関数。
pub fn identity_function<D: Dimension>(x: ArrayView<f64, D>) -> Array<f64, D> {
    x.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn returns_same_values_as_input() {
        let x = array![[0.5, -1.0, 2.0], [4.0, -5.0, 10.0]];

        let actual = identity_function(x.view());

        assert_eq!(actual, x);
    }
}
