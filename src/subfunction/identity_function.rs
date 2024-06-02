use ndarray::{Array, ArrayView, Dimension};

pub fn identity_function<D: Dimension>(x: ArrayView<f64, D>) -> Array<f64, D> {
    x.to_owned()
}
