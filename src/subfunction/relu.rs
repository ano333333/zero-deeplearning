use ndarray::{Array, ArrayView, Dimension};

pub fn relu<D: Dimension>(x: ArrayView<f64, D>) -> Array<f64, D> {
    x.mapv(|x| x.max(0.0))
}
