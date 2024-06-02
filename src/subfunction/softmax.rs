use ndarray::{Array, ArrayView, Dimension};

pub fn softmax<D: Dimension>(x: ArrayView<f64, D>) -> Array<f64, D> {
    let max = x.fold(-1.0 / 0.0, |acc, &x| x.max(acc));
    let c = x.mapv(|x| (x - max).exp()).sum();
    x.mapv(|x| (x - max).exp() / c)
}
