use ndarray::{Array, ArrayView, Axis, Dimension, RemoveAxis};

pub fn softmax_batch<D: Dimension + RemoveAxis>(x: ArrayView<f64, D>) -> Array<f64, D> {
    let mut res = Array::zeros(x.raw_dim());
    for i in 0..x.raw_dim()[0] {
        let row = x.index_axis(Axis(0), i).to_owned();
        let max = row.fold(-1.0 / 0.0, |acc, &x| x.max(acc));
        let c = row.mapv(|x| (x - max).exp()).sum();
        let row = row.mapv(|x| (x - max).exp() / c);
        res.index_axis_mut(Axis(0), i).assign(&row);
    }
    res
}
