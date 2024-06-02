use ndarray::ArrayView2;

pub fn cross_entropy_error(y: ArrayView2<f64>, t: ArrayView2<f64>) -> f64 {
    let delta = 1e-7;
    let batch_size = y.raw_dim()[0];
    let log_y = y.mapv(|y| (y + delta).ln());
    -(log_y * t).sum() / batch_size as f64
}
