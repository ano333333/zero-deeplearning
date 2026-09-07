use ndarray::ArrayView2;

pub fn cross_entropy_error(y: ArrayView2<f64>, t: ArrayView2<f64>) -> f64 {
    let delta = 1e-7;
    let batch_size = y.raw_dim()[0];
    let log_y = y.mapv(|y| (y + delta).ln());
    -(log_y * t).sum() / batch_size as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const EPSILON: f64 = 1e-6;

    #[test]
    fn returns_zero_when_prediction_perfectly_matches_target() {
        let y = array![[1.0, 0.0, 0.0]];
        let t = array![[1.0, 0.0, 0.0]];

        let error = cross_entropy_error(y.view(), t.view());

        assert!(error.abs() < EPSILON);
    }

    #[test]
    fn returns_larger_error_for_less_confident_prediction() {
        let t = array![[1.0, 0.0, 0.0]];
        let confident = array![[0.8, 0.1, 0.1]];
        let unconfident = array![[0.4, 0.3, 0.3]];

        let confident_error = cross_entropy_error(confident.view(), t.view());
        let unconfident_error = cross_entropy_error(unconfident.view(), t.view());

        assert!(confident_error < unconfident_error);
    }

    #[test]
    fn averages_error_over_batch() {
        let y = array![[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let t = array![[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let single_y = array![[1.0, 0.0, 0.0]];
        let single_t = array![[1.0, 0.0, 0.0]];

        let batch_error = cross_entropy_error(y.view(), t.view());
        let single_error = cross_entropy_error(single_y.view(), single_t.view());

        assert!((batch_error - single_error).abs() < EPSILON);
    }
}
