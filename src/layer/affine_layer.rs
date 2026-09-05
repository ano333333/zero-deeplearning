use crate::layer::layer::Layer;
use ndarray::{
    prelude::{Array1, Array2},
    Axis,
};

pub struct AffineLayer<'a> {
    w: &'a Array2<f64>,
    b: &'a Array1<f64>,
    x: Array2<f64>,
    pub dw: Array2<f64>,
    pub db: Array1<f64>,
}

impl<'a> AffineLayer<'a> {
    pub fn new(w: &'a Array2<f64>, b: &'a Array1<f64>) -> Self {
        assert_eq!(w.ncols(), b.len(), "w.ncols() must equal b.len()");
        AffineLayer {
            w,
            b,
            x: Array2::zeros((0, 0)),
            dw: Array2::zeros((0, 0)),
            db: Array1::zeros(0),
        }
    }
}

impl<'a> Layer<Array2<f64>, Array2<f64>> for AffineLayer<'a> {
    fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        assert_eq!(x.ncols(), self.w.nrows(), "x.ncols() must equal w.nrows()");
        self.x = x.clone();
        x.dot(self.w) + self.b
    }
    fn backward(&mut self, dout: &Array2<f64>) -> Array2<f64> {
        assert_eq!(
            dout.nrows(),
            self.x.nrows(),
            "dout.nrows() must equal x.nrows()"
        );
        assert_eq!(
            dout.ncols(),
            self.w.ncols(),
            "dout.ncols() must equal w.ncols()"
        );
        self.dw = self.x.t().dot(dout);
        self.db = dout.sum_axis(Axis(0));
        dout.dot(&self.w.t())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use ndarray::prelude::{Array1, Array2};
    use std::panic::AssertUnwindSafe;

    const H: f64 = 1e-6;
    const EPSILON: f64 = 1e-6;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < EPSILON,
            "expected {expected}, got {actual}"
        );
    }

    fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| {
                payload
                    .downcast_ref::<&str>()
                    .map(|message| (*message).to_owned())
            })
            .expect("panic payload should be a string")
    }

    #[test]
    fn forward_matches_direct_affine_calculation() {
        let x = array![[0.7, -1.2, 2.3]];
        let w = array![[-0.4], [1.5], [0.8]];
        let b = array![-0.6];
        let expected = x.dot(&w) + &b;
        let mut layer = AffineLayer::new(&w, &b);

        let actual = layer.forward(&x);

        assert_eq!(actual, expected);
    }

    #[test]
    fn new_rejects_bias_with_wrong_length() {
        let w = Array2::zeros((2, 3));
        let b = Array1::zeros(2);
        let panic = match std::panic::catch_unwind(|| AffineLayer::new(&w, &b)) {
            Ok(_) => panic!("expected AffineLayer::new to panic"),
            Err(panic) => panic,
        };

        assert!(panic_message(panic).contains("w.ncols() must equal b.len()"));
    }

    #[test]
    fn forward_rejects_input_with_wrong_width() {
        let w = Array2::zeros((2, 3));
        let b = Array1::zeros(3);
        let x = Array2::zeros((4, 1));
        let mut layer = AffineLayer::new(&w, &b);
        let panic = match std::panic::catch_unwind(AssertUnwindSafe(|| layer.forward(&x))) {
            Ok(_) => panic!("expected AffineLayer::forward to panic"),
            Err(panic) => panic,
        };

        assert!(panic_message(panic).contains("x.ncols() must equal w.nrows()"));
    }

    #[test]
    fn backward_rejects_gradient_with_wrong_batch_size() {
        let w = Array2::zeros((2, 3));
        let b = Array1::zeros(3);
        let x = Array2::zeros((4, 2));
        let dout = Array2::zeros((5, 3));
        let mut layer = AffineLayer::new(&w, &b);
        layer.forward(&x);
        let panic = match std::panic::catch_unwind(AssertUnwindSafe(|| layer.backward(&dout))) {
            Ok(_) => panic!("expected AffineLayer::backward to panic"),
            Err(panic) => panic,
        };

        assert!(panic_message(panic).contains("dout.nrows() must equal x.nrows()"));
    }

    #[test]
    fn backward_rejects_gradient_with_wrong_width() {
        let w = Array2::zeros((2, 3));
        let b = Array1::zeros(3);
        let x = Array2::zeros((4, 2));
        let dout = Array2::zeros((4, 2));
        let mut layer = AffineLayer::new(&w, &b);
        layer.forward(&x);
        let panic = match std::panic::catch_unwind(AssertUnwindSafe(|| layer.backward(&dout))) {
            Ok(_) => panic!("expected AffineLayer::backward to panic"),
            Err(panic) => panic,
        };

        assert!(panic_message(panic).contains("dout.ncols() must equal w.ncols()"));
    }

    #[test]
    fn backward_matches_central_difference_for_all_inputs_and_parameters() {
        let x = array![[0.7, -1.2, 2.3]];
        let w = array![[-0.4], [1.5], [0.8]];
        let b = array![-0.6];
        let dout = array![[1.0]];
        let mut layer = AffineLayer::new(&w, &b);

        layer.forward(&x);
        let dx = layer.backward(&dout);

        for (index, _) in x.indexed_iter() {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[index] += H;
            x_minus[index] -= H;

            let plus = AffineLayer::new(&w, &b).forward(&x_plus)[[0, 0]];
            let minus = AffineLayer::new(&w, &b).forward(&x_minus)[[0, 0]];
            assert_close(dx[index], (plus - minus) / (2.0 * H));
        }

        for (index, _) in w.indexed_iter() {
            let mut w_plus = w.clone();
            let mut w_minus = w.clone();
            w_plus[index] += H;
            w_minus[index] -= H;

            let plus = AffineLayer::new(&w_plus, &b).forward(&x)[[0, 0]];
            let minus = AffineLayer::new(&w_minus, &b).forward(&x)[[0, 0]];
            assert_close(layer.dw[index], (plus - minus) / (2.0 * H));
        }

        for (index, _) in b.indexed_iter() {
            let mut b_plus = b.clone();
            let mut b_minus = b.clone();
            b_plus[index] += H;
            b_minus[index] -= H;

            let plus = AffineLayer::new(&w, &b_plus).forward(&x)[[0, 0]];
            let minus = AffineLayer::new(&w, &b_minus).forward(&x)[[0, 0]];
            assert_close(layer.db[index], (plus - minus) / (2.0 * H));
        }
    }
}
