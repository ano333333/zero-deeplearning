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
        self.x = x.clone();
        x.dot(self.w) + self.b
    }
    fn backward(&mut self, dout: &Array2<f64>) -> Array2<f64> {
        self.dw = self.x.t().dot(dout);
        self.db = dout.sum_axis(Axis(0));
        dout.dot(&self.w.t())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const H: f64 = 1e-6;
    const EPSILON: f64 = 1e-6;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < EPSILON,
            "expected {expected}, got {actual}"
        );
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
