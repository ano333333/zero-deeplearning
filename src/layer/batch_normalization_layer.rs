use crate::layer::layer::Layer;
use ndarray::{Array1, Array2, Axis};
pub struct BatchNormalizationLayer<'a> {
    aff: &'a Array1<f64>, // [gamma, beta]
    pub daff: Array1<f64>,
    input_size: usize,
    xhat: Array2<f64>,
    u9: Array2<f64>,
    u8: Array2<f64>,
    u7: Array2<f64>,
    u6: Array2<f64>,
    u5: Array2<f64>,
    u4: Array2<f64>,
    u3: Array2<f64>,
    u2: Array2<f64>,
    u1: Array2<f64>,
}

impl<'a> BatchNormalizationLayer<'a> {
    pub fn new(input_size: usize, aff: &'a Array1<f64>) -> Self {
        Self {
            aff,
            daff: Array1::zeros(2),
            input_size,
            xhat: Array2::zeros((0, 0)),
            u9: Array2::zeros((0, 0)),
            u8: Array2::zeros((0, 0)),
            u7: Array2::zeros((0, 0)),
            u6: Array2::zeros((0, 0)),
            u5: Array2::zeros((0, 0)),
            u4: Array2::zeros((0, 0)),
            u3: Array2::zeros((0, 0)),
            u2: Array2::zeros((0, 0)),
            u1: Array2::zeros((0, 0)),
        }
    }
}

impl<'a> Layer<Array2<f64>, Array2<f64>> for BatchNormalizationLayer<'a> {
    fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        self.u9 = x.clone();
        self.u8 = (x.sum_axis(Axis(1)) / self.input_size as f64).insert_axis(Axis(1));
        self.u7 = x.clone();
        self.u6 = x - &self.u8;
        self.u5 = self.u6.mapv(|v| v * v);
        self.u4 = (self.u5.sum_axis(Axis(1)) / self.input_size as f64).insert_axis(Axis(1));
        self.u3 = self.u4.mapv(|v| (v + 1.0e-7).sqrt());
        self.u2 = 1.0 / &self.u3;
        self.u1 = self.u6.clone();
        self.xhat = &self.u6 * &self.u2;
        &self.xhat * self.aff[0] + self.aff[1]
    }
    fn backward(&mut self, dout: &Array2<f64>) -> Array2<f64> {
        self.daff[0] = (&self.xhat * dout).sum();
        self.daff[1] = dout.sum();
        let dxhat = dout * self.aff[0];
        let du1 = &dxhat * &self.u2;
        let du3 = -&self.u2 * &self.u2 * dout;
        let du4 = &du3 / (2.0 * &self.u3);
        let du5 = Array2::from_elem(self.u5.raw_dim(), 1.0) * &du4;
        let du6 = 2.0 * &self.u6 * &du5;
        let du7 = &du1 + &du6;
        let du8 = -&du7;
        let du9 = Array2::from_elem(self.u9.raw_dim(), 1.0) * &du8;
        &du7 + &du9
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const H: f64 = 1e-6;
    const EPSILON: f64 = 1e-4;

    #[test]
    fn forward_normalizes_each_row_before_scale_and_shift() {
        let aff = array![1.0, 0.0];
        let x = array![[1.0, 2.0, 3.0, 4.0], [10.0, 0.0, -5.0, 15.0]];
        let mut layer = BatchNormalizationLayer::new(x.ncols(), &aff);

        let out = layer.forward(&x);

        for row in out.rows() {
            let mean = row.sum() / row.len() as f64;
            let variance = row.mapv(|v| (v - mean).powi(2)).sum() / row.len() as f64;
            assert!(mean.abs() < EPSILON, "expected mean close to 0, got {mean}");
            assert!(
                (variance - 1.0).abs() < EPSILON,
                "expected variance close to 1, got {variance}"
            );
        }
    }

    #[test]
    fn forward_scales_and_shifts_each_row_by_gamma_and_beta() {
        let aff = array![2.0, 3.0];
        let x = array![[1.0, 2.0, 3.0, 4.0], [10.0, 0.0, -5.0, 15.0]];
        let mut layer = BatchNormalizationLayer::new(x.ncols(), &aff);

        let out = layer.forward(&x);

        for row in out.rows() {
            let mean = row.sum() / row.len() as f64;
            let variance = row.mapv(|v| (v - mean).powi(2)).sum() / row.len() as f64;
            assert!(
                (mean - aff[1]).abs() < EPSILON,
                "expected mean close to {}, got {mean}",
                aff[1]
            );
            assert!(
                (variance - aff[0] * aff[0]).abs() < EPSILON,
                "expected variance close to {}, got {variance}",
                aff[0] * aff[0]
            );
        }
    }

    #[test]
    fn backward_matches_central_difference_for_input() {
        let aff = array![1.5, -0.5];
        let x = array![[1.0, 2.0, 3.0, 4.0], [10.0, 0.0, -5.0, 15.0]];
        let dout = Array2::ones(x.raw_dim());
        let mut layer = BatchNormalizationLayer::new(x.ncols(), &aff);

        layer.forward(&x);
        let dx = layer.backward(&dout);

        for index in ndarray::indices(x.raw_dim()) {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[index] += H;
            x_minus[index] -= H;

            let plus = BatchNormalizationLayer::new(x.ncols(), &aff)
                .forward(&x_plus)
                .sum();
            let minus = BatchNormalizationLayer::new(x.ncols(), &aff)
                .forward(&x_minus)
                .sum();
            let numerical_gradient = (plus - minus) / (2.0 * H);

            assert!(
                (dx[index] - numerical_gradient).abs() < EPSILON,
                "gradient mismatch at {index:?}: backward={}, numerical={numerical_gradient}",
                dx[index]
            );
        }
    }

    #[test]
    fn backward_matches_central_difference_for_aff() {
        let aff = array![1.5, -0.5];
        let x = array![[1.0, 2.0, 3.0, 4.0], [10.0, 0.0, -5.0, 15.0]];
        let dout = Array2::ones(x.raw_dim());
        let mut layer = BatchNormalizationLayer::new(x.ncols(), &aff);

        layer.forward(&x);
        layer.backward(&dout);

        for (index, _) in aff.indexed_iter() {
            let mut aff_plus = aff.clone();
            let mut aff_minus = aff.clone();
            aff_plus[index] += H;
            aff_minus[index] -= H;

            let plus = BatchNormalizationLayer::new(x.ncols(), &aff_plus)
                .forward(&x)
                .sum();
            let minus = BatchNormalizationLayer::new(x.ncols(), &aff_minus)
                .forward(&x)
                .sum();
            let numerical_gradient = (plus - minus) / (2.0 * H);

            assert!(
                (layer.daff[index] - numerical_gradient).abs() < EPSILON,
                "gradient mismatch at aff[{index}]: backward={}, numerical={numerical_gradient}",
                layer.daff[index]
            );
        }
    }
}
