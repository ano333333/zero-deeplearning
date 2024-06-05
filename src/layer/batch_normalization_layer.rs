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
        let y = &self.xhat * self.aff[0] + self.aff[1];
        y
    }
    fn backward(&mut self, dout: &Array2<f64>) -> Array2<f64> {
        self.daff[0] = (&self.xhat * dout).sum();
        self.daff[1] = dout.sum();
        let dxhat = dout * self.aff[0];
        let du1 = &dxhat * &self.u2;
        let du2 = (&dxhat * &self.u1).sum_axis(Axis(1));
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
