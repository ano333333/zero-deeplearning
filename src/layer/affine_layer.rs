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
