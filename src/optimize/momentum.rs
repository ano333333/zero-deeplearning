use ndarray::{Array, Dimension};

use super::optimize::Optimize;

pub struct Momentum<D: Dimension> {
    learning_rate: f64,
    momentum: f64,
    v: Array<f64, D>,
}

impl<D: Dimension> Momentum<D> {
    pub fn new(learning_rate: f64, momentum: f64) -> Self {
        Self {
            learning_rate,
            momentum,
            v: Array::zeros(D::default()),
        }
    }
}

impl<D: Dimension> Optimize<D> for Momentum<D> {
    fn update(&mut self, w: &mut Array<f64, D>, grad: &Array<f64, D>) {
        // 初めての呼び出し時vのdimは(0,0,...)なので、wの形に揃える
        if self.v.len() == 0 {
            self.v = Array::zeros(w.raw_dim());
        }
        let v = self.momentum * &self.v - self.learning_rate * grad;
        *w += &v;
        self.v = v;
    }
}
