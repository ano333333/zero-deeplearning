use ndarray::{Array, Dimension};

use super::optimize::Optimize;

pub struct AdaGrad<D: Dimension> {
    learning_rate: f64,
    h: Array<f64, D>,
}

impl<D: Dimension> AdaGrad<D> {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            h: Array::zeros(D::default()),
        }
    }
}

impl<D: Dimension> Optimize<D> for AdaGrad<D> {
    fn update(&mut self, w: &mut Array<f64, D>, grad: &Array<f64, D>) {
        // 初めての呼び出し時はhのdimが(0,0,...)なので、wの形に揃える
        if self.h.len() == 0 {
            self.h = Array::zeros(w.raw_dim());
        }
        let h = &self.h + &(grad * grad);
        let h_sqrt = h.map(|x| x.sqrt() + 1e-7);
        *w -= &(grad * self.learning_rate / h_sqrt);
        self.h = h;
    }
}
