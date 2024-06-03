use ndarray::{Array, Dimension};

pub trait Optimize<D: Dimension> {
    fn update(&mut self, w: &mut Array<f64, D>, grad: &Array<f64, D>);
}
