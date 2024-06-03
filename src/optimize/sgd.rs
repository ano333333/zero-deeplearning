use std::marker::PhantomData;

use ndarray::{Array, Dimension};

use super::optimize::Optimize;

pub struct SGD<D: Dimension> {
    learning_rate: f64,
    d: PhantomData<D>,
}

impl<D: Dimension> SGD<D> {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            d: PhantomData,
        }
    }
}

impl<D: Dimension> Optimize<D> for SGD<D> {
    fn update(&mut self, w: &mut Array<f64, D>, grad: &Array<f64, D>) {
        *w -= &(grad * self.learning_rate);
    }
}
