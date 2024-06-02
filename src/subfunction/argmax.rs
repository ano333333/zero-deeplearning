use ndarray::{ArrayView, Dimension, NdIndex};

pub fn argmax<D: Dimension>(x: ArrayView<f64, D>) -> <D as Dimension>::Pattern
where
    <D as ndarray::Dimension>::Pattern: NdIndex<D>,
{
    let mut max_index: Option<<D as Dimension>::Pattern> = None;
    for iter in x.indexed_iter() {
        match max_index {
            None => max_index = Some(iter.0),
            Some(ref index) => {
                if *iter.1 > x[index.clone()] {
                    max_index = Some(iter.0);
                }
            }
        }
    }
    max_index.unwrap()
}
