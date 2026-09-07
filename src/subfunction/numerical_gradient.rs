use ndarray::{Array, ArrayView, Dimension, NdIndex};

pub fn numerical_gradient<D: Dimension>(
    f: &dyn Fn(ArrayView<f64, D>) -> f64,
    x: ArrayView<f64, D>,
) -> Array<f64, D>
where
    <D as ndarray::Dimension>::Pattern: NdIndex<D>,
{
    // let h = 1e-4;
    let h = 1.0;
    let mut grad = Array::<f64, D>::zeros(x.raw_dim());
    let mut x_mut = x.to_owned();
    for iter in x.indexed_iter() {
        println!("{:?}", iter);
        x_mut[iter.0.clone()] = iter.1 + h;
        let fxh1 = f(x_mut.view());
        x_mut[iter.0.clone()] = iter.1 - h;
        let fxh2 = f(x_mut.view());
        grad[iter.0.clone()] = (fxh1 - fxh2) / (2.0 * h);
        println!("fxh1: {}, fxh2: {}", fxh1, fxh2);
        println!("grad[iter.0.clone()]: {}", grad[iter.0.clone()]);
        x_mut[iter.0.clone()] = *iter.1;
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const EPSILON: f64 = 1e-4;

    #[test]
    fn matches_analytical_gradient_of_sum_of_squares() {
        let x = array![1.0, 2.0, -3.0];
        let f = |x: ArrayView<f64, _>| x.iter().map(|value| value * value).sum();

        let grad = numerical_gradient(&f, x.view());
        let expected = x.mapv(|value| 2.0 * value);

        for (actual, expected) in grad.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < EPSILON);
        }
    }

    #[test]
    fn returns_zero_gradient_for_constant_function() {
        let x = array![1.0, 2.0, -3.0];
        let f = |_: ArrayView<f64, _>| 5.0;

        let grad = numerical_gradient(&f, x.view());

        for value in grad {
            assert!(value.abs() < EPSILON);
        }
    }
}
