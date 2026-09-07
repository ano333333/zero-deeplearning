use ndarray::{ArrayView, Dimension, NdIndex};

/// `x` の中で最大値を持つ要素のインデックスを返す。
///
/// 最大値が複数存在する場合は、最初に見つかったもののインデックスを返す。
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn returns_index_of_max_element_1d() {
        let x = array![1.0, 3.0, 2.0];

        let index = argmax(x.view());

        assert_eq!(index, 1);
    }

    #[test]
    fn returns_first_index_when_max_value_is_duplicated() {
        let x = array![1.0, 3.0, 3.0, 2.0];

        let index = argmax(x.view());

        assert_eq!(index, 1);
    }

    #[test]
    fn returns_index_of_max_element_2d() {
        let x = array![[1.0, 5.0], [4.0, 2.0]];

        let index = argmax(x.view());

        assert_eq!(index, (0, 1));
    }
}
