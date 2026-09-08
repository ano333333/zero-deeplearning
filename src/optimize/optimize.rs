use ndarray::{Array, Dimension};

pub trait Optimize<D: Dimension> {
    fn update(&mut self, w: &mut Array<f64, D>, grad: &Array<f64, D>);
}

/// `Optimize` の生成器。学習率など次元によらない母数のみを保持し、
/// 更新対象パラメータの次元 `dim` を受け取ってから状態を持つ `Optimize` を作る。
///
/// `Factory` 自体は特定の次元に紐付かない。`create` を呼ぶ際に次元 `D` を
/// 指定することで、同じ `Factory` から `Ix1` 用・`Ix2` 用など異なる次元の
/// `Optimize` をいずれも生成できる。
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use zero_deeplearning::optimize::optimize::{Optimize, OptimizeFactory};
/// use zero_deeplearning::optimize::sgd::SGDFactory;
///
/// let mut w = array![1.0, 2.0];
/// let grad = array![0.1, 0.2];
///
/// // 母数(learning_rate)のみを持つFactoryを作る。
/// let factory = SGDFactory::new(0.1);
/// // 更新対象パラメータwの次元を渡してOptimizeを生成する。
/// let mut sgd = factory.create(w.raw_dim());
/// sgd.update(&mut w, &grad);
/// ```
pub trait OptimizeFactory {
    type Optimize<D: Dimension>: Optimize<D>;
    fn create<D: Dimension>(&self, dim: D) -> Self::Optimize<D>;
}
