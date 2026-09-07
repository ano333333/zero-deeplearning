/// 順伝播と逆伝播を持つ計算グラフの1層を表すトレイト。
pub trait Layer<In, Out> {
    /// 入力 `x` を受け取り、この層の出力を返す。内部で `backward` に必要な状態を保持する。
    fn forward(&mut self, x: &In) -> Out;
    /// 出力側から伝わってきた勾配 `dout` を受け取り、入力側への勾配を返す。
    fn backward(&mut self, dout: &Out) -> In;
}
