pub trait Layer<In, Out, Grad> {
    fn forward(&mut self, x: In) -> Out;
    fn backward(&mut self, dout: Out) -> Grad;
}
