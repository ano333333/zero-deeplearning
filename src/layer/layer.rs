pub trait Layer<In, Out> {
    fn forward(&mut self, x: &In) -> Out;
    fn backward(&mut self, dout: &Out) -> In;
}
