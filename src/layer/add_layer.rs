use crate::layer::layer::Layer;

pub struct AddLayer {
    x: f64,
    y: f64,
}

impl AddLayer {
    pub fn new() -> Self {
        AddLayer { x: 0.0, y: 0.0 }
    }
}

impl Layer<(f64, f64), f64, (f64, f64)> for AddLayer {
    fn forward(&mut self, x: (f64, f64)) -> f64 {
        self.x = x.0;
        self.y = x.1;
        x.0 + x.1
    }
    fn backward(&mut self, dout: f64) -> (f64, f64) {
        (dout, dout)
    }
}
