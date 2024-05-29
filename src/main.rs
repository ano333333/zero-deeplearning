use ndarray::prelude::*;
use plotters::prelude::*;

fn separator() -> String {
    (0..20).map(|_| "-").collect::<String>()
}

fn step_function<D: Dimension>(x: ArrayView<f64, D>) -> Array<f64, D> {
    x.mapv(|x| (x > 0.0) as i32 as f64)
}

fn sigmoid<D: Dimension>(x: ArrayView<f64, D>) -> Array<f64, D> {
    x.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}

fn main() {
    let root = BitMapBackend::new("images/step_function.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let x_vec = (-1000..1000)
        .map(|x| x as f64 / 100.0)
        .collect::<Vec<f64>>();
    let x_array_1: Array1<f64> = Array::from_vec(x_vec.clone());
    let y_1: Array1<f64> = step_function(x_array_1.view());
    let y_1_vec = y_1.to_vec();
    let x_vec_y_1_vec_iter = x_vec.clone().into_iter().zip(y_1_vec.into_iter());
    let x_array_2 = Array::from_vec(x_vec.clone());
    let y_2: Array1<f64> = sigmoid(x_array_2.view());
    let y_2_vec = y_2.to_vec();
    let x_vec_y_2_vec_iter = x_vec.clone().into_iter().zip(y_2_vec.into_iter());
    let mut chart = ChartBuilder::on(&root)
        .caption("Step Function", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(-10.0..10.0, -0.5..2.0)
        .unwrap();
    chart.configure_mesh().draw().unwrap();
    chart
        .draw_series(LineSeries::new(x_vec_y_1_vec_iter, &RED))
        .unwrap();
    chart
        .draw_series(LineSeries::new(x_vec_y_2_vec_iter, &BLUE))
        .unwrap();
    root.present().unwrap();
}
