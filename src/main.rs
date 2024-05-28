use plotters::prelude::*;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("images/0.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let x_min = 0f32;
    let x_max = 6.0f32;
    let y_min = -1.0f32;
    let y_max = 1.0f32;
    let x_iter = (0..=1200).map(|x| x as f32 / 200.0);
    let mut chart = ChartBuilder::on(&root)
        .caption("y=sin(x)&cos(x)", ("Arial", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
    chart.configure_mesh().draw()?;
    let sin_iter = x_iter.clone().map(|x| x.sin());
    let x_sin_iter = x_iter.zip(sin_iter);
    chart
        .draw_series(LineSeries::new(x_sin_iter, &RED))?
        .label("y = sin(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    let x_iter = (0..=1200).map(|x| x as f32 / 200.0);
    let cos_iter = x_iter.clone().map(|x| x.cos());
    let x_cos_iter = x_iter.zip(cos_iter);
    chart
        .draw_series(LineSeries::new(x_cos_iter, &BLUE))?
        .label("y = cos(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    root.present()?;

    Ok(())
}
