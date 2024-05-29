use ndarray::prelude::*;
use plotters::{
    backend::RGBPixel,
    coord::{types::RangedCoordf64, Shift},
    prelude::*,
};

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
    let a = array![1, 2, 3, 4];
    println!("a: {:?}", a);
    println!("{:?}", a.ndim());
    println!("{:?}", a.shape());
    println!("{:?}", a.shape()[0]);
    println!("{}", separator());

    let b = array![[1, 2], [3, 4], [4, 5]];
    println!("b: {:?}", b);
    println!("{:?}", b.ndim());
    println!("{:?}", b.shape());
    println!("{}", separator());

    let a = array![[1, 2], [3, 4]];
    println!("{:?}", a.shape());
    let b = array![[5, 6], [7, 8]];
    println!("{:?}", b.shape());
    println!("{:?}", a.dot(&b));
    println!("{}", separator());

    let a = array![[1, 2, 3], [4, 5, 6]];
    println!("{:?}", a.shape());
    let b = array![[1, 2], [3, 4], [5, 6]];
    println!("{:?}", b.shape());
    println!("{:?}", a.dot(&b));
    println!("{}", separator());

    // let c = array![[1, 2], [3, 4]];
    // println!("{:?}", c.shape());
    // let d = array![[1, 2, 3], [4, 5, 6]];
    // println!("{:?}", a.shape());
    // println!("{:?}", a.dot(&c));

    let a = array![[1, 2], [3, 4], [5, 6]];
    println!("{:?}", a.shape());
    let b = array![7, 8];
    println!("{:?}", b.shape());
    println!("{:?}", a.dot(&b));
    println!("{}", separator());

    let x = array![1, 2];
    println!("{:?}", x);
    let w = array![[1, 3, 5], [2, 4, 6]];
    println!("{:?}", w);
    let y = x.dot(&w);
    println!("{:?}", y);
}
