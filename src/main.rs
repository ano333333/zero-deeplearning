use ndarray::prelude::*;
fn separator() -> String {
    return (0..20)
        .map(|_| String::from("-"))
        .collect::<Vec<String>>()
        .concat();
}

fn and(x1: f64, x2: f64) -> i32 {
    let x = array![x1, x2];
    let w = array![0.5, 0.5];
    let b = -0.7;
    let y = (x * w).sum() + b;
    return if y <= 0.0 { 0 } else { 1 };
}

fn nand(x1: f64, x2: f64) -> i32 {
    let x = array![x1, x2];
    let w = array![-0.5, -0.5];
    let b = 0.7;
    let y = (x * w).sum() + b;
    return if y <= 0.0 { 0 } else { 1 };
}

fn or(x1: f64, x2: f64) -> i32 {
    let x = array![x1, x2];
    let w = array![0.5, 0.5];
    let b = -0.2;
    let y = (x * w).sum() + b;
    return if y <= 0.0 { 0 } else { 1 };
}

fn main() {
    println!("AND(0, 0) = {}", and(0.0, 0.0));
    println!("AND(1, 0) = {}", and(1.0, 0.0));
    println!("AND(0, 1) = {}", and(0.0, 1.0));
    println!("AND(1, 1) = {}", and(1.0, 1.0));
    println!("{}", separator());
    println!("NAND(0, 0) = {}", nand(0.0, 0.0));
    println!("NAND(1, 0) = {}", nand(1.0, 0.0));
    println!("NAND(0, 1) = {}", nand(0.0, 1.0));
    println!("NAND(1, 1) = {}", nand(1.0, 1.0));
    println!("{}", separator());
    println!("OR(0, 0) = {}", or(0.0, 0.0));
    println!("OR(1, 0) = {}", or(1.0, 0.0));
    println!("OR(0, 1) = {}", or(0.0, 1.0));
    println!("OR(1, 1) = {}", or(1.0, 1.0));
}
