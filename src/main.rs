use ndarray::prelude::*;

fn main() {
    let x = array![1.0, 2.0, 3.0];
    let y = array![2.0, 4.0, 6.0];
    println!("a");
    println!("{:?}", x + y);
    let x = array![1.0, 2.0, 3.0];
    let y = array![2.0, 4.0, 6.0];
    println!("{:?}", x - y);
    let x = array![1.0, 2.0, 3.0];
    let y = array![2.0, 4.0, 6.0];
    println!("{:?}", x * y);
    let x = array![1.0, 2.0, 3.0];
    let y = array![2.0, 4.0, 6.0];
    println!("{:?}", x / y);

    let mut x = array![1.0, 2.0, 3.0];
    x /= 2.0;
    println!("{:?}", x);

    let a = array![[1, 2], [3, 4]];
    println!("{:?}", a);
    println!("{:?}", a.shape());

    let b = array![[3, 0], [0, 6]];
    println!("{:?}", a + b);
    let a = array![[1, 2], [3, 4]];
    let b = array![[3, 0], [0, 6]];
    println!("{:?}", a * b);
    let a = array![[1, 2], [3, 4]];
    println!("{:?}", a * 10);

    let a = array![[1, 2], [3, 4]];
    let b = array![10, 20];
    println!("{:?}", a * b);

    let x = array![[51, 55], [14, 19], [0, 4]];
    println!("{:?}", x);
    println!("{:?}", x.slice(s![0, ..]));
    println!("{:?}", x[[0, 1]]);

    for row in x.outer_iter() {
        println!("{:?}", row);
    }

    let x = Array::from_iter(x.iter());
    println!("{:?}", x);
}
