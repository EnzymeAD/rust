// We want a batch size of 4.
// The original function processes 2 elements a 64 bit, so for our vfoo we have an offset of 16 bytes.
// Both vfoo and bfoo return [f64; 4].

#[batch(vfoo, 4, Leaf(16))]
#[batch(bfoo, 4, Batch)]
fn foo(x: &[f64]) -> f64 {
    assert!(x.len() == 2);
    x.iter().map(|&x| x * x).sum()
}

fn main() {
    // 8 elements
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let x2 = vec![1.0, 2.0];
    let x3 = vec![3.0, 4.0];
    let x4 = vec![5.0, 6.0];
    let x5 = vec![7.0, 8.0];

    let mut res1 = [0.0;4];
    for i in 0..4 {
        res1[i] = foo(&x1[i..i + 1]);
    }

    let res2: [f64; 4] = bfoo(&x2, &x3, &x4, &x5);

    let res3: [f64; 4] = vfoo(&x1);

    for i in 0..4 {
        assert_eq!(res1[i], res2[i]);
        assert_eq!(res1[i], res3[i]);
    }
}
