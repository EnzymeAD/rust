// Problem: The user might want to pass either [f64; 4], (f64, f64, f64, f64), or S to the
// function. All of these are valid (modulo we have to force the user to set the right repr).
// Our current design doesn't allow users to specify those, so we will want at least one iteration.
// However, for the sake of similarity to the current autodiff (where we'd also want a change),
// leave it as is.

struct _S {
    x1: f64,
    x2: f64,
    x3: f64,
    x4: f64,
}

#[batch(bsquare4, 4, Const, Leaf(8))]
#[batch(vsquare4, 4, Const, Vector)]
fn square(multiplier: f64, x: f64) -> f64 {
    x * x * multiplier
}

fn main() {
    let vals = [23.1, 10.0, 100.0, 3.14];
    let expected = [square(3.14, vals[0]), square(3.14, vals[1]), square(3.14, vals[2]), square(3.14, vals[3])];
    let result1 = bsquare4(3.14, vals[0], vals[1], vals[2], vals[3]);
    let result2 = vsquare4(3.14, vals);
    assert_eq!(result.x1, expected[0]);
    assert_eq!(result.x2, expected[1]);
    assert_eq!(result.x3, expected[2]);
    assert_eq!(result.x4, expected[3]);
    assert_eq!(result2.x1, expected[0]);
    assert_eq!(result2.x2, expected[1]);
    assert_eq!(result2.x3, expected[2]);
    assert_eq!(result2.x4, expected[3]);
}
