// Showcasing a slightly more complex type.


#[repr(C, packed)]
struct Foo {
    arr: [i32; 3],
    x: f64,
    y: f32,
    res: f64,
}

//#pragma pack(1)
//struct Foo {
//  int arr[3];
//  double x;
//  float y;
//  double res;
//};

#[batch(df, 4, Vector)]
unsafe fn f(foo: *mut i32) {
    let xptr = foo.add(3) as *mut f64;
    let yptr = foo.add(5) as *mut f32;
    let resptr = foo.add(6) as *mut f64;
    let x: f64 = *xptr;
    let y: f32 = *yptr;
    *resptr = x * y;
}

fn main() {
    let foo1: Foo = Foo { [0,0,0], 10.0, 9.0, 0.0 };
    let foo2: Foo = Foo { [0,0,0], 99.0, 7.0, 0.0 };
    let foo3: Foo = Foo { [0,0,0], 1.10, 9.0, 0.0 };
    let foo4: Foo = Foo { [0,0,0], 3.14, 0.1, 0.0 };

    let expected = [90.0, 693.0, 9.9, 0.314};

    df(&foo1.as_ptr() as *mut i32,
       &foo2.as_ptr() as *mut i32,
       &foo3.as_ptr() as *mut i32,
       &foo4.as_ptr() as *mut i32);

    assert_eq!(foo1.res, expected[0]);
    assert_eq!(foo2.res, expected[1]);
    assert_eq!(foo3.res, expected[2]);
    assert_eq!(foo4.res, expected[3]);
}
