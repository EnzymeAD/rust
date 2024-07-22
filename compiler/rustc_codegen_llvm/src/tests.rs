use super::*;

use crate::llvm::LLVMRustContextCreate;
//pub fn LLVMRustContextCreate(shouldDiscardNames: bool) -> &'static mut Context;
use rustc_ast::expand::typetree::{Kind, Type, TypeTree};

// (&[f32], i32) will be represented as:
// (ptr, int, int)
#[test]
fn test_to_enzyme_typetree() {
    let llcx = unsafe {LLVMRustContextCreate(false) };
    let child = TypeTree(vec![
        Type {
            offset: -1,
            size: 4,
            kind: Kind::Float,
            child: TypeTree::new(),
        },
    ]);
    let forest = vec![
        TypeTree(vec![
            Type {
                offset: -1,
                size: 8,
                kind: Kind::FatPointer,
                child,
            },
        ]),
        TypeTree(vec![
            Type {
                offset: -1,
                size: 8,
                kind: Kind::Integer,
                child: TypeTree::new(),
            },
        ]),
    ];
    let llvm_data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128";
    let ret = to_enzyme_typetree(&forest, llvm_data_layout, &llcx);
    assert_eq!(ret.len(), 3);
}
