use crate::llvm;
use rustc_ast::expand::typetree::{Kind, TypeTree};

// Here we take a vector of tt and return a vec.
// This simplifies cases like fat-ptr, where we need to return two types.
#[allow(unused_variables)]
pub fn to_enzyme_typetree(
    forest: &[TypeTree],
    llvm_data_layout: &str,
    llcx: &llvm::Context,
) -> Vec<llvm::TypeTree> {
    //let mut ret = Vec::new();
    let ret = Vec::new();
    let mut offset = 0;
    for (_i, tree) in forest.iter().enumerate() {
        let _len = tree.0.len();
        let mut obj = llvm::TypeTree::new();
        for (_j, x) in tree.0.iter().enumerate() {
            let mut extra_int = false;
            let scalar = match x.kind {
                Kind::Integer => llvm::CConcreteType::DT_Integer,
                Kind::Float => llvm::CConcreteType::DT_Float,
                Kind::Double => llvm::CConcreteType::DT_Double,
                Kind::FatPointer => { extra_int = true; llvm::CConcreteType::DT_Pointer },
                Kind::Pointer => llvm::CConcreteType::DT_Pointer,
                _ => panic!("Unknown kind {:?}", x.kind),
            };
            let tt = llvm::TypeTree::from_type(scalar, llcx).only(-1);

            let tt = if !x.child.0.is_empty() {
                let inner_tt = to_enzyme_typetree(&[x.child.clone()], llvm_data_layout, llcx);
                assert!(inner_tt.len() == 1);
                let inner = if extra_int {
                    inner_tt[0].clone().only(0)
                } else {
                    inner_tt[0].clone().only(-1)
                };
                tt.merge(inner)
            } else {
                tt
            };

            obj = if extra_int {
                offset += 1;
                let int = llvm::CConcreteType::DT_Integer;
                let int_tt = llvm::TypeTree::from_type(int, llcx).only(0);
                let int_tt = int_tt.shift(llvm_data_layout, 0, 0, 8);
                obj.merge(int_tt)
            } else {
                obj
            };

            if x.offset != -1 {
                obj = obj.merge(tt.shift(llvm_data_layout, 0, x.size as isize, x.offset as usize));
            } else {
                obj = obj.merge(tt);
            }
        }
    }
    ret
}

#[cfg(test)]
mod tests;
