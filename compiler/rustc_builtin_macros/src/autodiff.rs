#![allow(unused_imports)]
#![allow(unused_variables)]
//use crate::util::check_builtin_macro_attribute;
//use crate::util::check_autodiff;

use std::string::String;
use crate::errors;
use rustc_ast::ptr::P;
use rustc_ast::{BindingAnnotation, ByRef};
use rustc_ast::{self as ast, FnHeader, FnSig, Generics, StmtKind, NestedMetaItem, MetaItemKind};
use rustc_ast::{Fn, ItemKind, Stmt, TyKind, Unsafe, PatKind};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{kw, sym, Ident};
use rustc_span::Span;
use thin_vec::{thin_vec, ThinVec};
use rustc_span::Symbol;
use rustc_ast::expand::autodiff_attrs::{AutoDiffAttrs, DiffActivity, DiffMode};

pub fn expand(
    ecx: &mut ExtCtxt<'_>,
    expand_span: Span,
    meta_item: &ast::MetaItem,
    item: Annotatable,
) -> Vec<Annotatable> {
    //check_builtin_macro_attribute(ecx, meta_item, sym::alloc_error_handler);

    let meta_item_vec: ThinVec<NestedMetaItem> = match meta_item.kind {
        ast::MetaItemKind::List(ref vec) => vec.clone(),
        _ => {
            ecx.sess
                .dcx()
                .emit_err(errors::AutoDiffInvalidApplication { span: item.span() });
            return vec![item];
        }
    };
    let orig_item: P<ast::Item> = item.clone().expect_item();
    let primal = orig_item.ident.clone();

    // Allow using `#[autodiff(...)]` only on a Fn
    let (fn_item, has_ret, sig, sig_span) = if let Annotatable::Item(item) = &item
        && let ItemKind::Fn(box ast::Fn { sig, .. }) = &item.kind
    {
        (item, sig.decl.output.has_ret(), sig, ecx.with_call_site_ctxt(sig.span))
    } else {
        ecx.sess
            .dcx()
            .emit_err(errors::AutoDiffInvalidApplication { span: item.span() });
        return vec![item];
    };
    let x: AutoDiffAttrs = AutoDiffAttrs::from_ast(&meta_item_vec, has_ret);
    dbg!(&x);
    //let span = ecx.with_def_site_ctxt(sig_span);
    let span = ecx.with_def_site_ctxt(expand_span);
    //let span = ecx.with_def_site_ctxt(fn_item.span);

    let (d_sig, old_names, new_args) = gen_enzyme_decl(ecx, &sig, &x, span, sig_span);
    let new_decl_span = d_sig.span;
    //let d_body = gen_enzyme_body(ecx, primal, &old_names, &new_args, span, span);
    let d_body = gen_enzyme_body(ecx, primal, &old_names, &new_args, span, sig_span, new_decl_span);
    let d_ident = meta_item_vec[0].meta_item().unwrap().path.segments[0].ident;

    // The first element of it is the name of the function to be generated
    let asdf = ItemKind::Fn(Box::new(ast::Fn {
        defaultness: ast::Defaultness::Final,
        sig: d_sig,
        generics: Generics::default(),
        body: Some(d_body),
    }));
    let d_fn = ecx.item(span, d_ident, rustc_ast::AttrVec::default(), asdf);

    let orig_annotatable = Annotatable::Item(orig_item.clone());
    let d_annotatable = Annotatable::Item(d_fn);
    return vec![orig_annotatable, d_annotatable];
}

// shadow arguments must be mutable references or ptrs, because Enzyme will write into them.
fn assure_mut_ref(ty: &ast::Ty) -> ast::Ty {
    let mut ty = ty.clone();
    match ty.kind {
        TyKind::Ptr(ref mut mut_ty) => {
            mut_ty.mutbl = ast::Mutability::Mut;
        }
        TyKind::Ref(_, ref mut mut_ty) => {
            mut_ty.mutbl = ast::Mutability::Mut;
        }
        _ => {
            panic!("unsupported type: {:?}", ty);
        }
    }
    ty
}


// The body of our generated functions will consist of three black_Box calls.
// The first will call the primal function with the original arguments.
// The second will just take the shadow arguments.
// The third will (unsafely) call std::mem::zeroed(), to match the return type of the new function
// (whatever that might be). This way we surpress rustc from optimizing anyt argument away.
fn gen_enzyme_body(ecx: &ExtCtxt<'_>, primal: Ident, old_names: &[String], new_names: &[String], span: Span, sig_span: Span, new_decl_span: Span) -> P<ast::Block> {
    let blackbox_path = ecx.std_path(&[Symbol::intern("hint"), Symbol::intern("black_box")]);
    let zeroed_path = ecx.std_path(&[Symbol::intern("mem"), Symbol::intern("zeroed")]);
    let empty_loop_block = ecx.block(span, ThinVec::new());
    let loop_expr = ecx.expr_loop(span, empty_loop_block);


    let primal_call_expr = ecx.expr_path(ecx.path_ident(span, primal));
    let blackbox_call_expr = ecx.expr_path(ecx.path(span, blackbox_path));
    let zeroed_call_expr = ecx.expr_path(ecx.path(span, zeroed_path));

    let mem_zeroed_call: Stmt = ecx.stmt_expr(ecx.expr_call(
        span,
        zeroed_call_expr.clone(),
        thin_vec![],
    ));
    let unsafe_block_with_zeroed_call: P<ast::Expr> = ecx.expr_block(P(ast::Block {
        stmts: thin_vec![mem_zeroed_call],
        id: ast::DUMMY_NODE_ID,
        rules: ast::BlockCheckMode::Unsafe(ast::UserProvided),
        span: sig_span,
        tokens: None,
        could_be_bare_literal: false,
    }));
    // create ::core::hint::black_box(array(arr));
    let primal_call = ecx.expr_call(
        new_decl_span,
        primal_call_expr,
        old_names.iter().map(|name| {
            ecx.expr_path(ecx.path_ident(new_decl_span, Ident::from_str(name)))
        }).collect(),
    );
    let black_box0 = ecx.expr_call(
        new_decl_span,
        blackbox_call_expr.clone(),
        thin_vec![primal_call.clone()],
    );

    // create ::core::hint::black_box(grad_arr, tang_y));
    let black_box1 = ecx.expr_call(
        sig_span,
        blackbox_call_expr.clone(),
        new_names.iter().map(|arg| {
            ecx.expr_path(ecx.path_ident(span, Ident::from_str(arg)))
        }).collect(),
    );

    // create ::core::hint::black_box(unsafe { ::core::mem::zeroed() })
    let black_box2 = ecx.expr_call(
        sig_span,
        blackbox_call_expr.clone(),
        thin_vec![unsafe_block_with_zeroed_call.clone()],
    );

    let mut body = ecx.block(span, ThinVec::new());
    //body.stmts.push(ecx.stmt_expr(primal_call));
    //body.stmts.push(ecx.stmt_expr(black_box0));
    //body.stmts.push(ecx.stmt_expr(black_box1));
    body.stmts.push(ecx.stmt_expr(black_box2));
    body.stmts.push(ecx.stmt_expr(loop_expr));
    body
}

// Generate the new function declaration. Const arguments are kept as is. Duplicated arguments must
// be pointers or references. Those receive a shadow argument, which is a mutable reference/pointer.
// Active arguments must be scalars. Their shadow argument is added to the return type (and will be
// zero-initialized by Enzyme). Active arguments are not handled yet.
// Each argument of the primal function (and the return type if existing) must be annotated with an
// activity.
fn gen_enzyme_decl(_ecx: &ExtCtxt<'_>, sig: &ast::FnSig, x: &AutoDiffAttrs, span: Span, _sig_span: Span)
        -> (ast::FnSig, Vec<String>, Vec<String>) {
    let decl: P<ast::FnDecl> = sig.decl.clone();
    assert!(decl.inputs.len() == x.input_activity.len());
    assert!(decl.output.has_ret() == x.has_ret_activity());
    let mut d_decl = decl.clone();
    let mut d_inputs = Vec::new();
    let mut new_inputs = Vec::new();
    let mut old_names = Vec::new();
    for (arg, activity) in decl.inputs.iter().zip(x.input_activity.iter()) {
        //dbg!(&arg);
        d_inputs.push(arg.clone());
        match activity {
            DiffActivity::Duplicated => {
                let mut shadow_arg = arg.clone();
                shadow_arg.ty = P(assure_mut_ref(&arg.ty));
                // adjust name depending on mode
                let old_name = if let PatKind::Ident(_, ident, _) = shadow_arg.pat.kind {
                    ident.name
                } else {
                    dbg!(&shadow_arg.pat);
                    panic!("not an ident?");
                };
                old_names.push(old_name.to_string());
                let name: String = match x.mode {
                    DiffMode::Reverse => format!("d{}", old_name),
                    DiffMode::Forward => format!("b{}", old_name),
                    _ => panic!("unsupported mode: {}", old_name),
                };
                dbg!(&name);
                new_inputs.push(name.clone());
                shadow_arg.pat = P(ast::Pat {
                    // TODO: Check id
                    id: ast::DUMMY_NODE_ID,
                    kind: PatKind::Ident(BindingAnnotation::NONE,
                        Ident::from_str_and_span(&name, shadow_arg.pat.span),
                        None,
                    ),
                    span: shadow_arg.pat.span,
                    tokens: shadow_arg.pat.tokens.clone(),
                });

                d_inputs.push(shadow_arg);
            }
            _ => {},
        }
    }
    d_decl.inputs = d_inputs.into();
    let d_sig = FnSig {
        header: sig.header.clone(),
        decl: d_decl,
        span,
    };
    (d_sig, old_names, new_inputs)
}
