//! This module contains the implementation of the `#[batch]` attribute.
//! Currently our linter isn't smart enough to see that each import is used in one of the two
//! configs (batch enabled or disabled), so we have to add cfg's to each import.
//! FIXME(ZuseZ4): Remove this once we have a smarter linter.

//#[cfg(llvm_enzyme)]
mod llvm_enzyme {
    use std::str::FromStr;
    use std::string::String;

    use rustc_ast::expand::batch_attrs::{
        BatchAttrs, BatchActivity, BatchMode,
    };
    use rustc_ast::ptr::P;
    use rustc_ast::token::{Token, TokenKind};
    use rustc_ast::tokenstream::*;
    use rustc_ast::visit::AssocCtxt::*;
    use rustc_ast::{
        self as ast, AssocItemKind, BindingMode, FnRetTy, FnSig, Generics, ItemKind, MetaItemInner,
        PatKind, TyKind,
    };
    use rustc_expand::base::{Annotatable, ExtCtxt};
    use rustc_span::{Ident, Span, Symbol, kw, sym};
    use thin_vec::{ThinVec, thin_vec};
    use tracing::{debug, trace};

    use crate::errors;

    // If we have a default `()` return type or explicitley `()` return type,
    // then we often can skip doing some work.
    fn has_ret(ty: &FnRetTy) -> bool {
        match ty {
            FnRetTy::Ty(ty) => !ty.kind.is_unit(),
            FnRetTy::Default(_) => false,
        }
    }
    fn first_ident(x: &MetaItemInner) -> rustc_span::Ident {
        let segments = &x.meta_item().unwrap().path.segments;
        assert!(segments.len() == 1);
        segments[0].ident
    }

    fn width(x: &MetaItemInner) -> Option<usize> {
        first_ident(x).name.as_str().parse().ok()
    }

    fn name(x: &MetaItemInner) -> String {
        first_ident(x).name.to_string()
    }

    pub(crate) fn from_ast(
        ecx: &mut ExtCtxt<'_>,
        meta_item: &ThinVec<MetaItemInner>,
    ) -> BatchAttrs {
        let dcx = ecx.sess.dcx();

        let width = width(&meta_item[1]);
        let Some(width) = width else {
            dcx.emit_err(errors::BatchInvalidMode { span: meta_item[1].span(), arg: name(&meta_item[1]) });
            return BatchAttrs::error();
        };

        let mut input_activity: Vec<BatchActivity> = vec![];
        let mut errors = false;
        for x in &meta_item[2..] {
            let activity_str = name(&x);
            let res = BatchActivity::from_str(&activity_str);
            match res {
                Ok(x) => input_activity.push(x),
                Err(_) => {
                    dcx.emit_err(errors::BatchUnknownActivity {
                        span: x.span(),
                        act: activity_str,
                    });
                    errors = true;
                }
            };
        }
        if errors {
            return BatchAttrs::error();
        }

        let mode = BatchMode::Source;
        BatchAttrs { mode, width, input_activity }
    }

    /// We expand the batch macro to generate a new placeholder function which passes
    /// type-checking and can be called by users. The function body of the placeholder function will
    /// later be replaced on LLVM-IR level, so the design of the body is less important and for now
    /// should just prevent early inlining and optimizations which alter the function signature.
    /// The exact signature of the generated function depends on the configuration provided by the
    /// user, but here is an example:
    ///
    /// ```
    /// #[batch(vsin, 4, Vector)]
    /// fn sin(x: &Box<f32>) -> f32 {
    ///     f32::sin(**x)
    /// }
    /// ```
    /// which becomes expanded to:
    /// ```
    /// #[rustc_batch]
    /// #[inline(never)]
    /// fn sin(x: &Box<f32>) -> f32 {
    ///     f32::sin(**x)
    /// }
    /// #[rustc_batch(Reverse, 4, Vector)]
    /// #[inline(never)]
    /// fn vsin(x: &Box<f32>, x2: &Box<f32>, x3: &Box<f32>, x4: &Box<f32>) -> [f32;4] {
    ///     unsafe {
    ///         asm!("NOP");
    ///     };
    ///     ::core::hint::black_box(sin(x));
    ///     ::core::hint::black_box((dx, dret));
    ///     ::core::hint::black_box(Default::default())
    /// }
    /// ```
    /// FIXME(ZuseZ4): Once batch is enabled by default, make this a doc comment which is checked
    /// in CI.
    pub(crate) fn expand(
        ecx: &mut ExtCtxt<'_>,
        expand_span: Span,
        meta_item: &ast::MetaItem,
        mut item: Annotatable,
    ) -> Vec<Annotatable> {
        let dcx = ecx.sess.dcx();
        // first get the annotable item:
        let (sig, is_impl): (FnSig, bool) = match &item {
            Annotatable::Item(ref iitem) => {
                let sig = match &iitem.kind {
                    ItemKind::Fn(box ast::Fn { sig, .. }) => sig,
                    _ => {
                        dcx.emit_err(errors::BatchInvalidApplication { span: item.span() });
                        return vec![item];
                    }
                };
                (sig.clone(), false)
            }
            Annotatable::AssocItem(ref assoc_item, _) => {
                let sig = match &assoc_item.kind {
                    ast::AssocItemKind::Fn(box ast::Fn { sig, .. }) => sig,
                    _ => {
                        dcx.emit_err(errors::BatchInvalidApplication { span: item.span() });
                        return vec![item];
                    }
                };
                (sig.clone(), true)
            }
            _ => {
                dcx.emit_err(errors::BatchInvalidApplication { span: item.span() });
                return vec![item];
            }
        };

        let meta_item_vec: ThinVec<MetaItemInner> = match meta_item.kind {
            ast::MetaItemKind::List(ref vec) => vec.clone(),
            _ => {
                dcx.emit_err(errors::BatchInvalidApplication { span: item.span() });
                return vec![item];
            }
        };

        let sig_span = ecx.with_call_site_ctxt(sig.span);

        let (vis, primal) = match &item {
            Annotatable::Item(ref iitem) => (iitem.vis.clone(), iitem.ident.clone()),
            Annotatable::AssocItem(ref assoc_item, _) => {
                (assoc_item.vis.clone(), assoc_item.ident.clone())
            }
            _ => {
                dcx.emit_err(errors::BatchInvalidApplication { span: item.span() });
                return vec![item];
            }
        };

        // create TokenStream from vec elemtents:
        // meta_item doesn't have a .tokens field
        let comma: Token = Token::new(TokenKind::Comma, Span::default());
        let mut ts: Vec<TokenTree> = vec![];
        if meta_item_vec.len() < 2 {
            // At the bare minimum, we need a fnc name and a width, even for a dummy function with no
            // input and output args.
            dcx.emit_err(errors::BatchMissingConfig { span: item.span() });
            return vec![item];
        } else {
            for t in meta_item_vec.clone()[1..].iter() {
                let val = first_ident(t);
                let t = Token::from_ast_ident(val);
                ts.push(TokenTree::Token(t, Spacing::Joint));
                ts.push(TokenTree::Token(comma.clone(), Spacing::Alone));
            }
        }
        let ts: TokenStream = TokenStream::from_iter(ts);

        let x: BatchAttrs = from_ast(ecx, &meta_item_vec);
        if x.mode == BatchMode::Error {
            // We encountered an error, so we return the original item.
            // This allows us to potentially parse other attributes.
            return vec![item];
        }
        let span = ecx.with_def_site_ctxt(expand_span);

        let (d_sig, new_args, idents, errored) = gen_enzyme_decl(ecx, &sig, &x, span);


        let new_decl_span = d_sig.span;
        let d_body = gen_enzyme_body(
            ecx,
            &x,
            &sig,
            &d_sig,
            primal,
            &new_args,
            span,
            sig_span,
            new_decl_span,
            idents,
            errored,
        );
        let d_ident = first_ident(&meta_item_vec[0]);

        // The first element of it is the name of the function to be generated
        let asdf = Box::new(ast::Fn {
            defaultness: ast::Defaultness::Final,
            sig: d_sig,
            generics: Generics::default(),
            body: Some(d_body),
        });
        let mut rustc_batch_attr =
            P(ast::NormalAttr::from_ident(Ident::with_dummy_span(sym::rustc_batch)));

        let ts2: Vec<TokenTree> = vec![TokenTree::Token(
            Token::new(TokenKind::Ident(sym::never, false.into()), span),
            Spacing::Joint,
        )];
        let never_arg = ast::DelimArgs {
            dspan: ast::tokenstream::DelimSpan::from_single(span),
            delim: ast::token::Delimiter::Parenthesis,
            tokens: ast::tokenstream::TokenStream::from_iter(ts2),
        };
        let inline_item = ast::AttrItem {
            unsafety: ast::Safety::Default,
            path: ast::Path::from_ident(Ident::with_dummy_span(sym::inline)),
            args: ast::AttrArgs::Delimited(never_arg),
            tokens: None,
        };
        let inline_never_attr = P(ast::NormalAttr { item: inline_item, tokens: None });
        let new_id = ecx.sess.psess.attr_id_generator.mk_attr_id();
        let attr: ast::Attribute = ast::Attribute {
            kind: ast::AttrKind::Normal(rustc_batch_attr.clone()),
            id: new_id,
            style: ast::AttrStyle::Outer,
            span,
        };
        let new_id = ecx.sess.psess.attr_id_generator.mk_attr_id();
        let inline_never: ast::Attribute = ast::Attribute {
            kind: ast::AttrKind::Normal(inline_never_attr),
            id: new_id,
            style: ast::AttrStyle::Outer,
            span,
        };

        // Don't add it multiple times:
        let orig_annotatable: Annotatable = match item {
            Annotatable::Item(ref mut iitem) => {
                if !iitem.attrs.iter().any(|a| a.id == attr.id) {
                    iitem.attrs.push(attr.clone());
                }
                if !iitem.attrs.iter().any(|a| a.id == inline_never.id) {
                    iitem.attrs.push(inline_never.clone());
                }
                Annotatable::Item(iitem.clone())
            }
            Annotatable::AssocItem(ref mut assoc_item, i @ Impl) => {
                if !assoc_item.attrs.iter().any(|a| a.id == attr.id) {
                    assoc_item.attrs.push(attr.clone());
                }
                if !assoc_item.attrs.iter().any(|a| a.id == inline_never.id) {
                    assoc_item.attrs.push(inline_never.clone());
                }
                Annotatable::AssocItem(assoc_item.clone(), i)
            }
            _ => {
                unreachable!("annotatable kind checked previously")
            }
        };
        // Now update for d_fn
        rustc_batch_attr.item.args = rustc_ast::AttrArgs::Delimited(rustc_ast::DelimArgs {
            dspan: DelimSpan::dummy(),
            delim: rustc_ast::token::Delimiter::Parenthesis,
            tokens: ts,
        });
        let d_attr: ast::Attribute = ast::Attribute {
            kind: ast::AttrKind::Normal(rustc_batch_attr.clone()),
            id: new_id,
            style: ast::AttrStyle::Outer,
            span,
        };

        let d_annotatable = if is_impl {
            let assoc_item: AssocItemKind = ast::AssocItemKind::Fn(asdf);
            let d_fn = P(ast::AssocItem {
                attrs: thin_vec![d_attr.clone(), inline_never],
                id: ast::DUMMY_NODE_ID,
                span,
                vis,
                ident: d_ident,
                kind: assoc_item,
                tokens: None,
            });
            Annotatable::AssocItem(d_fn, Impl)
        } else {
            let mut d_fn = ecx.item(
                span,
                d_ident,
                thin_vec![d_attr.clone(), inline_never],
                ItemKind::Fn(asdf),
            );
            d_fn.vis = vis;
            Annotatable::Item(d_fn)
        };

        return vec![orig_annotatable, d_annotatable];
    }

    /// We only want this function to type-check, since we will replace the body
    /// later on llvm level. Using `loop {}` does not cover all return types anymore,
    /// so instead we build something that should pass. We also add a inline_asm
    /// line, as one more barrier for rustc to prevent inlining of this function.
    /// FIXME(ZuseZ4): We still have cases of incorrect inlining across modules, see
    /// <https://github.com/EnzymeAD/rust/issues/173>, so this isn't sufficient.
    /// It also triggers an Enzyme crash if we due to a bug ever try to differentiate
    /// this function (which should never happen, since it is only a placeholder).
    /// Finally, we also add back_box usages of all input arguments, to prevent rustc
    /// from optimizing any arguments away.
    fn gen_enzyme_body(
        ecx: &ExtCtxt<'_>,
        _x: &BatchAttrs,
        _sig: &ast::FnSig,
        d_sig: &ast::FnSig,
        primal: Ident,
        new_names: &[String],
        span: Span,
        sig_span: Span,
        new_decl_span: Span,
        idents: Vec<Ident>,
        errored: bool,
    ) -> P<ast::Block> {
        let blackbox_path = ecx.std_path(&[sym::hint, sym::black_box]);
        let noop = ast::InlineAsm {
            asm_macro: ast::AsmMacro::Asm,
            template: vec![ast::InlineAsmTemplatePiece::String("NOP".into())],
            template_strs: Box::new([]),
            operands: vec![],
            clobber_abis: vec![],
            options: ast::InlineAsmOptions::PURE | ast::InlineAsmOptions::NOMEM,
            line_spans: vec![],
        };
        let noop_expr = ecx.expr_asm(span, P(noop));
        let unsf = ast::BlockCheckMode::Unsafe(ast::UnsafeSource::CompilerGenerated);
        let unsf_block = ast::Block {
            stmts: thin_vec![ecx.stmt_semi(noop_expr)],
            id: ast::DUMMY_NODE_ID,
            tokens: None,
            rules: unsf,
            span,
            could_be_bare_literal: false,
        };
        let unsf_expr = ecx.expr_block(P(unsf_block));
        let blackbox_call_expr = ecx.expr_path(ecx.path(span, blackbox_path));
        let primal_call = gen_primal_call(ecx, span, primal, idents);
        let black_box_primal_call =
            ecx.expr_call(new_decl_span, blackbox_call_expr.clone(), thin_vec![
                primal_call.clone()
            ]);
        let tup_args = new_names
            .iter()
            .map(|arg| ecx.expr_path(ecx.path_ident(span, Ident::from_str(arg))))
            .collect();

        let black_box_remaining_args =
            ecx.expr_call(sig_span, blackbox_call_expr.clone(), thin_vec![
                ecx.expr_tuple(sig_span, tup_args)
            ]);

        let mut body = ecx.block(span, ThinVec::new());
        body.stmts.push(ecx.stmt_semi(unsf_expr));

        // This uses primal args which won't be available if we errored before
        if !errored {
            body.stmts.push(ecx.stmt_semi(black_box_primal_call.clone()));
        }
        body.stmts.push(ecx.stmt_semi(black_box_remaining_args));

        if !has_ret(&d_sig.decl.output) {
            // there is no return type that we have to match, () works fine.
            return body;
        }

        // TODO: this only calls default(), we want Default::default()
        let sl: Vec<Symbol> = vec![kw::Default];
        //let sl: Vec<Symbol> = vec![arg, kw::Default];
        let tmp = ecx.def_site_path(&sl);
        let default_call_expr = ecx.expr_path(ecx.path(span, tmp));
        let default_call_expr = ecx.expr_call(new_decl_span, default_call_expr, thin_vec![]);
        body.stmts.push(ecx.stmt_expr(default_call_expr));

        body
    }

    fn gen_primal_call(
        ecx: &ExtCtxt<'_>,
        span: Span,
        primal: Ident,
        idents: Vec<Ident>,
    ) -> P<ast::Expr> {
        let has_self = idents.len() > 0 && idents[0].name == kw::SelfLower;
        if has_self {
            let args: ThinVec<_> =
                idents[1..].iter().map(|arg| ecx.expr_path(ecx.path_ident(span, *arg))).collect();
            let self_expr = ecx.expr_self(span);
            ecx.expr_method_call(span, self_expr, primal, args.clone())
        } else {
            let args: ThinVec<_> =
                idents.iter().map(|arg| ecx.expr_path(ecx.path_ident(span, *arg))).collect();
            let primal_call_expr = ecx.expr_path(ecx.path_ident(span, primal));
            ecx.expr_call(span, primal_call_expr, args)
        }
    }

    // Generate the new function declaration. Const arguments are kept as is. Duplicated arguments must
    // be pointers or references. Those receive a shadow argument, which is a mutable reference/pointer.
    // Active arguments must be scalars. Their shadow argument is added to the return type (and will be
    // zero-initialized by Enzyme).
    // Each argument of the primal function (and the return type if existing) must be annotated with an
    // activity.
    //
    // Error handling: If the user provides an invalid configuration (incorrect numbers, types, or
    // both), we emit an error and return the original signature. This allows us to continue parsing.
    fn gen_enzyme_decl(
        ecx: &ExtCtxt<'_>,
        sig: &ast::FnSig,
        x: &BatchAttrs,
        span: Span,
    ) -> (ast::FnSig, Vec<String>, Vec<Ident>, bool) {
        let dcx = ecx.sess.dcx();
        let has_ret = has_ret(&sig.decl.output);
        let sig_args = sig.decl.inputs.len() + if has_ret { 1 } else { 0 };
        let num_activities = x.input_activity.len();
        if sig_args != num_activities {
            dcx.emit_err(errors::BatchInvalidNumberActivities {
                span,
                expected: sig_args,
                found: num_activities,
            });
            // This is not the right signature, but we can continue parsing.
            return (sig.clone(), vec![], vec![], true);
        }
        assert!(sig.decl.inputs.len() == x.input_activity.len());
        let mut d_decl = sig.decl.clone();
        let mut d_inputs = Vec::new();
        let mut new_inputs = Vec::new();
        let mut idents = Vec::new();

        let unsafe_activities = x
            .input_activity
            .iter()
            .any(|&act| matches!(act, BatchActivity::Leaf));

        for (arg, activity) in sig.decl.inputs.iter().zip(x.input_activity.iter()) {
            match activity {
                BatchActivity::Vector => {
                    // We now have N instead of 1 argument. Name the new args with _1, _2, ...
                    let mut shadow_arg = arg.clone();
                    let old_name = if let PatKind::Ident(_, ident, _) = arg.pat.kind {
                        ident.name
                    } else {
                        debug!("{:#?}", &arg.pat);
                        panic!("not an ident?");
                    };
                    for i in 0..x.width {
                        let name: String = format!("{}_{}", old_name, i.to_string());
                        new_inputs.push(name.clone());
                        let ident = Ident::from_str_and_span(&name, shadow_arg.pat.span);
                        shadow_arg.pat = P(ast::Pat {
                            id: ast::DUMMY_NODE_ID,
                            kind: PatKind::Ident(BindingMode::NONE, ident, None),
                            span: shadow_arg.pat.span,
                            tokens: shadow_arg.pat.tokens.clone(),
                        });
                        d_inputs.push(shadow_arg.clone());
                    }
                }
                BatchActivity::Leaf => {
                    // Now our argument is width times larger.
                    // So &[T] stays the same.
                    // Vec<T> too.
                    // [T; N] becomes [T; N * width]
                    // T becomes [T; width]
                    // For now we assume we have a slice or vec.
                    d_inputs.push(arg.clone());
                }
                BatchActivity::Const => {
                    // Nothing special to do here.
                    d_inputs.push(arg.clone());
                }
            }
            if let PatKind::Ident(_, ident, _) = arg.pat.kind {
                idents.push(ident.clone());
            } else {
                panic!("not an ident?");
            }
        }

        d_decl.inputs = d_inputs.into();


        // If we return a value of type T, we will now return [T; width] instead of T.
        d_decl.output = match sig.decl.output {
            FnRetTy::Ty(ref ty) => {
                let kind = TyKind::Tup(thin_vec![ty.clone(); x.width]);
                FnRetTy::Ty(P(rustc_ast::Ty { kind, id: ty.id, span: ty.span, tokens: None }))
            }
            FnRetTy::Default(span) => FnRetTy::Default(span),
        };

        let mut d_header = sig.header.clone();
        if unsafe_activities {
            d_header.safety = rustc_ast::Safety::Unsafe(span);
        }
        let d_sig = FnSig { header: d_header, decl: d_decl, span };
        trace!("Generated signature: {:?}", d_sig);
        (d_sig, new_inputs, idents, false)
    }
}

#[cfg(not(llvm_enzyme))]
mod batch_fallback {
    use rustc_ast::ast;
    use rustc_expand::base::{Annotatable, ExtCtxt};
    use rustc_span::Span;

    use crate::errors;
    pub(crate) fn expand(
        ecx: &mut ExtCtxt<'_>,
        _expand_span: Span,
        meta_item: &ast::MetaItem,
        item: Annotatable,
    ) -> Vec<Annotatable> {
        ecx.sess.dcx().emit_err(errors::BatchSupportNotBuild { span: meta_item.span });
        return vec![item];
    }
}

#[cfg(not(llvm_enzyme))]
pub(crate) use batch_fallback::expand;
#[cfg(llvm_enzyme)]
pub(crate) use llvm_enzyme::expand;
