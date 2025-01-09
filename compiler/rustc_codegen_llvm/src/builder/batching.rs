use std::ptr;

use rustc_ast::expand::batch_attrs::{BatchAttrs, BatchItem, BatchActivity};
use rustc_codegen_ssa::ModuleCodegen;
use rustc_codegen_ssa::back::write::ModuleConfig;
use rustc_codegen_ssa::traits::{BaseTypeCodegenMethods, BuilderMethods};
use rustc_errors::FatalError;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::Lto;
use tracing::{debug, trace};

use crate::back::write::{llvm_err, llvm_optimize};
use crate::builder::Builder;
use crate::declare::declare_raw_fn;
use crate::errors::LlvmError;
use crate::llvm::AttributePlace::Function;
use crate::llvm::{Metadata, True};
use crate::value::Value;
use crate::{CodegenContext, LlvmCodegenBackend, ModuleLlvm, attributes, context, llvm};

fn get_params(fnc: &Value) -> Vec<&Value> {
    unsafe {
        let param_num = llvm::LLVMCountParams(fnc) as usize;
        let mut fnc_args: Vec<&Value> = vec![];
        fnc_args.reserve(param_num);
        llvm::LLVMGetParams(fnc, fnc_args.as_mut_ptr());
        fnc_args.set_len(param_num);
        fnc_args
    }
}

/// When differentiating `fn_to_diff`, take a `outer_fn` and generate another
/// function with expected naming and calling conventions[^1] which will be
/// discovered by the enzyme LLVM pass and its body populated with the differentiated
/// `fn_to_diff`. `outer_fn` is then modified to have a call to the generated
/// function and handle the differences between the Rust calling convention and
/// Enzyme.
/// [^1]: <https://enzyme.mit.edu/getting_started/CallingConvention/>
// FIXME(ZuseZ4): `outer_fn` should include upstream safety checks to
// cover some assumptions of enzyme/batch, which could lead to UB otherwise.
fn generate_enzyme_call<'ll, 'tcx>(
    cx: &context::CodegenCx<'ll, 'tcx>,
    fn_to_diff: &'ll Value,
    outer_fn: &'ll Value,
    attrs: BatchAttrs,
) {
    let inputs = attrs.input_activity;
    let width = attrs.width;
    let mut ad_name: String = "__enzyme_batch".to_string();

    // add outer_fn name to ad_name to make it unique, in case users apply batch to multiple
    // functions. Unwrap will only panic, if LLVM gave us an invalid string.
    let name = llvm::get_value_name(outer_fn);
    let outer_fn_name = std::ffi::CStr::from_bytes_with_nul(name).unwrap().to_str().unwrap();
    ad_name.push_str(outer_fn_name.to_string().as_str());

    // Let us assume the user wrote the following function square:
    //
    // ```llvm
    // define double @square(double %x) {
    // entry:
    //  %0 = fmul double %x, %x
    //  ret double %0
    // }
    // ```
    //
    // The user now applies batching to the function square, in which case fn_to_diff will be `square`.
    // Our macro generates the following placeholder code (slightly simplified):
    //
    // ```llvm
    // define double @dsquare(double %x) {
    //  ; placeholder code
    //  return 0.0;
    // }
    // ```
    //
    // so our `outer_fn` will be `dsquare`. The unsafe code section below now removes the placeholder
    // code and inserts an batching call. We also add a declaration for the __enzyme_batch call.
    // Again, the arguments to all functions are slightly simplified.
    // ```llvm
    // declare double @__enzyme_batch_square(...)
    //
    // define double @dsquare(double %x0, double %x1, double %x2, double %x3) {
    // entry:
    //   %0 = tail call double (...) @__enzyme_batch_square(double (double)* nonnull @square, metadata !"enzyme_width", i64 4,
    //   metadata !"enzyme_vector", double %x0, double %x1, double %x2, double %x3)
    //   ret double %0
    // }
    // ```
    unsafe {
        // On LLVM-IR, we can luckily declare __enzyme_ functions without specifying the input
        // arguments. We do however need to declare them with their correct return type.
        // We already figured the correct return type out in our frontend, when generating the outer_fn,
        // so we can now just go ahead and use that. FIXME(ZuseZ4): This doesn't handle sret yet.
        let fn_ty = llvm::LLVMGlobalGetValueType(outer_fn);
        let ret_ty = llvm::LLVMGetReturnType(fn_ty);

        // LLVM can figure out the input types on it's own, so we take a shortcut here.
        let enzyme_ty = llvm::LLVMFunctionType(ret_ty, ptr::null(), 0, True);

        //FIXME(ZuseZ4): the CC/Addr/Vis values are best effort guesses, we should look at tests and
        // think a bit more about what should go here.
        let cc = llvm::LLVMGetFunctionCallConv(outer_fn);
        let ad_fn = declare_raw_fn(
            cx,
            &ad_name,
            llvm::CallConv::try_from(cc).expect("invalid callconv"),
            llvm::UnnamedAddr::No,
            llvm::Visibility::Default,
            enzyme_ty,
        );

        // Otherwise LLVM might inline our temporary code before the enzyme pass has a chance to
        // do it's work.
        let attr = llvm::AttributeKind::NoInline.create_attr(cx.llcx);
        attributes::apply_to_llfn(ad_fn, Function, &[attr]);

        // first, remove all calls from fnc
        let entry = llvm::LLVMGetFirstBasicBlock(outer_fn);
        let br = llvm::LLVMRustGetTerminator(entry);
        llvm::LLVMRustEraseInstFromParent(br);

        let last_inst = llvm::LLVMRustGetLastInstruction(entry).unwrap();
        let mut builder = Builder::build(cx, entry);

        let num_args = llvm::LLVMCountParams(&fn_to_diff);
        let mut args = Vec::with_capacity(num_args as usize + 1);
        args.push(fn_to_diff);

        let enzyme_const = cx.create_metadata("enzyme_const".to_string()).unwrap();
        let enzyme_vector = cx.create_metadata("enzyme_vector".to_string()).unwrap();
        let enzyme_buffer = cx.create_metadata("enzyme_buffer".to_string()).unwrap();

        trace!("matching batch arguments");
        // We now handle the issue that Rust level arguments not always match the llvm-ir level
        // arguments. A slice, `&[f32]`, for example, is represented as a pointer and a length on
        // llvm-ir level. The number of activities matches the number of Rust level arguments, so we
        // need to match those.
        // FIXME(ZuseZ4): This logic is a bit more complicated than it should be, can we simplify it
        // using iterators and peek()?
        let mut outer_pos: usize = 0;
        let mut activity_pos = 0;
        let outer_args: Vec<&llvm::Value> = get_params(outer_fn);
        while activity_pos < inputs.len() {
            let activity = inputs[activity_pos];
            let (activity, vectorized): (&Metadata, bool) = match activity {
                BatchActivity::Const => (enzyme_const, false),
                BatchActivity::Vector => (enzyme_vector, true),
                BatchActivity::Leaf => (enzyme_buffer, false),
                BatchActivity::FakeActivitySize => (enzyme_const, false),
            };
            let outer_arg = outer_args[outer_pos];
            args.push(cx.get_metadata_value(activity));
            args.push(outer_arg);
            if vectorized {
                // We know that vectorized args by construction have <width-1> following arguments,
                // so this can not be out of bounds.
                let next_outer_arg = outer_args[outer_pos + width - 1];
                let next_outer_ty = cx.val_ty(next_outer_arg);
                // FIXME(ZuseZ4): We should add support for Vec here too, but it's less urgent since
                // vectors behind references (&Vec<T>) are already supported. Users can not pass a
                // Vec by value for reverse mode, so this would only help forward mode batch.
                let slice = {
                    if activity_pos + 1 >= inputs.len() {
                        // If there is no arg following our ptr, it also can't be a slice,
                        // since that would lead to a ptr, int pair.
                        false
                    } else {
                        let next_activity = inputs[activity_pos + 1];
                        // We analyze the MIR types and add this dummy activity if we visit a slice.
                        next_activity == BatchActivity::FakeActivitySize
                    }
                };
                if slice {
                    // A 4x batched slice will have the following two outer_fn arguments:
                    // (..., ptr0, int0, ptr1, int1, ...). We add the following llvm-ir to our __enzyme call:
                    // (..., metadata! enzyme_vector, ptr0, ptr1, ptr2, ptr3, int1, ...).
                    // FIXME(ZuseZ4): We will upstream a safety check later which asserts that
                    // int2 >= int1, which means the shadow args are equally large

                    args.push(cx.get_metadata_value(enzyme_const));
                    // Now we verify that we have width pairs of (ptr/int)
                    for i in 0..width {
                        let next_outer_arg = outer_args[outer_pos + 2 * i];
                        let next_outer_ty = cx.val_ty(next_outer_arg);
                        assert!(llvm::LLVMRustGetTypeKind(next_outer_ty) == llvm::TypeKind::Pointer);
                        let next_outer_arg2 = outer_args[outer_pos + 2 * i + 1];
                        let next_outer_ty2 = cx.val_ty(next_outer_arg2);
                        assert!(llvm::LLVMRustGetTypeKind(next_outer_ty2) == llvm::TypeKind::Integer);
                        args.push(next_outer_arg);
                        args.push(next_outer_arg2);
                    }
                    args.push(cx.get_metadata_value(enzyme_const));
                    args.push(next_outer_arg);
                    outer_pos += 4;
                    activity_pos += 2;
                } else {
                    // A vectorized pointer will have the following two outer_fn arguments:
                    // (..., ptr, ptr, ...). We add the following llvm-ir to our __enzyme call:
                    // (..., metadata! enzyme_dup, ptr, ptr, ...).
                    assert!(llvm::LLVMRustGetTypeKind(next_outer_ty) == llvm::TypeKind::Pointer);
                    args.push(next_outer_arg);
                    outer_pos += 2;
                    activity_pos += 1;
                }
            } else {
                // We do not differentiate with resprect to this argument.
                // We already added the metadata and argument above, so just increase the counters.
                outer_pos += 1;
                activity_pos += 1;
            }
        }

        let call = builder.call(enzyme_ty, None, None, ad_fn, &args, None, None);

        // This part is a bit iffy. LLVM requires that a call to an inlineable function has some
        // metadata attachted to it, but we just created this code oota. Given that the
        // differentiated function already has partly confusing metadata, and given that this
        // affects nothing but the auttodiff IR, we take a shortcut and just steal metadata from the
        // dummy code which we inserted at a higher level.
        // FIXME(ZuseZ4): Work with Enzyme core devs to clarify what debug metadata issues we have,
        // and how to best improve it for enzyme core and rust-enzyme.
        let md_ty = cx.get_md_kind_id("dbg");
        if llvm::LLVMRustHasMetadata(last_inst, md_ty) {
            let md = llvm::LLVMRustDIGetInstMetadata(last_inst)
                .expect("failed to get instruction metadata");
            let md_todiff = cx.get_metadata_value(md);
            llvm::LLVMSetMetadata(call, md_ty, md_todiff);
        } else {
            // We don't panic, since depending on whether we are in debug or release mode, we might
            // have no debug info to copy, which would then be ok.
            trace!("no dbg info");
        }
        // Now that we copied the metadata, get rid of dummy code.
        llvm::LLVMRustEraseInstBefore(entry, last_inst);
        llvm::LLVMRustEraseInstFromParent(last_inst);

        if cx.val_ty(outer_fn) != cx.type_void() {
            builder.ret(call);
        } else {
            builder.ret_void();
        }

        // Let's crash in case that we messed something up above and generated invalid IR.
        llvm::LLVMRustVerifyFunction(
            outer_fn,
            llvm::LLVMRustVerifierFailureAction::LLVMAbortProcessAction,
        );
    }
}

pub(crate) fn batch<'ll, 'tcx>(
    module: &'ll ModuleCodegen<ModuleLlvm>,
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    tcx: TyCtxt<'tcx>,
    batch_items: Vec<BatchItem>,
    config: &ModuleConfig,
) -> Result<(), FatalError> {
    for item in &batch_items {
        trace!("{}", item);
    }

    let diag_handler = cgcx.create_dcx();
    let (_, _, cgus) = tcx.collect_and_partition_mono_items(());
    let cx = context::CodegenCx::new(tcx, &cgus.first().unwrap(), &module.module_llvm);

    // Before dumping the module, we want all the TypeTrees to become part of the module.
    for item in batch_items.iter() {
        let name = item.source.clone();
        let fn_def: Option<&llvm::Value> = cx.get_function(&name);
        let Some(fn_def) = fn_def else {
            return Err(llvm_err(diag_handler.handle(), LlvmError::PrepareBatching {
                src: item.source.clone(),
                target: item.target.clone(),
                error: "could not find source function".to_owned(),
            }));
        };
        debug!(?item.target);
        let fn_target: Option<&llvm::Value> = cx.get_function(&item.target);
        let Some(fn_target) = fn_target else {
            return Err(llvm_err(diag_handler.handle(), LlvmError::PrepareBatching {
                src: item.source.clone(),
                target: item.target.clone(),
                error: "could not find target function".to_owned(),
            }));
        };

        generate_enzyme_call(&cx, fn_def, fn_target, item.attrs.clone());
    }

    // FIXME(ZuseZ4): support SanitizeHWAddress and prevent illegal/unsupported opts

    if let Some(opt_level) = config.opt_level {
        let opt_stage = match cgcx.lto {
            Lto::Fat => llvm::OptStage::PreLinkFatLTO,
            Lto::Thin | Lto::ThinLocal => llvm::OptStage::PreLinkThinLTO,
            _ if cgcx.opts.cg.linker_plugin_lto.enabled() => llvm::OptStage::PreLinkThinLTO,
            _ => llvm::OptStage::PreLinkNoLTO,
        };
        // This is our second opt call, so now we run all opts,
        // to make sure we get the best performance.
        let skip_size_increasing_opts = false;
        trace!("running Module Optimization after differentiation");
        unsafe {
            llvm_optimize(
                cgcx,
                diag_handler.handle(),
                module,
                config,
                opt_level,
                opt_stage,
                skip_size_increasing_opts,
            )?
        };
    }
    trace!("done with differentiate()");

    Ok(())
}
