//! The Rust compiler.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(bool_to_option)]
#![feature(crate_visibility_modifier)]
#![feature(let_chains)]
#![feature(let_else)]
#![feature(extern_types)]
#![feature(once_cell)]
#![feature(nll)]
#![feature(iter_intersperse)]
#![recursion_limit = "256"]
#![allow(rustc::potential_query_instability)]

#[macro_use]
extern crate rustc_macros;

use back::write::{create_informational_target_machine, create_target_machine};

use llvm::TypeTree;
pub use llvm_util::target_features;
use rustc_ast::expand::allocator::AllocatorKind;
use rustc_codegen_ssa::back::lto::{LtoModuleCodegen, SerializedModule, ThinModule};
use rustc_codegen_ssa::back::write::{
    CodegenContext, FatLTOInput, ModuleConfig, TargetMachineFactoryConfig, TargetMachineFactoryFn,
};
use rustc_codegen_ssa::traits::*;
use rustc_codegen_ssa::ModuleCodegen;
use rustc_codegen_ssa::{CodegenResults, CompiledModule};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{ErrorGuaranteed, FatalError, Handler};
use rustc_metadata::EncodedMetadata;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{TyCtxt, Ty, ParamEnvAnd, ParamEnv, Param, Adt};
use rustc_middle::middle::autodiff_attrs::AutoDiffItem;
use rustc_session::config::{OptLevel, OutputFilenames, PrintRequest};
use rustc_session::Session;
use rustc_span::symbol::Symbol;
use rustc_target::abi::FieldsShape;

use std::any::Any;
use std::ffi::CStr;

mod back {
    pub mod archive;
    pub mod lto;
    mod profiling;
    pub mod write;
}

mod abi;
mod allocator;
mod asm;
mod attributes;
mod base;
mod builder;
mod callee;
mod common;
mod consts;
mod context;
mod coverageinfo;
mod debuginfo;
mod declare;
mod intrinsic;

// The following is a work around that replaces `pub mod llvm;` and that fixes issue 53912.
#[path = "llvm/mod.rs"]
mod llvm_;
pub mod llvm {
    pub use super::llvm_::*;
}

mod llvm_util;
mod mono_item;
mod type_;
mod type_of;
mod va_arg;
mod value;

#[derive(Clone)]
pub struct LlvmCodegenBackend(());

struct TimeTraceProfiler {
    enabled: bool,
}

impl TimeTraceProfiler {
    fn new(enabled: bool) -> Self {
        if enabled {
            unsafe { llvm::LLVMTimeTraceProfilerInitialize() }
        }
        TimeTraceProfiler { enabled }
    }
}

impl Drop for TimeTraceProfiler {
    fn drop(&mut self) {
        if self.enabled {
            unsafe { llvm::LLVMTimeTraceProfilerFinishThread() }
        }
    }
}

impl ExtraBackendMethods for LlvmCodegenBackend {
    fn new_metadata(&self, tcx: TyCtxt<'_>, mod_name: &str) -> ModuleLlvm {
        ModuleLlvm::new_metadata(tcx, mod_name)
    }

    fn codegen_allocator<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        module_llvm: &mut ModuleLlvm,
        module_name: &str,
        kind: AllocatorKind,
        has_alloc_error_handler: bool,
        ) {
        unsafe { allocator::codegen(tcx, module_llvm, module_name, kind, has_alloc_error_handler) }
    }
    fn compile_codegen_unit(
        &self,
        tcx: TyCtxt<'_>,
        cgu_name: Symbol,
        ) -> (ModuleCodegen<ModuleLlvm>, u64) {
        base::compile_codegen_unit(tcx, cgu_name)
    }
    fn target_machine_factory(
        &self,
        sess: &Session,
        optlvl: OptLevel,
        target_features: &[String],
        ) -> TargetMachineFactoryFn<Self> {
        back::write::target_machine_factory(sess, optlvl, target_features)
    }
    fn target_cpu<'b>(&self, sess: &'b Session) -> &'b str {
        llvm_util::target_cpu(sess)
    }
    fn tune_cpu<'b>(&self, sess: &'b Session) -> Option<&'b str> {
        llvm_util::tune_cpu(sess)
    }

    fn spawn_thread<F, T>(time_trace: bool, f: F) -> std::thread::JoinHandle<T>
        where
            F: FnOnce() -> T,
            F: Send + 'static,
            T: Send + 'static,
            {
                std::thread::spawn(move || {
                    let _profiler = TimeTraceProfiler::new(time_trace);
                    f()
                })
            }

    fn spawn_named_thread<F, T>(
        time_trace: bool,
        name: String,
        f: F,
        ) -> std::io::Result<std::thread::JoinHandle<T>>
        where
            F: FnOnce() -> T,
            F: Send + 'static,
            T: Send + 'static,
            {
                std::thread::Builder::new().name(name).spawn(move || {
                    let _profiler = TimeTraceProfiler::new(time_trace);
                    f()
                })
            }
}

impl WriteBackendMethods for LlvmCodegenBackend {
    type Module = ModuleLlvm;
    type ModuleBuffer = back::lto::ModuleBuffer;
    type Context = llvm::Context;
    type TargetMachine = &'static mut llvm::TargetMachine;
    type ThinData = back::lto::ThinData;
    type ThinBuffer = back::lto::ThinBuffer;
    type TypeTree = DiffTypeTree;

    fn print_pass_timings(&self) {
        unsafe {
            llvm::LLVMRustPrintPassTimings();
        }
    }
    fn run_link(
        cgcx: &CodegenContext<Self>,
        diag_handler: &Handler,
        modules: Vec<ModuleCodegen<Self::Module>>,
        ) -> Result<ModuleCodegen<Self::Module>, FatalError> {
        back::write::link(cgcx, diag_handler, modules)
    }
    fn run_fat_lto(
        cgcx: &CodegenContext<Self>,
        modules: Vec<FatLTOInput<Self>>,
        cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>,
        ) -> Result<LtoModuleCodegen<Self>, FatalError> {
        back::lto::run_fat(cgcx, modules, cached_modules)
    }
    fn run_thin_lto(
        cgcx: &CodegenContext<Self>,
        modules: Vec<(String, Self::ThinBuffer)>,
        cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>,
        ) -> Result<(Vec<LtoModuleCodegen<Self>>, Vec<WorkProduct>), FatalError> {
        back::lto::run_thin(cgcx, modules, cached_modules)
    }
    unsafe fn optimize(
        cgcx: &CodegenContext<Self>,
        diag_handler: &Handler,
        module: &ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
        ) -> Result<(), FatalError> {
        back::write::optimize(cgcx, diag_handler, module, config)
    }
    unsafe fn optimize_thin(
        cgcx: &CodegenContext<Self>,
        thin: &mut ThinModule<Self>,
        ) -> Result<ModuleCodegen<Self::Module>, FatalError> {
        back::lto::optimize_thin_module(thin, cgcx)
    }
    unsafe fn codegen(
        cgcx: &CodegenContext<Self>,
        diag_handler: &Handler,
        module: ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
        ) -> Result<CompiledModule, FatalError> {
        back::write::codegen(cgcx, diag_handler, module, config)
    }
    fn prepare_thin(module: ModuleCodegen<Self::Module>) -> (String, Self::ThinBuffer) {
        back::lto::prepare_thin(module)
    }
    fn serialize_module(module: ModuleCodegen<Self::Module>) -> (String, Self::ModuleBuffer) {
        (module.name, back::lto::ModuleBuffer::new(module.module_llvm.llmod()))
    }
    fn run_lto_pass_manager(
        cgcx: &CodegenContext<Self>,
        module: &ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
        thin: bool,
        ) -> Result<(), FatalError> {
        let diag_handler = cgcx.create_diag_handler();
        back::lto::run_pass_manager(cgcx, &diag_handler, module, config, thin)
    }
    /// Generate autodiff rules
    fn autodiff(
        cgcx: &CodegenContext<Self>,
        module: &ModuleCodegen<Self::Module>,
        diff_fncs: Vec<AutoDiffItem>,
        typetrees: FxHashMap<String, Self::TypeTree>,
        ) -> Result<(), FatalError> {
        unsafe {
            back::write::differentiate(module, cgcx, diff_fncs, typetrees)
        }
    }
    
    fn typetrees(module: &mut Self::Module) -> FxHashMap<String, Self::TypeTree> {
        module.typetrees.drain().collect()
    }
}

unsafe impl Send for LlvmCodegenBackend {} // Llvm is on a per-thread basis
unsafe impl Sync for LlvmCodegenBackend {}

impl LlvmCodegenBackend {
    pub fn new() -> Box<dyn CodegenBackend> {
        Box::new(LlvmCodegenBackend(()))
    }
}

impl CodegenBackend for LlvmCodegenBackend {
    fn init(&self, sess: &Session) {
        llvm_util::init(sess); // Make sure llvm is inited
    }

    fn provide(&self, providers: &mut Providers) {
        providers.global_backend_features =
            |tcx, ()| llvm_util::global_llvm_features(tcx.sess, true)
    }

    fn print(&self, req: PrintRequest, sess: &Session) {
        match req {
            PrintRequest::RelocationModels => {
                println!("Available relocation models:");
                for name in &[
                    "static",
                    "pic",
                    "pie",
                    "dynamic-no-pic",
                    "ropi",
                    "rwpi",
                    "ropi-rwpi",
                    "default",
                ] {
                    println!("    {}", name);
                }
                println!();
            }
            PrintRequest::CodeModels => {
                println!("Available code models:");
                for name in &["tiny", "small", "kernel", "medium", "large"] {
                    println!("    {}", name);
                }
                println!();
            }
            PrintRequest::TlsModels => {
                println!("Available TLS models:");
                for name in &["global-dynamic", "local-dynamic", "initial-exec", "local-exec"] {
                    println!("    {}", name);
                }
                println!();
            }
            PrintRequest::StackProtectorStrategies => {
                println!(
                    r#"Available stack protector strategies:
    all
        Generate stack canaries in all functions.

    strong
        Generate stack canaries in a function if it either:
        - has a local variable of `[T; N]` type, regardless of `T` and `N`
        - takes the address of a local variable.

          (Note that a local variable being borrowed is not equivalent to its
          address being taken: e.g. some borrows may be removed by optimization,
          while by-value argument passing may be implemented with reference to a
          local stack variable in the ABI.)

    basic
        Generate stack canaries in functions with:
        - local variables of `[T; N]` type, where `T` is byte-sized and `N` > 8.

    none
        Do not generate stack canaries.
"#
);
            }
            req => llvm_util::print(req, sess),
        }
    }

    fn print_passes(&self) {
        llvm_util::print_passes();
    }

    fn print_version(&self) {
        llvm_util::print_version();
    }

    fn target_features(&self, sess: &Session) -> Vec<Symbol> {
        target_features(sess)
    }

    fn codegen_crate<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        metadata: EncodedMetadata,
        need_metadata_module: bool,
        ) -> Box<dyn Any> {
        Box::new(rustc_codegen_ssa::base::codegen_crate(
                LlvmCodegenBackend(()),
                tcx,
                crate::llvm_util::target_cpu(tcx.sess).to_string(),
                metadata,
                need_metadata_module,
                ))
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        outputs: &OutputFilenames,
        ) -> Result<(CodegenResults, FxHashMap<WorkProductId, WorkProduct>), ErrorGuaranteed> {
        let (codegen_results, work_products) = ongoing_codegen
            .downcast::<rustc_codegen_ssa::back::write::OngoingCodegen<LlvmCodegenBackend>>()
            .expect("Expected LlvmCodegenBackend's OngoingCodegen, found Box<Any>")
            .join(sess);

        sess.time("llvm_dump_timing_file", || {
            if sess.opts.debugging_opts.llvm_time_trace {
                let file_name = outputs.with_extension("llvm_timings.json");
                llvm_util::time_trace_profiler_finish(&file_name);
            }
        });

        Ok((codegen_results, work_products))
    }

    fn link(
        &self,
        sess: &Session,
        codegen_results: CodegenResults,
        outputs: &OutputFilenames,
        ) -> Result<(), ErrorGuaranteed> {
        use crate::back::archive::LlvmArchiveBuilder;
        use rustc_codegen_ssa::back::link::link_binary;

        // Run the linker on any artifacts that resulted from the LLVM run.
        // This should produce either a finished executable or library.
        link_binary::<LlvmArchiveBuilder<'_>>(sess, &codegen_results, outputs)
    }
}

pub fn get_enzyme_typtree<'tcx>(id: Ty<'tcx>, llvm_data_layout: &str,
                                tcx: TyCtxt<'tcx>, llcx: &'_ llvm::Context, depth: u8) -> TypeTree {
    let mut tt = TypeTree::new();
    if depth > 6 {
        panic!("depth > 6! Abort");
    }

    if id.is_unsafe_ptr() || id.is_ref() || id.is_box() {
        if  id.is_fn_ptr() {
            unimplemented!("what to do whith fn ptr?");
        }

        if id.is_unsafe_ptr() {
            dbg!("a raw ptr");
        } else if id.is_ref() {
            dbg!("a ref");
        } else {
            dbg!("a box");
        }

        tt = TypeTree::from_type(llvm_::CConcreteType::DT_Pointer, llcx).only(-1);
        let inner_id = id.builtin_deref(true).unwrap().ty;
        let inner_tt = get_enzyme_typtree(inner_id, llvm_data_layout, tcx, llcx, depth+1);
        tt.merge(inner_tt.only(-1));
        println!("returning tt with indirection: {}", tt);
        return tt;
    }

    if id.is_scalar() {
        assert!(!id.is_any_ptr());
        dbg!("a scalar");
        let scalar_type = if id.is_integral() {
            llvm::CConcreteType::DT_Integer
        } else {
            assert!(id.is_floating_point());
            match id {
                x if x == tcx.types.f32 => llvm::CConcreteType::DT_Float ,
                x if x == tcx.types.f64 => llvm::CConcreteType::DT_Double,
                _ => panic!("floatTy scalar that is neither f32 nor f64"),
            }
        };
        return llvm::TypeTree::from_type(scalar_type, llcx).only(-1);
    }

    if let Param(_) = id.kind() {
        return TypeTree::new();
    }

    let param_env_and = ParamEnvAnd {
        param_env: ParamEnv::empty(),
        value: id,
    };

    let layout = tcx.layout_of(param_env_and);
    assert!(layout.is_ok());

    let layout = layout.unwrap().layout;
    let fields = layout.fields();
    let _abi = layout.abi();
    let max_size = layout.size();
    dbg!(layout);
    // dbg!(fields);
    // dbg!(abi);
    // dbg!(max_size);

    if id.is_adt() {
        dbg!("an ADT");
        let adt_def = id.ty_adt_def().unwrap();
        let substs = match id.kind() {
            Adt(_, subst_ref) => subst_ref,
            _ => unreachable!(),
        };

        if adt_def.is_struct() {
            dbg!("a struct");
            let (offsets, memory_index) = match fields {
                FieldsShape::Arbitrary{ offsets: o, memory_index: m } => (o,m),
                _ => panic!(""),
            };
            let fields = adt_def.all_fields();
            let mut field_tt = vec![];
            for field in fields {
                let field_ty: Ty<'_> = field.ty(tcx, substs);
                let inner_tt = get_enzyme_typtree(field_ty, llvm_data_layout, tcx, llcx, depth+1).data0();
                dbg!(field_ty);
                println!("inner tt: {}", inner_tt);
                field_tt.push(inner_tt);
            }
            dbg!(offsets);
            dbg!(memory_index);
            // Now let's move those typeTrees in the order that rustc mandates.
            let mut ret_tt = TypeTree::new();
            for i in 0..field_tt.len() {
                let tt = &field_tt[i];
                let offset = offsets[i];
                ret_tt.merge(tt.clone().only(offset.bytes_usize() as isize));
            }
            println!("ret_tt: {}", ret_tt);
            return ret_tt;
        } else {
            unimplemented!("adt that isn't a struct");
        }
    }


    if id.is_array() {
        let (stride, count) = match fields {
            FieldsShape::Array{ stride: s, count: c } => (s,c),
            _ => panic!(""),
        };
        dbg!("an array");
        let byte_stride = stride.bytes_usize();
        let byte_max_size = max_size.bytes_usize();
        let isize_count: isize = (*count).try_into().unwrap();

        assert!(byte_stride * *count as usize == byte_max_size);
        assert!(*count > 0); // return empty TT for empty?
        let sub_id = id.builtin_index().unwrap();
        let sub_tt = get_enzyme_typtree(sub_id, llvm_data_layout, tcx, llcx, depth+1).data0();
        for i in 0isize..isize_count {
            println!("tt: {}", tt);
            println!("sub_tt: {}", sub_tt);
            tt.merge(sub_tt.clone().only(i * (byte_stride as isize)));
        }
    }
    println!("returning tt: {}", tt);
    return tt;
}

#[derive(Clone, Debug)]
pub struct DiffTypeTree {
    pub ret_tt: TypeTree,
    pub input_tt: Vec<TypeTree>,
}

#[allow(dead_code)]
pub struct ModuleLlvm {
    llcx: &'static mut llvm::Context,
    llmod_raw: *const llvm::Module,
    tm: &'static mut llvm::TargetMachine,
    typetrees: FxHashMap<String, DiffTypeTree>,
}

unsafe impl Send for ModuleLlvm {}
unsafe impl Sync for ModuleLlvm {}

impl ModuleLlvm {
    fn new(tcx: TyCtxt<'_>, mod_name: &str) -> Self {
        unsafe {
            let llcx = llvm::LLVMRustContextCreate(tcx.sess.fewer_names());
            let llmod_raw = context::create_module(tcx, llcx, mod_name) as *const _;
            ModuleLlvm { llmod_raw, llcx, tm: create_target_machine(tcx, mod_name), typetrees: Default::default(), }
        }
    }

    fn new_metadata(tcx: TyCtxt<'_>, mod_name: &str) -> Self {
        unsafe {
            let llcx = llvm::LLVMRustContextCreate(tcx.sess.fewer_names());
            let llmod_raw = context::create_module(tcx, llcx, mod_name) as *const _;
            ModuleLlvm { llmod_raw, llcx, tm: create_informational_target_machine(tcx.sess) , typetrees: Default::default() }
        }
    }

    fn parse(
        cgcx: &CodegenContext<LlvmCodegenBackend>,
        name: &CStr,
        buffer: &[u8],
        handler: &Handler,
        ) -> Result<Self, FatalError> {
        unsafe {
            let llcx = llvm::LLVMRustContextCreate(cgcx.fewer_names);
            let llmod_raw = back::lto::parse_module(llcx, name, buffer, handler)?;
            let tm_factory_config = TargetMachineFactoryConfig::new(cgcx, name.to_str().unwrap());
            let tm = match (cgcx.tm_factory)(tm_factory_config) {
                Ok(m) => m,
                Err(e) => {
                    handler.struct_err(&e).emit();
                    return Err(FatalError);
                }
            };


            Ok(ModuleLlvm { llmod_raw, llcx, tm, typetrees: Default::default() })
        }
    }

    fn llmod(&self) -> &llvm::Module {
        unsafe { &*self.llmod_raw }
    }
}

impl Drop for ModuleLlvm {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMRustDisposeTargetMachine(&mut *(self.tm as *mut _));
            llvm::LLVMContextDispose(&mut *(self.llcx as *mut _));
        }
    }
}
