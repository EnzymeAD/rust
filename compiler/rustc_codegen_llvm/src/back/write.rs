#![allow(unused_imports)]
#![allow(unused_variables)]
use crate::llvm::LLVMGetFirstBasicBlock;
use crate::llvm::LLVMBuildCondBr;
use crate::llvm::LLVMBuildICmp;
use crate::llvm::LLVMBuildRetVoid;
use crate::llvm::LLVMRustEraseInstBefore;
use crate::llvm::LLVMRustHasDbgMetadata;
use crate::llvm::LLVMRustHasMetadata;
use crate::llvm::LLVMRustRemoveFncAttr;
use crate::llvm::LLVMMetadataAsValue;
use crate::llvm::LLVMRustGetLastInstruction;
use crate::llvm::LLVMRustDIGetInstMetadata;
use crate::llvm::LLVMRustDIGetInstMetadataOfTy;
use crate::llvm::LLVMRustgetFirstNonPHIOrDbgOrLifetime;
use crate::llvm::LLVMRustGetTerminator;
use crate::llvm::LLVMRustEraseInstFromParent;
use crate::llvm::LLVMRustEraseBBFromParent;
//use crate::llvm::LLVMEraseFromParent;
use crate::back::lto::ThinBuffer;
use crate::back::owned_target_machine::OwnedTargetMachine;
use crate::back::profiling::{
    selfprofile_after_pass_callback, selfprofile_before_pass_callback, LlvmSelfProfiler,
};
use crate::base;
use crate::common;
use crate::errors::{
    CopyBitcode, FromLlvmDiag, FromLlvmOptimizationDiag, LlvmError, UnknownCompression,
    WithLlvmError, WriteBytecode,
};
use crate::llvm::{self, DiagnosticInfo, PassManager};
use crate::llvm::{
    enzyme_rust_forward_diff, enzyme_rust_reverse_diff, AttributeKind, BasicBlock, FreeTypeAnalysis,
    CreateEnzymeLogic, CreateTypeAnalysis, EnzymeLogicRef, EnzymeTypeAnalysisRef, LLVMAddFunction,
    LLVMAppendBasicBlockInContext, LLVMBuildCall2, LLVMBuildExtractValue, LLVMBuildRet,
    LLVMCountParams, LLVMCountStructElementTypes, LLVMCreateBuilderInContext,
    LLVMCreateStringAttribute, LLVMDeleteFunction, LLVMDisposeBuilder, LLVMDumpModule,
    LLVMGetBasicBlockTerminator, LLVMGetFirstFunction, LLVMGetModuleContext,
    LLVMGetNextFunction, LLVMGetParams, LLVMGetReturnType, LLVMRustGetFunctionType, LLVMGetStringAttributeAtIndex,
    LLVMGlobalGetValueType, LLVMIsEnumAttribute, LLVMIsStringAttribute, LLVMPositionBuilderAtEnd,
    LLVMRemoveStringAttributeAtIndex, LLVMReplaceAllUsesWith, LLVMRustAddEnumAttributeAtIndex,
    LLVMRustAddFunctionAttributes, LLVMRustGetEnumAttributeAtIndex,
    LLVMRustRemoveEnumAttributeAtIndex, LLVMSetValueName2, LLVMVerifyFunction,
    LLVMVoidTypeInContext, Value,
};
use crate::llvm_util;
use crate::type_::Type;
use crate::typetree::to_enzyme_typetree;
use crate::DiffTypeTree;
use crate::LlvmCodegenBackend;
use crate::ModuleLlvm;
use llvm::IntPredicate;
use llvm::LLVMRustDISetInstMetadata;
use llvm::{
    LLVMRustLLVMHasZlibCompressionForDebugSymbols, LLVMRustLLVMHasZstdCompressionForDebugSymbols, LLVMGetNextBasicBlock,
};
use rustc_ast::expand::autodiff_attrs::{AutoDiffItem, DiffActivity, DiffMode};
use rustc_codegen_ssa::back::link::ensure_removed;
use rustc_codegen_ssa::back::write::{
    BitcodeSection, CodegenContext, EmitObj, ModuleConfig, TargetMachineFactoryConfig,
    TargetMachineFactoryFn,
};
use rustc_codegen_ssa::traits::*;
use rustc_codegen_ssa::{CompiledModule, ModuleCodegen};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_errors::{DiagCtxt, FatalError, Level};
use rustc_fs_util::{link_or_copy, path_to_c_string};
use rustc_middle::ty::TyCtxt;
use rustc_session::config::{self, Lto, OutputType, Passes, SplitDwarfKind, SwitchWithOptPath};
use rustc_session::Session;
use rustc_span::symbol::sym;
use rustc_span::InnerSpan;
use rustc_target::spec::{CodeModel, RelocModel, SanitizerSet, SplitDebuginfo, TlsModel};

use crate::llvm::diagnostic::OptimizationDiagnosticKind;
use libc::{c_char, c_int, c_uint, c_void, size_t};
use std::ffi::{CStr, CString};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::slice;
use std::str;
use std::sync::Arc;

pub fn llvm_err<'a>(dcx: &rustc_errors::DiagCtxt, err: LlvmError<'a>) -> FatalError {
    match llvm::last_error() {
        Some(llvm_err) => dcx.emit_almost_fatal(WithLlvmError(err, llvm_err)),
        None => dcx.emit_almost_fatal(err),
    }
}

pub fn write_output_file<'ll>(
    dcx: &rustc_errors::DiagCtxt,
    target: &'ll llvm::TargetMachine,
    pm: &llvm::PassManager<'ll>,
    m: &'ll llvm::Module,
    output: &Path,
    dwo_output: Option<&Path>,
    file_type: llvm::FileType,
    self_profiler_ref: &SelfProfilerRef,
) -> Result<(), FatalError> {
    debug!("write_output_file output={:?} dwo_output={:?}", output, dwo_output);
    unsafe {
        let output_c = path_to_c_string(output);
        let dwo_output_c;
        let dwo_output_ptr = if let Some(dwo_output) = dwo_output {
            dwo_output_c = path_to_c_string(dwo_output);
            dwo_output_c.as_ptr()
        } else {
            std::ptr::null()
        };
        let result = llvm::LLVMRustWriteOutputFile(
            target,
            pm,
            m,
            output_c.as_ptr(),
            dwo_output_ptr,
            file_type,
        );

        // Record artifact sizes for self-profiling
        if result == llvm::LLVMRustResult::Success {
            let artifact_kind = match file_type {
                llvm::FileType::ObjectFile => "object_file",
                llvm::FileType::AssemblyFile => "assembly_file",
            };
            record_artifact_size(self_profiler_ref, artifact_kind, output);
            if let Some(dwo_file) = dwo_output {
                record_artifact_size(self_profiler_ref, "dwo_file", dwo_file);
            }
        }

        result.into_result().map_err(|()| llvm_err(dcx, LlvmError::WriteOutput { path: output }))
    }
}

pub fn create_informational_target_machine(sess: &Session) -> OwnedTargetMachine {
    let config = TargetMachineFactoryConfig { split_dwarf_file: None, output_obj_file: None };
    // Can't use query system here quite yet because this function is invoked before the query
    // system/tcx is set up.
    let features = llvm_util::global_llvm_features(sess, false);
    target_machine_factory(sess, config::OptLevel::No, &features)(config)
        .unwrap_or_else(|err| llvm_err(sess.dcx(), err).raise())
}

pub fn create_target_machine(tcx: TyCtxt<'_>, mod_name: &str) -> OwnedTargetMachine {
    let split_dwarf_file = if tcx.sess.target_can_use_split_dwarf() {
        tcx.output_filenames(()).split_dwarf_path(
            tcx.sess.split_debuginfo(),
            tcx.sess.opts.unstable_opts.split_dwarf_kind,
            Some(mod_name),
        )
    } else {
        None
    };

    let output_obj_file =
        Some(tcx.output_filenames(()).temp_path(OutputType::Object, Some(mod_name)));
    let config = TargetMachineFactoryConfig { split_dwarf_file, output_obj_file };

    target_machine_factory(
        tcx.sess,
        tcx.backend_optimization_level(()),
        tcx.global_backend_features(()),
    )(config)
    .unwrap_or_else(|err| llvm_err(tcx.sess.dcx(), err).raise())
}

pub fn to_llvm_opt_settings(
    cfg: config::OptLevel,
) -> (llvm::CodeGenOptLevel, llvm::CodeGenOptSize) {
    use self::config::OptLevel::*;
    match cfg {
        No => (llvm::CodeGenOptLevel::None, llvm::CodeGenOptSizeNone),
        Less => (llvm::CodeGenOptLevel::Less, llvm::CodeGenOptSizeNone),
        Default => (llvm::CodeGenOptLevel::Default, llvm::CodeGenOptSizeNone),
        Aggressive => (llvm::CodeGenOptLevel::Aggressive, llvm::CodeGenOptSizeNone),
        Size => (llvm::CodeGenOptLevel::Default, llvm::CodeGenOptSizeDefault),
        SizeMin => (llvm::CodeGenOptLevel::Default, llvm::CodeGenOptSizeAggressive),
    }
}

fn to_pass_builder_opt_level(cfg: config::OptLevel) -> llvm::PassBuilderOptLevel {
    use config::OptLevel::*;
    match cfg {
        No => llvm::PassBuilderOptLevel::O0,
        Less => llvm::PassBuilderOptLevel::O1,
        Default => llvm::PassBuilderOptLevel::O2,
        Aggressive => llvm::PassBuilderOptLevel::O3,
        Size => llvm::PassBuilderOptLevel::Os,
        SizeMin => llvm::PassBuilderOptLevel::Oz,
    }
}

fn to_llvm_relocation_model(relocation_model: RelocModel) -> llvm::RelocModel {
    match relocation_model {
        RelocModel::Static => llvm::RelocModel::Static,
        // LLVM doesn't have a PIE relocation model, it represents PIE as PIC with an extra attribute.
        RelocModel::Pic | RelocModel::Pie => llvm::RelocModel::PIC,
        RelocModel::DynamicNoPic => llvm::RelocModel::DynamicNoPic,
        RelocModel::Ropi => llvm::RelocModel::ROPI,
        RelocModel::Rwpi => llvm::RelocModel::RWPI,
        RelocModel::RopiRwpi => llvm::RelocModel::ROPI_RWPI,
    }
}

pub(crate) fn to_llvm_code_model(code_model: Option<CodeModel>) -> llvm::CodeModel {
    match code_model {
        Some(CodeModel::Tiny) => llvm::CodeModel::Tiny,
        Some(CodeModel::Small) => llvm::CodeModel::Small,
        Some(CodeModel::Kernel) => llvm::CodeModel::Kernel,
        Some(CodeModel::Medium) => llvm::CodeModel::Medium,
        Some(CodeModel::Large) => llvm::CodeModel::Large,
        None => llvm::CodeModel::None,
    }
}

pub fn target_machine_factory(
    sess: &Session,
    optlvl: config::OptLevel,
    target_features: &[String],
) -> TargetMachineFactoryFn<LlvmCodegenBackend> {
    let reloc_model = to_llvm_relocation_model(sess.relocation_model());

    let (opt_level, _) = to_llvm_opt_settings(optlvl);
    let use_softfp = sess.opts.cg.soft_float;

    let ffunction_sections =
        sess.opts.unstable_opts.function_sections.unwrap_or(sess.target.function_sections);
    let fdata_sections = ffunction_sections;
    let funique_section_names = !sess.opts.unstable_opts.no_unique_section_names;

    let code_model = to_llvm_code_model(sess.code_model());

    let mut singlethread = sess.target.singlethread;

    // On the wasm target once the `atomics` feature is enabled that means that
    // we're no longer single-threaded, or otherwise we don't want LLVM to
    // lower atomic operations to single-threaded operations.
    if singlethread && sess.target.is_like_wasm && sess.target_features.contains(&sym::atomics) {
        singlethread = false;
    }

    let triple = SmallCStr::new(&sess.target.llvm_target);
    let cpu = SmallCStr::new(llvm_util::target_cpu(sess));
    let features = CString::new(target_features.join(",")).unwrap();
    let abi = SmallCStr::new(&sess.target.llvm_abiname);
    let trap_unreachable =
        sess.opts.unstable_opts.trap_unreachable.unwrap_or(sess.target.trap_unreachable);
    let emit_stack_size_section = sess.opts.unstable_opts.emit_stack_sizes;

    let asm_comments = sess.opts.unstable_opts.asm_comments;
    let relax_elf_relocations =
        sess.opts.unstable_opts.relax_elf_relocations.unwrap_or(sess.target.relax_elf_relocations);

    let use_init_array =
        !sess.opts.unstable_opts.use_ctors_section.unwrap_or(sess.target.use_ctors_section);

    let path_mapping = sess.source_map().path_mapping().clone();

    let use_emulated_tls = matches!(sess.tls_model(), TlsModel::Emulated);

    // copy the exe path, followed by path all into one buffer
    // null terminating them so we can use them as null terminated strings
    let args_cstr_buff = {
        let mut args_cstr_buff: Vec<u8> = Vec::new();
        let exe_path = std::env::current_exe().unwrap_or_default();
        let exe_path_str = exe_path.into_os_string().into_string().unwrap_or_default();

        args_cstr_buff.extend_from_slice(exe_path_str.as_bytes());
        args_cstr_buff.push(0);

        for arg in sess.expanded_args.iter() {
            args_cstr_buff.extend_from_slice(arg.as_bytes());
            args_cstr_buff.push(0);
        }

        args_cstr_buff
    };

    let debuginfo_compression = sess.opts.debuginfo_compression.to_string();
    match sess.opts.debuginfo_compression {
        rustc_session::config::DebugInfoCompression::Zlib => {
            if !unsafe { LLVMRustLLVMHasZlibCompressionForDebugSymbols() } {
                sess.emit_warning(UnknownCompression { algorithm: "zlib" });
            }
        }
        rustc_session::config::DebugInfoCompression::Zstd => {
            if !unsafe { LLVMRustLLVMHasZstdCompressionForDebugSymbols() } {
                sess.emit_warning(UnknownCompression { algorithm: "zstd" });
            }
        }
        rustc_session::config::DebugInfoCompression::None => {}
    };
    let debuginfo_compression = SmallCStr::new(&debuginfo_compression);

    let should_prefer_remapped_for_split_debuginfo_paths =
        sess.should_prefer_remapped_for_split_debuginfo_paths();

    Arc::new(move |config: TargetMachineFactoryConfig| {
        let path_to_cstring_helper = |path: Option<PathBuf>| -> CString {
            let path = path.unwrap_or_default();
            let path = if should_prefer_remapped_for_split_debuginfo_paths {
                path_mapping.map_prefix(path).0
            } else {
                path.into()
            };
            CString::new(path.to_str().unwrap()).unwrap()
        };

        let split_dwarf_file = path_to_cstring_helper(config.split_dwarf_file);
        let output_obj_file = path_to_cstring_helper(config.output_obj_file);

        OwnedTargetMachine::new(
            &triple,
            &cpu,
            &features,
            &abi,
            code_model,
            reloc_model,
            opt_level,
            use_softfp,
            ffunction_sections,
            fdata_sections,
            funique_section_names,
            trap_unreachable,
            singlethread,
            asm_comments,
            emit_stack_size_section,
            relax_elf_relocations,
            use_init_array,
            &split_dwarf_file,
            &output_obj_file,
            &debuginfo_compression,
            use_emulated_tls,
            &args_cstr_buff,
        )
    })
}

pub(crate) fn save_temp_bitcode(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    module: &ModuleCodegen<ModuleLlvm>,
    name: &str,
) {
    if !cgcx.save_temps {
        return;
    }
    unsafe {
        let ext = format!("{name}.bc");
        let cgu = Some(&module.name[..]);
        let path = cgcx.output_filenames.temp_path_ext(&ext, cgu);
        let cstr = path_to_c_string(&path);
        let llmod = module.module_llvm.llmod();
        llvm::LLVMWriteBitcodeToFile(llmod, cstr.as_ptr());
    }
}

/// In what context is a dignostic handler being attached to a codegen unit?
pub enum CodegenDiagnosticsStage {
    /// Prelink optimization stage.
    Opt,
    /// LTO/ThinLTO postlink optimization stage.
    LTO,
    /// Code generation.
    Codegen,
}

pub struct DiagnosticHandlers<'a> {
    data: *mut (&'a CodegenContext<LlvmCodegenBackend>, &'a DiagCtxt),
    llcx: &'a llvm::Context,
    old_handler: Option<&'a llvm::DiagnosticHandler>,
}

impl<'a> DiagnosticHandlers<'a> {
    pub fn new(
        cgcx: &'a CodegenContext<LlvmCodegenBackend>,
        dcx: &'a DiagCtxt,
        llcx: &'a llvm::Context,
        module: &ModuleCodegen<ModuleLlvm>,
        stage: CodegenDiagnosticsStage,
    ) -> Self {
        let remark_passes_all: bool;
        let remark_passes: Vec<CString>;
        match &cgcx.remark {
            Passes::All => {
                remark_passes_all = true;
                remark_passes = Vec::new();
            }
            Passes::Some(passes) => {
                remark_passes_all = false;
                remark_passes =
                    passes.iter().map(|name| CString::new(name.as_str()).unwrap()).collect();
            }
        };
        let remark_passes: Vec<*const c_char> =
            remark_passes.iter().map(|name: &CString| name.as_ptr()).collect();
        let remark_file = cgcx
            .remark_dir
            .as_ref()
            // Use the .opt.yaml file suffix, which is supported by LLVM's opt-viewer.
            .map(|dir| {
                let stage_suffix = match stage {
                    CodegenDiagnosticsStage::Codegen => "codegen",
                    CodegenDiagnosticsStage::Opt => "opt",
                    CodegenDiagnosticsStage::LTO => "lto",
                };
                dir.join(format!("{}.{stage_suffix}.opt.yaml", module.name))
            })
            .and_then(|dir| dir.to_str().and_then(|p| CString::new(p).ok()));

        let pgo_available = cgcx.opts.cg.profile_use.is_some();
        let data = Box::into_raw(Box::new((cgcx, dcx)));
        unsafe {
            let old_handler = llvm::LLVMRustContextGetDiagnosticHandler(llcx);
            llvm::LLVMRustContextConfigureDiagnosticHandler(
                llcx,
                diagnostic_handler,
                data.cast(),
                remark_passes_all,
                remark_passes.as_ptr(),
                remark_passes.len(),
                // The `as_ref()` is important here, otherwise the `CString` will be dropped
                // too soon!
                remark_file.as_ref().map(|dir| dir.as_ptr()).unwrap_or(std::ptr::null()),
                pgo_available,
            );
            DiagnosticHandlers { data, llcx, old_handler }
        }
    }
}

impl<'a> Drop for DiagnosticHandlers<'a> {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMRustContextSetDiagnosticHandler(self.llcx, self.old_handler);
            drop(Box::from_raw(self.data));
        }
    }
}

fn report_inline_asm(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    msg: String,
    level: llvm::DiagnosticLevel,
    mut cookie: c_uint,
    source: Option<(String, Vec<InnerSpan>)>,
) {
    // In LTO build we may get srcloc values from other crates which are invalid
    // since they use a different source map. To be safe we just suppress these
    // in LTO builds.
    if matches!(cgcx.lto, Lto::Fat | Lto::Thin) {
        cookie = 0;
    }
    let level = match level {
        llvm::DiagnosticLevel::Error => Level::Error { lint: false },
        llvm::DiagnosticLevel::Warning => Level::Warning(None),
        llvm::DiagnosticLevel::Note | llvm::DiagnosticLevel::Remark => Level::Note,
    };
    cgcx.diag_emitter.inline_asm_error(cookie as u32, msg, level, source);
}

unsafe extern "C" fn diagnostic_handler(info: &DiagnosticInfo, user: *mut c_void) {
    if user.is_null() {
        return;
    }
    let (cgcx, dcx) = *(user as *const (&CodegenContext<LlvmCodegenBackend>, &DiagCtxt));

    match llvm::diagnostic::Diagnostic::unpack(info) {
        llvm::diagnostic::InlineAsm(inline) => {
            report_inline_asm(cgcx, inline.message, inline.level, inline.cookie, inline.source);
        }

        llvm::diagnostic::Optimization(opt) => {
            dcx.emit_note(FromLlvmOptimizationDiag {
                filename: &opt.filename,
                line: opt.line,
                column: opt.column,
                pass_name: &opt.pass_name,
                kind: match opt.kind {
                    OptimizationDiagnosticKind::OptimizationRemark => "success",
                    OptimizationDiagnosticKind::OptimizationMissed
                    | OptimizationDiagnosticKind::OptimizationFailure => "missed",
                    OptimizationDiagnosticKind::OptimizationAnalysis
                    | OptimizationDiagnosticKind::OptimizationAnalysisFPCommute
                    | OptimizationDiagnosticKind::OptimizationAnalysisAliasing => "analysis",
                    OptimizationDiagnosticKind::OptimizationRemarkOther => "other",
                },
                message: &opt.message,
            });
        }
        llvm::diagnostic::PGO(diagnostic_ref) | llvm::diagnostic::Linker(diagnostic_ref) => {
            let message = llvm::build_string(|s| {
                llvm::LLVMRustWriteDiagnosticInfoToString(diagnostic_ref, s)
            })
            .expect("non-UTF8 diagnostic");
            dcx.emit_warning(FromLlvmDiag { message });
        }
        llvm::diagnostic::Unsupported(diagnostic_ref) => {
            let message = llvm::build_string(|s| {
                llvm::LLVMRustWriteDiagnosticInfoToString(diagnostic_ref, s)
            })
            .expect("non-UTF8 diagnostic");
            dcx.emit_err(FromLlvmDiag { message });
        }
        llvm::diagnostic::UnknownDiagnostic(..) => {}
    }
}

fn get_pgo_gen_path(config: &ModuleConfig) -> Option<CString> {
    match config.pgo_gen {
        SwitchWithOptPath::Enabled(ref opt_dir_path) => {
            let path = if let Some(dir_path) = opt_dir_path {
                dir_path.join("default_%m.profraw")
            } else {
                PathBuf::from("default_%m.profraw")
            };

            Some(CString::new(format!("{}", path.display())).unwrap())
        }
        SwitchWithOptPath::Disabled => None,
    }
}

fn get_pgo_use_path(config: &ModuleConfig) -> Option<CString> {
    config
        .pgo_use
        .as_ref()
        .map(|path_buf| CString::new(path_buf.to_string_lossy().as_bytes()).unwrap())
}

fn get_pgo_sample_use_path(config: &ModuleConfig) -> Option<CString> {
    config
        .pgo_sample_use
        .as_ref()
        .map(|path_buf| CString::new(path_buf.to_string_lossy().as_bytes()).unwrap())
}

fn get_instr_profile_output_path(config: &ModuleConfig) -> Option<CString> {
    config.instrument_coverage.then(|| CString::new("default_%m_%p.profraw").unwrap())
}

pub(crate) unsafe fn llvm_optimize(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    dcx: &DiagCtxt,
    module: &ModuleCodegen<ModuleLlvm>,
    config: &ModuleConfig,
    opt_level: config::OptLevel,
    opt_stage: llvm::OptStage,
    first_run: bool,
) -> Result<(), FatalError> {
    // Enzyme:
    // We want to simplify / optimize functions before AD.
    // However, benchmarks show that optimizations increasing the code size
    // tend to reduce AD performance. Therefore activate them first, then differentiate the code
    // and finally re-optimize the module, now with all optimizations available.
    // RIP compile time.

    let unroll_loops;
    let vectorize_slp;
    let vectorize_loop;

    if first_run {
        unroll_loops = false;
        vectorize_slp = false;
        vectorize_loop = false;
    } else {
        unroll_loops =
        opt_level != config::OptLevel::Size && opt_level != config::OptLevel::SizeMin;
        vectorize_slp = config.vectorize_slp;
        vectorize_loop = config.vectorize_loop;
        dbg!("Enzyme: Running with unroll_loops: {}, vectorize_slp: {}, vectorize_loop: {}", unroll_loops, vectorize_slp, vectorize_loop);
    }

    let using_thin_buffers = opt_stage == llvm::OptStage::PreLinkThinLTO || config.bitcode_needed();
    let pgo_gen_path = get_pgo_gen_path(config);
    let pgo_use_path = get_pgo_use_path(config);
    let pgo_sample_use_path = get_pgo_sample_use_path(config);
    let is_lto = opt_stage == llvm::OptStage::ThinLTO || opt_stage == llvm::OptStage::FatLTO;
    let instr_profile_output_path = get_instr_profile_output_path(config);
    // Sanitizer instrumentation is only inserted during the pre-link optimization stage.
    let sanitizer_options = if !is_lto {
        Some(llvm::SanitizerOptions {
            sanitize_address: config.sanitizer.contains(SanitizerSet::ADDRESS),
            sanitize_address_recover: config.sanitizer_recover.contains(SanitizerSet::ADDRESS),
            sanitize_cfi: config.sanitizer.contains(SanitizerSet::CFI),
            sanitize_kcfi: config.sanitizer.contains(SanitizerSet::KCFI),
            sanitize_memory: config.sanitizer.contains(SanitizerSet::MEMORY),
            sanitize_memory_recover: config.sanitizer_recover.contains(SanitizerSet::MEMORY),
            sanitize_memory_track_origins: config.sanitizer_memory_track_origins as c_int,
            sanitize_thread: config.sanitizer.contains(SanitizerSet::THREAD),
            sanitize_hwaddress: config.sanitizer.contains(SanitizerSet::HWADDRESS),
            sanitize_hwaddress_recover: config.sanitizer_recover.contains(SanitizerSet::HWADDRESS),
            sanitize_kernel_address: config.sanitizer.contains(SanitizerSet::KERNELADDRESS),
            sanitize_kernel_address_recover: config
                .sanitizer_recover
                .contains(SanitizerSet::KERNELADDRESS),
        })
    } else {
        None
    };

    let mut llvm_profiler = cgcx
        .prof
        .llvm_recording_enabled()
        .then(|| LlvmSelfProfiler::new(cgcx.prof.get_self_profiler().unwrap()));

    let llvm_selfprofiler =
        llvm_profiler.as_mut().map(|s| s as *mut _ as *mut c_void).unwrap_or(std::ptr::null_mut());

    let extra_passes = if !is_lto { config.passes.join(",") } else { "".to_string() };

    let llvm_plugins = config.llvm_plugins.join(",");

    // FIXME: NewPM doesn't provide a facility to pass custom InlineParams.
    // We would have to add upstream support for this first, before we can support
    // config.inline_threshold and our more aggressive default thresholds.
    let result = llvm::LLVMRustOptimize(
        module.module_llvm.llmod(),
        &*module.module_llvm.tm,
        to_pass_builder_opt_level(opt_level),
        opt_stage,
        cgcx.opts.cg.linker_plugin_lto.enabled(),
        config.no_prepopulate_passes,
        config.verify_llvm_ir,
        using_thin_buffers,
        config.merge_functions,
        unroll_loops,
        vectorize_slp,
        vectorize_loop,
        config.emit_lifetime_markers,
        sanitizer_options.as_ref(),
        pgo_gen_path.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
        pgo_use_path.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
        config.instrument_coverage,
        instr_profile_output_path.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
        config.instrument_gcov,
        pgo_sample_use_path.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
        config.debug_info_for_profiling,
        llvm_selfprofiler,
        selfprofile_before_pass_callback,
        selfprofile_after_pass_callback,
        extra_passes.as_ptr().cast(),
        extra_passes.len(),
        llvm_plugins.as_ptr().cast(),
        llvm_plugins.len(),
    );
    result.into_result().map_err(|()| llvm_err(dcx, LlvmError::RunLlvmPasses))
}

fn get_params(fnc: &Value) -> Vec<&Value> {
    unsafe {
        let param_num = LLVMCountParams(fnc) as usize;
        let mut fnc_args: Vec<&Value> = vec![];
        fnc_args.reserve(param_num);
        LLVMGetParams(fnc, fnc_args.as_mut_ptr());
        fnc_args.set_len(param_num);
        fnc_args
    }
}

// DESIGN:
// Today we have our placeholder function, and our Enzyme generated one.
// We create a wrapper function and delete the placeholder body.
// We then call the wrapper from the placeholder.
//
// Soon, we won't delete the whole placeholder, but just the loop,
// and the two inline asm sections. For now we can still call the wrapper.
// In the future we call our Enzyme generated function directly and unwrap the return
// struct in our original placeholder.
//
// define internal double @_ZN2ad3bar17ha38374e821680177E(ptr align 8 %0, ptr align 8 %1, double %2) unnamed_addr #17 !dbg !13678 {
//  %4 = alloca double, align 8
//  %5 = alloca ptr, align 8
//  %6 = alloca ptr, align 8
//  %7 = alloca { ptr, double }, align 8
//  store ptr %0, ptr %6, align 8
//  call void @llvm.dbg.declare(metadata ptr %6, metadata !13682, metadata !DIExpression()), !dbg !13685
//  store ptr %1, ptr %5, align 8
//  call void @llvm.dbg.declare(metadata ptr %5, metadata !13683, metadata !DIExpression()), !dbg !13685
//  store double %2, ptr %4, align 8
//  call void @llvm.dbg.declare(metadata ptr %4, metadata !13684, metadata !DIExpression()), !dbg !13686
//  call void asm sideeffect alignstack inteldialect "NOP", "~{dirflag},~{fpsr},~{flags},~{memory}"(), !dbg !13687, !srcloc !23
//  %8 = call double @_ZN2ad3foo17h95b548a9411653b2E(ptr align 8 %0), !dbg !13687
//  %9 = call double @_ZN4core4hint9black_box17h7bd67a41b0f12bdfE(double %8), !dbg !13687
//  store ptr %1, ptr %7, align 8, !dbg !13687
//  %10 = getelementptr inbounds { ptr, double }, ptr %7, i32 0, i32 1, !dbg !13687
//  store double %2, ptr %10, align 8, !dbg !13687
//  %11 = getelementptr inbounds { ptr, double }, ptr %7, i32 0, i32 0, !dbg !13687
//  %12 = load ptr, ptr %11, align 8, !dbg !13687, !nonnull !23, !align !1047, !noundef !23
//  %13 = getelementptr inbounds { ptr, double }, ptr %7, i32 0, i32 1, !dbg !13687
//  %14 = load double, ptr %13, align 8, !dbg !13687, !noundef !23
//  %15 = call { ptr, double } @_ZN4core4hint9black_box17h669f3b22afdcb487E(ptr align 8 %12, double %14), !dbg !13687
//  %16 = extractvalue { ptr, double } %15, 0, !dbg !13687
//  %17 = extractvalue { ptr, double } %15, 1, !dbg !13687
//  br label %18, !dbg !13687
//
//18:                                               ; preds = %18, %3
//  br label %18, !dbg !13687


unsafe fn create_call<'a>(tgt: &'a Value, src: &'a Value, rev_mode: bool,
    llmod: &'a llvm::Module, llcx: &llvm::Context, size_positions: &[usize]) {
    dbg!("size_positions: {:?}", size_positions);
   // first, remove all calls from fnc
   let bb = LLVMGetFirstBasicBlock(tgt);
   let br = LLVMRustGetTerminator(bb);
   LLVMRustEraseInstFromParent(br);

   // now add a call to inner.
    // append call to src at end of bb.
    let f_ty = LLVMRustGetFunctionType(src);

    let inner_param_num = LLVMCountParams(src);
    let outer_param_num = LLVMCountParams(tgt);
    let outer_args: Vec<&Value> = get_params(tgt);
    let inner_args: Vec<&Value> = get_params(src);
    let mut call_args: Vec<&Value> = vec![];

    let mut safety_vals = vec![];
    let builder = LLVMCreateBuilderInContext(llcx);
    let last_inst = LLVMRustGetLastInstruction(bb).unwrap();
    LLVMPositionBuilderAtEnd(builder, bb);

    if inner_param_num == outer_param_num {
        call_args = outer_args;
    } else {
        trace!("Different number of args, adjusting");
        let mut outer_pos: usize = 0;
        let mut inner_pos: usize = 0;
        // copy over if they are identical.
        // If not, skip the outer arg (and assert it's int).
        while outer_pos < outer_param_num as usize {
            let inner_arg = inner_args[inner_pos];
            let outer_arg = outer_args[outer_pos];
            let inner_arg_ty = llvm::LLVMTypeOf(inner_arg);
            let outer_arg_ty = llvm::LLVMTypeOf(outer_arg);
            if inner_arg_ty == outer_arg_ty {
                call_args.push(outer_arg);
                inner_pos += 1;
                outer_pos += 1;
            } else {
                // out: (ptr, <>int1, ptr, int2)
                // inner: (ptr, <>ptr, int)
                // goal: (ptr, ptr, int1), skipping int2
                // we are here: <>
                assert!(llvm::LLVMRustGetTypeKind(outer_arg_ty) == llvm::TypeKind::Integer);
                assert!(llvm::LLVMRustGetTypeKind(inner_arg_ty) == llvm::TypeKind::Pointer);
                let next_outer_arg = outer_args[outer_pos + 1];
                let next_inner_arg = inner_args[inner_pos + 1];
                let next_outer_arg_ty = llvm::LLVMTypeOf(next_outer_arg);
                let next_inner_arg_ty = llvm::LLVMTypeOf(next_inner_arg);
                assert!(llvm::LLVMRustGetTypeKind(next_outer_arg_ty) == llvm::TypeKind::Pointer);
                assert!(llvm::LLVMRustGetTypeKind(next_inner_arg_ty) == llvm::TypeKind::Integer);
                let next2_outer_arg = outer_args[outer_pos + 2];
                let next2_outer_arg_ty = llvm::LLVMTypeOf(next2_outer_arg);
                assert!(llvm::LLVMRustGetTypeKind(next2_outer_arg_ty) == llvm::TypeKind::Integer);
                call_args.push(next_outer_arg);
                call_args.push(outer_arg);

                outer_pos += 3;
                inner_pos += 2;

                // Now we assert if int1 <= int2
                let res = LLVMBuildICmp(
                    builder,
                    IntPredicate::IntULE as u32,
                    outer_arg,
                    next2_outer_arg,
                    "safety_check".as_ptr() as *const c_char);
                safety_vals.push(res);
            }
        }
    }

    if inner_param_num as usize != call_args.len() {
        panic!("Args len shouldn't differ. Please report this. {} : {}", inner_param_num, call_args.len());
    }

    // Now add the safety checks.
    if !safety_vals.is_empty() {
        dbg!("Adding safety checks");
        // first we create one bb per check and two more for the fail and success case.
        let fail_bb = LLVMAppendBasicBlockInContext(llcx, tgt, "ad_safety_fail".as_ptr() as *const c_char);
        let success_bb = LLVMAppendBasicBlockInContext(llcx, tgt, "ad_safety_success".as_ptr() as *const c_char);
        let mut err_bb = vec![];
        for i in 0..safety_vals.len() {
            let name: String = format!("ad_safety_err_{}", i);
            err_bb.push(LLVMAppendBasicBlockInContext(llcx, tgt, name.as_ptr() as *const c_char));
        }
        for (i, &val) in safety_vals.iter().enumerate() {
            LLVMBuildCondBr(builder, val, err_bb[i], fail_bb);
            LLVMPositionBuilderAtEnd(builder, err_bb[i]);
        }
        LLVMBuildCondBr(builder, safety_vals.last().unwrap(), success_bb, fail_bb);
        LLVMPositionBuilderAtEnd(builder, fail_bb);


        let panic_name: CString = get_panic_name(llmod);

        let mut arg_vec = vec![add_panic_msg_to_global(llmod, llcx)];

        let fnc1 = llvm::LLVMGetNamedFunction(llmod, panic_name.as_ptr() as *const c_char);
        assert!(fnc1.is_some());
        let fnc1 = fnc1.unwrap();
        let ty = LLVMRustGetFunctionType(fnc1);
        let call = LLVMBuildCall2(builder, ty, fnc1, arg_vec.as_mut_ptr(), arg_vec.len(), panic_name.as_ptr() as *const c_char);
        llvm::LLVMSetTailCall(call, 1);
        llvm::LLVMBuildUnreachable(builder);
        LLVMPositionBuilderAtEnd(builder, success_bb);
    }

    let inner_fnc_name = llvm::get_value_name(src);
    let c_inner_fnc_name = CString::new(inner_fnc_name).unwrap();

    let mut struct_ret = LLVMBuildCall2(
        builder,
        f_ty,
        src,
        call_args.as_mut_ptr(),
        call_args.len(),
        c_inner_fnc_name.as_ptr(),
    );


    // Add dummy dbg info to our newly generated call, if we have any.
    let inst = LLVMRustgetFirstNonPHIOrDbgOrLifetime(bb).unwrap();
    let md_ty = llvm::LLVMGetMDKindIDInContext(
            llcx,
            "dbg".as_ptr() as *const c_char,
            "dbg".len() as c_uint,
        );

    if LLVMRustHasMetadata(last_inst, md_ty) {
        let md = LLVMRustDIGetInstMetadata(last_inst);
        let md_val = LLVMMetadataAsValue(llcx, md);
        let md2 = llvm::LLVMSetMetadata(struct_ret, md_ty, md_val);
    } else {
        trace!("No dbg info");
    }

    // Now clean up placeholder code.
    LLVMRustEraseInstBefore(bb, last_inst);
    //dbg!(&tgt);

    let f_return_type = LLVMGetReturnType(LLVMGlobalGetValueType(src));
    let void_type = LLVMVoidTypeInContext(llcx);
    // Now unwrap the struct_ret if it's actually a struct
    if rev_mode && f_return_type != void_type {
        let num_elem_in_ret_struct = LLVMCountStructElementTypes(f_return_type);
        if num_elem_in_ret_struct == 1 {
            let inner_grad_name = "foo".to_string();
            let c_inner_grad_name = CString::new(inner_grad_name).unwrap();
            struct_ret = LLVMBuildExtractValue(builder, struct_ret, 0, c_inner_grad_name.as_ptr());
        }
    }
    if f_return_type != void_type {
        let _ret = LLVMBuildRet(builder, struct_ret);
    } else {
        let _ret = LLVMBuildRetVoid(builder);
    }
    LLVMDisposeBuilder(builder);
    let _fnc_ok =
        LLVMVerifyFunction(tgt, llvm::LLVMVerifierFailureAction::LLVMAbortProcessAction);
}
unsafe fn get_panic_name(llmod: &llvm::Module) -> CString {
    // The names are mangled and their ending changes based on a hash, so just take whichever.
    let mut f = LLVMGetFirstFunction(llmod);
    loop {
        if let Some(lf) = f {
            f = LLVMGetNextFunction(lf);
            let fnc_name = llvm::get_value_name(lf);
            let fnc_name: String = String::from_utf8(fnc_name.to_vec()).unwrap();
            if fnc_name.starts_with("_ZN4core9panicking14panic_explicit") {
                return CString::new(fnc_name).unwrap();
            } else if fnc_name.starts_with("_RN4core9panicking14panic_explicit") {
                return CString::new(fnc_name).unwrap();
            }
        } else {
            break;
        }
    }
    panic!("Could not find panic function");
}
unsafe fn add_panic_msg_to_global<'a>(llmod: &'a llvm::Module, llcx: &'a llvm::Context) -> &'a llvm::Value {
    use llvm::*;

    // Convert the message to a CString
    let msg = "autodiff safety check failed!";
    let cmsg = CString::new(msg).unwrap();

    let msg_global_name = "ad_safety_msg".to_string();
    let cmsg_global_name = CString::new(msg_global_name).unwrap();

    // Get the length of the message
    let msg_len = msg.len();

    // Create the array type
    let i8_array_type = LLVMRustArrayType(LLVMInt8TypeInContext(llcx), msg_len as u64);

    // Create the string constant
    let string_const_val = LLVMConstStringInContext(llcx, cmsg.as_ptr() as *const i8, msg_len as u32, 0);

    // Create the array initializer
    let mut array_elems: Vec<_> = Vec::with_capacity(msg_len);
    for i in 0..msg_len {
        let char_value = LLVMConstInt(LLVMInt8TypeInContext(llcx), cmsg.as_bytes()[i] as u64, 0);
        array_elems.push(char_value);
    }
    let array_initializer = LLVMConstArray(LLVMInt8TypeInContext(llcx), array_elems.as_mut_ptr(), msg_len as u32);

    // Create the struct type
    let global_type = LLVMStructTypeInContext(llcx, [i8_array_type].as_mut_ptr(), 1, 0);

    // Create the struct initializer
    let struct_initializer = LLVMConstStructInContext(llcx, [array_initializer].as_mut_ptr(), 1, 0);

    // Add the global variable to the module
    let global_var = LLVMAddGlobal(llmod, global_type, cmsg_global_name.as_ptr() as *const i8);
    LLVMRustSetLinkage(global_var, Linkage::PrivateLinkage);
    LLVMSetInitializer(global_var, struct_initializer);

        //let msg_global_name = "ad_safety_msg".to_string();
        //let cmsg_global_name = CString::new(msg_global_name).unwrap();
        //let msg = "autodiff safety check failed!";
        //let cmsg = CString::new(msg).unwrap();
        //let msg_len = msg.len();
        //let i8_array_type = llvm::LLVMRustArrayType(llvm::LLVMInt8TypeInContext(llcx), msg_len as u64);
        //let global_type  = llvm::LLVMStructTypeInContext(llcx, [i8_array_type].as_mut_ptr(), 1, 0);
        //let string_const_val = llvm::LLVMConstStringInContext(llcx, cmsg.as_ptr() as *const c_char, msg_len as u32, 0);
        //let initializer = llvm::LLVMConstStructInContext(llcx, [string_const_val].as_mut_ptr(), 1, 0);
        //let global = llvm::LLVMAddGlobal(llmod, global_type, cmsg_global_name.as_ptr() as *const c_char);
        //llvm::LLVMRustSetLinkage(global, llvm::Linkage::PrivateLinkage);
        //llvm::LLVMSetInitializer(global, initializer);
        //llvm::LLVMSetUnnamedAddress(global, llvm::UnnamedAddr::Global);

        global_var
}

// As unsafe as it can be.
#[allow(unused_variables)]
#[allow(unused)]
pub(crate) unsafe fn enzyme_ad(
    llmod: &llvm::Module,
    llcx: &llvm::Context,
    diag_handler: &DiagCtxt,
    item: AutoDiffItem,
) -> Result<(), FatalError> {
    let autodiff_mode = item.attrs.mode;
    let rust_name = item.source;
    let rust_name2 = &item.target;

    let args_activity = item.attrs.input_activity.clone();
    let ret_activity: DiffActivity = item.attrs.ret_activity;

    // get target and source function
    let name = CString::new(rust_name.to_owned()).unwrap();
    let name2 = CString::new(rust_name2.clone()).unwrap();
    let src_fnc_opt = llvm::LLVMGetNamedFunction(llmod, name.as_c_str().as_ptr());
    let src_fnc = match src_fnc_opt {
        Some(x) => x,
        None => {
            return Err(llvm_err(
                diag_handler,
                LlvmError::PrepareAutoDiff {
                    src: rust_name.to_owned(),
                    target: rust_name2.to_owned(),
                    error: "could not find src function".to_owned(),
                },
            ));
        }
    };
    let target_fnc_opt = llvm::LLVMGetNamedFunction(llmod, name2.as_ptr());
    let target_fnc = match target_fnc_opt {
        Some(x) => x,
        None => {
            return Err(llvm_err(
                diag_handler,
                LlvmError::PrepareAutoDiff {
                    src: rust_name.to_owned(),
                    target: rust_name2.to_owned(),
                    error: "could not find target function".to_owned(),
                },
            ));
        }
    };
    let src_num_args = llvm::LLVMCountParams(src_fnc);
    let target_num_args = llvm::LLVMCountParams(target_fnc);
    // A really simple check
    assert!(src_num_args <= target_num_args);

    // create enzyme typetrees
    let llvm_data_layout = unsafe { llvm::LLVMGetDataLayoutStr(&*llmod) };
    let llvm_data_layout =
        std::str::from_utf8(unsafe { CStr::from_ptr(llvm_data_layout) }.to_bytes())
            .expect("got a non-UTF8 data-layout from LLVM");

    let input_tts =
        item.inputs.into_iter().map(|x| to_enzyme_typetree(x, llvm_data_layout, llcx)).collect();
    let output_tt = to_enzyme_typetree(item.output, llvm_data_layout, llcx);

    let mut fnc_opt = false;
    if std::env::var("ENZYME_ENABLE_FNC_OPT").is_ok() {
        dbg!("Disabling optimizations for Enzyme");
        fnc_opt = true;
    }

    let logic_ref: EnzymeLogicRef = CreateEnzymeLogic(fnc_opt as u8);
    let type_analysis: EnzymeTypeAnalysisRef =
        CreateTypeAnalysis(logic_ref, std::ptr::null_mut(), std::ptr::null_mut(), 0);

    llvm::set_strict_aliasing(false);

    if std::env::var("ENZYME_PRINT_TA").is_ok() {
        llvm::set_print_type(true);
    }
    if std::env::var("ENZYME_PRINT_AA").is_ok() {
        llvm::set_print_activity(true);
    }
    if std::env::var("ENZYME_PRINT_PERF").is_ok() {
        llvm::set_print_perf(true);
    }
    if std::env::var("ENZYME_PRINT").is_ok() {
        llvm::set_print(true);
    }

    let mut tmp = match item.attrs.mode {
        DiffMode::Forward => enzyme_rust_forward_diff(
            logic_ref,
            type_analysis,
            src_fnc,
            args_activity,
            ret_activity,
            input_tts,
            output_tt,
        ),
        DiffMode::Reverse => enzyme_rust_reverse_diff(
            logic_ref,
            type_analysis,
            src_fnc,
            args_activity,
            ret_activity,
            input_tts,
            output_tt,
        ),
        _ => unreachable!(),
    };
    let mut res: &Value = tmp.0;
    let size_positions: Vec<usize> = tmp.1;

    let f_return_type = LLVMGetReturnType(LLVMGlobalGetValueType(res));

    let void_type = LLVMVoidTypeInContext(llcx);
    let rev_mode = item.attrs.mode == DiffMode::Reverse;
    create_call(target_fnc, res, rev_mode, llmod, llcx, &size_positions);
    // TODO: implement drop for wrapper type?
    FreeTypeAnalysis(type_analysis);

    Ok(())
}


pub(crate) unsafe fn differentiate(
    module: &ModuleCodegen<ModuleLlvm>,
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    diff_items: Vec<AutoDiffItem>,
    _typetrees: FxHashMap<String, DiffTypeTree>,
    config: &ModuleConfig,
) -> Result<(), FatalError> {
    for item in &diff_items {
        trace!("{}", item);
    }

    let llmod = module.module_llvm.llmod();
    let llcx = &module.module_llvm.llcx;
    let diag_handler = cgcx.create_dcx();

    llvm::set_strict_aliasing(false);

    if std::env::var("ENZYME_LOOSE_TYPES").is_ok() {
        dbg!("Setting loose types to true");
        llvm::set_loose_types(true);
    }

    if std::env::var("ENZYME_PRINT_MOD").is_ok() {
        unsafe {
            LLVMDumpModule(llmod);
        }
    }
    if std::env::var("ENZYME_TT_DEPTH").is_ok() {
        let depth = std::env::var("ENZYME_TT_DEPTH").unwrap();
        let depth = depth.parse::<u64>().unwrap();
        assert!(depth >= 1);
        llvm::set_max_int_offset(depth);
    }
    if std::env::var("ENZYME_TT_WIDTH").is_ok() {
        let width = std::env::var("ENZYME_TT_WIDTH").unwrap();
        let width = width.parse::<u64>().unwrap();
        assert!(width >= 1);
        llvm::set_max_type_offset(width);
    }

    let differentiate = !diff_items.is_empty();
    for item in diff_items {
        let res = enzyme_ad(llmod, llcx, &diag_handler, item);
        assert!(res.is_ok());
    }

    let mut f = LLVMGetFirstFunction(llmod);
    loop {
        if let Some(lf) = f {
            f = LLVMGetNextFunction(lf);
            let myhwattr = "enzyme_hw";
            let attr = LLVMGetStringAttributeAtIndex(
                lf,
                c_uint::MAX,
                myhwattr.as_ptr() as *const c_char,
                myhwattr.as_bytes().len() as c_uint,
            );
            if LLVMIsStringAttribute(attr) {
                LLVMRemoveStringAttributeAtIndex(
                    lf,
                    c_uint::MAX,
                    myhwattr.as_ptr() as *const c_char,
                    myhwattr.as_bytes().len() as c_uint,
                );
            } else {
                LLVMRustRemoveEnumAttributeAtIndex(
                    lf,
                    c_uint::MAX,
                    AttributeKind::SanitizeHWAddress,
                );
            }
        } else {
            break;
        }
    }
    if std::env::var("ENZYME_PRINT_MOD_AFTER").is_ok() {
        unsafe {
            LLVMDumpModule(llmod);
        }
    }


    if std::env::var("ENZYME_NO_MOD_OPT_AFTER").is_ok() || !differentiate {
        trace!("Skipping module optimization after automatic differentiation");
    } else {
        if let Some(opt_level) = config.opt_level {
            let opt_stage = match cgcx.lto {
                Lto::Fat => llvm::OptStage::PreLinkFatLTO,
                Lto::Thin | Lto::ThinLocal => llvm::OptStage::PreLinkThinLTO,
                _ if cgcx.opts.cg.linker_plugin_lto.enabled() => llvm::OptStage::PreLinkThinLTO,
                _ => llvm::OptStage::PreLinkNoLTO,
            };
            let first_run = false;
            dbg!("Running Module Optimization after differentiation");
            llvm_optimize(cgcx, &diag_handler, module, config, opt_level, opt_stage, first_run)?;
        }
    }

    Ok(())
}

// Unsafe due to LLVM calls.
pub(crate) unsafe fn optimize(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    dcx: &DiagCtxt,
    module: &ModuleCodegen<ModuleLlvm>,
    config: &ModuleConfig,
) -> Result<(), FatalError> {
    let _timer = cgcx.prof.generic_activity_with_arg("LLVM_module_optimize", &*module.name);

    let llmod = module.module_llvm.llmod();
    let llcx = &*module.module_llvm.llcx;
    let _handlers = DiagnosticHandlers::new(cgcx, dcx, llcx, module, CodegenDiagnosticsStage::Opt);

    let module_name = module.name.clone();
    let module_name = Some(&module_name[..]);

    if config.emit_no_opt_bc {
        let out = cgcx.output_filenames.temp_path_ext("no-opt.bc", module_name);
        let out = path_to_c_string(&out);
        llvm::LLVMWriteBitcodeToFile(llmod, out.as_ptr());
    }

    // This code enables Enzyme to differentiate code containing Rust enums.
    // By adding the SanitizeHWAddress attribute we prevent LLVM from Optimizing
    // away the enums and allows Enzyme to understand why a value can be of different types in
    // different code sections. We remove this attribute after Enzyme is done, to not affect the
    // rest of the compilation.
    {
        let mut f = LLVMGetFirstFunction(llmod);
        loop {
            if let Some(lf) = f {
                f = LLVMGetNextFunction(lf);
                let myhwattr = "enzyme_hw";
                let myhwv = "";
                let prevattr = LLVMRustGetEnumAttributeAtIndex(
                    lf,
                    c_uint::MAX,
                    AttributeKind::SanitizeHWAddress,
                );
                if LLVMIsEnumAttribute(prevattr) {
                    let attr = LLVMCreateStringAttribute(
                        llcx,
                        myhwattr.as_ptr() as *const c_char,
                        myhwattr.as_bytes().len() as c_uint,
                        myhwv.as_ptr() as *const c_char,
                        myhwv.as_bytes().len() as c_uint,
                    );
                    LLVMRustAddFunctionAttributes(lf, c_uint::MAX, &attr, 1);
                } else {
                    LLVMRustAddEnumAttributeAtIndex(
                        llcx,
                        lf,
                        c_uint::MAX,
                        AttributeKind::SanitizeHWAddress,
                    );
                }
            } else {
                break;
            }
        }
    }

    if let Some(opt_level) = config.opt_level {
        let opt_stage = match cgcx.lto {
            Lto::Fat => llvm::OptStage::PreLinkFatLTO,
            Lto::Thin | Lto::ThinLocal => llvm::OptStage::PreLinkThinLTO,
            _ if cgcx.opts.cg.linker_plugin_lto.enabled() => llvm::OptStage::PreLinkThinLTO,
            _ => llvm::OptStage::PreLinkNoLTO,
        };
        // Second run only relevant for AD
        let first_run = true;
        return llvm_optimize(cgcx, dcx, module, config, opt_level, opt_stage, first_run);
    }
    Ok(())
}

pub(crate) fn link(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    dcx: &DiagCtxt,
    mut modules: Vec<ModuleCodegen<ModuleLlvm>>,
) -> Result<ModuleCodegen<ModuleLlvm>, FatalError> {
    use super::lto::{Linker, ModuleBuffer};
    // Sort the modules by name to ensure deterministic behavior.
    modules.sort_by(|a, b| a.name.cmp(&b.name));
    let (first, elements) =
        modules.split_first().expect("Bug! modules must contain at least one module.");

    let mut linker = Linker::new(first.module_llvm.llmod());
    for module in elements {
        let _timer = cgcx.prof.generic_activity_with_arg("LLVM_link_module", &*module.name);
        let buffer = ModuleBuffer::new(module.module_llvm.llmod());
        linker
            .add(buffer.data())
            .map_err(|()| llvm_err(dcx, LlvmError::SerializeModule { name: &module.name }))?;
    }
    drop(linker);
    Ok(modules.remove(0))
}

pub(crate) unsafe fn codegen(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    dcx: &DiagCtxt,
    module: ModuleCodegen<ModuleLlvm>,
    config: &ModuleConfig,
) -> Result<CompiledModule, FatalError> {
    let _timer = cgcx.prof.generic_activity_with_arg("LLVM_module_codegen", &*module.name);
    {
        let llmod = module.module_llvm.llmod();
        let llcx = &*module.module_llvm.llcx;
        let tm = &*module.module_llvm.tm;
        let module_name = module.name.clone();
        let module_name = Some(&module_name[..]);
        let _handlers =
            DiagnosticHandlers::new(cgcx, dcx, llcx, &module, CodegenDiagnosticsStage::Codegen);

        if cgcx.msvc_imps_needed {
            create_msvc_imps(cgcx, llcx, llmod);
        }

        // A codegen-specific pass manager is used to generate object
        // files for an LLVM module.
        //
        // Apparently each of these pass managers is a one-shot kind of
        // thing, so we create a new one for each type of output. The
        // pass manager passed to the closure should be ensured to not
        // escape the closure itself, and the manager should only be
        // used once.
        unsafe fn with_codegen<'ll, F, R>(
            tm: &'ll llvm::TargetMachine,
            llmod: &'ll llvm::Module,
            f: F,
        ) -> R
        where
            F: FnOnce(&'ll mut PassManager<'ll>) -> R,
        {
            let cpm = llvm::LLVMCreatePassManager();
            llvm::LLVMAddAnalysisPasses(tm, cpm);
            llvm::LLVMRustAddLibraryInfo(cpm, llmod);
            f(cpm)
        }

        // Two things to note:
        // - If object files are just LLVM bitcode we write bitcode, copy it to
        //   the .o file, and delete the bitcode if it wasn't otherwise
        //   requested.
        // - If we don't have the integrated assembler then we need to emit
        //   asm from LLVM and use `gcc` to create the object file.

        let bc_out = cgcx.output_filenames.temp_path(OutputType::Bitcode, module_name);
        let obj_out = cgcx.output_filenames.temp_path(OutputType::Object, module_name);

        if config.bitcode_needed() {
            let _timer = cgcx
                .prof
                .generic_activity_with_arg("LLVM_module_codegen_make_bitcode", &*module.name);
            let thin = ThinBuffer::new(llmod, config.emit_thin_lto);
            let data = thin.data();

            if let Some(bitcode_filename) = bc_out.file_name() {
                cgcx.prof.artifact_size(
                    "llvm_bitcode",
                    bitcode_filename.to_string_lossy(),
                    data.len() as u64,
                );
            }

            if config.emit_bc || config.emit_obj == EmitObj::Bitcode {
                let _timer = cgcx
                    .prof
                    .generic_activity_with_arg("LLVM_module_codegen_emit_bitcode", &*module.name);
                if let Err(err) = fs::write(&bc_out, data) {
                    dcx.emit_err(WriteBytecode { path: &bc_out, err });
                }
            }

            if config.emit_obj == EmitObj::ObjectCode(BitcodeSection::Full) {
                let _timer = cgcx
                    .prof
                    .generic_activity_with_arg("LLVM_module_codegen_embed_bitcode", &*module.name);
                embed_bitcode(cgcx, llcx, llmod, &config.bc_cmdline, data);
            }
        }

        if config.emit_ir {
            let _timer =
                cgcx.prof.generic_activity_with_arg("LLVM_module_codegen_emit_ir", &*module.name);
            let out = cgcx.output_filenames.temp_path(OutputType::LlvmAssembly, module_name);
            let out_c = path_to_c_string(&out);

            extern "C" fn demangle_callback(
                input_ptr: *const c_char,
                input_len: size_t,
                output_ptr: *mut c_char,
                output_len: size_t,
            ) -> size_t {
                let input =
                    unsafe { slice::from_raw_parts(input_ptr as *const u8, input_len as usize) };

                let Ok(input) = str::from_utf8(input) else { return 0 };

                let output = unsafe {
                    slice::from_raw_parts_mut(output_ptr as *mut u8, output_len as usize)
                };
                let mut cursor = io::Cursor::new(output);

                let Ok(demangled) = rustc_demangle::try_demangle(input) else { return 0 };

                if write!(cursor, "{demangled:#}").is_err() {
                    // Possible only if provided buffer is not big enough
                    return 0;
                }

                cursor.position() as size_t
            }

            let result = llvm::LLVMRustPrintModule(llmod, out_c.as_ptr(), demangle_callback);

            if result == llvm::LLVMRustResult::Success {
                record_artifact_size(&cgcx.prof, "llvm_ir", &out);
            }

            result.into_result().map_err(|()| llvm_err(dcx, LlvmError::WriteIr { path: &out }))?;
        }

        if config.emit_asm {
            let _timer =
                cgcx.prof.generic_activity_with_arg("LLVM_module_codegen_emit_asm", &*module.name);
            let path = cgcx.output_filenames.temp_path(OutputType::Assembly, module_name);

            // We can't use the same module for asm and object code output,
            // because that triggers various errors like invalid IR or broken
            // binaries. So we must clone the module to produce the asm output
            // if we are also producing object code.
            let llmod = if let EmitObj::ObjectCode(_) = config.emit_obj {
                llvm::LLVMCloneModule(llmod)
            } else {
                llmod
            };
            with_codegen(tm, llmod, |cpm| {
                write_output_file(
                    dcx,
                    tm,
                    cpm,
                    llmod,
                    &path,
                    None,
                    llvm::FileType::AssemblyFile,
                    &cgcx.prof,
                )
            })?;
        }

        match config.emit_obj {
            EmitObj::ObjectCode(_) => {
                let _timer = cgcx
                    .prof
                    .generic_activity_with_arg("LLVM_module_codegen_emit_obj", &*module.name);

                let dwo_out = cgcx.output_filenames.temp_path_dwo(module_name);
                let dwo_out = match (cgcx.split_debuginfo, cgcx.split_dwarf_kind) {
                    // Don't change how DWARF is emitted when disabled.
                    (SplitDebuginfo::Off, _) => None,
                    // Don't provide a DWARF object path if split debuginfo is enabled but this is
                    // a platform that doesn't support Split DWARF.
                    _ if !cgcx.target_can_use_split_dwarf => None,
                    // Don't provide a DWARF object path in single mode, sections will be written
                    // into the object as normal but ignored by linker.
                    (_, SplitDwarfKind::Single) => None,
                    // Emit (a subset of the) DWARF into a separate dwarf object file in split
                    // mode.
                    (_, SplitDwarfKind::Split) => Some(dwo_out.as_path()),
                };

                with_codegen(tm, llmod, |cpm| {
                    write_output_file(
                        dcx,
                        tm,
                        cpm,
                        llmod,
                        &obj_out,
                        dwo_out,
                        llvm::FileType::ObjectFile,
                        &cgcx.prof,
                    )
                })?;
            }

            EmitObj::Bitcode => {
                debug!("copying bitcode {:?} to obj {:?}", bc_out, obj_out);
                if let Err(err) = link_or_copy(&bc_out, &obj_out) {
                    dcx.emit_err(CopyBitcode { err });
                }

                if !config.emit_bc {
                    debug!("removing_bitcode {:?}", bc_out);
                    ensure_removed(dcx, &bc_out);
                }
            }

            EmitObj::None => {}
        }

        record_llvm_cgu_instructions_stats(&cgcx.prof, llmod);
    }

    // `.dwo` files are only emitted if:
    //
    // - Object files are being emitted (i.e. bitcode only or metadata only compilations will not
    //   produce dwarf objects, even if otherwise enabled)
    // - Target supports Split DWARF
    // - Split debuginfo is enabled
    // - Split DWARF kind is `split` (i.e. debuginfo is split into `.dwo` files, not different
    //   sections in the `.o` files).
    let dwarf_object_emitted = matches!(config.emit_obj, EmitObj::ObjectCode(_))
        && cgcx.target_can_use_split_dwarf
        && cgcx.split_debuginfo != SplitDebuginfo::Off
        && cgcx.split_dwarf_kind == SplitDwarfKind::Split;
    Ok(module.into_compiled_module(
        config.emit_obj != EmitObj::None,
        dwarf_object_emitted,
        config.emit_bc,
        &cgcx.output_filenames,
    ))
}

fn create_section_with_flags_asm(section_name: &str, section_flags: &str, data: &[u8]) -> Vec<u8> {
    let mut asm = format!(".section {section_name},\"{section_flags}\"\n").into_bytes();
    asm.extend_from_slice(b".ascii \"");
    asm.reserve(data.len());
    for &byte in data {
        if byte == b'\\' || byte == b'"' {
            asm.push(b'\\');
            asm.push(byte);
        } else if byte < 0x20 || byte >= 0x80 {
            // Avoid non UTF-8 inline assembly. Use octal escape sequence, because it is fixed
            // width, while hex escapes will consume following characters.
            asm.push(b'\\');
            asm.push(b'0' + ((byte >> 6) & 0x7));
            asm.push(b'0' + ((byte >> 3) & 0x7));
            asm.push(b'0' + ((byte >> 0) & 0x7));
        } else {
            asm.push(byte);
        }
    }
    asm.extend_from_slice(b"\"\n");
    asm
}

fn target_is_apple(cgcx: &CodegenContext<LlvmCodegenBackend>) -> bool {
    cgcx.opts.target_triple.triple().contains("-ios")
        || cgcx.opts.target_triple.triple().contains("-darwin")
        || cgcx.opts.target_triple.triple().contains("-tvos")
        || cgcx.opts.target_triple.triple().contains("-watchos")
}

fn target_is_aix(cgcx: &CodegenContext<LlvmCodegenBackend>) -> bool {
    cgcx.opts.target_triple.triple().contains("-aix")
}

//FIXME use c string literals here too
pub(crate) fn bitcode_section_name(cgcx: &CodegenContext<LlvmCodegenBackend>) -> &'static str {
    if target_is_apple(cgcx) {
        "__LLVM,__bitcode\0"
    } else if target_is_aix(cgcx) {
        ".ipa\0"
    } else {
        ".llvmbc\0"
    }
}

/// Embed the bitcode of an LLVM module in the LLVM module itself.
///
/// This is done primarily for iOS where it appears to be standard to compile C
/// code at least with `-fembed-bitcode` which creates two sections in the
/// executable:
///
/// * __LLVM,__bitcode
/// * __LLVM,__cmdline
///
/// It appears *both* of these sections are necessary to get the linker to
/// recognize what's going on. A suitable cmdline value is taken from the
/// target spec.
///
/// Furthermore debug/O1 builds don't actually embed bitcode but rather just
/// embed an empty section.
///
/// Basically all of this is us attempting to follow in the footsteps of clang
/// on iOS. See #35968 for lots more info.
unsafe fn embed_bitcode(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    llcx: &llvm::Context,
    llmod: &llvm::Module,
    cmdline: &str,
    bitcode: &[u8],
) {
    // We're adding custom sections to the output object file, but we definitely
    // do not want these custom sections to make their way into the final linked
    // executable. The purpose of these custom sections is for tooling
    // surrounding object files to work with the LLVM IR, if necessary. For
    // example rustc's own LTO will look for LLVM IR inside of the object file
    // in these sections by default.
    //
    // To handle this is a bit different depending on the object file format
    // used by the backend, broken down into a few different categories:
    //
    // * Mach-O - this is for macOS. Inspecting the source code for the native
    //   linker here shows that the `.llvmbc` and `.llvmcmd` sections are
    //   automatically skipped by the linker. In that case there's nothing extra
    //   that we need to do here.
    //
    // * Wasm - the native LLD linker is hard-coded to skip `.llvmbc` and
    //   `.llvmcmd` sections, so there's nothing extra we need to do.
    //
    // * COFF - if we don't do anything the linker will by default copy all
    //   these sections to the output artifact, not what we want! To subvert
    //   this we want to flag the sections we inserted here as
    //   `IMAGE_SCN_LNK_REMOVE`.
    //
    // * ELF - this is very similar to COFF above. One difference is that these
    //   sections are removed from the output linked artifact when
    //   `--gc-sections` is passed, which we pass by default. If that flag isn't
    //   passed though then these sections will show up in the final output.
    //   Additionally the flag that we need to set here is `SHF_EXCLUDE`.
    //
    // * XCOFF - AIX linker ignores content in .ipa and .info if no auxiliary
    //   symbol associated with these sections.
    //
    // Unfortunately, LLVM provides no way to set custom section flags. For ELF
    // and COFF we emit the sections using module level inline assembly for that
    // reason (see issue #90326 for historical background).
    let is_aix = target_is_aix(cgcx);
    let is_apple = target_is_apple(cgcx);
    if is_apple || is_aix || cgcx.opts.target_triple.triple().starts_with("wasm") {
        // We don't need custom section flags, create LLVM globals.
        let llconst = common::bytes_in_context(llcx, bitcode);
        let llglobal = llvm::LLVMAddGlobal(
            llmod,
            common::val_ty(llconst),
            c"rustc.embedded.module".as_ptr().cast(),
        );
        llvm::LLVMSetInitializer(llglobal, llconst);

        let section = bitcode_section_name(cgcx);
        llvm::LLVMSetSection(llglobal, section.as_ptr().cast());
        llvm::LLVMRustSetLinkage(llglobal, llvm::Linkage::PrivateLinkage);
        llvm::LLVMSetGlobalConstant(llglobal, llvm::True);

        let llconst = common::bytes_in_context(llcx, cmdline.as_bytes());
        let llglobal = llvm::LLVMAddGlobal(
            llmod,
            common::val_ty(llconst),
            c"rustc.embedded.cmdline".as_ptr().cast(),
        );
        llvm::LLVMSetInitializer(llglobal, llconst);
        let section = if is_apple {
            c"__LLVM,__cmdline"
        } else if is_aix {
            c".info"
        } else {
            c".llvmcmd"
        };
        llvm::LLVMSetSection(llglobal, section.as_ptr().cast());
        llvm::LLVMRustSetLinkage(llglobal, llvm::Linkage::PrivateLinkage);
    } else {
        // We need custom section flags, so emit module-level inline assembly.
        let section_flags = if cgcx.is_pe_coff { "n" } else { "e" };
        let asm = create_section_with_flags_asm(".llvmbc", section_flags, bitcode);
        llvm::LLVMAppendModuleInlineAsm(llmod, asm.as_ptr().cast(), asm.len());
        let asm = create_section_with_flags_asm(".llvmcmd", section_flags, cmdline.as_bytes());
        llvm::LLVMAppendModuleInlineAsm(llmod, asm.as_ptr().cast(), asm.len());
    }
}

// Create a `__imp_<symbol> = &symbol` global for every public static `symbol`.
// This is required to satisfy `dllimport` references to static data in .rlibs
// when using MSVC linker. We do this only for data, as linker can fix up
// code references on its own.
// See #26591, #27438
fn create_msvc_imps(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    llcx: &llvm::Context,
    llmod: &llvm::Module,
) {
    if !cgcx.msvc_imps_needed {
        return;
    }
    // The x86 ABI seems to require that leading underscores are added to symbol
    // names, so we need an extra underscore on x86. There's also a leading
    // '\x01' here which disables LLVM's symbol mangling (e.g., no extra
    // underscores added in front).
    let prefix = if cgcx.target_arch == "x86" { "\x01__imp__" } else { "\x01__imp_" };

    unsafe {
        let ptr_ty = Type::ptr_llcx(llcx);
        let globals = base::iter_globals(llmod)
            .filter(|&val| {
                llvm::LLVMRustGetLinkage(val) == llvm::Linkage::ExternalLinkage
                    && llvm::LLVMIsDeclaration(val) == 0
            })
            .filter_map(|val| {
                // Exclude some symbols that we know are not Rust symbols.
                let name = llvm::get_value_name(val);
                if ignored(name) { None } else { Some((val, name)) }
            })
            .map(move |(val, name)| {
                let mut imp_name = prefix.as_bytes().to_vec();
                imp_name.extend(name);
                let imp_name = CString::new(imp_name).unwrap();
                (imp_name, val)
            })
            .collect::<Vec<_>>();

        for (imp_name, val) in globals {
            let imp = llvm::LLVMAddGlobal(llmod, ptr_ty, imp_name.as_ptr().cast());
            llvm::LLVMSetInitializer(imp, val);
            llvm::LLVMRustSetLinkage(imp, llvm::Linkage::ExternalLinkage);
        }
    }

    // Use this function to exclude certain symbols from `__imp` generation.
    fn ignored(symbol_name: &[u8]) -> bool {
        // These are symbols generated by LLVM's profiling instrumentation
        symbol_name.starts_with(b"__llvm_profile_")
    }
}

fn record_artifact_size(
    self_profiler_ref: &SelfProfilerRef,
    artifact_kind: &'static str,
    path: &Path,
) {
    // Don't stat the file if we are not going to record its size.
    if !self_profiler_ref.enabled() {
        return;
    }

    if let Some(artifact_name) = path.file_name() {
        let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        self_profiler_ref.artifact_size(artifact_kind, artifact_name.to_string_lossy(), file_size);
    }
}

fn record_llvm_cgu_instructions_stats(prof: &SelfProfilerRef, llmod: &llvm::Module) {
    if !prof.enabled() {
        return;
    }

    let raw_stats =
        llvm::build_string(|s| unsafe { llvm::LLVMRustModuleInstructionStats(llmod, s) })
            .expect("cannot get module instruction stats");

    #[derive(serde::Deserialize)]
    struct InstructionsStats {
        module: String,
        total: u64,
    }

    let InstructionsStats { module, total } =
        serde_json::from_str(&raw_stats).expect("cannot parse llvm cgu instructions stats");
    prof.artifact_size("cgu_instructions", module, total);
}
