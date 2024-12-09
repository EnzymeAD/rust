#![allow(non_camel_case_types)]

use libc::{c_char, c_uint, size_t};
use super::ffi::*;

extern "C" {
    // Enzyme
    pub fn LLVMRustAddFncParamAttr<'a>(F: &'a Value, index: c_uint, Attr: &'a Attribute);
    pub fn LLVMRustAddRetFncAttr(F: &Value, attr: &Attribute);
    pub fn LLVMRustHasMetadata(I: &Value, KindID: c_uint) -> bool;
    pub fn LLVMRustEraseInstBefore(BB: &BasicBlock, I: &Value);
    pub fn LLVMRustGetLastInstruction<'a>(BB: &BasicBlock) -> Option<&'a Value>;
    pub fn LLVMRustDIGetInstMetadata(I: &Value) -> &Metadata;
    pub fn LLVMRustEraseInstFromParent(V: &Value);
    pub fn LLVMRustGetTerminator<'a>(B: &BasicBlock) -> &'a Value;
    pub fn LLVMRustAddEnumAttributeAtIndex(
        C: &Context,
        V: &Value,
        index: c_uint,
        attr: AttributeKind,
    );
    pub fn LLVMRustRemoveEnumAttributeAtIndex(V: &Value, index: c_uint, attr: AttributeKind);
    pub fn LLVMRustGetEnumAttributeAtIndex(
        V: &Value,
        index: c_uint,
        attr: AttributeKind,
    ) -> &Attribute;
    pub fn LLVMRustAddParamAttr<'a>(Instr: &'a Value, index: c_uint, Attr: &'a Attribute);


    pub fn LLVMGetReturnType(T: &Type) -> &Type;
    pub fn LLVMDumpModule(M: &Module);
    pub fn LLVMCountStructElementTypes(T: &Type) -> c_uint;
    pub fn LLVMVerifyFunction(V: &Value, action: LLVMVerifierFailureAction) -> bool;
    pub fn LLVMGetParams(Fnc: &Value, parms: *mut &Value);
    pub fn LLVMBuildCall2<'a>(
        arg1: &Builder<'a>,
        ty: &Type,
        func: &Value,
        args: *mut &Value,
        num_args: size_t,
        name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMGetFirstFunction(M: &Module) -> Option<&Value>;
    pub fn LLVMGetNextFunction(V: &Value) -> Option<&Value>;
    pub fn LLVMGetNamedFunction(M: &Module, Name: *const c_char) -> Option<&Value>;
    pub fn LLVMRustGetFunctionType(fnc: &Value) -> &Type;

    pub fn LLVMRemoveStringAttributeAtIndex(F: &Value, Idx: c_uint, K: *const c_char, KLen: c_uint);
    pub fn LLVMGetStringAttributeAtIndex(
        F: &Value,
        Idx: c_uint,
        K: *const c_char,
        KLen: c_uint,
    ) -> &Attribute;
    pub fn LLVMIsEnumAttribute(A: &Attribute) -> bool;
    pub fn LLVMIsStringAttribute(A: &Attribute) -> bool;

}

#[repr(C)]
pub enum LLVMVerifierFailureAction {
    LLVMAbortProcessAction,
    LLVMPrintMessageAction,
    LLVMReturnStatusAction,
}
