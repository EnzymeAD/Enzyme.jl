module API

import LLVM.API: LLVMValueRef, LLVMModuleRef, LLVMTypeRef, LLVMContextRef
using Enzyme_jll
using Libdl
using LLVM
using CEnum

const EnzymeLogicRef = Ptr{Cvoid}
const EnzymeTypeAnalysisRef = Ptr{Cvoid}
const EnzymeAugmentedReturnPtr = Ptr{Cvoid}

const UP = Cint(1)
const DOWN = Cint(2)
const BOTH = Cint(3)

struct IntList
    data::Ptr{Int64}
    size::Csize_t
end
IntList() = IntList(Ptr{Int64}(0),0)

@cenum(CConcreteType,
  DT_Anything = 0,
  DT_Integer = 1,
  DT_Pointer = 2,
  DT_Half = 3,
  DT_Float = 4,
  DT_Double = 5,
  DT_Unknown = 6
)

struct EnzymeTypeTree end
const CTypeTreeRef = Ptr{EnzymeTypeTree}

EnzymeNewTypeTree() = ccall((:EnzymeNewTypeTree, libEnzyme), CTypeTreeRef, ())
EnzymeNewTypeTreeCT(T, ctx) = ccall((:EnzymeNewTypeTreeCT, libEnzyme), CTypeTreeRef, (CConcreteType, LLVMContextRef), T, ctx)
EnzymeNewTypeTreeTR(tt) = ccall((:EnzymeNewTypeTreeTR, libEnzyme), CTypeTreeRef, (CTypeTreeRef,), tt)

EnzymeFreeTypeTree(tt) = ccall((:EnzymeFreeTypeTree, libEnzyme), Cvoid, (CTypeTreeRef,), tt)
EnzymeSetTypeTree(dst, src) = ccall((:EnzymeSetTypeTree, libEnzyme), UInt8, (CTypeTreeRef, CTypeTreeRef), dst, src)
EnzymeMergeTypeTree(dst, src) = ccall((:EnzymeMergeTypeTree, libEnzyme), UInt8, (CTypeTreeRef, CTypeTreeRef), dst, src)
EnzymeTypeTreeOnlyEq(dst, x) = ccall((:EnzymeTypeTreeOnlyEq, libEnzyme), Cvoid, (CTypeTreeRef, Int64), dst, x)
EnzymeTypeTreeShiftIndiciesEq(dst, dl, offset, maxSize, addOffset) =
    ccall((:EnzymeTypeTreeShiftIndiciesEq, libEnzyme), Cvoid, (CTypeTreeRef, Cstring, Int64, Int64, UInt64),
        dst, dl, offset, maxSize, addOffset)


EnzymeTypeTreeToString(tt) = ccall((:EnzymeTypeTreeToString, libEnzyme), Cstring, (CTypeTreeRef,), tt)
EnzymeTypeTreeToStringFree(str) = ccall((:EnzymeTypeTreeToStringFree, libEnzyme), Cvoid, (Cstring,), str)

struct CFnTypeInfo
    arguments::Ptr{CTypeTreeRef}
    ret::CTypeTreeRef

    known_values::Ptr{IntList}
end

@cenum(CDIFFE_TYPE,
  DFT_OUT_DIFF = 0,  # add differential to an output struct
  DFT_DUP_ARG = 1,   # duplicate the argument and store differential inside
  DFT_CONSTANT = 2,  # no differential
  DFT_DUP_NONEED = 3 # duplicate this argument and store differential inside,
                     # but don't need the forward
)

@cenum(CDerivativeMode,
  DEM_ForwardMode = 0,
  DEM_ReverseModePrimal = 1,
  DEM_ReverseModeGradient = 2,
  DEM_ReverseModeCombined = 3
)

# Create the derivative function itself.
#  \p todiff is the function to differentiate
#  \p retType is the activity info of the return
#  \p constant_args is the activity info of the arguments
#  \p returnValue is whether the primal's return should also be returned
#  \p dretUsed is whether the shadow return value should also be returned
#  \p additionalArg is the type (or null) of an additional type in the signature
#  to hold the tape.
#  \p typeInfo is the type info information about the calling context
#  \p _uncacheable_args marks whether an argument may be rewritten before loads in
#  the generated function (and thus cannot be cached).
#  \p augmented is the data structure created by prior call to an augmented forward
#  pass
#  \p AtomicAdd is whether to perform all adjoint updates to memory in an atomic way
#  \p PostOpt is whether to perform basic optimization of the function after synthesis
function EnzymeCreatePrimalAndGradient(logic, todiff, retType, constant_args, TA, 
                                       returnValue, dretUsed, topLevel, additionalArg, typeInfo,
                                       uncacheable_args, augmented, atomicAdd, postOpt)
    mode = topLevel ? DEM_ReverseModeCombined : DEM_ReverseModeGradient
    ccall((:EnzymeCreatePrimalAndGradient, libEnzyme), LLVMValueRef, 
        (EnzymeLogicRef, LLVMValueRef, CDIFFE_TYPE, Ptr{CDIFFE_TYPE}, Csize_t,
         EnzymeTypeAnalysisRef, UInt8, UInt8, CDerivativeMode, LLVMTypeRef, CFnTypeInfo,
         Ptr{UInt8}, Csize_t, EnzymeAugmentedReturnPtr, UInt8, UInt8),
        logic, todiff, retType, constant_args, length(constant_args), TA, returnValue,
        dretUsed, mode, additionalArg, typeInfo, uncacheable_args, length(uncacheable_args),
        augmented, atomicAdd, postOpt)
end

# Create an augmented forward pass.
#  \p todiff is the function to differentiate
#  \p retType is the activity info of the return
#  \p constant_args is the activity info of the arguments
#  \p returnUsed is whether the primal's return should also be returned
#  \p typeInfo is the type info information about the calling context
#  \p _uncacheable_args marks whether an argument may be rewritten before loads in
#  the generated function (and thus cannot be cached).
#  \p forceAnonymousTape forces the tape to be an i8* rather than the true tape structure
#  \p AtomicAdd is whether to perform all adjoint updates to memory in an atomic way
#  \p PostOpt is whether to perform basic optimization of the function after synthesis
function EnzymeCreateAugmentedPrimal(logic, todiff, retType, constant_args, TA,  returnUsed,
                                     typeInfo, uncacheable_args, forceAnonymousTape, atomicAdd, postOpt)
    ccall((:EnzymeCreateAugmentedPrimal, libEnzyme), EnzymeAugmentedReturnPtr, 
        (EnzymeLogicRef, LLVMValueRef, CDIFFE_TYPE, Ptr{CDIFFE_TYPE}, Csize_t, 
         EnzymeTypeAnalysisRef, UInt8, 
         CFnTypeInfo, Ptr{UInt8}, Csize_t, UInt8, UInt8, UInt8),
        logic, todiff, retType, constant_args, length(constant_args), TA,  returnUsed,
        typeInfo, uncacheable_args, length(uncacheable_args), forceAnonymousTape, atomicAdd, postOpt)
end

# typedef uint8_t (*CustomRuleType)(int /*direction*/, CTypeTreeRef /*return*/,
#                                   CTypeTreeRef * /*args*/,
#                                   struct IntList * /*knownValues*/,
#                                   size_t /*numArgs*/, LLVMValueRef);
const CustomRuleType = Ptr{Cvoid}

function CreateTypeAnalysis(triple, rulenames, rules)
    @assert length(rulenames) == length(rules)
    ccall((:CreateTypeAnalysis, libEnzyme), EnzymeTypeAnalysisRef, (Cstring, Ptr{Cstring}, Ptr{CustomRuleType}, Csize_t), triple, rulenames, rules, length(rules))
end

function ClearTypeAnalysis(ta)
    ccall((:ClearTypeAnalysis, libEnzyme), Cvoid, (EnzymeTypeAnalysisRef,), ta)
end

function FreeTypeAnalysis(ta)
    ccall((:FreeTypeAnalysis, libEnzyme), Cvoid, (EnzymeTypeAnalysisRef,), ta)
end

const CustomShadowAlloc = Ptr{Cvoid}
const CustomShadowFree = Ptr{Cvoid}
EnzymeRegisterAllocationHandler(name, ahandle, fhandle) = ccall((:EnzymeRegisterAllocationHandler, libEnzyme), Cvoid, (Cstring, CustomShadowAlloc, CustomShadowFree), name, ahandle, fhandle)


const CustomForwardPass = Ptr{Cvoid}
const CustomReversePass = Ptr{Cvoid}
EnzymeRegisterCallHandler(name, fwdhandle, revhandle) = ccall((:EnzymeRegisterCallHandler, libEnzyme), Cvoid, (Cstring, CustomForwardPass, CustomReversePass), name, fwdhandle, revhandle)

const EnzymeGradientUtilsRef = Ptr{Cvoid}

EnzymeGradientUtilsNewFromOriginal(gutils, val) = ccall((:EnzymeGradientUtilsNewFromOriginal, libEnzyme), LLVMValueRef, (EnzymeGradientUtilsRef, LLVMValueRef), gutils, val)
EnzymeGradientUtilsSetDebugLocFromOriginal(gutils, val, orig) = ccall((:EnzymeGradientUtilsSetDebugLocFromOriginal, libEnzyme), Cvoid, (EnzymeGradientUtilsRef, LLVMValueRef, LLVMValueRef), gutils, val, orig)
EnzymeGradientUtilsLookup(gutils, val, B) = ccall((:EnzymeGradientUtilsLookup, libEnzyme), LLVMValueRef, (EnzymeGradientUtilsRef, LLVMValueRef, LLVM.API.LLVMBuilderRef), gutils, val, B)
EnzymeGradientUtilsInvertPointer(gutils, val, B) = ccall((:EnzymeGradientUtilsInvertPointer, libEnzyme), LLVMValueRef, (EnzymeGradientUtilsRef, LLVMValueRef, LLVM.API.LLVMBuilderRef), gutils, val, B)
EnzymeGradientUtilsDiffe(gutils, val, B) = ccall((:EnzymeGradientUtilsDiffe, libEnzyme), LLVMValueRef, (EnzymeGradientUtilsRef, LLVMValueRef, LLVM.API.LLVMBuilderRef), gutils, val, B)
EnzymeGradientUtilsAddToDiffe(gutils, val, diffe, B, T) = ccall((:EnzymeGradientUtilsAddToDiffe, libEnzyme), Cvoid, (EnzymeGradientUtilsRef, LLVMValueRef, LLVMValueRef, LLVM.API.LLVMBuilderRef, LLVMTypeRef), gutils, val, diffe, B, T)
EnzymeGradientUtilsSetDiffe(gutils, val, diffe, B) = ccall((:EnzymeGradientUtilsSetDiffe, libEnzyme), Cvoid, (EnzymeGradientUtilsRef, LLVMValueRef, LLVMValueRef, LLVM.API.LLVMBuilderRef), gutils, val, diffe, B)
EnzymeGradientUtilsIsConstantValue(gutils, val) = ccall((:EnzymeGradientUtilsIsConstantValue, libEnzyme), UInt8, (EnzymeGradientUtilsRef, LLVMValueRef), gutils, val)
EnzymeGradientUtilsIsConstantInstruction(gutils, val) = ccall((:EnzymeGradientUtilsIsConstantInstruction, libEnzyme), UInt8, (EnzymeGradientUtilsRef, LLVMValueRef), gutils, val)
EnzymeGradientUtilsAllocationBlock(gutils) = ccall((:EnzymeGradientUtilsAllocationBlock, libEnzyme), LLVM.API.LLVMBasicBlockRef, (EnzymeGradientUtilsRef,), gutils)

function CreateLogic()
    ccall((:CreateEnzymeLogic, libEnzyme), EnzymeLogicRef, ())
end

function ClearLogic(logic)
    ccall((:ClearEnzymeLogic, libEnzyme), Cvoid, (EnzymeLogicRef,), logic)
end

function FreeLogic(logic)
    ccall((:FreeEnzymeLogic, libEnzyme), Cvoid, (EnzymeLogicRef,), logic)
end

function EnzymeExtractReturnInfo(ret, data, existed)
    @assert length(data) == length(existed)
    ccall((:EnzymeExtractReturnInfo, libEnzyme),
           Cvoid, (EnzymeAugmentedReturnPtr, Ptr{Int64}, Ptr{UInt8}, Csize_t),
           ret, data, existed, length(data))
end

function EnzymeExtractFunctionFromAugmentation(ret)
    ccall((:EnzymeExtractFunctionFromAugmentation, libEnzyme), LLVMValueRef, (EnzymeAugmentedReturnPtr,), ret)
end


function EnzymeExtractTapeTypeFromAugmentation(ret)
    ccall((:EnzymeExtractTapeTypeFromAugmentation, libEnzyme), LLVMTypeRef, (EnzymeAugmentedReturnPtr,), ret)
end

import Libdl
function EnzymeSetCLBool(name, val)
    handle = Libdl.dlopen(libEnzyme)
    ptr = Libdl.dlsym(handle, name)
    ccall((:EnzymeSetCLBool, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end
function EnzymeGetCLBool(name)
    handle = Libdl.dlopen(libEnzyme)
    ptr = Libdl.dlsym(handle, name)
    ccall((:EnzymeGetCLBool, libEnzyme), UInt8, (Ptr{Cvoid},), ptr)
end
# void EnzymeSetCLInteger(void *, int64_t);

function printperf!(val)
    ptr = cglobal((:EnzymePrintPerf, libEnzyme))
    ccall((:EnzymeSetCLBool, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end

function printtype!(val)
    ptr = cglobal((:EnzymePrintType, libEnzyme))
    ccall((:EnzymeSetCLBool, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end

function printactivity!(val)
    ptr = cglobal((:EnzymePrintActivity, libEnzyme))
    ccall((:EnzymeSetCLBool, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end

function printall!(val)
    ptr = cglobal((:EnzymePrint, libEnzyme))
    ccall((:EnzymeSetCLBool, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end

function inlineall!(val)
    ptr = cglobal((:EnzymeInline, libEnzyme))
    ccall((:EnzymeSetCLBool, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end

function maxtypeoffset!(val)
    ptr = cglobal((:MaxTypeOffset, libEnzyme))
    ccall((:EnzymeSetCLInteger, libEnzyme), Cvoid, (Ptr{Cvoid}, Int64), ptr, val)
end

function looseTypeAnalysis!(val)
    ptr = cglobal((:looseTypeAnalysis, libEnzyme))
    ccall((:EnzymeSetCLInteger, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end

function instname!(val)
    ptr = cglobal((:EnzymeNameInstructions, libEnzyme))
    ccall((:EnzymeSetCLBool, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end

function EnzymeRemoveTrivialAtomicIncrements(func)
    ccall((:EnzymeRemoveTrivialAtomicIncrements, libEnzyme), Cvoid, (LLVMValueRef,), func)
end

function EnzymeAddAttributorLegacyPass(PM)
    ccall((:EnzymeAddAttributorLegacyPass, libEnzyme),Cvoid,(LLVM.API.LLVMPassManagerRef,), PM)
end

function __init__()
    ptr = cglobal((:EnzymeJuliaAddrLoad, libEnzyme))
    val = true
    ccall((:EnzymeSetCLBool, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end

function moveBefore(i1, i2)
    ccall((:EnzymeMoveBefore, libEnzyme),Cvoid,(LLVM.API.LLVMValueRef,LLVM.API.LLVMValueRef), i1, i2)
end

end
