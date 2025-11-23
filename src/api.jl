module API

import LLVM.API: LLVMValueRef, LLVMModuleRef, LLVMTypeRef, LLVMContextRef
using Enzyme_jll
using EnzymeCore
using Libdl
using LLVM
using CEnum

const EnzymeLogicRef = Ptr{Cvoid}
const EnzymeTypeAnalysisRef = Ptr{Cvoid}
const EnzymeAugmentedReturnPtr = Ptr{Cvoid}
const EnzymeTypeAnalyzerRef = Ptr{Cvoid}
const EnzymeGradientUtilsRef = Ptr{Cvoid}

const UP = Cint(1)
const DOWN = Cint(2)
const BOTH = Cint(3)

struct IntList
    data::Ptr{Int64}
    size::Csize_t
end
IntList() = IntList(Ptr{Int64}(0), 0)

@cenum(
    CConcreteType,
    DT_Anything = 0,
    DT_Integer = 1,
    DT_Pointer = 2,
    DT_Half = 3,
    DT_Float = 4,
    DT_Double = 5,
    DT_Unknown = 6,
    DT_FP80 = 7,
    DT_BFloat16 = 8
)

function EnzymeConcreteTypeIsFloat(cc::CConcreteType)
    if cc == DT_Half
        return LLVM.HalfType()
    elseif cc == DT_Float
        return LLVM.FloatType()
    elseif cc == DT_Double
        return LLVM.DoubleType()
    elseif cc == DT_FP80
        return LLVM.X86FP80Type()
    elseif cc == DT_BFloat16
        return LLVM.BFloatType()
    else
        return nothing
    end
end

@cenum(CValueType, VT_None = 0, VT_Primal = 1, VT_Shadow = 2, VT_Both = 3)

function EnzymeBitcodeReplacement(mod, NotToReplace, found)
    foundSize = Ref{Csize_t}(0)
    foundP = Ref{Ptr{Cstring}}(C_NULL)
    res = ccall(
        (:EnzymeBitcodeReplacement, libEnzymeBCLoad),
        UInt8,
        (LLVM.API.LLVMModuleRef, Ptr{Cstring}, Csize_t, Ptr{Ptr{Cstring}}, Ptr{Csize_t}),
        mod,
        NotToReplace,
        length(NotToReplace),
        foundP,
        foundSize,
    )
    foundNum = foundSize[]
    if foundNum != 0
        foundP = foundP[]
        for i = 1:foundNum
            str = unsafe_load(foundP, i)
            push!(found, Base.unsafe_string(str))
            Libc.free(str)

        end
        Libc.free(foundP)
    end
    return res
end

struct EnzymeTypeTree end
const CTypeTreeRef = Ptr{EnzymeTypeTree}

EnzymeNewTypeTree() = ccall((:EnzymeNewTypeTree, libEnzyme), CTypeTreeRef, ())
EnzymeNewTypeTreeCT(T, ctx) = ccall(
    (:EnzymeNewTypeTreeCT, libEnzyme),
    CTypeTreeRef,
    (CConcreteType, LLVMContextRef),
    T,
    ctx,
)
EnzymeNewTypeTreeTR(tt) =
    ccall((:EnzymeNewTypeTreeTR, libEnzyme), CTypeTreeRef, (CTypeTreeRef,), tt)

EnzymeFreeTypeTree(tt) = ccall((:EnzymeFreeTypeTree, libEnzyme), Cvoid, (CTypeTreeRef,), tt)
EnzymeSetTypeTree(dst, src) =
    ccall((:EnzymeSetTypeTree, libEnzyme), UInt8, (CTypeTreeRef, CTypeTreeRef), dst, src)
EnzymeMergeTypeTree(dst, src) =
    ccall((:EnzymeMergeTypeTree, libEnzyme), UInt8, (CTypeTreeRef, CTypeTreeRef), dst, src)
function EnzymeCheckedMergeTypeTree(dst, src)
    legal = Ref{UInt8}(0)
    res = ccall(
        (:EnzymeCheckedMergeTypeTree, libEnzyme),
        UInt8,
        (CTypeTreeRef, CTypeTreeRef, Ptr{UInt8}),
        dst,
        src,
        legal,
    )
    return res != 0, legal[] != 0
end
EnzymeTypeTreeOnlyEq(dst, x) =
    ccall((:EnzymeTypeTreeOnlyEq, libEnzyme), Cvoid, (CTypeTreeRef, Int64), dst, x)
EnzymeTypeTreeLookupEq(dst, x, dl) = ccall(
    (:EnzymeTypeTreeLookupEq, libEnzyme),
    Cvoid,
    (CTypeTreeRef, Int64, Cstring),
    dst,
    x,
    dl,
)
EnzymeTypeTreeCanonicalizeInPlace(dst, x, dl) = ccall(
    (:EnzymeTypeTreeCanonicalizeInPlace, libEnzyme),
    Cvoid,
    (CTypeTreeRef, Int64, Cstring),
    dst,
    x,
    dl,
)
EnzymeTypeTreeData0Eq(dst) =
    ccall((:EnzymeTypeTreeData0Eq, libEnzyme), Cvoid, (CTypeTreeRef,), dst)
EnzymeTypeTreeInner0(dst) =
    ccall((:EnzymeTypeTreeInner0, libEnzyme), CConcreteType, (CTypeTreeRef,), dst)
EnzymeTypeTreeShiftIndiciesEq(dst, dl, offset, maxSize, addOffset) = ccall(
    (:EnzymeTypeTreeShiftIndiciesEq, libEnzyme),
    Cvoid,
    (CTypeTreeRef, Cstring, Int64, Int64, UInt64),
    dst,
    dl,
    offset,
    maxSize,
    addOffset,
)

EnzymeTypeTreeToString(tt) =
    ccall((:EnzymeTypeTreeToString, libEnzyme), Cstring, (CTypeTreeRef,), tt)
EnzymeStringFree(str) = ccall((:EnzymeStringFree, libEnzyme), Cvoid, (Cstring,), str)

struct CFnTypeInfo
    arguments::Ptr{CTypeTreeRef}
    ret::CTypeTreeRef

    known_values::Ptr{IntList}
end

SetMD(v::Union{LLVM.Instruction,LLVM.GlobalVariable}, kind::String, node::LLVM.Metadata) =
    ccall(
        (:EnzymeSetStringMD, libEnzyme),
        Cvoid,
        (LLVM.API.LLVMValueRef, Cstring, LLVM.API.LLVMValueRef),
        v,
        kind,
        LLVM.Value(node),
    )

@static if !isdefined(LLVM, :ValueMetadataDict)
    Base.haskey(md::LLVM.InstructionMetadataDict, kind::String) =
        ccall(
            (:EnzymeGetStringMD, libEnzyme),
            Cvoid,
            (LLVM.API.LLVMValueRef, Cstring),
            md.inst,
            kind,
        ) != C_NULL

    function Base.getindex(md::LLVM.InstructionMetadataDict, kind::String)
        objref =
            ccall(
                (:EnzymeGetStringMD, libEnzyme),
                Cvoid,
                (LLVM.API.LLVMValueRef, Cstring),
                md.inst,
                kind,
            ) != C_NULL
        objref == C_NULL && throw(KeyError(kind))
        return LLVM.Metadata(LLVM.MetadataAsValue(objref))
    end

    Base.setindex!(md::LLVM.InstructionMetadataDict, node::LLVM.Metadata, kind::String) =
        ccall(
            (:EnzymeSetStringMD, libEnzyme),
            Cvoid,
            (LLVM.API.LLVMValueRef, Cstring, LLVM.API.LLVMValueRef),
            md.inst,
            kind,
            LLVM.Value(node),
        )
end

@cenum(
    CDIFFE_TYPE,
    DFT_OUT_DIFF = 0,  # add differential to an output struct
    DFT_DUP_ARG = 1,   # duplicate the argument and store differential inside
    DFT_CONSTANT = 2,  # no differential
    DFT_DUP_NONEED = 3 # duplicate this argument and store differential inside,
    # but don't need the forward
)

@inline Base.convert(::Type{API.CDIFFE_TYPE}, ::Type{A}) where {A<:EnzymeCore.Const} = API.DFT_CONSTANT
@inline Base.convert(::Type{API.CDIFFE_TYPE}, ::Type{A}) where {A<:EnzymeCore.Active} =
    API.DFT_OUT_DIFF
@inline Base.convert(::Type{API.CDIFFE_TYPE}, ::Type{A}) where {A<:EnzymeCore.Duplicated} =
    API.DFT_DUP_ARG
@inline Base.convert(::Type{API.CDIFFE_TYPE}, ::Type{A}) where {A<:EnzymeCore.BatchDuplicated} =
    API.DFT_DUP_ARG
@inline Base.convert(::Type{API.CDIFFE_TYPE}, ::Type{A}) where {A<:EnzymeCore.BatchDuplicatedFunc} =
    API.DFT_DUP_ARG
@inline Base.convert(::Type{API.CDIFFE_TYPE}, ::Type{A}) where {A<:EnzymeCore.DuplicatedNoNeed} =
    API.DFT_DUP_NONEED
@inline Base.convert(::Type{API.CDIFFE_TYPE}, ::Type{A}) where {A<:EnzymeCore.BatchDuplicatedNoNeed} =
    API.DFT_DUP_NONEED

@cenum(
    CDerivativeMode,
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
function EnzymeCreatePrimalAndGradient(
    logic,
    todiff,
    retType,
    constant_args,
    TA,
    returnValue,
    dretUsed,
    mode,
    runtimeActivity,
    strongZero,
    width,
    additionalArg,
    forceAnonymousTape,
    typeInfo,
    uncacheable_args,
    augmented,
    atomicAdd,
)
    freeMemory = true
    subsequent_calls_may_write = mode != DEM_ReverseModeCombined
    ccall(
        (:EnzymeCreatePrimalAndGradient, libEnzyme),
        LLVMValueRef,
        (
            EnzymeLogicRef,
            LLVMValueRef,
            LLVM.API.LLVMBuilderRef,
            LLVMValueRef,
            CDIFFE_TYPE,
            Ptr{CDIFFE_TYPE},
            Csize_t,
            EnzymeTypeAnalysisRef,
            UInt8,
            UInt8,
            CDerivativeMode,
            UInt8,
            UInt8,
            Cuint,
            UInt8,
            LLVMTypeRef,
            UInt8,
            CFnTypeInfo,
            UInt8,
            Ptr{UInt8},
            Csize_t,
            EnzymeAugmentedReturnPtr,
            UInt8,
        ),
        logic,
        C_NULL,
        C_NULL,
        todiff,
        retType,
        constant_args,
        length(constant_args),
        TA,
        returnValue,
        dretUsed,
        mode,
        runtimeActivity,
        strongZero,
        width,
        freeMemory,
        additionalArg,
        forceAnonymousTape,
        typeInfo,
        subsequent_calls_may_write,
        uncacheable_args,
        length(uncacheable_args),
        augmented,
        atomicAdd,
    )
end

function EnzymeCreateForwardDiff(
    logic,
    todiff,
    retType,
    constant_args,
    TA,
    returnValue,
    mode,
    runtimeActivity,
    strongZero,
    width,
    additionalArg,
    typeInfo,
    uncacheable_args,
)
    freeMemory = true
    aug = C_NULL
    subsequent_calls_may_write = false
    ccall(
        (:EnzymeCreateForwardDiff, libEnzyme),
        LLVMValueRef,
        (
            EnzymeLogicRef,
            LLVMValueRef,
            LLVM.API.LLVMBuilderRef,
            LLVMValueRef,
            CDIFFE_TYPE,
            Ptr{CDIFFE_TYPE},
            Csize_t,
            EnzymeTypeAnalysisRef,
            UInt8,
            CDerivativeMode,
            UInt8,
            UInt8,
            UInt8,
            Cuint,
            LLVMTypeRef,
            CFnTypeInfo,
            UInt8,
            Ptr{UInt8},
            Csize_t,
            EnzymeAugmentedReturnPtr,
        ),
        logic,
        C_NULL,
        C_NULL,
        todiff,
        retType,
        constant_args,
        length(constant_args),
        TA,
        returnValue,
        mode,
        freeMemory,
        runtimeActivity,
        strongZero,
        width,
        additionalArg,
        typeInfo,
        subsequent_calls_may_write,
        uncacheable_args,
        length(uncacheable_args),
        aug,
    )
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
function EnzymeCreateAugmentedPrimal(
    logic,
    todiff,
    retType,
    constant_args,
    TA,
    returnUsed,
    shadowReturnUsed,
    typeInfo,
    uncacheable_args,
    forceAnonymousTape,
    runtimeActivity,
    strongZero,
    width,
    atomicAdd,
)
    subsequent_calls_may_write = true
    ccall(
        (:EnzymeCreateAugmentedPrimal, libEnzyme),
        EnzymeAugmentedReturnPtr,
        (
            EnzymeLogicRef,
            LLVMValueRef,
            LLVM.API.LLVMBuilderRef,
            LLVMValueRef,
            CDIFFE_TYPE,
            Ptr{CDIFFE_TYPE},
            Csize_t,
            EnzymeTypeAnalysisRef,
            UInt8,
            UInt8,
            CFnTypeInfo,
            UInt8,
            Ptr{UInt8},
            Csize_t,
            UInt8,
            UInt8,
            UInt8,
            Cuint,
            UInt8,
        ),
        logic,
        C_NULL,
        C_NULL,
        todiff,
        retType,
        constant_args,
        length(constant_args),
        TA,
        returnUsed,
        shadowReturnUsed,
        typeInfo,
        subsequent_calls_may_write,
        uncacheable_args,
        length(uncacheable_args),
        forceAnonymousTape,
        runtimeActivity,
        strongZero,
        width,
        atomicAdd,
    )
end

# typedef uint8_t (*CustomRuleType)(int /*direction*/, CTypeTreeRef /*return*/,
#                                   CTypeTreeRef * /*args*/,
#                                   struct IntList * /*knownValues*/,
#                                   size_t /*numArgs*/, LLVMValueRef);
const CustomRuleType = Ptr{Cvoid}

function CreateTypeAnalysis(logic, rulenames, rules)
    @assert length(rulenames) == length(rules)
    ccall(
        (:CreateTypeAnalysis, libEnzyme),
        EnzymeTypeAnalysisRef,
        (EnzymeLogicRef, Ptr{Cstring}, Ptr{CustomRuleType}, Csize_t),
        logic,
        rulenames isa Tuple{} ? C_NULL : rulenames,
        rules isa Tuple{} ? C_NULL : rules,
        rulenames isa Tuple{} ? 0 : length(rules),
    )
end

function ClearTypeAnalysis(ta)
    ccall((:ClearTypeAnalysis, libEnzyme), Cvoid, (EnzymeTypeAnalysisRef,), ta)
end

function FreeTypeAnalysis(ta)
    ccall((:FreeTypeAnalysis, libEnzyme), Cvoid, (EnzymeTypeAnalysisRef,), ta)
end

function EnzymeAnalyzeTypes(ta, CTI, F)
    ccall(
        (:EnzymeAnalyzeTypes, libEnzyme),
        EnzymeTypeAnalyzerRef,
        (EnzymeTypeAnalysisRef, CFnTypeInfo, LLVMValueRef),
        ta,
        CTI,
        F,
    )
end

const CustomShadowAlloc = Ptr{Cvoid}
const CustomShadowFree = Ptr{Cvoid}
EnzymeRegisterAllocationHandler(name, ahandle, fhandle) = ccall(
    (:EnzymeRegisterAllocationHandler, libEnzyme),
    Cvoid,
    (Cstring, CustomShadowAlloc, CustomShadowFree),
    name,
    ahandle,
    fhandle,
)


const CustomAugmentedForwardPass = Ptr{Cvoid}
const CustomForwardPass = Ptr{Cvoid}
const CustomReversePass = Ptr{Cvoid}
EnzymeRegisterCallHandler(name, fwdhandle, revhandle) = ccall(
    (:EnzymeRegisterCallHandler, libEnzyme),
    Cvoid,
    (Cstring, CustomAugmentedForwardPass, CustomReversePass),
    name,
    fwdhandle,
    revhandle,
)
EnzymeRegisterFwdCallHandler(name, fwdhandle) = ccall(
    (:EnzymeRegisterFwdCallHandler, libEnzyme),
    Cvoid,
    (Cstring, CustomForwardPass),
    name,
    fwdhandle,
)

EnzymeInsertValue(
    B::LLVM.IRBuilder,
    v::LLVM.Value,
    v2::LLVM.Value,
    insts::Vector{Cuint},
    name = "",
) = LLVM.Value(
    ccall(
        (:EnzymeInsertValue, libEnzyme),
        LLVMValueRef,
        (LLVM.API.LLVMBuilderRef, LLVMValueRef, LLVMValueRef, Ptr{Cuint}, Int64, Cstring),
        B,
        v,
        v2,
        insts,
        length(insts),
        name,
    ),
)

const CustomDiffUse = Ptr{Cvoid}
EnzymeRegisterDiffUseCallHandler(name, handle) = ccall(
    (:EnzymeRegisterDiffUseCallHandler, libEnzyme),
    Cvoid,
    (Cstring, CustomDiffUse),
    name,
    handle,
)
EnzymeSetCalledFunction(ci::LLVM.CallInst, fn::LLVM.Function, toremove) = ccall(
    (:EnzymeSetCalledFunction, libEnzyme),
    Cvoid,
    (LLVMValueRef, LLVMValueRef, Ptr{Int64}, Int64),
    ci,
    fn,
    toremove,
    length(toremove),
)
EnzymeCloneFunctionWithoutReturnOrArgs(fn::LLVM.Function, keepret, args) = ccall(
    (:EnzymeCloneFunctionWithoutReturnOrArgs, libEnzyme),
    LLVMValueRef,
    (LLVMValueRef, UInt8, Ptr{Int64}, Int64),
    fn,
    keepret,
    args,
    length(args),
)
EnzymeGetShadowType(width, T) =
    ccall((:EnzymeGetShadowType, libEnzyme), LLVMTypeRef, (UInt64, LLVMTypeRef), width, T)

EnzymeGradientUtilsReplaceAWithB(gutils, a, b) = ccall(
    (:EnzymeGradientUtilsReplaceAWithB, libEnzyme),
    Cvoid,
    (EnzymeGradientUtilsRef, LLVMValueRef, LLVMValueRef),
    gutils,
    a,
    b,
)
EnzymeGradientUtilsErase(gutils, a) = ccall(
    (:EnzymeGradientUtilsErase, libEnzyme),
    Cvoid,
    (EnzymeGradientUtilsRef, LLVMValueRef),
    gutils,
    a,
)
EnzymeReplaceOriginalToNew(gutils, orig, rep) = ccall(
    (:EnzymeReplaceOriginalToNew, libEnzyme),
    Cvoid,
    (EnzymeGradientUtilsRef, LLVMValueRef, LLVMValueRef),
    gutils,
    orig,
    rep
)
EnzymeGradientUtilsEraseWithPlaceholder(gutils, a, orig, erase) = ccall(
    (:EnzymeGradientUtilsEraseWithPlaceholder, libEnzyme),
    Cvoid,
    (EnzymeGradientUtilsRef, LLVMValueRef, LLVMValueRef, UInt8),
    gutils,
    a,
    orig,
    erase,
)
EnzymeGradientUtilsGetMode(gutils) = ccall(
    (:EnzymeGradientUtilsGetMode, libEnzyme),
    CDerivativeMode,
    (EnzymeGradientUtilsRef,),
    gutils,
)
EnzymeGradientUtilsGetWidth(gutils) = ccall(
    (:EnzymeGradientUtilsGetWidth, libEnzyme),
    UInt64,
    (EnzymeGradientUtilsRef,),
    gutils,
)
EnzymeGradientUtilsGetRuntimeActivity(gutils) =
    ccall(
        (:EnzymeGradientUtilsGetRuntimeActivity, libEnzyme),
        UInt8,
        (EnzymeGradientUtilsRef,),
        gutils,
    ) != 0
EnzymeGradientUtilsGetStrongZero(gutils) =
    ccall(
        (:EnzymeGradientUtilsGetStrongZero, libEnzyme),
        UInt8,
        (EnzymeGradientUtilsRef,),
        gutils,
    ) != 0
EnzymeGradientUtilsNewFromOriginal(gutils, val) = ccall(
    (:EnzymeGradientUtilsNewFromOriginal, libEnzyme),
    LLVMValueRef,
    (EnzymeGradientUtilsRef, LLVMValueRef),
    gutils,
    val,
)
EnzymeGradientUtilsSetDebugLocFromOriginal(gutils, val, orig) = ccall(
    (:EnzymeGradientUtilsSetDebugLocFromOriginal, libEnzyme),
    Cvoid,
    (EnzymeGradientUtilsRef, LLVMValueRef, LLVMValueRef),
    gutils,
    val,
    orig,
)
EnzymeGradientUtilsLookup(gutils, val, B) = ccall(
    (:EnzymeGradientUtilsLookup, libEnzyme),
    LLVMValueRef,
    (EnzymeGradientUtilsRef, LLVMValueRef, LLVM.API.LLVMBuilderRef),
    gutils,
    val,
    B,
)
EnzymeGradientUtilsInvertPointer(gutils, val, B) = ccall(
    (:EnzymeGradientUtilsInvertPointer, libEnzyme),
    LLVMValueRef,
    (EnzymeGradientUtilsRef, LLVMValueRef, LLVM.API.LLVMBuilderRef),
    gutils,
    val,
    B,
)
EnzymeGradientUtilsDiffe(gutils, val, B) = ccall(
    (:EnzymeGradientUtilsDiffe, libEnzyme),
    LLVMValueRef,
    (EnzymeGradientUtilsRef, LLVMValueRef, LLVM.API.LLVMBuilderRef),
    gutils,
    val,
    B,
)
EnzymeGradientUtilsAddToDiffe(gutils, val, diffe, B, T) = ccall(
    (:EnzymeGradientUtilsAddToDiffe, libEnzyme),
    Cvoid,
    (
        EnzymeGradientUtilsRef,
        LLVMValueRef,
        LLVMValueRef,
        LLVM.API.LLVMBuilderRef,
        LLVMTypeRef,
    ),
    gutils,
    val,
    diffe,
    B,
    T,
)
function EnzymeGradientUtilsAddToInvertedPointerDiffeTT(
    gutils,
    orig,
    origVal,
    vd,
    size,
    origptr,
    prediff,
    B,
    align,
    premask,
)
    ccall(
        (:EnzymeGradientUtilsAddToInvertedPointerDiffeTT, libEnzyme),
        Cvoid,
        (
            EnzymeGradientUtilsRef,
            LLVMValueRef,
            LLVMValueRef,
            CTypeTreeRef,
            Cuint,
            LLVMValueRef,
            LLVMValueRef,
            LLVM.API.LLVMBuilderRef,
            Cuint,
            LLVMValueRef,
        ),
        gutils,
        orig,
        origVal,
        vd,
        size,
        origptr,
        prediff,
        B,
        align,
        premask,
    )
end

EnzymeGradientUtilsSetDiffe(gutils, val, diffe, B) = ccall(
    (:EnzymeGradientUtilsSetDiffe, libEnzyme),
    Cvoid,
    (EnzymeGradientUtilsRef, LLVMValueRef, LLVMValueRef, LLVM.API.LLVMBuilderRef),
    gutils,
    val,
    diffe,
    B,
)
EnzymeGradientUtilsIsConstantValue(gutils, val) = ccall(
    (:EnzymeGradientUtilsIsConstantValue, libEnzyme),
    UInt8,
    (EnzymeGradientUtilsRef, LLVMValueRef),
    gutils,
    val,
)
EnzymeGradientUtilsIsConstantInstruction(gutils, val) = ccall(
    (:EnzymeGradientUtilsIsConstantInstruction, libEnzyme),
    UInt8,
    (EnzymeGradientUtilsRef, LLVMValueRef),
    gutils,
    val,
)
EnzymeGradientUtilsAllocationBlock(gutils) = ccall(
    (:EnzymeGradientUtilsAllocationBlock, libEnzyme),
    LLVM.API.LLVMBasicBlockRef,
    (EnzymeGradientUtilsRef,),
    gutils,
)

EnzymeGradientUtilsTypeAnalyzer(gutils) = ccall(
    (:EnzymeGradientUtilsTypeAnalyzer, libEnzyme),
    EnzymeTypeAnalyzerRef,
    (EnzymeGradientUtilsRef,),
    gutils,
)

EnzymeGradientUtilsAllocAndGetTypeTree(gutils, val) = ccall(
    (:EnzymeGradientUtilsAllocAndGetTypeTree, libEnzyme),
    CTypeTreeRef,
    (EnzymeGradientUtilsRef, LLVMValueRef),
    gutils,
    val,
)

EnzymeGradientUtilsGetUncacheableArgs(gutils, orig, uncacheable, size) = ccall(
    (:EnzymeGradientUtilsGetUncacheableArgs, libEnzyme),
    UInt8,
    (EnzymeGradientUtilsRef, LLVMValueRef, Ptr{UInt8}, UInt64),
    gutils,
    orig,
    uncacheable,
    size,
)

EnzymeGradientUtilsGetDiffeType(gutils, op, isforeign) = ccall(
    (:EnzymeGradientUtilsGetDiffeType, libEnzyme),
    CDIFFE_TYPE,
    (EnzymeGradientUtilsRef, LLVMValueRef, UInt8),
    gutils,
    op,
    isforeign,
)

EnzymeGradientUtilsGetReturnDiffeType(gutils, orig, needsPrimalP, needsShadowP, mode) =
    ccall(
        (:EnzymeGradientUtilsGetReturnDiffeType, libEnzyme),
        CDIFFE_TYPE,
        (EnzymeGradientUtilsRef, LLVMValueRef, Ptr{UInt8}, Ptr{UInt8}, CDerivativeMode),
        gutils,
        orig,
        needsPrimalP,
        needsShadowP,
        mode,
    )

EnzymeGradientUtilsSubTransferHelper(
    gutils,
    mode,
    secretty,
    intrinsic,
    dstAlign,
    srcAlign,
    offset,
    dstConstant,
    origdst,
    srcConstant,
    origsrc,
    length,
    isVolatile,
    MTI,
    allowForward,
    shadowsLookedUp,
) = ccall(
    (:EnzymeGradientUtilsSubTransferHelper, libEnzyme),
    Cvoid,
    (
        EnzymeGradientUtilsRef,
        CDerivativeMode,
        LLVMTypeRef,
        UInt64,
        UInt64,
        UInt64,
        UInt64,
        UInt8,
        LLVMValueRef,
        UInt8,
        LLVMValueRef,
        LLVMValueRef,
        LLVMValueRef,
        LLVMValueRef,
        UInt8,
        UInt8,
    ),
    gutils,
    mode,
    secretty,
    intrinsic,
    dstAlign,
    srcAlign,
    offset,
    dstConstant,
    origdst,
    srcConstant,
    origsrc,
    length,
    isVolatile,
    MTI,
    allowForward,
    shadowsLookedUp,
)

EnzymeGradientUtilsAddReverseBlock(
    gutils,
    block,
    name,
    forkCache,
    push
) = ccall(
    (:EnzymeGradientUtilsAddReverseBlock, libEnzyme),
    LLVM.API.LLVMBasicBlockRef,
    (
        EnzymeGradientUtilsRef,
        LLVM.API.LLVMBasicBlockRef,
        Cstring,
        UInt8,
        UInt8,
    ),
    gutils,
    block,
    name,
    forkCache,
    push
)

EnzymeGradientUtilsSetReverseBlock(
    gutils,
    block,
) = ccall(
    (:EnzymeGradientUtilsSetReverseBlock, libEnzyme),
    Cvoid,
    (
        EnzymeGradientUtilsRef,
        LLVM.API.LLVMBasicBlockRef,
    ),
    gutils,
    block,
)

EnzymeGradientUtilsCallWithInvertedBundles(
    gutils,
    func,
    funcTy,
    argvs,
    argc,
    orig,
    valTys,
    valCnt,
    B,
    lookup,
) = ccall(
    (:EnzymeGradientUtilsCallWithInvertedBundles, libEnzyme),
    LLVMValueRef,
    (
        EnzymeGradientUtilsRef,
        LLVMValueRef,
        LLVMTypeRef,
        Ptr{LLVMValueRef},
        UInt64,
        LLVMValueRef,
        Ptr{CValueType},
        UInt64,
        LLVM.API.LLVMBuilderRef,
        UInt8,
    ),
    gutils,
    func,
    funcTy,
    argvs,
    argc,
    orig,
    valTys,
    valCnt,
    B,
    lookup,
)

function sub_transfer(
    gutils,
    mode,
    secretty,
    intrinsic,
    dstAlign,
    srcAlign,
    offset,
    dstConstant,
    origdst,
    srcConstant,
    origsrc,
    length,
    isVolatile,
    MTI,
    allowForward,
    shadowsLookedUp,
)
    GC.@preserve secretty begin
        if secretty === nothing
            secretty = Base.unsafe_convert(LLVMTypeRef, C_NULL)
        else
            secretty = Base.unsafe_convert(LLVMTypeRef, secretty)
        end

        EnzymeGradientUtilsSubTransferHelper(
            gutils,
            mode,
            secretty,
            intrinsic,
            dstAlign,
            srcAlign,
            offset,
            dstConstant,
            origdst,
            srcConstant,
            origsrc,
            length,
            isVolatile,
            MTI,
            allowForward,
            shadowsLookedUp,
        )
    end
end

function CreateLogic(postOpt = false)
    ccall((:CreateEnzymeLogic, libEnzyme), EnzymeLogicRef, (UInt8,), postOpt)
end

EnzymeLogicErasePreprocessedFunctions(logic) = ccall(
    (:EnzymeLogicErasePreprocessedFunctions, libEnzyme),
    Cvoid,
    (EnzymeLogicRef,),
    logic,
)

function ClearLogic(logic)
    ccall((:ClearEnzymeLogic, libEnzyme), Cvoid, (EnzymeLogicRef,), logic)
end

function FreeLogic(logic)
    ccall((:FreeEnzymeLogic, libEnzyme), Cvoid, (EnzymeLogicRef,), logic)
end

function LogicSetExternalContext(logic, ctx)
    ccall((:EnzymeLogicSetExternalContext, libEnzyme), Cvoid, (EnzymeLogicRef, Ptr{Cvoid}), logic, ctx)
end

function LogicGetExternalContext(logic)
    ccall((:EnzymeLogicGetExternalContext, libEnzyme), Ptr{Cvoid}, (EnzymeLogicRef,), logic)
end

function EnzymeExtractReturnInfo(ret, data, existed)
    @assert length(data) == length(existed)
    ccall(
        (:EnzymeExtractReturnInfo, libEnzyme),
        Cvoid,
        (EnzymeAugmentedReturnPtr, Ptr{Int64}, Ptr{UInt8}, Csize_t),
        ret,
        data,
        existed,
        length(data),
    )
end

function EnzymeExtractFunctionFromAugmentation(ret)
    ccall(
        (:EnzymeExtractFunctionFromAugmentation, libEnzyme),
        LLVMValueRef,
        (EnzymeAugmentedReturnPtr,),
        ret,
    )
end


function EnzymeExtractTapeTypeFromAugmentation(ret)
    ccall(
        (:EnzymeExtractTapeTypeFromAugmentation, libEnzyme),
        LLVMTypeRef,
        (EnzymeAugmentedReturnPtr,),
        ret,
    )
end

function EnzymeExtractUnderlyingTapeTypeFromAugmentation(ret)
    ccall(
        (:EnzymeExtractUnderlyingTapeTypeFromAugmentation, libEnzyme),
        LLVMTypeRef,
        (EnzymeAugmentedReturnPtr,),
        ret,
    )
end

import Libdl
function EnzymeSetCLBool(name, val)
    handle = Libdl.dlopen(libEnzyme)
    ptr = Libdl.dlsym(handle, name)
    ccall((:EnzymeSetCLBool, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end
function EnzymeGetCLBool(ptr)
    ccall((:EnzymeGetCLBool, libEnzyme), UInt8, (Ptr{Cvoid},), ptr)
end
# void EnzymeSetCLInteger(void *, int64_t);

function zcache!(val)
    ptr = cglobal((:EnzymeZeroCache, libEnzyme))
    ccall((:EnzymeSetCLBool, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end


"""
    printperf!(val::Bool)

An debugging option for developers of Enzyme. If one sets this flag prior
to the first differentiation of a function, Enzyme will print (to stderr)
performance information about generated derivative programs. It will provide
debug information that warns why particular values are cached for the
reverse pass, and thus require additional computation/storage. This is particularly
helpful for debugging derivatives which OOM or otherwise run slow.
ff by default
"""
function printperf!(val)
    ptr = cglobal((:EnzymePrintPerf, libEnzyme))
    ccall((:EnzymeSetCLBool, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end

"""
    printdiffuse!(val::Bool)

An debugging option for developers of Enzyme. If one sets this flag prior
to the first differentiation of a function, Enzyme will print (to stderr)
information about each LLVM value -- specifically whether it and its shadow
is required for computing the derivative. In contrast to [`printunnecessary!`](@ref),
this flag prints debug log for the analysis which determines for each value
and shadow value, whether it can find a user which would require it to be kept
around (rather than being deleted). This is prior to any cache optimizations
and a debug log of Differential Use Analysis. This may be helpful for debugging
caching, phi node deletion, performance, and other errors.
Off by default
"""
function printdiffuse!(val)
    ptr = cglobal((:EnzymePrintDiffUse, libEnzyme))
    ccall((:EnzymeSetCLBool, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end

"""
    printtype!(val::Bool)

An debugging option for developers of Enzyme. If one sets this flag prior
to the first differentiation of a function, Enzyme will print (to stderr)
a log of all decisions made during Type Analysis (the analysis which
Enzyme determines the type of all values in the program). This may be useful
for debugging correctness errors, illegal type analysis errors, insufficient
type information errors, correctness, and performance errors.
Off by default
"""
function printtype!(val)
    ptr = cglobal((:EnzymePrintType, libEnzyme))
    ccall((:EnzymeSetCLBool, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end

"""
    printactivity!(val::Bool)

An debugging option for developers of Enzyme. If one sets this flag prior
to the first differentiation of a function, Enzyme will print (to stderr)
a log of all decisions made during Activity Analysis (the analysis which
determines what values/instructions are differentiated). This may be useful
for debugging MixedActivity errors, correctness, and performance errors.
Off by default
"""
function printactivity!(val)
    ptr = cglobal((:EnzymePrintActivity, libEnzyme))
    ccall((:EnzymeSetCLBool, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end

"""
    printall!(val::Bool)

An debugging option for developers of Enzyme. If one sets this flag prior
to the first differentiation of a function, Enzyme will print (to stderr)
the LLVM function being differentiated, as well as all generated derivatives
immediately after running Enzyme (but prior to any other optimizations).
Off by default
"""
function printall!(val)
    ptr = cglobal((:EnzymePrint, libEnzyme))
    ccall((:EnzymeSetCLBool, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end

"""
    printunnecessary!(val::Bool)

An debugging option for developers of Enzyme. If one sets this flag prior
to the first differentiation of a function, Enzyme will print (to stderr)
information about each LLVM value -- specifically whether it and its shadow
is required for computing the derivative. In contrast to [`printdiffuse!`](@ref),
this flag prints the final results after running cache optimizations such
as minCut (see Recompute vs Cache Heuristics from [this paper](https://c.wsmoses.com/papers/EnzymeGPU.pdf)
and slides 31-33 from [this presentation](https://c.wsmoses.com/presentations/enzyme-sc.pdf)) for a
description of the caching algorithm. This may be helpful for debugging
caching, phi node deletion, performance, and other errors.
Off by default
"""
function printunnecessary!(val)
    ptr = cglobal((:EnzymePrintUnnecessary, libEnzyme))
    ccall((:EnzymeSetCLBool, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end

"""
    inlineall!(val::Bool)

Whether to inline all (non-recursive) functions generated by Julia within a 
single compilation unit. This may improve Enzyme's ability to successfully
differentiate code and improve performance of the original and generated 
derivative program. It often, however, comes with an increase in compile time.
This is off by default.
"""
function inlineall!(val)
    ptr = cglobal((:EnzymeInline, libEnzyme))
    ccall((:EnzymeSetCLBool, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end


"""
    maxtypeoffset!(val::Int)

Enzyme runs a type analysis to deduce the corresponding types of all values being
differentiated. This is necessary to compute correct derivatives of various values.
To ensure this analysis temrinates, it operates on a finite lattice of possible
states. This function sets the maximum offset into a type that Enzyme will consider.
A smaller value will cause type analysis to run faster, but may result in some
necessary types not being found and result in unknown type errors. A larger value
may result in unknown type errors being resolved by searching a larger space, but
may run longer. The default setting is 512.
"""
function maxtypeoffset!(val)
    ptr = cglobal((:MaxTypeOffset, libEnzyme))
    ccall((:EnzymeSetCLInteger, libEnzyme), Cvoid, (Ptr{Cvoid}, Int64), ptr, val)
end

"""
    maxtypedepth!(val::Int)

Enzyme runs a type analysis to deduce the corresponding types of all values being
differentiated. This is necessary to compute correct derivatives of various values.
To ensure this analysis temrinates, it operates on a finite lattice of possible
states. This function sets the maximum depth into a type that Enzyme will consider.
A smaller value will cause type analysis to run faster, but may result in some
necessary types not being found and result in unknown type errors. A larger value
may result in unknown type errors being resolved by searching a larger space, but
may run longer. The default setting is 6.
"""
function maxtypedepth!(val)
    ptr = cglobal((:EnzymeMaxTypeDepth, libEnzyme))
    ccall((:EnzymeSetCLInteger, libEnzyme), Cvoid, (Ptr{Cvoid}, Int64), ptr, val)
end



"""
    looseTypeAnalysis!(val::Bool)

Enzyme runs a type analysis to deduce the corresponding types of all values being
differentiated. This is necessary to compute correct derivatives of various values.
For example, a copy of Float32's requires a different derivative than a memcpy of
Float64's, Ptr's, etc. In some cases Enzyme may not be able to deduce all the types
necessary and throw an unknown type error. If this is the case, open an issue. 
One can silence these issues by setting `looseTypeAnalysis!(true)` which tells 
Enzyme to make its best guess. This will remove the error and allow differentiation
to continue, however, it may produce incorrect results. Alternatively one can
consider increasing the space of the evaluated type lattice which gives Enzyme
more time to run a more thorough analysis through the use of [`maxtypeoffset!`](@ref)
"""
function looseTypeAnalysis!(val)
    ptr = cglobal((:looseTypeAnalysis, libEnzyme))
    ccall((:EnzymeSetCLInteger, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end


"""
    strictAliasing!(val::Bool)

Whether Enzyme's type analysis will assume strict aliasing semantics. When strict
aliasing semantics are on (the default), Enzyme can propagate type information up
through conditional branches. This may lead to illegal type errors when analyzing
code with unions. Disabling strict aliasing will enable these union types to be
correctly analyzed. However, it may lead to some errors that sufficient type information
cannot be deduced. One can turn these insufficient type information errors into to
warnings by calling [`looseTypeAnalysis!`](@ref)`(true)` which tells Enzyme to use its best
guess in such scenarios.
"""
function strictAliasing!(val)
    ptr = cglobal((:EnzymeStrictAliasing, libEnzyme))
    ccall((:EnzymeSetCLInteger, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end

"""
    fast_math!(val::Bool)

Whether generated derivatives have fast math on or off, default on.
"""
function fast_math!(val)
    ptr = cglobal((:EnzymeFastMath, libEnzyme))
    ccall((:EnzymeSetCLInteger, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end

"""
    typeWarning!(val::Bool)

Whether to print a warning when Type Analysis learns informatoin about a value's type
which cannot be represented in the current size of the lattice. See [`maxtypeoffset!`](@ref) for
more information.
Off by default.
"""
function typeWarning!(val)
    ptr = cglobal((:EnzymeTypeWarning, libEnzyme))
    ccall((:EnzymeSetCLInteger, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end

"""
    instname!(val::Bool)

Whether to add a name to all LLVM values. This may be helpful for debugging generated
programs, both primal and derivative.
Off by default.
"""
function instname!(val)
    ptr = cglobal((:EnzymeNameInstructions, libEnzyme))
    ccall((:EnzymeSetCLBool, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end

"""
    memmove_warning!(val::Bool)

Whether to issue a warning when differentiating memmove.
Off by default.
"""
function memmove_warning!(val)
    ptr = cglobal((:EnzymeMemmoveWarning, libEnzyme))
    ccall((:EnzymeSetCLBool, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end

function EnzymeNonPower2Cache!(val)
    ptr = cglobal((:EnzymeNonPower2Cache, libEnzyme))
    ccall((:EnzymeSetCLInteger, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
end

function EnzymeRemoveTrivialAtomicIncrements(func)
    ccall((:EnzymeRemoveTrivialAtomicIncrements, libEnzyme), Cvoid, (LLVMValueRef,), func)
end

function EnzymeAddAttributorLegacyPass(PM)
    ccall(
        (:EnzymeAddAttributorLegacyPass, libEnzyme),
        Cvoid,
        (LLVM.API.LLVMPassManagerRef,),
        PM,
    )
end

@cenum(
    ErrorType,
    ET_NoDerivative = 0,
    ET_NoShadow = 1,
    ET_IllegalTypeAnalysis = 2,
    ET_NoType = 3,
    ET_IllegalFirstPointer = 4,
    ET_InternalError = 5,
    ET_TypeDepthExceeded = 6,
    ET_MixedActivityError = 7,
    ET_IllegalReplaceFicticiousPHIs = 8,
    ET_GetIndexError = 9,
    ET_NoTruncate = 10,
    ET_GCRewrite = 11
)

function EnzymeTypeAnalyzerToString(typeanalyzer)
    ccall(
        (:EnzymeTypeAnalyzerToString, libEnzyme),
        Cstring,
        (EnzymeTypeAnalyzerRef,),
        typeanalyzer,
    )
end

function EnzymeGradientUtilsInvertedPointersToString(gutils)
    ccall(
        (:EnzymeGradientUtilsInvertedPointersToString, libEnzyme),
        Cstring,
        (Ptr{Cvoid},),
        gutils,
    )
end

function EnzymeSetHandler(handler)
    ptr = cglobal((:CustomErrorHandler, libEnzyme), Ptr{Ptr{Cvoid}})
    unsafe_store!(ptr, handler)
end
function EnzymeSetSanitizeDerivatives(handler)
    ptr = cglobal((:EnzymeSanitizeDerivatives, libEnzyme), Ptr{Ptr{Cvoid}})
    unsafe_store!(ptr, handler)
end
function EnzymeSetRuntimeInactiveError(handler)
    ptr = cglobal((:CustomRuntimeInactiveError, libEnzyme), Ptr{Ptr{Cvoid}})
    unsafe_store!(ptr, handler)
end
function EnzymeHasCustomInactiveSupport()
    try
        EnzymeSetRuntimeInactiveError(C_NULL)
    catch
        return false
    end
    return true
end

function EnzymeSetPostCacheStore(handler)
    ptr = cglobal((:EnzymePostCacheStore, libEnzyme), Ptr{Ptr{Cvoid}})
    unsafe_store!(ptr, handler)
end
function EnzymeSetUndefinedValueForType(handler)
    ptr = cglobal((:EnzymeUndefinedValueForType, libEnzyme), Ptr{Ptr{Cvoid}})
    unsafe_store!(ptr, handler)
end
function EnzymeSetShadowAllocRewrite(handler)
    ptr = cglobal((:EnzymeShadowAllocRewrite, libEnzyme), Ptr{Ptr{Cvoid}})
    unsafe_store!(ptr, handler)
end
function EnzymeSetDefaultTapeType(handler)
    ptr = cglobal((:EnzymeDefaultTapeType, libEnzyme), Ptr{Ptr{Cvoid}})
    unsafe_store!(ptr, handler)
end
function EnzymeSetCustomAllocator(handler)
    ptr = cglobal((:CustomAllocator, libEnzyme), Ptr{Ptr{Cvoid}})
    unsafe_store!(ptr, handler)
end
function EnzymeSetCustomDeallocator(handler)
    ptr = cglobal((:CustomDeallocator, libEnzyme), Ptr{Ptr{Cvoid}})
    unsafe_store!(ptr, handler)
end
function EnzymeSetCustomZero(handler)
    ptr = cglobal((:CustomZero, libEnzyme), Ptr{Ptr{Cvoid}})
    unsafe_store!(ptr, handler)
end
function EnzymeSetFixupReturn(handler)
    ptr = cglobal((:EnzymeFixupReturn, libEnzyme), Ptr{Ptr{Cvoid}})
    unsafe_store!(ptr, handler)
end

function EnzymeHasCustomAllocatorSupport()
    try
        EnzymeSetCustomAllocator(C_NULL)
        EnzymeSetCustomDeallocator(C_NULL)
    catch
        return false
    end
    return true
end

function __init__()
    ptr = cglobal((:EnzymeJuliaAddrLoad, libEnzyme))
    val = true
    ccall((:EnzymeSetCLBool, libEnzyme), Cvoid, (Ptr{Cvoid}, UInt8), ptr, val)
    zcache!(true)
end

function moveBefore(i1, i2, BR)
    ccall(
        (:EnzymeMoveBefore, libEnzyme),
        Cvoid,
        (LLVM.API.LLVMValueRef, LLVM.API.LLVMValueRef, LLVM.API.LLVMBuilderRef),
        i1,
        i2,
        BR,
    )
end

function EnzymeCloneFunctionDISubprogramInto(i1, i2)
    ccall(
        (:EnzymeCloneFunctionDISubprogramInto, libEnzyme),
        Cvoid,
        (LLVM.API.LLVMValueRef, LLVM.API.LLVMValueRef),
        i1,
        i2,
    )
end

function EnzymeCopyMetadata(i1, i2)
    ccall(
        (:EnzymeCopyMetadata, libEnzyme),
        Cvoid,
        (LLVM.API.LLVMValueRef, LLVM.API.LLVMValueRef),
        i1,
        i2,
    )
end

function SetMustCache!(i1)
    ccall((:EnzymeSetMustCache, libEnzyme), Cvoid, (LLVM.API.LLVMValueRef,), i1)
end

function SetForMemSet!(i1)
    ccall((:EnzymeSetForMemSet, libEnzyme), Cvoid, (LLVM.API.LLVMValueRef,), i1)
end

function HasFromStack(i1)
    ccall((:EnzymeHasFromStack, libEnzyme), UInt8, (LLVM.API.LLVMValueRef,), i1) != 0
end

function AddPreserveNVVMPass!(pm, i8)
    ccall(
        (:AddPreserveNVVMPass, libEnzyme),
        Cvoid,
        (LLVM.API.LLVMPassManagerRef, UInt8),
        pm,
        i8,
    )
end

function EnzymeReplaceFunctionImplementation(mod)
    ccall(
        (:EnzymeReplaceFunctionImplementation, libEnzyme),
        Cvoid,
        (LLVM.API.LLVMModuleRef,),
        mod,
    )
end

function EnzymeDetectReadonlyOrThrow(mod)
    ccall(
        (:EnzymeDetectReadonlyOrThrow, libEnzyme),
        Cvoid,
        (LLVM.API.LLVMModuleRef,),
        mod,
    )
end

function EnzymeDumpModuleRef(mod)
    ccall((:EnzymeDumpModuleRef, libEnzyme), Cvoid, (LLVM.API.LLVMModuleRef,), mod)
end

EnzymeComputeByteOffsetOfGEP(B, V, T) = LLVM.Value(
    ccall(
        (:EnzymeComputeByteOffsetOfGEP, libEnzyme),
        LLVM.API.LLVMValueRef,
        (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, LLVM.API.LLVMTypeRef),
        B,
        V,
        T,
    ),
)

EnzymeAllocaType(al) = LLVM.LLVMType(
    ccall(
        (:EnzymeAllocaType, libEnzyme),
        LLVM.API.LLVMTypeRef,
        (LLVM.API.LLVMValueRef,),
        al,
    ),
)

EnzymeAttributeKnownFunctions(f) =
    ccall((:EnzymeAttributeKnownFunctions, libEnzyme), Cvoid, (LLVM.API.LLVMValueRef,), f)

EnzymeAnonymousAliasScopeDomain(str, ctx) = LLVM.Metadata(
    ccall(
        (:EnzymeAnonymousAliasScopeDomain, libEnzyme),
        LLVM.API.LLVMMetadataRef,
        (Cstring, LLVMContextRef),
        str,
        ctx,
    ),
)
EnzymeAnonymousAliasScope(dom::LLVM.Metadata, str) = LLVM.Metadata(
    ccall(
        (:EnzymeAnonymousAliasScope, libEnzyme),
        LLVM.API.LLVMMetadataRef,
        (LLVM.API.LLVMMetadataRef, Cstring),
        dom.ref,
        str,
    ),
)
EnzymeFixupJuliaCallingConvention(f) = ccall(
    (:EnzymeFixupJuliaCallingConvention, libEnzyme),
    Cvoid,
    (LLVM.API.LLVMValueRef,),
    f,
)
EnzymeFixupBatchedJuliaCallingConvention(f) = ccall(
    (:EnzymeFixupBatchedJuliaCallingConvention, libEnzyme),
    Cvoid,
    (LLVM.API.LLVMValueRef,),
    f,
)

e_extract_value!(builder, AggVal, Index, Name::String = "") = GC.@preserve Index begin
    LLVM.Value(
        ccall(
            (:EnzymeBuildExtractValue, libEnzyme),
            LLVM.API.LLVMValueRef,
            (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, Ptr{Cuint}, Cuint, Cstring),
            builder,
            AggVal,
            Index,
            length(Index),
            Name,
        ),
    )
end

e_insert_value!(builder, AggVal, EltVal, Index, Name::String = "") =
    GC.@preserve Index begin
        LLVM.Value(
            ccall(
                (:EnzymeBuildInsertValue, libEnzyme),
                LLVM.API.LLVMValueRef,
                (
                    LLVM.API.LLVMBuilderRef,
                    LLVM.API.LLVMValueRef,
                    LLVM.API.LLVMValueRef,
                    Ptr{Cuint},
                    Cuint,
                    Cstring,
                ),
                builder,
                AggVal,
                EltVal,
                Index,
                length(Index),
                Name,
            ),
        )
    end

end
