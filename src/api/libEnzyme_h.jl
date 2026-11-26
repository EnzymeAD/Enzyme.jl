using CEnum: CEnum, @cenum

mutable struct EnzymeOpaqueTypeAnalysis end

const EnzymeTypeAnalysisRef = Ptr{EnzymeOpaqueTypeAnalysis}

mutable struct EnzymeOpaqueLogic end

const EnzymeLogicRef = Ptr{EnzymeOpaqueLogic}

mutable struct EnzymeOpaqueAugmentedReturn end

const EnzymeAugmentedReturnPtr = Ptr{EnzymeOpaqueAugmentedReturn}

mutable struct EnzymeOpaqueTraceInterface end

const EnzymeTraceInterfaceRef = Ptr{EnzymeOpaqueTraceInterface}

struct IntList
    data::Ptr{Int64}
    size::Csize_t
end

@cenum CConcreteType::UInt32 begin
    DT_Anything = 0x0000000000000000
    DT_Integer = 0x0000000000000001
    DT_Pointer = 0x0000000000000002
    DT_Half = 0x0000000000000003
    DT_Float = 0x0000000000000004
    DT_Double = 0x0000000000000005
    DT_Unknown = 0x0000000000000006
    DT_X86_FP80 = 0x0000000000000007
    DT_BFloat16 = 0x0000000000000008
    DT_FP128 = 0x0000000000000009
end

struct CDataPair
    offsets::IntList
    datatype::CConcreteType
end

@cenum CValueType::UInt32 begin
    VT_None = 0x0000000000000000
    VT_Primal = 0x0000000000000001
    VT_Shadow = 0x0000000000000002
    VT_Both = 0x0000000000000003
end

mutable struct EnzymeTypeTree end

const CTypeTreeRef = Ptr{EnzymeTypeTree}

function EnzymeNewTypeTree()
    return @ccall libEnzyme.EnzymeNewTypeTree()::CTypeTreeRef
end

function EnzymeNewTypeTreeCT(arg1, ctx)
    return @ccall libEnzyme.EnzymeNewTypeTreeCT(arg1::CConcreteType, ctx::LLVMContextRef)::CTypeTreeRef
end

function EnzymeNewTypeTreeTR(arg1)
    return @ccall libEnzyme.EnzymeNewTypeTreeTR(arg1::CTypeTreeRef)::CTypeTreeRef
end

function EnzymeFreeTypeTree(CTT)
    return @ccall libEnzyme.EnzymeFreeTypeTree(CTT::CTypeTreeRef)::Cvoid
end

function EnzymeSetTypeTree(dst, src)
    return @ccall libEnzyme.EnzymeSetTypeTree(dst::CTypeTreeRef, src::CTypeTreeRef)::UInt8
end

function EnzymeMergeTypeTree(dst, src)
    return @ccall libEnzyme.EnzymeMergeTypeTree(dst::CTypeTreeRef, src::CTypeTreeRef)::UInt8
end

function EnzymeTypeTreeOnlyEq(dst, x)
    return @ccall libEnzyme.EnzymeTypeTreeOnlyEq(dst::CTypeTreeRef, x::Int64)::Cvoid
end

function EnzymeTypeTreeData0Eq(dst)
    return @ccall libEnzyme.EnzymeTypeTreeData0Eq(dst::CTypeTreeRef)::Cvoid
end

function EnzymeTypeTreeShiftIndiciesEq(dst, datalayout, offset, maxSize, addOffset)
    return @ccall libEnzyme.EnzymeTypeTreeShiftIndiciesEq(dst::CTypeTreeRef, datalayout::Cstring, offset::Int64, maxSize::Int64, addOffset::UInt64)::Cvoid
end

function EnzymeTypeTreeInsertEq(dst, indices, len, ct, ctx)
    return @ccall libEnzyme.EnzymeTypeTreeInsertEq(dst::CTypeTreeRef, indices::Ptr{Int64}, len::Csize_t, ct::CConcreteType, ctx::LLVMContextRef)::Cvoid
end

function EnzymeTypeTreeToString(src)
    return @ccall libEnzyme.EnzymeTypeTreeToString(src::CTypeTreeRef)::Cstring
end

function EnzymeTypeTreeToStringFree(cstr)
    return @ccall libEnzyme.EnzymeTypeTreeToStringFree(cstr::Cstring)::Cvoid
end

function EnzymeSetCLBool(arg1, arg2)
    return @ccall libEnzyme.EnzymeSetCLBool(arg1::Ptr{Cvoid}, arg2::UInt8)::Cvoid
end

function EnzymeSetCLInteger(arg1, arg2)
    return @ccall libEnzyme.EnzymeSetCLInteger(arg1::Ptr{Cvoid}, arg2::Int64)::Cvoid
end

function EnzymeSetCLString(arg1, arg2)
    return @ccall libEnzyme.EnzymeSetCLString(arg1::Ptr{Cvoid}, arg2::Cstring)::Cvoid
end

"""
    CFnTypeInfo

| Field       | Note                                                                       |
| :---------- | :------------------------------------------------------------------------- |
| Arguments   | Types of arguments, assumed of size len(Arguments)                         |
| Return      | Type of return                                                             |
| KnownValues | The specific constant(s) known to represented by an argument, if constant  |
"""
struct CFnTypeInfo
    Arguments::Ptr{CTypeTreeRef}
    Return::CTypeTreeRef
    KnownValues::Ptr{IntList}
end

@cenum CDIFFE_TYPE::UInt32 begin
    DFT_OUT_DIFF = 0x0000000000000000
    DFT_DUP_ARG = 0x0000000000000001
    DFT_CONSTANT = 0x0000000000000002
    DFT_DUP_NONEED = 0x0000000000000003
end

@cenum CBATCH_TYPE::UInt32 begin
    BT_SCALAR = 0x0000000000000000
    BT_VECTOR = 0x0000000000000001
end

@cenum CDerivativeMode::UInt32 begin
    DEM_ForwardMode = 0x0000000000000000
    DEM_ReverseModePrimal = 0x0000000000000001
    DEM_ReverseModeGradient = 0x0000000000000002
    DEM_ReverseModeCombined = 0x0000000000000003
    DEM_ForwardModeSplit = 0x0000000000000004
    DEM_ForwardModeError = 0x0000000000000005
end

@cenum CProbProgMode::UInt32 begin
    DEM_Trace = 0x0000000000000000
    DEM_Condition = 0x0000000000000001
end

# typedef uint8_t ( * CustomRuleType ) ( int /*direction*/ , CTypeTreeRef /*return*/ , CTypeTreeRef * /*args*/ , struct IntList * /*knownValues*/ , size_t /*numArgs*/ , LLVMValueRef , void * /*TA*/ )
const CustomRuleType = Ptr{Cvoid}

function CreateTypeAnalysis(Log, customRuleNames, customRules, numRules)
    return @ccall libEnzyme.CreateTypeAnalysis(Log::EnzymeLogicRef, customRuleNames::Ptr{Cstring}, customRules::Ptr{CustomRuleType}, numRules::Csize_t)::EnzymeTypeAnalysisRef
end

function ClearTypeAnalysis(arg1)
    return @ccall libEnzyme.ClearTypeAnalysis(arg1::EnzymeTypeAnalysisRef)::Cvoid
end

function FreeTypeAnalysis(arg1)
    return @ccall libEnzyme.FreeTypeAnalysis(arg1::EnzymeTypeAnalysisRef)::Cvoid
end

function EnzymeTypeAnalysisGetLogic(TAR)
    return @ccall libEnzyme.EnzymeTypeAnalysisGetLogic(TAR::EnzymeTypeAnalysisRef)::EnzymeLogicRef
end

function EnzymeGetTypeAnalysisFromTypeAnalyzer(TAR)
    return @ccall libEnzyme.EnzymeGetTypeAnalysisFromTypeAnalyzer(TAR::Ptr{Cvoid})::EnzymeTypeAnalysisRef
end

function FindEnzymeStaticTraceInterface(M)
    return @ccall libEnzyme.FindEnzymeStaticTraceInterface(M::LLVMModuleRef)::EnzymeTraceInterfaceRef
end

function CreateEnzymeStaticTraceInterface(C, getTraceFunction, getChoiceFunction, insertCallFunction, insertChoiceFunction, insertArgumentFunction, insertReturnFunction, insertFunctionFunction, insertChoiceGradientFunction, insertArgumentGradientFunction, newTraceFunction, freeTraceFunction, hasCallFunction, hasChoiceFunction)
    return @ccall libEnzyme.CreateEnzymeStaticTraceInterface(C::LLVMContextRef, getTraceFunction::LLVMValueRef, getChoiceFunction::LLVMValueRef, insertCallFunction::LLVMValueRef, insertChoiceFunction::LLVMValueRef, insertArgumentFunction::LLVMValueRef, insertReturnFunction::LLVMValueRef, insertFunctionFunction::LLVMValueRef, insertChoiceGradientFunction::LLVMValueRef, insertArgumentGradientFunction::LLVMValueRef, newTraceFunction::LLVMValueRef, freeTraceFunction::LLVMValueRef, hasCallFunction::LLVMValueRef, hasChoiceFunction::LLVMValueRef)::EnzymeTraceInterfaceRef
end

function CreateEnzymeDynamicTraceInterface(interface, F)
    return @ccall libEnzyme.CreateEnzymeDynamicTraceInterface(interface::LLVMValueRef, F::LLVMValueRef)::EnzymeTraceInterfaceRef
end

function CreateEnzymeLogic(PostOpt)
    return @ccall libEnzyme.CreateEnzymeLogic(PostOpt::UInt8)::EnzymeLogicRef
end

function ClearEnzymeLogic(arg1)
    return @ccall libEnzyme.ClearEnzymeLogic(arg1::EnzymeLogicRef)::Cvoid
end

function FreeEnzymeLogic(arg1)
    return @ccall libEnzyme.FreeEnzymeLogic(arg1::EnzymeLogicRef)::Cvoid
end

function EnzymeLogicSetExternalContext(arg1, ExternalContext)
    return @ccall libEnzyme.EnzymeLogicSetExternalContext(arg1::EnzymeLogicRef, ExternalContext::Ptr{Cvoid})::Cvoid
end

function EnzymeLogicGetExternalContext(arg1)
    return @ccall libEnzyme.EnzymeLogicGetExternalContext(arg1::EnzymeLogicRef)::Ptr{Cvoid}
end

function EnzymeExtractReturnInfo(ret, data, existed, len)
    return @ccall libEnzyme.EnzymeExtractReturnInfo(ret::EnzymeAugmentedReturnPtr, data::Ptr{Int64}, existed::Ptr{UInt8}, len::Csize_t)::Cvoid
end

function EnzymeExtractFunctionFromAugmentation(ret)
    return @ccall libEnzyme.EnzymeExtractFunctionFromAugmentation(ret::EnzymeAugmentedReturnPtr)::LLVMValueRef
end

function EnzymeExtractTapeTypeFromAugmentation(ret)
    return @ccall libEnzyme.EnzymeExtractTapeTypeFromAugmentation(ret::EnzymeAugmentedReturnPtr)::LLVMTypeRef
end

# typedef LLVMValueRef ( * CustomShadowAlloc ) ( LLVMBuilderRef , LLVMValueRef , size_t /*numArgs*/ , LLVMValueRef * , GradientUtils * )
const CustomShadowAlloc = Ptr{Cvoid}

# typedef LLVMValueRef ( * CustomShadowFree ) ( LLVMBuilderRef , LLVMValueRef )
const CustomShadowFree = Ptr{Cvoid}

function EnzymeRegisterAllocationHandler(Name, AHandle, FHandle)
    return @ccall libEnzyme.EnzymeRegisterAllocationHandler(Name::Cstring, AHandle::CustomShadowAlloc, FHandle::CustomShadowFree)::Cvoid
end

# typedef uint8_t ( * CustomFunctionForward ) ( LLVMBuilderRef , LLVMValueRef , GradientUtils * , LLVMValueRef * , LLVMValueRef * )
const CustomFunctionForward = Ptr{Cvoid}

# typedef uint8_t ( * CustomFunctionDiffUse ) ( LLVMValueRef , const GradientUtils * , LLVMValueRef , uint8_t , CDerivativeMode , uint8_t * )
const CustomFunctionDiffUse = Ptr{Cvoid}

# typedef uint8_t ( * CustomAugmentedFunctionForward ) ( LLVMBuilderRef , LLVMValueRef , GradientUtils * , LLVMValueRef * , LLVMValueRef * , LLVMValueRef * )
const CustomAugmentedFunctionForward = Ptr{Cvoid}

# typedef void ( * CustomFunctionReverse ) ( LLVMBuilderRef , LLVMValueRef , DiffeGradientUtils * , LLVMValueRef )
const CustomFunctionReverse = Ptr{Cvoid}

function EnzymeCreateForwardDiff(Logic, request_req, request_ip, todiff, retType, constant_args, constant_args_size, TA, returnValue, mode, freeMemory, runtimeActivity, strongZero, width, additionalArg, typeInfo, subsequent_calls_may_write, _overwritten_args, overwritten_args_size, augmented)
    return @ccall libEnzyme.EnzymeCreateForwardDiff(Logic::EnzymeLogicRef, request_req::LLVMValueRef, request_ip::LLVMBuilderRef, todiff::LLVMValueRef, retType::CDIFFE_TYPE, constant_args::Ptr{CDIFFE_TYPE}, constant_args_size::Csize_t, TA::EnzymeTypeAnalysisRef, returnValue::UInt8, mode::CDerivativeMode, freeMemory::UInt8, runtimeActivity::UInt8, strongZero::UInt8, width::Cuint, additionalArg::LLVMTypeRef, typeInfo::CFnTypeInfo, subsequent_calls_may_write::UInt8, _overwritten_args::Ptr{UInt8}, overwritten_args_size::Csize_t, augmented::EnzymeAugmentedReturnPtr)::LLVMValueRef
end

function EnzymeCreatePrimalAndGradient(Logic, request_req, request_ip, todiff, retType, constant_args, constant_args_size, TA, returnValue, dretUsed, mode, runtimeActivity, strongZero, width, freeMemory, additionalArg, forceAnonymousTape, typeInfo, subsequent_calls_may_write, _overwritten_args, overwritten_args_size, augmented, AtomicAdd)
    return @ccall libEnzyme.EnzymeCreatePrimalAndGradient(Logic::EnzymeLogicRef, request_req::LLVMValueRef, request_ip::LLVMBuilderRef, todiff::LLVMValueRef, retType::CDIFFE_TYPE, constant_args::Ptr{CDIFFE_TYPE}, constant_args_size::Csize_t, TA::EnzymeTypeAnalysisRef, returnValue::UInt8, dretUsed::UInt8, mode::CDerivativeMode, runtimeActivity::UInt8, strongZero::UInt8, width::Cuint, freeMemory::UInt8, additionalArg::LLVMTypeRef, forceAnonymousTape::UInt8, typeInfo::CFnTypeInfo, subsequent_calls_may_write::UInt8, _overwritten_args::Ptr{UInt8}, overwritten_args_size::Csize_t, augmented::EnzymeAugmentedReturnPtr, AtomicAdd::UInt8)::LLVMValueRef
end

function EnzymeRegisterCallHandler(Name, FwdHandle, RevHandle)
    return @ccall libEnzyme.EnzymeRegisterCallHandler(Name::Cstring, FwdHandle::CustomAugmentedFunctionForward, RevHandle::CustomFunctionReverse)::Cvoid
end

function EnzymeGradientUtilsNewFromOriginal(gutils, val)
    return @ccall libEnzyme.EnzymeGradientUtilsNewFromOriginal(gutils::Ptr{Cvoid}, val::LLVMValueRef)::LLVMValueRef
end

function EnzymeGradientUtilsLookup(gutils, val, builder)
    return @ccall libEnzyme.EnzymeGradientUtilsLookup(gutils::Ptr{Cvoid}, val::LLVMValueRef, builder::LLVMBuilderRef)::LLVMValueRef
end

function EnzymeGradientUtilsInvertPointer(gutils, val, builder)
    return @ccall libEnzyme.EnzymeGradientUtilsInvertPointer(gutils::Ptr{Cvoid}, val::LLVMValueRef, builder::LLVMBuilderRef)::LLVMValueRef
end

function EnzymeGradientUtilsGetExternalContext(gutils)
    return @ccall libEnzyme.EnzymeGradientUtilsGetExternalContext(gutils::Ptr{Cvoid})::Ptr{Cvoid}
end

function EnzymeTypeAnalyzerGetLogic(analyzer)
    return @ccall libEnzyme.EnzymeTypeAnalyzerGetLogic(analyzer::Ptr{Cvoid})::EnzymeLogicRef
end

function EnzymeGradientUtilsGetLogic(gutils)
    return @ccall libEnzyme.EnzymeGradientUtilsGetLogic(gutils::Ptr{Cvoid})::EnzymeLogicRef
end

function EnzymeGradientUtilsGetAtomicAdd(gutils)
    return @ccall libEnzyme.EnzymeGradientUtilsGetAtomicAdd(gutils::Ptr{Cvoid})::UInt8
end
