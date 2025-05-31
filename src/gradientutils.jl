import LLVM: refcheck
import GPUCompiler
LLVM.@checked struct GradientUtils
    ref::API.EnzymeGradientUtilsRef
end
Base.unsafe_convert(::Type{API.EnzymeGradientUtilsRef}, gutils::GradientUtils) = gutils.ref
LLVM.dispose(gutils::GradientUtils) = throw("Cannot free gutils")

function call_samefunc_with_inverted_bundles!(
    B::LLVM.IRBuilder,
    gutils::GradientUtils,
    orig::LLVM.CallInst,
    args::Vector{<:LLVM.Value},
    valTys::Vector{API.CValueType},
    lookup::Bool,
)
    @assert length(args) == length(valTys)
    return LLVM.Value(
        API.EnzymeGradientUtilsCallWithInvertedBundles(
            gutils,
            LLVM.called_operand(orig),
            LLVM.called_type(orig),
            args,
            length(args),
            orig,
            valTys,
            length(valTys),
            B,
            false,
        ),
    ) #=lookup=#
end

get_width(gutils::GradientUtils) = API.EnzymeGradientUtilsGetWidth(gutils)
get_mode(gutils::GradientUtils) = API.EnzymeGradientUtilsGetMode(gutils)
get_runtime_activity(gutils::GradientUtils) =
    API.EnzymeGradientUtilsGetRuntimeActivity(gutils)

get_strong_zero(gutils::GradientUtils) =
    API.EnzymeGradientUtilsGetStrongZero(gutils)

function get_shadow_type(gutils::GradientUtils, T::LLVM.LLVMType)
    w = get_width(gutils)
    if w == 1
        return T
    else
        return LLVM.ArrayType(T, Int(w))
    end
end
function get_uncacheable(gutils::GradientUtils, orig::LLVM.CallInst)
    uncacheable = Vector{UInt8}(undef, length(collect(LLVM.operands(orig))) - 1)
    if API.EnzymeGradientUtilsGetUncacheableArgs(
        gutils,
        orig,
        uncacheable,
        length(uncacheable),
    ) != 1
        uncacheable .= 1
    end
    return uncacheable
end

erase_with_placeholder(
    gutils::GradientUtils,
    inst::LLVM.Instruction,
    orig::LLVM.Instruction,
    erase::Bool = true,
) = API.EnzymeGradientUtilsEraseWithPlaceholder(gutils, inst, orig, erase)
is_constant_value(gutils::GradientUtils, val::LLVM.Value) =
    API.EnzymeGradientUtilsIsConstantValue(gutils, val) != 0

is_constant_inst(gutils::GradientUtils, inst::LLVM.Instruction) =
    API.EnzymeGradientUtilsIsConstantInstruction(gutils, inst) != 0

new_from_original(gutils::GradientUtils, val::LLVM.Value) =
    LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, val))

lookup_value(gutils::GradientUtils, val::LLVM.Value, B::LLVM.IRBuilder) =
    LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, val, B))

invert_pointer(gutils::GradientUtils, val::LLVM.Value, B::LLVM.IRBuilder) =
    LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, val, B))

function debug_from_orig!(
    gutils::GradientUtils,
    nval::LLVM.Instruction,
    oval::LLVM.Instruction,
)
    API.EnzymeGradientUtilsSetDebugLocFromOriginal(gutils, nval, oval)
    nothing
end
