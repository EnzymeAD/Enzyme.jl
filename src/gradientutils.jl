import LLVM: refcheck
import GPUCompiler
LLVM.@checked struct GradientUtils
    ref::API.EnzymeGradientUtilsRef
end
Base.unsafe_convert(::Type{API.EnzymeGradientUtilsRef}, gutils::GradientUtils) = gutils.ref
LLVM.dispose(gutils::GradientUtils) = throw("Cannot free gutils")

function call_samefunc_with_inverted_bundles!(B::LLVM.IRBuilder, gutils::GradientUtils, orig::LLVM.CallInst, args::Vector{LLVM.Value}, vals::Vector{API.CValueType}, lookup::Bool)
    @assert length(args) == length(vals)
    return LLVM.Value(API.EnzymeGradientUtilsCallWithInvertedBundles(gutils, LLVM.called_value(LLVM.Value(OrigCI)), vals, length(vals), orig, valTys, length(valTys), B, #=lookup=#false))
end