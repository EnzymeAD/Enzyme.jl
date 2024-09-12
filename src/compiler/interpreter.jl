module Interpreter
import Enzyme: API
using Core.Compiler: AbstractInterpreter, InferenceResult, InferenceParams, InferenceState, OptimizationParams, MethodInstance
using GPUCompiler: @safe_debug
if VERSION < v"1.11.0-DEV.1552"
    using GPUCompiler: CodeCache, WorldView, @safe_debug
end
const HAS_INTEGRATED_CACHE = VERSION >= v"1.11.0-DEV.1552"

import ..Enzyme
import ..EnzymeRules

@static if VERSION ≥ v"1.11.0-DEV.1498"
    import Core.Compiler: get_inference_world
    using Base: get_world_counter
else
    import Core.Compiler: get_world_counter, get_world_counter as get_inference_world
end

struct EnzymeInterpreter <: AbstractInterpreter
@static if HAS_INTEGRATED_CACHE
    token::Any
else
    code_cache::CodeCache
end
    method_table::Union{Nothing,Core.MethodTable}

    # Cache of inference results for this particular interpreter
    local_cache::Vector{InferenceResult}
    # The world age we're working inside of
    world::UInt

    # Parameters for inference and optimization
    inf_params::InferenceParams
    opt_params::OptimizationParams

    mode::API.CDerivativeMode
end

function EnzymeInterpreter(cache_or_token, mt::Union{Nothing,Core.MethodTable}, world::UInt, mode::API.CDerivativeMode)
    @assert world <= Base.get_world_counter()

    return EnzymeInterpreter(
        cache_or_token,
        mt,

        # Initially empty cache
        Vector{InferenceResult}(),

        # world age counter
        world,

        # parameters for inference and optimization
        InferenceParams(unoptimize_throw_blocks=false),
        OptimizationParams(),
        mode
    )
end

Core.Compiler.InferenceParams(interp::EnzymeInterpreter) = interp.inf_params
Core.Compiler.OptimizationParams(interp::EnzymeInterpreter) = interp.opt_params
get_inference_world(interp::EnzymeInterpreter) = interp.world
Core.Compiler.get_inference_cache(interp::EnzymeInterpreter) = interp.local_cache
@static if HAS_INTEGRATED_CACHE
    Core.Compiler.cache_owner(interp::EnzymeInterpreter) = interp.token
else
    Core.Compiler.code_cache(interp::EnzymeInterpreter) = WorldView(interp.code_cache, interp.world)
end

# No need to do any locking since we're not putting our results into the runtime cache
Core.Compiler.lock_mi_inference(interp::EnzymeInterpreter, mi::MethodInstance) = nothing
Core.Compiler.unlock_mi_inference(interp::EnzymeInterpreter, mi::MethodInstance) = nothing

function Core.Compiler.add_remark!(interp::EnzymeInterpreter, sv::InferenceState, msg)
end

Core.Compiler.may_optimize(interp::EnzymeInterpreter) = true
Core.Compiler.may_compress(interp::EnzymeInterpreter) = true
# From @aviatesk:
#     `may_discard_trees = true`` means a complicated (in terms of inlineability) source will be discarded,
#      but as far as I understand Enzyme wants "always inlining, except special cased functions",
#      so I guess we really don't want to discard sources?
Core.Compiler.may_discard_trees(interp::EnzymeInterpreter) = false
Core.Compiler.verbose_stmt_info(interp::EnzymeInterpreter) = false

if isdefined(Base.Experimental, Symbol("@overlay"))
Core.Compiler.method_table(interp::EnzymeInterpreter, sv::InferenceState) =
    Core.Compiler.OverlayMethodTable(interp.world, interp.method_table)
else

# On 1.6- CUDA.jl will poison the method table at the end of the world
# using GPUCompiler: WorldOverlayMethodTable
# Core.Compiler.method_table(interp::EnzymeInterpreter, sv::InferenceState) =
#     WorldOverlayMethodTable(interp.world)
end

function is_alwaysinline_func(@nospecialize(TT))
    isa(TT, DataType) || return false
    return false
end

function is_primitive_func(@nospecialize(TT))
    isa(TT, DataType) || return false
    ft = TT.parameters[1]
    if ft == typeof(Enzyme.pmap)
        return true
    end
    match = Enzyme.Compiler.find_math_method(ft, TT.parameters[2:end])[1]
    if match !== nothing
        return true
    end

    # FIXME(@wsmoses): For which types should we not inline?
    if ft === typeof(Base.wait) || ft === typeof(Base._wait) || ft === typeof(Base.enq_work) ||
       ft === typeof(Base.Threads.threadid) || ft == typeof(Base.Threads.nthreads) ||
       ft === typeof(Base.Threads.threading_run)
        return true
    end
    return false
end

function isKWCallSignature(@nospecialize(TT))
    return TT <: Tuple{typeof(Core.kwcall), Any, Any, Vararg}
end

function simplify_kw(specTypes)
    if isKWCallSignature(specTypes)
        return Base.tuple_type_tail(Base.tuple_type_tail(specTypes))
    else
        return specTypes
    end
end

# https://github.com/JuliaLang/julia/pull/46965
import Core.Compiler: CallInfo
function Core.Compiler.inlining_policy(interp::EnzymeInterpreter,
    @nospecialize(src), @nospecialize(info::CallInfo), stmt_flag::UInt8, mi::MethodInstance, argtypes::Vector{Any})

    method_table = Core.Compiler.method_table(interp)
    specTypes = simplify_kw(mi.specTypes)

    if is_primitive_func(specTypes)
        @safe_debug "Blocking inlining for primitive func" mi.specTypes
        return nothing
    end

    if is_alwaysinline_func(specTypes)
        @safe_debug "Forcing inlining for primitive func" mi.specTypes
        @assert src !== nothing
        return src
    end

    if EnzymeRules.is_inactive_from_sig(specTypes; world = interp.world, method_table)
        @safe_debug "Blocking inlining due to inactive rule" mi.specTypes
        return nothing
    end

    if interp.mode == API.DEM_ForwardMode
        if EnzymeRules.has_frule_from_sig(specTypes; world = interp.world, method_table)
            @safe_debug "Blocking inlining due to frule" mi.specTypes
            return nothing
        end
    else
        if EnzymeRules.has_rrule_from_sig(specTypes; world = interp.world, method_table)
            @safe_debug "Blocking inling due to rrule" mi.specTypes
            return nothing
        end
    end

    return Base.@invoke Core.Compiler.inlining_policy(interp::AbstractInterpreter,
        src::Any, info::CallInfo, stmt_flag::UInt8, mi::MethodInstance, argtypes::Vector{Any})
end

end
