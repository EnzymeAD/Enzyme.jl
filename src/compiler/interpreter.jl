module Interpreter
import Enzyme: API
using Core.Compiler: AbstractInterpreter, InferenceResult, InferenceParams, InferenceState, OptimizationParams, MethodInstance
using GPUCompiler: CodeCache, WorldView, @safe_debug
import ..Enzyme
import ..EnzymeRules

struct EnzymeInterpreter <: AbstractInterpreter
    global_cache::CodeCache
    method_table::Union{Nothing,Core.MethodTable}

    # Cache of inference results for this particular interpreter
    local_cache::Vector{InferenceResult}
    # The world age we're working inside of
    world::UInt

    # Parameters for inference and optimization
    inf_params::InferenceParams
    opt_params::OptimizationParams

    mode::API.CDerivativeMode

    function EnzymeInterpreter(cache::CodeCache, mt::Union{Nothing,Core.MethodTable}, world::UInt, mode::API.CDerivativeMode)
        @assert world <= Base.get_world_counter()

        return new(
            cache,
            mt,

            # Initially empty cache
            Vector{InferenceResult}(),

            # world age counter
            world,

            # parameters for inference and optimization
            InferenceParams(unoptimize_throw_blocks=false),
            VERSION >= v"1.8.0-DEV.486" ? OptimizationParams() :
                                          OptimizationParams(unoptimize_throw_blocks=false),
            mode
        )
    end
end

Core.Compiler.InferenceParams(interp::EnzymeInterpreter) = interp.inf_params
Core.Compiler.OptimizationParams(interp::EnzymeInterpreter) = interp.opt_params
Core.Compiler.get_world_counter(interp::EnzymeInterpreter) = interp.world
Core.Compiler.get_inference_cache(interp::EnzymeInterpreter) = interp.local_cache
Core.Compiler.code_cache(interp::EnzymeInterpreter) = WorldView(interp.global_cache, interp.world)

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
if VERSION >= v"1.7.0-DEV.577"
Core.Compiler.verbose_stmt_info(interp::EnzymeInterpreter) = false
end

if isdefined(Base.Experimental, Symbol("@overlay"))
Core.Compiler.method_table(interp::EnzymeInterpreter, sv::InferenceState) =
    Core.Compiler.OverlayMethodTable(interp.world, interp.method_table)
else

# On 1.6- CUDA.jl will poison the method table at the end of the world
# using GPUCompiler: WorldOverlayMethodTable
# Core.Compiler.method_table(interp::EnzymeInterpreter, sv::InferenceState) =
#     WorldOverlayMethodTable(interp.world)
end

function is_primitive_func(@nospecialize(TT))
    isa(TT, DataType) || return false
    ft = TT.parameters[1]
    if ft == typeof(Enzyme.pmap)
        return true
    end
    if ft === typeof(Base.rem2pi)
        if TT <: Tuple{ft, Float32, <:Any} || TT <: Tuple{ft, Float64, <:Any} || TT <: Tuple{ft, Float16, <:Any}
            return true
        end
    end
    @static if VERSION >= v"1.9-"
    if ft === typeof(Base.rem)
        if TT <: Tuple{ft, Float32, Float32} || TT <: Tuple{ft, Float64, Float64}
            return true
        end
        end
    end

    if ft === typeof(Base.cbrt) || ft === typeof(Base.sin) || ft === typeof(Base.cos) ||
       ft === typeof(Base.sinc) ||
       ft === typeof(Base.tan) || ft === typeof(Base.exp) || ft === typeof(Base.FastMath.exp_fast) ||
       ft === typeof(Base.exp10) ||
       ft === typeof(Base.exp2) ||
       ft === typeof(Base.expm1) ||
       ft === typeof(Base.log) || ft === typeof(Base.FastMath.log) ||
       ft === typeof(Base.log1p) ||
       ft === typeof(Base.log2) ||
       ft === typeof(Base.log10) ||
       ft === typeof(Base.asin) ||
       ft === typeof(Base.acos) ||
       ft === typeof(Base.atan) ||
       ft === typeof(Base.sinpi) ||
       ft === typeof(Base.cospi) ||
       ft === typeof(Base.sinh) || ft === typeof(Base.FastMath.sinh_fast) ||
       ft === typeof(Base.cosh) || ft === typeof(Base.FastMath.cosh_fast) ||
       ft === typeof(Base.tanh) || ft === typeof(Base.FastMath.tanh_fast) ||
       ft === typeof(Base.sqrt) || ft === typeof(Base.sincos) || ft === typeof(Base.sincospi)
        if TT <: Tuple{ft, Float32} || TT <: Tuple{ft, Float64} || TT <: Tuple{ft, Float16}
            return true
        end
    end
@static if VERSION < v"1.8.0"
else
    if ft === typeof(Base.fma_emulated)
        if TT <: Tuple{ft, Float32, Float32, Float32} || TT <: Tuple{ft, Float64, Float64, Float64}
            return true
        end
    end
end
    if ft === typeof(Base.:^) || ft === typeof(Base.atan)
        if TT <: Tuple{ft, Float32, Float32} || TT <: Tuple{ft, Float64, Float64}
            return true
        end
        if TT <: Tuple{ft, Float32, <:Integer} || TT <: Tuple{ft, Float64, <:Integer}
            return true
        end
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
    if VERSION >= v"1.9.0-DEV.1598"
        return TT <: Tuple{typeof(Core.kwcall), Any, Any, Vararg}
    else
        if hasproperty(TT, :parameters) && length(TT.parameters) >= 3
            kwftype = TT.parameters[1]
            ft = TT.parameters[3]
            if ccall(:jl_argument_method_table, Any, (Any,), ft) === nothing
                return false
            end
            if Core.kwftype(ft) == kwftype
                return true
            end
        end
        return false
    end
end

function simplify_kw(specTypes)
    if isKWCallSignature(specTypes)
        return Base.tuple_type_tail(Base.tuple_type_tail(specTypes))
    else
        return specTypes
    end
end

# https://github.com/JuliaLang/julia/pull/46965
@static if VERSION ≥ v"1.9.0-DEV.1535"

import Core.Compiler: CallInfo
function Core.Compiler.inlining_policy(interp::EnzymeInterpreter,
    @nospecialize(src), @nospecialize(info::CallInfo), stmt_flag::UInt8, mi::MethodInstance, argtypes::Vector{Any})

    method_table = Core.Compiler.method_table(interp)
    specTypes = simplify_kw(mi.specTypes)

    if is_primitive_func(specTypes)
        @safe_debug "Blocking inlining for primitive func" mi.specTypes
        return nothing
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

# https://github.com/JuliaLang/julia/pull/41328
elseif isdefined(Core.Compiler, :is_stmt_inline)

function Core.Compiler.inlining_policy(interp::EnzymeInterpreter,
    @nospecialize(src), stmt_flag::UInt8, mi::MethodInstance, argtypes::Vector{Any})

    method_table = Core.Compiler.method_table(interp)
    specTypes = simplify_kw(mi.specTypes)

    if is_primitive_func(specTypes)
        return nothing
    end
    if EnzymeRules.is_inactive_from_sig(specTypes; world = interp.world, method_table)
        return nothing
    end
    if interp.mode == API.DEM_ForwardMode
        if EnzymeRules.has_frule_from_sig(specTypes; world = interp.world, method_table)
            return nothing
        end
    else
        if EnzymeRules.has_rrule_from_sig(specTypes; world = interp.world, method_table)
            return nothing
        end
    end

    return Base.@invoke Core.Compiler.inlining_policy(interp::AbstractInterpreter,
        src::Any, stmt_flag::UInt8, mi::MethodInstance, argtypes::Vector{Any})
end

elseif isdefined(Core.Compiler, :inlining_policy)

import Core.Compiler: InliningTodo, InliningState
struct EnzymeInliningPolicy
    interp::EnzymeInterpreter
end
(::EnzymeInliningPolicy)(@nospecialize(src)) = Core.Compiler.default_inlining_policy(src)
Core.Compiler.inlining_policy(interp::EnzymeInterpreter) = EnzymeInliningPolicy(interp)

function Core.Compiler.resolve_todo(todo::InliningTodo, state::InliningState{S, T, <:EnzymeInliningPolicy}) where {S<:Union{Nothing, Core.Compiler.EdgeTracker}, T}
    mi = todo.mi
    specTypes = simplify_kw(mi.specTypes)

    if is_primitive_func(specTypes)
        return Core.Compiler.compileable_specialization(state.et, todo.spec.match)
    end
    interp = state.policy.interp
    method_table = Core.Compiler.method_table(interp)
    if EnzymeRules.is_inactive_from_sig(specTypes; world = interp.world, method_table)
        return Core.Compiler.compileable_specialization(state.et, todo.spec.match)
    end
    if interp.mode == API.DEM_ForwardMode
        if EnzymeRules.has_frule_from_sig(specTypes; world = interp.world, method_table)
            return Core.Compiler.compileable_specialization(state.et, todo.spec.match)
        end
    else
        if EnzymeRules.has_rrule_from_sig(specTypes; world = interp.world, method_table)
            return Core.Compiler.compileable_specialization(state.et, todo.spec.match)
        end
    end

    return Base.@invoke Core.Compiler.resolve_todo(
        todo::InliningTodo, state::InliningState)
end

end # @static if isdefined(Core.Compiler, :is_stmt_inline)

end
