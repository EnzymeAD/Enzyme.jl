module Interpreter

using Core.Compiler: AbstractInterpreter, InferenceResult, InferenceParams, InferenceState, OptimizationParams, MethodInstance
using GPUCompiler: CodeCache, WorldView
using ....Enzyme: pmap
struct EnzymeInterpeter <: AbstractInterpreter
    global_cache::CodeCache
    method_table::Union{Nothing,Core.MethodTable}

    # Cache of inference results for this particular interpreter
    local_cache::Vector{InferenceResult}
    # The world age we're working inside of
    world::UInt

    # Parameters for inference and optimization
    inf_params::InferenceParams
    opt_params::OptimizationParams

    function EnzymeInterpeter(cache::CodeCache, mt::Union{Nothing,Core.MethodTable}, world::UInt)
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
        )
    end
end

Core.Compiler.InferenceParams(interp::EnzymeInterpeter) = interp.inf_params
Core.Compiler.OptimizationParams(interp::EnzymeInterpeter) = interp.opt_params
Core.Compiler.get_world_counter(interp::EnzymeInterpeter) = interp.world
Core.Compiler.get_inference_cache(interp::EnzymeInterpeter) = interp.local_cache
Core.Compiler.code_cache(interp::EnzymeInterpeter) = WorldView(interp.global_cache, interp.world)

# No need to do any locking since we're not putting our results into the runtime cache
Core.Compiler.lock_mi_inference(interp::EnzymeInterpeter, mi::MethodInstance) = nothing
Core.Compiler.unlock_mi_inference(interp::EnzymeInterpeter, mi::MethodInstance) = nothing

function Core.Compiler.add_remark!(interp::EnzymeInterpeter, sv::InferenceState, msg)
end

Core.Compiler.may_optimize(interp::EnzymeInterpeter) = true
Core.Compiler.may_compress(interp::EnzymeInterpeter) = true
# From @aviatesk:
#     `may_discard_trees = true`` means a complicated (in terms of inlineability) source will be discarded,
#      but as far as I understand Enzyme wants "always inlining, except special cased functions",
#      so I guess we really don't want to discard sources?
Core.Compiler.may_discard_trees(interp::EnzymeInterpeter) = false
if VERSION >= v"1.7.0-DEV.577"
Core.Compiler.verbose_stmt_info(interp::EnzymeInterpeter) = false
end

if isdefined(Base.Experimental, Symbol("@overlay"))
Core.Compiler.method_table(interp::EnzymeInterpeter, sv::InferenceState) =
    Core.Compiler.OverlayMethodTable(interp.world, interp.method_table)
else
Core.Compiler.method_table(interp::EnzymeInterpeter, sv::InferenceState) =
    GPUCompiler.WorldOverlayMethodTable(interp.world)
end

const PrimitiveFuncs = Set([typeof(Base.string), typeof(Base.eps), typeof(Base.nextfloat), typeof(Base.prevfloat), typeof(pmap),
                            typeof(Base.to_tuple_type)])

function is_primitive_func(@nospecialize(TT))
    isa(TT, DataType) || return false
    ft = TT.parameters[1]
    if in(ft, PrimitiveFuncs)
       return true
    end
    if ft === typeof(Base.cbrt) || ft === typeof(Base.sin) || ft === typeof(Base.cos) ||
       ft === typeof(Base.tan) || ft === typeof(Base.exp) || 
       ft === typeof(Base.log) ||
       ft === typeof(Base.log2) ||
       ft === typeof(Base.log10) ||
       ft === typeof(Base.asin) || ft === typeof(Base.tanh) || ft === typeof(Base.FastMath.tanh_fast) ||
       ft === typeof(Base.sqrt) || ft === typeof(Base.sincos)
        if TT <: Tuple{ft, Float32} || TT <: Tuple{ft, Float64} || TT <: Tuple{ft, Float16}
            return true
        end
    end
    if ft === typeof(Base.:^)
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


# branch on https://github.com/JuliaLang/julia/pull/41328
@static if isdefined(Core.Compiler, :is_stmt_inline)

function Core.Compiler.inlining_policy(
    interp::EnzymeInterpeter, @nospecialize(src), stmt_flag::UInt8,
    mi::MethodInstance, argtypes::Vector{Any})

    if is_primitive_func(mi.specTypes)
        return nothing
    end

    return Base.@invoke Core.Compiler.inlining_policy(
        interp::AbstractInterpreter, src::Any, stmt_flag::UInt8,
        mi::MethodInstance, argtypes::Vector{Any})
end

elseif isdefined(Core.Compiler, :inlining_policy)

import Core.Compiler: InliningTodo, InliningState
enzyme_inlining_policy(@nospecialize(src)) = Core.Compiler.default_inlining_policy(src)
Core.Compiler.inlining_policy(::EnzymeInterpeter) = enzyme_inlining_policy
function Core.Compiler.resolve_todo(todo::InliningTodo, state::InliningState{S, T, <:typeof(enzyme_inlining_policy)}) where {S<:Union{Nothing, Core.Compiler.EdgeTracker}, T}
    mi = todo.mi
    if is_primitive_func(mi.specTypes)
        return Core.Compiler.compileable_specialization(state.et, todo.spec.match)
    end

    return Base.@invoke Core.Compiler.resolve_todo(
        todo::InliningTodo, state::InliningState)
end

end # @static if isdefined(Core.Compiler, :is_stmt_inline)

end
