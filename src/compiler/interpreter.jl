using Core.Compiler: AbstractInterpreter, InferenceResult, InferenceParams, InferenceState, OptimizationParams, MethodInstance
using GPUCompiler: CodeCache, WorldView

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
Core.Compiler.may_discard_trees(interp::EnzymeInterpeter) = true
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