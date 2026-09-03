using Enzyme, Test
using Enzyme.Compiler: EnzymeTarget, EnzymeCompilerParams, PrimalCompilerParams, UnknownTapeType
using Enzyme: API, FFIABI
const GPUCompiler = Enzyme.Compiler.GPUCompiler
using .GPUCompiler: CompilerJob, CompilerConfig

# A pretend backend that, like CUDA.jl, defines its method table only for its own job type.
struct MTTestTarget <: GPUCompiler.AbstractCompilerTarget end
struct MTTestParams <: GPUCompiler.AbstractCompilerParams end
Base.Experimental.@MethodTable(mt_test_table)
GPUCompiler.method_table(@nospecialize(job::CompilerJob{MTTestTarget, MTTestParams})) = mt_test_table

mt_test_fn(x::Float64) = x
Base.Experimental.@overlay mt_test_table mt_test_fn(x::Float64) = "overlaid"
# The overlay is picked up at call sites, so infer a caller.
mt_test_caller(x::Float64) = mt_test_fn(x)

function enzyme_job(inner_target, inner_params, mode, world)
    mi = Enzyme.Compiler.my_methodinstance(Reverse, typeof(mt_test_caller), Tuple{Float64}, world)
    params = EnzymeCompilerParams(
        Tuple{Const{typeof(mt_test_caller)}, Active{Float64}}, mode, 1, Active{Float64},
        true, true, (false, false), false, false, UnknownTapeType, FFIABI, false, false, false,
    )
    target = GPUCompiler.nest_target(EnzymeTarget(), inner_target)
    params = GPUCompiler.nest_params(params, inner_params)
    return mi, CompilerJob(mi, CompilerConfig(target, params; kernel = false), world)
end

@testset "method_table forwards through EnzymeTarget" begin
    world = Base.get_world_counter()
    mode = API.DEM_ReverseModeCombined

    mi, native_job = enzyme_job(GPUCompiler.NativeCompilerTarget(), PrimalCompilerParams(mode), mode, world)
    @test GPUCompiler.method_table(native_job) === GPUCompiler.GLOBAL_METHOD_TABLE
    @test Enzyme.Compiler.return_type(GPUCompiler.get_interpreter(native_job), mi) === Float64

    mi, backend_job = enzyme_job(MTTestTarget(), MTTestParams(), mode, world)
    @test backend_job.config.target isa EnzymeTarget{MTTestTarget}
    @test backend_job.config.params isa EnzymeCompilerParams{MTTestParams}
    @test GPUCompiler.method_table(backend_job) === mt_test_table

    interp = GPUCompiler.get_interpreter(backend_job)
    @test Core.Compiler.method_table(interp).mt === mt_test_table
    # Inference under the Enzyme job now sees the backend's overlay, as the primal job does.
    @test Enzyme.Compiler.return_type(interp, mi) === String
    @static if VERSION >= v"1.11.0-DEV.1552"
        @test GPUCompiler.ci_cache_token(backend_job).method_table === mt_test_table
    end
end
