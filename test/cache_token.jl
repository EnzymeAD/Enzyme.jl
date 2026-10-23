using Enzyme, Test
using Enzyme.Compiler: EnzymeTarget, EnzymeCompilerParams, PrimalCompilerParams, UnknownTapeType
using Enzyme: API, FFIABI
const GPUCompiler = Enzyme.Compiler.GPUCompiler
using .GPUCompiler: CompilerJob, CompilerConfig

cache_token_fn(x::Float64) = sin(x) * x

function token_jobs(mode, world)
    mi = Enzyme.Compiler.my_methodinstance(Reverse, typeof(cache_token_fn), Tuple{Float64}, world)
    params = EnzymeCompilerParams(
        Tuple{Const{typeof(cache_token_fn)}, Active{Float64}}, mode, 1, Active{Float64},
        true, true, (false, false), false, false, UnknownTapeType, FFIABI, false, false, false,
    )
    # The job `thunk` compiles, and the primal job `codegen` derives from it.
    thunk_job = CompilerJob(mi, CompilerConfig(EnzymeTarget(), params; kernel = false), world)
    primal_job = CompilerJob(mi, CompilerConfig(thunk_job.config.target.target, params.params; kernel = false), world)
    return thunk_job, primal_job
end

@static if VERSION >= v"1.11.0-DEV.1552"
    @testset "one cache owner per mode" begin
        world = Base.get_world_counter()
        for (mode, sugar) in ((API.DEM_ReverseModeCombined, Reverse), (API.DEM_ForwardMode, Forward))
            thunk_job, primal_job = token_jobs(mode, world)
            thunk_token = Enzyme.Compiler.enzyme_cache_owner(thunk_job)
            primal_token = Enzyme.Compiler.enzyme_cache_owner(primal_job)
            query_token = Enzyme.Compiler.primal_interp_world(sugar, world).token
            # Owners are matched with jl_egal, so `===` is exactly the comparison that matters.
            @test thunk_token === primal_token
            @test thunk_token === query_token
            @test Core.Compiler.cache_owner(GPUCompiler.get_interpreter(thunk_job)) === primal_token
        end
        # Forward and reverse mode still get separate owners: rule inlining differs between them.
        rev_job, _ = token_jobs(API.DEM_ReverseModeCombined, world)
        fwd_job, _ = token_jobs(API.DEM_ForwardMode, world)
        @test Enzyme.Compiler.enzyme_cache_owner(rev_job) !== Enzyme.Compiler.enzyme_cache_owner(fwd_job)
    end
end
