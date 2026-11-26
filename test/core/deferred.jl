using Enzyme, Test
using Enzyme: API
using Enzyme.Compiler: DeferredSpec, DeferredOnHostError, UnknownTapeType
const GPUCompiler = Enzyme.Compiler.GPUCompiler

deferred_square(x) = x * x

@testset "deferred" begin

    @testset "Host calls are an error" begin
        function dot2(A)
            return A[1] * A[1] + A[2] * A[2]
        end
        A = [3.0, 5.0]
        dA = zeros(2)
        @test_throws DeferredOnHostError autodiff_deferred(ReverseWithPrimal, Const(dot2), Active, Duplicated(A, dA))
        @test_throws DeferredOnHostError autodiff_deferred(Forward, Const(dot2), Duplicated, Duplicated(A, dA))
        TapeType = Enzyme.EnzymeCore.tape_type(ReverseSplitWithPrimal, Const{typeof(dot2)}, Active, Duplicated{typeof(A)})
        @test_throws DeferredOnHostError autodiff_deferred_thunk(
            ReverseSplitWithPrimal, TapeType, Const{typeof(dot2)}, Active, Duplicated{typeof(A)}
        )

        # Invalid requests raise their error before the host check.
        @test_throws MethodError autodiff_deferred(Reverse, Const(dot2), Active, Active(1.0), Active(2.0))

        # The host entry points do the job.
        @test autodiff(ReverseWithPrimal, dot2, Active, Duplicated(A, dA))[2] == 34.0
        @test dA == [6.0, 10.0]
    end

    @testset "Requests are specified at the type level" begin
        spec = Enzyme.Compiler.deferred_spec(
            Const{typeof(deferred_square)}, Active, Tuple{Active{Float64}},
            Val(API.DEM_ReverseModeCombined), Val(1), Val((false, false)), Val(false),
            Val(false), UnknownTapeType, Val(false), Val(false), Val(false),
        )
        @test spec isa Val
        S = typeof(spec).parameters[1]
        @test S <: DeferredSpec

        # Nothing is registered with GPUCompiler while inferring or running the request.
        n = length(GPUCompiler.deferred_codegen_jobs)
        @test_throws DeferredOnHostError autodiff_deferred(Reverse, Const(deferred_square), Active, Active(2.0))
        @test length(GPUCompiler.deferred_codegen_jobs) == n

        # GPUCompiler creates the job from the specification when it compiles the request.
        world = Base.get_world_counter()
        job = Enzyme.Compiler.deferred_job(S, world)
        @test job isa GPUCompiler.CompilerJob
        @test job.source.specTypes == Tuple{typeof(deferred_square), Float64}
        @test !job.config.kernel
        @test GPUCompiler.deferred_job(S, job) isa GPUCompiler.CompilerJob
    end

    @testset "Deferred upgrade" begin
        function gradsin(x)
            return gradient(Reverse, sin, x)[1]
        end
        res = Enzyme.gradient(Reverse, gradsin, 3.1)[1]
        @test res ≈ -sin(3.1)
    end

end # testset "deferred"
