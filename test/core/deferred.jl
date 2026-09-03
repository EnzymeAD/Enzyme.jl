using Enzyme, Test
using Enzyme: API
using Enzyme.Compiler: DeferredSpec, deferred_id, DeferredOnHostError, DEFERRED_SPECS,
    rebuild_deferred_registry!, UnknownTapeType
const GPUCompiler = Enzyme.Compiler.GPUCompiler

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

        # The host entry points do the job.
        @test autodiff(ReverseWithPrimal, dot2, Active, Duplicated(A, dA))[2] == 34.0
        @test dA == [6.0, 10.0]
    end

    @testset "Ids derive from the specification" begin
        make_spec(width) = DeferredSpec(
            Const{typeof(sin)}, Active, Tuple{Active{Float64}}, API.DEM_ReverseModeCombined,
            width, (false, false), false, false, UnknownTapeType, false, false, false,
        )
        id = deferred_id(make_spec(1))
        @test id > 0
        @test deferred_id(make_spec(1)) == id
        @test deferred_id(make_spec(2)) != id
        # Never one of GPUCompiler's own ids; the twin id is distinct and keeps that bit.
        @test reinterpret(UInt, id) & Enzyme.Compiler.DEFERRED_ID_BIT != 0
        twin = Enzyme.Compiler.deferred_twin_id(id)
        @test twin != id
        @test reinterpret(UInt, twin) & Enzyme.Compiler.DEFERRED_ID_BIT != 0
        @test Enzyme.Compiler.deferred_twin_id(twin) == id

        # Another session derives the same id.
        code = """
        using Enzyme
        using Enzyme: API
        using Enzyme.Compiler: DeferredSpec, deferred_id, UnknownTapeType
        print(deferred_id(DeferredSpec(
            Const{typeof(sin)}, Active, Tuple{Active{Float64}}, API.DEM_ReverseModeCombined,
            1, (false, false), false, false, UnknownTapeType, false, false, false,
        )))
        """
        cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $code`
        @test parse(Int, read(cmd, String)) == id
    end

    @testset "Registry rebuild" begin
        ids = collect(keys(DEFERRED_SPECS))
        @test !isempty(ids)
        for id in ids
            delete!(GPUCompiler.deferred_codegen_jobs, id)
        end
        @test rebuild_deferred_registry!() >= length(ids)
        for id in ids
            @test GPUCompiler.deferred_codegen_jobs[id] isa GPUCompiler.CompilerJob
            # Both markers of a call site resolve to one job.
            @test GPUCompiler.deferred_codegen_jobs[Enzyme.Compiler.deferred_twin_id(id)] ===
                GPUCompiler.deferred_codegen_jobs[id]
        end
    end

    @testset "Deferred upgrade" begin
        function gradsin(x)
            return gradient(Reverse, sin, x)[1]
        end
        res = Enzyme.gradient(Reverse, gradsin, 3.1)[1]
        @test res ≈ -sin(3.1)
    end

end # testset "deferred"
