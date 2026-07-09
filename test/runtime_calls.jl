using Enzyme, Test, Logging

@testset "jl_typeof" begin
    # https://github.com/EnzymeAD/Enzyme.jl/issues/2405
    function foo(x)
        @ccall jl_typeof(Ref(x)::Ref{Float64})::Any
        x + 1
    end
    @test autodiff(Reverse, foo, Active(1.0))[1][1] == 1.0
end

@testset "jl_f_current_scope" begin
    function foo_logger(p)
        sol = Logging.with_logger(Logging.current_logger()) do
            p[1] * p[2]
        end
        return sol
    end
    g = Enzyme.gradient(set_runtime_activity(Reverse), foo_logger, [2.0, 3.0])[1]
    @test g ≈ [3.0, 2.0]
end

