using Enzyme, Test

@testset "jl_typeof" begin
    # https://github.com/EnzymeAD/Enzyme.jl/issues/2405
    function foo(x)
        @ccall jl_typeof(Ref(x)::Ref{Float64})::Nothing
        x + 1
    end
    @test autodiff(Reverse, foo, Active(1.0))[1][1] == 1.0
end
