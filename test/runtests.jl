using Test

using InteractiveUtils
using Enzyme
Enzyme.API.printall!(true)

@testset "Simple tests" begin
    f1(x) = 1.0 + x
    f2(x) = x*x
    for T in (Float16,)
        @show @code_typed autodiff(Reverse, tanh, Active, Active(T(1)))
        res = autodiff(Reverse, tanh, Active, Active(T(1)))[1][1]
        @test res isa T
        cmp = if T == Float64
            T(0.41997434161402606939)
        else
            T(0.41997434161402606939f0)
        end
        @test res ≈ cmp
        @show @code_typed autodiff(Forward, tanh, Duplicated(T(1), T(1)))
        res = autodiff(Forward, tanh, Duplicated(T(1), T(1)))[1]
        @test res isa T
        @test res ≈ cmp
    end

end
