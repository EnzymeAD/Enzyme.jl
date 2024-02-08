using InteractiveUtils
using Enzyme
using Test
using FiniteDifferences

Enzyme.API.printall!(true)

# Test against FiniteDifferences
function test_scalar(f, x; rtol=1e-9, atol=1e-9, fdm=central_fdm(5, 1), kwargs...)
    @show @code_llvm autodiff(ForwardMode{InlineABI}(), f, Duplicated(x, one(typeof(x))))
    ∂x, = autodiff(ForwardMode{InlineABI}(), f, Duplicated(x, one(typeof(x))))
    @show "inline", x, ∂x, f(x), one(typeof(x)) * f(x)
    
    @show @code_llvm autodiff(Forward, f, Duplicated(x, one(typeof(x))))
    ∂x, = autodiff(Forward, f, Duplicated(x, one(typeof(x))))
    @show "normal", x, ∂x, f(x), one(typeof(x)) * f(x)
    @test isapprox(∂x, fdm(f, x); rtol=rtol, atol=atol, kwargs...)
end

@testset "Simple tests" begin
    test_scalar(Base.FastMath.exp_fast, 1.0)
end
