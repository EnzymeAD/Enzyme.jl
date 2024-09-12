module TuringIntegrationTests

using Test: @test, @testset
using Random: randn
using Enzyme: Enzyme
using Turing: Turing
using FiniteDifferences: FiniteDifferences

# TODO(mhauru) Could we at some point make do without this?
Enzyme.API.runtimeActivity!(true)

"""
Turn a Turing model, possibly with given example values, into a log density function and a
random value that it can be evaluated at.
"""
function build_turing_problem(model)
    ctx = Turing.DefaultContext()
    vi = Turing.VarInfo(model)
    vi_linked = Turing.link(vi, model)
    ldp = Turing.LogDensityFunction(vi_linked, model, ctx)
    test_function = Base.Fix1(Turing.LogDensityProblems.logdensity, ldp)
    d = Turing.LogDensityProblems.dimension(ldp)
    return test_function, randn(d)
end

"""
Test Enzyme.gradient, both Forward and Reverse mode, against FiniteDifferences.grad, for a
given function f and argument x.
"""
function test_grad(f, x; rtol=1e-6, atol=1e-6)
    finitediff = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), f, x)[1]
    # TODO(mhauru) The Val(1) works around https://github.com/EnzymeAD/Enzyme.jl/issues/1807
    @test(
        collect(Enzyme.gradient(Enzyme.Forward, Enzyme.Const(f), x, Val(1))) ≈ finitediff,
        rtol = rtol,
        atol = atol
    )
    @test(
        Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(f), x) ≈ finitediff,
        rtol = rtol, atol = atol
    )
    return nothing
end

# Turing models to test with. These come from Turing's test suite.
models = collect(Turing.DynamicPPL.TestUtils.DEMO_MODELS)

# Add some other models that use features that have previously been problematic for Enzyme.

Turing.@model function MvDirichletWithManualAccumulation(w, doc)
    β ~ Turing.filldist(Turing.Dirichlet([1.0, 1.0]), 2)
    log_product = log.(β)
    Turing.@addlogprob! sum(log_product[CartesianIndex.(w, doc)])
end

push!(models, MvDirichletWithManualAccumulation([1, 1, 1, 1], [1, 1, 2, 2]))

Turing.@model function demo_lkjchol(d::Int=2)
    x ~ Turing.LKJCholesky(d, 1.0)
    return (x=x,)
end

push!(models, demo_lkjchol())

# Test each model in turn, checking Enzyme's gradient against FiniteDifferences.
@testset "Turing integration tests" begin
    @testset "$(typeof(model.f))" for model in models
        f, x = build_turing_problem(model)
        test_grad(f, x)
    end
end

end
