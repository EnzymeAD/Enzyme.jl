module TuringIntegrationTests

using Distributions: Distributions
using DynamicPPL: DynamicPPL
using Enzyme: Enzyme
using FiniteDifferences: FiniteDifferences
using LinearAlgebra: LinearAlgebra
using Random: randn
using Test: @test, @testset

# TODO(mhauru) Could we at some point make do without this?
Enzyme.API.runtimeActivity!(true)

"""
Turn a Turing model, possibly with given example values, into a log density function and a
random value that it can be evaluated at.
"""
function build_turing_problem(model)
    ctx = DynamicPPL.DefaultContext()
    vi = DynamicPPL.VarInfo(model)
    vi_linked = DynamicPPL.link(vi, model)
    ldp = DynamicPPL.LogDensityFunction(vi_linked, model, ctx)
    test_function = Base.Fix1(DynamicPPL.LogDensityProblems.logdensity, ldp)
    d = DynamicPPL.LogDensityProblems.dimension(ldp)
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
models = collect(DynamicPPL.TestUtils.DEMO_MODELS)

# Add some other models that use features that have previously been problematic for Enzyme.

DynamicPPL.@model function MvDirichletWithManualAccumulation(w, doc)
    β ~ DynamicPPL.filldist(Distributions.Dirichlet([1.0, 1.0]), 2)
    log_product = log.(β)
    DynamicPPL.@addlogprob! sum(log_product[CartesianIndex.(w, doc)])
end

push!(models, MvDirichletWithManualAccumulation([1, 1, 1, 1], [1, 1, 2, 2]))

DynamicPPL.@model function demo_lkjchol(d::Int=2)
    x ~ DynamicPPL.LKJCholesky(d, 1.0)
    return (x=x,)
end

push!(models, demo_lkjchol())

DynamicPPL.@model function hmcmatrixsup()
    return v ~ Distributions.Wishart(7, [1 0.5; 0.5 1])
end

push!(models, hmcmatrixsup())

DynamicPPL.@model function mvnormal_with_transpose(x=transpose([1.5 2.0;]))
    m ~ Distributions.MvNormal(LinearAlgebra.Diagonal([1.0, 1.0]))
    x .~ Distributions.MvNormal(m, LinearAlgebra.Diagonal([1.0, 1.0]))
    return nothing
end

push!(models, mvnormal_with_transpose())

DynamicPPL.@model function mvnorm_with_argtype(::Type{TV}=Matrix{Float64}) where {TV}
    P0 = vcat([0.1 0.0], [0.0 0.1])
    x = TV(undef, 2, 2)
    fill!(x, zero(eltype(x)))
    x[:, 2] ~ Distributions.MvNormal(x[:, 1], P0)
    return nothing
end

push!(models, mvnorm_with_argtype())

# Test each model in turn, checking Enzyme's gradient against FiniteDifferences.
@testset "Turing integration tests" begin
    @testset "$(typeof(model.f))" for model in models
        f, x = build_turing_problem(model)
        test_grad(f, x)
    end
end

end
