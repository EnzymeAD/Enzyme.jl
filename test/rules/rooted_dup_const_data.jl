# Custom rules whose argument is an immutable struct with duplicated GC-tracked fields and
# constant inline data of a differentiable type (a closure capturing duplicated arrays and
# a constant scalar). The argument is passed as duplicated, with a zero shadow for the
# constant data.
# https://github.com/EnzymeAD/Enzyme.jl/issues/3875
module RootedDupConstData

using Enzyme
using Enzyme.EnzymeCore: EnzymeRules
using Test

struct Job{F}
    f::F
    n::Int
end
(j::Job)(c) = j.f(c)

@noinline function run_job(f, n::Int)
    for c in 1:n
        f(c)
    end
    return nothing
end

call(g, c) = (g(c); nothing)

function EnzymeRules.forward(
        config, ::Const{typeof(run_job)}, ::Type{RT}, f::Annotation, n::Const{Int}
    ) where {RT}
    for c in 1:n.val
        autodiff(Forward, Const(call), Const, f, Const(c))
    end
    return nothing
end

function EnzymeRules.augmented_primal(
        config, ::Const{typeof(run_job)}, ::Type{RT}, f::Annotation, n::Const{Int}
    ) where {RT}
    run_job(f.val, n.val)
    return EnzymeRules.AugmentedReturn(nothing, nothing, nothing)
end

function EnzymeRules.reverse(
        config, ::Const{typeof(run_job)}, ::Type{RT}, tape, f::Annotation, n::Const{Int}
    ) where {RT}
    for c in n.val:-1:1
        autodiff(Reverse, Const(call), Const, f, Const(c))
    end
    return (nothing, nothing)
end

function scaled_square_sum(x, a)
    y = similar(x)
    run_job(Job(i -> (y[i] = a * x[i]^2), length(x)), length(x))
    return sum(y)
end

function const_square_sum(x)
    y = similar(x)
    c = 2.0
    run_job(Job(i -> (y[i] = c * x[i]^2), length(x)), length(x))
    return sum(y)
end

@testset "Duplicated roots with constant float data" begin
    x = collect(1.0:4.0)
    # The constant `a` must have a zero tangent: `sum(2a .* x)`, not `+ sum(x .^ 2)`
    @test autodiff(Forward, scaled_square_sum, Duplicated(x, ones(4)), Const(2.0))[1] ≈
        sum(2 * 2.0 .* x)
    @test autodiff(Forward, scaled_square_sum, Duplicated(x, ones(4)), Duplicated(2.0, 1.0))[1] ≈
        sum(2 * 2.0 .* x) + sum(x .^ 2)
    @test Enzyme.gradient(Reverse, const_square_sum, x)[1] ≈ 4x
    @test autodiff(Forward, const_square_sum, Duplicated(x, ones(4)))[1] ≈ sum(4x)
end

end # module
