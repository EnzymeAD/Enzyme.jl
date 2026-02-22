module InactiveKWRules

using Enzyme
using Enzyme.EnzymeRules
using Test

import .EnzymeRules: forward, augmented_primal, reverse

function f_kw(out; tmp = [2.0, 0.0])
    out[1] *= tmp[1]
    tmp[2] += 1
    return nothing
end

function forward(config, ::Const{typeof(f_kw)}, ::Type{<:Const}, x::Duplicated; kwargs...)
    f_kw(x.val; kwargs...)
    f_kw(x.dval; kwargs...)
    return nothing
end

function augmented_primal(config, ::Const{typeof(f_kw)}, ::Type{<:Const}, x::Duplicated; kwargs...)
    f_kw(x.val; kwargs...)
    return EnzymeRules.AugmentedReturn(nothing, nothing, nothing)
end

function reverse(config, ::Const{typeof(f_kw)}, ::Type{<:Const}, tape, x::Duplicated; kwargs...)
    f_kw(x.dval; kwargs...)
    return (nothing,)
end

function g_kw(out)
    tmp = [2.0, 0.0]
    f_kw(out; tmp)
    return nothing
end

function h_kw(out, tmp)
    f_kw(out; tmp)
    return nothing
end

@testset "Forward Inactive allocated kwarg error" begin
    x = [2.7]
    dx = [3.1]
    @test_throws Enzyme.Compiler.NonConstantKeywordArgException autodiff(Forward, g_kw, Duplicated(x, dx))
end

@testset "Reverse Inactive allocated kwarg error" begin
    x = [2.7]
    dx = [3.1]
    @test_throws Enzyme.Compiler.NonConstantKeywordArgException autodiff(Reverse, g_kw, Duplicated(x, dx))
end

@testset "Forward Inactive arg kwarg error" begin
    x = [2.7]
    dx = [3.1]

    tmp = [2.0, 0.0]
    dtmp = [7.1, 9.4]
    @test_throws Enzyme.Compiler.NonConstantKeywordArgException autodiff(Forward, h_kw, Duplicated(x, dx), Duplicated(tmp, dtmp))
end

@testset "Reverse Inactive arg kwarg error" begin
    x = [2.7]
    dx = [3.1]

    tmp = [2.0, 0.0]
    dtmp = [7.1, 9.4]
    @test_throws Enzyme.Compiler.NonConstantKeywordArgException autodiff(Forward, h_kw, Duplicated(x, dx), Duplicated(tmp, dtmp))
end

Enzyme.EnzymeRules.inactive_kwarg(::typeof(f_kw), out; tmp = [2.0]) = nothing

@testset "Forward Inactive allocated kwarg success" begin
    x = [2.7]
    dx = [3.1]
    autodiff(Forward, g_kw, Duplicated(x, dx))
    @test x ≈ [2.7 * 2.0]
    @test dx ≈ [3.1 * 2.0]
end

@testset "Reverse Inactive allocated kwarg success" begin
    x = [2.7]
    dx = [3.1]
    autodiff(Reverse, g_kw, Duplicated(x, dx))
    @test x ≈ [2.7 * 2.0]
    @test dx ≈ [3.1 * 2.0]
end

@testset "Forward Inactive arg kwarg success" begin
    x = [2.7]
    dx = [3.1]

    tmp = [2.0, 0.0]
    dtmp = [7.1, 9.4]
    autodiff(Forward, h_kw, Duplicated(x, dx), Duplicated(tmp, dtmp))

    @test x ≈ [2.7 * 2.0]
    @test dx ≈ [3.1 * 2.0]
end

@testset "Reverse Inactive arg kwarg success" begin
    x = [2.7]
    dx = [3.1]

    tmp = [2.0, 0.0]
    dtmp = [7.1, 9.4]
    autodiff(Reverse, h_kw, Duplicated(x, dx), Duplicated(tmp, dtmp))

    @test x ≈ [2.7 * 2.0]
    @test dx ≈ [3.1 * 2.0]
end

end # InactiveKWRules
