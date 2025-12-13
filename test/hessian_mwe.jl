#!/usr/bin/env julia
# Minimal Working Example (MWE) for testing enzyme_sret type handling
# This file isolates the core functionality being tested to help diagnose timeouts

using Enzyme
using Test
using LinearAlgebra

println("Testing Forward-over-Reverse (Hessian) computation...")
println("This test exercises sret type handling in nested AD")

@testset "Hessian MWE" begin
    # Function: f(x) = x[1]^2 + x[1]*x[2]
    # Gradient: ∇f = [2*x[1] + x[2], x[1]]
    # Hessian: H = [[2, 1], [1, 0]]
    
    function origf(x::Array{Float64}, y::Array{Float64})
        y[1] = x[1] * x[1] + x[2] * x[1]
        return nothing
    end

    function grad(x, dx, y, dy)
        Enzyme.autodiff(Reverse, Const(origf), Duplicated(x, dx), DuplicatedNoNeed(y, dy))
        nothing
    end

    x = [2.0, 2.0]
    y = Vector{Float64}(undef, 1)
    dx = [0.0, 0.0]
    dy = [1.0]

    grad(x, dx, y, dy)

    vx = ([1.0, 0.0], [0.0, 1.0])
    hess = ([0.0, 0.0], [0.0, 0.0])
    dx2 = [0.0, 0.0]
    dy = [1.0]

    Enzyme.autodiff(
        Enzyme.Forward, grad,
        Enzyme.BatchDuplicated(x, vx),
        Enzyme.BatchDuplicated(dx2, hess),
        Const(y),
        Const(dy)
    )

    @test dx ≈ dx2
    @test hess[1][1] ≈ 2.0
    @test hess[1][2] ≈ 1.0
    @test hess[2][1] ≈ 1.0
    @test hess[2][2] ≈ 0.0
end

println("\nTest completed successfully!")
println("If this test hangs or times out, the issue is with Forward-over-Reverse AD")
println("with BatchDuplicated arguments, which requires proper enzyme_sret handling.")
