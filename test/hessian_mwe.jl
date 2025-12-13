using Enzyme
using Test
using LinearAlgebra

# Minimal Working Example for Hessian computation
# This test exercises Forward-over-Reverse mode which requires proper handling of sret types

@testset "Hessian MWE" begin
    # Original function that computes f(x) = x[1]^2 + x[1]*x[2]
    function origf(x::Array{Float64}, y::Array{Float64})
        y[1] = x[1] * x[1] + x[2] * x[1]
        return nothing
    end

    # Gradient computation using reverse mode
    function grad(x, dx, y, dy)
        Enzyme.autodiff(Reverse, Const(origf), Duplicated(x, dx), DuplicatedNoNeed(y, dy))
        nothing
    end

    # Setup input
    x = [2.0, 2.0]
    y = Vector{Float64}(undef, 1)
    dx = [0.0, 0.0]
    dy = [1.0]

    # Compute gradient
    grad(x, dx, y, dy)

    # Setup for Hessian computation via Forward-over-Reverse
    vx = ([1.0, 0.0], [0.0, 1.0])
    hess = ([0.0, 0.0], [0.0, 0.0])
    dx2 = [0.0, 0.0]
    dy = [1.0]

    # Compute Hessian using Forward mode over gradient function
    Enzyme.autodiff(
        Enzyme.Forward, grad,
        Enzyme.BatchDuplicated(x, vx),
        Enzyme.BatchDuplicated(dx2, hess),
        Const(y),
        Const(dy)
    )

    # Verify results
    @test dx ≈ dx2
    @test hess[1][1] ≈ 2.0  # ∂²f/∂x₁² = 2
    @test hess[1][2] ≈ 1.0  # ∂²f/∂x₁∂x₂ = 1
    @test hess[2][1] ≈ 1.0  # ∂²f/∂x₂∂x₁ = 1
    @test hess[2][2] ≈ 0.0  # ∂²f/∂x₂² = 0
end
