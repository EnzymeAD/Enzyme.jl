using Test
using Enzyme
import Enzyme: ignore_derivatives

@testset "ignore_derivatives" begin
    @test autodiff(Enzyme.Forward, ignore_derivatives, Duplicated(1.0, 2.0)) == (0.0,)
    @test autodiff(Enzyme.Reverse, ignore_derivatives, Active(1.0)) == ((0.0,),)
end

N(xᵢ, θ) = θ[1] * xᵢ^2 + θ[2] * xᵢ
N_stop(xᵢ, θ) = θ[1] * ignore_derivatives(xᵢ^2) + θ[2] * ignore_derivatives(xᵢ)

@testset "simulate with ignore_derivatives" begin
    x₀ = -0.3
    θ = (-4.0, 4.0)

    dθ = MixedDuplicated(θ, Ref(Enzyme.make_zero(θ)))
    @test Enzyme.autodiff(Enzyme.Reverse, N, Active(x₀), dθ) == ((6.4, nothing),)
    @test dθ.dval[] == (0.09, -0.3)


    dθ = MixedDuplicated(θ, Ref(Enzyme.make_zero(θ)))
    @test Enzyme.autodiff(Enzyme.Reverse, N_stop, Active(x₀), dθ) == ((0.0, nothing),)
    @test dθ.dval[] == (0.09, -0.3)
end
