using Test
using Enzyme
using Checkpointing
using LinearAlgebra

# Explicit 1D heat equation
mutable struct Heat
    Tnext::Vector{Float64}
    Tlast::Vector{Float64}
    n::Int
    λ::Float64
    tsteps::Int
end

function advance(heat)
    next = heat.Tnext
    last = heat.Tlast
    λ = heat.λ
    n = heat.n
    for i in 2:(n - 1)
        next[i] = last[i] + λ * (last[i - 1] - 2 * last[i] + last[i + 1])
    end
    return nothing
end


function sumheat(heat::Heat, chkpscheme::Scheme, tsteps::Int64)
    @ad_checkpoint chkpscheme for i in 1:tsteps
        heat.Tlast .= heat.Tnext
        advance(heat)
    end
    return reduce(+, heat.Tnext)
end

function heat(scheme::Scheme, tsteps::Int)
    n = 100
    Δx = 0.1
    Δt = 0.001
    # Select μ such that λ ≤ 0.5 for stability with μ = (λ*Δt)/Δx^2
    λ = 0.5

    # Create object from struct. tsteps is not needed for a for-loop
    heat = Heat(zeros(n), zeros(n), n, λ, tsteps)
    # Shadow copy for Enzyme
    dheat = Heat(zeros(n), zeros(n), n, λ, tsteps)

    # Boundary conditions
    heat.Tnext[1] = 20.0
    heat.Tnext[end] = 0

    # Compute gradient
    autodiff(Enzyme.ReverseWithPrimal, sumheat, Duplicated(heat, dheat), Const(scheme), Const(tsteps))

    return heat.Tnext, dheat.Tnext[2:(end - 1)]
end

T, dT = heat(Revolve(4), 500)
@test norm(T) == 66.21987468492061
@test norm(dT) == 6.970279349365908
