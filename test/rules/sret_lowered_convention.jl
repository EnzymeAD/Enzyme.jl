using Enzyme
using Enzyme.EnzymeCore: EnzymeRules
using Test

function cfun_sret_lowered(v::Vector{Float64})
    z = ComplexF64(v[1], v[2])
    return inv(z) * inv(z + 1)
end

mysum_sret_lowered(x::Vector{Float64}) = sum(x)

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(mysum_sret_lowered)},
        ::Type{RT},
        x::Annotation{<:Vector{Float64}},
    ) where {RT}
    primal = EnzymeRules.needs_primal(config) ? mysum_sret_lowered(x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(mysum_sret_lowered)},
        dret::Active,
        tape,
        x::Annotation{<:Vector{Float64}},
    )
    if !isa(x, Const)
        seed = ones(length(x.val))
        dz = Enzyme.autodiff(
            Forward, cfun_sret_lowered, Duplicated, Duplicated(x.val, seed)
        )[1]
        x.dval .+= dret.val .* real(dz)
    end
    return (nothing,)
end

@testset "sret lowered convention relinked into nested module" begin
    x = [1.0, 2.0, 3.0]
    dx = zero(x)

    Enzyme.autodiff(Reverse, mysum_sret_lowered, Active, Duplicated(x, dx))

    @test all(dx .≈ 0.085)
end
