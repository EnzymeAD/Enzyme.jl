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

# EnzymeAD/Enzyme.jl#3639: a rule builds split-mode thunks for a function that
# calls `sincos`. The cached thunk IR holds the wrapper that `lower_convention`
# made for `sincos`. The nested module imports that IR.
function rtkm_inner!(f, x)
    for i in eachindex(x)
        @inbounds x[i] = f(x[i])
    end
    return nothing
end
@noinline rtkm_outer!(f, x) = rtkm_inner!(f, x)

function rtkm_thunks(f, x)
    return autodiff_thunk(
        ReverseSplitWithPrimal, Const{typeof(rtkm_inner!)}, Const{Nothing}, Core.Typeof(f), Core.Typeof(x)
    )
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1}, ::Const{typeof(rtkm_outer!)}, ::Type{RT}, f::Const, x::Duplicated
    ) where {RT}
    fwd, _ = rtkm_thunks(f, x)
    return EnzymeRules.AugmentedReturn{Nothing, Nothing, Any}(nothing, nothing, fwd(Const(rtkm_inner!), f, x)[1])
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1}, ::Const{typeof(rtkm_outer!)}, ::Type{RT}, tape, f::Const, x::Duplicated
    ) where {RT}
    _, rev = rtkm_thunks(f, x)
    rev(Const(rtkm_inner!), f, x, tape)
    return (nothing, nothing)
end

function rtkm_g(y)
    s, c = sincos(y)
    return s + 2c
end
rtkm_J(x) = (rtkm_outer!(rtkm_g, x); sum(x))

@testset "Rule that builds thunks for a known math function" begin
    x = [0.1, 0.2]
    dx = zero(x)
    autodiff(Reverse, rtkm_J, Active, Duplicated(x, dx))
    @test dx ≈ cos.([0.1, 0.2]) .- 2 .* sin.([0.1, 0.2])
end

# The same import without a rule: the top-level module calls the cached thunk.
const rtkm_fwd, rtkm_rev = autodiff_thunk(
    ReverseSplitWithPrimal, Const{typeof(rtkm_inner!)}, Const{Nothing}, Const{typeof(rtkm_g)}, Duplicated{Vector{Float64}}
)

function rtkm_h(a, x, dx)
    tape = rtkm_fwd(Const(rtkm_inner!), Const(rtkm_g), Duplicated(x, dx))[1]
    rtkm_rev(Const(rtkm_inner!), Const(rtkm_g), Duplicated(x, dx), tape)
    return a * a
end

@testset "Top-level call to a thunk for a known math function" begin
    res = autodiff(set_runtime_activity(Reverse), rtkm_h, Active, Active(3.0), Const([0.1, 0.2]), Const(ones(2)))
    @test res[1][1] ≈ 6.0
end
