using Enzyme
using Enzyme: EnzymeRules
using Test

# On Julia 1.14, `Core.Typeof(T)` for a type value `T` is `Core.TypeEgal{T}`, and
# `Const(T)` is `Const{Core.TypeEgal{T}}`. These tests make sure that a rule that is
# written with `::Const{Type{T}}` still applies. Each rule returns a derivative that is
# different from the true derivative, so a test fails if Enzyme differentiates the primal.

struct TypeArgCtor
    x::Float64
end

@noinline ctor_use(x) = TypeArgCtor(x).x

@noinline scale_by(::Type{T}, x) where {T} = T(2) * x
@noinline scale_after(x, ::Type{T}) where {T} = T(2) * x

const CTOR_FACTOR = 42.0
const SCALE_FACTOR = 17.0

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig, ::Const{Type{TypeArgCtor}},
        RT::Type{<:Union{Duplicated, BatchDuplicated, DuplicatedNoNeed, BatchDuplicatedNoNeed}},
        x::Union{Duplicated{Float64}, BatchDuplicated{Float64}}
    )
    primal = TypeArgCtor(x.val)
    shadow = if EnzymeRules.width(config) == 1
        TypeArgCtor(CTOR_FACTOR * x.dval)
    else
        ntuple(i -> TypeArgCtor(CTOR_FACTOR * x.dval[i]), Val(EnzymeRules.width(config)))
    end
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return EnzymeRules.width(config) == 1 ? Duplicated(primal, shadow) : BatchDuplicated(primal, shadow)
    elseif EnzymeRules.needs_shadow(config)
        return shadow
    elseif EnzymeRules.needs_primal(config)
        return primal
    end
    return nothing
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, ::Const{Type{TypeArgCtor}},
        RT::Type{<:Active}, x::Active{Float64}
    )
    primal = EnzymeRules.needs_primal(config) ? TypeArgCtor(x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, ::Const{Type{TypeArgCtor}},
        dret::Active, tape, x::Active{Float64}
    )
    return (CTOR_FACTOR * dret.val.x,)
end

for (fn, Targ) in ((:scale_by, :(Ty::Const{Type{Float64}})), (:scale_after, :(Ty::Const{Type{Float64}})))
    fwdargs = fn == :scale_by ? (Targ, :(x::Union{Duplicated, BatchDuplicated})) : (:(x::Union{Duplicated, BatchDuplicated}), Targ)
    augargs = fn == :scale_by ? (Targ, :(x::Active)) : (:(x::Active), Targ)
    revret = fn == :scale_by ? :((nothing, SCALE_FACTOR * dret.val)) : :((SCALE_FACTOR * dret.val, nothing))
    @eval begin
        function EnzymeRules.forward(
                config::EnzymeRules.FwdConfig, ::Const{typeof($fn)},
                RT::Type{<:Union{Duplicated, BatchDuplicated, DuplicatedNoNeed, BatchDuplicatedNoNeed}},
                $(fwdargs...)
            )
            @assert Ty.val === Float64
            primal = Ty.val(2) * x.val
            shadow = if EnzymeRules.width(config) == 1
                SCALE_FACTOR * x.dval
            else
                ntuple(i -> SCALE_FACTOR * x.dval[i], Val(EnzymeRules.width(config)))
            end
            if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
                return EnzymeRules.width(config) == 1 ? Duplicated(primal, shadow) : BatchDuplicated(primal, shadow)
            elseif EnzymeRules.needs_shadow(config)
                return shadow
            elseif EnzymeRules.needs_primal(config)
                return primal
            end
            return nothing
        end

        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfig, ::Const{typeof($fn)},
                RT::Type{<:Active}, $(augargs...)
            )
            @assert Ty.val === Float64
            primal = EnzymeRules.needs_primal(config) ? Ty.val(2) * x.val : nothing
            return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
        end

        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfig, ::Const{typeof($fn)},
                dret::Active, tape, $(augargs...)
            )
            return $revret
        end
    end
end

@testset "Rule with Const{Type{T}} for a type as the function" begin
    @test autodiff(Forward, ctor_use, Duplicated(3.0, 1.0))[1] == CTOR_FACTOR
    @test autodiff(ForwardWithPrimal, ctor_use, Duplicated(3.0, 1.0)) == (CTOR_FACTOR, 3.0)
    res = autodiff(Forward, ctor_use, BatchDuplicated(3.0, (1.0, 2.0)))[1]
    @test res[1] == CTOR_FACTOR
    @test res[2] == 2 * CTOR_FACTOR
    @test autodiff(Reverse, ctor_use, Active, Active(3.0))[1][1] == CTOR_FACTOR
    @test autodiff(ReverseWithPrimal, ctor_use, Active, Active(3.0)) == ((CTOR_FACTOR,), 3.0)
end

@testset "Rule with Const{Type{T}} for a later type argument" begin
    wrap_by(x) = scale_by(Float64, x)
    wrap_after(x) = scale_after(x, Float64)

    @test autodiff(Forward, wrap_by, Duplicated(3.0, 1.0))[1] == SCALE_FACTOR
    @test autodiff(Forward, wrap_after, Duplicated(3.0, 1.0))[1] == SCALE_FACTOR
    @test autodiff(ForwardWithPrimal, wrap_by, Duplicated(3.0, 1.0)) == (SCALE_FACTOR, 6.0)

    res = autodiff(Forward, wrap_by, BatchDuplicated(3.0, (1.0, 2.0)))[1]
    @test res[1] == SCALE_FACTOR
    @test res[2] == 2 * SCALE_FACTOR
    res = autodiff(Forward, wrap_after, BatchDuplicated(3.0, (1.0, 2.0)))[1]
    @test res[1] == SCALE_FACTOR
    @test res[2] == 2 * SCALE_FACTOR

    @test autodiff(Reverse, wrap_by, Active, Active(3.0))[1][1] == SCALE_FACTOR
    @test autodiff(Reverse, wrap_after, Active, Active(3.0))[1][1] == SCALE_FACTOR
    @test autodiff(ReverseWithPrimal, wrap_after, Active, Active(3.0)) == ((SCALE_FACTOR,), 6.0)

    # The type is the argument that `autodiff` passes in.
    @test autodiff(Forward, scale_by, Const(Float64), Duplicated(3.0, 1.0))[1] == SCALE_FACTOR
    @test autodiff(Reverse, scale_after, Active, Active(3.0), Const(Float64))[1][1] == SCALE_FACTOR
end

@static if isdefined(Core, :TypeEgal)
    # When a rule for `Const{TypeEgal{T}}` and a rule for `Const{Type{T}}` both exist,
    # Enzyme uses the rule for `Const{TypeEgal{T}}`.
    struct TypeArgBoth
        x::Float64
    end
    @noinline both_use(x) = TypeArgBoth(x).x

    function EnzymeRules.forward(
            config::EnzymeRules.FwdConfig, ::Const{Type{TypeArgBoth}},
            RT::Type{<:Union{Duplicated, DuplicatedNoNeed}}, x::Duplicated{Float64}
        )
        shadow = TypeArgBoth(1.0e3 * x.dval)
        return EnzymeRules.needs_primal(config) ? Duplicated(TypeArgBoth(x.val), shadow) : shadow
    end
    function EnzymeRules.forward(
            config::EnzymeRules.FwdConfig, ::Const{Core.TypeEgal{TypeArgBoth}},
            RT::Type{<:Union{Duplicated, DuplicatedNoNeed}}, x::Duplicated{Float64}
        )
        shadow = TypeArgBoth(5.0 * x.dval)
        return EnzymeRules.needs_primal(config) ? Duplicated(TypeArgBoth(x.val), shadow) : shadow
    end

    @testset "Rule for Const{TypeEgal{T}} has priority" begin
        @test autodiff(Forward, both_use, Duplicated(3.0, 1.0))[1] == 5.0
    end
end
