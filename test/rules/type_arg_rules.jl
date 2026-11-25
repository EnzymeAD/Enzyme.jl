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

# A rule for all `Const` arguments and a rule for `Const{Type{Float64}}`. The second rule is
# more specific, so Enzyme uses it. This is the selection of Julia dispatch before Julia 1.14.
@noinline specific_fn(::Type{T}, x) where {T} = T(2) * x

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig, ::Const{typeof(specific_fn)},
        RT::Type{<:Union{Duplicated, DuplicatedNoNeed}}, Ty::Const, x::Annotation
    )
    shadow = 1.0e3 * x.dval
    return EnzymeRules.needs_primal(config) ? Duplicated(2 * x.val, shadow) : shadow
end
function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig, ::Const{typeof(specific_fn)},
        RT::Type{<:Union{Duplicated, DuplicatedNoNeed}}, Ty::Const{Type{Float64}}, x::Duplicated{Float64}
    )
    shadow = 7.0 * x.dval
    return EnzymeRules.needs_primal(config) ? Duplicated(2 * x.val, shadow) : shadow
end

@testset "Rule with Const{Type{T}} is more specific than a rule for all Const" begin
    specific_f64(x) = specific_fn(Float64, x)
    specific_f32(x) = specific_fn(Float32, x)
    @test autodiff(Forward, specific_f64, Duplicated(3.0, 1.0))[1] == 7.0
    @test autodiff(Forward, specific_f32, Duplicated(3.0, 1.0))[1] == 1.0e3
end

# A rule with a `Const{Type{T}}` argument and keyword arguments.
@noinline scale_kw(::Type{T}, x; s = 1.0) where {T} = T(2) * x * s

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig, ::Const{typeof(scale_kw)},
        RT::Type{<:Union{Duplicated, DuplicatedNoNeed}}, Ty::Const{Type{Float64}}, x::Duplicated{Float64};
        s = 1.0
    )
    shadow = 11.0 * s * x.dval
    return EnzymeRules.needs_primal(config) ? Duplicated(2 * x.val * s, shadow) : shadow
end

@testset "Rule with Const{Type{T}} and keyword arguments" begin
    kw_default(x) = scale_kw(Float64, x)
    kw_given(x) = scale_kw(Float64, x; s = 3.0)
    @test autodiff(Forward, kw_default, Duplicated(3.0, 1.0))[1] == 11.0
    @test autodiff(Forward, kw_given, Duplicated(3.0, 1.0))[1] == 33.0
end

@testset "Rule check finds a rule with Const{Type{T}}" begin
    @test EnzymeRules.has_frule_from_sig(Tuple{typeof(scale_by), Core.Typeof(Float64), Float64})
    @test EnzymeRules.has_rrule_from_sig(Tuple{typeof(scale_by), Core.Typeof(Float64), Float64})
    @test EnzymeRules.has_frule_from_sig(Tuple{Core.Typeof(TypeArgCtor), Float64})
    @test !EnzymeRules.has_frule_from_sig(Tuple{typeof(scale_by), Core.Typeof(Float32), Float64})
end

# Enzyme shows no warning when it applies a rule with a `Const{Type{T}}` argument.
@noinline quiet_fn(::Type{T}, x) where {T} = T(2) * x

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig, ::Const{typeof(quiet_fn)},
        RT::Type{<:Union{Duplicated, DuplicatedNoNeed}}, Ty::Const{Type{Float64}}, x::Duplicated{Float64}
    )
    shadow = 13.0 * x.dval
    return EnzymeRules.needs_primal(config) ? Duplicated(2 * x.val, shadow) : shadow
end

@testset "No warning for a rule with Const{Type{T}}" begin
    quiet_caller(x) = quiet_fn(Float64, x)
    res = @test_nowarn autodiff(Forward, quiet_caller, Duplicated(3.0, 1.0))
    @test res[1] == 13.0
end

@static if isdefined(Core, :TypeEgal)
    # The same as `TypeArgBoth`, but the rule for `Const{TypeEgal{T}}` is the first definition.
    struct TypeArgBothRev
        x::Float64
    end
    @noinline both_rev_use(x) = TypeArgBothRev(x).x

    function EnzymeRules.forward(
            config::EnzymeRules.FwdConfig, ::Const{Core.TypeEgal{TypeArgBothRev}},
            RT::Type{<:Union{Duplicated, DuplicatedNoNeed}}, x::Duplicated{Float64}
        )
        shadow = TypeArgBothRev(5.0 * x.dval)
        return EnzymeRules.needs_primal(config) ? Duplicated(TypeArgBothRev(x.val), shadow) : shadow
    end
    function EnzymeRules.forward(
            config::EnzymeRules.FwdConfig, ::Const{Type{TypeArgBothRev}},
            RT::Type{<:Union{Duplicated, DuplicatedNoNeed}}, x::Duplicated{Float64}
        )
        shadow = TypeArgBothRev(1.0e3 * x.dval)
        return EnzymeRules.needs_primal(config) ? Duplicated(TypeArgBothRev(x.val), shadow) : shadow
    end

    @testset "Rule for Const{TypeEgal{T}} has priority, first definition" begin
        @test autodiff(Forward, both_rev_use, Duplicated(3.0, 1.0))[1] == 5.0
    end

    # One rule with a `Const{Type{T}}` argument and a `Const{TypeEgal{T}}` argument.
    @noinline mixed_fn(::Type{A}, ::Type{B}, x) where {A, B} = A(2) * B(1) * x

    function EnzymeRules.forward(
            config::EnzymeRules.FwdConfig, ::Const{typeof(mixed_fn)},
            RT::Type{<:Union{Duplicated, DuplicatedNoNeed}},
            A::Const{Type{Float64}}, B::Const{Core.TypeEgal{Int}}, x::Duplicated{Float64}
        )
        @assert A.val === Float64
        @assert B.val === Int
        shadow = 19.0 * x.dval
        return EnzymeRules.needs_primal(config) ? Duplicated(2 * x.val, shadow) : shadow
    end
    function EnzymeRules.augmented_primal(
            config::EnzymeRules.RevConfig, ::Const{typeof(mixed_fn)}, RT::Type{<:Active},
            A::Const{Type{Float64}}, B::Const{Core.TypeEgal{Int}}, x::Active{Float64}
        )
        primal = EnzymeRules.needs_primal(config) ? 2 * x.val : nothing
        return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
    end
    function EnzymeRules.reverse(
            config::EnzymeRules.RevConfig, ::Const{typeof(mixed_fn)}, dret::Active, tape,
            A::Const{Type{Float64}}, B::Const{Core.TypeEgal{Int}}, x::Active{Float64}
        )
        return (nothing, nothing, 19.0 * dret.val)
    end

    @testset "Rule with Const{Type{T}} and Const{TypeEgal{T}} arguments" begin
        mixed_caller(x) = mixed_fn(Float64, Int, x)
        @test autodiff(Forward, mixed_caller, Duplicated(3.0, 1.0))[1] == 19.0
        @test autodiff(Reverse, mixed_caller, Active, Active(3.0))[1][1] == 19.0
        @test autodiff(Forward, mixed_fn, Const(Float64), Const(Int), Duplicated(3.0, 1.0))[1] == 19.0
    end
end
