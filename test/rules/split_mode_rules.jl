using Enzyme
using Test
using Enzyme.EnzymeRules

# Custom reverse rules reached through a dynamic call, which differentiates the
# caller in split mode (separate augmented and gradient passes). The structs
# are immutable with an inline Float64 and a heap field, so they are passed by
# reference with a separate roots array, and their shadows are MixedDuplicated.

struct SplitP
    x::Float64
    v::Vector{Float64}
end

struct SplitSol
    u::Vector{Float64}
    a::Float64
end

# The rule reads the tape, so the gradient pass must configure it (needs_primal)
# as the augmented pass did to find the same tape type.
@noinline split_tape(p::SplitP) = p.v .* p.x

function EnzymeRules.augmented_primal(config::RevConfigWidth{1}, ::Const{typeof(split_tape)},
        RT::Type{<:Duplicated}, p)
    res = split_tape(p.val)
    return EnzymeRules.augmented_rule_return_type(config, RT, p.val.x)(res, zero(res), p.val.x)
end
function EnzymeRules.reverse(config::RevConfigWidth{1}, ::Const{typeof(split_tape)},
        ::Type{<:Duplicated}, tape::Float64, p)
    if !(p isa Const)
        dp = p isa MixedDuplicated ? p.dval[] : p.dval
        dp.v .+= tape
    end
    return (nothing,)
end

# The return is a struct with a heap field, returned through an sret whose
# shadow only exists in the reverse pass.
@noinline split_sret(v::Vector{Float64}) = SplitSol(2 .* v, 1.0)

function EnzymeRules.augmented_primal(config::RevConfigWidth{1}, ::Const{typeof(split_sret)},
        RT::Type{<:Duplicated}, v)
    res = split_sret(v.val)
    primal = EnzymeRules.needs_primal(config) ? res : nothing
    return EnzymeRules.augmented_rule_return_type(config, RT)(primal, Enzyme.make_zero(res), nothing)
end
function EnzymeRules.reverse(config::RevConfigWidth{1}, ::Const{typeof(split_sret)},
        ::Type{<:Duplicated}, tape, v)
    v.dval .+= fill(2.0, length(v.val))
    return (nothing,)
end

# The primal body never reads `p`, but lowering the rule does: even with the
# body's attributes kept (as for an easy rule, here declared as DiffEqBase does
# for solve_up), `p` and its roots array must not be inferred readnone, which
# would let the caller drop the stores that lowering reads.
@noinline split_unread(p::SplitP, w::Vector{Float64}) = 2 .* w
EnzymeRules.has_easy_rule(::typeof(split_unread), p, w) = nothing

function EnzymeRules.augmented_primal(config::RevConfigWidth{1}, ::Const{typeof(split_unread)},
        RT::Type{<:Duplicated}, p, w)
    res = split_unread(p.val, w.val)
    s = sum(p.val.v)
    return EnzymeRules.augmented_rule_return_type(config, RT, s)(res, zero(res), s)
end
function EnzymeRules.reverse(config::RevConfigWidth{1}, ::Const{typeof(split_unread)},
        ::Type{<:Duplicated}, tape::Float64, p, w)
    w.dval .+= tape .+ sum(p.val.v)
    return (nothing, nothing)
end

# The MixedDuplicated argument's shadow is built in the forward pass from memory
# that only exists in the reverse pass.
@noinline split_mixed(p::SplitP, w::Vector{Float64}) = w .* p.x

function EnzymeRules.augmented_primal(config::RevConfigWidth{1}, ::Const{typeof(split_mixed)},
        RT::Type{<:Duplicated}, p, w)
    res = split_mixed(p.val, w.val)
    return EnzymeRules.augmented_rule_return_type(config, RT)(res, zero(res), nothing)
end
function EnzymeRules.reverse(config::RevConfigWidth{1}, ::Const{typeof(split_mixed)},
        ::Type{<:Duplicated}, tape, p, w)
    w.dval .+= 3.0
    return (nothing, nothing)
end

split_dyn(f, x) = Base.inferencebarrier(f)(x)

@testset "Custom rules in split mode" begin
    RA = set_runtime_activity(Reverse)

    @testset "tape type ($mode)" for mode in (Reverse, RA)
        loss(v) = sum(split_dyn(s -> split_tape(s), SplitP(2.0, v)))
        @test Enzyme.gradient(mode, loss, [1.0, 2.0])[1] ≈ [2.0, 2.0]
    end

    @testset "struct return" begin
        loss(v) = sum(split_dyn(w -> split_sret(w), v).u)
        @test Enzyme.gradient(Reverse, loss, [1.0, 2.0])[1] ≈ [2.0, 2.0]
    end

    @testset "argument unread by the primal" begin
        loss(w) = sum(split_dyn(q -> split_unread(SplitP(q[1], [3.0]), q), w))
        @test Enzyme.gradient(RA, loss, [1.0, 2.0])[1] ≈ [6.0, 6.0]
    end

    @testset "mixed argument shadow" begin
        loss(w) = sum(split_dyn(q -> split_mixed(SplitP(q[1], [3.0]), q), w))
        @test Enzyme.gradient(RA, loss, [1.0, 2.0])[1] ≈ [3.0, 3.0]
    end
end
