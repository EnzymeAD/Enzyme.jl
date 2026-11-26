using Enzyme
using Enzyme.EnzymeRules
using Test

# Primal function
function my_func_union(x)
    return x * x
end

# Custom augmented primal returning a Union of AugmentedReturn types
@inline function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, ::Const{typeof(my_func_union)}, ::Type{<:Active}, x
    )
    primal = EnzymeRules.needs_primal(config) ? my_func_union(x.val) : nothing
    if x.val > 0.0
        return EnzymeRules.AugmentedReturn(primal, nothing, 1)
    else
        return EnzymeRules.AugmentedReturn(primal, nothing, 2.0)
    end
end

# Custom reverse rule consuming the union tape
@inline function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, ::Const{typeof(my_func_union)}, dret::Active, tape, x
    )
    # tape is a Union{Int64, Float64}
    dx = if tape isa Int64
        2.0 * x.val * tape
    else
        2.0 * x.val * tape
    end
    return (dx,)
end

@testset "Union sret return rule" begin
    grads = Enzyme.gradient(Reverse, my_func_union, 1.0)
    @test grads[1] ≈ 2.0
end

# Since Julia 1.14 (julia#55045) a union return also holds members with GC
# pointers unboxed, with the pointers in a return-roots argument.

# The tape is a `Vector` (a pointer in the member) or a `Float64`.
@noinline union_tape_vec(x) = x * x
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, ::Const{typeof(union_tape_vec)}, ::Type{<:Active}, x::Active
    )
    primal = EnzymeRules.needs_primal(config) ? union_tape_vec(x.val) : nothing
    if x.val > 0.0
        return EnzymeRules.AugmentedReturn(primal, nothing, [2 * x.val])
    else
        return EnzymeRules.AugmentedReturn(primal, nothing, 2 * x.val)
    end
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, ::Const{typeof(union_tape_vec)}, dret::Active, tape, x::Active
    )
    return ((tape isa Vector ? tape[1] : tape) * dret.val,)
end

# The tape is an inline tuple with a pointer, or `nothing`.
@noinline union_tape_tup(x) = x * x
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, ::Const{typeof(union_tape_tup)}, ::Type{<:Active}, x::Active
    )
    primal = EnzymeRules.needs_primal(config) ? union_tape_tup(x.val) : nothing
    if x.val > 0.0
        return EnzymeRules.AugmentedReturn(primal, nothing, (x.val, [2.0]))
    else
        return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
    end
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, ::Const{typeof(union_tape_tup)}, dret::Active, tape, x::Active
    )
    return (tape === nothing ? 0.0 : tape[1] * tape[2][1] * dret.val,)
end

# The primal holds a pointer and the tape is a union.
@noinline union_tape_prim(x, v) = (x * x, v)
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, ::Const{typeof(union_tape_prim)}, ::Type, x::Active, v::Const
    )
    primal = EnzymeRules.needs_primal(config) ? union_tape_prim(x.val, v.val) : nothing
    shadow = EnzymeRules.needs_shadow(config) ? (0.0, zero(v.val)) : nothing
    if x.val > 0.0
        return EnzymeRules.AugmentedReturn(primal, shadow, 1)
    else
        return EnzymeRules.AugmentedReturn(primal, shadow, 2.0)
    end
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, ::Const{typeof(union_tape_prim)}, dret, tape, x::Active, v::Const
    )
    return (2 * x.val * tape, nothing)
end
union_tape_prim_caller(x, v) = (r = union_tape_prim(x, v); r[1] * r[2][1])

# Before Julia 1.14 these unions are boxed, and the reverse pass fails with
# "Illegal calling convention fixup".
@static if VERSION >= v"1.14.0-DEV.165"
    @testset "Union sret return rule with GC pointers in the tape" begin
        @test autodiff(Reverse, union_tape_vec, Active, Active(3.0))[1][1] ≈ 6.0
        @test autodiff(Reverse, union_tape_vec, Active, Active(-3.0))[1][1] ≈ -6.0
        res = autodiff(ReverseWithPrimal, union_tape_vec, Active, Active(3.0))
        @test res[1][1] ≈ 6.0
        @test res[2] ≈ 9.0
        @test autodiff(Reverse, union_tape_tup, Active, Active(3.0))[1][1] ≈ 6.0
        @test autodiff(Reverse, union_tape_tup, Active, Active(-3.0))[1][1] ≈ 0.0
        @test autodiff(Reverse, union_tape_prim_caller, Active, Active(3.0), Const([1.0]))[1][1] ≈ 6.0
    end
end

# The primal returns a union with a member that holds a pointer.
const UnionWithRoots = Union{Nothing, Tuple{Float64, Vector{Float64}}}
@noinline union_roots_primal(x, v, b) = b ? (x, v) : nothing
function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig, ::Const{typeof(union_roots_primal)}, ::Type, x::Duplicated, v::Duplicated, b::Const
    )
    primal::UnionWithRoots = union_roots_primal(x.val, v.val, b.val)
    shadow::UnionWithRoots = b.val ? (x.dval, v.dval) : nothing
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated{UnionWithRoots}(primal, shadow)
    elseif EnzymeRules.needs_shadow(config)
        return shadow
    elseif EnzymeRules.needs_primal(config)
        return primal
    else
        return nothing
    end
end
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, ::Const{typeof(union_roots_primal)}, ::Type, x::Active, v::Duplicated, b::Const
    )
    primal::UnionWithRoots = union_roots_primal(x.val, v.val, b.val)
    shadow::UnionWithRoots = b.val ? (0.0, v.dval) : nothing
    P = EnzymeRules.needs_primal(config) ? UnionWithRoots : Nothing
    S = EnzymeRules.needs_shadow(config) ? UnionWithRoots : Nothing
    return EnzymeRules.AugmentedReturn{P, S, Nothing}(
        EnzymeRules.needs_primal(config) ? primal : nothing,
        EnzymeRules.needs_shadow(config) ? shadow : nothing,
        nothing,
    )
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, ::Const{typeof(union_roots_primal)}, dret, tape, x::Active, v::Duplicated, b::Const
    )
    return (7.0, nothing, nothing)
end
union_roots_caller(x, v, b) = (r = union_roots_primal(x, v, b); r === nothing ? 0.0 : r[1] * sum(r[2]))
union_roots_direct(x, v, b) = union_roots_primal(x, v, b)

@testset "Custom rule for a union return with GC pointers" begin
    v = [1.0, 2.0]
    @test autodiff(Forward, union_roots_caller, Duplicated(3.0, 1.0), Duplicated(v, ones(2)), Const(true))[1] ≈ 9.0
    res = autodiff(Forward, union_roots_direct, Duplicated(3.0, 1.0), Duplicated(v, ones(2)), Const(true))[1]
    @test res[1] ≈ 1.0
    @test res[2] == [1.0, 1.0]
    dv = zeros(2)
    res = autodiff(Reverse, union_roots_caller, Active, Active(3.0), Duplicated(v, dv), Const(true))
    @test res[1][1] ≈ 7.0
    @test dv ≈ [3.0, 3.0]
end

# Batched forward rules for a union return with GC pointers. The shadows are an
# `NTuple` or a `BatchDuplicated`, which hold each union boxed.
@noinline union_roots_batch(x, v, b) = b ? (x, v) : nothing
function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig, ::Const{typeof(union_roots_batch)}, ::Type,
        x::Union{Duplicated, BatchDuplicated}, v::Union{Duplicated, BatchDuplicated}, b::Const
    )
    primal::UnionWithRoots = union_roots_batch(x.val, v.val, b.val)
    if x isa Duplicated
        shadow::UnionWithRoots = b.val ? (x.dval, v.dval) : nothing
        needs_both = EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        needs_both && return Duplicated{UnionWithRoots}(primal, shadow)
        EnzymeRules.needs_shadow(config) && return shadow
    else
        N = EnzymeRules.width(config)
        shadows = ntuple(i -> (b.val ? (x.dval[i], v.dval[i]) : nothing)::UnionWithRoots, Val(N))
        needs_both = EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        needs_both && return BatchDuplicated{UnionWithRoots, N}(primal, shadows)
        EnzymeRules.needs_shadow(config) && return shadows
    end
    return EnzymeRules.needs_primal(config) ? primal : nothing
end
union_roots_batch_caller(x, v, b) = (r = union_roots_batch(x, v, b); r === nothing ? 0.0 : r[1] * sum(r[2]))
union_roots_batch_direct(x, v, b) = union_roots_batch(x, v, b)

# The union return is inactive (its members hold only integers), so the rule
# only returns the primal, in place of the original split union.
const ConstUnionWithRoots = Union{Nothing, Tuple{Int, Vector{Int}}}
@noinline union_const_primal(x, n) = x > 0 ? (n, [n]) : nothing
function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig, ::Const{typeof(union_const_primal)}, ::Type{<:Const},
        x::Union{Duplicated, BatchDuplicated}, n::Const
    )
    primal::ConstUnionWithRoots = union_const_primal(x.val, n.val)
    return EnzymeRules.needs_primal(config) ? primal : nothing
end
union_const_caller(x, n) = (r = union_const_primal(x, n); r === nothing ? 0.0 : x * r[1] * r[2][1])

@testset "Batched forward rule for a union return with GC pointers" begin
    v = [1.0, 2.0]
    @test autodiff(Forward, union_roots_batch_caller, Duplicated(3.0, 1.0), Duplicated(v, ones(2)), Const(true))[1] ≈ 9.0
    res = autodiff(
        Forward, union_roots_batch_caller, BatchDuplicated(3.0, (1.0, 2.0)),
        BatchDuplicated(v, (ones(2), 2 .* ones(2))), Const(true)
    )[1]
    @test res[1] ≈ 9.0
    @test res[2] ≈ 18.0
    @test autodiff(
        Forward, union_roots_batch_caller, BatchDuplicated(3.0, (1.0, 2.0)),
        BatchDuplicated(v, (ones(2), 2 .* ones(2))), Const(false)
    )[1] |> values == (0.0, 0.0)

    dv = (ones(2), 2 .* ones(2))
    res = autodiff(Forward, union_roots_batch_direct, BatchDuplicated(3.0, (1.0, 2.0)), BatchDuplicated(v, dv), Const(true))[1]
    @test res[1][1] ≈ 1.0
    @test res[1][2] === dv[1]
    @test res[2][1] ≈ 2.0
    @test res[2][2] === dv[2]

    res = autodiff(ForwardWithPrimal, union_roots_batch_direct, BatchDuplicated(3.0, (1.0, 2.0)), BatchDuplicated(v, dv), Const(true))
    @test res[1][1][1] ≈ 1.0
    @test res[1][2][1] ≈ 2.0
    @test res[2][1] ≈ 3.0
    @test res[2][2] === v

    @test autodiff(Forward, union_const_caller, Duplicated(3.0, 1.0), Const(2))[1] ≈ 4.0
    res = autodiff(Forward, union_const_caller, BatchDuplicated(3.0, (1.0, 2.0)), Const(2))[1]
    @test res[1] ≈ 4.0
    @test res[2] ≈ 8.0
    @test autodiff(Forward, union_const_caller, BatchDuplicated(-3.0, (1.0, 2.0)), Const(2))[1] |> values == (0.0, 0.0)
end

# The tape is an isbits union, which a member stores inline with a selector byte.
# The rules scale by 100, so the tests can tell that they ran.
@noinline union_tape_bits(x) = x * x
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, ::Const{typeof(union_tape_bits)}, ::Type{<:Active}, x::Active
    )
    primal = EnzymeRules.needs_primal(config) ? union_tape_bits(x.val) : nothing
    P = typeof(primal)
    if x.val > 0.0
        tape::Union{Nothing, Float64} = x.val > 1.0 ? 2 * x.val : nothing
        return EnzymeRules.AugmentedReturn{P, Nothing, Union{Nothing, Float64}}(primal, nothing, tape)
    else
        return EnzymeRules.AugmentedReturn(primal, nothing, Int(round(2 * x.val)))
    end
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, ::Const{typeof(union_tape_bits)}, dret::Active, tape, x::Active
    )
    return (100 * (tape === nothing ? -1.0 : Float64(tape)) * dret.val,)
end

@static if VERSION >= v"1.14.0-DEV.165"
    @testset "Union sret return rule with an isbits union tape" begin
        @test autodiff(Reverse, union_tape_bits, Active, Active(3.0))[1][1] ≈ 600.0
        @test autodiff(Reverse, union_tape_bits, Active, Active(0.5))[1][1] ≈ -100.0
        @test autodiff(Reverse, union_tape_bits, Active, Active(-3.0))[1][1] ≈ -600.0
    end
end

# Forward rules for an isbits union return. A rule result stores the union inline,
# and a batched rule holds the shadows boxed in an `NTuple`.
for (fname, FT) in ((:union_bits_f64, Float64), (:union_bits_f16, Float16))
    @eval begin
        @noinline $fname(x, b) = b ? x * x : nothing
        function EnzymeRules.forward(
                config::EnzymeRules.FwdConfig, ::Const{typeof($fname)}, ::Type,
                x::Union{Duplicated, BatchDuplicated}, b::Const
            )
            U = Union{Nothing, $FT}
            primal::U = $fname(x.val, b.val)
            if x isa Duplicated
                shadow::U = b.val ? $FT(100) * x.dval : nothing
                needs_both = EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
                needs_both && return Duplicated{U}(primal, shadow)
                EnzymeRules.needs_shadow(config) && return shadow
            else
                N = EnzymeRules.width(config)
                shadows = ntuple(i -> (b.val ? $FT(100) * x.dval[i] : nothing)::U, Val(N))
                needs_both = EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
                needs_both && return BatchDuplicated{U, N}(primal, shadows)
                EnzymeRules.needs_shadow(config) && return shadows
            end
            return EnzymeRules.needs_primal(config) ? primal : nothing
        end
    end
end
union_bits_caller(f::F, x, b) where {F} = (r = f(x, b); r === nothing ? zero(x) : 3 * r)
union_bits_f64_caller(x, b) = union_bits_caller(union_bits_f64, x, b)
union_bits_f16_caller(x, b) = union_bits_caller(union_bits_f16, x, b)
union_bits_f64_direct(x, b) = union_bits_f64(x, b)

@testset "Forward rule for an isbits union return" begin
    @test autodiff(Forward, union_bits_f64_caller, Duplicated(3.0, 1.0), Const(true))[1] ≈ 300.0
    @test autodiff(Forward, union_bits_f64_caller, Duplicated(3.0, 1.0), Const(false))[1] ≈ 0.0
    @test autodiff(Forward, union_bits_f64_direct, Duplicated(3.0, 1.0), Const(true))[1] ≈ 100.0
    @test autodiff(Forward, union_bits_f16_caller, Duplicated(Float16(3), Float16(1)), Const(true))[1] ≈ Float16(300)

    res = autodiff(Forward, union_bits_f64_caller, BatchDuplicated(3.0, (1.0, 2.0)), Const(true))[1]
    @test values(res) == (300.0, 600.0)
    @test values(autodiff(Forward, union_bits_f64_caller, BatchDuplicated(3.0, (1.0, 2.0)), Const(false))[1]) == (0.0, 0.0)
    res = autodiff(Forward, union_bits_f64_direct, BatchDuplicated(3.0, (1.0, 2.0)), Const(true))[1]
    @test values(res) == (100.0, 200.0)
    res = autodiff(ForwardWithPrimal, union_bits_f64_direct, BatchDuplicated(3.0, (1.0, 2.0)), Const(true))
    @test values(res[1]) == (100.0, 200.0)
    @test res[2] ≈ 9.0
    res = autodiff(Forward, union_bits_f16_caller, BatchDuplicated(Float16(3), (Float16(1), Float16(2))), Const(true))[1]
    @test values(res) == (Float16(300), Float16(600))
end

# The data of the union return is active, but its GC pointers (a `Vector{Int}`)
# are not: since Julia 1.14 the union buffer and the return roots differ in activity.
const UnionInactiveRoots = Union{Nothing, Tuple{Float64, Vector{Int}}}
@noinline union_inactive_roots(x, n, b) = b ? (x, [n]) : nothing
function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig, ::Const{typeof(union_inactive_roots)}, ::Type,
        x::Union{Duplicated, BatchDuplicated}, n::Const, b::Const
    )
    primal::UnionInactiveRoots = union_inactive_roots(x.val, n.val, b.val)
    if x isa Duplicated
        shadow::UnionInactiveRoots = b.val ? (100 * x.dval, [0]) : nothing
        needs_both = EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        needs_both && return Duplicated{UnionInactiveRoots}(primal, shadow)
        EnzymeRules.needs_shadow(config) && return shadow
    else
        N = EnzymeRules.width(config)
        shadows = ntuple(i -> (b.val ? (100 * x.dval[i], [0]) : nothing)::UnionInactiveRoots, Val(N))
        needs_both = EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        needs_both && return BatchDuplicated{UnionInactiveRoots, N}(primal, shadows)
        EnzymeRules.needs_shadow(config) && return shadows
    end
    return EnzymeRules.needs_primal(config) ? primal : nothing
end
union_inactive_roots_caller(x, n, b) = (r = union_inactive_roots(x, n, b); r === nothing ? 0.0 : r[1] * r[2][1])

@testset "Forward rule for a union return with inactive GC pointers" begin
    @test autodiff(Forward, union_inactive_roots_caller, Duplicated(3.0, 1.0), Const(2), Const(true))[1] ≈ 200.0
    res = autodiff(Forward, union_inactive_roots_caller, BatchDuplicated(3.0, (1.0, 2.0)), Const(2), Const(true))[1]
    @test values(res) == (200.0, 400.0)
end

# The union buffer holds only an integer. The return roots hold the active data.
const UnionWithConstBuffer = Union{Nothing, Tuple{Int, Vector{Float64}}}
@noinline union_const_buffer(n, v, b) = b ? (n, v) : nothing
function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig, ::Const{typeof(union_const_buffer)}, ::Type,
        n::Const, v::Union{Duplicated, BatchDuplicated}, b::Const
    )
    primal::UnionWithConstBuffer = union_const_buffer(n.val, v.val, b.val)
    needs_both = EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
    if v isa Duplicated
        shadow::UnionWithConstBuffer = b.val ? (n.val, v.dval) : nothing
        needs_both && return Duplicated{UnionWithConstBuffer}(primal, shadow)
        EnzymeRules.needs_shadow(config) && return shadow
    else
        N = EnzymeRules.width(config)
        shadows = ntuple(i -> (b.val ? (n.val, v.dval[i]) : nothing)::UnionWithConstBuffer, Val(N))
        needs_both && return BatchDuplicated{UnionWithConstBuffer, N}(primal, shadows)
        EnzymeRules.needs_shadow(config) && return shadows
    end
    return EnzymeRules.needs_primal(config) ? primal : nothing
end
union_const_buffer_caller(n, v, b) = (r = union_const_buffer(n, v, b); r === nothing ? 0.0 : r[1] * sum(r[2]))
union_const_buffer_direct(n, v, b) = union_const_buffer(n, v, b)

@testset "Forward rule for a union return with an inactive buffer" begin
    v = [1.0, 2.0]
    @test autodiff(Forward, union_const_buffer_caller, Const(5), Duplicated(v, ones(2)), Const(true))[1] ≈ 10.0
    res = autodiff(Forward, union_const_buffer_caller, Const(5), BatchDuplicated(v, (ones(2), 2 .* ones(2))), Const(true))[1]
    @test res[1] ≈ 10.0
    @test res[2] ≈ 20.0
    @test autodiff(Forward, union_const_buffer_caller, Const(5), BatchDuplicated(v, (ones(2), 2 .* ones(2))), Const(false))[1] |> values == (0.0, 0.0)

    # Only the shadow is needed, so the rule returns the split union itself.
    dv = ones(2)
    res = autodiff(Forward, union_const_buffer_direct, Const(5), Duplicated(v, dv), Const(true))[1]
    @test res[2] === dv
    @test autodiff(Forward, union_const_buffer_direct, Const(5), Duplicated(v, dv), Const(false))[1] === nothing
end
