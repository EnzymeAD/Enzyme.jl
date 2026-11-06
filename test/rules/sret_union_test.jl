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
        2.0 * x.val * tape * dret.val
    else
        2.0 * x.val * tape * dret.val
    end
    return (dx,)
end

# Mixed pointer-free (sret) and pointer-bearing (boxed) union arms
function mixed_sret_union_op!(y::Vector{Float64}, x::Vector{Float64}, save_x::Bool)
    @inbounds for i in eachindex(y, x)
        y[i] = x[i]^2
    end
    return nothing
end

function EnzymeRules.augmented_primal(
        ::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(mixed_sret_union_op!)},
        ::Type{Const{Nothing}},
        y::Annotation{Vector{Float64}},
        x::Annotation{Vector{Float64}},
        save_x::Const{Bool},
    )
    mixed_sret_union_op!(y.val, x.val, save_x.val)
    cache = save_x.val ? (2, copy(x.val)) : (1, nothing)
    return EnzymeRules.AugmentedReturn(nothing, nothing, cache)
end

function EnzymeRules.reverse(
        ::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(mixed_sret_union_op!)},
        ::Type{Const{Nothing}},
        cache,
        y::Annotation{Vector{Float64}},
        x::Annotation{Vector{Float64}},
        ::Const{Bool},
    )
    x_used = if cache[1] == 2
        cache[2]::Vector{Float64}
    elseif cache[1] == 1 && cache[2] === nothing
        x.val
    else
        error("unexpected tape: $(repr(cache))")
    end
    if !(y isa Const) && !(x isa Const)
        @inbounds for i in eachindex(y.dval, x.dval, x_used)
            x.dval[i] += 2.0 * x_used[i] * y.dval[i]
            y.dval[i] = 0.0
        end
    end
    return (nothing, nothing, nothing)
end

# Mixed singleton/unboxed and boxed tape with non-void primal return (needs_primal = true)
function mixed_sret_union_primal(x::Vector{Float64}, save_x::Bool)
    return sum(x)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(mixed_sret_union_primal)},
        ::Type{<:Active},
        x::Annotation{Vector{Float64}},
        save_x::Const{Bool},
    )
    primal = EnzymeRules.needs_primal(config) ? mixed_sret_union_primal(x.val, save_x.val) : nothing
    cache = save_x.val ? copy(x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, cache)
end

function EnzymeRules.reverse(
        ::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(mixed_sret_union_primal)},
        dret::Active,
        cache,
        x::Annotation{Vector{Float64}},
        save_x::Const{Bool},
    )
    if save_x.val
        @assert cache isa Vector{Float64}
        if !(x isa Const)
            @inbounds for i in eachindex(x.dval, cache)
                x.dval[i] += dret.val * (cache[i] / cache[i])
            end
        end
    else
        @assert cache === nothing
        if !(x isa Const)
            @inbounds for i in eachindex(x.dval)
                x.dval[i] += dret.val
            end
        end
    end
    return (nothing, nothing)
end

# sret_union whose extracted TapeT collapses to a single concrete scalar or struct type
const CollapsedScalarAugRT = Union{
    EnzymeRules.AugmentedReturn{Nothing, Nothing, Int64},
    EnzymeRules.AugmentedReturn{Nothing, Nothing, T} where {T <: Int64},
}
const CollapsedStructAugRT = Union{
    EnzymeRules.AugmentedReturn{Nothing, Nothing, Tuple{Int64, Nothing}},
    EnzymeRules.AugmentedReturn{Nothing, Nothing, T} where {T <: Tuple{Int64, Nothing}},
}

@noinline _make_collapsed_scalar_aug()::CollapsedScalarAugRT =
    EnzymeRules.AugmentedReturn(nothing, nothing, 3)
@noinline _make_collapsed_struct_aug()::CollapsedStructAugRT =
    EnzymeRules.AugmentedReturn(nothing, nothing, (5, nothing))

function collapsed_scalar_tape_fn(r::Ref{Float64}, x::Float64)
    r[] = x * x
    return nothing
end

function EnzymeRules.augmented_primal(
        ::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(collapsed_scalar_tape_fn)},
        ::Type{Const{Nothing}},
        r::Duplicated{Base.RefValue{Float64}},
        x::Active{Float64},
    )
    collapsed_scalar_tape_fn(r.val, x.val)
    return _make_collapsed_scalar_aug()
end

function EnzymeRules.reverse(
        ::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(collapsed_scalar_tape_fn)},
        ::Type{Const{Nothing}},
        tape::Int64,
        r::Duplicated{Base.RefValue{Float64}},
        x::Active{Float64},
    )
    dr = r.dval[]
    r.dval[] = 0.0
    return (nothing, tape * 2.0 * x.val * dr)
end

function collapsed_struct_tape_fn(r::Ref{Float64}, x::Float64)
    r[] = x * x
    return nothing
end

function EnzymeRules.augmented_primal(
        ::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(collapsed_struct_tape_fn)},
        ::Type{Const{Nothing}},
        r::Duplicated{Base.RefValue{Float64}},
        x::Active{Float64},
    )
    collapsed_struct_tape_fn(r.val, x.val)
    return _make_collapsed_struct_aug()
end

function EnzymeRules.reverse(
        ::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(collapsed_struct_tape_fn)},
        ::Type{Const{Nothing}},
        tape::Tuple{Int64, Nothing},
        r::Duplicated{Base.RefValue{Float64}},
        x::Active{Float64},
    )
    dr = r.dval[]
    r.dval[] = 0.0
    return (nothing, tape[1] * 2.0 * x.val * dr)
end

@testset "Union sret return rule" begin
    @test Enzyme.gradient(Reverse, my_func_union, 1.0)[1] ≈ 2.0
    @test Enzyme.gradient(Reverse, my_func_union, -1.0)[1] ≈ -4.0

    @testset "Mixed isbits and boxed sret union tape" begin
        for save_x in (false, true)
            x = [2.0, 3.0]
            dx = zeros(2)
            Enzyme.autodiff(
                Reverse,
                (x_in, flag) -> begin
                    y = similar(x_in)
                    mixed_sret_union_op!(y, x_in, flag)
                    if flag
                        fill!(x_in, -99.0)
                    end
                    return sum(y)
                end,
                Active,
                Duplicated(copy(x), dx),
                Const(save_x),
            )
            @test dx ≈ [4.0, 6.0]
        end
    end

    @testset "Mixed sret union with needs_primal = true" begin
        for save_x in (false, true)
            x = [1.5, 2.5]
            dx = zeros(2)
            (_, primal) = Enzyme.autodiff(
                ReverseWithPrimal,
                (x_in, flag) -> begin
                    s = mixed_sret_union_primal(x_in, flag)
                    return s * s
                end,
                Active,
                Duplicated(copy(x), dx),
                Const(save_x),
            )
            @test primal ≈ 16.0
            @test dx ≈ [8.0, 8.0]
        end
    end

    @testset "Collapsed concrete TapeT under sret union" begin
        g_scalar = Enzyme.gradient(
            Reverse,
            x -> begin
                r = Ref(0.0)
                collapsed_scalar_tape_fn(r, x)
                return r[]
            end,
            2.5,
        )
        @test g_scalar[1] ≈ 15.0

        g_struct = Enzyme.gradient(
            Reverse,
            x -> begin
                r = Ref(0.0)
                collapsed_struct_tape_fn(r, x)
                return r[]
            end,
            2.5,
        )
        @test g_struct[1] ≈ 25.0
    end
end
