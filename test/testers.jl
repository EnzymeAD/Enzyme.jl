using Enzyme
using EnzymeCore: Annotation
using FiniteDifferences
using Random
using Test

# _make_jvp_call and _wrap_function adapted from ChainRulesTestUtils
# https://github.com/JuliaDiff/ChainRulesTestUtils.jl/blob/f76f3fc7be221e07ba9be28ef33a22238ef13661/src/finite_difference_calls.jl
# Copyright (c) 2020 JuliaDiff

"""
    _make_jvp_call(fdm, f, dret, y, xs, ẋs, ignores)

Call `FiniteDifferences.jvp`, with the option to ignore certain `xs`.

# Arguments
- `fdm::FiniteDifferenceMethod`: How to numerically differentiate `f`.
- `f`: The function to differentiate.
- `dret`: Return activity type
- `y`: The primal output `y=f(xs...)` or at least something of the right type
- `xs`: Inputs to `f`, such that `y = f(xs...)`.
- `ẋs`: The directional derivatives of `xs` w.r.t. some real number `t`.
- `ignores`: Collection of `Bool`s, the same length as `xs` and `ẋs`.
   If `ignores[i] === true`, then `ẋs[i]` is ignored for derivative estimation.

# Returns
- `Ω̇`: Derivative of output w.r.t. `t` estimated by finite differencing.
"""
function _make_jvp_call(fdm, f, rettype, y, activities)
    xs = map(x -> x.val, activities)
    ẋs = map(a -> a isa Const ? nothing : a.dval, activities)
    ignores = map(a -> a isa Const, activities)
    f2 = _wrap_function(f, xs, ignores)
    ignores = collect(ignores)
    if all(ignores)
        y isa Tuple && return map(_ -> nothing, y)
        return (nothing,)
    end
    if rettype <: Union{Duplicated,DuplicatedNoNeed}
        sigargs = zip(xs[.!ignores], ẋs[.!ignores])
        return FiniteDifferences.jvp(fdm, f2, sigargs...)
    else
        throw(ArgumentError("Unsupported return type: $rettype"))
    end
end
_make_jvp_call(fdm, f, ::Type{<:Const}, y, activities) = ()

"""
    _wrap_function(f, xs, ignores)

Return a new version of `f`, `fnew`, that ignores some of the arguments `xs`.

# Arguments
- `f`: The function to be wrapped.
- `xs`: Inputs to `f`, such that `y = f(xs...)`.
- `ignores`: Collection of `Bool`s, the same length as `xs`.
  If `ignores[i] === true`, then `xs[i]` is ignored; `∂xs[i] === NoTangent()`.
"""
function _wrap_function(f, xs, ignores)
    function fnew(sigargs...)
        callargs = Any[]
        j = 1

        for (i, (x, ignore)) in enumerate(zip(xs, ignores))
            if ignore
                push!(callargs, x)
            else
                push!(callargs, sigargs[j])
                j += 1
            end
        end
        @assert j == length(sigargs) + 1
        @assert length(callargs) == length(xs)
        return f(callargs...)
    end
    return fnew
end

rand_tangent(x) = rand_tangent(Random.default_rng(), x)
# base case: recursively call rand_tangent. Only actually generate random tangents for
# floating point numbers. all other fields are preserved exactly.
function rand_tangent(rng, x::T) where {T}
    fields = fieldnames(T)
    isempty(fields) && return x
    return typeof(x)((rand_tangent(rng, getfield(x, k)) for k in fields)...)
end
rand_tangent(rng, x::Array) = map(xi -> rand_tangent(rng, xi), x)
# make numbers prettier sometimes when errors are printed.
rand_tangent(rng, ::T) where {T<:AbstractFloat} = rand(rng, -9:T(0.01):9)

function zero_tangent(x::T) where {T}
    fields = fieldnames(T)
    isempty(fields) && return x
    return typeof(x)((zero_tangent(getfield(x, k)) for k in fields)...)
end
zero_tangent(x::Array) = map(zero_tangent, x)
# make numbers prettier sometimes when errors are printed.
zero_tangent(::T) where {T<:AbstractFloat} = zero(T)

function test_approx(x::Number, y::Number; kwargs...)
    @test isapprox(x, y; kwargs...)
end
function test_approx(x::AbstractArray{<:Number}, y::AbstractArray{<:Number}; kwargs...)
    @test isapprox(x, y; kwargs...)
end
function test_approx(x, y; kwargs...)
    for k in fieldnames(typeof(x))
        test_approx(getfield(x, k), getfield(y, k); kwargs...)
    end
end

function auto_forward_activity(arg::Tuple)
    if length(arg) == 2 && arg[2] isa Type && arg[2] <: Annotation
        return _build_activity(arg...)
    end
    return Const(arg)
end
auto_forward_activity(activity::Annotation) = activity
auto_forward_activity(activity) = Const(activity)

_build_activity(primal, ::Type{<:Const}) = Const(primal)
_build_activity(primal, ::Type{<:Duplicated}) = Duplicated(primal, rand_tangent(primal))
function _build_activity(primal, ::Type{<:BatchDuplicated})
    return BatchDuplicated(primal, ntuple(_ -> rand_tangent(primal), 2))
end
function _build_activity(primal, T::Type{<:Annotation})
    throw(ArgumentError("Unsupported activity type: $T"))
end

"""
    test_forward(f, return_activity, args...; kwargs...)

# Examples

```julia
x = randn(5)
for Tret in (Const, Duplicated), Tx in (Const, Duplicated)
    test_forward(Const(prod), Tret, (x, Tx))
end
```
"""
function test_forward(
    f,
    ret_activity,
    args...;
    fdm=FiniteDifferences.central_fdm(5, 1),
    fkwargs::NamedTuple=NamedTuple(),
    rtol::Real=1e-9,
    atol::Real=1e-9,
    testset_name=nothing,
)
    call_on_copy(f, xs...) = deepcopy(f)(deepcopy(xs)...; deepcopy(fkwargs)...)
    if testset_name === nothing
        testset_name = "test_forward: $(f isa Const ? f.val : f) with return activity $ret_activity on $(args)"
    end
    @testset "$testset_name" begin
        activities = map(auto_forward_activity, (f, args...))
        primals = map(x -> x.val, activities)
        y = call_on_copy(primals...)
        # TODO: handle batch activities
        dy_fdm = _make_jvp_call(fdm, call_on_copy, ret_activity, y, activities)
        y_and_dy_ad = autodiff(
            Forward, first(activities), ret_activity, Base.tail(activities)...; fkwargs...
        )
        if ret_activity <: Union{Duplicated,BatchDuplicated}
            y_ad, dy_ad = y_and_dy_ad
            @test test_approx(y_ad, y; atol, rtol)
        elseif ret_activity <: Const
            @test isempty(y_and_dy_ad)
            dy_ad = ()
        else
            dy_ad = only(y_and_dy_ad)
        end
        for (dy_ad_i, dy_fdm_i) in zip(dy_ad, dy_fdm)
            if dy_fdm_i === nothing
                @test iszero(dy_ad_i)
            else
                test_approx(dy_ad, dy_fdm; atol, rtol)
            end
        end
    end
end
