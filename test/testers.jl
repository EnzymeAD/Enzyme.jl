using Enzyme
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
function _make_jvp_call(fdm, f, dret, y, xs, ẋs, ignores)
    dret <: Const && return ()
    f2 = _wrap_function(f, xs, ignores)
    ignores = collect(ignores)
    if all(ignores)
        y isa Tuple && return map(_ -> nothing, y)
        return (nothing,)
    end
    sigargs = zip(xs[.!ignores], ẋs[.!ignores])
    return FiniteDifferences.jvp(fdm, f2, sigargs...)
end

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

# TODO: handle more cases
rand_tangent(x) = rand_tangent(Random.default_rng(), x)
function rand_tangent(rng, x::AbstractArray)
    ẋ = deepcopy(x)  # preserve output type
    ẋ .= rand_tangent.(rng, x)
    return ẋ
end
# make numbers prettier sometimes when errors are printed.
rand_tangent(rng, ::T) where {T<:AbstractFloat} = rand(rng, -9:T(0.01):9)
function rand_tangent(rng, x::Complex)
    return complex(rand_tangent(rng, real(x)), rand_tangent(rng, imag(x)))
end
rand_tangent(rng, x::Tuple) = map(xi -> rand_tangent(rng, xi), x)
rand_tangent(rng, x::NamedTuple) = map(xi -> rand_tangent(rng, xi), x)

function test_approx(x::Tuple, y::Tuple; kwargs...)
    return all(zip(x, y)) do (xi, yi)
        test_approx(xi, yi; kwargs...)
    end
end
function test_approx(x::NamedTuple, y::NamedTuple; kwargs...)
    return all(zip(x, y)) do (xi, yi)
        test_approx(xi, yi; kwargs...)
    end
end
test_approx(x, y; kwargs...) = isapprox(x, y; kwargs...)

function auto_forward_activity(arg::Tuple)
    primal, T = arg
    T <: Const && return T(primal)
    T <: Duplicated && return T(primal, rand_tangent(primal))
    if T <: BatchDuplicated
        tangents = ntuple(_ -> rand_tangent(primal), 2)
        return T(primal, tangents)
    end
    throw(ArgumentError("Unsupported activity type: $T"))
end
auto_forward_activity(activity) = activity

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
        tangents = map(a -> a isa Const ? nothing : a.dval, activities)
        ignores = map(a -> a isa Const, activities)
        y = call_on_copy(primals...)
        # TODO: handle batch activities
        dy_fdm = _make_jvp_call(
            fdm, call_on_copy, ret_activity, y, primals, tangents, ignores
        )
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
                @test test_approx(dy_ad_i, dy_fdm_i; atol, rtol)
            end
        end
    end
end
