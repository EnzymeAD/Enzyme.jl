# _make_jvp_call and _wrap_function adapted from ChainRulesTestUtils
# https://github.com/JuliaDiff/ChainRulesTestUtils.jl/blob/f76f3fc7be221e07ba9be28ef33a22238ef13661/src/finite_difference_calls.jl
# Copyright (c) 2020 JuliaDiff

#=
    _fd_forward(fdm, f, rettype, y, activities)

Call `FiniteDifferences.jvp` on `f` with the arguments `xs` determined by `activities`.

# Arguments
- `fdm::FiniteDifferenceMethod`: How to numerically differentiate `f`.
- `f`: The function to differentiate.
- `rettype`: Return activity type
- `y`: The primal output `y=f(xs...)` or at least something of the right type.
- `activities`: activities that would be passed to `Enzyme.autodiff`

# Returns
- `ẏ`: Derivative of output w.r.t. `t` estimated by finite differencing. If `rettype` is a
    batch return type, then `ẏ` is a `NamedTuple` of derivatives.
=#
function _fd_forward(fdm, f, rettype, y, activities)
    xs = map(x -> x.val, activities)
    ẋs = map(a -> a isa Const ? nothing : a.dval, activities)
    ignores = map(a -> a isa Const, activities)
    f2 = _wrap_forward_function(f, xs, ignores)
    ignores = collect(ignores)
    if rettype <: Union{Duplicated,DuplicatedNoNeed}
        all(ignores) && return zero_tangent(y)
        sigargs = zip(xs[.!ignores], ẋs[.!ignores])
        return FiniteDifferences.jvp(fdm, f2, sigargs...)
    elseif rettype <: Union{BatchDuplicated,BatchDuplicatedNoNeed}
        all(ignores) && return (var"1"=zero_tangent(y),)
        sig_arg_vals = xs[.!ignores]
        ret_dvals = map(ẋs[.!ignores]...) do sig_args_dvals...
            FiniteDifferences.jvp(fdm, f2, zip(sig_arg_vals, sig_args_dvals)...)
        end
        return NamedTuple{ntuple(Symbol, length(ret_dvals))}(ret_dvals)
    else
        throw(ArgumentError("Unsupported return type: $rettype"))
    end
end
_fd_forward(fdm, f, ::Type{<:Const}, y, activities) = ()

#=
    _fd_reverse(fdm, f, ȳ, activities)

Call `FiniteDifferences.j′vp` on `f` with the arguments `xs` determined by `activities`.

# Arguments
- `fdm::FiniteDifferenceMethod`: How to numerically differentiate `f`.
- `f`: The function to differentiate.
- `ȳ`: The cotangent of the primal output `y=f(xs...)`.
- `activities`: activities that would be passed to `Enzyme.autodiff`

# Returns
- `x̄s`: Derivatives of output `s` w.r.t. `xs` estimated by finite differencing.
=#
function _fd_reverse(fdm, f, ȳ, activities)
    xs = map(x -> x.val, activities)
    ignores = map(a -> a isa Const, activities)
    f2 = _wrap_reverse_function(f, xs, ignores)
    all(ignores) && return map(zero_tangent, xs)
    ignores = collect(ignores)
    x̄s = map(collect(activities)) do a
        a isa Union{Const,Active} && return zero_tangent(a.val)
        return a.dval
    end
    sigargs = xs[.!ignores]
    s̄igargs = x̄s[.!ignores]
    sigarginds = eachindex(x̄s)[.!ignores]
    fd = FiniteDifferences.j′vp(fdm, f2, (ȳ, s̄igargs...), sigargs...)
    @assert length(fd) == length(sigarginds)
    x̄s[sigarginds] = collect(fd)
    return (x̄s...,)
end

#=
    _wrap_forward_function(f, xs, ignores)

Return a new version of `f`, `fnew`, that ignores some of the arguments `xs`.

# Arguments
- `f`: The function to be wrapped.
- `xs`: Inputs to `f`, such that `y = f(xs...)`.
- `ignores`: Collection of `Bool`s, the same length as `xs`.
  If `ignores[i] === true`, then `xs[i]` is ignored; `∂xs[i] === NoTangent()`.
=#
function _wrap_forward_function(f, xs, ignores)
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

#=
    _wrap_reverse_function(f, xs, ignores)

Return a new version of `f`, `fnew`, that ignores some of the arguments `xs` and returns
also non-ignored arguments.

All arguments are copied before being passed to `f`, so that `fnew` is non-mutating.

# Arguments
- `f`: The function to be wrapped.
- `xs`: Inputs to `f`, such that `y = f(xs...)`.
- `ignores`: Collection of `Bool`s, the same length as `xs`.
  If `ignores[i] === true`, then `xs[i]` is ignored; `∂xs[i] === NoTangent()`.
=#
function _wrap_reverse_function(f, xs, ignores)
    function fnew(sigargs...)
        callargs = Any[]
        retargs = Any[]
        j = 1

        for (i, (x, ignore)) in enumerate(zip(xs, ignores))
            if ignore
                push!(callargs, deepcopy(x))
            else
                arg = deepcopy(sigargs[j])
                push!(callargs, arg)
                push!(retargs, arg)
                j += 1
            end
        end
        @assert j == length(sigargs) + 1
        @assert length(callargs) == length(xs)
        @assert length(retargs) == count(!, ignores)
        return (deepcopy(f)(callargs...), retargs...)
    end
    return fnew
end
