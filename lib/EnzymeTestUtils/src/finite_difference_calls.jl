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
    f_sig_args = _wrap_forward_function(f, xs, ignores)
    ignores = collect(ignores)
    _, from_vec_out = to_vec(y)
    sig_arg_val_vec, from_vec_in = to_vec(xs[.!ignores])
    # vectorize inputs and outputs of function
    f_vec = first ∘ to_vec ∘ Base.splat(f_sig_args) ∘ from_vec_in
    if rettype <: Union{Duplicated,DuplicatedNoNeed}
        all(ignores) && return Enzyme.make_zero(y)
        sig_arg_dval_vec, _ = to_vec(ẋs[.!ignores])
        ret_deval_vec = FiniteDifferences.jvp(fdm, f_vec,
                                              (sig_arg_val_vec, sig_arg_dval_vec))
        return from_vec_out(ret_deval_vec)
    elseif rettype <: Union{BatchDuplicated,BatchDuplicatedNoNeed}
        all(ignores) && return (var"1"=Enzyme.make_zero(y),)
        ret_dvals = map(ẋs[.!ignores]...) do sig_args_dvals...
            sig_args_dvals_vec, _ = to_vec(sig_args_dvals)
            ret_dval_vec = FiniteDifferences.jvp(fdm, f_vec,
                                                 (sig_arg_val_vec, sig_args_dvals_vec))
            return from_vec_out(ret_dval_vec)
        end
        return NamedTuple{ntuple(Symbol, length(ret_dvals))}(ret_dvals)
    else
        throw(ArgumentError("Unsupported return type: $rettype"))
    end
end
_fd_forward(fdm, f, ::Type{<:Const}, y, activities) = ()

#=
    _fd_reverse(fdm, f, ȳ, activities, active_return)

Call `FiniteDifferences.j′vp` on `f` with the arguments `xs` determined by `activities`.

# Arguments
- `fdm::FiniteDifferenceMethod`: How to numerically differentiate `f`.
- `f`: The function to differentiate.
- `ȳ`: The cotangent of the primal output `y=f(xs...)`.
- `activities`: activities that would be passed to `Enzyme.autodiff`
- `active_return`: whether the return is non-constant
# Returns
- `x̄s`: Derivatives of output `s` w.r.t. `xs` estimated by finite differencing.
=#
function _fd_reverse(fdm, f, ȳ, activities, active_return)
    xs = map(x -> x.val, activities)
    ignores = map(a -> a isa Const, activities)
    f_sig_args = _wrap_reverse_function(active_return, f, xs, ignores)
    all(ignores) && return map(Enzyme.make_zero, xs)
    ignores = collect(ignores)
    is_batch = _any_batch_duplicated(map(typeof, activities)...)
    batch_size = is_batch ? _batch_size(map(typeof, activities)...) : 1
    x̄s = map(collect(activities)) do a
        if a isa Union{Const,Active}
            dval = ntuple(_ -> Enzyme.make_zero(a.val), batch_size)
            return is_batch ? dval : dval[1]
        else
            return a.dval
        end
    end
    sigargs = xs[.!ignores]
    s̄igargs = x̄s[.!ignores]
    sigarginds = eachindex(x̄s)[.!ignores]
    sigargs_vec, from_vec_in = to_vec(sigargs)
    # vectorize inputs and outputs of function
    f_vec = first ∘ to_vec ∘ Base.splat(f_sig_args) ∘ from_vec_in
    if !is_batch
        ȳ_extended = (ȳ, s̄igargs...)
        ȳ_extended_vec, _ = to_vec(ȳ_extended)
        fd_vec = only(FiniteDifferences.j′vp(fdm, f_vec, ȳ_extended_vec, sigargs_vec))
        fd = from_vec_in(fd_vec)
    else
        fd = Tuple(zip(map(ȳ, s̄igargs...) do ȳ_extended...
                           ȳ_extended_vec, _ = to_vec(ȳ_extended)
                           fd_vec = only(FiniteDifferences.j′vp(fdm, f_vec, ȳ_extended_vec,
                                                                sigargs_vec))
                           return from_vec_in(fd_vec)
                       end...))
    end
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
function _wrap_reverse_function(active_return, f, xs, ignores)
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

        # if an arg and a return alias, do not consider the contribution from the arg as returned here,
        # it will already be taken into account. This is implemented using the deepcopy_internal, which
        # will add all objects inside the return into the dict `zeros`.
        zeros = IdDict()
        origRet = Base.deepcopy_internal(deepcopy(f)(callargs...), zeros)

        # we will now explicitly zero all objects returned, and replace any of the args with this
        # zero, if the input and output alias.
        if active_return
            for k in keys(zeros)
                zeros[k] = Enzyme.make_zero(k)
            end
        end

        return (origRet, Base.deepcopy_internal(retargs, zeros)...)
    end
    return fnew
end
