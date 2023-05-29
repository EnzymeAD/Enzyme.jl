using ConstructionBase
using Enzyme
using EnzymeCore: Annotation
using EnzymeCore.EnzymeRules: EnzymeRules
using FiniteDifferences
using MetaTesting
using Random
using Test

# _make_jvp_call and _wrap_function adapted from ChainRulesTestUtils
# https://github.com/JuliaDiff/ChainRulesTestUtils.jl/blob/f76f3fc7be221e07ba9be28ef33a22238ef13661/src/finite_difference_calls.jl
# Copyright (c) 2020 JuliaDiff

"""
    _make_jvp_call(fdm, f, rettype, y, activities)

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
"""
function _make_jvp_call(fdm, f, rettype, y, activities)
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
_make_jvp_call(fdm, f, ::Type{<:Const}, y, activities) = ()


"""
    _make_j′vp_call(fdm, f, ȳ, activities)

Call `FiniteDifferences.j′vp` on `f` with the arguments `xs` determined by `activities`.

# Arguments
- `fdm::FiniteDifferenceMethod`: How to numerically differentiate `f`.
- `f`: The function to differentiate.
- `ȳ`: The cotangent of the primal output `y=f(xs...)`.
- `activities`: activities that would be passed to `Enzyme.autodiff`

# Returns
- `x̄s`: Derivatives of output `s` w.r.t. `xs` estimated by finite differencing.
"""
function _make_j′vp_call(fdm, f, ȳ, activities)
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

"""
    _wrap_forward_function(f, xs, ignores)

Return a new version of `f`, `fnew`, that ignores some of the arguments `xs`.

# Arguments
- `f`: The function to be wrapped.
- `xs`: Inputs to `f`, such that `y = f(xs...)`.
- `ignores`: Collection of `Bool`s, the same length as `xs`.
  If `ignores[i] === true`, then `xs[i]` is ignored; `∂xs[i] === NoTangent()`.
"""
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

"""
    _wrap_reverse_function(f, xs, ignores)

Return a new version of `f`, `fnew`, that ignores some of the arguments `xs` and returns
also non-ignored arguments.

All arguments are copied before being passed to `f`, so that `fnew` is non-mutating.

# Arguments
- `f`: The function to be wrapped.
- `xs`: Inputs to `f`, such that `y = f(xs...)`.
- `ignores`: Collection of `Bool`s, the same length as `xs`.
  If `ignores[i] === true`, then `xs[i]` is ignored; `∂xs[i] === NoTangent()`.
"""
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

# recursively apply f to all fields of x for which f is implemented; all other fields are
# left unchanged
function map_fields_recursive(f, x::T...) where {T}
    fields = map(ConstructionBase.getfields, x)
    all(isempty, fields) && return first(x)
    new_fields = map(fields...) do xi...
        map_fields_recursive(f, xi...)
    end
    return ConstructionBase.constructorof(T)(new_fields...)
end
function map_fields_recursive(f, x::T...) where {T<:Union{Array,Tuple,NamedTuple}}
    map(x...) do xi...
        map_fields_recursive(f, xi...)
    end
end
map_fields_recursive(f, x::T...) where {T<:AbstractFloat} = f(x...)
map_fields_recursive(f, x::Array{<:Number}...) = f(x...)

rand_tangent(x) = rand_tangent(Random.default_rng(), x)
rand_tangent(rng, x) = map_fields_recursive(Base.Fix1(rand_tangent, rng), x)
# make numbers prettier sometimes when errors are printed.
rand_tangent(rng, ::T) where {T<:AbstractFloat} = rand(rng, -9:T(0.01):9)
rand_tangent(rng, x::T) where {T<:Array{<:Number}} = rand_tangent.(rng, x)

zero_tangent(x) = map_fields_recursive(zero_tangent, x)
zero_tangent(::T) where {T<:AbstractFloat} = zero(T)
zero_tangent(x::T) where {T<:Array{<:Number}} = zero_tangent.(x)

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

function auto_activity(arg::Tuple)
    if length(arg) == 2 && arg[2] isa Type && arg[2] <: Annotation
        return _build_activity(arg...)
    end
    return Const(arg)
end
auto_activity(activity::Annotation) = activity
auto_activity(activity) = Const(activity)

_build_activity(primal, ::Type{<:Const}) = Const(primal)
_build_activity(primal, ::Type{<:Active}) = Active(primal)
_build_activity(primal, ::Type{<:Duplicated}) = Duplicated(primal, rand_tangent(primal))
function _build_activity(primal, ::Type{<:BatchDuplicated})
    return BatchDuplicated(primal, ntuple(_ -> rand_tangent(primal), 2))
end
function _build_activity(primal, T::Type{<:Annotation})
    throw(ArgumentError("Unsupported activity type: $T"))
end

"""
    test_forward(f, Activity, args...; kwargs...)

Test `Enzyme.autodiff` of `f` in `Forward`-mode against finite differences.

`f` has the constraints of the same argument passed to `Enzyme.autodiff`.

# Arguments

- `Activity`: the activity of the return value of `f`
- `args`: Each entry is either an argument to `f`, an activity type accepted by `autodiff`,
    or a tuple of the form `(arg, Activity)`, where `Activity` is the activity type of
    `arg`. If the activity type specified requires a tangent, a random tangent will be
    automatically generated.

# Keywords

- `fdm=FiniteDifferences.central_fdm(5, 1)`: The finite differences method to use.
- `fkwargs`: Keyword arguments to pass to `f`.
- `rtol`: Relative tolerance for `isapprox`.
- `atol`: Absolute tolerance for `isapprox`.
- `testset_name`: Name to use for a testset in which all tests are evaluated.

# Examples

```julia
x = randn()
y = randn()  # will be Const
for Tret in (Const, Duplicated, DuplicatedNoNeed), Tx in (Const, Duplicated)
    test_forward(*, Tret, (x, Tx), y)
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
    call_with_copy(f, xs...) = deepcopy(f)(deepcopy(xs)...; deepcopy(fkwargs)...)
    call_with_kwargs(f, xs...) = f(xs...; fkwargs...)
    if testset_name === nothing
        testset_name = "test_forward: $(f isa Const ? f.val : f) with return activity $ret_activity on $(args)"
    end
    @testset "$testset_name" begin
        # format arguments for autodiff and FiniteDifferences
        activities = map(auto_activity, (f, args...))
        primals = map(x -> x.val, activities)
        # call primal, avoid mutating original arguments
        y = call_with_copy(primals...)
        # call finitedifferences, avoid mutating original arguments
        dy_fdm = _make_jvp_call(fdm, call_with_copy, ret_activity, y, activities)
        # call autodiff, allow mutating original arguments
        y_and_dy_ad = autodiff(Forward, call_with_kwargs, ret_activity, activities...)
        if ret_activity <: Union{Duplicated,BatchDuplicated}
            y_ad, dy_ad = y_and_dy_ad
            # check primal agrees with primal function
            test_approx(y_ad, y; atol, rtol)
        elseif ret_activity <: Union{DuplicatedNoNeed,BatchDuplicatedNoNeed}
            # check primal is not returned
            @test length(y_and_dy_ad) == 1
            dy_ad = y_and_dy_ad[1]
        elseif ret_activity <: Const
            # check Const activity returns an empty tuple
            @test isempty(y_and_dy_ad)
            dy_ad = ()
        else
            throw(ArgumentError("Unsupported return activity type: $ret_activity"))
        end
        if y isa Tuple
            # check Enzyme and FiniteDifferences return the same number of derivatives
            @test length(dy_ad) == length(dy_fdm)
            # check all returned derivatives against FiniteDifferences
            for (dy_ad_i, dy_fdm_i) in zip(dy_ad, dy_fdm)
                test_approx(dy_ad_i, dy_fdm_i; atol, rtol)
            end
        else
            test_approx(dy_ad, dy_fdm; atol, rtol)
        end
    end
end

"""
    test_reverse(f, Activity, args...; kwargs...)

Test `Enzyme.autodiff` of `f` in `Reverse`-mode against finite differences.

`f` has all constraints of the same argument passed to `Enzyme.autodiff_thunk`, with several
additional constraints:
- If it mutates one of its arguments, it must not also return that argument.
- If the return value is a struct, then all floating point numbers contained in the struct
    or its fields must be in arrays.

# Arguments

- `Activity`: the activity of the return value of `f`.
- `args`: Each entry is either an argument to `f`, an activity type accepted by `autodiff`,
    or a tuple of the form `(arg, Activity)`, where `Activity` is the activity type of
    `arg`. If the activity type specified requires a shadow, one will be automatically
    generated.

# Keywords

- `fdm=FiniteDifferences.central_fdm(5, 1)`: The finite differences method to use.
- `fkwargs`: Keyword arguments to pass to `f`.
- `rtol`: Relative tolerance for `isapprox`.
- `atol`: Absolute tolerance for `isapprox`.
- `testset_name`: Name to use for a testset in which all tests are evaluated.

# Examples

Testing a function that returns a scalar:

```julia
x = randn()
y = randn()  # will be Const
for Tret in (Const, Active), Tx in (Const, Active)
    test_reverse(*, Tret, (x, Tx), y)
end
```

Testing a function that returns an array:

```julia
x = randn(3)
y = randn()  # will be Const
for Tret in (Const, Duplicated), Tx in (Const, Duplicated)
    test_reverse(*, Tret, (x, Tx), y)
end
```
"""
function test_reverse(
    f,
    ret_activity,
    args...;
    fdm=FiniteDifferences.central_fdm(5, 1),
    fkwargs::NamedTuple=NamedTuple(),
    rtol::Real=1e-9,
    atol::Real=1e-9,
    testset_name=nothing,
)
    call_with_copy(f, xs...) = deepcopy(f)(deepcopy(xs)...; deepcopy(fkwargs)...)
    call_with_kwargs(f, xs...) = f(xs...; fkwargs...)
    if testset_name === nothing
        testset_name = "test_reverse: $(f isa Const ? f.val : f) with return activity $ret_activity on $(args)"
    end
    @testset "$testset_name" begin
        # format arguments for autodiff and FiniteDifferences
        activities = map(auto_activity, (f, args...))
        primals = map(x -> x.val, activities)
        # call primal, avoid mutating original arguments
        y = call_with_copy(primals...)
        # generate tangent for output
        ȳ = ret_activity <: Const ? zero_tangent(y) : rand_tangent(y)
        # call finitedifferences, avoid mutating original arguments
        dx_fdm = _make_j′vp_call(fdm, call_with_kwargs, ȳ, activities)
        # call autodiff, allow mutating original arguments
        c_act = Const(call_with_kwargs)
        forward, reverse = autodiff_thunk(
            ReverseSplitWithPrimal, typeof(c_act), ret_activity, map(typeof, activities)...
        )
        tape, y_ad, shadow_result = forward(c_act, activities...)
        if ret_activity <: Active
            dx_ad = only(reverse(c_act, activities..., ȳ, tape))
        else
            # if there's a shadow result, then we need to set it to our random adjoint
            if !(shadow_result === nothing)
                map_fields_recursive(copyto!, shadow_result, ȳ)
            end
            dx_ad = only(reverse(c_act, activities..., tape))
        end
        # check primal agrees with primal function
        test_approx(y_ad, y; atol, rtol)
        @test length(dx_ad) == length(dx_fdm) == length(activities)
        # check all returned derivatives against FiniteDifferences
        for (act_i, dx_ad_i, dx_fdm_i) in zip(activities, dx_ad, dx_fdm)
            if act_i isa Active
                test_approx(dx_ad_i, dx_fdm_i; atol, rtol)
                continue
            end
            # if not Active, returned derivative must be nothing
            @test dx_ad_i === nothing
            act_i isa Const && continue
            # if not Active or Const, derivative stored in Duplicated
            test_approx(act_i.dval, dx_fdm_i; atol, rtol)
        end
    end
end

"""
    all_or_no_batch(activities...) -> Bool

Returns `true` if `activities` are compatible in terms of batched activities.

When a test set loops over many activities, some of which may be `BatchedDuplicated` or
`BatchedDuplicatedNoNeed`, this is useful for skipping those combinations that are
incompatible and will raise errors.
"""
function all_or_no_batch(activities...)
    no_batch = !any(activities) do T
        T <: Union{BatchDuplicated,BatchDuplicatedNoNeed}
    end
    all_batch_or_const = all(activities) do T
        T <: Union{BatchDuplicated,BatchDuplicatedNoNeed,Const}
    end
    return all_batch_or_const || no_batch
end

f_array(x) = sum(exp, x)
f_tuple(x) = (-3 * x[1], 2 * x[2])
f_namedtuple(x) = (s=sin(x.a), c=cos(x.b))
struct TestStruct{X,A}
    x::X
    a::A
end
f_struct(x::TestStruct) = TestStruct(sinh.(x.a .* x.x), exp(x.a))
f_multiarg(x::AbstractArray, a) = sin.(a .* x)
function f_mut!(y, x, a)
    y .= x .* a
    return y
end

f_kwargs(x; a=3.0, kwargs...) = a .* x .^ 2

struct MutatedCallable{T}
    x::T
end
function (c::MutatedCallable)(y)
    s = c.x'y
    c.x ./= s
    return s
end

function EnzymeRules.forward(
    func::Const{typeof(f_kwargs)},
    RT::Type{
        <:Union{Const,Duplicated,DuplicatedNoNeed,BatchDuplicated,BatchDuplicatedNoNeed}
    },
    x::Union{Const,Duplicated,BatchDuplicated};
    a=4.0, # mismatched keyword
    incorrect_primal=false,
    incorrect_tangent=false,
    incorrect_batched_tangent=false,
    kwargs...,
)
    if RT <: Const
        return func.val(x.val; a=(incorrect_primal ? a - 1 : a), kwargs...)
    end
    dval = if x isa Duplicated
        2 * (incorrect_tangent ? (a + 2) : a) .* x.val .* x.dval
    elseif x isa BatchDuplicated
        map(x.dval) do dx
            2 * (incorrect_batched_tangent ? (a - 2) : a) .* x.val .* dx
        end
    else
        (incorrect_tangent | incorrect_batched_tangent) ? 2 * x.val : zero(a) * x.val
    end

    if RT <: Union{DuplicatedNoNeed,BatchDuplicatedNoNeed}
        return dval
    else
        val = func.val(x.val; a=(incorrect_primal ? a - 1 : a), kwargs...)
        return RT(val, dval)
    end
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.ConfigWidth{1},
    func::Const{typeof(f_kwargs)},
    RT::Type{<:Union{Const,Duplicated,DuplicatedNoNeed}},
    x::Union{Const,Duplicated};
    a=4.0, # mismatched keyword
    incorrect_primal=false,
    incorrect_tape=false,
    kwargs...,
)
    xtape = incorrect_tape ? x.val * 3 : copy(x.val)
    if EnzymeRules.needs_primal(config) || EnzymeRules.needs_shadow(config)
        val = func.val(x.val; a=(incorrect_primal ? a - 1 : a), kwargs...)
    else
        val = nothing
    end
    primal = EnzymeRules.needs_primal(config) ? val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(val) : nothing
    tape = (xtape, shadow)
    return EnzymeRules.AugmentedReturn(primal, shadow, tape)
end

function EnzymeRules.reverse(
    config::EnzymeRules.ConfigWidth{1},
    func::Const{typeof(f_kwargs)},
    dret::Type{<:Union{Const,Duplicated,DuplicatedNoNeed}},
    tape,
    x::Union{Const,Duplicated};
    a=4.0, # mismatched keyword
    incorrect_tangent=false,
    kwargs...,
)
    xval, dval = tape
    if !(x isa Const) && (dval !== nothing)
        x.dval .+= 2 .* (incorrect_tangent ? (a + 2) : a) .* dval .* xval
    end
    return (nothing,)
end

@testset "EnzymeRules testing functions" begin
    @testset "test_forward" begin
        @testset "tests pass for functions with no rules" begin
            @testset "unary function tests" begin
                combinations = [
                    "vector arguments" => (Vector, f_array),
                    "matrix arguments" => (Matrix, f_array),
                    "multidimensional array arguments" => (Array{<:Any,3}, f_array),
                    "tuple argument and return" => (Tuple, f_tuple),
                    "namedtuple argument and return" => (NamedTuple, f_namedtuple),
                    # "struct argument and return" => (TestStruct, f_struct),
                ]
                sz = (2, 3, 4)
                @testset "$name" for (name, (TT, fun)) in combinations
                    @testset for Tret in (
                            Const,
                            Duplicated,
                            DuplicatedNoNeed,
                            BatchDuplicated,
                            BatchDuplicatedNoNeed,
                        ),
                        Tx in (Const, Duplicated, BatchDuplicated),
                        T in (Float32, Float64, ComplexF32, ComplexF64)

                        # skip invalid combinations
                        all_or_no_batch(Tret, Tx) || continue

                        if TT <: Array
                            x = randn(T, sz[1:ndims(TT)])
                        elseif TT <: Tuple
                            x = (randn(T), randn(T))
                        elseif TT <: NamedTuple
                            x = (a=randn(T), b=randn(T))
                        else  # TT <: TestStruct
                            x = TestStruct(randn(T, 5), randn(T))
                        end
                        atol = rtol = sqrt(eps(real(T)))
                        test_forward(fun, Tret, (x, Tx); atol, rtol)
                    end
                end
            end

            @testset "multi-argument function" begin
                @testset for Tret in (
                        Const,
                        Duplicated,
                        DuplicatedNoNeed,
                        BatchDuplicated,
                        BatchDuplicatedNoNeed,
                    ),
                    Tx in (Const, Duplicated, BatchDuplicated),
                    Ta in (Const, Duplicated, BatchDuplicated),
                    T in (Float32, Float64, ComplexF32, ComplexF64)

                    # skip invalid combinations
                    all_or_no_batch(Tret, Tx, Ta) || continue

                    x = randn(T, 3)
                    a = randn(T)
                    atol = rtol = sqrt(eps(real(T)))
                    test_forward(f_multiarg, Tret, (x, Tx), (a, Ta); atol, rtol)
                end
            end

            @testset "mutating function" begin
                Enzyme.API.runtimeActivity!(true)
                sz = (2, 3)
                @testset for Tret in (Const, Duplicated, BatchDuplicated),
                    Tx in (Const, Duplicated, BatchDuplicated),
                    Ta in (Const, Duplicated, BatchDuplicated),
                    T in (Float32, Float64, ComplexF32, ComplexF64)

                    # if some are batch, all non-Const must be batch
                    all_or_no_batch(Tret, Tx, Ta) || continue
                    # since y is returned, it needs the same activity as the return type
                    Ty = Tret

                    x = randn(T, sz)
                    y = zeros(T, sz)
                    a = randn(T)

                    atol = rtol = sqrt(eps(real(T)))
                    test_forward(f_mut!, Tret, (y, Ty), (x, Tx), (a, Ta); atol, rtol)
                end
                Enzyme.API.runtimeActivity!(false)
            end

            @testset "mutated callable" begin
                n = 3
                @testset for Tret in (Const, Duplicated, BatchDuplicated),
                    Tc in (Const, Duplicated, BatchDuplicated),
                    Ty in (Const, Duplicated, BatchDuplicated),
                    T in (Float32, Float64, ComplexF32, ComplexF64)

                    # if some are batch, all non-Const must be batch
                    all_or_no_batch(Tret, Tc, Ty) || continue

                    c = MutatedCallable(randn(T, n))
                    y = randn(T, n)

                    atol = rtol = sqrt(eps(real(T)))
                    test_forward((c, Tc), Tret, (y, Ty); atol, rtol)
                end
            end
        end

        @testset "kwargs correctly forwarded" begin
            @testset for Tret in (Duplicated, BatchDuplicated),
                Tx in (Const, Duplicated, BatchDuplicated)

                all_or_no_batch(Tret, Tx) || continue

                x = randn(3)
                a = randn()

                @test fails() do
                    test_forward(f_kwargs, Tret, (x, Tx))
                end
                test_forward(f_kwargs, Tret, (x, Tx); fkwargs=(; a))
            end
        end

        @testset "incorrect primal detected" begin
            @testset for Tret in (Duplicated, BatchDuplicated),
                Tx in (Const, Duplicated, BatchDuplicated)

                all_or_no_batch(Tret, Tx) || continue

                x = randn(3)
                a = randn()

                test_forward(f_kwargs, Tret, (x, Tx); fkwargs=(; a))
                fkwargs = (; a, incorrect_primal=true)
                @test fails() do
                    test_forward(f_kwargs, Tret, (x, Tx); fkwargs)
                end
            end
        end

        @testset "incorrect tangent detected" begin
            @testset for Tret in (Duplicated, DuplicatedNoNeed), Tx in (Const, Duplicated)
                x = randn(3)
                a = randn()

                test_forward(f_kwargs, Tret, (x, Tx); fkwargs=(; a))
                fkwargs = (; a, incorrect_tangent=true)
                @test fails() do
                    test_forward(f_kwargs, Tret, (x, Tx); fkwargs)
                end
            end
        end

        @testset "incorrect batch tangent detected" begin
            @testset for Tret in (BatchDuplicated, BatchDuplicatedNoNeed),
                Tx in (Const, BatchDuplicated)

                x = randn(3)
                a = randn()

                test_forward(f_kwargs, Tret, (x, Tx); fkwargs=(; a))
                fkwargs = (; a, incorrect_batched_tangent=true)
                @test fails() do
                    test_forward(f_kwargs, Tret, (x, Tx); fkwargs)
                end
            end
        end
    end

    @testset "test_reverse" begin
        @testset "tests pass for functions with no rules" begin
            @testset "unary function tests" begin
                combinations = [
                    "vector arguments" => (Vector, f_array),
                    "matrix arguments" => (Matrix, f_array),
                    "multidimensional array arguments" => (Array{<:Any,3}, f_array),
                ]
                sz = (2, 3, 4)
                @testset "$name" for (name, (TT, fun)) in combinations
                    @testset for Tret in (Active, Const),
                        Tx in (Const, Duplicated),
                        T in (Float32, Float64, ComplexF32, ComplexF64)

                        x = randn(T, sz[1:ndims(TT)])
                        atol = rtol = sqrt(eps(real(T)))
                        test_reverse(fun, Tret, (x, Tx); atol, rtol)
                    end
                end
            end

            @testset "multi-argument function" begin
                @testset for Tret in (Const, Duplicated),
                    Tx in (Const, Duplicated),
                    Ta in (Const, Active),
                    T in (Float32, Float64, ComplexF32, ComplexF64)

                    x = randn(T, 3)
                    a = randn(T)
                    atol = rtol = sqrt(eps(real(T)))
                    test_reverse(f_multiarg, Tret, (x, Tx), (a, Ta); atol, rtol)
                end
            end

            @testset "mutating function" begin
                sz = (2, 3)
                Enzyme.API.runtimeActivity!(true)
                @testset for Ty in (Const, Duplicated),
                    Tx in (Const, Duplicated),
                    Ta in (Const, Active),
                    T in (Float32, Float64, ComplexF32, ComplexF64)

                    # return value is nothing
                    Tret = Const

                    x = randn(T, sz)
                    y = zeros(T, sz)
                    a = randn(T)

                    atol = rtol = sqrt(eps(real(T)))
                    test_reverse(f_mut!, Tret, (y, Ty), (x, Tx), (a, Ta); atol, rtol)
                end
                Enzyme.API.runtimeActivity!(false)
            end

            @testset "mutated callable" begin
                n = 3
                @testset for Tret in (Const, Active),
                    Tc in (Const, Duplicated),
                    Ty in (Const, Duplicated),
                    T in (Float32, Float64, ComplexF32, ComplexF64)

                    c = MutatedCallable(randn(T, n))
                    y = randn(T, n)

                    atol = rtol = sqrt(eps(real(T)))
                    test_reverse((c, Tc), Tret, (y, Ty); atol, rtol)
                end
            end
        end

        @testset "kwargs correctly forwarded" begin
            @testset for Tx in (Const, Duplicated)
                x = randn(3)
                a = randn()

                @test fails() do
                    test_reverse(f_kwargs, Duplicated, (x, Tx))
                end
                test_reverse(f_kwargs, Duplicated, (x, Tx); fkwargs=(; a))
            end
        end

        @testset "incorrect primal detected" begin
            @testset for Tx in (Const, Duplicated)
                x = randn(3)
                a = randn()

                test_reverse(f_kwargs, Duplicated, (x, Tx); fkwargs=(; a))
                fkwargs = (; a, incorrect_primal=true)
                @test fails() do
                    test_reverse(f_kwargs, Duplicated, (x, Tx); fkwargs)
                end
            end
        end

        @testset "incorrect tangent detected" begin
            @testset for Tx in (Duplicated,)
                x = randn(3)
                a = randn()

                test_reverse(f_kwargs, Duplicated, (x, Tx); fkwargs=(; a))
                fkwargs = (; a, incorrect_tangent=true)
                @test fails() do
                    test_reverse(f_kwargs, Duplicated, (x, Tx); fkwargs)
                end
            end
        end

        @testset "incorrect tape detected" begin
            @testset for Tx in (Duplicated,)
                x = randn(3)
                a = randn()

                function f_kwargs_overwrite(x; kwargs...)
                    y = f_kwargs(x; kwargs...)
                    x[1] = 0.0
                    return y
                end

                test_reverse(f_kwargs_overwrite, Duplicated, (x, Tx); fkwargs=(; a))
                fkwargs = (; a, incorrect_tape=true)
                @test fails() do
                    test_reverse(f_kwargs_overwrite, Duplicated, (x, Tx); fkwargs)
                end
            end
        end
    end
end
