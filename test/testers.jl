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
# special-case containers that can't be constructed from type and field
rand_tangent(rng, x::Union{Array,Tuple,NamedTuple}) = map(xi -> rand_tangent(rng, xi), x)
# make numbers prettier sometimes when errors are printed.
rand_tangent(rng, ::T) where {T<:AbstractFloat} = rand(rng, -9:T(0.01):9)

function zero_tangent(x::T) where {T}
    fields = fieldnames(T)
    isempty(fields) && return x
    return typeof(x)((zero_tangent(getfield(x, k)) for k in fields)...)
end
# special-case containers that can't be constructed from type and field
zero_tangent(x::Union{Array,Tuple,NamedTuple}) = map(zero_tangent, x)
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
    test_forward(f, Activity, args...; kwargs...)

Test `Enzyme.autodiff` of `f` in `Forward`-mode against finite differences.

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
        activities = map(auto_forward_activity, (f, args...))
        primals = map(x -> x.val, activities)
        # call primal, avoid mutating original arguments
        y = call_with_copy(primals...)
        # TODO: handle batch activities
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

# meta-testing functions adapted from https://github.com/JuliaDiff/ChainRulesTestUtils.jl/blob/v1.10.1/test/meta_testing_tools.jl
# Copyright (c) 2020 JuliaDiff

"""
    EncasedTestSet(desc, results) <: AbstractTestset

A custom testset that encases all test results within, not letting them out.
It doesn't let anything propagate up to the parent testset
(or to the top-level fallback testset, which throws an error on any non-passing result).
Not passes, not failures, not even errors.


This is useful for being able to observe the testsets results programatically;
without them triggering actual passes/failures/errors.
"""
struct EncasedTestSet <: Test.AbstractTestSet
    description::String
    results::Vector{Any}
end
EncasedTestSet(desc) = EncasedTestSet(desc, [])

Test.record(ts::EncasedTestSet, t) = (push!(ts.results, t); t)

function Test.finish(ts::EncasedTestSet)
    if Test.get_testset_depth() != 0
        # Attach this test set to the parent test set *if* it is also a NonPassingTestset
        # Otherwise don't as we don't want to push the errors and failures further up.
        parent_ts = Test.get_testset()
        parent_ts isa EncasedTestSet && Test.record(parent_ts, ts)
        return ts
    end
    return ts
end

"""
    nonpassing_results(f)

`f` should be a function that takes no argument, and calls some code that used `@test`.
Invoking it via `nonpassing_results(f)` will prevent those `@test` being added to the
current testset, and will return a collection of all nonpassing test results.
"""
function nonpassing_results(f)
    # Specify testset type to hijack system
    ts = @testset EncasedTestSet "nonpassing internal" begin
        f()
    end
    return _extract_nonpasses(ts)
end

"extracts as flat collection of failures from a (potential nested) testset"
_extract_nonpasses(x::Test.Result) = [x]
_extract_nonpasses(x::Test.Pass) = Test.Result[]
_extract_nonpasses(ts::EncasedTestSet) = _extract_nonpasses(ts.results)
function _extract_nonpasses(xs::Vector)
    if isempty(xs)
        return Test.Result[]
    else
        return mapreduce(_extract_nonpasses, vcat, xs)
    end
end

"""
    fails(f)

`f` should be a function that takes no argument, and calls some code that used `@test`.
`fails(f)` returns true if at least 1 `@test` fails.
If a test errors then it will display that error and throw an error of its own.
"""
function fails(f)
    results = nonpassing_results(f)
    did_fail = false
    for result in results
        did_fail |= result isa Test.Fail
        if result isa Test.Error
            # Log a error message, with original backtrace
            # Sadly we can't throw the original exception as it is only stored as a String
            error("Error occurred during `fails`")
        end
    end
    return did_fail
end

"""
    errors(f, msg_pattern="")

Returns true if at least 1 error is recorded into a testset
with a failure matching the given pattern.

`f` should be a function that takes no argument, and calls some code that uses `@testset`.
`msg_pattern` is a regex or a string, that should be contained in the error message.
If nothing is passed then it default to the empty string, which matches any error message.

If a test fails (rather than passing or erroring) then `errors` will throw an error.
"""
function errors(f, msg_pattern="")
    results = nonpassing_results(f)

    for result in results
        result isa Test.Fail && error("Test actually failed (not errored): \n $result")
        result isa Test.Error && occursin(msg_pattern, result.value) && return true
    end
    return false  # no matching error occured
end

f_array(x) = sum(sin, x)
f_tuple(x) = (sin(x[1]), cos(x[2]))
f_namedtuple(x) = (s=sin(x.a), c=cos(x.b))
struct Foo{X,A}
    x::X
    a::A
end
f_struct(x::Foo) = Foo(sinh.(x.a .* x.x), exp(x.a))
f_multiarg(x::AbstractArray, a) = sin.(a .* x)
function f_mut!(y, x, a)
    y .= x .* a
    return y
end

@testset "test_forward" begin
    @testset "tests pass for functions with no rules" begin
        @testset "unary function tests" begin
            combinations = [
                "vector arguments" => (Vector, f_array),
                "matrix arguments" => (Matrix, f_array),
                "multidimensional array arguments" => (Array{<:Any,3}, f_array),
                "tuple argument and return" => (Tuple, f_tuple),
                "namedtuple argument and return" => (NamedTuple, f_namedtuple),
                "struct argument and return" => (Foo, f_struct),
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
                    T in (Float32, Float64) # TODO: test complex

                    # skip invalid combinations
                    all_or_no_batch(Tret, Tx) || continue

                    if TT <: Array
                        x = randn(T, sz[1:ndims(TT)])
                    elseif TT <: Tuple
                        x = (randn(T), randn(T))
                    elseif TT <: NamedTuple
                        x = (a=randn(T), b=randn(T))
                    else  # TT <: Foo
                        x = Foo(randn(T, 5), randn(T))
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
                T in (Float32, Float64) # TODO: test complex

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
                T in (Float64, Float32, Float64, ComplexF32, ComplexF64)

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
    end
end
