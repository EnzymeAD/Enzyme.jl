"""
    test_rewind(f, Activity, args...; kwargs...)

Test `Enzyme.autodiff` of `f` in `Forward`-mode by backtracking using `Reverse`-mode,
which itself is checked against finite differences. This mode can be useful when computing
derivatives on functions such as matrix factorizations, where a particular choice of
gauge is important and the finite-differences approach generates tangents in an arbitrary
gauge. In effect, this plays the derivatives _forward_, then in _reverse_, "rewinding" the
tape.

`f` has all constraints of the same argument passed to `Enzyme.autodiff`, with additional
constraints:
- If it mutates one of its arguments, it _must_ return that argument.

To use this test mode, `f` _must_ have both forward and reverse rules defined.

# Arguments

- `Activity`: the activity of the return value of `f`
- `args`: Each entry is either an argument to `f`, an activity type accepted by `autodiff`,
    or a tuple of the form `(arg, Activity)`, where `Activity` is the activity type of
    `arg`. If the activity type specified requires a tangent, a random tangent will be
    automatically generated.

# Keywords

- `rng::AbstractRNG`: The random number generator to use for generating random tangents.
- `fdm=FiniteDifferences.central_fdm(5, 1)`: The finite differences method to use.
- `fkwargs`: Keyword arguments to pass to `f`.
- `rtol`: Relative tolerance for `isapprox`.
- `atol`: Absolute tolerance for `isapprox`.
- `testset_name`: Name to use for a testset in which all tests are evaluated.
- `output_tangent`: Optional final tangent to provide at the beginning of the reverse-mode differentiation 

# Examples

Here we test a rule for a function of scalars. Because we don't provide an activity
annotation for `y`, it is assumed to be `Const`.

```julia
using Enzyme, EnzymeTestUtils

x, y = randn(2)
for Tret in (Const, Duplicated, DuplicatedNoNeed), Tx in (Const, Duplicated)
    test_forward(*, Tret, (x, Tx), y)
end
```

Here we test a rule for a function of an array in batch forward-mode:

```julia
x = randn(3)
y = randn()
for Tret in (Const, BatchDuplicated, BatchDuplicatedNoNeed),
    Tx in (Const, BatchDuplicated),
    Ty in (Const, BatchDuplicated)

    test_forward(*, Tret, (x, Tx), (y, Ty))
end
```
"""

function test_rewind(
    f,
    fwd_ret_activity,
    rvs_ret_activity,
    args...;
    rng::Random.AbstractRNG=Random.default_rng(),
    fdm=FiniteDifferences.central_fdm(5, 1),
    fkwargs::NamedTuple=NamedTuple(),
    rtol::Real=1e-9,
    atol::Real=1e-9,
    testset_name=nothing,
    runtime_activity::Bool=false,
    output_tangent=nothing,
)
    # first, test reverse as normal with finite differences
    test_reverse(f, rvs_ret_activity, args...; rng=rng, fdm=fdm, fkwargs=fkwargs, rtol=rtol, atol=atol, testset_name=testset_name, runtime_activity=runtime_activity, output_tangent=output_tangent)
    # now, use the reverse rule to compare with the forward result 
    if testset_name === nothing
        testset_name = "test_rewind: $f with return activity $fwd_ret_activity on $(_string_activity(args))"
    end
    @testset "$testset_name" begin
        # test reverse rule to make sure it works with FD
        # run fwd mode first

        # format arguments for autodiff and FiniteDifferences
        activities = map(Base.Fix1(auto_activity, rng), (f, args...))
        primals = map(x -> x.val, activities)
        # call primal, avoid mutating original arguments
        fcopy = deepcopy(first(primals))
        args_copy = deepcopy(Base.tail(primals))
        y = fcopy(args_copy...; deepcopy(fkwargs)...)
        mode = if fwd_ret_activity <: Union{DuplicatedNoNeed, BatchDuplicatedNoNeed, Const}
            Forward
        else
            ForwardWithPrimal
        end
        mode = set_runtime_activity(mode, runtime_activity)
        ret_activity2 = if fwd_ret_activity <: DuplicatedNoNeed
            Duplicated
        elseif fwd_ret_activity <: BatchDuplicatedNoNeed
            BatchDuplicated
        else
            fwd_ret_activity
        end
        call_with_kwargs(f, xs...) = f(xs...; fkwargs...)
        y_and_dy_ad = autodiff(mode, call_with_kwargs, ret_activity2, activities...)
        dy_ad = y_and_dy_ad[1]
        # now run this back through reverse mode, using dy_ad from forward mode
        # as the output tangent
        test_reverse(f, rvs_ret_activity, args...; rng=rng, fdm=fdm, fkwargs=fkwargs, rtol=rtol, atol=atol, testset_name=testset_name, runtime_activity=runtime_activity, output_tangent=dy_ad)
    end
end
