"""
    test_forward(f, Activity, args...; kwargs...)

Test `Enzyme.autodiff` of `f` in `Forward`-mode against finite differences.

`f` has all constraints of the same argument passed to `Enzyme.autodiff`, with additional
constraints:
- If it mutates one of its arguments, it _must_ return that argument.

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

`isapprox(...; rtol, atol)` is used in the comparison of the *primal* and *the partial
derivative(s)*.

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
function test_forward(
    f,
    ret_activity,
    args...;
    rng::Random.AbstractRNG=Random.default_rng(),
    fdm=FiniteDifferences.central_fdm(5, 1),
    fkwargs::NamedTuple=NamedTuple(),
    rtol::Real=1e-9,
    atol::Real=1e-9,
    testset_name=nothing,
    runtime_activity::Bool=false
)
    call_with_copy(f, xs...) = deepcopy(f)(deepcopy(xs)...; deepcopy(fkwargs)...)
    call_with_kwargs(f, xs...) = f(xs...; fkwargs...)
    if testset_name === nothing
        testset_name = "test_forward: $f with return activity $ret_activity on $(_string_activity(args))"
    end
    @testset "$testset_name" begin
        # format arguments for autodiff and FiniteDifferences
        activities = map(Base.Fix1(auto_activity, rng), (f, args...))
        primals = map(x -> x.val, activities)
        # call primal, avoid mutating original arguments
        fcopy = deepcopy(first(primals))
        args_copy = deepcopy(Base.tail(primals))
        y = fcopy(args_copy...; deepcopy(fkwargs)...)
        # call finitedifferences, avoid mutating original arguments
        dy_fdm = _fd_forward(fdm, call_with_copy, ret_activity, y, activities)
        # call autodiff, allow mutating original arguments
        mode = if ret_activity <: Union{DuplicatedNoNeed,BatchDuplicatedNoNeed, Const}
            Forward
        else
            ForwardWithPrimal
        end
        mode = set_runtime_activity(mode, runtime_activity)

        ret_activity2 = if ret_activity <: DuplicatedNoNeed
            Duplicated
        elseif ret_activity <: BatchDuplicatedNoNeed
            BatchDuplicated
        else
            ret_activity
        end
        y_and_dy_ad = autodiff(mode, call_with_kwargs, ret_activity2, activities...)
        if ret_activity <: Union{Duplicated,BatchDuplicated}
            @test_msg(
                "For return type $ret_activity the return value and derivative must be returned",
                length(y_and_dy_ad) == 2,
            )
            dy_ad, y_ad = y_and_dy_ad
            test_approx(
                y_ad, y, "The return value of the rule and function must agree"; atol, rtol
            )
        elseif ret_activity <: Union{DuplicatedNoNeed,BatchDuplicatedNoNeed}
            @test_msg(
                "For return type $ret_activity only the derivative should be returned",
                length(y_and_dy_ad) == 1,
            )
            dy_ad = y_and_dy_ad[1]
        elseif ret_activity <: Const
            @test_msg(
                "For return type $ret_activity an empty tuple must be returned",
                isempty(y_and_dy_ad),
            )
            dy_ad = ()
        else
            throw(ArgumentError("Unsupported return activity type: $ret_activity"))
        end
        test_approx(
            first(activities).val,
            fcopy,
            "The rule must mutate the callable the same way as the function";
            atol,
            rtol,
        )
        for (i, (act_i, arg_i)) in enumerate(zip(Base.tail(activities), args_copy))
            test_approx(
                act_i.val,
                arg_i,
                "The rule must mutate argument $i the same way as the function";
                atol,
                rtol,
            )
        end
        if y isa Tuple
            @assert length(dy_ad) == length(dy_fdm)
            # check all returned derivatives against FiniteDifferences
            for (i, (dy_ad_i, dy_fdm_i)) in enumerate(zip(dy_ad, dy_fdm))
                target_str = i == 1 ? "callable" : "argument $(i - 1)"
                test_approx(
                    dy_ad_i,
                    dy_fdm_i,
                    "derivative for $target_str should agree with finite differences";
                    atol,
                    rtol,
                )
            end
        else
            test_approx(
                dy_ad, dy_fdm, "derivative should agree with finite differences"; atol, rtol
            )
        end
    end
end
