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
