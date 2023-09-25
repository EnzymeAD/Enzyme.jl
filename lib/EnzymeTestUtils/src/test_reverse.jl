"""
    test_reverse(f, Activity, args...; kwargs...)

Test `Enzyme.autodiff_thunk` of `f` in `ReverseSplitWithPrimal`-mode against finite
differences.

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

Here we test a rule for a function of scalars. Because we don't provide an activity
annotation for `y`, it is assumed to be `Const`.

```julia
using Enzyme, EnzymeTestUtils

x = randn()
y = randn()
for Tret in (Const, Active), Tx in (Const, Active)
    test_reverse(*, Tret, (x, Tx), y)
end
```

Here we test a rule for a function of an array in batch reverse-mode:

```julia
x = randn(3)
for Tret in (Const, Active), Tx in (Const, BatchDuplicated)
    test_reverse(prod, Tret, (x, Tx))
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
        testset_name = "test_reverse: $f with return activity $ret_activity on $(_string_activity(args))"
    end
    @testset "$testset_name" begin
        # format arguments for autodiff and FiniteDifferences
        activities = map(auto_activity, (f, args...))
        primals = map(x -> x.val, activities)
        # call primal, avoid mutating original arguments
        y = call_with_copy(primals...)
        # generate tangent for output
        if !_any_batch_duplicated(map(typeof, activities)...)
            ȳ = ret_activity <: Const ? zero_tangent(y) : rand_tangent(y)
        else
            batch_size = _batch_size(map(typeof, activities)...)
            ks = ntuple(Symbol ∘ string, batch_size)
            ȳ = ntuple(batch_size) do _
                ret_activity <: Const ? zero_tangent(y) : rand_tangent(y)
            end
        end
        # call finitedifferences, avoid mutating original arguments
        dx_fdm = _fd_reverse(fdm, call_with_kwargs, ȳ, activities, !(ret_activity <: Const))
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
                if !_any_batch_duplicated(map(typeof, activities)...)
                    map_fields_recursive(copyto!, shadow_result, ȳ)
                else
                    for (sr, dy) in zip(shadow_result, ȳ)
                        map_fields_recursive(copyto!, sr, dy)
                    end
                end
            end
            dx_ad = only(reverse(c_act, activities..., tape))
        end
        test_approx(
            y_ad, y, "The return value of the rule and function must agree"; atol, rtol
        )
        @test length(dx_ad) == length(dx_fdm) == length(activities)
        # check all returned derivatives against FiniteDifferences
        for (i, (act_i, dx_ad_i, dx_fdm_i)) in enumerate(zip(activities, dx_ad, dx_fdm))
            target_str = if i == 1
                "active derivative for callable"
            else
                "active derivative for argument $(i - 1)"
            end
            if act_i isa Active
                test_approx(
                    dx_ad_i,
                    dx_fdm_i,
                    "$target_str should agree with finite differences";
                    atol,
                    rtol,
                )
            else
                @test_msg(
                    "returned derivative for argument $(i-1) with activity $act_i must be `nothing`",
                    dx_ad_i === nothing,
                )
                target_str = if i == 1
                    "shadow derivative for callable"
                else
                    "shadow derivative for argument $(i - 1)"
                end
                if act_i isa Duplicated
                    msg_deriv = "$target_str should agree with finite differences"
                    test_approx(act_i.dval, dx_fdm_i, msg_deriv; atol, rtol)
                elseif act_i isa BatchDuplicated
                    @assert length(act_i.dval) == length(dx_fdm_i)
                    for (j, (act_i_j, dx_fdm_i_j)) in enumerate(zip(act_i.dval, dx_fdm_i))
                        msg_deriv = "$target_str for batch index $j should agree with finite differences"
                        test_approx(act_i_j, dx_fdm_i_j, msg_deriv; atol, rtol)
                    end
                end
            end
        end
    end
end
