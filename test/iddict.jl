using Enzyme, Test, GPUArraysCore

# Test for https://github.com/EnzymeAD/Enzyme.jl/issues/3042
# @allowscalar uses delete!(task_local_storage(), :ScalarIndexing) which calls
# jl_eqtable_pop — Enzyme needs a rule for this even on CPU.
@testset "allowscalar in autodiff (#3042)" begin
    using GPUArraysCore: @allowscalar

    function compute_allowscalar!(arr::Vector{Float64})
        arr .= 1 ./ (arr .^ 2 .+ 1)
        @allowscalar arr[1] = 0.0
        return nothing
    end

    ev = fill(0.5, 4)
    dev = ones(Float64, 4)
    # Should not error — this exercises the jl_eqtable_pop rule (Reverse mode)
    autodiff(set_runtime_activity(Reverse), compute_allowscalar!, Const, Duplicated(ev, dev))

    # arr[1] was set to 0.0, so d/darr[1] of the overall computation
    # The @allowscalar sets arr[1]=0, overwriting whatever 1/(x^2+1) gave,
    # so dout/darr[1] = 0 (the set clobbers the computation).
    @test dev[1] ≈ 0.0
    # arr[2:4] = 1/(0.5^2+1) = 0.8, so d(1/(x^2+1))/dx at x=0.5 = -2x/(x^2+1)^2
    # = -1.0/1.5625 ≈ -0.64, but gradient of sum reduction is 1 * d/dx, so dev[2:4] ≈ -0.64
    @test all(isfinite, dev[2:end])

    # Forward mode test
    ev_fwd = fill(0.5, 4)
    dev_fwd = ones(Float64, 4)
    dev_fwd[1] = 1.0 # Set tangent to 1.0
    autodiff(set_runtime_activity(Forward), compute_allowscalar!, Const, Duplicated(ev_fwd, dev_fwd))
    @test dev_fwd[1] == 0.0
    @test dev_fwd[2] ≈ -0.64

    # Batch Reverse mode test
    ev_batch = fill(0.5, 4)
    dev_batch_1 = ones(Float64, 4)
    dev_batch_2 = ones(Float64, 4)
    autodiff(set_runtime_activity(Reverse), compute_allowscalar!, Const, BatchDuplicated(ev_batch, (dev_batch_1, dev_batch_2)))
    @test dev_batch_1[1] ≈ 0.0
    @test dev_batch_2[1] ≈ 0.0
    @test all(isfinite, dev_batch_1[2:end])
    @test all(isfinite, dev_batch_2[2:end])
end

@testset "Active data in IdDict" begin
    function compute_iddict_active_float(x::Float64)
        d = IdDict{Symbol, Float64}()
        d[:val] = x * 2.0
        v = d[:val]
        return v * 3.0
    end

    # Reverse mode
    grad = autodiff(set_runtime_activity(Reverse), compute_iddict_active_float, Active, Active(1.5))[1][1]
    @test grad ≈ 6.0

    # Forward mode
    fwd_res = autodiff(set_runtime_activity(Forward), compute_iddict_active_float, Duplicated, Duplicated(1.5, 1.0))[1]
    @test fwd_res ≈ 6.0

    function compute_iddict_active_array(x::Vector{Float64})
        d = IdDict{Symbol, Vector{Float64}}()
        d[:val] = x
        v = d[:val]
        v[1] = v[1] * 2.0
        return nothing
    end

    x = [1.5]
    dx = [1.0]
    autodiff(set_runtime_activity(Reverse), compute_iddict_active_array, Const, Duplicated(x, dx))
    @test dx[1] ≈ 2.0

    x_fwd = [1.5]
    dx_fwd = [1.0]
    autodiff(set_runtime_activity(Forward), compute_iddict_active_array, Const, Duplicated(x_fwd, dx_fwd))
    @test dx_fwd[1] ≈ 2.0
end

@testset "Active IdDict" begin
    function compute_active_iddict_arg(d::IdDict{Symbol, Vector{Float64}}, x::Vector{Float64})
        d[:val] = x
        v = d[:val]
        v[1] = v[1] * 2.0
        return nothing
    end

    d = IdDict{Symbol, Vector{Float64}}()
    d_shadow = IdDict{Symbol, Vector{Float64}}()
    x = [1.5]
    dx = [1.0]
    autodiff(set_runtime_activity(Reverse), compute_active_iddict_arg, Duplicated(d, d_shadow), Duplicated(x, dx))
    @test dx[1] ≈ 2.0

    d_fwd = IdDict{Symbol, Vector{Float64}}()
    d_shadow_fwd = IdDict{Symbol, Vector{Float64}}()
    x_fwd = [1.5]
    dx_fwd = [1.0]
    autodiff(set_runtime_activity(Forward), compute_active_iddict_arg, Duplicated(d_fwd, d_shadow_fwd), Duplicated(x_fwd, dx_fwd))
    @test dx_fwd[1] ≈ 2.0

    d_batch = IdDict{Symbol, Vector{Float64}}()
    d_batch_1 = IdDict{Symbol, Vector{Float64}}()
    d_batch_2 = IdDict{Symbol, Vector{Float64}}()
    x_batch = [1.5]
    dx_batch_1 = [1.0]
    dx_batch_2 = [2.0]
    autodiff(set_runtime_activity(Forward), compute_active_iddict_arg, BatchDuplicated(d_batch, (d_batch_1, d_batch_2)), BatchDuplicated(x_batch, (dx_batch_1, dx_batch_2)))
    @test dx_batch_1[1] ≈ 2.0
    @test dx_batch_2[1] ≈ 4.0

    d_batch_rev = IdDict{Symbol, Vector{Float64}}()
    d_batch_rev_1 = IdDict{Symbol, Vector{Float64}}()
    d_batch_rev_2 = IdDict{Symbol, Vector{Float64}}()
    x_batch_rev = [1.5]
    dx_batch_rev_1 = [1.0]
    dx_batch_rev_2 = [2.0]
    autodiff(set_runtime_activity(Reverse), compute_active_iddict_arg, BatchDuplicated(d_batch_rev, (d_batch_rev_1, d_batch_rev_2)), BatchDuplicated(x_batch_rev, (dx_batch_rev_1, dx_batch_rev_2)))
    @test dx_batch_rev_1[1] ≈ 2.0
    @test dx_batch_rev_2[1] ≈ 4.0
end

@testset "make_zero and remake_zero for IdDict" begin
    # empty
    d1 = IdDict{Symbol, Vector{Float64}}()
    d1_zero = Enzyme.make_zero(d1)
    @test typeof(d1_zero) == typeof(d1)
    @test length(d1_zero) == 0

    # populated
    d2 = IdDict{Symbol, Vector{Float64}}()
    d2[:a] = [1.0, 2.0]
    d2_zero = Enzyme.make_zero(d2)
    @test typeof(d2_zero) == typeof(d2)
    @test length(d2_zero) == 1
    @test haskey(d2_zero, :a)
    @test d2_zero[:a] == [0.0, 0.0]

    # make_zero!
    d3_shadow = IdDict{Symbol, Vector{Float64}}()
    d3_shadow[:a] = [3.0, 4.0]
    Enzyme.make_zero!(d3_shadow)
    @test length(d3_shadow) == 1
    @test d3_shadow[:a] == [0.0, 0.0]

    # remake_zero!
    d4_shadow = IdDict{Symbol, Vector{Float64}}()
    d4_shadow[:a] = [5.0, 6.0]
    Enzyme.remake_zero!(d4_shadow)
    @test length(d4_shadow) == 1
    @test d4_shadow[:a] == [0.0, 0.0]
end

@testset "Jacobian with IdDict" begin
    function compute_iddict_jacobian(x::Vector{Float64})
        d = IdDict{Symbol, Vector{Float64}}()
        d[:val] = x
        v = d[:val]
        return [v[1] * 2.0, v[2] * 3.0]
    end

    x = [1.5, 2.5]

    J_fwd = Enzyme.jacobian(Forward, compute_iddict_jacobian, x)[1]
    @test J_fwd ≈ [2.0 0.0; 0.0 3.0]

    J_rev = Enzyme.jacobian(Reverse, compute_iddict_jacobian, x)[1]
    @test J_rev ≈ [2.0 0.0; 0.0 3.0]
end

@testset "Jacobian with IdDict as input" begin
    function compute_iddict_jacobian_input(d::IdDict{Symbol, Vector{Float64}})
        v = d[:val]
        return [v[1] * 2.0, v[2] * 3.0]
    end

    d = IdDict{Symbol, Vector{Float64}}()
    d[:val] = [1.5, 2.5]

    d_s1 = IdDict{Symbol, Vector{Float64}}()
    d_s1[:val] = [1.0, 0.0]
    d_s2 = IdDict{Symbol, Vector{Float64}}()
    d_s2[:val] = [0.0, 1.0]

    J_fwd = Enzyme.jacobian(Forward, compute_iddict_jacobian_input, d; shadows = ((d_s1, d_s2),))[1]
    @test J_fwd == ([2.0, 0.0], [0.0, 3.0])

    J_rev = Enzyme.jacobian(Reverse, compute_iddict_jacobian_input, d)[1]
    @test length(J_rev) == 2
    @test J_rev[1][:val] ≈ [2.0, 0.0]
    @test J_rev[2][:val] ≈ [0.0, 3.0]
end

@testset "Heterogeneous IdDict" begin
    function compute_hetero_iddict(x::Float64, y::Vector{Float64})
        d = IdDict{Symbol, Any}()
        d[:val_float] = x
        d[:val_array] = y
        d[:val_int] = 42

        v1 = d[:val_float]::Float64
        v2 = d[:val_array]::Vector{Float64}
        v3 = d[:val_int]::Int

        v2[1] = v2[1] * v1 + v3
        return v1 * 3.0
    end

    x = 1.5
    y = [2.0]
    dy = [1.0]
    grad_x = autodiff(set_runtime_activity(Reverse), compute_hetero_iddict, Active, Active(x), Duplicated(y, dy))[1][1]

    # Primal result of v1 * 3.0 is 1.5 * 3.0 = 4.5
    # y[1] becomes 2.0 * 1.5 + 42 = 45.0
    # dy is [1.0], so we accumulate dy[1] * d(y[1])/dx = 1.0 * v2_old = 1.0 * 2.0 = 2.0
    # d(v1 * 3.0)/dx = 3.0
    # Total grad_x = 3.0 + 2.0 = 5.0.
    @test grad_x ≈ 5.0
    # Total grad_y[1] = dy[1] * d(y[1])/dy = 1.0 * v1 = 1.5.
    @test dy[1] ≈ 1.5

    x_fwd = 1.5
    y_fwd = [2.0]
    dy_fwd = [1.0]
    fwd_res = autodiff(set_runtime_activity(Forward), compute_hetero_iddict, Duplicated, Duplicated(x_fwd, 1.0), Duplicated(y_fwd, dy_fwd))[1]

    # Forward mode returns the derivative of the return value.
    # Return value is v1 * 3.0. Derivative is 3.0 * dx = 3.0.
    @test fwd_res ≈ 3.0
    # Array y is modified to v2[1] * v1 + 42.
    # Derivative of y_fwd[1] = dy_fwd[1] * x + y_fwd[1] * dx = 1.0 * 1.5 + 2.0 * 1.0 = 3.5.
    @test dy_fwd[1] ≈ 3.5
end

@testset "pop! return value used in computation" begin
    function f_pop_scalar(x::Float64)
        d = IdDict{Symbol, Any}()
        d[:val] = x
        v = pop!(d, :val)::Float64
        return v * 2.0
    end

    grad = autodiff(set_runtime_activity(Reverse), f_pop_scalar, Active, Active(1.5))[1][1]
    @test grad ≈ 2.0

    fwd = autodiff(set_runtime_activity(Forward), f_pop_scalar, Duplicated, Duplicated(1.5, 1.0))[1]
    @test fwd ≈ 2.0

    # Non-trivial stored expression: shadow of pop! must match what put! stored,
    # not the default's shadow.
    function f_pop_computed(x::Float64, y::Float64)
        d = IdDict{Symbol, Any}()
        d[:val] = x * y
        v = pop!(d, :val)::Float64
        return v * 3.0
    end

    grads = autodiff(set_runtime_activity(Reverse), f_pop_computed, Active, Active(1.5), Active(2.0))[1]
    @test grads[1] ≈ 6.0  # 3y
    @test grads[2] ≈ 4.5  # 3x

    fwd_x = autodiff(set_runtime_activity(Forward), f_pop_computed, Duplicated, Duplicated(1.5, 1.0), Duplicated(2.0, 0.0))[1]
    @test fwd_x ≈ 6.0

    fwd_y = autodiff(set_runtime_activity(Forward), f_pop_computed, Duplicated, Duplicated(1.5, 0.0), Duplicated(2.0, 1.0))[1]
    @test fwd_y ≈ 4.5

    # Key present: default shadow must not be returned instead of the stored shadow.
    function f_pop_with_default(x::Float64)
        d = IdDict{Symbol, Any}()
        d[:val] = x
        v = pop!(d, :val, 0.0)::Float64
        return v * 5.0
    end

    grad_def = autodiff(set_runtime_activity(Reverse), f_pop_with_default, Active, Active(1.5))[1][1]
    @test grad_def ≈ 5.0

    fwd_def = autodiff(set_runtime_activity(Forward), f_pop_with_default, Duplicated, Duplicated(1.5, 1.0))[1]
    @test fwd_def ≈ 5.0
end

@testset "pop! in split reverse mode" begin
    function f_pop_split(d::IdDict{Symbol, Vector{Float64}})
        v = pop!(d, :val)::Vector{Float64}
        return v[1] * 2.0
    end

    fwd, rev = autodiff_thunk(
        ReverseSplitNoPrimal,
        Const{typeof(f_pop_split)},
        Active{Float64},
        Duplicated{IdDict{Symbol, Vector{Float64}}},
    )

    d = IdDict{Symbol, Vector{Float64}}()
    d[:val] = [1.5]
    val_shadow = [0.0]
    d_shadow = IdDict{Symbol, Vector{Float64}}()
    d_shadow[:val] = val_shadow

    tape, = fwd(Const(f_pop_split), Duplicated(d, d_shadow))
    rev(Const(f_pop_split), Duplicated(d, d_shadow), 1.0, tape)
    @test val_shadow[1] ≈ 2.0

    # Mutation between passes: insert a wrong vector into the now-empty shadow
    # dict to confirm the reverse pass uses the jlvalue captured during the
    # forward pass, not the current dict contents.
    d2 = IdDict{Symbol, Vector{Float64}}()
    d2[:val] = [1.5]
    val_shadow2 = [0.0]
    d_shadow2 = IdDict{Symbol, Vector{Float64}}()
    d_shadow2[:val] = val_shadow2

    tape2, = fwd(Const(f_pop_split), Duplicated(d2, d_shadow2))
    d_shadow2[:val] = [99.0]
    rev(Const(f_pop_split), Duplicated(d2, d_shadow2), 1.0, tape2)

    @test val_shadow2[1] ≈ 2.0
    @test d_shadow2[:val][1] ≈ 99.0
end

@testset "pop! missing key returns active default" begin
    # Key absent: pop! returns the default; gradient must flow through it.
    function f_pop_missing(x::Float64)
        d = IdDict{Symbol, Any}()
        v = pop!(d, :val, x)::Float64
        return v * 3.0
    end

    grad = autodiff(set_runtime_activity(Reverse), f_pop_missing, Active, Active(1.5))[1][1]
    @test grad ≈ 3.0

    fwd = autodiff(set_runtime_activity(Forward), f_pop_missing, Duplicated, Duplicated(1.5, 1.0))[1]
    @test fwd ≈ 3.0
end

@testset "pop! after overwrite uses second shadow" begin
    # Two puts to the same key: the second overwrites the first in both the
    # primal and shadow tables. pop! must return y's shadow, not x's.
    function f_overwrite_pop(x::Float64, y::Float64)
        d = IdDict{Symbol, Any}()
        d[:val] = x
        d[:val] = y
        v = pop!(d, :val)::Float64
        return v * 2.0
    end

    grads = autodiff(set_runtime_activity(Reverse), f_overwrite_pop, Active, Active(1.5), Active(2.0))[1]
    @test grads[1] ≈ 0.0
    @test grads[2] ≈ 2.0

    fwd_x = autodiff(set_runtime_activity(Forward), f_overwrite_pop, Duplicated, Duplicated(1.5, 1.0), Duplicated(2.0, 0.0))[1]
    @test fwd_x ≈ 0.0

    fwd_y = autodiff(set_runtime_activity(Forward), f_overwrite_pop, Duplicated, Duplicated(1.5, 0.0), Duplicated(2.0, 1.0))[1]
    @test fwd_y ≈ 2.0
end

@testset "double pop same key" begin
    # First pop gets the stored value; second pop misses and returns the default.
    # Their shadows must not bleed into each other.
    function f_double_pop(x::Float64)
        d = IdDict{Symbol, Any}()
        d[:val] = x
        v1 = pop!(d, :val, 0.0)::Float64  # key present
        v2 = pop!(d, :val, 1.0)::Float64  # key absent, gets inactive default
        return v1 * 2.0 + v2 * 3.0        # gradient wrt x: 2, not 2+3
    end

    grad = autodiff(set_runtime_activity(Reverse), f_double_pop, Active, Active(1.5))[1][1]
    @test grad ≈ 2.0

    fwd = autodiff(set_runtime_activity(Forward), f_double_pop, Duplicated, Duplicated(1.5, 1.0))[1]
    @test fwd ≈ 2.0
end

@testset "pop! two distinct keys" begin
    # Each key has its own shadow; popping them must not mix gradients.
    function f_two_keys(x::Float64, y::Float64)
        d = IdDict{Symbol, Any}()
        d[:x] = x
        d[:y] = y
        vx = pop!(d, :x)::Float64
        vy = pop!(d, :y)::Float64
        return vx * 2.0 + vy * 3.0
    end

    grads = autodiff(set_runtime_activity(Reverse), f_two_keys, Active, Active(1.5), Active(2.0))[1]
    @test grads[1] ≈ 2.0
    @test grads[2] ≈ 3.0

    fwd_x = autodiff(set_runtime_activity(Forward), f_two_keys, Duplicated, Duplicated(1.5, 1.0), Duplicated(2.0, 0.0))[1]
    @test fwd_x ≈ 2.0

    fwd_y = autodiff(set_runtime_activity(Forward), f_two_keys, Duplicated, Duplicated(1.5, 0.0), Duplicated(2.0, 1.0))[1]
    @test fwd_y ≈ 3.0
end

@testset "pop! then put back" begin
    # Pop a value, transform it, put it back, then retrieve it.
    # The shadow table must reflect the re-inserted (transformed) shadow.
    function f_pop_put_get(x::Float64)
        d = IdDict{Symbol, Any}()
        d[:val] = x
        v = pop!(d, :val)::Float64
        d[:val] = v * 2.0
        w = d[:val]::Float64
        return w * 3.0  # = 6x, gradient = 6
    end

    grad = autodiff(set_runtime_activity(Reverse), f_pop_put_get, Active, Active(1.5))[1][1]
    @test grad ≈ 6.0

    fwd = autodiff(set_runtime_activity(Forward), f_pop_put_get, Duplicated, Duplicated(1.5, 1.0))[1]
    @test fwd ≈ 6.0
end
