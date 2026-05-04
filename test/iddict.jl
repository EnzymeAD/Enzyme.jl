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

    J_fwd = Enzyme.jacobian(Forward, compute_iddict_jacobian_input, d; shadows=((d_s1, d_s2),))[1]
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




