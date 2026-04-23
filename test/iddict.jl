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
end
