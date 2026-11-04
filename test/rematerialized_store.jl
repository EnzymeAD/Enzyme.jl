using Enzyme, Test

@noinline function remat_store_value(x, a, b)
    y = 0.0
    for i in eachindex(a, b)
        y += a[i] * exp(-x * b[i])
    end
    return y
end

@noinline function remat_unused_array(x, a, b)
    out = zeros(length(a))
    for i in eachindex(a)
        out[i] = a[i] * remat_store_value(x, a, b)
    end
    return nothing
end

function remat_unused_array_loss(x, a, b)
    remat_unused_array(x, a, b)
    return x * x
end

@testset "Stores in an inactive rematerialized array" begin
    # Initializing the unused array still introduces control flow based on its
    # length. The cache graph must connect values stored into the array to the
    # allocation and its required loads, even through recursive use analysis.
    for n in (0, 1, 3), mode in (ReverseSplitNoPrimal, ReverseSplitWithPrimal)
        a = collect(1.0:n)
        b = copy(a)
        da = ones(n)
        db = fill(2.0, n)
        args = (Active(0.2), Duplicated(a, da), Duplicated(b, db))
        fwd, rev = autodiff_thunk(
            mode, Const{typeof(remat_unused_array)}, Const{Nothing}, typeof.(args)...
        )
        tape, primal, shadow = fwd(Const(remat_unused_array), args...)
        @test primal === nothing
        @test shadow === nothing
        @test rev(Const(remat_unused_array), args..., tape)[1][1] == 0.0
        @test da == ones(n)
        @test db == fill(2.0, n)
        @test a == b == collect(1.0:n)

        res = autodiff(Reverse, remat_unused_array_loss, Active, args...)
        @test res[1][1] ≈ 0.4
        @test da == ones(n)
        @test db == fill(2.0, n)
    end
end
