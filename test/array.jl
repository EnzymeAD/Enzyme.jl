using Enzyme, Test

function ptrcopy(B, A)
@static if VERSION < v"1.11"
	Base.unsafe_copyto!(B, 1, A, 1, 2)
else
	Base.unsafe_copyto!(B.ref, A.ref, 2)
end
	nothing
end

@testset "Array of Pointer Copy" begin
	A = [[2.7, 3.1], [4.7, 5.6]]
	dA1 = [1.1, 4.3]
	dA2 = [17.2, 0.26]
	dA = [dA1, dA2]

	B = [[2.0, 4.0], [7.0, 11.0]]
	dB = Enzyme.make_zero(B)

	Enzyme.autodiff(set_runtime_activity(Reverse), ptrcopy, Duplicated(B, dB), Duplicated(A, dA))

	@test dB[1] === dA1
	@test dB[2] === dA2
end

function unsafe_wrap_test(a, i, x)
	GC.@preserve a begin
		 ptr = pointer(a)
		 b = Base.unsafe_wrap(Array, ptr, length(a))
		 b[i] = x
	end
	a[i]
end


@testset "Array tests" begin

    function arsum(f::Array{T}) where {T}
        g = zero(T)
        for elem in f
            g += elem
        end
        return g
    end

    inp = Float64[1.0, 2.0]
    dinp = Float64[0.0, 0.0]
    autodiff(Reverse, arsum, Active, Duplicated(inp, dinp))
    @test inp ≈ Float64[1.0, 2.0]
    @test dinp ≈ Float64[1.0, 1.0]

    @test autodiff(Forward, arsum, Duplicated(inp, dinp))[1] ≈ 2.0

    function f1(m)
        s = 0.0
        for (i, col) in enumerate(eachcol(m))
            s += i * sum(col)
        end
        return s
    end

    m = Float64[1 2 3; 4 5 6; 7 8 9]
    dm = zero(m)
    autodiff(Reverse, f1, Active, Duplicated(m, dm))
    @test dm == Float64[1 2 3; 1 2 3; 1 2 3]

    function f2(m)
        s = 0.0
        for (i, col) in enumerate(eachrow(m))
            s += i * sum(col)
        end
        return s
    end

    dm = zero(m)
    autodiff(Reverse, f2, Active, Duplicated(m, dm))
    @test dm == Float64[1 1 1; 2 2 2; 3 3 3]

    function my_conv_3(x, w)
        y = zeros(Float64, 2, 3, 4, 5)
        for hi in axes(y, 3)
            y[1] += w * x
        end
        return y
    end
    loss3(x, w) = sum(my_conv_3(x, w))
    x = 2.0
    w = 3.0
    dx, dw = Enzyme.autodiff(Reverse, loss3, Active(x), Active(w))[1]
    @test dw ≈ 4 * x
    @test dx ≈ 4 * w
end

@testset "Advanced array tests" begin
    function arsum2(f::Array{T}) where {T}
        return sum(f)
    end
    inp = Float64[1.0, 2.0]
    dinp = Float64[0.0, 0.0]
    autodiff(Reverse, arsum2, Active, Duplicated(inp, dinp))
    @test inp ≈ Float64[1.0, 2.0]
    @test dinp ≈ Float64[1.0, 1.0]

    @test autodiff(Forward, arsum2, Duplicated(inp, dinp))[1] ≈ 2.0
end

@testset "Advanced array tests sq" begin
    function arsumsq(f::Array{T}) where {T}
        return sum(f) * sum(f)
    end
    inp = Float64[1.0, 2.0]
    dinp = Float64[0.0, 0.0]
    autodiff(Reverse, arsumsq, Active, Duplicated(inp, dinp))
    @test inp ≈ Float64[1.0, 2.0]
    @test dinp ≈ Float64[6.0, 6.0]
end

@testset "Reshape Activity" begin
    function f(x, bias)
        mout = x + @inbounds vec(bias)[1]
        sin(mout)
    end

    x = [2.0]

    bias = Float32[0.0;;;]
    res = Enzyme.autodiff(Reverse, f, Active, Active(x[1]), Const(bias))

    @test bias[1][1] ≈ 0.0
    @test res[1][1] ≈ cos(x[1])
end

@testset "Reshape" begin

    function rs(x)
        y = reshape(x, 2, 2)
        y[1, 1] *= y[1, 2]
        y[2, 2] *= y[2, 1]
        nothing
    end

    data = Float64[1.0, 2.0, 3.0, 4.0]
    ddata = ones(4)

    autodiff(Reverse, rs, Duplicated(data, ddata))
    @test ddata ≈ [3.0, 5.0, 2.0, 2.0]

    data = Float64[1.0, 2.0, 3.0, 4.0]
    ddata = ones(4)
    autodiff(Forward, rs, Duplicated(data, ddata))
    @test ddata ≈ [4.0, 1.0, 1.0, 6.0]
end

@testset "Array Copy" begin
    F = [2.0, 3.0]

    dF = [0.0, 0.0]

    function copytest(F)
        F2 = copy(F)
        @inbounds F[1] = 1.234
        @inbounds F[2] = 5.678
        @inbounds F2[1] * F2[2]
    end
    autodiff(Reverse, copytest, Duplicated(F, dF))
    @test F ≈ [1.234, 5.678]
    @test dF ≈ [3.0, 2.0]

    @test 31.0 ≈ autodiff(Forward, copytest, Duplicated([2.0, 3.0], [7.0, 5.0]))[1]

    function sh(x)
        Base.sizehint!(x, length(x))
        nothing
    end

    autodiff(Reverse, sh, Duplicated([1.0], [0.0]))
end


for RTA in (false, true)
@testset "Array push runtime activity=$RTA" begin

    function pusher(x, y)
        push!(x, y)
        x[1] + x[2]
    end

    x = [2.3]
    dx = [0.0]
    rf = @static if VERSION < v"1.11-"
        nothing
    else
        dx.ref.mem
    end
    @test 1.0 ≈ first(Enzyme.autodiff(set_runtime_activity(Reverse, RTA), pusher, Duplicated(x, dx), Active(2.0)))[2]
    @static if VERSION < v"1.11-"
        @test dx ≈ [1.0]
    else
        @test dx ≈ [0.0, 0.0]
        @test rf ≈ [1.0]
    end
    @test x ≈ [2.3, 2.0]

    function double_push(x)
        a = [0.5]
        push!(a, 1.0)
        push!(a, 1.0)
        return x
    end
    y, = Enzyme.autodiff(set_runtime_activity(Reverse, RTA), double_push, Active(1.0))[1]
    @test y == 1.0

    function aloss(a, arr)
        for i in 1:2500
            push!(arr, a)
        end
        return @inbounds arr[2500]
    end
    arr = Float64[]
    darr = Float64[]

    y = autodiff(
        set_runtime_activity(Reverse, RTA),
        aloss,
        Active,
        Active(1.0),
        Duplicated(arr, darr)
    )[1][1]
    @test y == 1.0
    @test arr ≈ ones(2500)

    arr = Float64[]

    y = autodiff(
        set_runtime_activity(Reverse, RTA),
        aloss,
        Active,
        Active(1.0),
        Const(arr)
    )[1][1]
    @test y == 0.0
    @test arr ≈ ones(2500)

    if RTA
        arr = Float64[]
        y = autodiff(
            set_runtime_activity(Reverse),
            aloss,
            Active,
            Active(1.0),
            Duplicated(arr, arr)
        )[1][1]
        @test y == 0.0
        @test arr ≈ ones(2500)
    end

end
end

@testset "Unsafe wrap" begin
   autodiff(Forward, unsafe_wrap_test,  Duplicated(zeros(1), zeros(1)), Const(1), Duplicated(1.0, 2.0))

	# TODO test for batch and reverse
end
