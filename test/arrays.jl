using Enzyme, Test
using SparseArrays, Statistics


@testset "Array tests" begin

    function arsum(f::Array{T}) where T
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
    function arsum2(f::Array{T}) where T
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
    function arsumsq(f::Array{T}) where T
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

    x  = [2.0,]

    bias = Float32[0.0;;;]
    res = Enzyme.autodiff(Reverse, f, Active, Active(x[1]), Const(bias))
    
    @test bias[1][1] ≈ 0.0
    @test res[1][1] ≈ cos(x[1])
end

# dot product (https://github.com/EnzymeAD/Enzyme.jl/issues/495)
@testset "Dot product" for T in (Float32, Float64)
    xx = rand(T, 10)
    grads = zeros(T, size(xx))
    autodiff(Reverse, (y) -> mapreduce(x -> x*x, +, y), Duplicated(xx, grads))
    @test xx .* 2 == grads

    xx = rand(T, 10)
    grads = zeros(T, size(xx))
    autodiff(Reverse, (x) -> sum(x .* x), Duplicated(xx, grads))
    @test xx .* 2 == grads

    xx = rand(T, 10)
    grads = zeros(T, size(xx))
    autodiff(Reverse, (x) -> x' * x, Duplicated(xx, grads))
    @test xx .* 2 == grads
end

@testset "broadcast" begin
    A = rand(10); B = rand(10); R = similar(A)
    dA = zero(A); dB = zero(B); dR = fill!(similar(R), 1)

    function foo_bc!(R, A, B)
        R .= A .+ B
        return nothing
    end

    autodiff(Reverse, foo_bc!, Const, Duplicated(R, dR), Duplicated(A, dA), Duplicated(B, dB))

    # works since aliasing is "simple"
    autodiff(Reverse, foo_bc!, Const, Duplicated(R, dR), Duplicated(R, dR), Duplicated(B, dB))

    A = rand(10,10); B = rand(10, 10)
    dA = zero(A); dB = zero(B); dR = fill!(similar(A), 1)

    autodiff(Reverse, foo_bc!, Const, Duplicated(A, dR), Duplicated(transpose(A), transpose(dA)), Duplicated(B, dB))
end

@testset "DuplicatedReturn" begin
    moo(x) = fill(x, 10)

    @test_throws ErrorException autodiff(Reverse, moo, Active(2.1))
    fo, = autodiff(Forward, moo, Duplicated(2.1, 1.0))
    for i in 1:10
        @test 1.0 ≈ fo[i]
    end

    @test_throws ErrorException autodiff(Forward, x->x, Active(2.1))
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

@testset "Arrays are double pointers" begin
    @noinline function func_scalar(X)
        return X
    end

    function timsteploop_scalar(FH1)
        G = Float64[FH1]
        k1 = @inbounds func_scalar(G[1])
        return k1
    end
    @test Enzyme.autodiff(Reverse, timsteploop_scalar, Active(2.0))[1][1] ≈ 1.0
    @test Enzyme.autodiff(Forward, timsteploop_scalar, Duplicated(2.0, 1.0))[1] ≈ 1.0

    @noinline function func(X)
        return @inbounds X[1]
    end
    function timsteploop(FH1)
        G = Float64[FH1]
        k1 = func(G)
        return k1
    end
    @test Enzyme.autodiff(Reverse, timsteploop, Active(2.0))[1][1] ≈ 1.0
    @test Enzyme.autodiff(Forward, timsteploop, Duplicated(2.0, 1.0))[1] ≈ 1.0
end

function bc0_test_function(ps)
    z = view(ps, 26:30)
    C = Matrix{Float64}(undef, 5, 1)
    C .= z
    return C[1]
end

@noinline function bc1_bcs2(x, y)
    x != y && error(2)
    return x
end

@noinline function bc1_affine_normalize(x::AbstractArray)
    _axes = bc1_bcs2(axes(x), axes(x))
    dest = similar(Array{Float32}, _axes)
    bc = convert(Broadcast.Broadcasted{Nothing}, Broadcast.instantiate(Base.broadcasted(+, x, x)))
    copyto!(dest, bc)
    return x
end

function bc1_loss_function(x)
    return bc1_affine_normalize(x)[1]
end

function bc2_affine_normalize(::typeof(identity), x::AbstractArray, xmean, xvar,
    scale::AbstractArray, bias::AbstractArray, epsilon::Real)
    _scale = @. scale / sqrt(xvar + epsilon)
    _bias = @. bias - xmean * _scale
    return @. x * _scale + _bias
end

function bc2_loss_function(x, scale, bias)
    x_ = reshape(x, 6, 6, 3, 2, 2)
    scale_ = reshape(scale, 1, 1, 3, 2, 1)
    bias_ = reshape(bias, 1, 1, 3, 2, 1)

    xmean = mean(x_, dims=(1, 2, 5))
    xvar = var(x_, corrected=false, mean=xmean, dims=(1, 2, 5))

    return sum(abs2, bc2_affine_normalize(identity, x_, xmean, xvar, scale_, bias_, 1e-5))
end

#FUCK: I'm *still* getting segfaults here????
#=
@testset "Broadcast noalias" begin
    x = ones(30)
    autodiff(Reverse, bc0_test_function, Active, Const(x))
    
    x = rand(Float32, 2, 3)
    Enzyme.autodiff(Reverse, bc1_loss_function, Duplicated(x, zero(x)))

    x = rand(Float32, 6, 6, 6, 2)
    sc = rand(Float32, 6)
    bi = rand(Float32, 6)
    Enzyme.autodiff(Reverse, bc2_loss_function, Active, Duplicated(x, Enzyme.make_zero(x)),
        Duplicated(sc, Enzyme.make_zero(sc)), Duplicated(bi, Enzyme.make_zero(bi)))
end
=#

@testset "BLAS" begin
    x = [2.0, 3.0]
    dx = [0.2,0.3]
    y = [5.0, 7.0]
    dy = [0.5,0.7]
    Enzyme.autodiff(Reverse, (x,y)->x' * y, Duplicated(x, dx), Duplicated(y, dy))
    @show x, dx, y, dy
    @test dx ≈ [5.2, 7.3]
    @test dy ≈ [2.5, 3.7]

    f_exc(x) = sum(x*x)
    y = [[1.0, 2.0] [3.0,4.0]]
    f_x = zero.(y)
    Enzyme.autodiff(Reverse, f_exc, Duplicated(y, f_x))
    @test f_x ≈ [7.0 9.0; 11.0 13.0]
end

@testset "Array push" begin

    function pusher(x, y)
        push!(x, y)
        x[1] + x[2]
    end

    x  = [2.3]
    dx = [0.0]
    @test 1.0 ≈ first(Enzyme.autodiff(Reverse, pusher, Duplicated(x, dx), Active(2.0)))[2]
    @test x ≈ [2.3, 2.0]
    @test dx ≈ [1.0]

    function double_push(x)
        a = [0.5]
        push!(a, 1.0)
        push!(a, 1.0)
        return x
    end
    y, = Enzyme.autodiff(Reverse, double_push,Active(1.0))[1]
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
        Reverse,
        aloss,
        Active,
        Active(1.0),
        Duplicated(arr, darr)
       )[1][1]
    @test y == 1.0
end


function sparse_eval(x::Vector{Float64})
    A = sparsevec([1, 1, 2, 3], [2.0*x[2]^3.0, 1.0-x[1], 2.0+x[3], -1.0])
    B = sparsevec([1, 1, 2, 3], [2.0*x[2], 1.0-x[1], 2.0+x[3], -1.0])
    C = A + B
    return A[1]
end

@testset "Type Unstable SparseArrays" begin
    x = [3.1, 2.7, 8.2]
    dx = [0.0, 0.0, 0.0]

    autodiff(Reverse, sparse_eval, Duplicated(x, dx))
    
    @test x ≈ [3.1, 2.7, 8.2]
    @test dx ≈ [-1.0, 43.74, 0]
end

@testset "Reshape" begin

	function rs(x)
		y = reshape(x, 2, 2)
		y[1,1] *= y[1, 2]
		y[2, 2] *= y[2, 1]
		nothing
	end

    data = Float64[1.,2.,3.,4.]
	ddata = ones(4)

	autodiff(Reverse, rs, Duplicated(data, ddata))
	@test ddata ≈ [3.0, 5.0, 2.0, 2.0]

    data = Float64[1.,2.,3.,4.]
	ddata = ones(4)
	autodiff(Forward, rs, Duplicated(data, ddata))
	@test ddata ≈ [4.0, 1.0, 1.0, 6.0]
end

function absset(out, x)
    @inbounds out[1] = (x,)
    return nothing
end

@testset "Abstract Array element type" begin
    out = Tuple{Any}[(9.7,)]
    dout = Tuple{Any}[(4.3,)]

    autodiff(Enzyme.Forward, absset, Duplicated(out, dout), Duplicated(3.1, 2.4))
    @test dout[1][1] ≈ 2.4
end


@testset "hvcat_fill" begin
    ar = Matrix{Float64}(undef, 2, 3)
    dar = [1.0 2.0 3.0; 4.0 5.0 6.0]

    res = first(Enzyme.autodiff(Reverse, Base.hvcat_fill!, Const, Duplicated(ar, dar), Active((1, 2.2, 3, 4.4, 5, 6.6))))

    @test res[2][1] == 0
    @test res[2][2] ≈ 2.0
    @test res[2][3] ≈ 0
    @test res[2][4] ≈ 4.0
    @test res[2][5] ≈ 0
    @test res[2][6] ≈ 6.0
end

