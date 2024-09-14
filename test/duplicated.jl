using Enzyme, Test

@noinline function prt_sret(A)
    A[1] *= 2
    return (A, A[2])
end

@noinline function sretf(A2, x, c)
    x[3] = c * A2[3]
end

@noinline function batchdecaysret0(x, A, b)
    A2, c = prt_sret(A)
    sretf(A2, x, c)
    return nothing
end

function batchdecaysret(x, A, b)
    batchdecaysret0(x, A, b)
    A[2] = 0
    return nothing
end

@testset "Batch Reverse sret fix" begin
    Enzyme.autodiff(Reverse, batchdecaysret,
                    BatchDuplicated(ones(3), (ones(3), ones(3))),
                    BatchDuplicated(ones(3), (ones(3), ones(3))),
                    BatchDuplicated(ones(3), (ones(3), ones(3))))
end

struct MyClosure{A}
    a::A
end

function (mc::MyClosure)(x)
    # computes x^2 using internal storage
    mc.a[1] = x
    return mc.a[1]^2
end

@testset "Batch Closure" begin
    g = MyClosure([0.0])
    g_and_dgs = BatchDuplicated(g, (make_zero(g), make_zero(g)))
    x_and_dxs = BatchDuplicated(3.0, (5.0, 7.0))
    autodiff(Forward, g_and_dgs, BatchDuplicated, x_and_dxs)  # error
end

# @testset "Split Tape" begin
#     f(x) = x[1] * x[1]

#     thunk_split = Enzyme.Compiler.thunk(f, Tuple{Duplicated{Array{Float64,1}}}, Val(Enzyme.API.DEM_ReverseModeGradient))
#     @test thunk_split.primal !== C_NULL
#     @test thunk_split.primal !== thunk_split.adjoint
# end

@testset "Duplicated" begin
    x = Ref(1.0)
    y = Ref(2.0)

    ∇x = Ref(0.0)
    ∇y = Ref(0.0)

    autodiff(Reverse, (a,b)->a[]*b[], Active, Duplicated(x, ∇x), Duplicated(y, ∇y))

    @test ∇y[] == 1.0
    @test ∇x[] == 2.0
end

@testset "Batch Generics" begin
    function mul2ip(y)
        y[1] *= 2
        return nothing
    end

    function fwdlatestfooip(y)
        Base.invokelatest(mul2ip, y)
    end

    x = [1.0, 2.0, 3.0]
    dx = [1.0, 1.0, 1.0]
    dx2 = [10.0, 20.0, 30.0]

    res = Enzyme.autodiff(Forward, fwdlatestfooip, Const, BatchDuplicated(x, (dx, dx2)))
    @test 2.0 ≈ dx[1]
    @test 20.0 ≈ dx2[1]

    function mul2(y)
        return y[1] * 2
    end

    function fwdlatestfoo(y)
        Base.invokelatest(mul2, y)
    end

    x = [1.0, 2.0, 3.0]
    dx = [1.0, 1.0, 1.0]
    dx2 = [10.0, 20.0, 30.0]

    res = Enzyme.autodiff(Forward, fwdlatestfoo, BatchDuplicated, BatchDuplicated(x, (dx, dx2)))

    @test 2.0 ≈ res[1][1]
    @test 2.0 ≈ res[2][1]
    @test 20.0 ≈ res[2][2]

    res = Enzyme.autodiff(Forward, fwdlatestfoo, BatchDuplicatedNoNeed, BatchDuplicated(x, (dx, dx2)))
    @test 2.0 ≈ res[1][1]
    @test 20.0 ≈ res[1][2]


    function revfoo(out, x)
        out[] = x*x
        nothing
    end

    out = Ref(0.0)
    dout = Ref(1.0)
    dout2 = Ref(10.0)

    res = Enzyme.autodiff(Reverse, revfoo, BatchDuplicated(out, (dout, dout2)), Active(2.0))[1][2]
    @test 4.0 ≈ res[1]
    @test 40.0 ≈ res[2]
    @test 0.0 ≈ dout[]
    @test 0.0 ≈ dout2[]

    out = Ref(0.0)
    dout = Ref(1.0)
    dout2 = Ref(10.0)

    function rev_lq(y)
        return y * y
    end
    function revfoo2(out, x)
        out[] = Base.invokelatest(rev_lq, x)::Float64
        nothing
    end
    res = Enzyme.autodiff(Reverse, revfoo2, BatchDuplicated(out, (dout, dout2)), Active(2.0))[1][2]
    @test 4.0 ≈ res[1]
    @test 40.0 ≈ res[2]
    @test 0.0 ≈ dout[]
    @test 0.0 ≈ dout2[]

end


function batchgf(out, args)
	res = 0.0
    x = Base.inferencebarrier((args[1][1],))
	for v in x
		v = v::Float64
		res += v
        break
	end
    out[] = res
	nothing
end

@testset "Batch Getfield" begin
    x = [(2.0, 3.0)]
    dx = [(0.0, 0.0)]
    dx2 = [(0.0, 0.0)]
    dx3 = [(0.0, 0.0)]
    out = Ref(0.0)
    dout = Ref(1.0)
    dout2 = Ref(3.0)
    dout3 = Ref(5.0)
    Enzyme.autodiff(Reverse, batchgf, Const, BatchDuplicatedNoNeed(out, (dout, dout2, dout3)), BatchDuplicated(x, (dx, dx2, dx3)))
    @test dx[1][1] ≈ 1.0
    @test dx[1][2] ≈ 0.0
    @test dx2[1][1] ≈ 3.0
    @test dx2[1][2] ≈ 0.0
    @test dx3[1][1] ≈ 5.0
    @test dx2[1][2] ≈ 0.0
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

@testset "Batch Forward" begin
    square(x)=x*x
    bres = autodiff(Forward, square, BatchDuplicatedNoNeed, BatchDuplicated(3.0, (1.0, 2.0, 3.0)))
    @test length(bres) == 1
    @test length(bres[1]) == 3
    @test bres[1][1] ≈  6.0
    @test bres[1][2] ≈ 12.0
    @test bres[1][3] ≈ 18.0

    bres = autodiff(Forward, square, BatchDuplicatedNoNeed, BatchDuplicated(3.0 + 7.0im, (1.0+0im, 2.0+0im, 3.0+0im)))
    @test bres[1][1] ≈  6.0 + 14.0im
    @test bres[1][2] ≈ 12.0 + 28.0im
    @test bres[1][3] ≈ 18.0 + 42.0im

    squareidx(x)=x[1]*x[1]
    inp = Float32[3.0]

    # Shadow offset is not the same as primal so following doesn't work
    # d_inp = Float32[1.0, 2.0, 3.0]
    # autodiff(Forward, squareidx, BatchDuplicatedNoNeed, BatchDuplicated(view(inp, 1:1), (view(d_inp, 1:1), view(d_inp, 2:2), view(d_inp, 3:3))))

    d_inp = (Float32[1.0], Float32[2.0], Float32[3.0])
    bres = autodiff(Forward, squareidx, BatchDuplicatedNoNeed, BatchDuplicated(inp, d_inp))
    @test bres[1][1] ≈  6.0
    @test bres[1][2] ≈ 12.0
    @test bres[1][3] ≈ 18.0
end

@testset "Batch Reverse" begin
    function refbatchbwd(out, x)
        v = x[]
        out[1] = v
        out[2] = v*v
        out[3] = v*v*v
        nothing
    end

    dxs = (Ref(0.0), Ref(0.0), Ref(0.0))
    out = Float64[0,0,0]
    x = Ref(2.0)

    autodiff(Reverse, refbatchbwd, BatchDuplicated(out, Enzyme.onehot(out)), BatchDuplicated(x, dxs))
    @test dxs[1][] ≈  1.0
    @test dxs[2][] ≈  4.0
    @test dxs[3][] ≈ 12.0

    function batchbwd(out, v)
        out[1] = v
        out[2] = v*v
        out[3] = v*v*v
        nothing
    end

    bres = Enzyme.autodiff(Reverse, batchbwd, BatchDuplicated(out, Enzyme.onehot(out)), Active(2.0))[1]
    @test length(bres) == 2
    @test length(bres[2]) == 3
    @test bres[2][1] ≈  1.0
    @test bres[2][2] ≈  4.0
    @test bres[2][3] ≈ 12.0

    times2(x) = x * 2
    xact = BatchDuplicated([1.0, 2.0, 3.0, 4.0, 5.0], (zeros(5), zeros(5)))
    forward, pullback = Enzyme.autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(times2)}, BatchDuplicated, typeof(xact))

    tape, primal, shadow = forward(Const(times2), xact)
    dy1 = [0.07, 0.011, 0.013, 0.017, 0.019]
    dy2 = [0.23, 0.029, 0.031, 0.037, 0.041]
    copyto!(shadow[1], dy1)
    copyto!(shadow[2], dy2)
    r = pullback(Const(times2), xact, tape)
    @test xact.dval[1] ≈ dy1 * 2
    @test xact.dval[2] ≈ dy2 * 2
end

@testset "Uncached batch sizes" begin
    genericsin(x) = Base.invokelatest(sin, x)
    res = Enzyme.autodiff(Forward, genericsin, BatchDuplicated(2.0, NTuple{10,Float64}((Float64(i) for i in 1:10))))[1]
    for (i, v) in enumerate(res)
        @test v ≈ i * -0.4161468365471424
    end
    @assert length(res) == 10
    res = Enzyme.autodiff(Forward, genericsin, BatchDuplicated(2.0, NTuple{40,Float64}((Float64(i) for i in 1:40))))[1]
    for (i, v) in enumerate(res)
        @test v ≈ i * -0.4161468365471424
    end
    @assert length(res) == 40
end

