using Enzyme
using Test
using LinearAlgebra

f0(x) = 1.0 + x
function vrec(start, x)
    if start > length(x)
        return 1.0
    else
        return x[start] * vrec(start + 1, x)
    end
end

import Enzyme: API

@testset "Internal tests" begin
    # issue #1935
    struct Incomplete
        x::Float64
        y
        Incomplete(x) = new(x)
        # incomplete constructor & non-bitstype field => !Base.allocatedinline(Incomplete)
    end
    @test Enzyme.Compiler.active_reg(Tuple{Incomplete}, Base.get_world_counter(); justActive = false) == Enzyme.Compiler.MixedState
    @test Enzyme.Compiler.active_reg(Tuple{Incomplete}, Base.get_world_counter(); justActive = true) == Enzyme.Compiler.ActiveState

    thunk_a = Enzyme.Compiler.thunk(Val(0), Const{typeof(f0)}, Active, Tuple{Active{Float64}}, Val(API.DEM_ReverseModeCombined), Val(1), Val((false, false)), Val(false), Val(false), DefaultABI, Val(false), Val(false), Val(false))
    thunk_b = Enzyme.Compiler.thunk(Val(0), Const{typeof(f0)}, Const, Tuple{Const{Float64}}, Val(API.DEM_ReverseModeCombined), Val(1), Val((false, false)), Val(false), Val(false), DefaultABI, Val(false), Val(false), Val(false))
    thunk_c = Enzyme.Compiler.thunk(Val(0), Const{typeof(f0)}, Active{Float64}, Tuple{Active{Float64}}, Val(API.DEM_ReverseModeCombined), Val(1), Val((false, false)), Val(false), Val(false), DefaultABI, Val(false), Val(false), Val(false))
    thunk_d = Enzyme.Compiler.thunk(Val(0), Const{typeof(f0)}, Active{Float64}, Tuple{Active{Float64}}, Val(API.DEM_ReverseModeCombined), Val(1), Val((false, false)), Val(false), Val(false), DefaultABI, Val(false), Val(false), Val(false))
    @test thunk_a.adjoint !== thunk_b.adjoint
    @test thunk_c.adjoint === thunk_a.adjoint
    @test thunk_c.adjoint === thunk_d.adjoint

    @test thunk_a(Const(f0), Active(2.0), 1.0) == ((1.0,),)
    @test thunk_a(Const(f0), Active(2.0), 2.0) == ((2.0,),)
    @test thunk_b(Const(f0), Const(2.0)) === ((nothing,),)

    forward, pullback = Enzyme.Compiler.thunk(Val(0), Const{typeof(f0)}, Active, Tuple{Active{Float64}}, Val(Enzyme.API.DEM_ReverseModeGradient), Val(1), Val((false, false)), Val(false), Val(false), DefaultABI, Val(false), Val(false), Val(false))

    @test forward(Const(f0), Active(2.0)) == (nothing, nothing, nothing)
    @test pullback(Const(f0), Active(2.0), 1.0, nothing) == ((1.0,),)

    function mul2(x)
        x[1] * x[2]
    end
    d = Duplicated([3.0, 5.0], [0.0, 0.0])

    forward, pullback = Enzyme.Compiler.thunk(Val(0), Const{typeof(mul2)}, Active, Tuple{Duplicated{Vector{Float64}}}, Val(Enzyme.API.DEM_ReverseModeGradient), Val(1), Val((false, true)), Val(false), Val(false), DefaultABI, Val(false), Val(false), Val(false))
    res = forward(Const(mul2), d)

    @static if VERSION < v"1.11-"
        @test typeof(res[1]) == Tuple{Float64, Float64}
    else
        @test typeof(res[1]) == NamedTuple{(Symbol("1"), Symbol("2"), Symbol("3")), Tuple{Any, Float64, Float64}}
    end

    pullback(Const(mul2), d, 1.0, res[1])
    @test d.dval[1] ≈ 5.0
    @test d.dval[2] ≈ 3.0

    d = Duplicated([3.0, 5.0], [0.0, 0.0])
    forward, pullback = Enzyme.Compiler.thunk(Val(0), Const{typeof(vrec)}, Active, Tuple{Const{Int}, Duplicated{Vector{Float64}}}, Val(Enzyme.API.DEM_ReverseModeGradient), Val(1), Val((false, false, true)), Val(false), Val(false), DefaultABI, Val(false), Val(false), Val(false))
    res = forward(Const(vrec), Const(Int(1)), d)
    pullback(Const(vrec), Const(1), d, 1.0, res[1])
    @test d.dval[1] ≈ 5.0
    @test d.dval[2] ≈ 3.0

    # @test thunk_split.primal !== C_NULL
    # @test thunk_split.primal !== thunk_split.adjoint
    # @test thunk_a.adjoint !== thunk_split.adjoint
    #
    z = ([3.14, 21.5, 16.7], [0, 1], [5.6, 8.9])
    Enzyme.make_zero!(z)
    @test z[1] ≈ [0.0, 0.0, 0.0]
    @test z[2][1] == 0
    @test z[2][2] == 1
    @test z[3] ≈ [0.0, 0.0]

    z2 = ([3.14, 21.5, 16.7], [0, 1], [5.6, 8.9])
    Enzyme.make_zero!(z2)
    @test z2[1] ≈ [0.0, 0.0, 0.0]
    @test z2[2][1] == 0
    @test z2[2][2] == 1
    @test z2[3] ≈ [0.0, 0.0]

    z3 = [3.4, "foo"]
    Enzyme.make_zero!(z3)
    @test z3[1] ≈ 0.0
    @test z3[2] == "foo"

    z4 = sin
    Enzyme.make_zero!(z4)

    struct Dense
        n_inp::Int
        b::Vector{Float64}
    end

    function Dense(n)
        Dense(n, rand(n))
    end

    nn = Dense(4)
    Enzyme.make_zero!(nn)
    @test nn.b ≈ [0.0, 0.0, 0.0, 0.0]
end

@testset "Reflection" begin
    Enzyme.Compiler.enzyme_code_typed(Active, Tuple{Active{Float64}}) do x
        x^2
    end
    sprint() do io
        Enzyme.Compiler.enzyme_code_native(io, f0, Active, Tuple{Active{Float64}})
    end

    sprint() do io
        Enzyme.Compiler.enzyme_code_llvm(io, f0, Active, Tuple{Active{Float64}})
    end
end

sumsq2(x) = sum(abs2, x)
sumsin(x) = sum(sin, x)
sqrtsumsq2(x) = (sum(abs2, x) * sum(abs2, x))
@testset "Recursion optimization" begin
    # Test that we can successfully optimize out the augmented primal from the recursive divide and conquer
    fn = sprint() do io
        Enzyme.Compiler.enzyme_code_llvm(io, sum, Active, Tuple{Duplicated{Vector{Float64}}}; dump_module = true)
    end
    @test occursin("diffe", fn)
    # TODO we need to fix julia to remove unused bounds checks
    # @test !occursin("aug",fn)

    fn = sprint() do io
        Enzyme.Compiler.enzyme_code_llvm(io, sumsq2, Active, Tuple{Duplicated{Vector{Float64}}}; dump_module = true)
    end
    @test occursin("diffe", fn)
    # TODO we need to fix julia to remove unused bounds checks
    # @test !occursin("aug",fn)

    fn = sprint() do io
        Enzyme.Compiler.enzyme_code_llvm(io, sumsin, Active, Tuple{Duplicated{Vector{Float64}}}; dump_module = true)
    end
    @test occursin("diffe", fn)
    # TODO we need to fix julia to remove unused bounds checks
    # @test !occursin("aug",fn)

    fn = sprint() do io
        Enzyme.Compiler.enzyme_code_llvm(io, sqrtsumsq2, Active, Tuple{Duplicated{Vector{Float64}}}; dump_module = true)
    end
    @test occursin("diffe", fn)
    # if count("call fastcc void @diffejulia__mapreduce", fn) != 1
    #     println(sprint() do io
    #        Enzyme.Compiler.enzyme_code_llvm(io, sqrtsumsq2, Active, Tuple{Duplicated{Vector{Float64}}}; dump_module=true, run_enzyme=false, optimize=false)
    #    end)
    #     println(sprint() do io
    #        Enzyme.Compiler.enzyme_code_llvm(io, sqrtsumsq2, Active, Tuple{Duplicated{Vector{Float64}}}; dump_module=true, run_enzyme=false)
    #    end)
    #     println(fn)
    # end
    # TODO per system being run on the indexing in the mapreduce is broken
    @test_broken count("call fastcc void @diffejulia__mapreduce", fn) == 1
    # TODO we need to have enzyme circumvent the double pointer issue by also considering a broader
    # no memory overwritten state [in addition to the arg-based variant]
    @test_broken !occursin("aug", fn)

    x = ones(100)
    dx = zeros(100)
    Enzyme.autodiff(Reverse, sqrtsumsq2, Duplicated(x, dx))
end

# @testset "Split Tape" begin
#     f(x) = x[1] * x[1]

#     thunk_split = Enzyme.Compiler.thunk(f, Tuple{Duplicated{Array{Float64,1}}}, Val(Enzyme.API.DEM_ReverseModeGradient))
#     @test thunk_split.primal !== C_NULL
#     @test thunk_split.primal !== thunk_split.adjoint
# end

function deadarg_pow(z::T, i) where {T <: Real}
    zabs = abs(z)
    if sign(z) < zero(T)
        return (zabs^i) * (cos(T(π) * i) + sin(T(π) * i)im)
    end
    return zabs^i + zero(T)im
end

function deadargtest(n)
    wp = 1 + deadarg_pow(-n, 0.5)

    deadarg_pow(-n, 0.5)

    return real(wp)
end

@testset "Dead arg elim" begin
    res = autodiff(Enzyme.ReverseWithPrimal, deadargtest, Active, Active(0.25))
    @test res[2] ≈ 1.0
end

@testset "1.11 vcat" begin

    function fcat(x)
        r = vcat(Any[1])
        return x
    end

    res = Enzyme.autodiff(Reverse, fcat, Active(2.0))
    @test res[1][1] ≈ 1.0

end

function fillsum(x)
    a = similar(rand(3, 3))
    fill!(a, x)
    return sum(a)
end

@testset "Fill sum" begin
    res = autodiff(Forward, fillsum, Duplicated(2.0, 1.0))[1]
    @test 9.0 ≈ res
end


mutable struct RTGData
    x
end

@noinline function rtg_sub(V, cv)
    return cv
end

@noinline function rtg_cast(cv)
    return cv
end

function rtg_f(V, @nospecialize(cv))
    s = rtg_sub(V, Base.inferencebarrier(cv))::RTGData
    s = rtg_cast(Base.inferencebarrier(s.x))::Float64
    return s
end

@testset "RuntimeActivity generic call" begin
    res = autodiff(set_runtime_activity(ForwardWithPrimal), rtg_f, Duplicated, Duplicated([0.2], [1.0]), Const(RTGData(3.14)))
    @test 3.14 ≈ res[2]
    @test 0.0 ≈ res[1]
end

@inline function myquantile(v::AbstractVector, p::Real; alpha)
    n = length(v)

    m = 1.0 + p * (1.0 - alpha - 1.0)
    aleph = n * p + oftype(p, m)
    j = clamp(trunc(Int, aleph), 1, n - 1)
    γ = clamp(aleph - j, 0, 1)

    if n == 1
        a = @inbounds v[1]
        b = @inbounds v[1]
    else
        a = @inbounds v[j]
        b = @inbounds v[j + 1]
    end

    return a + γ * (b - a)
end

function fquantile(x)
    v = [1.0, x]
    return @inbounds (map(y -> myquantile(v, y, alpha = 1.0), [0.7]))[1]
end

@testset "Attributor issues" begin

    cor = fquantile(2.0)
    res = autodiff(ForwardWithPrimal, fquantile, Duplicated, Duplicated(2.0, 1.0))
    @test cor ≈ res[2]
    @test 0.7 ≈ res[1]

end

@testset "IO" begin

    function printsq(x)
        println(x)
        x * x
    end

    @test 4.6 ≈ first(autodiff(Reverse, printsq, Active, Active(2.3)))[1]
    @test 4.6 ≈ first(autodiff(Forward, printsq, Duplicated(2.3, 1.0)))

    function tostring(x)
        string(x)
        x * x
    end

    @test 4.6 ≈ first(autodiff(Reverse, tostring, Active, Active(2.3)))[1]
    @test 4.6 ≈ first(autodiff(Forward, tostring, Duplicated(2.3, 1.0)))
end

@testset "hmlstm" begin
    sigm(x) = @fastmath 1 / (1 + exp(-x))
    @fastmath function hmlstm_update_c_scalar(z, zb, c, f, i, g)
        if z == 1.0f0 # FLUSH
            return sigm(i) * tanh(g)
        elseif zb == 0.0f0 # COPY
            return c
        else # UPDATE
            return sigm(f) * c + sigm(i) * tanh(g)
        end
    end

    N = 64
    Z = round.(rand(Float32, N))
    Zb = round.(rand(Float32, N))
    C = rand(Float32, N, N)
    F = rand(Float32, N, N)
    I = rand(Float32, N, N)
    G = rand(Float32, N, N)

    function broadcast_hmlstm(out, Z, Zb, C, F, I, G)
        out .= hmlstm_update_c_scalar.(Z, Zb, C, F, I, G)
        return nothing
    end

    ∇C = zeros(Float32, N, N)
    ∇F = zeros(Float32, N, N)
    ∇I = zeros(Float32, N, N)
    ∇G = zeros(Float32, N, N)

    # TODO(wsmoses): Check after updating Enzyme_jll
    # autodiff(broadcast_hmlstm, Const,
    #          Const(zeros(Float32, N, N)), Const(Z), Const(Zb),
    #          Duplicated(C, ∇C), Duplicated(F, ∇F), Duplicated(I, ∇I), Duplicated(G, ∇G))
    # fwddiff(broadcast_hmlstm, Const,
    #          Const(zeros(Float32, N, N)), Const(Z), Const(Zb),
    #          Duplicated(C, ∇C), Duplicated(F, ∇F), Duplicated(I, ∇I), Duplicated(G, ∇G))
end

genlatestsin(x)::Float64 = Base.invokelatest(sin, x)
function genlatestsinx(xp)
    x = @inbounds xp[1]
    @inbounds xp[1] = 0.0
    return Base.invokelatest(sin, x)::Float64 + 1
end

function loadsin(xp)
    x = @inbounds xp[1]
    @inbounds xp[1] = 0.0
    return sin(x)
end
function invsin(xp)
    xp = Base.invokelatest(convert, Vector{Float64}, xp)
    return loadsin(xp)
end

@testset "generic" begin
    @test -0.4161468365471424 ≈ Enzyme.autodiff(Reverse, genlatestsin, Active, Active(2.0))[1][1]
    @test -0.4161468365471424 ≈ Enzyme.autodiff(Forward, genlatestsin, Duplicated(2.0, 1.0))[1]

    x = [2.0]
    dx = [0.0]
    Enzyme.autodiff(Reverse, genlatestsinx, Active, Duplicated(x, dx))
    @test 0 ≈ x[1]
    @test -0.4161468365471424 ≈ dx[1]

    x = [2.0]
    dx = [0.0]
    Enzyme.autodiff(Reverse, invsin, Active, Duplicated(x, dx))
    @test 0 ≈ x[1]
    @test -0.4161468365471424 ≈ dx[1]

    function inactive_gen(x)
        n = 1
        for k in 1:2
            y = falses(n)
        end
        return x
    end
    @test 1.0 ≈ Enzyme.autodiff(Reverse, inactive_gen, Active, Active(1.0e4))[1][1]
    @test 1.0 ≈ Enzyme.autodiff(Forward, inactive_gen, Duplicated(1.0e4, 1.0))[1]

    function whocallsmorethan30args(R)
        temp = diag(R)
        R_inv = [
            temp[1] 0.0 0.0 0.0 0.0 0.0;
            0.0 temp[2] 0.0 0.0 0.0 0.0;
            0.0 0.0 temp[3] 0.0 0.0 0.0;
            0.0 0.0 0.0 temp[4] 0.0 0.0;
            0.0 0.0 0.0 0.0 temp[5] 0.0;
        ]

        return sum(R_inv)
    end

    R = zeros(6, 6)
    dR = zeros(6, 6)

    @static if VERSION ≥ v"1.11-"
    elseif VERSION ≥ v"1.10.8"
        autodiff(Reverse, whocallsmorethan30args, Active, Duplicated(R, dR))
        @test 1.0 ≈ dR[1, 1]
        @test 1.0 ≈ dR[2, 2]
        @test 1.0 ≈ dR[3, 3]
        @test 1.0 ≈ dR[4, 4]
        @test 1.0 ≈ dR[5, 5]
        @test 0.0 ≈ dR[6, 6]
    else
        @test_broken autodiff(Reverse, whocallsmorethan30args, Active, Duplicated(R, dR))
    end
end

@testset "invoke" begin
    @noinline apply(@nospecialize(func)) = func()

    function invtest(arr)
        function f()
            arr[1] *= 5.0
            nothing
        end
        apply(f)
    end

    x = [2.0]
    dx = [1.0]

    Enzyme.autodiff(Reverse, invtest, Duplicated(x, dx))

    @test 10.0 ≈ x[1]
    @test 5.0 ≈ dx[1]
end

@testset "Dynamic Val Construction" begin

    dyn_f(::Val{D}) where {D} = prod(D)
    dyn_mwe(x, t) = x / dyn_f(Val(t))

    @test 0.5 ≈ Enzyme.autodiff(Reverse, dyn_mwe, Active, Active(1.0), Const((1, 2)))[1][1]
end


sinadd(x, y) = (sin.(x) .+ (y))

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

    A = rand(10, 10); B = rand(10, 10)
    dA = zero(A); dB = zero(B); dR = fill!(similar(A), 1)

    autodiff(Reverse, foo_bc!, Const, Duplicated(A, dR), Duplicated(transpose(A), transpose(dA)), Duplicated(B, dB))

    # no runtime activity required
    res = autodiff(Forward, sinadd, Duplicated([2.7], [4.2]), Const([0.31]))[1]
    @test [-3.7971029964716574] ≈ res
end


@testset "DuplicatedReturn" begin
    moo(x) = fill(x, 10)

    @test_throws ErrorException autodiff(Reverse, moo, Active(2.1))
    fo, = autodiff(Forward, moo, Duplicated(2.1, 1.0))
    for i in 1:10
        @test 1.0 ≈ fo[i]
    end

    @test_throws ErrorException autodiff(Forward, x -> x, Active(2.1))
end

@testset "Mismatched return" begin
    @test_throws ErrorException autodiff(Reverse, _ -> missing, Active, Active(2.1))
    @test_throws ErrorException autodiff_deferred(Reverse, Const(_ -> missing), Active, Active(2.1))
end

@testset "Copy" begin
    function advance(u_v_eta)
        eta = copy(u_v_eta)
        return @inbounds eta[1]
    end

    u_v_eta = [0.0]
    ad_struct = [1.0]

    autodiff(Reverse, advance, Active, Duplicated(u_v_eta, ad_struct))
    @test ad_struct[1] ≈ 2.0

    function advance2(u_v_eta)
        eta = copy(u_v_eta)
        return @inbounds eta[1][]
    end

    u_v_eta = [Ref(0.0)]
    ad_struct = [Ref(1.0)]

    autodiff(Reverse, advance2, Active, Duplicated(u_v_eta, ad_struct))
    @test ad_struct[1][] ≈ 2.0


    function incopy(u_v_eta, val, i)
        eta = copy(u_v_eta)
        eta[1] = val
        return @inbounds eta[i]
    end

    u_v_eta = [0.0]

    v = autodiff(Reverse, incopy, Active, Const(u_v_eta), Active(3.14), Const(1))[1][2]
    @test v ≈ 1.0
    @test u_v_eta[1] ≈ 0.0

    function incopy2(val, i)
        eta = Float64[2.3]
        eta[1] = val
        return @inbounds eta[i]
    end

    v = autodiff(Reverse, incopy2, Active, Active(3.14), Const(1))[1][1]
    @test v ≈ 1.0
end

@testset "No inference" begin
    c = 5.0
    @test 5.0 ≈ autodiff(Reverse, (A,) -> c * A, Active, Active(2.0))[1][1]
    @test 5.0 ≈ autodiff(Forward, (A,) -> c * A, Duplicated(2.0, 1.0))[1]
end

@testset "Type-instable capture" begin
    L = Array{Float64, 1}(undef, 2)

    F = [1.0, 0.0]

    function main()
        t = 0.0

        function cap(m)
            t = m
        end

        @noinline function inner(F, cond)
            if cond
                genericcall(F)
            end
        end

        function tobedifferentiated(F, cond)
            inner(F, cond)
            # Force an apply generic
            -t
            nothing
        end
        autodiff(Reverse, Const(tobedifferentiated), Duplicated(F, L), Const(false))
        autodiff(Forward, Const(tobedifferentiated), Duplicated(F, L), Const(false))
    end

    main()
end

function solve_cubic_eq(poly::AbstractVector{Complex{T}}) where {T}
    a1 = 1 / @inbounds poly[1]
    E1 = 2 * a1
    E12 = E1 * E1
    s1 = log(E12)
    return nothing
end

@testset "Extract Tuple for Reverse" begin
    autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(solve_cubic_eq)}, Const, Duplicated{Vector{Complex{Float64}}})
end

@testset "BLAS" begin
    x = [2.0, 3.0]
    dx = [0.2, 0.3]
    y = [5.0, 7.0]
    dy = [0.5, 0.7]
    Enzyme.autodiff(Reverse, (x, y) -> x' * y, Duplicated(x, dx), Duplicated(y, dy))
    @test dx ≈ [5.2, 7.3]
    @test dy ≈ [2.5, 3.7]

    f_exc(x) = sum(x * x)
    y = [[1.0, 2.0] [3.0, 4.0]]
    f_x = zero.(y)
    Enzyme.autodiff(Reverse, f_exc, Duplicated(y, f_x))
    @test f_x ≈ [7.0 9.0; 11.0 13.0]
end

function indirectfltret(a)::DataType
    a[] *= 2
    return Float64
end
@testset "Partial return information" begin
    d = Duplicated(Ref(3.0), Ref(0.0))
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(indirectfltret)}, Const{DataType}, typeof(d))

    tape, primal, shadow = fwd(Const(indirectfltret), d)
    @test tape == nothing
    @test primal == Float64
    @test shadow == nothing
end

@testset "Gradient & NamedTuples" begin
    xy = (x = [1.0, 2.0], y = [3.0, 4.0])
    grad = Enzyme.gradient(Reverse, z -> sum(z.x .* z.y), xy)[1]
    @test grad == (x = [3.0, 4.0], y = [1.0, 2.0])

    xp = (x = [1.0, 2.0], p = 3)  # 3::Int is non-diff
    grad = Enzyme.gradient(Reverse, z -> sum(z.x .^ z.p), xp)[1]
    @test grad.x == [3.0, 12.0]

    xp2 = (x = [1.0, 2.0], p = 3.0)  # mixed activity
    grad = Enzyme.gradient(Reverse, z -> sum(z.x .^ z.p), xp2)[1]
    @test grad.x == [3.0, 12.0]
    @test grad.p ≈ 5.545177444479562

    xy = (x = [1.0, 2.0], y = [3, 4])  # y is non-diff
    grad = Enzyme.gradient(Reverse, z -> sum(z.x .* z.y), xy)[1]
    @test grad.x == [3.0, 4.0]
    @test grad.y === xy.y  # make_zero did not copy this

    grad = Enzyme.gradient(Reverse, z -> (z.x * z.y), (x = 5.0, y = 6.0))[1]
    @test grad == (x = 6.0, y = 5.0)

    grad = Enzyme.gradient(Reverse, abs2, 7.0)[1]
    @test grad == 14.0
end

@testset "Forward on Reverse" begin

    function speelpenning(y, x)
        ccall(
            :memmove, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
            y, x, 2 * 8
        )
        return nothing
    end

    x = [0.5, 0.3]
    y = zeros(2)
    dx = ones(2)
    rx = zeros(2)
    drx = zeros(2)
    dy = zeros(2)
    ry = ones(2)
    dry = zeros(2)

    function foo(y, dy, x, dx)
        autodiff(Reverse, speelpenning, Const, Duplicated(y, dy), Duplicated(x, dx))
        return nothing
    end

    autodiff(Forward, foo, Duplicated(x, dx), Duplicated(rx, drx), Duplicated(y, dy), Duplicated(ry, dry))
end

using Random

@testset "Random" begin
    f_rand(x) = x * rand()
    f_randn(x, N) = x * sum(randn(N))
    @test 0 <= autodiff(Reverse, f_rand, Active, Active(1.0))[1][1] < 1
    @test !iszero(autodiff(Reverse, f_randn, Active, Active(1.0), Const(64))[1][1])
    @test iszero(autodiff(Reverse, x -> rand(), Active, Active(1.0))[1][1])
    @test iszero(autodiff(Reverse, (x, N) -> sum(randn(N)), Active, Active(1.0), Const(64))[1][1])
    @test autodiff(Reverse, x -> x * sum(randcycle(5)), Active, Active(1.0))[1][1] == 15
    @test autodiff(Reverse, x -> x * sum(randperm(5)), Active, Active(1.0))[1][1] == 15
    @test autodiff(Reverse, x -> x * sum(shuffle(1:5)), Active, Active(1.0))[1][1] == 15
end

struct DensePE
    n_inp::Int
    W::Matrix{Float64}
end

struct NNPE
    layers::Tuple{DensePE, DensePE}
end


function set_paramsPE(nn, params)
    i = 1
    for l in nn.layers
        W = l.W # nn.layers[1].W
        Base.copyto!(W, reshape(view(params, i:(i + length(W) - 1)), size(W)))
    end
    return
end

@testset "Illegal phi erasure" begin
    # just check that it compiles
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(set_paramsPE)}, Const, Duplicated{NNPE}, Duplicated{Vector{Float64}})
    @test fwd !== nothing
    @test rev !== nothing
    nn = NNPE((DensePE(1, Matrix{Float64}(undef, 4, 4)), DensePE(1, Matrix{Float64}(undef, 4, 4))))
    dnn = NNPE((DensePE(1, Matrix{Float64}(undef, 4, 4)), DensePE(1, Matrix{Float64}(undef, 4, 4))))
    l = Vector{Float64}(undef, 32)
    dl = Vector{Float64}(undef, 32)
    fwd(Const(set_paramsPE), Duplicated(nn, dnn), Duplicated(l, dl))
end

@static if VERSION >= v"1.11-"
    mutable struct MyDict{K, V}
        slots::Memory{UInt8}
        keys::Memory{K}
        vals::Memory{V}
        idxfloor::Int
        maxprobe::Int
        function MyDict{K, V}() where {V} where {K}
            slots = Memory{UInt8}(undef, 0)
            fill!(slots, 0x00)
            new(slots, Memory{K}(undef, 0), Memory{V}(undef, 0), 1, 0)
        end
    end
    function my_rehash!(h::MyDict{K, V}) where {V} where {K}
        newsz = 16
        h.idxfloor = 1
        slots = Memory{UInt8}(undef, newsz)
        fill!(slots, 0x00)
        h.slots = slots
        h.keys = Memory{K}(undef, newsz)
        h.vals = Memory{V}(undef, newsz)
        h.maxprobe = 0
        return h
    end
    function ht_keyindex2_shorthash!(h::MyDict{K, V}, key) where {V} where {K}
        sz = length(h.keys)
        if sz == 0
            my_rehash!(h)
            index, sh = Base.hashindex(key, length(h.keys))
            return -index, sh
        end
        index, sh = Base.hashindex(key, sz)
        keys = h.keys
        while true
            if h.slots[index] == 0x00
                return -index, sh
            end
            if h.slots[index] == sh
                k = keys[index]
                if key === k || isequal(key, k)
                    return index, sh
                end
            end
            1 > h.maxprobe && break
        end
        return 0, sh
    end
    function my_setindex!(h::MyDict{K, V}, v, key::K) where {V} where {K}
        index, sh = ht_keyindex2_shorthash!(h, key)
        h.slots[-index] = sh
        h.keys[-index] = key
        h.vals[-index] = v
        return h
    end

    struct E{T}
        e::T
    end
    struct D{sym, T <: E}
        d::T
    end
    function Base.:(==)(d1::D{sym1}, d2::D{sym2}) where {sym1, sym2}
        # changing either == to === makes it no longer crash
        return sym1 == sym2 && d1.d == d2.d
    end
    function f(x)  # x is unused now
        d = MyDict{D, Int}()
        my_setindex!(d, 1, D{:s, E{Int}}(E(1)))
        return 1.0
    end

    @testset "Error handler for jlcall" begin
        res = Enzyme.gradient(set_runtime_activity(Reverse), f, [0.0])
        @test res[1] ≈ [0.0]
    end

end

@inline function uns_mymean(f, A, ::Type{T}, c) where {T}
    c && return Base.inferencebarrier(nothing)
    x1 = f(@inbounds A[1]) / 1
    return @inbounds A[1][1]
end

function uns_sum2(x::Array{T})::T where {T}
    op = Base.add_sum
    itr = x
    y = iterate(itr)::Tuple{T, Int}
    v = y[1]::T
    while true
        y = iterate(itr, y[2])
        y === nothing && break
        v = (v + y[1])::T
    end
    return v
end

function uns_ad_forward(scale_diag::Vector{T}, c) where {T}
    res = uns_mymean(uns_sum2, [scale_diag], T, c)
    return res
end

@testset "Split box float32" begin
    q = ones(Float32, 1)
    dx = make_zero(q)
    res, y = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        uns_ad_forward,
        Enzyme.Active,
        Enzyme.Duplicated(q, dx),
        Enzyme.Const(false),
    )
    @test dx ≈ Float32[1.0]
    q = ones(Float64, 1)
    dx = make_zero(q)
    res, y = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        uns_ad_forward,
        Enzyme.Active,
        Enzyme.Duplicated(q, dx),
        Enzyme.Const(false),
    )
    @test dx ≈ Float64[1.0]
end

@inline extract_bc(bc, ::Val{:north}) = (bc.north)
@inline extract_bc(bc, ::Val{:top}) = (bc.top)

function permute_boundary_conditions(boundary_conditions)
    sides = [:top, :north] # changing the order of these actually changes the error
    boundary_conditions = Tuple(extract_bc(boundary_conditions, Val(side)) for side in sides)

    return nothing
end

@testset "Extract abstype" begin

    parameters = (a = 1, b = 0.1)

    bc = (north = 1, top = tuple(parameters, tuple(:c)))
    d_bc = Enzyme.make_zero(bc)
    Enzyme.API.looseTypeAnalysis!(true)

    dc²_dκ = autodiff(
        Enzyme.Reverse,
        permute_boundary_conditions,
        Duplicated(bc, d_bc)
    )

    Enzyme.API.looseTypeAnalysis!(false)
end

const N_SAMPLES = 500
const N_COMPONENTS = 4

const data = [-0.5560268761463861, -0.444383357109696, 0.027155338009193845, -0.29948409035891055, 1.7778610980573246, -1.14490153172882, -0.46860588216767457, 0.15614346264074028, -2.641991008076796, 1.0033099014594844, 1.0823812056084292, 0.18702790710363, 0.5181487878771377, 1.4913791170403063, 0.3675627461748204, -0.8862052960481365, 0.6845647041648603, -1.590579974922555, 0.410653382404333, -0.856349830552304, -1.0509877103612828, 0.502079457623539, -0.2162480073298127, -0.7064242722014349, -3.663034802576991, 0.1683412659309176, 0.28425906701710857, 0.5698286489101805, -1.4220589095735332, -0.37240087577993225, 0.36901028455183293, -0.007612980079313577, 0.562668812321259, 0.10686911035365092, 0.5694584949295476, 0.6810849274435286, -1.3391251213773154, -0.23828371819888622, 1.0193587887377156, 0.7017713284136731, -0.14521050140729616, 0.6428964706243068, 1.8193484973888463, -0.3672598918343543, 0.7565689715912548, 0.08701681344541234, -0.8511936279150268, 0.9242378649817816, 1.6120908802143608, -0.9258028623888832, 0.49199249434769243, -0.22608145131612992, -1.351640432408328, 0.3023655023653634, 1.2252567101879008, -0.4579776898846808, -0.36873503034471294, -0.5879094743345893, 0.8901285443512134, 0.23942258932052793, -0.195126767304092, 0.6541265910968944, -0.8180579409672205, -1.6505670679051851, -0.41299157333934094, 0.027291621540711814, 0.29759513748684024, 0.07136737211887596, -0.8945184707955289, -1.9947538311302822, -0.18728482002197755, -0.5854431752183636, -1.5094025801991475, -0.10841979845993832, -0.37149972405672654, 1.209427254258146, -0.20401001223483511, 0.012484378414156184, 0.14058032536255227, -0.9923922290801328, -1.1484589871168687, 1.3715759475375402, -0.05906784259018913, -0.3530786655768097, -1.4488810057984256, 2.3879153026952267, 0.12580788649805388, -1.9725913559635857, 0.7009118499232365, 0.31700578675698954, 0.2762925198922067, -1.625619456664758, -0.9373153541158463, -0.9304928802188113, -1.9905539723314605, 0.13753980192836776, 3.1495759241275225, -0.7214121984874265, -0.577092696986848, 0.4593837753047561, 0.24720770605505335, -0.02566249380406308, -0.6320809680268722, -1.0204193548690512, -1.311507800204285, 0.687066447881175, -0.09460071173365793, -0.28474772376166907, -0.3387627167144301, -0.09536748694092163, 0.7689715736798227, 1.442443602597533, -0.27503213774867397, -0.37963749230903393, -0.7226963203200736, 0.13966112558116067, -1.4093392453795512, 1.0554566357077169, -2.237822231313888, 1.15915397311866, -0.6901150042613587, -1.3821310931236412, 0.5938504216651738, 0.6609960603802911, 2.7589164663644565, 0.46763556069956336, -0.08060548262614446, 0.0795712995405885, -0.36251216727772734, -0.6883308052782828, -0.6957669363357581, -0.4298941588197229, -0.6170193131914518, -0.7875381233595508, 0.9793681640487545, -0.16689001105724968, -0.4410353697187671, -0.0106585238662763, 0.4075406194010163, -0.3824860969744455, -0.4306357340413131, -0.05292489819141157, 0.7631065933222378, -1.7078224461664113, -0.665051961459703, 1.5950208497188643, 0.5424677877448506, 0.2702908010561237, 1.1637402735830906, -0.9752391996154888, 0.591290372307129, -0.5811624500841813, -1.0412662750801527, -0.19245292741043743, -0.4348102015339658, 0.08422740657017527, 1.0438125328282608, -0.4927174182298662, 1.2458570216388754, 1.1205311509593492, -0.12330863869436813, -1.0664768973086367, 0.30470144397407023, -1.8010263359728695, -0.13268665889419914, 0.630295337858245, -2.3417617931253183, -0.15973244410747056, 0.6795317720001368, -0.7447645337583819, 1.2306970723625588, 1.090597639554929, 1.8958640522777934, -0.26786676662796843, 2.0394271629613625, 0.055740331653061796, -0.7193540879657702, -0.1628910819232136, -0.2882790142695281, -2.0534491466053764, -2.233319106981269, -1.1534087585837147, -1.5591843759684125, -1.3523434858414614, -0.35519147266027956, -0.9383662929403082, 0.5502010944298217, -1.6530342319857132, -1.0177969113873517, 1.3546070391530678, -0.7303143540080486, 1.4594015115819061, 1.1755531107578732, 1.469591632664121, 2.155036256421912, -0.1978160724116997, -1.238444066448837, 0.4762431627842421, 0.5664035042754365, 2.191213907869695, -0.16697076769423774, -1.8401747205811765, -0.5935878583224328, -0.4447185333005777, 1.8811333529161927, -0.857003256515023, 0.5308971512660459, 0.9475262937475932, 0.5065543926241618, -1.426876319667508, 0.27277024759538804, 1.6832914767785863, 0.8794490994419152, -0.37229135630818994, 1.2103793835305425, -0.8145152351424103, -0.6637107250031519, -0.3642002730267983, 0.128180863566292, -0.8555397517942166, 0.7463496214304585, -0.21349745615233298, 0.6069963236292358, -0.15043631758004797, 0.2865438734856877, 0.9689530290178722, 0.4645944393389331, -0.10075844220363214, 0.9719135191686711, 1.359272322470581, 0.9198388707805807, -0.003947750510092571, 0.6651494514356097, -0.4642945862879513, -0.09632384020701204, -1.4640316974713914, 0.03411735606151582, 0.192612577289544, 1.2723529571406376, -0.6797254154698416, 0.7121446020587434, 1.6474227377969937, -1.3612960160873442, 1.639942921300844, -1.2934385805566249, -0.6093348312692207, -0.4929035640975793, 0.07652791635611562, 0.15922627820027674, 0.4446393063910561, -1.212412143247565, -1.3517775856153358, -1.0318877340986508, 1.074228515227531, 1.0673524298941364, -0.17007000912445897, 0.19378013515263207, -2.4816300047227666, -0.4592570057697022, -1.1897921624102403, 0.26056620827295174, -0.6309468513820905, 1.5399524139961005, -0.10352720131589113, 1.0498414218815497, -0.08560706218145639, -1.968271606952006, 0.9137220126511895, 1.5165903941194543, -0.9634560368533389, 1.1884250536535346, -0.23295101440683805, 0.9553533369282987, -0.3098984978002516, -0.042208017646174, -1.9930838186185373, 0.6230463669791857, -2.7605050117797982, 1.2120139690044167, 1.5742425795634203, -0.8464907448939961, 0.7321425928205605, 1.044045122552859, 1.6213749963355164, 2.750115535545452, 1.7347194078359887, -1.2300524375750894, -0.36190025258293085, 0.16420959796469084, -0.2846711718337991, 1.815652557702069, -0.7696456318219987, -0.3758835433623372, -0.31538987759342985, 0.23203583241300865, -0.9042757791617796, 0.14623184817251003, 0.22324769054960142, -0.07430046379776922, 0.8598376944077396, 0.8094753739829023, 0.7780695563934816, -0.9880838058116279, -0.17529075003709038, -1.4320848711398677, 0.49819547701694217, 1.455253953626022, -0.8646238569013837, 0.6274090322589988, 0.7214994440898491, 1.0249395310099292, 1.6051684957766426, -0.41752946512010924, -1.187706044484646, 1.9667607247158339, -0.8273416405870931, -1.8095812957335915, 0.21946085689792014, 1.6959323077028474, 0.07606410600663914, -0.0005899969679317951, -0.6300575938012335, 0.7168660929110295, -1.6957830800502658, -2.378949781992021, 0.1614508487249226, -0.34807928510100733, -0.19506959830062723, -0.5497606187369344, 1.0808323747949233, -2.2125060475463463, 0.8718983568753548, -0.007206357093314457, -1.575891948273875, -2.2088301903139564, -0.6163495955240155, -0.5801739513350528, -1.5612897485472592, -1.3002596896606895, -1.0059040824152614, 0.6796485760534101, -0.043207370167515954, -0.039839626218822005, -0.4385362125324239, -0.09173118968880091, 1.22561207137653, -0.232131676978592, -1.2139396904067505, -0.23690460123730323, -0.3827075659919168, -1.9688978438045297, -1.5797479817238906, -0.5654974841550139, -1.7170129974387656, -2.446261897220929, -0.26713540122649804, 0.6778692338783507, -0.1689008828898645, 1.604831121738095, 1.7480788262457672, 0.9166815687612451, 1.1341703209400371, -2.1775754411288144, -0.330419660506073, -0.2672312785080624, 1.200964147356692, 1.3170491877286854, 0.6023924017880021, -0.9827718177516547, -1.1457095184571038, 0.25819205428715203, -2.282547976724439, -3.049187087985662, -1.281790532097414, 2.8015194483397003, 0.5639614209301308, -0.45753014518031915, -0.7991285413217107, 1.0753460926351635, -0.5569593865129592, 0.049550548181828, -0.8913053693383933, -0.7053146611866707, 1.3970437844025363, -1.9127587026035067, -0.6408264977866664, -0.4027208663305603, 0.2116527896752755, 2.2400517749401025, -0.881636745059659, -0.6176341167004805, -0.3916247912848145, 0.9513607440394651, 0.14123219972588208, 1.2053043404475006, 0.023930401450278135, -0.8132651104773965, 1.010114660634259, 0.14932562573657962, 1.7774482649689414, 0.8427840155284196, -0.9801124442248111, 1.2865644225495012, 0.4389849473895256, -0.505701456587577, -1.6549980227323258, 0.6515278406227536, -0.5295755930612868, 0.9056306662830947, 0.08972278324911305, 0.23264270532152218, 2.627396969584777, 0.27204169314700904, 0.9247523784433901, 0.39543449166392586, -3.0074905237355902, 1.1183821383894414, 0.17479140498380819, -1.2141175099769368, 0.19312543732457393, 0.3046417196295455, -2.1349686721255985, 0.5660453142567702, 0.025065143849368067, -0.3915696698906197, 0.8816658350282802, 0.8266483514919597, 0.9493314906580934, -0.0032065446136149488, -1.7961894556919296, 0.4130469384119612, -1.28722133892104, -1.119436381330247, 0.773218214182339, -0.3837586462966479, 0.07777043635090204, -0.7542646791925891, 0.08136609240300065, 0.2746677233237281, 1.1122181237929718, 0.5326574958161293, 0.7823989790674563, -0.31307892473155574, -0.04580883976550116, 0.1691926964033846, 0.37103104440006834, -0.9432191269564248, -0.7609096208689287, 0.2804422856751161, -0.048373157442897496, -0.981155307666483, 1.3029831269962606, 0.059610387285286434, 0.12450090856951314, 0.11358777574045617, -1.3306646401495767, -0.34798310991558473, -0.2866743445913757, 0.674272748688434, -0.4239489256735372, -0.9229714850145041, 0.3113603292165489, -0.4846778890580723, -0.013595195649033436, 1.2403767852654752, 1.0331480262937687, -0.11947562831616687, -0.6057052894354995, -0.7754699190529916, 1.1616052799742849, 1.2438648692130239, 0.027783265850463142, -1.2121179280013439, 0.6370251861203581, 0.6834320658257506, 0.6241655870590277, -1.353228100410462, 0.8938693570362417, 0.8374026807814964, -0.3732266794347597, 1.1790529100520817, 0.7911863169212741, 0.2189687616385222, -0.6113204774701082, 0.19900691423850023, 0.31468457309049136, 1.2458549461519632, 0.5053751217075103, -0.4497826390316608, 0.6003636947378631, 0.7879125998061005, 0.4768361874753698, -0.7096215982620703, 0.09448322860785968, -1.6374823189754906, 1.1567167561713774, 0.7983310036650442, -1.3254511689721826, -0.2200472053270165, 0.629399242764823]
const params0 = [0.25733304995705586, 0.4635056170085754, 0.5285451129509773, 0.7120981447127772, 0.835601264145011, -1.4646785862195637, 0.24736086263101278, -0.21967358320549735, 1.0624643704713206, 1.628664511492019, 1.8530572439128092, 0.6276756477143253]

# ========== Objective function ==========
normal_pdf(x::Real, mean::Real, var::Real) =
    exp(-(x - mean)^2 / (2var)) / sqrt(2π * var)

normal_pdf(x, mean, var) =
    exp(-(x - mean)^2 / (2var)) / sqrt(2π * var)

# original objective (doesn't work)
function mixture_loglikelihood1(params::AbstractVector{<:Real}, data::AbstractVector{<:Real})::Real
    K = length(params) ÷ 3
    weights, means, stds = @views params[1:K], params[(K + 1):2K], params[(2K + 1):end]
    mat = normal_pdf.(data, means', stds' .^ 2) # (N, K)
    return sum(mat .* weights', dims = 2) .|> log |> sum
end

# another form of original objective (doesn't work)
function mixture_loglikelihood2(params::AbstractVector{<:Real}, data::AbstractVector{<:Real})::Real
    K = length(params) ÷ 3
    weights, means, stds = @views params[1:K], params[(K + 1):2K], params[(2K + 1):end]
    mat = normal_pdf.(data, means', stds' .^ 2) # (N, K)
    return obj_true = sum(
        sum(
                weight * normal_pdf(x, mean, std^2)
                for (weight, mean, std) in zip(weights, means, stds)
            ) |> log
            for x in data
    )
end

# objective re-written by me
function mixture_loglikelihood3(params::AbstractVector{<:Real}, data::AbstractVector{<:Real})::Real
    K = length(params) ÷ 3
    weights, means, stds = @views params[1:K], params[(K + 1):2K], params[(2K + 1):end]
    mat = normal_pdf.(data, means', stds' .^ 2) # (N, K)

    obj = zero(eltype(mat))
    for x in data
        obj_i = zero(eltype(mat))
        for (weight, mean, std) in zip(weights, means, stds)
            obj_i += weight * normal_pdf(x, mean, std^2)
        end
        obj += log(obj_i)
    end
    return obj
end

const objective1 = params -> mixture_loglikelihood1(params, data)
const objective2 = params -> mixture_loglikelihood2(params, data)
const objective3 = params -> mixture_loglikelihood3(params, data)

@testset "Type unstable return" begin
    expected = [
        289.7308495620467,
        199.27559524985728,
        236.6894577756876,
        292.0612340227955,
        -9.429799389881452,
        26.722295646439047,
        -1.9180355546752244,
        37.98749089573396,
        -24.095620148778277,
        -13.935687326484112,
        -38.00044665702692,
        12.87712891527131,
    ]
    @test expected ≈ Enzyme.gradient(Reverse, objective1, params0)[1]
    @test expected ≈ Enzyme.gradient(set_runtime_activity(Reverse), objective1, params0)[1]

    # objective2 fails from runtime activity requirements
    # @test expected ≈ Enzyme.gradient(Reverse, objective2, params0)[1]
    @test expected ≈ Enzyme.gradient(Reverse, objective3, params0)[1]
    @test expected ≈ Enzyme.gradient(set_runtime_activity(Reverse), objective3, params0)[1]
end

struct HarmonicAngle
    k::Float64
    t0::Float64
end

function harmonic_g(a, coords_i)
    return (a.k) * a.t0
end

function harmonic_f!(inter_list, coords, inters)
    si = 0.0
    for (i, b) in zip(inter_list, inters)
        si += harmonic_g(b, coords[i])
    end
    return si
end

function invwsumsq(w::AbstractVector, a::AbstractVector)
    s = zero(zero(eltype(a)) / zero(eltype(w)))
    for i in eachindex(w)
        s += abs2(a[i]) / w[i]
    end
    return s
end

_logpdf(d, x) = invwsumsq(d.Σ.diag, x .- d.μ)

function demo_func(x::Any = transpose([1.5 2.0;]))
    m = [-0.30725218207431315, 0.5492115788562757]
    d = (; Σ = LinearAlgebra.Diagonal([1.0, 1.0]), μ = m)
    logp = _logpdf(d, reshape(x, (2,)))
    return logp
end

demof(x) = demo_func()

@testset "Type checks" begin
    x = [0.0, 0.0]
    Enzyme.autodiff(
        Enzyme.Reverse,
        Enzyme.Const(demof),
        Enzyme.Active,
        Enzyme.Duplicated(x, zero(x)),
    )
end

@testset "Decay preservation" begin
    inters = [HarmonicAngle(1.0, 0.1), HarmonicAngle(2.0, 0.3)]
    inter_list = [1, 3]
    dinters = [HarmonicAngle(0.0, 0.0), HarmonicAngle(0.0, 0.0)]
    coords = [(1.0, 2.0, 3.0), (1.1, 2.1, 3.1), (1.2, 2.2, 3.2)]
    d_coords = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]

    autodiff(
        Reverse,
        harmonic_f!,
        Active,
        Const(inter_list),
        Duplicated(coords, d_coords),
        Duplicated(inters, dinters),
    )

    @test dinters[1].k ≈ 0.1
    @test dinters[1].t0 ≈ 1.0
    @test dinters[2].k ≈ 0.3
    @test dinters[2].t0 ≈ 2.0
end

function multisum(M)
  return sum(i -> sum(j -> M[j, i], 1:size(M, 1)), 1:size(M, 2))
end

@testset "Switch decay" begin
    M_sym = Symmetric([0.0 0.5624902964298075 0.017384758763595687 0.7231016696278582; 0.5624902964298075 0.0 0.33640285380605206 0.6064246107556078; 0.017384758763595687 0.33640285380605206 0.0 0.771113711577472; 0.7231016696278582 0.6064246107556078 0.771113711577472 0.0])

    dM_sym = make_zero(M_sym)

    Enzyme.autodiff(Reverse, test, Active, Duplicated(M_sym, dM_sym))

    @test dM_sym ≈ [1.0 2.0 2.0 2.0; 2.0 1.0 2.0 2.0; 2.0 2.0 1.0 2.0; 2.0 2.0 2.0 1.0]
end
