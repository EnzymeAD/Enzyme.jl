# HACK: work around Pkg.jl#2500
if VERSION < v"1.8-"
test_project = Base.active_project()
preferences_file = joinpath(dirname(@__DIR__), "LocalPreferences.toml")
test_preferences_file = joinpath(dirname(test_project), "LocalPreferences.toml")
if isfile(preferences_file) && !isfile(test_preferences_file)
    cp(preferences_file, test_preferences_file)
end
end

using Enzyme
using Test
using FiniteDifferences
using ForwardDiff
using Statistics

using Enzyme_jll
@info "Testing against" Enzyme_jll.libEnzyme

# Test against FiniteDifferences
function test_scalar(f, x; rtol=1e-9, atol=1e-9, fdm=central_fdm(5, 1), kwargs...)
    ∂x, = autodiff(f, Active, Active(x))
    if typeof(x) <: Complex
    else
      @test isapprox(∂x, fdm(f, x); rtol=rtol, atol=atol, kwargs...)
    end
  
    rm = ∂x 
    if typeof(x) <: Integer
        x = Float64(x)
    end
    ∂x, = fwddiff(f, Duplicated(x, one(typeof(x))))
    if typeof(x) <: Complex
      @test ∂x ≈ rm
    else
      @test isapprox(∂x, fdm(f, x); rtol=rtol, atol=atol, kwargs...)
    end
end

include("abi.jl")
include("typetree.jl")

@testset "Internal tests" begin
    f(x) = 1.0 + x
    thunk_a = Enzyme.Compiler.thunk(f, nothing, Active, Tuple{Active{Float64}}, Val(API.DEM_ReverseModeCombined), Val(1))
    thunk_b = Enzyme.Compiler.thunk(f, nothing, Const, Tuple{Const{Float64}}, Val(API.DEM_ReverseModeCombined), Val(1))
    thunk_c = Enzyme.Compiler.thunk(f, nothing, Active{Float64}, Tuple{Active{Float64}}, Val(API.DEM_ReverseModeCombined), Val(1))
    @test thunk_a.adjoint !== thunk_b.adjoint
    @test thunk_c.adjoint === thunk_a.adjoint

    @test thunk_a(Active(2.0), 1.0) == (1.0,)
    @test thunk_a(Active(2.0), 2.0) == (2.0,)
    @test thunk_b(Const(2.0)) === ()

    forward, pullback = Enzyme.Compiler.thunk(f, nothing, Active, Tuple{Active{Float64}}, Val(Enzyme.API.DEM_ReverseModeGradient), Val(1))
    # @test thunk_split.primal !== C_NULL
    # @test thunk_split.primal !== thunk_split.adjoint
    # @test thunk_a.adjoint !== thunk_split.adjoint
end

@testset "Reflection" begin
    Enzyme.Compiler.enzyme_code_typed(Active, Tuple{Active{Float64}}) do x
        x ^ 2
    end
    f(x) = 1.0 + x
    sprint() do io
        Enzyme.Compiler.enzyme_code_native(io, f, Active, Tuple{Active{Float64}})
    end

    sprint() do io
        Enzyme.Compiler.enzyme_code_llvm(io, f, Active, Tuple{Active{Float64}})
    end
end


# @testset "Split Tape" begin
#     f(x) = x[1] * x[1]

#     thunk_split = Enzyme.Compiler.thunk(f, Tuple{Duplicated{Array{Float64,1}}}, Val(Enzyme.API.DEM_ReverseModeGradient))
#     @test thunk_split.primal !== C_NULL
#     @test thunk_split.primal !== thunk_split.adjoint
# end

@testset "Simple tests" begin
    f1(x) = 1.0 + x
    f2(x) = x*x
    @test autodiff(f1, Active, Active(1.0))[1] ≈ 1.0
    @test fwddiff(f1, DuplicatedNoNeed, Duplicated(1.0, 1.0))[1] ≈ 1.0
    @test fwddiff(f1, Duplicated, Duplicated(1.0, 1.0))[2] ≈ 1.0
    @test autodiff(f2, Active, Active(1.0))[1] ≈ 2.0
    @test fwddiff(f2, Duplicated(1.0, 1.0))[1] ≈ 2.0
    @test autodiff(tanh, Active, Active(1.0))[1] ≈ 0.41997434161402606939
    @test fwddiff(tanh, Duplicated(1.0, 1.0))[1] ≈ 0.41997434161402606939
    @test autodiff(tanh, Active, Active(1.0f0))[1] ≈ Float32(0.41997434161402606939)
    @test fwddiff(tanh, Duplicated(1.0f0, 1.0f0))[1] ≈ Float32(0.41997434161402606939)
    test_scalar(f1, 1.0)
    test_scalar(f2, 1.0)
    test_scalar(log2, 1.0)
    test_scalar(log10, 1.0)

    @test autodiff((x)->log(x), Active(2.0)) == (0.5,)
end

@testset "Simple Exception" begin
    f_simple_exc(x, i) = ccall(:jl_, Cvoid, (Any,), x[i])
    y = [1.0, 2.0]
    f_x = zero.(y)
    @test_throws BoundsError autodiff(f_simple_exc, Duplicated(y, f_x), 0)
end


@testset "Duplicated" begin
    x = Ref(1.0)
    y = Ref(2.0)

    ∇x = Ref(0.0)
    ∇y = Ref(0.0)

    autodiff((a,b)->a[]*b[], Active, Duplicated(x, ∇x), Duplicated(y, ∇y))

    @test ∇y[] == 1.0
    @test ∇x[] == 2.0
end

@testset "Simple tests" begin
    g(x) = real((x + im)*(1 - im*x))
    @test first(autodiff(g, Active, Active(2.0))) ≈ 2.0
    @test first(fwddiff(g, Duplicated(2.0, 1.0))) ≈ 2.0
    @test first(autodiff(g, Active, Active(3.0))) ≈ 2.0
    @test first(fwddiff(g, Duplicated(3.0, 1.0))) ≈ 2.0
    test_scalar(g, 2.0)
    test_scalar(g, 3.0)
end

@testset "Taylor series tests" begin

# Taylor series for `-log(1-x)`
# eval at -log(1-1/2) = -log(1/2)
function euroad(f::T) where T
    g = zero(T)
    for i in 1:10^7
        g += f^i / i
    end
    return g
end

euroad′(x) = first(autodiff(euroad, Active, Active(x)))

@test euroad(0.5) ≈ -log(0.5) # -log(1-x)
@test euroad′(0.5) ≈ 2.0 # d/dx -log(1-x) = 1/(1-x)
test_scalar(euroad, 0.5)
end

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
    autodiff(arsum, Active, Duplicated(inp, dinp))
    @test inp ≈ Float64[1.0, 2.0]
    @test dinp ≈ Float64[1.0, 1.0]
    
    @test fwddiff(arsum, Duplicated(inp, dinp))[1] ≈ 2.0 
end

@testset "Advanced array tests" begin
    function arsum2(f::Array{T}) where T
        return sum(f)
    end
    inp = Float64[1.0, 2.0]
    dinp = Float64[0.0, 0.0]
    autodiff(arsum2, Active, Duplicated(inp, dinp))
    @test inp ≈ Float64[1.0, 2.0]
    @test dinp ≈ Float64[1.0, 1.0]
    
    @test fwddiff(arsum2, Duplicated(inp, dinp))[1] ≈ 2.0
end

@testset "Dict" begin
    params = Dict{Symbol, Float64}()
    dparams = Dict{Symbol, Float64}()

    params[:var] = 10.0
    dparams[:var] = 0.0

    f_dict(params, x) = params[:var] * x

    @test autodiff(f_dict, Const(params), Active(5.0)) == (10.0,)
    @test autodiff(f_dict, Duplicated(params, dparams), Active(5.0)) == (10.0,)
    @test dparams[:var] == 5.0
end

function grad_closure(f, x)
    function noretval(x,res)
        y = f(x)
        copyto!(res,y)
        return nothing
    end
    n = length(x)
    dx = zeros(n)
    y  = zeros(n)
    dy = zeros(n)
    dy[1] = 1.0

    autodiff(noretval, Duplicated(x,dx), Duplicated(y, dy))
    return dx
end

@testset "Closure" begin
    x = [2.0,6.0]
    dx = grad_closure(x->[x[1], x[2]], x)
    @test dx == [1.0, 0.0]
end


@testset "Bithacks" begin
    function fneg(x::Float64)
        xptr = reinterpret(Int64, x)
        y = Int64(-9223372036854775808)
        out = y ⊻ xptr;
        return reinterpret(Float64, out)
    end
    @test autodiff(fneg, Active, Active(2.0))[1] ≈ -1.0
    @test fwddiff(fneg, Duplicated(2.0, 1.0))[1] ≈ -1.0
    function expor(x::Float64)
        xptr = reinterpret(Int64, x)
        y = UInt64(4607182418800017408)
        out = y | xptr;
        return reinterpret(Float64, out)
    end
    @test autodiff(expor, Active, Active(0.42))[1] ≈ 4.0
    @test fwddiff(expor, Duplicated(0.42, 1.0))[1] ≈ 4.0
end

@testset "GC" begin
    function gc_alloc(x)  # Basically g(x) = x^2
        a = Array{Float64, 1}(undef, 10)
        for n in 1:length(a)
            a[n] = x^2
        end
        return mean(a)
    end
    @test autodiff(gc_alloc, Active, Active(5.0))[1] ≈ 10
    @test fwddiff(gc_alloc, Duplicated(5.0, 1.0))[1] ≈ 10

    # TODO (after BLAS)
    # A = Vector[2.0, 3.0]
    # B = Vector[4.0, 5.0]
    # dB = Vector[0.0, 0.0]
    # f = (X, Y) -> sum(X .* Y)
    # Enzyme.autodiff(f, Active, A, Duplicated(B, dB))

    function gc_copy(x)  # Basically g(x) = x^2
        a = x * ones(10)
        for n in 1:length(a)
            a[n] = x^2
        end
        return mean(a)
    end
    # TODO(wsmoses): Assertion failed: (pp->getNumUses() == 0), function eraseFictiousPHIs
    # @test Enzyme.autodiff(gc_copy, Active, Active(5.0))[1] ≈ 10
    # TODO: https://github.com/wsmoses/Enzyme/issues/393
    # @test Enzyme.fwddiff(gc_copy, Duplicated(5.0, 1.0))[1] ≈ 10
end


@testset "Compare against" begin
    x = 3.0
    fd = central_fdm(5, 1)(sin, x)

    @test fd ≈ ForwardDiff.derivative(sin, x)
    @test fd ≈ first(autodiff(sin, Active, Active(x)))
    @test fd ≈ first(fwddiff(sin, Duplicated(x, 1.0)))

    x = 0.2 + sin(3.0)
    fd = central_fdm(5, 1)(asin, x)

    @test fd ≈ ForwardDiff.derivative(asin, x)
    @test fd ≈ first(autodiff(asin, Active, Active(x)))
    @test fd ≈ first(fwddiff(asin, Duplicated(x, 1.0)))
    test_scalar(asin, x)

    function foo(x)
        a = sin(x)
        b = 0.2 + a
        c = asin(b)
        return c
    end

    x = 3.0
    fd = central_fdm(5, 1)(foo, x)

    @test fd ≈ ForwardDiff.derivative(foo, x)
    @test fd ≈ first(autodiff(foo, Active, Active(x)))
    @test fd ≈ first(fwddiff(foo, Duplicated(x, 1.0)))
    test_scalar(foo, x)

    # Input type shouldn't matter
    x = 3
    @test fd ≈ ForwardDiff.derivative(foo, x)
    @test fd ≈ first(autodiff(foo, Active, Active(x)))
    # They do matter for duplicated, which can't be auto promoted
    # @test fd ≈ first(fwddiff(foo, Duplicated(x, 1)))

    f74(a, c) = a * √c
    @test √3 ≈ first(autodiff(f74, Active, Active(2), 3))
    @test √3 ≈ first(fwddiff(f74, Duplicated(2.0, 1.0), 3))
end

@testset "SinCos" begin
	function sumsincos(theta)
		a, b = sincos(theta)
		return a + b
	end
    test_scalar(sumsincos, 1.0, rtol=1e-5, atol=1e-5)
end

"""
    J(ν, z) := ∑ (−1)^k / Γ(k+1) / Γ(k+ν+1) * (z/2)^(ν+2k)
"""
function mybesselj(ν, z, atol=1e-8)
    k = 0
    s = (z/2)^ν / factorial(ν)
    out = s
    while abs(s) > atol
        k += 1
        s *= (-1) / k / (k+ν) * (z/2)^2
        out += s
    end
    out
end
mybesselj0(z) = mybesselj(0, z)
mybesselj1(z) = mybesselj(1, z)

@testset "Bessel" begin
    autodiff(mybesselj, Active, Const(0), Active(1.0))
    autodiff(mybesselj, Active, 0, Active(1.0))
    fwddiff(mybesselj, Const(0), Duplicated(1.0, 1.0))
    fwddiff(mybesselj, 0, Duplicated(1.0, 1.0))
    @testset "besselj0/besselj1" for x in (1.0, -1.0, 0.0, 0.5, 10, -17.1,) # 1.5 + 0.7im)
        test_scalar(mybesselj0, x, rtol=1e-5, atol=1e-5)
        test_scalar(mybesselj1, x, rtol=1e-5, atol=1e-5)
    end
end

## https://github.com/JuliaDiff/ChainRules.jl/tree/master/test/rulesets
if !Sys.iswindows()
    include("packages/specialfunctions.jl")
end

@testset "Threads" begin
    cmd = `$(Base.julia_cmd()) --threads=1 --startup-file=no threads.jl`
   	@test success(pipeline(cmd, stderr=stderr, stdout=stdout))
    cmd = `$(Base.julia_cmd()) --threads=2 --startup-file=no threads.jl`
   	@test success(pipeline(cmd, stderr=stderr, stdout=stdout))
end

@testset "DiffTest" begin
    include("DiffTests.jl")

    n = rand()
    x, y = rand(5, 5), rand(26)
    A, B = rand(5, 5), rand(5, 5)

    # f returns Number
    @testset "Number to Number" for f in DiffTests.NUMBER_TO_NUMBER_FUNCS
        test_scalar(f, n)
    end

    # TODO(vchuravy/wsmoses): Enable these tests
    # for f in DiffTests.VECTOR_TO_NUMBER_FUNCS
    #     @test isa(f(y), Number)
    # end

    # for f in DiffTests.MATRIX_TO_NUMBER_FUNCS
    #     @test isa(f(x), Number)
    # end

    # for f in DiffTests.TERNARY_MATRIX_TO_NUMBER_FUNCS
    #     @test isa(f(A, B, x), Number)
    # end

    # # f returns Array

    # for f in DiffTests.NUMBER_TO_ARRAY_FUNCS
    #     @test isa(f(n), Array)
    # end

    # for f in DiffTests.ARRAY_TO_ARRAY_FUNCS
    #     @test isa(f(A), Array)
    #     @test isa(f(y), Array)
    # end

    # for f in DiffTests.MATRIX_TO_MATRIX_FUNCS
    #     @test isa(f(A), Array)
    # end

    # for f in DiffTests.BINARY_MATRIX_TO_MATRIX_FUNCS
    #     @test isa(f(A, B), Array)
    # end

    # # f! returns Nothing

    # for f! in DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS
    #     @test isa(f!(y, x), Nothing)
    # end

    # for f! in DiffTests.INPLACE_NUMBER_TO_ARRAY_FUNCS
    #     @test isa(f!(y, n), Nothing)
    # end

end

@testset "IO" begin

    function printsq(x)
        println(x)
        x*x
    end

    @test 4.6 ≈ first(autodiff(printsq, Active, Active(2.3)))
    @test 4.6 ≈ first(fwddiff(printsq, Duplicated(2.3, 1.0)))

    function tostring(x)
        string(x)
        x*x
    end

    @test 4.6 ≈ first(autodiff(tostring, Active, Active(2.3)))
    @test 4.6 ≈ first(fwddiff(tostring, Duplicated(2.3, 1.0)))
end

@testset "hmlstm" begin
    sigm(x)  = @fastmath 1 / (1 + exp(-x))
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
    Base.invokelatest(sin, x)::Float64 + 1
end

function loadsin(xp)
    x = @inbounds xp[1]
    @inbounds xp[1] = 0.0
    sin(x)
end
function invsin(xp)
    xp = Base.invokelatest(convert, Vector{Float64}, xp)
    loadsin(xp)
end

@testset "generic" begin
    @test -0.4161468365471424 ≈ Enzyme.autodiff(genlatestsin, Active, Active(2.0))[1]
    @test -0.4161468365471424 ≈ Enzyme.fwddiff(genlatestsin, Duplicated(2.0, 1.0))[1]

    x = [2.0]
    dx = [0.0]
    Enzyme.autodiff(genlatestsinx, Active, Duplicated(x, dx))
    @test 0 ≈ x[1]
    @test -0.4161468365471424 ≈ dx[1]

    x = [2.0]
    dx = [0.0]
    Enzyme.autodiff(invsin, Active, Duplicated(x, dx))
    @test 0 ≈ x[1]
    @test -0.4161468365471424 ≈ dx[1]
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

    x  = [2.0]
    dx = [1.0]

    Enzyme.autodiff(invtest, Duplicated(x, dx))
    
    @test 10.0 ≈ x[1]
    @test 5.0 ≈ dx[1]
end

@testset "broadcast" begin
    A = rand(10); B = rand(10); R = similar(A)
    dA = zero(A); dB = zero(B); dR = fill!(similar(R), 1)

    function foo_bc!(R, A, B)
        R .= A .+ B
        return nothing
    end

    autodiff(foo_bc!, Const, Duplicated(R, dR), Duplicated(A, dA), Duplicated(B, dB))

    # works since aliasing is "simple"
    autodiff(foo_bc!, Const, Duplicated(R, dR), Duplicated(R, dR), Duplicated(B, dB))

    A = rand(10,10); B = rand(10, 10)
    dA = zero(A); dB = zero(B); dR = fill!(similar(A), 1)

    autodiff(foo_bc!, Const, Duplicated(A, dR), Duplicated(transpose(A), transpose(dA)), Duplicated(B, dB))
end


@testset "DuplicatedReturn" begin
    moo(x) = fill(x, 10)

    @test_throws ErrorException autodiff(moo, Active(2.1))
    fo, = fwddiff(moo, Duplicated(2.1, 1.0))
    for i in 1:10
        @test 1.0 ≈ fo[i]
    end
end

@testset "GCPreserve" begin
    function f(x, y)
        GC.@preserve x y begin
            ccall(:memcpy, Cvoid,
                (Ptr{Float64},Ptr{Float64},Csize_t), x, y, 8)
        end
        nothing
    end
    autodiff(f, Duplicated([1.0], [0.0]), Duplicated([1.0], [0.0]))
    fwddiff(f, Duplicated([1.0], [0.0]), Duplicated([1.0], [0.0]))
end

@testset "GCPreserve2" begin
    function f!(a_out, a_in)
           a_out[1:end-1] .= a_in[2:end]
           return nothing
    end
    a_in = rand(4)
    a_out = a_in

    shadow_a_out = ones(4)
    shadow_a_in = shadow_a_out

    autodiff(f!, Const, Duplicated(a_out, shadow_a_out), Duplicated(a_in, shadow_a_in))
    
    @test shadow_a_in ≈ Float64[0.0, 1.0, 1.0, 2.0]
    @test shadow_a_out ≈ Float64[0.0, 1.0, 1.0, 2.0]
    
    fwddiff(f!, Const, Duplicated(a_out, shadow_a_out), Duplicated(a_in, shadow_a_in))
    
    @test shadow_a_in ≈ Float64[1.0, 1.0, 2.0, 2.0]
    @test shadow_a_out ≈ Float64[1.0, 1.0, 2.0, 2.0]
end

@testset "UndefVar" begin
    function f(x, y)
        if x
            undefinedfnthowmagic()
        end
        y
    end
    @test 1.0 ≈ autodiff(f, false, Active(2.14))[1]
    @test_throws Base.UndefVarError autodiff(f, true, Active(2.14))
    
    @test 1.0 ≈ fwddiff(f, false, Duplicated(2.14, 1.0))[1]
    @test_throws Base.UndefVarError fwddiff(f, true, Duplicated(2.14, 1.0))

    function foo(x, y)
        if x
            Threads.@threads for N in 1:5:20
                println("The number of this iteration is $N")
            end
        end
        y
    end
    @test 1.0 ≈ autodiff(foo, false, Active(2.14))[1]
    @test 1.0 ≈ fwddiff(foo, false, Duplicated(2.14, 1.0))[1]
end

@testset "Return GC error" begin
	t = 0.0

	function tobedifferentiated(cond, a)::Float64
		if cond
			t + t
		else
			0.0
		end
	end

	@test 0.0 ≈ autodiff(tobedifferentiated, true, Active(2.1))[1]
	@test 0.0 ≈ fwddiff(tobedifferentiated, true, Duplicated(2.1, 1.0))[1]
	
	function tobedifferentiated2(cond, a)::Float64
		if cond
			a + t
		else
			0.0
		end
	end

	@test 1.0 ≈ autodiff(tobedifferentiated2, true, Active(2.1))[1]
	@test 1.0 ≈ fwddiff(tobedifferentiated2, true, Duplicated(2.1, 1.0))[1]

    @noinline function copy(dest, p1, cond)
        bc = convert(Broadcast.Broadcasted{Nothing}, Broadcast.instantiate(p1))

        if cond
            return nothing
        end

        bc2 = Broadcast.preprocess(dest, bc)
        @inbounds    dest[1] = bc2[1]

        nothing
    end

    function mer(F, F_H, cond)
        p1 = Base.broadcasted(Base.identity, F_H)
        copy(F, p1, cond)

        # Force an apply generic
        flush(stdout)
        nothing
    end

    L_H = Array{Float64, 1}(undef, 2)
    L = Array{Float64, 1}(undef, 2)

    F_H = [1.0, 0.0]
    F = [1.0, 0.0]

    autodiff(mer, Duplicated(F, L), Duplicated(F_H, L_H), true)
    fwddiff(mer, Duplicated(F, L), Duplicated(F_H, L_H), true)
end


@testset "Split GC" begin
    @noinline function bmat(x)
        data = [x]
        return data
    end

    function f(x::Float64)
        @inbounds return bmat(x)[1]
    end
    @test 1.0 ≈ autodiff(f, Active(0.1))[1]
    @test 1.0 ≈ fwddiff(f, Duplicated(0.1, 1.0))[1]
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
	autodiff(copytest, Duplicated(F, dF))
	@test F ≈ [1.234, 5.678] 
	@test dF ≈ [3.0, 2.0]
	
    @test 31.0 ≈ fwddiff(copytest, Duplicated([2.0, 3.0], [7.0, 5.0]))[1]
end

@testset "No inference" begin
    c = 5.0
    @test 5.0 ≈ autodiff((A,)->c * A, Active, Active(2.0))[1]
    @test 5.0 ≈ fwddiff((A,)->c * A, Duplicated(2.0, 1.0))[1]
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
        autodiff(tobedifferentiated, Duplicated(F, L), false)
        fwddiff(tobedifferentiated, Duplicated(F, L), false)
    end

    main()
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
    @test Enzyme.autodiff(timsteploop_scalar, Active(2.0))[1] ≈ 1.0
    @test Enzyme.fwddiff(timsteploop_scalar, Duplicated(2.0, 1.0))[1] ≈ 1.0

    @noinline function func(X)
        return @inbounds X[1]
    end
    function timsteploop(FH1)
        G = Float64[FH1]
        k1 = func(G)
        return k1
    end
    @test Enzyme.autodiff(timsteploop, Active(2.0))[1] ≈ 1.0
    @test Enzyme.fwddiff(timsteploop, Duplicated(2.0, 1.0))[1] ≈ 1.0
end

@testset "Type" begin
    function foo(in::Ptr{Cvoid}, out::Ptr{Cvoid})
        markType(Float64, in)
        ccall(:memcpy,Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t), out, in, 8)
    end

    x = [2.0]
    y = [3.0]
    dx = [5.0]
    dy = [7.0]

    @test markType(x) === nothing
    @test markType(zeros(Float32, 64)) === nothing
    @test markType(view(zeros(64), 16:32)) === nothing

    GC.@preserve x y begin
        foo(Base.unsafe_convert(Ptr{Cvoid}, x), Base.unsafe_convert(Ptr{Cvoid}, y))
    end

    GC.@preserve x y dx dy begin
      autodiff(foo,
                Duplicated(Base.unsafe_convert(Ptr{Cvoid}, x), Base.unsafe_convert(Ptr{Cvoid}, dx)), 
                Duplicated(Base.unsafe_convert(Ptr{Cvoid}, y), Base.unsafe_convert(Ptr{Cvoid}, dy)))
    end
end

@testset "BLAS" begin
    x = [2.0, 3.0]
    dx = [0.2,0.3]
    y = [5.0, 7.0]
    dy = [0.5,0.7]
    Enzyme.autodiff((x,y)->x' * y, Duplicated(x, dx), Duplicated(y, dy))
    @show x, dx, y, dy
    @test dx ≈ [5.2, 7.3]
    @test dy ≈ [2.5, 3.7]
end

@testset "Exception" begin
    f_exc(x) = sum(x*x)
    y = [[1.0, 2.0] [3.0,4.0]]
    f_x = zero.(y)
    @test_throws Enzyme.Compiler.NoDerivativeException autodiff(f_exc, Duplicated(y, f_x))

    f_no_derv(x) = ccall("extern doesnotexist", llvmcall, Float64, (Float64,), x)
    @test_throws Enzyme.Compiler.NoDerivativeException autodiff(f_no_derv, Active, Active(0.5))

    f_union(cond, x) = cond ? x : 0
    g_union(cond, x) = f_union(cond,x)*x
    @test_throws Enzyme.Compiler.IllegalTypeAnalysisException autodiff(g_union, Active, true, Active(1.0))

    # TODO: Add test for NoShadowException
end

@testset "Array push" begin

    function pusher(x, y)
        push!(x, y)
        x[1] + x[2]
    end

    x  = [2.3]
    dx = [0.0]
    @test 1.0 ≈ first(Enzyme.autodiff(pusher, Duplicated(x, dx), Active(2.0)))
    @test x ≈ [2.3, 2.0]
    @test dx ≈ [1.0]
end

@testset "Batch" begin
    square(x)=x*x
    bres = fwddiff(square, BatchDuplicatedNoNeed, BatchDuplicated(3.0, (1.0, 2.0, 3.0)))
    @test length(bres) == 1
    @test length(bres[1]) == 3
    @test bres ≈ ((6.0, 12.0, 18.0),)

    bres = fwddiff(square, BatchDuplicatedNoNeed, BatchDuplicated(3.0 + 7.0im, (1.0+0im, 2.0+0im, 3.0+0im)))
    @test bres ≈ ((6.0 + 14.0im, 12.0 + 28.0im, 18.0 + 42.0im),)

    squareidx(x)=x[1]*x[1]
    inp = Float32[3.0]

    # Shadow offset is not the same as primal so following doesn't work
    # d_inp = Float32[1.0, 2.0, 3.0]
    # fwddiff(squareidx, BatchDuplicatedNoNeed, BatchDuplicated(view(inp, 1:1), (view(d_inp, 1:1), view(d_inp, 2:2), view(d_inp, 3:3))))

    d_inp = (Float32[1.0], Float32[2.0], Float32[3.0])
    bres = fwddiff(squareidx, BatchDuplicatedNoNeed, BatchDuplicated(inp, d_inp))
    @test bres ≈ ((6.0, 12.0, 18.0),)
end
