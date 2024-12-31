# # work around https://github.com/JuliaLang/Pkg.jl/issues/1585
# using Pkg
# Pkg.develop(PackageSpec(; path=joinpath(dirname(@__DIR__), "lib", "EnzymeTestUtils")))

using GPUCompiler
using Enzyme
using Test
using FiniteDifferences
using Aqua
using Statistics
using LinearAlgebra
using InlineStrings

using Enzyme_jll
@info "Testing against" Enzyme_jll.libEnzyme

function isapproxfn(fn, args...; kwargs...)
    isapprox(args...; kwargs...)
end
# Test against FiniteDifferences
function test_scalar(f, x; rtol=1e-9, atol=1e-9, fdm=central_fdm(5, 1), kwargs...)
    ∂x, = autodiff(ReverseHolomorphic, f, Active, Active(x))[1]

    finite_diff = if typeof(x) <: Complex
      RT = typeof(x).parameters[1]
      (fdm(dx -> f(x+dx), RT(0)) - im * fdm(dy -> f(x+im*dy), RT(0)))/2
    else
      fdm(f, x)
    end

    @test isapproxfn((Enzyme.Reverse, f), ∂x, finite_diff; rtol=rtol, atol=atol, kwargs...)

    if typeof(x) <: Integer
        x = Float64(x)
    end

    if typeof(x) <: Complex
        ∂re, = autodiff(Forward, f, Duplicated(x, one(typeof(x))))
        ∂im, = autodiff(Forward, f, Duplicated(x, im*one(typeof(x))))
        ∂x = (∂re - im*∂im)/2
    else
        ∂x, = autodiff(Forward, f, Duplicated(x, one(typeof(x))))
    end

    @test isapproxfn((Enzyme.Reverse, f), ∂x, finite_diff; rtol=rtol, atol=atol, kwargs...)

end

function test_matrix_to_number(f, x; rtol=1e-9, atol=1e-9, fdm=central_fdm(5, 1), kwargs...)
    dx_fd = map(eachindex(x)) do i
        fdm(x[i]) do xi
            x2 = copy(x)
            x2[i] = xi
            f(x2)
        end
    end

    dx = zero(x)
    autodiff(Reverse, f, Active, Duplicated(x, dx))
    @test isapproxfn((Enzyme.Reverse, f), reshape(dx, length(dx)), dx_fd; rtol=rtol, atol=atol, kwargs...)

    dx_fwd = map(eachindex(x)) do i
        dx = zero(x)
        dx[i] = 1
        ∂x = autodiff(Forward, f, Duplicated(x, dx))
        isempty(∂x) ? zero(eltype(dx)) : ∂x[1]
    end
    @test isapproxfn((Enzyme.Forward, f), dx_fwd, dx_fd; rtol=rtol, atol=atol, kwargs...)
end

# Aqua.test_all(Enzyme, unbound_args=false, piracies=false, deps_compat=false, stale_deps=(;:ignore=>[:EnzymeTestUtils]))
# Aqua.test_all(Enzyme, unbound_args=false, piracies=false, deps_compat=false, stale_deps=(;:ignore=>[:EnzymeTestUtils]))

include("abi.jl")
include("typetree.jl")
include("passes.jl")
include("optimize.jl")
include("make_zero.jl")

include("rules.jl")
include("rrules.jl")
include("kwrules.jl")
include("kwrrules.jl")
include("internal_rules.jl")
include("ruleinvalidation.jl")
include("typeunstable.jl")
include("absint.jl")

@static if !Sys.iswindows()
    include("blas.jl")
end

f0(x) = 1.0 + x
function vrec(start, x)
    if start > length(x)
        return 1.0
    else
        return x[start] * vrec(start+1, x)
    end
end

struct Ints{A, B}
    v::B
    q::Int
end

mutable struct MInts{A, B}
    v::B
    q::Int
end

@testset "Internal tests" begin
    @static if VERSION < v"1.11-"
    else
    @assert Enzyme.Compiler.active_reg_inner(Memory{Float64}, (), nothing) == Enzyme.Compiler.DupState
    end
    @assert Enzyme.Compiler.active_reg_inner(Type{Array}, (), nothing) == Enzyme.Compiler.AnyState
    @assert Enzyme.Compiler.active_reg_inner(Ints{<:Any, Integer}, (), nothing) == Enzyme.Compiler.AnyState
    @assert Enzyme.Compiler.active_reg_inner(Ints{<:Any, Float64}, (), nothing) == Enzyme.Compiler.DupState
    @assert Enzyme.Compiler.active_reg_inner(Ints{Integer, <:Any}, (), nothing) == Enzyme.Compiler.DupState
    @assert Enzyme.Compiler.active_reg_inner(Ints{Integer, <:Integer}, (), nothing) == Enzyme.Compiler.AnyState
    @assert Enzyme.Compiler.active_reg_inner(Ints{Integer, <:AbstractFloat}, (), nothing) == Enzyme.Compiler.DupState
    @assert Enzyme.Compiler.active_reg_inner(Ints{Integer, Float64}, (), nothing) == Enzyme.Compiler.ActiveState
    @assert Enzyme.Compiler.active_reg_inner(MInts{Integer, Float64}, (), nothing) == Enzyme.Compiler.DupState

    @assert Enzyme.Compiler.active_reg(Tuple{Float32,Float32,Int})
    @assert !Enzyme.Compiler.active_reg(Tuple{NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}})
    @assert !Enzyme.Compiler.active_reg(Base.RefValue{Float32})
    @assert Enzyme.Compiler.active_reg_inner(Ptr, (), nothing) == Enzyme.Compiler.DupState
    @assert Enzyme.Compiler.active_reg_inner(Base.RefValue{Float32}, (), nothing) == Enzyme.Compiler.DupState
    @assert Enzyme.Compiler.active_reg_inner(Colon, (), nothing) == Enzyme.Compiler.AnyState
    @assert Enzyme.Compiler.active_reg_inner(Symbol, (), nothing) == Enzyme.Compiler.AnyState
    @assert Enzyme.Compiler.active_reg_inner(String, (), nothing) == Enzyme.Compiler.AnyState
    @assert Enzyme.Compiler.active_reg_inner(Tuple{Any,Int64}, (), nothing) == Enzyme.Compiler.DupState
    @assert Enzyme.Compiler.active_reg_inner(Tuple{S,Int64} where S, (), Base.get_world_counter()) == Enzyme.Compiler.DupState
    @assert Enzyme.Compiler.active_reg_inner(Union{Float64,Nothing}, (), nothing) == Enzyme.Compiler.DupState
    @assert Enzyme.Compiler.active_reg_inner(Union{Float64,Nothing}, (), nothing, #=justActive=#Val(false), #=unionSret=#Val(true)) == Enzyme.Compiler.ActiveState
    @test Enzyme.Compiler.active_reg_inner(Tuple, (), nothing) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg_inner(Tuple, (), nothing, #=justactive=#Val(false), #=unionsret=#Val(false), #=abstractismixed=#Val(true)) == Enzyme.Compiler.MixedState
    @test Enzyme.Compiler.active_reg_inner(Tuple{A,A} where A, (), nothing, #=justactive=#Val(false), #=unionsret=#Val(false), #=abstractismixed=#Val(true)) == Enzyme.Compiler.MixedState

    # issue #1935
    struct Incomplete
        x::Float64
        y
        Incomplete(x) = new(x)
        # incomplete constructor & non-bitstype field => !Base.allocatedinline(Incomplete)
    end
    @test Enzyme.Compiler.active_reg_inner(Tuple{Incomplete}, (), nothing, #=justActive=#Val(false)) == Enzyme.Compiler.MixedState
    @test Enzyme.Compiler.active_reg_inner(Tuple{Incomplete}, (), nothing, #=justActive=#Val(true)) == Enzyme.Compiler.ActiveState

    thunk_a = Enzyme.Compiler.thunk(Val(0), Const{typeof(f0)}, Active, Tuple{Active{Float64}}, Val(API.DEM_ReverseModeCombined), Val(1), Val((false, false)), Val(false), Val(false), DefaultABI, Val(false), Val(false))
    thunk_b = Enzyme.Compiler.thunk(Val(0), Const{typeof(f0)}, Const, Tuple{Const{Float64}}, Val(API.DEM_ReverseModeCombined), Val(1), Val((false, false)), Val(false), Val(false), DefaultABI, Val(false), Val(false))
    thunk_c = Enzyme.Compiler.thunk(Val(0), Const{typeof(f0)}, Active{Float64}, Tuple{Active{Float64}}, Val(API.DEM_ReverseModeCombined), Val(1), Val((false, false)), Val(false), Val(false), DefaultABI, Val(false), Val(false))
    thunk_d = Enzyme.Compiler.thunk(Val(0), Const{typeof(f0)}, Active{Float64}, Tuple{Active{Float64}}, Val(API.DEM_ReverseModeCombined), Val(1), Val((false, false)), Val(false), Val(false), DefaultABI, Val(false), Val(false))
    @test thunk_a.adjoint !== thunk_b.adjoint
    @test thunk_c.adjoint === thunk_a.adjoint
    @test thunk_c.adjoint === thunk_d.adjoint

    @test thunk_a(Const(f0), Active(2.0), 1.0) == ((1.0,),)
    @test thunk_a(Const(f0), Active(2.0), 2.0) == ((2.0,),)
    @test thunk_b(Const(f0), Const(2.0)) === ((nothing,),)

    forward, pullback = Enzyme.Compiler.thunk(Val(0), Const{typeof(f0)}, Active, Tuple{Active{Float64}}, Val(Enzyme.API.DEM_ReverseModeGradient), Val(1), Val((false, false)), Val(false), Val(false), DefaultABI, Val(false), Val(false))

    @test forward(Const(f0), Active(2.0)) == (nothing,nothing,nothing)
    @test pullback(Const(f0), Active(2.0), 1.0, nothing) == ((1.0,),)

    function mul2(x)
        x[1] * x[2]
    end
    d = Duplicated([3.0, 5.0], [0.0, 0.0])

    forward, pullback = Enzyme.Compiler.thunk(Val(0), Const{typeof(mul2)}, Active, Tuple{Duplicated{Vector{Float64}}}, Val(Enzyme.API.DEM_ReverseModeGradient), Val(1), Val((false, true)), Val(false), Val(false), DefaultABI, Val(false), Val(false))
    res = forward(Const(mul2), d)

    @static if VERSION < v"1.11-"
    @test typeof(res[1]) == Tuple{Float64, Float64}
    else
    @test typeof(res[1]) == NamedTuple{(Symbol("1"),Symbol("2"),Symbol("3")), Tuple{Any, Float64, Float64}}
    end

    pullback(Const(mul2), d, 1.0, res[1])
    @test d.dval[1] ≈ 5.0
    @test d.dval[2] ≈ 3.0

    d = Duplicated([3.0, 5.0], [0.0, 0.0])
    forward, pullback = Enzyme.Compiler.thunk(Val(0), Const{typeof(vrec)}, Active, Tuple{Const{Int}, Duplicated{Vector{Float64}}}, Val(Enzyme.API.DEM_ReverseModeGradient), Val(1), Val((false, false, true)), Val(false), Val(false), DefaultABI, Val(false), Val(false))
    res = forward(Const(vrec), Const(Int(1)), d)
    pullback(Const(vrec), Const(1), d, 1.0, res[1])
    @test d.dval[1] ≈ 5.0
    @test d.dval[2] ≈ 3.0

    # @test thunk_split.primal !== C_NULL
    # @test thunk_split.primal !== thunk_split.adjoint
    # @test thunk_a.adjoint !== thunk_split.adjoint
    #
    z = ([3.14, 21.5, 16.7], [0,1], [5.6, 8.9])
    Enzyme.make_zero!(z)
    @test z[1] ≈ [0.0, 0.0, 0.0]
    @test z[2][1] == 0
    @test z[2][2] == 1
    @test z[3] ≈ [0.0, 0.0]
    
    z2 = ([3.14, 21.5, 16.7], [0,1], [5.6, 8.9])
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
        x ^ 2
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
sqrtsumsq2(x) = (sum(abs2, x)*sum(abs2,x))
@testset "Recursion optimization" begin
    # Test that we can successfully optimize out the augmented primal from the recursive divide and conquer
    fn = sprint() do io
       Enzyme.Compiler.enzyme_code_llvm(io, sum, Active, Tuple{Duplicated{Vector{Float64}}}; dump_module=true)
    end
    @test occursin("diffe",fn)
    # TODO we need to fix julia to remove unused bounds checks
    # @test !occursin("aug",fn)
    
    fn = sprint() do io
       Enzyme.Compiler.enzyme_code_llvm(io, sumsq2, Active, Tuple{Duplicated{Vector{Float64}}}; dump_module=true)
    end
    @test occursin("diffe",fn)
    # TODO we need to fix julia to remove unused bounds checks
    # @test !occursin("aug",fn)
    
    fn = sprint() do io
       Enzyme.Compiler.enzyme_code_llvm(io, sumsin, Active, Tuple{Duplicated{Vector{Float64}}}; dump_module=true)
    end
    @test occursin("diffe",fn)
    # TODO we need to fix julia to remove unused bounds checks
    # @test !occursin("aug",fn)
    
    fn = sprint() do io
       Enzyme.Compiler.enzyme_code_llvm(io, sqrtsumsq2, Active, Tuple{Duplicated{Vector{Float64}}}; dump_module=true)
    end
    @test occursin("diffe",fn)
    if count("call fastcc void @diffejulia__mapreduce", fn) != 1
        println(sprint() do io
           Enzyme.Compiler.enzyme_code_llvm(io, sqrtsumsq2, Active, Tuple{Duplicated{Vector{Float64}}}; dump_module=true, run_enzyme=false, optimize=false)
       end)
        println(sprint() do io
           Enzyme.Compiler.enzyme_code_llvm(io, sqrtsumsq2, Active, Tuple{Duplicated{Vector{Float64}}}; dump_module=true, run_enzyme=false)
       end)
        println(fn)
    end
    # TODO per system being run on the indexing in the mapreduce is broken
    # @test count("call fastcc void @diffejulia__mapreduce", fn) == 1
    # TODO we need to have enzyme circumvent the double pointer issue by also considering a broader
    # no memory overwritten state [in addition to the arg-based variant]
    @test_broken !occursin("aug",fn)

    x = ones(100)
    dx = zeros(100)
    Enzyme.autodiff(Reverse, sqrtsumsq2, Duplicated(x,dx))
end

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

make3() = (1.0, 2.0, 3.0)


@testset "Simple tests" begin
    f1(x) = 1.0 + x
    f2(x) = x*x
    @test autodiff(Reverse, f1, Active, Active(1.0))[1][1] ≈ 1.0
    @test autodiff(Forward, f1, Duplicated, Duplicated(1.0, 1.0))[1] ≈ 1.0
    @test autodiff(ForwardWithPrimal, f1, Duplicated, Duplicated(1.0, 1.0))[1] ≈ 1.0
    @test autodiff(Reverse, f2, Active, Active(1.0))[1][1] ≈ 2.0
    @test autodiff(Forward, f2, Duplicated(1.0, 1.0))[1] ≈ 2.0
    tup = autodiff(Forward, f2, BatchDuplicated(1.0, (1.0, 2.0, 3.0)))[1]
    @test tup[1] ≈ 2.0
    @test tup[2] ≈ 4.0
    @test tup[3] ≈ 6.0
    tup = autodiff(Forward, f2, BatchDuplicatedFunc{Float64, 3, typeof(make3)}(1.0))[1]
    @test tup[1] ≈ 2.0
    @test tup[2] ≈ 4.0
    @test tup[3] ≈ 6.0
    @test autodiff(Reverse, tanh, Active, Active(1.0))[1][1] ≈ 0.41997434161402606939
    @test autodiff(Forward, tanh, Duplicated(1.0, 1.0))[1] ≈ 0.41997434161402606939
    @test autodiff(Reverse, tanh, Active, Active(1.0f0))[1][1] ≈ Float32(0.41997434161402606939)
    @test autodiff(Forward, tanh, Duplicated(1.0f0, 1.0f0))[1] ≈ Float32(0.41997434161402606939)

    for T in (Float64, Float32, Float16)
        if T == Float16 && Sys.isapple()
            continue
        end
        res = autodiff(Reverse, tanh, Active, Active(T(1)))[1][1]
        @test res isa T
        cmp = if T == Float64
            T(0.41997434161402606939)
        else
            T(0.41997434161402606939f0)
        end
        @test res ≈ cmp
        res = autodiff(Forward, tanh, Duplicated(T(1), T(1)))[1]
        @test res isa T
        @test res ≈ cmp
    end

    test_scalar(f1, 1.0)
    test_scalar(f2, 1.0)
    test_scalar(log2, 1.0)
    test_scalar(log1p, 1.0)

    test_scalar(log10, 1.0)
    test_scalar(Base.acos, 0.9)

    test_scalar(Base.atan, 0.9)

    res = autodiff(Reverse, Base.atan, Active, Active(0.9), Active(3.4))[1]
    @test res[1] ≈ 3.4 / (0.9 * 0.9 + 3.4 * 3.4)
    @test res[2] ≈ -0.9 / (0.9 * 0.9 + 3.4 * 3.4)

    test_scalar(cbrt, 1.0)
    test_scalar(cbrt, 1.0f0; rtol = 1.0e-5, atol = 1.0e-5)
    test_scalar(Base.sinh, 1.0)
    test_scalar(Base.cosh, 1.0)
    test_scalar(Base.sinc, 2.2)
    test_scalar(Base.FastMath.sinh_fast, 1.0)
    test_scalar(Base.FastMath.cosh_fast, 1.0)
    test_scalar(Base.FastMath.exp_fast, 1.0)
    test_scalar(Base.exp10, 1.0)
    test_scalar(Base.exp2, 1.0)
    test_scalar(Base.expm1, 1.0)
    test_scalar(x->rem(x, 1), 0.7)
    test_scalar(x->rem2pi(x,RoundDown), 0.7)
    test_scalar(x->fma(x,x+1,x/3), 2.3)
    test_scalar(sqrt, 1.7+2.1im)
    
    @test autodiff(Forward, sincos, Duplicated(1.0, 1.0))[1][1] ≈ cos(1.0)

    @test autodiff(Reverse, (x)->log(x), Active(2.0)) == ((0.5,),)

    a = [3.14]
    da = [0.0]
    sumcopy(x) = sum(copy(x))
    autodiff(Reverse, sumcopy, Duplicated(a, da))
    @test da[1] ≈ 1.0

    da = [2.7]
    @test autodiff(Forward, sumcopy, Duplicated(a, da))[1] ≈ 2.7

    da = [0.0]
    sumdeepcopy(x) = sum(deepcopy(x))
    autodiff(Reverse, sumdeepcopy, Duplicated(a, da))
    @test da[1] ≈ 1.0

    da = [2.7]
    @test autodiff(Forward, sumdeepcopy, Duplicated(a, da))[1] ≈ 2.7

end

@testset "Deferred and deferred thunk" begin
    function dot(A)
        return A[1] * A[1] + A[2] * A[2] 
    end
    dA = zeros(2)
    A = [3.0, 5.0]
    thunk_dA, def_dA = copy(dA), copy(dA)
    def_A, thunk_A = copy(A), copy(A)
    primal = Enzyme.autodiff(ReverseWithPrimal, dot, Active, Duplicated(A, dA))[2]
    @test primal == 34.0
    primal = Enzyme.autodiff_deferred(ReverseWithPrimal, Const(dot), Active, Duplicated(def_A, def_dA))[2]
    @test primal == 34.0

    dup = Duplicated(thunk_A, thunk_dA)
    TapeType = Enzyme.EnzymeCore.tape_type(
        ReverseSplitWithPrimal,
        Const{typeof(dot)}, Active, Duplicated{typeof(thunk_A)}
    )
    @test Tuple{Float64,Float64}  === TapeType
    Ret = Active
    fwd, rev = Enzyme.autodiff_deferred_thunk(
        ReverseSplitWithPrimal,
        TapeType,
        Const{typeof(dot)},
        Ret,
        Duplicated{typeof(thunk_A)}
    )
    tape, primal, _  = fwd(Const(dot), dup)
    @test isa(tape, Tuple{Float64,Float64})
    rev(Const(dot), dup, 1.0, tape)
    @test all(primal == 34)
    @test all(dA .== [6.0, 10.0])
    @test all(dA .== def_dA)
    @test all(dA .== thunk_dA)

    function kernel(len, A)
        for i in 1:len
            A[i] *= A[i]
        end
    end

    A = Array{Float64}(undef, 64)
    dA = Array{Float64}(undef, 64)

    A .= (1:1:64)
    dA .= 1

    function aug_fwd(ctx, f::FT, ::Val{ModifiedBetween}, args...) where {ModifiedBetween, FT}
        TapeType = Enzyme.tape_type(ReverseSplitModified(ReverseSplitWithPrimal, Val(ModifiedBetween)), Const{Core.Typeof(f)}, Const, Const{Core.Typeof(ctx)}, map(Core.Typeof, args)...)
        forward, reverse = Enzyme.autodiff_deferred_thunk(ReverseSplitModified(ReverseSplitWithPrimal, Val(ModifiedBetween)), TapeType, Const{Core.Typeof(f)}, Const, Const{Core.Typeof(ctx)}, map(Core.Typeof, args)...)
        forward(Const(f), Const(ctx), args...)[1]
        return nothing
    end

    ModifiedBetween = Val((false, false, true))

    aug_fwd(64, kernel, ModifiedBetween, Duplicated(A, dA))

end

@testset "Deferred upgrade" begin
    function gradsin(x)
        return gradient(Reverse, sin, x)[1]
    end
    res = Enzyme.gradient(Reverse, gradsin, 3.1)[1]
    @test res ≈ -sin(3.1)
end

@testset "Simple Complex tests" begin
    mul2(z) = 2 * z
    square(z) = z * z

    z = 1.0+1.0im

    @test_throws ErrorException autodiff(Reverse, mul2, Active, Active(z))
    @test_throws ErrorException autodiff(ReverseWithPrimal, mul2, Active, Active(z))
    @test autodiff(ReverseHolomorphic, mul2, Active, Active(z))[1][1] ≈ 2.0 + 0.0im
    @test autodiff(ReverseHolomorphicWithPrimal, mul2, Active, Active(z))[1][1] ≈ 2.0 + 0.0im
    @test autodiff(ReverseHolomorphicWithPrimal, mul2, Active, Active(z))[2] ≈ 2 * z

    z = 3.4 + 2.7im
    @test autodiff(ReverseHolomorphic, square, Active, Active(z))[1][1] ≈ 2 * z
    @test autodiff(ReverseHolomorphic, identity, Active, Active(z))[1][1] ≈ 1

    @test autodiff(ReverseHolomorphic, Base.inv, Active, Active(3.0 + 4.0im))[1][1] ≈ 0.0112 + 0.0384im

    mul3(z) = Base.inferencebarrier(2 * z)

    @test_throws MethodError autodiff(ReverseHolomorphic, mul3, Active, Active(z))
    @test_throws MethodError autodiff(ReverseHolomorphic, mul3, Active{Complex}, Active(z))

    vals = Complex{Float64}[3.4 + 2.7im]
    dvals = Complex{Float64}[0.0]
    autodiff(ReverseHolomorphic, sum, Active, Duplicated(vals, dvals))
    @test vals[1] ≈ 3.4 + 2.7im
    @test dvals[1] ≈ 1.0

    sumsq(x) = sum(x .* x)

    vals = Complex{Float64}[3.4 + 2.7im]
    dvals = Complex{Float64}[0.0]
    autodiff(ReverseHolomorphic, sumsq, Active, Duplicated(vals, dvals))
    @test vals[1] ≈ 3.4 + 2.7im
    @test dvals[1] ≈ 2 * (3.4 + 2.7im)

    sumsq2(x) = sum(abs2.(x))
    vals = Complex{Float64}[3.4 + 2.7im]
    dvals = Complex{Float64}[0.0]
    autodiff(ReverseHolomorphic, sumsq2, Active, Duplicated(vals, dvals))
    @test vals[1] ≈ 3.4 + 2.7im
    @test dvals[1] ≈ 2 * (3.4 + 2.7im)

    sumsq2C(x) = Complex{Float64}(sum(abs2.(x)))
    vals = Complex{Float64}[3.4 + 2.7im]
    dvals = Complex{Float64}[0.0]
    autodiff(ReverseHolomorphic, sumsq2C, Active, Duplicated(vals, dvals))
    @test vals[1] ≈ 3.4 + 2.7im
    @test dvals[1] ≈ 3.4 - 2.7im

    sumsq3(x) = sum(x .* conj(x))
    vals = Complex{Float64}[3.4 + 2.7im]
    dvals = Complex{Float64}[0.0]
    autodiff(ReverseHolomorphic, sumsq3, Active, Duplicated(vals, dvals))
    @test vals[1] ≈ 3.4 + 2.7im
    @test dvals[1] ≈ 3.4 - 2.7im

    sumsq3R(x) = Float64(sum(x .* conj(x)))
    vals = Complex{Float64}[3.4 + 2.7im]
    dvals = Complex{Float64}[0.0]
    autodiff(ReverseHolomorphic, sumsq3R, Active, Duplicated(vals, dvals))
    @test vals[1] ≈ 3.4 + 2.7im
    @test dvals[1] ≈ 2 * (3.4 + 2.7im)

    function setinact(z)
        z[1] *= 2
        nothing
    end

    vals = Complex{Float64}[3.4 + 2.7im]
    dvals = Complex{Float64}[0.0]
    autodiff(ReverseHolomorphic, setinact, Const, Duplicated(vals, dvals))
    @test vals[1] ≈ 2 * (3.4 + 2.7im)
    @test dvals[1] ≈ 0.0


    function setinact2(z)
        z[1] *= 2
        return 0.0+1.0im
    end

    vals = Complex{Float64}[3.4 + 2.7im]
    dvals = Complex{Float64}[0.0]
    autodiff(ReverseHolomorphic, setinact2, Const, Duplicated(vals, dvals))
    @test vals[1] ≈ 2 * (3.4 + 2.7im)
    @test dvals[1] ≈ 0.0

    vals = Complex{Float64}[3.4 + 2.7im]
    dvals = Complex{Float64}[0.0]
    autodiff(ReverseHolomorphic, setinact2, Active, Duplicated(vals, dvals))
    @test vals[1] ≈ 2 * (3.4 + 2.7im)
    @test dvals[1] ≈ 0.0


    function setact(z)
        z[1] *= 2
        return z[1]
    end

    vals = Complex{Float64}[3.4 + 2.7im]
    dvals = Complex{Float64}[0.0]
    autodiff(ReverseHolomorphic, setact, Const, Duplicated(vals, dvals))
    @test vals[1] ≈ 2 * (3.4 + 2.7im)
    @test dvals[1] ≈ 0.0

    vals = Complex{Float64}[3.4 + 2.7im]
    dvals = Complex{Float64}[0.0]
    autodiff(ReverseHolomorphic, setact, Active, Duplicated(vals, dvals))
    @test vals[1] ≈ 2 * (3.4 + 2.7im)
    @test dvals[1] ≈ 2.0

    function upgrade(z)
        z = ComplexF64(z)
        return z*z
    end
    @test autodiff(ReverseHolomorphic, upgrade, Active, Active(3.1))[1][1] ≈ 6.2
end

@testset "Simple Exception" begin
    f_simple_exc(x, i) = ccall(:jl_, Cvoid, (Any,), x[i])
    y = [1.0, 2.0]
    f_x = zero.(y)
    @test_throws BoundsError autodiff(Reverse, f_simple_exc, Duplicated(y, f_x), Const(0))
end


@testset "Duplicated" begin
    x = Ref(1.0)
    y = Ref(2.0)

    ∇x = Ref(0.0)
    ∇y = Ref(0.0)

    autodiff(Reverse, (a,b)->a[]*b[], Active, Duplicated(x, ∇x), Duplicated(y, ∇y))

    @test ∇y[] == 1.0
    @test ∇x[] == 2.0
end

@testset "Simple tests" begin
    g(x) = real((x + im)*(1 - im*x))
    @test first(autodiff(Reverse, g, Active, Active(2.0))[1]) ≈ 2.0
    @test first(autodiff(Forward, g, Duplicated(2.0, 1.0))) ≈ 2.0
    @test first(autodiff(Reverse, g, Active, Active(3.0))[1]) ≈ 2.0
    @test first(autodiff(Forward, g, Duplicated(3.0, 1.0))) ≈ 2.0
    test_scalar(g, 2.0)
    test_scalar(g, 3.0)
    test_scalar(Base.inv, 3.0 + 4.0im)
end

@testset "Base functions" begin
    f1(x) = prod(ntuple(i -> i * x, 3))
    @test autodiff(Reverse, f1, Active, Active(2.0))[1][1] == 72
    @test autodiff(Forward, f1, Duplicated(2.0, 1.0))[1]   == 72

    f2(x) = x * something(nothing, 2)
    @test autodiff(Reverse, f2, Active, Active(1.0))[1][1] == 2
    @test autodiff(Forward, f2, Duplicated(1.0, 1.0))[1]   == 2

    f3(x) = x * sum(unique([x, 2.0, 2.0, 3.0]))
    @test autodiff(Reverse, f3, Active, Active(1.0))[1][1] == 7
    @test autodiff(Forward, f3, Duplicated(1.0, 1.0))[1]   == 7

    for rf in (reduce, foldl, foldr)
        f4(x) = rf(*, [1.0, x, x, 3.0])
        @test autodiff(Reverse, f4, Active, Active(2.0))[1][1] == 12
        @test autodiff(Forward, f4, Duplicated(2.0, 1.0))[1]   == 12
    end

    f5(x) = sum(accumulate(+, [1.0, x, x, 3.0]))
    @test autodiff(Reverse, f5, Active, Active(2.0))[1][1] == 5
    @test autodiff(Forward, f5, Duplicated(2.0, 1.0))[1]   == 5

    f6(x) = x |> inv |> abs
    @test autodiff(Reverse, f6, Active, Active(-2.0))[1][1] == 1/4
    @test autodiff(Forward, f6, Duplicated(-2.0, 1.0))[1]   == 1/4

    f7(x) = (inv ∘ abs)(x)
    @test autodiff(Reverse, f7, Active, Active(-2.0))[1][1] == 1/4
    @test autodiff(Forward, f7, Duplicated(-2.0, 1.0))[1]   == 1/4

    f8(x) = x * count(i -> i > 1, [0.5, x, 1.5])
    @test autodiff(Reverse, f8, Active, Active(2.0))[1][1] == 2
    @test autodiff(Forward, f8, Duplicated(2.0, 1.0))[1]   == 2

    Enzyme.API.strictAliasing!(false)
    function f9(x)
        y = []
        foreach(i -> push!(y, i^2), [1.0, x, x])
        return sum(y)
    end
    @test autodiff(Reverse, f9, Active, Active(2.0))[1][1] == 8
    @test autodiff(Forward, f9, Duplicated(2.0, 1.0))[1]   == 8

    Enzyme.API.strictAliasing!(true)
    f10(x) = hypot(x, 2x)
    @test autodiff(Reverse, f10, Active, Active(2.0))[1][1] == sqrt(5)
    @test autodiff(Forward, f10, Duplicated(2.0, 1.0))[1]   == sqrt(5)

    f11(x) = x * sum(LinRange(x, 10.0, 6))
    @test autodiff(Reverse, f11, Active, Active(2.0))[1][1] == 42
    @test autodiff(Forward, f11, Duplicated(2.0, 1.0))[1]   == 42

    f12(x, k) = get(Dict(1 => 1.0, 2 => x, 3 => 3.0), k, 1.0)
    @test autodiff(Reverse, f12, Active, Active(2.0),  Const(2))[1] == (1.0, nothing)
    @test autodiff(Forward, f12, Duplicated(2.0, 1.0), Const(2))    == (1.0,)
    @test autodiff(Reverse, f12, Active, Active(2.0),  Const(3))[1] == (0.0, nothing)
    @test autodiff(Forward, f12, Duplicated(2.0, 1.0), Const(3))    == (0.0,)
    @test autodiff(Reverse, f12, Active, Active(2.0),  Const(4))[1] == (0.0, nothing)
    @test autodiff(Forward, f12, Duplicated(2.0, 1.0), Const(4))    == (0.0,)

    f13(x) = muladd(x, 3, x)
    @test autodiff(Reverse, f13, Active, Active(2.0))[1][1] == 4
    @test autodiff(Forward, f13, Duplicated(2.0, 1.0))[1]   == 4

    f14(x) = x * cmp(x, 3)
    @test autodiff(Reverse, f14, Active, Active(2.0))[1][1] == -1
    @test autodiff(Forward, f14, Duplicated(2.0, 1.0))[1]   == -1

    f15(x) = x * argmax([1.0, 3.0, 2.0])
    @test autodiff(Reverse, f15, Active, Active(3.0))[1][1] == 2
    @test autodiff(Forward, f15, Duplicated(3.0, 1.0))[1]   == 2

    f16(x) = evalpoly(2, (1, 2, x))
    @test autodiff(Reverse, f16, Active, Active(3.0))[1][1] == 4
    @test autodiff(Forward, f16, Duplicated(3.0, 1.0))[1]   == 4

    f17(x) = @evalpoly(2, 1, 2, x)
    @test autodiff(Reverse, f17, Active, Active(3.0))[1][1] == 4
    @test autodiff(Forward, f17, Duplicated(3.0, 1.0))[1]   == 4

    f18(x) = widemul(x, 5.0f0)
    @test autodiff(Reverse, f18, Active, Active(2.0f0))[1][1] == 5
    @test autodiff(Forward, f18, Duplicated(2.0f0, 1.0f0))[1] == 5

    f19(x) = copysign(x, -x)
    @test autodiff(Reverse, f19, Active, Active(2.0))[1][1] == -1
    @test autodiff(Forward, f19, Duplicated(2.0, 1.0))[1]   == -1

    f20(x) = sum([ifelse(i > 5, i, zero(i)) for i in [x, 2x, 3x, 4x]])
    @test autodiff(Reverse, f20, Active, Active(2.0))[1][1] == 7
    @test autodiff(Forward, f20, Duplicated(2.0, 1.0))[1]   == 7

    function f21(x)
        nt = (a=x, b=2x, c=3x)
        return nt.c
    end
    @test autodiff(Reverse, f21, Active, Active(2.0))[1][1] == 3
    @test autodiff(Forward, f21, Duplicated(2.0, 1.0))[1]   == 3

    f22(x) = sum(fill(x, (3, 3)))
    @test autodiff(Reverse, f22, Active, Active(2.0))[1][1] == 9
    @test autodiff(Forward, f22, Duplicated(2.0, 1.0))[1]   == 9

    function f23(x)
        a = similar(rand(3, 3))
        fill!(a, x)
        return sum(a)
    end
    @test autodiff(Reverse, f23, Active, Active(2.0))[1][1] == 9
    @test autodiff(Forward, f23, Duplicated(2.0, 1.0))[1]   == 9

    function f24(x)
        try
            return 3x
        catch
            return 2x
        end
    end
    @test autodiff(Reverse, f24, Active, Active(2.0))[1][1] == 3
    @test autodiff(Forward, f24, Duplicated(2.0, 1.0))[1]   == 3

    function f25(x)
        try
            sqrt(-1.0)
            return 3x
        catch
            return 2x
        end
    end
    @test autodiff(Reverse, f25, Active, Active(2.0))[1][1] == 2
    @test autodiff(Forward, f25, Duplicated(2.0, 1.0))[1]   == 2

    f26(x) = circshift([1.0, 2x, 3.0], 1)[end]
    @test autodiff(Reverse, f26, Active, Active(2.0))[1][1] == 2
    @test autodiff(Forward, f26, Duplicated(2.0, 1.0))[1]   == 2

    f27(x) = repeat([x 3x], 3)[2, 2]
    @test autodiff(Reverse, f27, Active, Active(2.0))[1][1] == 3
    @test autodiff(Forward, f27, Duplicated(2.0, 1.0))[1]   == 3

    f28(x) = x * sum(trues(4, 3))
    @test autodiff(Reverse, f28, Active, Active(2.0))[1][1] == 12
    @test autodiff(Forward, f28, Duplicated(2.0, 1.0))[1]   == 12

    f29(x) = sum(Set([1.0, x, 2x, x]))
    @static if VERSION ≥ v"1.11-"
        @test autodiff(set_runtime_activity(Reverse), f29, Active, Active(2.0))[1][1] == 3
        @test autodiff(set_runtime_activity(Forward), f29, Duplicated(2.0, 1.0))[1]   == 3
    else
        @test autodiff(Reverse, f29, Active, Active(2.0))[1][1] == 3
        @test autodiff(Forward, f29, Duplicated(2.0, 1.0))[1]   == 3
    end

    f30(x) = reverse([x 2.0 3x])[1]
    @test autodiff(Reverse, f30, Active, Active(2.0))[1][1] == 3
    @test autodiff(Forward, f30, Duplicated(2.0, 1.0))[1]   == 3
end

function deadarg_pow(z::T, i) where {T<:Real}
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

euroad′(x) = first(autodiff(Reverse, euroad, Active, Active(x)))[1]

@test euroad(0.5) ≈ -log(0.5) # -log(1-x)
@test euroad′(0.5) ≈ 2.0 # d/dx -log(1-x) = 1/(1-x)
test_scalar(euroad, 0.5)
end
@noinline function womylogpdf(X::AbstractArray{<:Real})
  map(womylogpdf, X)
end

function womylogpdf(x::Real)
    (x - 2)
end


function wologpdf_test(x)
    return womylogpdf(x)
end

@testset "Ensure writeonly deduction combines with capture" begin
    res = Enzyme.autodiff(Enzyme.Forward, wologpdf_test, Duplicated([0.5], [0.7]))
    @test res[1] ≈ [0.7]
end

@testset "Nested AD" begin
    tonest(x,y) = (x + y)^2

    @test autodiff(Forward, (x,y) -> autodiff(Forward, Const(tonest), Duplicated(x, 1.0), Const(y))[1], Const(1.0), Duplicated(2.0, 1.0))[1] ≈ 2.0
end

@testset "Hessian" begin
    function origf(x::Array{Float64}, y::Array{Float64})
        y[1] = x[1] * x[1] + x[2] * x[1]
        return nothing
    end

    function grad(x, dx, y, dy)
      Enzyme.autodiff(Reverse, Const(origf), Duplicated(x, dx), DuplicatedNoNeed(y, dy))
      nothing
    end

    x = [2.0, 2.0]
    y = Vector{Float64}(undef, 1)
    dx = [0.0, 0.0]
    dy = [1.0]

    grad(x, dx, y, dy)

    vx = ([1.0, 0.0], [0.0, 1.0])
    hess = ([0.0, 0.0], [0.0, 0.0])
    dx2 = [0.0, 0.0]
    dy = [1.0]

    Enzyme.autodiff(Enzyme.Forward, grad,
                    Enzyme.BatchDuplicated(x, vx),
                    Enzyme.BatchDuplicated(dx2, hess),
                    Const(y),
                    Const(dy))

    @test dx ≈ dx2
    @test hess[1][1] ≈ 2.0
    @test hess[1][2] ≈ 1.0
    @test hess[2][1] ≈ 1.0
    @test hess[2][2] ≈ 0.0

    function f_ip(x, tmp)
        tmp .= x ./ 2
        return dot(tmp, x)
    end

    function f_gradient_deferred!(dx, x, tmp)
        dtmp = make_zero(tmp)
        autodiff_deferred(Reverse, Const(f_ip), Active, Duplicated(x, dx), Duplicated(tmp, dtmp))
        return nothing
    end

    function f_hvp!(hv, x, v, tmp)
        dx = make_zero(x)
        btmp = make_zero(tmp)
        autodiff(
            Forward,
            f_gradient_deferred!,
            Duplicated(dx, hv),
            Duplicated(x, v),
            Duplicated(tmp, btmp),
        )
        return nothing
    end

    x = [1.0]
    v = [-1.0]
    hv = make_zero(v)
    tmp = similar(x)

    f_hvp!(hv, x, v, tmp)
    @test hv ≈ [-1.0]
end

@testset "Nested Type Error" begin
    nested_f(x) = sum(tanh, x)

    function nested_df!(dx, x)
        make_zero!(dx)
        autodiff_deferred(Reverse, Const(nested_f), Active, Duplicated(x, dx))
        return nothing
    end

    function nested_hvp!(hv, v, x)
        make_zero!(hv)
        autodiff(Forward, nested_df!, Const, Duplicated(make_zero(x), hv), Duplicated(x, v))
        return nothing
    end

    x = [0.5]

    # primal: sanity check
    @test nested_f(x) ≈ sum(tanh, x)

    # gradient: works
    dx = make_zero(x)
    nested_df!(dx, x)

    @test dx ≈ (sech.(x).^2)

    v = first(onehot(x))
    hv = make_zero(v)
    nested_hvp!(hv, v, x)
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

@testset "Dict" begin
    params = Dict{Symbol, Float64}()
    dparams = Dict{Symbol, Float64}()

    params[:var] = 10.0
    dparams[:var] = 0.0

    f_dict(params, x) = params[:var] * x

    @test autodiff(Reverse, f_dict, Const(params), Active(5.0)) == ((nothing, 10.0,),)
    @test autodiff(Reverse, f_dict, Duplicated(params, dparams), Active(5.0)) == ((nothing, 10.0,),)
    @test dparams[:var] == 5.0


    mutable struct MD
        v::Float64
        d::Dict{Symbol, MD}
    end

    # TODO without Float64 on return
    # there is a potential phi bug
    function sum_rec(d::Dict{Symbol,MD})::Float64
        s = 0.0
        for k in keys(d)
            s += d[k].v
            s += sum_rec(d[k].d)
        end
        return s
    end

    par = Dict{Symbol, MD}()
    par[:var] = MD(10.0, Dict{Symbol, MD}())
    par[:sub] = MD(2.0, Dict{Symbol, MD}(:a=>MD(3.0, Dict{Symbol, MD}())))

    dpar = Dict{Symbol, MD}()
    dpar[:var] = MD(0.0, Dict{Symbol, MD}())
    dpar[:sub] = MD(0.0, Dict{Symbol, MD}(:a=>MD(0.0, Dict{Symbol, MD}())))

    # TODO
    # autodiff(Reverse, sum_rec, Duplicated(par, dpar))
    # @show par, dpar, sum_rec(par)
    # @test dpar[:var].v ≈ 1.0
    # @test dpar[:sub].v ≈ 1.0
    # @test dpar[:sub].d[:a].v ≈ 1.0
end

let
    function loadsin2(xp)
        x = @inbounds xp[1]
        @inbounds xp[1] = 0.0
        sin(x)
    end
    global invsin2
    function invsin2(xp)
        xp = Base.invokelatest(convert, Vector{Float64}, xp)
        loadsin2(xp)
    end
    x = [2.0]
end

@testset "Struct return" begin
    x = [2.0]
    dx = [0.0]
    @test Enzyme.autodiff(Reverse, invsin2, Active, Duplicated(x, dx)) == ((nothing,),)
    @test dx[1] == -0.4161468365471424
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

    autodiff(Reverse, Const(noretval), Duplicated(x,dx), Duplicated(y, dy))
    return dx
end

@testset "Closure" begin
    x = [2.0,6.0]
    dx = grad_closure(x->[x[1], x[2]], x)
    @test dx == [1.0, 0.0]
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

@testset "Bithacks" begin
    function fneg(x::Float64)
        xptr = reinterpret(Int64, x)
        y = Int64(-9223372036854775808)
        out = y ⊻ xptr;
        return reinterpret(Float64, out)
    end
    @test autodiff(Reverse, fneg, Active, Active(2.0))[1][1] ≈ -1.0
    @test autodiff(Forward, fneg, Duplicated(2.0, 1.0))[1] ≈ -1.0
    function expor(x::Float64)
        xptr = reinterpret(Int64, x)
        y = UInt64(4607182418800017408)
        out = y | xptr;
        return reinterpret(Float64, out)
    end
    @test autodiff(Reverse, expor, Active, Active(0.42))[1][1] ≈ 4.0
    @test autodiff(Forward, expor, Duplicated(0.42, 1.0))[1] ≈ 4.0
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

@testset "GC" begin
    function gc_alloc(x)  # Basically g(x) = x^2
        a = Array{Float64, 1}(undef, 10)
        for n in 1:length(a)
            a[n] = x^2
        end
        return mean(a)
    end
    @test autodiff(Reverse, gc_alloc, Active, Active(5.0))[1][1] ≈ 10
    @test autodiff(Forward, gc_alloc, Duplicated(5.0, 1.0))[1] ≈ 10

    A = Float64[2.0, 3.0]
    B = Float64[4.0, 5.0]
    dB = Float64[0.0, 0.0]
    f = (X, Y) -> sum(X .* Y)
    Enzyme.autodiff(Reverse, f, Active, Const(A), Duplicated(B, dB))

    function gc_copy(x)  # Basically g(x) = x^2
        a = x * ones(10)
        for n in 1:length(a)
            a[n] = x^2
        end
        return mean(a)
    end

    @test Enzyme.autodiff(Reverse, gc_copy, Active, Active(5.0))[1][1] ≈ 10
    @test Enzyme.autodiff(Forward, gc_copy, Duplicated(5.0, 1.0))[1] ≈ 10
end

@testset "Null init tape" begin
    struct Leaf
        params::NamedTuple
    end

    function LeafF(n::Leaf)::Float32
        y = first(n.params.b2)
        r = convert(Tuple{Float32}, (y,))
        return r[1]
    end

    ps =
        (
            b2 = 1.0f0,
        )

    grads =
        (
            b2 = 0.0f0,
        )

    t1 = Leaf(ps)
    t1Grads = Leaf(grads)

    forward, pullback = Enzyme.autodiff_thunk(ReverseSplitModified(ReverseSplitNoPrimal, Val((false, true))), Const{typeof(LeafF)}, Active, Duplicated{Leaf})
    tape, primal, shadow = forward(Const(LeafF), Duplicated(t1, t1Grads))


    struct Foo2{X,Y}
        x::X
        y::Y
    end

    test_f(f::Foo2) = f.x^2
    res = autodiff(Reverse, test_f, Active(Foo2(3.0, :two)))[1][1]
    @test res.x ≈ 6.0
    @test res.y == nothing
end

@testset "Method errors" begin
     fwd = Enzyme.autodiff_thunk(Forward, Const{typeof(sum)}, Duplicated, Duplicated{Vector{Float64}})
     @test_throws MethodError fwd(ones(10))
     @test_throws MethodError fwd(Duplicated(ones(10), ones(10)))
     @test_throws MethodError fwd(Const(first), Duplicated(ones(10), ones(10)))
     # TODO
     # @test_throws MethodError fwd(Const(sum), Const(ones(10)))
     fwd(Const(sum), Duplicated(ones(10), ones(10)))
end

@testset "Generic Active Union Return" begin

    function generic_union_ret(A)
            if 0 < length(A)
                @inbounds A[1]
            else
                nothing
                Base._InitialValue()
            end
    end

    function outergeneric(weights::Vector{Float64})::Float64
        v = generic_union_ret(Base.inferencebarrier(weights))
        return v::Float64
    end

    weights = [0.2]
    dweights = [0.0]

    Enzyme.autodiff(Enzyme.Reverse, outergeneric, Enzyme.Duplicated(weights, dweights))

    @test dweights[1] ≈ 1.
end

function Valuation1(z,Ls1)
    @inbounds Ls1[1] = sum(Base.inferencebarrier(z))
    return nothing
end
@testset "Active setindex!" begin
    v=ones(5)
    dv=zero(v)

    DV1=Float32[0]
    DV2=Float32[1]

    Enzyme.autodiff(Reverse,Valuation1,Duplicated(v,dv),Duplicated(DV1,DV2))
    @test dv[1] ≈ 1.
    
    DV1=Float32[0]
    DV2=Float32[1]
    v=ones(5)
    dv=zero(v)
    dv[1] = 1.    
    Enzyme.autodiff(Forward,Valuation1,Duplicated(v,dv),Duplicated(DV1,DV2))
    @test DV2[1] ≈ 1.
end

@testset "Null init union" begin
    @noinline function unionret(itr, cond)
        if cond
            return Base._InitialValue()
        else
            return itr[1]
        end
    end

    function fwdunion(data::Vector{Float64})::Real
        unionret(data, false)
    end

    data = ones(Float64, 500)
    ddata = zeros(Float64, 500)

    forward, pullback = Enzyme.autodiff_thunk(ReverseSplitModified(ReverseSplitNoPrimal, Val((false, true))), Const{typeof(fwdunion)}, Active, Duplicated{Vector{Float64}})
    tape, primal, shadow = forward(Const(fwdunion), Duplicated(data, ddata))

	function firstimpl(itr)
		v = firstfold(itr)
		@assert !(v isa Base._InitialValue)
		return v
	end

	function firstfold(itr)
		op, itr = Base._xfadjoint(Base.BottomRF(Base.add_sum), Base.Generator(Base.identity, itr))
		y = iterate(itr)
		init = Base._InitialValue()
		y === nothing && return init
		v = op(init, y[1])
		return v
	end

	function smallrf(weights::Vector{Float64}, data::Vector{Float64})::Float64
		itr1 = (weight for (weight, mean) in zip(weights, weights))

		itr2 = (firstimpl(itr1) for x in data)

		firstimpl(itr2)
	end

	data = ones(Float64, 1)

	weights = [0.2]
	dweights = [0.0]
    # Technically this test doesn't need runtimeactivity since the closure combo of active itr1 and const data
    # doesn't use any of the const data values, but now that we error for activity confusion, we need to
    # mark runtimeActivity to let this pass
    Enzyme.autodiff(set_runtime_activity(Enzyme.Reverse), Const(smallrf), Enzyme.Duplicated(weights, dweights), Enzyme.Const(data))
    @test dweights[1] ≈ 1.

    function invokesum(weights::Vector{Float64}, data::Vector{Float64})::Float64
        sum(
            sum(
                weight
                for (weight, mean) in zip(weights, weights)
            )
            for x in data
        )
    end

    data = ones(Float64, 20)

    weights = [0.2, 0.8]
    dweights = [0.0, 0.0]

    Enzyme.autodiff(set_runtime_activity(Enzyme.Reverse), invokesum, Enzyme.Duplicated(weights, dweights), Enzyme.Const(data))
    @test dweights[1] ≈ 20.
    @test dweights[2] ≈ 20.
end


abstract type AbsFwdType end

# Two copies of the same type.
struct FwdNormal1{T<:Real} <: AbsFwdType
σ::T
end

struct FwdNormal2{T<:Real} <: AbsFwdType
σ::T
end

fwdlogpdf(d) = d.σ

function absactfunc(x)
	dists = AbsFwdType[FwdNormal1{Float64}(1.0), FwdNormal2{Float64}(x)]
	res = Vector{Float64}(undef, 2)
	for i in 1:length(dists)
	    @inbounds res[i] = fwdlogpdf(dists[i])
	end
	return @inbounds res[1] + @inbounds res[2]
end

@testset "Forward Mode active runtime activity" begin
    @static if VERSION ≥ v"1.11-"
        res = Enzyme.autodiff(set_runtime_activity(Enzyme.Forward), Enzyme.Const(absactfunc), Duplicated(2.7, 3.1))
    else
        res = Enzyme.autodiff(Enzyme.Forward, Enzyme.Const(absactfunc), Duplicated(2.7, 3.1))
    end
    @test res[1] ≈ 3.1
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

@testset "Compare against" begin
    x = 3.0
    fd = central_fdm(5, 1)(sin, x)

    @test fd ≈ cos(x)
    @test fd ≈ first(autodiff(Reverse, sin, Active, Active(x)))[1]
    @test fd ≈ first(autodiff(Forward, sin, Duplicated(x, 1.0)))

    x = 0.2 + sin(3.0)
    fd = central_fdm(5, 1)(asin, x)

    @test fd ≈ 1/sqrt(1-x*x)
    @test fd ≈ first(autodiff(Reverse, asin, Active, Active(x)))[1]
    @test fd ≈ first(autodiff(Forward, asin, Duplicated(x, 1.0)))
    test_scalar(asin, x)

    function foo(x)
        a = sin(x)
        b = 0.2 + a
        c = asin(b)
        return c
    end

    x = 3.0
    fd = central_fdm(5, 1)(foo, x)

    @test fd ≈ cos(x)/sqrt(1-(0.2+sin(x))*(0.2+sin(x)))
    @test fd ≈ first(autodiff(Reverse, foo, Active, Active(x)))[1]
    @test fd ≈ first(autodiff(Forward, foo, Duplicated(x, 1.0)))
    test_scalar(foo, x)

    # Input type shouldn't matter
    x = 3
    @test fd ≈ cos(x)/sqrt(1-(0.2+sin(x))*(0.2+sin(x)))
    @test fd ≈ first(autodiff(Reverse, foo, Active, Active(x)))[1]
    # They do matter for duplicated, which can't be auto promoted
    # @test fd ≈ first(autodiff(Forward, foo, Duplicated(x, 1)))

    f74(a, c) = a * √c
    @test √3 ≈ first(autodiff(Reverse, f74, Active, Active(2), Const(3)))[1]
    @test √3 ≈ first(autodiff(Forward, f74, Duplicated(2.0, 1.0), Const(3)))
end

@testset "SinCos" begin
	function sumsincos(theta)
		a, b = sincos(theta)
		return a + b
	end
    test_scalar(sumsincos, 1.0, rtol=1e-5, atol=1e-5)
end

@testset "BoxFloat" begin
    function boxfloat(x)
        x = ccall(:jl_box_float64, Any, (Float64,), x)
        (sin(x)::Float64 + x)::Float64
    end
    @test 0.5838531634528576 ≈ Enzyme.autodiff(Reverse, boxfloat, Active, Active(2.0))[1][1]
    @test 0.5838531634528576 ≈ Enzyme.autodiff(Forward, boxfloat, Duplicated, Duplicated(2.0, 1.0))[1]
    res = Enzyme.autodiff(Forward, boxfloat, BatchDuplicated, BatchDuplicated(2.0, (1.0, 2.0)))[1]
    @test 0.5838531634528576 ≈ res[1]
    @test 1.1677063269057153 ≈ res[2]
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
    autodiff(Reverse, mybesselj, Active, Const(0), Active(1.0))
    autodiff(Reverse, mybesselj, Active, Const(0), Active(1.0))
    autodiff(Forward, mybesselj, Const(0), Duplicated(1.0, 1.0))
    autodiff(Forward, mybesselj, Const(0), Duplicated(1.0, 1.0))
    @testset "besselj0/besselj1" for x in (1.0, -1.0, 0.0, 0.5, 10, -17.1,) # 1.5 + 0.7im)
        test_scalar(mybesselj0, x, rtol=1e-5, atol=1e-5)
        test_scalar(mybesselj1, x, rtol=1e-5, atol=1e-5)
    end
end

# Ensure that this returns an error, and not a crash
# https://github.com/EnzymeAD/Enzyme.jl/issues/368
abstract type TensorProductBasis <: Function end

struct LegendreBasis <: TensorProductBasis
    n::Int
end

function (basis::LegendreBasis)(x)
    return x
end

struct MyTensorLayer
    model::Array{TensorProductBasis}
end

function fn(layer::MyTensorLayer, x)
    model = layer.model
    return model[1](x)
end

const nn = MyTensorLayer([LegendreBasis(10)])

function dxdt_pred(x)
  return fn(nn, x)
end

@testset "AbstractType calling convention" begin
    # TODO get rid of runtime activity
    @test 1.0 ≈ Enzyme.autodiff(set_runtime_activity(Reverse), dxdt_pred, Active(1.0))[1][1]
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

function rtg_f(V,@nospecialize(cv))
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
    aleph = n*p + oftype(p, m)
    j = clamp(trunc(Int, aleph), 1, n-1)
    γ = clamp(aleph - j, 0, 1)

    if n == 1
        a = @inbounds v[1]
        b = @inbounds v[1]
    else
        a = @inbounds v[j]
        b = @inbounds v[j + 1]
    end
    
    return a + γ*(b-a)
end

function fquantile(x)
    v = [1.0, x]
    return @inbounds (map(y->myquantile(v, y, alpha=1.), [0.7]))[1]
end

@testset "Attributor issues" begin

    cor = fquantile(2.0)
    res = autodiff(ForwardWithPrimal, fquantile, Duplicated,Duplicated(2.0, 1.0))
    @test cor ≈ res[2]
    @test 0.7 ≈ res[1]

end

## https://github.com/JuliaDiff/ChainRules.jl/tree/master/test/rulesets
if !Sys.iswindows()
    include("ext/specialfunctions.jl")
end

@testset "Threads" begin
    cmd = `$(Base.julia_cmd()) --threads=1 --startup-file=no threads.jl`
   	@test success(pipeline(cmd, stderr=stderr, stdout=stdout))
    cmd = `$(Base.julia_cmd()) --threads=2 --startup-file=no threads.jl`
   	@test success(pipeline(cmd, stderr=stderr, stdout=stdout))
end

@testset "DiffTest" begin
    include("DiffTests.jl")

    n = 1 + rand()
    x, y = 1 .+ rand(5, 5), 1 .+ rand(5)
    A, B = 1 .+ rand(5, 5), 1 .+ rand(5, 5)

    # f returns Number
    @testset "Number to Number" for f in DiffTests.NUMBER_TO_NUMBER_FUNCS
        test_scalar(f, n; rtol=1e-6, atol=1e-6)
    end

    @testset "Vector to Number" for f in DiffTests.VECTOR_TO_NUMBER_FUNCS
        test_matrix_to_number(f, y; rtol=1e-6, atol=1e-6)
    end

    @testset "Matrix to Number" for f in DiffTests.MATRIX_TO_NUMBER_FUNCS
        test_matrix_to_number(f, x; rtol=1e-6, atol=1e-6)
    end

    # TODO(vchuravy/wsmoses): Enable these tests
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

    @test 4.6 ≈ first(autodiff(Reverse, printsq, Active, Active(2.3)))[1]
    @test 4.6 ≈ first(autodiff(Forward, printsq, Duplicated(2.3, 1.0)))

    function tostring(x)
        string(x)
        x*x
    end

    @test 4.6 ≈ first(autodiff(Reverse, tostring, Active, Active(2.3)))[1]
    @test 4.6 ≈ first(autodiff(Forward, tostring, Duplicated(2.3, 1.0)))
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

@testset "No speculation" begin
	mutable struct SpecFoo

		iters::Int
		a::Float64
		b::Vector{Float64}

	end

	function f(Foo)
		for i = 1:Foo.iters

			c = -1.0

			if Foo.a < 0.0
				X = (-Foo.a)^0.25
				c = 2*log(X)
			end

			# set b equal to desired result
			Foo.b[1] = 1.0 / c

			return nothing
		end
	end

	foo  = SpecFoo(1, 1.0, zeros(Float64, 1))
	dfoo = SpecFoo(0, 0.0, zeros(Float64, 1))

	# should not throw a domain error, which
	# will occur if the pow is mistakenly speculated
	Enzyme.autodiff(Reverse, f, Duplicated(foo, dfoo))
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
    @test 1.0 ≈ Enzyme.autodiff(Reverse, inactive_gen, Active, Active(1E4))[1][1]
	@test 1.0 ≈ Enzyme.autodiff(Forward, inactive_gen, Duplicated(1E4, 1.0))[1]

    function whocallsmorethan30args(R)
        temp = diag(R)     
         R_inv = [temp[1] 0. 0. 0. 0. 0.; 
             0. temp[2] 0. 0. 0. 0.; 
             0. 0. temp[3] 0. 0. 0.; 
             0. 0. 0. temp[4] 0. 0.; 
             0. 0. 0. 0. temp[5] 0.; 
         ]
    
        return sum(R_inv)
    end
    
    R = zeros(6,6)    
    dR = zeros(6, 6)

    @static if VERSION ≥ v"1.11-"
    else
        @test_broken autodiff(Reverse, whocallsmorethan30args, Active, Duplicated(R, dR))
        # autodiff(Reverse, whocallsmorethan30args, Active, Duplicated(R, dR))
    	# @test 1.0 ≈ dR[1, 1]
    	# @test 1.0 ≈ dR[2, 2]
    	# @test 1.0 ≈ dR[3, 3]
    	# @test 1.0 ≈ dR[4, 4]
    	# @test 1.0 ≈ dR[5, 5]
    	# @test 0.0 ≈ dR[6, 6]
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

    x  = [2.0]
    dx = [1.0]

    Enzyme.autodiff(Reverse, invtest, Duplicated(x, dx))

    @test 10.0 ≈ x[1]
    @test 5.0 ≈ dx[1]
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

    res = Enzyme.autodiff(ForwardWithPrimal, fwdlatestfoo, BatchDuplicated, BatchDuplicated(x, (dx, dx2)))

    @test 2.0 ≈ res[1][1]
    @test 20.0 ≈ res[1][2]
    @test 2.0 ≈ res[2][1]

    res = Enzyme.autodiff(Forward, fwdlatestfoo, BatchDuplicated, BatchDuplicated(x, (dx, dx2)))
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

include("mixed.jl")
include("applyiter.jl")

@testset "Dynamic Val Construction" begin

    dyn_f(::Val{D}) where D = prod(D)
    dyn_mwe(x, t) = x / dyn_f(Val(t))

    @test 0.5 ≈ Enzyme.autodiff(Reverse, dyn_mwe, Active, Active(1.0), Const((1, 2)))[1][1]
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

@testset "Mismatched return" begin
    @test_throws ErrorException autodiff(Reverse, _->missing, Active, Active(2.1))
    @test_throws ErrorException autodiff_deferred(Reverse, Const(_->missing), Active, Active(2.1))
end

@testset "GCPreserve" begin
    function f(x, y)
        GC.@preserve x y begin
            ccall(:memcpy, Cvoid,
                (Ptr{Float64},Ptr{Float64},Csize_t), x, y, 8)
        end
        nothing
    end
    autodiff(Reverse, f, Duplicated([1.0], [0.0]), Duplicated([1.0], [0.0]))
    autodiff(Forward, f, Duplicated([1.0], [0.0]), Duplicated([1.0], [0.0]))
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


@testset "GCPreserve2" begin
    function f!(a_out, a_in)
           a_out[1:end-1] .= a_in[2:end]
           return nothing
    end
    a_in = rand(4)
    a_out = a_in

    shadow_a_out = ones(4)
    shadow_a_in = shadow_a_out

    autodiff(Reverse, f!, Const, Duplicated(a_out, shadow_a_out), Duplicated(a_in, shadow_a_in))

    @test shadow_a_in ≈ Float64[0.0, 1.0, 1.0, 2.0]
    @test shadow_a_out ≈ Float64[0.0, 1.0, 1.0, 2.0]

    autodiff(Forward, f!, Const, Duplicated(a_out, shadow_a_out), Duplicated(a_in, shadow_a_in))

    @test shadow_a_in ≈ Float64[1.0, 1.0, 2.0, 2.0]
    @test shadow_a_out ≈ Float64[1.0, 1.0, 2.0, 2.0]
end

@testset "UndefVar" begin
    function f_undef(x, y)
        if x
            undefinedfnthowmagic()
        end
        y
    end
    @test 1.0 ≈ autodiff(Reverse, f_undef, Const(false), Active(2.14))[1][2]
    @test_throws Base.UndefVarError autodiff(Reverse, f_undef, Const(true), Active(2.14))

    @test 1.0 ≈ autodiff(Forward, f_undef, Const(false), Duplicated(2.14, 1.0))[1]
    @test_throws Base.UndefVarError autodiff(Forward, f_undef, Const(true), Duplicated(2.14, 1.0))
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

    @test 0.0 ≈ autodiff(Reverse, tobedifferentiated, Const(true), Active(2.1))[1][2]
	@test 0.0 ≈ autodiff(Forward, tobedifferentiated, Const(true), Duplicated(2.1, 1.0))[1]

	function tobedifferentiated2(cond, a)::Float64
		if cond
			a + t
		else
			0.0
		end
	end

    @test 1.0 ≈ autodiff(Reverse, tobedifferentiated2, Const(true), Active(2.1))[1][2]
	@test 1.0 ≈ autodiff(Forward, tobedifferentiated2, Const(true), Duplicated(2.1, 1.0))[1]

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

    autodiff(Reverse, mer, Duplicated(F, L), Duplicated(F_H, L_H), Const(true))
    autodiff(Forward, mer, Duplicated(F, L), Duplicated(F_H, L_H), Const(true))
end

@testset "GC Sret" begin
    @noinline function _get_batch_statistics(x)
        batchmean = @inbounds x[1]
        return (x, x)
    end

    @noinline function _normalization_impl(x)
        _stats = _get_batch_statistics(x)
        return x
    end

    function gcloss(x)
        _normalization_impl(x)[1]
        return nothing
    end

    x = randn(10)
    dx = zero(x)

    Enzyme.autodiff(Reverse, gcloss, Duplicated(x, dx))
end

typeunknownvec = Float64[]

@testset "GC Sret 2" begin

    struct AGriddedInterpolation{K<:Tuple{Vararg{AbstractVector}}} <: AbstractArray{Float64, 1}
        knots::K
        v::Int
    end

    function AGriddedInterpolation(A::AbstractArray{Float64, 1})
        knots = (A,)
        use(A)
        AGriddedInterpolation{typeof(knots)}(knots, 2)
    end

    function ainterpolate(A::AbstractArray{Float64,1})
        AGriddedInterpolation(A)
    end

    function cost(C::Vector{Float64})
        zs = typeunknownvec
        ainterpolate(zs)
        return nothing
    end

    A = Float64[]
    dA = Float64[]
    @test_throws Base.UndefVarError autodiff(Reverse, cost, Const, Duplicated(A, dA))
end

@testset "No Decayed / GC" begin
    @noinline function deduplicate_knots!(knots)
        last_knot = first(knots)
        for i = eachindex(knots)
            if i == 1
                continue
            end
            if knots[i] == last_knot
                @warn knots[i]
                @inbounds knots[i] *= knots[i]
            else
                last_knot = @inbounds knots[i]
            end
        end
    end

    function cost(C::Vector{Float64})
        deduplicate_knots!(C)
        @inbounds C[1] = 0
        return nothing
    end
    A = Float64[1, 3, 3, 7]
    dA = Float64[1, 1, 1, 1]
    autodiff(Reverse, cost, Const, Duplicated(A, dA))
    @test dA ≈ [0.0, 1.0, 6.0, 1.0]
end

@testset "Split GC" begin
    @noinline function bmat(x)
        data = [x]
        return data
    end

    function f(x::Float64)
        @inbounds return bmat(x)[1]
    end
    @test 1.0 ≈ autodiff(Reverse, f, Active(0.1))[1][1]
    @test 1.0 ≈ autodiff(Forward, f, Duplicated(0.1, 1.0))[1]
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

@testset "No inference" begin
    c = 5.0
    @test 5.0 ≈ autodiff(Reverse, (A,)->c * A, Active, Active(2.0))[1][1]
    @test 5.0 ≈ autodiff(Forward, (A,)->c * A, Duplicated(2.0, 1.0))[1]
end

@testset "Recursive GC" begin
    function modf!(a)
        as = [zero(a) for _ in 1:2]
        a .+= sum(as)
        return nothing
    end

    a = rand(5)
    da = zero(a)
    autodiff(Reverse, modf!, Duplicated(a, da))
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

@testset "Broadcast noalias" begin

    x = ones(30)
    
    @static if VERSION < v"1.11-"
        autodiff(Reverse, bc0_test_function, Active, Const(x))
    else
        # TODO
        @test_broken autodiff(Reverse, bc0_test_function, Active, Const(x))
    end
    
    x = rand(Float32, 2, 3)
    Enzyme.autodiff(Reverse, bc1_loss_function, Duplicated(x, zero(x)))

    x = rand(Float32, 6, 6, 6, 2)
    sc = rand(Float32, 6)
    bi = rand(Float32, 6)
    Enzyme.autodiff(Reverse, bc2_loss_function, Active, Duplicated(x, Enzyme.make_zero(x)),
        Duplicated(sc, Enzyme.make_zero(sc)), Duplicated(bi, Enzyme.make_zero(bi)))
end

function solve_cubic_eq(poly::AbstractVector{Complex{T}}) where T
    a1  =  1 / @inbounds poly[1]
    E1  = 2*a1
    E12 =  E1*E1
    s1 = log(E12)
    return nothing
end

@testset "Extract Tuple for Reverse" begin
    autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(solve_cubic_eq)}, Const, Duplicated{Vector{Complex{Float64}}})
end


@testset "GetField" begin
    mutable struct MyType
       x::Float64
    end

    getfield_idx(v, idx) = ccall(:jl_get_nth_field_checked, Any, (Any, UInt), v, idx)

    function gf(v::MyType, fld::Symbol)
       x = getfield(v, fld)
       x = x::Float64
       2 * x
    end

    function gf(v::MyType, fld::Integer)
       x = getfield_idx(v, fld)
       x = x::Float64
       2 * x
    end

    function gf2(v::MyType, fld::Integer, fld2::Integer)
       x = getfield_idx(v, fld)
       y = getfield_idx(v, fld2)
       x + y
    end

    function gf2(v::MyType, fld::Symbol, fld2::Symbol)
       x = getfield(v, fld)
       y = getfield(v, fld2)
       x + y
    end

    mx = MyType(3.0)
    dx = MyType(0.0)

    Enzyme.autodiff(Reverse, gf, Active, Duplicated(mx, dx), Const(:x))
    @test mx.x ≈ 3.0
    @test dx.x ≈ 2.0


    mx = MyType(3.0)
    dx = MyType(0.0)

    Enzyme.autodiff(Reverse, gf, Active, Duplicated(mx, dx), Const(0))
    @test mx.x ≈ 3.0
    @test dx.x ≈ 2.0


    mx = MyType(3.0)
    dx = MyType(0.0)

    Enzyme.autodiff(Reverse, gf2, Active, Duplicated(mx, dx), Const(:x), Const(:x))
    @test mx.x ≈ 3.0
    @test dx.x ≈ 2.0

    mx = MyType(3.0)
    dx = MyType(0.0)

    Enzyme.autodiff(Reverse, gf2, Active, Duplicated(mx, dx), Const(0), Const(0))
    @test mx.x ≈ 3.0
    @test dx.x ≈ 2.0

    function forbatch(v, fld::Symbol, out)
        x = getfield(v, fld)
        x = x::Float64
        out[] = 2 * x
        nothing
    end
    function forbatch(v, fld::Integer, out)
        x = getfield_idx(v, fld)
        x = x::Float64
        out[] = 2 * x
        nothing
    end

    mx = MyType(3.0)
    dx = MyType(0.0)
    dx2 = MyType(0.0)

    Enzyme.autodiff(Reverse, forbatch, Const, BatchDuplicated(mx, (dx, dx2)), Const(:x), BatchDuplicated(Ref(0.0), (Ref(1.0), Ref(3.14))))
    @test mx.x ≈ 3.0
    @test dx.x ≈ 2.0
    @test dx2.x ≈ 6.28

    mx = MyType(3.0)
    dx = MyType(0.0)
    dx2 = MyType(0.0)

    Enzyme.autodiff(Reverse, forbatch, Const, BatchDuplicated(mx, (dx, dx2)), Const(0), BatchDuplicated(Ref(0.0), (Ref(1.0), Ref(3.14))))
    @test mx.x ≈ 3.0
    @test dx.x ≈ 2.0
    @test dx2.x ≈ 6.28

    mutable struct MyType2
       x::Float64
       y::Float64
    end

    function sf2(v::MyType2, fld, fld2)
       x = getfield(v, fld)
       x = x::Float64
       r = 2 * x
       x = setfield!(v, fld2, r)
       return nothing
    end

    mt2 = MyType2(3.0, 642.0)
    dmt2 = MyType2(1.2, 541.0)

    Enzyme.autodiff(Forward, sf2, Duplicated(mt2, dmt2), Const(:x), Const(:y))
    @test mt2.x ≈ 3.0
    @test mt2.y ≈ 6.0
    @test dmt2.x ≈ 1.2
    @test dmt2.y ≈ 2.4

    function sf_for2(v, fld, fld2, x)
       setfield!(v, fld, 0.0)
       for i in 1:100
            setfield!(v, fld2, getfield(v, fld)::Float64 + x * i)
       end
       return getfield(v, fld)::Float64
    end

    mt2 = MyType2(0.0, 0.0)
    dmt2 = MyType2(0.0, 0.0)

    adres = Enzyme.autodiff(Reverse, sf_for2, Duplicated(mt2, dmt2), Const(:x), Const(:x), Active(3.1))
    @test adres[1][4] ≈ 5050.0

    mutable struct MyType3
       x::Base.RefValue{Float64}
       y::Base.RefValue{Float64}
    end

    function sf_for3(v, fld, fld2, x)
       setfield!(v, fld, Ref(0.0))
       for i in 1:100
            setfield!(v, fld2, Base.Ref((getfield(v, fld)::Base.RefValue{Float64})[] + x * i))
       end
       return (getfield(v, fld)::Base.RefValue{Float64})[]
    end

    mt3 = MyType3(Ref(0.0), Ref(0.0))
    dmt3 = MyType3(Ref(0.0), Ref(0.0))

    adres = Enzyme.autodiff(Reverse, sf_for3, Duplicated(mt3, dmt3), Const(:x), Const(:x), Active(3.1))
    @test adres[1][4] ≈ 5050.0
    
    mutable struct MyTypeM
       x::Float64
       y
    end

    @noinline function unstable_mul(x, y)
        return (x*y)::Float64
    end

    function gf3(y, v::MyTypeM, fld::Symbol)
       x = getfield(v, fld)
       unstable_mul(x, y)
    end

    function gf3(y, v::MyTypeM, fld::Integer)
       x = getfield_idx(v, fld)
       unstable_mul(x, y)
    end
    
    mx = MyTypeM(3.0, 1)
    res = Enzyme.autodiff(Reverse, gf3, Active, Active(2.7), Const(mx), Const(:x))
    @test mx.x ≈ 3.0
    @test res[1][1] ≈ 3.0
    
    mx = MyTypeM(3.0, 1)
    res = Enzyme.autodiff(Reverse, gf3, Active, Active(2.7), Const(mx), Const(0))
    @test mx.x ≈ 3.0
    @test res[1][1] ≈ 3.0
end


struct GFUniform{T}
    a::T
    b::T
end
GFlogpdf(d::GFUniform, ::Real) = -log(d.b - d.a)

struct GFNormal{T}
    μ::T
    σ::T
end
GFlogpdf(d::GFNormal, x::Real) = -(x - d.μ)^2 / (2 * d.σ^2)

struct GFProductDist{V}
    dists::V
end
function GFlogpdf(d::GFProductDist, x::Vector)
    dists = d.dists
    s = zero(eltype(x))
    for i in eachindex(x)
	s += GFlogpdf(dists[i], x[i])
    end
    return s
end

struct GFNamedDist{Names, D<:NamedTuple{Names}}
    dists::D
end

function GFlogpdf(d::GFNamedDist{N}, x::NamedTuple{N}) where {N}
    vt = values(x)
    dists = d.dists
    return mapreduce((dist, acc) -> GFlogpdf(dist, acc), +, dists, vt)
end


@testset "Getfield with reference" begin
    d = GFNamedDist((;a = GFNormal(0.0, 1.0), b = GFProductDist([GFUniform(0.0, 1.0), GFUniform(0.0, 1.0)])))
    p = (a = 1.0, b = [0.5, 0.5])
    dp = Enzyme.make_zero(p)
    GFlogpdf(d, p)
    autodiff(set_runtime_activity(Reverse), GFlogpdf, Active, Const(d), Duplicated(p, dp))
end

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

@testset "Exception" begin

    f_no_derv(x) = ccall("extern doesnotexist", llvmcall, Float64, (Float64,), x)
    @test_throws Enzyme.Compiler.EnzymeNoDerivativeError autodiff(Reverse, f_no_derv, Active, Active(0.5))

    f_union(cond, x) = cond ? x : 0
    g_union(cond, x) = f_union(cond,x)*x
    if sizeof(Int) == sizeof(Int64)
        @test_throws Enzyme.Compiler.IllegalTypeAnalysisException autodiff(Reverse, g_union, Active, Const(true), Active(1.0))
    else
        @test_throws Enzyme.Compiler.IllegalTypeAnalysisException autodiff(Reverse, g_union, Active, Const(true), Active(1.0f0))
    end
    # TODO: Add test for NoShadowException
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

function objective!(x, loss, R)
    for i in 1:1000
        y = zeros(3)
        y[1] = R[1,1] * x[1] + R[1,2] * x[2] + R[1,3] * x[3]

        loss[] = y[1]
    end
    return nothing
end;

@testset "Static tape allocation" begin
    x = zeros(3)
    R = [1.0 0.0 0.0
        0.0 1.0 0.0
        0.0 0.0 1.0]
    loss = Ref(0.0)
    dloss = Ref(1.0)

    autodiff(Reverse, objective!, Duplicated(x, zero(x)), Duplicated(loss, dloss), Const(R))

    @test loss[] ≈ 0.0
    @show dloss[] ≈ 0.0
end

@testset "Union return" begin
    function unionret(a, out, cond)
        if cond
            out[] = a
        end
    end

    out = Ref(0.0)
    dout = Ref(1.0)
    @test 2.0 ≈ Enzyme.autodiff(Reverse, unionret, Active, Active(2.0), Duplicated(out, dout), Const(true))[1][1]
end


function assured_err(x)
    throw(AssertionError("foo"))
end

@testset "UnionAll" begin
    @test_throws AssertionError Enzyme.autodiff(Reverse, assured_err, Active, Active(2.0))
end

struct MyFlux
end

@testset "Union i8" begin
    args = (
        Val{(false, false, false)},
        Val(false),
        Val(1),
        Val((true, true, true)),
        Base.Val(NamedTuple{(Symbol("1"), Symbol("2"), Symbol("3")), Tuple{Any, Any, Any}}),
        Base.getindex,
        nothing,
        ((nothing,), MyFlux()),
        ((nothing,), MyFlux()),
        1,
        nothing
    )
    
    nt1 = Enzyme.Compiler.runtime_generic_augfwd(args...)
    @test nt1[1] == (nothing,)
    @test nt1[2] == (nothing,)
    
    args2 = (
        Val{(false, false, false)},
        Val(false),
        Val(1),
        Val((true, true, true)),
        Base.Val(NamedTuple{(Symbol("1"), Symbol("2"), Symbol("3")), Tuple{Any, Any, Any}}),
        Base.getindex,
        nothing,
        ((nothing,), MyFlux()),
        ((nothing,), MyFlux()),
        2,
        nothing
    )
    
    nt = Enzyme.Compiler.runtime_generic_augfwd(args2...)
    @test nt[1] == MyFlux()
    @test nt[2] == MyFlux()
end

@testset "Batched inactive" begin
    augres = Enzyme.Compiler.runtime_generic_augfwd(Val{(false, false, false)}, Val(false), Val(2), Val((true, true, true)),
                                                    Val(Enzyme.Compiler.AnyArray(2+Int(2))),
                                ==, nothing, nothing,
                                :foo, nothing, nothing,
                                :bar, nothing, nothing)

    Enzyme.Compiler.runtime_generic_rev(Val{(false, false, false)}, Val(false), Val(2), Val((true, true, true)), augres[end],
                                ==, nothing, nothing,
                                :foo, nothing, nothing,
                                :bar, nothing, nothing)
end

@testset "Array push" begin

    function pusher(x, y)
        push!(x, y)
        x[1] + x[2]
    end

    x  = [2.3]
    dx = [0.0]
    rf = @static if VERSION < v"1.11-"
        nothing
    else
        dx.ref.mem
    end
    @test 1.0 ≈ first(Enzyme.autodiff(Reverse, pusher, Duplicated(x, dx), Active(2.0)))[2]
    @static if VERSION < v"1.11-"
        @test dx ≈ [1.0]
    else
        @test dx ≈ [0.0, 0.0]
        @test rf ≈ [1.0,]
    end

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

@testset "Batch Forward" begin
    square(x)=x*x
    bres = autodiff(Forward, square, BatchDuplicated, BatchDuplicated(3.0, (1.0, 2.0, 3.0)))
    @test length(bres) == 1
    @test length(bres[1]) == 3
    @test bres[1][1] ≈  6.0
    @test bres[1][2] ≈ 12.0
    @test bres[1][3] ≈ 18.0

    bres = autodiff(Forward, square, BatchDuplicated, BatchDuplicated(3.0 + 7.0im, (1.0+0im, 2.0+0im, 3.0+0im)))
    @test bres[1][1] ≈  6.0 + 14.0im
    @test bres[1][2] ≈ 12.0 + 28.0im
    @test bres[1][3] ≈ 18.0 + 42.0im

    squareidx(x)=x[1]*x[1]
    inp = Float32[3.0]

    # Shadow offset is not the same as primal so following doesn't work
    # d_inp = Float32[1.0, 2.0, 3.0]
    # autodiff(Forward, squareidx, BatchDuplicated, BatchDuplicated(view(inp, 1:1), (view(d_inp, 1:1), view(d_inp, 2:2), view(d_inp, 3:3))))

    d_inp = (Float32[1.0], Float32[2.0], Float32[3.0])
    bres = autodiff(Forward, squareidx, BatchDuplicated, BatchDuplicated(inp, d_inp))
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

    grad = Enzyme.gradient(Reverse, z -> (z.x * z.y), (x=5.0, y=6.0))[1]
    @test grad == (x = 6.0, y = 5.0)

    grad = Enzyme.gradient(Reverse, abs2, 7.0)[1]
    @test grad == 14.0
end

include("sugar.jl")

@testset "Forward on Reverse" begin

	function speelpenning(y, x)
		ccall(:memmove, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
								  y, x, 2 * 8)
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

using  Documenter
DocMeta.setdocmeta!(Enzyme, :DocTestSetup, :(using Enzyme); recursive=true)
@testset "DocTests" begin
    doctest(Enzyme; manual = false)
end

using Random

@testset "Random" begin
    f_rand(x) = x*rand()
    f_randn(x, N) = x*sum(randn(N))
    @test 0 <= autodiff(Reverse, f_rand, Active, Active(1.0))[1][1] < 1
    @test !iszero(autodiff(Reverse, f_randn, Active, Active(1.0), Const(64))[1][1])
    @test iszero(autodiff(Reverse, x -> rand(), Active, Active(1.0))[1][1])
    @test iszero(autodiff(Reverse, (x, N) -> sum(randn(N)), Active, Active(1.0), Const(64))[1][1])
    @test autodiff(Reverse, x -> x * sum(randcycle(5)), Active, Active(1.0))[1][1] == 15
    @test autodiff(Reverse, x -> x * sum(randperm( 5)), Active, Active(1.0))[1][1] == 15
    @test autodiff(Reverse, x -> x * sum(shuffle(1:5)), Active, Active(1.0))[1][1] == 15
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
        Base.copyto!(W, reshape(view(params,i:(i+length(W)-1)), size(W)))
    end
end

@testset "Illegal phi erasure" begin
    # just check that it compiles
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(set_paramsPE)}, Const, Duplicated{NNPE}, Duplicated{Vector{Float64}})
    @test fwd !== nothing
    @test rev !== nothing
    nn = NNPE( ( DensePE(1, Matrix{Float64}(undef, 4, 4)), DensePE(1, Matrix{Float64}(undef, 4, 4)) ) )
    dnn = NNPE( ( DensePE(1, Matrix{Float64}(undef, 4, 4)), DensePE(1, Matrix{Float64}(undef, 4, 4)) ) )
    l = Vector{Float64}(undef, 32)
    dl = Vector{Float64}(undef, 32)
    fwd(Const(set_paramsPE), Duplicated(nn, dnn), Duplicated(l, dl))
end

@testset "Copy Broadcast arg" begin
	x = Float32[3]
	w = Float32[1]
	dw = zero(w)

	function inactiveArg(w, x, cond)
	   if cond
		  x = copy(x)
	   end
	  @inbounds w[1] * x[1]
	end

    @static if VERSION < v"1.11-"
    	Enzyme.autodiff(Reverse, inactiveArg, Active, Duplicated(w, dw), Const(x), Const(false))

        @test x ≈ [3.0]
        @test w ≈ [1.0]
        @test dw ≈ [3.0]
    else
        # TODO broken should not throw
        @test_throws Enzyme.Compiler.EnzymeRuntimeActivityError Enzyme.autodiff(Reverse, inactiveArg, Active, Duplicated(w, dw), Const(x), Const(false))
	Enzyme.autodiff(set_runtime_activity(Reverse), inactiveArg, Active, Duplicated(w, dw), Const(x), Const(false))
    end

    x = Float32[3]

    function loss(w, x, cond)
      dest = Array{Float32}(undef, 1)
      r = cond ? copy(x) : x
      res = @inbounds w[1] * r[1]
      @inbounds dest[1] = res
      res
    end

    @static if VERSION < v"1.11-"
        dw = Enzyme.autodiff(Reverse, loss, Active, Active(1.0), Const(x), Const(false))[1]

    else
        # TODO broken should not throw
        @test_throws Enzyme.Compiler.EnzymeRuntimeActivityError Enzyme.autodiff(Reverse, loss, Active, Active(1.0), Const(x), Const(false))[1]
	dw = Enzyme.autodiff(set_runtime_activity(Reverse), loss, Active, Active(1.0), Const(x), Const(false))[1]
    end
    
    @test x ≈ [3.0]
    @test dw[1] ≈ 3.0

    c = ones(3)
    inner(e) = c .+ e

    @static if VERSION < v"1.11-"
        fres = Enzyme.autodiff(Enzyme.Forward, Const(inner), Duplicated{Vector{Float64}}, Duplicated([0., 0., 0.], [1., 1., 1.]))[1]
    else
        # TODO broken should not throw
        @test_throws Enzyme.Compiler.EnzymeRuntimeActivityError Enzyme.autodiff(Enzyme.Forward, Const(inner), Duplicated{Vector{Float64}}, Duplicated([0., 0., 0.], [1., 1., 1.]))
	fres = Enzyme.autodiff(set_runtime_activity(Enzyme.Forward), Const(inner), Duplicated{Vector{Float64}}, Duplicated([0., 0., 0.], [1., 1., 1.]))
    end
    @test c ≈ [1.0, 1.0, 1.0]
    @test fres ≈ [1.0, 1.0, 1.0]
end

@testset "View Splat" begin
	function getloc(locs, i)
		loss = 0.0
		if i==1
			x, y = 0.0, 0.0
		else
		# correct
			# x, y = locs[1,i-1], locs[2,i-1]
		# incorrect
		x, y = @inbounds locs[:,i-1]
		end
		loss += y
		return loss
	end

	x0 = ones(2, 9)
	din = zeros(2, 9)
	Enzyme.autodiff(Reverse, getloc, Duplicated(x0, din), Const(2))
	@test din[1, 1] ≈ 0.0
	@test din[2, 1] ≈ 1.0
end

@testset "View Vars" begin

    x = [Float32(0.25)]
    dx = [Float32(0.0)]
    rng = Base.UnitRange{Int64}(1, 0)

    f = Const(Base.SubArray{T, N, P, I, L} where L where I where P where N where T)
    a1 = Const(Base.IndexLinear())
    a2 = Duplicated(x, dx)
    a3 = Const((rng,))
    a4 = Const((true,))

    fwd, rev = autodiff_thunk(ReverseSplitWithPrimal,
         typeof(f),
         Duplicated,
         typeof(a1),
         typeof(a2),
         typeof(a3),
         typeof(a4)
    )

    res = fwd(f,a1,a2,a3,a4)
    @test res[2].indices == (rng,)
    @test res[3].indices == (rng,)
    @test res[2].offset1 == 0
    @test res[3].offset1 == 0
    @test res[2].stride1 == 1
    @test res[3].stride1 == 1

    x = [Float32(0.25)]
    dx = [Float32(0.0)]
    rng = Base.UnitRange{Int64}(1, 0)

    f = Const(Base.SubArray{T, N, P, I, L} where L where I where P where N where T)
    a1 = Const(Base.IndexLinear())
    a2 = Duplicated(x, dx)
    a3 = Const((rng,))
    a4 = Const((true,))

    fwd, rev = autodiff_thunk(set_runtime_activity(ReverseSplitWithPrimal),
         typeof(f),
         Duplicated,
         typeof(a1),
         typeof(a2),
         typeof(a3),
         typeof(a4)
    )

    res = fwd(f,a1,a2,a3,a4)
    @test res[2].indices == (rng,)
    @test res[3].indices == (rng,)
    @test res[2].offset1 == 0
    @test res[3].offset1 == 0
    @test res[2].stride1 == 1
    @test res[3].stride1 == 1
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
@testset "Large dynamic tape" begin

	function ldynloss(X, Y, ps, bs)
		ll = 0.0f0
		for (x, y) in zip(X, Y)
			yhat = ps * x .+ bs
			ll += (yhat[1] - y)^2
		end
		return ll
	end

	ps = randn(Float32, (1, 5))
	bs = randn(Float32)

	X = map(x->rand(Float32, 5), 1:1000)
	Y = map(x->rand(Float32), 1:1000)

	grads = zero(ps)
	for epoch=1:1000
		fill!(grads, 0)
		autodiff(Reverse, ldynloss, Const(X), Const(Y), Duplicated(ps, grads), Active(bs))
	end

end

@testset "Union return getproperty" begin
	using Enzyme

	struct DOSData
		interp_func
	end

	function get_dos(Ef=0.)
		return x->x+Ef
	end

	struct MyMarcusHushChidseyDOS
		A::Float64
		dos::DOSData
	end

	mhcd = MyMarcusHushChidseyDOS(0.3,  DOSData(get_dos()));

	function myintegrand(V, a_r)
		function z(E)
			dos = mhcd.dos

			interp = dos.interp_func

			res = interp(V)

			return res
		end
		return z
	end

	function f2(V)
		fn = myintegrand(V, 1.0)

		fn(0.0)
	end

    res = autodiff(set_runtime_activity(ForwardWithPrimal), Const(f2), Duplicated, Duplicated(0.2, 1.0))
    @test res[2] ≈ 0.2
    # broken as the return of an apply generic is {primal, primal}
    # but since the return is abstractfloat doing the 
    @test res[1] ≈ 1.0
end

@inline function uns_mymean(f, A, ::Type{T}, c) where T
    c && return Base.inferencebarrier(nothing)
    x1 = f(@inbounds A[1]) / 1
    return @inbounds A[1][1]
end

function uns_sum2(x::Array{T})::T where T
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

function uns_ad_forward(scale_diag::Vector{T}, c) where T 
    ccall(:jl_, Cvoid, (Any,), scale_diag) 
    res = uns_mymean(uns_sum2, [scale_diag,], T, c)
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

    bc   = (north=1, top=tuple(parameters, tuple(:c)))
    d_bc = Enzyme.make_zero(bc)
    Enzyme.API.looseTypeAnalysis!(true)

    dc²_dκ = autodiff(Enzyme.Reverse,
                      permute_boundary_conditions,
                      Duplicated(bc, d_bc))

    Enzyme.API.looseTypeAnalysis!(false)
end


@testset "Static activity" begin

    struct Test2{T}
        obs::T
    end

    function test(t, x)
        o = t.obs
        y = (x .- o)
        yv = @inbounds y[1]
        return yv*yv
    end

    obs = [1.0]
    t = Test2(obs)

    x0 = [0.0]
    dx0 = [0.0]

    autodiff(Reverse, test, Const(t), Duplicated(x0, dx0))

    @test obs[1] ≈ 1.0
    @test x0[1] ≈ 0.0
    @test dx0[1] ≈ -2.0

end

@testset "Const Activity through intermediate" begin
    struct RHS_terms
        eta1::Vector{Float64}
        u_t::Vector{Float64}
        eta_t::Vector{Float64}
    end

    @noinline function comp_u_v_eta_t(rhs)
        Base.unsafe_copyto!(rhs.eta_t, 1, rhs.u_t, 1, 1)
        return nothing
    end

    function advance(eta, rhs)

        @inbounds rhs.eta1[1] = @inbounds eta[1]

        comp_u_v_eta_t(rhs)

        @inbounds eta[1] = @inbounds rhs.eta_t[1]

        return nothing

    end

    rhs_terms = RHS_terms(zeros(1), zeros(1), zeros(1))

    u_v_eta = Float64[NaN]
    ad_eta = zeros(1)

    autodiff(Reverse, advance,
        Duplicated(u_v_eta, ad_eta),
        Const(rhs_terms),
    )
    @test ad_eta[1] ≈ 0.0
end

function absset(out, x)
    @inbounds out[1] = (x,)
    return nothing
end

@testset "Abstract Array element type" begin
    out = Tuple{Any}[(9.7,)]
    dout = Tuple{Any}[(4.3,)]

    autodiff(Enzyme.Forward,
                      absset,
                      Duplicated(out, dout),
                      Duplicated(3.1, 2.4)
                      )
    @test dout[1][1] ≈ 2.4
end

@testset "Tape Width" begin
    struct Roo
        x::Float64
        bar::String63
    end

    struct Moo
        x::Float64
        bar::String63
    end

    function g(f)
        return f.x*5.0
    end

    res = autodiff(Reverse, g, Active, Active(Roo(3.0, "a")))[1][1]

    @test res.x == 5.0

    res = autodiff(Reverse, g, Active, Active(Moo(3.0, "a")))[1][1]

    @test res.x == 5.0
end

@testset "Type preservation" begin
    # Float16 fails due to #870
    for T in (Float64, Float32, #=Float16=#)
        res = autodiff(Reverse, x -> x * 2.0, Active, Active(T(1.0)))[1][1]
        @test res isa T
        @test res == 2
    end
end

struct GDoubleField{T}
    this_field_does_nothing::T
    b::T
end

GDoubleField() = GDoubleField{Float64}(0.0, 1.0)
function fexpandempty(vec)
    x = vec[1]
    empty = []
    d = GDoubleField(empty...)
    return x ≤ d.b ? x * d.b : zero(x)
end

@testset "Constant Complex return" begin
    vec = [0.5]
    @test Enzyme.gradient(Enzyme.Reverse, fexpandempty, vec)[1] ≈ [1.0]
    @test Enzyme.gradient(Enzyme.Forward, fexpandempty, vec)[1] ≈ [1.0]
end

const CUmemoryPool2 = Ptr{Float64} 

struct CUmemPoolProps2
    reserved::NTuple{31,Char}
end

mutable struct CuMemoryPool2
    handle::CUmemoryPool2
end

function ccall_macro_lower(func, rettype, types, args, nreq)
    # instead of re-using ccall or Expr(:foreigncall) to perform argument conversion,
    # we need to do so ourselves in order to insert a jl_gc_safe_enter|leave
    # just around the inner ccall

    cconvert_exprs = []
    cconvert_args = []
    for (typ, arg) in zip(types, args)
        var = gensym("$(func)_cconvert")
        push!(cconvert_args, var)
        push!(cconvert_exprs, quote
            $var = Base.cconvert($(esc(typ)), $(esc(arg)))
        end)
    end

    unsafe_convert_exprs = []
    unsafe_convert_args = []
    for (typ, arg) in zip(types, cconvert_args)
        var = gensym("$(func)_unsafe_convert")
        push!(unsafe_convert_args, var)
        push!(unsafe_convert_exprs, quote
            $var = Base.unsafe_convert($(esc(typ)), $arg)
        end)
    end

    quote
        $(cconvert_exprs...)

        $(unsafe_convert_exprs...)

        ret = ccall($(esc(func)), $(esc(rettype)), $(Expr(:tuple, map(esc, types)...)),
                    $(unsafe_convert_args...))
    end
end

macro gcsafe_ccall(expr)
    ccall_macro_lower(Base.ccall_macro_parse(expr)...)
end

function cuMemPoolCreate2(pool, poolProps)
    # CUDA.initialize_context()
    #CUDA.
    gc_state = @ccall(jl_gc_safe_enter()::Int8)
    @gcsafe_ccall cuMemPoolCreate(pool::Ptr{CUmemoryPool2},
                                          poolProps::Ptr{CUmemPoolProps2})::Cvoid
    @ccall(jl_gc_safe_leave(gc_state::Int8)::Cvoid)
end

function cual()
        props = Ref(CUmemPoolProps2( 
            ntuple(i->Char(0), 31)
        ))
        handle_ref = Ref{CUmemoryPool2}()
        cuMemPoolCreate2(handle_ref, props)

        CuMemoryPool2(handle_ref[])
end

@testset "Unused shadow phi rev" begin
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(cual)}, Duplicated)
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
    weights, means, stds = @views params[1:K], params[K+1:2K], params[2K+1:end]
    mat = normal_pdf.(data, means', stds' .^2) # (N, K)
    sum(mat .* weights', dims=2) .|> log |> sum
end

# another form of original objective (doesn't work)
function mixture_loglikelihood2(params::AbstractVector{<:Real}, data::AbstractVector{<:Real})::Real
    K = length(params) ÷ 3
    weights, means, stds = @views params[1:K], params[K+1:2K], params[2K+1:end]
    mat = normal_pdf.(data, means', stds' .^2) # (N, K)
    obj_true = sum(
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
    weights, means, stds = @views params[1:K], params[K+1:2K], params[2K+1:end]
    mat = normal_pdf.(data, means', stds' .^2) # (N, K)

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

@testset "Type unsstable return" begin
    expected =  [289.7308495620467,
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
                 12.87712891527131]
    @test expected ≈ Enzyme.gradient(Reverse, objective1, params0)[1]
    # objective2 fails from runtime activity requirements
    # @test expected ≈ Enzyme.gradient(Reverse, objective2, params0)[1]
    @test expected ≈ Enzyme.gradient(Reverse, objective3, params0)[1]
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

function demo_func(x::Any=transpose([1.5 2.0;]);)
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
    coords   = [(1.0, 2.0, 3.0), (1.1, 2.1, 3.1), (1.2, 2.2, 3.2)]
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

@testset "Statistics" begin
    f1(x) = var([x, 2.0, 3.0])
    @test autodiff(Reverse, f1, Active, Active(0.0))[1][1] ≈ -5/3
    @test autodiff(Forward, f1, Duplicated(0.0, 1.0))[1]   ≈ -5/3

    f2(x) = varm([x, 2.0, 3.0], 5/3)
    @test autodiff(Reverse, f2, Active, Active(0.0))[1][1] ≈ -5/3
    @test autodiff(Forward, f2, Duplicated(0.0, 1.0))[1]   ≈ -5/3

    f3(x) = std([x, 2.0, 3.0])
    @test autodiff(Reverse, f3, Active, Active(0.0))[1][1] ≈ -0.54554472559
    @test autodiff(Forward, f3, Duplicated(0.0, 1.0))[1]   ≈ -0.54554472559

    f4(x) = stdm([x, 2.0, 3.0], 5/3)
    @test autodiff(Reverse, f4, Active, Active(0.0))[1][1] ≈ -0.54554472559
    @test autodiff(Forward, f4, Duplicated(0.0, 1.0))[1]   ≈ -0.54554472559

    f5(x) = cor([2.0, x, 1.0], [1.0, 2.0, 3.0])
    @test autodiff(Reverse, f5, Active, Active(4.0))[1][1] ≈ 0.11690244120
    @test autodiff(Forward, f5, Duplicated(4.0, 1.0))[1]   ≈ 0.11690244120

    f6(x) = cov([2.0, x, 1.0])
    @test autodiff(Reverse, f6, Active, Active(4.0))[1][1] ≈ 5/3
    @test autodiff(Forward, f6, Duplicated(4.0, 1.0))[1]   ≈ 5/3

    f7(x) = median([2.0, 1.0, x])
    @test autodiff(Reverse, f7, Active, Active(1.5))[1][1] == 1
    @test autodiff(Forward, f7, Duplicated(1.5, 1.0))[1]   == 1
    @test autodiff(Reverse, f7, Active, Active(2.5))[1][1] == 0
    @test autodiff(Forward, f7, Duplicated(2.5, 1.0))[1]   == 0

    f8(x) = middle([2.0, x, 1.0])
    @test autodiff(Reverse, f8, Active, Active(2.5))[1][1] == 0.5
    @test autodiff(Forward, f8, Duplicated(2.5, 1.0))[1]   == 0.5
    @test autodiff(Reverse, f8, Active, Active(1.5))[1][1] == 0
    @test autodiff(Forward, f8, Duplicated(1.5, 1.0))[1]   == 0

    f9(x) = sum(quantile([1.0, x], [0.5, 0.7]))
    @test autodiff(Reverse, f9, Active, Active(2.0))[1][1] == 1.2
    @test autodiff(Forward, f9, Duplicated(2.0, 1.0))[1]   == 1.2
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

# TEST EXTENSIONS 
using SpecialFunctions
@testset "SpecialFunctions ext" begin
    lgabsg(x) = SpecialFunctions.logabsgamma(x)[1]
    test_scalar(lgabsg, 1.0; rtol = 1.0e-5, atol = 1.0e-5)
    test_scalar(lgabsg, 1.0f0; rtol = 1.0e-5, atol = 1.0e-5)
end

using ChainRulesCore
@testset "ChainRulesCore ext" begin
    include("ext/chainrulescore.jl")
end
include("ext/logexpfunctions.jl")

@testset "BFloat16s ext" begin
    include("ext/bfloat16s.jl")
end

include("ext/jlarrays.jl")
include("ext/sparsearrays.jl")
include("ext/staticarrays.jl")
