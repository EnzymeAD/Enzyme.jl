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
include("optimize.jl")

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

    world = codegen_world_age(typeof(f0), Tuple{Float64})
    thunk_a = Enzyme.Compiler.thunk(Val(world), Const{typeof(f0)}, Active, Tuple{Active{Float64}}, Val(API.DEM_ReverseModeCombined), Val(1), Val((false, false)), Val(false), Val(false), DefaultABI, Val(false), Val(false))
    thunk_b = Enzyme.Compiler.thunk(Val(world), Const{typeof(f0)}, Const, Tuple{Const{Float64}}, Val(API.DEM_ReverseModeCombined), Val(1), Val((false, false)), Val(false), Val(false), DefaultABI, Val(false), Val(false))
    thunk_c = Enzyme.Compiler.thunk(Val(world), Const{typeof(f0)}, Active{Float64}, Tuple{Active{Float64}}, Val(API.DEM_ReverseModeCombined), Val(1), Val((false, false)), Val(false), Val(false), DefaultABI, Val(false), Val(false))
    thunk_d = Enzyme.Compiler.thunk(Val(world), Const{typeof(f0)}, Active{Float64}, Tuple{Active{Float64}}, Val(API.DEM_ReverseModeCombined), Val(1), Val((false, false)), Val(false), Val(false), DefaultABI, Val(false), Val(false))
    @test thunk_a.adjoint !== thunk_b.adjoint
    @test thunk_c.adjoint === thunk_a.adjoint
    @test thunk_c.adjoint === thunk_d.adjoint

    @test thunk_a(Const(f0), Active(2.0), 1.0) == ((1.0,),)
    @test thunk_a(Const(f0), Active(2.0), 2.0) == ((2.0,),)
    @test thunk_b(Const(f0), Const(2.0)) === ((nothing,),)

    forward, pullback = Enzyme.Compiler.thunk(Val(world), Const{typeof(f0)}, Active, Tuple{Active{Float64}}, Val(Enzyme.API.DEM_ReverseModeGradient), Val(1), Val((false, false)), Val(false), Val(false), DefaultABI, Val(false), Val(false))

    @test forward(Const(f0), Active(2.0)) == (nothing,nothing,nothing)
    @test pullback(Const(f0), Active(2.0), 1.0, nothing) == ((1.0,),)

    function mul2(x)
        x[1] * x[2]
    end
    d = Duplicated([3.0, 5.0], [0.0, 0.0])

    world = codegen_world_age(typeof(mul2), Tuple{Vector{Float64}})
    forward, pullback = Enzyme.Compiler.thunk(Val(world), Const{typeof(mul2)}, Active, Tuple{Duplicated{Vector{Float64}}}, Val(Enzyme.API.DEM_ReverseModeGradient), Val(1), Val((false, true)), Val(false), Val(false), DefaultABI, Val(false), Val(false))
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
    world = codegen_world_age(typeof(vrec), Tuple{Int, Vector{Float64}})
    forward, pullback = Enzyme.Compiler.thunk(Val(world), Const{typeof(vrec)}, Active, Tuple{Const{Int}, Duplicated{Vector{Float64}}}, Val(Enzyme.API.DEM_ReverseModeGradient), Val(1), Val((false, false, true)), Val(false), Val(false), DefaultABI, Val(false), Val(false))
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
end
