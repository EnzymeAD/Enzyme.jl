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
include("runtime_calls.jl")

include("rules.jl")
include("rrules.jl")
include("kwrules.jl")
include("kwrrules.jl")
include("internal_rules.jl")
include("ruleinvalidation.jl")
include("typeunstable.jl")
include("absint.jl")
include("array.jl")

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

    @test forward(Const(f0), Active(2.0)) == (nothing,nothing,nothing)
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
    @test typeof(res[1]) == NamedTuple{(Symbol("1"),Symbol("2"),Symbol("3")), Tuple{Any, Float64, Float64}}
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