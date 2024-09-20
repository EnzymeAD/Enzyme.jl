using Enzyme, Test

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
    @test typeof(res[1]) == Tuple{Float64, Float64}
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

