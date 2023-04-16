using Enzyme
using Test

Enzyme.API.printall!(true)

@testset "ABI & Calling convention" begin
    struct LList
        next::Union{LList,Nothing}
        val::Float64
    end

    function sumlist(n::LList)
        sum = 0.0
        while n !== nothing
            sum += n.val
            n = n.next
        end
        sum
    end

    regular = LList(LList(nothing, 1.0), 2.0)
    shadow  = LList(LList(nothing, 0.0), 0.0)
    ad = autodiff(Reverse, sumlist, Active, Duplicated(regular, shadow))
    @test ad === ((nothing,),)
    @test shadow.val ≈ 1.0 && shadow.next.val ≈ 1.0

    @test 2.0 ≈ first(autodiff(Forward, sumlist, DuplicatedNoNeed, Duplicated(regular, shadow)))

    mulr(x, y) = x[] * y[]
    x = Ref(2.0)
    y = Ref(3.0)
    dx = Ref(0.0)
    dy = Ref(0.0)
    n = autodiff(Reverse, mulr, Active, Duplicated(x, dx), Duplicated(y, dy))
    @test n === ((nothing,nothing),)
    @test dx[] ≈ 3.0
    @test dy[] ≈ 2.0

    x = Ref(2.0)
    y = Ref(3.0)
    dx = Ref(5.0)
    dy = Ref(7.0)
    @test 5.0*3.0 + 2.0*7.0≈ first(autodiff(Forward, mulr, DuplicatedNoNeed, Duplicated(x, dx), Duplicated(y, dy)))

    _, mid = Enzyme.autodiff(Reverse, (fs, x) -> fs[1](x), Active, (x->x*x,), Active(2.0))[1]
    @test mid ≈ 4.0

    _, mid = Enzyme.autodiff(Reverse, (fs, x) -> fs[1](x), Active, [x->x*x], Active(2.0))[1]
    @test mid ≈ 4.0

    mid, = Enzyme.autodiff(Forward, (fs, x) -> fs[1](x), DuplicatedNoNeed, (x->x*x,), Duplicated(2.0, 1.0))
    @test mid ≈ 4.0

    mid, = Enzyme.autodiff(Forward, (fs, x) -> fs[1](x), DuplicatedNoNeed, [x->x*x], Duplicated(2.0, 1.0))
    @test mid ≈ 4.0


    # deserves_argbox yes and no
    struct Bar
        r::Ref{Int}
    end

    # ConstType

    # primitive type Int128, Float64, Float128

    # returns: sret, const/ghost, !deserve_retbox
end

@testset "Mutable Struct ABI" begin
    mutable struct MStruct
        val::Float32
    end

    function sqMStruct(domain::Vector{MStruct}, x::Float32)
       @inbounds domain[1] = MStruct(x*x)
       return nothing
    end

    orig   = [MStruct(0.0)]
    shadow = [MStruct(17.0)]
    Enzyme.autodiff(Forward, sqMStruct, Duplicated(orig, shadow), Duplicated(Float32(3.14), Float32(1.0)))
     @test 2.0*3.14 ≈ shadow[1].val 

end

@testset "Thread helpers" begin
    function timesID(x)
        x * Threads.threadid()
    end
    function timesNum(x)
        x * Threads.nthreads()
    end
    @test Threads.threadid() ≈ Enzyme.autodiff(Reverse, timesID, Active(2.0))[1][1]
    @test Threads.nthreads() ≈ Enzyme.autodiff(Reverse, timesNum, Active(2.0))[1][1]
end

@testset "Closure ABI" begin
    function clo2(x)
        V = [x]
        y -> V[1] * y
    end
    f = clo2(2.0)
    @test 2.0 ≈ Enzyme.autodiff(Reverse, f, Active(3.0))[1][1][1]

    @test 2.0 ≈ Enzyme.autodiff(Forward, f, Duplicated(3.0, 1.0))[1]
    
    df = clo2(0.0)
    @test 2.0 ≈ Enzyme.autodiff(Reverse, Duplicated(f, df), Active(3.0))[1][1]
    @test 3.0 ≈ df.V[1] 

    @test 2.0 * 7.0 + 3.0 * 5.0 ≈ first(Enzyme.autodiff(Forward, Duplicated(f, df), Duplicated(5.0, 7.0)))
end

@testset "Union return" begin
    @noinline function fwdunion(itr, cond)
        if cond
            return Base._InitialValue()
        else
            return itr[1]
        end
    end

    forward, pullback0 = Enzyme.autodiff_thunk(ReverseSplitModified(ReverseSplitWithPrimal, Val((true, true, false))), Const{typeof(fwdunion)}, Duplicated, Duplicated{Vector{Float64}}, Const{Bool})
    tape, primal, shadow = forward(Const(fwdunion), Duplicated(Float64[2.0], Float64[0.0]), Const(false))
    @test primal ≈ 2.0 
    @test shadow ≈ 0.0 
    
    forward, pullback1 = Enzyme.autodiff_thunk(ReverseSplitModified(ReverseSplitWithPrimal, Val((true, true, false))), Const{typeof(fwdunion)}, Duplicated, Duplicated{Vector{Float64}}, Const{Bool})
    tape, primal, shadow = forward(Const(fwdunion), Duplicated(Float64[2.0], Float64[0.0]), Const(true))
    @test primal == Base._InitialValue() 
    @test shadow == Base._InitialValue()
    @test pullback0 == pullback1
    
    forward, pullback2 = Enzyme.autodiff_thunk(ReverseSplitModified(ReverseSplitNoPrimal, Val((true, true, false))), Const{typeof(fwdunion)}, Duplicated, Duplicated{Vector{Float64}}, Const{Bool})
    tape, primal, shadow = forward(Const(fwdunion), Duplicated(Float64[2.0], Float64[0.0]), Const(false))
    @test primal == nothing
    @test shadow ≈ 0.0 
    @test pullback0 != pullback2
    
    forward, pullback3 = Enzyme.autodiff_thunk(ReverseSplitModified(ReverseSplitNoPrimal, Val((true, true, false))), Const{typeof(fwdunion)}, Duplicated, Duplicated{Vector{Float64}}, Const{Bool})
    tape, primal, shadow = forward(Const(fwdunion), Duplicated(Float64[2.0], Float64[0.0]), Const(true))
    @test primal == nothing
    @test shadow == Base._InitialValue()    
    @test pullback2 == pullback3
end

@testset "Callable ABI" begin
    function method(f, x)
        return f(x)
    end

    struct AFoo
       x::Float64
    end

    function (f::AFoo)(x::Float64)
       return f.x * x
    end

    @test Enzyme.autodiff(Reverse, method, Active, AFoo(2.0), Active(3.0))[1][2] ≈ 2.0
    @test Enzyme.autodiff(Reverse, AFoo(2.0), Active, Active(3.0))[1][1] ≈ 2.0

    @test Enzyme.autodiff(Forward, method, DuplicatedNoNeed, AFoo(2.0), Duplicated(3.0, 1.0))[1] ≈ 2.0
    @test Enzyme.autodiff(Forward, AFoo(2.0), DuplicatedNoNeed, Duplicated(3.0, 1.0))[1] ≈ 2.0

    struct ABar
    end

    function (f::ABar)(x::Float64)
       return 2.0 * x
    end

    @test Enzyme.autodiff(Reverse, method, Active, ABar(), Active(3.0))[1][2] ≈ 2.0
    @test Enzyme.autodiff(Reverse, ABar(), Active, Active(3.0))[1][1] ≈ 2.0

    @test Enzyme.autodiff(Forward, method, DuplicatedNoNeed, ABar(), Duplicated(3.0, 1.0))[1] ≈ 2.0
    @test Enzyme.autodiff(Forward, ABar(), DuplicatedNoNeed, Duplicated(3.0, 1.0))[1] ≈ 2.0
end

@testset "Promotion" begin
    x = [1.0, 2.0]; dx_1 = [1.0, 0.0]; dx_2 = [0.0, 1.0];
    rosenbrock_inp(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    r = autodiff(Forward, rosenbrock_inp, Duplicated, BatchDuplicated(x, (dx_1, dx_2)))
    @test r[1] ≈ 100.0
    @test r[2][1] ≈ -400.0
    @test r[2][2] ≈ 200.0
    r = autodiff_deferred(Forward, rosenbrock_inp, Duplicated, BatchDuplicated(x, (dx_1, dx_2)))
    @test r[1] ≈ 100.0
    @test r[2][1] ≈ -400.0
    @test r[2][2] ≈ 200.0
end
