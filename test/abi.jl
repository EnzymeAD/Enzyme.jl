using Enzyme
using Test

retty() = Float64

@testset "Const Return" begin
    res = Enzyme.autodiff(ForwardWithPrimal, retty, Const)
    @test res === NamedTuple{(Symbol("1"),), Tuple{Type{Float64}}}((Float64,))
    res = Enzyme.autodiff(Forward, retty, Const)
    @test res === ()
end

@testset "ABI & Calling convention" begin

    f(x) = x

    # GhostType -> Nothing
    res = autodiff(Reverse, f, Const, Const(nothing))
    @test res === ((nothing,),)
    
    res = autodiff(Enzyme.set_abi(Reverse, NonGenABI), f, Const, Const(nothing))
    @test res === ((nothing,),)
    
    @test () === autodiff(Forward, f, Const, Const(nothing))
    @test () === autodiff(Enzyme.set_abi(Forward, NonGenABI), f, Const, Const(nothing))

    res = autodiff(Reverse, f, Const(nothing))
    @test res === ((nothing,),)
    
    @test () === autodiff(Forward, f, Const(nothing))

    res = autodiff_deferred(Reverse, Const(f), Const, Const(nothing))
    @test res === ((nothing,),)
    res = autodiff_deferred(Enzyme.set_abi(Reverse, NonGenABI), Const(f), Const, Const(nothing))
    @test res === ((nothing,),)
    
    @test () === autodiff_deferred(Forward, Const(f), Const, Const(nothing))
    @test () === autodiff_deferred(Enzyme.set_abi(Forward, NonGenABI), Const(f), Const, Const(nothing))

    # ConstType -> Type{Int}
    res = autodiff(Reverse, f, Const, Const(Int))
    @test res === ((nothing,),)
    @test () === autodiff(Forward, f, Const, Const(Int))

    res = autodiff(Reverse, f, Const(Int))
    @test res === ((nothing,),)
    @test () === autodiff(Forward, f, Const(Int))

    res = autodiff_deferred(Reverse, Const(f), Const, Const(Int))
    @test res === ((nothing,),)
    @test () === autodiff_deferred(Forward, Const(f), Const, Const(Int))

    # Complex numbers
    @test_throws ErrorException autodiff(Reverse, f, Active, Active(1.5 + 0.7im))
    cres,  = autodiff(ReverseHolomorphic, f, Active, Active(1.5 + 0.7im))[1]
    @test cres ≈ 1.0 + 0.0im
    cres,  = autodiff(Forward, f, Duplicated, Duplicated(1.5 + 0.7im, 1.0 + 0im))
    @test cres ≈ 1.0 + 0.0im

    @test_throws ErrorException autodiff(Reverse, f, Active(1.5 + 0.7im))
    cres,  = autodiff(ReverseHolomorphic, f, Active(1.5 + 0.7im))[1]
    @test cres ≈ 1.0 + 0.0im
    cres,  = autodiff(Forward, f, Duplicated(1.5 + 0.7im, 1.0+0im))
    @test cres ≈ 1.0 + 0.0im

    @test_throws ErrorException autodiff_deferred(Reverse, Const(f), Active, Active(1.5 + 0.7im))
    @test_throws ErrorException autodiff_deferred(ReverseHolomorphic, Const(f), Active, Active(1.5 + 0.7im))

    cres,  = autodiff_deferred(Forward, Const(f), Duplicated, Duplicated(1.5 + 0.7im, 1.0+0im))
    @test cres ≈ 1.0 + 0.0im

    # Unused singleton argument
    unused(_, y) = y
    _, res0 = autodiff(Reverse, unused, Active, Const(nothing), Active(2.0))[1]
    @test res0 ≈ 1.0
    
    _, res0 = autodiff(Enzyme.set_abi(Reverse, NonGenABI), unused, Active, Const(nothing), Active(2.0))[1]
    @test res0 ≈ 1.0
    
    res0, = autodiff(Forward, unused, Duplicated, Const(nothing), Duplicated(2.0, 1.0))
    @test res0 ≈ 1.0
    res0, = autodiff(Forward, unused, Duplicated, Const(nothing), DuplicatedNoNeed(2.0, 1.0))
    @test res0 ≈ 1.0
    
    res0, = autodiff(Enzyme.set_abi(Forward, NonGenABI), unused, Duplicated, Const(nothing), Duplicated(2.0, 1.0))
    @test res0 ≈ 1.0

    _, res0 = autodiff(Reverse, unused, Const(nothing), Active(2.0))[1]
    @test res0 ≈ 1.0
    res0, = autodiff(Forward, unused, Const(nothing), Duplicated(2.9, 1.0))
    @test res0 ≈ 1.0

    # returning an Array, with Const
    function squareRetArray(x)
        x[1] *= 2
        x
    end
    x = [0.0]
    dx = [1.2]
    autodiff(Reverse, squareRetArray, Const, Duplicated(x, dx))
    @test dx[1] ≈ 2.4

    dx = [1.2]
    @test () === autodiff(Forward, squareRetArray, Const, Duplicated(x, dx))
    @test dx[1] ≈ 2.4

    x = [0.0]
    dx = [1.2]
    autodiff_deferred(Reverse, Const(squareRetArray), Const, Duplicated(x, dx))

    dx = [1.2]
    @test () === autodiff(Forward, squareRetArray, Const, Duplicated(x, dx))
    @test dx[1] ≈ 2.4

    # Multi arg => sret
    mul(x, y) = x * y
    pair = autodiff(Reverse, mul, Active, Active(2.0), Active(3.0))[1]
    @test pair[1] ≈ 3.0
    @test pair[2] ≈ 2.0

    pair = autodiff(Reverse, mul, Active(2.0), Active(3.0))[1]
    @test pair[1] ≈ 3.0
    @test pair[2] ≈ 2.0

    pair = autodiff_deferred(Reverse, Const(mul), Active, Active(2.0), Active(3.0))[1]
    @test pair[1] ≈ 3.0
    @test pair[2] ≈ 2.0
    
    pair, orig = autodiff(ReverseWithPrimal, mul, Active(2.0), Active(3.0))
    @test pair[1] ≈ 3.0
    @test pair[2] ≈ 2.0
    @test orig ≈ 6.0
    
    pair, orig = autodiff_deferred(ReverseWithPrimal, Const(mul), Active, Active(2.0), Active(3.0))
    @test pair[1] ≈ 3.0
    @test pair[2] ≈ 2.0
    @test orig ≈ 6.0

    function inplace(x)
        x[] *= 2
        return Float64
    end

    res = Ref(3.0)
    dres = Ref(1.0)
    pair, orig = autodiff(ReverseWithPrimal, inplace, Const, Duplicated(res, dres))
    @test pair == (nothing,)
    @test res[] ≈ 6.0
    @test dres[] ≈ 2.0
    @test orig == Float64
    
    res = Ref(3.0)
    dres = Ref(1.0)
    pair, orig = autodiff_deferred(ReverseWithPrimal, Const(inplace), Const, Duplicated(res, dres))
    @test pair == (nothing,)
    @test res[] ≈ 6.0
    @test dres[] ≈ 2.0
    @test orig == Float64
    
    function inplace2(x)
        x[] *= 2
        return nothing
    end

    res = Ref(3.0)
    dres = Ref(1.0)
    pair, orig = autodiff(ReverseWithPrimal, inplace2, Const, Duplicated(res, dres))
    @test pair == (nothing,)
    @test res[] ≈ 6.0
    @test dres[] ≈ 2.0
    @test orig == nothing

    res = Ref(3.0)
    dres = Ref(1.0)
    pair, orig = autodiff_deferred(ReverseWithPrimal, Const(inplace2), Const, Duplicated(res, dres))
    @test pair == (nothing,)
    @test res[] ≈ 6.0
    @test dres[] ≈ 2.0
    @test orig == nothing

    # Multi output
    # TODO broken arg convention?
    # tup(x) = (x, x*2)
    # pair = first(autodiff(Forward, tup, DuplicatedNoNeed, Duplicated(3.14, 1.0)))
    # @test pair[1] ≈ 1.0
    # @test pair[2] ≈ 2.0
    # pair = first(autodiff(Forward, tup, Duplicated(3.14, 1.0)))
    # @test pair[1] ≈ 1.0
    # @test pair[2] ≈ 2.0
    # pair = first(autodiff_deferred(Forward, tup, Duplicated(3.14, 1.0)))
    # @test pair[1] ≈ 1.0
    # @test pair[2] ≈ 2.0


    # SequentialType
    struct Foo
        baz::Int
        qux::Float64
    end

    g(x) = x.qux
    res2,  = autodiff(Reverse, g, Active, Active(Foo(3, 1.2)))[1]
    @test res2.qux ≈ 1.0

    @test 1.0≈ first(autodiff(Forward, g, Duplicated, Duplicated(Foo(3, 1.2), Foo(0, 1.0))))

    res2,  = autodiff(Reverse, g, Active(Foo(3, 1.2)))[1]
    @test res2.qux ≈ 1.0

    @test 1.0≈ first(autodiff(Forward, g, Duplicated(Foo(3, 1.2), Foo(0, 1.0))))

    unused2(_, y) = y.qux
    _, resF = autodiff(Reverse, unused2, Active, Const(nothing), Active(Foo(3, 2.0)))[1]
    @test resF.qux ≈ 1.0

    @test 1.0≈ first(autodiff(Forward, unused2, Duplicated, Const(nothing), Duplicated(Foo(3, 1.2), Foo(0, 1.0))))

    _, resF = autodiff(Reverse, unused2, Const(nothing), Active(Foo(3, 2.0)))[1]
    @test resF.qux ≈ 1.0

    @test 1.0≈ first(autodiff(Forward, unused2, Const(nothing), Duplicated(Foo(3, 1.2), Foo(0, 1.0))))

    h(x, y) = x.qux * y.qux
    res3 = autodiff(Reverse, h, Active, Active(Foo(3, 1.2)), Active(Foo(5, 3.4)))[1]
    @test res3[1].qux ≈ 3.4
    @test res3[2].qux ≈ 1.2

    @test 7*3.4 + 9 * 1.2 ≈ first(autodiff(Forward, h, Duplicated, Duplicated(Foo(3, 1.2), Foo(0, 7.0)), Duplicated(Foo(5, 3.4), Foo(0, 9.0))))

    res3 = autodiff(Reverse, h, Active(Foo(3, 1.2)), Active(Foo(5, 3.4)))[1]
    @test res3[1].qux ≈ 3.4
    @test res3[2].qux ≈ 1.2

    @test 7*3.4 + 9 * 1.2 ≈ first(autodiff(Forward, h, Duplicated(Foo(3, 1.2), Foo(0, 7.0)), Duplicated(Foo(5, 3.4), Foo(0, 9.0))))

    caller(f, x) = f(x)
    _, res4 = autodiff(Reverse, caller, Active, Const((x)->x), Active(3.0))[1]
    @test res4 ≈ 1.0

    res4, = autodiff(Forward, caller, Duplicated, Const((x)->x), Duplicated(3.0, 1.0))
    @test res4 ≈ 1.0

    _, res4 = autodiff(Reverse, caller, Const((x)->x), Active(3.0))[1]
    @test res4 ≈ 1.0

    res4, = autodiff(Forward, caller, Const((x)->x), Duplicated(3.0, 1.0))
    @test res4 ≈ 1.0

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

    @test 2.0 ≈ first(autodiff(Forward, sumlist, Duplicated, Duplicated(regular, shadow)))

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
    @test 5.0*3.0 + 2.0*7.0≈ first(autodiff(Forward, mulr, Duplicated, Duplicated(x, dx), Duplicated(y, dy)))

    _, mid = Enzyme.autodiff(Reverse, (fs, x) -> fs[1](x), Active, Const((x->x*x,)), Active(2.0))[1]
    @test mid ≈ 4.0

    _, mid = Enzyme.autodiff(Reverse, (fs, x) -> fs[1](x), Active, Const([x->x*x]), Active(2.0))[1]
    @test mid ≈ 4.0

    mid, = Enzyme.autodiff(Forward, (fs, x) -> fs[1](x), Duplicated, Const((x->x*x,)), Duplicated(2.0, 1.0))
    @test mid ≈ 4.0

    mid, = Enzyme.autodiff(Forward, (fs, x) -> fs[1](x), Duplicated, Const([x->x*x]), Duplicated(2.0, 1.0))
    @test mid ≈ 4.0


    # deserves_argbox yes and no
    struct Bar
        r::Ref{Int}
    end

    # ConstType

    # primitive type Int128, Float64, Float128

    # returns: sret, const/ghost, !deserve_retbox
end

unstable_load(x) = Base.inferencebarrier(x)[1]

@testset "Any Return" begin
    x = [2.7]
    dx = [0.0]
    Enzyme.autodiff(Reverse, Const(unstable_load), Active, Duplicated(x, dx))
    @test dx ≈ [1.0] 

    x = [2.7]
    dx = [0.0]
    Enzyme.autodiff_deferred(Reverse, Const(unstable_load), Active, Duplicated(x, dx))
    @test dx ≈ [1.0] 
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
    @test shadow[] ≈ 0.0 
    
    forward, pullback1 = Enzyme.autodiff_thunk(ReverseSplitModified(ReverseSplitWithPrimal, Val((true, true, false))), Const{typeof(fwdunion)}, Duplicated, Duplicated{Vector{Float64}}, Const{Bool})
    tape, primal, shadow = forward(Const(fwdunion), Duplicated(Float64[2.0], Float64[0.0]), Const(true))
    @test primal == Base._InitialValue() 
    @test shadow == Base._InitialValue()
    @test pullback0 == pullback1
    
    forward, pullback2 = Enzyme.autodiff_thunk(ReverseSplitModified(ReverseSplitNoPrimal, Val((true, true, false))), Const{typeof(fwdunion)}, Duplicated, Duplicated{Vector{Float64}}, Const{Bool})
    tape, primal, shadow = forward(Const(fwdunion), Duplicated(Float64[2.0], Float64[0.0]), Const(false))
    @test primal == nothing
    @test shadow[] ≈ 0.0 
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

    @test Enzyme.autodiff(Reverse, method, Active, Const(AFoo(2.0)), Active(3.0))[1][2] ≈ 2.0
    @test Enzyme.autodiff(Reverse, AFoo(2.0), Active, Active(3.0))[1][1] ≈ 2.0

    @test Enzyme.autodiff(Forward, method, Duplicated, Const(AFoo(2.0)), Duplicated(3.0, 1.0))[1] ≈ 2.0
    @test Enzyme.autodiff(Forward, AFoo(2.0), Duplicated, Duplicated(3.0, 1.0))[1] ≈ 2.0

    struct ABar
    end

    function (f::ABar)(x::Float64)
       return 2.0 * x
    end

    @test Enzyme.autodiff(Reverse, method, Active, Const(ABar()), Active(3.0))[1][2] ≈ 2.0
    @test Enzyme.autodiff(Reverse, ABar(), Active, Active(3.0))[1][1] ≈ 2.0

    @test Enzyme.autodiff(Forward, method, Duplicated, Const(ABar()), Duplicated(3.0, 1.0))[1] ≈ 2.0
    @test Enzyme.autodiff(Forward, ABar(), Duplicated, Duplicated(3.0, 1.0))[1] ≈ 2.0

    struct RWClos
        x::Vector{Float64}
    end

    function (c::RWClos)(y)
       c.x[1] *= y
       return y
    end

    c = RWClos([4.])

    @test_throws Enzyme.Compiler.EnzymeMutabilityException autodiff(Reverse, c, Active(3.0))

    @test autodiff(Reverse, Const(c), Active(3.0))[1][1] ≈ 1.0
    @test autodiff(Reverse, Duplicated(c, RWClos([2.7])), Active(3.0))[1][1] ≈ (1.0 + 2.7 * 4 * 3)

    struct RWClos2
        x::Vector{Float64}
    end

    function (c::RWClos2)(y)
       return y + c.x[1]
    end

    c2 = RWClos2([4.])

    @test autodiff(Reverse, c2, Active(3.0))[1][1] ≈ 1.0
    @test autodiff(Reverse, Const(c2), Active(3.0))[1][1] ≈ 1.0
    @test autodiff(Reverse, Duplicated(c2, RWClos2([2.7])), Active(3.0))[1][1] ≈ 1.0
end



@testset "Promotion" begin
    x = [1.0, 2.0]; dx_1 = [1.0, 0.0]; dx_2 = [0.0, 1.0];
    rosenbrock_inp(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    r = autodiff(ForwardWithPrimal, rosenbrock_inp, Duplicated, BatchDuplicated(x, (dx_1, dx_2)))
    @test r[2] ≈ 100.0
    @test r[1][1] ≈ -400.0
    @test r[1][2] ≈ 200.0
    r = autodiff_deferred(ForwardWithPrimal, Const(rosenbrock_inp), Duplicated, BatchDuplicated(x, (dx_1, dx_2)))
    @test r[2] ≈ 100.0
    @test r[1][1] ≈ -400.0
    @test r[1][2] ≈ 200.0
end

abssum(x) = sum(abs2, x);

mulsin(x) = sin(x[1] * x[2])

@testset "within_autodiff" begin
    @test !Enzyme.within_autodiff()
    @test_broken Enzyme.autodiff(ForwardWithPrimal, Enzyme.within_autodiff)[1]
    @test Enzyme.autodiff(ForwardWithPrimal, () -> Enzyme.within_autodiff())[1]
end

mutable struct ConstVal
    x::Float64
    const y::Float64
end

@testset "Make Zero" begin
    v = ConstVal(2.0, 3.0)
    dv = make_zero(v)
    @test dv isa ConstVal
    @test dv.x ≈ 0.0
    @test dv.y ≈ 0.0
end

@testset "Type inference" begin
    x = ones(10)
    @inferred autodiff(Enzyme.Reverse, abssum, Duplicated(x,x))
    @inferred autodiff(Enzyme.ReverseWithPrimal, abssum, Duplicated(x,x))
    @inferred autodiff(Enzyme.ReverseHolomorphic, abssum, Duplicated(x,x))
    @inferred autodiff(Enzyme.ReverseHolomorphicWithPrimal, abssum, Duplicated(x,x))
    @inferred autodiff(Enzyme.Forward, abssum, Duplicated(x,x))
    @inferred autodiff(Enzyme.ForwardWithPrimal, abssum, Duplicated, Duplicated(x,x))
    @inferred autodiff(Enzyme.Forward, abssum, Duplicated, Duplicated(x,x))
    
    @inferred gradient(Reverse, abssum, x)
    @inferred gradient!(Reverse, x, abssum, x)

    @inferred gradient(ReverseWithPrimal, abssum, x)
    @inferred gradient!(ReverseWithPrimal, x, abssum, x)
    
    cx = ones(10)
    @inferred autodiff(Enzyme.ReverseHolomorphic, sum, Duplicated(cx,cx))
    @inferred autodiff(Enzyme.ReverseHolomorphicWithPrimal, sum, Duplicated(cx,cx))
    @inferred autodiff(Enzyme.Forward, sum, Duplicated(cx,cx))
    
    @inferred Enzyme.make_zero(x)
    @inferred Enzyme.make_zero(cx)
    
    tx =  (1.0, 2.0, 3.0)

    @inferred Enzyme.Compiler.active_reg_inner(Tuple{Float64,Float64,Float64}, (), nothing, Val(true))
    @inferred Enzyme.make_zero(tx)
    
    @inferred gradient(Reverse, abssum, tx)
    @inferred gradient(Forward, abssum, tx)

    @inferred gradient(ReverseWithPrimal, abssum, tx)
    @inferred gradient(ForwardWithPrimal, abssum, tx)

    @inferred hvp(mulsin, [2.0, 3.0], [5.0, 2.7])

    @inferred hvp!(zeros(2), mulsin, [2.0, 3.0], [5.0, 2.7])

    @inferred hvp_and_gradient!(zeros(2), zeros(2), mulsin, [2.0, 3.0], [5.0, 2.7])
end

function ulogistic(x)
    return x > 36 ? one(x) : 1 / (one(x) + 1/x)
end

@noinline function u_transform_tuple(x)
    yfirst = ulogistic(@inbounds x[1])
    yfirst, 2
end


@noinline function mytransform(ts, x)
    yfirst = ulogistic(@inbounds x[1])
    yrest, _ = u_transform_tuple(x)
    (yfirst, yrest)
end

function undefsret(trf, x)
    p =  mytransform(trf, x)
    return 1/(p[2])
end

@testset "Undef sret" begin
    trf = 0.1

    x = randn(3)
    dx = zero(x)
    undefsret(trf, x)
    autodiff(Reverse, undefsret, Active, Const(trf), Duplicated(x, dx))
end

struct ByRefStruct
    x::Vector{Float64}
    v::Vector{Float64}
end

@noinline function byrefg(bref)
    return bref.x[1] .+ bref.v[1]
end
function byrefs(x, v)
    byrefg(ByRefStruct(x, v))
end

@testset "Batched byref struct" begin

    Enzyme.autodiff(Forward, byrefs, BatchDuplicated([1.0], ([1.0], [1.0])), BatchDuplicated([1.0], ([1.0], [1.0]) ) )
end

include("usermixed.jl")
