using Enzyme
using Test

@testset "ABI & Calling convention" begin

    f(x) = x

    # GhostType -> Nothing
    res = autodiff(Reverse, f, Const, Const(nothing))
    @test res === ()
    
    @test () === autodiff(Forward, f, Const, Const(nothing))

    res = autodiff(Reverse, f, Const(nothing))
    @test res === ()
    
    @test () === autodiff(Forward, f, Const(nothing))

    res = Enzyme.autodiff_deferred(f, Const(nothing))
    @test res === ()
    @test () === Enzyme.fwddiff_deferred(f, Const(nothing))

    # ConstType -> Type{Int}
    res = autodiff(Reverse, f, Const, Const(Int))
    @test res === ()
    @test () === autodiff(Forward, f, Const, Const(Int))

    res = autodiff(Reverse, f, Const(Int))
    @test res === ()
    @test () === autodiff(Forward, f, Const(Int))

    res = Enzyme.autodiff_deferred(f, Const(Int))
    @test res === ()
    @test () === Enzyme.fwddiff_deferred(f, Const(Int))

    # Complex numbers
    cres,  = Enzyme.autodiff(Reverse, f, Active, Active(1.5 + 0.7im))
    @test cres ≈ 1.0 + 0.0im
    cres,  = Enzyme.autodiff(Forward, f, DuplicatedNoNeed, Duplicated(1.5 + 0.7im, 1.0 + 0im))
    @test cres ≈ 1.0 + 0.0im

    cres,  = Enzyme.autodiff(Reverse, f, Active(1.5 + 0.7im))
    @test cres ≈ 1.0 + 0.0im
    cres,  = Enzyme.autodiff(Forward, f, Duplicated(1.5 + 0.7im, 1.0+0im))
    @test cres ≈ 1.0 + 0.0im

    cres, = Enzyme.autodiff_deferred(f, Active(1.5 + 0.7im))
    @test cres ≈ 1.0 + 0.0im
    cres,  = Enzyme.fwddiff_deferred(f, Duplicated(1.5 + 0.7im, 1.0+0im))
    @test cres ≈ 1.0 + 0.0im

    # Unused singleton argument
    unused(_, y) = y
    res0, = autodiff(Reverse, unused, Active, Const(nothing), Active(2.0))
    @test res0 ≈ 1.0
    res0, = autodiff(Forward, unused, DuplicatedNoNeed, Const(nothing), Duplicated(2.0, 1.0))
    @test res0 ≈ 1.0
    res0, = autodiff(Forward, unused, DuplicatedNoNeed, Const(nothing), DuplicatedNoNeed(2.0, 1.0))
    @test res0 ≈ 1.0

    res0, = autodiff(Reverse, unused, Const(nothing), Active(2.0))
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
    Enzyme.autodiff_deferred(squareRetArray, Const, Duplicated(x, dx))

    dx = [1.2]
    @test () === autodiff(Forward, squareRetArray, Const, Duplicated(x, dx))
    @test dx[1] ≈ 2.4

    # Multi arg => sret
    mul(x, y) = x * y
    pair = autodiff(Reverse, mul, Active, Active(2.0), Active(3.0))
    @test pair[1] ≈ 3.0
    @test pair[2] ≈ 2.0

    pair = autodiff(Reverse, mul, Active(2.0), Active(3.0))
    @test pair[1] ≈ 3.0
    @test pair[2] ≈ 2.0

    pair = Enzyme.autodiff_deferred(mul, Active(2.0), Active(3.0))
    @test pair[1] ≈ 3.0
    @test pair[2] ≈ 2.0

    # Multi output
    # TODO broken arg convention?
    # tup(x) = (x, x*2)
    # pair = first(autodiff(Forward, tup, DuplicatedNoNeed, Duplicated(3.14, 1.0)))
    # @test pair[1] ≈ 1.0
    # @test pair[2] ≈ 2.0
    # pair = first(autodiff(Forward, tup, Duplicated(3.14, 1.0)))
    # @test pair[1] ≈ 1.0
    # @test pair[2] ≈ 2.0
    # pair = first(Enzyme.fwddiff_deferred(tup, Duplicated(3.14, 1.0)))
    # @test pair[1] ≈ 1.0
    # @test pair[2] ≈ 2.0


    # SequentialType
    struct Foo
        baz::Int
        qux::Float64
    end

    g(x) = x.qux
    res2,  = autodiff(Reverse, g, Active, Active(Foo(3, 1.2)))
    @test res2.qux ≈ 1.0

    @test 1.0≈ first(autodiff(Forward, g, DuplicatedNoNeed, Duplicated(Foo(3, 1.2), Foo(0, 1.0))))

    res2,  = autodiff(Reverse, g, Active(Foo(3, 1.2)))
    @test res2.qux ≈ 1.0

    @test 1.0≈ first(autodiff(Forward, g, Duplicated(Foo(3, 1.2), Foo(0, 1.0))))

    unused2(_, y) = y.qux
    resF, = autodiff(Reverse, unused2, Active, Const(nothing), Active(Foo(3, 2.0)))
    @test resF.qux ≈ 1.0

    @test 1.0≈ first(autodiff(Forward, unused2, DuplicatedNoNeed, Const(nothing), Duplicated(Foo(3, 1.2), Foo(0, 1.0))))

    resF, = autodiff(Reverse, unused2, Const(nothing), Active(Foo(3, 2.0)))
    @test resF.qux ≈ 1.0

    @test 1.0≈ first(autodiff(Forward, unused2, Const(nothing), Duplicated(Foo(3, 1.2), Foo(0, 1.0))))

    h(x, y) = x.qux * y.qux
    res3 = autodiff(Reverse, h, Active, Active(Foo(3, 1.2)), Active(Foo(5, 3.4)))
    @test res3[1].qux ≈ 3.4
    @test res3[2].qux ≈ 1.2

    @test 7*3.4 + 9 * 1.2 ≈ first(autodiff(Forward, h, DuplicatedNoNeed, Duplicated(Foo(3, 1.2), Foo(0, 7.0)), Duplicated(Foo(5, 3.4), Foo(0, 9.0))))

    res3 = autodiff(Reverse, h, Active(Foo(3, 1.2)), Active(Foo(5, 3.4)))
    @test res3[1].qux ≈ 3.4
    @test res3[2].qux ≈ 1.2

    @test 7*3.4 + 9 * 1.2 ≈ first(autodiff(Forward, h, Duplicated(Foo(3, 1.2), Foo(0, 7.0)), Duplicated(Foo(5, 3.4), Foo(0, 9.0))))

    caller(f, x) = f(x)
    res4, = autodiff(Reverse, caller, Active, (x)->x, Active(3.0))
    @test res4 ≈ 1.0

    res4, = autodiff(Forward, caller, DuplicatedNoNeed, (x)->x, Duplicated(3.0, 1.0))
    @test res4 ≈ 1.0

    res4, = autodiff(Reverse, caller, (x)->x, Active(3.0))
    @test res4 ≈ 1.0

    res4, = autodiff(Forward, caller, (x)->x, Duplicated(3.0, 1.0))
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
    @test ad === ()
    @test shadow.val ≈ 1.0 && shadow.next.val ≈ 1.0

    @test 2.0 ≈ first(autodiff(Forward, sumlist, DuplicatedNoNeed, Duplicated(regular, shadow)))

    mulr(x, y) = x[] * y[]
    x = Ref(2.0)
    y = Ref(3.0)
    dx = Ref(0.0)
    dy = Ref(0.0)
    n = autodiff(Reverse, mulr, Active, Duplicated(x, dx), Duplicated(y, dy))
    @test n === ()
    @test dx[] ≈ 3.0
    @test dy[] ≈ 2.0

    x = Ref(2.0)
    y = Ref(3.0)
    dx = Ref(5.0)
    dy = Ref(7.0)
    @test 5.0*3.0 + 2.0*7.0≈ first(autodiff(Forward, mulr, DuplicatedNoNeed, Duplicated(x, dx), Duplicated(y, dy)))

    mid, = Enzyme.autodiff(Reverse, (fs, x) -> fs[1](x), Active, (x->x*x,), Active(2.0))
    @test mid ≈ 4.0

    mid, = Enzyme.autodiff(Reverse, (fs, x) -> fs[1](x), Active, [x->x*x], Active(2.0))
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
    @test Threads.threadid() ≈ Enzyme.autodiff(Reverse, timesID, Active(2.0))[1]
    @test Threads.nthreads() ≈ Enzyme.autodiff(Reverse, timesNum, Active(2.0))[1]
end

@testset "Closure ABI" begin
    function clo2(x)
        V = [x]
        y -> V[1] * y
    end
    f = clo2(2.0)
    @test 2.0 ≈ Enzyme.autodiff(Reverse, f, Active(3.0))[1]

    @test 2.0 ≈ Enzyme.autodiff(Forward, f, Duplicated(3.0, 1.0))[1]
    
    df = clo2(0.0)
    @test 2.0 ≈ Enzyme.autodiff(Reverse, Duplicated(f, df), Active(3.0))[1]
    @test 3.0 ≈ df.V[1] 

    @test 2.0 * 7.0 + 3.0 * 5.0 ≈ first(Enzyme.autodiff(Forward, Duplicated(f, df), Duplicated(5.0, 7.0)))
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

    @test Enzyme.autodiff(Reverse, method, Active, AFoo(2.0), Active(3.0))[1]≈ 2.0
    @test Enzyme.autodiff(Reverse, AFoo(2.0), Active, Active(3.0))[1]≈ 2.0

    @test Enzyme.autodiff(Forward, method, DuplicatedNoNeed, AFoo(2.0), Duplicated(3.0, 1.0))[1]≈ 2.0
    @test Enzyme.autodiff(Forward, AFoo(2.0), DuplicatedNoNeed, Duplicated(3.0, 1.0))[1]≈ 2.0

    struct ABar
    end

    function (f::ABar)(x::Float64)
       return 2.0 * x
    end

    @test Enzyme.autodiff(Reverse, method, Active, ABar(), Active(3.0))[1]≈ 2.0
    @test Enzyme.autodiff(Reverse, ABar(), Active, Active(3.0))[1]≈ 2.0

    @test Enzyme.autodiff(Forward, method, DuplicatedNoNeed, ABar(), Duplicated(3.0, 1.0))[1]≈ 2.0
    @test Enzyme.autodiff(Forward, ABar(), DuplicatedNoNeed, Duplicated(3.0, 1.0))[1]≈ 2.0
end
