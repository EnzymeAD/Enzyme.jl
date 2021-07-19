using Enzyme
using Test

@testset "ABI & Calling convention" begin

    f(x) = x

    # GhostType -> Nothing
    res = autodiff(f, Const, Const(nothing))
    @test res === ()

    res = autodiff(f, Const(nothing))
    @test res === ()

    res = Enzyme.autodiff_deferred(f, Const(nothing))
    @test res === ()

    # ConstType -> Type{Int}
    res = autodiff(f, Const, Const(Int))
    @test res === ()

    res = autodiff(f, Const(Int))
    @test res === ()

    res = Enzyme.autodiff_deferred(f, Const(Int))
    @test res === ()

    # Complex numbers
    cres,  = Enzyme.autodiff(f, Active, Active(1.5 + 0.7im))
    @test cres ≈ 1.0 + 0.0im

    cres,  = Enzyme.autodiff(f, Active(1.5 + 0.7im))
    @test cres ≈ 1.0 + 0.0im

    cres, = Enzyme.autodiff_deferred(f, Active(1.5 + 0.7im))
    @test cres ≈ 1.0 + 0.0im

    # Unused singleton argument
    unused(_, y) = y
    res0, = autodiff(unused, Active, Const(nothing), Active(2.0))
    @test res0 ≈ 1.0

    res0, = autodiff(unused, Const(nothing), Active(2.0))
    @test res0 ≈ 1.0

    # returning an Array, with Const
    function squareRetArray(x)
        x[1] *= 2
        x
    end
    x = [0.0]
    dx = [1.2]
    autodiff(squareRetArray, Const, Duplicated(x, dx))
    @test dx[1] ≈ 2.4

    x = [0.0]
    dx = [1.2]
    autodiff_deferred(squareRetArray, Const, Duplicated(x, dx))
    @test dx[1] ≈ 2.4

    # Multi arg => sret
    mul(x, y) = x * y
    pair = autodiff(mul, Active, Active(2.0), Active(3.0))
    @test pair[1] ≈ 3.0
    @test pair[2] ≈ 2.0

    pair = autodiff(mul, Active(2.0), Active(3.0))
    @test pair[1] ≈ 3.0
    @test pair[2] ≈ 2.0

    pair = autodiff_deferred(mul, Active(2.0), Active(3.0))
    @test pair[1] ≈ 3.0
    @test pair[2] ≈ 2.0

    # SequentialType
    struct Foo
        baz::Int
        qux::Float64
    end

    g(x) = x.qux
    res2,  = autodiff(g, Active, Active(Foo(3, 1.2)))
    @test res2.qux ≈ 1.0

    res2,  = autodiff(g, Active(Foo(3, 1.2)))
    @test res2.qux ≈ 1.0

    unused2(_, y) = y.qux
    resF, = autodiff(unused2, Active, Const(nothing), Active(Foo(3, 2.0)))
    @test resF.qux ≈ 1.0

    resF, = autodiff(unused2, Const(nothing), Active(Foo(3, 2.0)))
    @test resF.qux ≈ 1.0

    h(x, y) = x.qux * y.qux
    res3 = autodiff(h, Active, Active(Foo(3, 1.2)), Active(Foo(5, 3.4)))
    @test res3[1].qux ≈ 3.4
    @test res3[2].qux ≈ 1.2

    res3 = autodiff(h, Active(Foo(3, 1.2)), Active(Foo(5, 3.4)))
    @test res3[1].qux ≈ 3.4
    @test res3[2].qux ≈ 1.2

    caller(f, x) = f(x)
    res4, = autodiff(caller, Active, (x)->x, Active(3.0))
    @test res4 ≈ 1.0

    res4, = autodiff(caller, (x)->x, Active(3.0))
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
    ad = autodiff(sumlist, Active, Duplicated(regular, shadow))
    @test ad === ()
    @test shadow.val ≈ 1.0 && shadow.next.val ≈ 1.0

    mulr(x, y) = x[] * y[]
    x = Ref(2.0)
    y = Ref(3.0)
    dx = Ref(0.0)
    dy = Ref(0.0)
    n = autodiff(mulr, Active, Duplicated(x, dx), Duplicated(y, dy))
    @test n === ()
    @test dx[] ≈ 3.0
    @test dy[] ≈ 2.0

    mid, = Enzyme.autodiff((fs, x) -> fs[1](x), Active, (x->x*x,), Active(2.0))
    @test mid ≈ 4.0

    mid, = Enzyme.autodiff((fs, x) -> fs[1](x), Active, [x->x*x], Active(2.0))
    @test mid ≈ 4.0

    # deserves_argbox yes and no
    struct Bar
        r::Ref{Int}
    end

    # ConstType

    # primitive type Int128, Float64, Float128

    # returns: sret, const/ghost, !deserve_retbox
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

    @test Enzyme.autodiff(method, Active, AFoo(2.0), Active(3.0))[1]≈ 2.0
    @test Enzyme.autodiff(AFoo(2.0), Active, Active(3.0))[1]≈ 2.0

    struct ABar
    end

    function (f::ABar)(x::Float64)
       return 2.0 * x
    end

    @test Enzyme.autodiff(method, Active, ABar(), Active(3.0))[1]≈ 2.0
    @test Enzyme.autodiff(ABar(), Active, Active(3.0))[1]≈ 2.0
end
