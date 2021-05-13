using Enzyme
using Test

@testset "ABI & Calling convention" begin

    f(x) = x 

    # GhostType -> Nothing
    res = autodiff(f, Const(nothing))
    @test res === nothing

    # ConstType -> Type{Int}
    res = autodiff(f, Const(Int))
    @test res === nothing


    cres,  = Enzyme.autodiff(f, Active(1.5 + 0.7im))
    @test cres ≈ 1.0 + 0.0im

    unused(_, y) = y
    res0, = autodiff(unused, Const(nothing), Active(2.0))
    @test res0 ≈ 1.0

    # Multi arg => sret
    mul(x, y) = x * y
    pair = autodiff(mul, Active(2.0), Active(3.0))
    @test pair[1] ≈ 3.0
    @test pair[2] ≈ 2.0

    # SeqeuntialType
    struct Foo
        baz::Int
        qux::Float64
    end

    g(x) = x.qux
    res2,  = autodiff(g, Active(Foo(3, 1.2)))
    @test res2.qux ≈ 1.0


    unused2(_, y) = y.qux
    resF, = autodiff(unused2, Const(nothing), Active(Foo(3, 2.0)))
    @test resF.qux ≈ 1.0

    h(x, y) = x.qux * y.qux
    res3 = autodiff(h, Active(Foo(3, 1.2)), Active(Foo(5, 3.4)))
    @test res3[1].qux ≈ 3.4
    @test res3[2].qux ≈ 1.2

    caller(f, x) = f(x)
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
    ad = autodiff(sumlist, Duplicated(regular, shadow))
    @test ad === nothing
    @test shadow.val ≈ 1.0 && shadow.next.val ≈ 1.0


    mulr(x, y) = x[] * y[]
    x = Ref(2.0)
    y = Ref(3.0)
    dx = Ref(0.0)
    dy = Ref(0.0)
    n = autodiff(mulr, Duplicated(x, dx), Duplicated(y, dy))
    @test n === nothing
    @test dx[] ≈ 3.0
    @test dy[] ≈ 2.0

    # deserves_argbox yes and no
    struct Bar
        r::Ref{Int}
    end

    # ConstType

    # primitive type Int128, Float64, Float128

    # returns: sret, const/ghost, !deserve_retbox
end
