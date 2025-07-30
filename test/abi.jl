using Enzyme
using Test

retty() = Float64

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

end
