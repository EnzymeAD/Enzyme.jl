using Enzyme
using EnzymeTestUtils
using FiniteDifferences
using Test

@testset "BigFloat arithmetic" begin
    a = BigFloat(1.234)
    da = BigFloat(-0.23)
    b = BigFloat(0.56)
    db = BigFloat(0.27)
    af64 = 1.234 # for testing mixed methods
    daf64 = -0.23 # for testing mixed methods
    bf64 = 0.56 # for testing mixed methods
    dbf64 = 0.27 # for testing mixed methods
    
    @test autodiff(Enzyme.Forward, BigFloat, Duplicated)[:1] isa BigFloat
    @test autodiff(Enzyme.Forward, zero, Duplicated, Const(BigFloat))[:1] ≈ 0

    @test autodiff(Enzyme.Forward, +, Duplicated, Duplicated(a, da), Duplicated(b, db))[:1] ≈ da+db 
    @test autodiff(Enzyme.Forward, +, Duplicated, Duplicated(a, da), Duplicated(bf64, dbf64))[:1] ≈ da+dbf64 
    @test autodiff(Enzyme.Forward, -, Duplicated, Duplicated(a, da), Duplicated(b, db))[:1] ≈ da-db 
    @test autodiff(Enzyme.Forward, -, Duplicated, Duplicated(a, da), Duplicated(bf64, dbf64))[:1] ≈ da-dbf64 
    @test autodiff(Enzyme.Forward, *, Duplicated, Duplicated(a, da), Duplicated(b, db))[:1] ≈ b*da + a*db 
    @test autodiff(Enzyme.Forward, *, Duplicated, Duplicated(a, da), Duplicated(bf64, dbf64))[:1] ≈ bf64*da + a*dbf64
    @test autodiff(Enzyme.Forward, /, Duplicated, Duplicated(a, da), Duplicated(b, db))[:1] ≈ da/b  - db * a/b^2
    @test autodiff(Enzyme.Forward, /, Duplicated, Duplicated(a, da), Duplicated(bf64, dbf64))[:1] ≈ da/bf64 - dbf64 * a/bf64^2 

    @test autodiff(Enzyme.Forward, inv, Duplicated, Duplicated(a, da))[:1] ≈ -(one(BigFloat)/a^2) * da
    @test autodiff(Enzyme.Forward, sin, Duplicated, Duplicated(a, da))[:1] ≈ cos(a) * da
    @test autodiff(Enzyme.Forward, cos, Duplicated, Duplicated(a, da))[:1] ≈ -sin(a) * da 
    @test autodiff(Enzyme.Forward, tan, Duplicated, Duplicated(a, da))[:1] ≈ autodiff(Enzyme.Forward, tan, Duplicated, Duplicated(af64, daf64))[1]
end

@testset "BigFloat integer constructors" begin
    a = BigFloat(1.234)
    da = BigFloat(-0.23)

    # BigFloat(::Clong) / BigFloat(::Culong) construct a constant, so the shadow is zero
    @test autodiff(Enzyme.Forward, BigFloat, Duplicated, Const(2))[:1] ≈ 0
    @test autodiff(Enzyme.Forward, BigFloat, Duplicated, Const(UInt(2)))[:1] ≈ 0

    # ... but the primal value must still be right, and propagate through arithmetic
    mul2(x) = x * BigFloat(2)
    @test autodiff(Enzyme.Forward, mul2, Duplicated, Duplicated(a, da))[:1] ≈ 2 * da
    umul2(x) = x * BigFloat(UInt(2))
    @test autodiff(Enzyme.Forward, umul2, Duplicated, Duplicated(a, da))[:1] ≈ 2 * da

    # the constructor also has to keep working under a non-default precision
    setprecision(BigFloat, 512) do
        @test precision(BigFloat(3)) == 512
        mul3(x) = x * BigFloat(3)
        @test autodiff(Enzyme.Forward, mul3, Duplicated, Duplicated(a, da))[:1] ≈ 3 * da
    end

    # `eps(::Type{BigFloat})` is `nextfloat(BigFloat(1)) - BigFloat(1)`, so it goes
    # through the integer constructor; without a rule this fails to compile.
    # NOTE: using the result of `eps(BigFloat)` in an *active* position (e.g.
    # `x * eps(BigFloat)`) still segfaults -- Enzyme miscompiles the BigFloat
    # construction inside `nextfloat`/`_duplicate` when it has to differentiate it.
    epsconst(x) = x * x + zero(eps(BigFloat))
    @test autodiff(Enzyme.Forward, epsconst, Duplicated, Duplicated(a, da))[:1] ≈ 2 * a * da
end
