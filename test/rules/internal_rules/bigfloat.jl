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
