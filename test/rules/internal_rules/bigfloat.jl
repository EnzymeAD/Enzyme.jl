using Enzyme
using EnzymeTestUtils
using FiniteDifferences
using Test

@testset "BigFloat arithmetic" begin
    a = rand(BigFloat)
    da = rand(BigFloat)
    b = rand(BigFloat)
    db = rand(BigFloat)
    af64 = Float64(a) # for testing mixed methods
    daf64 = Float64(da) # for testing mixed methods
    bf64 = Float64(b) # for testing mixed methods
    dbf64 = Float64(db) # for testing mixed methods

    @test autodiff(Enzyme.Forward, +, Duplicated, Duplicated(a, da), Duplicated(b, db))[:1] ≈ autodiff(Enzyme.Forward, +, Duplicated, Duplicated(af64, daf64), Duplicated(bf64, dbf64))[1]
    @test autodiff(Enzyme.Forward, +, Duplicated, Duplicated(a, da), Duplicated(bf64, dbf64))[:1] ≈ autodiff(Enzyme.Forward, +, Duplicated, Duplicated(af64, daf64), Duplicated(bf64, dbf64))[1]
    @test autodiff(Enzyme.Forward, -, Duplicated, Duplicated(a, da), Duplicated(b, db))[:1] ≈ autodiff(Enzyme.Forward, -, Duplicated, Duplicated(af64, daf64), Duplicated(bf64, dbf64))[1]
    @test autodiff(Enzyme.Forward, -, Duplicated, Duplicated(a, da), Duplicated(bf64, dbf64))[:1] ≈ autodiff(Enzyme.Forward, -, Duplicated, Duplicated(af64, daf64), Duplicated(bf64, dbf64))[1]
    @test autodiff(Enzyme.Forward, *, Duplicated, Duplicated(a, da), Duplicated(b, db))[:1] ≈ autodiff(Enzyme.Forward, *, Duplicated, Duplicated(af64, daf64), Duplicated(bf64, dbf64))[1]
    @test autodiff(Enzyme.Forward, *, Duplicated, Duplicated(a, da), Duplicated(bf64, dbf64))[:1] ≈ autodiff(Enzyme.Forward, *, Duplicated, Duplicated(af64, daf64), Duplicated(bf64, dbf64))[1]
    @test autodiff(Enzyme.Forward, /, Duplicated, Duplicated(a, da), Duplicated(b, db))[:1] ≈ autodiff(Enzyme.Forward, /, Duplicated, Duplicated(af64, daf64), Duplicated(bf64, dbf64))[1]
    @test autodiff(Enzyme.Forward, /, Duplicated, Duplicated(a, da), Duplicated(bf64, dbf64))[:1] ≈ autodiff(Enzyme.Forward, /, Duplicated, Duplicated(af64, daf64), Duplicated(bf64, dbf64))[1]
    
    @test autodiff(Enzyme.Forward, inv, Duplicated, Duplicated(a, da))[:1] ≈ autodiff(Enzyme.Forward, inv, Duplicated, Duplicated(af64, daf64))[1]
    @test autodiff(Enzyme.Forward, sin, Duplicated, Duplicated(a, da))[:1] ≈ autodiff(Enzyme.Forward, sin, Duplicated, Duplicated(af64, daf64))[1]
    @test autodiff(Enzyme.Forward, cos, Duplicated, Duplicated(a, da))[:1] ≈ autodiff(Enzyme.Forward, cos, Duplicated, Duplicated(af64, daf64))[1]
    @test autodiff(Enzyme.Forward, tan, Duplicated, Duplicated(a, da))[:1] ≈ autodiff(Enzyme.Forward, tan, Duplicated, Duplicated(af64, daf64))[1]
end
