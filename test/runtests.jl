using Enzyme
using Test

Enzyme.API.printall!(true)
Enzyme.Compiler.DumpPostOpt[] = true

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
end
