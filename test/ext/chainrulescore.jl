using Enzyme
using Test
using ChainRulesCore
using EnzymeCore
using LinearAlgebra
using EnzymeTestUtils

# utility functions
fdiff(f, x::Number) = autodiff(Forward, f, Duplicated, Duplicated(x, one(x)))[2]
rdiff(f, x::Number) = autodiff(Reverse, f, Active, Active(x))[1][1]
rdiff(f, x::AbstractArray) = autodiff(Reverse, f, Active, Duplicated(x, zero(x)))[1][1]


# @testset "import_frule" begin
    f1(x) = 2*x
    ChainRulesCore.@scalar_rule f1(x)  (5.0,)
    Enzyme.@import_frule typeof(Main.f1) Any
    @test fdiff(f1, 1.0) === 5.0

    # specific signature    
    f2(x) = 2*x
    ChainRulesCore.@scalar_rule f2(x)  (5*one(x),)
    Enzyme.@import_frule typeof(Main.f2) Float32
    @test fdiff(f2, 1f0) === 5f0
    @test fdiff(f2, 1.0) === 2.0
# end


# function EnzymeRules.forward(func::Const{typeof(f)}, RT, x::Duplicated)
#     println("using custom Enzyme forward rule")
#     y = func.val(x.val)
#     @show x x.dval typeof(x.dval) RT
#     if RT <: Const
#         return y
#     elseif RT <: Duplicated
#         return Duplicated(y, 5.0)
#     end
# end






