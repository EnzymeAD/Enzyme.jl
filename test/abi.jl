using Enzyme
using Test

@testset "ABI & Calling convention" begin

# GhostType -> Nothing
f(x) = x 
# res = autodiff(f, Const(nothing))
# @test res === nothing

# ConstType -> Type{Int}
# res = autodiff(f, Int)
# @test res === Int

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
res2 = autodiff(g, Active(Foo(3, 1.2)))
@test res2[1].qux ≈ 1.0


h(x, y) = x.qux * y.qux
res3 = autodiff(h, Active(Foo(3, 1.2)), Active(Foo(5, 3.4)))
@test res3[1].qux ≈ 3.4
@test res3[2].qux ≈ 1.2


# deserves_argbox yes and no
struct Bar
    r::Ref{Int}
end

struct LList
    next::LList
    val::Float64
end

# Multi arg => sret

# ConstType

# primitive type Int128, Float64, Float128

# returns: sret, const/ghost, !deserve_retbox
end
