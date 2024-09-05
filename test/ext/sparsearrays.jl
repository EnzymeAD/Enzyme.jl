using Enzyme
using SparseArrays
using LinearAlgebra

@testset "SparseArrays reverse rule" begin
    C = zeros(128)
    M = sprand(128, 96, 0.1)
    v = randn(96)
    α = 2.0
    β = 1.0

    for Tret in (Duplicated,), Tv in (Const, Duplicated), 
        Tα in (Const, Active), Tβ in (Const, Active)
        test_reverse(mul!, Tret, (C, Tret), (M, Const), (v, Tv), (α, Tα), (β, Tβ))
    end

    for Tret in (Duplicated,), Tv in (Const, Duplicated)
        test_reverse(mul!, Tret, (C, Tret), (M, Const), (v, Tv), (true, Const), (true, Const))
    end

    for Tret in (Duplicated,), Tv in (Const, Duplicated), bα in (true, false), bβ in (true, false)
        test_reverse(mul!, Tret, (C, Tret), (M, Const), (v, Tv), (bα, Const), (bβ, Const))
    end
end