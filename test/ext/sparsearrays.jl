using Enzyme
using SparseArrays
using LinearAlgebra

@testset "SparseArrays spmatvec reverse rule" begin
    C = zeros(18)
    M = sprand(18, 9, 0.1)
    v = randn(9)
    α = 2.0
    β = 1.0

    for Tret in (Duplicated,), Tv in (Const, Duplicated,), 
        Tα in (Const, Active), Tβ in (Const, Active)

        are_activities_compatible(Tret, Tret, Tv, Tα, Tβ) || continue

        test_reverse(mul!, Tret, (C, Tret), (M, Const), (v, Tv), (α, Tα), (β, Tβ))
    end

    # Batch mode crashes so we skip it
    for Tret in (BatchDuplicated,), Tv in (Const,), 
        Tα in (Const, Active), Tβ in (Const, Active)

        are_activities_compatible(Tret, Tret, Tv, Tα, Tβ) || continue

        test_reverse(mul!, Tret, (C, Tret), (M, Const), (v, Tv), (α, Tα), (β, Tβ))
    end



    for Tret in (Duplicated,), Tv in (Const, Duplicated), bα in (true, false), bβ in (true, false)
        are_activities_compatible(Tret, Tret, Tv) || continue
        test_reverse(mul!, Tret, (C, Tret), (M, Const), (v, Tv), (bα, Const), (bβ, Const))
    end
end

@testset "SparseArrays spmatmat reverse rule" begin
    C = zeros(18, 11)
    M = sprand(18, 9, 0.1)
    v = randn(9, 11)
    α = 2.0
    β = 1.0

    for Tret in (Duplicated,), Tv in (Const, Duplicated,), 
        Tα in (Const, Active), Tβ in (Const, Active)

        are_activities_compatible(Tret, Tv, Tα, Tβ) || continue
        test_reverse(mul!, Tret, (C, Tret), (M, Const), (v, Tv), (α, Tα), (β, Tβ))
    end

    for Tret in (Duplicated,), Tv in (Const, Duplicated), bα in (true, false), bβ in (true, false)
        are_activities_compatible(Tret, Tv) || continue
        test_reverse(mul!, Tret, (C, Tret), (M, Const), (v, Tv), (bα, Const), (bβ, Const))
    end
end