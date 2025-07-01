module InternalRules

using Enzyme
using Enzyme.EnzymeRules
using LinearAlgebra
using SparseArrays
using Test
import Random

@testset "SparseArrays spmatvec reverse rule" begin
    Ts = (Float64, ComplexF64)

    Ms = sprandn.(Ts, 5, 3, 0.3)
    vs = rand.(Ts, 3)
    αs = rand.(Ts)
    βs = rand.(Ts)

    for M in Ms, v in vs, α in αs, β in βs
        tout = promote_type(eltype(M), eltype(v), typeof(α), typeof(β))
        C = zeros(tout, 5)

        for Tret in (Duplicated, BatchDuplicated), TM in (Const, Duplicated, BatchDuplicated), Tv in (Const, Duplicated, BatchDuplicated), 
            Tα in (Const, Active), Tβ in (Const, Active)

            are_activities_compatible(Tret, Tret, TM, Tv, Tα, Tβ) || continue
            test_reverse(LinearAlgebra.mul!, Tret, (C, Tret), (M, TM), (v, Tv), (α, Tα), (β, Tβ))
        end

        for Tret in (Duplicated, BatchDuplicated), TM in (Const, Duplicated, BatchDuplicated), 
            Tv in (Const, Duplicated, BatchDuplicated), bα in (true, false), bβ in (true, false)
            are_activities_compatible(Tret, Tret, TM, Tv) || continue
            test_reverse(LinearAlgebra.mul!, Tret, (C, Tret), (M, Const), (v, Tv), (bα, Const), (bβ, Const))
        end

        test_reverse(LinearAlgebra.mul!, Const, (C, Const), (M, Const), (v, Const), (α, Active), (β, Active))

    end
end



end # InternalRules
