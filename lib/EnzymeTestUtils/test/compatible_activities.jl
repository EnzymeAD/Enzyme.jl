using Test
using EnzymeTestUtils
using Enzyme

@testset "are_activities_compatible" begin
    not_batch = (Active, Const, Duplicated)
    not_batch_ret = (not_batch..., DuplicatedNoNeed)
    batch = (Const, BatchDuplicated)
    batch_ret = (batch..., BatchDuplicatedNoNeed)
    @testset for Tret in not_batch_ret, Tx in not_batch, Ty in not_batch
        @test are_activities_compatible(Tret, Tx, Ty)
    end
    @testset for Tret in batch_ret, Tx in not_batch, Ty in not_batch
        if Tret <: Const || (Tx <: Union{Const,Active} && Ty <: Union{Const,Active})
            continue
        end
        @test !are_activities_compatible(Tret, Tx, Ty)
    end
    @testset for Tret in not_batch_ret, Tx in batch, Ty in not_batch
        if Tx <: Const || (Tret <: Union{Const,Active} && Ty <: Union{Const,Active})
            continue
        end
        @test !are_activities_compatible(Tret, Tx, Ty)
    end
    @testset for Tret in not_batch_ret, Tx in not_batch, Ty in batch
        if Ty <: Const || (Tret <: Union{Const,Active} && Tx <: Union{Const,Active})
            continue
        end
        @test !are_activities_compatible(Tret, Tx, Ty)
    end
end
