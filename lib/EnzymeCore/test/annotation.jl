using EnzymeCore
using EnzymeCore: batchify_activity
using Test

@testset "Batchify activity" begin
    @test batchify_activity(Active{Float64}, Val(2)) == Active{Float64}
    @test batchify_activity(Duplicated{Vector{Float64}}, Val(2)) == BatchDuplicated{Vector{Float64},2}
    @test batchify_activity(DuplicatedNoNeed{Vector{Float64}}, Val(2)) == BatchDuplicatedNoNeed{Vector{Float64},2}
    @test batchify_activity(MixedDuplicated{Tuple{Float64,Vector{Float64}}}, Val(2)) == BatchMixedDuplicated{Tuple{Float64,Vector{Float64}},2}
end
