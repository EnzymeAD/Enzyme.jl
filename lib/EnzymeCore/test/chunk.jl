using Test
using EnzymeCore

@testset "SingleChunk" begin
    @test pick_chunksize(SingleChunk(), ones(10)) == Val(10)
    @test pick_chunksize(SingleChunk(), ones(100)) == Val(100)
end

@testset "FixedChunk" begin
    @test_throws ErrorException pick_chunksize(FixedChunk{3}(), ones(2))
    @test pick_chunksize(FixedChunk{3}(), ones(10)) == Val(3)
    @test pick_chunksize(FixedChunk{3}(), ones(100)) == Val(3)
    @test pick_chunksize(FixedChunk{4}(), ones(100)) == Val(4)
end

@testset "AutoChunk" begin
    @test pick_chunksize(AutoChunk(), ones(10)) == Val(10)
    @test pick_chunksize(AutoChunk(), ones(100)) == Val(16)
end
