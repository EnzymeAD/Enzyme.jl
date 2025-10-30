using Test
using EnzymeCore

@testset "SmallestChunk" begin
    @test pick_chunksize(SmallestChunk(), 10) == Val(1)
    @test pick_chunksize(SmallestChunk(), ones(10)) == Val(1)
    @test pick_chunksize(SmallestChunk(), 100) == Val(1)
    @test pick_chunksize(SmallestChunk(), ones(100)) == Val(1)
end

@testset "LargestChunk" begin
    @test pick_chunksize(LargestChunk(), 10) == Val(10)
    @test pick_chunksize(LargestChunk(), ones(10)) == Val(10)
    @test pick_chunksize(LargestChunk(), 100) == Val(100)
    @test pick_chunksize(LargestChunk(), ones(100)) == Val(100)
end

@testset "FixedChunk" begin
    @test_throws ErrorException pick_chunksize(FixedChunk{3}(), 2)
    @test_throws ErrorException pick_chunksize(FixedChunk{3}(), ones(2))
    @test pick_chunksize(FixedChunk{3}(), 10) == Val(3)
    @test pick_chunksize(FixedChunk{3}(), ones(10)) == Val(3)
    @test pick_chunksize(FixedChunk{3}(), 100) == Val(3)
    @test pick_chunksize(FixedChunk{3}(), ones(100)) == Val(3)
    @test pick_chunksize(FixedChunk{4}(), 100) == Val(4)
    @test pick_chunksize(FixedChunk{4}(), ones(100)) == Val(4)
end

@testset "AutoChunk" begin
    @test pick_chunksize(AutoChunk(), 10) == Val(10)
    @test pick_chunksize(AutoChunk(), ones(10)) == Val(10)
    @test pick_chunksize(AutoChunk(), 100) == Val(16)
    @test pick_chunksize(AutoChunk(), ones(100)) == Val(16)
end
