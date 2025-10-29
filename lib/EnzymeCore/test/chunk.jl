using Test
using EnzymeCore

@testset "SingleChunk" begin
    @test pick_chunksize(SingleChunk(), ones(10)) == Val(10)
    @test pick_chunksize(SingleChunk(), ones(100)) == Val(100)
end

@testset "AutoChunk" begin
    @test pick_chunksize(AutoChunk(), ones(10)) == Val(10)
    @test pick_chunksize(AutoChunk(), ones(100)) == Val(16)
end
