using Test
using EnzymeCore

@testset "OneChunk" begin
    @test pick_chunksize(OneChunk(), ones(10)) == 10
    @test pick_chunksize(OneChunk(), ones(100)) == 100
end

@testset "AutoChunk" begin
    @test pick_chunksize(AutoChunk(), ones(10)) == 10
    @test pick_chunksize(AutoChunk(), ones(100)) == 16
end
