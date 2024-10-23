using Test
using EnzymeCore

@testset verbose = true "EnzymeCore" begin
    @testset "Miscellaneous" begin
        include("misc.jl")
    end
end
