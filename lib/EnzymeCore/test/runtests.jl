using Test
using EnzymeCore

@testset verbose = true "EnzymeCore" begin
    @testset "Miscellaneous" begin
        include("misc.jl")
    end
    @testset "Mode modification" begin
        include("mode_modification.jl")
    end
end
