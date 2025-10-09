using Enzyme
using Documenter
using Test

DocMeta.setdocmeta!(Enzyme, :DocTestSetup, :(using Enzyme); recursive = true)

@testset "DocTests" begin
    doctest(Enzyme; manual = false)
end
