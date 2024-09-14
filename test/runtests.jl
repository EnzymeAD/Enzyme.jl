# # work around https://github.com/JuliaLang/Pkg.jl/issues/1585
# using Pkg
# Pkg.develop(PackageSpec(; path=joinpath(dirname(@__DIR__), "lib", "EnzymeTestUtils")))

using GPUCompiler
using Enzyme
using Test
using FiniteDifferences
using Aqua
using SparseArrays
using StaticArrays
using Statistics
using LinearAlgebra
using InlineStrings

using Enzyme_jll
@info "Testing against" Enzyme_jll.libEnzyme

include("utils.jl")

# Aqua.test_all(Enzyme, unbound_args=false, piracies=false, deps_compat=false, stale_deps=(;:ignore=>[:EnzymeTestUtils]))
# Aqua.test_all(Enzyme, unbound_args=false, piracies=false, deps_compat=false, stale_deps=(;:ignore=>[:EnzymeTestUtils]))

include("abi.jl")
include("typetree.jl")

@static if Enzyme.EnzymeRules.issupported()
    include("rules/rules.jl")
    include("rules/rrules.jl")
    include("rules/mixedrrule.jl")
    include("rules/kwrules.jl")
    include("rules/kwrrules.jl")
    include("rules/internal_rules.jl")
    include("rules/ruleinvalidation.jl")
end
@static if !Sys.iswindows()
    include("blas.jl")
end

include("internals.jl")

include("duplicated.jl")
include("simple.jl")
include("deferred.jl")
include("basestd.jl")

include("arrays.jl")
include("memory.jl")
include("gradjac.jl")

@testset "Threads" begin
    cmd = `$(Base.julia_cmd()) --threads=1 --startup-file=no threads.jl`
   	@test success(pipeline(cmd, stderr=stderr, stdout=stdout))
    cmd = `$(Base.julia_cmd()) --threads=2 --startup-file=no threads.jl`
   	@test success(pipeline(cmd, stderr=stderr, stdout=stdout))
end

include("mixed.jl")
include("applyiter.jl")
include("mixedapplyiter.jl")

# TEST EXTENSIONS 
Sys.iswindows() || include("ext/specialfunctions.jl")
include("ext/chainrulescore.jl")
include("ext/logexpfunctions.jl")
include("ext/bfloat16s.jl")

using  Documenter
DocMeta.setdocmeta!(Enzyme, :DocTestSetup, :(using Enzyme); recursive=true)
@testset "DocTests" begin
    doctest(Enzyme; manual = false)
end

