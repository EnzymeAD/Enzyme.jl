using Enzyme, Test

using Enzyme_jll
@info "Testing against" Enzyme_jll.libEnzyme

# Aqua.test_all(Enzyme, unbound_args=false, piracies=false, deps_compat=false, stale_deps=(;:ignore=>[:EnzymeTestUtils]))
# Aqua.test_all(Enzyme, unbound_args=false, piracies=false, deps_compat=false, stale_deps=(;:ignore=>[:EnzymeTestUtils]))

@testset "internals" begin
    @testset "abi" include("internals/abi.jl")
    @testset "typetree" include("internals/typetree.jl")
    @testset "usermixed" include("internals/usermixed.jl")
    @testset "misc internals" include("internals/internals.jl")
end

@static Enzyme.EnzymeRules.issupported() && @testset "rules" begin
    @testset "rules" include("rules/rules.jl")
    @testset "rrules" include("rules/rrules.jl")
    @testset "kwrules" include("rules/kwrules.jl")
    @testset "kwrrules" include("rules/kwrrules.jl")
    @testset "internal rules" include("rules/internal_rules.jl")
    @testset "rule invalidation" include("rules/ruleinvalidation.jl")
end

@static Sys.iswindows() || @testset "blas" include("blas.jl")

@testset "simple tests" include("simple.jl")
@testset "base and stdlibs" include("basestd.jl")
@testset "arrays" include("arrays.jl")
@testset "activity" include("activity.jl")
@testset "duplicated" include("duplicated.jl")
@testset "type system" include("typesystem.jl")
@testset "errors" include("errors.jl")
@testset "gradients and jacobians" include("gradjac.jl")
@testset "higher order derivatives" include("higherorder.jl")
@testset "memory" include("memory.jl")
@testset "mixed" include("mixed.jl")
@testset "applyiter" include("applyiter.jl")
@testset "mixed applyiter" include("mixedapplyiter.jl")
@testset "difftest" include("difftest.jl")
@testset "against finite diff" include("finitediff.jl")
@testset "misc" include("misc.jl")

@testset "Threads" begin
    cmd = `$(Base.julia_cmd()) --threads=1 --startup-file=no threads.jl`
   	@test success(pipeline(cmd, stderr=stderr, stdout=stdout))
    cmd = `$(Base.julia_cmd()) --threads=2 --startup-file=no threads.jl`
   	@test success(pipeline(cmd, stderr=stderr, stdout=stdout))
end

@testset "Extensions" begin
    @testset "ChainRules" include("ext/chainrulescore.jl")
    @testset "LogExpFunctions" include("ext/logexpfunctions.jl")
    @testset "BFloat16s" include("ext/bfloat16s.jl")
    ## https://github.com/JuliaDiff/ChainRules.jl/tree/master/test/rulesets
    Sys.iswindows() || @testset "SpecialFunctions" include("ext/specialfunctions.jl")
end

@testset "doctests" include("docs.jl")
