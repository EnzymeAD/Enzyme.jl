using Enzyme, Test

using Enzyme_jll
@info "Testing against" Enzyme_jll.libEnzyme

# Aqua.test_all(Enzyme, unbound_args=false, piracies=false, deps_compat=false, stale_deps=(;:ignore=>[:EnzymeTestUtils]))
# Aqua.test_all(Enzyme, unbound_args=false, piracies=false, deps_compat=false, stale_deps=(;:ignore=>[:EnzymeTestUtils]))

include("internals/internals.jl")
include("internals/abi.jl")
include("internals/usermixed.jl")
include("internals/typetree.jl")

@static if Enzyme.EnzymeRules.issupported()
    include("rules/rules.jl")
    include("rules/rrules.jl")
    include("rules/mixedrrule.jl")
    include("rules/kwrules.jl")
    include("rules/kwrrules.jl")
    include("rules/internal_rules.jl")
    include("rules/ruleinvalidation.jl")
end
@static Sys.iswindows() || include("blas.jl")

include("duplicated.jl")
include("simple.jl")
include("deferred.jl")
include("basestd.jl")

include("errors.jl")
include("arrays.jl")  #WARN: this currently broken somehow
include("memory.jl")
include("gradjac.jl")

include("mixed.jl")
include("applyiter.jl")
include("mixedapplyiter.jl")
include("misc.jl")

@testset "Threads" begin
    cmd = `$(Base.julia_cmd()) --threads=1 --startup-file=no threads.jl`
   	@test success(pipeline(cmd, stderr=stderr, stdout=stdout))
    cmd = `$(Base.julia_cmd()) --threads=2 --startup-file=no threads.jl`
   	@test success(pipeline(cmd, stderr=stderr, stdout=stdout))
end

# TEST EXTENSIONS 
@static Sys.iswindows() || include("ext/specialfunctions.jl")
include("ext/chainrulescore.jl")
include("ext/logexpfunctions.jl")
include("ext/bfloat16s.jl")
include("ext/staticarrays.jl")

include("docs.jl")
