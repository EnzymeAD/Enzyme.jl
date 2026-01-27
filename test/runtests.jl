#=import Enzyme
import Enzyme_jll
using ParallelTestRunner: addworker, filter_tests!, find_tests, parse_args, runtests

# Start with autodiscovered tests
testsuite = find_tests(@__DIR__)
# Add threads tests to be run with multiple Julia threads (will be configured in
# `test_worker`).
testsuite["threads/2"] = :(include($(joinpath(@__DIR__, "threads.jl"))))
# Exclude integration tests, they're handled differently (they each run in their
# own environment)
for (k, _) in testsuite
    startswith(k, "integration/") && delete!(testsuite, k)
end

# Parse arguments
args = parse_args(ARGS)

if filter_tests!(testsuite, args)
    # Skip GPU-specific tests by default.
    delete!(testsuite, "metal")
    delete!(testsuite, "cuda")
    delete!(testsuite, "amdgpu")
    delete!(testsuite, "common")

    # Skipped until https://github.com/EnzymeAD/Enzyme.jl/issues/2620 is fixed.
    if Sys.iswindows()
        delete!(testsuite, "ext/specialfunctions")
    end
end

function test_worker(name)
    if name == "threads/2"
        # Run the `threads/2` testset, with multiple threads.
        return addworker(; exeflags = ["--threads=2"])
    end
end

const init_code = quote end

@info "Testing against" Enzyme_jll.libEnzyme
runtests(Enzyme, args; testsuite, init_code, test_worker)=#
include("rules/internal_rules/bigfloat.jl")
