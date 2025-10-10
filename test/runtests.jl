using ParallelTestRunner: runtests
using Enzyme_jll: Enzyme_jll

include("setup.jl")     # make sure everything is precompiled

function testfilter(test)
    if test ∈ ("metal", "cuda", "amdgpu", "setup")
        return false
    end
    if Sys.iswindows() && test == "ext/specialfunctions"
        return false
    end
    return true
end

@info "Testing against" Enzyme_jll.libEnzyme
runtests(ARGS; testfilter, init_code=:(include("setup.jl")))
