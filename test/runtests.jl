import ParallelTestRunner: runtests

include("setup.jl")     # make sure everything is precompiled

function testfilter(test)
    if test ∈ ("metal", "cuda", "amdgpu")
        return false
    end
    if Sys.iswindows() && test == "ext/specialfunctions"
        return false
    end
    return true
end

runtests(ARGS; testfilter)
