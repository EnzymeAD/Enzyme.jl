# HACK: work around Pkg.jl#2500
if VERSION < v"1.8-"
test_project = Base.active_project()
preferences_file = joinpath(dirname(@__DIR__), "LocalPreferences.toml")
test_preferences_file = joinpath(dirname(test_project), "LocalPreferences.toml")
if isfile(preferences_file) && !isfile(test_preferences_file)
    cp(preferences_file, test_preferences_file)
end
end

using Enzyme
using Test
using FiniteDifferences
using ForwardDiff
using Statistics
using LinearAlgebra

using Enzyme_jll
@info "Testing against" Enzyme_jll.libEnzyme

import LLVM_jll
using LLVM
# TODO: Add to LLVM_jll
function lit(; adjust_PATH=true, adjust_LIBPATH=true)
    lit_path = joinpath(LLVM_jll.artifact_dir, "tools", "lit", "lit.py")
    env = LLVM_jll.JLLWrappers.adjust_ENV!(
            copy(ENV),
            LLVM_jll.PATH[],
            LLVM_jll.LIBPATH[],
            adjust_PATH,
            adjust_LIBPATH,
        )
    return Cmd(Cmd([lit_path]); env)
end

@testset "Threads" begin
    cmd = `$(Base.julia_cmd()) --threads=1 --startup-file=no threads.jl`
   	@test success(pipeline(cmd, stderr=stderr, stdout=stdout))
end

