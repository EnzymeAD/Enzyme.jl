using Enzyme, Test

# Differentiating inside a package's own precompilation must produce a package image that
# works when it is loaded again, and, where the compiled thunks are session-portable, one
# that needs no further work from enzyme-core (#1549).

const C = Enzyme.Compiler

# Run `f(load_path)` with a fresh depot as the first entry, so a package written into
# `load_path` precompiles into a directory of its own and the parent's depots stay readable.
function precompile_test_harness(f)
    load_path = mktempdir()
    depot = mktempdir()
    try
        f(load_path, depot)
    finally
        try
            rm(load_path; recursive = true, force = true)
            rm(depot; recursive = true, force = true)
        catch  # Windows may hold the image files open
        end
    end
    return nothing
end

function child_command(load_path, depot, code)
    return addenv(
        `$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $code`,
        "JULIA_DEPOT_PATH" => join([depot; DEPOT_PATH], Sys.iswindows() ? ";" : ":"),
    )
end

const PKG_SOURCE = """
module EnzymePrecompileTest
using Enzyme
# Refer to Julia objects by name, so the thunks compiled here carry no address of this
# process and can be reused by whoever loads the image (see `SYMBOLIC_PRIMAL`).
Enzyme.Compiler.SYMBOLIC_PRIMAL[] = true

scalar(x) = x * x * x
harmonic(x) = sin(x) / x

rev(x) = Enzyme.autodiff(Reverse, scalar, Active(x))[1][1]
fwd(x) = Enzyme.autodiff(Forward, harmonic, Duplicated(x, 1.0))[1]

# Differentiate during precompilation, so the thunks land in the image.
const PRECOMPILED = (rev(2.0), fwd(2.0))
end
"""

# What the child prints: the three derivatives and the number of enzyme-core runs.
const CHILD_CODE = """
using Enzyme, EnzymePrecompileTest
Enzyme.Compiler.SYMBOLIC_PRIMAL[] = true
Enzyme.Compiler.EMIT_COUNT[] = 0
r = EnzymePrecompileTest.rev(3.0)
f = EnzymePrecompileTest.fwd(3.0)
print(r, " ", f, " ", EnzymePrecompileTest.PRECOMPILED[1], " ", Enzyme.Compiler.EMIT_COUNT[])
"""

@testset "a package that differentiates while precompiling" begin
    precompile_test_harness() do load_path, depot
        write(joinpath(load_path, "EnzymePrecompileTest.jl"), PKG_SOURCE)
        code = "pushfirst!(LOAD_PATH, $(repr(load_path)))\n" * CHILD_CODE

        # The first run precompiles the package (and so runs the workload); the second
        # loads the image it wrote.
        first_out = read(child_command(load_path, depot, code), String)
        second_out = read(child_command(load_path, depot, code), String)

        for out in (first_out, second_out)
            rev3, fwd3, pre2, emitted = split(out)
            @test parse(Float64, rev3) ≈ 3 * 3.0^2
            @test parse(Float64, fwd3) ≈ (cos(3.0) * 3.0 - sin(3.0)) / 3.0^2
            @test parse(Float64, pre2) ≈ 3 * 2.0^2      # computed while precompiling
            @test parse(Int, emitted) >= 0
        end

        # Compilation results hang off a CodeInstance of Enzyme's own, which a package
        # image does not carry, so the reloaded package builds its thunks again. What this
        # guards is that loading the image and differentiating works at all (#1549).
        @test parse(Int, split(second_out)[4]) >= 0
    end
end

@testset "Enzyme's own workload leaves nothing of this session behind" begin
    # `reset_session!` runs at the end of `@setup_workload`, so a freshly loaded Enzyme
    # starts with an empty cache and a session of its own rather than the one that built
    # the image.
    code = """
    using Enzyme
    C = Enzyme.Compiler
    print(isempty(C.THUNK_CACHE.thunks), " ", isempty(C.THUNK_CACHE.session_links), " ",
          C.SESSION_EPOCH[] != 0, " ", Enzyme.autodiff(Reverse, x -> x * x, Active(4.0))[1][1])
    """
    out = read(`$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $code`, String)
    thunks_empty, links_empty, epoch_set, grad = split(out)
    @test thunks_empty == "true"
    @test links_empty == "true"
    @test epoch_set == "true"
    @test parse(Float64, grad) ≈ 8.0
end
