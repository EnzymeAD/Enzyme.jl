using Enzyme, Test

# Enzyme keeps what it compiles in globals of its own, and a thunk is held as the address of
# the code the JIT emitted for it. Neither survives the process that produced it, so a
# package that differentiates while it precompiles writes an image that refers to a dead
# session (EnzymeAD/Enzyme.jl#1549). What follows pins down which parts of that already work
# and which do not: the `@test_broken` cases are what precompilation support has to fix, and
# the plain `@test` cases are what it must not regress.

# Write the packages into a directory of their own and put it first on the child's load
# path. The depot is the one this test runs under, so the packages precompile next to the
# Enzyme the parent is testing instead of rebuilding the world in a fresh depot.
function precompile_test_harness(f)
    load_path = mktempdir()
    try
        f(load_path)
    finally
        try
            rm(load_path; recursive = true, force = true)
        catch  # Windows may still hold the image files open
        end
    end
    return nothing
end

function child_command(load_path, code)
    return addenv(
        `$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $code`,
        "JULIA_LOAD_PATH" => join([load_path, "@", "@v#.#", "@stdlib"], Sys.iswindows() ? ";" : ":"),
    )
end

# Run a child, and give back whether it exited cleanly along with what it printed.
function run_child(load_path, code)
    out = IOBuffer()
    ok = success(pipeline(child_command(load_path, code); stdout = out, stderr = devnull))
    return ok, String(take!(out))
end

const USES_AD = """
module EnzymePrecompileUsesAD
using Enzyme

scalar(x) = x * x * x
harmonic(x) = sin(x) / x

# Only differentiated once the package is loaded, never while it precompiles.
rev(x) = Enzyme.autodiff(Reverse, scalar, Active(x))[1][1]
fwd(x) = Enzyme.autodiff(Forward, harmonic, Duplicated(x, 1.0))[1]
end
"""

const AD_WHILE_PRECOMPILING = """
module EnzymePrecompileAtBuild
using Enzyme

scalar(x) = x * x * x
harmonic(x) = sin(x) / x

rev(x) = Enzyme.autodiff(Reverse, scalar, Active(x))[1][1]
fwd(x) = Enzyme.autodiff(Forward, harmonic, Duplicated(x, 1.0))[1]

# Differentiate as the package precompiles, so the thunks are compiled in the process that
# writes the image and whatever refers to them is what the image carries.
const PRECOMPILED = (rev(2.0), fwd(2.0))
end
"""

const rev_at_3 = 3 * 3.0^2
const fwd_at_3 = (cos(3.0) * 3.0 - sin(3.0)) / 3.0^2

@testset "a package that differentiates only once it is loaded" begin
    precompile_test_harness() do load_path
        write(joinpath(load_path, "EnzymePrecompileUsesAD.jl"), USES_AD)
        code = """
        using Enzyme, EnzymePrecompileUsesAD
        print(EnzymePrecompileUsesAD.rev(3.0), " ", EnzymePrecompileUsesAD.fwd(3.0))
        """

        # The first child writes the image, the second loads it.
        for _ in 1:2
            ok, out = run_child(load_path, code)
            @test ok
            if ok
                rev3, fwd3 = split(out)
                @test parse(Float64, rev3) ≈ rev_at_3
                @test parse(Float64, fwd3) ≈ fwd_at_3
            end
        end
    end
end

@testset "a package that differentiates while it precompiles" begin
    precompile_test_harness() do load_path
        write(joinpath(load_path, "EnzymePrecompileAtBuild.jl"), AD_WHILE_PRECOMPILING)

        # Differentiating during precompilation itself works, and the values it computed are
        # in the image: the first child precompiles the package, the second only loads it.
        values_code = """
        using Enzyme, EnzymePrecompileAtBuild
        print(EnzymePrecompileAtBuild.PRECOMPILED[1], " ", EnzymePrecompileAtBuild.PRECOMPILED[2])
        """
        for _ in 1:2
            ok, out = run_child(load_path, values_code)
            @test ok
            if ok
                rev2, fwd2 = split(out)
                @test parse(Float64, rev2) ≈ 3 * 2.0^2
                @test parse(Float64, fwd2) ≈ (cos(2.0) * 2.0 - sin(2.0)) / 2.0^2
            end
        end

        # Differentiating again from a loaded image is what does not work yet. The thunk the
        # package image refers to was compiled by the process that precompiled it, so the
        # call goes to an address that means nothing here and the child dies on it.
        call_code = """
        using Enzyme, EnzymePrecompileAtBuild
        print(EnzymePrecompileAtBuild.rev(3.0), " ", EnzymePrecompileAtBuild.fwd(3.0))
        """
        ok, out = run_child(load_path, call_code)
        @test_broken ok
        if ok
            rev3, fwd3 = split(out)
            @test parse(Float64, rev3) ≈ rev_at_3
            @test parse(Float64, fwd3) ≈ fwd_at_3
        end
    end
end

@testset "Enzyme's own precompilation" begin
    precompile_test_harness() do load_path
        # Enzyme differentiates in its `@compile_workload`, and the cache that fills is a
        # global of Enzyme's, so it is serialized into Enzyme's image along with the
        # addresses that session's JIT handed out. Every session that loads Enzyme inherits
        # them.
        code = """
        using Enzyme
        print(length(Enzyme.Compiler.cache), " ",
              Enzyme.autodiff(Reverse, x -> x * x, Active(4.0))[1][1])
        """
        ok, out = run_child(load_path, code)
        @test ok
        if ok
            cached, grad = split(out)
            # Differentiating something new is unaffected by what the cache was left holding.
            @test parse(Float64, grad) ≈ 8.0
            # Nothing of the session that built the image should still be in the cache.
            @test_broken parse(Int, cached) == 0
        end

        # And they are reachable. The workload differentiates a function that stays in
        # Enzyme, so asking for that same derivative in a fresh session finds the entry the
        # image carries and calls where that code used to be.
        workload_code = """
        using Enzyme
        mods = [getfield(Enzyme, n) for n in names(Enzyme; all = true) if
                startswith(string(n), "#") && isdefined(Enzyme, n) &&
                getfield(Enzyme, n) isa Module]
        fns = [m.f for m in mods if isdefined(m, :f)]
        if isempty(fns)
            print("no workload function")   # the workload stopped leaving one behind
        else
            print(Enzyme.autodiff(Reverse, first(fns), Active(2.0))[1][1])
        end
        """
        ok, out = run_child(load_path, workload_code)
        if ok && out == "no workload function"
            @test_skip ok
        else
            @test_broken ok
        end
    end
end
