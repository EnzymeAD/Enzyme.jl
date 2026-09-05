using Enzyme, Test

# Enzyme keeps what it compiles in globals of its own, and a thunk is held as the address of
# the code the JIT emitted for it. Neither survives the process that produced it, so a
# package that differentiates while it precompiles writes an image that refers to a dead
# session (EnzymeAD/Enzyme.jl#1549). What follows pins down which parts of that already work
# and which do not: the `@test_broken` cases are what precompilation support has to fix, and
# the plain `@test` cases are what it must not regress.
#
# One of them Enzyme can already keep out of an image, and does: the caches its own workload
# fills are emptied before the image is written. What no cache can help with is an address
# baked into code that was compiled during precompilation, which is what the remaining broken
# cases are.

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

# Options under which Julia ignores the package images it has and compiles from source
# instead. The test process may well be running under them, `--check-bounds=yes` and coverage
# being what CI tests with, and `Base.julia_cmd` would pass them on to the children, where
# they would leave nothing here testing what it says it does.
function ignores_pkgimages(arg::AbstractString)
    return startswith(arg, "--check-bounds") || startswith(arg, "--code-coverage") ||
        startswith(arg, "--track-allocation") || startswith(arg, "--inline") ||
        startswith(arg, "--pkgimages")
end

function child_julia_cmd()
    exec = String[arg for arg in Base.julia_cmd().exec if !ignores_pkgimages(arg)]
    push!(exec, "--pkgimages=yes")
    return Cmd(exec)
end

# Each child reports first whether it does have package images, so that a child which never
# loaded one cannot quietly stand in for one that did.
const CHILD_PREAMBLE = """
print(Int(Base.JLOptions().use_pkgimages), " ", Int(Base.JLOptions().code_coverage), " ")
flush(stdout)
"""

function child_command(load_path, code)
    return addenv(
        `$(child_julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $code`,
        "JULIA_LOAD_PATH" => join([load_path, "@", "@v#.#", "@stdlib"], Sys.iswindows() ? ";" : ":"),
    )
end

# Run a child, and give back whether it exited cleanly along with the fields it printed. The
# preamble is flushed before anything else runs, so those two come back even from a child
# that dies partway.
function run_child(load_path, code)
    out = IOBuffer()
    ok = success(
        pipeline(
            child_command(load_path, CHILD_PREAMBLE * code);
            stdout = out, stderr = devnull,
        )
    )
    fields = split(String(take!(out)))
    @test length(fields) >= 2 && fields[1] == "1" && fields[2] == "0"
    return ok, fields[3:end]
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
            ok, fields = run_child(load_path, code)
            @test ok
            if ok
                rev3, fwd3 = fields
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
            ok, fields = run_child(load_path, values_code)
            @test ok
            if ok
                rev2, fwd2 = fields
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
        ok, fields = run_child(load_path, call_code)
        @test_broken ok
        if ok
            rev3, fwd3 = fields
            @test parse(Float64, rev3) ≈ rev_at_3
            @test parse(Float64, fwd3) ≈ fwd_at_3
        end
    end
end

@testset "Enzyme's own precompilation" begin
    precompile_test_harness() do load_path
        # Enzyme differentiates in its `@compile_workload`, and the caches that fills are
        # globals of Enzyme's, so whatever is left in them is serialized into Enzyme's image
        # along with the addresses that session's JIT handed out. `clear_caches!` at the end
        # of the workload is what keeps them out of it.
        code = """
        using Enzyme
        C = Enzyme.Compiler
        sizes = (length(C.cache), length(C.autodiff_cache), length(Enzyme.tape_cache),
                 length(C.FRULE_CACHE), length(C.RRULE_CACHE), length(C.INACTIVE_CACHE),
                 length(C.EASY_RULE_CACHE), length(C.NOALIAS_CACHE),
                 length(C.Interpreter.SigCache), length(C.ActivityCache),
                 length(C.ActivityMethodCache), Int(C.ActivityWorldCache[]),
                 length(C.JIT.hnd_string_map), length(C.JIT.hnd_int_map))
        print(sum(sizes), " ", Enzyme.autodiff(Reverse, x -> x * x, Active(4.0))[1][1])
        """
        ok, fields = run_child(load_path, code)
        @test ok
        if ok
            cached, grad = fields
            # Nothing the session that built the image compiled or looked up is left.
            @test parse(Int, cached) == 0
            # And a session that starts from nothing differentiates.
            @test parse(Float64, grad) ≈ 8.0
        end

        # An emptied cache is not the whole of it. The workload differentiates a function
        # that stays in Enzyme, and the thunk that call goes through was compiled while
        # Enzyme precompiled, so Enzyme's own image bakes in an address of that session just
        # as a package image does. Asking for that derivative again calls it.
        workload_code = """
        using Enzyme
        mods = [getfield(Enzyme, n) for n in names(Enzyme; all = true) if
                startswith(string(n), "#") && isdefined(Enzyme, n) &&
                getfield(Enzyme, n) isa Module]
        fns = [m.f for m in mods if isdefined(m, :f)]
        if isempty(fns)
            print("none")   # the workload stopped leaving a function behind
        else
            print(Enzyme.autodiff(Reverse, first(fns), Active(2.0))[1][1])
        end
        """
        ok, fields = run_child(load_path, workload_code)
        if ok && fields == ["none"]
            @test_skip ok
        else
            @test_broken ok
        end
    end
end
