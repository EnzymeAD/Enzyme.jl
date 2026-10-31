using Enzyme, Test

# Enzyme in a program compiled by `juliac --trim`. Such a program runs without Julia's
# compiler, so nothing can be compiled once it runs: Enzyme generates the derivatives while
# the program is built, in the process that writes the image, and links them into the code
# that calls them (`InlineABI`, which stands in for the default `FFIABI` when trimming). The
# image compiler verifies that everything it keeps is statically resolved, and it keeps every
# `__init__`: Enzyme's must not reach the compiler, which initializes on first use instead.
#
# What this test builds exercises all of that: reverse mode over an array and over a scalar,
# forward mode, and the plain `Reverse` and `Forward` modes, with nothing said about the ABI.

const JULIAC = joinpath(Sys.BINDIR, Base.DATAROOTDIR, "julia", "juliac", "juliac.jl")

# `juliac` compiles with the default compiler of the system; `JULIA_CC` overrides it.
function c_compiler()
    haskey(ENV, "JULIA_CC") && return ENV["JULIA_CC"]
    for c in ("gcc", "clang", "cc")
        Sys.which(c) === nothing || return c
    end
    return nothing
end

# `juliac` links the image against libjulia alone. A derivative calls into the C math
# library (the derivative of `sin` calls `cos`), which the linker of Linux wants named on the
# command line, before the image at that, so keep it needed.
function build_env()
    cc = c_compiler()
    Sys.islinux() || return ()
    return ("JULIA_CC" => "$cc -Wl,--no-as-needed -lm",)
end

# `juliac` and `--trim` need Julia 1.12; Windows builds want a compiler artifact this test
# does not provide for.
const CAN_RUN = VERSION >= v"1.12" && isfile(JULIAC) && !Sys.iswindows() && c_compiler() !== nothing

# Options the test process may run under that would make the build compile Enzyme from
# source (they disable package images), or add instrumentation of their own. The driver's
# `Base.julia_cmd` passes them on to the process that builds the image.
function build_relevant(arg::AbstractString)
    return !(
        startswith(arg, "--check-bounds") || startswith(arg, "--code-coverage") ||
            startswith(arg, "--track-allocation") || startswith(arg, "--inline") ||
            startswith(arg, "--pkgimages") || startswith(arg, "--depwarn")
    )
end

driver_julia_cmd() = Cmd(String[arg for arg in Base.julia_cmd().exec if build_relevant(arg)])

const APP = """
using Enzyme

# Not yet `--trim`-clean, and nothing Enzyme can do about: the `__init__` of Pkg's REPL
# extension (loaded because LazyArtifacts depends on Pkg and GPUCompiler on REPL), and that
# of GPUCompiler, which asks `stderr` whether it has color. Both are replaced the way
# `juliac-trim-stdlib.jl` replaces the `__init__` of Pkg itself.
let
    Pkg = Base.maybe_root_module(Base.PkgId(Base.UUID("44cfe95a-1eb2-52ea-b672-e2afdf69b78f"), "Pkg"))
    if Pkg !== nothing
        REPLExt = Base.get_extension(Pkg, :REPLExt)
        REPLExt === nothing || @eval REPLExt __init__() = Base.time_ns()
    end
    GPUCompiler = Base.maybe_root_module(Base.PkgId(Base.UUID("61eb1bfa-7361-4325-ad38-22787b887f55"), "GPUCompiler"))
    GPUCompiler === nothing || @eval GPUCompiler __init__() = Base.time_ns()
end

cubic(x) = x * x * x
harmonic(x) = sin(x) / x

# Reverse mode over an array, through a `Duplicated` shadow.
function grad_sum_cubic!(g::Vector{Float64}, x::Vector{Float64})
    fill!(g, 0.0)
    Enzyme.autodiff(Reverse, x -> sum(cubic, x), Duplicated(x, g))
    return nothing
end

# Reverse mode of a scalar, returned as a value.
d_cubic(x::Float64) = Enzyme.autodiff(Reverse, cubic, Active(x))[1][1]

# Forward mode.
d_harmonic(x::Float64) = Enzyme.autodiff(Forward, harmonic, Duplicated(x, 1.0))[1]

Base.@ccallable function main()::Cint
    x = [1.0, 2.0, 3.0]
    g = zeros(3)
    grad_sum_cubic!(g, x)
    for v in g
        print(Core.stdout, v, " ")
    end
    print(Core.stdout, d_cubic(2.0), " ")
    print(Core.stdout, d_harmonic(3.0))
    println(Core.stdout)
    return 0
end
"""

const EXPECTED = [3 * 1.0^2, 3 * 2.0^2, 3 * 3.0^2, 3 * 2.0^2, (cos(3.0) * 3.0 - sin(3.0)) / 3.0^2]

@testset "a program compiled by juliac --trim" begin
    if !CAN_RUN
        @info "juliac test skipped" VERSION isfile(JULIAC) Sys.iswindows() c_compiler()
    else
        mktempdir() do dir
            app = joinpath(dir, "app.jl")
            write(app, APP)
            exe = joinpath(dir, "app")
            build = addenv(
                `$(driver_julia_cmd()) --project=$(Base.active_project()) --startup-file=no $JULIAC --output-exe $exe --experimental --trim=safe --compile-ccallable $app`,
                "JULIA_LOAD_PATH" => join([Base.active_project(), "@stdlib"], Sys.iswindows() ? ";" : ":"),
                build_env()...,
            )
            err = IOBuffer()
            built = success(pipeline(build; stdout = devnull, stderr = err))
            built || println(stderr, "juliac failed:\n", String(take!(err)))
            @test built
            if built
                out = IOBuffer()
                ran = success(pipeline(`$exe`; stdout = out, stderr = err))
                ran || println(stderr, "the trimmed program failed:\n", String(take!(err)))
                @test ran
                if ran
                    values = parse.(Float64, split(String(take!(out))))
                    @test length(values) == length(EXPECTED)
                    if length(values) == length(EXPECTED)
                        @test values ≈ EXPECTED
                    end
                end
            end
        end
    end
end
