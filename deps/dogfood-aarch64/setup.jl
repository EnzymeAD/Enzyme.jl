# julia --startup-file=no setup.jl ENV_DIR ENZYME_SOURCE RESOLVER_PROGRAM
using Pkg, TOML
length(ARGS) == 3 || error("Expected ENV_DIR ENZYME_SOURCE RESOLVER_PROGRAM")
env_dir, enzyme_dir, resolver = abspath.(ARGS)
expected = ENV["ENZYME_EXPECT_BACKEND"]
occursin(r"^[0-9a-f]{40}$", expected) || error("ENZYME_EXPECT_BACKEND must be a 40-hex commit")
TOML.parsefile(joinpath(enzyme_dir, "deps", "dogfood_libenzyme.toml"))["commit"] == expected || error("Unexpected backend declaration")
isfile(resolver) || error("Missing maintained resolver: $resolver")
Pkg.activate(env_dir)
Pkg.develop([PackageSpec(path = joinpath(enzyme_dir, "lib", "EnzymeCore")), PackageSpec(path = enzyme_dir)])
step = read(resolver, String)
isempty(strip(step)) && error("Resolver generated no program")
include_string(Main, step, "dogfood_libenzyme")
Pkg.instantiate(; allow_autoprecomp = false)
println("SETUP_COMPLETE ", env_dir)
