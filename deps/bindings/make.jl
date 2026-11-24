# Generate the low-level ccall wrappers in src/api/libEnzyme_h.jl from Enzyme's C API header.
#
# Usage:
#
#     julia --project=deps/bindings deps/bindings/make.jl [path/to/Enzyme]
#
# `path/to/Enzyme` is a checkout of https://github.com/EnzymeAD/Enzyme. Without it, the
# Enzyme release matching the Enzyme_jll compat bound in Enzyme.jl's Project.toml is cloned.

using Pkg: Pkg
Pkg.instantiate()

using Clang.Generators
using Clang: CLLinkageSpec, children
using LLVM_full_jll: LLVM_full_jll
using Runic: Runic
using TOML: TOML

# CApi.h wraps its declarations in an `extern "C"` block, which Clang.jl does not descend into.
function Generators.collect_top_level_nodes!(
        nodes::Vector{ExprNode}, cursor::CLLinkageSpec, options
    )
    for child in children(cursor)
        Generators.collect_top_level_nodes!(nodes, child, options)
    end
    return nodes
end

function enzyme_source(args)
    isempty(args) || return abspath(only(args))
    project = TOML.parsefile(joinpath(@__DIR__, "..", "..", "Project.toml"))
    version = project["compat"]["Enzyme_jll"]
    dir = mktempdir()
    @info "Cloning Enzyme v$version"
    run(`git clone --quiet --depth 1 --branch v$version https://github.com/EnzymeAD/Enzyme $dir`)
    return dir
end

function llvm_includedir()
    return LLVM_full_jll.llvm_config() do exe
        readchomp(`$exe --includedir`)
    end
end

function main(args)
    header = joinpath(enzyme_source(args), "enzyme", "Enzyme", "CApi.h")
    isfile(header) || error("$header not found")

    options = load_options(joinpath(@__DIR__, "wrap.toml"))
    output = normpath(joinpath(@__DIR__, options["general"]["output_file_path"]))
    options["general"]["output_file_path"] = output
    options["general"]["prologue_file_path"] =
        joinpath(@__DIR__, options["general"]["prologue_file_path"])

    # CApi.h forward-declares C++ classes, so it has to be parsed as C++.
    clang_args = get_default_args(; is_cxx = true)
    append!(clang_args, ["-x", "c++", "-isystem", llvm_includedir()])

    ctx = create_context([header], clang_args, options)
    build!(ctx)

    Runic.main(["--inplace", output])
    return
end

isinteractive() || main(ARGS)
