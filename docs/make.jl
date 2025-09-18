using Enzyme
using EnzymeCore
using EnzymeTestUtils
using Documenter

DocMeta.setdocmeta!(Enzyme, :DocTestSetup, :(using Enzyme); recursive=true)
DocMeta.setdocmeta!(EnzymeCore, :DocTestSetup, :(using EnzymeCore); recursive=true)
DocMeta.setdocmeta!(EnzymeTestUtils, :DocTestSetup, :(using EnzymeTestUtils); recursive=true)
@eval EnzymeCore begin
    const Enzyme = $(Enzyme)
end

# Generate examples

using Literate

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR = joinpath(@__DIR__, "src/generated")

examples = Pair{String,String}[
    "Basics" => "autodiff"
    "Box model" => "box"
    "Custom rules" => "custom_rule"
]

for (_, name) in examples
    example_filepath = joinpath(EXAMPLES_DIR, string(name, ".jl"))
    Literate.markdown(example_filepath, OUTPUT_DIR, documenter = true)
end

examples = [title => joinpath("generated", string(name, ".md")) for (title, name) in examples]

# Generate Pluto notebooks

using PlutoStaticHTML
const NOTEBOOK_DIR = joinpath(@__DIR__, "src", "notebooks")

"""
    build()

Run all Pluto notebooks (".jl" files) in `NOTEBOOK_DIR`.
"""
function build()
    println("Building notebooks in $NOTEBOOK_DIR")
    oopts = OutputOptions(; append_build_context = false)
    output_format = documenter_output
    bopts = BuildOptions(NOTEBOOK_DIR; output_format)
    build_notebooks(bopts, oopts)
    return nothing
end

# Build the notebooks; defaults to true.
if get(ENV, "BUILD_DOCS_NOTEBOOKS", "true") == "true"
    build()
end


makedocs(;
    modules=[Enzyme, EnzymeCore, EnzymeTestUtils],
    authors="William Moses <wmoses@mit.edu>, Valentin Churavy <vchuravy@mit.edu>",
    repo="https://github.com/EnzymeAD/Enzyme.jl/blob/{commit}{path}#{line}",
    sitename="Enzyme.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://enzyme.mit.edu/julia/",
        assets = [
            asset("https://plausible.io/js/plausible.js",
                    class=:js,
                    attributes=Dict(Symbol("data-domain") => "enzyme.mit.edu", :defer => "")
                )
	    ],
        mathengine = MathJax3(),
        size_threshold = 10_000_000
    ),
    pages = [
        "Home" => "index.md",
        "Examples" => examples,
        "Notebooks" => [
            "Ignore derivatives" => "notebooks/ignore_derivatives.md",
        ],
        "FAQ" => "faq.md",
        "API reference" => "api.md",
        "Advanced" => [
            "For developers" => "dev_docs.md",
            "Internal API" => "internal_api.md",
        ]
    ],
    doctest = true,
)

deploydocs(;
    repo="github.com/EnzymeAD/Enzyme.jl",
    devbranch = "main",
    push_preview = true,
)
