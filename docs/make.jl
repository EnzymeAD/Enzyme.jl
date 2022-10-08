pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..")) # add Enzyme to environment stack
pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", "lib")) # add EnzymeCore to environment stack

using Enzyme
using EnzymeCore
using Documenter

DocMeta.setdocmeta!(Enzyme, :DocTestSetup, :(using Enzyme); recursive=true)

# Generate examples

using Literate

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR = joinpath(@__DIR__, "src/generated")

examples = Pair{String,String}[
    "Box model" => "box"
]

for (_, name) in examples
    example_filepath = joinpath(EXAMPLES_DIR, string(name, ".jl"))
    Literate.markdown(example_filepath, OUTPUT_DIR, documenter = true)
end

examples = [title => joinpath("generated", string(name, ".md")) for (title, name) in examples]

makedocs(;
    modules=[Enzyme, EnzymeCore],
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
    ),
    pages = [
        "Home" => "index.md",
        "Examples" => examples,
        "API" => "api.md",
        "Implementing pullbacks" => "pullbacks.md",
        "For developers" => "dev_docs.md",
        "Internal API" => "internal_api.md",
    ],
    doctest = true,
    linkcheck = true,
    strict = true,
)

deploydocs(;
    repo="github.com/EnzymeAD/Enzyme.jl",
    devbranch = "main",
    push_preview = true,
)
