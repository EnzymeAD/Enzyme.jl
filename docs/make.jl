pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..")) # add Enzyme to environment stack

using Enzyme
using Documenter

DocMeta.setdocmeta!(Enzyme, :DocTestSetup, :(using Enzyme); recursive=true)

makedocs(;
    modules=[Enzyme],
    authors="William Moses <wmoses@mit.edu>, Valentin Churavy <vchuravy@mit.edu>",
    repo="https://github.com/wsmoses/Enzyme.jl/blob/{commit}{path}#{line}",
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
    repo="github.com/wsmoses/Enzyme.jl",
    devbranch = "master",
    push_preview = true,
)
