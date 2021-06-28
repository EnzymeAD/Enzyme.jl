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
        canonical="https://wsmoses.github.io/Enzyme.jl",
        assets=String[],
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
    forcepush = true,
    push_preview = true,
)
