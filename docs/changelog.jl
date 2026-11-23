# Update the link list at the end of CHANGELOG.md.
# Usage: julia --project=docs docs/changelog.jl

using Changelog

Changelog.generate(
    Changelog.CommonMark(),
    joinpath(@__DIR__, "..", "CHANGELOG.md");
    repo = "EnzymeAD/Enzyme.jl",
)
