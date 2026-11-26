# Update the link lists at the end of the CHANGELOG.md files.
# Usage: julia --project=docs docs/changelog.jl

using Changelog

const ROOT = joinpath(@__DIR__, "..")

for file in (
        joinpath(ROOT, "CHANGELOG.md"),
        joinpath(ROOT, "lib", "EnzymeCore", "CHANGELOG.md"),
        joinpath(ROOT, "lib", "EnzymeTestUtils", "CHANGELOG.md"),
    )
    Changelog.generate(Changelog.CommonMark(), file; repo = "EnzymeAD/Enzyme.jl")
end
