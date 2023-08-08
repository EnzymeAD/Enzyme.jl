# HACK: work around Pkg.jl#2500
if VERSION < v"1.8-"
test_project = Base.active_project()
preferences_file = joinpath(dirname(@__DIR__), "LocalPreferences.toml")
test_preferences_file = joinpath(dirname(test_project), "LocalPreferences.toml")
if isfile(preferences_file) && !isfile(test_preferences_file)
    cp(preferences_file, test_preferences_file)
end
end

using Enzyme
using Test

using Enzyme_jll
@info "Testing against" Enzyme_jll.libEnzyme

f(x) = hypot(x, 2x)
@test autodiff(Reverse, f, Active, Active(2.0))[1][1] == sqrt(5)
