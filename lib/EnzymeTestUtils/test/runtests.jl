using EnzymeTestUtils
using Random
using Test

Random.seed!(0)

@testset "EnzymeTestUtils.jl" begin
    include("helpers.jl")
    include("test_approx.jl")
    include("compatible_activities.jl")
    include("to_vec.jl")
    include("generate_tangent.jl")
    include("test_forward.jl")
    include("test_reverse.jl")
end
