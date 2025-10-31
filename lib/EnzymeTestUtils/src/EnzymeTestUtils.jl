module EnzymeTestUtils

using ConstructionBase: ConstructionBase
using Enzyme
using EnzymeCore: Annotation
using FiniteDifferences: FiniteDifferences
using Random: Random
using Test

export test_forward, test_reverse, test_rewind, are_activities_compatible

include("output_control.jl")
include("to_vec.jl")
include("test_approx.jl")
include("compatible_activities.jl")
include("finite_difference_calls.jl")
include("generate_tangent.jl")
include("test_forward.jl")
include("test_reverse.jl")
include("test_rewind.jl")

end  # module
