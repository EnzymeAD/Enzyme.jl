module EnzymeTestUtils

using ConstructionBase
using Enzyme
using EnzymeCore: Annotation
using FiniteDifferences
using Random
using Test

export test_forward, test_reverse, all_or_no_batch

include("finite_difference_calls.jl")
include("generate_tangent.jl")
include("testers.jl")

function test_approx(x::Number, y::Number; kwargs...)
    @test isapprox(x, y; kwargs...)
end
function test_approx(x::AbstractArray{<:Number}, y::AbstractArray{<:Number}; kwargs...)
    @test isapprox(x, y; kwargs...)
end
function test_approx(x, y; kwargs...)
    for k in fieldnames(typeof(x))
        test_approx(getfield(x, k), getfield(y, k); kwargs...)
    end
end

"""
    all_or_no_batch(activities...) -> Bool

Returns `true` if `activities` are compatible in terms of batched activities.

When a test set loops over many activities, some of which may be `BatchedDuplicated` or
`BatchedDuplicatedNoNeed`, this is useful for skipping those combinations that are
incompatible and will raise errors.
"""
function all_or_no_batch(activities...)
    no_batch = !any(activities) do T
        T <: Union{BatchDuplicated,BatchDuplicatedNoNeed}
    end
    all_batch_or_const = all(activities) do T
        T <: Union{BatchDuplicated,BatchDuplicatedNoNeed,Const}
    end
    return all_batch_or_const || no_batch
end

end  # module
