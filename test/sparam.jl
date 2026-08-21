using Enzyme, Test

dual_type(c::NTuple{N, Float64}) where {N} = Val{N}

function loss(x::Vector{Float64})
    t = ntuple(_ -> 0.0, length(x))   # NTuple{_A, Float64} where _A — non-inferable N
    return sum(x) + (dual_type(t) === Val{2} ? 0.0 : 1.0)
end

@testset "broadcast" begin
    res = Enzyme.gradient(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        Enzyme.Const(loss),
        [1.0, 2.0]
    )[1]
    @test res ≈ [1.0, 1.0]
end