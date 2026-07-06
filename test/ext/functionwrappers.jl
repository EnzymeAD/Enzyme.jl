using Enzyme, Test
using FunctionWrappers: FunctionWrapper

function fw_test_f!(out, u, p)
    out[1] = u[1] - p[1]
    return nothing
end

function wrapper_loss(p)
    out = [0.0]
    u = [1.0]
    fw = FunctionWrapper{Nothing, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}(fw_test_f!)
    fw(out, u, p)
    return out[1]
end

@testset "FunctionWrappers" begin
    res = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), wrapper_loss, [2.0])
    @test res[1] ≈ [-1.0]
end
