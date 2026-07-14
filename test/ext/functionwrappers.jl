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

# The FunctionWrapper type visible to abstract interpretation may be a
# UnionAll when the wrapper comes from a container with a partially known
# element type. Compilation must not fail on such calls, even if they are
# never reached at runtime:
function partially_known_wrapper(x::Float64, take_wrapper::Bool, fws::Vector{FunctionWrapper{Float64}})
    if take_wrapper
        return fws[1](x)::Float64
    else
        return sin(x)
    end
end

@testset "FunctionWrappers with partially known wrapper type" begin
    fws = Vector{FunctionWrapper{Float64}}(undef, 1)
    fws[1] = FunctionWrapper{Float64,Tuple{Float64}}(cos)
    res = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse), partially_known_wrapper,
        Enzyme.Active, Enzyme.Active(0.5), Enzyme.Const(false), Enzyme.Const(fws)
    )
    @test res[1][1] ≈ cos(0.5)
end
