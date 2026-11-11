using Enzyme, SpecialFunctions, Test

# `loggamma` may throw, so LLVM assumes the `@noinline` call may write memory and reloads
# the data pointer of `y` after it. GVN then carries that pointer around the loop as a phi of
# the load before the loop and the reload after the call. On Julia 1.10 Enzyme used to cache
# that pointer in the reverse pass and fail with "Illegal cached pointer".
@noinline data_pointer_phi_call(x::Float64) = loggamma(x)

function data_pointer_phi_sum(eta::Vector{Float64}, y::AbstractVector)
    s = 0.0
    for k in eachindex(eta)
        @inbounds begin
            r = exp(eta[k])
            a = y[k] > 0 ? -data_pointer_phi_call(y[k] + 1.0) : 0.0
            b = y[k] >= 0 ? (y[k] == 0 ? -r : y[k] * eta[k] - r) : -Inf
        end
        s += a + b
    end
    return s
end

# The same with a data vector of length one that is stretched over `eta`.
function data_pointer_phi_stretched(eta::Vector{Float64}, y::AbstractVector)
    s = 0.0
    stretch = length(y) == 1
    for k in eachindex(eta)
        j = stretch ? 1 : k
        @inbounds begin
            r = exp(eta[k])
            a = y[j] > 0 ? -data_pointer_phi_call(y[j] + 1.0) : 0.0
            b = y[j] >= 0 ? (y[j] == 0 ? -r : y[j] * eta[k] - r) : -Inf
        end
        s += a + b
    end
    return s
end

@testset "Data pointer reloaded after a call" begin
    eta = [0.1, -0.2, 0.3, 0.4, -0.5]
    for y in ([0, 2, 1, 4, 3], [0.0, 2.0, 1.0, 4.0, 3.0])
        d = zero(eta)
        Enzyme.autodiff(Reverse, data_pointer_phi_sum, Active, Duplicated(eta, d), Const(y))
        @test d ≈ y .- exp.(eta)
    end
    for y in ([2], [0, 2, 1, 4, 3])
        d = zero(eta)
        Enzyme.autodiff(Reverse, data_pointer_phi_stretched, Active, Duplicated(eta, d), Const(y))
        yy = length(y) == 1 ? fill(y[1], length(eta)) : y
        @test d ≈ yy .- exp.(eta)
    end
    # The same data, now differentiated.
    y = [0.0, 2.0, 1.0, 4.0, 3.0]
    d = zero(eta)
    dy = zero(y)
    Enzyme.autodiff(Reverse, data_pointer_phi_sum, Active, Duplicated(eta, d), Duplicated(y, dy))
    @test d ≈ y .- exp.(eta)
    @test dy ≈ [i == 1 ? 0.0 : eta[i] - digamma(y[i] + 1) for i in eachindex(y)]
end
