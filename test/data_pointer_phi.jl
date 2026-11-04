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

# `y` comes out of a vector of arrays and the index depends on its length, both read before
# `p[rows]`, a call Julia does not inline and that may write memory. So the reverse pass has to
# keep `y` and the index, and the cheapest value it could keep instead is the address of `y[k]`.
# A derived pointer cannot be kept, and Enzyme used to fall back to `y`'s data pointer and fail
# with "Illegal cached pointer". It must keep `y` itself and reload the data pointer from it.
@inline function array_of_arrays_read(p, arrays, i, rows)
    y = arrays[1]
    n = length(y)
    w = p[rows]
    k = n == 1 ? 1 : i
    return y, k, w
end

function array_of_arrays_sum(p, arrays, i, rows)
    y, k, w = array_of_arrays_read(p, arrays, i, rows)
    s = 0.0
    @inbounds for j in eachindex(w)
        s += (y[k] - w[j])^2
    end
    return s
end

# The same with the terms in an array, the shape of a broadcast with a fresh output.
function array_of_arrays_fresh(p, arrays, i, rows)
    y, k, w = array_of_arrays_read(p, arrays, i, rows)
    out = Vector{Float64}(undef, length(w))
    @inbounds for j in eachindex(out)
        out[j] = (y[k] - w[j])^2
    end
    return sum(out)
end

# Each of these changes what a reload of `y`'s data would read. The derivative needs the values
# the sum read, so they must not be reloaded from the new state.
function array_of_arrays_replaced(p, arrays, i, rows)
    s = array_of_arrays_sum(p, arrays, i, rows)
    arrays[1] = arrays[2]
    return s
end

function array_of_arrays_replaced_inline(p, arrays, i, rows)
    y, k, w = array_of_arrays_read(p, arrays, i, rows)
    s = 0.0
    @inbounds for j in eachindex(w)
        s += (y[k] - w[j])^2
    end
    arrays[1] = arrays[2]
    return s
end

function array_of_arrays_overwritten(p, arrays, i, rows)
    y, k, w = array_of_arrays_read(p, arrays, i, rows)
    s = 0.0
    @inbounds for j in eachindex(w)
        s += (y[k] - w[j])^2
    end
    y[k] = -100.0
    return s
end

function array_of_arrays_grown(p, arrays, i, rows)
    y, k, w = array_of_arrays_read(p, arrays, i, rows)
    s = 0.0
    @inbounds for j in eachindex(w)
        s += (y[k] - w[j])^2
    end
    append!(y, zeros(1000))
    y[k] = -100.0
    return s
end

@testset "Data pointer of an array read out of a vector of arrays" begin
    p = [0.3, -0.2, 0.5]
    rows = [1, 3]
    data() = [[1.0, 2.0, 3.0], [4.0]]
    expected = [-2 * (2.0 - 0.3), 0.0, -2 * (2.0 - 0.5)]
    for f in (
            array_of_arrays_sum, array_of_arrays_fresh, array_of_arrays_replaced,
            array_of_arrays_replaced_inline, array_of_arrays_overwritten, array_of_arrays_grown,
        )
        d = zero(p)
        res = Enzyme.autodiff(ReverseWithPrimal, Const(f), Active, Duplicated(p, d), Const(data()), Const(2), Const(rows))
        @test d ≈ expected
        @test res[2] ≈ (2.0 - 0.3)^2 + (2.0 - 0.5)^2

        d = zero(p)
        Enzyme.autodiff(set_runtime_activity(Reverse), Const(f), Active, Duplicated(p, d), Const(data()), Const(2), Const(rows))
        @test d ≈ expected
    end

    # A data vector of length one is stretched: the index does not depend on `i`.
    d = zero(p)
    Enzyme.autodiff(Reverse, Const(array_of_arrays_sum), Active, Duplicated(p, d), Const([[1.5], [4.0]]), Const(2), Const(rows))
    @test d ≈ [-2 * (1.5 - 0.3), 0.0, -2 * (1.5 - 0.5)]

    # The reverse pass runs later, from a tape.
    fwd, rev = Enzyme.autodiff_thunk(
        ReverseSplitNoPrimal, Const{typeof(array_of_arrays_sum)}, Active,
        Duplicated{typeof(p)}, Const{typeof(data())}, Const{Int}, Const{typeof(rows)},
    )
    d = zero(p)
    args = (Duplicated(p, d), Const(data()), Const(2), Const(rows))
    tape, = fwd(Const(array_of_arrays_sum), args...)
    rev(Const(array_of_arrays_sum), args..., 1.0, tape)
    @test d ≈ expected

    # Second derivatives go through the same reverse pass.
    gradient_of(f, q, arrays) = (g = zero(q); Enzyme.autodiff(Reverse, Const(f), Active, Duplicated(q, g), Const(arrays), Const(2), Const(rows)); g)
    v = [1.0, 0.5, -2.0]
    for f in (array_of_arrays_sum, array_of_arrays_fresh)
        hv = only(Enzyme.autodiff(Forward, gradient_of, Const(f), Duplicated(p, v), Const(data())))
        @test hv ≈ [2.0, 0.0, -4.0]
    end
end
