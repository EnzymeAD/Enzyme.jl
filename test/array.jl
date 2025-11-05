using Enzyme, Test

function ptrcopy(B, A)
    @static if VERSION < v"1.11"
        Base.unsafe_copyto!(B, 1, A, 1, 2)
    else
        Base.unsafe_copyto!(B.ref, A.ref, 2)
    end
    return nothing
end

@testset "Array of Pointer Copy" begin
    A = [[2.7, 3.1], [4.7, 5.6]]
    dA1 = [1.1, 4.3]
    dA2 = [17.2, 0.26]
    dA = [dA1, dA2]

    B = [[2.0, 4.0], [7.0, 11.0]]
    dB = Enzyme.make_zero(B)

    Enzyme.autodiff(set_runtime_activity(Reverse), ptrcopy, Duplicated(B, dB), Duplicated(A, dA))

    @test dB[1] === dA1
    @test dB[2] === dA2
end

function unsafe_wrap_test(a, i, x)
    GC.@preserve a begin
        ptr = pointer(a)
        b = Base.unsafe_wrap(Array, ptr, length(a))
        b[i] = x
    end
    return a[i]
end

mutable struct Container
    u::Array{Float64, 2}
    neighbor_ids::Array{Int, 2}
    # internal `resize!`able storage
    _u::Vector{Float64}
    _neighbor_ids::Vector{Int}
end

function Base.resize!(c::Container, capacity)
    resize!(c._u, 2 * capacity)
    c.u = unsafe_wrap(
        Array, pointer(c._u),
        (2, capacity)
    )

    resize!(c._neighbor_ids, 2 * capacity)
    c.neighbor_ids = unsafe_wrap(
        Array, pointer(c._neighbor_ids),
        (2, capacity)
    )
    return nothing
end

function Container(capacity::Integer)
    # Initialize fields with defaults
    _u = fill(NaN, 2 * capacity)
    u = unsafe_wrap(
        Array, pointer(_u),
        (2, capacity)
    )

    _neighbor_ids = fill(typemin(Int), 2 * capacity)
    neighbor_ids = unsafe_wrap(
        Array, pointer(_neighbor_ids),
        (2, capacity)
    )
    return Container(u, neighbor_ids, _u, _neighbor_ids)
end

function unsafe_wrap_test2(x)
    c = Container(0)
    resize!(c, 10)
    c.u[:] .= x
    resize!(c, 3)
    return prod(c.u)
end


@testset "Unsafe wrap" begin
    # TODO test for batch and reverse
    autodiff(Forward, unsafe_wrap_test, Duplicated(zeros(1), zeros(1)), Const(1), Duplicated(1.0, 2.0))

    autodiff(Forward, unsafe_wrap_test2, Duplicated(3.0, 1.0))
end
