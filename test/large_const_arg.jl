using Enzyme, Test

struct LargeConstArg{T}
    data::T
    scale::Vector{Float64}
end

@noinline large_const_unused(x, _) = sum(x)
@noinline large_const_used(x, c) = sum(x) * (c.data[1] + c.scale[1])

function check_large_const_arg(::Val{N}) where {N}
    c = LargeConstArg(ntuple(Returns(1.0), Val(N)), [2.0])
    @test sizeof(typeof(c)) == 8N + 8

    x = [1.0, 2.0, 3.0]
    dx = zeros(3)
    result = autodiff(ReverseWithPrimal, large_const_unused, Active, Duplicated(x, dx), Const(c))
    @test result[2] == 6.0
    @test dx == ones(3)

    fill!(dx, 0.0)
    result = autodiff(ReverseWithPrimal, large_const_used, Active, Duplicated(x, dx), Const(c))
    @test result[2] == 18.0
    return @test dx == fill(3.0, 3)
end

@testset "Large constant argument convention" begin
    # Julia 1.10 gives by-reference aggregates beyond 32 KiB an empty
    # struct pointee. The smaller case exercises the ordinary layout.
    check_large_const_arg(Val(4095))
    check_large_const_arg(Val(4096))
end
