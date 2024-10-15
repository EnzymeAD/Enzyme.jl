using Enzyme, Test

using StaticArrays

@testset "Gradient & StaticArrays" begin

    x = @SArray [5.0 0.0 6.0]
    dx = Enzyme.gradient(Reverse, prod, x)[1]
    @test dx isa SArray
    @test dx ≈ [0 30 0]

    x = @SVector [1.0, 2.0, 3.0]
    y = onehot(x)
    # this should be a very specific type of SArray, but there
    # is a bizarre issue with older julia versions where it can be MArray
    @test eltype(y) <: StaticVector
    @test length(y) == 3
    @test y[1] == [1.0, 0.0, 0.0]
    @test y[2] == [0.0, 1.0, 0.0]
    @test y[3] == [0.0, 0.0, 1.0]

    y = onehot(x, 2, 3)
    @test eltype(y) <: StaticVector
    @test length(y) == 2
    @test y[1] == [0.0, 1.0, 0.0]
    @test y[2] == [0.0, 0.0, 1.0]

    x = @SArray [5.0 0.0 6.0]
    dx = Enzyme.gradient(Forward, prod, x)[1]
    @test dx isa SArray
    @test dx ≈ [0 30 0]

    f0 = x -> sum(2*x)
    f1 = x -> @SVector Float64[x[2], 2*x[2]]
    f2 = x -> @SMatrix Float64[x[2] x[1]; 2*x[2] 2*x[1]]

    x = @SVector Float64[1, 2]

    @inferred gradient(Forward, f0, x)
    dx = gradient(Forward, f0, x)[1]
    @test dx isa SVector
    @test dx == [2.0, 2.0]  # test to make sure conversion works
    @test gradient(Forward, f1, x)[1] isa SMatrix
    @test gradient(Forward, f1, x)[1] == [0 1.0; 0 2.0]
    @test Enzyme.jacobian(Forward, f2, x)[1] isa SArray
    @test Enzyme.jacobian(Forward, f2, x)[1] == reshape(Float64[0,0,1,2,1,2,0,0], (2,2,2))

    x = @SMatrix Float64[1 2; 3 4]

    @inferred gradient(Forward, f0, x)
    dx = gradient(Forward, f0, x)[1]
    @test dx isa SMatrix
    @test dx == fill(2.0, (2,2))
    @test gradient(Forward, f1, x)[1] isa SArray
    @test gradient(Forward, f1, x)[1] == reshape(Float64[0,0,1,2,0,0,0,0], (2,2,2))
    @test Enzyme.jacobian(Forward, f2, x)[1] isa SArray
    @test Enzyme.jacobian(Forward, f2, x)[1] == reshape(
        Float64[0,0,1,2,1,2,0,0,0,0,0,0,0,0,0,0], (2,2,2,2),
    )

    x = @SVector Float64[1, 2]

    @inferred gradient(Reverse, f0, x)
    dx = gradient(Reverse, f0, x)[1]
    @test dx isa SVector
    @test dx == [2.0, 2.0]  # test to make sure conversion works
    @test_broken gradient(Reverse, f1, x)[1] isa SMatrix
    @test_broken gradient(Reverse, f1, x)[1] == [0 1.0; 0 2.0]
    @test_broken Enzyme.jacobian(Reverse, f2, x)[1] isa SArray
    @test_broken Enzyme.jacobian(Reverse, f2, x)[1] == reshape(Float64[0,0,1,2,1,2,0,0], (2,2,2))

    x = @SMatrix Float64[1 2; 3 4]

    @test_broken gradient(Reverse, f1, x)[1] isa SArray
    @test_broken gradient(Reverse, f1, x)[1] == reshape(Float64[0,0,1,2,0,0,0,0], (2,2,2))
    @test_broken Enzyme.jacobian(Reverse, f2, x)[1] isa SArray
    @test_broken Enzyme.jacobian(Reverse, f2, x)[1] == reshape(
        Float64[0,0,1,2,1,2,0,0,0,0,0,0,0,0,0,0], (2,2,2,2),
    )
end

function unstable_fun(A0)
    A = 'N' in ('H', 'h', 'S', 's') ? wrap(A0) : A0
    (@inbounds A[1])::eltype(A0)
end
@testset "Type unstable static array index" begin
    inp = ones(SVector{2, Float64})
    res = Enzyme.gradient(Enzyme.Reverse, unstable_fun, inp)[1]
    @test res ≈ [1.0, 0.0]
    res = Enzyme.gradient(Enzyme.Forward, unstable_fun, inp)[1]
    @test res ≈ [1.0, 0.0]
end