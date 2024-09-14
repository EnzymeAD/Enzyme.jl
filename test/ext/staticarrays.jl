using Enzyme, Test
using StaticArrays

using Enzyme: gradient


@testset "StaticArrays ext" begin
    @testset "basics" begin
        f = x -> @SVector([x[1],0.0])

        x = @SVector ones(2)

        df = values(autodiff(Enzyme.Forward, f, BatchDuplicatedNoNeed, BatchDuplicated(x, Enzyme.onehot(x)))[1])
        @test df[1] ≈ [1.0, 0.0]
        @test df[2] ≈ zeros(2)
    end

    @testset "gradient" begin
        x = @SArray [5.0 0.0 6.0]
        dx = gradient(Reverse, prod, x)
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
        dx = gradient(Forward, prod, x)
        @test dx[1] ≈ 0
        @test dx[2] ≈ 30
        @test dx[3] ≈ 0
    end
end
