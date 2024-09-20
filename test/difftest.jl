using Enzyme, Test

@isdefined(UTILS_INCLUDE) || include("utils.jl")


@testset "DiffTest" begin
    include("DiffTests.jl")

    n = 1 + rand()
    x, y = 1 .+ rand(5, 5), 1 .+ rand(5)
    A, B = 1 .+ rand(5, 5), 1 .+ rand(5, 5)

    # f returns Number
    @testset "Number to Number" for f in DiffTests.NUMBER_TO_NUMBER_FUNCS
        test_scalar(f, n; rtol=1e-6, atol=1e-6)
    end

    @testset "Vector to Number" for f in DiffTests.VECTOR_TO_NUMBER_FUNCS
        test_matrix_to_number(f, y; rtol=1e-6, atol=1e-6)
    end

    @testset "Matrix to Number" for f in DiffTests.MATRIX_TO_NUMBER_FUNCS
        test_matrix_to_number(f, x; rtol=1e-6, atol=1e-6)
    end

    # TODO(vchuravy/wsmoses): Enable these tests
    # for f in DiffTests.TERNARY_MATRIX_TO_NUMBER_FUNCS
    #     @test isa(f(A, B, x), Number)
    # end

    # # f returns Array

    # for f in DiffTests.NUMBER_TO_ARRAY_FUNCS
    #     @test isa(f(n), Array)
    # end

    # for f in DiffTests.ARRAY_TO_ARRAY_FUNCS
    #     @test isa(f(A), Array)
    #     @test isa(f(y), Array)
    # end

    # for f in DiffTests.MATRIX_TO_MATRIX_FUNCS
    #     @test isa(f(A), Array)
    # end

    # for f in DiffTests.BINARY_MATRIX_TO_MATRIX_FUNCS
    #     @test isa(f(A, B), Array)
    # end

    # # f! returns Nothing

    # for f! in DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS
    #     @test isa(f!(y, x), Nothing)
    # end

    # for f! in DiffTests.INPLACE_NUMBER_TO_ARRAY_FUNCS
    #     @test isa(f!(y, n), Nothing)
    # end

end


