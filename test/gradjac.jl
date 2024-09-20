using Enzyme, Test
using LinearAlgebra, SparseArrays, StaticArrays

@isdefined(UTILS_INCLUDE) || include("utils.jl")


@testset "Gradient & NamedTuples" begin
    xy = (x = [1.0, 2.0], y = [3.0, 4.0])
    grad = Enzyme.gradient(Reverse, z -> sum(z.x .* z.y), xy)[1]
    @test grad == (x = [3.0, 4.0], y = [1.0, 2.0])

    xp = (x = [1.0, 2.0], p = 3)  # 3::Int is non-diff
    grad = Enzyme.gradient(Reverse, z -> sum(z.x .^ z.p), xp)[1]
    @test grad.x == [3.0, 12.0]

    xp2 = (x = [1.0, 2.0], p = 3.0)  # mixed activity
    grad = Enzyme.gradient(Reverse, z -> sum(z.x .^ z.p), xp2)[1]
    @test grad.x == [3.0, 12.0]
    @test grad.p ≈ 5.545177444479562

    xy = (x = [1.0, 2.0], y = [3, 4])  # y is non-diff
    grad = Enzyme.gradient(Reverse, z -> sum(z.x .* z.y), xy)[1]
    @test grad.x == [3.0, 4.0]
    @test grad.y === xy.y  # make_zero did not copy this

    grad = Enzyme.gradient(Reverse, z -> (z.x * z.y), (x=5.0, y=6.0))[1]
    @test grad == (x = 6.0, y = 5.0)

    grad = Enzyme.gradient(Reverse, abs2, 7.0)[1]
    @test grad == 14.0
end

@testset "Gradient & SparseArrays / StaticArrays" begin
    x = sparse([5.0, 0.0, 6.0])
    dx = Enzyme.gradient(Reverse, sum, x)[1]
    @test dx isa SparseVector
    @test dx ≈ [1, 0, 1]

    x = sparse([5.0 0.0 6.0])
    dx = Enzyme.gradient(Reverse, sum, x)[1]
    @test dx isa SparseMatrixCSC
    @test dx ≈ [1 0 1]

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
    @test dx[1] ≈ 0
    @test dx[2] ≈ 30
    @test dx[3] ≈ 0
end


# these are used in gradient and jacobian tests
struct InpStruct
    i1::Float64
    i2::Float64
    i3::Float64
end
struct OutStruct
    i1::Float64
    i2::Float64
    i3::Float64
end

for A ∈ (:InpStruct, :OutStruct)
    @eval (≃)(a::$A, b::$A) = (a.i1 ≃ b.i1) && (a.i2 ≃ b.i2) && (a.i3 ≃ b.i3)
    @eval function (≃)(a::AbstractArray{<:$A}, b::AbstractArray{<:$A})
        size(a) == size(b) || return false
        all(xy -> xy[1] ≃ xy[2], zip(a, b))
    end
end


#NOTE: this is needed because of problems with hvcat on 1.10 and something inexplicable on 1.6
# suffice it to say it's not good that this is required, please remove when possible
mkarray(sz, args...) = reshape(vcat(args...), sz)

@testset "Gradient and Jacobian Outputs" begin

    scalar = 3.0

    @testset "∂ scalar / ∂ scalar" begin
        @test Enzyme.gradient(Enzyme.Forward, x -> x^2, scalar)[1] ≈ 6.0
        @test Enzyme.gradient(Enzyme.Reverse, x -> x^2, scalar)[1] ≈ 6.0
        @test Enzyme.jacobian(Enzyme.Forward, x -> x^2, scalar)[1] ≈ 6.0
        @test Enzyme.jacobian(Enzyme.Reverse, x -> x^2, scalar)[1] ≈ 6.0
        @test Enzyme.gradient(Enzyme.Forward, x -> 2*x, scalar)[1] ≈ 2.0
        @test Enzyme.gradient(Enzyme.Reverse, x -> 2*x, scalar)[1] ≈ 2.0
        @test Enzyme.jacobian(Enzyme.Forward, x -> 2*x, scalar)[1] ≈ 2.0
        @test Enzyme.jacobian(Enzyme.Reverse, x -> 2*x, scalar)[1] ≈ 2.0
    end

    @testset "∂ vector / ∂ scalar" begin
        @test Enzyme.gradient(Enzyme.Forward, x -> [2*x, x^2], scalar)[1] ≈ [2.0, 6.0]
        @test_broken Enzyme.gradient(Enzyme.Reverse, x -> [2*x, x^2], scalar)[1] ≈ [2.0, 6.0]
    
        @test Enzyme.jacobian(Enzyme.Forward, x -> [2*x, x^2], scalar)[1] ≈ [2.0, 6.0]
        @test Enzyme.jacobian(Enzyme.Reverse, x -> [2*x, x^2], scalar)[1] ≈ [2.0, 6.0]
    end

    @testset "∂ tuple / ∂ scalar" begin
        @test Enzyme.gradient(Enzyme.Forward, x -> (2*x, x^2), scalar)[1] ≃ (2.0, 6.0)
        @test_broken Enzyme.gradient(Enzyme.Reverse, x -> (2*x, x^2), scalar)[1] ≈ [2.0, 6.0]
    
        @test Enzyme.jacobian(Enzyme.Forward, x -> (2*x, x^2), scalar)[1] ≃ (2.0, 6.0)
        @test_broken Enzyme.jacobian(Enzyme.Reverse, x -> (2*x, x^2), scalar)[1] ≃ (2.0, 6.0)
    end

    mkarray1 = x -> mkarray((2,2),2*x,sin(x),x^2,exp(x))

    @testset "∂ matrix / ∂ scalar" begin
        @test Enzyme.gradient(Enzyme.Forward, mkarray1, scalar)[1] ≈ [2.0 6.0; cos(scalar) exp(scalar)]
        @test_broken Enzyme.gradient(Enzyme.Reverse, mkarray1, scalar)[1] ≈ [2.0 6.0; cos(scalar) exp(scalar)]
    
        @test Enzyme.jacobian(Enzyme.Forward, mkarray1, scalar)[1] ≈ [2.0 6.0; cos(scalar) exp(scalar)]
        @test Enzyme.jacobian(Enzyme.Reverse, mkarray1, scalar)[1] ≈ [2.0 6.0; cos(scalar) exp(scalar)]
    end

    @testset "∂ struct / ∂ scalar" begin
        @test Enzyme.gradient(Enzyme.Forward, x -> OutStruct(x, x^2, x^3), scalar)[1] == OutStruct(1.0,2*scalar,3*scalar^2)
        @test_broken Enzyme.gradient(Enzyme.Reverse, x -> InpStruct(x, x^2, x^3), scalar)[1] == (OutStruct(1.0,2.0,3.0),)
        @test Enzyme.jacobian(Enzyme.Forward, x -> OutStruct(x, x^2, x^3), scalar)[1] == OutStruct(1.0,2*scalar,3*scalar^2)
        @test_broken Enzyme.jacobian(Enzyme.Reverse, x -> InpStruct(x, x^2, x^3), scalar)[1] == (OutStruct(1.0,2.0,3.0),)
    end    


    vector = [2.7, 3.1]

    @testset "∂ scalar / ∂ vector" begin
        @test Enzyme.gradient(Enzyme.Forward, x -> x[1] * x[2], vector)[1] ≈ [vector[2],vector[1]]
        @test Enzyme.gradient(Enzyme.Reverse, x -> x[1] * x[2], vector)[1] ≈ [vector[2], vector[1]]
        @test Enzyme.jacobian(Enzyme.Forward, x -> x[1] * x[2], vector)[1] ≈ [vector[2], vector[1]]
        @test Enzyme.jacobian(Enzyme.Reverse, x -> x[1] * x[2], vector)[1] ≈ [vector[2], vector[1]]
    end

    @testset "∂ vector / ∂ vector" begin
        @test Enzyme.gradient(Enzyme.Forward, x -> [x[1] * x[2], cos(x[1]) + x[2]], vector)[1] ≈
                            [vector[2] vector[1]; -sin(vector[1])  1.0]
        @test_broken Enzyme.gradient(Enzyme.Reverse, x -> [x[1] * x[2], cos(x[1]) + x[2]], vector)[1] ≈
                            [vector[2] vector[1]; -sin(vector[1])  1.0]
        @test Enzyme.jacobian(Enzyme.Forward, x -> [x[1] * x[2], cos(x[1]) + x[2]], vector)[1] ≈
                            [vector[2] vector[1]; -sin(vector[1])  1.0]
        @test Enzyme.jacobian(Enzyme.Reverse, x -> [x[1] * x[2], cos(x[1]) + x[2]], vector)[1] ≈
                            [vector[2] vector[1]; -sin(vector[1])  1.0]
    end

    @testset "∂ tuple / ∂ vector" begin
        @test Enzyme.gradient(Enzyme.Forward, x -> (x[1] * x[2], cos(x[1]) + x[2]), vector)[1] ≃
            [(vector[2], -sin(vector[1])), (vector[1], 1.0)]
        @test_broken Enzyme.gradient(Enzyme.Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), vector)[1] ≈
            ([vector[2], -sin(vector[1])], [vector[1], 1.0])
        @test Enzyme.jacobian(Enzyme.Forward, x -> (x[1] * x[2], cos(x[1]) + x[2]), vector)[1] ≃
            [(vector[2], -sin(vector[1])), (vector[1], 1.0)]
        @test_broken Enzyme.jacobian(Enzyme.Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), vector)[1]
    end

    mkarray2 = x -> mkarray((2,2), x[1]*x[2], exp(x[2]), cos(x[1])+x[2], x[1])

    @testset "∂ matrix / ∂ vector" begin
        @test Enzyme.gradient(Enzyme.Forward, mkarray2, vector)[1] ≈
            mkarray((2,2,2), vector[2], 0.0, -sin(vector[1]), 1.0, vector[1], exp(vector[2]), 1.0, 0.0)
        @test_broken Enzyme.gradient(Enzyme.Reverse, mkarray2, vector)[1]
        @test Enzyme.jacobian(Enzyme.Forward, mkarray2, vector)[1] ≈
            mkarray((2,2,2), vector[2], 0.0, -sin(vector[1]), 1.0, vector[1], exp(vector[2]), 1.0, 0.0)
        @test Enzyme.jacobian(Enzyme.Reverse, mkarray2, vector)[1] ≈
            mkarray((2,2,2), vector[2], 0.0, -sin(vector[1]), 1.0, vector[1], exp(vector[2]), 1.0, 0.0)
    end

    @testset "∂ struct / ∂ vector" begin
        @test Enzyme.gradient(Enzyme.Forward, x -> OutStruct(x[1] * x[2], cos(x[1]) + x[2], exp(x[2])), vector)[1] ≃
            [OutStruct(vector[2], -sin(vector[1]), 0.0), OutStruct(vector[1], 1.0, exp(vector[2]))]
        @test_broken Enzyme.gradient(Enzyme.Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), vector)[1] ≈
            ([vector[2], -sin(vector[1])], [vector[1], 1.0])
        @test Enzyme.jacobian(Enzyme.Forward, x -> OutStruct(x[1] * x[2], cos(x[1]) + x[2], exp(x[2])), vector)[1] ≃
            [OutStruct(vector[2], -sin(vector[1]), 0.0), OutStruct(vector[1], 1.0, exp(vector[2]))]
        @test_broken Enzyme.jacobian(Enzyme.Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), vector)[1] ≈
            ([vector[2], -sin(vector[1])], [vector[1], 1.0])
    end


    tuplev = (2.7, 3.1)

    @testset "∂ scalar / ∂ tuple" begin
        @test Enzyme.gradient(Enzyme.Forward, x -> x[1] * x[2], tuplev)[1] ≃ (tuplev[2],tuplev[1])
        @test Enzyme.gradient(Enzyme.Reverse, x -> x[1] * x[2], tuplev)[1] ≃ (tuplev[2],tuplev[1])
        @test Enzyme.jacobian(Enzyme.Forward, x -> x[1] * x[2], tuplev)[1] ≃ (tuplev[2],tuplev[1])
        @test Enzyme.jacobian(Enzyme.Reverse, x -> x[1] * x[2], tuplev)[1] ≃ (tuplev[2],tuplev[1])
    end

    @testset "∂ vector / ∂ tuple" begin
        @test Enzyme.gradient(Enzyme.Forward, x -> [x[1] * x[2], cos(x[1]) + x[2]], tuplev)[1] ≃
            ([tuplev[2], -sin(tuplev[1])], [tuplev[1], 1.0])
        @test_broken Enzyme.gradient(Enzyme.Reverse, x -> [x[1] * x[2], cos(x[1]) + x[2]], tuplev)[1] ≈
            ([tuplev[2], -sin(tuplev[1])], [tuplev[1], 1.0])
        @test_broken Enzyme.jacobian(Enzyme.Forward, x -> [x[1] * x[2], cos(x[1]) + x[2]], tuplev)[1] ≈
            [tuplev[2] tuplev[1]; -sin(tuplev[1])  1.0]
        @test Enzyme.jacobian(Enzyme.Reverse, x -> [x[1] * x[2], cos(x[1]) + x[2]], tuplev)[1] ≃
            [(tuplev[2], tuplev[1]), (-sin(tuplev[1]), 1.0)]
    end

    @testset "∂ tuple / ∂ tuple" begin
        @test Enzyme.gradient(Enzyme.Forward, x -> (x[1] * x[2], cos(x[1]) + x[2]), tuplev)[1] ≃
            ((vector[2], -sin(vector[1])), (vector[1], 1.0))
        @test_broken Enzyme.gradient(Enzyme.Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), tuplev)[1] ≈
            ([tuplev[2], -sin(tuplev[1])], [tuplev[1], 1.0])
        @test Enzyme.jacobian(Enzyme.Forward, x -> (x[1] * x[2], cos(x[1]) + x[2]), tuplev)[1] ≃
            ((tuplev[2], -sin(tuplev[1])), (tuplev[1], 1.0))
        @test_broken Enzyme.jacobian(Enzyme.Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), tuplev)[1] ≈
                            [tuplev[2] tuplev[1]; -sin(tuplev[1])  1.0]
    end

    @testset "∂ matrix / ∂ tuple" begin
        @test Enzyme.gradient(Enzyme.Forward, mkarray2, tuplev)[1] ≃
            ([tuplev[2] -sin(tuplev[1]); 0.0 1.0], [tuplev[1] 1.0; exp(tuplev[2]) 0.0])
        @test_broken Enzyme.gradient(Enzyme.Reverse, mkarray2, tuplev)[1]
        @test_broken Enzyme.jacobian(Enzyme.Forward, mkarray2, tuplev)[1] ≈
                            [tuplev[2] -sin(tuplev[1]); 0.0 1.0;;; tuplev[1] 1.0;  exp(tuplev[2]) 0.0]
        @test_broken Enzyme.jacobian(Enzyme.Reverse, x -> mkarray2, tuplev)[1] ≈
                            [tuplev[2] -sin(tuplev[1]); 0.0 1.0;;; tuplev[1] 1.0;  exp(tuplev[2]) 0.0]
    end

    @testset "∂ struct / ∂ tuple" begin
        @test Enzyme.gradient(Enzyme.Forward, x -> OutStruct(x[1] * x[2], cos(x[1]) + x[2], exp(x[2])), tuplev)[1] ≃
            (OutStruct(tuplev[2], -sin(tuplev[1]), 0.0), OutStruct(tuplev[1], 1.0, exp(tuplev[2])))
        @test_broken Enzyme.gradient(Enzyme.Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), tuplev)[1] ≈
            ([tuplev[2], -sin(tuplev[1])], [tuplev[1], 1.0])
        @test_broken Enzyme.jacobian(Enzyme.Forward, x -> OutStruct(x[1] * x[2], cos(x[1]) + x[2], exp(x[2])), tuplev)[1] ≃
            [OutStruct(tuplev[2], -sin(tuplev[1]), 0.0), OutStruct(tuplev[1], 1.0, exp(tuplev[2]))]
        @test_broken Enzyme.jacobian(Enzyme.Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), tuplev)[1] ≈
            ([tuplev[2], -sin(tuplev[1])], [tuplev[1], 1.0])
    end


    matrix = [2.7 3.1; 4.7 5.6]

    @testset "∂ scalar / ∂ matrix" begin
        @test Enzyme.gradient(Enzyme.Forward, x->x[1,1]*x[1,2]+x[2,1]*x[2,2], matrix)[1] ≈
            [matrix[1,2] matrix[1,1]; matrix[2,2] matrix[2,1]]
        @test Enzyme.gradient(Enzyme.Reverse, x->x[1,1]*x[1,2]+x[2,1]*x[2,2], matrix)[1] ≈
            [matrix[1,2] matrix[1,1]; matrix[2,2] matrix[2,1]]
        @test Enzyme.jacobian(Enzyme.Forward, x->x[1,1]*x[1,2]+x[2,1]*x[2,2], matrix)[1] ≈
            [matrix[1,2] matrix[1,1]; matrix[2,2] matrix[2,1]]
        @test Enzyme.jacobian(Enzyme.Reverse, x->x[1,1]*x[1,2]+x[2,1]*x[2,2], matrix)[1] ≈
            [matrix[1,2] matrix[1,1]; matrix[2,2] matrix[2,1]]
    end

    @testset "∂ vector / ∂ matrix" begin
        @test Enzyme.gradient(Enzyme.Forward, x->[x[1,1]*x[1,2],x[2,1]*x[2,2]], matrix)[1] ≈
            mkarray((2,2,2), matrix[1,2], 0.0, 0.0, matrix[2,2], matrix[1,1], 0.0, 0.0, matrix[2,1])
        @test_broken Enzyme.gradient(Enzyme.Reverse, x->[x[1,1]*x[1,2],x[2,1]*x[2,2]], matrix)[1]
        # again we can't use array construction syntax because of 1.6
        @test Enzyme.jacobian(Enzyme.Forward, x->[x[1,1]*x[1,2],x[2,1]*x[2,2]], matrix)[1] ≈
            mkarray((2,2,2), matrix[1,2], 0.0, 0.0, matrix[2,2], matrix[1,1], 0.0, 0.0, matrix[2,1])
        @test Enzyme.jacobian(Enzyme.Reverse, x->[x[1,1]*x[1,2],x[2,1]*x[2,2]], matrix)[1] ≈
            mkarray((2,2,2), matrix[1,2], 0.0, 0.0, matrix[2,2], matrix[1,1], 0.0, 0.0, matrix[2,1])
    end

    @testset "∂ tuple / ∂ matrix" begin
        @test Enzyme.gradient(Enzyme.Forward, x->(x[1,1]*x[1,2],x[2,1]*x[2,2]), matrix)[1] ≃ 
            [(matrix[1,2],0.0) (matrix[1,1],0.0); (0.0,matrix[2,2]) (0.0,matrix[2,1])]
        @test_broken Enzyme.gradient(Enzyme.Reverse, x->(x[1,1]*x[1,2],x[2,1]*x[2,2]), matrix)
        @test Enzyme.jacobian(Enzyme.Forward, x->(x[1,1]*x[1,2],x[2,1]*x[2,2]), matrix)[1] ≃
            [(matrix[1,2],0.0) (matrix[1,1],0.0); (0.0,matrix[2,2]) (0.0,matrix[2,1])]
        @test_broken Enzyme.jacobian(Enzyme.Reverse, x->(x[1,1]*x[1,2],x[2,1]*x[2,2]), matrix)[1]
    end

    mkarray3 = x -> mkarray((2,2), x[1,1]*x[1,2], exp(x[1,1])+x[2,2], x[2,1]*x[2,2], sin(x[1,2])+x[2,1])

    @testset "∂ matrix / ∂ matrix" begin
        @test Enzyme.gradient(Enzyme.Forward, mkarray3, matrix)[1] ≈
            mkarray((2,2,2,2), matrix[1,2],exp(matrix[1,1]),0.0,0.0,0.0,0.0,matrix[2,2],1.0,
                    matrix[1,1],0.0,0.0,cos(matrix[1,2]),0.0,1.0,matrix[2,1],0.0)
        @test_broken Enzyme.gradient(Enzyme.Reverse, mkarray3, matrix)[1]
        # array construction syntax broken on 1.6
        @test Enzyme.jacobian(Enzyme.Forward, mkarray3, matrix)[1] ≈
            mkarray((2,2,2,2), matrix[1,2],exp(matrix[1,1]),0.0,0.0,0.0,0.0,matrix[2,2],1.0,
                    matrix[1,1],0.0,0.0,cos(matrix[1,2]),0.0,1.0,matrix[2,1],0.0)
        @test Enzyme.jacobian(Enzyme.Reverse, mkarray3, matrix)[1] ≈
            mkarray((2,2,2,2), matrix[1,2],exp(matrix[1,1]),0.0,0.0,0.0,0.0,matrix[2,2],1.0,
                    matrix[1,1],0.0,0.0,cos(matrix[1,2]),0.0,1.0,matrix[2,1],0.0)
    end

    @testset "∂ tuple / ∂ matrix" begin
        @test Enzyme.gradient(Enzyme.Forward, x->OutStruct(x[1,1]*x[1,2],x[2,1]*x[2,2], exp(x[1,1])+x[2,2]), matrix)[1] ≃
            [OutStruct(matrix[1,2],0.0, exp(matrix[1,1])) OutStruct(matrix[1,1],0.0,0.0);
             OutStruct(0.0,matrix[2,2],0.0) OutStruct(0.0,matrix[2,1], 1.0)]
        @test_broken Enzyme.gradient(Enzyme.Reverse, x->OutStruct(x[1,1]*x[1,2],x[2,1]*x[2,2],
                                                                  exp(x[1,1])+x[2,2]), matrix)[1]
        @test Enzyme.jacobian(Enzyme.Forward, x->OutStruct(x[1,1]*x[1,2],x[2,1]*x[2,2], exp(x[1,1])+x[2,2]), matrix)[1] ≃ 
            [OutStruct(matrix[1,2],0.0, exp(matrix[1,1])) OutStruct(matrix[1,1],0.0,0.0);
            OutStruct(0.0,matrix[2,2],0.0) OutStruct(0.0,matrix[2,1], 1.0)]
        @test_broken Enzyme.jacobian(Enzyme.Reverse, x->OutStruct(x[1,1]*x[1,2],x[2,1]*x[2,2],
                                                                  exp(x[1,1])+x[2,2]), matrix)[1]
    end


    istruct = InpStruct(2.7, 3.1, 4.7)

    @testset "∂ scalar / ∂ struct" begin
        @test_broken Enzyme.gradient(Enzyme.Forward, x -> x.i1 * x.i2 + x.i3, istruct)[1]
        @test Enzyme.gradient(Enzyme.Reverse, x -> x.i1 * x.i2 + x.i3, istruct)[1] ≃ InpStruct(istruct.i2, istruct.i1, 1.0)
        @test_broken Enzyme.jacobian(Enzyme.Forward, x -> x.i1 * x.i2 + x.i3, istruct)[1]
        @test Enzyme.jacobian(Enzyme.Reverse, x -> x.i1 * x.i2 + x.i3, istruct)[1] ≃ InpStruct(istruct.i2, istruct.i1, 1.0)
    end
    
    @testset "∂ vector / ∂ struct" begin
        @test_broken Enzyme.gradient(Enzyme.Forward, x -> [x.i1 * x.i2, cos(x.i3) + x.i1], istruct)[1]
        @test_broken Enzyme.gradient(Enzyme.Reverse, x -> [x.i1 * x.i2, cos(x.i3) + x.i1], istruct)[1]
        @test_broken Enzyme.jacobian(Enzyme.Forward, x -> [x.i1 * x.i2, cos(x.i3) + x.i1], istruct)[1]
        @test Enzyme.jacobian(Enzyme.Reverse, x -> [x.i1 * x.i2, cos(x.i3) + x.i1], istruct)[1] ≃
            [InpStruct(istruct.i2, istruct.i1, 0.0), InpStruct(1.0, 0.0, -sin(istruct.i3))]
    end
    
    @testset "∂ tuple / ∂ struct" begin
        @test_broken Enzyme.gradient(Enzyme.Forward, x -> (x.i1 * x.i2, cos(x.i3) + x.i1), istruct)[1]
        @test_broken Enzyme.gradient(Enzyme.Reverse, x -> (x.i1 * x.i2, cos(x.i3) + x.i1), istruct)[1]
        @test_broken Enzyme.jacobian(Enzyme.Forward, x -> (x.i1 * x.i2, cos(x.i3) + x.i1), istruct)[1]
        @test_broken Enzyme.jacobian(Enzyme.Reverse, x -> (x.i1 * x.i2, cos(x.i3) + x.i1), istruct)[1]
    end

    mkarray4 = x -> mkarray((2,2), x.i1*x.i2, exp(x.i2), cos(x.i3)+x.i1, x.i1)

    @testset "∂ matrix / ∂ struct" begin
        @test_broken Enzyme.gradient(Enzyme.Forward, x -> [x.i1 * x.i2  cos(x.i3) + x.i1; exp(x.i2) x.i1], istruct)[1]
        @test_broken Enzyme.gradient(Enzyme.Reverse, x -> [x.i1 * x.i2  cos(x.i3) + x.i1; exp(x.i2) x.i1], istruct)[1]
        @test_broken Enzyme.jacobian(Enzyme.Forward, x -> [x.i1 * x.i2  cos(x.i3) + x.i1; exp(x.i2) x.i1], istruct)[1]
        @test Enzyme.jacobian(Enzyme.Reverse, mkarray4, istruct)[1] ≃
            [InpStruct(istruct.i2, istruct.i1, 0.0) InpStruct(1.0, 0.0, -sin(istruct.i3));
            InpStruct(0.0, exp(istruct.i2), 0.0) InpStruct(1.0, 0.0, 0.0)]
    end

    @testset "∂ struct / ∂ struct" begin
        @test_broken Enzyme.gradient(Enzyme.Forward, x -> OutStruct(x.i1 * x.i2, cos(x.i3) + x.i1, exp(x.i2)), istruct)[1]
        @test_broken Enzyme.gradient(Enzyme.Reverse, x -> OutStruct(x.i1 * x.i2, cos(x.i3) + x.i1, exp(x.i2)), istruct)[1]
        @test_broken Enzyme.jacobian(Enzyme.Forward, x -> OutStruct(x.i1 * x.i2, cos(x.i3) + x.i1, exp(x.i2)), istruct)[1]
        @test_broken Enzyme.jacobian(Enzyme.Reverse, x -> OutStruct(x.i1 * x.i2, cos(x.i3) + x.i1, exp(x.i2)), istruct)[1]
    end
end

@testset "Simple Jacobian" begin
    @test Enzyme.jacobian(Enzyme.Forward, x->2*x, 3.0)[1] ≈ 2.0
    @test Enzyme.jacobian(Enzyme.Forward, x->[x, 2*x], 3.0)[1] ≈ [1.0, 2.0]
    @test Enzyme.jacobian(Enzyme.Forward, x->sum(abs2, x), [2.0, 3.0])[1] ≈ [4.0, 6.0]

    @test Enzyme.jacobian(Enzyme.Forward, x->2*x, 3.0, chunk=Val(1))[1] ≈ 2.0
    @test Enzyme.jacobian(Enzyme.Forward, x->[x, 2*x], 3.0, chunk=Val(1))[1] ≈ [1.0, 2.0]
    @test Enzyme.jacobian(Enzyme.Forward, x->sum(abs2, x), [2.0, 3.0], chunk=Val(1))[1] ≈ [4.0, 6.0]

    @test Enzyme.jacobian(Enzyme.Forward, x->2*x, 3.0, chunk=Val(2))[1] ≈ 2.0
    @test Enzyme.jacobian(Enzyme.Forward, x->[x, 2*x], 3.0, chunk=Val(2))[1] ≈ [1.0, 2.0]
    @test Enzyme.jacobian(Enzyme.Forward, x->sum(abs2, x), [2.0, 3.0], chunk=Val(2))[1] ≈ [4.0, 6.0]

    @test Enzyme.jacobian(Enzyme.Reverse, x->[x, 2*x], 3.0, n_outs=Val((2,)))[1] ≈ [1.0, 2.0]
    @test Enzyme.jacobian(Enzyme.Reverse, x->[x, 2*x], 3.0, n_outs=Val((2,)), chunk=Val(1))[1] ≈ [1.0, 2.0]
    @test Enzyme.jacobian(Enzyme.Reverse, x->[x, 2*x], 3.0, n_outs=Val((2,)), chunk=Val(2))[1] ≈ [1.0, 2.0]

    x = float.(reshape(1:6, 2, 3))

    fillabs2(x) = [sum(abs2, x), 10*sum(abs2, x), 100*sum(abs2, x), 1000*sum(abs2, x)]

    jac = Enzyme.jacobian(Enzyme.Forward, fillabs2, x)[1]

    @test jac[1, :, :] ≈ [2.0 6.0 10.0; 4.0 8.0 12.0]
    @test jac[2, :, :] ≈ [20.0 60.0 100.0; 40.0 80.0 120.0]
    @test jac[3, :, :] ≈ [200.0 600.0 1000.0; 400.0 800.0 1200.0]
    @test jac[4, :, :] ≈ [2000.0 6000.0 10000.0; 4000.0 8000.0 12000.0]

    jac = Enzyme.jacobian(Enzyme.Forward, fillabs2, x, chunk=Val(1))[1]

    @test jac[1, :, :] ≈ [2.0 6.0 10.0; 4.0 8.0 12.0]
    @test jac[2, :, :] ≈ [20.0 60.0 100.0; 40.0 80.0 120.0]
    @test jac[3, :, :] ≈ [200.0 600.0 1000.0; 400.0 800.0 1200.0]
    @test jac[4, :, :] ≈ [2000.0 6000.0 10000.0; 4000.0 8000.0 12000.0]

    jac = Enzyme.jacobian(Enzyme.Forward, fillabs2, x, chunk=Val(2))[1]

    @test jac[1, :, :] ≈ [2.0 6.0 10.0; 4.0 8.0 12.0]
    @test jac[2, :, :] ≈ [20.0 60.0 100.0; 40.0 80.0 120.0]
    @test jac[3, :, :] ≈ [200.0 600.0 1000.0; 400.0 800.0 1200.0]
    @test jac[4, :, :] ≈ [2000.0 6000.0 10000.0; 4000.0 8000.0 12000.0]


    jac = Enzyme.jacobian(Enzyme.Reverse, fillabs2, x, n_outs=Val((4,)), chunk=Val(1))[1]

    @test jac[1, :, :] ≈ [2.0 6.0 10.0; 4.0 8.0 12.0]
    @test jac[2, :, :] ≈ [20.0 60.0 100.0; 40.0 80.0 120.0]
    @test jac[3, :, :] ≈ [200.0 600.0 1000.0; 400.0 800.0 1200.0]
    @test jac[4, :, :] ≈ [2000.0 6000.0 10000.0; 4000.0 8000.0 12000.0]

    jac = Enzyme.jacobian(Enzyme.Reverse, fillabs2, x, n_outs=Val((4,)), chunk=Val(2))[1]

    @test jac[1, :, :] ≈ [2.0 6.0 10.0; 4.0 8.0 12.0]
    @test jac[2, :, :] ≈ [20.0 60.0 100.0; 40.0 80.0 120.0]
    @test jac[3, :, :] ≈ [200.0 600.0 1000.0; 400.0 800.0 1200.0]
    @test jac[4, :, :] ≈ [2000.0 6000.0 10000.0; 4000.0 8000.0 12000.0]

    fillinpabs2(x) = [(x.i1*x.i1+x.i2*x.i2+x.i3*x.i3), 10*(x.i1*x.i1+x.i2*x.i2+x.i3*x.i3), 100*(x.i1*x.i1+x.i2*x.i2+x.i3*x.i3), 1000*(x.i1*x.i1+x.i2*x.i2+x.i3*x.i3)]

    x2 = InpStruct(1.0, 2.0, 3.0)

    jac = Enzyme.jacobian(Enzyme.Reverse, fillinpabs2, x2, n_outs=Val((4,)), chunk=Val(1))[1]

    @test jac[1] == InpStruct(2.0, 4.0, 6.0)
    @test jac[2] == InpStruct(20.0, 40.0, 60.0)
    @test jac[3] == InpStruct(200.0, 400.0, 600.0)
    @test jac[4] == InpStruct(2000.0, 4000.0, 6000.0)

    jac = Enzyme.jacobian(Enzyme.Reverse, fillinpabs2, x2, n_outs=Val((4,)), chunk=Val(2))[1]

    @test jac[1] == InpStruct(2.0, 4.0, 6.0)
    @test jac[2] == InpStruct(20.0, 40.0, 60.0)
    @test jac[3] == InpStruct(200.0, 400.0, 600.0)
    @test jac[4] == InpStruct(2000.0, 4000.0, 6000.0)

    filloutabs2(x) = OutStruct(sum(abs2, x), 10*sum(abs2, x), 100*sum(abs2, x))

    jac = Enzyme.jacobian(Enzyme.Forward, filloutabs2, x)[1]

    @test jac[1, 1] == OutStruct(2.0, 20.0, 200.0)
    @test jac[2, 1] == OutStruct(4.0, 40.0, 400.0)

    @test jac[1, 2] == OutStruct(6.0, 60.0, 600.0)
    @test jac[2, 2] == OutStruct(8.0, 80.0, 800.0)

    @test jac[1, 3] == OutStruct(10.0, 100.0, 1000.0)
    @test jac[2, 3] == OutStruct(12.0, 120.0, 1200.0)

    jac = Enzyme.jacobian(Enzyme.Forward, filloutabs2, x, chunk=Val(1))[1]

    @test jac[1, 1] == OutStruct(2.0, 20.0, 200.0)
    @test jac[2, 1] == OutStruct(4.0, 40.0, 400.0)

    @test jac[1, 2] == OutStruct(6.0, 60.0, 600.0)
    @test jac[2, 2] == OutStruct(8.0, 80.0, 800.0)

    @test jac[1, 3] == OutStruct(10.0, 100.0, 1000.0)
    @test jac[2, 3] == OutStruct(12.0, 120.0, 1200.0)

    jac = Enzyme.jacobian(Enzyme.Forward, filloutabs2, x, chunk=Val(2))[1]

    @test jac[1, 1] == OutStruct(2.0, 20.0, 200.0)
    @test jac[2, 1] == OutStruct(4.0, 40.0, 400.0)

    @test jac[1, 2] == OutStruct(6.0, 60.0, 600.0)
    @test jac[2, 2] == OutStruct(8.0, 80.0, 800.0)

    @test jac[1, 3] == OutStruct(10.0, 100.0, 1000.0)
    @test jac[2, 3] == OutStruct(12.0, 120.0, 1200.0)
end


@testset "Jacobian" begin
    function inout(v)
       [v[2], v[1]*v[1], v[1]*v[1]*v[1]]
    end

    jac = Enzyme.jacobian(Reverse, inout, [2.0, 3.0], n_outs=Val((3,)), chunk=Val(1))[1]
    @test size(jac) == (3, 2)
    @test jac ≈ [ 0.0   1.0;
                  4.0   0.0;
                  12.0  0.0]

    jac = Enzyme.jacobian(Forward, inout, [2.0, 3.0], chunk=Val(1))[1]
    @test size(jac) == (3, 2)
    @test jac ≈ [ 0.0   1.0;
                  4.0   0.0;
                  12.0  0.0]

    @test jac == Enzyme.jacobian(Forward, inout, [2.0, 3.0])[1]

    jac = Enzyme.jacobian(Reverse, inout, [2.0, 3.0], n_outs=Val((3,)), chunk=Val(2))[1]
    @test size(jac) == (3, 2)
    @test jac ≈ [ 0.0   1.0;
                  4.0   0.0;
                  12.0  0.0]

    jac = Enzyme.jacobian(Forward, inout, [2.0, 3.0], chunk=Val(2))[1]
    @test size(jac) == (3, 2)
    @test jac ≈ [ 0.0   1.0;
                  4.0   0.0;
                  12.0  0.0]

    function f_test_1(A, x)
        utmp = A*x[2:end] .+ x[1]
        return utmp
    end

    function f_test_2(A, x)
        utmp = Vector{Float64}(undef, length(x)-1)
        utmp .= A*x[2:end] .+ x[1]
        return utmp
    end

    function f_test_3!(u, A, x)
        utmp .= A*x[2:end] .+ x[1]
    end

    J_r_1(A, x) = Enzyme.jacobian(Reverse, θ -> f_test_1(A, θ), x, n_outs=Val((5,)))[1]
    J_r_2(A, x) = Enzyme.jacobian(Reverse, θ -> f_test_2(A, θ), x, n_outs=Val((5,)))[1]
    J_r_3(u, A, x) = Enzyme.jacobian(Reverse, θ -> f_test_3!(u, A, θ), x, n_outs=Val((5,)))[1]

    J_f_1(A, x) = Enzyme.jacobian(Forward, Const(θ -> f_test_1(A, θ)), x)[1]
    J_f_2(A, x) = Enzyme.jacobian(Forward, Const(θ -> f_test_2(A, θ)), x)[1]
    J_f_3(u, A, x) = Enzyme.jacobian(Forward, Const(θ -> f_test_3!(u, A, θ)), x)[1]

    x = ones(6)
    A = Matrix{Float64}(LinearAlgebra.I, 5, 5)
    u = Vector{Float64}(undef, 5)

    @test J_r_1(A, x) == [
        1.0  1.0  0.0  0.0  0.0  0.0;
        1.0  0.0  1.0  0.0  0.0  0.0;
        1.0  0.0  0.0  1.0  0.0  0.0;
        1.0  0.0  0.0  0.0  1.0  0.0;
        1.0  0.0  0.0  0.0  0.0  1.0;
    ]

    @test J_r_2(A, x) == [
        1.0  1.0  0.0  0.0  0.0  0.0;
        1.0  0.0  1.0  0.0  0.0  0.0;
        1.0  0.0  0.0  1.0  0.0  0.0;
        1.0  0.0  0.0  0.0  1.0  0.0;
        1.0  0.0  0.0  0.0  0.0  1.0;
    ]

    @test J_f_1(A, x) == [
        1.0  1.0  0.0  0.0  0.0  0.0;
        1.0  0.0  1.0  0.0  0.0  0.0;
        1.0  0.0  0.0  1.0  0.0  0.0;
        1.0  0.0  0.0  0.0  1.0  0.0;
        1.0  0.0  0.0  0.0  0.0  1.0;
    ]
    @test J_f_2(A, x) == [
        1.0  1.0  0.0  0.0  0.0  0.0;
        1.0  0.0  1.0  0.0  0.0  0.0;
        1.0  0.0  0.0  1.0  0.0  0.0;
        1.0  0.0  0.0  0.0  1.0  0.0;
        1.0  0.0  0.0  0.0  0.0  1.0;
    ]

    # @show J_r_3(u, A, x)
    # @show J_f_3(u, A, x)
end


