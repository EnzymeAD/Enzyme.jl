using Enzyme, Test
using LinearAlgebra
using SparseArrays

using Enzyme: gradient, jacobian

@isdefined(UTILS) || include("utils.jl")


@testset "Gradient & NamedTuples" begin
    xy = (x = [1.0, 2.0], y = [3.0, 4.0])
    grad = gradient(Reverse, z -> sum(z.x .* z.y), xy)
    @test grad == (x = [3.0, 4.0], y = [1.0, 2.0])

    xp = (x = [1.0, 2.0], p = 3)  # 3::Int is non-diff
    grad = gradient(Reverse, z -> sum(z.x .^ z.p), xp)
    @test grad.x == [3.0, 12.0]

    xp2 = (x = [1.0, 2.0], p = 3.0)  # mixed activity
    grad = gradient(Reverse, z -> sum(z.x .^ z.p), xp2)
    @test grad.x == [3.0, 12.0]
    @test grad.p ≈ 5.545177444479562

    xy = (x = [1.0, 2.0], y = [3, 4])  # y is non-diff
    grad = gradient(Reverse, z -> sum(z.x .* z.y), xy)
    @test grad.x == [3.0, 4.0]
    @test grad.y === xy.y  # make_zero did not copy this

    grad = gradient(Reverse, z -> (z.x * z.y), (x=5.0, y=6.0))
    @test grad == (x = 6.0, y = 5.0)

    grad = gradient(Reverse, abs2, 7.0)
    @test grad == 14.0
end

@testset "Gradient & SparseArrays" begin
    x = sparse([5.0, 0.0, 6.0])
    dx = gradient(Reverse, sum, x)
    @test dx isa SparseVector
    @test dx ≈ [1, 0, 1]

    x = sparse([5.0 0.0 6.0])
    dx = gradient(Reverse, sum, x)
    @test dx isa SparseMatrixCSC
    @test dx ≈ [1 0 1]
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

    @testset  "∂ scalar / ∂ scalar" begin
        @test gradient(Forward, x -> x^2, scalar) ≈ 6.0
        @test gradient(Reverse, x -> x^2, scalar) ≈ 6.0
        @test jacobian(Forward, x -> x^2, scalar) ≈ 6.0
        @test jacobian(Reverse, x -> x^2, scalar) ≈ 6.0
        @test gradient(Forward, x -> 2*x, scalar) ≈ 2.0
        @test gradient(Reverse, x -> 2*x, scalar) ≈ 2.0
        @test jacobian(Forward, x -> 2*x, scalar) ≈ 2.0
        @test jacobian(Reverse, x -> 2*x, scalar) ≈ 2.0
    end

    @testset "∂ vector / ∂ scalar" begin
        @test gradient(Forward, x -> [2*x, x^2], scalar) ≈ [2.0, 6.0]
        @test_broken gradient(Reverse, x -> [2*x, x^2], scalar) ≈ [2.0, 6.0]
    
        @test jacobian(Forward, x -> [2*x, x^2], scalar) ≈ [2.0, 6.0]
        @test jacobian(Reverse, x -> [2*x, x^2], scalar) ≈ [2.0, 6.0]
    end

    @testset "∂ tuple / ∂ scalar" begin
        @test gradient(Forward, x -> (2*x, x^2), scalar) ≃ (2.0, 6.0)
        @test_broken gradient(Reverse, x -> (2*x, x^2), scalar) ≈ [2.0, 6.0]
    
        @test jacobian(Forward, x -> (2*x, x^2), scalar) ≃ (2.0, 6.0)
        @test_broken jacobian(Reverse, x -> (2*x, x^2), scalar) ≃ (2.0, 6.0)
    end

    mkarray1 = x -> mkarray((2,2),2*x,sin(x),x^2,exp(x))

    @testset "∂ matrix / ∂ scalar" begin
        @test gradient(Forward, mkarray1, scalar) ≈ [2.0 6.0; cos(scalar) exp(scalar)]
        @test_broken gradient(Reverse, mkarray1, scalar) ≈ [2.0 6.0; cos(scalar) exp(scalar)]
    
        @test jacobian(Forward, mkarray1, scalar) ≈ [2.0 6.0; cos(scalar) exp(scalar)]
        @test jacobian(Reverse, mkarray1, scalar) ≈ [2.0 6.0; cos(scalar) exp(scalar)]
    end

    @testset "∂ struct / ∂ scalar" begin
        @test gradient(Forward, x -> OutStruct(x, x^2, x^3), scalar) == OutStruct(1.0,2*scalar,3*scalar^2)
        @test_broken gradient(Reverse, x -> InpStruct(x, x^2, x^3), scalar) == (OutStruct(1.0,2.0,3.0),)
        @test jacobian(Forward, x -> OutStruct(x, x^2, x^3), scalar) == OutStruct(1.0,2*scalar,3*scalar^2)
        @test_broken jacobian(Reverse, x -> InpStruct(x, x^2, x^3), scalar) == (OutStruct(1.0,2.0,3.0),)
    end

    vector = [2.7, 3.1]

    @testset "∂ scalar / ∂ vector" begin
        @test gradient(Forward, x -> x[1] * x[2], vector) ≃ (vector[2],vector[1])
        @test gradient(Reverse, x -> x[1] * x[2], vector) ≈ [vector[2], vector[1]]
        @test jacobian(Forward, x -> x[1] * x[2], vector) ≈ [vector[2], vector[1]]
        @test jacobian(Reverse, x -> x[1] * x[2], vector) ≈ [vector[2], vector[1]]
    end

    @testset "∂ vector / ∂ vector" begin
        @test gradient(Forward, x -> [x[1] * x[2], cos(x[1]) + x[2]], vector) ≃
            ([vector[2], -sin(vector[1])], [vector[1], 1.0])
        @test_broken gradient(Reverse, x -> [x[1] * x[2], cos(x[1]) + x[2]], vector) ≈
            ([vector[2], -sin(vector[1])], [vector[1], 1.0])
        @test jacobian(Forward, x -> [x[1] * x[2], cos(x[1]) + x[2]], vector) ≈
            [vector[2] vector[1]; -sin(vector[1])  1.0]
        @test jacobian(Reverse, x -> [x[1] * x[2], cos(x[1]) + x[2]], vector) ≈
            [vector[2] vector[1]; -sin(vector[1])  1.0]
    end

    @testset "∂ tuple / ∂ vector" begin
        @test gradient(Forward, x -> (x[1] * x[2], cos(x[1]) + x[2]), vector) ≃
            ((vector[2], -sin(vector[1])), (vector[1], 1.0))
        @test_broken gradient(Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), vector) ≈
            ([vector[2], -sin(vector[1])], [vector[1], 1.0])
        @test jacobian(Forward, x -> (x[1] * x[2], cos(x[1]) + x[2]), vector) ≃
            [(vector[2], -sin(vector[1])), (vector[1], 1.0)]
        @test_broken jacobian(Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), vector)
    end

    mkarray2 = x -> mkarray((2,2), x[1]*x[2], exp(x[2]), cos(x[1])+x[2], x[1])

    @testset "∂ matrix / ∂ vector" begin
        @test gradient(Forward, mkarray2, vector) ≃
            ([vector[2] -sin(vector[1]); 0.0 1.0], [vector[1] 1.0; exp(vector[2]) 0.0])
        @test_broken gradient(Reverse, mkarray2, vector)
        @test jacobian(Forward, mkarray2, vector) ≈
            mkarray((2,2,2), vector[2], 0.0, -sin(vector[1]), 1.0, vector[1], exp(vector[2]), 1.0, 0.0)
        @test jacobian(Reverse, mkarray2, vector) ≈
            mkarray((2,2,2), vector[2], 0.0, -sin(vector[1]), 1.0, vector[1], exp(vector[2]), 1.0, 0.0)
    end

    @testset "∂ struct / ∂ vector" begin
        @test gradient(Forward, x -> OutStruct(x[1] * x[2], cos(x[1]) + x[2], exp(x[2])), vector) ≃
            (OutStruct(vector[2], -sin(vector[1]), 0.0), OutStruct(vector[1], 1.0, exp(vector[2])))
        @test_broken gradient(Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), vector) ≈
            ([vector[2], -sin(vector[1])], [vector[1], 1.0])
    
        @test jacobian(Forward, x -> OutStruct(x[1] * x[2], cos(x[1]) + x[2], exp(x[2])), vector) ≃
            [OutStruct(vector[2], -sin(vector[1]), 0.0), OutStruct(vector[1], 1.0, exp(vector[2]))]
        @test_broken jacobian(Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), vector) ≈ ([vector[2], -sin(vector[1])], [vector[1], 1.0])
    end


    tuplev = (2.7, 3.1)

    @testset "∂ scalar / ∂ tuple" begin
        @test gradient(Forward, x -> x[1] * x[2], tuplev) ≃ (tuplev[2],tuplev[1])
        @test gradient(Reverse, x -> x[1] * x[2], tuplev) ≃ (tuplev[2],tuplev[1])
        @test jacobian(Forward, x -> x[1] * x[2], tuplev) ≃ (tuplev[2],tuplev[1])
        @test jacobian(Reverse, x -> x[1] * x[2], tuplev) ≃ (tuplev[2],tuplev[1])
    end

    @testset "∂ vector / ∂ tuple" begin
        @test gradient(Forward, x -> [x[1] * x[2], cos(x[1]) + x[2]], tuplev) ≃
            ([tuplev[2], -sin(tuplev[1])], [tuplev[1], 1.0])
        @test_broken gradient(Reverse, x -> [x[1] * x[2], cos(x[1]) + x[2]], tuplev) ≈
            ([tuplev[2], -sin(tuplev[1])], [tuplev[1], 1.0])
        @test_broken jacobian(Forward, x -> [x[1] * x[2], cos(x[1]) + x[2]], tuplev) ≈
            [tuplev[2] tuplev[1]; -sin(tuplev[1])  1.0]
        @test jacobian(Reverse, x -> [x[1] * x[2], cos(x[1]) + x[2]], tuplev) ≃
            [(tuplev[2], tuplev[1]), (-sin(tuplev[1]), 1.0)]
    end

    @testset "∂ tuple / ∂ tuple" begin
        @test gradient(Forward, x -> (x[1] * x[2], cos(x[1]) + x[2]), tuplev) ≃
            ((vector[2], -sin(vector[1])), (vector[1], 1.0))
        @test_broken gradient(Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), tuplev) ≈
            ([tuplev[2], -sin(tuplev[1])], [tuplev[1], 1.0])
        @test jacobian(Forward, x -> (x[1] * x[2], cos(x[1]) + x[2]), tuplev) ≃
            ((tuplev[2], -sin(tuplev[1])), (tuplev[1], 1.0))
        @test_broken jacobian(Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), tuplev) ≈
            [tuplev[2] tuplev[1]; -sin(tuplev[1])  1.0]
    end

    @testset "∂ matrix / ∂ tuple" begin
        @test gradient(Forward, mkarray2, tuplev) ≃
            ([tuplev[2] -sin(tuplev[1]); 0.0 1.0], [tuplev[1] 1.0; exp(tuplev[2]) 0.0])
        @test_broken gradient(Reverse, mkarray2, tuplev)
        @test_broken jacobian(Forward, mkarray2, tuplev) ≈
            [tuplev[2] -sin(tuplev[1]); 0.0 1.0;;; tuplev[1] 1.0;  exp(tuplev[2]) 0.0]
        @test_broken jacobian(Reverse, x -> mkarray2, tuplev) ≈
            [tuplev[2] -sin(tuplev[1]); 0.0 1.0;;; tuplev[1] 1.0;  exp(tuplev[2]) 0.0]
    end

    @testset "∂ struct / ∂ tuple" begin
        @test gradient(Forward, x -> OutStruct(x[1] * x[2], cos(x[1]) + x[2], exp(x[2])), tuplev) ≃
            (OutStruct(tuplev[2], -sin(tuplev[1]), 0.0), OutStruct(tuplev[1], 1.0, exp(tuplev[2])))
        @test_broken gradient(Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), tuplev) ≈
            ([tuplev[2], -sin(tuplev[1])], [tuplev[1], 1.0])
    
        @test_broken jacobian(Forward, x -> OutStruct(x[1] * x[2], cos(x[1]) + x[2], exp(x[2])), tuplev) ≃
            [OutStruct(tuplev[2], -sin(tuplev[1]), 0.0), OutStruct(tuplev[1], 1.0, exp(tuplev[2]))]
        @test_broken jacobian(Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), tuplev) ≈
            ([tuplev[2], -sin(tuplev[1])], [tuplev[1], 1.0])
    end

    matrix = [2.7 3.1; 4.7 5.6]

    @testset "∂ scalar / ∂ matrix" begin
        @test gradient(Forward, x->x[1,1]*x[1,2]+x[2,1]*x[2,2], matrix) ≃
            (matrix[1,2], matrix[2,2], matrix[1,1], matrix[2,1])
        @test gradient(Reverse, x->x[1,1]*x[1,2]+x[2,1]*x[2,2], matrix) ≈
            [matrix[1,2] matrix[1,1]; matrix[2,2] matrix[2,1]]
        @test jacobian(Forward, x->x[1,1]*x[1,2]+x[2,1]*x[2,2], matrix) ≈
            [matrix[1,2] matrix[1,1]; matrix[2,2] matrix[2,1]]
        @test jacobian(Reverse, x->x[1,1]*x[1,2]+x[2,1]*x[2,2], matrix) ≈
            [matrix[1,2] matrix[1,1]; matrix[2,2] matrix[2,1]]
    end

    @testset "∂ vector / ∂ matrix" begin
        @test gradient(Forward, x->[x[1,1]*x[1,2],x[2,1]*x[2,2]], matrix) ≃
            ([matrix[1,2], 0.0], [0.0, matrix[2,2]], [matrix[1,1], 0.0], [0.0, matrix[2,1]])
        @test_broken gradient(Reverse, x->[x[1,1]*x[1,2],x[2,1]*x[2,2]], matrix)
        # again we can't use array construction syntax because of 1.6
        @test jacobian(Forward, x->[x[1,1]*x[1,2],x[2,1]*x[2,2]], matrix) ≈
            mkarray((2,2,2), matrix[1,2], 0.0, 0.0, matrix[2,2], matrix[1,1], 0.0, 0.0, matrix[2,1])
        @test jacobian(Reverse, x->[x[1,1]*x[1,2],x[2,1]*x[2,2]], matrix) ≈
            mkarray((2,2,2), matrix[1,2], 0.0, 0.0, matrix[2,2], matrix[1,1], 0.0, 0.0, matrix[2,1])
    end

    @testset "∂ tuple / ∂ matrix" begin
        @test gradient(Forward, x->(x[1,1]*x[1,2],x[2,1]*x[2,2]), matrix) ≃
            ((matrix[1,2], 0.0), (0.0, matrix[2,2]), (matrix[1,1], 0.0), (0.0, matrix[2,1]))
        @test_broken gradient(Reverse, x->(x[1,1]*x[1,2],x[2,1]*x[2,2]), matrix)
        @test jacobian(Forward, x->(x[1,1]*x[1,2],x[2,1]*x[2,2]), matrix) ≃
            [(matrix[1,2],0.0) (matrix[1,1],0.0); (0.0,matrix[2,2]) (0.0,matrix[2,1])]
        @test_broken jacobian(Reverse, x->(x[1,1]*x[1,2],x[2,1]*x[2,2]), matrix)
    end

    mkarray3 = x -> mkarray((2,2), x[1,1]*x[1,2], exp(x[1,1])+x[2,2], x[2,1]*x[2,2], sin(x[1,2])+x[2,1])

    @testset "∂ matrix / ∂ matrix" begin
        @test gradient(Forward, mkarray3, matrix) ≃
            ([matrix[1,2] 0.0; exp(matrix[1,1]) 0.0], [0.0 matrix[2,2]; 0.0 1.0],
            [matrix[1,1] 0.0; 0.0 cos(matrix[1,2])], [0.0 matrix[2,1]; 1.0 0.0])
        @test_broken gradient(Reverse, mkarray3, matrix)
        # array construction syntax broken on 1.6
        @test jacobian(Forward, mkarray3, matrix) ≈
            mkarray((2,2,2,2), matrix[1,2],exp(matrix[1,1]),0.0,0.0,0.0,0.0,matrix[2,2],1.0,
                    matrix[1,1],0.0,0.0,cos(matrix[1,2]),0.0,1.0,matrix[2,1],0.0)
        @test jacobian(Reverse, mkarray3, matrix) ≈
            mkarray((2,2,2,2), matrix[1,2],exp(matrix[1,1]),0.0,0.0,0.0,0.0,matrix[2,2],1.0,
                    matrix[1,1],0.0,0.0,cos(matrix[1,2]),0.0,1.0,matrix[2,1],0.0)
    end

    @testset "∂ tuple / ∂ matrix" begin
        @test gradient(Forward, x->OutStruct(x[1,1]*x[1,2],x[2,1]*x[2,2], exp(x[1,1])+x[2,2]), matrix) ≃
            (OutStruct(matrix[1,2], 0.0, exp(matrix[1,1])), OutStruct(0.0, matrix[2,2], 0.0),
            OutStruct(matrix[1,1], 0.0, 0.0), OutStruct(0.0, matrix[2,1], 1.0))
        @test_broken gradient(Reverse, x->OutStruct(x[1,1]*x[1,2],x[2,1]*x[2,2], exp(x[1,1])+x[2,2]), matrix)
        @test jacobian(Forward, x->OutStruct(x[1,1]*x[1,2],x[2,1]*x[2,2], exp(x[1,1])+x[2,2]), matrix) ≃ 
            [OutStruct(matrix[1,2],0.0, exp(matrix[1,1])) OutStruct(matrix[1,1],0.0,0.0);
            OutStruct(0.0,matrix[2,2],0.0) OutStruct(0.0,matrix[2,1], 1.0)]
        @test_broken jacobian(Reverse, x->OutStruct(x[1,1]*x[1,2],x[2,1]*x[2,2], exp(x[1,1])+x[2,2]), matrix)
    end


    istruct = InpStruct(2.7, 3.1, 4.7)

    @testset "∂ scalar / ∂ struct" begin
        @test_broken gradient(Forward, x -> x.i1 * x.i2 + x.i3, istruct)
        @test gradient(Reverse, x -> x.i1 * x.i2 + x.i3, istruct) ≃ InpStruct(istruct.i2, istruct.i1, 1.0)
        @test_broken jacobian(Forward, x -> x.i1 * x.i2 + x.i3, istruct)
        @test jacobian(Reverse, x -> x.i1 * x.i2 + x.i3, istruct) ≃ InpStruct(istruct.i2, istruct.i1, 1.0)
    end

    @testset "∂ vector / ∂ struct" begin
        @test_broken gradient(Forward, x -> [x.i1 * x.i2, cos(x.i3) + x.i1], istruct)
        @test_broken gradient(Reverse, x -> [x.i1 * x.i2, cos(x.i3) + x.i1], istruct)
        @test_broken jacobian(Forward, x -> [x.i1 * x.i2, cos(x.i3) + x.i1], istruct)
        @test jacobian(Reverse, x -> [x.i1 * x.i2, cos(x.i3) + x.i1], istruct) ≃
            [InpStruct(istruct.i2, istruct.i1, 0.0), InpStruct(1.0, 0.0, -sin(istruct.i3))]
    end

    @testset "∂ tuple / ∂ struct" begin
        @test_broken gradient(Forward, x -> (x.i1 * x.i2, cos(x.i3) + x.i1), istruct)
        @test_broken gradient(Reverse, x -> (x.i1 * x.i2, cos(x.i3) + x.i1), istruct)
        @test_broken jacobian(Forward, x -> (x.i1 * x.i2, cos(x.i3) + x.i1), istruct)
        @test_broken jacobian(Reverse, x -> (x.i1 * x.i2, cos(x.i3) + x.i1), istruct)
    end

    mkarray4 = x -> mkarray((2,2), x.i1*x.i2, exp(x.i2), cos(x.i3)+x.i1, x.i1)

    @testset "∂ matrix / ∂ struct" begin
        @test_broken gradient(Forward, x -> [x.i1 * x.i2  cos(x.i3) + x.i1; exp(x.i2) x.i1], istruct)
        @test_broken gradient(Reverse, x -> [x.i1 * x.i2  cos(x.i3) + x.i1; exp(x.i2) x.i1], istruct)
        @test_broken jacobian(Forward, x -> [x.i1 * x.i2  cos(x.i3) + x.i1; exp(x.i2) x.i1], istruct)
        @test jacobian(Reverse, mkarray4, istruct) ≃
            [InpStruct(istruct.i2, istruct.i1, 0.0) InpStruct(1.0, 0.0, -sin(istruct.i3));
            InpStruct(0.0, exp(istruct.i2), 0.0) InpStruct(1.0, 0.0, 0.0)]
    end

    @testset "∂ struct / ∂ struct" begin
        @test_broken gradient(Forward, x -> OutStruct(x.i1 * x.i2, cos(x.i3) + x.i1, exp(x.i2)), istruct)
        @test_broken gradient(Reverse, x -> OutStruct(x.i1 * x.i2, cos(x.i3) + x.i1, exp(x.i2)), istruct)
        @test_broken jacobian(Forward, x -> OutStruct(x.i1 * x.i2, cos(x.i3) + x.i1, exp(x.i2)), istruct)
        @test_broken jacobian(Reverse, x -> OutStruct(x.i1 * x.i2, cos(x.i3) + x.i1, exp(x.i2)), istruct)
    end
end

@testset "Simple Jacobian" begin
    @test jacobian(Forward, x->2*x, 3.0) ≈ 2.0
    @test jacobian(Forward, x->[x, 2*x], 3.0) ≈ [1.0, 2.0]
    @test jacobian(Forward, x->sum(abs2, x), [2.0, 3.0]) ≈ [4.0, 6.0]

    @test jacobian(Forward, x->2*x, 3.0, Val(1)) ≈ 2.0
    @test jacobian(Forward, x->[x, 2*x], 3.0, Val(1)) ≈ [1.0, 2.0]
    @test jacobian(Forward, x->sum(abs2, x), [2.0, 3.0], Val(1)) ≈ [4.0, 6.0]

    @test jacobian(Forward, x->2*x, 3.0, Val(2)) ≈ 2.0
    @test jacobian(Forward, x->[x, 2*x], 3.0, Val(2)) ≈ [1.0, 2.0]
    @test jacobian(Forward, x->sum(abs2, x), [2.0, 3.0], Val(2)) ≈ [4.0, 6.0]

    @test jacobian(Reverse, x->[x, 2*x], 3.0, Val(2)) ≈ [1.0, 2.0]
    @test jacobian(Reverse, x->[x, 2*x], 3.0, Val(2), Val(1)) ≈ [1.0, 2.0]
    @test jacobian(Reverse, x->[x, 2*x], 3.0, Val(2), Val(2)) ≈ [1.0, 2.0]

    x = float.(reshape(1:6, 2, 3))

    fillabs2(x) = [sum(abs2, x), 10*sum(abs2, x), 100*sum(abs2, x), 1000*sum(abs2, x)]

    jac = jacobian(Forward, fillabs2, x)

    @test jac[1, :, :] ≈ [2.0 6.0 10.0; 4.0 8.0 12.0]
    @test jac[2, :, :] ≈ [20.0 60.0 100.0; 40.0 80.0 120.0]
    @test jac[3, :, :] ≈ [200.0 600.0 1000.0; 400.0 800.0 1200.0]
    @test jac[4, :, :] ≈ [2000.0 6000.0 10000.0; 4000.0 8000.0 12000.0]

    jac = jacobian(Forward, fillabs2, x, Val(1))

    @test jac[1, :, :] ≈ [2.0 6.0 10.0; 4.0 8.0 12.0]
    @test jac[2, :, :] ≈ [20.0 60.0 100.0; 40.0 80.0 120.0]
    @test jac[3, :, :] ≈ [200.0 600.0 1000.0; 400.0 800.0 1200.0]
    @test jac[4, :, :] ≈ [2000.0 6000.0 10000.0; 4000.0 8000.0 12000.0]

    jac = jacobian(Forward, fillabs2, x, Val(2))

    @test jac[1, :, :] ≈ [2.0 6.0 10.0; 4.0 8.0 12.0]
    @test jac[2, :, :] ≈ [20.0 60.0 100.0; 40.0 80.0 120.0]
    @test jac[3, :, :] ≈ [200.0 600.0 1000.0; 400.0 800.0 1200.0]
    @test jac[4, :, :] ≈ [2000.0 6000.0 10000.0; 4000.0 8000.0 12000.0]


    jac = jacobian(Reverse, fillabs2, x, Val(4), Val(1))

    @test jac[1, :, :] ≈ [2.0 6.0 10.0; 4.0 8.0 12.0]
    @test jac[2, :, :] ≈ [20.0 60.0 100.0; 40.0 80.0 120.0]
    @test jac[3, :, :] ≈ [200.0 600.0 1000.0; 400.0 800.0 1200.0]
    @test jac[4, :, :] ≈ [2000.0 6000.0 10000.0; 4000.0 8000.0 12000.0]

    jac = jacobian(Reverse, fillabs2, x, Val(4), Val(2))

    @test jac[1, :, :] ≈ [2.0 6.0 10.0; 4.0 8.0 12.0]
    @test jac[2, :, :] ≈ [20.0 60.0 100.0; 40.0 80.0 120.0]
    @test jac[3, :, :] ≈ [200.0 600.0 1000.0; 400.0 800.0 1200.0]
    @test jac[4, :, :] ≈ [2000.0 6000.0 10000.0; 4000.0 8000.0 12000.0]

    fillinpabs2(x) = [(x.i1*x.i1+x.i2*x.i2+x.i3*x.i3), 10*(x.i1*x.i1+x.i2*x.i2+x.i3*x.i3), 100*(x.i1*x.i1+x.i2*x.i2+x.i3*x.i3), 1000*(x.i1*x.i1+x.i2*x.i2+x.i3*x.i3)]

    x2 = InpStruct(1.0, 2.0, 3.0)

    jac = jacobian(Reverse, fillinpabs2, x2, Val(4), Val(1))

    @test jac[1] == InpStruct(2.0, 4.0, 6.0)
    @test jac[2] == InpStruct(20.0, 40.0, 60.0)
    @test jac[3] == InpStruct(200.0, 400.0, 600.0)
    @test jac[4] == InpStruct(2000.0, 4000.0, 6000.0)

    jac = jacobian(Reverse, fillinpabs2, x2, Val(4), Val(2))

    @test jac[1] == InpStruct(2.0, 4.0, 6.0)
    @test jac[2] == InpStruct(20.0, 40.0, 60.0)
    @test jac[3] == InpStruct(200.0, 400.0, 600.0)
    @test jac[4] == InpStruct(2000.0, 4000.0, 6000.0)

    filloutabs2(x) = OutStruct(sum(abs2, x), 10*sum(abs2, x), 100*sum(abs2, x))

    jac = jacobian(Forward, filloutabs2, x)

    @test jac[1, 1] == OutStruct(2.0, 20.0, 200.0)
    @test jac[2, 1] == OutStruct(4.0, 40.0, 400.0)

    @test jac[1, 2] == OutStruct(6.0, 60.0, 600.0)
    @test jac[2, 2] == OutStruct(8.0, 80.0, 800.0)

    @test jac[1, 3] == OutStruct(10.0, 100.0, 1000.0)
    @test jac[2, 3] == OutStruct(12.0, 120.0, 1200.0)

    jac = jacobian(Forward, filloutabs2, x, Val(1))

    @test jac[1, 1] == OutStruct(2.0, 20.0, 200.0)
    @test jac[2, 1] == OutStruct(4.0, 40.0, 400.0)

    @test jac[1, 2] == OutStruct(6.0, 60.0, 600.0)
    @test jac[2, 2] == OutStruct(8.0, 80.0, 800.0)

    @test jac[1, 3] == OutStruct(10.0, 100.0, 1000.0)
    @test jac[2, 3] == OutStruct(12.0, 120.0, 1200.0)

    jac = jacobian(Forward, filloutabs2, x, Val(2))

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

    jac = jacobian(Reverse, inout, [2.0, 3.0], #=n_outs=# Val(3), Val(1))
    @test size(jac) == (3, 2)
    @test jac ≈ [ 0.0   1.0;
                  4.0   0.0;
                  12.0  0.0]

    jac = jacobian(Forward, inout, [2.0, 3.0], Val(1))
    @test size(jac) == (3, 2)
    @test jac ≈ [ 0.0   1.0;
                  4.0   0.0;
                  12.0  0.0]

    @test jac == jacobian(Forward, inout, [2.0, 3.0])

    jac = jacobian(Reverse, inout, [2.0, 3.0], #=n_outs=# Val(3), Val(2))
    @test size(jac) == (3, 2)
    @test jac ≈ [ 0.0   1.0;
                  4.0   0.0;
                  12.0  0.0]

    jac = jacobian(Forward, inout, [2.0, 3.0], Val(2))
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

    J_r_1(A, x) = jacobian(Reverse, θ -> f_test_1(A, θ), x, Val(5))
    J_r_2(A, x) = jacobian(Reverse, θ -> f_test_2(A, θ), x, Val(5))
    J_r_3(u, A, x) = jacobian(Reverse, θ -> f_test_3!(u, A, θ), x, Val(5))

    J_f_1(A, x) = jacobian(Forward, Const(θ -> f_test_1(A, θ)), x)
    J_f_2(A, x) = jacobian(Forward, Const(θ -> f_test_2(A, θ)), x)
    J_f_3(u, A, x) = jacobian(Forward, Const(θ -> f_test_3!(u, A, θ)), x)

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

