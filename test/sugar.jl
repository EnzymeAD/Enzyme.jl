using Enzyme, Test
using LinearAlgebra

mul_scalar(x, y) = x[1]*y[2] + x[2]*y[1]
mul_vector(x, y) = [x[1]*y[2], x[2]*y[1]]

@testset "Forward Multi-Arg Gradient" begin
	res = gradient(Forward, mul_scalar, [2.0, 3.0], [2.7, 3.1])
	@test res[1] ≈ [3.1, 2.7]
	@test res[2] ≈ [3.0, 2.0]

	res = gradient(Forward, mul_scalar, [2.0, 3.0], [2.7, 3.1]; chunk=Val(1))
    @test res[1] ≈ [3.1, 2.7]
	@test res[2] ≈ [3.0, 2.0]

	res = gradient(Forward, mul_scalar, [2.0, 3.0], [2.7, 3.1]; chunk=Val(2))
	@test res[1] ≈ [3.1, 2.7]
	@test res[2] ≈ [3.0, 2.0]

	res = gradient(ForwardWithPrimal, mul_scalar, [2.0, 3.0], [2.7, 3.1])
	@test res.val ≈ mul_scalar([2.0, 3.0], [2.7, 3.1])
	@test res.derivs[1] ≈ [3.1, 2.7]
	@test res.derivs[2] ≈ [3.0, 2.0]

	res = gradient(ForwardWithPrimal, mul_scalar, [2.0, 3.0], [2.7, 3.1]; chunk=Val(1))
	@test res.val ≈ mul_scalar([2.0, 3.0], [2.7, 3.1])
	@test res.derivs[1] ≈ [3.1, 2.7]
	@test res.derivs[2] ≈ [3.0, 2.0]

	res = gradient(ForwardWithPrimal, mul_scalar, [2.0, 3.0], [2.7, 3.1]; chunk=Val(2))
	@test res.val ≈ mul_scalar([2.0, 3.0], [2.7, 3.1])
	@test res.derivs[1] ≈ [3.1, 2.7]
	@test res.derivs[2] ≈ [3.0, 2.0]



	res = gradient(Forward, mul_scalar, Const([2.0, 3.0]), [2.7, 3.1])
	@test res[1] == nothing
	@test res[2] ≈ [3.0, 2.0]

	res = gradient(Forward, mul_scalar, Const([2.0, 3.0]), [2.7, 3.1]; chunk=Val(1))
	@test res[1] == nothing
	@test res[2] ≈ [3.0, 2.0]

	res = gradient(Forward, mul_scalar, Const([2.0, 3.0]), [2.7, 3.1]; chunk=Val(2))
	@test res[1] == nothing
	@test res[2] ≈ [3.0, 2.0]

	res = gradient(ForwardWithPrimal, mul_scalar, Const([2.0, 3.0]), [2.7, 3.1])
	@test res.val ≈ mul_scalar([2.0, 3.0], [2.7, 3.1])
	@test res.derivs[1] == nothing
	@test res.derivs[2] ≈ [3.0, 2.0]

	res = gradient(ForwardWithPrimal, mul_scalar, Const([2.0, 3.0]), [2.7, 3.1]; chunk=Val(1))
	@test res.val ≈ mul_scalar([2.0, 3.0], [2.7, 3.1])
	@test res.derivs[1] == nothing
	@test res.derivs[2] ≈ [3.0, 2.0]

	res = gradient(ForwardWithPrimal, mul_scalar, Const([2.0, 3.0]), [2.7, 3.1]; chunk=Val(2))
	@test res.val ≈ mul_scalar([2.0, 3.0], [2.7, 3.1])
	@test res.derivs[1] == nothing
	@test res.derivs[2] ≈ [3.0, 2.0]


	res = gradient(Forward, mul_scalar, [2.0, 3.0], Const([2.7, 3.1]))
	@test res[1] ≈ [3.1, 2.7]
	@test res[2] == nothing

	res = gradient(Forward, mul_scalar, [2.0, 3.0], Const([2.7, 3.1]); chunk=Val(1))
	@test res[1] ≈ [3.1, 2.7]
	@test res[2] == nothing

	res = gradient(Forward, mul_scalar, [2.0, 3.0], Const([2.7, 3.1]); chunk=Val(2))
	@test res[1] ≈ [3.1, 2.7]
	@test res[2] == nothing

	res = gradient(ForwardWithPrimal, mul_scalar, [2.0, 3.0], Const([2.7, 3.1]))
	@test res.val ≈ mul_scalar([2.0, 3.0], [2.7, 3.1])
	@test res.derivs[1] ≈ [3.1, 2.7]
	@test res.derivs[2] == nothing

	res = gradient(ForwardWithPrimal, mul_scalar, [2.0, 3.0], Const([2.7, 3.1]); chunk=Val(1))
	@test res.val ≈ mul_scalar([2.0, 3.0], [2.7, 3.1])
	@test res.derivs[1] ≈ [3.1, 2.7]
	@test res.derivs[2] == nothing

	res = gradient(ForwardWithPrimal, mul_scalar, [2.0, 3.0], Const([2.7, 3.1]); chunk=Val(2))
	@test res.val ≈ mul_scalar([2.0, 3.0], [2.7, 3.1])
	@test res.derivs[1] ≈ [3.1, 2.7]
	@test res.derivs[2] == nothing



	res = gradient(Forward, mul_vector, [2.0, 3.0], [2.7, 3.1])
	@test res[1] ≈ [3.1 0.0; 0.0 2.7]
	@test res[2] ≈ [0.0 2.0; 3.0 0.0]

    res = gradient(Forward, mul_vector, [2.0, 3.0], [2.7, 3.1]; chunk=Val(1))
    @test res[1] ≈ [3.1 0.0; 0.0 2.7]
    @test res[2] ≈ [0.0 2.0; 3.0 0.0]

    res = gradient(Forward, mul_vector, [2.0, 3.0], [2.7, 3.1]; chunk=Val(2))
    @test res[1] ≈ [3.1 0.0; 0.0 2.7]
    @test res[2] ≈ [0.0 2.0; 3.0 0.0]

    res = gradient(ForwardWithPrimal, mul_vector, [2.0, 3.0], [2.7, 3.1])
    @test res.val ≈ mul_vector([2.0, 3.0], [2.7, 3.1])
    @test res.derivs[1] ≈ [3.1 0.0; 0.0 2.7]
    @test res.derivs[2] ≈ [0.0 2.0; 3.0 0.0]

    res = gradient(ForwardWithPrimal, mul_vector, [2.0, 3.0], [2.7, 3.1]; chunk=Val(1))
    @test res.val ≈ mul_vector([2.0, 3.0], [2.7, 3.1])
    @test res.derivs[1] ≈ [3.1 0.0; 0.0 2.7]
    @test res.derivs[2] ≈ [0.0 2.0; 3.0 0.0]

    res = gradient(ForwardWithPrimal, mul_vector, [2.0, 3.0], [2.7, 3.1]; chunk=Val(2))
    @test res.val ≈ mul_vector([2.0, 3.0], [2.7, 3.1])
    @test res.derivs[1] ≈ [3.1 0.0; 0.0 2.7]
    @test res.derivs[2] ≈ [0.0 2.0; 3.0 0.0]

    

    res = gradient(Forward, mul_vector, Const([2.0, 3.0]), [2.7, 3.1])
    @test res[1] == nothing
    @test res[2] ≈ [0.0 2.0; 3.0 0.0]

    res = gradient(Forward, mul_vector, Const([2.0, 3.0]), [2.7, 3.1]; chunk=Val(1))
    @test res[1] == nothing
    @test res[2] ≈ [0.0 2.0; 3.0 0.0]

    res = gradient(Forward, mul_vector, Const([2.0, 3.0]), [2.7, 3.1]; chunk=Val(2))
    @test res[1] == nothing
    @test res[2] ≈ [0.0 2.0; 3.0 0.0]

    res = gradient(ForwardWithPrimal, mul_vector, Const([2.0, 3.0]), [2.7, 3.1])
    @test res.val ≈ mul_vector([2.0, 3.0], [2.7, 3.1])
    @test res.derivs[1] == nothing
    @test res.derivs[2] ≈ [0.0 2.0; 3.0 0.0]

    res = gradient(ForwardWithPrimal, mul_vector, Const([2.0, 3.0]), [2.7, 3.1]; chunk=Val(1))
    @test res.val ≈ mul_vector([2.0, 3.0], [2.7, 3.1])
    @test res.derivs[1] == nothing
    @test res.derivs[2] ≈ [0.0 2.0; 3.0 0.0]

    res = gradient(ForwardWithPrimal, mul_vector, Const([2.0, 3.0]), [2.7, 3.1]; chunk=Val(2))
    @test res.val ≈ mul_vector([2.0, 3.0], [2.7, 3.1])
    @test res.derivs[1] == nothing
    @test res.derivs[2] ≈ [0.0 2.0; 3.0 0.0]


    res = gradient(Forward, mul_vector, [2.0, 3.0], Const([2.7, 3.1]))
    @test res[1] ≈ [3.1 0.0; 0.0 2.7]
    @test res[2] == nothing

    res = gradient(Forward, mul_vector, [2.0, 3.0], Const([2.7, 3.1]); chunk=Val(1))
    @test res[1] ≈ [3.1 0.0; 0.0 2.7]
    @test res[2] == nothing

    res = gradient(Forward, mul_vector, [2.0, 3.0], Const([2.7, 3.1]); chunk=Val(2))
    @test res[1] ≈ [3.1 0.0; 0.0 2.7]
    @test res[2] == nothing

    res = gradient(ForwardWithPrimal, mul_vector, [2.0, 3.0], Const([2.7, 3.1]))
    @test res.val ≈ mul_vector([2.0, 3.0], [2.7, 3.1])
    @test res.derivs[1] ≈ [3.1 0.0; 0.0 2.7]
    @test res.derivs[2] == nothing

    res = gradient(ForwardWithPrimal, mul_vector, [2.0, 3.0], Const([2.7, 3.1]); chunk=Val(1))
    @test res.val ≈ mul_vector([2.0, 3.0], [2.7, 3.1])
    @test res.derivs[1] ≈ [3.1 0.0; 0.0 2.7]
    @test res.derivs[2] == nothing

    res = gradient(ForwardWithPrimal, mul_vector, [2.0, 3.0], Const([2.7, 3.1]); chunk=Val(2))
    @test res.val ≈ mul_vector([2.0, 3.0], [2.7, 3.1])
    @test res.derivs[1] ≈ [3.1 0.0; 0.0 2.7]
    @test res.derivs[2] == nothing

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

# symbol is \simeq
# this is basically a more flexible version of ≈
(≃)(a, b) = (≈)(a, b)
(≃)(a::Tuple, b::Tuple) = all(xy -> xy[1] ≃ xy[2], zip(a,b))
function (≃)(a::AbstractArray{<:Tuple}, b::AbstractArray{<:Tuple})
    size(a) == size(b) || return false
    all(xy -> xy[1] ≃ xy[2], zip(a,b))
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

    # ∂ scalar / ∂ scalar
    @test Enzyme.gradient(Enzyme.Forward, x -> x^2, scalar)[1] ≈ 6.0
    @test Enzyme.gradient(Enzyme.Reverse, x -> x^2, scalar)[1] ≈ 6.0
    @test Enzyme.jacobian(Enzyme.Forward, x -> x^2, scalar)[1] ≈ 6.0
    @test Enzyme.jacobian(Enzyme.Reverse, x -> x^2, scalar)[1] ≈ 6.0
    @test Enzyme.gradient(Enzyme.Forward, x -> 2*x, scalar)[1] ≈ 2.0
    @test Enzyme.gradient(Enzyme.Reverse, x -> 2*x, scalar)[1] ≈ 2.0
    @test Enzyme.jacobian(Enzyme.Forward, x -> 2*x, scalar)[1] ≈ 2.0
    @test Enzyme.jacobian(Enzyme.Reverse, x -> 2*x, scalar)[1] ≈ 2.0

    # ∂ vector / ∂ scalar
    @test Enzyme.gradient(Enzyme.Forward, x -> [2*x, x^2], scalar)[1] ≈ [2.0, 6.0]
    @test_broken Enzyme.gradient(Enzyme.Reverse, x -> [2*x, x^2], scalar)[1] ≈ [2.0, 6.0]

    @test Enzyme.jacobian(Enzyme.Forward, x -> [2*x, x^2], scalar)[1] ≈ [2.0, 6.0]
    @test Enzyme.jacobian(Enzyme.Reverse, x -> [2*x, x^2], scalar)[1] ≈ [2.0, 6.0]


    # ∂ tuple / ∂ scalar
    @test Enzyme.gradient(Enzyme.Forward, x -> (2*x, x^2), scalar)[1] ≃ (2.0, 6.0)
    @test_broken Enzyme.gradient(Enzyme.Reverse, x -> (2*x, x^2), scalar)[1] ≈ [2.0, 6.0]

    @test Enzyme.jacobian(Enzyme.Forward, x -> (2*x, x^2), scalar)[1] ≃ (2.0, 6.0)
    @test_broken Enzyme.jacobian(Enzyme.Reverse, x -> (2*x, x^2), scalar)[1] ≃ (2.0, 6.0)

    mkarray1 = x -> mkarray((2,2),2*x,sin(x),x^2,exp(x))

    # ∂ matrix / ∂ scalar
    @test Enzyme.gradient(Enzyme.Forward, mkarray1, scalar)[1] ≈ [2.0 6.0; cos(scalar) exp(scalar)]
    @test_broken Enzyme.gradient(Enzyme.Reverse, mkarray1, scalar)[1] ≈ [2.0 6.0; cos(scalar) exp(scalar)]

    @test Enzyme.jacobian(Enzyme.Forward, mkarray1, scalar)[1] ≈ [2.0 6.0; cos(scalar) exp(scalar)]
    @test Enzyme.jacobian(Enzyme.Reverse, mkarray1, scalar)[1] ≈ [2.0 6.0; cos(scalar) exp(scalar)]

    # ∂ struct / ∂ scalar
    @test Enzyme.gradient(Enzyme.Forward, x -> OutStruct(x, x^2, x^3), scalar)[1] == OutStruct(1.0,2*scalar,3*scalar^2)
    @test_broken Enzyme.gradient(Enzyme.Reverse, x -> InpStruct(x, x^2, x^3), scalar)[1] == (OutStruct(1.0,2.0,3.0),)
    @test Enzyme.jacobian(Enzyme.Forward, x -> OutStruct(x, x^2, x^3), scalar)[1] == OutStruct(1.0,2*scalar,3*scalar^2)
    @test_broken Enzyme.jacobian(Enzyme.Reverse, x -> InpStruct(x, x^2, x^3), scalar)[1] == (OutStruct(1.0,2.0,3.0),)



    vector = [2.7, 3.1]

    # ∂ scalar / ∂ vector
    @test Enzyme.gradient(Enzyme.Forward, x -> x[1] * x[2], vector)[1] ≈ [vector[2],vector[1]]
    @test Enzyme.gradient(Enzyme.Reverse, x -> x[1] * x[2], vector)[1] ≈ [vector[2], vector[1]]
    @test Enzyme.jacobian(Enzyme.Forward, x -> x[1] * x[2], vector)[1] ≈ [vector[2], vector[1]]
    @test Enzyme.jacobian(Enzyme.Reverse, x -> x[1] * x[2], vector)[1] ≈ [vector[2], vector[1]]


    # ∂ vector / ∂ vector
    @test Enzyme.gradient(Enzyme.Forward, x -> [x[1] * x[2], cos(x[1]) + x[2]], vector)[1] ≈
                        [vector[2] vector[1]; -sin(vector[1])  1.0]
    @test_broken Enzyme.gradient(Enzyme.Reverse, x -> [x[1] * x[2], cos(x[1]) + x[2]], vector)[1] ≈
                        [vector[2] vector[1]; -sin(vector[1])  1.0]
    @test Enzyme.jacobian(Enzyme.Forward, x -> [x[1] * x[2], cos(x[1]) + x[2]], vector)[1] ≈
                        [vector[2] vector[1]; -sin(vector[1])  1.0]
    @test Enzyme.jacobian(Enzyme.Reverse, x -> [x[1] * x[2], cos(x[1]) + x[2]], vector)[1] ≈
                        [vector[2] vector[1]; -sin(vector[1])  1.0]

    # ∂ tuple / ∂ vector
    @test Enzyme.gradient(Enzyme.Forward, x -> (x[1] * x[2], cos(x[1]) + x[2]), vector)[1] ≃
        [(vector[2], -sin(vector[1])), (vector[1], 1.0)]
    @test_broken Enzyme.gradient(Enzyme.Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), vector)[1] ≈
        ([vector[2], -sin(vector[1])], [vector[1], 1.0])
    @test Enzyme.jacobian(Enzyme.Forward, x -> (x[1] * x[2], cos(x[1]) + x[2]), vector)[1] ≃
        [(vector[2], -sin(vector[1])), (vector[1], 1.0)]
    @test_broken Enzyme.jacobian(Enzyme.Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), vector)[1]

    mkarray2 = x -> mkarray((2,2), x[1]*x[2], exp(x[2]), cos(x[1])+x[2], x[1])

    # ∂ matrix / ∂ vector
    @test Enzyme.gradient(Enzyme.Forward, mkarray2, vector)[1] ≈
        mkarray((2,2,2), vector[2], 0.0, -sin(vector[1]), 1.0, vector[1], exp(vector[2]), 1.0, 0.0)
    @test_broken Enzyme.gradient(Enzyme.Reverse, mkarray2, vector)[1]
    @test Enzyme.jacobian(Enzyme.Forward, mkarray2, vector)[1] ≈
        mkarray((2,2,2), vector[2], 0.0, -sin(vector[1]), 1.0, vector[1], exp(vector[2]), 1.0, 0.0)
    @test Enzyme.jacobian(Enzyme.Reverse, mkarray2, vector)[1] ≈
        mkarray((2,2,2), vector[2], 0.0, -sin(vector[1]), 1.0, vector[1], exp(vector[2]), 1.0, 0.0)

    # ∂ struct / ∂ vector
    @test Enzyme.gradient(Enzyme.Forward, x -> OutStruct(x[1] * x[2], cos(x[1]) + x[2], exp(x[2])), vector)[1] ≃
        [OutStruct(vector[2], -sin(vector[1]), 0.0), OutStruct(vector[1], 1.0, exp(vector[2]))]
    @test_broken Enzyme.gradient(Enzyme.Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), vector)[1] ≈ ([vector[2], -sin(vector[1])], [vector[1], 1.0])

    @test Enzyme.jacobian(Enzyme.Forward, x -> OutStruct(x[1] * x[2], cos(x[1]) + x[2], exp(x[2])), vector)[1] ≃
        [OutStruct(vector[2], -sin(vector[1]), 0.0), OutStruct(vector[1], 1.0, exp(vector[2]))]
    @test_broken Enzyme.jacobian(Enzyme.Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), vector)[1] ≈ ([vector[2], -sin(vector[1])], [vector[1], 1.0])



    tuplev = (2.7, 3.1)

    # ∂ scalar / ∂ tuple
    @test Enzyme.gradient(Enzyme.Forward, x -> x[1] * x[2], tuplev)[1] ≃ (tuplev[2],tuplev[1])
    @test Enzyme.gradient(Enzyme.Reverse, x -> x[1] * x[2], tuplev)[1] ≃ (tuplev[2],tuplev[1])
    @test Enzyme.jacobian(Enzyme.Forward, x -> x[1] * x[2], tuplev)[1] ≃ (tuplev[2],tuplev[1])
    @test Enzyme.jacobian(Enzyme.Reverse, x -> x[1] * x[2], tuplev)[1] ≃ (tuplev[2],tuplev[1])

    # ∂ vector / ∂ tuple
    @test Enzyme.gradient(Enzyme.Forward, x -> [x[1] * x[2], cos(x[1]) + x[2]], tuplev)[1] ≃
        ([tuplev[2], -sin(tuplev[1])], [tuplev[1], 1.0])
    @test_broken Enzyme.gradient(Enzyme.Reverse, x -> [x[1] * x[2], cos(x[1]) + x[2]], tuplev)[1] ≈ ([tuplev[2], -sin(tuplev[1])], [tuplev[1], 1.0])
    @test_broken Enzyme.jacobian(Enzyme.Forward, x -> [x[1] * x[2], cos(x[1]) + x[2]], tuplev)[1] ≈
                        [tuplev[2] tuplev[1]; -sin(tuplev[1])  1.0]
    @test Enzyme.jacobian(Enzyme.Reverse, x -> [x[1] * x[2], cos(x[1]) + x[2]], tuplev)[1] ≃
        [(tuplev[2], tuplev[1]), (-sin(tuplev[1]), 1.0)]

    # ∂ tuple / ∂ tuple
    @test Enzyme.gradient(Enzyme.Forward, x -> (x[1] * x[2], cos(x[1]) + x[2]), tuplev)[1] ≃
        ((vector[2], -sin(vector[1])), (vector[1], 1.0))
    @test_broken Enzyme.gradient(Enzyme.Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), tuplev)[1] ≈ ([tuplev[2], -sin(tuplev[1])], [tuplev[1], 1.0])
    @test Enzyme.jacobian(Enzyme.Forward, x -> (x[1] * x[2], cos(x[1]) + x[2]), tuplev)[1] ≃
        ((tuplev[2], -sin(tuplev[1])), (tuplev[1], 1.0))
    @test_broken Enzyme.jacobian(Enzyme.Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), tuplev)[1] ≈
                        [tuplev[2] tuplev[1]; -sin(tuplev[1])  1.0]

    # ∂ matrix / ∂ tuple
    @test Enzyme.gradient(Enzyme.Forward, mkarray2, tuplev)[1] ≃
        ([tuplev[2] -sin(tuplev[1]); 0.0 1.0], [tuplev[1] 1.0; exp(tuplev[2]) 0.0])
    @test_broken Enzyme.gradient(Enzyme.Reverse, mkarray2, tuplev)[1]
    @test_broken Enzyme.jacobian(Enzyme.Forward, mkarray2, tuplev)[1] ≈
                        [tuplev[2] -sin(tuplev[1]); 0.0 1.0;;; tuplev[1] 1.0;  exp(tuplev[2]) 0.0]
    @test_broken Enzyme.jacobian(Enzyme.Reverse, x -> mkarray2, tuplev)[1] ≈
                        [tuplev[2] -sin(tuplev[1]); 0.0 1.0;;; tuplev[1] 1.0;  exp(tuplev[2]) 0.0]

    # ∂ struct / ∂ tuple
    @test Enzyme.gradient(Enzyme.Forward, x -> OutStruct(x[1] * x[2], cos(x[1]) + x[2], exp(x[2])), tuplev)[1] ≃
        (OutStruct(tuplev[2], -sin(tuplev[1]), 0.0), OutStruct(tuplev[1], 1.0, exp(tuplev[2])))
    @test_broken Enzyme.gradient(Enzyme.Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), tuplev)[1] ≈ ([tuplev[2], -sin(tuplev[1])], [tuplev[1], 1.0])

    @test_broken Enzyme.jacobian(Enzyme.Forward, x -> OutStruct(x[1] * x[2], cos(x[1]) + x[2], exp(x[2])), tuplev)[1] ≃
        [OutStruct(tuplev[2], -sin(tuplev[1]), 0.0), OutStruct(tuplev[1], 1.0, exp(tuplev[2]))]
    @test_broken Enzyme.jacobian(Enzyme.Reverse, x -> (x[1] * x[2], cos(x[1]) + x[2]), tuplev)[1] ≈ ([tuplev[2], -sin(tuplev[1])], [tuplev[1], 1.0])



    matrix = [2.7 3.1; 4.7 5.6]

    # ∂ scalar / ∂ matrix
    @test Enzyme.gradient(Enzyme.Forward, x->x[1,1]*x[1,2]+x[2,1]*x[2,2], matrix)[1] ≈ [matrix[1,2] matrix[1,1]; matrix[2,2] matrix[2,1]]
    @test Enzyme.gradient(Enzyme.Reverse, x->x[1,1]*x[1,2]+x[2,1]*x[2,2], matrix)[1] ≈ [matrix[1,2] matrix[1,1]; matrix[2,2] matrix[2,1]]
    @test Enzyme.jacobian(Enzyme.Forward, x->x[1,1]*x[1,2]+x[2,1]*x[2,2], matrix)[1] ≈ [matrix[1,2] matrix[1,1]; matrix[2,2] matrix[2,1]]
    @test Enzyme.jacobian(Enzyme.Reverse, x->x[1,1]*x[1,2]+x[2,1]*x[2,2], matrix)[1] ≈ [matrix[1,2] matrix[1,1]; matrix[2,2] matrix[2,1]]

    # ∂ vector / ∂ matrix
    @test Enzyme.gradient(Enzyme.Forward, x->[x[1,1]*x[1,2],x[2,1]*x[2,2]], matrix)[1] ≈
        mkarray((2,2,2), matrix[1,2], 0.0, 0.0, matrix[2,2], matrix[1,1], 0.0, 0.0, matrix[2,1])
    @test_broken Enzyme.gradient(Enzyme.Reverse, x->[x[1,1]*x[1,2],x[2,1]*x[2,2]], matrix)[1]
    # again we can't use array construction syntax because of 1.6
    @test Enzyme.jacobian(Enzyme.Forward, x->[x[1,1]*x[1,2],x[2,1]*x[2,2]], matrix)[1] ≈
        mkarray((2,2,2), matrix[1,2], 0.0, 0.0, matrix[2,2], matrix[1,1], 0.0, 0.0, matrix[2,1])
    @test Enzyme.jacobian(Enzyme.Reverse, x->[x[1,1]*x[1,2],x[2,1]*x[2,2]], matrix)[1] ≈
        mkarray((2,2,2), matrix[1,2], 0.0, 0.0, matrix[2,2], matrix[1,1], 0.0, 0.0, matrix[2,1])

    # ∂ tuple / ∂ matrix
    @test Enzyme.gradient(Enzyme.Forward, x->(x[1,1]*x[1,2],x[2,1]*x[2,2]), matrix)[1] ≃ 
        [(matrix[1,2],0.0) (matrix[1,1],0.0); (0.0,matrix[2,2]) (0.0,matrix[2,1])]
    @test_broken Enzyme.gradient(Enzyme.Reverse, x->(x[1,1]*x[1,2],x[2,1]*x[2,2]), matrix)
    @test Enzyme.jacobian(Enzyme.Forward, x->(x[1,1]*x[1,2],x[2,1]*x[2,2]), matrix)[1] ≃
        [(matrix[1,2],0.0) (matrix[1,1],0.0); (0.0,matrix[2,2]) (0.0,matrix[2,1])]
    @test_broken Enzyme.jacobian(Enzyme.Reverse, x->(x[1,1]*x[1,2],x[2,1]*x[2,2]), matrix)[1]

    mkarray3 = x -> mkarray((2,2), x[1,1]*x[1,2], exp(x[1,1])+x[2,2], x[2,1]*x[2,2], sin(x[1,2])+x[2,1])

    # ∂ matrix / ∂ matrix
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

    # ∂ tuple / ∂ matrix
    @test Enzyme.gradient(Enzyme.Forward, x->OutStruct(x[1,1]*x[1,2],x[2,1]*x[2,2], exp(x[1,1])+x[2,2]), matrix)[1] ≃
        [OutStruct(matrix[1,2],0.0, exp(matrix[1,1])) OutStruct(matrix[1,1],0.0,0.0); OutStruct(0.0,matrix[2,2],0.0) OutStruct(0.0,matrix[2,1], 1.0)]
    @test_broken Enzyme.gradient(Enzyme.Reverse, x->OutStruct(x[1,1]*x[1,2],x[2,1]*x[2,2], exp(x[1,1])+x[2,2]), matrix)[1]
    @test Enzyme.jacobian(Enzyme.Forward, x->OutStruct(x[1,1]*x[1,2],x[2,1]*x[2,2], exp(x[1,1])+x[2,2]), matrix)[1] ≃ 
        [OutStruct(matrix[1,2],0.0, exp(matrix[1,1])) OutStruct(matrix[1,1],0.0,0.0); OutStruct(0.0,matrix[2,2],0.0) OutStruct(0.0,matrix[2,1], 1.0)]
    @test_broken Enzyme.jacobian(Enzyme.Reverse, x->OutStruct(x[1,1]*x[1,2],x[2,1]*x[2,2], exp(x[1,1])+x[2,2]), matrix)[1]


    istruct = InpStruct(2.7, 3.1, 4.7)

    # ∂ scalar / ∂ struct
    @test_broken Enzyme.gradient(Enzyme.Forward, x -> x.i1 * x.i2 + x.i3, istruct)[1]
    @test Enzyme.gradient(Enzyme.Reverse, x -> x.i1 * x.i2 + x.i3, istruct)[1] ≃ InpStruct(istruct.i2, istruct.i1, 1.0)
    @test_broken Enzyme.jacobian(Enzyme.Forward, x -> x.i1 * x.i2 + x.i3, istruct)[1]
    @test Enzyme.jacobian(Enzyme.Reverse, x -> x.i1 * x.i2 + x.i3, istruct)[1] ≃ InpStruct(istruct.i2, istruct.i1, 1.0)

    # ∂ vector / ∂ struct
    @test_broken Enzyme.gradient(Enzyme.Forward, x -> [x.i1 * x.i2, cos(x.i3) + x.i1], istruct)[1]
    @test_broken Enzyme.gradient(Enzyme.Reverse, x -> [x.i1 * x.i2, cos(x.i3) + x.i1], istruct)[1]
    @test_broken Enzyme.jacobian(Enzyme.Forward, x -> [x.i1 * x.i2, cos(x.i3) + x.i1], istruct)[1]
    @test Enzyme.jacobian(Enzyme.Reverse, x -> [x.i1 * x.i2, cos(x.i3) + x.i1], istruct)[1] ≃ [InpStruct(istruct.i2, istruct.i1, 0.0), InpStruct(1.0, 0.0, -sin(istruct.i3))]

    # ∂ tuple / ∂ struct
    @test_broken Enzyme.gradient(Enzyme.Forward, x -> (x.i1 * x.i2, cos(x.i3) + x.i1), istruct)[1]
    @test_broken Enzyme.gradient(Enzyme.Reverse, x -> (x.i1 * x.i2, cos(x.i3) + x.i1), istruct)[1]
    @test_broken Enzyme.jacobian(Enzyme.Forward, x -> (x.i1 * x.i2, cos(x.i3) + x.i1), istruct)[1]
    @test_broken Enzyme.jacobian(Enzyme.Reverse, x -> (x.i1 * x.i2, cos(x.i3) + x.i1), istruct)[1]

    mkarray4 = x -> mkarray((2,2), x.i1*x.i2, exp(x.i2), cos(x.i3)+x.i1, x.i1)

    # ∂ matrix / ∂ struct
    @test_broken Enzyme.gradient(Enzyme.Forward, x -> [x.i1 * x.i2  cos(x.i3) + x.i1; exp(x.i2) x.i1], istruct)[1]
    @test_broken Enzyme.gradient(Enzyme.Reverse, x -> [x.i1 * x.i2  cos(x.i3) + x.i1; exp(x.i2) x.i1], istruct)[1]
    @test_broken Enzyme.jacobian(Enzyme.Forward, x -> [x.i1 * x.i2  cos(x.i3) + x.i1; exp(x.i2) x.i1], istruct)[1]
    @test Enzyme.jacobian(Enzyme.Reverse, mkarray4, istruct)[1] ≃
        [InpStruct(istruct.i2, istruct.i1, 0.0) InpStruct(1.0, 0.0, -sin(istruct.i3));
        InpStruct(0.0, exp(istruct.i2), 0.0) InpStruct(1.0, 0.0, 0.0)]

    # ∂ struct / ∂ struct
    @test_broken Enzyme.gradient(Enzyme.Forward, x -> OutStruct(x.i1 * x.i2, cos(x.i3) + x.i1, exp(x.i2)), istruct)[1]
    @test_broken Enzyme.gradient(Enzyme.Reverse, x -> OutStruct(x.i1 * x.i2, cos(x.i3) + x.i1, exp(x.i2)), istruct)[1]
    @test_broken Enzyme.jacobian(Enzyme.Forward, x -> OutStruct(x.i1 * x.i2, cos(x.i3) + x.i1, exp(x.i2)), istruct)[1]
    @test_broken Enzyme.jacobian(Enzyme.Reverse, x -> OutStruct(x.i1 * x.i2, cos(x.i3) + x.i1, exp(x.i2)), istruct)[1]
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
