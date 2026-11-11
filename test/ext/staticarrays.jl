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

    f0 = x -> sum(2 * x)
    f1 = x -> @SVector Float64[x[2], 2 * x[2]]
    f2 = x -> @SMatrix Float64[x[2] x[1]; 2 * x[2] 2 * x[1]]

    x = @SVector Float64[1, 2]

    @inferred gradient(Forward, f0, x)
    dx = gradient(Forward, f0, x)[1]
    @test dx isa SVector
    @test dx == [2.0, 2.0]  # test to make sure conversion works
    @test gradient(Forward, f1, x)[1] isa SMatrix
    @test gradient(Forward, f1, x)[1] == [0 1.0; 0 2.0]
    @test Enzyme.jacobian(Forward, f2, x)[1] isa SArray
    @test Enzyme.jacobian(Forward, f2, x)[1] == reshape(Float64[0, 0, 1, 2, 1, 2, 0, 0], (2, 2, 2))

    x = @SMatrix Float64[1 2; 3 4]

    @inferred gradient(Forward, f0, x)
    dx = gradient(Forward, f0, x)[1]
    @test dx isa SMatrix
    @test dx == fill(2.0, (2, 2))
    @test gradient(Forward, f1, x)[1] isa SArray
    @test gradient(Forward, f1, x)[1] == reshape(Float64[0, 0, 1, 2, 0, 0, 0, 0], (2, 2, 2))
    @test Enzyme.jacobian(Forward, f2, x)[1] isa SArray
    @test Enzyme.jacobian(Forward, f2, x)[1] == reshape(
        Float64[0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], (2, 2, 2, 2),
    )

    x = @SVector Float64[1, 2]

    @inferred gradient(Reverse, f0, x)
    dx = gradient(Reverse, f0, x)[1]
    @test dx isa SVector
    @test dx == [2.0, 2.0]  # test to make sure conversion works
    @test_broken gradient(Reverse, f1, x)[1] isa SMatrix
    @test_broken gradient(Reverse, f1, x)[1] == [0 1.0; 0 2.0]
    @test_broken Enzyme.jacobian(Reverse, f2, x)[1] isa SArray
    @test_broken Enzyme.jacobian(Reverse, f2, x)[1] == reshape(Float64[0, 0, 1, 2, 1, 2, 0, 0], (2, 2, 2))

    x = @SMatrix Float64[1 2; 3 4]

    @test_broken gradient(Reverse, f1, x)[1] isa SArray
    @test_broken gradient(Reverse, f1, x)[1] == reshape(Float64[0, 0, 1, 2, 0, 0, 0, 0], (2, 2, 2))
    @test_broken Enzyme.jacobian(Reverse, f2, x)[1] isa SArray
    @test_broken Enzyme.jacobian(Reverse, f2, x)[1] == reshape(
        Float64[0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], (2, 2, 2, 2),
    )
end

function unstable_fun(A0)
    A = 'N' in ('H', 'h', 'S', 's') ? wrap(A0) : A0
    return (@inbounds A[1])::eltype(A0)
end
@testset "Type unstable static array index" begin
    inp = ones(SVector{2, Float64})
    res = Enzyme.gradient(Enzyme.Reverse, unstable_fun, inp)[1]
    @test res ≈ [1.0, 0.0]
    res = Enzyme.gradient(Enzyme.Forward, unstable_fun, inp)[1]
    @test res ≈ [1.0, 0.0]
end

function inner_forhess(x)
    return tanh.(x)
end

function for_hess(x)
    return sum(inner_forhess(x))
end

grad_forhess(x) = autodiff(Reverse, for_hess, Active, Active(x))[1][1]
hess(x) = jacobian(Forward, grad_forhess, x)[1]

@testset "StaticArrays hessian" begin
    # sum(tanh.(x)) has a diagonal hessian, with entries tanh''(xᵢ) = -2 tanh(xᵢ) (1 - tanh(xᵢ)^2)
    x = @SVector fill(0.5, 10)
    d = -2 * tanh(0.5) * (1 - tanh(0.5)^2)
    res = [i == j ? d : 0.0 for i in 1:10, j in 1:10]
    @test jacobian(Forward, grad_forhess, x)[1] ≈ res
end

abstract type AbstractBasisType1263 end
struct Contravariant1263 <: AbstractBasisType1263 end

struct CurvilinearBasisVectors1263{N, T, C, B, V <: AbstractBasisType1263} <: StaticMatrix{N, N, T}
    __x::Union{SMatrix{N, N, T}}
    function CurvilinearBasisVectors1263{N, T, C, B, V}(b::AbstractMatrix) where {N, T, C, B, V}
        return new{N, T, C, B, V}(SMatrix{N, N, T}(b))
    end
end

Base.@propagate_inbounds function Base.getindex(
        v::CurvilinearBasisVectors1263{N, T, C, B, V},
        i::Int,
    ) where {N, T, C, B, V}
    return view(getfield(v, :__x), i)[]
end

mutable struct MutableBasisHolder1263{N, T}
    __x::Union{SMatrix{N, N, T}}
end

const CBV3_1263 = CurvilinearBasisVectors1263{3, Float64, (:∇x, :∇y, :∇z), (:x, :y, :z), Contravariant1263}

@testset "Struct with single-member Union{SMatrix} field (#1263)" begin
    a = CBV3_1263(reshape(collect(1.0:9.0), 3, 3))
    da = CBV3_1263(zeros(3, 3))
    f_1263(x, y) = sum(sum(y .* x))

    res_dup = autodiff(Reverse, f_1263, Active, Duplicated(a, da), Active(2.0))
    @test res_dup[1][2] ≈ sum(a)

    da_ref = Ref(CBV3_1263(zeros(3, 3)))
    res_mdup = autodiff(Reverse, f_1263, Active, MixedDuplicated(a, da_ref), Active(2.0))
    @test res_mdup[1][2] ≈ sum(a)
    @test da_ref[].__x ≈ fill(2.0, 3, 3)

    grads = Enzyme.gradient(Reverse, f_1263, a, 2.0)
    @test grads[1].__x ≈ fill(2.0, 3, 3)
    @test grads[2] ≈ sum(a)

    mh = MutableBasisHolder1263{3, Float64}(SMatrix{3, 3, Float64}(reshape(collect(1.0:9.0), 3, 3)))
    dmh = MutableBasisHolder1263{3, Float64}(zeros(SMatrix{3, 3, Float64}))
    f_mut_1263(m, y) = y * sum(m.__x)
    res_mut = autodiff(Reverse, f_mut_1263, Active, Duplicated(mh, dmh), Active(2.0))
    @test res_mut[1][2] ≈ 45.0
    @test dmh.__x ≈ fill(2.0, 3, 3)

    res_fwd = autodiff(Forward, f_1263, Duplicated(a, CBV3_1263(ones(3, 3))), Const(2.0))
    @test res_fwd[1] ≈ 18.0
end
