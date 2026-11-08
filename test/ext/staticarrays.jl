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

@testset "SVector \\ SMatrix (#1805)" begin
    f_rdiv(v, m) = sum(v / m)
    v = @SVector [1.0, 2.0, 3.0]
    m = @SMatrix [2.0;;]

    dv, dm = autodiff(Reverse, f_rdiv, Active, Active(v), Active(m))[1]
    @test dv isa SVector{3, Float64}
    @test dm isa SMatrix{1, 1, Float64, 1}
    @test dv ≈ @SVector [0.5, 0.5, 0.5]
    @test dm ≈ @SMatrix [-1.5;;]

    dv_dir = @SVector [1.0, -0.5, 2.0]
    dm_dir = @SMatrix [0.25;;]
    dfwd = autodiff(Forward, f_rdiv, Duplicated(v, dv_dir), Duplicated(m, dm_dir))[1]
    @test dfwd ≈ sum(dv .* dv_dir) + sum(dm .* dm_dir)

    @test jacobian(Forward, x -> x / m, v)[1] ≈ [0.5 0.0 0.0; 0.0 0.5 0.0; 0.0 0.0 0.5]

    v0 = @SVector [0.0, 0.0, 0.0]
    m0 = @SMatrix [1.0;;]
    @test_throws Enzyme.Compiler.EnzymeNonScalarReturnException autodiff(
        Reverse, /, Active, Active(v0), Active(m0)
    )

    f_ldiv_adj(m, v) = sum(m \ v')
    dm_adj, dv_adj = autodiff(Reverse, f_ldiv_adj, Active, Active(m), Active(v))[1]
    @test dv_adj ≈ @SVector [0.5, 0.5, 0.5]
    @test dm_adj ≈ @SMatrix [-1.5;;]

    f_ldiv(v, m) = sum(v \ m)
    m32 = @SMatrix [1.0 2.0; 3.0 4.0; 5.0 6.0]
    dv_l, dm_l = autodiff(Reverse, f_ldiv, Active, Active(v), Active(m32))[1]
    @test dv_l ≈ @SVector [-29 / 98, -51 / 98, -73 / 98]
    @test dm_l ≈ @SMatrix [1 / 14 1 / 14; 2 / 14 2 / 14; 3 / 14 3 / 14]
    dm32_dir = @SMatrix [0.5 -1.0; 0.25 0.75; -0.5 1.5]
    dfwd_l = autodiff(Forward, f_ldiv, Duplicated(v, dv_dir), Duplicated(m32, dm32_dir))[1]
    @test dfwd_l ≈ sum(dv_l .* dv_dir) + sum(dm_l .* dm32_dir)

    f_rdiv_adj(m, v) = sum(m / v')
    m23 = @SMatrix [1.0 2.0 3.0; 4.0 5.0 6.0]
    dm_r, dv_r = autodiff(Reverse, f_rdiv_adj, Active, Active(m23), Active(v))[1]
    @test dm_r ≈ @SMatrix [1 / 14 2 / 14 3 / 14; 1 / 14 2 / 14 3 / 14]
    @test dv_r ≈ @SVector [-11 / 98, -43 / 98, -75 / 98]
end
