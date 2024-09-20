using Enzyme, Test


function f_ip(x, tmp)
    tmp .= x ./ 2
    return dot(tmp, x)
end

function f_gradient_deferred!(dx, x, tmp)
    dtmp = make_zero(tmp)
    autodiff_deferred(Reverse, Const(f_ip), Active, Duplicated(x, dx), Duplicated(tmp, dtmp))
    return nothing
end

function f_hvp!(hv, x, v, tmp)
    dx = make_zero(x)
    btmp = make_zero(tmp)
    autodiff(
        Forward,
        f_gradient_deferred!,
        Duplicated(dx, hv),
        Duplicated(x, v),
        Duplicated(tmp, btmp),
    )
    return nothing
end

@testset "Hessian" begin
    function origf(x::Array{Float64}, y::Array{Float64})
        y[1] = x[1] * x[1] + x[2] * x[1]
        return nothing
    end

    function grad(x, dx, y, dy)
      Enzyme.autodiff(Reverse, Const(origf), Duplicated(x, dx), DuplicatedNoNeed(y, dy))
      nothing
    end

    x = [2.0, 2.0]
    y = Vector{Float64}(undef, 1)
    dx = [0.0, 0.0]
    dy = [1.0]

    grad(x, dx, y, dy)

    vx = ([1.0, 0.0], [0.0, 1.0])
    hess = ([0.0, 0.0], [0.0, 0.0])
    dx2 = [0.0, 0.0]
    dy = [1.0]

    Enzyme.autodiff(Enzyme.Forward, grad,
                    Enzyme.BatchDuplicated(x, vx),
                    Enzyme.BatchDuplicated(dx2, hess),
                    Const(y),
                    Const(dy))

    @test dx ≈ dx2
    @test hess[1][1] ≈ 2.0
    @test hess[1][2] ≈ 1.0
    @test hess[2][1] ≈ 1.0
    @test hess[2][2] ≈ 0.0

    x = [1.0]
    v = [-1.0]
    hv = make_zero(v)
    tmp = similar(x)

    #WARN: this fails with an assertion error from somewhere deep inside enzyme/compiler and I 
    # have absolutely no idea why
    # definitely fails when file is run in isolation, not sure about otherwise
    f_hvp!(hv, x, v, tmp)
    @test hv ≈ [-1.0]
end

