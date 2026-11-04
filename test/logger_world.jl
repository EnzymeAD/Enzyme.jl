using Enzyme, Test, Logging

# When Enzyme differentiates code that itself calls `autodiff`, the generated `thunk`
# compiles the inner derivative in place, in the world `thunk` was defined in. On Julia
# 1.10 that compilation binds a scoped value, which asks the task's logger for its level.
# A logger type defined after Enzyme, as Test's `TestLogger` is when Test loads after
# Enzyme, has no methods in that world. The nested derivative then failed with
# `EnzymeNoDerivativeError` ("indirect call through an inactive function pointer"), for
# the rest of the process, under any logger.

struct LateLogger <: Logging.AbstractLogger
    records::Vector{Any}
end
Logging.min_enabled_level(::LateLogger) = Logging.Warn
Logging.shouldlog(::LateLogger, args...) = true
Logging.handle_message(logger::LateLogger, args...; kwargs...) = (push!(logger.records, args); nothing)
Logging.catch_exceptions(::LateLogger) = false

# Each derivative is first compiled under the logger, so each needs functions of its own.
late_f(x) = sum(abs2, x) * x[1] + exp(x[2])
function late_gradient(x)
    dx = zero(x)
    autodiff(Reverse, late_f, Active, Duplicated(x, dx))
    return dx
end
late_hvp(x, v) = autodiff(Forward, late_gradient, Duplicated(x, v))[1]

late_g(x) = x[1]^3 * x[2] + sin(x[2])
late_tangent(x, v) = autodiff(Forward, late_g, Duplicated(x, v))[1]
function late_hvp_rev(x, v)
    dx = zero(x)
    autodiff(Reverse, late_tangent, Active, Duplicated(x, dx), Const(v))
    return dx
end

late_h(x) = x^4 + cos(x)
late_derivative(x) = autodiff(Reverse, late_h, Active, Active(x))[1][1]

@testset "Nested derivatives first compiled under a logger defined after Enzyme" begin
    logger = LateLogger(Any[])
    x = [1.0, 2.0, 3.0]
    v = [1.0, 0.0, 0.0]
    # the Hessian of `late_f` times v = e₁: (6x₁, 2x₂, 2x₃)
    @test with_logger(() -> late_hvp(x, v), logger) == [6.0, 4.0, 6.0]
    @test late_hvp(x, v) == [6.0, 4.0, 6.0]
    @test with_logger(() -> late_hvp(x, v), logger) == [6.0, 4.0, 6.0]

    y = [1.5, 0.5]
    w = [0.0, 1.0]
    # the Hessian of `late_g` times w = e₂: (3y₁², -sin(y₂))
    @test with_logger(() -> late_hvp_rev(y, w), logger) ≈ [3 * 1.5^2, -sin(0.5)]
    @test late_hvp_rev(y, w) ≈ [3 * 1.5^2, -sin(0.5)]

    # a second derivative through a first-order derivative compiled under the logger
    @test with_logger(() -> late_derivative(0.7), logger) ≈ 4 * 0.7^3 - sin(0.7)
    @test with_logger(() -> autodiff(Forward, late_derivative, Duplicated(0.7, 1.0))[1], logger) ≈
        12 * 0.7^2 - cos(0.7)

    @test isempty(logger.records)
end

# As the global logger, such a type broke every derivative on Julia 1.10, first-order ones
# included: compiling or inferring inside a generated function asked it for its level.
global_f(x) = x^3 + sin(x)
global_g(x) = sum(abs2, x) * x[1] + exp(x[2])
function global_gradient(x)
    dx = zero(x)
    autodiff(Reverse, global_g, Active, Duplicated(x, dx))
    return dx
end

@testset "Derivatives first compiled under a global logger defined after Enzyme" begin
    logger = LateLogger(Any[])
    old = global_logger(logger)
    try
        @test autodiff(Reverse, global_f, Active, Active(0.7))[1][1] ≈ 3 * 0.7^2 + cos(0.7)
        @test autodiff(Forward, global_f, Duplicated(0.7, 1.0))[1] ≈ 3 * 0.7^2 + cos(0.7)
        @test autodiff(Forward, global_gradient, Duplicated([1.0, 2.0, 3.0], [1.0, 0.0, 0.0]))[1] ==
            [6.0, 4.0, 6.0]
    finally
        global_logger(old)
    end
    @test isempty(logger.records)
end
