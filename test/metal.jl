using Metal
using Enzyme
using Test

function fun_cpu!(A, B, a)
    for ix in axes(A, 1)
        A[ix] += a * B[ix] * Float32(100.65)
    end
    return nothing
end

function fun_gpu!(A, B, a)
    ix = Metal.thread_position_in_grid_1d()
    A[ix] += a * B[ix] * Float32(100.65)
    return nothing
end

function ∇_fun_cpu!(A, Ā, B, B̄, a)
    Enzyme.autodiff_deferred(Reverse, Const(fun_cpu!), Const, DuplicatedNoNeed(A, Ā), DuplicatedNoNeed(B, B̄), Const(a))
    return nothing
end

function ∇_fun_gpu!(A_d, Ā_d, B_d, B̄_d, a)
    Enzyme.autodiff_deferred(Reverse, Const(fun_gpu!), Const, Duplicated(A_d, Ā_d), Duplicated(B_d, B̄_d), Const(a))
    return nothing
end

@testset "Metal autodiff" begin
    N = 16

    A = rand(Float32, N)
    B = rand(Float32, N)
    a = Float32(6.5)

    A_d = MtlArray(copy(A))
    B_d = MtlArray(copy(B))

    Ā = ones(Float32, size(A))
    B̄ = ones(Float32, size(B))
    Ā_d = Metal.ones(Float32, size(A_d))
    B̄_d = Metal.ones(Float32, size(B_d))

    ∇_fun_cpu!(A, Ā, B, B̄, a)

    @sync @metal threads = N groups = 1 ∇_fun_gpu!(A_d, Ā_d, B_d, B̄_d, a)

    @test Array(Ā_d) ≈ Ā
    @test Array(B̄_d) ≈ B̄

end

# Metal.jl implements the Float32 math functions with Metal's library, `air.<name>.f32`
# (`air.fast_<name>.f32` under `@fastmath`).
function math!(y, x, f)
    i = Metal.thread_position_in_grid_1d()
    @inbounds y[i] = f(x[i])
    return nothing
end

function ∇math!(y, ȳ, x, x̄, f)
    Enzyme.autodiff_deferred(Reverse, Const(math!), Const, Duplicated(y, ȳ), Duplicated(x, x̄), Const(f))
    return nothing
end

function fwd_math!(y, ẏ, x, ẋ, f)
    Enzyme.autodiff_deferred(Forward, Const(math!), Const, Duplicated(y, ẏ), Duplicated(x, ẋ), Const(f))
    return nothing
end

function metal_math_gradient(f, x)
    N = length(x)
    x_d = MtlArray(x)
    y_d = Metal.zeros(Float32, N)
    x̄_d = Metal.zeros(Float32, N)
    Metal.@sync @metal threads = N ∇math!(y_d, Metal.ones(Float32, N), x_d, x̄_d, f)
    return Array(y_d), Array(x̄_d)
end

function metal_math_tangent(f, x)
    N = length(x)
    ẏ_d = Metal.zeros(Float32, N)
    Metal.@sync @metal threads = N fwd_math!(Metal.zeros(Float32, N), ẏ_d, MtlArray(x), Metal.ones(Float32, N), f)
    return Array(ẏ_d)
end

cpu_gradient(f, x) = Enzyme.autodiff(Reverse, f, Active, Active(x))[1][1]

# `sincos` is `air.sincos.f32`, which returns the sine and writes the cosine through a pointer.
# libEnzyme has no rule for that form: Enzyme.jl splits the call into `air.sin.f32` and
# `air.cos.f32` (`split_gpu_sincos!`).
@testset "Metal sincos" begin
    x = collect(range(0.1f0, 0.9f0; length = 256))
    @testset "$name" for (name, f, rtol) in (
            ("sincos", x -> sum(sincos(x)), sqrt(eps(Float32))),
            ("sincos (cos only)", x -> last(sincos(x)), sqrt(eps(Float32))),
            # Metal's fast variants trade accuracy for speed
            ("fastmath sincos", x -> sum(@fastmath(sincos(x))), 1.0f-3),
        )
        y, x̄ = metal_math_gradient(f, x)
        @test isapprox(y, f.(x); rtol)
        @test isapprox(x̄, cpu_gradient.(f, x); rtol)
        @test isapprox(metal_math_tangent(f, x), cpu_gradient.(f, x); rtol)
    end
end
