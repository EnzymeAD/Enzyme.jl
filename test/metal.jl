using Metal
using Enzyme
using Test

# 3-point stencil: every thread reads its neighbours, so the adjoint of thread `i` adds into
# x̄[i-1], x̄[i] and x̄[i+1], concurrently with its neighbours. Enzyme has to accumulate these
# with atomic adds on device memory.
function stencil!(y, x)
    i = Metal.thread_position_in_grid_1d()
    if 2 <= i <= length(x) - 1
        @inbounds y[i] = x[i - 1] - 2.0f0 * x[i] + x[i + 1] * x[i + 1]
    end
    return nothing
end

function ∇stencil!(y, ȳ, x, x̄)
    Enzyme.autodiff_deferred(Reverse, Const(stencil!), Const, Duplicated(y, ȳ), Duplicated(x, x̄))
    return nothing
end

@testset "Metal stencil adjoint" begin
    N = 1024
    x = Float32[sin(i) for i in 1:N]
    ȳ = Float32[cos(i) for i in 1:N]
    x̄ = zeros(Float32, N)
    for i in 2:(N - 1)
        x̄[i - 1] += ȳ[i]
        x̄[i] -= 2 * ȳ[i]
        x̄[i + 1] += 2 * x[i + 1] * ȳ[i]
    end

    x_d = MtlArray(x)
    x̄_d = Metal.zeros(Float32, N)
    Metal.@sync @metal threads = 256 groups = cld(N, 256) ∇stencil!(Metal.zeros(Float32, N), MtlArray(ȳ), x_d, x̄_d)
    @test Array(x̄_d) ≈ x̄
end

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

# `tape_type` with a Metal parent job gives the tape of the derivative that a kernel then compiles with
# `autodiff_deferred_thunk` (KernelAbstractions' pattern; the twin of "tape_type with a CUDA parent job,
# as the kernel's derivative" in cuda.jl). The parent job is built from Metal.jl's compiler config.
function exp_store!(x, y)
    x[1] = exp(y[1]) * y[1]
    return nothing
end
const exp_tape_mode = ReverseSplitModified(ReverseSplitWithPrimal, Val((true, true, true)))
function exp_aug!(tape, x, dx, y, dy, ::Val{TT}) where {TT}
    fwd, _ = autodiff_deferred_thunk(
        exp_tape_mode, TT, Const{typeof(exp_store!)}, Const{Nothing},
        Duplicated{typeof(x)}, Duplicated{typeof(y)}
    )
    tape[1] = fwd(Const(exp_store!), Duplicated(x, dx), Duplicated(y, dy))[1]
    return nothing
end
function exp_rev!(tape, x, dx, y, dy, ::Val{TT}) where {TT}
    _, rev = autodiff_deferred_thunk(
        exp_tape_mode, TT, Const{typeof(exp_store!)}, Const{Nothing},
        Duplicated{typeof(x)}, Duplicated{typeof(y)}
    )
    rev(Const(exp_store!), Duplicated(x, dx), Duplicated(y, dy), tape[1])
    return nothing
end
@testset "tape_type with a Metal parent job, as the kernel's derivative" begin
    GPUCompiler = Enzyme.Compiler.GPUCompiler
    mi = GPUCompiler.methodinstance(typeof(() -> return), Tuple{})
    job = GPUCompiler.CompilerJob(mi, Metal.compiler_config(Metal.device()))
    TT = Enzyme.tape_type(
        job, exp_tape_mode, Const{typeof(exp_store!)}, Const{Nothing},
        Duplicated{MtlDeviceVector{Float32, 1}}, Duplicated{MtlDeviceVector{Float32, 1}}
    )
    x = Metal.zeros(Float32, 1); dx = Metal.ones(Float32, 1); y = MtlArray(Float32[0.7]); dy = Metal.zeros(Float32, 1)
    tape = MtlArray{TT}(undef, 1)
    Metal.@sync @metal exp_aug!(tape, x, dx, y, dy, Val(TT))
    Metal.@sync @metal exp_rev!(tape, x, dx, y, dy, Val(TT))
    @test Array(dy)[1] ≈ exp(0.7f0) * 1.7f0
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
