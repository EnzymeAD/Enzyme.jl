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

# Indexing through the `CartesianIndices` of a 4-D array computes its `length`, the product of the four sizes. After
# differentiation, Enzyme ran LLVM's vectorizers on the kernel, which GPUCompiler doesn't do for Metal, and the SLP
# vectorizer turned that product into `llvm.vector.reduce.mul.v4i64`, which Apple's back-end can't compile.
function cartesian4!(A)
    I = CartesianIndices(size(A))[Metal.thread_position_in_grid_1d()]
    A[I] = 2.0f0 * A[I]
    return nothing
end
∇cartesian4!(A, Ā) = (autodiff_deferred(Reverse, Const(cartesian4!), Const, Duplicated(A, Ā)); nothing)

@testset "Metal reverse mode through a 4-D CartesianIndices" begin
    A = MtlArray(rand(Float32, 2, 2, 2, 2))
    Ā = Metal.ones(Float32, 2, 2, 2, 2)
    @metal threads = 16 ∇cartesian4!(A, Ā)
    @test all(==(2.0f0), Array(Ā))
end
