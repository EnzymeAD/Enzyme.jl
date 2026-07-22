using CUDA
using Enzyme
using LinearAlgebra: dot
using Test

@testset "CUDA memory copies" begin
    #= Exercise the reverse copy rules across CUDA's memory types. A host<->device
    roundtrip gradient must recover `2x`. =#
    grad_roundtrip = function (to_gpu)
        x = Float32[1, 2, 3]
        dx = zeros(Float32, 3)
        Enzyme.autodiff(
            Reverse,
            x -> sum(abs2, Array(to_gpu(x))),
            Active,
            Duplicated(x, dx),
        )
        return dx
    end

    @testset "device memory" begin
        @test grad_roundtrip(x -> cu(x)) == Float32[2, 4, 6]
    end
    # Unified/host memory: `pointer(::CuArray{…,Unified/Host})` is inferred as `Union{CuPtr,Ptr}`, so the `pointer` rule cannot yet return a concrete
    @testset "unified memory" begin
        @test_broken grad_roundtrip(x -> cu(x; unified = true)) == Float32[2, 4, 6]
    end
    @testset "host memory" begin
        @test_broken grad_roundtrip(x -> cu(x; host = true)) == Float32[2, 4, 6]
    end

    x = CuArray(Float32[1, 2, 3])
    original = copy(x)
    dx = CUDA.zeros(Float32, 3)
    Enzyme.autodiff(
        Reverse,
        x -> sum(abs2, Array(x)),
        Active,
        Duplicated(x, dx),
    )
    @test Array(x) == Array(original)
    @test Array(dx) == 2 .* Array(x)

    fill!(dx, 0)
    Enzyme.autodiff(
        Reverse,
        x -> sum(abs2, Array(copy(x))),
        Active,
        Duplicated(x, dx),
    )
    @test Array(x) == Array(original)
    @test Array(dx) == 2 .* Array(x)
end

@testset "CUDA array-memory copies" begin
    ER = Enzyme.EnzymeRules
    config = ER.RevConfig{true, true, 1, (false, false, false, false, false), false, false}()
    n = 3
    offset = sizeof(Float32)
    src = Float32[4, 5, 6]
    dsrc = Float32[10, 10, 10]
    seed = Float32[1, 2, 3]
    memory_src = Float32[99, 4, 5, 6]
    memory_dsrc = Float32[77, 10, 10, 10]
    memory_seed = Float32[88, 1, 2, 3]

    dest_mem = CUDA.alloc(CUDA.ArrayMemory{Float32}, (n + 1,))
    ddest_mem = CUDA.alloc(CUDA.ArrayMemory{Float32}, (n + 1,))
    src_mem = CUDA.alloc(CUDA.ArrayMemory{Float32}, (n + 1,))
    dsrc_mem = CUDA.alloc(CUDA.ArrayMemory{Float32}, (n + 1,))

    write_memory! = function (memory, values)
        GC.@preserve values begin
            Base.unsafe_copyto!(pointer(memory), 0, pointer(values), length(values))
        end
    end
    read_memory = function (memory)
        values = zeros(Float32, n + 1)
        GC.@preserve values begin
            Base.unsafe_copyto!(pointer(values), pointer(memory), 0, length(values))
        end
        return values
    end
    reverse_copy! = function (seed!, dest_ann, src_ann; doff = nothing, soff = nothing)
        args = if doff !== nothing
            (dest_ann, Const(doff), src_ann, Const(n))
        elseif soff !== nothing
            (dest_ann, src_ann, Const(soff), Const(n))
        else
            (dest_ann, src_ann, Const(n))
        end
        return_type = Duplicated{typeof(dest_ann.val)}
        ER.augmented_primal(
            config, Const(Base.unsafe_copyto!), return_type, args...,
        )
        seed!()
        ER.reverse(
            config, Const(Base.unsafe_copyto!), return_type, nothing, args...,
        )
    end

    try
        # Ptr -> CuArrayPtr
        dest = pointer(dest_mem)
        ddest = pointer(ddest_mem)
        GC.@preserve src dsrc begin
            dest_ann = Duplicated(dest, ddest)
            src_ann = Duplicated(pointer(src), pointer(dsrc))
            reverse_copy!(dest_ann, src_ann; doff = offset) do
                write_memory!(ddest_mem, memory_seed)
            end
        end
        @test dsrc == Float32[11, 12, 13]
        @test read_memory(ddest_mem) == Float32[88, 0, 0, 0]

        # CuArrayPtr -> Ptr
        write_memory!(src_mem, memory_src)
        write_memory!(dsrc_mem, memory_dsrc)
        dest = zeros(Float32, n)
        ddest = copy(seed)
        GC.@preserve dest ddest begin
            dest_ann = Duplicated(pointer(dest), pointer(ddest))
            src_ann = Duplicated(pointer(src_mem), pointer(dsrc_mem))
            reverse_copy!(dest_ann, src_ann; soff = offset) do
                copyto!(ddest, seed)
            end
        end
        @test dest == src
        @test read_memory(dsrc_mem) == Float32[77, 11, 12, 13]
        @test ddest == zeros(Float32, n)

        # CuPtr -> CuArrayPtr
        fill!(dsrc, 10)
        device_src = CuArray(src)
        device_dsrc = CuArray(dsrc)
        dest = pointer(dest_mem)
        ddest = pointer(ddest_mem)
        dest_ann = Duplicated(dest, ddest)
        src_ann = Duplicated(pointer(device_src), pointer(device_dsrc))
        reverse_copy!(dest_ann, src_ann; doff = offset) do
            write_memory!(ddest_mem, memory_seed)
        end
        @test Array(device_dsrc) == Float32[11, 12, 13]
        @test read_memory(ddest_mem) == Float32[88, 0, 0, 0]

        # CuArrayPtr -> CuPtr
        write_memory!(src_mem, memory_src)
        write_memory!(dsrc_mem, memory_dsrc)
        device_dest = CUDA.zeros(Float32, n)
        device_ddest = CuArray(seed)
        dest_ann = Duplicated(pointer(device_dest), pointer(device_ddest))
        src_ann = Duplicated(pointer(src_mem), pointer(dsrc_mem))
        reverse_copy!(dest_ann, src_ann; soff = offset) do
            copyto!(device_ddest, seed)
        end
        @test Array(device_dest) == src
        @test read_memory(dsrc_mem) == Float32[77, 11, 12, 13]
        @test Array(device_ddest) == zeros(Float32, n)
    finally
        CUDA.free(dest_mem)
        CUDA.free(ddest_mem)
        CUDA.free(src_mem)
        CUDA.free(dsrc_mem)
    end
end

function mul_kernel(A)
    i = threadIdx().x
    if i <= length(A)
        A[i] *= A[i]
    end
    return nothing
end

function grad_mul_kernel(A, dA)
    autodiff_deferred(Reverse, Const(mul_kernel), Const, Duplicated(A, dA))
    return nothing
end

@testset "mul_kernel" begin
    A = CUDA.ones(64,)
    @cuda threads=length(A) mul_kernel(A)
    A = CUDA.ones(64,)
    dA = similar(A)
    dA .= 1
    @cuda threads=length(A) grad_mul_kernel(A, dA)
    @test all(dA .== 2)
end

function exp_kernel(A)
    i = threadIdx().x
    if i <= length(A)
        A[i] = exp(A[i])
    end
    return nothing
end

function grad_exp_kernel(A, dA)
    autodiff_deferred(Reverse, Const(exp_kernel), Const, Duplicated(A, dA))
    return nothing
end

@testset "exp_kernel" begin
    A = CUDA.ones(64,)
    @cuda threads=length(A) exp_kernel(A)
    A = CUDA.ones(64,)
    dA = similar(A)
    dA .= 1
    @cuda threads=length(A) grad_exp_kernel(A, dA)
    @test all(dA .== exp(1.f0))
end

function cos_kernel(A)
    i = threadIdx().x
    if i <= length(A)
        A[i] = cos(A[i])
    end
    return nothing
end

function grad_cos_kernel(A, dA)
    autodiff_deferred(Reverse, Const(cos_kernel), Const, Duplicated(A, dA))
    return nothing
end

@testset "cos_kernel" begin
    A = CUDA.ones(64,)
    @cuda threads=length(A) cos_kernel(A)
    A = CUDA.ones(64,)
    dA = similar(A)
    dA .= 1
    @cuda threads=length(A) grad_cos_kernel(A, dA)
    @test all(dA .≈ -sin(1.f0))
end

function val_kernel!(_, ::Val{N}) where N
    return nothing
end

function dval_kernel!(du, ::Val{N}) where {N}
    autodiff_deferred(Reverse, Const(val_kernel!), Const, du, Const(Val(N)))
    return nothing
end

# Test for https://github.com/EnzymeAD/Enzyme.jl/issues/358
@testset "Test val kernel" begin
    n = 10
    u = CUDA.rand(n)
    dzdu = CUDA.rand(n)
    @cuda threads=4 dval_kernel!(Duplicated(u, dzdu), Val(n))
end

# https://github.com/EnzymeAD/Enzyme.jl/issues/367

relu(x) = ifelse(x<0, zero(x), x)
## Define CUDA kernel for dense layer
function dense!(
  feats_out, feats_in, W, b,
  ::Val{nfeat_out}, ::Val{nfeat_in}, ::Val{ndof}
) where {nfeat_out, nfeat_in, ndof}

  ## Each thread will update features for a single sample
  idof = (blockIdx().x - 1) * blockDim().x + threadIdx().x

  ## Prevent out-of-bounds array access
  (idof > ndof) && return nothing

  ## Compute `feats_out`
  for i = 1:nfeat_out
    for k = 1:nfeat_in
      feats_out[i, idof] += W[i, k] * feats_in[k, idof]
    end

    feats_out[i, idof] = relu(feats_out[i, idof] + b[i])
  end

  return nothing
end # dense!

## Wrapper for Enzyme to differentiate `dense!`
function ddense!(
  dfeats_out, dfeats_in, dW, db,
  ::Val{nfeat_out}, ::Val{nfeat_in}, ::Val{ndof}
) where {nfeat_out, nfeat_in, ndof}

  autodiff_deferred(
    Reverse,
    Const(dense!),
    Const,
    dfeats_out, dfeats_in, dW, db,
    Const(Val(nfeat_out)), Const(Val(nfeat_in)), Const(Val(ndof))
  )
  return nothing

end # ddense!

function call_ddense()
  nthread      = 32
  ndof         = 32
  nblock       = ceil(Int, ndof / nthread)
  nfeat_out    = 32
  nfeat_in     = 16
  feats_out    = zeros(nfeat_out, ndof)    |> cu
  feats_in     = rand(nfeat_in, ndof)      |> cu
  W            = rand(nfeat_out, nfeat_in) |> cu
  b            = rand(nfeat_out)           |> cu
  dzdfeats_out = rand(nfeat_out, ndof)     |> cu
  dfeats_out   = Duplicated(feats_out, dzdfeats_out)
  dzdfeats_in  = zero(feats_in)
  dfeats_in    = Duplicated(feats_in, dzdfeats_in)
  dzdW         = zero(W)
  dW           = Duplicated(W, dzdW)
  dzdb         = zero(b)
  db           = Duplicated(b, dzdb)

  @cuda threads=nthread blocks=nblock ddense!(
    dfeats_out, dfeats_in, dW, db,
    Val(nfeat_out), Val(nfeat_in), Val(ndof)
  )

end # call_ddense

@testset "DDense" begin
    call_ddense()
end

function square_kernel!(x)
    i = threadIdx().x
    x[i] *= x[i]
    sync_threads()
    return nothing
end

# basic squaring on GPU
function square!(x)
    @cuda blocks = 1 threads = length(x) square_kernel!(x)
    return nothing
end

@testset "Reverse Kernel" begin
    A = CUDA.rand(64)
    dA = CUDA.ones(64)
    A .= (1:1:64)
    dA .= 1
    Enzyme.autodiff(Reverse, square!, Duplicated(A, dA))
    @test all(dA .≈ (2:2:128))

    A = CUDA.rand(32)
    dA = CUDA.ones(32)
    dA2 = CUDA.ones(32)
    A .= (1:1:32)
    dA .= 1
    dA2 .= 3
    Enzyme.autodiff(Reverse, square!, BatchDuplicated(A, (dA, dA2)))
    @test all(dA .≈ (2:2:64))
    @test all(dA2 .≈ 3*(2:2:64))
end
