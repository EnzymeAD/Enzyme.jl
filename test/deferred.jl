using Enzyme, Test
using LLVM
import GPUCompiler

# GPUCompiler compiles each deferred job of a kernel to its own module, and links the modules
# into the module of the kernel. Since julia#60031, the counters in Julia's function names are
# local to the module, so the derivatives of two deferred jobs of one function have the same
# names. The tests compile the kernels as the GPU backends do, with a host target.

function deferred_store!(x, y)
    x[1] = exp(y[1]) * y[1]
    return nothing
end

const deferred_split_mode = ReverseSplitModified(ReverseSplitWithPrimal, Val((true, true, true)))

# The two halves of one split derivative, in one kernel (KernelAbstractions uses two kernels).
function deferred_split_kernel!(x, dx, y, dy, ::Val{TT}) where {TT}
    fwd, rev = autodiff_deferred_thunk(
        deferred_split_mode, TT, Const{typeof(deferred_store!)}, Const{Nothing},
        Duplicated{typeof(x)}, Duplicated{typeof(y)}
    )
    tape = fwd(Const(deferred_store!), Duplicated(x, dx), Duplicated(y, dy))[1]
    rev(Const(deferred_store!), Duplicated(x, dx), Duplicated(y, dy), tape)
    return nothing
end

# Two derivatives of one function that differ only in the activity of the arguments.
function deferred_activity_kernel!(x, dx, y, dy)
    autodiff_deferred(Reverse, Const(deferred_store!), Const, Duplicated(x, dx), Duplicated(y, dy))
    autodiff_deferred(Reverse, Const(deferred_store!), Const, Duplicated(x, dx), Const(y))
    return nothing
end

function deferred_kernel_ir(f, tt)
    world = Base.get_world_counter()
    mi = GPUCompiler.methodinstance(typeof(f), tt, world)
    config = GPUCompiler.CompilerConfig(
        Enzyme.Compiler.DefaultCompilerTarget(),
        Enzyme.Compiler.PrimalCompilerParams(Enzyme.API.DEM_ReverseModeCombined);
        kernel = false, validate = false,
    )
    job = GPUCompiler.CompilerJob(mi, config, world)
    return GPUCompiler.JuliaContext() do _
        mod, meta = GPUCompiler.compile(:llvm, job)
        LLVM.verify(mod)
        ir = string(mod)
        LLVM.dispose(mod)
        ir
    end
end

@testset "Deferred jobs of one function in one kernel" begin
    V = Vector{Float32}
    TT = Enzyme.tape_type(
        deferred_split_mode, Const{typeof(deferred_store!)}, Const{Nothing},
        Duplicated{V}, Duplicated{V}
    )
    ir = deferred_kernel_ir(deferred_split_kernel!, Tuple{V, V, V, V, Val{TT}})
    @test !occursin("deferred_codegen", ir)

    ir = deferred_kernel_ir(deferred_activity_kernel!, Tuple{V, V, V, V})
    @test !occursin("deferred_codegen", ir)
end
