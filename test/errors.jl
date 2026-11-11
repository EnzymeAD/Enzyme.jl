using Enzyme, Test
import LLVM, GPUCompiler

array_square(x) = 2 .* x

@testset "Array of Pointer Copy" begin
    @test_throws Enzyme.Compiler.EnzymeNonScalarReturnException Enzyme.gradient(Reverse, array_square, [2.0])
end


function sumsin(x)
    return sin(sum(x))
end

@testset "Incorrect thunk arguments" begin
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(sumsin)}, Active, Duplicated{Vector{Float64}})

    @test_throws Enzyme.Compiler.ThunkCallError fwd(Duplicated([1.0], [2.0]))

    @test_throws Enzyme.Compiler.ThunkCallError fwd(Const(sumsin), Duplicated([1.0], [2.0]), Active(3.14))
end

@testset "emit_error with a method instance" begin
    # Some errors carry the method instance they occur in (`EnzymeRuntimeExceptionMI`). For GPU
    # targets, and for conditional errors, `emit_error` instantiated that type with two type
    # parameters, which it does not have: a TypeError during compilation instead of the error.
    mi = GPUCompiler.methodinstance(typeof(sin), Tuple{Float64})
    world = Base.get_world_counter()
    @testset "$triple" for (triple, conditional) in (
            ("nvptx64-nvidia-cuda", false), ("amdgcn-amd-amdhsa", false), (Sys.MACHINE, true),
        )
        LLVM.Context() do ctx
            mod = LLVM.Module("m")
            LLVM.triple!(mod, triple)
            f = LLVM.Function(mod, "f", LLVM.FunctionType(LLVM.VoidType(), [LLVM.Int1Type()]))
            B = LLVM.IRBuilder()
            LLVM.position!(B, LLVM.BasicBlock(f, "entry"))
            cond = conditional ? LLVM.parameters(f)[1] : nothing
            Enzyme.Compiler.emit_error(B, nothing, ("Enzyme: example", mi, world), Enzyme.Compiler.EnzymeRuntimeExceptionMI, cond)
            fs = [LLVM.name(g) for g in LLVM.functions(mod)]
            if conditional
                # only the message is stored by the conditional throw
                @test "jl_conditional_throw_$(Enzyme.Compiler.EnzymeRuntimeException)" in fs
            else
                @test "gpu_report_exception" in fs
            end
        end
    end
    # A conditional error of a parametric type keeps its type, without method instance and world:
    # e.g. a missing derivative under runtime activity is still an `EnzymeNoDerivativeError`.
    LLVM.Context() do ctx
        mod = LLVM.Module("m")
        LLVM.triple!(mod, Sys.MACHINE)
        f = LLVM.Function(mod, "f", LLVM.FunctionType(LLVM.VoidType(), [LLVM.Int1Type()]))
        B = LLVM.IRBuilder()
        LLVM.position!(B, LLVM.BasicBlock(f, "entry"))
        # as `src/errors.jl` raises a missing derivative under runtime activity
        Enzyme.Compiler.emit_error(B, nothing, ("Enzyme: example", mi, world), Enzyme.Compiler.EnzymeNoDerivativeError{Core.MethodInstance, UInt}, LLVM.parameters(f)[1])
        fs = [LLVM.name(g) for g in LLVM.functions(mod)]
        @test "jl_conditional_throw_$(Enzyme.Compiler.EnzymeNoDerivativeError{Nothing, Nothing})" in fs
    end
end
