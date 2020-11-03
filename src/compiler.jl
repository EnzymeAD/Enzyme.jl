module Compiler

import ..Enzyme: Const, Active, Duplicated, DuplicatedNoNeed

using LLVM, GPUCompiler, Libdl
import Enzyme_jll

import GPUCompiler: CompilerJob, FunctionSpec, codegen
using LLVM.Interop
import LLVM: Target, TargetMachine

# We have one global JIT and TM
const jit = Ref{OrcJIT}()
const tm  = Ref{TargetMachine}()

function __init__()
    opt_level = Base.JLOptions().opt_level
    if opt_level < 2
        optlevel = LLVM.API.LLVMCodeGenLevelNone
    elseif opt_level == 2
        optlevel = LLVM.API.LLVMCodeGenLevelDefault
    else
        optlevel = LLVM.API.LLVMCodeGenLevelAggressive
    end

    tm[] = GPUCompiler.JITTargetMachine(optlevel=optlevel)
    LLVM.asm_verbosity!(tm[], true)

    jit[] = OrcJIT(tm[]) # takes ownership of tm
    atexit() do
        dispose(jit[])
    end
end

# Define EnzymeTarget
Base.@kwdef struct EnzymeTarget <: AbstractCompilerTarget
end
GPUCompiler.llvm_triple(::EnzymeTarget) = Sys.MACHINE

# GPUCompiler.llvm_datalayout(::EnzymeTarget) =  nothing

function GPUCompiler.llvm_machine(::EnzymeTarget)
    return tm[]
end

module Runtime
    # the runtime library
    signal_exception() = return
    malloc(sz) = Base.Libc.malloc(sz)
    report_oom(sz) = return
    report_exception(ex) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return
end


struct EnzymeCompilerParams <: AbstractCompilerParams end

## job

# TODO: We shouldn't blancket opt-out
GPUCompiler.check_invocation(job::CompilerJob{EnzymeTarget}, entry::LLVM.Function) = nothing

GPUCompiler.runtime_module(target::CompilerJob{EnzymeTarget}) = Runtime
GPUCompiler.isintrinsic(::CompilerJob{EnzymeTarget}, fn::String) = true
GPUCompiler.can_throw(::CompilerJob{EnzymeTarget}) = true

# TODO: encode debug build or not in the compiler job
#       https://github.com/JuliaGPU/CUDAnative.jl/issues/368
GPUCompiler.runtime_slug(job::CompilerJob{EnzymeTarget}) = "enzyme"

include("compiler/optimize.jl")
include("compiler/cassette.jl")

"""
Create the `FunctionSpec` pair, and lookup the primal return type.
"""
@inline function fspec(f::F, tt::TT) where {F, TT}
    # Entry for the cache look-up
    adjoint = FunctionSpec(f, tt, #=kernel=# false, #=name=# nothing)

    # primal function. Inferred here to get return type
    _tt = (tt.parameters...,)
    overdub_tt = Tuple{typeof(Compiler.CTX), F, map(eltype, _tt)...}
    primal = FunctionSpec(Cassette.overdub, overdub_tt, #=kernel=# false, #=name=# nothing)

    # can't return array since that's complicated.
    rt = Core.Compiler.return_type(Cassette.overdub, overdub_tt)
    @assert rt<:Union{AbstractFloat, Nothing}
    return primal, adjoint, rt
end


"""
    wrapper!(::LLVM.Module, ::LLVM.Function, ::FunctionSpec, ::Type)

Generates a wrapper function that will call `__enzyme_autodiff` on the primal function,
named `enzyme_entry`.
"""
function wrapper!(mod, primalf, adjoint, rt, name = "enzyme_entry")
    # create a wrapper function that will call `__enzyme_autodiff`
    ctx     = context(mod)
    rettype = convert(LLVMType, rt)

    tt = [adjoint.tt.parameters...,]
    params = parameters(primalf)
    adjoint_tt = LLVMType[]
    for (i, T) in enumerate(tt)
        llvmT = llvmtype(params[i])
        push!(adjoint_tt, llvmT)
        if T <: Duplicated
            push!(adjoint_tt, llvmT)
        end
    end

    llvmf = LLVM.Function(mod, name, LLVM.FunctionType(rettype, adjoint_tt))
    push!(function_attributes(llvmf), EnumAttribute("alwaysinline", 0, ctx))

    # Create the FunctionType and funtion declaration for the intrinsic
    pt       = LLVM.PointerType(LLVM.Int8Type(ctx))
    ftd      = LLVM.FunctionType(rettype, LLVMType[pt], vararg = true)
    autodiff = LLVM.Function(mod, string("__enzyme_autodiff.", rt), ftd)

    params = LLVM.Value[]
    llvm_params = parameters(llvmf)
    i = 1
    for T in tt
        if T <: Const
            push!(params, MDString("enzyme_const"))
        elseif T <: Active
            push!(params, MDString("enzyme_out"))
        elseif T <: Duplicated
            push!(params, MDString("enzyme_dup"))
            push!(params, llvm_params[i])
            i += 1
        elseif T <: DuplicatedNoNeed
            push!(params, MDString("enzyme_dupnoneed"))
            push!(params, llvm_params[i])
            i += 1
        else
            @assert("illegal annotation type")
        end
        push!(params, llvm_params[i])
        i += 1
    end

    Builder(ctx) do builder
        entry = BasicBlock(llvmf, "entry", ctx)
        position!(builder, entry)

        tc = bitcast!(builder, primalf,  pt)
        pushfirst!(params, tc)

        val = call!(builder, autodiff, params)

        ret!(builder, val)
    end

    return llvmf
end

include("compiler/thunk.jl")
include("compiler/reflection.jl")
# include("compiler/validation.jl")

end
