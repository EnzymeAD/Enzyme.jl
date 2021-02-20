module Compiler

import ..Enzyme: Const, Active, Duplicated, DuplicatedNoNeed
import ..Enzyme: API, TypeTree, typetree, only!, shift!, TypeAnalysis, FnTypeInfo

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
    malloc(sz) = Base.reinterpret(Ptr{Int8}, ccall("extern malloc", llvmcall, Core.LLVMPtr{Int8, 0}, (Int64,), sz))
    report_oom(sz) = return
    report_exception(ex) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return
end

struct EnzymeCompilerParams <: AbstractCompilerParams
    adjoint::FunctionSpec
    rt::DataType
    split::Bool
end

## job

# TODO: We shouldn't blanket opt-out
GPUCompiler.check_invocation(job::CompilerJob{EnzymeTarget}, entry::LLVM.Function) = nothing

GPUCompiler.runtime_module(target::CompilerJob{EnzymeTarget}) = Runtime
GPUCompiler.isintrinsic(::CompilerJob{EnzymeTarget}, fn::String) = true
GPUCompiler.can_throw(::CompilerJob{EnzymeTarget}) = true

# TODO: encode debug build or not in the compiler job
#       https://github.com/JuliaGPU/CUDAnative.jl/issues/368
GPUCompiler.runtime_slug(job::CompilerJob{EnzymeTarget}) = "enzyme"

include("compiler/optimize.jl")
include("compiler/cassette.jl")

function alloc_obj_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    @info "alloc_obj_rule" direction ret args numArgs val known_values
    return UInt8(false)
end

function i64_box_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    TT = TypeTree(API.DT_Integer, LLVM.context(LLVM.Value(val)))
    only!(TT, -1)
    API.EnzymeSetTypeTree(unsafe_load(args), TT)
    dl = string(LLVM.datalayout(LLVM.parent(LLVM.parent(LLVM.parent(LLVM.Instruction(val))))))
    shift!(TT,  dl, #=off=#0, #=maxSize=#8, #=addOffset=#0)
    API.EnzymeSetTypeTree(ret, TT)
    return UInt8(false)
end

"""
Create the `FunctionSpec` pair, and lookup the primal return type.
"""
@inline function fspec(f::F, tt::TT) where {F, TT}
    # Entry for the cache look-up
    adjoint = FunctionSpec(f, tt, #=kernel=# false, #=name=# nothing)

    # primal function. Inferred here to get return type
    _tt = (tt.parameters...,)
    overdub_tt = Tuple{map(eltype, _tt)...}
    #primal = FunctionSpec(Cassette.overdub, overdub_tt, #=kernel=# false, #=name=# nothing)
    primal = FunctionSpec(f, overdub_tt, #=kernel=# false, #=name=# nothing)

    # can't return array since that's complicated.
    #rt = Core.Compiler.return_type(Cassette.overdub, overdub_tt)
    rt = Core.Compiler.return_type(f, overdub_tt)
    if !(rt<:Union{AbstractFloat, Nothing})
        @error "Return type should be <:Union{Nothing, AbstractFloat}" rt adjoint primal
        error("Internal Enzyme Error")
    end
    return primal, adjoint, rt
end


function annotate!(mod)
    inactive = LLVM.StringAttribute("enzyme_inactive", "", context(mod))
    for inactivefn in ["julia.ptls_states", "julia.write_barrier", "julia.typeof", "jl_box_int64"]
        if haskey(functions(mod), inactivefn)
            fn = functions(mod)[inactivefn]
            push!(function_attributes(fn), inactive)
        end
    end
end


function enzyme!(mod, primalf, adjoint, rt, split)
    @show mod
    ctx     = context(mod)
    rettype = convert(LLVMType, rt, ctx)
    dl      = string(LLVM.datalayout(mod))

    tt = [adjoint.tt.parameters...,]

    args_activity     = API.CDIFFE_TYPE[]
    uncacheable_args  = Bool[]
    args_typeInfo     = TypeTree[]
    args_known_values = API.IntList[]

    for T in tt
        if T <: Const
            push!(args_activity, API.DFT_CONSTANT)
        elseif T <: Active
            push!(args_activity, API.DFT_OUT_DIFF)
        elseif  T <: Duplicated
            push!(args_activity, API.DFT_DUP_ARG)
        elseif T <: DuplicatedNoNeed
            push!(args_activity, API.DFT_DUP_NONEED)
        else 
            @assert("illegal annotation type")
        end
        typeTree = typetree(T, ctx, dl)
        push!(args_typeInfo, typeTree)
        if split
            push!(uncacheable_args, true)
        else
            push!(uncacheable_args, false)
        end
        push!(args_known_values, API.IntList())
    end

    # TODO ABI returned
    # The return of createprimal and gradient has this ABI
    #  It returns a struct containing the following values
    #     If requested, the original return value of the function
    #     If requested, the shadow return value of the function
    #     For each active (non duplicated) argument
    #       The adjoint of that argument

    if rt <: Integer
        retType = API.DFT_CONSTANT
    elseif rt <: AbstractFloat
        retType = API.DFT_OUT_DIFF
    elseif rt == Nothing
        retType = API.DFT_CONSTANT
    else
        error("What even is $rt")
    end

    rules = Dict{String, API.CustomRuleType}(
        "julia.gc_alloc_obj" => @cfunction(alloc_obj_rule, 
                                           UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                   Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "jl_box_int64" => @cfunction(i64_box_rule, 
                                           UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                   Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef))
    )
    TA = TypeAnalysis(triple(mod), rules) 
    global_AA = API.EnzymeGetGlobalAA(mod)
    retTT = typetree(rt, ctx, dl)

    typeInfo = FnTypeInfo(retTT, args_typeInfo, args_known_values)

    if split
        augmented = API.EnzymeCreateAugmentedPrimal(
            primalf, retType, args_activity, TA, global_AA, #=returnUsed=# true,
            typeInfo, uncacheable_args, #=forceAnonymousTape=# false, #=atomicAdd=# false, #=postOpt=# false)

        # 2. get new_primalf
        augmented_primalf = LLVM.Function(API.EnzymeExtractFunctionFromAugmentation(augmented))

        # TODOs:
        # 1. Handle mutable or !pointerfree arguments by introducing caching
        #     + specifically by setting uncacheable_args[i] = true
        # 2. Forward tape from augmented primalf to adjoint (as last arg)
        # 3. Make creation of augumented primalf vs joint forward and reverse optional

        tape = API.EnzymeExtractTapeTypeFromAugmentation(augmented)
        data = Array{Int64, 1}(undef, 3)
        existed = Array{UInt8, 1}(undef, 3)

        API.EnzymeExtractReturnInfo(augmented, data, existed)

        adjointf = LLVM.Function(API.EnzymeCreatePrimalAndGradient(
            primalf, retType, args_activity, TA, global_AA,
            #=returnValue=#false, #=dretUsed=#false, #=topLevel=#false,
            #=additionalArg=#tape, typeInfo,
            uncacheable_args, augmented, #=atomicAdd=#false, #=postOpt=#false))
    else
        adjointf = LLVM.Function(API.EnzymeCreatePrimalAndGradient(
            primalf, retType, args_activity, TA, global_AA,
            #=returnValue=#false, #=dretUsed=#false, #=topLevel=#true,
            #=additionalArg=#C_NULL, typeInfo,
            uncacheable_args, #=augmented=#C_NULL, #=atomicAdd=#false, #=postOpt=#false))
        augmented_primalf = nothing
    end
    
    API.EnzymeFreeGlobalAA(global_AA)
    return adjointf, augmented_primalf
end

include("compiler/thunk.jl")
include("compiler/reflection.jl")
# include("compiler/validation.jl")

end
