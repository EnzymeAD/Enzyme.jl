module Compiler

import ..Enzyme: Const, Active, Duplicated, DuplicatedNoNeed, Annotation, guess_activity, eltype
import ..Enzyme: API, TypeTree, typetree, only!, shift!, data0!,
                 TypeAnalysis, FnTypeInfo, Logic, allocatedinline

using LLVM, GPUCompiler, Libdl
import Enzyme_jll

import GPUCompiler: CompilerJob, FunctionSpec, codegen, safe_name
using LLVM.Interop
import LLVM: Target, TargetMachine

using Printf

if LLVM.has_orc_v1()
    include("compiler/orcv1.jl")
else
    include("compiler/orcv2.jl")
end

# User facing interface
abstract type AbstractThunk{F, RT, TT, Width, DF} end

struct CombinedAdjointThunk{F, RT, TT, Width, DF} <: AbstractThunk{F, RT, TT, Width, DF}
    fn::F
    adjoint::Ptr{Cvoid}
    dfn::DF
end

struct ForwardModeThunk{F, RT, TT, Width, DF} <: AbstractThunk{F, RT, TT, Width, DF}
    fn::F
    adjoint::Ptr{Cvoid}
    dfn::DF
end

struct AugmentedForwardThunk{F, RT, TT, Width, DF} <: AbstractThunk{F, RT, TT, Width, DF}
    fn::F
    primal::Ptr{Cvoid}
    dfn::DF
end

struct AdjointThunk{F, RT, TT, Width, DF} <: AbstractThunk{F, RT, TT, Width, DF}
    fn::F
    adjoint::Ptr{Cvoid}
    dfn::DF
end
return_type(::AbstractThunk{F, RT, TT, Width, DF}) where {F, RT, TT, Width, DF} = RT

using .JIT

# Define EnzymeTarget
Base.@kwdef struct EnzymeTarget <: AbstractCompilerTarget
end
GPUCompiler.llvm_triple(::EnzymeTarget) = Sys.MACHINE

# GPUCompiler.llvm_datalayout(::EnzymeTarget) =  nothing

function GPUCompiler.llvm_machine(::EnzymeTarget)
    return tm[]
end

module Runtime
end

abstract type AbstractEnzymeCompilerParams <: AbstractCompilerParams end
struct EnzymeCompilerParams <: AbstractEnzymeCompilerParams
    adjoint::FunctionSpec
    mode::API.CDerivativeMode
    width::Int64
    rt::Type{<:Annotation}
    run_enzyme::Bool
    dupClosure::Bool
    abiwrap::Bool
    # Whether, in split mode, acessible primal argument data is modified
    # between the call and the split
    modifiedBetween::Bool
end

struct PrimalCompilerParams <: AbstractEnzymeCompilerParams
end

include("compiler/interpreter.jl")
## job

# TODO: We shouldn't blanket opt-out
GPUCompiler.check_invocation(job::CompilerJob{EnzymeTarget}, entry::LLVM.Function) = nothing

GPUCompiler.runtime_module(::CompilerJob{<:Any,<:AbstractEnzymeCompilerParams}) = Runtime
# GPUCompiler.isintrinsic(::CompilerJob{EnzymeTarget}, fn::String) = true
# GPUCompiler.can_throw(::CompilerJob{EnzymeTarget}) = true

# TODO: encode debug build or not in the compiler job
#       https://github.com/JuliaGPU/CUDAnative.jl/issues/368
GPUCompiler.runtime_slug(job::CompilerJob{EnzymeTarget}) = "enzyme"

# provide a specific interpreter to use.
GPUCompiler.get_interpreter(job::CompilerJob{<:Any,<:AbstractEnzymeCompilerParams}) =
    EnzymeInterpeter(GPUCompiler.ci_cache(job), GPUCompiler.method_table(job), job.source.world)

include("compiler/utils.jl")
include("compiler/passes.jl")
include("compiler/optimize.jl")

"""
Create the `FunctionSpec` pair, and lookup the primal return type.
"""
@inline function fspec(@nospecialize(F), @nospecialize(TT))
    # Entry for the cache look-up
    adjoint = FunctionSpec(F, TT, #=kernel=# false, #=name=# nothing)

    # primal function. Inferred here to get return type
    _tt = (TT.parameters...,)

    primal_tt = Tuple{map(eltype, _tt)...}
    primal = FunctionSpec(F, primal_tt, #=kernel=# false, #=name=# nothing)

    return primal, adjoint
end

##
# Enzyme compiler step
##

include("compiler/rules.jl")

const inactivefns = Set{String}((
    "jl_gc_queue_root", "gpu_report_exception", "gpu_signal_exception",
    "julia.ptls_states", "julia.write_barrier", "julia.typeof", "jl_box_int64", "jl_box_int32",
    "jl_subtype", "julia.get_pgcstack", "jl_in_threaded_region",
    "jl_object_id_", "jl_object_id", "ijl_object_id_", "ijl_object_id",
    "jl_breakpoint",
    "llvm.julia.gc_preserve_begin","llvm.julia.gc_preserve_end", "jl_get_ptls_states",
    "jl_f_fieldtype",
    "jl_symbol_n",
    "jl_stored_inline", "ijl_stored_inline",
    "jl_f_apply_type", "jl_f_issubtype", "jl_isa",
    "jl_matching_methods", "ijl_matching_methods",
    "jl_excstack_state", "jl_current_exception"
    # "jl_"
))

const activefns = Set{String}((
    "jl_",
))

function annotate!(mod, mode)
    ctx = context(mod)
    inactive = LLVM.StringAttribute("enzyme_inactive", ""; ctx)
    active = LLVM.StringAttribute("enzyme_active", ""; ctx)
    fns = functions(mod)

    for inactivefn in inactivefns
        if haskey(fns, inactivefn)
            fn = fns[inactivefn]
            push!(function_attributes(fn), inactive)
        end
    end

    for activefn in activefns
        if haskey(fns, activefn)
            fn = fns[activefn]
            push!(function_attributes(fn), active)
        end
    end

    for fname in ("julia.typeof",)
        if haskey(fns, fname)
            fn = fns[fname]
            push!(function_attributes(fn), LLVM.EnumAttribute("readnone", 0; ctx))
            push!(function_attributes(fn), LLVM.StringAttribute("enzyme_shouldrecompute"; ctx))
        end
    end

    for fname in ("jl_excstack_state",)
        if haskey(fns, fname)
            fn = fns[fname]
            push!(function_attributes(fn), LLVM.EnumAttribute("readonly", 0; ctx))
            push!(function_attributes(fn), LLVM.StringAttribute("inaccessiblememonly"; ctx))
        end
    end

    for fname in ("julia.get_pgcstack", "julia.ptls_states", "jl_get_ptls_states")
        if haskey(fns, fname)
            fn = fns[fname]
            # TODO per discussion w keno perhaps this should change to readonly / inaccessiblememonly
            push!(function_attributes(fn), LLVM.EnumAttribute("readnone", 0; ctx))
        end
    end

    for fname in ("julia.pointer_from_objref",)
        if haskey(fns, fname)
            fn = fns[fname]
            push!(function_attributes(fn), LLVM.EnumAttribute("readnone", 0; ctx))
        end
    end

    for boxfn in ("jl_box_float32", "jl_box_float64", "jl_box_int32", "jl_box_int64", "julia.gc_alloc_obj",
                  "jl_alloc_array_1d", "jl_alloc_array_2d", "jl_alloc_array_3d",
                  "ijl_alloc_array_1d", "ijl_alloc_array_2d", "ijl_alloc_array_3d",
                  "jl_f_tuple", "ijl_f_tuple")
        if haskey(fns, boxfn)
            fn = fns[boxfn]
            push!(return_attributes(fn), LLVM.EnumAttribute("noalias", 0; ctx))
            push!(function_attributes(fn), LLVM.EnumAttribute("inaccessiblememonly", 0; ctx))
        end
    end

    for gc in ("llvm.julia.gc_preserve_begin", "llvm.julia.gc_preserve_end")
        if haskey(fns, gc)
            fn = fns[gc]
            push!(function_attributes(fn), LLVM.EnumAttribute("inaccessiblememonly", 0; ctx))
        end
    end

    for rfn in ("jl_object_id_", "jl_object_id", "ijl_object_id_", "ijl_object_id")
        if haskey(fns, rfn)
            fn = fns[rfn]
            push!(function_attributes(fn), LLVM.EnumAttribute("readonly", 0; ctx))
        end
    end

    for rfn in ("jl_in_threaded_region_", "jl_in_threaded_region")
        if haskey(fns, rfn)
            fn = fns[rfn]
            push!(function_attributes(fn), LLVM.EnumAttribute("readonly", 0; ctx))
            push!(function_attributes(fn), LLVM.EnumAttribute("inaccessiblememonly", 0; ctx))
        end
    end
end

function noop_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    return UInt8(false)
end

function alloc_obj_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    inst = LLVM.Instruction(val)
    ce = operands(inst)[3]
    while isa(ce, ConstantExpr)
        ce = operands(ce)[1]
    end
    ptr = reinterpret(Ptr{Cvoid}, convert(UInt64, ce))
    typ = Base.unsafe_pointer_to_objref(ptr)

    ctx = LLVM.context(LLVM.Value(val))
    dl = string(LLVM.datalayout(LLVM.parent(LLVM.parent(LLVM.parent(inst)))))

    rest = typetree(typ, ctx, dl)
    only!(rest, -1)
    API.EnzymeMergeTypeTree(ret, rest)
    return UInt8(false)
end

function int_return_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    TT = TypeTree(API.DT_Integer, LLVM.context(LLVM.Value(val)))
    only!(TT, -1)
    API.EnzymeSetTypeTree(ret, TT)
    return UInt8(false)
end

function i64_box_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    TT = TypeTree(API.DT_Pointer, LLVM.context(LLVM.Value(val)))
    only!(TT, -1)
    API.EnzymeSetTypeTree(ret, TT)
    return UInt8(false)
end


function f32_box_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    TT = TypeTree(API.DT_Float, LLVM.context(LLVM.Value(val)))
    only!(TT, -1)
    API.EnzymeSetTypeTree(unsafe_load(args), TT)
    dl = string(LLVM.datalayout(LLVM.parent(LLVM.parent(LLVM.parent(LLVM.Instruction(val))))))
    shift!(TT,  dl, #=off=#0, #=maxSize=#8, #=addOffset=#0)
    API.EnzymeSetTypeTree(ret, TT)
    return UInt8(false)
end

function ptr_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    TT = TypeTree(API.DT_Pointer, LLVM.context(LLVM.Value(val)))
    only!(TT, -1)
    API.EnzymeSetTypeTree(ret, TT)
    return UInt8(false)
end

function inout_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    if (direction & API.UP) != 0
        API.EnzymeMergeTypeTree(unsafe_load(args), ret)
    end
    if (direction & API.DOWN) != 0
        API.EnzymeMergeTypeTree(ret, unsafe_load(args))
    end
    return UInt8(false)
end

function alloc_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    inst = LLVM.Instruction(val)
    ce = operands(inst)[1]
    while isa(ce, ConstantExpr)
        ce = operands(ce)[1]
    end
    ptr = reinterpret(Ptr{Cvoid}, convert(UInt64, ce))
    typ = Base.unsafe_pointer_to_objref(ptr)

    ctx = LLVM.context(LLVM.Value(val))
    dl = string(LLVM.datalayout(LLVM.parent(LLVM.parent(LLVM.parent(inst)))))

    rest = typetree(typ, ctx, dl)
    only!(rest, -1)
    API.EnzymeMergeTypeTree(ret, rest)

    for i = 1:numArgs
        API.EnzymeMergeTypeTree(unsafe_load(args, i), TypeTree(API.DT_Integer, -1, ctx))
    end
    return UInt8(false)
end

function enzyme!(job, mod, primalf, adjoint, mode, width, parallel, actualRetType, dupClosure, wrap, modifiedBetween)
    rt  = job.params.rt
    ctx = context(mod)
    dl  = string(LLVM.datalayout(mod))
    F   = adjoint.f

    tt = [adjoint.tt.parameters...,]

    if eltype(rt) === Union{}
        error("return type is Union{}, giving up.")
    end

    args_activity     = API.CDIFFE_TYPE[]
    uncacheable_args  = Bool[]
    args_typeInfo     = TypeTree[]
    args_known_values = API.IntList[]

    ctx = LLVM.context(mod)
    if !GPUCompiler.isghosttype(F) && !Core.Compiler.isconstType(F)
        typeTree = typetree(F, ctx, dl)
        push!(args_typeInfo, typeTree)
        if dupClosure
            push!(args_activity, API.DFT_DUP_ARG)
        else
            push!(args_activity, API.DFT_CONSTANT)
        end
        if !modifiedBetween
            push!(uncacheable_args, false)
        else
            push!(uncacheable_args, true)
        end
        push!(args_known_values, API.IntList())
    end

    for T in tt
        source_typ = eltype(T)
        if GPUCompiler.isghosttype(source_typ) || Core.Compiler.isconstType(source_typ)
            if !(T <: Const)
                error("Type of ghost or constant is marked as differentiable.")
            end
            continue
        end
        isboxed = GPUCompiler.deserves_argbox(source_typ)

        if T <: Const
            push!(args_activity, API.DFT_CONSTANT)
        elseif T <: Active

            if isboxed
                push!(args_activity, API.DFT_DUP_ARG)
            else
                push!(args_activity, API.DFT_OUT_DIFF)
            end
        elseif  T <: Duplicated || T<: BatchDuplicated
            push!(args_activity, API.DFT_DUP_ARG)
        elseif T <: DuplicatedNoNeed || T<: BatchDuplicatedNoNeed
            push!(args_activity, API.DFT_DUP_NONEED)
        else
            error("illegal annotation type")
        end
        T = source_typ
        if isboxed
            T = Ptr{T}
        end
        typeTree = typetree(T, ctx, dl)
        push!(args_typeInfo, typeTree)
        if !modifiedBetween
            push!(uncacheable_args, false)
        else
            push!(uncacheable_args, true)
        end
        push!(args_known_values, API.IntList())
    end

    # The return of createprimal and gradient has this ABI
    #  It returns a struct containing the following values
    #     If requested, the original return value of the function
    #     If requested, the shadow return value of the function
    #     For each active (non duplicated) argument
    #       The adjoint of that argument
    if rt <: Const
        retType = API.DFT_CONSTANT
    elseif rt <: Active
        retType = API.DFT_OUT_DIFF
    elseif rt <: Duplicated || rt <: BatchDuplicated
        retType = API.DFT_DUP_ARG
    elseif rt <: DuplicatedNoNeed || rt <: BatchDuplicatedNoNeed
        retType = API.DFT_DUP_NONEED
    else
        error("Unhandled return type $rt")
    end

    rules = Dict{String, API.CustomRuleType}(
        "jl_apply_generic" => @cfunction(ptr_rule,
                                           UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                   Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "ijl_apply_generic" => @cfunction(ptr_rule,
                                           UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                   Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "julia.gc_alloc_obj" => @cfunction(alloc_obj_rule,
                                           UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                   Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "jl_box_float32" => @cfunction(f32_box_rule,
                                           UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                   Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "jl_box_int64" => @cfunction(i64_box_rule,
                                           UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                   Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "jl_box_uint64" => @cfunction(i64_box_rule,
                                            UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                    Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "jl_array_copy" => @cfunction(inout_rule,
                                           UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                   Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "jl_alloc_array_1d" => @cfunction(alloc_rule,
                                            UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                    Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "jl_alloc_array_2d" => @cfunction(alloc_rule,
                                            UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                    Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "jl_alloc_array_3d" => @cfunction(alloc_rule,
                                            UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                    Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "ijl_alloc_array_1d" => @cfunction(alloc_rule,
                                            UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                    Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "ijl_alloc_array_2d" => @cfunction(alloc_rule,
                                            UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                    Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "ijl_alloc_array_3d" => @cfunction(alloc_rule,
                                            UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                    Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "julia.pointer_from_objref" => @cfunction(inout_rule,
                                            UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                    Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "jl_wait" => @cfunction(noop_rule,
                                            UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                    Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "jl_enq_work" => @cfunction(noop_rule,
                                            UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                    Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),

        "enz_noop" => @cfunction(noop_rule,
                                            UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                    Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "jl_inactive_inout" => @cfunction(inout_rule,
                                           UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                   Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "jl_excstack_state" => @cfunction(int_return_rule,
                                           UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                   Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "julia.except_enter" => @cfunction(int_return_rule,
                                           UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                   Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
    )

    logic = Logic()
    TA = TypeAnalysis(logic, rules)

    retTT = typetree(GPUCompiler.deserves_argbox(actualRetType) ? Ptr{actualRetType} : actualRetType, ctx, dl)

    typeInfo = FnTypeInfo(retTT, args_typeInfo, args_known_values)

    if mode == API.DEM_ReverseModePrimal || mode == API.DEM_ReverseModeGradient
        returnUsed = !(GPUCompiler.isghosttype(actualRetType) || Core.Compiler.isconstType(actualRetType)) 
        shadowReturnUsed = returnUsed && (retType == API.DFT_DUP_ARG || retType == API.DFT_DUP_NONEED)
        returnUsed &= retType != API.DFT_DUP_NONEED
        augmented = API.EnzymeCreateAugmentedPrimal(
            logic, primalf, retType, args_activity, TA, #=returnUsed=# returnUsed,
            #=shadowReturnUsed=#shadowReturnUsed,
            typeInfo, uncacheable_args, #=forceAnonymousTape=# true, width, #=atomicAdd=# parallel)

        # 2. get new_primalf and tape
        augmented_primalf = LLVM.Function(API.EnzymeExtractFunctionFromAugmentation(augmented))
        tape = API.EnzymeExtractTapeTypeFromAugmentation(augmented)
        if wrap
          augmented_primalf = create_abi_wrapper(augmented_primalf, F, tt, rt, actualRetType, API.DEM_ReverseModePrimal, augmented, dupClosure, width)
        end

        # TODOs:
        # 1. Handle mutable or !pointerfree arguments by introducing caching
        #     + specifically by setting uncacheable_args[i] = true

        adjointf = LLVM.Function(API.EnzymeCreatePrimalAndGradient(
            logic, primalf, retType, args_activity, TA,
            #=returnValue=#false, #=dretUsed=#false, #=mode=#API.DEM_ReverseModeGradient, width,
            #=additionalArg=#tape, typeInfo,
            uncacheable_args, augmented, #=atomicAdd=# parallel))
        if wrap
          adjointf = create_abi_wrapper(adjointf, F, tt, rt, actualRetType, API.DEM_ReverseModeGradient, augmented, dupClosure, width)
        end
    elseif mode == API.DEM_ReverseModeCombined
        adjointf = LLVM.Function(API.EnzymeCreatePrimalAndGradient(
            logic, primalf, retType, args_activity, TA,
            #=returnValue=#false, #=dretUsed=#false, #=mode=#API.DEM_ReverseModeCombined, width,
            #=additionalArg=#C_NULL, typeInfo,
            uncacheable_args, #=augmented=#C_NULL, #=atomicAdd=# parallel))
        augmented_primalf = nothing
        if wrap
          adjointf = create_abi_wrapper(adjointf, F, tt, rt, actualRetType, API.DEM_ReverseModeCombined, nothing, dupClosure, width)
        end
    elseif mode == API.DEM_ForwardMode
        adjointf = LLVM.Function(API.EnzymeCreateForwardDiff(
            logic, primalf, retType, args_activity, TA,
            #=returnValue=#rt <: Duplicated || rt <: BatchDuplicated, #=mode=#API.DEM_ForwardMode, width,
            #=additionalArg=#C_NULL, typeInfo,
            uncacheable_args))
        augmented_primalf = nothing
        if wrap
          adjointf = create_abi_wrapper(adjointf, F, tt, rt, actualRetType, API.DEM_ForwardMode, nothing, dupClosure, width)
        end
    else
        @assert "Unhandled derivative mode", mode
    end
    return adjointf, augmented_primalf
end

function create_abi_wrapper(enzymefn::LLVM.Function, F, argtypes, rettype, actualRetType, Mode::API.CDerivativeMode, augmented, dupClosure, width)
    is_adjoint = Mode == API.DEM_ReverseModeGradient || Mode == API.DEM_ReverseModeCombined
    is_split   = Mode == API.DEM_ReverseModeGradient || Mode == API.DEM_ReverseModePrimal
    needs_tape = Mode == API.DEM_ReverseModeGradient

    mod = LLVM.parent(enzymefn)
    ctx = LLVM.context(mod)

    push!(function_attributes(enzymefn), EnumAttribute("alwaysinline", 0; ctx))
    T_void = convert(LLVMType, Nothing; ctx)
    ptr8 = LLVM.PointerType(LLVM.IntType(8; ctx))
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)

    # Create Enzyme calling convention
    T_wrapperargs = LLVMType[] # Arguments of the wrapper

    T_JuliaSRet = LLVMType[]  # Struct return of all objects
                              # + If the adjoint this will be all Active variables (includes all of T_EnzymeSRet)
                              # + If the forward, this will be ?return, ?shadowReturn, ?tape

    if !GPUCompiler.isghosttype(F) && !Core.Compiler.isconstType(F)
        isboxed = GPUCompiler.deserves_argbox(F)
        llvmT = isboxed ? T_prjlvalue : convert(LLVMType, F; ctx)
        push!(T_wrapperargs, llvmT)
        if dupClosure
            push!(T_wrapperargs, llvmT)
        end
    end

    for T in argtypes
        source_typ = eltype(T)
        if GPUCompiler.isghosttype(source_typ) || Core.Compiler.isconstType(source_typ)
            @assert T <: Const
            continue
        end

        isboxed = GPUCompiler.deserves_argbox(source_typ)
        llvmT = isboxed ? T_prjlvalue : convert(LLVMType, source_typ; ctx)

        push!(T_wrapperargs, llvmT)

        T <: Const && continue

        if T <: Active
            if is_adjoint
                # Use deserves_argbox??
                llvmT = LLVM.LLVMType(API.EnzymeGetShadowType(width, convert(LLVMType, source_typ; ctx)))
                push!(T_JuliaSRet, llvmT)
            end
        elseif T <: Duplicated || T <: DuplicatedNoNeed
            @assert width == 1
            push!(T_wrapperargs, LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmT)))
        elseif T <: BatchDuplicated || T <: BatchDuplicatedNoNeed
            push!(T_wrapperargs, LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmT)))
        else
            error("calling convention should be annotated, got $T")
        end
    end

    # API.DFT_OUT_DIFF
    if is_adjoint && rettype <: Active
        @assert allocatedinline(actualRetType)
        push!(T_wrapperargs, convert(LLVMType, actualRetType; ctx))
    end

    data    = Array{Int64}(undef, 3)
    existed = Array{UInt8}(undef, 3)
    if Mode == API.DEM_ReverseModePrimal
        API.EnzymeExtractReturnInfo(augmented, data, existed)
        # tape -- todo ??? on wrap
        if existed[1] != 0
            tape = API.EnzymeExtractTapeTypeFromAugmentation(augmented)
            push!(T_JuliaSRet, LLVM.LLVMType(tape))
        else
            error("Assuming always has a tape")
        end
        
        isboxed = GPUCompiler.deserves_argbox(actualRetType)
        llvmT = isboxed ? T_prjlvalue : convert(LLVMType, actualRetType; ctx)
        
        # primal return
        if existed[2] != 0 
            push!(T_JuliaSRet, llvmT)
        end
        # shadow return
        if existed[3] != 0
            push!(T_JuliaSRet, LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmT)))
        else
            @assert rettype <: Const || rettype <: Active
        end
    end
    if Mode == API.DEM_ForwardMode
        returnUsed = !(GPUCompiler.isghosttype(actualRetType) || Core.Compiler.isconstType(actualRetType))
        if returnUsed
            isboxed = GPUCompiler.deserves_argbox(actualRetType)
            llvmT = isboxed ? T_prjlvalue : convert(LLVMType, actualRetType; ctx)
            if rettype <: Duplicated || rettype <: BatchDuplicated
                push!(T_JuliaSRet, llvmT)
            end
            if !(rettype <: Const)
                push!(T_JuliaSRet, LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmT)))
            end
        end
    end

    # sret argument
    if !isempty(T_JuliaSRet)
        pushfirst!(T_wrapperargs, LLVM.PointerType(LLVM.StructType(T_JuliaSRet; ctx)))
    end

    if needs_tape
        push!(T_wrapperargs, LLVM.LLVMType(API.EnzymeExtractTapeTypeFromAugmentation(augmented)))
    end

    FT = LLVM.FunctionType(T_void, T_wrapperargs)
    llvm_f = LLVM.Function(mod, safe_name(LLVM.name(enzymefn)*"wrap"), FT)
    dl = datalayout(mod)

    params = [parameters(llvm_f)...]
    target =  !isempty(T_JuliaSRet) ? 2 : 1

    intrinsic_typ = LLVM.FunctionType(T_void, [ptr8, LLVM.IntType(8; ctx), LLVM.IntType(64; ctx), LLVM.IntType(1; ctx)])
    memsetIntr = LLVM.Function(mod, "llvm.memset.p0i8.i64", intrinsic_typ)
    LLVM.Builder(ctx) do builder
        entry = BasicBlock(llvm_f, "entry"; ctx)
        position!(builder, entry)

        realparms = LLVM.Value[]
        i = target

        if !isempty(T_JuliaSRet)
            sret = params[1]
        end

        activeNum = 0

        if !GPUCompiler.isghosttype(F) && !Core.Compiler.isconstType(F)
            push!(realparms, params[i])
            i+=1
            if dupClosure
                push!(realparms, params[i])
                i+=1
            end
        end

        for T in argtypes
            T′ = eltype(T)

            if GPUCompiler.isghosttype(T′) || Core.Compiler.isconstType(T′)
                continue
            end
            push!(realparms, params[i])
            i += 1
            if T <: Const
            elseif T <: Active
                isboxed = GPUCompiler.deserves_argbox(T′)
                if isboxed
                    @assert !is_split
                    ptr = gep!(builder, sret, [LLVM.ConstantInt(LLVM.IntType(64; ctx), 0), LLVM.ConstantInt(LLVM.IntType(32; ctx), activeNum)])
                    cst = pointercast!(builder, ptr, ptr8)
                    push!(realparms, ptr)

                    cparms = LLVM.Value[cst,
                    LLVM.ConstantInt(LLVM.IntType(8; ctx), 0),
                    LLVM.ConstantInt(LLVM.IntType(64; ctx), LLVM.storage_size(dl, Base.eltype(LLVM.llvmtype(ptr)) )),
                    LLVM.ConstantInt(LLVM.IntType(1; ctx), 0)]
                    call!(builder, memsetIntr, cparms)
                end
                activeNum += 1
            elseif T <: Duplicated || T <: DuplicatedNoNeed
                push!(realparms, params[i])
                i += 1
            elseif T <: BatchDuplicated || T <: BatchDuplicatedNoNeed
                isboxed = GPUCompiler.deserves_argbox(NTuple{width, T′})
                val = params[i]
                if isboxed
                  val = load!(builder, val)
                end
                i += 1
                push!(realparms, val)
            end
        end

        if is_adjoint && rettype <: Active
            push!(realparms, params[i])
            i += 1
        end

        if needs_tape
            push!(realparms, params[i])
            i += 1
        end

        val = call!(builder, enzymefn, realparms)

        if Mode == API.DEM_ReverseModePrimal
            returnNum = 0
            for i in 1:3
                if existed[i] != 0
                    eval = val
                    if data[i] != -1
                        eval = extract_value!(builder, val, data[i])
                    end
                    store!(builder, eval, gep!(builder, sret, [LLVM.ConstantInt(LLVM.IntType(64; ctx), 0), LLVM.ConstantInt(LLVM.IntType(32; ctx), returnNum)]))
                    returnNum+=1
                end
            end
        elseif Mode == API.DEM_ForwardMode
            for returnNum in 0:(length(T_JuliaSRet)-1)
                eval = val
                if length(T_JuliaSRet) > 1
                    eval = extract_value!(builder, val, returnNum)
                end
                store!(builder, eval, gep!(builder, sret, [LLVM.ConstantInt(LLVM.IntType(64; ctx), 0), LLVM.ConstantInt(LLVM.IntType(32; ctx), returnNum)]))
            end
        else
            activeNum = 0
            returnNum = 0
            for T in argtypes
                T′ = eltype(T)
                isboxed = GPUCompiler.deserves_argbox(T′)
                if T <: Active
                    if !isboxed
                        eval = extract_value!(builder, val, returnNum)
                        store!(builder, eval, gep!(builder, sret, [LLVM.ConstantInt(LLVM.IntType(64; ctx), 0), LLVM.ConstantInt(LLVM.IntType(32; ctx), activeNum)]))
                        returnNum+=1
                    end
                    activeNum+=1
                end
            end
        end
        ret!(builder)
    end

    # make sure that arguments are rooted if necessary
    reinsert_gcmarker!(llvm_f)
    return llvm_f
end

function fixup_metadata!(f::LLVM.Function)
    for param in parameters(f)
        if isa(llvmtype(param), LLVM.PointerType)
            # collect all uses of the pointer
            worklist = Vector{LLVM.Instruction}(user.(collect(uses(param))))
            while !isempty(worklist)
                value = popfirst!(worklist)

                # remove the invariant.load attribute
                md = metadata(value)
                if haskey(md, LLVM.MD_invariant_load)
                    delete!(md, LLVM.MD_invariant_load)
                end
                if haskey(md, LLVM.MD_tbaa)
                    delete!(md, LLVM.MD_tbaa)
                end

                # recurse on the output of some instructions
                if isa(value, LLVM.BitCastInst) ||
                   isa(value, LLVM.GetElementPtrInst) ||
                   isa(value, LLVM.AddrSpaceCastInst)
                    append!(worklist, user.(collect(uses(value))))
                end

                # IMPORTANT NOTE: if we ever want to inline functions at the LLVM level,
                # we need to recurse into call instructions here, and strip metadata from
                # called functions (see CUDAnative.jl#238).
            end
        end
    end
end

# Modified from GPUCompiler classify_arguments
function classify_arguments(source_sig::Type, codegen_ft::LLVM.FunctionType, has_sret)
    source_types = [source_sig.parameters...]
    codegen_types = parameters(codegen_ft)

    args = []
    codegen_i = has_sret ? 2 : 1
    for (source_i, source_typ) in enumerate(source_types)
        if isghosttype(source_typ) || Core.Compiler.isconstType(source_typ)
            push!(args, (cc=GPUCompiler.GHOST, typ=source_typ))
            continue
        end
        codegen_typ = codegen_types[codegen_i]
        if codegen_typ isa LLVM.PointerType && !issized(eltype(codegen_typ))
            push!(args, (cc=GPUCompiler.MUT_REF, typ=source_typ,
                         codegen=(typ=codegen_typ, i=codegen_i)))
        elseif codegen_typ isa LLVM.PointerType && issized(eltype(codegen_typ)) &&
               !(source_typ <: Ptr) && !(source_typ <: Core.LLVMPtr)
            push!(args, (cc=GPUCompiler.BITS_REF, typ=source_typ,
                         codegen=(typ=codegen_typ, i=codegen_i)))
        else
            push!(args, (cc=GPUCompiler.BITS_VALUE, typ=source_typ,
                         codegen=(typ=codegen_typ, i=codegen_i)))
        end
        codegen_i += 1
    end

    return args
end


# Modified from GPUCompiler/src/irgen.jl:365 lower_byval
function lower_convention(functy::Type, mod::LLVM.Module, entry_f::LLVM.Function, actualRetType::Type)
    ctx = context(mod)
    entry_ft = eltype(llvmtype(entry_f)::LLVM.PointerType)::LLVM.FunctionType

    RT = LLVM.return_type(entry_ft)

    # generate the wrapper function type & definition
    wrapper_types = LLVM.LLVMType[]
    sret = false
    if !isempty(parameters(entry_f)) && any(map(k->kind(k)==kind(EnumAttribute("sret"; ctx)), collect(parameter_attributes(entry_f, 1))))
        # TODO sret is now TypeAttribute
        RT = eltype(llvmtype(first(parameters(entry_f))))
        sret = true
    end
    
	args = classify_arguments(functy, entry_ft, sret)
    filter!(args) do arg
        arg.cc != GPUCompiler.GHOST
    end
    @assert length(args) == length(collect(parameters(entry_f))[1+sret:end]) 

    # TODO use rettype for sret calculation instead
    rettype = actualRetType
    
    for (parm, arg) in zip(collect(parameters(entry_f))[1+sret:end], args)
        typ = if !GPUCompiler.deserves_argbox(arg.typ) && arg.cc == GPUCompiler.BITS_REF
            eltype(arg.codegen.typ)
        else
            llvmtype(parm)
        end
        push!(wrapper_types, typ)
    end
    wrapper_fn = LLVM.name(entry_f)
    LLVM.name!(entry_f, safe_name(wrapper_fn * ".inner"))
    wrapper_ft = LLVM.FunctionType(RT, wrapper_types)
    wrapper_f = LLVM.Function(mod, LLVM.name(entry_f), wrapper_ft)

    hasReturnsTwice = any(map(k->kind(k)==kind(EnumAttribute("returns_twice"; ctx)), collect(function_attributes(entry_f))))
    push!(function_attributes(wrapper_f), EnumAttribute("returns_twice"; ctx))
    push!(function_attributes(entry_f), EnumAttribute("returns_twice"; ctx))

    # emit IR performing the "conversions"
    let builder = Builder(ctx)
        toErase = LLVM.CallInst[]
        for u in LLVM.uses(entry_f)
            ci = LLVM.user(u)
            if !isa(ci, LLVM.CallInst) || called_value(ci) != entry_f
                continue
            end
            ops = collect(operands(ci))[1:end-1]
            position!(builder, ci)
            nops = LLVM.Value[]
            start = sret ? 2 : 1
            for (parm, arg) in zip(ops[1+sret:end], args)
                if !GPUCompiler.deserves_argbox(arg.typ) && arg.cc == GPUCompiler.BITS_REF
                    push!(nops, load!(builder, parm))
                else
                    push!(nops, parm)
                end
            end
            res = call!(builder, wrapper_f, nops)
            if sret
              @assert llvmtype(res) == eltype(llvmtype(ops[1]))
              store!(builder, res, ops[1])
            else
              LLVM.replace_uses!(ci, res)
            end
            push!(toErase, ci)
        end
        for e in toErase
            if !isempty(collect(uses(e)))
                @show mod
                @show entry_f
                @show e
                throw(AssertionError("Use after deletion"))
            end
            LLVM.API.LLVMInstructionEraseFromParent(e)
        end

        entry = BasicBlock(wrapper_f, "entry"; ctx)
        position!(builder, entry)

        wrapper_args = Vector{LLVM.Value}()

        if sret
            sretPtr = alloca!(builder, eltype(llvmtype(parameters(entry_f)[1])))
            push!(wrapper_args, sretPtr)
        end

        # perform argument conversions
        for (parm, arg) in zip(collect(parameters(entry_f))[1+sret:end], args)
            wrapparm = parameters(wrapper_f)[arg.codegen.i-sret]
            if !GPUCompiler.deserves_argbox(arg.typ) && arg.cc == GPUCompiler.BITS_REF
                # copy the argument value to a stack slot, and reference it.
                ty = llvmtype(parm)
                if !isa(ty, LLVM.PointerType)
                    @show entry_f, args, parm, ty
                end
                @assert isa(ty, LLVM.PointerType)
                ptr = alloca!(builder, eltype(ty))
                if LLVM.addrspace(ty) != 0
                    ptr = addrspacecast!(builder, ptr, ty)
                end
                @assert eltype(ty) == llvmtype(wrapparm)
                store!(builder, wrapparm, ptr)
                push!(wrapper_args, ptr)
            else
                push!(wrapper_args, wrapparm)
                for attr in collect(parameter_attributes(entry_f, arg.codegen.i))
                    push!(parameter_attributes(wrapper_f, arg.codegen.i-sret), attr)
                end
            end
        end
        res = call!(builder, entry_f, wrapper_args)

        if LLVM.get_subprogram(entry_f) !== nothing
            metadata(res)[LLVM.MD_dbg] = DILocation(ctx, 0, 0, LLVM.get_subprogram(entry_f) )
        end
    
        LLVM.API.LLVMSetInstructionCallConv(res, LLVM.callconv(entry_f))

        if sret
            ret!(builder, load!(builder, sretPtr))
        elseif LLVM.return_type(entry_ft) == LLVM.VoidType(ctx)
            ret!(builder)
        else
            ret!(builder, res)
        end

        dispose(builder)
    end

    # early-inline the original entry function into the wrapper
    push!(function_attributes(entry_f), EnumAttribute("alwaysinline", 0; ctx))
    linkage!(entry_f, LLVM.API.LLVMInternalLinkage)

    fixup_metadata!(entry_f)
	
	ModulePassManager() do pm
        always_inliner!(pm)
        run!(pm, mod)
    end
    if !hasReturnsTwice
        LLVM.API.LLVMRemoveEnumAttributeAtIndex(wrapper_f, reinterpret(LLVM.API.LLVMAttributeIndex, LLVM.API.LLVMAttributeFunctionIndex), kind(EnumAttribute("returns_twice"; ctx)))
    end
    ModulePassManager() do pm
        # Kill the temporary staging function
        global_dce!(pm)
        global_optimizer!(pm)
        run!(pm, mod)
    end
    if haskey(globals(mod), "llvm.used")
        unsafe_delete!(mod, globals(mod)["llvm.used"])
        for u in user.(collect(uses(entry_f)))
            if isa(u, LLVM.GlobalVariable) && endswith(LLVM.name(u), "_slot") && startswith(LLVM.name(u), "julia")
                unsafe_delete!(mod, u)
            end
        end
    end
    verify(wrapper_f)
    return wrapper_f
end

function adim(::Array{T, N}) where {T, N}
    return N
end

function GPUCompiler.codegen(output::Symbol, job::CompilerJob{<:EnzymeTarget};
                 libraries::Bool=true, deferred_codegen::Bool=true, optimize::Bool=true, ctx = nothing,
                 strip::Bool=false, validate::Bool=true, only_entry::Bool=false, parent_job::Union{Nothing, CompilerJob} = nothing)
    params  = job.params
    mode   = params.mode
    adjoint = params.adjoint
    dupClosure = params.dupClosure
    width = params.width
    abiwrap = params.abiwrap
    primal  = job.source
    modifiedBetween = params.modifiedBetween

    if parent_job === nothing
        primal_target = GPUCompiler.NativeCompilerTarget()
        primal_params = Compiler.PrimalCompilerParams()
        primal_job    = CompilerJob(primal_target, primal, primal_params)
    else
        primal_job = similar(parent_job, job.source)
    end
    mod, meta = GPUCompiler.codegen(:llvm, primal_job; optimize=false, validate=false, parent_job=parent_job, ctx)
    primalf = meta.entry
    check_ir(job, mod)
    if Enzyme.API.EnzymeBitcodeReplacement(mod) != 0
        ModulePassManager() do pm
            instruction_combining!(pm)
            run!(pm, mod)
        end
        toremove = []
        for f in functions(mod)
            if !any(map(k->kind(k)==kind(EnumAttribute("alwaysinline"; ctx)), collect(function_attributes(f))))
                continue
            end
            if !any(map(k->kind(k)==kind(EnumAttribute("returns_twice"; ctx)), collect(function_attributes(f))))
                push!(function_attributes(f), EnumAttribute("returns_twice"; ctx))
                push!(toremove, name(f))
            end
            todo = LLVM.CallInst[]
            for u in LLVM.uses(f)
                ci = LLVM.user(u)
                if isa(ci, LLVM.CallInst) && called_value(ci) == f
                    push!(todo, ci)
                end
            end
            for ci in todo
                b = Builder(ctx)
                position!(b, ci)
                args = collect(collect(operands(ci))[1:LLVM.API.LLVMGetNumArgOperands(ci)]) 
                nc = call!(b, f, args)
                replace_uses!(ci, nc)
                LLVM.API.LLVMInstructionEraseFromParent(ci)
            end
        end

        for fname in ["cblas_xerbla"]
            if in(fname, functions(mod))
                f = functions(mod)[fname]
                if isempty(LLVM.blocks(f))
                    entry = BasicBlock(f, "entry"; ctx)
                    b = Builder(ctx)
                    position!(b, entry)
                    emit_error(b, "BLAS Error")
                    ret!(b)
                end
            end
        end
        
        ModulePassManager() do pm
            always_inliner!(pm)
            run!(pm, mod)
        end
        for fname in toremove
            if in(fname, functions(mod))
                f = functions(mod)[fname]
                LLVM.API.LLVMRemoveEnumAttributeAtIndex(f, reinterpret(LLVM.API.LLVMAttributeIndex, LLVM.API.LLVMAttributeFunctionIndex), kind(EnumAttribute("returns_twice"; ctx)))
            end
        end
        GPUCompiler.@safe_warn "Using fallback BLAS replacements, performance may be degraded"
        ModulePassManager() do pm
            global_optimizer!(pm)
            run!(pm, mod)
        end
    end

    @assert ctx == context(mod)
    custom = Dict{String, LLVM.API.LLVMLinkage}()
    must_wrap = false

    # Julia function to LLVM stem and arity
    known_ops = Dict(
        Base.cbrt => (:cbrt, 1),
        Base.sqrt => (:sqrt, 1),
        Base.sin => (:sin, 1),
        Base.sincos => (:__fd_sincos_1, 1),
        Base.:^ => (:pow, 2),
        Base.cos => (:cos, 1),
        Base.tan => (:tan, 1),
        Base.exp => (:exp, 1),
        Base.log => (:log, 1),
        Base.log2 => (:log2, 1),
        Base.log10 => (:log10, 1),
        Base.asin => (:asin, 1),
        Base.tanh => (:tanh, 1),
        Base.ldexp => (:ldexp, 2),
        Base.FastMath.tanh_fast => (:tanh, 1)
    )
    actualRetType = nothing
    for (mi, k) in meta.compiled
        k_name = GPUCompiler.safe_name(k.specfunc)
        haskey(functions(mod), k_name) || continue

        llvmfn = functions(mod)[k_name]
        if llvmfn == primalf
            actualRetType = k.ci.rettype
        end

        meth = mi.def
        name = meth.name
        jlmod  = meth.module

        function handleCustom(name, attrs=[], setlink=true)
            attributes = function_attributes(llvmfn)
            custom[k_name] = linkage(llvmfn)
            if setlink
              linkage!(llvmfn, LLVM.API.LLVMExternalLinkage)
            end
            for a in attrs
                push!(attributes, a)
            end
            push!(attributes, StringAttribute("enzymejl_mi", string(convert(Int, pointer_from_objref(mi))); ctx))
            push!(attributes, StringAttribute("enzyme_math", name; ctx))
            push!(attributes, EnumAttribute("noinline", 0; ctx))
            must_wrap |= llvmfn == primalf
            nothing
        end

        Base.isbindingresolved(jlmod, name) && isdefined(jlmod, name) || continue
        func = getfield(jlmod, name)

        sparam_vals = mi.specTypes.parameters[2:end] # mi.sparam_vals
        if func == Base.println || func == Base.print || func == Base.show ||
            func == Base.flush || func == Base.string || func == Base.print_to_string
            handleCustom("enz_noop", [StringAttribute("enzyme_inactive"; ctx)])
            continue
        end
        if func == Base.eps || func == Base.nextfloat || func == Base.prevfloat
            handleCustom("jl_inactive_inout", [StringAttribute("enzyme_inactive"; ctx),
                                      EnumAttribute("readnone", 0; ctx),
                                      EnumAttribute("speculatable", 0; ctx),
                                      StringAttribute("enzyme_shouldrecompute"; ctx)
                                                      ])
            continue
        end
        if func == Base.enq_work && length(sparam_vals) == 1 && first(sparam_vals) <: Task 
            handleCustom("jl_enq_work")
            continue
        end
        if func == Base.Threads.threadid || func == Base.Threads.nthreads
            name = (func == Base.Threads.threadid) ? "jl_threadid" : "jl_nthreads"
            handleCustom(name,
                   [EnumAttribute("readonly", 0; ctx),
                    EnumAttribute("inaccessiblememonly", 0; ctx),
                    EnumAttribute("speculatable", 0; ctx),
                    StringAttribute("enzyme_shouldrecompute"; ctx),
                    StringAttribute("enzyme_inactive"; ctx),
                                  ])
            continue
        end
        if func == Base.wait || func == Base._wait
            if length(sparam_vals) == 0 || 
                (length(sparam_vals) == 1 && first(sparam_vals) <: Task)
                handleCustom("jl_wait")
            end
            continue
        end
        if func == Base.Threads.threading_run
            if length(sparam_vals) == 1 || length(sparam_vals) == 2
                handleCustom("jl_threadsfor")
            end
            continue
        end
        if func == Enzyme.Parallel.pmap
            source_sig = Base.signature_type(func, sparam_vals)
            primal = llvmfn == primalf
            llvmfn = lower_convention(source_sig, mod, llvmfn, k.ci.rettype)
            k_name = LLVM.name(llvmfn)
            if primal
                primalf = llvmfn
            end
            handleCustom("jl_pmap", [], false)
            continue
        end

        func ∈ keys(known_ops) || continue
        name, arity = known_ops[func]
        length(sparam_vals) == arity || continue

        T = first(sparam_vals)
        isfloat = T ∈ (Float32, Float64)
        if !isfloat
            continue
        end
        if name == :ldexp
           sparam_vals[2] <: Integer || continue
        elseif name == :pow
           if sparam_vals[2] <: Integer 
              name = :powi
           elseif sparam_vals[2] != T
              continue
           end
        else
           all(==(T), sparam_vals) || continue
        end

        if name == :__fd_sincos_1
          source_sig = Base.signature_type(func, sparam_vals)
          llvmfn = lower_convention(source_sig, mod, llvmfn, k.ci.rettype)
          k_name = LLVM.name(llvmfn)
        end

        name = string(name)
        name = T == Float32 ? name*"f" : name

        handleCustom(name, [EnumAttribute("readnone", 0; ctx),
                    StringAttribute("enzyme_shouldrecompute"; ctx)])
    end

    @assert actualRetType !== nothing

    if must_wrap
        llvmfn = primalf
        FT = eltype(llvmtype(llvmfn)::LLVM.PointerType)::LLVM.FunctionType

        wrapper_f = LLVM.Function(mod, safe_name(LLVM.name(llvmfn)*"mustwrap"), FT)

        let builder = Builder(ctx)
            entry = BasicBlock(wrapper_f, "entry"; ctx)
            position!(builder, entry)

            res = call!(builder, llvmfn, collect(parameters(wrapper_f)))

            if LLVM.return_type(FT) == LLVM.VoidType(ctx)
                ret!(builder)
            else
                ret!(builder, res)
            end

            dispose(builder)
        end
        primalf = wrapper_f
    end

    source_sig = GPUCompiler.typed_signature(job)::Type
    primalf = lower_convention(source_sig, mod, primalf, actualRetType)

    if primal_job.target isa GPUCompiler.NativeCompilerTarget
        target_machine = JIT.get_tm()
    else
        target_machine = GPUCompiler.llvm_machine(primal_job.target)
    end

    parallel = Threads.nthreads() > 1
    process_module = false
    if parent_job !== nothing
        if parent_job.target isa GPUCompiler.PTXCompilerTarget ||
           parent_job.target isa GPUCompiler.GCNCompilerTarget
            parallel = true
        end
        if parent_job.target isa GPUCompiler.GCNCompilerTarget
           process_module = true
        end
    end


    # annotate
    annotate!(mod, mode)

    # Run early pipeline
    optimize!(mod, target_machine)

    if process_module
        GPUCompiler.optimize_module!(parent_job, mod)
    end

    if params.run_enzyme
        # Generate the adjoint
        adjointf, augmented_primalf = enzyme!(job, mod, primalf, adjoint, mode, width, parallel, actualRetType, dupClosure, abiwrap, modifiedBetween)
        toremove = []
        # Inline the wrapper
        for f in functions(mod)
            if !any(map(k->kind(k)==kind(EnumAttribute("alwaysinline"; ctx)), collect(function_attributes(f))))
                continue
            end
            if !any(map(k->kind(k)==kind(EnumAttribute("returns_twice"; ctx)), collect(function_attributes(f))))
                push!(function_attributes(f), EnumAttribute("returns_twice"; ctx))
                push!(toremove, name(f))
            end
        end 
        ModulePassManager() do pm
            always_inliner!(pm)
            run!(pm, mod)
        end
        for fname in toremove
            if in(fname, functions(mod))
                f = functions(mod)[fname]
                LLVM.API.LLVMRemoveEnumAttributeAtIndex(f, reinterpret(LLVM.API.LLVMAttributeIndex, LLVM.API.LLVMAttributeFunctionIndex), kind(EnumAttribute("returns_twice"; ctx)))
            end
        end
    else
        adjointf = primalf
        augmented_primalf = nothing
    end
    
    for (fname, lnk) in custom
        haskey(functions(mod), fname) || continue
        f = functions(mod)[fname]
        linkage!(f, lnk)
        iter = function_attributes(f)
        elems = Vector{LLVM.API.LLVMAttributeRef}(undef, length(iter))
        LLVM.API.LLVMGetAttributesAtIndex(iter.f, iter.idx, elems)
        for eattr in elems
            at = Attribute(eattr)
            if isa(at, LLVM.EnumAttribute)
                if kind(at) == kind(EnumAttribute("noinline"; ctx))
                    delete!(iter, at)
                    break
                end
            end
        end
    end
    for fname in ["__enzyme_float", "__enzyme_double", "__enzyme_integer", "__enzyme_pointer"]
        haskey(functions(mod), fname) || continue
        f = functions(mod)[fname]
        for u in uses(f)
            st = LLVM.user(u)
            LLVM.API.LLVMInstructionEraseFromParent(st)
        end
        LLVM.unsafe_delete!(mod, f)
    end

    linkage!(adjointf, LLVM.API.LLVMExternalLinkage)
    adjointf_name = name(adjointf)

    if augmented_primalf !== nothing
        linkage!(augmented_primalf, LLVM.API.LLVMExternalLinkage)
        augmented_primalf_name = name(augmented_primalf)
    end

    restore_lookups(mod)

    if parent_job !== nothing
        reinsert_gcmarker!(adjointf)
        augmented_primalf !== nothing && reinsert_gcmarker!(augmented_primalf)
        post_optimze!(mod, target_machine)
    end

    adjointf = functions(mod)[adjointf_name]

    # API.EnzymeRemoveTrivialAtomicIncrements(adjointf)

    if process_module
        GPUCompiler.process_module!(parent_job, mod)
    end

    adjointf = functions(mod)[adjointf_name]
    push!(function_attributes(adjointf), EnumAttribute("alwaysinline", 0; ctx=context(mod)))
    if augmented_primalf !== nothing
        augmented_primalf = functions(mod)[augmented_primalf_name]
    end

    for fn in functions(mod)
        fn == adjointf && continue
        augmented_primalf !== nothing && fn === augmented_primalf && continue
        isempty(LLVM.blocks(fn)) && continue
        linkage!(fn, LLVM.API.LLVMLinkerPrivateLinkage)
    end

    return mod, (;adjointf, augmented_primalf, entry=adjointf, compiled=meta.compiled)
end

##
# Thunk
##

# Compiler result
struct Thunk
    adjoint::Ptr{Cvoid}
    primal::Ptr{Cvoid}
end

@inline (thunk::CombinedAdjointThunk{F, RT, TT, Width, DF})(args...) where {F, Width, DF, RT, TT} =
   enzyme_call(thunk.adjoint, CombinedAdjointThunk, Width, TT, RT, thunk.fn, thunk.dfn, args...)

@inline (thunk::ForwardModeThunk{F, RT, TT, Width, DF})(args...) where {F, Width, DF, RT, TT} =
   enzyme_call(thunk.adjoint, ForwardModeThunk, Width, TT, RT, thunk.fn, thunk.dfn, args...)

@inline (thunk::AdjointThunk{F, RT, TT, Width, DF})(args...) where {F, Width, DF, RT, TT} =
   enzyme_call(thunk.adjoint, AdjointThunk, Width, TT, RT, thunk.fn, thunk.dfn, args...)

@inline (thunk::AugmentedForwardThunk{F, RT, TT, Width, DF})(args...) where {F, Width, DF, RT, TT} =
   enzyme_call(thunk.primal, AugmentedForwardThunk, Width, TT, RT, thunk.fn, thunk.dfn, args...)

function jl_set_typeof(v::Ptr{Cvoid}, T)
    tag = reinterpret(Ptr{Any}, reinterpret(UInt, v) - 8)
    Base.unsafe_store!(tag, T) # set tag
    return nothing
end

@generated function enzyme_call(fptr::Ptr{Cvoid}, ::Type{CC}, ::Type{Val{width}}, tt::Type{T},
                                rt::Type{RT}, f::F, df::DF, args::Vararg{Any, N}) where {F, T, RT, DF, N, CC, width}

    is_forward = CC <: AugmentedForwardThunk || CC <: ForwardModeThunk
    is_adjoint = CC <: AdjointThunk || CC <: CombinedAdjointThunk
    is_split   = CC <: AdjointThunk || CC <: AugmentedForwardThunk
    needs_tape = CC <: AdjointThunk

    argtt    = tt.parameters[1]
    rettype  = rt.parameters[1]
    argtypes = DataType[argtt.parameters...]
    argexprs = Union{Expr, Symbol}[:(args[$i]) for i in 1:N]
    if rettype <: Active
        @assert length(argtypes) + is_adjoint + needs_tape == length(argexprs)
    elseif rettype <: Const
        @assert length(argtypes)              + needs_tape == length(argexprs)
    else
        @assert length(argtypes)              + needs_tape == length(argexprs)
    end

    types = DataType[]

    if eltype(rettype) === Union{}
        error("return type is Union{}, giving up.")
    end

    sret_types  = []  # Julia types of all returned variables
    # By ref values we create and need to preserve
    ccexprs = Union{Expr, Symbol}[] # The expressions passed to the `llvmcall`

    if !GPUCompiler.isghosttype(F) && !Core.Compiler.isconstType(F)
        isboxed = GPUCompiler.deserves_argbox(F)
        argexpr = :(f)
        if isboxed
            push!(types, Any)
        else
            push!(types, F)
        end

        push!(ccexprs, argexpr)
        if DF != Nothing
            argexpr = :(df)
            if isboxed
                push!(types, Any)
            else
                push!(types, F)
            end
            push!(ccexprs, argexpr)
        end
    end

    for (i, T) in enumerate(argtypes)
        source_typ = eltype(T)
        if GPUCompiler.isghosttype(source_typ) || Core.Compiler.isconstType(source_typ)
            @assert T <: Const
            continue
        end
        expr = argexprs[i]

        isboxed = GPUCompiler.deserves_argbox(source_typ)
        argexpr = Expr(:., expr, QuoteNode(:val))
        if isboxed
            push!(types, Any)
        else
            push!(types, source_typ)
        end

        push!(ccexprs, argexpr)

        T <: Const && continue

        if T <: Active
            if is_adjoint
                if width == 1
                    push!(sret_types, source_typ)
                else
                    push!(sret_types, NTuple{width, source_typ})
                end
            end
        elseif T <: Duplicated || T <: DuplicatedNoNeed
            argexpr =  Expr(:., expr, QuoteNode(:dval))
            if isboxed
                push!(types, Any)
            else
                push!(types, source_typ)
            end
            push!(ccexprs, argexpr)
        elseif T <: BatchDuplicated || T <: BatchDuplicatedNoNeed
            argexpr =  Expr(:., expr, QuoteNode(:dval))
            isboxedvec = GPUCompiler.deserves_argbox(NTuple{width, source_typ})
            if isboxedvec
                push!(types, Any)
            else
                push!(types, NTuple{width, source_typ})
            end
            push!(ccexprs, argexpr)
        else
            error("calling convention should be annotated, got $T")
        end
    end

    # API.DFT_OUT_DIFF
    if is_adjoint && rettype <: Active
        # TODO handle batch width
        @assert allocatedinline(eltype(rettype))
        push!(types, eltype(rettype))
        idx = length(argtypes) + 1
        push!(ccexprs, argexprs[idx])
    end

    if needs_tape
        # TODO
        push!(types, Ptr{Cvoid})
        push!(ccexprs, last(argexprs))
    end

    if is_forward
        # Tape
        if CC <: AugmentedForwardThunk 
            push!(sret_types, Ptr{Cvoid})
        end

        returnUsed = !(GPUCompiler.isghosttype(eltype(rettype)) || Core.Compiler.isconstType(eltype(rettype)))
        if returnUsed
            jlRT = eltype(rettype)
            if typeof(jlRT) == UnionAll
              # Future improvement, add tye assertion on load
              jlRT = DataType
            end
            if (CC <: AugmentedForwardThunk && !(rettype <: DuplicatedNoNeed || rettype <: BatchDuplicatedNoNeed ) ) || ((CC <: ForwardModeThunk) && (rettype <: Duplicated || rettype <: BatchDuplicated))
                push!(sret_types, jlRT)
            end
            if rettype <: Duplicated || rettype <: DuplicatedNoNeed
                push!(sret_types, jlRT)
            elseif rettype <: BatchDuplicated || rettype <: BatchDuplicatedNoNeed
                push!(sret_types, NTuple{width, jlRT})
            end
        end
    end


	# calls fptr
	ctx = LLVM.Context()
	llvmtys = LLVMType[convert(LLVMType, x; ctx, allow_boxed=true) for x in types]
    if !isempty(sret_types)
      llsret_types = LLVMType[convert(LLVMType, x; ctx, allow_boxed=true) for x in sret_types]
      T_sjoint = LLVM.StructType(llsret_types; ctx)
      if in(Any, sret_types)
        for T in llsret_types
          pushfirst!(llvmtys, convert(LLVMType, Ptr{Cvoid}; ctx))
        end
      else
		@assert allocatedinline(Tuple{sret_types...})
        pushfirst!(llvmtys, convert(LLVMType, Ptr{Cvoid}; ctx))
      end
	end
    pushfirst!(llvmtys, convert(LLVMType, Ptr{Cvoid}; ctx))
    T_void = convert(LLVMType, Nothing; ctx)
    T_ret = T_void
	llvm_f, _ = LLVM.Interop.create_function(T_ret, llvmtys)

	mod = LLVM.parent(llvm_f)
    i64 = LLVM.IntType(64; ctx)
	LLVM.Builder(ctx) do builder
		entry = BasicBlock(llvm_f, "entry"; ctx)
		position!(builder, entry)
		params = collect(LLVM.Value, parameters(llvm_f))
		lfn = @inbounds params[1]
		params = params[2:end]
        callparams = params
        if in(Any, sret_types)
            callparams = params[(length(sret_types)+1):end]
            alloc = LLVM.alloca!(builder, T_sjoint)
            pushfirst!(callparams, alloc)
        end
		lfn = inttoptr!(builder, lfn, LLVM.PointerType(LLVM.FunctionType(T_void, [llvmtype(x) for x in callparams])))
		call!(builder, lfn, callparams)
        if in(Any, sret_types)
            for (i, (parm, sret)) in enumerate(zip(params[1:length(sret_types)], llsret_types))
                out = LLVM.load!(builder, LLVM.gep!(builder, alloc, [LLVM.ConstantInt(LLVM.IntType(64; ctx), 0), LLVM.ConstantInt(LLVM.IntType(32; ctx), i-1)]))
                parm = LLVM.inttoptr!(builder, parm, LLVM.PointerType(sret))
                LLVM.store!(builder, out, parm)
            end
		end
        ret!(builder)
	end

	ir = string(mod)
	fn = LLVM.name(llvm_f)

    @assert length(types) == length(ccexprs)
    if !isempty(sret_types)
		# Any case needed since this cannot be included inside a tuple
        if in(Any, sret_types)

        msrets = (:($(Symbol(:ref, i)) = Ref{$x}()) for (i, x) in enumerate(sret_types))
        gcsrets = (:($(Symbol(:ref, i))) for (i, x) in enumerate(sret_types))
        tptrs = (:($(Symbol(:tptr, i)) = Base.unsafe_convert(Ptr{Cvoid}, Base.unsafe_convert(Ptr{$x}, $(Symbol(:ref,i)) ) ) ) for (i, x) in enumerate(sret_types))
        voidptrs = (:(Ptr{Cvoid}) for _ in 1:length(sret_types))
        tptrres = (:($(Symbol(:tptr, i)) ) for (i, x) in enumerate(sret_types))
        results = (:($(Symbol(:ref, i))[] ) for (i, x) in enumerate(sret_types))
        return quote
            Base.@_inline_meta

            let $(msrets...)
            GC.@preserve $(gcsrets...) begin
                $(tptrs...)
                Base.llvmcall(($ir, $fn), Cvoid,
                    Tuple{Ptr{Cvoid},
                    $(voidptrs...),
                    $(types...)},
                    fptr,
                    $(tptrres...),
                    $(ccexprs...))
            end
            return ( $(results...), )
            end
        end

        else
            return quote
                Base.@_inline_meta
                    sret = Ref{$(Tuple{sret_types...})}()
                    GC.@preserve sret begin
                       tret = Base.pointer_from_objref(sret)
                       Base.llvmcall(($ir, $fn), Cvoid,
                        Tuple{Ptr{Cvoid}, Ptr{Cvoid}, $(types...)},
                        fptr, tret, $(ccexprs...))
                    end
                    sret[]
            end
        end
    else
        return quote
            Base.@_inline_meta
            Base.llvmcall(($ir, $fn), Cvoid,
                Tuple{Ptr{Cvoid}, $(types...),},
                fptr, $(ccexprs...))
            return ()
        end
    end
end

##
# JIT
##

function _link(job, (mod, adjoint_name, primal_name, ctx))
    params = job.params
    adjoint = params.adjoint

    primal = job.source

    # Now invoke the JIT
    jitted_mod = JIT.add!(mod)
    if VERSION >= v"1.9.0-DEV.115"
        LLVM.dispose(ctx)
    else
        # we cannot dispose of the global unique context
    end
    adjoint_addr = JIT.lookup(jitted_mod, adjoint_name)

    adjoint_ptr  = pointer(adjoint_addr)
    if adjoint_ptr === C_NULL
        throw(GPUCompiler.InternalCompilerError(job, "Failed to compile Enzyme thunk, adjoint not found"))
    end
    if primal_name === nothing
        primal_ptr = C_NULL
    else
        primal_addr = JIT.lookup(jitted_mod, primal_name)
        primal_ptr  = pointer(primal_addr)
        if primal_ptr === C_NULL
            throw(GPUCompiler.InternalCompilerError(job, "Failed to compile Enzyme thunk, primal not found"))
        end
    end

    return Thunk(adjoint_ptr, primal_ptr)
end

# actual compilation
function _thunk(job)
    params = job.params

    # TODO: on 1.9, this actually creates a context. cache those.
    ctx = JuliaContext()
    mod, meta = codegen(:llvm, job; optimize=false, ctx)

    adjointf, augmented_primalf = meta.adjointf, meta.augmented_primalf

    adjoint_name = name(adjointf)

    if augmented_primalf !== nothing
        primal_name = name(augmented_primalf)
    else
        primal_name = nothing
    end

    # Enzyme kills dead instructions, and removes the ptls call
    # which we need for correct GC handling in LateLowerGC
    reinsert_gcmarker!(adjointf)
    augmented_primalf !== nothing && reinsert_gcmarker!(augmented_primalf)

    # Run post optimization pipeline
    post_optimze!(mod, JIT.get_tm())
    return (mod, adjoint_name, primal_name, ctx)
end

const cache = Dict{UInt, Thunk}()

const cache_lock = ReentrantLock()
@inline function cached_compilation(@nospecialize(job::CompilerJob), key, specid)
    # XXX: CompilerJob contains a world age, so can't be respecialized.
    #      have specialization_id take a f/tt and return a world to construct a CompilerJob?
    key = hash(hash(job, specid), key)

    # XXX: by taking the hash, we index the compilation cache directly with the world age.
    #      that's wrong; we should perform an intersection with the entry its bounds.

    # NOTE: no use of lock(::Function)/@lock/get! to keep stack traces clean
    lock(cache_lock)
    try
        obj = get(cache, key, nothing)
        if obj === nothing
            asm = _thunk(job)
            obj = _link(job, asm)
            cache[key] = obj
        end
        obj
    finally
        unlock(cache_lock)
    end
end


@generated function genthunk(f::F, df::DF, ::Type{A}, tt::Type{TT},::Val{Mode}, ::Val{ModifiedBetween}, ::Val{width}, ::Val{specid}) where {F, DF, A<:Annotation, TT, Mode, ModifiedBetween, width, specid}
    primal, adjoint = fspec(F, TT)

    target = Compiler.EnzymeTarget()
    params = Compiler.EnzymeCompilerParams(adjoint, Mode, width, A, true, DF != Nothing, #=abiwrap=#true, ModifiedBetween)
    job    = Compiler.CompilerJob(target, primal, params)

    sig = Tuple{F, map(eltype, TT.parameters)...}

    # world = ...
    
    interp = Core.Compiler.NativeInterpreter(job.source.world)

    # TODO check compile return here, early
    # rrt = Core.Compiler.return_type(f, primal.tt) # nothing
    rrt = nothing
    for m in Base._methods_by_ftype(sig, -1, job.source.world)::Vector
        m = m::Core.MethodMatch
        ty = Core.Compiler.typeinf_type(interp, m.method, m.spec_types, m.sparams)
        rrt = something(ty, Any)
        break
    end

    if rrt == Union{}
        error("Return type inferred to be Union{}. Giving up.")
    end
 
    if A isa UnionAll
        rt = A{rrt}
    else
        @assert A isa DataType
        # Can we relax this condition?
        # @assert eltype(A) == rrt
        rt = A
    end

    if rrt == Nothing && !(A <: Const)
        error("Return of nothing must be marked Const")
    end

    # @assert isa(rrt, DataType)

    # We need to use primal as the key, to lookup the right method
    # but need to mixin the hash of the adjoint to avoid cache collisions
    # This is counter-intuitive since we would expect the cache to be split
    # by the primal, but we want the generated code to be invalidated by
    # invalidations of the primal, which is managed by GPUCompiler.


    thunk = cached_compilation(job, hash(hash(hash(adjoint, hash(rt, UInt64(Mode))), UInt64(width)), UInt64(ModifiedBetween)), specid)::Thunk
    if Mode == API.DEM_ReverseModePrimal || Mode == API.DEM_ReverseModeGradient
        return quote
            augmented = AugmentedForwardThunk{F, $rt, $(adjoint.tt), Val{width} , DF}(f, $(thunk.primal), df)
            adjoint  = AdjointThunk{F, $rt, $(adjoint.tt), Val{width}, DF}(f, $(thunk.adjoint), df)
            (augmented, adjoint)
        end
    elseif Mode == API.DEM_ReverseModeCombined
        return quote
            CombinedAdjointThunk{F, $rt, $(adjoint.tt), Val{width}, DF}(f, $(thunk.adjoint), df)
        end
    elseif Mode == API.DEM_ForwardMode
        return quote
            ForwardModeThunk{F, $rt, $(adjoint.tt), Val{width}, DF}(f, $(thunk.adjoint), df)
        end
    else
        @assert false
    end
end

@inline function thunk(f::F,df::DF, ::Type{A}, tt::Type{TT},::Val{Mode}, ::Val{width}, ::Val{ModifiedBetween}=Val(Mode != API.DEM_ReverseModeCombined)) where {F, DF, A<:Annotation, TT, Mode, width, ModifiedBetween}
    primal, adjoint = fspec(F, TT)
    target = Compiler.EnzymeTarget()
    params = Compiler.EnzymeCompilerParams(adjoint, Mode, width, A, true, DF != Nothing, #=abiwrap=#true, ModifiedBetween)
    job    = Compiler.CompilerJob(target, primal, params)

    specid = GPUCompiler.specialization_id(job)

    genthunk(f, df, A, TT, Val(Mode), Val(ModifiedBetween), Val(width), Val(specid))
end

import GPUCompiler: deferred_codegen_jobs

@generated function deferred_codegen(f::F, ::Val{tt}, ::Val{rt}, ::Val{DupClosure},::Val{Mode},
                                     ::Val{width}, ::Val{ModifiedBetween}=Val(Mode != API.DEM_ReverseModeCombined)) where {F,tt, rt, DupClosure, Mode, width, ModifiedBetween}
    primal, adjoint = fspec(F, tt)
    target = EnzymeTarget()
    params = EnzymeCompilerParams(adjoint, Mode, width, rt, true, DupClosure, #=abiwrap=#true, ModifiedBetween)
    job    = CompilerJob(target, primal, params)

    addr = get_trampoline(job)
    id = Base.reinterpret(Int, pointer(addr))

    deferred_codegen_jobs[id] = job
    trampoline = reinterpret(Ptr{Cvoid}, id)

    quote
        ccall("extern deferred_codegen", llvmcall, Ptr{Cvoid}, (Ptr{Cvoid},), $trampoline)
    end
end

include("compiler/reflection.jl")
include("compiler/validation.jl")

end
