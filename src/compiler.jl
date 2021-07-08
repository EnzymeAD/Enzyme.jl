module Compiler

import ..Enzyme: Const, Active, Duplicated, DuplicatedNoNeed
import ..Enzyme: API, TypeTree, typetree, only!, shift!, data0!, TypeAnalysis, FnTypeInfo, Logic

using LLVM, GPUCompiler, Libdl
import Enzyme_jll

import GPUCompiler: CompilerJob, FunctionSpec, codegen
using LLVM.Interop
import LLVM: Target, TargetMachine

if LLVM.has_orc_v1()
    include("compiler/orcv1.jl")
else
    include("compiler/orcv2.jl")
end

using .JIT

function array_inner(::Type{<:Array{T}}) where T
    return T
end
function array_shadow_handler(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, numArgs::Csize_t, Args::Ptr{LLVM.API.LLVMValueRef})::LLVM.API.LLVMValueRef
    inst = LLVM.Instruction(OrigCI)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(inst)))
    ctx = LLVM.context(LLVM.Value(OrigCI))

    ce = operands(inst)[1]
    while isa(ce, ConstantExpr)
        ce = operands(ce)[1]
    end
    ptr = reinterpret(Ptr{Cvoid}, convert(UInt64, ce))
    typ = array_inner(Base.unsafe_pointer_to_objref(ptr))

    b = LLVM.Builder(B)

    vals = Vector{LLVM.Value}()
    for i = 1:numArgs
        push!(vals, LLVM.Value(unsafe_load(Args, i)))
    end

    anti = LLVM.call!(b, LLVM.Value(LLVM.API.LLVMGetCalledValue(OrigCI)), vals)

    prod = LLVM.Value(unsafe_load(Args, 2))
    for i = 3:numArgs
        prod = LLVM.mul!(b, prod, LLVM.Value(unsafe_load(Args, i)))
    end

    isunboxed = typ.isinlinealloc
    elsz = sizeof(typ)

    isunion = typ <: Union

    LLT_ALIGN(x, sz) = (((x) + (sz)-1) & ~((sz)-1))

    if !isunboxed
        elsz = sizeof(Ptr{Cvoid})
        al = elsz;
    else
        al = 1 # check
        elsz = LLT_ALIGN(elsz, al)
    end

    tot = prod
    tot = LLVM.mul!(b, tot, LLVM.ConstantInt(LLVM.llvmtype(tot), elsz, false))

    if elsz == 1 && !isunion
        tot = LLVM.add!(b, t, LLVM.ConstantInt(LLVM.llvmtype(tot), 1, false))
    end
    if (isunion)
        # an extra byte for each isbits union array element, stored after a->maxsize
        tot = LLVM.add!(b, tot, prod)
    end

    i1 = LLVM.IntType(1; ctx)
    i8 = LLVM.IntType(8; ctx)
    ptrty = LLVM.PointerType(i8) #, LLVM.addrspace(LLVM.llvmtype(anti)))
    toset = LLVM.load!(b, LLVM.pointercast!(b, anti, LLVM.PointerType(ptrty, LLVM.addrspace(LLVM.llvmtype(anti)))))

    memtys = LLVM.LLVMType[ptrty, LLVM.llvmtype(tot)]
    memset = LLVM.Function(mod, LLVM.Intrinsic("llvm.memset"), memtys)
    memargs = LLVM.Value[toset, LLVM.ConstantInt(i8, 0, false), tot, LLVM.ConstantInt(i1, 0, false)]

    mcall = LLVM.call!(b, memset, memargs)
    ref::LLVM.API.LLVMValueRef = Base.unsafe_convert(LLVM.API.LLVMValueRef, anti)
    return ref
end

function null_free_handler(B::LLVM.API.LLVMBuilderRef, ToFree::LLVM.API.LLVMValueRef, Fn::LLVM.API.LLVMValueRef)::LLVM.API.LLVMValueRef
    return C_NULL
end

function __init__()
    API.EnzymeRegisterAllocationHandler(
        "jl_alloc_array_1d",
        @cfunction(array_shadow_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, Csize_t, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(null_free_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, LLVM.API.LLVMValueRef))
    )
    API.EnzymeRegisterAllocationHandler(
        "jl_alloc_array_2d",
        @cfunction(array_shadow_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, Csize_t, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(null_free_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, LLVM.API.LLVMValueRef))
    )
    API.EnzymeRegisterAllocationHandler(
        "jl_alloc_array_3d",
        @cfunction(array_shadow_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, Csize_t, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(null_free_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, LLVM.API.LLVMValueRef))
    )
    # TODO handle array copy
    # API.EnzymeRegisterAllocationHandler(
    #     "jl_array_copy",
    #     @cfunction(copy_shadow_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, Csize_t, Ptr{LLVM.API.LLVMValueRef})),
    #     @cfunction(copy_free_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, LLVM.API.LLVMValueRef))
    # )
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
    malloc(sz) = ccall("extern malloc", llvmcall, Csize_t, (Csize_t,), sz)
    report_oom(sz) = return
    report_exception(ex) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return
end

abstract type AbstractEnzymeCompilerParams <: AbstractCompilerParams end
struct EnzymeCompilerParams <: AbstractEnzymeCompilerParams
    adjoint::FunctionSpec
    split::Bool
    run_enzyme::Bool
end

struct PrimalCompilerParams <: AbstractEnzymeCompilerParams
end

## job

# TODO: We shouldn't blanket opt-out
GPUCompiler.check_invocation(job::CompilerJob{EnzymeTarget}, entry::LLVM.Function) = nothing

GPUCompiler.runtime_module(::CompilerJob{<:Any,<:AbstractEnzymeCompilerParams}) = Runtime
# GPUCompiler.isintrinsic(::CompilerJob{EnzymeTarget}, fn::String) = true
# GPUCompiler.can_throw(::CompilerJob{EnzymeTarget}) = true

# TODO: encode debug build or not in the compiler job
#       https://github.com/JuliaGPU/CUDAnative.jl/issues/368
GPUCompiler.runtime_slug(job::CompilerJob{EnzymeTarget}) = "enzyme"

include("compiler/optimize.jl")

"""
Create the `FunctionSpec` pair, and lookup the primal return type.
"""
@inline function fspec(f::F, tt::TT) where {F, TT}
    # Entry for the cache look-up
    adjoint = FunctionSpec(f, tt, #=kernel=# false, #=name=# nothing)

    # primal function. Inferred here to get return type
    _tt = (tt.parameters...,)

    primal_tt = Tuple{map(eltype, _tt)...}
    primal = FunctionSpec(f, primal_tt, #=kernel=# false, #=name=# nothing)

    return primal, adjoint
end

##
# Enzyme compiler step
##

function annotate!(mod)
    ctx = context(mod)
    inactive = LLVM.StringAttribute("enzyme_inactive", ""; ctx)
    fns = functions(mod)
    for inactivefn in ["jl_gc_queue_root", "gpu_report_exception", "gpu_signal_exception", "julia.ptls_states", "julia.write_barrier", "julia.typeof", "jl_box_int64", "jl_subtype"]
        if haskey(fns, inactivefn)
            fn = fns[inactivefn]
            push!(function_attributes(fn), inactive)
        end
    end

    if haskey(fns, "julia.ptls_states")
        fn = fns["julia.ptls_states"]
        push!(function_attributes(fn), LLVM.EnumAttribute("readnone", 0; ctx))
    end


    for boxfn in ["jl_box_int64"]
        if haskey(fns, boxfn)
            fn = fns[boxfn]
            push!(return_attributes(fn), LLVM.EnumAttribute("noalias", 0; ctx))
            push!(function_attributes(fn), LLVM.EnumAttribute("inaccessiblememonly", 0; ctx))
        end
    end

end

function alloc_obj_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
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

function enzyme!(job, mod, primalf, adjoint, split, parallel)
    primal = job.source
    rt = Core.Compiler.return_type(primal.f, primal.tt)
    ctx     = context(mod)
    dl      = string(LLVM.datalayout(mod))

    tt = [adjoint.tt.parameters...,]

    if rt === Union{}
        error("return type is Union{}, giving up.")
    end

    args_activity     = API.CDIFFE_TYPE[]
    uncacheable_args  = Bool[]
    args_typeInfo     = TypeTree[]
    args_known_values = API.IntList[]

    ctx = LLVM.context(mod)
    if !GPUCompiler.isghosttype(typeof(adjoint.f)) && !Core.Compiler.isconstType(typeof(adjoint.f))
        push!(args_activity, API.DFT_CONSTANT)
        typeTree = typetree(typeof(adjoint.f), ctx, dl)
        push!(args_typeInfo, typeTree)
        if split
            push!(uncacheable_args, true)
        else
            push!(uncacheable_args, false)
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
        elseif  T <: Duplicated
            push!(args_activity, API.DFT_DUP_ARG)
        elseif T <: DuplicatedNoNeed
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
    if rt <: Integer || rt <: DataType
        retType = API.DFT_CONSTANT
    elseif rt <: AbstractFloat
        retType = API.DFT_OUT_DIFF
    elseif rt <: Complex{<:AbstractFloat}
        retType = API.DFT_OUT_DIFF
    elseif GPUCompiler.isghosttype(rt) || Core.Compiler.isconstType(rt)
        retType = API.DFT_CONSTANT
    else
        error("Unhandled return type $rt")
    end

    rules = Dict{String, API.CustomRuleType}(
        "julia.gc_alloc_obj" => @cfunction(alloc_obj_rule,
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
                                                    Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef))
    )

    TA = TypeAnalysis(triple(mod), rules)
    logic = Logic()

    if GPUCompiler.isghosttype(rt)|| Core.Compiler.isconstType(rt)
        retTT = typetree(Nothing, ctx, dl)
    else
        retTT = typetree(rt, ctx, dl)
    end

    typeInfo = FnTypeInfo(retTT, args_typeInfo, args_known_values)

    if split
        augmented = API.EnzymeCreateAugmentedPrimal(
            logic, primalf, retType, args_activity, TA, #=returnUsed=# true,
            typeInfo, uncacheable_args, #=forceAnonymousTape=# false, #=atomicAdd=# parallel, #=postOpt=# false)

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
            logic, primalf, retType, args_activity, TA,
            #=returnValue=#false, #=dretUsed=#false, #=topLevel=#false,
            #=additionalArg=#tape, typeInfo,
            uncacheable_args, augmented, #=atomicAdd=# parallel, #=postOpt=#false))
    else
        adjointf = LLVM.Function(API.EnzymeCreatePrimalAndGradient(
            logic, primalf, retType, args_activity, TA,
            #=returnValue=#false, #=dretUsed=#false, #=topLevel=#true,
            #=additionalArg=#C_NULL, typeInfo,
            # uncacheable_args, #=augmented=#C_NULL, #=atomicAdd=# parallel, #=postOpt=#false))
            uncacheable_args, #=augmented=#C_NULL, #=atomicAdd=# parallel, #=postOpt=#false))
        augmented_primalf = nothing
    end
    return adjointf, augmented_primalf
end

function Base.in(attr::LLVM.EnumAttribute, iter::LLVM.FunctionAttrSet)
    elems = Vector{LLVM.API.LLVMAttributeRef}(undef, length(iter))
    if length(iter) > 0
      # FIXME: this prevents a nullptr ref in LLVM similar to D26392
      LLVM.API.LLVMGetAttributesAtIndex(iter.f, iter.idx, elems)
    end
    for eattr in elems
        at = Attribute(eattr)
        if isa(at, LLVM.EnumAttribute)
            if kind(at) == kind(attr)
                return true
            end
        end
    end
    return false
end

# Modified from GPUCompiler/src/irgen.jl:365 lower_byval
function lower_convention(@nospecialize(job::CompilerJob), mod::LLVM.Module, entry_f::LLVM.Function)
    ctx = context(mod)
    entry_ft = eltype(llvmtype(entry_f)::LLVM.PointerType)::LLVM.FunctionType

    RT = return_type(entry_ft)
    args = GPUCompiler.classify_arguments(job, entry_f)
    filter!(args) do arg
        arg.cc != GPUCompiler.GHOST
    end

    # generate the wrapper function type & definition
    wrapper_types = LLVM.LLVMType[]
    sret = 0
    if !isempty(parameters(entry_f)) && EnumAttribute("sret"; ctx) in parameter_attributes(entry_f, 1)
        RT = eltype(llvmtype(first(parameters(entry_f))))
        sret = 1
    end
    for (parm, arg) in zip(collect(parameters(entry_f))[1+sret:end], args)
        typ = if !GPUCompiler.deserves_argbox(arg.typ) && arg.cc == GPUCompiler.BITS_REF
            eltype(arg.codegen.typ)
        else
            llvmtype(parm)
        end
        push!(wrapper_types, typ)
    end
    wrapper_fn = LLVM.name(entry_f)
    LLVM.name!(entry_f, wrapper_fn * ".inner")
    wrapper_ft = LLVM.FunctionType(RT, wrapper_types)
    wrapper_f = LLVM.Function(mod, wrapper_fn, wrapper_ft)

    # emit IR performing the "conversions"
    let builder = Builder(ctx)
        entry = BasicBlock(wrapper_f, "entry"; ctx)
        position!(builder, entry)

        wrapper_args = Vector{LLVM.Value}()

        if !isempty(parameters(entry_f)) && EnumAttribute("sret"; ctx) in parameter_attributes(entry_f, 1)
            sretPtr = alloca!(builder, llvmtype(parameters(wrapper_f)[1]))
            push!(wrapper_args, sretPtr)
        end

        # perform argument conversions
        for (parm, arg) in zip(collect(parameters(entry_f))[1+sret:end], args)
            if !GPUCompiler.deserves_argbox(arg.typ) && arg.cc == GPUCompiler.BITS_REF
                # copy the argument value to a stack slot, and reference it.
                ty = llvmtype(parm)
                ptr = alloca!(builder, eltype(ty))
                if LLVM.addrspace(ty) != 0
                    ptr = addrspacecast!(builder, ptr, ty)
                end
                store!(builder, parameters(wrapper_f)[arg.codegen.i], ptr)
                push!(wrapper_args, ptr)
            else
                push!(wrapper_args, parameters(wrapper_f)[arg.codegen.i])
                for attr in collect(parameter_attributes(entry_f, arg.codegen.i+sret))
                    push!(parameter_attributes(wrapper_f, arg.codegen.i), attr)
                end
            end
        end
        res = call!(builder, entry_f, wrapper_args)

        if sret == 1
            ret!(builder, load!(builder, sretPtr))
        elseif return_type(entry_ft) == LLVM.VoidType(ctx)
            ret!(builder)
        else
            ret!(builder, res)
        end

        dispose(builder)
    end

    # early-inline the original entry function into the wrapper
    push!(function_attributes(entry_f), EnumAttribute("alwaysinline", 0; ctx))
    linkage!(entry_f, LLVM.API.LLVMInternalLinkage)

    # copy debug info
    sp = LLVM.get_subprogram(entry_f)
    if sp !== nothing
        LLVM.set_subprogram!(wrapper_f, sp)
    end

    GPUCompiler.fixup_metadata!(entry_f)
    ModulePassManager() do pm
        always_inliner!(pm)
        run!(pm, mod)
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
    return wrapper_f
end

function adim(::Array{T, N}) where {T, N}
    return N
end

function GPUCompiler.codegen(output::Symbol, job::CompilerJob{<:EnzymeTarget};
                 libraries::Bool=true, deferred_codegen::Bool=true, optimize::Bool=true,
                 strip::Bool=false, validate::Bool=true, only_entry::Bool=false, parent_job::Union{Nothing, CompilerJob} = nothing)
    params  = job.params
    split   = params.split
    adjoint = params.adjoint
    primal  = job.source

    if parent_job === nothing
        primal_target = GPUCompiler.NativeCompilerTarget()
        primal_params = Compiler.PrimalCompilerParams()
        primal_job    = CompilerJob(primal_target, primal, primal_params)
    else
        primal_job = similar(parent_job, job.source)
    end
    mod, meta = GPUCompiler.codegen(:llvm, primal_job, optimize=false, validate=false, parent_job=parent_job)
    primalf = meta.entry

    known_fns = check_ir(job, mod)

    ctx = context(mod)
    custom = []
    must_wrap = false

    # Julia function to LLVM stem and arity
    known_ops = Dict(
        Base.sin => (:sin, 1),
        Base.cos => (:cos, 1),
        Base.tan => (:tan, 1),
        Base.exp => (:exp, 1),
        Base.log => (:log, 1),
        Base.asin => (:asin, 1),
        Base.tanh => (:tanh, 1)
    )
    for (mi, k) in meta.compiled
        meth = mi.def
        name = meth.name
        jlmod  = meth.module

        Base.isbindingresolved(jlmod, name) && isdefined(jlmod, name) || continue
        func = getfield(jlmod, name)

        sparam_vals = mi.sparam_vals

        if func == Base.copy && length(sparam_vals) == 1 && first(sparam_vals) <: Array
            AT = first(sparam_vals)
            T = eltype(AT)
            N = adim(AT)
            bitsunion = Base.isbitsunion(T)
            error("jl_copy unhandled")
        end

        func ∈ keys(known_ops) || continue

        name, arity = known_ops[func]

        length(sparam_vals) == arity || continue

        T = first(sparam_vals)
        T ∈ (Float32, Float64) && all(==(T), sparam_vals) || continue
        name = string(name)
        name = T == Float32 ? name*"f" : name

        llvmfn = functions(mod)[k.specfunc]
        push!(custom, llvmfn)

        attributes = function_attributes(llvmfn)
        push!(attributes, EnumAttribute("noinline", 0; ctx))
        push!(attributes, StringAttribute("enzyme_math", name; ctx))

        # Need to wrap the code when outermost
        must_wrap |= llvmfn == primalf
    end

    if must_wrap
        llvmfn = primalf
        FT = eltype(llvmtype(llvmfn)::LLVM.PointerType)::LLVM.FunctionType

        wrapper_f = LLVM.Function(mod, LLVM.name(llvmfn)*"wrap", FT)

        let builder = Builder(ctx)
            entry = BasicBlock(wrapper_f, "entry"; ctx)
            position!(builder, entry)

            res = call!(builder, llvmfn, collect(parameters(wrapper_f)))

            if return_type(FT) == LLVM.VoidType(ctx)
                ret!(builder)
            else
                ret!(builder, res)
            end

            dispose(builder)
        end
        primalf = wrapper_f
    end

    if primal_job.target isa GPUCompiler.NativeCompilerTarget
        target_machine = JIT.get_tm()
    else
        target_machine = GPUCompiler.llvm_machine(primal_job.target)
    end

    parallel = false
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

    primalf = lower_convention(job, mod, primalf)

    # annotate
    annotate!(mod)

    # Run early pipeline
    optimize!(mod, target_machine)

    if process_module
        GPUCompiler.optimize_module!(parent_job, mod)
    end

    if params.run_enzyme
        # Generate the adjoint
        adjointf, augmented_primalf = enzyme!(job, mod, primalf, adjoint, split, parallel)
    else
        adjointf = primalf
        augmented_primalf = nothing
    end

    for f in custom
        iter = function_attributes(f)
        elems = Vector{LLVM.API.LLVMAttributeRef}(undef, length(iter))
        LLVM.API.LLVMGetAttributesAtIndex(iter.f, iter.idx, elems)
        for eattr in elems
            at = Attribute(eattr)
            if isa(at, LLVM.EnumAttribute)
                if kind(at) == "noinline"
                    delete!(iter, at)
                end
            end
        end
    end

    linkage!(adjointf, LLVM.API.LLVMExternalLinkage)
    adjointf_name = name(adjointf)

    if augmented_primalf !== nothing
        linkage!(augmented_primalf, LLVM.API.LLVMExternalLinkage)
        augmented_primalf_name = name(augmented_primalf)
    end

    restore_lookups(mod, known_fns)

    if parent_job !== nothing
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
    return mod, (;adjointf, augmented_primalf)
end

##
# Thunk
##

struct CombinedAdjointThunk{F, RT, TT, Pullback}
    fn::F
    # primal::Ptr{Cvoid}
    adjoint::Ptr{Cvoid}
end

@inline (thunk::CombinedAdjointThunk{F, RT, TT, Pullback})(args...) where {F, RT, TT, Pullback} =
   enzyme_call(thunk.adjoint, thunk.fn, Pullback, TT, RT, args...)

@generated function enzyme_call(fptr::Ptr{Cvoid}, f::F, pullback::Val{Pullback}, tt::Type{T}, rt::Type{RT}, args::Vararg{Any, N}) where {F, T, RT, N, Pullback}
    argtt    = tt.parameters[1]
    rettype  = rt.parameters[1]
    argtypes = DataType[argtt.parameters...]
    argexprs = Union{Expr, Symbol}[:(args[$i]) for i in 1:N]
    if Pullback && (rettype <: AbstractFloat || rettype <: Complex{<:AbstractFloat})
        @assert length(argtypes)+1 == length(argexprs)
    else
        @assert length(argtypes)   == length(argexprs)
    end

    types = DataType[]

    if rettype === Union{}
        error("return type is Union{}, giving up.")
    end

    LLVM.Context() do ctx
        T_void = convert(LLVMType, Nothing; ctx)
        ptr8 = LLVM.PointerType(LLVM.IntType(8; ctx))
        T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
        T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)

        # Create Enzyme calling convention
        T_wrapperargs = LLVMType[] # Arguments of the wrapper
        T_EnzymeSRet = LLVMType[] # Struct returns of Active variables in the enzyme call
                                  # Equal to all Active vars passed by value
        T_JuliaSRet = LLVMType[]  # Struct return of all Active variables (includes all of T_EnzymeSRet)
        sret_types  = DataType[]  # Julia types of all Active variables
        inputexprs = Union{Expr, Symbol}[]
        # By ref values we create and need to preserve
        ccexprs = Union{Expr, Symbol}[] # The expressions passed to the `llvmcall`

        if !GPUCompiler.isghosttype(F) && !Core.Compiler.isconstType(F)
            isboxed = GPUCompiler.deserves_argbox(F)
            llvmT = isboxed ? T_prjlvalue : convert(LLVMType, F; ctx)
            argexpr = :(f)
            if isboxed
                push!(types, Any)
            else
                push!(types, F)
            end

            push!(ccexprs, argexpr)
            push!(T_wrapperargs, llvmT)
        end

        for (i, T) in enumerate(argtypes)
            source_typ = eltype(T)
            if GPUCompiler.isghosttype(source_typ) || Core.Compiler.isconstType(source_typ)
                @assert T <: Const
                continue
            end
            expr = argexprs[i]

            isboxed = GPUCompiler.deserves_argbox(source_typ)
            llvmT = isboxed ? T_prjlvalue : convert(LLVMType, source_typ; ctx)
            argexpr = Expr(:., expr, QuoteNode(:val))
            if isboxed
                push!(types, Any)
            else
                push!(types, source_typ)
            end


            push!(ccexprs, argexpr)
            push!(T_wrapperargs, llvmT)

            T <: Const && continue

            if T <: Active
                # Use deserves_argbox??
                llvmT = convert(LLVMType, source_typ; ctx)
                push!(sret_types, source_typ)
                push!(T_JuliaSRet, llvmT)
                if !isboxed # XXX: Not consistent
                    push!(T_EnzymeSRet, llvmT)
                end
            elseif T <: Duplicated || T <: DuplicatedNoNeed
                argexpr =  Expr(:., expr, QuoteNode(:dval))
                if isboxed
                    push!(types, Any)
                else
                    push!(types, source_typ)
                end
                push!(ccexprs, argexpr)
                push!(T_wrapperargs, llvmT)
            else
                error("calling convention should be annotated, got $T")
            end
        end

        # API.DFT_OUT_DIFF
        if rettype <: AbstractFloat || rettype <: Complex{<:AbstractFloat}
            push!(types, rettype)
            push!(T_wrapperargs, convert(LLVMType, rettype; ctx))
            if !Pullback
                push!(ccexprs, :(one($rettype)))
            else
                push!(ccexprs, last(argexprs))
            end
        end
        # XXX: What if not `Nothing`/`Missing` what if struct or array or...

        if !isempty(T_EnzymeSRet)
            ret = LLVM.StructType(T_EnzymeSRet; ctx)
        else
            ret = T_void
        end

        # pointer to call
        pushfirst!(T_wrapperargs, convert(LLVMType, Int; ctx))

        # sret argument
        if !isempty(sret_types)
            pushfirst!(T_wrapperargs, convert(LLVMType, Int; ctx))
        end

        llvm_f, _ = LLVM.Interop.create_function(T_void, T_wrapperargs)
        mod = LLVM.parent(llvm_f)
        dl = datalayout(mod)

        params = [parameters(llvm_f)...]
        target =  !isempty(sret_types) ? 2 : 1

        intrinsic_typ = LLVM.FunctionType(T_void, [ptr8, LLVM.IntType(8; ctx), LLVM.IntType(64; ctx), LLVM.IntType(1; ctx)])
        memsetIntr = LLVM.Function(mod, "llvm.memset.p0i8.i64", intrinsic_typ)
        LLVM.Builder(ctx) do builder
            entry = BasicBlock(llvm_f, "entry"; ctx)
            position!(builder, entry)

            realparms = LLVM.Value[]
            i = target+1

            if !isempty(T_JuliaSRet)
                sret = inttoptr!(builder, params[1], LLVM.PointerType(LLVM.StructType(T_JuliaSRet; ctx)))
            end

            activeNum = 0

            if !GPUCompiler.isghosttype(F) && !Core.Compiler.isconstType(F)
                push!(realparms, params[i])
                i+=1
            end

            for T in argtypes
                T′ = eltype(T)

                if GPUCompiler.isghosttype(T′) || Core.Compiler.isconstType(T′)
                    continue
                end
                push!(realparms, params[i])
                i+=1
                if T <: Const
                elseif T <: Active
                    isboxed = GPUCompiler.deserves_argbox(T′)
                    if isboxed
                        ptr = gep!(builder, sret, [LLVM.ConstantInt(LLVM.IntType(64; ctx), 0), LLVM.ConstantInt(LLVM.IntType(32; ctx), activeNum)])
                        cst = pointercast!(builder, ptr, ptr8)
                        push!(realparms, ptr)

                        cparms = LLVM.Value[cst,
                        LLVM.ConstantInt(LLVM.IntType(8; ctx), 0),
                        LLVM.ConstantInt(LLVM.IntType(64; ctx), LLVM.storage_size(dl, Base.eltype(LLVM.llvmtype(ptr)) )),
                        LLVM.ConstantInt(LLVM.IntType(1; ctx), 0)]
                        call!(builder, memsetIntr, cparms)
                    end
                    activeNum+=1
                elseif T <: Duplicated || T <: DuplicatedNoNeed
                    push!(realparms, params[i])
                    i+=1
                end
            end

            # Primal Differential Return type
            if rettype <: AbstractFloat || rettype <: Complex{<:AbstractFloat}
                push!(realparms, params[i])
            end


            E_types = LLVM.LLVMType[]
            for p in realparms
                push!(E_types, LLVM.llvmtype(p))
            end
            ft = LLVM.FunctionType(ret, E_types)

            ptr = inttoptr!(builder, params[target], LLVM.PointerType(ft))
            val = call!(builder, ptr, realparms)
            if !isempty(T_JuliaSRet)
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

        ir = string(mod)
        fn = LLVM.name(llvm_f)

        if !isempty(T_JuliaSRet)
            quote
                Base.@_inline_meta
                sret = Ref{$(Tuple{sret_types...})}()
                GC.@preserve sret begin
                    tptr = Base.unsafe_convert(Ptr{$(Tuple{sret_types...})}, sret)
                    tptr = Base.unsafe_convert(Ptr{Cvoid}, tptr)
                    Base.llvmcall(($ir,$fn), Cvoid,
                        $(Tuple{Ptr{Cvoid}, Ptr{Cvoid}, types...}),
                        tptr, fptr, $(ccexprs...))
                end
                sret[]
            end
        else
            quote
                Base.@_inline_meta
                Base.llvmcall(($ir,$fn), Cvoid,
                    $(Tuple{Ptr{Cvoid}, types...}),
                    fptr, $(ccexprs...))
            end
        end
    end
end

##
# JIT
##

function _link(job, (mod, adjoint_name, primal_name))
    params = job.params
    adjoint = params.adjoint
    split = params.split

    primal = job.source
    rt = Core.Compiler.return_type(primal.f, primal.tt)

    # Now invoke the JIT
    jitted_mod = JIT.add!(mod)
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

    @assert primal_name === nothing
    return CombinedAdjointThunk{typeof(adjoint.f), rt, adjoint.tt, #=pullback=#Val(false)}(adjoint.f, #=primal_ptr,=# adjoint_ptr)
end

# actual compilation
function _thunk(job)
    params = job.params

    mod, meta = codegen(:llvm, job, optimize=false)

    adjointf, augmented_primalf = meta.adjointf, meta.augmented_primalf

    adjoint_name = name(adjointf)

    if augmented_primalf !== nothing
        primal_name = name(augmented_primalf)
    else
        primal_name = nothing
    end

    # Run post optimization pipeline
    post_optimze!(mod, JIT.get_tm())

    return (mod, adjoint_name, primal_name)
end

const cache = Dict{UInt, Dict{UInt, Any}}()

function thunk(f::F,tt::TT=Tuple{},::Val{Split}=Val(false)) where {F<:Core.Function, TT<:Type, Split}
    primal, adjoint = fspec(f, tt)
    # We need to use primal as the key, to lookup the right method
    # but need to mixin the hash of the adjoint to avoid cache collisions
    # This is counter-intuitive since we would expect the cache to be split
    # by the primal, but we want the generated code to be invalidated by
    # invalidations of the primal, which is managed by GPUCompiler.
    local_cache = get!(Dict{Int, Any}, cache, hash(adjoint, UInt64(Split)))

    target = Compiler.EnzymeTarget()
    params = Compiler.EnzymeCompilerParams(adjoint, Split, true)
    job    = Compiler.CompilerJob(target, primal, params)

    rt = Core.Compiler.return_type(primal.f, primal.tt)

    GPUCompiler.cached_compilation(local_cache, job, _thunk, _link)::CombinedAdjointThunk{F,rt,tt}
end

import GPUCompiler: deferred_codegen_jobs

@generated function deferred_codegen(::Val{f}, ::Val{tt}) where {f,tt}
    primal, adjoint = fspec(f, tt)
    target = EnzymeTarget()
    params = EnzymeCompilerParams(adjoint, false, true)
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
