module Compiler

import ..Enzyme: Const, Active, Duplicated, DuplicatedNoNeed
import ..Enzyme: API, TypeTree, typetree, TypeAnalysis, FnTypeInfo, Logic

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
include("compiler/cassette.jl")

"""
Create the `FunctionSpec` pair, and lookup the primal return type.
"""
@inline function fspec(f::F, tt::TT, ::Val{cassette}=Val(true)) where {F, TT, cassette}
    # Entry for the cache look-up
    adjoint = FunctionSpec(f, tt, #=kernel=# false, #=name=# nothing)

    # primal function. Inferred here to get return type
    _tt = (tt.parameters...,)

    if cassette
        overdub_tt = Tuple{typeof(Compiler.CTX), F, map(eltype, _tt)...}
        primal = FunctionSpec(Cassette.overdub, overdub_tt, #=kernel=# false, #=name=# nothing)
    else
        primal_tt = Tuple{map(eltype, _tt)...}
        primal = FunctionSpec(f, primal_tt, #=kernel=# false, #=name=# nothing)
    end

    return primal, adjoint
end

##
# Enzyme compiler step
##

function annotate!(mod)
    inactive = LLVM.StringAttribute("enzyme_inactive", "", context(mod))
    fns = functions(mod)
    for inactivefn in ["jl_gc_queue_root", "gpu_report_exception", "gpu_signal_exception", "julia.ptls_states", "julia.write_barrier", "julia.typeof", "jl_box_int64"]
        if haskey(fns, inactivefn)
            fn = fns[inactivefn]
            push!(function_attributes(fn), inactive)
        end
    end

    if haskey(fns, "julia.ptls_states")
        fn = fns["julia.ptls_states"]
        push!(function_attributes(fn), LLVM.EnumAttribute("readnone", 0, context(mod)))
        # push!(function_attributes(fn), LLVM.EnumAttribute("readonly", 0, context(mod)))
    end


    for boxfn in ["jl_box_int64"]
        if haskey(fns, boxfn)
            fn = fns[boxfn]
            push!(return_attributes(fn), LLVM.EnumAttribute("noalias", 0, context(mod)))
            push!(function_attributes(fn), LLVM.EnumAttribute("inaccessiblememonly", 0, context(mod)))
        end
    end

end

function passbyref(T::DataType)
    if T <: Array
        return false 
    else
        # LLVM.Interop.isboxed(T)
        return !isprimitivetype(T)
    end
end


function enzyme!(job, mod, primalf, adjoint, split, parallel)
    primal = job.source
    rt = Core.Compiler.return_type(primal.f, primal.tt)
    ctx     = context(mod)
    dl      = string(LLVM.datalayout(mod))

    tt = [adjoint.tt.parameters...,]

    # argClassification = GPUCompiler.classify_arguments(job, primalf)
    # filter!(argClassification) do arg
    #     arg.cc != GPUCompiler.GHOST
    # end

    args_activity     = API.CDIFFE_TYPE[]
    uncacheable_args  = Bool[]
    args_typeInfo     = TypeTree[]
    args_known_values = API.IntList[]

    ctx = LLVM.context(mod)

    for T in tt
        source_typ = eltype(T)
        if GPUCompiler.isghosttype(source_typ) || Core.Compiler.isconstType(source_typ)
            @assert T <: Const
            continue
        end
        isboxed = GPUCompiler.deserves_argbox(source_typ) || isa(convert(LLVMType, source_typ, ctx), LLVM.SequentialType)
        
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
            @assert("illegal annotation type")
        end
        T = eltype(T)
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
    if rt <: Integer
        retType = API.DFT_CONSTANT
    elseif rt <: AbstractFloat
        retType = API.DFT_OUT_DIFF
    elseif rt == Nothing || Core.Compiler.isconstType(rt)
        retType = API.DFT_CONSTANT
    else
        error("What even is $rt")
    end

    TA = TypeAnalysis(triple(mod)) 
    logic = Logic()

    if rt == Nothing || Core.Compiler.isconstType(rt)
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
    mod, primalf = GPUCompiler.codegen(:llvm, primal_job, optimize=false, validate=false, parent_job=parent_job)
    check_ir(job, mod)
    if primal_job.target isa GPUCompiler.NativeCompilerTarget
        target_machine = tm[]
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

    linkage!(adjointf, LLVM.API.LLVMExternalLinkage)
    adjointf_name = name(adjointf)

    if augmented_primalf !== nothing
        linkage!(augmented_primalf, LLVM.API.LLVMExternalLinkage)
        augmented_primalf_name = name(augmented_primalf)
    end

    if parent_job !== nothing
        post_optimze!(mod, target_machine)
    end

    adjointf = functions(mod)[adjointf_name]
    # API.EnzymeRemoveTrivialAtomicIncrements(adjointf)

    if process_module
        GPUCompiler.process_module!(parent_job, mod)
    end
    adjointf = functions(mod)[adjointf_name]
    push!(function_attributes(adjointf), EnumAttribute("alwaysinline", 0, context(mod)))
    if augmented_primalf === nothing
        return mod, adjointf
    else
        augmented_primalf = functions(mod)[augmented_primalf_name]
        return mod, (adjointf, augmented_primalf)
    end
end

##
# Thunk
## 

struct Thunk{f, RT, TT#=, Split=#}
    # primal::Ptr{Cvoid}
    adjoint::Ptr{Cvoid}
end

@inline (thunk::Thunk{F, RT, TT})(args...) where {F, RT, TT} =
   enzyme_call(thunk.adjoint, TT, RT, args...)

@generated function enzyme_call(f::Ptr{Cvoid}, tt::Type{T}, rt::Type{RT}, args::Vararg{Any, N}) where {T, RT, N}
    argtt    = tt.parameters[1]
    rettype  = rt.parameters[1]
    argtypes = DataType[argtt.parameters...]
    argexprs = Union{Expr, Symbol}[:(args[$i]) for i in 1:N]
    @assert length(argtypes) == length(argexprs)

    types = DataType[]

    LLVM.Interop.JuliaContext() do ctx
        T_void = convert(LLVMType, Nothing, ctx)
        ptr8 = LLVM.PointerType(LLVM.IntType(8, ctx))
        T_jlvalue = LLVM.StructType(LLVMType[], ctx)
        T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)

        # Create Enzyme calling convention
        T_wrapperargs = LLVMType[] # Arguments of the wrapper
        T_EnzymeSRet = LLVMType[] # Struct returns of Active variables in the enzyme call
                                  # Equal to all Active vars passed by value
        T_JuliaSRet = LLVMType[]  # Struct return of all Active variables (includes all of T_EnzymeSRet)
        sret_types  = DataType[]  # Julia types of all Active variables
        inputexprs = Union{Expr, Symbol}[]

        argpreserve = Union{Symbol}[]   # By ref values we create and need to preserve
        ccexprs = Union{Expr, Symbol}[] # The expressions passed to the `llvmcall`

        # function byref(expr, T)
        #     val = gensym(:val)
        #     push!(inputexprs, :($val = Ref($expr)))
        #     push!(argpreserve, val)
        #     :(Base.unsafe_convert($T, Base.cconvert($T, $val)))
        # end

        for (i, T) in enumerate(argtypes)
            source_typ = eltype(T)
            if GPUCompiler.isghosttype(source_typ) || Core.Compiler.isconstType(source_typ)
                @assert T <: Const
                continue
            end
            expr = argexprs[i]

            isboxed = GPUCompiler.deserves_argbox(source_typ)
            llvmT = isboxed ? T_prjlvalue : convert(LLVMType, source_typ, ctx)

            argexpr = Expr(:., expr, QuoteNode(:val))
            if isboxed
                push!(types, Any)
            elseif isa(llvmT, LLVM.SequentialType) # et->isAggregateType
                # push!(types, Core.LLVMPtr{source_typ, 0}) # XXX: AS?
                push!(types, Base.RefValue{source_typ}) # XXX: AS?
                argexpr = Expr(:call, GlobalRef(Base, :Ref), argexpr)
                llvmT = T_prjlvalue # LLVM.PointerType(llvmT)
            else
                push!(types, source_typ)
            end
            push!(ccexprs, argexpr)
            push!(T_wrapperargs, llvmT)
            
            T <: Const && continue

            if T <: Active
                # Use deserves_argbox??
                llvmT = convert(LLVMType, source_typ, ctx)
                push!(sret_types, source_typ)
                push!(T_JuliaSRet, llvmT)
                if !passbyref(source_typ) # XXX: Not consistent
                    push!(T_EnzymeSRet, llvmT)
                end
            elseif T <: Duplicated || T <: DuplicatedNoNeed
                argexpr =  Expr(:., expr, QuoteNode(:dval))
                if isboxed
                    push!(types, Any)
                elseif isa(llvmT, LLVM.SequentialType) # et->isAggregateType
                    push!(types, Ptr{source_typ})
                    argexpr = Expr(:call, GlobalRef(Base, :Ref), argexpr)
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
        if rettype <: AbstractFloat
            push!(types, rettype)
            push!(T_wrapperargs, convert(LLVMType, rettype, ctx))
            push!(ccexprs, :(one($rettype)))
        end
        # XXX: What if not `Nothing`/`Missing` what if struct or array or...

        if !isempty(T_EnzymeSRet)
            ret = LLVM.StructType(T_EnzymeSRet)
        else
            ret = T_void
        end

        # pointer to call
        pushfirst!(T_wrapperargs, convert(LLVMType, Int, ctx))

        # sret argument
        if !isempty(sret_types)
            pushfirst!(T_wrapperargs, convert(LLVMType, Int, ctx))
        end

        llvm_f, _ = LLVM.Interop.create_function(T_void, T_wrapperargs)
        mod = LLVM.parent(llvm_f)
        dl = datalayout(mod)

        params = [parameters(llvm_f)...]
        target =  !isempty(sret_types) ? 2 : 1

        intrinsic_typ = LLVM.FunctionType(T_void, [ptr8, LLVM.IntType(8, ctx), LLVM.IntType(64, ctx), LLVM.IntType(1, ctx)])
        memsetIntr = LLVM.Function(mod, "llvm.memset.p0i8.i64", intrinsic_typ)
        LLVM.Builder(ctx) do builder
            entry = BasicBlock(llvm_f, "entry", ctx)
            position!(builder, entry)

            realparms = LLVM.Value[]
            i = target+1

            if !isempty(T_JuliaSRet) 
                sret = inttoptr!(builder, params[1], LLVM.PointerType(LLVM.StructType(T_JuliaSRet)))
            end

            activeNum = 0

            for T in argtypes
                T′ = eltype(T)
                ispassbyref = passbyref(T′)

                if GPUCompiler.isghosttype(T′) || Core.Compiler.isconstType(T′)
                    continue
                end
                push!(realparms, params[i])
                i+=1
                if T <: Const
                elseif T <: Active
                    if ispassbyref
                        #push!(realparms, params[i])
                        #store!(builder, params[i], gep!(builder, sret, [LLVM.ConstantInt(LLVM.IntType(64, ctx), 0), LLVM.ConstantInt(LLVM.IntType(32, ctx), activeNum)]))
                        
                        # llvmT = convert(LLVMType, T′, ctx, allow_boxed=true)
                        # @show llvmT
                        ptr = gep!(builder, sret, [LLVM.ConstantInt(LLVM.IntType(64, ctx), 0), LLVM.ConstantInt(LLVM.IntType(32, ctx), activeNum)])
                        # ptr = load!(builder, ptr)
                        # @show ptr
                        cst = pointercast!(builder, ptr, ptr8)
                        push!(realparms, ptr)

                        cparms = LLVM.Value[cst, 
                        LLVM.ConstantInt(LLVM.IntType(8, ctx), 0),
                        LLVM.ConstantInt(LLVM.IntType(64, ctx), LLVM.storage_size(dl, Base.eltype(LLVM.llvmtype(ptr)) )),
                        LLVM.ConstantInt(LLVM.IntType(1, ctx), 0)]
                        call!(builder, memsetIntr, cparms)
                        # i+=1
                    end
                    activeNum+=1
                elseif T <: Duplicated || T <: DuplicatedNoNeed
                    push!(realparms, params[i])
                    i+=1
                end
            end

            # Primal Return type
            if i <= size(params, 1)
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
                    ispassbyref = passbyref(T′)
                    if T <: Active
                        if !ispassbyref
                            eval = extract_value!(builder, val, returnNum)
                            store!(builder, eval, gep!(builder, sret, [LLVM.ConstantInt(LLVM.IntType(64, ctx), 0), LLVM.ConstantInt(LLVM.IntType(32, ctx), activeNum)]))
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

        # @show (ir, fn)
        # @show Tuple{Ptr{Cvoid}, Ptr{Cvoid}, types...}
        # @show f, (ccexprs...)
        if !isempty(T_JuliaSRet)  
            quote
                Base.@_inline_meta
                sret = Ref{$(Tuple{sret_types...})}()
                # $(inputexprs...)
                GC.@preserve sret $(argpreserve...) begin
                    ptr = Base.unsafe_convert(Ptr{$(Tuple{sret_types...})}, sret)
                    ptr = Base.unsafe_convert(Ptr{Cvoid}, ptr)
                    Base.llvmcall(($ir,$fn), Cvoid,
                        $(Tuple{Ptr{Cvoid}, Ptr{Cvoid}, types...}),
                        ptr, f, $(ccexprs...))
                end
                sret[]
            end
        else 
            quote
                Base.@_inline_meta
                # $(inputexprs...)
                GC.@preserve $(argpreserve...) begin
                    Base.llvmcall(($ir,$fn), Cvoid,
                        $(Tuple{Ptr{Cvoid}, types...}),
                        f, $(ccexprs...))
                end
            end
        end
    end
end

##
# JIT
##

function resolver(name, ctx)
    name = unsafe_string(name)
    ptr = try
        ## Step 0: Should have already resolved it iff it was in the
        ##         same module
        ## Step 1: See if it's something known to the execution enging
        # TODO: Do we need to do this?
        # address(jit[], name)

        ## Step 2: Search the program symbols
        #
        # SearchForAddressOfSymbol expects an unmangled 'C' symbol name.
        # Iff we are on Darwin, strip the leading '_' off.
        @static if Sys.isapple()
            if name[1] == '_'
                name = name[2:end]
            end
        end
        LLVM.API.LLVMSearchForAddressOfSymbol(name)
        ## Step 4: Lookup in libatomic
        # TODO: Do we need to do this?
    catch ex
        @error "Enzyme: Lookup failed" jl_name exception=(ex, Base.catch_backtrace())
        C_NULL
    end
    if ptr === C_NULL
        @show name
        error("Enzyme: Symbol lookup failed. Aborting!")
    end

    return UInt64(reinterpret(UInt, ptr))
end

function _link(job, (mod, adjoint_name, primal_name))
    params = job.params
    adjoint = params.adjoint
    split = params.split

    primal = job.source 
    rt = Core.Compiler.return_type(primal.f, primal.tt)

    # Now invoke the JIT
    orc = jit[]
    jitted_mod = compile!(orc, mod, @cfunction(resolver, UInt64, (Cstring, Ptr{Cvoid})))

    adjoint_addr = addressin(orc, jitted_mod, adjoint_name)
    adjoint_ptr  = pointer(adjoint_addr)
    if adjoint_ptr === C_NULL
        throw(GPUCompiler.InternalCompilerError(job, "Failed to compile Enzyme thunk, adjoint not found"))
    end
    if primal_name === nothing
        primal_ptr = C_NULL
    else
        primal_addr = addressin(orc, jitted_mod, primal_name)
        primal_ptr  = pointer(primal_addr)
        if primal_ptr === C_NULL
            throw(GPUCompiler.InternalCompilerError(job, "Failed to compile Enzyme thunk, primal not found"))
        end
    end

    @assert primal_name === nothing
    return Thunk{typeof(adjoint.f), rt, adjoint.tt #=, split=#}(#=primal_ptr,=# adjoint_ptr)
end

# actual compilation
function _thunk(job)
    params = job.params

    mod, fns = codegen(:llvm, job, optimize=false)

    if fns isa Tuple
        adjointf, augmented_primalf = fns
    else
        adjointf = fns
        augmented_primalf = nothing
    end

    adjoint_name = name(adjointf)

    if augmented_primalf !== nothing
        primal_name = name(augmented_primalf)
    else
        primal_name = nothing
    end

    # Run post optimization pipeline
    post_optimze!(mod, tm[])

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

    GPUCompiler.cached_compilation(local_cache, job, _thunk, _link)::Thunk{F,rt,tt,#=Split=#}
end

import GPUCompiler: deferred_codegen_jobs

mutable struct CallbackContext
    job::Compiler.CompilerJob
    stub::String
    compiled::Bool
end

const outstanding = IdDict{CallbackContext, Nothing}()

# Setup the lazy callback for creating a module
function callback(orc_ref::LLVM.API.LLVMOrcJITStackRef, callback_ctx::Ptr{Cvoid})
    orc = OrcJIT(orc_ref)
    cc = Base.unsafe_pointer_to_objref(callback_ctx)::CallbackContext

    @assert !cc.compiled
    job = cc.job

    thunk = Compiler._link(job, Compiler._thunk(job))
    cc.compiled = true
    delete!(outstanding, cc)

    # 4. Update the stub pointer to point to the recently compiled module
    set_stub!(orc, cc.stub, thunk.adjoint)

    # 5. Return the address of tie implementation, since we are going to call it now
    ptr = thunk.adjoint
    return UInt64(reinterpret(UInt, ptr))
end

@generated function deferred_codegen(::Val{f}, ::Val{tt}, ::Val{cassette}) where {f,tt,cassette}
    primal, adjoint = fspec(f, tt, Val(cassette))
    target = EnzymeTarget()
    params = EnzymeCompilerParams(adjoint, false, true)
    job    = CompilerJob(target, primal, params)

    cc = CallbackContext(job, String(gensym(:trampoline)), false)
    outstanding[cc] = nothing

    c_callback = @cfunction(callback, UInt64, (LLVM.API.LLVMOrcJITStackRef, Ptr{Cvoid}))

    orc = Compiler.jit[]
    initial_addr = callback!(orc, c_callback, pointer_from_objref(cc))
    create_stub!(orc, cc.stub, initial_addr)
    addr = address(orc, cc.stub)
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