module Compiler

import ..Enzyme
import Enzyme: Const, Active, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed,
               Annotation, guess_activity, eltype, 
               API, TypeTree, typetree, only!, shift!, data0!, merge!,
               TypeAnalysis, FnTypeInfo, Logic, allocatedinline

using Enzyme

using LLVM, GPUCompiler, Libdl
import Enzyme_jll

import GPUCompiler: CompilerJob, FunctionSpec, codegen, safe_name
using LLVM.Interop
import LLVM: Target, TargetMachine

using Random
using Printf

if LLVM.has_orc_v1()
    include("compiler/orcv1.jl")
else
    include("compiler/orcv2.jl")
end

unsafe_to_pointer(ptr) = ccall(Base.@cfunction(x->x, Ptr{Cvoid}, (Ptr{Cvoid},)), Ptr{Cvoid}, (Any,), ptr)
    
# Julia function to LLVM stem and arity
const known_ops = Dict(
    Base.cbrt => (:cbrt, 1),
    Base.sqrt => (:sqrt, 1),
    Base.sin => (:sin, 1),
    Base.sincos => (:__fd_sincos_1, 1),
    Base.:^ => (:pow, 2),
    Base.cos => (:cos, 1),
    Base.tan => (:tan, 1),
    Base.exp => (:exp, 1),
    Base.exp2 => (:exp2, 1),
    Base.expm1 => (:expm1, 1),
    Base.exp10 => (:exp10, 1),
    Base.FastMath.exp_fast => (:exp, 1),
    Base.log => (:log, 1),
    Base.log1p => (:log1p, 1),
    Base.log2 => (:log2, 1),
    Base.log10 => (:log10, 1),
    Base.asin => (:asin, 1),
    Base.sinh => (:sinh, 1),
    Base.FastMath.sinh_fast => (:sinh, 1),
    Base.cosh => (:cosh, 1),
    Base.FastMath.cosh_fast => (:cosh, 1),
    Base.tanh => (:tanh, 1),
    Base.ldexp => (:ldexp, 2),
    Base.FastMath.tanh_fast => (:tanh, 1)
)

const inactivefns = Set{String}((
    "jl_gc_queue_root", "gpu_report_exception", "gpu_signal_exception",
    "julia.ptls_states", "julia.write_barrier", "julia.typeof",
    "jl_box_int64", "jl_box_int32",
    "ijl_box_int64", "ijl_box_int32",
    "jl_subtype", "julia.get_pgcstack", "jl_in_threaded_region",
    "jl_object_id_", "jl_object_id", "ijl_object_id_", "ijl_object_id",
    "jl_breakpoint",
    "llvm.julia.gc_preserve_begin","llvm.julia.gc_preserve_end", "jl_get_ptls_states",
    "jl_f_fieldtype",
    "jl_symbol_n",
    "jl_stored_inline", "ijl_stored_inline",
    "jl_f_apply_type", "jl_f_issubtype", "jl_isa",
    "jl_matching_methods", "ijl_matching_methods",
    "jl_excstack_state", "jl_current_exception",
    "memhash_seed",
    "jl_f__typevar", "ijl_f__typevar",
    "jl_f_isa", "ijl_f_isa",
    "jl_set_task_threadpoolid", "ijl_set_task_threadpoolid"
    # "jl_"
))

const InactiveFunctions = Set([Base.CoreLogging.logmsg_code,
                               Base.CoreLogging.shouldlog,
                               Base.to_tuple_type,
                               Base.methods,
                               Base.println,
                               Base.print,
                               Base.show,
                               Base.flush,
                               Base.string,
                               Base.repr,
                               Base.print_to_string,
                               Base.Threads.threadid,
                               Base.Threads.nthreads,
                               Base.eps,
                               Base.nextfloat,
                               Base.prevfloat,
                               Base.Val,
                               Core.kwfunc,
                               Random.rand,
                               Random.rand!,
                               Random.randn,
                               Random.default_rng,
                               Random.seed!
                               ])

const activefns = Set{String}((
    "jl_",
))

# User facing interface
abstract type AbstractThunk{F, RT, TT, Width, DF} end

struct CombinedAdjointThunk{F, RT, TT, Width, DF, ReturnPrimal} <: AbstractThunk{F, RT, TT, Width, DF}
    fn::F
    adjoint::Ptr{Cvoid}
    dfn::DF
end

struct ForwardModeThunk{F, RT, TT, Width, DF, ReturnPrimal} <: AbstractThunk{F, RT, TT, Width, DF}
    fn::F
    adjoint::Ptr{Cvoid}
    dfn::DF
end

struct AugmentedForwardThunk{F, RT, TT, Width, DF, ReturnPrimal, TapeType} <: AbstractThunk{F, RT, TT, Width, DF}
    fn::F
    primal::Ptr{Cvoid}
    dfn::DF
end

struct AdjointThunk{F, RT, TT, Width, DF, TapeType} <: AbstractThunk{F, RT, TT, Width, DF}
    fn::F
    adjoint::Ptr{Cvoid}
    dfn::DF
end
return_type(::AbstractThunk{F, RT, TT, Width, DF}) where {F, RT, TT, Width, DF} = RT

using .JIT


safe_println(head, tail) =  ccall(:jl_safe_printf, Cvoid, (Cstring, Cstring...), "%s%s\n",head, tail)
macro safe_show(exs...)
    blk = Expr(:block)
    for ex in exs
        push!(blk.args, :($safe_println($(sprint(Base.show_unquoted, ex)*" = "),
            repr(begin local value = $(esc(ex)) end))))
    end
    isempty(exs) || push!(blk.args, :value)
    return blk
end

declare_allocobj!(mod) = get_function!(mod, "julia.gc_alloc_obj") do ctx
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)
    
	T_pjlvalue = LLVM.PointerType(T_jlvalue)
    T_ppjlvalue = LLVM.PointerType(T_pjlvalue)

	#TODO make size_t > 32 => 64, else 32
	T_size = LLVM.IntType((sizeof(Csize_t)*8) > 32 ? 64 : 32; ctx)
    LLVM.FunctionType(T_prjlvalue, [LLVM.PointerType(T_ppjlvalue), T_size, T_prjlvalue])
end
function emit_allocobj!(B, T, size)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
	ctx = context(mod)
    func = declare_allocobj!(mod)
	
	T_size = parameters(eltype(llvmtype(func)))[2]
    
	T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)

	ty = inttoptr!(B, LLVM.ConstantInt(convert(Int, pointer_from_objref(T)); ctx), LLVM.PointerType(T_jlvalue))
	ty = addrspacecast!(B, ty, T_prjlvalue)
	size = LLVM.ConstantInt(T_size, size)
    args = [reinsert_gcmarker!(fn), size, ty]
	args[1] = bitcast!(B, args[1], parameters(eltype(llvmtype(func)))[1])
    return call!(B, func, args)
end
function emit_allocobj!(B, T)
    emit_allocobj!(B, T, sizeof(T))
end
declare_pointerfromobjref!(mod) = get_function!(mod, "julia.pointer_from_objref") do ctx
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 11) 
    T_pjlvalue = LLVM.PointerType(T_jlvalue)
    LLVM.FunctionType(T_pjlvalue, [T_prjlvalue])
end
function emit_pointerfromobjref!(B, T)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    ctx = context(mod)
    func = declare_pointerfromobjref!(mod)
    return call!(B, func, [T])
end

declare_writebarrier!(mod) = get_function!(mod, "julia.write_barrier") do ctx
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, 10) 
    LLVM.FunctionType(LLVM.VoidType(ctx), [T_prjlvalue]; vararg=true)
end
function emit_writebarrier!(B, T)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    func = declare_writebarrier!(mod)
    return call!(B, func, T)
end

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

    vals = LLVM.Value[]
    for i = 1:numArgs
        push!(vals, LLVM.Value(unsafe_load(Args, i)))
    end

    anti = LLVM.call!(b, LLVM.Value(LLVM.API.LLVMGetCalledValue(OrigCI)), vals)

    prod = LLVM.Value(unsafe_load(Args, 2))
    for i = 3:numArgs
        prod = LLVM.mul!(b, prod, LLVM.Value(unsafe_load(Args, i)))
    end

    isunboxed = allocatedinline(typ)

    isunion = typ isa Union

    LLT_ALIGN(x, sz) = (((x) + (sz)-1) & ~((sz)-1))

    if !isunboxed
        elsz = sizeof(Ptr{Cvoid})
        al = elsz;
    else
        elsz = sizeof(typ)
        al = 1 # check
        elsz = LLT_ALIGN(elsz, al)
    end

    tot = prod
    tot = LLVM.mul!(b, tot, LLVM.ConstantInt(LLVM.llvmtype(tot), elsz, false))

    if elsz == 1 && !isunion
        # extra byte for all julia allocated byte arrays
        tot = LLVM.add!(b, tot, LLVM.ConstantInt(LLVM.llvmtype(tot), 1, false))
    end
    if (isunion)
        # an extra byte for each isbits union array element, stored after a->maxsize
        tot = LLVM.add!(b, tot, prod)
    end

    i8 = LLVM.IntType(8; ctx)
    toset = get_array_data(b, anti)

    mcall = LLVM.memset!(b, toset, LLVM.ConstantInt(i8, 0, false), tot, al)
    
    ref::LLVM.API.LLVMValueRef = Base.unsafe_convert(LLVM.API.LLVMValueRef, anti)
    return ref
end

function get_array_struct(ctx)
# JL_EXTENSION typedef struct {
#     JL_DATA_TYPE
#     void *data;
# #ifdef STORE_ARRAY_LEN (just true new newer versions)
# 	size_t length;
# #endif
#     jl_array_flags_t flags;
#     uint16_t elsize;  // element size including alignment (dim 1 memory stride)
#     uint32_t offset;  // for 1-d only. does not need to get big.
#     size_t nrows;
#     union {
#         // 1d
#         size_t maxsize;
#         // Nd
#         size_t ncols;
#     };
#     // other dim sizes go here for ndims > 2
# 
#     // followed by alignment padding and inline data, or owner pointer
# } jl_array_t;
    
    i8 = LLVM.IntType(8; ctx)
    ptrty = LLVM.PointerType(i8, 13)
    sizeT = LLVM.IntType(8*sizeof(Csize_t); ctx)
    arrayFlags = LLVM.IntType(16; ctx)
    elsz = LLVM.IntType(16; ctx)
    off = LLVM.IntType(32; ctx)
    nrows = LLVM.IntType(8*sizeof(Csize_t); ctx)
    
    return LLVM.StructType([ptrty, sizeT, arrayFlags, elsz, off, nrows]; packed=true, ctx)
end

function get_array_data(B, array)
    ctx = LLVM.context(array)
    i8 = LLVM.IntType(8; ctx)
    ptrty = LLVM.PointerType(i8, 13)
    array = LLVM.pointercast!(B, array, LLVM.PointerType(ptrty, LLVM.addrspace(LLVM.llvmtype(array))))
    return LLVM.load!(B, array)
end

function get_array_elsz(B, array)
    ctx = LLVM.context(array)
    ST = get_array_struct(ctx)
    array = LLVM.pointercast!(B, array, LLVM.PointerType(ST, LLVM.addrspace(LLVM.llvmtype(array))))
    v = inbounds_gep!(B, array, LLVM.Value[LLVM.ConstantInt(Int32(0); ctx), LLVM.ConstantInt(Int32(3); ctx)])
    return LLVM.load!(B, v)
end

function get_array_len(B, array)
    ctx = LLVM.context(array)
    ST = get_array_struct(ctx)
    array = LLVM.pointercast!(B, array, LLVM.PointerType(ST, LLVM.addrspace(LLVM.llvmtype(array))))
    v = inbounds_gep!(B, array, LLVM.Value[LLVM.ConstantInt(Int32(0); ctx), LLVM.ConstantInt(Int32(1); ctx)])
    return LLVM.load!(B, v)
end

function get_array_nrows(B, array)
    ctx = LLVM.context(array)
    ST = get_array_struct(ctx)
    array = LLVM.pointercast!(B, array, LLVM.PointerType(ST, LLVM.addrspace(LLVM.llvmtype(array))))
    v = inbounds_gep!(B, array, LLVM.Value[LLVM.ConstantInt(Int32(0); ctx), LLVM.ConstantInt(Int32(5); ctx)])
    return LLVM.load!(B, v)
end

function null_free_handler(B::LLVM.API.LLVMBuilderRef, ToFree::LLVM.API.LLVMValueRef, Fn::LLVM.API.LLVMValueRef)::LLVM.API.LLVMValueRef
    return C_NULL
end

dedupargs() = ()
dedupargs(a, da, args...) = (a, dedupargs(args...)...)

# Force sret
struct Return2
    ret1::Any
    ret2::Any
end

struct Return3
    ret1::Any
    ret2::Any
    ret3::Any
end
AnyArray(Length::Int64) = NamedTuple{ntuple(i->Symbol(i), Val(Length)),NTuple{Length,Any}}

function permit_inlining!(f::LLVM.Function)
    for bb in blocks(f), inst in instructions(bb)
        # remove illegal invariant.load and jtbaa_const invariants
        if isa(inst, LLVM.LoadInst)
            md = metadata(inst)
            if haskey(md, LLVM.MD_tbaa)
                modified = LLVM.Metadata(ccall((:EnzymeMakeNonConstTBAA, API.libEnzyme), LLVM.API.LLVMMetadataRef, (LLVM.API.LLVMMetadataRef,), md[LLVM.MD_tbaa]))
                setindex!(md, modified, LLVM.MD_tbaa)
            end
            if haskey(md, LLVM.MD_invariant_load)
                delete!(md, LLVM.MD_invariant_load)
            end
        end
    end
end

function runtime_newtask_fwd(fn::Any, dfn::Any, post::Any, ssize::Int, width)
    tt′ = Tuple{}
    args = ()
    tt = Tuple{map(x->eltype(Core.Typeof(x)), args)...}
    rt = Core.Compiler.return_type(fn, tt)
    forward = thunk(fn, dfn, Const, tt′, Val(API.DEM_ForwardMode), width)

    function fclosure()
        res = forward()
        if length(res) > 1
            return res[1]
        else
            return nothing
        end
    end

    return ccall(:jl_new_task, Ref{Task}, (Any, Any, Int), fclosure, post, ssize)
end

function runtime_newtask_augfwd(fn::Any, dfn::Any, post::Any, ssize::Int, ::Val{width}) where width

    tt′ = Tuple{}
    args = ()
    tt = Tuple{map(x->eltype(Core.Typeof(x)), args)...}
    rt = Core.Compiler.return_type(fn, tt)
    forward, adjoint = thunk(fn, dfn, Const, tt′, Val(API.DEM_ReverseModePrimal), Val(width))

    taperef = Ref{Any}()

    function fclosure()
        res = forward()
        taperef[] = res[1]
        if length(res) > 1
            return res[2]
        else
            return nothing
        end
    end

    ftask = ccall(:jl_new_task, Ref{Task}, (Any, Any, Int), fclosure, post, ssize)

    function rclosure()
        adjoint(taperef[])
        return 0
    end
    
    rtask = ccall(:jl_new_task, Ref{Task}, (Any, Any, Int), rclosure, post, ssize)
    
    return Return2(ftask, rtask)
end

struct Tape{T}
    thunk::AdjointThunk
    internal_tape::T
    shadow_return::Any
    resT::DataType
end

@inline function wrap_annotated_args(forwardMode, start, arg_ptr, shadow_ptr::NamedTuple{A,B}, activity_ptr::Ptr{UInt8}, ::Val{width}) where {width,A,B}
    # Note: We shall not unsafe_wrap any of the Ptr{Any}, since these are stack allocations
    #       As an example, if the Array created by unsafe_wrap get's moved to the remset it
    #       will constitute a leak of the stack allocation, and GC will find delicous garbage.
    __activity = Base.unsafe_wrap(Array, activity_ptr, length(arg_ptr))
    args = Any[]
    
    for i in start:length(arg_ptr)
        p = arg_ptr[i]
        T = Core.Typeof(p)
        if __activity[i] != 0 && !(GPUCompiler.isghosttype(T) || Core.Compiler.isconstType(T))
            if !forwardMode && (T <: AbstractFloat || T <: Complex{<:AbstractFloat})
                push!(args, Active(p))
            else
                if width == 1
                    s = shadow_ptr[(i-1)*width + 1]
                    push!(args, Duplicated(p, s))
                else
                    push!(args, BatchDuplicated(p, ntuple(w->shadow_ptr[(i-1)*width+w], width)))
                end
            end
        else
            push!(args, Const(p))
        end
    end
    return args
end
@inline function wrap_annotated_args(forwardMode, start, arg_ptr, shadow_ptr::Ptr{Any}, activity_ptr::Ptr{UInt8}, ::Val{width}) where {width,AddrSpace}
    # Note: We shall not unsafe_wrap any of the Ptr{Any}, since these are stack allocations
    #       As an example, if the Array created by unsafe_wrap get's moved to the remset it
    #       will constitute a leak of the stack allocation, and GC will find delicous garbage.
    __activity = Base.unsafe_wrap(Array, activity_ptr, length(arg_ptr))
    args = Any[]
    
    for i in start:length(arg_ptr)
        p = arg_ptr[i]
        T = Core.Typeof(p)
        if __activity[i] != 0 && !(GPUCompiler.isghosttype(T) || Core.Compiler.isconstType(T))
            if !forwardMode && (T <: AbstractFloat || T <: Complex{<:AbstractFloat})
                push!(args, Active(p))
            else
                if width == 1
                    s = unsafe_load(shadow_ptr, (i-1)*width + 1)
                    push!(args, Duplicated(p, s))
                else
                    push!(args, BatchDuplicated(p, ntuple(w->shadow_ptr[(i-1)*width+w], width)))
                end
            end
        else
            push!(args, Const(p))
        end
    end
    return args
end

@inline function common_interface_augfwd(annotation, forward, adjoint, args, width::Val{Width}, RT::Val{ReturnType}) where {Width,ReturnType}
    res = forward(args...)

    internal_tape = res[1]
    
    if length(res) == 1
        resT = Nothing
        shadow_return = nothing
        tape = Tape(adjoint, internal_tape, shadow_return, resT)
        if Width == 1
            return ReturnType((nothing, nothing, tape))
        else
            return ReturnType((nothing, ntuple(i->nothing, width)..., tape))
        end
    end
    if annotation <: Const
        origRet = res[2]
        resT = typeof(origRet)
        shadow_return = nothing
        tape = Tape(adjoint, internal_tape, shadow_return, resT)
        if Width == 1
            return ReturnType((origRet, origRet, tape))
        else
            return ReturnType((origRet, ntuple(i->origRet, width)..., tape))
        end
    end
    if annotation <: Active
        origRet = res[2]
        resT = typeof(origRet)
        if Width == 1
            shadow_return = Ref(zero(resT))
        else
            shadow_return = ntuple(i->Ref(zero(resT)), width)
        end
        tape = Tape(adjoint, internal_tape, shadow_return, resT)
        if Width == 1
            return ReturnType((origRet, shadow_return, tape))
        else
            return ReturnType((origRet, shadow_return..., tape))
        end
    end
    
    @assert annotation <: Duplicated || annotation <: DuplicatedNoNeed || annotation <: BatchDuplicated || annotation <: BatchDuplicatedNoNeed
    
    origRet = res[2]
    resT = typeof(origRet)
    shadow_return = nothing
    tape = Tape(adjoint, internal_tape, shadow_return, resT)
    if Width == 1
        return ReturnType((origRet, res[3], tape))
    else
        return ReturnType((origRet, ntuple(i->res[3], width)..., tape))
    end
end

function runtime_generic_fwd(fn::Any, arg_ptr, shadow_ptr, activity_ptr::Ptr{UInt8}, width::Val{Width}, ::Val{ReturnType}) where {Width,ReturnType}
    args = wrap_annotated_args(#=forwardMode=#true, #=start=#1, arg_ptr, shadow_ptr, activity_ptr, width)

    # TODO: Annotation of return value
    tt = Tuple{map(x->eltype(Core.Typeof(x)), args)...}
    rt = Core.Compiler.return_type(fn, tt)
    annotation = guess_activity(rt, API.DEM_ForwardMode)
    if annotation <: DuplicatedNoNeed
        annotation = Duplicated{rt}
    end
    if Width != 1
        if annotation <: Duplicated
            annotation = BatchDuplicated{rt, Width}
        end
    end

    tt′ = Tuple{map(Core.Typeof, args)...}
    forward = thunk(fn, #=dfn=#nothing, annotation, tt′, Val(API.DEM_ForwardMode), width, #=ModifiedBetween=#Val(false), #=returnPrimal=#Val(true))

    res = forward(args...)

    if length(res) == 0
        return ReturnType(ntuple(i->nothing, Val(Width+1)))
    end
    if annotation <: Const
        return ReturnType(ntuple(i->res[1], Val(Width+1)))
    end

    if Width == 1
       return ReturnType((res[1], res[2]))
    else
        return ReturnType((res[1], res[2]...))
    end
end

@inline function common_interface_rev(args, shadow_ptr, tape::Any, width::Val{Width}) where Width
    actives = []
    for (i, p) in enumerate(args)
        if typeof(p) <: Active
            push!(actives, i)
        end
    end

    tape = tape::Tape

    if tape.shadow_return !== nothing
        if Width == 1
            val = tape.shadow_return[]
        else
            val = ntuple(i->tape.shadow_return[i][], width)
        end
        push!(args, val)
    end

    push!(args, tape.internal_tape)

    tup = tape.thunk(args...)
    
    for (d, i) in zip(tup, actives)
        # While `RefValue{T}` and boxed T for immutable are bitwise compatible
        # they are not idempotent on the Julia level. We could "force" `a` to be
        # a boxed T, but would lose the mutable memory semantics that Enzyme requires
        for w in 1:Width
            dd = if Width == 1
                d
            else
                d[w]
            end
            a = unsafe_load(shadow_ptr, (i-1)*Width+w)
            if a isa Base.RefValue
                @assert eltype(a) == typeof(dd)
                a[] += dd
            else
                ref = unsafe_load(reinterpret(Ptr{Ptr{typeof(a)}}, shadow_ptr), (i-1)*Width+w)
                unsafe_store!(ref, dd+a)
            end
        end
    end
    return nothing
end

function runtime_generic_augfwd(fn::Any, arg_ptr, shadow_ptr, activity_ptr::Ptr{UInt8}, width::Val{Width}, RT::Val{ReturnType}) where {Width,ReturnType}
    args = wrap_annotated_args(#=forwardMode=#false, #=start=#1, arg_ptr, shadow_ptr, activity_ptr, width)

    # TODO: Annotation of return value
    tt = Tuple{map(x->eltype(Core.Typeof(x)), args)...}
    rt = Core.Compiler.return_type(fn, tt)
    annotation = guess_activity(rt)
    
    tt′ = Tuple{map(Core.Typeof, args)...}
    forward, adjoint = thunk(fn, #=dfn=#nothing, annotation, tt′, Val(API.DEM_ReverseModePrimal), width,
                                 #=ModifiedBetween=#Val(true), #=returnPrimal=#Val(true))
    return common_interface_augfwd(annotation, forward, adjoint, args, width, RT)
end

function runtime_generic_rev(fn::Any, arg_ptr, shadow_ptr, activity_ptr::Ptr{UInt8}, tape::Any, width::Val{Width}, ::Val{ReturnType}) where {Width,ReturnType}
    args = wrap_annotated_args(#=forwardMode=#false, #=start=#1, arg_ptr, shadow_ptr, activity_ptr, width)
    return common_interface_rev(args, shadow_ptr, tape, width)
end

function runtime_invoke_fwd(mi::Any, arg_ptr, shadow_ptr, activity_ptr::Ptr{UInt8}, width::Val{Width}, ::Val{ReturnType}) where {Width,ReturnType}
    args = wrap_annotated_args(#=forwardMode=#true, #=start=#2, arg_ptr, shadow_ptr, activity_ptr, width)
    
    fn = arg_ptr[1]

    specTypes = mi.specTypes.parameters
    F = specTypes[1]
    @assert F == typeof(fn)
    @assert in(mi.def, methods(fn))
    
    tt = Tuple{specTypes[2:end]...}
    rt = Core.Compiler.return_type(fn, tt)
    annotation = guess_activity(rt, API.DEM_ForwardMode)
    if annotation <: DuplicatedNoNeed
        annotation = Duplicated{rt}
    end
    if Width != 1
        if annotation <: Duplicated
            annotation = BatchDuplicated{rt, Width}
        end
    end

    tt′ = Tuple{map(Core.Typeof, args)...}

    forward = thunk(fn, #=dfn=#nothing, annotation, tt′, Val(API.DEM_ForwardMode), width,
                        #=ModifiedBetween=#Val(false), #=returnPrimal=#Val(true))

    res = forward(args...)
    
    if length(res) == 0
        return ReturnType(ntuple(i->nothing, Val(Width+1)))
    end
    if annotation <: Const
        return ReturnType(ntuple(i->res[1], Val(Width+1)))
    end

    if Width == 1
       return ReturnType((res[1], res[2]))
    else
        return ReturnType((res[1], res[2]...))
    end
end

function runtime_invoke_augfwd(mi::Any, arg_ptr, shadow_ptr, activity_ptr::Ptr{UInt8}, width::Val{Width}, RT::Val{ReturnType}) where {Width,ReturnType}
    args = wrap_annotated_args(#=forwardMode=#false, #=start=#2, arg_ptr, shadow_ptr, activity_ptr, width)

    fn = arg_ptr[1]
    
    # TODO actually use the mi rather than fn
    @assert in(mi.def, methods(fn))

    # TODO: Annotation of return value
    tt = Tuple{map(x->eltype(Core.Typeof(x)), args)...}
    rt = Core.Compiler.return_type(fn, tt)
    annotation = guess_activity(rt)

    tt′ = Tuple{map(Core.Typeof, args)...}
    forward, adjoint = thunk(fn, #=dfn=#nothing, annotation, tt′, Val(API.DEM_ReverseModePrimal), width,
                                 #=ModifiedBetween=#Val(true), #=returnPrimal=#Val(true))
    return common_interface_augfwd(annotation, forward, adjoint, args, width, RT)
end

function runtime_invoke_rev(mi::Any, arg_ptr, shadow_ptr, activity_ptr::Ptr{UInt8}, tape::Any, width::Val{Width}, ::Val{ReturnType}) where {Width,ReturnType}
    args = wrap_annotated_args(#=forwardMode=#false, #=start=#2, arg_ptr, shadow_ptr, activity_ptr, width)
    return common_interface_rev(args, shadow_ptr, tape, width)
end

function runtime_apply_latest_fwd(fn::Any, arg_ptr, shadow_ptr, activity_ptr::Ptr{UInt8}, width::Val{Width}, ::Val{ReturnType}) where {Width,ReturnType}
    args = wrap_annotated_args(#=forwardMode=#true, #=start=#1, arg_ptr, shadow_ptr, activity_ptr, width)

    tt = Tuple{map(x->eltype(Core.Typeof(x)), args)...}
    rt = Core.Compiler.return_type(fn, tt)
    annotation = guess_activity(rt, API.DEM_ForwardMode)
    if annotation <: DuplicatedNoNeed
        annotation = Duplicated{rt}
    end
    if Width != 1
        if annotation <: Duplicated
            annotation = BatchDuplicated{rt, Width}
        end
    end

    tt′ = Tuple{map(Core.Typeof, args)...}
    forward = thunk(fn, #=dfn=#nothing, annotation, tt′, Val(API.DEM_ForwardMode), width,
                        #=ModifiedBetween=#Val(false), #=returnPrimal=#Val(true))

    res = forward(args...)

    if length(res) == 0
        return ReturnType(ntuple(i->nothing, Val(Width+1)))
    end
    if annotation <: Const
        return ReturnType(ntuple(i->res[1], Val(Width+1)))
    end

    if Width == 1
       return ReturnType((res[1], res[2]))
    else
        return ReturnType((res[1], res[2]...))
    end
end

function runtime_apply_latest_augfwd(fn::Any, arg_ptr, shadow_ptr, activity_ptr::Ptr{UInt8}, width::Val{Width}, RT::Val{ReturnType}) where {Width,ReturnType}
    args = wrap_annotated_args(#=forwardMode=#false, #=start=#1, arg_ptr, shadow_ptr, activity_ptr, width)

    tt = Tuple{map(x->eltype(Core.Typeof(x)), args)...}
    rt = Core.Compiler.return_type(fn, tt)
    annotation = guess_activity(rt)
    if Width != 1
        if annotation <: Duplicated
            annotation = BatchDuplicated{rt, Width}
        end
    end

    tt′ = Tuple{map(Core.Typeof, args)...}
    forward, adjoint = thunk(fn, #=dfn=#nothing, annotation, tt′, Val(API.DEM_ReverseModePrimal), width,
                                 #=ModifiedBetween=#Val(true), #=returnPrimal=#Val(true))

    return common_interface_augfwd(annotation, forward, adjoint, args, width, RT)
end

function runtime_apply_latest_rev(fn::Any, arg_ptr, shadow_ptr, activity_ptr::Ptr{UInt8}, tape::Any, width::Val{Width}, ::Val{ReturnType}) where {Width,ReturnType}
    args = wrap_annotated_args(#=forwardMode=#false, #=start=#1, arg_ptr, shadow_ptr, activity_ptr, width)
    return common_interface_rev(args, shadow_ptr, tape, width)
end

function emit_gc_preserve_begin(B::LLVM.Builder, args=LLVM.Value[])
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    ctx = context(mod)

    func = get_function!(mod, "llvm.julia.gc_preserve_begin", LLVM.FunctionType(LLVM.TokenType(ctx), vararg=true))

    token = call!(B, func, args)
    return token
end

function emit_gc_preserve_end(B::LLVM.Builder, token)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    ctx = context(mod)

    func = get_function!(mod, "llvm.julia.gc_preserve_end", LLVM.FunctionType(LLVM.VoidType(ctx), [LLVM.TokenType(ctx)]))

    call!(B, func, [token])
    return
end

function generic_setup(orig, func, ReturnType, gutils, start, ctx::LLVM.Context, B::LLVM.Builder,  lookup; sret=nothing, tape=nothing)
    width = API.EnzymeGradientUtilsGetWidth(gutils)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    ops = collect(operands(orig))[(start+1):end-1]
    
    if tape === nothing
        llvmf = nested_codegen!(mod, func, Tuple{Any, AnyArray(length(ops)), AnyArray(Int64(width)*length(ops)), Ptr{UInt8}, Val{Int64(width)}, Val{ReturnType}})
    else
        llvmf = nested_codegen!(mod, func, Tuple{Any, AnyArray(length(ops)), Ptr{Any}, Ptr{UInt8}, Any, Val{Int64(width)}, Val{ReturnType}})
    end

    T_int8 = LLVM.Int8Type(ctx)
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)


    num = convert(UInt32, length(ops))

    EB = LLVM.Builder(ctx)
    position!(EB, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))

    # TODO: Optimization by emitting liverange
    primal = LLVM.alloca!(EB, LLVM.ArrayType(T_prjlvalue, num))
    shadow = LLVM.alloca!(EB, LLVM.ArrayType(T_prjlvalue, num*width))
    activity = LLVM.alloca!(EB, LLVM.ArrayType(T_int8, num))

    jl_fn = API.EnzymeGradientUtilsNewFromOriginal(gutils, operands(orig)[start])
    if lookup
        jl_fn = API.EnzymeGradientUtilsLookup(gutils, jl_fn, B)
    end

    to_preserve = LLVM.Value[]

    if length(ops) == 0
        vals = LLVM.Value[LLVM.Value(jl_fn), activity]
        if tape !== nothing
            push!(vals, LLVM.ConstantInt(Csize_t(0); ctx))
        end
    else
        for (i, op) in enumerate(ops)
            idx = LLVM.Value[LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(i-1; ctx)]
            val = LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, op))
            if lookup
                val = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, val, B))
            end
            
            LLVM.store!(B, val, LLVM.inbounds_gep!(B, primal, idx))

            active = API.EnzymeGradientUtilsIsConstantValue(gutils, op) == 0
            activeC = LLVM.ConstantInt(T_int8, active)
            LLVM.store!(B, activeC, LLVM.inbounds_gep!(B, activity, idx))

            inverted = nothing

            if active
                inverted = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, op, B))
                if lookup
                    inverted = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, inverted, B))
                end
            end

            for w in 1:width
                
                ev = LLVM.null(llvmtype(op))
                if inverted !== nothing
                    if width == 1
                        ev = inverted
                    else
                        ev = extract_value!(B, inverted, w-1)
                    end
                end

                idx = LLVM.Value[LLVM.ConstantInt(0; ctx), LLVM.ConstantInt((i-1)*Int64(width) + w-1; ctx)]
                LLVM.store!(B, ev,
                            LLVM.inbounds_gep!(B, shadow, idx))
            end
        end

        primal = addrspacecast!(B, primal, LLVM.PointerType(eltype(llvmtype(primal)), 11))
        if tape === nothing
            shadow = addrspacecast!(B, shadow, LLVM.PointerType(eltype(llvmtype(shadow)), 11))
        else
            # TODO ROOT
        end
        
        vals = LLVM.Value[LLVM.Value(jl_fn), primal, shadow, activity]
    end

    if tape !== nothing
        push!(vals, LLVM.Value(tape))
    end

    if sret !== nothing && any(map(k->kind(k)==kind(EnumAttribute("sret"; ctx)), collect(parameter_attributes(llvmf, 1))))
        pushfirst!(vals, sret)
    end

    params = parameters(eltype(LLVM.llvmtype(llvmf)))
    for (i, val) in enumerate(vals)
        sourceT = LLVM.llvmtype(val)
        targetT = params[i]
        if sourceT !== targetT
            vals[i] = if sourceT isa LLVM.PointerType && targetT isa LLVM.IntegerType
                LLVM.ptrtoint!(B, val, targetT)
            elseif sourceT isa LLVM.IntegerType && targetT isa LLVM.PointerType
                LLVM.inttoptr!(B, val, targetT)
            else
                LLVM.bitcast!(B, val, targetT)
            end
        end
    end

    if length(to_preserve) != 0
        token = emit_gc_preserve_begin(B, to_preserve)
    end

    cal = LLVM.call!(B, llvmf, vals)
    if sret !== nothing
        attr = if LLVM.version().major >= 12
            TypeAttribute("sret", eltype(llvmtype(sret)); ctx)
        else
            EnumAttribute("sret"; ctx)
        end
        LLVM.API.LLVMAddCallSiteAttribute(cal, LLVM.API.LLVMAttributeIndex(1), attr)
    end
    API.EnzymeGradientUtilsSetDebugLocFromOriginal(gutils, cal, orig)
    
    if length(to_preserve) != 0
        token = emit_gc_preserve_end(B, to_preserve)
    end
    
    if sret !== nothing && !any(map(k->kind(k)==kind(EnumAttribute("sret"; ctx)), collect(parameter_attributes(llvmf, 1))))
        emit_error(B, "Illegal generic_setup")
    end

    return cal
end

function allocate_sret!(B::LLVM.Builder, N, ctx)
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)
    al = LLVM.alloca!(B, LLVM.ArrayType(T_prjlvalue, N))
    for i in 1:N
        LLVM.store!(B, LLVM.null(T_prjlvalue),
                    LLVM.inbounds_gep!(B, al, LLVM.Value[LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(i-1; ctx)]))
    end
    return al
end

function allocate_sret!(gutils::API.EnzymeGradientUtilsRef, N, ctx)
    sret = LLVM.Builder(ctx) do B
        position!(B, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))
        allocate_sret!(B, N, ctx)
    end
end

function common_generic_fwd(offset, B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})
    orig = LLVM.Instruction(OrigCI)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing
    ctx = LLVM.context(orig)

    if API.EnzymeGradientUtilsIsConstantValue(gutils, orig) == 0 || API.EnzymeGradientUtilsIsConstantInstruction(gutils, orig) == 0 

        B = LLVM.Builder(B)
    
        width = API.EnzymeGradientUtilsGetWidth(gutils)
        
        sret = allocate_sret!(gutils, 1+width, ctx)

        generic_setup(orig, runtime_generic_fwd, AnyArray(1+Int64(width)), gutils, #=start=#offset, ctx, B, false; sret)

        if shadowR != C_NULL
            if width == 1
                gep = LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(1; ctx)])
                shadow = LLVM.load!(B, gep)
            else
                ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig)))
                shadow = LLVM.UndefValue(ST)
                for i in 1:width
                    gep = LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(i; ctx)])
                    ld = LLVM.load!(B, gep)
                    shadow = insert_value!(B, shadow, ld, i-1)
                end
            end
            unsafe_store!(shadowR, shadow.ref)
        end

        if normalR != C_NULL
            normal = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(0; ctx)]))
            unsafe_store!(normalR, normal.ref)
        end
    end

end

function generic_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    conv = LLVM.API.LLVMGetInstructionCallConv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
    @assert conv == 37
    common_generic_fwd(1, B, OrigCI, gutils, normalR, shadowR)
    return nothing
end

function common_generic_augfwd(offset, B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing
    ctx = LLVM.context(orig)

    B = LLVM.Builder(B)
    width = API.EnzymeGradientUtilsGetWidth(gutils)
    sret = allocate_sret!(gutils, 2+width, ctx)
    generic_setup(orig, runtime_generic_augfwd, AnyArray(2+Int64(width)), gutils, #=start=#offset, ctx, B, false; sret)

        if shadowR != C_NULL
            if width == 1
                gep = LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(1; ctx)])
                shadow = LLVM.load!(B, gep)
            else
                ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig)))
                shadow = LLVM.UndefValue(ST)
                for i in 1:width
                    gep = LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(i; ctx)])
                    ld = LLVM.load!(B, gep)
                    shadow = insert_value!(B, shadow, ld, i-1)
                end
            end
            unsafe_store!(shadowR, shadow.ref)
        end

    if normalR != C_NULL
        normal = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(0; ctx)]))
        unsafe_store!(normalR, normal.ref)
    end

    tape = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(1+width; ctx)]))
    unsafe_store!(tapeR, tape.ref)
    return nothing
end

function generic_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)

    conv = LLVM.API.LLVMGetInstructionCallConv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
    
    @assert conv == 37

    common_generic_augfwd(1, B, OrigCI, gutils, normalR, shadowR, tapeR)

    return nothing
end

function common_generic_rev(offset, B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)

    B = LLVM.Builder(B)
    ctx = LLVM.context(orig)

    @assert tape !== C_NULL
    generic_setup(orig, runtime_generic_rev, Nothing, gutils, #=start=#offset, ctx, B, true; tape)

    return nothing
end

function generic_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)

    conv = LLVM.API.LLVMGetInstructionCallConv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
    
    @assert conv == 37

    common_generic_rev(1, B, OrigCI, gutils, tape)
    return nothing
end

function common_apply_latest_fwd(offset, B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})
    orig = LLVM.Instruction(OrigCI)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    ctx = LLVM.context(orig)
    B = LLVM.Builder(B)

    width = API.EnzymeGradientUtilsGetWidth(gutils)
    sret = allocate_sret!(gutils, 1+width, ctx)
    generic_setup(orig, runtime_apply_latest_fwd, AnyArray(1+Int64(width)), gutils, #=start=#offset+1, ctx, B, false; sret)

    if shadowR != C_NULL
        if width == 1
            gep = LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(1; ctx)])
            shadow = LLVM.load!(B, gep)
        else
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig)))
            shadow = LLVM.UndefValue(ST)
            for i in 1:width
                gep = LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(i; ctx)])
                ld = LLVM.load!(B, gep)
                shadow = insert_value!(B, shadow, ld, i-1)
            end
        end
        unsafe_store!(shadowR, shadow.ref)
    end

    if normalR != C_NULL
        normal = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(0; ctx)]))
        unsafe_store!(normalR, normal.ref)
    end

    return nothing
end

function common_apply_latest_augfwd(offset, B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    ctx = LLVM.context(orig)

    B = LLVM.Builder(B)

    width = API.EnzymeGradientUtilsGetWidth(gutils)
    sret = allocate_sret!(gutils, 2+width, ctx)
    generic_setup(orig, runtime_apply_latest_augfwd, AnyArray(2+Int64(width)), gutils, #=start=#offset+1, ctx, B, false; sret)

    if shadowR != C_NULL
        if width == 1
            gep = LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(1; ctx)])
            shadow = LLVM.load!(B, gep)
        else
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig)))
            shadow = LLVM.UndefValue(ST)
            for i in 1:width
                gep = LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(i; ctx)])
                ld = LLVM.load!(B, gep)
                shadow = insert_value!(B, shadow, ld, i-1)
            end
        end
        unsafe_store!(shadowR, shadow.ref)
    end

    if normalR != C_NULL
        normal = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(0; ctx)]))
        unsafe_store!(normalR, normal.ref)
    end

    tape = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(1+width; ctx)]))
    unsafe_store!(tapeR, tape.ref)
    return nothing
end

function common_apply_latest_rev(offset, B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    ctx = LLVM.context(orig)

    B = LLVM.Builder(B)

    generic_setup(orig, runtime_apply_latest_rev, Nothing, gutils, #=start=#offset+1, ctx, B, true; tape)

    return nothing
end

function apply_latest_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)

    conv = LLVM.API.LLVMGetInstructionCallConv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
    @assert conv == 37
    
    common_apply_latest_fwd(1, B, OrigCI, gutils, normalR, shadowR)
    return nothing
end

function apply_latest_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)

    conv = LLVM.API.LLVMGetInstructionCallConv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
    @assert conv == 37
    
    common_apply_latest_augfwd(1, B, OrigCI, gutils, normalR, shadowR, tapeR)

    return nothing
end

function apply_latest_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)

    conv = LLVM.API.LLVMGetInstructionCallConv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
    @assert conv == 37
    
    common_apply_latest_rev(1, B, OrigCI, gutils, tape)
    return nothing
end

function jlcall_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_apply_generic", "jl_apply_generic"))
            common_generic_fwd(2, B, OrigCI, gutils, normalR, shadowR)
            return nothing
        end
        if in(name, ("ijl_f__apply_latest", "ijl_f__call_latest", "jl_f__apply_latest", "jl_f__call_latest"))
            common_apply_latest_fwd(2, B, OrigCI, gutils, normalR, shadowR)
            return nothing
        end
    end
    
    @assert false "jl_call calling convention not implemented yet", orig
    
    return nothing
end

function jlcall_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
   
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_apply_generic", "jl_apply_generic"))
            common_generic_augfwd(2, B, OrigCI, gutils, normalR, shadowR, tapeR)
            return nothing
        end
        if in(name, ("ijl_f__apply_latest", "ijl_f__call_latest", "jl_f__apply_latest", "jl_f__call_latest"))
            common_apply_latest_augfwd(2, B, OrigCI, gutils, normalR, shadowR, tapeR)
            return nothing
        end
    end

    @assert false "jl_call calling convention not implemented yet", orig
    
    return nothing
end

function jlcall_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_apply_generic", "jl_apply_generic"))
            common_generic_rev(2, B, OrigCI, gutils, tape)
            return nothing
        end
        if in(name, ("ijl_f__apply_latest", "ijl_f__call_latest", "jl_f__apply_latest", "jl_f__call_latest"))
            common_apply_latest_rev(2, B, OrigCI, gutils, tape)
            return nothing
        end
    end
    
    @assert false "jl_call calling convention not implemented yet", orig

    return nothing
end

function jlcall2_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_invoke", "jl_invoke"))
            common_invoke_fwd(2, B, OrigCI, gutils, normalR, shadowR)
            return nothing
        end
    end
    
    @assert false "jl_call calling convention not implemented yet", orig
    
    return nothing
end

function jlcall2_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
   
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_invoke", "jl_invoke"))
            common_invoke_augfwd(2, B, OrigCI, gutils, normalR, shadowR, tapeR)
            return nothing
        end
    end

    @assert false "jl_call calling convention not implemented yet", orig
    
    return nothing
end

function jlcall2_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_invoke", "jl_invoke"))
            common_invoke_rev(2, B, OrigCI, gutils, tape)
            return nothing
        end
    end
    
    @assert false "jl_call calling convention not implemented yet", orig

    return nothing
end

function common_invoke_fwd(offset, B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)

    if API.EnzymeGradientUtilsIsConstantValue(gutils, orig) == 0 || API.EnzymeGradientUtilsIsConstantInstruction(gutils, orig) == 0 
        normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
        shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing
        ctx = LLVM.context(orig)

        B = LLVM.Builder(B)
        width = API.EnzymeGradientUtilsGetWidth(gutils)
        sret = allocate_sret!(gutils, 1+width, ctx)
        generic_setup(orig, runtime_invoke_fwd, AnyArray(1+Int64(width)), gutils, #=start=#offset, ctx, B, false; sret)
        
        if shadowR != C_NULL
            if width == 1
                gep = LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(1; ctx)])
                shadow = LLVM.load!(B, gep)
            else
                ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig)))
                shadow = LLVM.UndefValue(ST)
                for i in 1:width
                    gep = LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(i; ctx)])
                    ld = LLVM.load!(B, gep)
                    shadow = insert_value!(B, shadow, ld, i-1)
                end
            end
            unsafe_store!(shadowR, shadow.ref)
        end

        if normalR != C_NULL
            normal = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(0; ctx)]))
            unsafe_store!(normalR, normal.ref)
        end
    end

    return nothing
end

function common_invoke_augfwd(offset, B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    
    if API.EnzymeGradientUtilsIsConstantValue(gutils, orig) == 0 || API.EnzymeGradientUtilsIsConstantInstruction(gutils, orig) == 0 
        normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
        shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing
        ctx = LLVM.context(orig)

        conv = LLVM.API.LLVMGetInstructionCallConv(orig)

        B = LLVM.Builder(B)
        width = API.EnzymeGradientUtilsGetWidth(gutils)
        sret = allocate_sret!(gutils, 2+width, ctx)
        generic_setup(orig, runtime_invoke_augfwd, AnyArray(2+Int64(width)), gutils, #=start=#offset, ctx, B, false; sret)
        
        if shadowR != C_NULL
            if width == 1
                gep = LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(1; ctx)])
                shadow = LLVM.load!(B, gep)
            else
                ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig)))
                shadow = LLVM.UndefValue(ST)
                for i in 1:width
                    gep = LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(i; ctx)])
                    ld = LLVM.load!(B, gep)
                    shadow = insert_value!(B, shadow, ld, i-1)
                end
            end
            unsafe_store!(shadowR, shadow.ref)
        end

        if normalR != C_NULL
            normal = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(0; ctx)]))
            unsafe_store!(normalR, normal.ref)
        end

        tape = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(1+width; ctx)]))
        unsafe_store!(tapeR, tape.ref)
    end

    return nothing
end

function common_invoke_rev(offset, B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    
    if API.EnzymeGradientUtilsIsConstantValue(gutils, orig) == 0 || API.EnzymeGradientUtilsIsConstantInstruction(gutils, orig) == 0 
        ctx = LLVM.context(orig)

        B = LLVM.Builder(B)

        generic_setup(orig, runtime_invoke_rev, Nothing, gutils, #=start=#offset, ctx, B, true; tape)
    end

    return nothing
end

function invoke_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)

        conv = LLVM.API.LLVMGetInstructionCallConv(orig)
        # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
        if conv != 38
            GPUCompiler.@safe_error "Illegal invoke convention ", orig, conv, API.EnzymeGradientUtilsIsConstantValue(gutils, orig),  API.EnzymeGradientUtilsIsConstantInstruction(gutils, orig)
        end
        @assert conv == 38
    common_invoke_fwd(1, B, OrigCI, gutils, normalR, shadowR)
    return nothing
end

function invoke_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)

        conv = LLVM.API.LLVMGetInstructionCallConv(orig)
        # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
        if conv != 38
            GPUCompiler.@safe_error "Illegal invoke convention ", orig, conv, API.EnzymeGradientUtilsIsConstantValue(gutils, orig),  API.EnzymeGradientUtilsIsConstantInstruction(gutils, orig)
        end
        @assert conv == 38
    common_invoke_augfwd(1, B, OrigCI, gutils, normalR, shadowR, tapeR)
    return nothing
end

function invoke_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)

        conv = LLVM.API.LLVMGetInstructionCallConv(orig)
        # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
        if conv != 38
            GPUCompiler.@safe_error "Illegal invoke convention ", orig, conv, API.EnzymeGradientUtilsIsConstantValue(gutils, orig),  API.EnzymeGradientUtilsIsConstantInstruction(gutils, orig)
        end
        @assert conv == 38
    common_invoke_rev(1, B, OrigCI, gutils, tape)
    return nothing
end

function emit_error(B::LLVM.Builder, string)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    ctx = context(mod)

    # 1. get the error function
    funcT = LLVM.FunctionType(LLVM.VoidType(ctx), LLVMType[LLVM.PointerType(LLVM.Int8Type(ctx))])
    func = get_function!(mod, "jl_error", funcT, [EnumAttribute("noreturn"; ctx)])

    # 2. Call error function and insert unreachable
    call!(B, func, LLVM.Value[globalstring_ptr!(B, string)])

    # FIXME(@wsmoses): Allow for emission of new BB in this code path
    # unreachable!(B)

    # 3. Change insertion point so that we don't stumble later
    # after_error = BasicBlock(fn, "after_error"; ctx)
    # position!(B, after_error)
end

function noop_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    return nothing
end

function noop_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    return nothing
end

function duplicate_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    newg = API.EnzymeGradientUtilsNewFromOriginal(gutils, orig)
    
    B = LLVM.Builder(B)

    real_ops = collect(operands(orig))
    ops = [LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, API.EnzymeGradientUtilsNewFromOriginal(gutils, o), B)) for o in real_ops]
    call!(B, ops[end], ops[1:end-1])
    return nothing
end

@static if VERSION < v"1.9-"
else
const ctxToThreadSafe = Dict{LLVM.Context, LLVM.ThreadSafeContext}()
end

function nested_codegen!(mod::LLVM.Module, f, tt)
    # TODO: Put a cache here index on `mod` and f->tt

    ctx = LLVM.context(mod)

@static if VERSION < v"1.9-"
else
    ctx = ctxToThreadSafe[ctx]
end
    funcspec = FunctionSpec(f, tt, #=kernel=# false, #=name=# nothing)

    # 3) Use the MI to create the correct augmented fwd/reverse
    # TODO:
    #  - GPU support
    #  - When OrcV2 only use a MaterializationUnit to avoid mutation of the module here

    target = GPUCompiler.NativeCompilerTarget()
    params = Compiler.PrimalCompilerParams()
    job    = CompilerJob(target, funcspec, params)

    otherMod, meta = GPUCompiler.codegen(:llvm, job; optimize=false, validate=false, ctx)
    entry = name(meta.entry)

    # Apply first stage of optimization's so that this module is at the same stage as `mod`
    optimize!(otherMod, JIT.get_tm())

    # 4) Link the corresponding module
    LLVM.link!(mod, otherMod)


    # 5) Call the function
    return functions(mod)[entry]
end

@inline GetTapeType(::Type{AugmentedForwardThunk{F, RT, TT, Width, DF, ReturnPrimal, TapeType}}) where {F, RT, TT, Width, DF, ReturnPrimal, TapeType} = TapeType
@inline GetTapeType(::Type{AdjointThunk{F, RT, TT, Width, DF, TapeType}}) where {F, RT, TT, Width, DF, TapeType} = TapeType

if VERSION < v"1.8-"
function runtime_pfor_fwd(func, ptr, dfunc, ::Type{ThunkTy})::Cvoid where ThunkTy
    thunk = ThunkTy(func, ptr, dfunc)
    Base.Threads.threading_run(thunk)
    return
end

function runtime_pfor_augfwd(func, ptr, dfunc, ::Type{ThunkTy}, ::Val{AnyJL}) where {ThunkTy, AnyJL}
    TapeType = GetTapeType(ThunkTy)
    thunk = ThunkTy(func, ptr, dfunc)
    tapes = if AnyJL
        Vector{TapeType}(undef, Base.Threads.nthreads())
    else
        Base.unsafe_convert(Ptr{TapeType}, Libc.malloc(sizeof(TapeType)*Base.Threads.nthreads()))
    end

    function fwd()
        tres = thunk()
        tid = Base.Threads.threadid()
        if !AnyJL
            unsafe_store!(tapes, tres[1], tid)
        else
            @inbounds tapes[tid] = tres[1]
        end
    end
    Base.Threads.threading_run(fwd)
    return tapes
end

function runtime_pfor_rev(func, ptr, dfunc, ::Type{ThunkTy}, ::Val{AnyJL}, tapes) where {ThunkTy, AnyJL}
    thunk = ThunkTy(func, ptr, dfunc)
    function rev()
        tid = Base.Threads.threadid()
        tres = if !AnyJL
            unsafe_load(tapes, tid)
        else
            @inbounds tapes[tid]
        end
        thunk(tres)
    end
    Base.Threads.threading_run(rev)
    if !AnyJL
        Libc.free(tapes)
    end 
    return nothing
end

else

function runtime_pfor_fwd(func, ptr, dfunc, ::Type{ThunkTy}, dynamic)::Cvoid where ThunkTy
    thunk = ThunkTy(func, ptr, dfunc)
    function fwd(tid)
        thunk(Const(tid))
    end
    Base.Threads.threading_run(fwd, dynamic)
    return
end

function runtime_pfor_augfwd(func, ptr, dfunc, ::Type{ThunkTy}, ::Val{AnyJL}, dynamic) where {ThunkTy, AnyJL}
    TapeType = GetTapeType(ThunkTy)
    thunk = ThunkTy(func, ptr, dfunc)
    tapes = if AnyJL
        Vector{TapeType}(undef, Base.Threads.nthreads())
    else
        Base.unsafe_convert(Ptr{TapeType}, Libc.malloc(sizeof(TapeType)*Base.Threads.nthreads()))
    end

    function fwd(tid)
        tres = thunk(Const(tid))
        if !AnyJL
            unsafe_store!(tapes, tres[1], tid)
        else
            @inbounds tapes[tid] = tres[1]
        end
    end
    Base.Threads.threading_run(fwd, dynamic)
    return tapes
end

function runtime_pfor_rev(func, ptr, dfunc, ::Type{ThunkTy}, ::Val{AnyJL}, tapes, dynamic) where {ThunkTy, AnyJL}
    thunk = ThunkTy(func, ptr, dfunc)
    function rev(tid)
        tres = if !AnyJL
            unsafe_load(tapes, tid)
        else
            @inbounds tapes[tid]
        end
        thunk(Const(tid), tres)
    end
    Base.Threads.threading_run(rev, dynamic)
    if !AnyJL
        Libc.free(tapes)
    end
    return nothing
end
end

@inline function threadsfor_common(orig, gutils, B, mode)

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    ctx = LLVM.context(orig)

    llvmfn = LLVM.called_value(orig)
    mi = nothing
    fwdmodenm = nothing
    augfwdnm = nothing
    adjointnm = nothing
    TapeType = nothing
    attributes = function_attributes(llvmfn)
    for fattr in collect(attributes)
        if isa(fattr, LLVM.StringAttribute)
            if kind(fattr) == "enzymejl_mi"
                ptr = reinterpret(Ptr{Cvoid}, parse(Int, LLVM.value(fattr)))
                mi = Base.unsafe_pointer_to_objref(ptr)
            end
            if kind(fattr) == "enzymejl_tapetype"
                ptr = reinterpret(Ptr{Cvoid}, parse(Int, LLVM.value(fattr)))
                TapeType = Base.unsafe_pointer_to_objref(ptr)
            end
            if kind(fattr) == "enzymejl_forward"
                fwdmodenm = value(fattr)
            end
            if kind(fattr) == "enzymejl_augforward"
                augfwdnm = value(fattr)
            end
            if kind(fattr) == "enzymejl_adjoint"
                adjointnm = value(fattr)
            end
        end
    end

    funcT = mi.specTypes.parameters[2]

@static if VERSION < v"1.8-"
    e_tt = Tuple{}
else
    e_tt = Tuple{Const{Int}}
end
    eprimal, eadjoint = fspec(funcT, e_tt)

    # TODO: Clean this up and add to `nested_codegen!` asa feature
    width = API.EnzymeGradientUtilsGetWidth(gutils)

    ops = collect(operands(orig))[1:end-1]
    dupClosure = !GPUCompiler.isghosttype(funcT) && !Core.Compiler.isconstType(funcT) && API.EnzymeGradientUtilsIsConstantValue(gutils, ops[1]) == 0

    subfunc = nothing
    if mode == API.DEM_ForwardMode
        if fwdmodenm === nothing
            etarget = Compiler.EnzymeTarget()
            eparams = Compiler.EnzymeCompilerParams(eadjoint, API.DEM_ForwardMode, width, Const{Nothing}, #=runEnzyme=#true, #=shadowfunc=#dupClosure, #=abiwrap=#true, #=modifiedBetween=#false, #=returnPrimal=#false)
            ejob    = Compiler.CompilerJob(etarget, eprimal, eparams)
    
            jctx = ctx
@static if VERSION < v"1.9-"
else
            jctx = ctxToThreadSafe[jctx]
end
            cmod, fwdmodenm, _, _ = _thunk(ejob, jctx)
            LLVM.link!(mod, cmod)

            push!(attributes, StringAttribute("enzymejl_forward", fwdmodenm; ctx))
            push!(function_attributes(functions(mod)[fwdmodenm]), EnumAttribute("alwaysinline"; ctx))
            permit_inlining!(functions(mod)[fwdmodenm])
        end
        dfuncT = dupClosure ? funcT : Nothing
        thunkTy = ForwardModeThunk{funcT, Const{Nothing}, eadjoint.tt, Val{width}, dfuncT, #=returnPrimal=#Val(false)}
        subfunc = functions(mod)[fwdmodenm]

    elseif mode == API.DEM_ReverseModePrimal || mode == API.DEM_ReverseModeGradient
        if augfwdnm === nothing || adjointnm === nothing
            etarget = Compiler.EnzymeTarget()
            eparams = Compiler.EnzymeCompilerParams(eadjoint, API.DEM_ReverseModePrimal, width, Const{Nothing}, #=runEnzyme=#true, #=shadowfunc=#dupClosure, #=abiwrap=#true, #=modifiedBetween=#true, #=returnPrimal=#false)
            ejob    = Compiler.CompilerJob(etarget, eprimal, eparams)
            jctx = ctx
@static if VERSION < v"1.9-"
else
            jctx = ctxToThreadSafe[jctx]
end
            cmod, adjointnm, augfwdnm, _, TapeType = _thunk(ejob, jctx)
            LLVM.link!(mod, cmod)

            push!(attributes, StringAttribute("enzymejl_augforward", augfwdnm; ctx))
            push!(function_attributes(functions(mod)[augfwdnm]), EnumAttribute("alwaysinline"; ctx))
            permit_inlining!(functions(mod)[augfwdnm])

            push!(attributes, StringAttribute("enzymejl_adjoint", adjointnm; ctx))
            push!(function_attributes(functions(mod)[adjointnm]), EnumAttribute("alwaysinline"; ctx))
            permit_inlining!(functions(mod)[adjointnm])
            
            push!(attributes, StringAttribute("enzymejl_tapetype", string(convert(Int, unsafe_to_pointer(TapeType))); ctx))
        end

        if mode == API.DEM_ReverseModePrimal
            dfuncT = dupClosure ? funcT : Nothing
            thunkTy = AugmentedForwardThunk{funcT, Const{Nothing}, eadjoint.tt, Val{width}, dfuncT, #=returnPrimal=#Val(true), TapeType}
            subfunc = functions(mod)[augfwdnm]
       else
            dfuncT = dupClosure ? funcT : Nothing
            thunkTy = AdjointThunk{funcT, Const{Nothing}, eadjoint.tt, Val{width}, dfuncT, TapeType}
            subfunc = functions(mod)[adjointnm]
        end
    else
        @assert "Unknown mode"
    end

    to_preserve = LLVM.Value[]
    vals = LLVM.Value[]

    if !GPUCompiler.isghosttype(funcT) && !Core.Compiler.isconstType(funcT)
        v = LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[1]))
        if mode == API.DEM_ReverseModeGradient
            v = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, v, B))
        end
        push!(vals, v)
        push!(to_preserve, v)
    end

    push!(vals, ptrtoint!(B, subfunc, convert(LLVMType, Ptr{Cvoid}; ctx)))

    if !GPUCompiler.isghosttype(dfuncT) && !Core.Compiler.isconstType(dfuncT)
        v = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, ops[1], B))
        if mode == API.DEM_ReverseModeGradient
            v = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, v, B))
        end
        push!(vals, v)
        push!(to_preserve, v)
    end

    return funcT, dfuncT, vals, thunkTy, to_preserve, TapeType
end

function threadsfor_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid

    orig = LLVM.Instruction(OrigCI)
    ctx = LLVM.context(orig)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing

    B = LLVM.Builder(B)

    funcT, dfuncT, vals, thunkTy, to_preserve, _ = threadsfor_common(orig, gutils, B, API.DEM_ForwardMode)

@static if VERSION < v"1.8-"
    tt = Tuple{funcT, Core.Ptr{Cvoid}, dfuncT, Type{thunkTy}}
else
    tt = Tuple{funcT, Core.Ptr{Cvoid}, dfuncT, Type{thunkTy}, Bool}
end
    entry = nested_codegen!(mod, runtime_pfor_fwd, tt)
    permit_inlining!(entry)
    push!(function_attributes(entry), EnumAttribute("alwaysinline"; ctx))

@static if VERSION < v"1.8-"
else
    push!(vals, LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, operands(orig)[end-1])))
end

    token = emit_gc_preserve_begin(B, to_preserve)

    cal = LLVM.call!(B, entry, vals)
    API.EnzymeGradientUtilsSetDebugLocFromOriginal(gutils, cal, orig)

    emit_gc_preserve_end(B, token)

    # Delete the primal code
    if normal !== nothing
        unsafe_store!(normalR, C_NULL)
    else
        LLVM.API.LLVMInstructionEraseFromParent(LLVM.Instruction(API.EnzymeGradientUtilsNewFromOriginal(gutils, orig)))
    end
end

function threadsfor_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid

    orig = LLVM.Instruction(OrigCI)
    ctx = LLVM.context(orig)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing

    B = LLVM.Builder(B)

    funcT, dfuncT, vals, thunkTy, to_preserve, _ = threadsfor_common(orig, gutils, B, API.DEM_ReverseModePrimal)

@static if VERSION < v"1.8-"
    tt = Tuple{funcT, Core.Ptr{Cvoid}, dfuncT, Type{thunkTy}, Val{any_jltypes(GetTapeType(thunkTy))}}
else
    tt = Tuple{funcT, Core.Ptr{Cvoid}, dfuncT, Type{thunkTy}, Val{any_jltypes(GetTapeType(thunkTy))}, Bool}
end
    entry = nested_codegen!(mod, runtime_pfor_augfwd, tt)
    permit_inlining!(entry)
    push!(function_attributes(entry), EnumAttribute("alwaysinline"; ctx))

@static if VERSION < v"1.8-"
else
    push!(vals, LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, operands(orig)[end-1])))
end

    token = emit_gc_preserve_begin(B, to_preserve)
    tape = LLVM.call!(B, entry, vals)
    API.EnzymeGradientUtilsSetDebugLocFromOriginal(gutils, tape, orig)

    emit_gc_preserve_end(B, token)

    # Delete the primal code
    if normal !== nothing
        unsafe_store!(normalR, C_NULL)
    else
        LLVM.API.LLVMInstructionEraseFromParent(LLVM.Instruction(API.EnzymeGradientUtilsNewFromOriginal(gutils, orig)))
    end

    GPUCompiler.@safe_warn "active variables passed by value to jl_threadsfor are not yet supported"

    unsafe_store!(tapeR, tape.ref)

    return nothing
end

function threadsfor_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid

    orig = LLVM.Instruction(OrigCI)
    ctx = LLVM.context(orig)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    tape = LLVM.Value(tape)

    B = LLVM.Builder(B)

    funcT, dfuncT, vals, thunkTy, to_preserve, TapeType = threadsfor_common(orig, gutils, B, API.DEM_ReverseModeGradient)

    STT = if !any_jltypes(TapeType)
        Ptr{TapeType}
    else
        Vector{TapeType}
    end

@static if VERSION < v"1.8-"
    tt = Tuple{funcT, Core.Ptr{Cvoid}, dfuncT, Type{thunkTy}, Val{any_jltypes(GetTapeType(thunkTy))}, STT }
else
    tt = Tuple{funcT, Core.Ptr{Cvoid}, dfuncT, Type{thunkTy}, Val{any_jltypes(GetTapeType(thunkTy))}, STT, Bool}
end
    entry = nested_codegen!(mod, runtime_pfor_rev, tt)
    permit_inlining!(entry)
    push!(function_attributes(entry), EnumAttribute("alwaysinline"; ctx))

    push!(vals, tape)

@static if VERSION < v"1.8-"
else
    push!(vals, LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, operands(orig)[end-1])))
end

    token = emit_gc_preserve_begin(B, to_preserve)

    cal = LLVM.call!(B, entry, vals)
    API.EnzymeGradientUtilsSetDebugLocFromOriginal(gutils, cal, orig)

    emit_gc_preserve_end(B, token)
    return nothing
end

include("compiler/pmap.jl")

function newtask_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    width = API.EnzymeGradientUtilsGetWidth(gutils)
    fun = nested_codegen!(mod, runtime_newtask_fwd, Tuple{Any, Any, Any, Int, Val{width}})

    B = LLVM.Builder(B)
    
    ops = collect(operands(orig))

    vals = LLVM.Value[ LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[1])),
                       LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, ops[1], B)),
                       LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[2])),
                       LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[3]))]

    to_preserve = LLVM.Value[vals[1], vals[2], vals[3]]
    token = emit_gc_preserve_begin(B, to_preserve)

    ntask = LLVM.call!(B, fun, vals)

    # TODO: GC, ret
    if shadowR != C_NULL
        unsafe_store!(shadowR, ntask.ref)
    end

    if normalR != C_NULL
        unsafe_store!(normalR, ntask.ref)
    end

    emit_gc_preserve_end(B, token)

    return nothing
end

function newtask_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    # fn, dfn = augmentAndGradient(fn)
    # t = jl_new_task(fn)
    # # shadow t
    # dt = jl_new_task(dfn)
    orig = LLVM.Instruction(OrigCI)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    ctx = LLVM.context(orig)

    GPUCompiler.@safe_warn "active variables passed by value to jl_new_task are not yet supported"
    width = API.EnzymeGradientUtilsGetWidth(gutils)
    fun = nested_codegen!(mod, runtime_newtask_augfwd, Tuple{Any, Any, Any, Int, Val{width}})

    B = LLVM.Builder(B)
    sret = allocate_sret!(gutils, 2, ctx)
    
    ops = collect(operands(orig))[1:end-1]

    vals = LLVM.Value[sret,
                       LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[1])),
                       LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, ops[1], B)),
                       LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[2])),
                       LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[3]))]

    to_preserve = LLVM.Value[vals[2], vals[3], vals[4]] # All Any should be preserved
    token = emit_gc_preserve_begin(B, to_preserve)

    cal = LLVM.call!(B, fun, vals)
    
    attr = if LLVM.version().major >= 12
        TypeAttribute("sret", eltype(llvmtype(sret)); ctx)
    else
        EnumAttribute("sret"; ctx)
    end
    LLVM.API.LLVMAddCallSiteAttribute(cal, LLVM.API.LLVMAttributeIndex(1), attr)

    if shadowR != C_NULL
        shadow = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(1; ctx)]))
        unsafe_store!(shadowR, shadow.ref)
    end

    if normalR != C_NULL
        normal = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(0; ctx)]))
        unsafe_store!(normalR, normal.ref)
    end

    emit_gc_preserve_end(B, token)

    return nothing
end

function newtask_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    return nothing
end

function set_task_tid_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    ops = collect(operands(orig))[1:end-1]
    if API.EnzymeGradientUtilsIsConstantValue(gutils, ops[1]) == 0

        inv = API.EnzymeGradientUtilsInvertPointer(gutils, ops[1], B)
        width = API.EnzymeGradientUtilsGetWidth(gutils)
        if width == 1
            nops =LLVM.API.LLVMValueRef[inv,
                                        API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[2])]
            valTys = API.CValueType[API.VT_Shadow, API.VT_Primal]
            cal = API.EnzymeGradientUtilsCallWithInvertedBundles(gutils, LLVM.called_value(orig), nops, length(nops), orig, valTys, length(valTys), B, #=lookup=#false)

            API.EnzymeGradientUtilsSetDebugLocFromOriginal(gutils, cal, orig)
            conv = LLVM.API.LLVMGetInstructionCallConv(orig)
            LLVM.API.LLVMSetInstructionCallConv(cal, conv)
        else
            for idx in 1:width
                nops =LLVM.API.LLVMValueRef[extract_value(B, inv, idx-1),
                                            API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[2])]
                valTys = API.CValueType[API.VT_Shadow, API.VT_Primal]
                cal = API.EnzymeGradientUtilsCallWithInvertedBundles(gutils, LLVM.called_value(orig), nops, length(nops), orig, valTys, length(valTys), B, #=lookup=#false)

                API.EnzymeGradientUtilsSetDebugLocFromOriginal(gutils, cal, orig)
                conv = LLVM.API.LLVMGetInstructionCallConv(orig)
                LLVM.API.LLVMSetInstructionCallConv(cal, conv)
            end
        end

    end
    return nothing
end

function set_task_tid_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    set_task_tid_fwd(B, OrigCI, gutils, normalR, shadowR)
end

function set_task_tid_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    return nothing
end

function enq_work_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return nothing
end

function enq_work_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return nothing
end

function find_match(mod, name)
    for f in functions(mod)
        iter = function_attributes(f)
        elems = Vector{LLVM.API.LLVMAttributeRef}(undef, length(iter))
        LLVM.API.LLVMGetAttributesAtIndex(iter.f, iter.idx, elems)
        for eattr in elems
            at = Attribute(eattr)
            if isa(at, LLVM.StringAttribute)
                if kind(at) == "enzyme_math"
                    if value(at) == name
                        return f
                    end
                end
            end
        end
    end
    return nothing
end

function enq_work_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    # jl_wait(shadow(t)) 
    orig = LLVM.Instruction(OrigCI)
    origops = LLVM.operands(orig)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    waitfn = find_match(mod, "jl_wait")
    @assert waitfn !== nothing
    shadowtask = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, API.EnzymeGradientUtilsInvertPointer(gutils, origops[1], B), B))
    cal = LLVM.call!(LLVM.Builder(B), waitfn, [shadowtask])
    API.EnzymeGradientUtilsSetDebugLocFromOriginal(gutils, cal, orig)
    conv = LLVM.API.LLVMGetInstructionCallConv(orig)
    LLVM.API.LLVMSetInstructionCallConv(cal, conv)
    return nothing
end

function wait_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return nothing
end

function wait_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return nothing
end

function wait_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    # jl_enq_work(shadow(t)) 
    orig = LLVM.Instruction(OrigCI)
    origops = LLVM.operands(orig)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    enq_work_fn = find_match(mod, "jl_enq_work")
    @assert enq_work_fn !== nothing
    shadowtask = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, API.EnzymeGradientUtilsInvertPointer(gutils, origops[1], B), B))
    cal = LLVM.call!(LLVM.Builder(B), enq_work_fn, [shadowtask])
    API.EnzymeGradientUtilsSetDebugLocFromOriginal(gutils, cal, orig)
    conv = LLVM.API.LLVMGetInstructionCallConv(orig)
    LLVM.API.LLVMSetInstructionCallConv(cal, conv)
    return nothing
end

function arraycopy_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    
    orig = LLVM.Instruction(OrigCI)
    origops = LLVM.operands(orig)

    width = API.EnzymeGradientUtilsGetWidth(gutils)

    shadowin = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, origops[1], B))

    B = LLVM.Builder(B)

    if width == 1
        shadowres = LLVM.call!(B, LLVM.called_value(orig), [shadowin])
        if API.runtimeActivity()
            prev = LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, orig))
            shadowres = LLVM.select!(B, LLVM.icmp!(B, LLVM.API.LLVMIntNE, shadowin, LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[1]))), shadowres, prev)
            API.moveBefore(prev, shadowres, C_NULL)
        end
    else
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig))))
        for idx in 1:width
            ev = extract_value!(B, shadowin, idx-1)
            callv = LLVM.call!(B, LLVM.called_value(orig), [ev])
            if API.runtimeActivity()
                prev = LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, orig))
                callv = LLVM.select!(B, LLVM.icmp!(B, LLVM.API.LLVMIntNE, ev, LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[1]))), callv, prev)
                if idx == 1
                    API.moveBefore(prev, callv, C_NULL)
                end
            end
            shadowres = insert_value!(B, shadowres, callv, idx-1)
        end
    end
 
    unsafe_store!(shadowR, shadowres.ref)
	
	return nothing
end

function arraycopy_common(fwd, B, orig, origArg, gutils, shadowdst)
    # size_t len = jl_array_len(ary);
    # size_t elsz = ary->elsize;
    # memcpy(new_ary->data, ary->data, len * elsz);
	# JL_EXTENSION typedef struct {
	# 	JL_DATA_TYPE
	# 	void *data;
	# #ifdef STORE_ARRAY_LEN
	# 	size_t length;
	# #endif
	# 	jl_array_flags_t flags;
	# 	uint16_t elsize;  // element size including alignment (dim 1 memory stride)

	tt = TypeTree(API.EnzymeGradientUtilsAllocAndGetTypeTree(gutils, orig))
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
	dl = string(LLVM.datalayout(mod))
	API.EnzymeTypeTreeLookupEq(tt, 1, dl)
	data0!(tt)
    ct = API.EnzymeTypeTreeInner0(tt)

    if ct == API.DT_Unknown
        # analyzer = API.EnzymeGradientUtilsTypeAnalyzer(gutils)
        # ip = API.EnzymeTypeAnalyzerToString(analyzer)
        # sval = Base.unsafe_string(ip)
        # API.EnzymeStringFree(ip)
        GPUCompiler.@safe_warn "Unknown concrete type" tt=string(tt) orig=string(orig)
        emit_error(B, "Enzyme: Unknown concrete type in arraycopy_common")
        return nothing
    end

    @assert ct != API.DT_Unknown
    ctx = LLVM.context(orig)
    secretty = API.EnzymeConcreteTypeIsFloat(ct, ctx)

    off = sizeof(Cstring)
    if true # STORE_ARRAY_LEN
        off += sizeof(Csize_t)
    end
    #jl_array_flags_t
    off += 2

    actualOp = LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origArg))
    if fwd
        B0 = B
    elseif typeof(actualOp) <: LLVM.Argument
        B0 = LLVM.Builder(ctx)
        position!(B0, first(instructions(LLVM.BasicBlock(API.EnzymeGradientUtilsNewFromOriginal(gutils, LLVM.entry(LLVM.parent(LLVM.parent(orig))))))))
    else
        B0 = LLVM.Builder(ctx)
        nextInst = LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(actualOp))
        while isa(nextInst, LLVM.PHIInst)
            nextInst = LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(nextInst))
        end
        position!(B0, nextInst)
    end
    
    elSize = get_array_elsz(B0, actualOp)
    elSize = LLVM.zext!(B0, elSize, LLVM.IntType(8*sizeof(Csize_t); ctx))

    len = get_array_len(B0, actualOp)

    length = LLVM.mul!(B0, len, elSize)
    isVolatile = LLVM.ConstantInt(LLVM.IntType(1; ctx), 0)

    # forward pass copy already done by underlying call
    allowForward = false
    intrinsic = LLVM.Intrinsic("llvm.memcpy").id

    if !fwd
        shadowdst = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, shadowdst, B))
    end
    shadowsrc = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, origArg, B))
    if !fwd
        shadowsrc = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, shadowsrc, B))
    end
    
    width = API.EnzymeGradientUtilsGetWidth(gutils)
    if width == 1
    
    shadowsrc = get_array_data(B, shadowsrc)
    shadowdst = get_array_data(B, shadowdst)

    API.sub_transfer(gutils, fwd ? API.DEM_ReverseModePrimal : API.DEM_ReverseModeGradient, secretty, intrinsic, #=dstAlign=#1, #=srcAlign=#1, #=offset=#0, false, shadowdst, false, shadowsrc, length, isVolatile, orig, allowForward, #=shadowsLookedUp=#!fwd)
    
    else
    for i in 1:width

    evsrc = extract_value!(B, shadowsrc, i-1)
    evdst = extract_value!(B, shadowdst, i-1)

    shadowsrc0 = load!(B, bitcast!(B, evsrc, LLVM.PointerType(LLVM.PointerType(LLVM.IntType(8; ctx), 13), LLVM.addrspace(LLVM.llvmtype(evsrc)))))
    shadowdst0 = load!(B, bitcast!(B, evdst, LLVM.PointerType(LLVM.PointerType(LLVM.IntType(8; ctx), 13), LLVM.addrspace(LLVM.llvmtype(evdst)))))

    API.sub_transfer(gutils, fwd ? API.DEM_ReverseModePrimal : API.DEM_ReverseModeGradient, secretty, intrinsic, #=dstAlign=#1, #=srcAlign=#1, #=offset=#0, false, shadowdst0, false, shadowsrc0, length, isVolatile, orig, allowForward, #=shadowsLookedUp=#!fwd)
    end

    end

    return nothing
end

function arraycopy_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid   
    arraycopy_fwd(B, OrigCI, gutils, normalR, shadowR)

    shadowres = LLVM.Value(unsafe_load(shadowR))
    
    orig = LLVM.Instruction(OrigCI)
    origops = LLVM.operands(orig)

    if API.EnzymeGradientUtilsIsConstantValue(gutils, origops[1]) == 0
      arraycopy_common(#=fwd=#true, LLVM.Builder(B), orig, origops[1], gutils, shadowres)
    end
	
	return nothing
end

function arraycopy_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    origops = LLVM.operands(orig)
    
    if API.EnzymeGradientUtilsIsConstantValue(gutils, origops[1]) == 0
      arraycopy_common(#=fwd=#false, LLVM.Builder(B), orig, origops[1], gutils, LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, orig, B)))
    end

    return nothing
end

function arrayreshape_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    
    orig = LLVM.Instruction(OrigCI)
    origops = LLVM.operands(orig)

    B = LLVM.Builder(B)    
    width = API.EnzymeGradientUtilsGetWidth(gutils)

    shadowin = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, origops[2], B))
    if width == 1
        args = LLVM.Value[
                          LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[1]))
                          shadowin
                          LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[3]))
                          ]
        shadowres = LLVM.call!(B, LLVM.called_value(orig), args)
    else
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig))))
        for idx in 1:width
            args = LLVM.Value[
                              LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[1]))
                              extract_value!(B, shadowin, idx-1)
                              LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[3]))
                              ]
            tmp = LLVM.call!(B, LLVM.called_value(orig), args)
            shadowres = insert_value!(B, shadowres, tmp, idx-1)
        end
    end
    unsafe_store!(shadowR, shadowres.ref)
	
	return nothing
end

function arrayreshape_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    arrayreshape_fwd(B, OrigCI, gutils, normalR, shadowR)
    return nothing
end

function arrayreshape_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    return nothing
end

function f_tuple_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    origops = collect(operands(orig))
    width = API.EnzymeGradientUtilsGetWidth(gutils)
    if API.EnzymeGradientUtilsIsConstantValue(gutils, orig) == 0
        B = LLVM.Builder(B)

        shadowsin = LLVM.Value[
                        LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, o, B)) for o in origops[1:end-1] ]
        if width == 1
            shadowres = LLVM.call!(B, LLVM.called_value(orig), shadowsin)
            conv = LLVM.API.LLVMGetInstructionCallConv(orig)
            LLVM.API.LLVMSetInstructionCallConv(shadowres, conv)
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig))))
            for idx in 1:width
                args = LLVM.Value[
                                  extract_value!(B, s, idx-1) for s in shadowsin
                                  ]
                tmp = LLVM.call!(B, LLVM.called_value(orig), args)
                conv = LLVM.API.LLVMGetInstructionCallConv(orig)
                LLVM.API.LLVMSetInstructionCallConv(tmp, conv)
                shadowres = insert_value!(B, shadowres, tmp, idx-1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return nothing
end

function f_tuple_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    f_tuple_fwd(B, OrigCI, gutils, normalR, shadowR)
    return nothing
end

function f_tuple_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    # This function allocates a new return which returns a pointer, thus this instruction itself cannot transfer
    # derivative info, only create a shadow pointer, which is handled by the forward pass.
    return nothing
end

function apply_iterate_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    emit_error(LLVM.Builder(B), "Enzyme: Not yet implemented augmented forward for jl_f__apply_iterate")

    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return nothing
end

function apply_iterate_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    emit_error(LLVM.Builder(B), "Enzyme: Not yet implemented reverse for jl_f__apply_iterate")
    return nothing
end

function new_structv_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    emit_error(LLVM.Builder(B), "Enzyme: Not yet implemented augmented forward for jl_new_struct")

    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return nothing
end

function new_structv_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    emit_error(LLVM.Builder(B), "Enzyme: Not yet implemented reverse for jl_new_struct")
    return nothing
end

function gcpreserve_begin_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)

    ops = collect(operands(orig))[1:end-1]

    to_preserve = LLVM.Value[]

    width = API.EnzymeGradientUtilsGetWidth(gutils)
    B = LLVM.Builder(B)
    for op in ops
        val = LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, op))
        push!(to_preserve, val)

        active = API.EnzymeGradientUtilsIsConstantValue(gutils, op) == 0

        if active
            shadowin = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, op, B))

            if width == 1
                push!(to_preserve, shadowin)
            else
                for idx in 1:width
                    push!(to_preserve, extract_value!(B, shadowin, idx-1))
                end
            end
        end
    end

    token = emit_gc_preserve_begin(B, to_preserve)
    unsafe_store!(normalR, token.ref)

    return nothing
end

function gcpreserve_begin_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)

    ops = collect(operands(orig))[1:end-1]

    to_preserve = LLVM.Value[]

    width = API.EnzymeGradientUtilsGetWidth(gutils)
    B = LLVM.Builder(B)
    for op in ops
        val = LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, op))
        push!(to_preserve, val)

        active = API.EnzymeGradientUtilsIsConstantValue(gutils, op) == 0

        if active
            shadowin = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, op, B))

            if width == 1
                push!(to_preserve, shadowin)
            else
                for idx in 1:width
                    push!(to_preserve, extract_value!(B, shadowin, idx-1))
                end
            end
        end
    end

    token = emit_gc_preserve_begin(B, to_preserve)
    unsafe_store!(normalR, token.ref)

    return nothing
end

const GCToks = Dict{LLVM.Instruction, LLVM.Instruction}()

function gcpreserve_begin_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    builder = LLVM.Builder(B)
    orig = LLVM.Instruction(OrigCI)
    if haskey(GCToks, orig)
        token = GCToks[orig]
        delete!(GCToks, orig)
    else
        f   = LLVM.parent(orig)
        mod = LLVM.parent(f)
        ctx = LLVM.context(mod)

        
        token = emit_gc_preserve_begin(LLVM.Builder(B))
        # token = LLVM.phi!(builder, LLVM.TokenType(ctx), "placeholder")
        GCToks[orig] = token
    end
    emit_gc_preserve_end(builder, token)
    return nothing
end

function gcpreserve_end_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    return nothing
end

function gcpreserve_end_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    origPres = operands(orig)[1]

    ops = collect(operands(origPres))[1:end-1]

    to_preserve = LLVM.Value[]

    for op in ops
        val = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, API.EnzymeGradientUtilsNewFromOriginal(gutils, op), B))
        push!(to_preserve, val)

        active = API.EnzymeGradientUtilsIsConstantValue(gutils, op) == 0

        if active
            push!(to_preserve, LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, API.EnzymeGradientUtilsInvertPointer(gutils, op, B), B)))
        end
    end

    token = emit_gc_preserve_begin(LLVM.Builder(B), to_preserve)

    if haskey(GCToks, origPres)
        placeHolder = GCToks[origPres]
        LLVM.replace_uses!(placeHolder, token)
        delete!(GCToks, origPres)
        LLVM.API.LLVMInstructionEraseFromParent(placeHolder)
    else
        GCToks[origPres] = token
    end

    return nothing
end

function jl_array_grow_end_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    origops = collect(operands(orig))
    if API.EnzymeGradientUtilsIsConstantValue(gutils, origops[1]) == 0
        B = LLVM.Builder(B)

        width = API.EnzymeGradientUtilsGetWidth(gutils)

        shadowin = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, origops[1], B))
        if width == 1
            args = LLVM.Value[
                              shadowin
                              LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[2]))
                              ]
            LLVM.call!(B, LLVM.called_value(orig), args)
        else
            for idx in 1:width
                args = LLVM.Value[
                                  extract_value!(B, shadowin, idx-1)
                                  LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[2]))
                                  ]
                LLVM.call!(B, LLVM.called_value(orig), args)
            end
        end
    end
    return nothing
end


function jl_array_grow_end_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    origops = collect(operands(orig))
    if API.EnzymeGradientUtilsIsConstantValue(gutils, origops[1]) == 0
        B = LLVM.Builder(B)

        width = API.EnzymeGradientUtilsGetWidth(gutils)

        shadowin = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, origops[1], B))
        ctx = LLVM.context(orig)
        i8 = LLVM.IntType(8; ctx)
        
        inc = LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[2]))
        
        al = 0
        
        if width == 1
            anti = shadowin
            
            idx = get_array_nrows(B, anti)
            elsz = zext!(B, get_array_elsz(B, anti), llvmtype(idx))
            off = mul!(B, idx, elsz)
            tot = mul!(B, inc, elsz)
            
            args = LLVM.Value[anti, inc]
            LLVM.call!(B, LLVM.called_value(orig), args)

            toset = get_array_data(B, anti)
            toset = gep!(B, toset, LLVM.Value[off])
            mcall = LLVM.memset!(B, toset, LLVM.ConstantInt(i8, 0, false), tot, al)
        else
            for idx in 1:width
                anti = extract_value!(B, shadowin, idx-1)

                idx = get_array_nrows(B, anti)
                elsz = zext!(B, get_array_elsz(B, anti), llvmtype(idx))
                off = mul!(B, idx, elsz)
                tot = mul!(B, inc, elsz)
                
                args = LLVM.Value[anti, inc]
                LLVM.call!(B, LLVM.called_value(orig), args)

                toset = get_array_data(B, anti)
                toset = gep!(B, toset, LLVM.Value[off])
                mcall = LLVM.memset!(B, toset, LLVM.ConstantInt(i8, 0, false), tot, al)
            end
        end
    end

    return nothing
end

function jl_array_grow_end_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    origops = collect(operands(orig))
    if API.EnzymeGradientUtilsIsConstantValue(gutils, origops[1]) == 0
        B = LLVM.Builder(B)

        width = API.EnzymeGradientUtilsGetWidth(gutils)
        
        called_value = origops[end]
        funcT = eltype(llvmtype(called_value)::LLVM.PointerType)::LLVM.FunctionType
        mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
        delF = get_function!(mod, "jl_array_del_end", funcT)

        shadowin = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, origops[1], B))
        shadowin = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, shadowin, B))

        offset = LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[2]))
        offset = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, offset, B))

        if width == 1
            args = LLVM.Value[
                              shadowin
                              offset
                              ]
            LLVM.call!(B, delF, args)
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig))))
            for idx in 1:width
                args = LLVM.Value[
                                  extract_value!(B, shadowin, idx-1)
                                  offset
                                  ]
                LLVM.call!(B, delF, args)
            end
        end
    end
    return nothing
end

function jl_array_del_end_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    jl_array_grow_end_fwd(B, OrigCI, gutils, normalR, shadowR)
    return nothing
end

function jl_array_del_end_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    jl_array_del_end_fwd(B, OrigCI, gutils, normalR, shadowR)
    return nothing
end

function jl_array_del_end_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    origops = collect(operands(orig))
    if API.EnzymeGradientUtilsIsConstantValue(gutils, origops[1]) == 0
        B = LLVM.Builder(B)

        width = API.EnzymeGradientUtilsGetWidth(gutils)
        
        called_value = origops[end]
        funcT = eltype(llvmtype(called_value)::LLVM.PointerType)::LLVM.FunctionType
        mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
        delf = get_function!(mod, "jl_array_grow_end", funcT)

        shadowin = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, origops[1], B))
        shadowin = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, shadowin, B))

        offset = LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[2]))
        offset = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, offset, B))

        if width == 1
            args = LLVM.Value[
                              shadowin
                              offset
                              ]
            LLVM.call!(B, delF, args)
        else
            for idx in 1:width
                args = LLVM.Value[
                                  extract_value!(B, shadowin, idx-1)
                                  offset
                                  ]
                LLVM.call!(B, delF, args)
            end
        end
        
        # GPUCompiler.@safe_warn "Not applying memsetUnknown concrete type" tt=string(tt)
        emit_error(B, "Not applying memset on reverse of jl_array_del_end")
        # memset(data + idx * elsz, 0, inc * elsz);
    end
    return nothing
end

function jl_getfield_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    if shadowR != C_NULL
        orig = LLVM.Instruction(OrigCI)
        origops = collect(operands(orig))
        width = API.EnzymeGradientUtilsGetWidth(gutils)
        if API.EnzymeGradientUtilsIsConstantValue(gutils, origops[2]) == 0
            B = LLVM.Builder(B)


            shadowin = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, origops[2], B))
            if width == 1
                args = LLVM.Value[
                                  LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[1]))
                                  shadowin
                                  ]
                for a in origops[3:end-1]
                    push!(args, LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, a)))
                end

                shadowres = LLVM.call!(B, LLVM.called_value(orig), args)
                conv = LLVM.API.LLVMGetInstructionCallConv(orig)
                LLVM.API.LLVMSetInstructionCallConv(shadowres, conv)
            else
                shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig))))
                for idx in 1:width
                    args = LLVM.Value[
                                      LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[1]))
                                      extract_value!(B, shadowin, idx-1)
                                      ]
                    for a in origops[3:end-1]
                        push!(args, LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, a)))
                    end
                    tmp = LLVM.call!(B, LLVM.called_value(orig), args)
                    conv = LLVM.API.LLVMGetInstructionCallConv(orig)
                    LLVM.API.LLVMSetInstructionCallConv(tmp, conv)
                    shadowres = insert_value!(B, shadowres, tmp, idx-1)
                end
            end
            unsafe_store!(shadowR, shadowres.ref)
        else
            normal = LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, orig))
            if width == 1
                shadowres = normal
            else
                shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(normal))))
                for idx in 1:width
                    shadowres = insert_value!(B, shadowres, normal, idx-1)
                end
            end
            unsafe_store!(shadowR, shadowres.ref)
        end
    end
    return nothing
end
function jl_getfield_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
  jl_getfield_fwd(B, OrigCI, gutils, normalR, shadowR)
end
function jl_getfield_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid 
    orig = LLVM.Instruction(OrigCI)
    if API.EnzymeGradientUtilsIsConstantValue(gutils, orig) == 0
        origops = collect(operands(orig))
        width = API.EnzymeGradientUtilsGetWidth(gutils)
        if API.EnzymeGradientUtilsIsConstantValue(gutils, origops[2]) == 0
            B = LLVM.Builder(B)

            shadowin = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, API.EnzymeGradientUtilsInvertPointer(gutils, origops[2], B), B))
            shadowout = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, API.EnzymeGradientUtilsInvertPointer(gutils, orig, B), B))
			callval = LLVM.called_value(orig)
			callval = first(operands(callval))::LLVM.Function
        	funcT = eltype(llvmtype(callval)::LLVM.PointerType)::LLVM.FunctionType
			setF = get_function!(LLVM.parent(callval), "jl_f_setfield", funcT)
			setF = LLVM.const_pointercast(setF, LLVM.PointerType(LLVM.FunctionType(llvmtype(orig), LLVM.LLVMType[]; vararg=true)))

            o1 = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[1]), B))
            o3 = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[3]), B))

            if width == 1
                args = LLVM.Value[o1
                                  shadowin
                                  o3
								  shadowout
                                  ]
                shadowres = LLVM.call!(B, setF, args)
                conv = LLVM.API.LLVMGetInstructionCallConv(orig)
                LLVM.API.LLVMSetInstructionCallConv(shadowres, conv)
            else
                shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig))))
                for idx in 1:width
                    args = LLVM.Value[o1
                                      extract_value!(B, shadowin, idx-1)
                                      o3
                                      extract_value!(B, shadowout, idx-1)
                                      ]
                    tmp = LLVM.call!(B, setF, args)
                    conv = LLVM.API.LLVMGetInstructionCallConv(orig)
                    LLVM.API.LLVMSetInstructionCallConv(tmp, conv)
                end
            end
        end
    end
    return nothing
end

function jl_array_sizehint_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    origops = collect(operands(orig))
    width = API.EnzymeGradientUtilsGetWidth(gutils)
    if API.EnzymeGradientUtilsIsConstantValue(gutils, origops[1]) == 0
        B = LLVM.Builder(B)


        shadowin = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, origops[1], B))
        if width == 1
            args = LLVM.Value[
                              shadowin
                              LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[2]))
                              ]
            LLVM.call!(B, LLVM.called_value(orig), args)
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig))))
            for idx in 1:width
                args = LLVM.Value[
                                  extract_value!(B, shadowin, idx-1)
                                  LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[2]))
                                  ]
                LLVM.call!(B, LLVM.called_value(orig), args)
            end
        end
    end
    return nothing
end

function jl_array_sizehint_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    jl_array_sizehint_fwd(B, OrigCI, gutils, normalR, shadowR, tapeR)
     return nothing
end

function jl_array_sizehint_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    return nothing
end

function jl_f__apply_iterate_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    emit_error(LLVM.Builder(B), "Enzyme: unhandled augmented forward for jl_f__apply_iterate")
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return nothing
end

function setfield_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    
    orig = LLVM.Instruction(OrigCI)
    origops = collect(operands(orig))
    if API.EnzymeGradientUtilsIsConstantValue(gutils, origops[4]) == 0
        width = API.EnzymeGradientUtilsGetWidth(gutils)

        B = LLVM.Builder(B)
        shadowin = if API.EnzymeGradientUtilsIsConstantValue(gutils, origops[2]) == 0
            LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, origops[2], B))
        else
            LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[2]))
        end

        shadowout = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, origops[4], B))
        if width == 1
            args = LLVM.Value[
                              LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[1]))
                              shadowin
                              LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[3]))
                              shadowout
                              ]
            shadowres = LLVM.call!(B, LLVM.called_value(orig), args)
            conv = LLVM.API.LLVMGetInstructionCallConv(orig)
            LLVM.API.LLVMSetInstructionCallConv(shadowres, conv)
        else
            for idx in 1:width
                args = LLVM.Value[
                                  LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[1]))
                                  extract_value!(B, shadowin, idx-1)
                                  LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[3]))
                                  extract_value!(B, shadowout, idx-1)
                                  ]
                tmp = LLVM.call!(B, setF, args)
                conv = LLVM.API.LLVMGetInstructionCallConv(orig)
                LLVM.API.LLVMSetInstructionCallConv(tmp, conv)
            end
        end
    end
    return nothing
end

function setfield_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    emit_error(LLVM.Builder(B), "Enzyme: unhandled augmented forward for jl_f_setfield")
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return nothing
end

function setfield_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
  emit_error(LLVM.Builder(B), "Enzyme: unhandled reverse for jl_f_setfield")
  return nothing
end

function get_binding_or_error_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    CI = API.EnzymeGradientUtilsNewFromOriginal(gutils, OrigCI)
    err = emit_error(LLVM.Builder(B), "Enzyme: unhandled forward for jl_get_binding_or_error")
    API.moveBefore(CI, err, C_NULL)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing

    if shadowR != C_NULL && normal !== nothing
        width = API.EnzymeGradientUtilsGetWidth(gutils)
        B = LLVM.Builder(B)
        if width == 1
            shadowres = normal
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(normal))))
            for idx in 1:width
                shadowres = insert_value!(B, shadowres, normal, idx-1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return nothing
end

function get_binding_or_error_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    CI = API.EnzymeGradientUtilsNewFromOriginal(gutils, OrigCI)
    err = emit_error(LLVM.Builder(B), "Enzyme: unhandled augmented forward for jl_get_binding_or_error")
    API.moveBefore(CI, err, C_NULL)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        width = API.EnzymeGradientUtilsGetWidth(gutils)
        B = LLVM.Builder(B)
        if width == 1
            shadowres = normal
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(normal))))
            for idx in 1:width
                shadowres = insert_value!(B, shadowres, normal, idx-1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return nothing
end

function get_binding_or_error_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    emit_error(LLVM.Builder(B), "Enzyme: unhandled reverse for jl_get_binding_or_error")
    return nothing
end

function finalizer_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    CI = API.EnzymeGradientUtilsNewFromOriginal(gutils, OrigCI)
    err = emit_error(LLVM.Builder(B), "Enzyme: unhandled augmented forward for jl_gc_add_finalizer_th")
    API.moveBefore(CI, err, C_NULL)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return nothing
end

function finalizer_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    CI = API.EnzymeGradientUtilsNewFromOriginal(gutils, OrigCI)
    # err = emit_error(LLVM.Builder(B), "Enzyme: unhandled augmented forward for jl_gc_add_finalizer_th")
    # API.moveBefore(CI, err, C_NULL)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    # Delete the primal code
    if normal !== nothing
        unsafe_store!(normalR, C_NULL)
    else
        LLVM.API.LLVMInstructionEraseFromParent(LLVM.Instruction(API.EnzymeGradientUtilsNewFromOriginal(gutils, orig)))
    end
    return nothing
end

function finalizer_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    # emit_error(LLVM.Builder(B), "Enzyme: unhandled reverse for jl_gc_add_finalizer_th")
    return nothing
end


function register_handler!(variants, augfwd_handler, rev_handler, fwd_handler=nothing)
    for variant in variants
        if augfwd_handler !== nothing && rev_handler !== nothing
            API.EnzymeRegisterCallHandler(variant, augfwd_handler, rev_handler)
        end
        if fwd_handler !== nothing
            API.EnzymeRegisterFwdCallHandler(variant, fwd_handler)
        end
    end
end

function register_alloc_handler!(variants, alloc_handler, free_handler)
    for variant in variants
        API.EnzymeRegisterAllocationHandler(variant, alloc_handler, free_handler)
    end
end

abstract type CompilationException <: Base.Exception end
struct NoDerivativeException <: CompilationException
    msg::String
    ir::Union{Nothing, String}
    bt::Union{Nothing, Vector{StackTraces.StackFrame}}
end

function Base.showerror(io::IO, ece::NoDerivativeException)
    print(io, "Enzyme compilation failed.\n")
    if ece.ir !== nothing
        print(io, "Current scope: \n")
        print(io, ece.ir)
    end
    print(io, '\n', ece.msg, '\n')
    if ece.bt !== nothing
        Base.show_backtrace(io, ece.bt)
        println(io)
    end
end

struct NoShadowException <: CompilationException
    msg::String
    sval::String
    ir::Union{Nothing, String}
    bt::Union{Nothing, Vector{StackTraces.StackFrame}}
end

function Base.showerror(io::IO, ece::NoShadowException)
    print(io, "Enzyme compilation failed due missing shadow.\n")
    if ece.ir !== nothing
        print(io, "Current scope: \n")
        print(io, ece.ir)
    end
    print(io, "\n Inverted pointers: \n")
    write(io, ece.sval)
    print(io, '\n', ece.msg, '\n')
    if ece.bt !== nothing
        print(io,"\nCaused by:")
        Base.show_backtrace(io, ece.bt)
        println(io)
    end
end

struct IllegalTypeAnalysisException <: CompilationException
    msg::String
    sval::String
    ir::Union{Nothing, String}
    bt::Union{Nothing, Vector{StackTraces.StackFrame}}
end

function Base.showerror(io::IO, ece::IllegalTypeAnalysisException)
    print(io, "Enzyme compilation failed due to illegal type analysis.\n")
    if ece.ir !== nothing
        print(io, "Current scope: \n")
        print(io, ece.ir)
    end
    print(io, "\n Type analysis state: \n")
    write(io, ece.sval)
    print(io, '\n', ece.msg, '\n')
    if ece.bt !== nothing
        print(io,"\nCaused by:")
        Base.show_backtrace(io, ece.bt)
        println(io)
    end
end

struct NoTypeException <: CompilationException
    msg::String
    sval::String
    ir::Union{Nothing, String}
    bt::Union{Nothing, Vector{StackTraces.StackFrame}}
    val::LLVM.Instruction
end

function Base.showerror(io::IO, ece::NoTypeException)
    print(io, "Enzyme cannot deduce type\n")
    if ece.ir !== nothing
        print(io, "Current scope: \n")
        print(io, ece.ir)
    end
    print(io, "\n Type analysis state: \n")
    write(io, ece.sval)
    print(io, '\n', ece.msg, '\n')
    if ece.bt !== nothing
        print(io,"\nCaused by:")
        Base.show_backtrace(io, ece.bt)
        println(io)
    end
end

parent_scope(val::LLVM.Function, depth=0) = depth==0 ? LLVM.parent(val) : val
parent_scope(val::LLVM.Module, depth=0) = val
parent_scope(val::LLVM.Value, depth=0) = parent_scope(LLVM.parent(val), depth+1)
parent_scope(val::LLVM.Argument, depth=0) = parent_scope(LLVM.Function(LLVM.API.LLVMGetParamParent(val)), depth+1)

function julia_error(cstr::Cstring, val::LLVM.API.LLVMValueRef, errtype::API.ErrorType, data::Ptr{Cvoid})
    msg = Base.unsafe_string(cstr)
    val = LLVM.Value(val)
    if isa(val, LLVM.Instruction)
        bt = GPUCompiler.backtrace(val)
    else
        bt = nothing
    end
    if isa(val, LLVM.ConstantExpr)
        ir = nothing
    else
        # Need to convert function to string, since when the error is going to be printed
        # the module might have been destroyed
        ir = sprint(io->show(io, parent_scope(val)))
    end
    if errtype == API.ET_NoDerivative
        throw(NoDerivativeException(msg, ir, bt))
    elseif errtype == API.ET_NoShadow
        data = API.EnzymeGradientUtilsRef(data)
        ip = API.EnzymeGradientUtilsInvertedPointersToString(data)
        sval = Base.unsafe_string(ip)
        API.EnzymeStringFree(ip)
        throw(NoShadowException(msg, sval, ir, bt))
    elseif errtype == API.ET_IllegalTypeAnalysis
        data = API.EnzymeTypeAnalyzerRef(data)
        ip = API.EnzymeTypeAnalyzerToString(data)
        sval = Base.unsafe_string(ip)
        API.EnzymeStringFree(ip)
        throw(IllegalTypeAnalysisException(msg, sval, ir, bt))
    elseif errtype == API.ET_NoType
        data = API.EnzymeTypeAnalyzerRef(data)
        ip = API.EnzymeTypeAnalyzerToString(data)
        sval = Base.unsafe_string(ip)
        API.EnzymeStringFree(ip)
        throw(NoTypeException(msg, sval, ir, bt, val))
    end
    throw(AssertionError("Unknown errtype"))
end

function any_jltypes(Type::LLVM.PointerType)
    if 10 <= LLVM.addrspace(Type) <= 12
        return true
    else
        # do we care about {} addrspace(11)**
        return false
    end
end

any_jltypes(Type::LLVM.StructType) = any(any_jltypes, LLVM.elements(Type))
any_jltypes(Type::Union{LLVM.VectorType, LLVM.ArrayType}) = any_jltypes(eltype(Type))
any_jltypes(::LLVM.IntegerType) = false
any_jltypes(::LLVM.FloatingPointType) = false


@inline any_jltypes(::Type{Nothing}) = false
@inline any_jltypes(::Type{T}) where {T<:AbstractFloat} = false
@inline any_jltypes(::Type{T}) where {T<:Integer} = false
@inline any_jltypes(::Type{NTuple{Size, T}}) where {Size, T} = any_jltypes(T)
@inline any_jltypes(::Type{Core.LLVMPtr{T, Addr}}) where {T, Addr} = 10 <= Addr <= 12
@inline any_jltypes(::Type{Any}) = true
@inline any_jltypes(::Type{NamedTuple{A,B}}) where {A,B} = any(any_jltypes(b) for b in B.parameters)

nfields(Type::LLVM.StructType) = length(LLVM.elements(Type))
nfields(Type::LLVM.VectorType) = size(Type)
nfields(Type::LLVM.ArrayType) = length(Type)
nfields(Type::LLVM.PointerType) = 1

mutable struct EnzymeTape{N, T}
    data::NTuple{N, T}
end

mutable struct EnzymeTapeToLoad{T}
    data::T
end
Base.eltype(::EnzymeTapeToLoad{T}) where T = T

const TapeTypes = Dict{String, DataType}()

base_type(T::UnionAll) = base_type(T.body)
base_type(T::DataType) = T

function to_tape_type(Type::LLVM.PointerType)
    if 10 <= LLVM.addrspace(Type) <= 12
        return Any
    else
        return Core.LLVMPtr{to_tape_type(eltype(Type)), Int(LLVM.addrspace(Type))}
    end
end

to_tape_type(Type::LLVM.StructType) = AnonymousStruct(Tuple{map(to_tape_type, LLVM.elements(Type))...})
to_tape_type(Type::LLVM.ArrayType) = AnonymousStruct(NTuple{Int(length(Type)), to_tape_type(eltype(Type))})
to_tape_type(Type::LLVM.VectorType) = AnonymousStruct(NTuple{Int(size(Type)), to_tape_type(eltype(Type))})

function to_tape_type(Type::LLVM.IntegerType)
    N = width(Type)
    if N == 1
        return Bool
    elseif N == 8
        return UInt8
    elseif N == 16
        return UInt16
    elseif N == 32
        return UInt32
    elseif N == 64 
        return UInt64
    elseif N == 128
        return UInt128
    else
        error("Can't construct tape type for $N")
    end
end

to_tape_type(::LLVM.LLVMHalf) = Float16
to_tape_type(::LLVM.LLVMFloat) = Float32
to_tape_type(::LLVM.LLVMDouble) = Float64
to_tape_type(::LLVM.LLVMFP128) = Float128

function tape_type(LLVMType) 
    TT = to_tape_type(LLVMType)
    return TT
end

from_tape_type(::Type{T}, ctx) where T<:AbstractFloat = convert(LLVMType, T; ctx)
from_tape_type(::Type{T}, ctx) where T<:Integer = convert(LLVMType, T; ctx)
from_tape_type(::Type{NTuple{Size, T}}, ctx) where {Size, T} = LLVM.ArrayType(from_tape_type(T, ctx), Size)
from_tape_type(::Type{Core.LLVMPtr{T, Addr}}, ctx) where {T, Addr} = LLVM.PointerType(from_tape_type(T, ctx), Addr)
from_tape_type(::Type{Any}, ctx) where {T, Addr} = LLVM.PointerType(LLVM.StructType(LLVM.LLVMType[]; ctx), 10)
function from_tape_type(::Type{NamedTuple{A,B}}, ctx) where {A,B}
    if length(B.parameters) >= 1 && all(B.parameters[1] == b for b in B.parameters)
        return LLVM.ArrayType(from_tape_type(B.parameters[1], ctx), length(B.parameters))
    else
        return LLVM.StructType(LLVM.LLVMType[from_tape_type(b, ctx) for b in B.parameters]; ctx)
    end
end
const Tracked = 10

# TODO: Calculate that constant... see get_current_task
current_task_offset() = -12
current_ptls_offset() = 14

function julia_post_cache_store(SI::LLVM.API.LLVMValueRef, B::LLVM.API.LLVMBuilderRef, R2)::LLVM.API.LLVMValueRef
    B = LLVM.Builder(B)
    SI = LLVM.Instruction(SI)
    v = operands(SI)[1]
    p = operands(SI)[2]
    while true
        if isa(p, LLVM.GetElementPtrInst) || isa(p, LLVM.BitCastInst) || isa(p, LLVM.AddrSpaceCastInst)
            p = operands(p)[1]
            continue
        end
        break
    end
    if any_jltypes(llvmtype(v)) && !isa(p, LLVM.AllocaInst)
        ctx = LLVM.context(v)
        T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
        T_prjlvalue = LLVM.PointerType(T_jlvalue, 10) 
        p = bitcast!(B, p, T_prjlvalue)
        if R2 != C_NULL
            unsafe_store!(R2, p.ref)
        end
        
        vals = LLVM.Value[p]
        todo = LLVM.Value[v]
        while length(todo) != 0
            cur = pop!(todo)
            ty = llvmtype(cur)
            if isa(ty, LLVM.PointerType)
                if any_jltypes(ty)
                    cur = bitcast!(B, cur, T_prjlvalue)
                    push!(vals, cur)
                end
                continue
            end
            if isa(ty, LLVM.ArrayType)
                if any_jltypes(ty)
                    for i=1:length(ty)
                        push!(todo, extract_value!(B, cur, i-1))
                    end
                end
                continue
            end
            if isa(ty, LLVM.StructType)
                for (i, t) in enumerate(LLVM.elements(ty))
                    if any_jltypes(t)
                        push!(todo, extract_value!(B, cur, i-1))
                    end
                end
                continue
            end
            GPUCompiler.@safe_warn "Enzyme illegal subtype", ty, cur, SI, p, v
            @assert false
        end
        r = emit_writebarrier!(B, vals)
        return LLVM.API.LLVMValueRef(r.ref)
    end
    return C_NULL
end

function julia_allocator(B::LLVM.API.LLVMBuilderRef, LLVMType::LLVM.API.LLVMTypeRef, Count::LLVM.API.LLVMValueRef, AlignedSize::LLVM.API.LLVMValueRef)
    B = LLVM.Builder(B)
    Count = LLVM.Value(Count)
    AlignedSize = LLVM.Value(AlignedSize)
    LLVMType = LLVM.LLVMType(LLVMType)
    return julia_allocator(B, LLVMType, Count, AlignedSize)
end

function julia_allocator(B, LLVMType, Count, AlignedSize)
    func = LLVM.parent(position(B))
    mod = LLVM.parent(func)
    ctx = context(mod)

    Size = nuwmul!(B, Count, AlignedSize) # should be nsw, nuw

    if any_jltypes(LLVMType)
        T_int8 = LLVM.Int8Type(ctx)
        T_int32 = LLVM.Int32Type(ctx)
        T_int64 = LLVM.Int64Type(ctx)
        T_jlvalue = LLVM.StructType(LLVM.LLVMType[]; ctx)
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
        T_pprjlvalue = LLVM.PointerType(T_prjlvalue)
        T_prjlvalue_UT = LLVM.PointerType(T_jlvalue)
        T_ppjlvalue = LLVM.PointerType(LLVM.PointerType(T_jlvalue))
        T_pint8 = LLVM.PointerType(T_int8)
        T_ppint8 = LLVM.PointerType(T_pint8)

        esizeof(X) = X == Any ? sizeof(Int) : sizeof(X)

        TT = tape_type(LLVMType)
        if esizeof(TT) != convert(Int, AlignedSize)
            @safe_show LLVMType
            GPUCompiler.@safe_warn "Enzyme aligned size and Julia size disagree" AlignedSize=convert(Int, AlignedSize) esizeof(TT) fieldtypes(TT)
        end
        @assert esizeof(TT) == convert(Int, AlignedSize)
        if Count isa LLVM.ConstantInt
            N = convert(Int, Count)

            ETT = N == 1 ? EnzymeTapeToLoad{TT} : EnzymeTape{N, TT}
            @assert sizeof(ETT) <= N*convert(Int, AlignedSize)

            # Obtain tag
            tag = LLVM.ConstantInt(reinterpret(Int, Base.pointer_from_objref(ETT)); ctx)  # do we need to root ETT
            tag = LLVM.const_inttoptr(tag, T_prjlvalue_UT)
            tag = LLVM.const_addrspacecast(tag, T_prjlvalue)
        else
            # TODO: Version switch to get jl_box_int64 and not call37...
        @static if VERSION < v"1.8-"
            box_int64 = get_function!(mod, "jl_box_int64", LLVM.FunctionType(T_prjlvalue, [T_int64]))
        else
            box_int64 = get_function!(mod, "ijl_box_int64", LLVM.FunctionType(T_prjlvalue, [T_int64]))
        end
            boxed_count = call!(B, box_int64, [Count])

            generic_FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_pprjlvalue, T_int32])
            f_apply_type = get_function!(mod, "jl_f_apply_type", generic_FT)


            ptr = unsafe_to_pointer(EnzymeTape)

            EnzymeTapeT = LLVM.ConstantInt(reinterpret(Int, ptr); ctx)
            EnzymeTapeT = LLVM.const_inttoptr(EnzymeTapeT, T_prjlvalue_UT)
            EnzymeTapeT = LLVM.const_addrspacecast(EnzymeTapeT, T_prjlvalue)

            TapeT = LLVM.ConstantInt(reinterpret(Int, Base.pointer_from_objref(TT)); ctx)
            TapeT = LLVM.const_inttoptr(TapeT, T_prjlvalue_UT)
            TapeT = LLVM.const_addrspacecast(TapeT, T_prjlvalue)

            if VERSION < v"1.9.0-"
                FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_prjlvalue, T_prjlvalue, T_prjlvalue])
                f_apply_type = bitcast!(B, f_apply_type, LLVM.PointerType(FT))
                # call cc37 nonnull {}* bitcast ({}* ({}*, {}**, i32)* @jl_f_apply_type to {}* ({}*, {}*, {}*, {}*)*)({}* null, {}* inttoptr (i64 140150176657296 to {}*), {}* %4, {}* inttoptr (i64 140149987564368 to {}*))
                tag = call!(B, f_apply_type, [LLVM.PointerNull(T_prjlvalue), EnzymeTapeT, boxed_count, TapeT])
                LLVM.callconv!(tag, 37)
            else
                # %5 = call nonnull {}* ({}* ({}*, {}**, i32)*, {}*, ...) @julia.call({}* ({}*, {}**, i32)* @jl_f_apply_type, {}* null, {}* inttoptr (i64 139640605802128 to {}*), {}* %4, {}* inttoptr (i64 139640590432896 to {}*))
                julia_call = get_function!(mod, "julia.call",
                    LLVM.FunctionType(T_prjlvalue, 
                                      [LLVM.PointerType(generic_FT), T_prjlvalue]; vararg=true))
                tag = call!(B, julia_call, [f_apply_type, LLVM.PointerNull(T_prjlvalue), EnzymeTapeT, boxed_count, TapeT])
            end
        end

        # Check if Julia version has https://github.com/JuliaLang/julia/pull/46914
        @static if VERSION >= v"1.9.0-DEV.1439"
            needs_dynamic_size_workaround = false
        else
            needs_dynamic_size_workaround = true
        end

        T_size_t = convert(LLVM.LLVMType, Int; ctx)
        if !needs_dynamic_size_workaround
            # TODO Call declare_allocobj/emit_allocobj
            alloc_obj = get_function!(mod, "julia.gc_alloc_obj",
                LLVM.FunctionType(T_prjlvalue, 
                    [T_ppjlvalue, T_size_t, T_prjlvalue]))
        else
            # This doesn't allow for optimizations
            alloc_obj = get_function!(mod, "jl_gc_alloc_typed",
                LLVM.FunctionType(T_prjlvalue,
                    [T_pint8, T_size_t, T_prjlvalue]))
        end

        if VERSION < v"1.7.0"
            ptls = reinsert_gcmarker!(func, B)
            ptls = bitcast!(B, ptls, T_pint8)
        else
            pgcstack = reinsert_gcmarker!(func, B)
            ct = inbounds_gep!(B, 
                bitcast!(B, pgcstack, T_ppjlvalue),
                [LLVM.ConstantInt(current_task_offset(); ctx)])
            ptls_field = inbounds_gep!(B, 
                ct, [LLVM.ConstantInt(current_ptls_offset(); ctx)])
            ptls = load!(B, bitcast!(B, ptls_field, T_ppint8))
        end

        if !needs_dynamic_size_workaround
            if VERSION < v"1.8.0"
                obj = call!(B, alloc_obj, [ptls, Size, tag])
            else
                obj = call!(B, alloc_obj, [ct, Size, tag])
            end
        else
            obj = call!(B, alloc_obj, [ptls, Size, tag])
        end
        AS = Tracked
    else
        ptr8 = LLVM.PointerType(LLVM.IntType(8; ctx))
        mallocF = get_function!(mod, "malloc", LLVM.FunctionType(ptr8, [llvmtype(Count)]))

        obj = call!(B, mallocF, [Size])
        AS = 0
    end

    LLVM.API.LLVMAddCallSiteAttribute(obj, LLVM.API.LLVMAttributeReturnIndex, EnumAttribute("noalias"; ctx))
    LLVM.API.LLVMAddCallSiteAttribute(obj, LLVM.API.LLVMAttributeReturnIndex, EnumAttribute("nonnull"; ctx))
    if isa(Count, LLVM.ConstantInt)
        val = convert(UInt64, AlignedSize)
        val *= convert(UInt64, Count)
        LLVM.API.LLVMAddCallSiteAttribute(obj, LLVM.API.LLVMAttributeReturnIndex, EnumAttribute("dereferenceable", val; ctx))
        LLVM.API.LLVMAddCallSiteAttribute(obj, LLVM.API.LLVMAttributeReturnIndex, EnumAttribute("dereferenceable_or_null", val; ctx))
    end

    mem = pointercast!(B, obj, LLVM.PointerType(LLVMType, AS))
    return LLVM.API.LLVMValueRef(mem.ref)
end

function julia_deallocator(B::LLVM.API.LLVMBuilderRef, Obj::LLVM.API.LLVMValueRef)
    B = LLVM.Builder(B)
    Obj = LLVM.Value(Obj)
    julia_deallocator(B, Obj)
end

function julia_deallocator(B::LLVM.Builder, Obj::LLVM.Value)
    mod = LLVM.parent(LLVM.parent(position(B)))
    ctx = context(mod)
    
    T_void = LLVM.VoidType(ctx)
    if any_jltypes(LLVM.llvmtype(Obj))
        return LLVM.API.LLVMValueRef(C_NULL)
    else
        ptr8 = LLVM.PointerType(LLVM.IntType(8; ctx))
        freeF = get_function!(mod, "free", LLVM.FunctionType(T_void, [ptr8]))
        callf = call!(B, freeF, [pointercast!(B, Obj, ptr8)])
        LLVM.API.LLVMAddCallSiteAttribute(callf, LLVM.API.LLVMAttributeIndex(1), EnumAttribute("nonnull"; ctx))
    end
    return LLVM.API.LLVMValueRef(callf.ref)
end

function emit_inacterror(B, V, orig)
    B = LLVM.Builder(B)
    curent_bb = position(B)
    orig = LLVM.Value(orig)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    ctx = context(mod)
    
    bt = GPUCompiler.backtrace(orig)
    bts = sprint(io->Base.show_backtrace(io, bt))
    fmt = globalstring_ptr!(B, "%s:\nBacktrace\n"*bts)

    funcT = LLVM.FunctionType(LLVM.VoidType(ctx), LLVMType[LLVM.PointerType(LLVM.Int8Type(ctx))], vararg=true)
    func = get_function!(mod, "jl_errorf", funcT, [EnumAttribute("noreturn"; ctx)])

    call!(B, func, LLVM.Value[fmt, LLVM.Value(V)])
    return nothing
end

function __init__()
    API.EnzymeSetHandler(@cfunction(julia_error, Cvoid, (Cstring, LLVM.API.LLVMValueRef, API.ErrorType, Ptr{Cvoid})))
    if API.EnzymeHasCustomInactiveSupport()
      API.EnzymeSetRuntimeInactiveError(@cfunction(emit_inacterror, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, LLVM.API.LLVMValueRef)))
    end
    if API.EnzymeHasCustomAllocatorSupport()
        API.EnzymeSetCustomAllocator(@cfunction(
            julia_allocator, LLVM.API.LLVMValueRef,
            (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMTypeRef, LLVM.API.LLVMValueRef, LLVM.API.LLVMValueRef)))
        API.EnzymeSetCustomDeallocator(@cfunction(
            julia_deallocator, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef)))
        API.EnzymeSetPostCacheStore(@cfunction(
            julia_post_cache_store, LLVM.API.LLVMValueRef,
            (LLVM.API.LLVMValueRef, LLVM.API.LLVMBuilderRef, Ptr{LLVM.API.LLVMValueRef})))
    end
    register_alloc_handler!(
        ("jl_alloc_array_1d", "ijl_alloc_array_1d"),
        @cfunction(array_shadow_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, Csize_t, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(null_free_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, LLVM.API.LLVMValueRef))
    )
    register_alloc_handler!(
        ("jl_alloc_array_2d", "ijl_alloc_array_2d"),
        @cfunction(array_shadow_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, Csize_t, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(null_free_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, LLVM.API.LLVMValueRef))
    )
    register_alloc_handler!(
        ("jl_alloc_array_3d", "ijl_alloc_array_3d"),
        @cfunction(array_shadow_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, Csize_t, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(null_free_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, LLVM.API.LLVMValueRef))
    )
    register_handler!(
        ("julia.call",),
        @cfunction(jlcall_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(jlcall_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(jlcall_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("julia.call2",),
        @cfunction(jlcall2_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(jlcall2_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(jlcall2_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_apply_generic", "ijl_apply_generic"),
        @cfunction(generic_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(generic_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(generic_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_invoke", "ijl_invoke", "jl_f_invoke"),
        @cfunction(invoke_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(invoke_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(invoke_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_f__apply_latest", "jl_f__call_latest"),
        @cfunction(apply_latest_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(apply_latest_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(apply_latest_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_threadsfor",),
        @cfunction(threadsfor_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(threadsfor_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(threadsfor_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_pmap",),
        @cfunction(pmap_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(pmap_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(pmap_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_new_task", "ijl_new_task"),
        @cfunction(newtask_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(newtask_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(newtask_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_set_task_threadpoolid", "ijl_set_task_threadpoolid"),
        @cfunction(set_task_tid_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(set_task_tid_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(set_task_tid_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_enq_work",),
        @cfunction(enq_work_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(enq_work_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(enq_work_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}))
    )
    register_handler!(
        ("jl_wait",),
        @cfunction(wait_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(wait_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(wait_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_","jl_breakpoint"),
        @cfunction(noop_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(duplicate_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(noop_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_array_copy","ijl_array_copy"),
        @cfunction(arraycopy_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(arraycopy_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(arraycopy_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_reshape_array","ijl_reshape_array"),
        @cfunction(arrayreshape_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(arrayreshape_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(arrayreshape_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("llvm.julia.gc_preserve_begin",),
        @cfunction(gcpreserve_begin_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(gcpreserve_begin_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(gcpreserve_begin_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("llvm.julia.gc_preserve_end",),
        @cfunction(gcpreserve_end_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(gcpreserve_end_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
    )
    register_handler!(
        ("jl_f_setfield",),
        @cfunction(setfield_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(setfield_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(setfield_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_f_tuple","ijl_f_tuple"),
        @cfunction(f_tuple_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(f_tuple_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(f_tuple_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_f__apply_iterate",),
        @cfunction(apply_iterate_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(apply_iterate_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
    )
    register_handler!(
        ("jl_new_structv",),
        @cfunction(new_structv_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(new_structv_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
    )
    register_handler!(
        ("jl_get_binding_or_error", "ijl_get_binding_or_error"),
        @cfunction(get_binding_or_error_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(get_binding_or_error_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(get_binding_or_error_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_gc_add_finalizer_th",),
        @cfunction(finalizer_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(finalizer_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(finalizer_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_array_grow_end","ijl_array_grow_end"),
        @cfunction(jl_array_grow_end_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(jl_array_grow_end_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(jl_array_grow_end_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_array_del_end","ijl_array_del_end"),
        @cfunction(jl_array_del_end_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(jl_array_del_end_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(jl_array_del_end_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_f_getfield",),
        @cfunction(jl_getfield_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(jl_getfield_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(jl_getfield_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_f__apply_iterate",),
        nothing,
        nothing,
        @cfunction(jl_f__apply_iterate_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_array_sizehint","ijl_array_sizehint"),
        @cfunction(jl_array_sizehint_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(jl_array_sizehint_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(jl_array_sizehint_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
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
    # Whether to also return the primal
    returnPrimal::Bool
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

# provide a specific interpreter to use.
GPUCompiler.get_interpreter(job::CompilerJob{<:Any,<:AbstractEnzymeCompilerParams}) =
    Interpreter.EnzymeInterpeter(GPUCompiler.ci_cache(job), GPUCompiler.method_table(job), job.source.world)

include("compiler/utils.jl")
include("compiler/passes.jl")
include("compiler/optimize.jl")
include("compiler/interpreter.jl")
include("compiler/validation.jl")

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

    for fname in ("jl_f_getfield",)
        if haskey(fns, fname)
            fn = fns[fname]
            push!(function_attributes(fn), LLVM.EnumAttribute("readonly", 0; ctx))
            push!(function_attributes(fn), LLVM.StringAttribute("enzyme_shouldrecompute"; ctx))
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

    for boxfn in ("julia.gc_alloc_obj", "jl_gc_alloc_typed", "ijl_gc_alloc_typed",
                  "jl_box_float32", "jl_box_float64", "jl_box_int32", "jl_box_int64",
                  "ijl_box_float32", "ijl_box_float64", "ijl_box_int32", "ijl_box_int64",
                  "jl_alloc_array_1d", "jl_alloc_array_2d", "jl_alloc_array_3d",
                  "ijl_alloc_array_1d", "ijl_alloc_array_2d", "ijl_alloc_array_3d",
                  "jl_f_tuple", "ijl_f_tuple")
        if haskey(fns, boxfn)
            fn = fns[boxfn]
            push!(return_attributes(fn), LLVM.EnumAttribute("noalias", 0; ctx))
            push!(function_attributes(fn), LLVM.EnumAttribute("inaccessiblememonly", 0; ctx))
        end
    end
    
    for boxfn in ("jl_array_copy", "ijl_array_copy")
        if haskey(fns, boxfn)
            fn = fns[boxfn]
            push!(return_attributes(fn), LLVM.EnumAttribute("noalias", 0; ctx))
            push!(function_attributes(fn), LLVM.EnumAttribute("inaccessiblemem_or_argmemonly", 0; ctx))
        end
    end

    for gc in ("llvm.julia.gc_preserve_begin", "llvm.julia.gc_preserve_end")
        if haskey(fns, gc)
            fn = fns[gc]
            push!(function_attributes(fn), LLVM.EnumAttribute("inaccessiblememonly", 0; ctx))
        end
    end

    for rfn in ("jl_object_id_", "jl_object_id", "ijl_object_id_", "ijl_object_id",
                "jl_eqtable_get", "ijl_eqtable_get")
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
    if API.HasFromStack(inst)
        return UInt8(false)
    end
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

function enzyme_custom_extract_mi(orig)
    mi = nothing
    job = nothing
    for fattr in collect(function_attributes(LLVM.called_value(orig)))
        if isa(fattr, LLVM.StringAttribute)
            if kind(fattr) == "enzymejl_mi"
                ptr = reinterpret(Ptr{Cvoid}, parse(Int, LLVM.value(fattr)))
                mi = Base.unsafe_pointer_to_objref(ptr)
            end
            if kind(fattr) == "enzymejl_job"
                ptr = reinterpret(Ptr{Cvoid}, parse(Int, LLVM.value(fattr)))
                job = Base.unsafe_pointer_to_objref(ptr)[]
            end
        end
    end
    if mi === nothing
        GPUCompiler.@safe_error "Enzyme: Custom handler, could not find mi", orig, LLVM.called_value(orig)
    end
    return mi, job
end

function julia_type_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    inst = LLVM.Instruction(val)
    ctx = LLVM.context(inst)

    mi, job = enzyme_custom_extract_mi(inst)

    ops = collect(operands(inst))
    called = ops[end]
    
    interp = GPUCompiler.get_interpreter(job)
    RT = Core.Compiler.typeinf_ext_toplevel(interp, mi).rettype
    
    sret = is_sret(RT, ctx) 
    returnRoots = false
    if sret
        lRT = eltype(llvmtype(ops[1]))
    	returnRoots = deserves_rooting(lRT)
    end

    jlargs = classify_arguments(mi.specTypes, eltype(llvmtype(called)), sret, returnRoots)

    dl = string(LLVM.datalayout(LLVM.parent(LLVM.parent(LLVM.parent(inst)))))
    
    for arg in jlargs
        if arg.cc == GPUCompiler.GHOST
            continue
        end
   
        op_idx = arg.codegen.i
        rest = typetree(arg.typ, ctx, dl)
        if arg.cc == GPUCompiler.BITS_REF
            merge!(rest, TypeTree(API.DT_Pointer, ctx))
            only!(rest, -1)
        else
            # canonicalize wrt size
        end
        API.EnzymeMergeTypeTree(unsafe_load(args, op_idx), rest)
    end
     
    rtt = typetree(RT, ctx, dl)
    
    if sret
        merge!(rtt, TypeTree(API.DT_Pointer, ctx))
        only!(rtt, -1)
        API.EnzymeMergeTypeTree(unsafe_load(args, 1), rtt)
        if returnRoots
            allpointer = TypeTree(API.DT_Pointer, -1, ctx)
            API.EnzymeMergeTypeTree(unsafe_load(args, 2), allpointer)
        end
    else
        if GPUCompiler.deserves_retbox(RT)
            merge!(rtt, TypeTree(API.DT_Pointer, ctx))
            only!(rtt, -1)
        end
        API.EnzymeMergeTypeTree(ret, rtt)
    end
    
    return UInt8(false)
end

function enzyme!(job, mod, primalf, adjoint, mode, width, parallel, actualRetType, dupClosure, wrap, modifiedBetween, returnPrimal, jlrules)
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
                error("Type of ghost or constant type "*string(T)*" is marked as differentiable.")
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
        "ijl_box_float32" => @cfunction(f32_box_rule,
                                           UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                   Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "jl_box_int64" => @cfunction(i64_box_rule,
                                           UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                   Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "ijl_box_int64" => @cfunction(i64_box_rule,
                                           UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                   Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "jl_box_uint64" => @cfunction(i64_box_rule,
                                            UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                    Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "ijl_box_uint64" => @cfunction(i64_box_rule,
                                            UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                    Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "jl_array_copy" => @cfunction(inout_rule,
                                           UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                   Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "ijl_array_copy" => @cfunction(inout_rule,
                                           UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                   Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "jl_alloc_array_1d" => @cfunction(alloc_rule,
                                            UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                    Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "ijl_alloc_array_1d" => @cfunction(alloc_rule,
                                            UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                    Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "jl_alloc_array_2d" => @cfunction(alloc_rule,
                                            UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                    Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "ijl_alloc_array_2d" => @cfunction(alloc_rule,
                                            UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                    Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef)),
        "jl_alloc_array_3d" => @cfunction(alloc_rule,
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
    for jl in jlrules
        rules[jl] = @cfunction(julia_type_rule,
                                           UInt8, (Cint, API.CTypeTreeRef, Ptr{API.CTypeTreeRef},
                                                   Ptr{API.IntList}, Csize_t, LLVM.API.LLVMValueRef))
    end

    logic = Logic()
    TA = TypeAnalysis(logic, rules)

    retTT = typetree((!isa(actualRetType, Union) && GPUCompiler.deserves_retbox(actualRetType)) ? Ptr{actualRetType} : actualRetType, ctx, dl)

    typeInfo = FnTypeInfo(retTT, args_typeInfo, args_known_values)

    TapeType = Cvoid

    if mode == API.DEM_ReverseModePrimal || mode == API.DEM_ReverseModeGradient
        returnUsed = !(GPUCompiler.isghosttype(actualRetType) || Core.Compiler.isconstType(actualRetType)) 
        shadowReturnUsed = returnUsed && (retType == API.DFT_DUP_ARG || retType == API.DFT_DUP_NONEED)
        returnUsed &= returnPrimal
        augmented = API.EnzymeCreateAugmentedPrimal(
            logic, primalf, retType, args_activity, TA, #=returnUsed=# returnUsed,
            #=shadowReturnUsed=#shadowReturnUsed,
            typeInfo, uncacheable_args, #=forceAnonymousTape=# false, width, #=atomicAdd=# parallel)

        # 2. get new_primalf and tape
        augmented_primalf = LLVM.Function(API.EnzymeExtractFunctionFromAugmentation(augmented))
        tape = API.EnzymeExtractTapeTypeFromAugmentation(augmented)
        utape = API.EnzymeExtractUnderlyingTapeTypeFromAugmentation(augmented)
        if utape != C_NULL
            TapeType = EnzymeTapeToLoad{LLVMType(tape_type(utape))}
            tape = utape
        elseif tape != C_NULL
            TapeType = tape_type(LLVMType(tape))
        else
            TapeType = Cvoid
        end

        if wrap
          augmented_primalf = create_abi_wrapper(augmented_primalf, F, tt, rt, actualRetType, API.DEM_ReverseModePrimal, augmented, dupClosure, width, returnUsed)
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
          adjointf = create_abi_wrapper(adjointf, F, tt, rt, actualRetType, API.DEM_ReverseModeGradient, augmented, dupClosure, width, #=returnPrimal=#false)
        end
    elseif mode == API.DEM_ReverseModeCombined
        returnUsed = !(GPUCompiler.isghosttype(actualRetType) || Core.Compiler.isconstType(actualRetType)) 
        returnUsed &= returnPrimal
        adjointf = LLVM.Function(API.EnzymeCreatePrimalAndGradient(
            logic, primalf, retType, args_activity, TA,
            #=returnValue=#returnUsed, #=dretUsed=#false, #=mode=#API.DEM_ReverseModeCombined, width,
            #=additionalArg=#C_NULL, typeInfo,
            uncacheable_args, #=augmented=#C_NULL, #=atomicAdd=# parallel))
        augmented_primalf = nothing
        if wrap
          adjointf = create_abi_wrapper(adjointf, F, tt, rt, actualRetType, API.DEM_ReverseModeCombined, nothing, dupClosure, width, returnUsed)
        end
    elseif mode == API.DEM_ForwardMode
        returnUsed = !(GPUCompiler.isghosttype(actualRetType) || Core.Compiler.isconstType(actualRetType)) 
        returnUsed &= returnPrimal
        adjointf = LLVM.Function(API.EnzymeCreateForwardDiff(
            logic, primalf, retType, args_activity, TA,
            #=returnValue=#returnUsed, #=mode=#API.DEM_ForwardMode, width,
            #=additionalArg=#C_NULL, typeInfo,
            uncacheable_args))
        augmented_primalf = nothing
        if wrap
          adjointf = create_abi_wrapper(adjointf, F, tt, rt, actualRetType, API.DEM_ForwardMode, nothing, dupClosure, width, returnUsed)
        end
    else
        @assert "Unhandled derivative mode", mode
    end
    API.EnzymeLogicErasePreprocessedFunctions(logic)
    return adjointf, augmented_primalf, TapeType
end

function create_abi_wrapper(enzymefn::LLVM.Function, F, argtypes, rettype, actualRetType, Mode::API.CDerivativeMode, augmented, dupClosure, width, returnPrimal)
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
        push!(T_wrapperargs, LLVM.LLVMType(API.EnzymeGetShadowType(width, convert(LLVMType, actualRetType; ctx))))
    end

    data    = Array{Int64}(undef, 3)
    existed = Array{UInt8}(undef, 3)
    if Mode == API.DEM_ReverseModePrimal
        API.EnzymeExtractReturnInfo(augmented, data, existed)
        # tape -- todo ??? on wrap
        if existed[1] != 0
            tape = API.EnzymeExtractTapeTypeFromAugmentation(augmented)
            push!(T_JuliaSRet, LLVM.LLVMType(tape))
        end
        
        isboxed = GPUCompiler.deserves_argbox(actualRetType)
        llvmT = isboxed ? T_prjlvalue : convert(LLVMType, actualRetType; ctx)
        
        # primal return
        if existed[2] != 0 
            @assert returnPrimal
            push!(T_JuliaSRet, llvmT)
        else
            @assert !returnPrimal
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
            if returnPrimal
                push!(T_JuliaSRet, llvmT)
            end
            if !(rettype <: Const)
                push!(T_JuliaSRet, LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmT)))
            end
        end
    end

    # sret argument
    if !isempty(T_JuliaSRet)
        ST = LLVM.StructType(T_JuliaSRet; ctx)
        pushfirst!(T_wrapperargs, LLVM.PointerType(ST, 10))
    end

    if needs_tape
        tape = API.EnzymeExtractTapeTypeFromAugmentation(augmented)
        utape = API.EnzymeExtractUnderlyingTapeTypeFromAugmentation(augmented)
        if utape != C_NULL
            tape = utape
        end
        if tape != C_NULL
            push!(T_wrapperargs, LLVM.LLVMType(tape))
        else
            needs_tape = false
        end
    end

    FT = LLVM.FunctionType(T_void, T_wrapperargs)
    llvm_f = LLVM.Function(mod, safe_name(LLVM.name(enzymefn)*"wrap"), FT)
    sfn = LLVM.get_subprogram(enzymefn)
    if sfn !== nothing
        LLVM.set_subprogram!(llvm_f, sfn)
    end
    dl = datalayout(mod)

    params = [parameters(llvm_f)...]
    target =  !isempty(T_JuliaSRet) ? 2 : 1

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

                    LLVM.memset!(builder, cst,  LLVM.ConstantInt(LLVM.IntType(8; ctx), 0),
                                                LLVM.ConstantInt(LLVM.IntType(64; ctx), LLVM.storage_size(dl, Base.eltype(LLVM.llvmtype(ptr)) )),
                                                #=align=#0 )
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
                    ptr = gep!(builder, sret, [LLVM.ConstantInt(LLVM.IntType(64; ctx), 0), LLVM.ConstantInt(LLVM.IntType(32; ctx), returnNum)])
                    si = store!(builder, eval, ptr)
                    julia_post_cache_store(si.ref, builder.ref, C_NULL)
                    returnNum+=1
                end
            end
        elseif Mode == API.DEM_ForwardMode
            for returnNum in 0:(length(T_JuliaSRet)-1)
                eval = val
                if length(T_JuliaSRet) > 1
                    eval = extract_value!(builder, val, returnNum)
                end
                ptr = gep!(builder, sret, [LLVM.ConstantInt(LLVM.IntType(64; ctx), 0), LLVM.ConstantInt(LLVM.IntType(32; ctx), returnNum)])
                si = store!(builder, eval, ptr)
                julia_post_cache_store(si.ref, builder.ref, C_NULL)
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
function classify_arguments(source_sig::Type, codegen_ft::LLVM.FunctionType, has_sret, has_returnroots)
    source_types = [source_sig.parameters...]
    codegen_types = parameters(codegen_ft)

    args = []
    codegen_i = has_sret ? 2 : 1
    if has_returnroots
        codegen_i += 1
    end
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

function isSpecialPtr(Ty)
    if !isa(Ty, LLVM.PointerType)
		return false
	end
	AS = LLVM.addrspace(Ty)
    return 10 <= AS && AS <= 13
end

mutable struct CountTrackedPointers
    count::UInt
    all::Bool
    derived::Bool
end

function CountTrackedPointers(T)
	res = CountTrackedPointers(0, true, false)

    if isa(T, LLVM.PointerType)
        if isSpecialPtr(T)
            res.count += 1
            if LLVM.addrspace(T) != #=AddressSpace::Tracked=# 10
                res.derived = true
			end
        end
    elseif isa(T, LLVM.StructType)
        for ElT in elements(T)
            sub = CountTrackedPointers(ElT)
            res.count += sub.count
            res.all &= sub.all
            res.derived |= sub.derived
        end
	elseif isa(T, LLVM.ArrayType)
		sub = CountTrackedPointers(eltype(T))
		res.count += sub.count
		res.all &= sub.all
		res.derived |= sub.derived
		res.count *= length(T)
	elseif isa(T, LLVM.VectorType)
		sub = CountTrackedPointers(eltype(T))
		res.count += sub.count
		res.all &= sub.all
		res.derived |= sub.derived
		res.count *= size(T)
    end
    if res.count == 0
        res.all = false
	end
	return res
end

# must deserve sret
function deserves_rooting(T)
	tracked = CountTrackedPointers(T)
	@assert !tracked.derived
	if tracked.count != 0 && !tracked.all
		return true # tracked.count;
	end
	return false
end

function for_each_uniontype_small(f, ty)
    if ty isa Union
        for_each_uniontype_small(f, ty.a)
        for_each_uniontype_small(f, ty.b)
        return
    end
    if Base.datatype_pointerfree(ty)
        f(ty)
        return
    end
end

# From https://github.com/JuliaLang/julia/blob/038d31463f0ef744c8308bdbe87339b9c3f0b890/src/cgutils.cpp#L3108
function union_alloca_type(UT)
    nbytes = 0
    function inner(jlrettype)
        if !(Base.issingletontype(jlrettype) &&isa(jlrettype, DataType))
           nbytes = max(nbytes, sizeof(jlrettype)) 
        end
    end
    for_each_uniontype_small(inner, UT)
    return nbytes
end

# From https://github.com/JuliaLang/julia/blob/e6bf81f39a202eedc7bd4f310c1ab60b5b86c251/src/codegen.cpp#L6447
function is_sret(jlrettype, ctx)
    if jlrettype === Union{}
        # jlrettype == (jl_value_t*)jl_bottom_type
        return false
    elseif Base.isstructtype(jlrettype) && Base.issingletontype(jlrettype) &&isa(jlrettype, DataType)
        # jl_is_structtype(jlrettype) && jl_is_datatype_singleton((jl_datatype_t*)jlrettype)
        return false
    elseif jlrettype isa Union # jl_is_uniontype(jlrettype)
        if union_alloca_type(jlrettype) > 0
            # sret, also a regular return here
            return true
        end
        return false
    elseif !GPUCompiler.deserves_retbox(jlrettype)
        rt = convert(LLVMType, jlrettype ; ctx)
        if !isa(rt, LLVM.VoidType) && GPUCompiler.deserves_sret(jlrettype, rt)
            return true
        end
    end
    return false
end

# Modified from GPUCompiler/src/irgen.jl:365 lower_byval
function lower_convention(functy::Type, mod::LLVM.Module, entry_f::LLVM.Function, actualRetType::Type)
    ctx = context(mod)
    entry_ft = eltype(llvmtype(entry_f)::LLVM.PointerType)::LLVM.FunctionType

    RT = LLVM.return_type(entry_ft)

    # generate the wrapper function type & definition
    wrapper_types = LLVM.LLVMType[]
    sret = is_sret(actualRetType, ctx)
    
    returnRoots = false
    if sret
        RT = eltype(llvmtype(first(parameters(entry_f))))
    	returnRoots = deserves_rooting(RT)
		if returnRoots
			GPUCompiler.@safe_warn "Returned rooting not fully handled, segfault likely"
		end
    end

	args = classify_arguments(functy, entry_ft, sret, returnRoots)
    filter!(args) do arg
        arg.cc != GPUCompiler.GHOST
    end
    @assert length(args) == length(collect(parameters(entry_f))[1+sret+returnRoots:end]) 
    
	# if returnRoots
	# 	push!(wrapper_types, llvmtype(parameters(entry_f)[1+sret]))
	# end
    for (parm, arg) in zip(collect(parameters(entry_f))[1+sret+returnRoots:end], args)
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
    sfn = LLVM.get_subprogram(entry_f)
    if sfn !== nothing
        LLVM.set_subprogram!(wrapper_f, sfn)
    end

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
			if returnRoots
				push!(nops, ops[1+sret])
			end
            for (parm, arg) in zip(ops[1+sret+returnRoots:end], args)
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
                @safe_show mod
                @safe_show entry_f
                @safe_show e
                throw(AssertionError("Use after deletion"))
            end
            LLVM.API.LLVMInstructionEraseFromParent(e)
        end

        entry = BasicBlock(wrapper_f, "entry"; ctx)
        position!(builder, entry)
        if LLVM.get_subprogram(entry_f) !== nothing
            debuglocation!(builder, DILocation(ctx, 0, 0, LLVM.get_subprogram(entry_f)))
        end

        wrapper_args = Vector{LLVM.Value}()

        if sret
            sretPtr = alloca!(builder, eltype(llvmtype(parameters(entry_f)[1])))
            push!(wrapper_args, sretPtr)
        end
		if returnRoots
            retRootPtr = alloca!(builder, eltype(llvmtype(parameters(entry_f)[1+sret])))
            # retRootPtr = alloca!(builder, parameters(wrapper_f)[1])
			push!(wrapper_args, retRootPtr)
		end

        # perform argument conversions
        for (parm, arg) in zip(collect(parameters(entry_f))[1+sret+returnRoots:end], args)
            wrapparm = parameters(wrapper_f)[arg.codegen.i-sret-returnRoots]
            if !GPUCompiler.deserves_argbox(arg.typ) && arg.cc == GPUCompiler.BITS_REF
                # copy the argument value to a stack slot, and reference it.
                ty = llvmtype(parm)
                if !isa(ty, LLVM.PointerType)
                    @safe_show entry_f, args, parm, ty
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
                    push!(parameter_attributes(wrapper_f, arg.codegen.i-sret-returnRoots), attr)
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

    if LLVM.API.LLVMVerifyFunction(wrapper_f, LLVM.API.LLVMReturnStatusAction) != 0
        @safe_show mod
        @safe_show LLVM.API.LLVMVerifyFunction(wrapper_f, LLVM.API.LLVMPrintMessageAction)
        @safe_show wrapper_f
        flush(stdout)
        throw(LLVM.LLVMException("broken function"))
    end

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
    
    if LLVM.API.LLVMVerifyFunction(wrapper_f, LLVM.API.LLVMReturnStatusAction) != 0
        @safe_show mod
        @safe_show LLVM.API.LLVMVerifyFunction(wrapper_f, LLVM.API.LLVMPrintMessageAction)
        @safe_show wrapper_f
        flush(stdout)
        throw(LLVM.LLVMException("broken function"))
    end
    return wrapper_f, returnRoots
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
    returnPrimal = params.returnPrimal

    if parent_job === nothing
        primal_target = GPUCompiler.NativeCompilerTarget()
        primal_params = Compiler.PrimalCompilerParams()
        primal_job    = CompilerJob(primal_target, primal, primal_params)
    else
        primal_job = similar(parent_job, job.source)
    end
    
    mod, meta = GPUCompiler.codegen(:llvm, primal_job; optimize=false, cleanup=false, validate=false, parent_job=parent_job, ctx)
    inserted_ts = false
    if ctx !== nothing && ctx isa LLVM.Context
        @assert ctx == context(mod)
        ts_ctx = nothing
    else
        ts_ctx = ctx
        ctx = context(mod)
@static if VERSION < v"1.9-"
else
        if !in(ctx, keys(ctxToThreadSafe))
            ctxToThreadSafe[ctx] = ts_ctx
            inserted_ts = true
        end
end
    end

    LLVM.ModulePassManager() do pm
        API.AddPreserveNVVMPass!(pm, #=Begin=#true)
        run!(pm, mod)
    end
    
    primalf = meta.entry
    check_ir(job, mod)
    if API.EnzymeBitcodeReplacement(mod) != 0
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

        for fname in ("cblas_xerbla",)
            if haskey(functions(mod), fname)
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
            if haskey(functions(mod), fname)
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

    custom = Dict{String, LLVM.API.LLVMLinkage}()
    must_wrap = false
    
    foundTys = Dict{String, Tuple{LLVM.FunctionType, Core.MethodInstance}}()

    jobref = Ref(job)

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
        
        foundTys[k_name] = (eltype(llvmtype(llvmfn)), mi)

        Base.isbindingresolved(jlmod, name) && isdefined(jlmod, name) || continue
        func = getfield(jlmod, name)

        sparam_vals = mi.specTypes.parameters[2:end] # mi.sparam_vals
        if func == Base.eps || func == Base.nextfloat || func == Base.prevfloat
            handleCustom("jl_inactive_inout", [StringAttribute("enzyme_inactive"; ctx),
                                      EnumAttribute("readnone", 0; ctx),
                                      EnumAttribute("speculatable", 0; ctx),
                                      StringAttribute("enzyme_shouldrecompute"; ctx)
                                                      ])
            continue
        end
        if func == Base.to_tuple_type
            handleCustom("jl_to_tuple_type",
                   [EnumAttribute("readonly", 0; ctx),
                    EnumAttribute("inaccessiblememonly", 0; ctx),
                    EnumAttribute("speculatable", 0; ctx),
                    StringAttribute("enzyme_shouldrecompute"; ctx),
                    StringAttribute("enzyme_inactive"; ctx),
                                  ])
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
        # Since this is noreturn and it can't write to any operations in the function
        # in a way accessible by the function. Ideally the attributor should actually
        # handle this and similar not impacting the read/write behavior of the calling
        # fn, but it doesn't presently so for now we will ensure this by hand
        if func == Base.Checked.throw_overflowerr_binaryop
            llvmfn = functions(mod)[k.specfunc]
            handleCustom("enz_noop", [StringAttribute("enzyme_inactive"; ctx), EnumAttribute("readonly"; ctx)])
            continue
        end
        if in(func, InactiveFunctions)
            handleCustom("enz_noop", [StringAttribute("enzyme_inactive"; ctx)])
            continue
        end
        if func == Base.enq_work && length(sparam_vals) == 1 && first(sparam_vals) <: Task 
            handleCustom("jl_enq_work")
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
        if func == Enzyme.pmap
            source_sig = Base.signature_type(func, sparam_vals)
            primal = llvmfn == primalf
            llvmfn, _ = lower_convention(source_sig, mod, llvmfn, k.ci.rettype)
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
          llvmfn, _ = lower_convention(source_sig, mod, llvmfn, k.ci.rettype)
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
    primalf, returnRoots = lower_convention(source_sig, mod, primalf, actualRetType)

    if primal_job.target isa GPUCompiler.NativeCompilerTarget
        target_machine = JIT.get_tm()
    else
        target_machine = GPUCompiler.llvm_machine(primal_job.target)
    end

    parallel = Threads.nthreads() > 1
    process_module = false
    device_module = false
    if parent_job !== nothing
        if parent_job.target isa GPUCompiler.PTXCompilerTarget ||
           parent_job.target isa GPUCompiler.GCNCompilerTarget
            parallel = true
            device_module = true
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

    TapeType::Type = Cvoid

    if params.run_enzyme
        # Generate the adjoint
        jlrules = String[]
        for (fname, (ftyp, mi)) in foundTys
            haskey(functions(mod), fname) || continue
            f = functions(mod)[fname]
            if eltype(llvmtype(f)) != ftyp
                continue
            end
            attributes = function_attributes(f)
            push!(attributes, StringAttribute("enzymejl_mi", string(convert(Int, pointer_from_objref(mi))); ctx))
            push!(attributes, StringAttribute("enzymejl_job", string(convert(Int, pointer_from_objref(jobref))); ctx))
            push!(jlrules, fname)
        end

        GC.@preserve jobref begin
            adjointf, augmented_primalf, TapeType = enzyme!(job, mod, primalf, adjoint, mode, width, parallel, actualRetType, dupClosure, abiwrap, modifiedBetween, returnPrimal, jlrules)
        end
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
            if haskey(functions(mod), fname)
                f = functions(mod)[fname]
                LLVM.API.LLVMRemoveEnumAttributeAtIndex(f, reinterpret(LLVM.API.LLVMAttributeIndex, LLVM.API.LLVMAttributeFunctionIndex), kind(EnumAttribute("returns_twice"; ctx)))
            end
        end
    else
        adjointf = primalf
        augmented_primalf = nothing
    end

    if ts_ctx !== nothing
@static if VERSION < v"1.9-"
else
        if inserted_ts
            delete!(ctxToThreadSafe, ctx)
        end
end
    end

    LLVM.ModulePassManager() do pm
        API.AddPreserveNVVMPass!(pm, #=Begin=#false)
        run!(pm, mod)
    end
    API.EnzymeReplaceFunctionImplementation(mod)

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

    if !device_module
        # Don't restore pointers when we are doing GPU compilation
        restore_lookups(mod)
    end
    
    if parent_job !== nothing
        reinsert_gcmarker!(adjointf)
        augmented_primalf !== nothing && reinsert_gcmarker!(augmented_primalf)
        post_optimze!(mod, target_machine, #=machine=#false)
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

    return mod, (;adjointf, augmented_primalf, entry=adjointf, compiled=meta.compiled, TapeType)
end

##
# Box
##

mutable struct Box{T} <: Base.Ref{T}
    x::T
    Box{T}() where {T} = new()
    Box{T}(x) where {T} = new(x)
end

Base.isassigned(x::Box) = isdefined(x, :x)
Base.getindex(b::Box) = b.x
Base.setindex(b::Box, x) = (b.x = x; b)
Base.eltype(::Box{T}) where T = T

function Base.unsafe_convert(::Type{Ptr{Cvoid}}, b::Box{T}) where T
    if Base.allocatedinline(T)
        p = Base.pointer_from_objref(b)
    elseif Base.isconcretetype(T) && ismutabletype(T)
        p = Base.pointer_from_objref(b.x)
    elseif !isassigned(b)
        # TODO: if Box{AbstractInt}() the RefValue equivalent would lead to C_NULL
        #       What is the semantics we want to have?
        throw(Core.UndefRefError())
    else
        # If the slot is not leaf type, it could be either immutable or not.
        # If it is actually an immutable, then we can't take it's pointer directly
        # Instead, explicitly load the pointer from the `RefValue`,
        # which also ensures this returns same pointer as the one rooted in the `RefValue` object.
        p = Base.pointerref(Ptr{Ptr{Cvoid}}(Base.pointer_from_objref(b)), 1, Core.sizeof(Ptr{Cvoid}))
    end
    return p
end

Base.unsafe_convert(::Type{Ptr{T}}, b::Box{T}) where T = Base.unsafe_convert(Ptr{T}, Base.unsafe_convert(Ptr{Cvoid}, b))

##
# Thunk
##

# Compiler result
struct Thunk
    adjoint::Ptr{Cvoid}
    primal::Ptr{Cvoid}
    TapeType::Type
end

@inline (thunk::CombinedAdjointThunk{F, RT, TT, Width, DF, ReturnPrimal})(args...) where {F, Width, DF, RT, TT, ReturnPrimal} =
enzyme_call(Val(false), thunk.adjoint, CombinedAdjointThunk, Width, ReturnPrimal, TT, RT, thunk.fn, thunk.dfn, Cvoid, args...)

@inline (thunk::ForwardModeThunk{F, RT, TT, Width, DF, ReturnPrimal})(args...) where {F, Width, DF, RT, TT, ReturnPrimal} =
enzyme_call(Val(false), thunk.adjoint, ForwardModeThunk, Width, ReturnPrimal, TT, RT, thunk.fn, thunk.dfn, Cvoid, args...)

@inline (thunk::AdjointThunk{F, RT, TT, Width, DF, TapeT})(args...) where {F, Width, DF, RT, TT, TapeT} =
enzyme_call(Val(false), thunk.adjoint, AdjointThunk, Width, #=ReturnPrimal=#Val(false), TT, RT, thunk.fn, thunk.dfn, TapeT, args...)
@inline raw_enzyme_call(thunk::AdjointThunk{F, RT, TT, Width, DF, TapeT}, args...) where {F, Width, DF, RT, TT, TapeT} =
enzyme_call(Val(true), thunk.adjoint, AdjointThunk, Width, #=ReturnPrimal=#Val(false), TT, RT, thunk.fn, thunk.dfn, TapeT, args...)

@inline (thunk::AugmentedForwardThunk{F, RT, TT, Width, DF, ReturnPrimal, TapeT})(args...) where {F, Width, DF, RT, TT, ReturnPrimal, TapeT} =
enzyme_call(Val(false), thunk.primal, AugmentedForwardThunk, Width, ReturnPrimal, TT, RT, thunk.fn, thunk.dfn, TapeT, args...)
@inline raw_enzyme_call(thunk::AugmentedForwardThunk{F, RT, TT, Width, DF, ReturnPrimal, TapeT}, args...) where {F, Width, DF, RT, TT, ReturnPrimal, TapeT} =
enzyme_call(Val(true), thunk.primal, AugmentedForwardThunk, Width, ReturnPrimal, TT, RT, thunk.fn, thunk.dfn, TapeT, args...)


function jl_set_typeof(v::Ptr{Cvoid}, T)
    tag = reinterpret(Ptr{Any}, reinterpret(UInt, v) - 8)
    Base.unsafe_store!(tag, T) # set tag
    return nothing
end

@generated function enzyme_call(::Val{RawCall}, fptr::Ptr{Cvoid}, ::Type{CC}, ::Type{Val{width}}, ::Val{returnPrimal}, tt::Type{T},
        rt::Type{RT}, f::F, df::DF, ::Type{TapeType}, args::Vararg{Any, N}) where {RawCall, F, T, RT, DF, TapeType, N, CC, width, returnPrimal}

    is_forward = CC <: AugmentedForwardThunk || CC <: ForwardModeThunk
    is_adjoint = CC <: AdjointThunk || CC <: CombinedAdjointThunk
    is_split   = CC <: AdjointThunk || CC <: AugmentedForwardThunk
    needs_tape = CC <: AdjointThunk

    argtt    = tt.parameters[1]
    rettype  = rt.parameters[1]
    argtypes = DataType[argtt.parameters...]
    argexprs = Union{Expr, Symbol}[:(args[$i]) for i in 1:N]

    if !RawCall
        if rettype <: Active
            @assert length(argtypes) + is_adjoint + needs_tape == length(argexprs)
        elseif rettype <: Const
            @assert length(argtypes)              + needs_tape == length(argexprs)
        else
            @assert length(argtypes)              + needs_tape == length(argexprs)
        end
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

    i = 1
    for T in argtypes
        source_typ = eltype(T)
        
		expr = argexprs[i]
        i+=1
        if GPUCompiler.isghosttype(source_typ) || Core.Compiler.isconstType(source_typ)
            @assert T <: Const
            continue
        end

        isboxed = GPUCompiler.deserves_argbox(source_typ)

        argexpr = if RawCall
            expr
        else
            Expr(:., expr, QuoteNode(:val))
        end

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
            if RawCall
                argexpr = argexprs[i]
                i+=1
            else
                argexpr = Expr(:., expr, QuoteNode(:dval))
            end
            if isboxed
                push!(types, Any)
            else
                push!(types, source_typ)
            end
            push!(ccexprs, argexpr)
        elseif T <: BatchDuplicated || T <: BatchDuplicatedNoNeed
            if RawCall
                argexpr = argexprs[i]
                i+=1
            else
                argexpr = Expr(:., expr, QuoteNode(:dval))
            end
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
        if width == 1
            push!(types, eltype(rettype))
        else
            push!(types, NTuple{width, eltype(rettype)})
        end
        push!(ccexprs, argexprs[i])
        i+=1
    end

    if needs_tape
        if !(GPUCompiler.isghosttype(TapeType) || Core.Compiler.isconstType(TapeType))
            push!(types, TapeType)
            push!(ccexprs, argexprs[i])
        end
        i+=1
    end

    @assert i == length(argexprs)+1

    # Tape
    if CC <: AugmentedForwardThunk 
        push!(sret_types, TapeType)
    end

    jlRT = eltype(rettype)
    if typeof(jlRT) == UnionAll
      # Future improvement, add tye assertion on load
      jlRT = DataType
    end
    if returnPrimal
        push!(sret_types, jlRT)
    end
    if is_forward
        if rettype <: Duplicated || rettype <: DuplicatedNoNeed
            push!(sret_types, jlRT)
        elseif rettype <: BatchDuplicated || rettype <: BatchDuplicatedNoNeed
            push!(sret_types, AnonymousStruct(NTuple{width, jlRT}))
        end
    end

	# calls fptr
	ctx = LLVM.Context()
    llvmtys = LLVMType[convert(LLVMType, x; ctx, allow_boxed=true) for x in types]
    llsret_types = LLVMType[]
    if !isempty(sret_types)
      for x in sret_types
          if !(GPUCompiler.isghosttype(x) || Core.Compiler.isconstType(x))
            push!(llsret_types, convert(LLVMType, x; ctx, allow_boxed=true))
          end
      end
    end
    if !isempty(llsret_types)
      T_sjoint = LLVM.StructType(llsret_types; ctx)
      pushfirst!(llvmtys, convert(LLVMType, Ptr{Cvoid}; allow_boxed=true, ctx))
      # pushfirst!(llvmtys, convert(LLVMType, Any; allow_boxed=true, ctx))
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
        callparams = collect(params)
        if needs_tape && !(GPUCompiler.isghosttype(TapeType) || Core.Compiler.isconstType(TapeType))
            tape = callparams[end]
            if TapeType <: EnzymeTapeToLoad
                llty = from_tape_type(eltype(TapeType), ctx)
                tape = bitcast!(builder, LLVM.PointerType(llty, LLVM.addrspace(llvmtype(tape))))
                tape = load!(builder, tape)
                API.SetMustCache!(tape)
                callparams[end] = tape
            else
                llty = from_tape_type(TapeType, ctx)
                @assert llvmtype(tape) == llty
            end
        end
		lfn = inttoptr!(builder, lfn, LLVM.PointerType(LLVM.FunctionType(T_void, [llvmtype(x) for x in callparams])))
		call!(builder, lfn, callparams)
        ret!(builder)
	end

	ir = string(mod)
	fn = LLVM.name(llvm_f)

    @assert length(types) == length(ccexprs)
    if !isempty(llsret_types)
        return quote
            Base.@_inline_meta
            sret = Box{$(AnonymousStruct(Tuple{sret_types...}))}()
            GC.@preserve sret begin
                sret_p = Base.unsafe_convert(Ptr{Cvoid}, sret)
                Base.llvmcall(($ir, $fn), Cvoid,
                  Tuple{Ptr{Cvoid}, Ptr{Cvoid}, $(types...)},
                fptr, sret_p, $(ccexprs...))
            end
            # TODO: https://github.com/EnzymeAD/Enzyme.jl/issues/347
            # We might find undefined fields here and when we try to load from them
            # Julia will throw an exception.
            values(sret[])
        end
    else
        return quote
            Base.@_inline_meta
            sret = Box{$(AnonymousStruct(Tuple{sret_types...}))}()
            Base.llvmcall(($ir, $fn), Cvoid,
                Tuple{Ptr{Cvoid}, $(types...),},
                fptr, $(ccexprs...))
            return values(sret[])
        end
    end
end

##
# JIT
##

function _link(job, (mod, adjoint_name, primal_name, ctx, TapeType))
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

    return Thunk(adjoint_ptr, primal_ptr, TapeType)
end

# actual compilation
function _thunk(job, ctx=nothing)
    params = job.params

    # TODO: on 1.9, this actually creates a context. cache those.
    if ctx === nothing
        ctx = JuliaContext()
    end
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
    return (mod, adjoint_name, primal_name, ctx, meta.TapeType)
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


@generated function genthunk(::Type{F}, f::Fn, df::DF, ::Type{A}, tt::Type{TT},::Val{Mode}, ::Val{ModifiedBetween}, ::Val{width}, ::Val{specid}, ::Val{ReturnPrimal}) where {F, Fn, DF, A<:Annotation, TT, Mode, ModifiedBetween, width, specid, ReturnPrimal}
    primal, adjoint = fspec(F, TT)

    target = Compiler.EnzymeTarget()
    params = Compiler.EnzymeCompilerParams(adjoint, Mode, width, A, true, DF != Nothing, #=abiwrap=#true, ModifiedBetween, ReturnPrimal)
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


    thunk = cached_compilation(job, hash(hash(hash(hash(adjoint, hash(rt, UInt64(Mode))), UInt64(width)), UInt64(ModifiedBetween)), UInt64(ReturnPrimal)), specid)::Thunk
    if Mode == API.DEM_ReverseModePrimal || Mode == API.DEM_ReverseModeGradient
        TapeType = thunk.TapeType
        AugT = AugmentedForwardThunk{F, rt, adjoint.tt, Val{width} , DF, Val(ReturnPrimal), TapeType}
        AdjT = AdjointThunk{F, rt, adjoint.tt, Val{width}, DF, TapeType}
        return quote
            augmented = $AugT(f, $(thunk.primal), df)
            adjoint  = $AdjT(f, $(thunk.adjoint), df)
            (augmented, adjoint)
        end
    elseif Mode == API.DEM_ReverseModeCombined
        CAdjT = CombinedAdjointThunk{F, rt, adjoint.tt, Val{width}, DF, Val(ReturnPrimal)}
        return quote
            $CAdjT(f, $(thunk.adjoint), df)
        end
    elseif Mode == API.DEM_ForwardMode
        FMT = ForwardModeThunk{F, rt, adjoint.tt, Val{width}, DF, Val(ReturnPrimal)}
        return quote
            $FMT(f, $(thunk.adjoint), df)
        end
    else
        @assert false
    end
end

@inline function thunk(f::F,df::DF, ::Type{A}, tt::Type{TT},::Val{Mode}, ::Val{width}, ::Val{ModifiedBetween}=Val(Mode != API.DEM_ReverseModeCombined), ::Val{ReturnPrimal}=Val(false)) where {F, DF, A<:Annotation, TT, Mode, width, ModifiedBetween, ReturnPrimal}
    primal, adjoint = fspec(Core.Typeof(f), TT)
    target = Compiler.EnzymeTarget()
    params = Compiler.EnzymeCompilerParams(adjoint, Mode, width, A, true, DF != Nothing, #=abiwrap=#true, ModifiedBetween, ReturnPrimal)
    job    = Compiler.CompilerJob(target, primal, params)

    specid = GPUCompiler.specialization_id(job)
    
    genthunk(Core.Typeof(f), f, df, A, TT, Val(Mode), Val(ModifiedBetween), Val(width), Val(specid), Val(ReturnPrimal))
end

import GPUCompiler: deferred_codegen_jobs

@generated function gendeferred_codegen(::Type{F}, ::Val{tt}, ::Val{rt}, ::Val{DupClosure},::Val{Mode},
                                     ::Val{width}, ::Val{ModifiedBetween}, ::Val{ReturnPrimal}) where {F,tt, rt, DupClosure, Mode, width, ModifiedBetween, ReturnPrimal}
    primal, adjoint = fspec(F, tt)
    target = EnzymeTarget()
    params = EnzymeCompilerParams(adjoint, Mode, width, rt, true, DupClosure, #=abiwrap=#true, ModifiedBetween, ReturnPrimal)
    job    = CompilerJob(target, primal, params)
    
    addr = get_trampoline(job)
    id = Base.reinterpret(Int, pointer(addr))

    deferred_codegen_jobs[id] = job
    trampoline = reinterpret(Ptr{Cvoid}, id)
    quote
        ccall("extern deferred_codegen", llvmcall, Ptr{Cvoid}, (Ptr{Cvoid},), $trampoline)
    end
end

@inline function deferred_codegen(f::F, ::Val{tt}, ::Val{rt}, ::Val{DupClosure},::Val{Mode},
                                     ::Val{width}, ::Val{ModifiedBetween}=Val(Mode != API.DEM_ReverseModeCombined), ::Val{ReturnPrimal}=Val(false)) where {F,tt, rt, DupClosure, Mode, width, ModifiedBetween, ReturnPrimal}
    gendeferred_codegen(Core.Typeof(f), Val(tt), Val(rt), Val(DupClosure), Val(Mode), Val(width), Val(ModifiedBetween), Val(ReturnPrimal))
end

include("compiler/reflection.jl")

end
