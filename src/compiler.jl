module Compiler

import ..Enzyme
import Enzyme: Const, Active, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed,
               Annotation, guess_activity, eltype, 
               API, TypeTree, typetree, only!, shift!, data0!, merge!,
               TypeAnalysis, FnTypeInfo, Logic, allocatedinline, ismutabletype
using Enzyme

import EnzymeCore: EnzymeRules

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

unsafe_to_pointer(ptr) = ccall(Base.@cfunction(x->x, Ptr{Cvoid}, (Ptr{Cvoid},)), Ptr{Cvoid}, (Any,), ptr)
    
# Julia function to LLVM stem and arity
const known_ops = Dict(
    Base.cbrt => (:cbrt, 1),
    Base.rem2pi => (:jl_rem2pi, 2),
    Base.sqrt => (:sqrt, 1),
    Base.sin => (:sin, 1),
    Base.sinc => (:sincn, 1),
    Base.sincos => (:__fd_sincos_1, 1),
    Base.:^ => (:pow, 2),
    Base.rem => (:fmod, 2),
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
    Base.acos => (:acos, 1),
    Base.atan => (:atan, 1),
    Base.atan => (:atan2, 2),
    Base.sinh => (:sinh, 1),
    Base.FastMath.sinh_fast => (:sinh, 1),
    Base.cosh => (:cosh, 1),
    Base.FastMath.cosh_fast => (:cosh, 1),
    Base.tanh => (:tanh, 1),
    Base.ldexp => (:ldexp, 2),
    Base.FastMath.tanh_fast => (:tanh, 1)
)

const nofreefns = Set{String}((
    "jl_gc_queue_root", "gpu_report_exception", "gpu_signal_exception",
    "julia.ptls_states", "julia.write_barrier", "julia.typeof",
    "jl_box_int64", "jl_box_int32",
    "ijl_box_int64", "ijl_box_int32",
    "jl_subtype", "julia.get_pgcstack", "jl_in_threaded_region",
    "jl_object_id_", "jl_object_id", "ijl_object_id_", "ijl_object_id",
    "jl_breakpoint",
    "llvm.julia.gc_preserve_begin","llvm.julia.gc_preserve_end", 
    "jl_get_ptls_states",
    "ijl_get_ptls_states",
    "jl_f_fieldtype",
    "jl_symbol_n",
    "jl_stored_inline", "ijl_stored_inline",
    "jl_f_apply_type", "jl_f_issubtype", 
    "jl_isa", "ijl_isa",
    "jl_matching_methods", "ijl_matching_methods",
    "jl_excstack_state", "ijl_excstack_state", 
    "jl_current_exception", "ijl_current_exception",
    "memhash_seed",
    "jl_f__typevar", "ijl_f__typevar",
    "jl_f_isa", "ijl_f_isa",
    "jl_set_task_threadpoolid", "ijl_set_task_threadpoolid",
    "jl_types_equal", "ijl_types_equal",
    "jl_invoke", "ijl_invoke",
    "jl_apply_generic", "ijl_apply_generic",
    "jl_egal__unboxed", "julia.pointer_from_objref", "_platform_memcmp",
    "memcmp",
    "julia.except_enter",
    "jl_array_grow_end",
    "ijl_array_grow_end",
    "jl_f_getfield",
    "ijl_f_getfield",
    "jl_pop_handler",
    "ijl_pop_handler",
    "jl_string_to_array",
    "ijl_string_to_array",
    "jl_alloc_string",
    "ijl_alloc_string",
    "getenv",
    "jl_cstr_to_string",
    "ijl_cstr_to_string",
    "jl_symbol_n",
    "ijl_symbol_n",
    "uv_os_homedir",
    "jl_array_to_string",
    "ijl_array_to_string",
    "pcre2_jit_compile_8"
))

const inactivefns = Set{String}((
    "jl_gc_queue_root", "gpu_report_exception", "gpu_signal_exception",
    "julia.ptls_states", "julia.write_barrier", "julia.typeof",
    "jl_box_int64", "jl_box_int32",
    "ijl_box_int64", "ijl_box_int32",
    "jl_subtype", "julia.get_pgcstack", "jl_in_threaded_region",
    "jl_object_id_", "jl_object_id", "ijl_object_id_", "ijl_object_id",
    "jl_breakpoint",
    "llvm.julia.gc_preserve_begin","llvm.julia.gc_preserve_end",
    "jl_get_ptls_states",
    "ijl_get_ptls_states",
    "jl_f_fieldtype",
    "jl_symbol_n",
    "jl_stored_inline", "ijl_stored_inline",
    "jl_f_apply_type", "jl_f_issubtype", 
    "jl_isa", "ijl_isa",
    "jl_matching_methods", "ijl_matching_methods",
    "jl_excstack_state", "ijl_excstack_state", 
    "jl_current_exception", "ijl_current_exception",
    "memhash_seed",
    "jl_f__typevar", "ijl_f__typevar",
    "jl_f_isa", "ijl_f_isa",
    "jl_set_task_threadpoolid", "ijl_set_task_threadpoolid",
    "jl_types_equal", "ijl_types_equal",
    "jl_string_to_array",
    "ijl_string_to_array",
    "jl_alloc_string",
    "ijl_alloc_string",
    "getenv",
    "jl_cstr_to_string",
    "ijl_cstr_to_string",
    "jl_symbol_n",
    "ijl_symbol_n",
    "uv_os_homedir",
    "jl_array_to_string",
    "ijl_array_to_string",
    "pcre2_jit_compile_8"
    # "jl_"
))

const activefns = Set{String}((
    "jl_",
))


Enzyme.guess_activity(::Type{T}, mode::Enzyme.Mode) where T = guess_activity(T, convert(API.CDerivativeMode, mode))

@inline function Enzyme.guess_activity(::Type{T}, Mode::API.CDerivativeMode) where {T}
    if T isa Union
        if !(guess_activity(T.a, Mode) <: Const) || !(guess_activity(T.b, Mode) <: Const)
            if Mode == API.DEM_ForwardMode
                return DuplicatedNoNeed{T}
            else
                return Duplicated{T}
            end
        end
    end
    if isghostty(T) || Core.Compiler.isconstType(T) || T === DataType
        return Const{T}
    end
    if Mode == API.DEM_ForwardMode
        return DuplicatedNoNeed{T}
    else
        return Duplicated{T}
    end
end

@inline function Enzyme.guess_activity(::Type{Union{}}, Mode::API.CDerivativeMode)
    return Const{Union{}}
end

@inline function Enzyme.guess_activity(::Type{T}, Mode::API.CDerivativeMode) where {T<:Integer}
    return Const{T}
end

@inline function Enzyme.guess_activity(::Type{T}, Mode::API.CDerivativeMode) where {T<:AbstractFloat}
    if Mode == API.DEM_ForwardMode
        return DuplicatedNoNeed{T}
    else
        return Active{T}
    end
end
@inline function Enzyme.guess_activity(::Type{T}, Mode::API.CDerivativeMode) where {T<:Complex{<:AbstractFloat}}
    if Mode == API.DEM_ForwardMode
        return DuplicatedNoNeed{T}
    else
        return Active{T}
    end
end

@inline function Enzyme.guess_activity(::Type{T}, Mode::API.CDerivativeMode) where {T<:AbstractArray}
    if Mode == API.DEM_ForwardMode
        return DuplicatedNoNeed{T}
    else
        return Duplicated{T}
    end
end

@inline function Enzyme.guess_activity(::Type{Real}, Mode::API.CDerivativeMode)
    if Mode == API.DEM_ForwardMode
        return DuplicatedNoNeed{Any}
    else
        return Duplicated{Any}
    end
end
@inline function Enzyme.guess_activity(::Type{Any}, Mode::API.CDerivativeMode)
    if Mode == API.DEM_ForwardMode
        return DuplicatedNoNeed{Any}
    else
        return Duplicated{Any}
    end
end

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

import GPUCompiler: @safe_debug, @safe_info, @safe_warn, @safe_error

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
    Tracked = 10
    T_jlvalue = LLVM.StructType(LLVM.LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_ppjlvalue = LLVM.PointerType(LLVM.PointerType(T_jlvalue))
    T_size_t = convert(LLVM.LLVMType, Int; ctx)
 
    @static if VERSION < v"1.8.0"
        T_int8 = LLVM.Int8Type(ctx)
        T_pint8 = LLVM.PointerType(T_int8)
        LLVM.FunctionType(T_prjlvalue, [T_pint8, T_size_t, T_prjlvalue])
    else
        LLVM.FunctionType(T_prjlvalue, [T_ppjlvalue, T_size_t, T_prjlvalue])
    end
end
function emit_allocobj!(B, tag::LLVM.Value, Size::LLVM.Value, needs_workaround)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
	ctx = context(mod)
	 
	T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_ppjlvalue = LLVM.PointerType(LLVM.PointerType(T_jlvalue))

    T_int8 = LLVM.Int8Type(ctx)
    T_pint8 = LLVM.PointerType(T_int8)
 
    @static if VERSION < v"1.7.0"
        ptls = reinsert_gcmarker!(fn, B)
        ptls = bitcast!(B, ptls, T_pint8)
    else
        pgcstack = reinsert_gcmarker!(fn, B)
        ct = inbounds_gep!(B, 
            bitcast!(B, pgcstack, T_ppjlvalue),
            [LLVM.ConstantInt(current_task_offset(); ctx)])
        ptls_field = inbounds_gep!(B, 
            ct, [LLVM.ConstantInt(current_ptls_offset(); ctx)])
        T_ppint8 = LLVM.PointerType(T_pint8)
        ptls = load!(B, bitcast!(B, ptls_field, T_ppint8))
    end

    if needs_workaround
        T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)
        T_size_t = convert(LLVM.LLVMType, Int; ctx)
        # This doesn't allow for optimizations
        alloc_obj = get_function!(mod, "jl_gc_alloc_typed",
            LLVM.FunctionType(T_prjlvalue,
                [T_pint8, T_size_t, T_prjlvalue]))
        return call!(B, alloc_obj, [ptls, Size, tag])
    end


    alloc_obj = declare_allocobj!(mod)

    @static if VERSION < v"1.8.0"
        return call!(B, alloc_obj, [ptls, Size, tag])
    else
        return call!(B, alloc_obj, [ct, Size, tag])
    end
end
function emit_allocobj!(B, T::DataType)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
	ctx = context(mod)
    
	T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue_UT = LLVM.PointerType(T_jlvalue)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)
    
    # Obtain tag
    tag = LLVM.ConstantInt(reinterpret(Int, Base.pointer_from_objref(T)); ctx)  # do we need to root ETT
    tag = LLVM.const_inttoptr(tag, T_prjlvalue_UT)
    tag = LLVM.const_addrspacecast(tag, T_prjlvalue)
    
    T_size_t = convert(LLVM.LLVMType, Int; ctx)
    Size = LLVM.ConstantInt(T_size_t, sizeof(T))
    emit_allocobj!(B, tag, Size, #=needs_workaround=#false)
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
@static if VERSION < v"1.8.0"
declare_apply_generic!(mod) = get_function!(mod, "jl_apply_generic") do ctx
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, 10) 
    LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, LLVM.PointerType(T_prjlvalue), LLVM.Int32Type(ctx)])
end
else
declare_apply_generic!(mod) = get_function!(mod, "ijl_apply_generic") do ctx
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, 10) 
    LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, LLVM.PointerType(T_prjlvalue), LLVM.Int32Type(ctx)])
end
end
declare_juliacall!(mod) = get_function!(mod, "julia.call") do ctx
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, 10) 
    LLVM.FunctionType(T_prjlvalue, [T_prjlvalue]; vararg=true)
end

function emit_jl!(B, val)::LLVM.Value 
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    ctx = LLVM.context(mod)
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, 10) 
    FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue])
    fn = get_function!(mod, "jl_", FT)
    call!(B, fn, [val])
end

function emit_box_int64!(B, val)::LLVM.Value  
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    ctx = LLVM.context(mod)
    
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, 10) 
    T_int64 = LLVM.Int64Type(ctx)

    @static if VERSION < v"1.8-"
        box_int64 = get_function!(mod, "jl_box_int64", LLVM.FunctionType(T_prjlvalue, [T_int64]))
    else
        box_int64 = get_function!(mod, "ijl_box_int64", LLVM.FunctionType(T_prjlvalue, [T_int64]))
    end
    call!(B, box_int64, [val])
end

function emit_apply_generic!(B, args)::LLVM.Value     
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    ctx = LLVM.context(mod)
    
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, 10) 
    T_pprjlvalue = LLVM.PointerType(T_prjlvalue)
    T_int32 = LLVM.Int32Type(ctx)

    gen_FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_pprjlvalue, T_int32]) 
    @static if VERSION < v"1.8-"
        inv = get_function!(mod, "jl_apply_generic", gen_FT)
    else
        inv = get_function!(mod, "ijl_apply_generic", gen_FT)
    end

    @static if VERSION < v"1.9.0-"
        FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_prjlvalue]; vararg=true)
        inv = bitcast!(B, inv, LLVM.PointerType(FT))
        # call cc37 nonnull {}* bitcast ({}* ({}*, {}**, i32)* @jl_f_apply_type to {}* ({}*, {}*, {}*, {}*)*)({}* null, {}* inttoptr (i64 140150176657296 to {}*), {}* %4, {}* inttoptr (i64 140149987564368 to {}*))
        res = call!(B, inv, args)
        LLVM.callconv!(res, 37)
    else
        # %5 = call nonnull {}* ({}* ({}*, {}**, i32)*, {}*, ...) @julia.call({}* ({}*, {}**, i32)* @jl_f_apply_type, {}* null, {}* inttoptr (i64 139640605802128 to {}*), {}* %4, {}* inttoptr (i64 139640590432896 to {}*))
        julia_call = get_function!(mod, "julia.call",
            LLVM.FunctionType(T_prjlvalue, 
                              [LLVM.PointerType(gen_FT), T_prjlvalue]; vararg=true))
        res = call!(B, julia_call, [inv, args...])
    end
    return res
end

function emit_invoke!(B, args)::LLVM.Value     
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    ctx = LLVM.context(mod)
    
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, 10) 
    T_pprjlvalue = LLVM.PointerType(T_prjlvalue)
    T_int32 = LLVM.Int32Type(ctx)

    # {} addrspace(10)* ({} addrspace(10)*, {} addrspace(10)**, i32, {} addrspace(10)*)* @ijl_invoke
    gen_FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_pprjlvalue, T_int32, T_prjlvalue]) 
    @static if VERSION < v"1.8-"
        inv = get_function!(mod, "jl_invoke", gen_FT)
    else
        inv = get_function!(mod, "ijl_invoke", gen_FT)
    end

    @static if VERSION < v"1.9.0-"
        FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_prjlvalue]; vararg=true)
        inv = bitcast!(B, inv, LLVM.PointerType(FT))
        # call cc37 nonnull {}* bitcast ({}* ({}*, {}**, i32)* @jl_f_apply_type to {}* ({}*, {}*, {}*, {}*)*)({}* null, {}* inttoptr (i64 140150176657296 to {}*), {}* %4, {}* inttoptr (i64 140149987564368 to {}*))
        res = call!(B, inv, args)
        LLVM.callconv!(res, 38)
    else
        # %5 = call nonnull {}* ({}* ({}*, {}**, i32)*, {}*, ...) @julia.call({}* ({}*, {}**, i32)* @jl_f_apply_type, {}* null, {}* inttoptr (i64 139640605802128 to {}*), {}* %4, {}* inttoptr (i64 139640590432896 to {}*))
        julia_call = get_function!(mod, "julia.call2",
            LLVM.FunctionType(T_prjlvalue, 
                              [LLVM.PointerType(generic_FT), T_prjlvalue]; vararg=true))
        res = call!(B, julia_call, [inv, args...])
    end
    return res
end

function emit_svec!(B, args)::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    ctx = LLVM.context(mod)
    
    fn = get_function!(mod, "jl_svec") do ctx
        sz = convert(LLVMType, Csize_t; ctx)
        T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
        T_prjlvalue = LLVM.PointerType(T_jlvalue, 10) 
        LLVM.FunctionType(T_prjlvalue, [sz]; vararg=true)
    end
    sz = convert(LLVMType, Csize_t; ctx)
    call!(B, fn, [LLVM.ConstantInt(sz, length(args)), args...])
end


function emit_apply_type!(B, Ty, args)::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    ctx = LLVM.context(mod)

    legal = true
    found = []
    for arg in args
        if !isa(arg, ConstantExpr)
            legal = false
            break
        end

        ce = arg
        while isa(ce, ConstantExpr)
            ce = operands(ce)[1]
        end
        if !isa(ce, LLVM.ConstantInt)
            legal = false
            break
        end
        ptr = reinterpret(Ptr{Cvoid}, convert(UInt64, ce))
        typ = Base.unsafe_pointer_to_objref(ptr)
        push!(found, typ, ctx)
    end

    if legal
        return unsafe_to_llvm(Ty{found...}, ctx)
    end
    
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, 10) 
    T_pprjlvalue = LLVM.PointerType(T_prjlvalue)
    T_int32 = LLVM.Int32Type(ctx)
    
    generic_FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_pprjlvalue, T_int32])
    f_apply_type = get_function!(mod, "jl_f_apply_type", generic_FT)
    Ty = unsafe_to_llvm(Ty, ctx)

    @static if VERSION < v"1.9.0-"
        FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_prjlvalue]; vararg=true)
        f_apply_type = bitcast!(B, f_apply_type, LLVM.PointerType(FT))
        # call cc37 nonnull {}* bitcast ({}* ({}*, {}**, i32)* @jl_f_apply_type to {}* ({}*, {}*, {}*, {}*)*)({}* null, {}* inttoptr (i64 140150176657296 to {}*), {}* %4, {}* inttoptr (i64 140149987564368 to {}*))
        tag = call!(B, f_apply_type, [LLVM.PointerNull(T_prjlvalue), Ty, args...])
        LLVM.callconv!(tag, 37)
    else
        # %5 = call nonnull {}* ({}* ({}*, {}**, i32)*, {}*, ...) @julia.call({}* ({}*, {}**, i32)* @jl_f_apply_type, {}* null, {}* inttoptr (i64 139640605802128 to {}*), {}* %4, {}* inttoptr (i64 139640590432896 to {}*))
        julia_call = get_function!(mod, "julia.call",
            LLVM.FunctionType(T_prjlvalue, 
                              [LLVM.PointerType(generic_FT), T_prjlvalue]; vararg=true))
        tag = call!(B, julia_call, [f_apply_type, LLVM.PointerNull(T_prjlvalue), Ty, args...])
    end
    return tag
end

function emit_jltypeof!(B, arg)::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    ctx = LLVM.context(mod)

    if isa(arg, ConstantExpr)
        ce = arg
        while isa(ce, ConstantExpr)
            ce = operands(ce)[1]
        end
        if isa(ce, LLVM.ConstantInt)
            ptr = reinterpret(Ptr{Cvoid}, convert(UInt64, ce))
            typ = Base.unsafe_pointer_to_objref(ptr)
            return unsafe_to_llvm(Core.Typeof(typ), ctx)
        end
    end

    fn = get_function!(mod, "jl_typeof") do ctx
        T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
        T_prjlvalue = LLVM.PointerType(T_jlvalue, 10) 
        LLVM.FunctionType(T_prjlvalue, [T_prjlvalue]; vararg=true)
    end
    call!(B, fn, [arg])
end

function emit_methodinstance!(B, func, args)::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    ctx = LLVM.context(mod)

    world = enzyme_extract_world(fn)

    sizeT = convert(LLVMType, Csize_t; ctx)
    psizeT = LLVM.PointerType(sizeT)

    primalvaltys = LLVM.Value[unsafe_to_llvm(Core.Typeof(func), ctx)]
    for a in args
        push!(primalvaltys, emit_jltypeof!(B, a))
    end
   
    meth = only(methods(func))
    tag = emit_apply_type!(B, Tuple, primalvaltys)

#    TT = meth.sig
#    while TT isa UnionAll
#        TT = TT.body
#    end
#    parms = TT.parameters
#    
#    tosv = primalvaltys
#    if length(parms) > 0 && typeof(parms[end]) == Core.TypeofVararg
#        tosv = LLVM.Value[tosv[1:length(parms)-1]..., emit_apply_type!(B, Tuple, tosv[length(parms):end])]
#    end
#    sv = emit_svec!(B, tosv[2:end])
#
    
    meth = unsafe_to_llvm(meth, ctx)

    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, 10) 
    @static if VERSION < v"1.8.0-"
    worlds = get_function!(mod, "jl_gf_invoke_lookup_worlds", 
        LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, sizeT, psizeT, psizeT]))
    else
    worlds = get_function!(mod, "jl_gf_invoke_lookup_worlds", 
        LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_prjlvalue, sizeT, psizeT, psizeT]))
    end
    EB = LLVM.Builder(ctx)
    position!(EB, first(LLVM.instructions(LLVM.entry(fn))))
    minworld = alloca!(EB, sizeT)
    maxworld = alloca!(EB, sizeT)
    store!(B, LLVM.ConstantInt(sizeT, 0), minworld)
    store!(B, LLVM.ConstantInt(sizeT, -1), maxworld)
    @static if VERSION < v"1.8.0-"
    methodmatch = call!(B, worlds, LLVM.Value[tag, LLVM.ConstantInt(sizeT, world), minworld, maxworld])
    else
    methodmatch = call!(B, worlds, LLVM.Value[tag, unsafe_to_llvm(nothing, ctx), LLVM.ConstantInt(sizeT, world), minworld, maxworld])
    end
    # emit_jl!(B, methodmatch)
    # emit_jl!(B, emit_jltypeof!(B, methodmatch))
    offset = 1
    methodmatch = bitcast!(B, methodmatch, LLVM.PointerType(LLVM.ArrayType(T_prjlvalue, offset+1), Tracked))
    gep = LLVM.inbounds_gep!(B, methodmatch, LLVM.Value[LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(offset; ctx)])
    sv = LLVM.load!(B, gep)
   
    fn = get_function!(mod, "jl_specializations_get_linfo",
                       LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_prjlvalue, T_prjlvalue]))

    mi = call!(B, fn, [meth, tag, sv])

    return mi
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
function array_shadow_handler(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, numArgs::Csize_t, Args::Ptr{LLVM.API.LLVMValueRef}, gutils::API.EnzymeGradientUtilsRef)::LLVM.API.LLVMValueRef
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
    valTys = API.CValueType[]
    for i = 1:numArgs
        push!(vals, LLVM.Value(unsafe_load(Args, i)))
        push!(valTys, API.VT_Primal)
    end

    anti = LLVM.Value(API.EnzymeGradientUtilsCallWithInvertedBundles(gutils, LLVM.Value(LLVM.API.LLVMGetCalledValue(OrigCI)), vals, length(vals), OrigCI, valTys, length(valTys), b, #=lookup=#false ))

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

# From https://github.com/JuliaLang/julia/blob/81813164963f38dcd779d65ecd222fad8d7ed437/src/cgutils.cpp#L570
@inline function isghostty(ty) 
    if ty === Union{}
        return true
    end
    if Base.isconcretetype(ty) && !ismutabletype(ty)
        if sizeof(ty) == 0
            return true
        end
        # TODO consider struct_to_llvm ?
    end
    return false
end

struct Tape{TapeTy,ShadowTy,ResT}
    internal_tape::TapeTy
    shadow_return::ShadowTy
end

function setup_macro_wraps(forwardMode::Bool, N::Int64, Width::Int64, base=nothing)
    primargs = Union{Symbol,Expr}[]
    shadowargs = Union{Symbol,Expr}[]
    primtypes = Union{Symbol,Expr}[]
    allargs = Expr[]
    typeargs = Symbol[]
    dfns = Union{Symbol,Expr}[:df]
    base_idx = 1
    for w in 2:Width
        if base === nothing
            shad = Symbol("df_$w")
            t = Symbol("DF__$w*")
            e = :($shad::$t)
            push!(allargs, e)
            push!(typeargs, t)
        else
            shad = :($base[$base_idx]) 
            base_idx += 1
        end
        push!(dfns, shad)
    end
    for i in 1:N
        if base === nothing
            prim = Symbol("primal_$i")
            t = Symbol("PT_$i")
            e = :($prim::$t)
            push!(allargs, e)
            push!(typeargs, t)
        else
            prim = :($base[$base_idx]) 
            t = :(typeof($prim))
            base_idx += 1
        end
        push!(primargs, prim)
        push!(primtypes, t)
        shadows = Union{Symbol,Expr}[]
        for w in 1:Width
            if base === nothing
                shad = Symbol("shadow_$(i)_$w")
                t = Symbol("ST_$(i)_$w")
                e = :($shad::$t)
                push!(allargs, e)
                push!(typeargs, t)
            else
                shad = :($base[$base_idx]) 
                t = :(typeof($shad))
                base_idx += 1
            end
            push!(shadows, shad)
        end
        if Width == 1
            push!(shadowargs, shadows[1])
        else
            push!(shadowargs, :(($(shadows...),)))
        end
    end
    wrapped = Expr[]
    for i in 1:N
        expr = :(
                if ActivityTup[$i+1] && !isghostty($(primtypes[i]))
                if !$forwardMode && ($(primtypes[i]) <: AbstractFloat || $(primtypes[i]) <: Complex{<:AbstractFloat})
                    Active($(primargs[i]))
                 else
                     $((Width == 1) ? :Duplicated : :BatchDuplicated)($(primargs[i]), $(shadowargs[i]))
                 end
             else
                 Const($(primargs[i]))
             end

            )
        push!(wrapped, expr)
    end
    return primargs, shadowargs, primtypes, allargs, typeargs, wrapped
end

@inline eltypeof(x) = eltype(typeof(x))

function body_runtime_generic_fwd(N, Width, wrapped, primtypes)
    nnothing = ntuple(i->nothing, Val(Width+1))
    nres = ntuple(i->:(res[1]), Val(Width+1))
    return quote
        args = ($(wrapped...),)

        fn = f
        dfn = ActivityTup[1] ? df : nothing

        # TODO: Annotation of return value
        # tt0 = Tuple{$(primtypes...)}
        tt = Tuple{map(eltypeof, args)...}
        tt′ = Tuple{map(typeof, args)...}
        rt = Core.Compiler.return_type(fn, tt)
        annotation = guess_activity(rt, API.DEM_ForwardMode)
        
        if annotation <: DuplicatedNoNeed
            annotation = Duplicated{rt}
        end
        if $Width != 1
            if annotation <: Duplicated
                annotation = BatchDuplicated{rt, $Width}
            end
        end

        forward = thunk(fn, dfn, annotation, tt′, Val(API.DEM_ForwardMode), width, #=ModifiedBetween=#Val(false), #=returnPrimal=#Val(true))

        res = forward(args...)

        if length(res) == 0
            return ReturnType($nnothing)
        end
        if annotation <: Const
            return ReturnType(($(nres...),))
        end

        if $Width == 1
            return ReturnType((res[1], res[2]))
        else
            return ReturnType((res[1], res[2]...))
        end
    end
end

function func_runtime_generic_fwd(N, Width)
    _, _, primtypes, allargs, typeargs, wrapped = setup_macro_wraps(true, N, Width)
    body = body_runtime_generic_fwd(N, Width, wrapped, primtypes)

    quote 
        function runtime_generic_fwd(activity::Val{ActivityTup}, width::Val{$Width}, RT::Val{ReturnType}, f::F, df::DF, $(allargs...)) where {ActivityTup, ReturnType, F, DF, $(typeargs...)}
            $body
        end
    end
end

@generated function runtime_generic_fwd(activity::Val{ActivityTup}, width::Val{Width}, RT::Val{ReturnType}, f::F, df::DF, allargs...) where {ActivityTup, Width, ReturnType, F, DF}
    N = div(length(allargs)+2, Width)-1
    _, _, primtypes, _, _, wrapped = setup_macro_wraps(true, N, Width, :allargs)
    return body_runtime_generic_fwd(N, Width, wrapped, primtypes)
end

function body_runtime_generic_augfwd(N, Width, wrapped, primttypes)
    nnothing = ntuple(i->nothing, Val(Width+1))
    nres = ntuple(i->:(origRet), Val(Width+1))
    nzeros = ntuple(i->:(Ref(zero(resT))), Val(Width))
    nres3 = ntuple(i->:(res[3]), Val(Width))
    return quote
        args = ($(wrapped...),)

        fn = f
        dfn = ActivityTup[1] ? df : nothing

        # TODO: Annotation of return value
        # tt0 = Tuple{$(primtypes...)}
        tt = Tuple{map(eltypeof, args)...}
        tt′ = Tuple{map(typeof, args)...}
        rt = Core.Compiler.return_type(fn, tt)
        annotation = guess_activity(rt, API.DEM_ReverseModePrimal)

        forward, adjoint = thunk(fn, dfn, annotation, tt′, Val(API.DEM_ReverseModePrimal), width,
                                     #=ModifiedBetween=#Val(true), #=returnPrimal=#Val(true))
        
        res = forward(args...)

        internal_tape = res[1]

        if length(res) == 1
            resT = Nothing
            shadow_return = nothing
            tape = Tape{typeof(internal_tape), typeof(shadow_return), resT}(internal_tape, shadow_return)
            return ReturnType(($nnothing..., tape))
        end
        if annotation <: Const
            let origRet = res[2], resT = typeof(origRet)
                shadow_return = nothing
                tape = Tape{typeof(internal_tape), typeof(shadow_return), resT}(internal_tape, shadow_return)
                return ReturnType(($(nres...), tape))
            end
        end
        if annotation <: Active
            let origRet = res[2], resT = typeof(origRet)
                if $Width == 1
                    shadow_return = Ref(zero(resT))
                else
                    shadow_return = ($(nzeros...),)
                end
                tape = Tape{typeof(internal_tape), typeof(shadow_return), resT}(internal_tape, shadow_return)
                if $Width == 1
                    return ReturnType((origRet, shadow_return, tape))
                else
                    return ReturnType((origRet, shadow_return..., tape))
                end
            end
        end
        
        @assert annotation <: Duplicated || annotation <: DuplicatedNoNeed || annotation <: BatchDuplicated || annotation <: BatchDuplicatedNoNeed
        
        origRet = res[2]
        resT = typeof(origRet)
        shadow_return = nothing
        tape = Tape{typeof(internal_tape), typeof(shadow_return), resT}(internal_tape, shadow_return)
        if $Width == 1
            return ReturnType((origRet, res[3], tape))
        else
            return ReturnType((origRet, $(nres3...), tape))
        end
    end
end

function func_runtime_generic_augfwd(N, Width)
    _, _, primtypes, allargs, typeargs, wrapped = setup_macro_wraps(false, N, Width)
    body = body_runtime_generic_augfwd(N, Width, wrapped, primtypes)

    quote 
        function runtime_generic_augfwd(activity::Val{ActivityTup}, width::Val{$Width}, RT::Val{ReturnType}, f::F, df::DF, $(allargs...)) where {ActivityTup, ReturnType, F, DF, $(typeargs...)}
            $body
        end
    end
end

@generated function runtime_generic_augfwd(activity::Val{ActivityTup}, width::Val{Width}, RT::Val{ReturnType}, f::F, df::DF, allargs...) where {ActivityTup, Width, ReturnType, F, DF}
    N = div(length(allargs)+2, Width)-1
    _, _, primtypes, _, _, wrapped = setup_macro_wraps(false, N, Width, :allargs)
    return body_runtime_generic_augfwd(N, Width, wrapped, primtypes)
end

function body_runtime_generic_rev(N, Width, wrapped, primttypes)
    outs = []
    for i in 1:N
        for w in 1:Width
            expr = if Width == 1
                :(tup[$i])
            else
                :(tup[$i][$w])
            end
            shad = Symbol("shadow_$(i)_$w")
            out = :(if $expr === nothing
              elseif $shad isa Base.RefValue
                  $shad[] += $expr
                else
                  ref = shadow_ptr[$(i*(Width)+w)]
                  ref = reinterpret(Ptr{typeof($shad)}, ref)
                  unsafe_store!(ref, $shad+$expr)
                end
               )
            push!(outs, out)
        end
    end
    shadow_ret = nothing
    if Width == 1
        shadowret = :(tape.shadow_return[])
    else
        shadowret = []
        for w in 1:Width
            push!(shadowret, :(tape.shadow_return[$w][]))
        end
        shadowret = :(($(shadowret...),))
    end

    quote
        args = ($(wrapped...),)

        fn = f
        dfn = ActivityTup[1] ? df : nothing

        # TODO: Annotation of return value
        # tt0 = Tuple{$(primtypes...)}
        tt = Tuple{map(eltypeof, args)...}
        tt′ = Tuple{map(typeof, args)...}
        rt = Core.Compiler.return_type(fn, tt)
        annotation = guess_activity(rt, API.DEM_ReverseModePrimal)
        
        forward, adjoint = thunk(fn, dfn, annotation, tt′, Val(API.DEM_ReverseModePrimal), width,
                                    #=ModifiedBetween=#Val(true), #=returnPrimal=#Val(true))
        if tape.shadow_return !== nothing
            args = (args..., $shadowret)
        end
    
        tup = adjoint(args..., tape.internal_tape)[1]

        $(outs...)
        return nothing
    end
end

function func_runtime_generic_rev(N, Width)
    _, _, primtypes, allargs, typeargs, wrapped = setup_macro_wraps(false, N, Width)
    body = body_runtime_generic_rev(N, Width, wrapped, primtypes)

    quote 
        function runtime_generic_rev(activity::Val{ActivityTup}, width::Val{$Width}, tape::TapeType, shadow_ptr, f::F, df::DF, $(allargs...)) where {ActivityTup, TapeType, F, DF, $(typeargs...)}
            $body
        end
    end
end

@generated function runtime_generic_rev(activity::Val{ActivityTup}, width::Val{Width}, tape::TapeType, f::F, df::DF, allargs...) where {ActivityTup, Width, TapeType, F, DF}
    N = div(length(allargs)+2, Width)-1
    _, _, primtypes, _, _, wrapped = setup_macro_wraps(false, N, Width, :allargs)
    return body_runtime_generic_rev(N, Width, wrapped, primtypes)
end

# Create specializations
for (N, Width) in Iterators.product(0:30, 1:10)
    eval(func_runtime_generic_fwd(N, Width))
    eval(func_runtime_generic_augfwd(N, Width))
    eval(func_runtime_generic_rev(N, Width))
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

function unsafe_to_llvm(val, ctx)
    T_jlvalue = LLVM.StructType(LLVM.LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_prjlvalue_UT = LLVM.PointerType(T_jlvalue)
    fill_val = unsafe_to_pointer(val)
    fill_val = LLVM.ConstantInt(reinterpret(Int, fill_val); ctx)
    fill_val = LLVM.const_inttoptr(fill_val, T_prjlvalue_UT)
    LLVM.const_addrspacecast(fill_val, T_prjlvalue)
end

function generic_setup(orig, func, ReturnType, gutils, start, ctx::LLVM.Context, B::LLVM.Builder,  lookup; sret=nothing, tape=nothing, firstconst=false)
    width = API.EnzymeGradientUtilsGetWidth(gutils)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    ops = collect(operands(orig))[start+firstconst:end-1]
    
    T_int8 = LLVM.Int8Type(ctx)
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)


    num = convert(UInt32, length(ops))

    ActivityList = Bool[]

    to_preserve = LLVM.Value[]

    @assert length(ops) != 0
    fill_val = unsafe_to_llvm(nothing, ctx)

    vals = LLVM.Value[]
    
    T_jlvalue = LLVM.StructType(LLVM.LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    
    if tape !== nothing
        NT = NTuple{length(ops)*Int64(width), Ptr{Nothing}}
        shadow_ptr = emit_allocobj!(B, NT)
        shadow = bitcast!(B, shadow_ptr, LLVM.PointerType(convert(LLVMType, NT; ctx), addrspace(llvmtype(shadow_ptr))))
    end

    if firstconst
        val = LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, operands(orig)[start]))
        if lookup
            val = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, val, B))
        end
        push!(vals, val)
    end

    for (i, op) in enumerate(ops)
        val = LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, op))
        if lookup
            val = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, val, B))
        end
        
        push!(vals, val)

        active = API.EnzymeGradientUtilsIsConstantValue(gutils, op) == 0
        push!(ActivityList, active)

        inverted = nothing

        if active
            inverted = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, op, B))
            if lookup
                inverted = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, inverted, B))
            end
        end

        for w in 1:width 
            ev = fill_val
            if inverted !== nothing
                if width == 1
                    ev = inverted
                else
                    ev = extract_value!(B, inverted, w-1)
                end
                if tape !== nothing
                    push!(to_preserve, ev)
                end
            end

            push!(vals, ev)
            if tape !== nothing
                idx = LLVM.Value[LLVM.ConstantInt(0; ctx), LLVM.ConstantInt((i-1)*Int64(width) + w-1; ctx)]
                ev = addrspacecast!(B, ev, LLVM.PointerType(eltype(llvmtype(ev)), 11))
                ev = emit_pointerfromobjref!(B, ev)
                ev = ptrtoint!(B, ev, convert(LLVMType, Int; ctx))
                LLVM.store!(B, ev, LLVM.inbounds_gep!(B, shadow, idx))
            end
        end
    end

    if tape !== nothing
        pushfirst!(vals, shadow_ptr)
        pushfirst!(vals, LLVM.Value(tape))
    else
        pushfirst!(vals, unsafe_to_llvm(Val(ReturnType), ctx))
    end
    pushfirst!(vals, unsafe_to_llvm(Val(Int64(width)), ctx))
    pushfirst!(vals, unsafe_to_llvm(Val((ActivityList...,)), ctx))
    
    @static if VERSION < v"1.7.0-" || true
    else
    mi = emit_methodinstance!(B, func, vals)
    end

    pushfirst!(vals, unsafe_to_llvm(func, ctx))
    
    @static if VERSION < v"1.7.0-" || true
    else
    pushfirst!(vals, mi)
    end

    @static if VERSION < v"1.7.0-" || true
    cal = emit_apply_generic!(B, vals)
    else
    cal = emit_invoke!(B, vals)
    end

    API.EnzymeGradientUtilsSetDebugLocFromOriginal(gutils, cal, orig)

    if tape === nothing
        llty = convert(LLVMType, ReturnType; ctx)
        cal = LLVM.pointercast!(B, cal, LLVM.PointerType(llty, Tracked))
    end

    return cal
end

function allocate_sret!(B::LLVM.Builder, N, ctx)
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)
    al = LLVM.alloca!(B, LLVM.ArrayType(T_prjlvalue, N))
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
        
        sret = generic_setup(orig, runtime_generic_fwd, AnyArray(1+Int64(width)), gutils, #=start=#offset, ctx, B, false)

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

    if API.EnzymeGradientUtilsIsConstantValue(gutils, orig) == 0 || API.EnzymeGradientUtilsIsConstantInstruction(gutils, orig) == 0 
        B = LLVM.Builder(B)
        width = API.EnzymeGradientUtilsGetWidth(gutils)
        sret = generic_setup(orig, runtime_generic_augfwd, AnyArray(2+Int64(width)), gutils, #=start=#offset, ctx, B, false)

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

    if API.EnzymeGradientUtilsIsConstantValue(gutils, orig) == 0 || API.EnzymeGradientUtilsIsConstantInstruction(gutils, orig) == 0 
        B = LLVM.Builder(B)
        ctx = LLVM.context(orig)

        @assert tape !== C_NULL
        width = API.EnzymeGradientUtilsGetWidth(gutils)
        generic_setup(orig, runtime_generic_rev, Nothing, gutils, #=start=#offset, ctx, B, true; tape)

    end
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
    sret = generic_setup(orig, runtime_generic_fwd, AnyArray(1+Int64(width)), gutils, #=start=#offset+1, ctx, B, false)

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
    # sret = generic_setup(orig, runtime_apply_latest_augfwd, AnyArray(2+Int64(width)), gutils, #=start=#offset+1, ctx, B, false)
    sret = generic_setup(orig, runtime_generic_augfwd, AnyArray(2+Int64(width)), gutils, #=start=#offset+1, ctx, B, false)

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

    width = API.EnzymeGradientUtilsGetWidth(gutils)
    generic_setup(orig, runtime_generic_rev, Nothing, gutils, #=start=#offset+1, ctx, B, true; tape)

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

function common_newstructv_fwd(offset, B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    origops = collect(operands(orig))
    width = API.EnzymeGradientUtilsGetWidth(gutils)
    if API.EnzymeGradientUtilsIsConstantValue(gutils, orig) == 0
        B = LLVM.Builder(B)

        shadowsin = LLVM.Value[
                        LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, o, B)) for o in origops[offset:end-1] ]
        if width == 1
            if offset != 1
                pushfirst!(shadowsin, origops[1])
            end
            shadowres = LLVM.call!(B, LLVM.called_value(orig), shadowsin)
            conv = LLVM.API.LLVMGetInstructionCallConv(orig)
            LLVM.API.LLVMSetInstructionCallConv(shadowres, conv)
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig))))
            for idx in 1:width
                args = LLVM.Value[
                                  extract_value!(B, s, idx-1) for s in shadowsin
                                  ]
                if offset != 1
                    pushfirst!(args, origops[1])
                end
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
function common_newstructv_augfwd(offset, B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    common_newstructv_fwd(offset, B, OrigCI, gutils, normalR, shadowR)
    return nothing
end

function common_newstructv_rev(offset, B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    emit_error(LLVM.Builder(B), orig, "Enzyme: Not yet implemented reverse for jl_new_struct")
    return nothing
end

function common_f_tuple_fwd(offset, B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    common_newstructv_fwd(offset, B, OrigCI, gutils, normalR, shadowR)
    return nothing
end
function common_f_tuple_augfwd(offset, B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    common_f_tuple_fwd(offset, B, OrigCI, gutils, normalR, shadowR)
    return nothing
end

function common_f_tuple_rev(offset, B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    # This function allocates a new return which returns a pointer, thus this instruction itself cannot transfer
    # derivative info, only create a shadow pointer, which is handled by the forward pass.
    return nothing
end


function f_tuple_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    common_f_tuple_fwd(1, B, OrigCI, gutils, normalR, shadowR)
    return nothing
end

function f_tuple_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    common_f_tuple_augfwd(1, B, OrigCI, gutils, normalR, shadowR, tapeR)
    return nothing
end

function f_tuple_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    common_f_tuple_rev(1, B, OrigCI, gutils, tape)
    return nothing
end

function new_structv_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    common_newstructv_fwd(1, B, OrigCI, gutils, normalR, shadowR)
    return nothing
end

function new_structv_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    common_newstructv_augfwd(1, B, OrigCI, gutils, normalR, shadowR, tapeR)
    return nothing
end

function new_structv_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    common_apply_latest_rev(1, B, OrigCI, gutils, tape)
    return nothing
end

function common_jl_getfield_fwd(offset, B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    if shadowR != C_NULL
        orig = LLVM.Instruction(OrigCI)
        origops = collect(operands(orig))[offset:end]
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
                if offset != 1
                    pushfirst!(args, first(operands(orig)))
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
                    if offset != 1
                        pushfirst!(args, first(operands(orig)))
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
function common_jl_getfield_augfwd(offset, B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
  common_jl_getfield_fwd(offset, B, OrigCI, gutils, normalR, shadowR)
end
function common_jl_getfield_rev(offset, B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid 
    orig = LLVM.Instruction(OrigCI)

    if API.EnzymeGradientUtilsIsConstantValue(gutils, orig) == 0
        emit_error(LLVM.Builder(B), orig, "Enzyme: not yet implemented in reverse mode, jl_getfield")
        return nothing
        origops = collect(operands(orig))[offset:end]
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
function jl_getfield_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    common_jl_getfield_fwd(1, B, OrigCI, gutils, normalR, shadowR)
end
function jl_getfield_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    common_jl_getfield_augfwd(1, B, OrigCI, gutils, normalR, shadowR, tapeR)
end
function jl_getfield_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid 
    common_jl_getfield_rev(1, B, OrigCI, gutils, tape)
end

function common_apply_iterate_fwd(offset, B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    emit_error(LLVM.Builder(B), orig, "Enzyme: unhandled augmented forward for jl_f__apply_iterate")
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return nothing
end

function common_apply_iterate_augfwd(offset, B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    
    if API.EnzymeGradientUtilsIsConstantValue(gutils, orig) == 0 || API.EnzymeGradientUtilsIsConstantInstruction(gutils, orig) == 0  
        emit_error(LLVM.Builder(B), orig, "Enzyme: Not yet implemented augmented forward for jl_f__apply_iterate")

        normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
        if shadowR != C_NULL && normal !== nothing
            unsafe_store!(shadowR, normal.ref)
        end
    end

    return nothing
end

function common_apply_iterate_rev(offset, B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    if API.EnzymeGradientUtilsIsConstantValue(gutils, orig) == 0 || API.EnzymeGradientUtilsIsConstantInstruction(gutils, orig) == 0  
        emit_error(LLVM.Builder(B), orig, "Enzyme: Not yet implemented reverse for jl_f__apply_iterate")
    end
    return nothing
end

function apply_iterate_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    common_apply_iterate_fwd(1, B, OrigCI, gutils, normalR, shadowR)
    return nothing
end

function apply_iterate_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    common_apply_iterate_augfwd(1, B, OrigCI, gutils, normalR, shadowR, tapeR)
    return nothing
end

function apply_iterate_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    common_apply_iterate_rev(1, B, OrigCI, gutils, tape)
    return nothing
end


function jlcall_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    ctx = LLVM.context(LLVM.parent(LLVM.parent(LLVM.parent(orig)))) 
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
        if in(name, ("ijl_new_structv", "jl_new_structv"))
            common_newstructv_fwd(2, B, OrigCI, gutils, normalR, shadowR)
            return nothing
        end
        if in(name, ("ijl_f_tuple", "jl_f_tuple"))
            common_f_tuple_fwd(2, B, OrigCI, gutils, normalR, shadowR)
            return nothing
        end
        if in(name, ("ijl_f_getfield", "jl_f_getfield"))
            common_jl_getfield_fwd(2, B, OrigCI, gutils, normalR, shadowR)
            return nothing
        end
        if in(name, ("ijl_f__apply_iterate", "jl_f__apply_iterate"))
            common_apply_iterate_fwd(2, B, OrigCI, gutils, normalR, shadowR)
            return nothing
        end
        if any(map(k->kind(k)==kind(StringAttribute("enzyme_inactive"; ctx)), collect(function_attributes(F))))
            return nothing
        end
    end
    
    @assert false "jl_call calling convention not implemented yet", orig
    
    return nothing
end

function jlcall_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    ctx = LLVM.context(LLVM.parent(LLVM.parent(LLVM.parent(orig)))) 
   
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
        if in(name, ("ijl_new_structv", "jl_new_structv"))
            common_newstructv_augfwd(2, B, OrigCI, gutils, normalR, shadowR, tapeR)
            return nothing
        end
        if in(name, ("ijl_f_tuple", "jl_f_tuple"))
            common_f_tuple_augfwd(2, B, OrigCI, gutils, normalR, shadowR, tapeR)
            return nothing
        end
        if in(name, ("ijl_f_getfield", "jl_f_getfield"))
            common_jl_getfield_augfwd(2, B, OrigCI, gutils, normalR, shadowR, tapeR)
            return nothing
        end
        if in(name, ("ijl_f__apply_iterate", "jl_f__apply_iterate"))
            common_apply_iterate_augfwd(2, B, OrigCI, gutils, normalR, shadowR, tapeR)
            return nothing
        end
        if any(map(k->kind(k)==kind(StringAttribute("enzyme_inactive"; ctx)), collect(function_attributes(F))))
            return nothing
        end
    end

    @assert false "jl_call calling convention not implemented yet", orig
    
    return nothing
end

function jlcall_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    ctx = LLVM.context(LLVM.parent(LLVM.parent(LLVM.parent(orig)))) 
    
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
        if in(name, ("ijl_new_structv", "jl_new_structv"))
            common_newstructv_rev(2, B, OrigCI, gutils, tape)
            return nothing
        end
        if in(name, ("ijl_f_tuple", "jl_f_tuple"))
            common_f_tuple_rev(2, B, OrigCI, gutils, tape)
            return nothing
        end
        if in(name, ("ijl_f_getfield", "jl_f_getfield"))
            common_jl_getfield_rev(2, B, OrigCI, gutils, tape)
            return nothing
        end
        if in(name, ("ijl_f__apply_iterate", "jl_f__apply_iterate"))
            common_apply_iterate_rev(2, B, OrigCI, gutils, tape)
            return nothing
        end
        if any(map(k->kind(k)==kind(StringAttribute("enzyme_inactive"; ctx)), collect(function_attributes(F))))
            return nothing
        end
    end
    
    @assert false "jl_call calling convention not implemented yet", orig

    return nothing
end

function jlcall2_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    ctx = LLVM.context(LLVM.parent(LLVM.parent(LLVM.parent(orig)))) 
    
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_invoke", "jl_invoke"))
            common_invoke_fwd(2, B, OrigCI, gutils, normalR, shadowR)
            return nothing
        end
        if any(map(k->kind(k)==kind(StringAttribute("enzyme_inactive"; ctx)), collect(function_attributes(F))))
            return nothing
        end
    end
    
    @assert false "jl_call calling convention not implemented yet", orig
    
    return nothing
end

function jlcall2_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    ctx = LLVM.context(LLVM.parent(LLVM.parent(LLVM.parent(orig)))) 
   
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_invoke", "jl_invoke"))
            common_invoke_augfwd(2, B, OrigCI, gutils, normalR, shadowR, tapeR)
            return nothing
        end
        if any(map(k->kind(k)==kind(StringAttribute("enzyme_inactive"; ctx)), collect(function_attributes(F))))
            return nothing
        end
    end

    @assert false "jl_call calling convention not implemented yet", orig
    
    return nothing
end

function jlcall2_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    ctx = LLVM.context(LLVM.parent(LLVM.parent(LLVM.parent(orig)))) 
    
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_invoke", "jl_invoke"))
            common_invoke_rev(2, B, OrigCI, gutils, tape)
            return nothing
        end
        if any(map(k->kind(k)==kind(StringAttribute("enzyme_inactive"; ctx)), collect(function_attributes(F))))
            return nothing
        end
    end
    
    @assert false "jl_call calling convention not implemented yet", orig

    return nothing
end

function jl_nthfield_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    if shadowR != C_NULL
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
                shadowres = LLVM.call!(B, LLVM.called_value(orig), args)
                conv = LLVM.API.LLVMGetInstructionCallConv(orig)
                LLVM.API.LLVMSetInstructionCallConv(shadowres, conv)
            else
                shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig))))
                for idx in 1:width
                    args = LLVM.Value[
                                      extract_value!(B, shadowin, idx-1)
                                      LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[2]))
                                      ]
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
function jl_nthfield_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
  jl_nthfield_fwd(B, OrigCI, gutils, normalR, shadowR)
end
function jl_nthfield_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid 
    orig = LLVM.Instruction(OrigCI)
    if API.EnzymeGradientUtilsIsConstantValue(gutils, orig) == 0
        emit_error(LLVM.Builder(B), orig, "Enzyme: not yet implemented in reverse mode, jl_nthfield")
    end
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
        sret = generic_setup(orig, runtime_generic_fwd, AnyArray(1+Int64(width)), gutils, #=start=#offset+1, ctx, B, false)
        
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
        sret = generic_setup(orig, runtime_generic_augfwd, AnyArray(2+Int64(width)), gutils, #=start=#offset+1, ctx, B, false)
        
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

        width = API.EnzymeGradientUtilsGetWidth(gutils)
        generic_setup(orig, runtime_generic_rev, Nothing, gutils, #=start=#offset+1, ctx, B, true; tape)
    end

    return nothing
end

function invoke_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    common_invoke_fwd(1, B, OrigCI, gutils, normalR, shadowR)
    return nothing
end

function invoke_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    common_invoke_augfwd(1, B, OrigCI, gutils, normalR, shadowR, tapeR)
    return nothing
end

function invoke_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    common_invoke_rev(1, B, OrigCI, gutils, tape)
    return nothing
end

function emit_error(B::LLVM.Builder, orig, string)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    ctx = context(mod)

    # 1. get the error function
    funcT = LLVM.FunctionType(LLVM.VoidType(ctx), LLVMType[LLVM.PointerType(LLVM.Int8Type(ctx))])
    func = get_function!(mod, "jl_error", funcT, [EnumAttribute("noreturn"; ctx)])


    if orig !== nothing
        bt = GPUCompiler.backtrace(orig)
        function printBT(io)
            print(io,"\nCaused by:")
            Base.show_backtrace(io, bt)
        end
        string*=sprint(io->Base.show_backtrace(io, bt)) 
    end

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

function nested_codegen!(mode::API.CDerivativeMode, mod::LLVM.Module, f, tt)
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

    target = DefaultCompilerTarget()
    params = PrimalCompilerParams(mode)
    job    = CompilerJob(target, funcspec, params)

    # TODO
    parent_job = nothing
    otherMod, meta = GPUCompiler.codegen(:llvm, job; optimize=false, cleanup=false, validate=false, parent_job=parent_job, ctx)
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
            eparams = Compiler.EnzymeCompilerParams(eadjoint, API.DEM_ForwardMode, width, Const{Nothing}, #=runEnzyme=#true, #=shadowfunc=#dupClosure, #=abiwrap=#true, #=modifiedBetween=#false, #=returnPrimal=#false, #=shadowInit=#false)
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
            eparams = Compiler.EnzymeCompilerParams(eadjoint, API.DEM_ReverseModePrimal, width, Const{Nothing}, #=runEnzyme=#true, #=shadowfunc=#dupClosure, #=abiwrap=#true, #=modifiedBetween=#true, #=returnPrimal=#false, #=shadowInit=#false)
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
    mode = API.EnzymeGradientUtilsGetMode(gutils)
    entry = nested_codegen!(mode, mod, runtime_pfor_fwd, tt)
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
    mode = API.EnzymeGradientUtilsGetMode(gutils)
    entry = nested_codegen!(mode, mod, runtime_pfor_augfwd, tt)
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
    mode = API.EnzymeGradientUtilsGetMode(gutils)
    entry = nested_codegen!(mode, mod, runtime_pfor_rev, tt)
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
    mode = API.EnzymeGradientUtilsGetMode(gutils)
    fun = nested_codegen!(mode, mod, runtime_newtask_fwd, Tuple{Any, Any, Any, Int, Val{width}})
    permit_inlining!(fun)

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
    mode = API.EnzymeGradientUtilsGetMode(gutils)
    fun = nested_codegen!(mode, mod, runtime_newtask_augfwd, Tuple{Any, Any, Any, Int, Val{width}})
    permit_inlining!(fun)

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

function enzyme_custom_setup_args(B, orig, gutils, mi, reverse)
    ctx = LLVM.context(orig)
    ops = collect(operands(orig))
    called = ops[end]
    ops = ops[1:end-1]
    width = API.EnzymeGradientUtilsGetWidth(gutils)

    args = LLVM.Value[]
    activity = Type[]
    overwritten = Bool[]

    actives = LLVM.Value[]
   
    uncacheable = Vector{UInt8}(undef, length(ops))
    API.EnzymeGradientUtilsGetUncacheableArgs(gutils, orig, uncacheable, length(uncacheable))

    sret = false
    returnRoots = false

	jlargs = classify_arguments(mi.specTypes, eltype(llvmtype(called)), sret, returnRoots)

    op_idx = 1
    
    alloctx = LLVM.Builder(ctx)
    position!(alloctx, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))
   
    for arg in jlargs
        if arg.cc == GPUCompiler.GHOST
            push!(activity, Const{arg.typ})
            push!(overwritten, false)
            continue
        end
        
        op = ops[op_idx]
        push!(overwritten, uncacheable[op_idx] != 0)
        op_idx+=1
        
        val = LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, op))
        if reverse
            val = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, val, B))
        end

        activep = API.EnzymeGradientUtilsGetDiffeType(gutils, op, #=isforeign=#false)
       
        # TODO type analysis deduce if duplicated vs active
        if activep == API.DFT_CONSTANT
            Ty = Const{arg.typ}
            llty = convert(LLVMType, Ty; ctx)
            al0 = al = emit_allocobj!(B, Ty)
            al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(llvmtype(al))))
            al = addrspacecast!(B, al, LLVM.PointerType(llty, 11))

            ptr = gep!(B, al, [LLVM.ConstantInt(LLVM.IntType(64; ctx), 0), LLVM.ConstantInt(LLVM.IntType(32; ctx), 0)])
            if llvmtype(val) != eltype(llvmtype(ptr))
                val = load!(B, val)
            end
            store!(B, val, ptr)
            
            if any_jltypes(llty)
                emit_writebarrier!(B, get_julia_inner_types(B, al0, val))
            end
            
            push!(args, al)

            push!(activity, Ty)

        elseif activep == API.DFT_OUT_DIFF
            Ty = Active{arg.typ}
            llty = convert(LLVMType, Ty; ctx)
            al0 = al = emit_allocobj!(B, Ty)
            al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(llvmtype(al))))
            al = addrspacecast!(B, al, LLVM.PointerType(llty, 11))
            
            ptr = gep!(B, al, [LLVM.ConstantInt(LLVM.IntType(64; ctx), 0), LLVM.ConstantInt(LLVM.IntType(32; ctx), 0)])
            if llvmtype(val) != eltype(llvmtype(ptr))
                val = load!(B, val)
            end
            store!(B, val, ptr)
            
            if any_jltypes(llty)
                emit_writebarrier!(B, get_julia_inner_types(B, al0, val))
            end
            
            push!(args, al)
            
            push!(activity, Ty)
            push!(actives, op)
        else
            ival = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, op, B))
            if reverse
                ival = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, ival, B))
            end
            if width == 1
                if activep == API.DFT_DUP_ARG
                    Ty = Duplicated{arg.typ}
                else
                    @assert activep == API.DFT_DUP_NONEED
                    Ty = DuplicatedNoNeed{arg.typ}
                end
            else
                if activep == API.DFT_DUP_ARG
                    Ty = BatchDuplicated{arg.typ, Int64(width)}
                else
                    @assert activep == API.DFT_DUP_NONEED
                    Ty = BatchDuplicatedNoNeed{arg.typ, Int64(width)}
                end
            end
            
            llty = convert(LLVMType, Ty; ctx)
            al0 = al = emit_allocobj!(B, Ty)
            al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(llvmtype(al))))
            al = addrspacecast!(B, al, LLVM.PointerType(llty, 11))

            ptr = gep!(B, al, [LLVM.ConstantInt(LLVM.IntType(64; ctx), 0), LLVM.ConstantInt(LLVM.IntType(32; ctx), 0)])
            if llvmtype(val) != eltype(llvmtype(ptr))
                val = load!(B, val)
                ival = load!(B, ival)
            end
            store!(B, val, ptr)
            
            iptr = gep!(B, al, [LLVM.ConstantInt(LLVM.IntType(64; ctx), 0), LLVM.ConstantInt(LLVM.IntType(32; ctx), 1)])
            store!(B, ival, iptr)

            if any_jltypes(llty)
                emit_writebarrier!(B, get_julia_inner_types(B, al0, val, ival))
            end
            
            push!(args, al)
            push!(activity, Ty)
        end

    end
    
    @assert op_idx-1 == length(ops)

    return args, activity, (overwritten...,), actives
end

function enzyme_custom_setup_ret(gutils, orig, mi, job)
    width = API.EnzymeGradientUtilsGetWidth(gutils)
    interp = GPUCompiler.get_interpreter(job)
    RealRt = Core.Compiler.typeinf_ext_toplevel(interp, mi).rettype

    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(gutils, orig, needsPrimalP, needsShadowP)
    needsPrimal = needsPrimalP[] != 0

    if !needsPrimal && activep == API.DFT_DUP_ARG
        activep = API.DFT_DUP_NONEED
    end
   
    if activep == API.DFT_CONSTANT
        RT = Const{RealRt}
    
    elseif activep == API.DFT_OUT_DIFF
        RT = Active{RealRt}
    
    elseif activep == API.DFT_DUP_ARG
        if width == 1
            RT = Duplicated{RealRt}
        else
            RT = BatchDuplicated{RealRt, Int64(width)}
        end
    else
        @assert activep == API.DFT_DUP_NONEED
        if width == 1
            RT = DuplicatedNoNeed{RealRt}
        else
            RT = BatchDuplicatedNoNeed{RealRt, Int64(width)}
        end
    end
    return RealRt, RT, needsPrimal, needsShadowP[] != 0
end

function enzyme_custom_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    ctx = LLVM.context(orig)
    B  = LLVM.Builder(B)
    
    width = API.EnzymeGradientUtilsGetWidth(gutils)

    if shadowR != C_NULL
        unsafe_store!(shadowR,UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig)))).ref)
    end

    # TODO: don't inject the code multiple times for multiple calls
    
    # 1) extract out the MI from attributes
    mi, job = enzyme_custom_extract_mi(orig)

    # 2) Create activity, and annotate function spec
    args, activity, overwritten, actives = enzyme_custom_setup_args(B, orig, gutils, mi, #=reverse=#false)
    RealRt, RT, needsPrimal, needsShadow = enzyme_custom_setup_ret(gutils, orig, mi, job)
    
    
    alloctx = LLVM.Builder(ctx)
    position!(alloctx, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))
    mode = API.EnzymeGradientUtilsGetMode(gutils)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    tt = copy(activity)
    insert!(tt, 2, Type{RT})
    TT = Tuple{tt...}

    # TODO get world
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    world = enzyme_extract_world(fn)
    if EnzymeRules.isapplicable(EnzymeRules.forward, TT; world)
        @safe_debug "Applying custom forward rule" TT
        llvmf = nested_codegen!(mode, mod, EnzymeRules.forward, TT)
    else
        @safe_debug "No custom forward rule is applicable for" TT
        emit_error(B, orig, "Enzyme: No custom rule was appliable for " * string(TT))
        return nothing
    end
    
    fwd_RT = Core.Compiler.return_type(EnzymeRules.forward, TT, world)


    sret = nothing
    if !isempty(parameters(llvmf)) && any(map(k->kind(k)==kind(EnumAttribute("sret"; ctx)), collect(parameter_attributes(llvmf, 1))))
        sret = alloca!(alloctx, eltype(llvmtype(parameters(llvmf)[1])))
        pushfirst!(args, sret)
    end

    for i in eachindex(args)
        party = llvmtype(parameters(llvmf)[i])
        if llvmtype(args[i]) == party
            continue
        end
        GPUCompiler.@safe_error "Calling convention mismatch", party, args[i], i, llvmf
        return
    end

    res = LLVM.call!(B, llvmf, args)
    API.EnzymeGradientUtilsSetDebugLocFromOriginal(gutils, res, orig)
    
    hasNoRet = any(map(k->kind(k)==kind(EnumAttribute("noreturn"; ctx)), collect(function_attributes(llvmf))))
                                                      
    if hasNoRet
        return nothing
    end
    
    if sret !== nothing
        attr = if LLVM.version().major >= 12
            TypeAttribute("sret", eltype(llvmtype(parameters(llvmf)[1])); ctx)
        else
            EnumAttribute("sret"; ctx)
        end
        LLVM.API.LLVMAddCallSiteAttribute(res, LLVM.API.LLVMAttributeIndex(1), attr)
        res = load!(B, sret)
    end

    shadowV = C_NULL
    normalV = C_NULL

    if RT <: Const
        if needsPrimal
            if RealRt != fwd_RT
                emit_error(B, orig, "Enzyme: incorrect return type of const primal-only forward custom rule - "*(string(RT))*" "*string(activity)*" want just return type "*string(RealRt)*" found "*string(fwd_RT))
                return
            end
            normalV = res.ref
        else
            if Nothing != fwd_RT
                emit_error(B, orig, "Enzyme: incorrect return type of const no-primal forward custom rule - "*(string(RT))*" "*string(activity)*" want just return type Nothing found "*string(fwd_RT))
                return
            end
        end
    else
        if !needsPrimal
            shadowV = res.ref
            ST = RealRt
            if width != 1
                ST = NTuple{Int64(width), ST}
            end
            if ST != fwd_RT
                emit_error(B, orig, "Enzyme: incorrect return type of shadow-only forward custom rule - "*(string(RT))*" "*string(activity)*" want just shadow type "*string(ST)*" found "*string(fwd_RT))
                return
            end
        else
            ST = if width == 1
                Duplicated{RealRt}
            else
                BatchDuplicated{RealRt, Int64(width)}
            end
            if ST != fwd_RT
                emit_error(B, orig, "Enzyme: incorrect return type of prima/shadow forward custom rule - "*(string(RT))*" "*string(activity)*" want just shadow type "*string(ST)*" found "*string(fwd_RT))
                return
            end
            normalV = extract_value!(B, res, 0).ref
            shadowV = extract_value!(B, res, 1).ref
        end
    end

    if shadowR != C_NULL
        unsafe_store!(shadowR, shadowV)
    end
    
    # Delete the primal code
    if needsPrimal
        unsafe_store!(normalR, normalV)
    else
        LLVM.API.LLVMInstructionEraseFromParent(LLVM.Instruction(API.EnzymeGradientUtilsNewFromOriginal(gutils, orig)))
    end

    return nothing
end

function enzyme_custom_common_rev(forward::Bool, B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR, shadowR, tape)::LLVM.API.LLVMValueRef
    
    orig = LLVM.Instruction(OrigCI)
    ctx = LLVM.context(orig)
    B  = LLVM.Builder(B)
    
    width = API.EnzymeGradientUtilsGetWidth(gutils)

    shadowType = LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig)))
    if shadowR != C_NULL
        unsafe_store!(shadowR,UndefValue(shadowType).ref)
    end
    
    # TODO: don't inject the code multiple times for multiple calls
    
    # 1) extract out the MI from attributes
    mi, job = enzyme_custom_extract_mi(orig)

    # 2) Create activity, and annotate function spec
    args, activity, overwritten, actives = enzyme_custom_setup_args(B, orig, gutils, mi, #=reverse=#!forward)
    RealRt, RT, needsPrimal, needsShadow = enzyme_custom_setup_ret(gutils, orig, mi, job)
    
    alloctx = LLVM.Builder(ctx)
    position!(alloctx, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))

    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    world = enzyme_extract_world(fn)

    C = EnzymeRules.Config{Bool(needsPrimal), Bool(needsShadow), Int(width), overwritten}
    augprimal_tt = copy(activity)
    insert!(augprimal_tt, 2, Type{RT})
    pushfirst!(augprimal_tt, C)
    augprimal_TT = Tuple{augprimal_tt...}

    rev_TT = nothing
    aug_RT = Core.Compiler.return_type(EnzymeRules.augmented_primal, augprimal_TT, world)
    rev_RT = nothing

    TapeT = Nothing
    
    if (aug_RT <: EnzymeRules.AugmentedReturn || aug_RT <: EnzymeRules.AugmentedReturnFlexShadow) && !(aug_RT isa Union) && !(aug_RT === Union{})
        TapeT = EnzymeRules.tape_type(aug_RT)
    end

    mode = API.EnzymeGradientUtilsGetMode(gutils)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    if forward
        if EnzymeRules.isapplicable(EnzymeRules.augmented_primal, augprimal_TT; world)
            @safe_debug "Applying custom augmented_primal rule" TT=augprimal_TT
            llvmf = nested_codegen!(mode, mod, EnzymeRules.augmented_primal, augprimal_TT)
        else
            @safe_debug "No custom augmented_primal rule is applicable for" augprimal_TT
            emit_error(B, orig, "Enzyme: No custom rule was appliable for " * string(augprimal_TT))
            return C_NULL
        end
    else
        tt = copy(activity)
        insert!(tt, 2, RT <: Active ? RT : Type{RT})
        insert!(tt, 3, TapeT)
        pushfirst!(tt, C)
        TT = Tuple{tt...}
        rev_TT = TT

        if EnzymeRules.isapplicable(EnzymeRules.reverse, TT; world)
            @safe_debug "Applying custom reverse rule" TT
            llvmf = nested_codegen!(mode, mod, EnzymeRules.reverse, TT)
        else
            @safe_debug "No custom reverse rule is applicable for" TT
            emit_error(B, orig, "Enzyme: No custom rule was appliable for " * string(TT))
            return C_NULL
        end
        rev_RT = Core.Compiler.return_type(EnzymeRules.reverse, TT, world)
    end
     
    needsTape = !GPUCompiler.isghosttype(TapeT) && !Core.Compiler.isconstType(TapeT)
    
    tapeV = C_NULL
    if forward && needsTape
        tapeV = LLVM.UndefValue(convert(LLVMType, TapeT; ctx, allow_boxed=true)).ref
    end

    # if !forward
    #     argTys = copy(activity)
    #     if RT <: Active
    #         if width == 1
    #             push!(argTys, RealRt)
    #         else
    #             push!(argTys, NTuple{RealRt, (Int64)width})
    #         end
    #     end
    #     push!(argTys, tapeType)
    #     llvmf = nested_codegen!(mode, mod, rev_func, Tuple{argTys...})
    # end
    
    sret = nothing
    if !isempty(parameters(llvmf)) && any(map(k->kind(k)==kind(EnumAttribute("sret"; ctx)), collect(parameter_attributes(llvmf, 1))))
        sret = alloca!(alloctx, eltype(llvmtype(parameters(llvmf)[1])))
        pushfirst!(args, sret)
    end

    if !forward
        if RT <: Active

            llty = convert(LLVMType, RT; ctx)

            val = LLVM.Value(API.EnzymeGradientUtilsDiffe(gutils, orig, B))
            al0 = al = emit_allocobj!(B, RT)
            al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(llvmtype(al))))
            al = addrspacecast!(B, al, LLVM.PointerType(llty, 11))
            
            ptr = gep!(B, al, [LLVM.ConstantInt(LLVM.IntType(64; ctx), 0), LLVM.ConstantInt(LLVM.IntType(32; ctx), 0)])
            store!(B, val, ptr)
            
            if any_jltypes(llty)
                emit_writebarrier!(B, get_julia_inner_types(B, al0, val))
            end

            pushfirst!(args, al)
        end
        if needsTape
            @assert tape != C_NULL
            pushfirst!(args, LLVM.Value(tape))
        end
    end

    for i in 1:length(args)
        party =  llvmtype(parameters(llvmf)[i])
        if llvmtype(args[i]) == party
            continue
        end
        GPUCompiler.@safe_error "Calling convention mismatch", party, args[i], i, llvmf, augprimal_TT, rev_TT
        return tapeV
    end

    res = LLVM.call!(B, llvmf, args)
    API.EnzymeGradientUtilsSetDebugLocFromOriginal(gutils, res, orig)
    
    hasNoRet = any(map(k->kind(k)==kind(EnumAttribute("noreturn"; ctx)), collect(function_attributes(llvmf))))
                                                      
    if hasNoRet
        return tapeV
    end
    
    if sret !== nothing
        attr = if LLVM.version().major >= 12
            TypeAttribute("sret", eltype(llvmtype(parameters(llvmf)[1])); ctx)
        else
            EnumAttribute("sret"; ctx)
        end
        LLVM.API.LLVMAddCallSiteAttribute(res, LLVM.API.LLVMAttributeIndex(1), attr)
        res = load!(B, sret)
    end

    shadowV = C_NULL
    normalV = C_NULL
    

    if forward
        ShadT = RealRt
        if width != 1
            ShadT = NTuple{Int64(width), RealRt}
        end
        ST = EnzymeRules.AugmentedReturn{needsPrimal ? RealRt : Nothing, needsShadow ? ShadT : Nothing, TapeT}
        if aug_RT != ST
            if aug_RT <: EnzymeRules.AugmentedReturnFlexShadow
                if convert(LLVMType, EnzymeRules.shadow_type(aug_RT); ctx, allow_boxed=true) != 
                   convert(LLVMType, EnzymeRules.shadow_type(ST)    ; ctx, allow_boxed=true) 
                    emit_error(B, orig, "Enzyme: Augmented forward pass custom rule " * string(augprimal_TT) * " flex shadow ABI return type mismatch, expected "*string(ST)*" found "* string(aug_RT))
            return C_NULL
                end
                ST = EnzymeRules.AugmentedReturnFlexShadow{needsPrimal ? RealRt : Nothing, needsShadow ? EnzymeRules.shadow_type(aug_RT) : Nothing, TapeT}
            end
        end
        if aug_RT != ST 
            ST = EnzymeRules.AugmentedReturn{needsPrimal ? RealRt : Nothing, needsShadow ? ShadT : Nothing, Any}
            emit_error(B, orig, "Enzyme: Augmented forward pass custom rule " * string(augprimal_TT) * " return type mismatch, expected "*string(ST)*" found "* string(aug_RT))
            return C_NULL
        end

        idx = 0
        if needsPrimal
            @assert !GPUCompiler.isghosttype(RealRt)
            normalV = extract_value!(B, res, idx)
            @assert llvmtype(normalV) == llvmtype(orig)
            normalV = normalV.ref
            idx+=1
        end
        if needsShadow
            @assert !GPUCompiler.isghosttype(RealRt)
            shadowV = extract_value!(B, res, idx)
            @assert llvmtype(shadowV) == shadowType
            shadowV = shadowV.ref
            idx+=1
        end
        if needsTape
            tapeV = extract_value!(B, res, idx).ref
            idx+=1
        end
    else
        if length(actives) >= 1 && !isa(llvmtype(res), LLVM.StructType) && !isa(llvmtype(res), LLVM.ArrayType)
            GPUCompiler.@safe_error "Shadow arg calling convention mismatch found return ", res
            return tapeV
        end
        Tys = Type[eltype(A) for A in activity if A <: Active]
        ST = Tuple{Tys...}
        if rev_RT != ST 
            emit_error(B, orig, "Enzyme: Reverse pass custom rule " * string(augprimal_TT) * " return type mismatch, expected "*string(ST)*" found "* string(rev_RT))
            return C_NULL
        end
 
        idx = 0
        for v in actives
            ext = extract_value!(B, res, idx)
            shadowVType = LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(v)))
            @assert llvmtype(ext) == shadowVType
            Typ = C_NULL
            API.EnzymeGradientUtilsAddToDiffe(gutils, v, ext, B, Typ)
            idx+=1
        end
    end

    if forward
        if shadowR != C_NULL
            unsafe_store!(shadowR, shadowV)
        end
        
        # Delete the primal code
        if needsPrimal
            unsafe_store!(normalR, normalV)
        else
            LLVM.API.LLVMInstructionEraseFromParent(LLVM.Instruction(API.EnzymeGradientUtilsNewFromOriginal(gutils, orig)))
        end
    end

    return tapeV
end


function enzyme_custom_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid 
    tape = enzyme_custom_common_rev(#=forward=#true, B, OrigCI, gutils, normalR, shadowR, #=tape=#nothing) 
    if tape != C_NULL
        unsafe_store!(tapeR, tape)
    end
    return nothing
end


function enzyme_custom_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    enzyme_custom_common_rev(#=forward=#false, B, OrigCI, gutils, #=normalR=#C_NULL, #=shadowR=#C_NULL, #=tape=#tape)
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
            API.moveBefore(prev, shadowres, B)
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
                    API.moveBefore(prev, callv, B)
                end
            end
            shadowres = insert_value!(B, shadowres, callv, idx-1)
        end
    end
 
    unsafe_store!(shadowR, shadowres.ref)
	
	return nothing
end

function arraycopy_common(fwd, B, orig, origArg, gutils, shadowdst)
    
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)

    activep = API.EnzymeGradientUtilsGetReturnDiffeType(gutils, orig, needsPrimalP, needsShadowP)
    needsPrimal = needsPrimalP[] != 0
    needsShadow = needsShadowP[] != 0
    if !needsShadow
        return nothing
    end

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
        emit_error(B, orig, "Enzyme: Unknown concrete type in arraycopy_common")
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

    # Zero the copy in the forward pass. 
    #   initshadow = 2.0
    #   dres = copy(initshadow) # 2.0
    #   
    #   This needs to be inserted
    #   memset(dres, 0, ...)
    #
    #   # removed return res[1]
    #   dres[1] += differeturn
    #   dmemcpy aka initshadow += dres
    algn = 0
    i8 = LLVM.IntType(8; ctx)

    if width == 1
    
    shadowsrc = get_array_data(B, shadowsrc)
    shadowdst = get_array_data(B, shadowdst)
   
    if fwd && secretty != nothing
        LLVM.memset!(B, shadowdst, LLVM.ConstantInt(i8, 0, false), length, algn)
    end

    API.sub_transfer(gutils, fwd ? API.DEM_ReverseModePrimal : API.DEM_ReverseModeGradient, secretty, intrinsic, #=dstAlign=#1, #=srcAlign=#1, #=offset=#0, false, shadowdst, false, shadowsrc, length, isVolatile, orig, allowForward, #=shadowsLookedUp=#!fwd)
    
    else
    for i in 1:width

    evsrc = extract_value!(B, shadowsrc, i-1)
    evdst = extract_value!(B, shadowdst, i-1)
    
    if fwd && secretty != nothing
        LLVM.memset!(B, shadowdst, LLVM.ConstantInt(i8, 0, false), length, algn)
    end

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

function boxfloat_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    origops = collect(operands(orig))
    width = API.EnzymeGradientUtilsGetWidth(gutils)
    if API.EnzymeGradientUtilsIsConstantValue(gutils, orig) == 0
        B = LLVM.Builder(B)
        flt = llvmtype(origops[1])
        shadowsin = LLVM.Value[
                               LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, origops[1], B))]
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

function boxfloat_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    origops = collect(operands(orig))
    width = API.EnzymeGradientUtilsGetWidth(gutils)
    if API.EnzymeGradientUtilsIsConstantValue(gutils, orig) == 0
        B = LLVM.Builder(B)
        
        flt = llvmtype(origops[1])
        TT = to_tape_type(flt)

        if width == 1
            obj = emit_allocobj!(B, TT)
            o2 = bitcast!(B, obj, LLVM.PointerType(flt, addrspace(llvmtype(obj))))
            store!(B, ConstantFP(flt, 0.0), o2)
            shadowres = obj
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, flt)))
            for idx in 1:width
                obj = emit_allocobj!(B, TT)
                o2 = bitcast!(B, obj, LLVM.PointerType(flt, addrspace(llvmtype(obj))))
                store!(B, ConstantFP(flt, 0.0), o2)
                shadowres = insert_value!(B, shadowres, obj, idx-1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return nothing
end

function boxfloat_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    origops = collect(operands(orig))
    width = API.EnzymeGradientUtilsGetWidth(gutils)
    if API.EnzymeGradientUtilsIsConstantValue(gutils, orig) == 0
        B = LLVM.Builder(B)
        ip = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, API.EnzymeGradientUtilsInvertPointer(gutils, orig, B), B))
        flt = llvmtype(origops[1])
        if width == 1
            ipc = bitcast!(B, ip, LLVM.PointerType(flt, addrspace(llvmtype(orig))))
            ld = load!(B, ipc)
            store!(B, ConstantFP(flt, 0.0), ipc)
            if API.EnzymeGradientUtilsIsConstantValue(gutils, origops[1]) == 0
                API.EnzymeGradientUtilsAddToDiffe(gutils, origops[1], ld, B, flt)
            end
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, flt)))
            for idx in 1:width
                ipc = extract_value!(B, ip, idx-1)
                ipc = bitcast!(B, ipc, LLVM.PointerType(flt, addrspace(llvmtype(orig))))
                ld = load!(B, ipc)
                store!(B, ConstantFP(flt, 0.0), ipc)
                shadowres = insert_value!(B, shadowres, ld, idx-1)
            end
            if API.EnzymeGradientUtilsIsConstantValue(gutils, origops[1]) == 0
                API.EnzymeGradientUtilsAddToDiffe(gutils, origops[1], shadowret, B, flt)
            end
        end
    end
    return nothing
end

function eqtableget_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    emit_error(LLVM.Builder(B), orig, "Enzyme: Not yet implemented forward for jl_eqtable_get")
    
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return nothing
end

function eqtableget_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    emit_error(LLVM.Builder(B), orig, "Enzyme: Not yet implemented augmented forward for jl_eqtable_get")

    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return nothing
end

function eqtableget_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    emit_error(LLVM.Builder(B), orig, "Enzyme: Not yet implemented reverse for jl_eqtable_get")
    return nothing
end

function eqtableput_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    emit_error(LLVM.Builder(B), orig, "Enzyme: Not yet implemented forward for jl_eqtable_put")
    
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return nothing
end

function eqtableput_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    emit_error(LLVM.Builder(B), orig, "Enzyme: Not yet implemented augmented forward for jl_eqtable_put")

    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return nothing
end

function eqtableput_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    emit_error(LLVM.Builder(B), orig, "Enzyme: Not yet implemented reverse for jl_eqtable_put")
    return nothing
end


function idtablerehash_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    emit_error(LLVM.Builder(B), orig, "Enzyme: Not yet implemented forward for jl_idtable_rehash")
    
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return nothing
end

function idtablerehash_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    emit_error(LLVM.Builder(B), orig, "Enzyme: Not yet implemented augmented forward for jl_idtable_rehash")

    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return nothing
end

function idtablerehash_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    emit_error(LLVM.Builder(B), orig, "Enzyme: Not yet implemented reverse for jl_idtable_rehash")
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
        delF = get_function!(mod, "jl_array_grow_end", funcT)

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
        emit_error(B, orig, "Not applying memset on reverse of jl_array_del_end")
        # memset(data + idx * elsz, 0, inc * elsz);
    end
    return nothing
end

function jl_array_ptr_copy_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    origops = collect(operands(orig))
    width = API.EnzymeGradientUtilsGetWidth(gutils)
    if API.EnzymeGradientUtilsIsConstantInstruction(gutils, orig) == 0
        B = LLVM.Builder(B)
        origops = collect(operands(orig))
        width = API.EnzymeGradientUtilsGetWidth(gutils)
            
        args = LLVM.Value[]
        for a in origops[1:end-2]
            v = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, a, B))
            push!(args, v)
        end
        push!(args, LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[end-1])))
        valTys = API.CValueType[API.VT_Shadow, API.VT_Shadow, API.VT_Shadow, API.VT_Shadow, API.VT_Primal]

        if width == 1
            vargs = args
            cal = API.EnzymeGradientUtilsCallWithInvertedBundles(gutils, LLVM.called_value(orig), vargs, length(vargs), orig, valTys, length(valTys), B, #=lookup=#false)
            API.EnzymeGradientUtilsSetDebugLocFromOriginal(gutils, cal, orig)
            conv = LLVM.API.LLVMGetInstructionCallConv(orig)
            LLVM.API.LLVMSetInstructionCallConv(cal, conv)
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig))))
            for idx in 1:width
                vargs = LLVM.Value[]
                for a in args[1:end-1]
                    push!(vargs, extract_value!(B, a, idx-1))
                end
                push!(vargs, args[end])
                cal = API.EnzymeGradientUtilsCallWithInvertedBundles(gutils, LLVM.called_value(orig), vargs, length(vargs), orig, valTys, length(valTys), B, #=lookup=#false)
                API.EnzymeGradientUtilsSetDebugLocFromOriginal(gutils, cal, orig)
                conv = LLVM.API.LLVMGetInstructionCallConv(orig)
                LLVM.API.LLVMSetInstructionCallConv(cal, conv)
            end
        end

    end
    return nothing
end
function jl_array_ptr_copy_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
  jl_array_ptr_copy_fwd(B, OrigCI, gutils, normalR, shadowR)
end
function jl_array_ptr_copy_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid 
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
    orig = LLVM.Instruction(OrigCI)
    emit_error(LLVM.Builder(B), orig, "Enzyme: unhandled augmented forward for jl_f_setfield")
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return nothing
end

function setfield_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
  orig = LLVM.Instruction(OrigCI)
  emit_error(LLVM.Builder(B), orig, "Enzyme: unhandled reverse for jl_f_setfield")
  return nothing
end

function get_binding_or_error_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    CI = API.EnzymeGradientUtilsNewFromOriginal(gutils, OrigCI)
    orig = LLVM.Instruction(OrigCI)
    err = emit_error(LLVM.Builder(B), orig, "Enzyme: unhandled forward for jl_get_binding_or_error")
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
    orig = LLVM.Instruction(OrigCI)
    err = emit_error(LLVM.Builder(B), orig, "Enzyme: unhandled augmented forward for jl_get_binding_or_error")
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
    orig = LLVM.Instruction(OrigCI)
    emit_error(LLVM.Builder(B), orig, "Enzyme: unhandled reverse for jl_get_binding_or_error")
    return nothing
end

function finalizer_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    CI = API.EnzymeGradientUtilsNewFromOriginal(gutils, OrigCI)
    orig = LLVM.Instruction(OrigCI)
    err = emit_error(LLVM.Builder(B), orig, "Enzyme: unhandled augmented forward for jl_gc_add_finalizer_th or jl_gc_add_ptr_finalizer")
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
    # err = emit_error(LLVM.Builder(B), orig, "Enzyme: unhandled augmented forward for jl_gc_add_finalizer_th")
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
    # emit_error(LLVM.Builder(B), orig, "Enzyme: unhandled reverse for jl_gc_add_finalizer_th")
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

struct IllegalFirstPointerException <: CompilationException
    msg::String
    ir::Union{Nothing, String}
    bt::Union{Nothing, Vector{StackTraces.StackFrame}}
end

function Base.showerror(io::IO, ece::IllegalFirstPointerException)
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

struct EnzymeInternalError <: CompilationException
    msg::String
    ir::Union{Nothing, String}
    bt::Union{Nothing, Vector{StackTraces.StackFrame}}
end

function Base.showerror(io::IO, ece::EnzymeInternalError)
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

parent_scope(val::LLVM.Function, depth=0) = depth==0 ? LLVM.parent(val) : val
parent_scope(val::LLVM.Module, depth=0) = val
parent_scope(val::LLVM.Value, depth=0) = parent_scope(LLVM.parent(val), depth+1)
parent_scope(val::LLVM.Argument, depth=0) = parent_scope(LLVM.Function(LLVM.API.LLVMGetParamParent(val)), depth+1)

function julia_error(cstr::Cstring, val::LLVM.API.LLVMValueRef, errtype::API.ErrorType, data::Ptr{Cvoid})
    msg = Base.unsafe_string(cstr)
    bt = nothing
    ir = nothing
    if val != C_NULL
        val = LLVM.Value(val)
        if isa(val, LLVM.Instruction)
            bt = GPUCompiler.backtrace(val)
        end
        if !isa(val, LLVM.ConstantExpr)
            # Need to convert function to string, since when the error is going to be printed
            # the module might have been destroyed
            ir = sprint(io->show(io, parent_scope(val)))
        end
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
    elseif errtype == API.ET_IllegalFirstPointer
        throw(IllegalFirstPointerException(msg, ir, bt))
    elseif errtype == API.ET_InternalError
        throw(EnzymeInternalError(msg, ir, bt))
    elseif errtype == API.ET_TypeDepthExceeded
        function c(io)
            print(io, msg)
            println(io)

            if val != C_NULL
                println(io, val)
            end
        
            st = API.EnzymeTypeTreeToString(data)
            println(io, Base.unsafe_string(st))
            API.EnzymeStringFree(st)

            if bt !== nothing
                Base.show_backtrace(io, bt)
            end
        end
        msg2 = sprint(c)
        GPUCompiler.@safe_warn msg2
        return
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
any_jltypes(::LLVM.VoidType) = false

@inline any_jltypes(::Type{Nothing}) = false
@inline any_jltypes(::Type{T}) where {T<:AbstractFloat} = false
@inline any_jltypes(::Type{T}) where {T<:Integer} = false
@inline any_jltypes(::Type{Complex{T}}) where T = any_jltypes(T)
@inline any_jltypes(::Type{NTuple{Size, T}}) where {Size, T} = any_jltypes(T)
@inline any_jltypes(::Type{Core.LLVMPtr{T, Addr}}) where {T, Addr} = 10 <= Addr <= 12
@inline any_jltypes(::Type{Any}) = true
@inline any_jltypes(::Type{NamedTuple{A,B}}) where {A,B} = any(any_jltypes(b) for b in B.parameters)

nfields(Type::LLVM.StructType) = length(LLVM.elements(Type))
nfields(Type::LLVM.VectorType) = size(Type)
nfields(Type::LLVM.ArrayType) = length(Type)
nfields(Type::LLVM.PointerType) = 1

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
    if TT == Any
        return AnonymousStruct(Tuple{Any})
    end
    return TT
end

from_tape_type(::Type{T}, ctx) where T<:AbstractFloat = convert(LLVMType, T; ctx)
from_tape_type(::Type{T}, ctx) where T<:Integer = convert(LLVMType, T; ctx)
from_tape_type(::Type{NTuple{Size, T}}, ctx) where {Size, T} = LLVM.ArrayType(from_tape_type(T, ctx), Size)
from_tape_type(::Type{Core.LLVMPtr{T, Addr}}, ctx) where {T, Addr} = LLVM.PointerType(from_tape_type(UInt8, ctx), Addr)
# from_tape_type(::Type{Core.LLVMPtr{T, Addr}}, ctx) where {T, Addr} = LLVM.PointerType(from_tape_type(T, ctx), Addr)
from_tape_type(::Type{Any}, ctx) = LLVM.PointerType(LLVM.StructType(LLVM.LLVMType[]; ctx), 10)
function from_tape_type(::Type{NamedTuple{A,B}}, ctx) where {A,B}
    if length(B.parameters) >= 1 && all(B.parameters[1] == b for b in B.parameters)
        return LLVM.ArrayType(from_tape_type(B.parameters[1], ctx), length(B.parameters))
    else
        return LLVM.StructType(LLVM.LLVMType[from_tape_type(b, ctx) for b in B.parameters]; ctx)
    end
end
const Tracked = 10

# TODO: Calculate that constant... see get_current_task
if VERSION >= v"1.9.0-"
  current_task_offset() = -13
else
  current_task_offset() = -12
end
current_ptls_offset() = 14

function get_julia_inner_types(B, p, startvals...; added=[])
    ctx = LLVM.context(p)
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, 10) 
    vals = LLVM.Value[p]
    todo = LLVM.Value[startvals...]
    while length(todo) != 0
        cur = popfirst!(todo)
        ty = llvmtype(cur)
        if isa(ty, LLVM.PointerType)
            if any_jltypes(ty)
                if addrspace(ty) != 10
                    cur = addrspacecast!(B, cur, LLVM.PointerType(eltype(ty), 10))
                    push!(added, cur.ref)
                end
                if llvmtype(cur) != T_prjlvalue
                    cur = bitcast!(B, cur, T_prjlvalue)
                    push!(added, cur.ref)
                end
                push!(vals, cur)
            end
            continue
        end
        if isa(ty, LLVM.ArrayType)
            if any_jltypes(ty)
                for i=1:length(ty)
                    ev = extract_value!(B, cur, i-1)
                    push!(added, ev.ref)
                    push!(todo, ev)
                end
            end
            continue
        end
        if isa(ty, LLVM.StructType)
            for (i, t) in enumerate(LLVM.elements(ty))
                if any_jltypes(t)
                    ev = extract_value!(B, cur, i-1)
                    push!(added, ev.ref)
                    push!(todo, ev)
                end
            end
            continue
        end
        GPUCompiler.@safe_warn "Enzyme illegal subtype", ty, cur, SI, p, v
        @assert false
    end
    return vals
end

function julia_post_cache_store(SI::LLVM.API.LLVMValueRef, B::LLVM.API.LLVMBuilderRef, R2)::Ptr{LLVM.API.LLVMValueRef}
    B = LLVM.Builder(B)
    SI = LLVM.Instruction(SI)
    v = operands(SI)[1]
    p = operands(SI)[2]
    added = LLVM.API.LLVMValueRef[]
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
        push!(added, p.ref)
        
        vals = get_julia_inner_types(B, p, v, added=added)
        r = emit_writebarrier!(B, vals)
        push!(added, r.ref)
    end
    if R2 != C_NULL
        unsafe_store!(R2, length(added))
        ptr = Base.unsafe_convert(Ptr{LLVM.API.LLVMValueRef}, Libc.malloc(sizeof(LLVM.API.LLVMValueRef)*length(added)))
        for (i, v) in enumerate(added)
            unsafe_store!(ptr, v, i)
        end
        return ptr
    end
    return C_NULL
end

function julia_default_tape_type(C::LLVM.API.LLVMContextRef)
    ctx = LLVM.Context(C)
    T_jlvalue = LLVM.StructType(LLVM.LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    return T_prjlvalue.ref
end

function julia_allocator(B::LLVM.API.LLVMBuilderRef, LLVMType::LLVM.API.LLVMTypeRef, Count::LLVM.API.LLVMValueRef, AlignedSize::LLVM.API.LLVMValueRef, IsDefault::UInt8, ZI)
    B = LLVM.Builder(B)
    Count = LLVM.Value(Count)
    AlignedSize = LLVM.Value(AlignedSize)
    LLVMType = LLVM.LLVMType(LLVMType)
    return julia_allocator(B, LLVMType, Count, AlignedSize, IsDefault, ZI)
end

function fixup_return(B, retval)
    B = LLVM.Builder(B)
    
    func = LLVM.parent(position(B))
    mod = LLVM.parent(func)
    ctx = context(mod)
    T_jlvalue = LLVM.StructType(LLVM.LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_prjlvalue_UT = LLVM.PointerType(T_jlvalue)
    
    retval = LLVM.Value(retval)
    ty = llvmtype(retval)
    # Special case the union return { {} addr(10)*, i8 }
    #   which can be [ null, 1 ], to not have null in the ptr
    #   field, but nothing
    if isa(ty, LLVM.StructType)
        elems = LLVM.elements(ty)
        if length(elems) == 2 && elems[1] == T_prjlvalue
            fill_val = unsafe_to_llvm(nothing, ctx)
            prev = extract_value!(B, retval, 0)
            eq = icmp!(B, LLVM.API.LLVMIntEQ, prev, LLVM.null(T_prjlvalue))
            retval = select!(B, eq, insert_value!(B, retval, fill_val, 0), retval)
        end
    end
    return retval.ref
end

function zero_allocation(B, LLVMType, obj, isTape::UInt8)
    B = LLVM.Builder(B)
    LLVMType = LLVM.LLVMType(LLVMType)
    obj = LLVM.Value(obj)
    jlType = tape_type(LLVMType)
    zeroAll = isTape == 0
    func = LLVM.parent(position(B))
    mod = LLVM.parent(func)
    ctx = context(mod)
    T_int64 = LLVM.Int64Type(ctx)
    zero_single_allocation(B, jlType, LLVMType, obj, zeroAll, LLVM.ConstantInt(T_int64, 0), ctx)
    return nothing
end

function zero_single_allocation(builder, jlType, LLVMType, nobj, zeroAll, idx, ctx)
    T_jlvalue = LLVM.StructType(LLVM.LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_prjlvalue_UT = LLVM.PointerType(T_jlvalue)

        todo = Tuple{Vector{LLVM.Value},LLVM.LLVMType,DataType}[(LLVM.Value[idx], LLVMType, jlType)]

        while length(todo) != 0
            path, ty, jlty = popfirst!(todo)
            if isa(ty, LLVM.PointerType)
                if any_jltypes(ty) 
                    loc = gep!(builder, nobj, path)
                    fill_val = unsafe_to_llvm(nothing, ctx)
                    loc = bitcast!(builder, loc, LLVM.PointerType(T_prjlvalue, addrspace(llvmtype(loc))))
                    store!(builder, fill_val, loc)
                elseif zeroAll
                    loc = gep!(builder, nobj, path)
                    store!(builder, LLVM.null(ty), loc)
                end
                continue
            end
            if isa(ty, LLVM.FloatingPointType) || isa(ty, LLVM.IntegerType)
                if zeroAll
                    loc = gep!(builder, nobj, path)
                    store!(builder, LLVM.null(ty), loc)
                end
                continue
            end
            if isa(ty, LLVM.ArrayType) || isa(ty, LLVM.VectorType)
                for i=1:length(ty)
                    npath = copy(path)
                    push!(npath, LLVM.ConstantInt(LLVM.IntType(32; ctx), i-1))
                    push!(todo, (npath, eltype(ty), eltype(jlty)))
                end
                continue
            end
            if isa(ty, LLVM.StructType)
                i = 1
                for ii in 1:fieldcount(jlty)
                    jlet = fieldtype(jlty, ii)
                    if GPUCompiler.isghosttype(jlet) || Core.Compiler.isconstType(jlet)
                        continue
                    end
                    t = LLVM.elements(ty)[i]
                    npath = copy(path)
                    push!(npath, LLVM.ConstantInt(LLVM.IntType(32; ctx), i-1))
                    push!(todo, (npath, t, jlet))
                    i+=1
                end
                @assert i == Int(length(LLVM.elements(ty)))+1
                continue
            end
        end
    return nothing

end


function zero_allocation(B::LLVM.Builder, jlType, LLVMType, obj, AlignedSize, Size, zeroAll::Bool)::LLVM.API.LLVMValueRef
    func = LLVM.parent(position(B))
    mod = LLVM.parent(func)
    ctx = context(mod)
    T_int8 = LLVM.Int8Type(ctx)
        
    T_jlvalue = LLVM.StructType(LLVM.LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_prjlvalue_UT = LLVM.PointerType(T_jlvalue)

    wrapper_f = LLVM.Function(mod, "zeroType", LLVM.FunctionType(LLVM.VoidType(ctx), [llvmtype(obj), T_int8, llvmtype(Size)]))
    push!(function_attributes(wrapper_f), EnumAttribute("alwaysinline", 0; ctx))
    linkage!(wrapper_f, LLVM.API.LLVMInternalLinkage)
    let builder = Builder(ctx)
        entry = BasicBlock(wrapper_f, "entry"; ctx)
        loop = BasicBlock(wrapper_f, "loop"; ctx)
        exit = BasicBlock(wrapper_f, "exit"; ctx)
        position!(builder, entry)
        nobj, _, nsize = collect(parameters(wrapper_f))
        nobj = pointercast!(builder, nobj, LLVM.PointerType(LLVMType, addrspace(llvmtype(nobj))))

        LLVM.br!(builder, loop)
        position!(builder, loop)
        idx = LLVM.phi!(builder, llvmtype(Size))
        inc = add!(builder, idx, LLVM.ConstantInt(llvmtype(Size), 1))
        append!(LLVM.incoming(idx), [(LLVM.ConstantInt(llvmtype(Size), 0), entry), (inc, loop)])
    
        zero_single_allocation(builder, jlType, LLVMType, nobj, zeroAll, idx, ctx)
        
        br!(builder, icmp!(builder, LLVM.API.LLVMIntEQ, inc, LLVM.Value(LLVM.API.LLVMBuildExactUDiv(builder, nsize, AlignedSize, ""))), exit, loop)
        position!(builder, exit)
        
        ret!(builder)

        dispose(builder)
    end
    return call!(B, wrapper_f, [obj, LLVM.ConstantInt(T_int8, 0), Size]).ref
end

function julia_allocator(B, LLVMType, Count, AlignedSize, IsDefault, ZI)
    func = LLVM.parent(position(B))
    mod = LLVM.parent(func)
    ctx = context(mod)

    Size = nuwmul!(B, Count, AlignedSize) # should be nsw, nuw
    T_int8 = LLVM.Int8Type(ctx)

    if any_jltypes(LLVMType) || IsDefault != 0
        T_int64 = LLVM.Int64Type(ctx)
        T_jlvalue = LLVM.StructType(LLVM.LLVMType[]; ctx)
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
        T_pint8 = LLVM.PointerType(T_int8)
        T_ppint8 = LLVM.PointerType(T_pint8)

        esizeof(X) = X == Any ? sizeof(Int) : sizeof(X)

        TT = tape_type(LLVMType)
        if esizeof(TT) != convert(Int, AlignedSize)
            GPUCompiler.@safe_error "Enzyme aligned size and Julia size disagree" AlignedSize=convert(Int, AlignedSize) esizeof(TT) fieldtypes(TT)
            emit_error(B, nothing, "Enzyme: Tape allocation failed.") # TODO: Pick appropriate orig
            return LLVM.API.LLVMValueRef(LLVM.UndefValue(LLVMType).ref)
        end
        @assert esizeof(TT) == convert(Int, AlignedSize)
        if Count isa LLVM.ConstantInt
            N = convert(Int, Count)

            ETT = N == 1 ? TT : NTuple{N, TT}
            if sizeof(ETT) !=  N*convert(Int, AlignedSize)
                GPUCompiler.@safe_error "Size of Enzyme tape is incorrect. Please report this issue" ETT sizeof(ETT) TargetSize = N*convert(Int, AlignedSize)
                emit_error(B, nothing, "Enzyme: Tape allocation failed.") # TODO: Pick appropriate orig

                return LLVM.API.LLVMValueRef(LLVM.UndefValue(LLVMType).ref)
            end

            # Obtain tag
            tag = unsafe_to_llvm(ETT, ctx)
        else
            boxed_count = emit_box_int64!(B, Count)
            tag = emit_apply_type!(B, NTuple, (boxed_count, unsafe_to_llvm(TT, ctx)))
        end

        # Check if Julia version has https://github.com/JuliaLang/julia/pull/46914
        # and also https://github.com/JuliaLang/julia/pull/47076
        # and also https://github.com/JuliaLang/julia/pull/48620
        @static if VERSION >= v"1.11.0-DEV.1571"
            needs_dynamic_size_workaround = false
        else
            needs_dynamic_size_workaround = !isa(Size, LLVM.ConstantInt) || convert(Int64, Size) != 1
        end
        
        obj = emit_allocobj!(B, tag, Size, needs_dynamic_size_workaround)
        
        if ZI != C_NULL
            unsafe_store!(ZI, zero_allocation(B, TT, LLVMType, obj, AlignedSize, Size, #=ZeroAll=#false))
        end
        AS = Tracked
    else
        ptr8 = LLVM.PointerType(LLVM.IntType(8; ctx))
        mallocF = get_function!(mod, "malloc", LLVM.FunctionType(ptr8, [llvmtype(Count)]))

        obj = call!(B, mallocF, [Size])
        if ZI != C_NULL
            unsafe_store!(ZI, LLVM.memset!(B, obj,  LLVM.ConstantInt(T_int8, 0),
                                                  Size, 
                                                 #=align=#0 ).ref)
        end
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
        API.EnzymeSetDefaultTapeType(@cfunction(
                                                julia_default_tape_type, LLVM.API.LLVMTypeRef, (LLVM.API.LLVMContextRef,)))
        API.EnzymeSetCustomAllocator(@cfunction(
            julia_allocator, LLVM.API.LLVMValueRef,
            (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMTypeRef, LLVM.API.LLVMValueRef, LLVM.API.LLVMValueRef, UInt8, Ptr{LLVM.API.LLVMValueRef})))
        API.EnzymeSetCustomDeallocator(@cfunction(
            julia_deallocator, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef)))
        API.EnzymeSetPostCacheStore(@cfunction(
             julia_post_cache_store, Ptr{LLVM.API.LLVMValueRef},
            (LLVM.API.LLVMValueRef, LLVM.API.LLVMBuilderRef, Ptr{UInt64})))
        
        API.EnzymeSetCustomZero(@cfunction(
            zero_allocation, Cvoid,
            (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMTypeRef, LLVM.API.LLVMValueRef, UInt8)))
        API.EnzymeSetFixupReturn(@cfunction(
            fixup_return, LLVM.API.LLVMValueRef,
            (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef)))
    end
    register_alloc_handler!(
        ("jl_alloc_array_1d", "ijl_alloc_array_1d"),
        @cfunction(array_shadow_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, Csize_t, Ptr{LLVM.API.LLVMValueRef}, API.EnzymeGradientUtilsRef)),
        @cfunction(null_free_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, LLVM.API.LLVMValueRef))
    )
    register_alloc_handler!(
        ("jl_alloc_array_2d", "ijl_alloc_array_2d"),
        @cfunction(array_shadow_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, Csize_t, Ptr{LLVM.API.LLVMValueRef}, API.EnzymeGradientUtilsRef)),
        @cfunction(null_free_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, LLVM.API.LLVMValueRef))
    )
    register_alloc_handler!(
        ("jl_alloc_array_3d", "ijl_alloc_array_3d"),
        @cfunction(array_shadow_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, Csize_t, Ptr{LLVM.API.LLVMValueRef}, API.EnzymeGradientUtilsRef)),
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
        ("enzyme_custom",),
        @cfunction(enzyme_custom_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(enzyme_custom_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(enzyme_custom_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}))
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
        ("jl_f_setfield","ijl_f_setfield"),
        @cfunction(setfield_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(setfield_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(setfield_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_box_float32","ijl_box_float32", "jl_box_float64", "ijl_box_float64"),
        @cfunction(boxfloat_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(boxfloat_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(boxfloat_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_f_tuple","ijl_f_tuple"),
        @cfunction(f_tuple_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(f_tuple_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(f_tuple_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_eqtable_get","ijl_eqtable_get"),
        @cfunction(eqtableget_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(eqtableget_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(eqtableget_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_eqtable_put","ijl_eqtable_put"),
        @cfunction(eqtableput_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(eqtableput_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(eqtableput_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_idtable_rehash","ijl_idtable_rehash"),
        @cfunction(idtablerehash_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(idtablerehash_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(idtablerehash_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_f__apply_iterate","ijl_f__apply_iterate"),
        @cfunction(apply_iterate_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(apply_iterate_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(apply_iterate_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_new_structv","ijl_new_structv"),
        @cfunction(new_structv_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(new_structv_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(new_structv_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_get_binding_or_error", "ijl_get_binding_or_error"),
        @cfunction(get_binding_or_error_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(get_binding_or_error_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(get_binding_or_error_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_gc_add_finalizer_th","ijl_gc_add_finalizer_th"),
        @cfunction(finalizer_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(finalizer_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(finalizer_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_gc_add_ptr_finalizer","ijl_gc_add_ptr_finalizer"),
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
        ("jl_f_getfield","ijl_f_getfield"),
        @cfunction(jl_getfield_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(jl_getfield_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(jl_getfield_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("ijl_get_nth_field_checked","jl_get_nth_field_checked"),
        @cfunction(jl_nthfield_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(jl_nthfield_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(jl_nthfield_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_array_sizehint","ijl_array_sizehint"),
        @cfunction(jl_array_sizehint_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(jl_array_sizehint_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(jl_array_sizehint_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_array_ptr_copy","ijl_array_ptr_copy"),
        @cfunction(jl_array_ptr_copy_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(jl_array_ptr_copy_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(jl_array_ptr_copy_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
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
    # Whether to (in aug fwd) += by one
    shadowInit::Bool
end

struct PrimalCompilerParams <: AbstractEnzymeCompilerParams
    mode::API.CDerivativeMode
end

DefaultCompilerTarget(;kwargs...) = GPUCompiler.NativeCompilerTarget(;jlruntime=true, kwargs...)

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
    Interpreter.EnzymeInterpreter(GPUCompiler.ci_cache(job), GPUCompiler.method_table(job), job.source.world, job.params.mode)

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

    for fname in inactivefns
        if haskey(fns, fname)
            fn = fns[fname]
            push!(function_attributes(fn), inactive)
            for u in LLVM.uses(fn)
                c = LLVM.user(u)
                if !isa(c, LLVM.CallInst)
                    continue
                end
                cf = LLVM.called_value(c)
                if !isa(cf, LLVM.Function)
                    continue
                end
                if LLVM.name(cf) != "julia.call" && LLVM.name(cf) != "julia.call2"
                    continue
                end
                if operands(c)[1] != fn
                    continue
                end
                LLVM.API.LLVMAddCallSiteAttribute(c, LLVM.API.LLVMAttributeFunctionIndex, inactive)
            end            
        end
    end
    
    for fname in nofreefns
        if haskey(fns, fname)
            fn = fns[fname]
            push!(function_attributes(fn), LLVM.EnumAttribute("nofree", 0; ctx))
        end
    end

    for fname in activefns
        if haskey(fns, fname)
            fn = fns[fname]
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

    for fname in ("jl_excstack_state","ijl_excstack_state")
        if haskey(fns, fname)
            fn = fns[fname]
            push!(function_attributes(fn), LLVM.EnumAttribute("readonly", 0; ctx))
            push!(function_attributes(fn), LLVM.StringAttribute("inaccessiblememonly"; ctx))
        end
    end

    for fname in ("jl_types_equal", "ijl_types_equal")
        if haskey(fns, fname)
            fn = fns[fname]
            push!(function_attributes(fn), LLVM.EnumAttribute("readonly", 0; ctx))
            push!(function_attributes(fn), LLVM.StringAttribute("enzyme_shouldrecompute"; ctx))
        end
    end

    for fname in ("jl_f_getfield","ijl_f_getfield","jl_get_nth_field_checked","ijl_get_nth_field_checked")
        if haskey(fns, fname)
            fn = fns[fname]
            push!(function_attributes(fn), LLVM.EnumAttribute("readonly", 0; ctx))
            for u in LLVM.uses(fn)
                c = LLVM.user(u)
                if !isa(c, LLVM.CallInst)
                    continue
                end
                cf = LLVM.called_value(c)
                if !isa(cf, LLVM.Function)
                    continue
                end
                if LLVM.name(cf) != "julia.call" && LLVM.name(cf) != "julia.call2"
                    continue
                end
                if operands(c)[1] != fn
                    continue
                end
                LLVM.API.LLVMAddCallSiteAttribute(c, LLVM.API.LLVMAttributeFunctionIndex, LLVM.EnumAttribute("readonly", 0; ctx))
            end     
        end
    end

    for fname in ("julia.get_pgcstack", "julia.ptls_states", "jl_get_ptls_states")
        if haskey(fns, fname)
            fn = fns[fname]
            # TODO per discussion w keno perhaps this should change to readonly / inaccessiblememonly
            push!(function_attributes(fn), LLVM.EnumAttribute("readnone", 0; ctx))
            push!(function_attributes(fn), LLVM.StringAttribute("enzyme_shouldrecompute"; ctx))
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
                  "jl_array_copy", "ijl_array_copy",
                  "jl_f_tuple", "ijl_f_tuple", "jl_new_structv", "ijl_new_structv")
        if haskey(fns, boxfn)
            fn = fns[boxfn]
            push!(return_attributes(fn), LLVM.EnumAttribute("noalias", 0; ctx))
            push!(function_attributes(fn), LLVM.EnumAttribute("inaccessiblememonly", 0; ctx))
            for u in LLVM.uses(fn)
                c = LLVM.user(u)
                if !isa(c, LLVM.CallInst)
                    continue
                end
                cf = LLVM.called_value(c)
                if !isa(cf, LLVM.Function)
                    continue
                end
                if LLVM.name(cf) != "julia.call" && LLVM.name(cf) != "julia.call2"
                    continue
                end
                if operands(c)[1] != fn
                    continue
                end
                LLVM.API.LLVMAddCallSiteAttribute(c, LLVM.API.LLVMAttributeReturnIndex, LLVM.EnumAttribute("noalias", 0; ctx))
                LLVM.API.LLVMAddCallSiteAttribute(c, LLVM.API.LLVMAttributeFunctionIndex, LLVM.EnumAttribute("inaccessiblememonly", 0; ctx))
            end            
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
    API.EnzymeMergeTypeTree(ret, TT)
    return UInt8(false)
end

function i64_box_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    TT = TypeTree(API.DT_Pointer, LLVM.context(LLVM.Value(val)))
    only!(TT, -1)
    API.EnzymeMergeTypeTree(ret, TT)
    return UInt8(false)
end


function f32_box_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    TT = TypeTree(API.DT_Float, LLVM.context(LLVM.Value(val)))
    only!(TT, -1)
    API.EnzymeMergeTypeTree(unsafe_load(args), TT)
    
    API.EnzymeMergeTypeTree(TT, TypeTree(API.DT_Pointer,LLVM.context(LLVM.Value(val))))
    only!(TT, -1)
    API.EnzymeMergeTypeTree(ret, TT)
    return UInt8(false)
end

function ptr_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    TT = TypeTree(API.DT_Pointer, LLVM.context(LLVM.Value(val)))
    only!(TT, -1)
    API.EnzymeSetTypeTree(ret, TT)
    return UInt8(false)
end

function inout_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    inst = LLVM.Instruction(val)
    ce = operands(inst)[1]
    while isa(ce, ConstantExpr)
        ce = operands(ce)[1]
    end
    if isa(ce, ConstantInt)
        if (direction & API.DOWN) != 0
            ptr = reinterpret(Ptr{Cvoid}, convert(UInt64, ce))
            typ = Base.unsafe_pointer_to_objref(ptr)
            ctx = LLVM.context(LLVM.Value(val))
            dl = string(LLVM.datalayout(LLVM.parent(LLVM.parent(LLVM.parent(inst)))))
            typ2 = Core.Typeof(typ)
            rest = typetree(typ2, ctx, dl)
            if GPUCompiler.deserves_retbox(typ2)
                merge!(rest, TypeTree(API.DT_Pointer, ctx))
                only!(rest, -1)
            end
            API.EnzymeMergeTypeTree(ret, rest)
        end
        return UInt8(false)
    end

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

function enzyme_extract_world(fn::LLVM.Function)::UInt64
    for fattr in collect(function_attributes(fn))
        if isa(fattr, LLVM.StringAttribute)
            if kind(fattr) == "enzymejl_world"
                return parse(UInt64, LLVM.value(fattr))
            end
        end
    end
    GPUCompiler.@safe_error "Enzyme: Could not find world", fn
end

function enzyme_custom_extract_mi(orig::LLVM.Instruction)
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
        PTT = unsafe_load(args, op_idx)
        changed, legal = API.EnzymeCheckedMergeTypeTree(PTT, rest)
        if !legal
            function c(io)
                println(io, "Illegal type analysis update from julia rule of method ", mi)
                println(io, "Found type ", arg.typ, " at index ", arg.codegen.i, " of ", string(rest))
                t = API.EnzymeTypeTreeToString(PTT)
                println(io, "Prior type ", Base.unsafe_string(t))
                println(io, inst)
                API.EnzymeStringFree(t)
            end
            msg = sprint(c)
            
            bt = GPUCompiler.backtrace(inst)
            ir = sprint(io->show(io, parent_scope(inst)))
        
            sval = ""
            # data = API.EnzymeTypeAnalyzerRef(data)
            # ip = API.EnzymeTypeAnalyzerToString(data)
            # sval = Base.unsafe_string(ip)
            # API.EnzymeStringFree(ip)
            throw(IllegalTypeAnalysisException(msg, sval, ir, bt))
        end
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
    shadow_init = job.params.shadowInit
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
        "ijl_excstack_state" => @cfunction(int_return_rule,
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
          augmented_primalf = create_abi_wrapper(augmented_primalf, F, tt, rt, actualRetType, API.DEM_ReverseModePrimal, augmented, dupClosure, width, returnUsed, shadow_init)
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
          adjointf = create_abi_wrapper(adjointf, F, tt, rt, actualRetType, API.DEM_ReverseModeGradient, augmented, dupClosure, width, #=returnPrimal=#false, shadow_init)
        end
    elseif mode == API.DEM_ReverseModeCombined
        returnUsed = !GPUCompiler.isghosttype(actualRetType)
        returnUsed &= returnPrimal
        adjointf = LLVM.Function(API.EnzymeCreatePrimalAndGradient(
            logic, primalf, retType, args_activity, TA,
            #=returnValue=#returnUsed, #=dretUsed=#false, #=mode=#API.DEM_ReverseModeCombined, width,
            #=additionalArg=#C_NULL, typeInfo,
            uncacheable_args, #=augmented=#C_NULL, #=atomicAdd=# parallel))
        augmented_primalf = nothing
        if wrap
          adjointf = create_abi_wrapper(adjointf, F, tt, rt, actualRetType, API.DEM_ReverseModeCombined, nothing, dupClosure, width, returnUsed, shadow_init)
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
          adjointf = create_abi_wrapper(adjointf, F, tt, rt, actualRetType, API.DEM_ForwardMode, nothing, dupClosure, width, returnUsed, shadow_init)
        end
    else
        @assert "Unhandled derivative mode", mode
    end
    API.EnzymeLogicErasePreprocessedFunctions(logic)
    fix_decayaddr!(mod)
    return adjointf, augmented_primalf, TapeType
end

function create_abi_wrapper(enzymefn::LLVM.Function, F, argtypes, rettype, actualRetType, Mode::API.CDerivativeMode, augmented, dupClosure, width, returnPrimal, shadow_init)
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

    sret_types  = Type[]  # Julia types of all returned variables
    
    # Number of sret values from Enzyme calling convention
    count_Sret = 0

    pactualRetType = actualRetType
    sret_union = is_sret_union(actualRetType)
    if sret_union
        actualRetType = Any
    end

    if !GPUCompiler.isghosttype(F) && !Core.Compiler.isconstType(F)
        isboxed = GPUCompiler.deserves_argbox(F)
        llvmT = isboxed ? T_prjlvalue : convert(LLVMType, F; ctx)
        push!(T_wrapperargs, llvmT)
        if dupClosure
            push!(T_wrapperargs, llvmT)
        end
    end

    ActiveRetTypes = Type[]
    for (i, T) in enumerate(argtypes)
        source_typ = eltype(T)
        if GPUCompiler.isghosttype(source_typ) || Core.Compiler.isconstType(source_typ)
            @assert T <: Const
            continue
        end

        isboxed = GPUCompiler.deserves_argbox(source_typ)
        llvmT = isboxed ? T_prjlvalue : convert(LLVMType, source_typ; ctx)

        push!(T_wrapperargs, llvmT)

        if T <: Const
            if is_adjoint
                push!(ActiveRetTypes, Nothing)
            end
            continue
        end

        if T <: Active
            if is_adjoint
                count_Sret += 1
                if width == 1
                    push!(ActiveRetTypes, source_typ)
                else
                    push!(ActiveRetTypes, NTuple{width, source_typ})
                end
            end
        elseif T <: Duplicated || T <: DuplicatedNoNeed
            @assert width == 1
            push!(T_wrapperargs, LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmT)))
            if is_adjoint
                push!(ActiveRetTypes, Nothing)
            end
        elseif T <: BatchDuplicated || T <: BatchDuplicatedNoNeed
            push!(T_wrapperargs, LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmT)))
            if is_adjoint
                push!(ActiveRetTypes, Nothing)
            end
        else
            error("calling convention should be annotated, got $T")
        end
    end

    if is_adjoint
        NT = Tuple{ActiveRetTypes...}
        if any(any_jltypes(convert(LLVM.LLVMType, b; ctx, allow_boxed=true)) for b in ActiveRetTypes)
            NT = AnonymousStruct(NT)
        end
        push!(sret_types, NT)
    end

    # API.DFT_OUT_DIFF
    if is_adjoint && rettype <: Active
        @assert !sret_union
        if !allocatedinline(actualRetType)
            @safe_show actualRetType, rettype
            @assert allocatedinline(actualRetType)
        end
        push!(T_wrapperargs, LLVM.LLVMType(API.EnzymeGetShadowType(width, convert(LLVMType, actualRetType; ctx))))
    end

    data    = Array{Int64}(undef, 3)
    existed = Array{UInt8}(undef, 3)
    if Mode == API.DEM_ReverseModePrimal
        API.EnzymeExtractReturnInfo(augmented, data, existed)
        # tape -- todo ??? on wrap
        if existed[1] != 0
            tape = API.EnzymeExtractTapeTypeFromAugmentation(augmented)
            count_Sret += 1
        end
        
        tape = API.EnzymeExtractTapeTypeFromAugmentation(augmented)
        utape = API.EnzymeExtractUnderlyingTapeTypeFromAugmentation(augmented)
        if utape != C_NULL
            TapeType = EnzymeTapeToLoad{LLVMType(tape_type(utape))}
        elseif tape != C_NULL
            TapeType = tape_type(LLVMType(tape))
        else
            TapeType = Cvoid
        end
        push!(sret_types, TapeType)
        
        # primal return
        if existed[2] != 0 
            @assert returnPrimal
            count_Sret += 1
            push!(sret_types, actualRetType)
        else
            @assert !returnPrimal
        end
        # shadow return
        if existed[3] != 0
            count_Sret += 1
            if rettype <: Duplicated || rettype <: DuplicatedNoNeed
                push!(sret_types, actualRetType)
            elseif rettype <: BatchDuplicated || rettype <: BatchDuplicatedNoNeed
                push!(sret_types, AnonymousStruct(NTuple{width, actualRetType}))
            end
        else
            @assert rettype <: Const || rettype <: Active
        end
    end
    if Mode == API.DEM_ReverseModeCombined
        if returnPrimal
            push!(sret_types, actualRetType)
        end
    end
    if Mode == API.DEM_ForwardMode
        returnUsed = !GPUCompiler.isghosttype(actualRetType)
        if returnUsed
            if returnPrimal
                count_Sret += 1
            end
            if !(rettype <: Const)
                count_Sret += 1
            end
        end
        if returnPrimal
            push!(sret_types, actualRetType)
        end
        if rettype <: Duplicated || rettype <: DuplicatedNoNeed
            push!(sret_types, actualRetType)
        elseif rettype <: BatchDuplicated || rettype <: BatchDuplicatedNoNeed
            push!(sret_types, AnonymousStruct(NTuple{width, actualRetType}))
        end
    end
    
    combinedReturn = AnonymousStruct(Tuple{sret_types...})

    uses_sret = is_sret(combinedReturn, ctx)
   
    jltype = convert(LLVM.LLVMType, combinedReturn; ctx)

    returnRoots = false
    if uses_sret
    	returnRoots = deserves_rooting(jltype)
		if returnRoots
	        tracked = CountTrackedPointers(jltype)
            pushfirst!(T_wrapperargs, LLVM.PointerType(LLVM.ArrayType(T_prjlvalue, tracked.count)))
        
            pushfirst!(T_wrapperargs, LLVM.PointerType(jltype))
		end
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

    T_ret = returnRoots ? T_void : jltype
    FT = LLVM.FunctionType(T_ret, T_wrapperargs)
    llvm_f = LLVM.Function(mod, safe_name(LLVM.name(enzymefn)*"wrap"), FT)
    sfn = LLVM.get_subprogram(enzymefn)
    if sfn !== nothing
        LLVM.set_subprogram!(llvm_f, sfn)
    end
    dl = datalayout(mod)

    params = [parameters(llvm_f)...]

    LLVM.Builder(ctx) do builder
        entry = BasicBlock(llvm_f, "entry"; ctx)
        position!(builder, entry)

        realparms = LLVM.Value[]
        i = 1

        if returnRoots
            sret = params[i]
            i+= 1

            attr = if LLVM.version().major >= 12
                TypeAttribute("sret", jltype; ctx)
            else
                EnumAttribute("sret"; ctx)
            end
            push!(parameter_attributes(llvm_f, 1), attr)
            push!(parameter_attributes(llvm_f, 1), EnumAttribute("noalias"; ctx))
            push!(parameter_attributes(llvm_f, 2), EnumAttribute("noalias"; ctx))
        elseif jltype != T_void
            sret = alloca!(builder, jltype)
        end
        rootRet = nothing
        if returnRoots
            rootRet = params[i]
            i+=1
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
                    # TODO replace with better enzyme_zero
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
        if LLVM.get_subprogram(enzymefn) !== nothing
            metadata(val)[LLVM.MD_dbg] = DILocation(ctx, 0, 0, LLVM.get_subprogram(enzymefn) )
        end

        if Mode == API.DEM_ReverseModePrimal
            returnNum = 0
            for i in 1:3
                if existed[i] != 0
                    eval = val
                    if data[i] != -1
                        eval = extract_value!(builder, val, data[i])
                    end
                    ptr = gep!(builder, sret, [LLVM.ConstantInt(LLVM.IntType(64; ctx), 0), LLVM.ConstantInt(LLVM.IntType(32; ctx), returnNum)])
                    ptr = pointercast!(builder, ptr, LLVM.PointerType(llvmtype(eval)))
                    si = store!(builder, eval, ptr)
                    returnNum+=1

                    if i == 3 && shadow_init
                        shadows = LLVM.Value[]
                        if width == 1
                            push!(shadows, eval)
                        else
                            for i in 1:width
                                push!(shadows, extract_value!(builder, eval, i-1))
                            end
                        end
                        #cf = add_one_in_place_gen(eltype(rettype))
                        #cf = inttoptr!(builder, cf, LLVM.PointerType(LLVM.FunctionType(T_void, [convert(LLVMType, eltype(rettype); ctx)])))
                       
                        cf = nested_codegen!(Mode, mod, add_one_in_place, Tuple{Any})
                        push!(function_attributes(cf), EnumAttribute("alwaysinline", 0; ctx))
                        permit_inlining!(cf)
                        for shadowv in shadows
                            c = call!(builder, cf, [shadowv])
                            if LLVM.get_subprogram(enzymefn) !== nothing
                                metadata(c)[LLVM.MD_dbg] = DILocation(ctx, 0, 0, LLVM.get_subprogram(enzymefn) )
                            end
                        end
                    end
                end
            end
        elseif Mode == API.DEM_ForwardMode
            for returnNum in 0:(count_Sret-1)
                eval = val
                if count_Sret > 1
                    eval = extract_value!(builder, val, returnNum)
                end
                ptr = gep!(builder, sret, [LLVM.ConstantInt(LLVM.IntType(64; ctx), 0), LLVM.ConstantInt(LLVM.IntType(32; ctx), returnNum)])
                ptr = pointercast!(builder, ptr, LLVM.PointerType(llvmtype(eval)))
                si = store!(builder, eval, ptr)
            end
        else
            activeNum = 0
            returnNum = 0
            if Mode == API.DEM_ReverseModeCombined
                if returnPrimal
                    if !GPUCompiler.isghosttype(actualRetType) 
                        eval = extract_value!(builder, val, returnNum)
                        store!(builder, eval, gep!(builder, sret, [LLVM.ConstantInt(LLVM.IntType(64; ctx), 0), LLVM.ConstantInt(LLVM.IntType(32; ctx), length(elements(jltype))-1 )]))
                        returnNum+=1
                    end
                end
            end
            for T in argtypes
                T′ = eltype(T)
                isboxed = GPUCompiler.deserves_argbox(T′)
                if T <: Active
                    if !isboxed
                        eval = extract_value!(builder, val, returnNum)
                        store!(builder, eval, gep!(builder, sret, [LLVM.ConstantInt(LLVM.IntType(64; ctx), 0), LLVM.ConstantInt(LLVM.IntType(32; ctx), 0), LLVM.ConstantInt(LLVM.IntType(32; ctx), activeNum)]))
                        returnNum+=1
                    end
                    activeNum+=1
                end
            end
        end
        
        if returnRoots
            count = 0
            todo = Tuple{Vector{LLVM.Value},LLVM.LLVMType}[([LLVM.ConstantInt(LLVM.IntType(64; ctx), 0)], jltype)]
            while length(todo) != 0
                path, ty = popfirst!(todo)
                if isa(ty, LLVM.PointerType)
                    loc = gep!(builder, rootRet, [LLVM.ConstantInt(LLVM.IntType(64; ctx), 0), LLVM.ConstantInt(LLVM.IntType(32; ctx), count)])
                    count+=1
                    outloc = gep!(builder, sret, path)
                    store!(builder, load!(builder, outloc), loc)
                    continue
                end
                if isa(ty, LLVM.ArrayType) || isa(ty, LLVM.VectorType)
                    if any_jltypes(ty)
                        for i=1:length(ty)
                            npath = copy(path)
                            push!(npath, LLVM.ConstantInt(LLVM.IntType(32; ctx), i-1))
                            push!(todo, (npath, eltype(ty)))
                        end
                    end
                    continue
                end
                if isa(ty, LLVM.StructType)
                    for (i, t) in enumerate(LLVM.elements(ty))
                        if any_jltypes(t)
                            npath = copy(path)
                            push!(npath, LLVM.ConstantInt(LLVM.IntType(32; ctx), i-1))
                            push!(todo, (npath, t))
                        end
                    end
                    continue
                end
            end
            @assert count == tracked.count
        end
        if T_ret != T_void
            ret!(builder, load!(builder, sret))
        else
            ret!(builder)
        end
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

# https://github.com/JuliaLang/julia/blob/64378db18b512677fc6d3b012e6d1f02077af191/src/cgutils.cpp#L823
function for_each_uniontype_small(f, ty)
    if ty isa Union
        for_each_uniontype_small(f, ty.a)
        for_each_uniontype_small(f, ty.b)
        return
    end
    # https://github.com/JuliaLang/julia/blob/170d6439445c86e640214620dad3423d2bb42337/src/codegen.cpp#L1233
    if Base.isconcretetype(ty) && !ismutabletype(ty) && Base.datatype_pointerfree(ty)
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
function is_sret_union(jlrettype)
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
    sret_union = is_sret_union(actualRetType)
    
    returnRoots = false
    if sret
        if sret_union
            T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
            T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
            RT = T_prjlvalue
        else
            RT = eltype(llvmtype(first(parameters(entry_f))))
            returnRoots = deserves_rooting(RT)
            if returnRoots
                GPUCompiler.@safe_warn "Returned rooting not fully handled, segfault likely"
            end
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
    #
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
            @assert !sret_union
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

        # Box union return, from https://github.com/JuliaLang/julia/blob/81813164963f38dcd779d65ecd222fad8d7ed437/src/cgutils.cpp#L3138
        if sret_union
            def = BasicBlock(wrapper_f, "defaultBB"; ctx)
            scase = extract_value!(builder, res, 1)
            sw = switch!(builder, scase, def)
            counter = 1
            T_int8 = LLVM.Int8Type(ctx)
            T_int64 = LLVM.Int64Type(ctx)
            T_jlvalue = LLVM.StructType(LLVM.LLVMType[]; ctx)
            T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
            T_prjlvalue_UT = LLVM.PointerType(T_jlvalue)
            function inner(jlrettype)
                BB = BasicBlock(wrapper_f, "box_union"; ctx)
                position!(builder, BB)
                
                if GPUCompiler.isghosttype(jlrettype) || Core.Compiler.isconstType(jlrettype)
                    fill_val = unsafe_to_llvm(jlrettype.instance, ctx)
                    ret!(builder, fill_val)
                else
                    obj = emit_allocobj!(builder, jlrettype)
                    llty = convert(LLVMType, jlrettype; ctx)
                    ld = load!(builder, bitcast!(builder, sretPtr, LLVM.PointerType(llty, addrspace(llvmtype(sretPtr)))))
                    store!(builder, ld, bitcast!(builder, obj, LLVM.PointerType(llty, addrspace(llvmtype(obj)))))
                    # memcpy!(builder, bitcast!(builder, obj, LLVM.PointerType(T_int8, addrspace(llvmtype(obj)))), 0, bitcast!(builder, sretPtr, LLVM.PointerType(T_int8)), 0, LLVM.ConstantInt(T_int64, sizeof(jlrettype)))

                    ret!(builder, obj)
                end

                LLVM.API.LLVMAddCase(sw, LLVM.ConstantInt(llvmtype(scase), counter), BB)
                counter+=1
                return
            end
            for_each_uniontype_small(inner, actualRetType)

            position!(builder, def)
            fill_val = unsafe_to_llvm(nothing, ctx)
            ret!(builder, fill_val)
        elseif sret 
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

    if !(params.rt <: Const)
        @assert !GPUCompiler.isghosttype(eltype(params.rt))
    end
    if parent_job === nothing
        primal_target = DefaultCompilerTarget()
        primal_params = PrimalCompilerParams(mode)
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
                    emit_error(b, nothing, "BLAS Error")
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
    world = job.source.world

    actualRetType = nothing
    customDerivativeNames = String[]
    for (mi, k) in meta.compiled
        k_name = GPUCompiler.safe_name(k.specfunc)
        has_custom_rule = false
        if mode == API.DEM_ForwardMode
            has_custom_rule = EnzymeRules.has_frule_from_sig(mi.specTypes; world)
            if has_custom_rule
                @safe_debug "Found frule for" mi.specTypes
            end
        else
            has_custom_rule = EnzymeRules.has_rrule_from_sig(mi.specTypes; world)
            if has_custom_rule
                @safe_debug "Found rrule for" mi.specTypes
            end
        end

        if !(haskey(functions(mod), k_name) || has_custom_rule)
            continue
        end

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
            push!(attributes, StringAttribute("enzymejl_job", string(convert(Int, pointer_from_objref(jobref))); ctx))
            push!(attributes, StringAttribute("enzyme_math", name; ctx))
            push!(attributes, EnumAttribute("noinline", 0; ctx))
            must_wrap |= llvmfn == primalf
            nothing
        end
        
        foundTys[k_name] = (eltype(llvmtype(llvmfn)), mi)
        if has_custom_rule
            handleCustom("enzyme_custom")
            continue
        end

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
        if EnzymeRules.is_inactive_from_sig(mi.specTypes; world)
            handleCustom("enz_noop", [StringAttribute("enzyme_inactive"; ctx), StringAttribute("nofree"; ctx)])
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
        elseif name == :jl_rem2pi
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
        for f in functions(mod)
            push!(function_attributes(f), StringAttribute("enzymejl_world", string(job.source.world); ctx))
        end

        GC.@preserve job jobref begin
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

function add_one_in_place(x) 
    ty = typeof(x)
    # ptr = Base.pointer_from_objref(x)
    ptr = unsafe_to_pointer(x)
    if ty <: Base.RefValue || ty == Base.RefValue{Float64}
        x[] += one(eltype(ty))
    elseif true
        res = x+one(ty)
        @assert typeof(res) == ty
        unsafe_store!(reinterpret(Ptr{ty}, ptr), res)
    end
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
    ActiveRetTypes = Type[]

    for T in argtypes
        source_typ = eltype(T)
        
		expr = argexprs[i]
        i+=1
        if GPUCompiler.isghosttype(source_typ) || Core.Compiler.isconstType(source_typ)
            @assert T <: Const
            if is_adjoint
                push!(ActiveRetTypes, Nothing)
            end
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

        if T <: Const
            if is_adjoint
                push!(ActiveRetTypes, Nothing)
            end
            continue
        end

        if T <: Active
            if is_adjoint
                if width == 1
                    push!(ActiveRetTypes, source_typ)
                else
                    push!(ActiveRetTypes, NTuple{width, source_typ})
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
            if is_adjoint
                push!(ActiveRetTypes, Nothing)
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
            if is_adjoint
                push!(ActiveRetTypes, Nothing)
            end
            push!(ccexprs, argexpr)
        else
            error("calling convention should be annotated, got $T")
        end
    end
    
    jlRT = eltype(rettype)
    if typeof(jlRT) == UnionAll
      # Future improvement, add type assertion on load
      jlRT = DataType
    end

    if is_sret_union(jlRT)
        jlRT = Any
    end

    # API.DFT_OUT_DIFF
    if is_adjoint && rettype <: Active
        # TODO handle batch width
        @assert allocatedinline(jlRT)
        if width == 1
            push!(types, jlRT)
        else
            push!(types, NTuple{width, jlRT})
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
	
    ctx = LLVM.Context()
    
    if is_adjoint
        NT = Tuple{ActiveRetTypes...}
        if any(any_jltypes(convert(LLVM.LLVMType, b; ctx, allow_boxed=true)) for b in ActiveRetTypes)
            NT = AnonymousStruct(NT)
        end
        push!(sret_types, NT)
    end

    @assert i == length(argexprs)+1
            
    # Tape
    if CC <: AugmentedForwardThunk 
        push!(sret_types, TapeType)
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
    llvmtys = LLVMType[convert(LLVMType, x; ctx, allow_boxed=true) for x in types]
    llsret_types = LLVMType[]
    if !isempty(sret_types)
      for x in sret_types
          if !(GPUCompiler.isghosttype(x) || Core.Compiler.isconstType(x))
            push!(llsret_types, convert(LLVMType, x; ctx, allow_boxed=true))
          end
      end
    end
    
    T_void = convert(LLVMType, Nothing; ctx)


    combinedReturn = AnonymousStruct(Tuple{sret_types...})
    uses_sret = is_sret(combinedReturn, ctx)
    jltype = convert(LLVM.LLVMType, combinedReturn; ctx)
    
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)

    returnRoots = false
    if uses_sret
    	returnRoots = deserves_rooting(jltype)
    end
    
    pushfirst!(llvmtys, convert(LLVMType, Ptr{Cvoid}; ctx))
    

    T_jlvalue = LLVM.StructType(LLVM.LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    
    T_ret = jltype
    # if returnRoots
    #     T_ret = T_prjlvalue
    # end
    llvm_f, _ = LLVM.Interop.create_function(T_ret, llvmtys)
    push!(function_attributes(llvm_f), EnumAttribute("alwaysinline", 0; ctx))

	mod = LLVM.parent(llvm_f)
    i64 = LLVM.IntType(64; ctx)
	LLVM.Builder(ctx) do builder
		entry = BasicBlock(llvm_f, "entry"; ctx)
		position!(builder, entry)
		callparams = collect(LLVM.Value, parameters(llvm_f))
        lfn = callparams[1]
        deleteat!(callparams, 1)

        if returnRoots
	        tracked = CountTrackedPointers(jltype)
            pushfirst!(callparams, alloca!(builder, LLVM.ArrayType(T_prjlvalue, tracked.count)))
            pushfirst!(callparams, alloca!(builder, jltype))
        end

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
		lfn = inttoptr!(builder, lfn, LLVM.PointerType(LLVM.FunctionType(returnRoots ? T_void : T_ret, [llvmtype(x) for x in callparams])))
		r = call!(builder, lfn, callparams)

        if returnRoots
            attr = if LLVM.version().major >= 12
                TypeAttribute("sret", jltype; ctx)
            else
                EnumAttribute("sret"; ctx)
            end
            LLVM.API.LLVMAddCallSiteAttribute(r, LLVM.API.LLVMAttributeIndex(1), attr)
            r = load!(builder, callparams[1])
        end

        if T_ret != T_void
            ret!(builder, r)
        else
            ret!(builder)
        end
	end
    reinsert_gcmarker!(llvm_f)

	ir = string(mod)
	fn = LLVM.name(llvm_f)

    @assert length(types) == length(ccexprs)
    if any_jltypes(jltype)
        return quote
            Base.@_inline_meta
            Base.llvmcall(($ir, $fn), $(AnonymousStruct(Tuple{sret_types...})),
                    Tuple{Ptr{Cvoid}, $(types...)},
                    fptr, $(ccexprs...))
        end
    else
        return quote
            Base.@_inline_meta
                Base.llvmcall(($ir, $fn), $(Tuple{sret_types...}),
                    Tuple{Ptr{Cvoid}, $(types...)},
                    fptr, $(ccexprs...))
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


@generated function genthunk(::Type{F}, f::Fn, df::DF, ::Type{A}, tt::Type{TT},::Val{Mode}, ::Val{ModifiedBetween}, ::Val{width}, ::Val{specid}, ::Val{ReturnPrimal}, ::Val{ShadowInit}) where {F, Fn, DF, A<:Annotation, TT, Mode, ModifiedBetween, width, specid, ReturnPrimal, ShadowInit}
    primal, adjoint = fspec(F, TT)

    target = Compiler.EnzymeTarget()
    params = Compiler.EnzymeCompilerParams(adjoint, Mode, width, A, true, DF != Nothing, #=abiwrap=#true, ModifiedBetween, ReturnPrimal, ShadowInit)
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

@inline function thunk(f::F,df::DF, ::Type{A}, tt::Type{TT},::Val{Mode}, ::Val{width}, ::Val{ModifiedBetween}=Val(Mode != API.DEM_ReverseModeCombined), ::Val{ReturnPrimal}=Val(false), ::Val{ShadowInit}=Val(false)) where {F, DF, A<:Annotation, TT, Mode, width, ModifiedBetween, ReturnPrimal, ShadowInit}
    primal, adjoint = fspec(Core.Typeof(f), TT)
    target = Compiler.EnzymeTarget()
    params = Compiler.EnzymeCompilerParams(adjoint, Mode, width, A, true, DF != Nothing, #=abiwrap=#true, ModifiedBetween, ReturnPrimal, ShadowInit)
    job    = Compiler.CompilerJob(target, primal, params)

    specid = GPUCompiler.specialization_id(job)
    
    genthunk(Core.Typeof(f), f, df, A, TT, Val(Mode), Val(ModifiedBetween), Val(width), Val(specid), Val(ReturnPrimal), Val(ShadowInit))
end

import GPUCompiler: deferred_codegen_jobs

@generated function gendeferred_codegen(::Type{F}, ::Val{tt}, ::Val{rt}, ::Val{DupClosure},::Val{Mode},
                                     ::Val{width}, ::Val{ModifiedBetween}, ::Val{ReturnPrimal},::Val{ShadowInit}=Val(false)) where {F,tt, rt, DupClosure, Mode, width, ModifiedBetween, ReturnPrimal, ShadowInit}
    primal, adjoint = fspec(F, tt)
    target = EnzymeTarget()
    params = EnzymeCompilerParams(adjoint, Mode, width, rt, true, DupClosure, #=abiwrap=#true, ModifiedBetween, ReturnPrimal, ShadowInit)
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
