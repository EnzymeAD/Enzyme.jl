module Compiler

import ..Enzyme
import Enzyme: Const, Active, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed,
               BatchDuplicatedFunc,
               Annotation, guess_activity, eltype,
               API, TypeTree, typetree, only!, shift!, data0!, merge!,
               TypeAnalysis, FnTypeInfo, Logic, allocatedinline, ismutabletype
using Enzyme


import EnzymeCore: EnzymeRules, ABI, FFIABI, DefaultABI

using LLVM, GPUCompiler, Libdl
import Enzyme_jll

import GPUCompiler: CompilerJob, codegen, safe_name
using LLVM.Interop
import LLVM: Target, TargetMachine

using Printf

function cpu_name()
    ccall(:jl_get_cpu_name, String, ())
end

function cpu_features()
    @static if Sys.ARCH == :x86_64 ||
               Sys.ARCH == :x86
        return "+mmx,+sse,+sse2,+fxsr,+cx8" # mandated by Julia
    else
        return ""
    end
end

if LLVM.has_orc_v1()
    include("compiler/orcv1.jl")
else
    include("compiler/orcv2.jl")
end

include("gradientutils.jl")

include("compiler/utils.jl")

# Julia function to LLVM stem and arity
@static if VERSION < v"1.8.0"
const known_ops = Dict(
    Base.cbrt => (:cbrt, 1),
    Base.rem2pi => (:jl_rem2pi, 2),
    Base.sqrt => (:sqrt, 1),
    Base.sin => (:sin, 1),
    Base.sinc => (:sincn, 1),
    Base.sincos => (:__fd_sincos_1, 1),
    Base.sincospi => (:sincospi, 1),
    Base.sinpi => (:sinpi, 1),
    Base.cospi => (:cospi, 1),
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
    Base.FastMath.log => (:log, 1),
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
else
const known_ops = Dict(
    Base.fma_emulated => (:fma, 3),
    Base.cbrt => (:cbrt, 1),
    Base.rem2pi => (:jl_rem2pi, 2),
    Base.sqrt => (:sqrt, 1),
    Base.sin => (:sin, 1),
    Base.sinc => (:sincn, 1),
    Base.sincos => (:__fd_sincos_1, 1),
    Base.sincospi => (:sincospi, 1),
    Base.sinpi => (:sinpi, 1),
    Base.cospi => (:cospi, 1),
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
    Base.FastMath.log => (:log, 1),
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
end

const nofreefns = Set{String}((
    "ijl_module_parent", "jl_module_parent",
    "julia.safepoint",
    "ijl_set_task_tid", "jl_set_task_tid",
    "ijl_get_task_tid", "jl_get_task_tid",
    "julia.get_pgcstack_or_new",
    "ijl_global_event_loop", "jl_global_event_loop",
    "ijl_gf_invoke_lookup", "jl_gf_invoke_lookup",
    "ijl_f_typeassert", "jl_f_typeassert",
    "ijl_type_unionall", "jl_type_unionall",
    "jl_gc_queue_root", "gpu_report_exception", "gpu_signal_exception",
    "julia.ptls_states", "julia.write_barrier", "julia.typeof",
    "jl_backtrace_from_here", "ijl_backtrace_from_here",
    "jl_box_int64", "jl_box_int32",
    "ijl_box_int64", "ijl_box_int32",
    "jl_box_uint64", "jl_box_uint32",
    "ijl_box_uint64", "ijl_box_uint32",
    "ijl_box_char", "jl_box_char",
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
    "ijl_module_parent", "jl_module_parent",
    "julia.safepoint",
    "ijl_set_task_tid", "jl_set_task_tid",
    "ijl_get_task_tid", "jl_get_task_tid",
    "julia.get_pgcstack_or_new",
    "ijl_global_event_loop", "jl_global_event_loop",
    "ijl_gf_invoke_lookup", "jl_gf_invoke_lookup",
    "ijl_f_typeassert", "jl_f_typeassert",
    "ijl_type_unionall", "jl_type_unionall",
    "jl_gc_queue_root", "gpu_report_exception", "gpu_signal_exception",
    "julia.ptls_states", "julia.write_barrier", "julia.typeof",
    "jl_backtrace_from_here", "ijl_backtrace_from_here",
    "jl_box_int64", "jl_box_int32",
    "ijl_box_int64", "ijl_box_int32",
    "jl_box_uint64", "jl_box_uint32",
    "ijl_box_uint64", "ijl_box_uint32",
    "ijl_box_char", "jl_box_char",
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
@inline function Enzyme.guess_activity(::Type{Complex{T}}, Mode::API.CDerivativeMode) where {T<:AbstractFloat}
    if Mode == API.DEM_ForwardMode
        return DuplicatedNoNeed{Complex{T}}
    else
        return Active{Complex{T}}
    end
end

@enum ActivityState begin
    AnyState = 0
    ActiveState = 1
    DupState = 2
    MixedState = 3
end

@inline function Base.:|(a1::ActivityState, a2::ActivityState)
    ActivityState(Int(a1) | Int(a2))
end

@inline active_reg_inner(::Type{Complex{T}}, seen) where {T<:AbstractFloat} = ActiveState
@inline active_reg_inner(::Type{T}, seen) where {T<:AbstractFloat} = ActiveState
@inline active_reg_inner(::Type{T}, seen) where {T<:Integer} = AnyState
@inline active_reg_inner(::Type{T}, seen) where {T<:Function} = AnyState
@inline active_reg_inner(::Type{T}, seen) where {T<:DataType} = AnyState
@inline active_reg_inner(::Type{T}, seen) where {T<:Module} = AnyState
@inline active_reg_inner(::Type{T}, seen) where {T<:AbstractString} = AnyState
# here we explicity make ref considered dup rather than active
@inline function active_reg_inner(::Type{<:Union{Ptr{T}, Core.LLVMPtr{T}, Base.RefValue{T}}}, seen) where T
    state = active_reg_inner(T, seen)
    if state == AnyState
        return AnyState
    end
    return DupState
end
@inline function active_reg_inner(PT::Type{Array{T}}, seen) where {T}
    state = active_reg_inner(T, seen)
    if state == AnyState
        return AnyState
    end
    return DupState
end

@inline function active_reg_inner(::Type{T}, seen) where T
    if T isa UnionAll || T isa Union || T == Union{}
        return AnyState
    end
    if T ∈ keys(seen)
        return seen[T]
    end
    seen[T] = MixedState

    @assert !Base.isabstracttype(T)
    @assert Base.isconcretetype(T)

    ty = AnyState
    for f in 1:fieldcount(T)
        subT    = fieldtype(T, f)

        if subT isa UnionAll || subT isa Union || subT == Union{} || subT <: Integer
            continue
        end

        # Allocated inline so adjust first path
        if allocatedinline(subT)
            ty |= active_reg_inner(subT, seen)
        else
            sub = active_reg_inner(subT, seen)
            if sub == AnyState
                continue
            end
            ty |= DupState
        end
    end
    seen[T] = ty
    return ty
end

@inline @generated function active_reg(::Type{T}) where {T}
    seen = Dict{DataType, ActivityState}()
    state = active_reg_inner(T, seen)
    str = string(T)*" has mixed internal activity types"
    @assert state != MixedState str
    return state == ActiveState
end

@inline @generated function active_reg_nothrow(::Type{T}) where {T}
    seen = Dict{DataType, ActivityState}()
    state = active_reg_inner(T, seen)
    return state == ActiveState
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
abstract type AbstractThunk{FA, RT, TT, Width} end

struct CombinedAdjointThunk{PT, FA, RT, TT, Width, ReturnPrimal} <: AbstractThunk{FA, RT, TT, Width}
    adjoint::PT
end

struct ForwardModeThunk{PT, FA, RT, TT, Width, ReturnPrimal} <: AbstractThunk{FA, RT, TT, Width}
    adjoint::PT
end

struct AugmentedForwardThunk{PT, FA, RT, TT, Width, ReturnPrimal, TapeType} <: AbstractThunk{FA, RT, TT, Width}
    primal::PT
end

struct AdjointThunk{PT, FA, RT, TT, Width, TapeType} <: AbstractThunk{FA, RT, TT, Width}
    adjoint::PT
end

@inline return_type(::AbstractThunk{FA, RT}) where {FA, RT} = RT
@inline return_type(::Type{AugmentedForwardThunk{PT, FA, RT, TT, Width, ReturnPrimal, TapeType}}) where {PT, FA, RT, TT, Width, ReturnPrimal, TapeType} = RT
@inline get_tape_type(::Type{AugmentedForwardThunk{PT, FA, RT, TT, Width, ReturnPrimal, TapeType}}) where {PT, FA, RT, TT, Width, ReturnPrimal, TapeType} = TapeType
@inline get_tape_type(::Type{AdjointThunk{PT, FA, RT, TT, Width, TapeType}}) where {PT, FA, RT, TT, Width, TapeType} = TapeType

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

declare_allocobj!(mod) = get_function!(mod, "julia.gc_alloc_obj") do
    T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_ppjlvalue = LLVM.PointerType(LLVM.PointerType(T_jlvalue))
    T_size_t = convert(LLVM.LLVMType, Int)

    @static if VERSION < v"1.8.0"
        T_int8 = LLVM.Int8Type()
        T_pint8 = LLVM.PointerType(T_int8)
        LLVM.FunctionType(T_prjlvalue, [T_pint8, T_size_t, T_prjlvalue])
    else
        LLVM.FunctionType(T_prjlvalue, [T_ppjlvalue, T_size_t, T_prjlvalue])
    end
end
function emit_allocobj!(B, tag::LLVM.Value, Size::LLVM.Value, needs_workaround::Bool)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

	T_jlvalue = LLVM.StructType(LLVMType[])
    T_pjlvalue = LLVM.PointerType(T_jlvalue)
    T_ppjlvalue = LLVM.PointerType(T_pjlvalue)

    T_int8 = LLVM.Int8Type()
    T_pint8 = LLVM.PointerType(T_int8)

    @static if VERSION < v"1.7.0"
        ptls = reinsert_gcmarker!(fn, B)
        ptls = bitcast!(B, ptls, T_pint8)
    else
        pgcstack = reinsert_gcmarker!(fn, B)
        ct = inbounds_gep!(B,
            T_pjlvalue,
            bitcast!(B, pgcstack, T_ppjlvalue),
            [LLVM.ConstantInt(current_task_offset())])
        ptls_field = inbounds_gep!(B,
            T_pjlvalue,
            ct, [LLVM.ConstantInt(current_ptls_offset())])
        T_ppint8 = LLVM.PointerType(T_pint8)
        ptls = load!(B, T_pint8, bitcast!(B, ptls_field, T_ppint8))
    end

    if needs_workaround
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
        T_size_t = convert(LLVM.LLVMType, Int)
        # This doesn't allow for optimizations
        alty = LLVM.FunctionType(T_prjlvalue, [T_pint8, T_size_t, T_prjlvalue])
        alloc_obj, _ = get_function!(mod, "jl_gc_alloc_typed", alty)
        if value_type(Size) != T_size_t # Fix Int32/Int64 issues on 32bit systems
            Size = trunc!(B, Size, T_size_t)
        end
        return call!(B, alty, alloc_obj, [ptls, Size, tag])
    end


    alloc_obj, alty = declare_allocobj!(mod)

    @static if VERSION < v"1.8.0"
        return call!(B, alty, alloc_obj, [ptls, Size, tag])
    else
        return call!(B, alty, alloc_obj, [ct, Size, tag])
    end
end
function emit_allocobj!(B, T::DataType)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

	T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue_UT = LLVM.PointerType(T_jlvalue)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    # Obtain tag
    tag = LLVM.ConstantInt(convert(UInt, Base.pointer_from_objref(T)))  # do we need to root ETT
    tag = LLVM.const_inttoptr(tag, T_prjlvalue_UT)
    tag = LLVM.const_addrspacecast(tag, T_prjlvalue)

    T_size_t = convert(LLVM.LLVMType, UInt)
    Size = LLVM.ConstantInt(T_size_t, sizeof(T))
    emit_allocobj!(B, tag, Size, #=needs_workaround=#false)
end
declare_pointerfromobjref!(mod) = get_function!(mod, "julia.pointer_from_objref") do
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Derived)
    T_pjlvalue = LLVM.PointerType(T_jlvalue)
    LLVM.FunctionType(T_pjlvalue, [T_prjlvalue])
end
function emit_pointerfromobjref!(B, T)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    func, fty = declare_pointerfromobjref!(mod)
    return call!(B, fty, func, [T])
end

declare_writebarrier!(mod) = get_function!(mod, "julia.write_barrier") do
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    LLVM.FunctionType(LLVM.VoidType(), [T_prjlvalue]; vararg=true)
end
@static if VERSION < v"1.8.0"
declare_apply_generic!(mod) = get_function!(mod, "jl_apply_generic") do
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, LLVM.PointerType(T_prjlvalue), LLVM.Int32Type()])
end
else
declare_apply_generic!(mod) = get_function!(mod, "ijl_apply_generic") do
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, LLVM.PointerType(T_prjlvalue), LLVM.Int32Type()])
end
end
declare_juliacall!(mod) = get_function!(mod, "julia.call") do
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    LLVM.FunctionType(T_prjlvalue, [T_prjlvalue]; vararg=true)
end

function emit_jl!(B, val)::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue])
    fn, _ = get_function!(mod, "jl_", FT)
    call!(B, FT, fn, [val])
end

function emit_box_int32!(B, val)::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_int32 = LLVM.Int32Type()

    FT = LLVM.FunctionType(T_prjlvalue, [T_int32])
    @static if VERSION < v"1.8-"
        box_int32, _ = get_function!(mod, "jl_box_int32", FT)
    else
        box_int32, _ = get_function!(mod, "ijl_box_int32", FT)
    end
    call!(B, FT, box_int32, [val])
end

function emit_box_int64!(B, val)::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_int64 = LLVM.Int64Type()

    FT = LLVM.FunctionType(T_prjlvalue, [T_int64])
    @static if VERSION < v"1.8-"
        box_int64, _ = get_function!(mod, "jl_box_int64", FT)
    else
        box_int64, _ = get_function!(mod, "ijl_box_int64", FT)
    end
    call!(B, FT, box_int64, [val])
end

function emit_apply_generic!(B, args)::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_pprjlvalue = LLVM.PointerType(T_prjlvalue)
    T_int32 = LLVM.Int32Type()

    gen_FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_pprjlvalue, T_int32])
    @static if VERSION < v"1.8-"
        inv, _ = get_function!(mod, "jl_apply_generic", gen_FT)
    else
        inv, _ = get_function!(mod, "ijl_apply_generic", gen_FT)
    end

    @static if VERSION < v"1.9.0-"
        FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_prjlvalue]; vararg=true)
        inv = bitcast!(B, inv, LLVM.PointerType(FT))
        # call cc37 nonnull {}* bitcast ({}* ({}*, {}**, i32)* @jl_f_apply_type to {}* ({}*, {}*, {}*, {}*)*)({}* null, {}* inttoptr (i64 140150176657296 to {}*), {}* %4, {}* inttoptr (i64 140149987564368 to {}*))
        res = call!(B, FT, inv, args)
        LLVM.callconv!(res, 37)
    else
        # %5 = call nonnull {}* ({}* ({}*, {}**, i32)*, {}*, ...) @julia.call({}* ({}*, {}**, i32)* @jl_f_apply_type, {}* null, {}* inttoptr (i64 139640605802128 to {}*), {}* %4, {}* inttoptr (i64 139640590432896 to {}*))
        julia_call, FT = get_function!(mod, "julia.call",
            LLVM.FunctionType(T_prjlvalue,
                              [LLVM.PointerType(gen_FT), T_prjlvalue]; vararg=true))
        res = call!(B, FT, julia_call, LLVM.Value[inv, args...])
    end
    return res
end

function emit_invoke!(B, args)::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_pprjlvalue = LLVM.PointerType(T_prjlvalue)
    T_int32 = LLVM.Int32Type()

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
        res = call!(B, FT, inv, args)
        LLVM.callconv!(res, 38)
    else
        # %5 = call nonnull {}* ({}* ({}*, {}**, i32)*, {}*, ...) @julia.call({}* ({}*, {}**, i32)* @jl_f_apply_type, {}* null, {}* inttoptr (i64 139640605802128 to {}*), {}* %4, {}* inttoptr (i64 139640590432896 to {}*))
        julia_call, FT = get_function!(mod, "julia.call2",
            LLVM.FunctionType(T_prjlvalue,
                              [LLVM.PointerType(generic_FT), T_prjlvalue]; vararg=true))
        res = call!(B, FT, julia_call, [inv, args...])
    end
    return res
end

function emit_svec!(B, args)::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    fn, fty = get_function!(mod, "jl_svec")
    sz = convert(LLVMType, Csize_t)
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    LLVM.FunctionType(T_prjlvalue, [sz]; vararg=true)
    
    sz = convert(LLVMType, Csize_t)
    call!(B, fty, fn, [LLVM.ConstantInt(sz, length(args)), args...])
end


function emit_apply_type!(B, Ty, args)::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    legal = true
    found = []
    for arg in args
        if isa(arg, LLVM.CallInst)
            fn = LLVM.called_operand(arg)
            nm = ""
            if isa(fn, LLVM.Function)
                nm = LLVM.name(fn)
            end
            match = false
            for (fname, ty) in (
                                 ("jl_box_int64", Int64), ("ijl_box_int64", Int64),
                                 ("jl_box_uint64", UInt64), ("ijl_box_uint64", UInt64),
                                 ("jl_box_int32", Int32), ("ijl_box_int32", Int32),
                                 ("jl_box_uint32", UInt32), ("ijl_box_uint32", UInt32),
                                )
                if nm == "jl_box_int64" || nm == "ijl_box_int64"
                    v = first(operands(arg))
                    if isa(v, ConstantInt)
                        push!(found, convert(Int64, v))
                        match = true
                        break
                    end
                end
            end
            if match
                continue
            end
        end
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
        ptr = reinterpret(Ptr{Cvoid}, convert(UInt, ce))
        typ = Base.unsafe_pointer_to_objref(ptr)
        push!(found, typ)
    end

    if legal
        return unsafe_to_llvm(Ty{found...})
    end

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_pprjlvalue = LLVM.PointerType(T_prjlvalue)
    T_int32 = LLVM.Int32Type()

    generic_FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_pprjlvalue, T_int32])
    f_apply_type, _ = get_function!(mod, "jl_f_apply_type", generic_FT)
    Ty = unsafe_to_llvm(Ty)

    @static if VERSION < v"1.9.0-"
        FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_prjlvalue]; vararg=true)
        f_apply_type = bitcast!(B, f_apply_type, LLVM.PointerType(FT))
        # call cc37 nonnull {}* bitcast ({}* ({}*, {}**, i32)* @jl_f_apply_type to {}* ({}*, {}*, {}*, {}*)*)({}* null, {}* inttoptr (i64 140150176657296 to {}*), {}* %4, {}* inttoptr (i64 140149987564368 to {}*))
        tag = call!(B, FT, f_apply_type, LLVM.Value[LLVM.PointerNull(T_prjlvalue), Ty, args...])
        LLVM.callconv!(tag, 37)
    else
        # %5 = call nonnull {}* ({}* ({}*, {}**, i32)*, {}*, ...) @julia.call({}* ({}*, {}**, i32)* @jl_f_apply_type, {}* null, {}* inttoptr (i64 139640605802128 to {}*), {}* %4, {}* inttoptr (i64 139640590432896 to {}*))
        julia_call, FT = get_function!(mod, "julia.call",
            LLVM.FunctionType(T_prjlvalue,
                              [LLVM.PointerType(generic_FT), T_prjlvalue]; vararg=true))
        tag = call!(B, FT, julia_call, LLVM.Value[f_apply_type, LLVM.PointerNull(T_prjlvalue), Ty, args...])
    end
    return tag
end

function emit_jltypeof!(B, arg)::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    if isa(arg, ConstantExpr)
        ce = arg
        while isa(ce, ConstantExpr)
            ce = operands(ce)[1]
        end
        if isa(ce, LLVM.ConstantInt)
            ptr = reinterpret(Ptr{Cvoid}, convert(UInt, ce))
            typ = Base.unsafe_pointer_to_objref(ptr)
            return unsafe_to_llvm(Core.Typeof(typ))
        end
    end

    fn, FT = get_function!(mod, "jl_typeof") do ctx
        T_jlvalue = LLVM.StructType(LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
        LLVM.FunctionType(T_prjlvalue, [T_prjlvalue]; vararg=true)
    end
    call!(B, FT, fn, [arg])
end

function emit_methodinstance!(B, func, args)::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    world = enzyme_extract_world(fn)

    sizeT = convert(LLVMType, Csize_t)
    psizeT = LLVM.PointerType(sizeT)

    primalvaltys = LLVM.Value[unsafe_to_llvm(Core.Typeof(func))]
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

    meth = unsafe_to_llvm(meth)

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    @static if VERSION < v"1.8.0-"
    worlds, FT = get_function!(mod, "jl_gf_invoke_lookup_worlds",
        LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, sizeT, psizeT, psizeT]))
    else
    worlds, FT = get_function!(mod, "jl_gf_invoke_lookup_worlds",
        LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_prjlvalue, sizeT, psizeT, psizeT]))
    end
    EB = LLVM.IRBuilder()
    position!(EB, first(LLVM.instructions(LLVM.entry(fn))))
    minworld = alloca!(EB, sizeT)
    maxworld = alloca!(EB, sizeT)
    store!(B, LLVM.ConstantInt(sizeT, 0), minworld)
    store!(B, LLVM.ConstantInt(sizeT, -1), maxworld)
    @static if VERSION < v"1.8.0-"
    methodmatch = call!(B, FT, worlds, LLVM.Value[tag, LLVM.ConstantInt(sizeT, world), minworld, maxworld])
    else
    methodmatch = call!(B, FT, worlds, LLVM.Value[tag, unsafe_to_llvm(nothing), LLVM.ConstantInt(sizeT, world), minworld, maxworld])
    end
    # emit_jl!(B, methodmatch)
    # emit_jl!(B, emit_jltypeof!(B, methodmatch))
    offset = 1
    AT = LLVM.ArrayType(T_prjlvalue, offset+1)
    methodmatch = addrspacecast!(B, methodmatch, LLVM.PointerType(T_jlvalue, Derived))
    methodmatch = bitcast!(B, methodmatch, LLVM.PointerType(AT, Derived))
    gep = LLVM.inbounds_gep!(B, AT, methodmatch, LLVM.Value[LLVM.ConstantInt(0), LLVM.ConstantInt(offset)])
    sv = LLVM.load!(B, T_prjlvalue, gep)

    fn, FT = get_function!(mod, "jl_specializations_get_linfo",
                       LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_prjlvalue, T_prjlvalue]))

    mi = call!(B, FT, fn, [meth, tag, sv])

    return mi
end

function emit_writebarrier!(B, T)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    func, FT = declare_writebarrier!(mod)
    return call!(B, FT, func, T)
end

function array_inner(::Type{<:Array{T}}) where T
    return T
end
function array_shadow_handler(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, numArgs::Csize_t, Args::Ptr{LLVM.API.LLVMValueRef}, gutils::API.EnzymeGradientUtilsRef)::LLVM.API.LLVMValueRef
    inst = LLVM.Instruction(OrigCI)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(inst)))
    ctx = LLVM.context(LLVM.Value(OrigCI))
    gutils = GradientUtils(gutils)

    ce = operands(inst)[1]
    while isa(ce, ConstantExpr)
        ce = operands(ce)[1]
    end
    ptr = reinterpret(Ptr{Cvoid}, convert(UInt, ce))
    typ = array_inner(Base.unsafe_pointer_to_objref(ptr))

    b = LLVM.IRBuilder(B)
    orig = LLVM.Value(OrigCI)

    vals = LLVM.Value[]
    valTys = API.CValueType[]
    for i = 1:numArgs
        push!(vals, LLVM.Value(unsafe_load(Args, i)))
        push!(valTys, API.VT_Primal)
    end

    anti = call_samefunc_with_inverted_bundles!(b, gutils, orig, vals, valTys, #=lookup=#false)

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
    tot = LLVM.mul!(b, tot, LLVM.ConstantInt(LLVM.value_type(tot), elsz, false))

    if elsz == 1 && !isunion
        # extra byte for all julia allocated byte arrays
        tot = LLVM.add!(b, tot, LLVM.ConstantInt(LLVM.value_type(tot), 1, false))
    end
    if (isunion)
        # an extra byte for each isbits union array element, stored after a->maxsize
        tot = LLVM.add!(b, tot, prod)
    end

    i8 = LLVM.IntType(8)
    toset = get_array_data(b, anti)

    mcall = LLVM.memset!(b, toset, LLVM.ConstantInt(i8, 0, false), tot, al)

    ref::LLVM.API.LLVMValueRef = Base.unsafe_convert(LLVM.API.LLVMValueRef, anti)
    return ref
end

function get_array_struct()
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

    i8 = LLVM.IntType(8)
    ptrty = LLVM.PointerType(i8, 13)
    sizeT = LLVM.IntType(8*sizeof(Csize_t))
    arrayFlags = LLVM.IntType(16)
    elsz = LLVM.IntType(16)
    off = LLVM.IntType(32)
    nrows = LLVM.IntType(8*sizeof(Csize_t))

    return LLVM.StructType([ptrty, sizeT, arrayFlags, elsz, off, nrows]; packed=true)
end

function get_array_data(B, array)
    i8 = LLVM.IntType(8)
    ptrty = LLVM.PointerType(i8, 13)
    array = LLVM.pointercast!(B, array, LLVM.PointerType(ptrty, LLVM.addrspace(LLVM.value_type(array))))
    return LLVM.load!(B, ptrty, array)
end

function get_array_elsz(B, array)
    ST = get_array_struct()
    elsz = LLVM.IntType(16)
    array = LLVM.pointercast!(B, array, LLVM.PointerType(ST, LLVM.addrspace(LLVM.value_type(array))))
    v = inbounds_gep!(B, ST, array, LLVM.Value[LLVM.ConstantInt(Int32(0)), LLVM.ConstantInt(Int32(3))])
    return LLVM.load!(B, elsz, v)
end

function get_array_len(B, array)
    ST = get_array_struct()
    array = LLVM.pointercast!(B, array, LLVM.PointerType(ST, LLVM.addrspace(LLVM.value_type(array))))
    v = inbounds_gep!(B, ST, array, LLVM.Value[LLVM.ConstantInt(Int32(0)), LLVM.ConstantInt(Int32(1))])
    sizeT = LLVM.IntType(8*sizeof(Csize_t))
    return LLVM.load!(B, sizeT, v)
end

function get_array_nrows(B, array)
    ST = get_array_struct()
    array = LLVM.pointercast!(B, array, LLVM.PointerType(ST, LLVM.addrspace(LLVM.value_type(array))))
    v = inbounds_gep!(B, ST, array, LLVM.Value[LLVM.ConstantInt(Int32(0)), LLVM.ConstantInt(Int32(5))])
    nrows = LLVM.IntType(8*sizeof(Csize_t))
    return LLVM.load!(B, nrows, v)
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
AnyArray(Length::Int) = NamedTuple{ntuple(i->Symbol(i), Val(Length)),NTuple{Length,Any}}

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

function runtime_newtask_fwd(world::Val{World}, fn::FT1, dfn::FT2, post::Any, ssize::Int, ::Val{width}) where {FT1, FT2, World, width}
    FT = Core.Typeof(fn)
    ghos = isghostty(FT) || Core.Compiler.isconstType(FT)
    forward = thunk(world, (ghos ? Const : Duplicated){FT}, Const, Tuple{}, Val(API.DEM_ForwardMode), Val(width), Val((false,)), #=returnPrimal=#Val(true), #=shadowinit=#Val(false), FFIABI)
    ft = ghos ? Const(fn) : Duplicated(fn, dfn)
    function fclosure()
        res = forward(ft)
        return res[1]
    end

    return ccall(:jl_new_task, Ref{Task}, (Any, Any, Int), fclosure, post, ssize)
end

function runtime_newtask_augfwd(world::Val{World}, fn::FT1, dfn::FT2, post::Any, ssize::Int, ::Val{width}, ::Val{ModifiedBetween}) where {FT1, FT2, World, width, ModifiedBetween}
    # TODO make this AD subcall type stable
    FT = Core.Typeof(fn)
    ghos = isghostty(FT) || Core.Compiler.isconstType(FT)
    forward, adjoint = thunk(world, (ghos ? Const : Duplicated){FT}, Const, Tuple{}, Val(API.DEM_ReverseModePrimal), Val(width), Val(ModifiedBetween), #=returnPrimal=#Val(true), #=shadowinit=#Val(false), FFIABI)
    ft = ghos ? Const(fn) : Duplicated(fn, dfn)
    taperef = Ref{Any}()

    function fclosure()
        res = forward(ft)
        taperef[] = res[1]
        return res[2]
    end

    ftask = ccall(:jl_new_task, Ref{Task}, (Any, Any, Int), fclosure, post, ssize)

    function rclosure()
        adjoint(ft, taperef[])
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

function setup_macro_wraps(forwardMode::Bool, N::Int, Width::Int, base=nothing)
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
            base_idx += 1
        end
        t = :(Core.Typeof($prim))
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
    @assert length(primargs) == N
    @assert length(primtypes) == N
    wrapped = Expr[]
    for i in 1:N
        expr = :(
                 if ActivityTup[$i+1] && !isghostty($(primtypes[i])) && !Core.Compiler.isconstType($(primtypes[i]))
                   @assert $(primtypes[i]) !== DataType
                    if !$forwardMode && active_reg($(primtypes[i]))
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

function body_runtime_generic_fwd(N, Width, wrapped, primtypes)
    nnothing = ntuple(i->nothing, Val(Width+1))
    nres = ntuple(i->:(res[1]), Val(Width+1))
    ModifiedBetween = ntuple(i->false, Val(N+1))
    ElTypes = ntuple(i->:(eltype(Core.Typeof(args[$i]))), Val(N))
    Types = ntuple(i->:(Core.Typeof(args[$i])), Val(N))
    return quote
        args = ($(wrapped...),)

        # TODO: Annotation of return value
        # tt0 = Tuple{$(primtypes...)}
        tt = Tuple{$(ElTypes...)}
        tt′ = Tuple{$(Types...)}
        rt = Core.Compiler.return_type(f, Tuple{$(ElTypes...)})
        annotation = guess_activity(rt, API.DEM_ForwardMode)

        if annotation <: DuplicatedNoNeed
            annotation = Duplicated{rt}
        end
        if $Width != 1
            if annotation <: Duplicated
                annotation = BatchDuplicated{rt, $Width}
            end
        end

        dupClosure = ActivityTup[1]
        FT = Core.Typeof(f)
        if dupClosure && (isghostty(FT) || Core.Compiler.isconstType(FT))
            dupClosure = false
        end

        world = codegen_world_age(FT, tt)

        forward = thunk(Val(world), (dupClosure ? Duplicated : Const){FT}, annotation, tt′, Val(API.DEM_ForwardMode), width, #=ModifiedBetween=#Val($ModifiedBetween), #=returnPrimal=#Val(true), #=shadowInit=#Val(false), FFIABI)

        res = forward(dupClosure ? Duplicated(f, df) : Const(f), args...)

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
    ElTypes = ntuple(i->:(eltype(Core.Typeof(args[$i]))), Val(N))
    Types = ntuple(i->:(Core.Typeof(args[$i])), Val(N))

    return quote
        args = ($(wrapped...),)

        # TODO: Annotation of return value
        # tt0 = Tuple{$(primtypes...)}
        tt′ = Tuple{$(Types...)}
        rt = Core.Compiler.return_type(f, Tuple{$(ElTypes...)})
        annotation = guess_activity(rt, API.DEM_ReverseModePrimal)

        dupClosure = ActivityTup[1]
        FT = Core.Typeof(f)
        if dupClosure && (isghostty(FT) || Core.Compiler.isconstType(FT))
            dupClosure = false
        end

        world = codegen_world_age(FT, Tuple{$(ElTypes...)})

        forward, adjoint = thunk(Val(world), (dupClosure ? Duplicated : Const){FT},
                                 annotation, tt′, Val(API.DEM_ReverseModePrimal), width,
                                 ModifiedBetween, #=returnPrimal=#Val(true), #=shadowInit=#Val(false), FFIABI)

        internal_tape, origRet, initShadow = forward(dupClosure ? Duplicated(f, df) : Const(f), args...)
        resT = typeof(origRet)

        if annotation <: Const
            shadow_return = nothing
            tape = Tape{typeof(internal_tape), typeof(shadow_return), resT}(internal_tape, shadow_return)
            return ReturnType(($(nres...), tape))
        elseif annotation <: Active
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

        @assert annotation <: Duplicated || annotation <: DuplicatedNoNeed || annotation <: BatchDuplicated || annotation <: BatchDuplicatedNoNeed

        shadow_return = nothing
        tape = Tape{typeof(internal_tape), typeof(shadow_return), resT}(internal_tape, shadow_return)
        if $Width == 1
            return ReturnType((origRet, initShadow, tape))
        else
            return ReturnType((origRet, initShadow..., tape))
        end
    end
end

function func_runtime_generic_augfwd(N, Width)
    _, _, primtypes, allargs, typeargs, wrapped = setup_macro_wraps(false, N, Width)
    body = body_runtime_generic_augfwd(N, Width, wrapped, primtypes)

    quote
        function runtime_generic_augfwd(activity::Val{ActivityTup}, width::Val{$Width}, ModifiedBetween::Val{MB}, RT::Val{ReturnType}, f::F, df::DF, $(allargs...)) where {ActivityTup, MB, ReturnType, F, DF, $(typeargs...)}
            $body
        end
    end
end

@generated function runtime_generic_augfwd(activity::Val{ActivityTup}, width::Val{Width}, ModifiedBetween::Val{MB}, RT::Val{ReturnType}, f::F, df::DF, allargs...) where {ActivityTup, MB, Width, ReturnType, F, DF}
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

    ElTypes = ntuple(i->:(eltype(Core.Typeof(args[$i]))), Val(N))
    Types = ntuple(i->:(Core.Typeof(args[$i])), Val(N))

    quote
        args = ($(wrapped...),)

        # TODO: Annotation of return value
        # tt0 = Tuple{$(primtypes...)}
        tt = Tuple{$(ElTypes...)}
        tt′ = Tuple{$(Types...)}
        rt = Core.Compiler.return_type(f, tt)
        annotation = guess_activity(rt, API.DEM_ReverseModePrimal)

        dupClosure = ActivityTup[1]
        FT = Core.Typeof(f)
        if dupClosure && (isghostty(FT) || Core.Compiler.isconstType(FT))
            dupClosure = false
        end
        world = codegen_world_age(FT, tt)

        forward, adjoint = thunk(Val(world), (dupClosure ? Duplicated : Const){FT}, annotation, tt′, Val(API.DEM_ReverseModePrimal), width,
                                 ModifiedBetween, #=returnPrimal=#Val(true), #=shadowInit=#Val(false), FFIABI)
        if tape.shadow_return !== nothing
            args = (args..., $shadowret)
        end

        tup = adjoint(dupClosure ? Duplicated(f, df) : Const(f), args..., tape.internal_tape)[1]

        $(outs...)
        return nothing
    end
end

function func_runtime_generic_rev(N, Width)
    _, _, primtypes, allargs, typeargs, wrapped = setup_macro_wraps(false, N, Width)
    body = body_runtime_generic_rev(N, Width, wrapped, primtypes)

    quote
        function runtime_generic_rev(activity::Val{ActivityTup}, width::Val{$Width}, ModifiedBetween::Val{MB}, tape::TapeType, shadow_ptr, f::F, df::DF, $(allargs...)) where {ActivityTup, MB, TapeType, F, DF, $(typeargs...)}
            $body
        end
    end
end

@generated function runtime_generic_rev(activity::Val{ActivityTup}, width::Val{Width}, ModifiedBetween::Val{MB}, tape::TapeType, shadow_ptr, f::F, df::DF, allargs...) where {ActivityTup, MB, Width, TapeType, F, DF}
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

function emit_gc_preserve_begin(B::LLVM.IRBuilder, args=LLVM.Value[])
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    func, FT = get_function!(mod, "llvm.julia.gc_preserve_begin", LLVM.FunctionType(LLVM.TokenType(), vararg=true))

    token = call!(B, FT, func, args)
    return token
end

function emit_gc_preserve_end(B::LLVM.IRBuilder, token)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    func, FT = get_function!(mod, "llvm.julia.gc_preserve_end", LLVM.FunctionType(LLVM.VoidType(), [LLVM.TokenType()]))

    call!(B, FT, func, [token])
    return
end

function generic_setup(orig, func, ReturnType, gutils, start, B::LLVM.IRBuilder,  lookup; sret=nothing, tape=nothing, firstconst=false)
    width = get_width(gutils)
    mode = get_mode(gutils)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    ops = collect(operands(orig))[start+firstconst:end-1]

    T_int8 = LLVM.Int8Type()
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    ActivityList = Bool[]

    to_preserve = LLVM.Value[]

    @assert length(ops) != 0
    fill_val = unsafe_to_llvm(nothing)

    vals = LLVM.Value[]

    T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    if tape !== nothing
        NT = NTuple{length(ops)*Int(width), Ptr{Nothing}}
        SNT = convert(LLVMType, NT)
        shadow_ptr = emit_allocobj!(B, NT)
        shadow = addrspacecast!(B, shadow_ptr, LLVM.PointerType(T_jlvalue, Derived))
        shadow = bitcast!(B, shadow, LLVM.PointerType(SNT, Derived))
    end

    if firstconst
        val = new_from_original(gutils, operands(orig)[start])
        if lookup
            val = lookup_value(gutils, val, B)
        end
        push!(vals, val)
    end

    for (i, op) in enumerate(ops)
        val = new_from_original(gutils, op)
        if lookup
            val = lookup_value(gutils, val, B)
        end

        push!(vals, val)

        active = !is_constant_value(gutils, op)
        push!(ActivityList, active)

        inverted = nothing

        if active
            inverted = invert_pointer(gutils, op, B)
            if lookup
                inverted = lookup_value(gutils, inverted, B)
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
                idx = LLVM.Value[LLVM.ConstantInt(0), LLVM.ConstantInt((i-1)*Int(width) + w-1)]
                ev = addrspacecast!(B, ev, is_opaque(value_type(ev)) ? LLVM.PointerType(Derived) : LLVM.PointerType(eltype(value_type(ev)), Derived))
                ev = emit_pointerfromobjref!(B, ev)
                ev = ptrtoint!(B, ev, convert(LLVMType, Int))
                LLVM.store!(B, ev, LLVM.inbounds_gep!(B, SNT, shadow, idx))
            end
        end
    end

    if tape !== nothing
        pushfirst!(vals, shadow_ptr)
        pushfirst!(vals, tape)
    else
        pushfirst!(vals, unsafe_to_llvm(Val(ReturnType)))
    end

    if mode != API.DEM_ForwardMode
        uncacheable = get_uncacheable(gutils, orig)
        sret = false
        returnRoots = false

        ModifiedBetween = Bool[]

        for idx in 1:(length(ops)+firstconst)
            push!(ModifiedBetween, uncacheable[(start-1)+idx] != 0)
        end
        pushfirst!(vals, unsafe_to_llvm(Val((ModifiedBetween...,))))
    end

    pushfirst!(vals, unsafe_to_llvm(Val(Int(width))))
    pushfirst!(vals, unsafe_to_llvm(Val((ActivityList...,))))

    @static if VERSION < v"1.7.0-" || true
    else
    mi = emit_methodinstance!(B, func, vals)
    end

    pushfirst!(vals, unsafe_to_llvm(func))

    @static if VERSION < v"1.7.0-" || true
    else
    pushfirst!(vals, mi)
    end

    @static if VERSION < v"1.7.0-" || true
    cal = emit_apply_generic!(B, vals)
    else
    cal = emit_invoke!(B, vals)
    end

    debug_from_orig!(gutils, cal, orig)

    if tape === nothing
        llty = convert(LLVMType, ReturnType)
        cal = LLVM.addrspacecast!(B, cal, LLVM.PointerType(T_jlvalue, Derived))
        cal = LLVM.pointercast!(B, cal, LLVM.PointerType(llty, Derived))
    end

    return cal
end

function allocate_sret!(B::LLVM.IRBuilder, N)
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    al = LLVM.alloca!(B, LLVM.ArrayType(T_prjlvalue, N))
    return al
end

function allocate_sret!(gutils::API.EnzymeGradientUtilsRef, N)
    sret = LLVM.IRBuilder() do B
        position!(B, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))
        allocate_sret!(B, N)
    end
end

function common_generic_fwd(offset, B, orig, gutils, normalR, shadowR)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end

    width = get_width(gutils)

    sret = generic_setup(orig, runtime_generic_fwd, AnyArray(1+Int(width)), gutils, #=start=#offset, B, false)
    AT = LLVM.ArrayType(T_prjlvalue, 1+Int(width))
    if shadowR != C_NULL
        if width == 1
            gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1)])
            shadow = LLVM.load!(B, T_prjlvalue, gep)
        else
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
            shadow = LLVM.UndefValue(ST)
            for i in 1:width
                gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(i)])
                ld = LLVM.load!(B, T_prjlvalue, gep)
                shadow = insert_value!(B, shadow, ld, i-1)
            end
        end
        unsafe_store!(shadowR, shadow.ref)
    end

    if normalR != C_NULL
        normal = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]))
        unsafe_store!(normalR, normal.ref)
    end
    return false
end

function generic_fwd(B, orig, gutils, normalR, shadowR)
    conv = LLVM.callconv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
    @assert conv == 37
    common_generic_fwd(1, B, orig, gutils, normalR, shadowR)
end

function common_generic_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end

    width = get_width(gutils)
    sret = generic_setup(orig, runtime_generic_augfwd, AnyArray(2+Int(width)), gutils, #=start=#offset, B, false)
    AT = LLVM.ArrayType(T_prjlvalue, 2+Int(width))

    if shadowR != C_NULL
        if width == 1
            gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1)])
            shadow = LLVM.load!(B, T_prjlvalue, gep)
        else
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
            shadow = LLVM.UndefValue(ST)
            for i in 1:width
                gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(i)])
                ld = LLVM.load!(B, T_prjlvalue, gep)
                shadow = insert_value!(B, shadow, ld, i-1)
            end
        end
        unsafe_store!(shadowR, shadow.ref)
    end

    if normalR != C_NULL
        normal = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]))
        unsafe_store!(normalR, normal.ref)
    end

    tape = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1+width)]))
    unsafe_store!(tapeR, tape.ref)
    return false
end

function generic_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    conv = LLVM.callconv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20

    @assert conv == 37

    common_generic_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end

function common_generic_rev(offset, B, orig, gutils, tape)::Cvoid
    if !is_constant_value(gutils, orig) || !is_constant_inst(gutils, orig)

        @assert tape !== C_NULL
        width = get_width(gutils)
        generic_setup(orig, runtime_generic_rev, Nothing, gutils, #=start=#offset, B, true; tape)
    end
    return nothing
end

function generic_rev(B, orig, gutils, tape)::Cvoid
    conv = LLVM.callconv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20

    @assert conv == 37

    common_generic_rev(1, B, orig, gutils, tape)
    return nothing
end

function common_apply_latest_fwd(offset, B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    width = get_width(gutils)
    AT = LLVM.ArrayType(T_prjlvalue, 1+Int(width))
    sret = generic_setup(orig, runtime_generic_fwd, AnyArray(1+Int(width)), gutils, #=start=#offset+1, B, false)

    if shadowR != C_NULL
        if width == 1
            gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1)])
            shadow = LLVM.load!(B, T_prjlvalue, gep)
        else
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
            shadow = LLVM.UndefValue(ST)
            for i in 1:width
                gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(i)])
                ld = LLVM.load!(B, T_prjlvalue, gep)
                shadow = insert_value!(B, shadow, ld, i-1)
            end
        end
        unsafe_store!(shadowR, shadow.ref)
    end

    if normalR != C_NULL
        normal = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]))
        unsafe_store!(normalR, normal.ref)
    end

    return false
end

function common_apply_latest_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    width = get_width(gutils)
    AT = LLVM.ArrayType(T_prjlvalue, 2+Int(width))
    # sret = generic_setup(orig, runtime_apply_latest_augfwd, AnyArray(2+Int(width)), gutils, #=start=#offset+1, ctx, B, false)
    sret = generic_setup(orig, runtime_generic_augfwd, AnyArray(2+Int(width)), gutils, #=start=#offset+1, B, false)

    if shadowR != C_NULL
        if width == 1
            gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1)])
            shadow = LLVM.load!(B, T_prjlvalue, gep)
        else
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
            shadow = LLVM.UndefValue(ST)
            for i in 1:width
                gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(i)])
                ld = LLVM.load!(B, T_prjlvalue, gep)
                shadow = insert_value!(B, shadow, ld, i-1)
            end
        end
        unsafe_store!(shadowR, shadow.ref)
    end

    if normalR != C_NULL
        normal = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]))
        unsafe_store!(normalR, normal.ref)
    end

    tape = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1+width)]))
    unsafe_store!(tapeR, tape.ref)
    return false
end

function common_apply_latest_rev(offset, B, orig, gutils, tape)::Cvoid
    if !is_constant_value(gutils, orig) || !is_constant_inst(gutils, orig)
        width = get_width(gutils)
        generic_setup(orig, runtime_generic_rev, Nothing, gutils, #=start=#offset+1, B, true; tape)
    end

    return nothing
end

function apply_latest_fwd(B, orig, gutils, normalR, shadowR)
    conv = LLVM.callconv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
    @assert conv == 37

    common_apply_latest_fwd(1, B, orig, gutils, normalR, shadowR)
end

function apply_latest_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    conv = LLVM.callconv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
    @assert conv == 37

    common_apply_latest_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end

function apply_latest_rev(B, orig, gutils, tape)
    conv = LLVM.callconv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
    @assert conv == 37

    common_apply_latest_rev(1, B, orig, gutils, tape)
    return nothing
end

function common_newstructv_fwd(offset, B, orig, gutils, normalR, shadowR)
    origops = collect(operands(orig))
    width = get_width(gutils)
    if is_constant_value(gutils, orig)
        return true
    end

    shadowsin = LLVM.Value[invert_pointer(gutils, o, B) for o in origops[offset:end-1] ]
    if width == 1
        if offset != 1
            pushfirst!(shadowsin, origops[1])
        end
        shadowres = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), shadowsin)
        callconv!(shadowres, callconv(orig))
    else
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx in 1:width
            args = LLVM.Value[
                              extract_value!(B, s, idx-1) for s in shadowsin
                              ]
            if offset != 1
                pushfirst!(args, origops[1])
            end
            tmp = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), args)
            callconv!(tmp, callconv(orig))
            shadowres = insert_value!(B, shadowres, tmp, idx-1)
        end
    end
    unsafe_store!(shadowR, shadowres.ref)
    return false
end
function common_newstructv_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
    common_newstructv_fwd(offset, B, orig, gutils, normalR, shadowR)
end

function common_newstructv_rev(offset, B, orig, gutils, tape)
    emit_error(B, orig, "Enzyme: Not yet implemented reverse for jl_new_struct")
    return nothing
end

function common_f_tuple_fwd(offset, B, orig, gutils, normalR, shadowR)
    common_newstructv_fwd(offset, B, orig, gutils, normalR, shadowR)
end
function common_f_tuple_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
    common_f_tuple_fwd(offset, B, orig, gutils, normalR, shadowR)
end

function common_f_tuple_rev(offset, B, orig, gutils, tape)
    # This function allocates a new return which returns a pointer, thus this instruction itself cannot transfer
    # derivative info, only create a shadow pointer, which is handled by the forward pass.
    return nothing
end


function f_tuple_fwd(B, orig, gutils, normalR, shadowR)
    common_f_tuple_fwd(1, B, orig, gutils, normalR, shadowR)
end

function f_tuple_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    common_f_tuple_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end

function f_tuple_rev(B, orig, gutils, tape)
    common_f_tuple_rev(1, B, orig, gutils, tape)
    return nothing
end

function new_structv_fwd(B, orig, gutils, normalR, shadowR)
    common_newstructv_fwd(1, B, orig, gutils, normalR, shadowR)
end

function new_structv_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    common_newstructv_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end

function new_structv_rev(B, orig, gutils, tape)
    common_apply_latest_rev(1, B, orig, gutils, tape)
    return nothing
end

function common_jl_getfield_fwd(offset, B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig)
        return true
    end

    origops = collect(operands(orig))[offset:end]
    width = get_width(gutils)
    if !is_constant_value(gutils, origops[2])
        shadowin = invert_pointer(gutils, origops[2], B)
        if width == 1
            args = LLVM.Value[new_from_original(gutils, origops[1]), shadowin]
            for a in origops[3:end-1]
                push!(args, new_from_original(gutils, a))
            end
            if offset != 1
                pushfirst!(args, first(operands(orig)))
            end
            shadowres = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), args)
            callconv!(shadowres, callconv(orig))
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
            for idx in 1:width
                args = LLVM.Value[new_from_original(gutils, origops[1]), extract_value!(B, shadowin, idx-1)]
                for a in origops[3:end-1]
                    push!(args, new_from_original(gutils, a))
                end
                if offset != 1
                    pushfirst!(args, first(operands(orig)))
                end
                tmp = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), args)
                callconv!(tmp, callconv(orig))
                shadowres = insert_value!(B, shadowres, tmp, idx-1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    else
        normal = new_from_original(gutils, orig)
        if width == 1
            shadowres = normal
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(normal))))
            for idx in 1:width
                shadowres = insert_value!(B, shadowres, normal, idx-1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return false
end

getfield_idx(v, idx) = ccall(:jl_get_nth_field_checked, Any, (Any, UInt), v, idx)
setfield_idx(v, idx, rhs) = ccall(:jl_set_nth_field, Cvoid, (Any, UInt, Any), v, idx, rhs)

@inline function make_zero(::Type{RT}) where RT
    return RT(0)
end

function rt_jl_getfield_aug(dptr::T, ::Type{Val{symname}}, ::Val{isconst}, dptrs...) where {T, symname, isconst}
    res = getfield(dptr, symname)
    RT = Core.Typeof(res)
    if active_reg(RT)
        if length(dptrs) == 0
            return Ref{RT}(make_zero(RT))
        else
            return ( (Ref{RT}(make_zero(RT)) for _ in 1:(1+length(dptrs)))..., )
        end
    else
        if length(dptrs) == 0
            return res
        else
            return (res, (getfield(dv, symname) for dv in dptrs)...)
        end
    end
end

function idx_jl_getfield_aug(dptr::T, ::Type{Val{symname}}, ::Val{isconst}, dptrs...) where {T, symname, isconst}
    res = getfield_idx(dptr, symname)
    RT = Core.Typeof(res)
    if active_reg(RT)
        if length(dptrs) == 0
            return Ref{RT}(make_zero(RT))
        else
            return ( (Ref{RT}(make_zero(RT)) for _ in 1:(1+length(dptrs)))..., )
        end
    else
        if length(dptrs) == 0
            return res
        else
            return (res, (getfield(dv, symname) for dv in dptrs)...)
        end
    end
end

function rt_jl_getfield_rev(dptr::T, dret, ::Type{Val{symname}}, ::Val{isconst}, dptrs...) where {T, symname, isconst}
    cur = getfield(dptr, symname)

    RT = Core.Typeof(cur)
    if active_reg(RT) && !isconst
        if length(dptrs) == 0
            setfield!(dptr, symname, cur+dret[])
        else
            setfield!(dptr, symname, cur+dret[1][])
            for i in 1:length(dptrs)
                setfield!(dptrs[i], symname, cur+dret[1+i][])
            end
        end
    end
    return nothing
end
function idx_jl_getfield_rev(dptr::T, dret, ::Type{Val{symname}}, ::Val{isconst}, dptrs...) where {T, symname, isconst}
    cur = getfield_idx(dptr, symname)

    RT = Core.Typeof(cur)
    if active_reg(RT) && !isconst
        if length(dptrs) == 0
            setfield_idx(dptr, symname, cur+dret[])
        else
            setfield_idx(dptr, symname, cur+dret[1][])
            for i in 1:length(dptrs)
                setfield_idx(dptrs[i], symname, cur+dret[1+i][])
            end
        end
    end
    return nothing
end

function common_jl_getfield_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig)
        return true
    end

    ops = collect(operands(orig))[offset:end]
    width = get_width(gutils)

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    T_int8 = LLVM.Int8Type()
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    if !is_constant_value(gutils, ops[2])
        inp = invert_pointer(gutils, ops[2], B)
        if width == 1
            inps = [inp]
        else
            inps = LLVM.Value[]
            for w in 1:width
                push!(inps, extract_value!(B, inp, w-1))
            end
        end
    else
        inps = [new_from_original(gutils, ops[2])]
    end

    vals = LLVM.Value[]
    push!(vals, inps[1])

    sym = new_from_original(gutils, ops[3])
    sym = emit_apply_type!(B, Base.Val, [sym])
    push!(vals, sym)

    push!(vals, unsafe_to_llvm(Val(is_constant_value(gutils, orig))))

    for v in inps[2:end]
        push!(vals, v)
    end

    pushfirst!(vals, unsafe_to_llvm(rt_jl_getfield_aug))

    cal = emit_apply_generic!(B, vals)

    debug_from_orig!(gutils, cal, orig)

    if width == 1
        shadowres = cal
    else
        AT = LLVM.ArrayType(T_prjlvalue, Int(width))

        forgep = cal
        if !is_constant_value(gutils, ops[2])
            forgep = LLVM.addrspacecast!(B, forgep, LLVM.PointerType(T_jlvalue, Derived))
            forgep = LLVM.pointercast!(B, forgep, LLVM.PointerType(AT, Derived))
        end    

        ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
        shadow = LLVM.UndefValue(ST)
        for i in 1:width
            if !is_constant_value(gutils, ops[2])
                gep = LLVM.inbounds_gep!(B, AT, forgep, [LLVM.ConstantInt(0), LLVM.ConstantInt(i-1)])
                ld = LLVM.load!(B, T_prjlvalue, gep)
            else
                ld = forgep
            end
            shadow = insert_value!(B, shadow, ld, i-1)
        end
        shadowres = shadow
    end

    unsafe_store!(shadowR, shadowres.ref)
    unsafe_store!(tapeR, cal.ref)
    return false
end

function common_jl_getfield_rev(offset, B, orig, gutils, tape)
    if is_constant_value(gutils, orig)
        return
    end

    ops = collect(operands(orig))[offset:end]
    width = get_width(gutils)

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    
    if !is_constant_value(gutils, ops[2])
        inp = invert_pointer(gutils, ops[2], B)
        inp = lookup_value(gutils, inp, B)
        if width == 1
            inps = [inp]
        else
            inps = LLVM.Value[]
            for w in 1:width
                push!(inps, extract_value!(B, inp, w-1))
            end
        end
    else
        inp = new_from_original(gutils, ops[2])
        inp = lookup_value(gutils, inp, B)
        inps = [inp]
    end

    vals = LLVM.Value[]
    push!(vals, inps[1])

    push!(vals, tape)

    sym = new_from_original(gutils, ops[3])
    sym = lookup_value(gutils, sym, B)
    sym = emit_apply_type!(B, Base.Val, [sym])
    push!(vals, sym)

    push!(vals, unsafe_to_llvm(Val(is_constant_value(gutils, orig))))

    for v in inps[2:end]
        push!(vals, v)
    end

    pushfirst!(vals, unsafe_to_llvm(rt_jl_getfield_rev))

    cal = emit_apply_generic!(B, vals)

    debug_from_orig!(gutils, cal, orig)
    return nothing
end

function jl_nthfield_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig)
        return true
    end
    origops = collect(operands(orig))
    width = get_width(gutils)
    if !is_constant_value(gutils, origops[1])
        shadowin = invert_pointer(gutils, origops[1], B)
        if width == 1
            args = LLVM.Value[
                              shadowin
                              new_from_original(gutils, origops[2])
                              ]
            shadowres = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), args)
            callconv!(shadowres, callconv(orig))
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
            for idx in 1:width
                args = LLVM.Value[
                                  extract_value!(B, shadowin, idx-1)
                                  new_from_original(gutils, origops[2])
                                  ]
                tmp = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), args)
                callconv!(tmp, callconv(orig))
                shadowres = insert_value!(B, shadowres, tmp, idx-1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    else
        normal = new_from_original(gutils, orig)
        if width == 1
            shadowres = normal
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(normal))))
            for idx in 1:width
                shadowres = insert_value!(B, shadowres, normal, idx-1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return false
end
function jl_nthfield_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig)
        return true
    end

    ops = collect(operands(orig))
    width = get_width(gutils)

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    
    T_int8 = LLVM.Int8Type()
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    if !is_constant_value(gutils, ops[1])
        inp = invert_pointer(gutils, ops[1], B)
        if width == 1
            inps = [inp]
        else
            inps = LLVM.Value[]
            for w in 1:width
                push!(inps, extract_value!(B, inp, w-1))
            end
        end
    else
        inps = [new_from_original(gutils, ops[1])]
    end

    vals = LLVM.Value[]
    push!(vals, inps[1])

    sym = new_from_original(gutils, ops[2])
    sym = (sizeof(Int) == sizeof(Int64) ? emit_box_int64! : emit_box_int32!)(B, sym)
    sym = emit_apply_type!(B, Base.Val, [sym])
    push!(vals, sym)

    push!(vals, unsafe_to_llvm(Val(is_constant_value(gutils, orig))))

    for v in inps[2:end]
        push!(vals, v)
    end

    pushfirst!(vals, unsafe_to_llvm(idx_jl_getfield_aug))

    cal = emit_apply_generic!(B, vals)

    debug_from_orig!(gutils, cal, orig)

    if width == 1
        shadowres = cal
    else
        AT = LLVM.ArrayType(T_prjlvalue, Int(width))
        forgep = cal
        if !is_constant_value(gutils, ops[1])
            forgep = LLVM.addrspacecast!(B, forgep, LLVM.PointerType(T_jlvalue, Derived))
            forgep = LLVM.pointercast!(B, forgep, LLVM.PointerType(AT, Derived))
        end    

        ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
        shadow = LLVM.UndefValue(ST)
        for i in 1:width
            if !is_constant_value(gutils, ops[1])
                gep = LLVM.inbounds_gep!(B, AT, forgep, [LLVM.ConstantInt(0), LLVM.ConstantInt(i-1)])
                ld = LLVM.load!(B, T_prjlvalue, gep)
            else
                ld = forgep
            end
            shadow = insert_value!(B, shadow, ld, i-1)
        end
        shadowres = shadow
    end

    unsafe_store!(shadowR, shadowres.ref)
    unsafe_store!(tapeR, cal.ref)
    return false
end
function jl_nthfield_rev(B, orig, gutils, tape)
    if is_constant_value(gutils, orig)
        return
    end

    ops = collect(operands(orig))
    width = get_width(gutils)

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    
    if !is_constant_value(gutils, ops[1])
        inp = invert_pointer(gutils, ops[1], B)
        inp = lookup_value(gutils, inp, B)
        if width == 1
            inps = [inp]
        else
            inps = LLVM.Value[]
            for w in 1:width
                push!(inps, extract_value!(B, inp, w-1))
            end
        end
    else
        inp = new_from_original(gutils, ops[1])
        inp = lookup_value(gutils, inp, B)
        inps = [inp]
    end

    vals = LLVM.Value[]
    push!(vals, inps[1])

    push!(vals, tape)

    sym = new_from_original(gutils, ops[2])
    sym = lookup_value(gutils, sym, B)
    sym = (sizeof(Int) == sizeof(Int64) ? emit_box_int64! : emit_box_int32!)(B, sym)
    sym = emit_apply_type!(B, Base.Val, [sym])
    push!(vals, sym)

    push!(vals, unsafe_to_llvm(Val(is_constant_value(gutils, orig))))

    for v in inps[2:end]
        push!(vals, v)
    end

    pushfirst!(vals, unsafe_to_llvm(idx_jl_getfield_rev))

    cal = emit_apply_generic!(B, vals)

    debug_from_orig!(gutils, cal, orig)
    return nothing
end

function jl_getfield_fwd(B, orig, gutils, normalR, shadowR)
    common_jl_getfield_fwd(1, B, orig, gutils, normalR, shadowR)
end
function jl_getfield_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    common_jl_getfield_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end
function jl_getfield_rev(B, orig, gutils, tape)
    common_jl_getfield_rev(1, B, orig, gutils, tape)
end

function common_setfield_fwd(offset, B, orig, gutils, normalR, shadowR)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    origops = collect(operands(orig))[offset:end]
    if !is_constant_value(gutils, origops[4])
        width = get_width(gutils)

        shadowin = if !is_constant_value(gutils, origops[2])
            invert_pointer(gutils, origops[2], B)
        else
            new_from_original(gutils, origops[2])
        end

        shadowout = invert_pointer(gutils, origops[4], B)
        if width == 1
            args = LLVM.Value[
                              new_from_original(gutils, origops[1])
                              shadowin
                              new_from_original(gutils, origops[3])
                              shadowout
                              ]
            valTys = API.CValueType[API.VT_Primal, API.VT_Shadow, API.VT_Primal, API.VT_Shadow]
            if offset != 1
                pushfirst!(args, first(operands(orig)))
                pushfirst!(valTys, API.VT_Primal)
            end

            shadowres = call_samefunc_with_inverted_bundles!(B, gutils, orig, args, valTys, #=lookup=#false)
            callconv!(shadowres, callconv(orig))
        else
            for idx in 1:width
                args = LLVM.Value[
                                  new_from_original(gutils, origops[1])
                                  extract_value!(B, shadowin, idx-1)
                                  new_from_original(gutils, origops[3])
                                  extract_value!(B, shadowout, idx-1)
                                  ]
                valTys = API.CValueType[API.VT_Primal, API.VT_Shadow, API.VT_Primal, API.VT_Shadow]
                if offset != 1
                    pushfirst!(args, first(operands(orig)))
                    pushfirst!(valTys, API.VT_Primal)
                end

                tmp = call_samefunc_with_inverted_bundles!(B, gutils, orig, args, valTys, #=lookup=#false)

                callconv!(tmp, callconv(orig))
            end
        end
    end
    return false
end

function common_setfield_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
    emit_error(B, orig, "Enzyme: unhandled augmented forward for jl_f_setfield")
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return false
end

function common_setfield_rev(offset, B, orig, gutils, tape)
  emit_error(B, orig, "Enzyme: unhandled reverse for jl_f_setfield")
  return nothing
end


function setfield_fwd(B, orig, gutils, normalR, shadowR)
    common_setfield_fwd(1, B, orig, gutils, normalR, shadowR)
end

function setfield_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    common_setfield_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end

function setfield_rev(B, orig, gutils, tape)
    common_setfield_rev(1, B, orig, gutils, tape)
end

function common_apply_iterate_fwd(offset, B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    emit_error(B, orig, "Enzyme: Not yet implemented, forward for jl_f__apply_iterate")
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return false
end

function common_apply_iterate_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    emit_error(B, orig, "Enzyme: Not yet implemented augmented forward for jl_f__apply_iterate")

    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return false
end

function common_apply_iterate_rev(offset, B, orig, gutils, tape)
    if !is_constant_value(gutils, orig) || !is_constant_inst(gutils, orig)
        emit_error(B, orig, "Enzyme: Not yet implemented reverse for jl_f__apply_iterate")
    end
    return nothing
end

function apply_iterate_fwd(B, orig, gutils, normalR, shadowR)
    common_apply_iterate_fwd(1, B, orig, gutils, normalR, shadowR)
end

function apply_iterate_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    common_apply_iterate_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end

function apply_iterate_rev(B, orig, gutils, tape)
    common_apply_iterate_rev(1, B, orig, gutils, tape)
    return nothing
end

function common_f_svec_ref_fwd(offset, B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    emit_error(B, orig, "Enzyme: unhandled augmented forward for jl_f__svec_ref")
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return false
end

function common_f_svec_ref_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    emit_error(B, orig, "Enzyme: Not yet implemented augmented forward for jl_f__svec_ref")

    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return false
end

function common_f_svec_ref_rev(offset, B, orig, gutils, tape)
    if !is_constant_value(gutils, orig) || !is_constant_inst(gutils, orig)
        emit_error(B, orig, "Enzyme: Not yet implemented reverse for jl_f__svec_ref")
    end
    return nothing
end

function f_svec_ref_fwd(B, orig, gutils, normalR, shadowR)
    common_f_svec_ref_fwd(1, B, orig, gutils, normalR, shadowR)
    return nothing
end

function f_svec_ref_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    common_f_svec_ref_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
    return nothing
end

function f_svec_ref_rev(B, orig, gutils, tape)
    common_f_svec_ref_rev(1, B, orig, gutils, tape)
    return nothing
end


function jlcall_fwd(B, orig, gutils, normalR, shadowR)
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_apply_generic", "jl_apply_generic"))
            return common_generic_fwd(2, B, orig, gutils, normalR, shadowR)
        end
        if in(name, ("ijl_f__apply_latest", "ijl_f__call_latest", "jl_f__apply_latest", "jl_f__call_latest"))
            return common_apply_latest_fwd(2, B, orig, gutils, normalR, shadowR)
        end
        if in(name, ("ijl_new_structv", "jl_new_structv"))
            return common_newstructv_fwd(2, B, orig, gutils, normalR, shadowR)
        end
        if in(name, ("ijl_f_tuple", "jl_f_tuple"))
            return common_f_tuple_fwd(2, B, orig, gutils, normalR, shadowR)
        end
        if in(name, ("ijl_f_getfield", "jl_f_getfield"))
            return common_jl_getfield_fwd(2, B, orig, gutils, normalR, shadowR)
        end
        if in(name, ("ijl_f_setfield", "jl_f_setfield"))
            return common_setfield_fwd(2, B, orig, gutils, normalR, shadowR)
        end
        if in(name, ("ijl_f__apply_iterate", "jl_f__apply_iterate"))
            return common_apply_iterate_fwd(2, B, orig, gutils, normalR, shadowR)
        end
        if in(name, ("ijl_f__svec_ref", "jl_f__svec_ref"))
            return common_f_svec_ref_fwd(2, B, orig, gutils, normalR, shadowR)
        end
        if any(map(k->kind(k)==kind(StringAttribute("enzyme_inactive")), collect(function_attributes(F))))
            return true
        end
    end

    emit_error(B, orig, "Enzyme: jl_call calling convention not implemented in forward for "*string(orig))

    return false
end

function jlcall_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_apply_generic", "jl_apply_generic"))
            return common_generic_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if in(name, ("ijl_f__apply_latest", "ijl_f__call_latest", "jl_f__apply_latest", "jl_f__call_latest"))
            return common_apply_latest_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if in(name, ("ijl_new_structv", "jl_new_structv"))
            return common_newstructv_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if in(name, ("ijl_f_tuple", "jl_f_tuple"))
            return common_f_tuple_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if in(name, ("ijl_f_getfield", "jl_f_getfield"))
            return common_jl_getfield_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if in(name, ("ijl_s_getfield", "jl_s_getfield"))
            return common_setfield_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if in(name, ("ijl_f__apply_iterate", "jl_f__apply_iterate"))
            return common_apply_iterate_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if in(name, ("ijl_f__svec_rev", "jl_f__svec_ref"))
            return common_f_svec_ref_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if any(map(k->kind(k)==kind(StringAttribute("enzyme_inactive")), collect(function_attributes(F))))
            return true
        end
    end

    emit_error(B, orig, "Enzyme: jl_call calling convention not implemented in aug_forward for "*string(orig))

    return false
end

function jlcall_rev(B, orig, gutils, tape)
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_apply_generic", "jl_apply_generic"))
            common_generic_rev(2, B, orig, gutils, tape)
            return nothing
        end
        if in(name, ("ijl_f__apply_latest", "ijl_f__call_latest", "jl_f__apply_latest", "jl_f__call_latest"))
            common_apply_latest_rev(2, B, orig, gutils, tape)
            return nothing
        end
        if in(name, ("ijl_new_structv", "jl_new_structv"))
            common_newstructv_rev(2, B, orig, gutils, tape)
            return nothing
        end
        if in(name, ("ijl_f_tuple", "jl_f_tuple"))
            common_f_tuple_rev(2, B, orig, gutils, tape)
            return nothing
        end
        if in(name, ("ijl_f_getfield", "jl_f_getfield"))
            common_jl_getfield_rev(2, B, orig, gutils, tape)
            return nothing
        end
        if in(name, ("ijl_f_setfield", "jl_f_setfield"))
            common_setfield_rev(2, B, orig, gutils, tape)
            return nothing
        end
        if in(name, ("ijl_f__apply_iterate", "jl_f__apply_iterate"))
            common_apply_iterate_rev(2, B, orig, gutils, tape)
            return nothing
        end
        if in(name, ("ijl_f__svec_ref", "jl_f__svec_ref"))
            common_f_svec_ref_rev(2, B, orig, gutils, tape)
            return nothing
        end
        if any(map(k->kind(k)==kind(StringAttribute("enzyme_inactive")), collect(function_attributes(F))))
            return nothing
        end
    end

    emit_error(B, orig, "Enzyme: jl_call calling convention not implemented in reverse for "*string(orig))

    return nothing
end

function jlcall2_fwd(B, orig, gutils, normalR, shadowR)
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_invoke", "jl_invoke"))
            return common_invoke_fwd(2, B, orig, gutils, normalR, shadowR)
        end
        if any(map(k->kind(k)==kind(StringAttribute("enzyme_inactive")), collect(function_attributes(F))))
            return true
        end
    end

    @assert false "jl_call calling convention not implemented yet", orig

    return false
end

function jlcall2_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_invoke", "jl_invoke"))
            return common_invoke_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if any(map(k->kind(k)==kind(StringAttribute("enzyme_inactive")), collect(function_attributes(F))))
            return true
        end
    end

    @assert false "jl_call calling convention not implemented yet", orig

    return false
end

function jlcall2_rev(B, orig, gutils, tape)
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_invoke", "jl_invoke"))
            common_invoke_rev(2, B, orig, gutils, tape)
            return nothing
        end
        if any(map(k->kind(k)==kind(StringAttribute("enzyme_inactive")), collect(function_attributes(F))))
            return nothing
        end
    end

    @assert false "jl_call calling convention not implemented yet", orig

    return nothing
end


function common_invoke_fwd(offset, B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end

    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing
    
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    width = get_width(gutils)
    sret = generic_setup(orig, runtime_generic_fwd, AnyArray(1+Int(width)), gutils, #=start=#offset+1, B, false)
    AT = LLVM.ArrayType(T_prjlvalue, 1+Int(width))

    if shadowR != C_NULL
        if width == 1
            gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1)])
            shadow = LLVM.load!(B, T_prjlvalue, gep)
        else
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
            shadow = LLVM.UndefValue(ST)
            for i in 1:width
                gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(i)])
                ld = LLVM.load!(B, T_prjlvalue, gep)
                shadow = insert_value!(B, shadow, ld, i-1)
            end
        end
        unsafe_store!(shadowR, shadow.ref)
    end

    if normalR != C_NULL
        normal = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]))
        unsafe_store!(normalR, normal.ref)
    end

    return false
end

function common_invoke_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing
    
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    conv = LLVM.callconv(orig)

    width = get_width(gutils)
    sret = generic_setup(orig, runtime_generic_augfwd, AnyArray(2+Int(width)), gutils, #=start=#offset+1, B, false)
    AT = LLVM.ArrayType(T_prjlvalue, 2+Int(width))

    if shadowR != C_NULL
        if width == 1
            gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1)])
            shadow = LLVM.load!(B, T_prjlvalue, gep)
        else
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
            shadow = LLVM.UndefValue(ST)
            for i in 1:width
                gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(i)])
                ld = LLVM.load!(B, T_prjlvalue, gep)
                shadow = insert_value!(B, shadow, ld, i-1)
            end
        end
        unsafe_store!(shadowR, shadow.ref)
    end

    if normalR != C_NULL
        normal = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]))
        unsafe_store!(normalR, normal.ref)
    end

    tape = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1+width)]))
    unsafe_store!(tapeR, tape.ref)

    return false
end

function common_invoke_rev(offset, B, orig, gutils, tape)
    if !is_constant_value(gutils, orig) || !is_constant_inst(gutils, orig)
        width = get_width(gutils)
        generic_setup(orig, runtime_generic_rev, Nothing, gutils, #=start=#offset+1, B, true; tape)
    end

    return nothing
end

function invoke_fwd(B, orig, gutils, normalR, shadowR)
    common_invoke_fwd(1, B, orig, gutils, normalR, shadowR)
end

function invoke_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    common_invoke_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end

function invoke_rev(B, orig, gutils, tape)
    common_invoke_rev(1, B, orig, gutils, tape)
    return nothing
end


struct EnzymeRuntimeException <: Base.Exception
    msg::Cstring
end

function Base.showerror(io::IO, ece::EnzymeRuntimeException)
    print(io, "Enzyme execution failed.\n")
    msg = Base.unsafe_string(ece.msg)
    print(io, msg, '\n')
end

function throwerr(cstr::Cstring)
    throw(EnzymeRuntimeException(cstr))
end

function emit_error(B::LLVM.IRBuilder, orig, string)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    # 1. get the error function
    funcT = LLVM.FunctionType(LLVM.VoidType(), LLVMType[LLVM.PointerType(LLVM.Int8Type())])
    ptr = @cfunction(throwerr, Union{}, (Cstring,))
    ptr = convert(UInt, ptr)
    ptr = LLVM.ConstantInt(ptr)
    func = inttoptr!(B, ptr, LLVM.PointerType(funcT))
    if orig !== nothing
        bt = GPUCompiler.backtrace(orig)
        function printBT(io)
            print(io,"\nCaused by:")
            Base.show_backtrace(io, bt)
        end
        string*=sprint(io->Base.show_backtrace(io, bt))
    end

    # 2. Call error function and insert unreachable
    call!(B, funcT, func, LLVM.Value[globalstring_ptr!(B, string)])

    # FIXME(@wsmoses): Allow for emission of new BB in this code path
    # unreachable!(B)

    # 3. Change insertion point so that we don't stumble later
    # after_error = BasicBlock(fn, "after_error"; ctx)
    # position!(B, after_error)
end

function noop_fwd(B, orig, gutils, normalR, shadowR)
    return true
end

function noop_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    return true
end

function duplicate_rev(B, orig, gutils, tape)
    newg = new_from_original(gutils, orig)

    real_ops = collect(operands(orig))[1:end-1]
    ops = [lookup_value(gutils, new_from_original(gutils, o), B) for o in real_ops]
    
    c = call_samefunc_with_inverted_bundles!(B, gutils, orig, ops, [API.VT_Primal for _ in ops], #=lookup=#false)
    callconv!(c, callconv(orig))

    return nothing
end

function nested_codegen!(mode::API.CDerivativeMode, mod::LLVM.Module, f, tt, world)
    funcspec = GPUCompiler.methodinstance(typeof(f), tt, world)
    nested_codegen!(mode, mod, funcspec, world)
end

function prepare_llvm(mod, job, meta)
    interp = GPUCompiler.get_interpreter(job)
    for f in functions(mod)
        attributes = function_attributes(f)
        push!(attributes, StringAttribute("enzymejl_world", string(job.world)))
    end
    for (mi, k) in meta.compiled
        k_name = GPUCompiler.safe_name(k.specfunc)
        if !haskey(functions(mod), k_name)
            continue
        end
        llvmfn = functions(mod)[k_name]
    
        RT = Core.Compiler.typeinf_ext_toplevel(interp, mi).rettype

        _, _, returnRoots = get_return_info(RT)
        returnRoots = returnRoots !== nothing

        attributes = function_attributes(llvmfn)
        push!(attributes, StringAttribute("enzymejl_mi", string(convert(UInt, pointer_from_objref(mi)))))
        push!(attributes, StringAttribute("enzymejl_rt", string(convert(UInt, unsafe_to_pointer(RT)))))
        if returnRoots
            attr = StringAttribute("enzymejl_returnRoots", "")
            push!(parameter_attributes(llvmfn, 2), attr)
            for u in LLVM.uses(llvmfn)
                u = LLVM.user(u)
                @assert isa(u, LLVM.CallInst)
                LLVM.API.LLVMAddCallSiteAttribute(u, LLVM.API.LLVMAttributeIndex(2), attr)
            end
        end
    end
end

function nested_codegen!(mode::API.CDerivativeMode, mod::LLVM.Module, funcspec::Core.MethodInstance, world)
    # TODO: Put a cache here index on `mod` and f->tt

    
    # 3) Use the MI to create the correct augmented fwd/reverse
    # TODO:
    #  - GPU support
    #  - When OrcV2 only use a MaterializationUnit to avoid mutation of the module here

    target = DefaultCompilerTarget()
    params = PrimalCompilerParams(mode)
    job    = CompilerJob(funcspec, CompilerConfig(target, params; kernel=false), world)

    # TODO
    parent_job = nothing

    otherMod, meta = GPUCompiler.codegen(:llvm, job; optimize=false, cleanup=false, validate=false, parent_job=parent_job)
    prepare_llvm(otherMod, job, meta)

    entry = name(meta.entry)
   
    for f in functions(otherMod)
        permit_inlining!(f)
    end

    # Apply first stage of optimization's so that this module is at the same stage as `mod`
    optimize!(otherMod, JIT.get_tm())
    # 4) Link the corresponding module
    LLVM.link!(mod, otherMod)
    # 5) Call the function

    return functions(mod)[entry]
end

function referenceCaller(fn::Ref{Clos}, args...) where Clos
    fval = fn[]
    fval = fval::Clos
    fval(args...)
end

function runtime_pfor_fwd(thunk::ThunkTy, ft::FT, threading_args...)::Cvoid where {ThunkTy, FT}
    function fwd(tid_args...)
        if length(tid_args) == 0
            thunk(ft)
        else
            thunk(ft, Const(tid_args[1]))
        end
    end
    Base.Threads.threading_run(fwd, threading_args...)
    return
end

function runtime_pfor_augfwd(thunk::ThunkTy, ft::FT, ::Val{AnyJL}, ::Val{byRef}, threading_args...) where {ThunkTy, FT, AnyJL, byRef}
    TapeType = get_tape_type(ThunkTy)
    tapes = if AnyJL
        Vector{TapeType}(undef, Base.Threads.nthreads())
    else
        Base.unsafe_convert(Ptr{TapeType}, Libc.malloc(sizeof(TapeType)*Base.Threads.nthreads()))
    end

    function fwd(tid_args...)
        if length(tid_args) == 0
            if byRef
                tres = thunk(Const(referenceCaller), ft)
            else
                tres = thunk(ft)
            end
            tid = Base.Threads.threadid()
        else
            tid = tid_args[1]
            if byRef
                tres = thunk(Const(referenceCaller), ft, Const(tid))
            else
                tres = thunk(ft, Const(tid))
            end
        end

        if !AnyJL
            unsafe_store!(tapes, tres[1], tid)
        else
            @inbounds tapes[tid] = tres[1]
        end
    end
    Base.Threads.threading_run(fwd, threading_args...)
    return tapes
end

function runtime_pfor_rev(thunk::ThunkTy, ft::FT, ::Val{AnyJL}, ::Val{byRef}, tapes, threading_args...) where {ThunkTy, FT, AnyJL, byRef}
    function rev(tid_args...)
        tid = if length(tid_args) == 0
            tid = Base.Threads.threadid()
        else
            tid_args[1]
        end

        tres = if !AnyJL
            unsafe_load(tapes, tid)
        else
            @inbounds tapes[tid]
        end

        if length(tid_args) == 0
            if byRef
                thunk(Const(referenceCaller), ft, tres)
            else
                thunk(ft, tres)
            end
        else
            if byRef
                thunk(Const(referenceCaller), ft, Const(tid), tres)
            else
                thunk(ft, Const(tid), tres)
            end
        end
    end

    Base.Threads.threading_run(rev, threading_args...)
    if !AnyJL
        Libc.free(tapes)
    end
    return nothing
end

@inline function threadsfor_common(orig, gutils, B, mode, tape=nothing)

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    llvmfn = LLVM.called_operand(orig)
    mi = nothing
    fwdmodenm = nothing
    augfwdnm = nothing
    adjointnm = nothing
    TapeType = nothing
    attributes = function_attributes(llvmfn)
    for fattr in collect(attributes)
        if isa(fattr, LLVM.StringAttribute)
            if kind(fattr) == "enzymejl_mi"
                ptr = reinterpret(Ptr{Cvoid}, parse(UInt, LLVM.value(fattr)))
                mi = Base.unsafe_pointer_to_objref(ptr)
            end
            if kind(fattr) == "enzymejl_tapetype"
                ptr = reinterpret(Ptr{Cvoid}, parse(UInt, LLVM.value(fattr)))
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


    # TODO actually do modifiedBetween
@static if VERSION < v"1.8-"
    e_tt = Tuple{}
    modifiedBetween = (mode != API.DEM_ForwardMode, )
else
    e_tt = Tuple{Const{Int}}
    modifiedBetween = (mode != API.DEM_ForwardMode, false)
end

    world = enzyme_extract_world(LLVM.parent(position(B)))

    pfuncT = funcT

    mi2 = fspec(funcT, e_tt, world)

    refed = false

    # TODO: Clean this up and add to `nested_codegen!` asa feature
    width = get_width(gutils)

    ops = collect(operands(orig))[1:end-1]
    dupClosure = !isghostty(funcT) && !Core.Compiler.isconstType(funcT) && !is_constant_value(gutils, ops[1])
    pdupClosure = dupClosure

    subfunc = nothing
    if mode == API.DEM_ForwardMode
        if fwdmodenm === nothing
            etarget = Compiler.EnzymeTarget()
            eparams = Compiler.EnzymeCompilerParams(Tuple{(dupClosure ? Duplicated : Const){funcT}, e_tt.parameters...}, API.DEM_ForwardMode, width, Const{Nothing}, #=runEnzyme=#true, #=abiwrap=#true, modifiedBetween, #=returnPrimal=#false, #=shadowInit=#false, UnknownTapeType, FFIABI)
            ejob    = Compiler.CompilerJob(mi2, CompilerConfig(etarget, eparams; kernel=false), world)

            cmod, fwdmodenm, _, _ = _thunk(ejob, #=postopt=#false)
            
            LLVM.link!(mod, cmod)

            push!(attributes, StringAttribute("enzymejl_forward", fwdmodenm))
            push!(function_attributes(functions(mod)[fwdmodenm]), EnumAttribute("alwaysinline"))
            permit_inlining!(functions(mod)[fwdmodenm])
        end
        thunkTy = ForwardModeThunk{Ptr{Cvoid}, dupClosure ? Duplicated{funcT} : Const{funcT}, Const{Nothing}, e_tt, Val{width},  #=returnPrimal=#Val(false)}
        subfunc = functions(mod)[fwdmodenm]

    elseif mode == API.DEM_ReverseModePrimal || mode == API.DEM_ReverseModeGradient

        # TODO can optimize to only do if could contain a float
        if dupClosure
            has_active = false
            todo = Type[funcT]
            while length(todo) != 0
                T = pop!(todo)
                if !allocatedinline(T)
                    continue
                end
                if fieldcount(T) == 0
                    if T <: Integer
                        continue
                    end
                    has_active = true
                    break
                end
                for f in 1:fieldcount(T)
                    push!(todo, fieldtype(T, f))
                end
            end

            if has_active
                refed = true
                e_tt = Tuple{Duplicated{Base.RefValue{funcT}}, e_tt.parameters...}
                funcT = Core.Typeof(referenceCaller)
                dupClosure = false
                modifiedBetween = (false, modifiedBetween...)
                mi2 = fspec(funcT, e_tt, world)
            end
        end

        if augfwdnm === nothing || adjointnm === nothing
            etarget = Compiler.EnzymeTarget()
            # TODO modifiedBetween
            eparams = Compiler.EnzymeCompilerParams(Tuple{(dupClosure ? Duplicated : Const){funcT}, e_tt.parameters...}, API.DEM_ReverseModePrimal, width, Const{Nothing}, #=runEnzyme=#true, #=abiwrap=#true, modifiedBetween, #=returnPrimal=#false, #=shadowInit=#false, UnknownTapeType, FFIABI)
            ejob    = Compiler.CompilerJob(mi2, CompilerConfig(etarget, eparams; kernel=false), world)

            cmod, adjointnm, augfwdnm, TapeType = _thunk(ejob, #=postopt=#false)

            LLVM.link!(mod, cmod)

            push!(attributes, StringAttribute("enzymejl_augforward", augfwdnm))
            push!(function_attributes(functions(mod)[augfwdnm]), EnumAttribute("alwaysinline"))
            permit_inlining!(functions(mod)[augfwdnm])

            push!(attributes, StringAttribute("enzymejl_adjoint", adjointnm))
            push!(function_attributes(functions(mod)[adjointnm]), EnumAttribute("alwaysinline"))
            permit_inlining!(functions(mod)[adjointnm])

            push!(attributes, StringAttribute("enzymejl_tapetype", string(convert(UInt, unsafe_to_pointer(TapeType)))))
            
        end

        if mode == API.DEM_ReverseModePrimal
            thunkTy = AugmentedForwardThunk{Ptr{Cvoid}, dupClosure ? Duplicated{funcT} : Const{funcT}, Const{Nothing}, e_tt, Val{width}, #=returnPrimal=#Val(true), TapeType}
            subfunc = functions(mod)[augfwdnm]
       else
           thunkTy = AdjointThunk{Ptr{Cvoid}, dupClosure ? Duplicated{funcT} : Const{funcT}, Const{Nothing}, e_tt, Val{width}, TapeType}
            subfunc = functions(mod)[adjointnm]
        end
    else
        @assert "Unknown mode"
    end

    ppfuncT = pfuncT
    dpfuncT = width == 1 ? pfuncT : NTuple{(Int)width, pfuncT}

    if refed
        dpfuncT = Base.RefValue{dpfuncT}
        pfuncT = Base.RefValue{pfuncT}
    end

    dfuncT = pfuncT
    if pdupClosure
        if width == 1
            dfuncT = Duplicated{dfuncT}
        else
            dfuncT = BatchDuplicated{dfuncT, Int(width)}
        end
    else
        dfuncT = Const{dfuncT}
    end

    vals = LLVM.Value[]

    alloctx = LLVM.IRBuilder()
    position!(alloctx, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))
    ll_th =  convert(LLVMType, thunkTy)
    al = alloca!(alloctx, ll_th)
    al = addrspacecast!(B, al, LLVM.PointerType(ll_th, Tracked))
    al = addrspacecast!(B, al, LLVM.PointerType(ll_th, Derived))
    push!(vals, al)

    copies = []
    if !isghostty(dfuncT)

        llty = convert(LLVMType, dfuncT)

        alloctx = LLVM.IRBuilder()
        position!(alloctx, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))
        al = alloca!(alloctx, llty)

        if !isghostty(ppfuncT)
            v = new_from_original(gutils, ops[1])
            if mode == API.DEM_ReverseModeGradient
                v = lookup_value(gutils, v, B)
            end

            pllty = convert(LLVMType, ppfuncT)
            pv = nothing
            if value_type(v) != pllty
                pv = v
                v = load!(B, pllty, v)
            end
        else
            v = makeInstanceOf(ppfuncT, ctx)
        end

        if refed
            val0 = val = emit_allocobj!(B, pfuncT)
            val = bitcast!(B, val, LLVM.PointerType(pllty, addrspace(value_type(val))))
            val = addrspacecast!(B, val, LLVM.PointerType(pllty, Derived))
            store!(B, v, val)
            if pv !== nothing
                push!(copies, (pv, val, pllty))
            end

            if any_jltypes(pllty)
                emit_writebarrier!(B, get_julia_inner_types(B, val0, v))
            end
        else
            val0 = v
        end

        ptr = inbounds_gep!(B, llty, al, [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), 0)])
        store!(B, val0, ptr)

        if pdupClosure

            if !isghostty(ppfuncT)
                dv = invert_pointer(gutils, ops[1], B)
                if mode == API.DEM_ReverseModeGradient
                    dv = lookup_value(gutils, dv, B)
                end

                spllty = LLVM.LLVMType(API.EnzymeGetShadowType(width, pllty))
                pv = nothing
                if value_type(dv) != spllty
                    pv = dv
                    dv = load!(B, spllty, dv)
                end
            else
                @assert false
            end

            if refed
                dval0 = dval = emit_allocobj!(B, dpfuncT)
                dval = bitcast!(B, dval, LLVM.PointerType(spllty, addrspace(value_type(dval))))
                dval = addrspacecast!(B, dval, LLVM.PointerType(spllty, Derived))
                store!(B, dv, dval)
                if pv !== nothing
                    push!(copies, (pv, dval, spllty))
                end
                if any_jltypes(spllty)
                    emit_writebarrier!(B, get_julia_inner_types(B, dval0, dv))
                end
            else
                dval0 = dv
            end

            dptr = inbounds_gep!(B, llty, al, [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), 1)])
            store!(B, dval0, dptr)
        end

        al = addrspacecast!(B, al, LLVM.PointerType(llty, Derived))

        push!(vals, al)
    end

    if tape !== nothing
        push!(vals, tape)
    end

    @static if VERSION < v"1.8-"
    else
        push!(vals, new_from_original(gutils, operands(orig)[end-1]))
    end
    return refed, LLVM.name(subfunc), dfuncT, vals, thunkTy, TapeType, copies
end

function threadsfor_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing

    _, sname, dfuncT, vals, thunkTy, _, _ = threadsfor_common(orig, gutils, B, API.DEM_ForwardMode)

@static if VERSION < v"1.8-"
    tt = Tuple{thunkTy, dfuncT}
else
    tt = Tuple{thunkTy, dfuncT, Bool}
end
    mode = get_mode(gutils)
    world = enzyme_extract_world(LLVM.parent(position(B)))
    entry = nested_codegen!(mode, mod, runtime_pfor_fwd, tt, world)
    push!(function_attributes(entry), EnumAttribute("alwaysinline"))

    pval = const_ptrtoint(functions(mod)[sname], convert(LLVMType, Ptr{Cvoid}))
    pval = LLVM.ConstantArray(value_type(pval), [pval])
    store!(B, pval, vals[1])

    cal = LLVM.call!(B, LLVM.function_type(entry), entry, vals)
    debug_from_orig!(gutils, cal, orig)

    # Delete the primal code
    if normal !== nothing
        unsafe_store!(normalR, C_NULL)
    else
        ni = new_from_original(gutils, orig)
        API.EnzymeGradientUtilsErase(gutils, ni)
    end
    return false
end

function threadsfor_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end

    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing

    byRef, sname, dfuncT, vals, thunkTy, _, copies = threadsfor_common(orig, gutils, B, API.DEM_ReverseModePrimal)

@static if VERSION < v"1.8-"
    if byRef
        emit_error(B, orig, "Enzyme: active variable in Threads.@threads closure "*(string(eltype(eltype(dfuncT))))*" not supported")
    end
end

@static if VERSION < v"1.8-"
    tt = Tuple{thunkTy, dfuncT, Val{any_jltypes(get_tape_type(thunkTy))}, Val{byRef}}
else
    tt = Tuple{thunkTy, dfuncT, Val{any_jltypes(get_tape_type(thunkTy))}, Val{byRef}, Bool}
end
    mode = get_mode(gutils)
    world = enzyme_extract_world(LLVM.parent(position(B)))
    entry = nested_codegen!(mode, mod, runtime_pfor_augfwd, tt, world)
    push!(function_attributes(entry), EnumAttribute("alwaysinline"))

    pval = const_ptrtoint(functions(mod)[sname], convert(LLVMType, Ptr{Cvoid}))
    pval = LLVM.ConstantArray(value_type(pval), [pval])
    store!(B, pval, vals[1])

    tape = LLVM.call!(B, LLVM.function_type(entry), entry, vals)
    debug_from_orig!(gutils, tape, orig)

    if !any_jltypes(get_tape_type(thunkTy))
        if value_type(tape) != convert(LLVMType, Ptr{Cvoid})
            tape = LLVM.ConstantInt(0)
            GPUCompiler.@safe_warn "Illegal calling convention for threadsfor augfwd"
        end
    end

    # Delete the primal code
    if normal !== nothing
        unsafe_store!(normalR, C_NULL)
    else
        ni = new_from_original(gutils, orig)
        API.EnzymeGradientUtilsErase(gutils, ni)
    end

    unsafe_store!(tapeR, tape.ref)

    return false
end

function threadsfor_rev(B, orig, gutils, tape)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    world = enzyme_extract_world(LLVM.parent(position(B)))
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return
    end

    byRef, sname, dfuncT, vals, thunkTy, TapeType, copies = threadsfor_common(orig, gutils, B, API.DEM_ReverseModeGradient, tape)

    STT = if !any_jltypes(TapeType)
        Ptr{TapeType}
    else
        Vector{TapeType}
    end

@static if VERSION < v"1.8-"
    tt = Tuple{thunkTy, dfuncT, Val{any_jltypes(get_tape_type(thunkTy))}, Val{byRef}, STT }
else
    tt = Tuple{thunkTy, dfuncT, Val{any_jltypes(get_tape_type(thunkTy))}, Val{byRef}, STT, Bool}
end
    mode = get_mode(gutils)
    entry = nested_codegen!(mode, mod, runtime_pfor_rev, tt, world)
    push!(function_attributes(entry), EnumAttribute("alwaysinline"))

    pval = const_ptrtoint(functions(mod)[sname], convert(LLVMType, Ptr{Cvoid}))
    pval = LLVM.ConstantArray(value_type(pval), [pval])
    store!(B, pval, vals[1])

    cal = LLVM.call!(B, LLVM.function_type(entry), entry, vals)
    debug_from_orig!(gutils, cal, orig)

    for (pv, val, pllty) in copies
        ld = load!(B, pllty, val)
        store!(B, ld, pv)
    end
    return nothing
end

include("compiler/pmap.jl")

function newtask_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    width = get_width(gutils)
    mode = get_mode(gutils)

    world = enzyme_extract_world(LLVM.parent(position(B)))

    ops = collect(operands(orig))

    vals = LLVM.Value[
                       unsafe_to_llvm(runtime_newtask_fwd),
                       unsafe_to_llvm(Val(world)),
                       new_from_original(gutils, ops[1]),
                       invert_pointer(gutils, ops[1], B),
                       new_from_original(gutils, ops[2]),
                       (sizeof(Int) == sizeof(Int64) ? emit_box_int64! : emit_box_int32!)(B, new_from_original(gutils, ops[3])),
                       unsafe_to_llvm(Val(width)),
                      ]

    ntask = emit_apply_generic!(B, vals)
    debug_from_orig!(gutils, ntask, orig)

    # TODO: GC, ret
    if shadowR != C_NULL
        unsafe_store!(shadowR, ntask.ref)
    end

    if normalR != C_NULL
        unsafe_store!(normalR, ntask.ref)
    end

    return false
end

function newtask_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    # fn, dfn = augmentAndGradient(fn)
    # t = jl_new_task(fn)
    # # shadow t
    # dt = jl_new_task(dfn)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    GPUCompiler.@safe_warn "active variables passed by value to jl_new_task are not yet supported"
    width = get_width(gutils)
    mode = get_mode(gutils)

    uncacheable = get_uncacheable(gutils, orig)
    ModifiedBetween = (uncacheable[1] != 0,)

    world = enzyme_extract_world(LLVM.parent(position(B)))

    ops = collect(operands(orig))

    vals = LLVM.Value[
                       unsafe_to_llvm(runtime_newtask_augfwd),
                       unsafe_to_llvm(Val(world)),
                       new_from_original(gutils, ops[1]),
                       invert_pointer(gutils, ops[1], B),
                       new_from_original(gutils, ops[2]),
                       (sizeof(Int) == sizeof(Int64) ? emit_box_int64! : emit_box_int32!)(B, new_from_original(gutils, ops[3])),
                       unsafe_to_llvm(Val(width)),
                       unsafe_to_llvm(Val(ModifiedBetween)),
                      ]

    ntask = emit_apply_generic!(B, vals)
    debug_from_orig!(gutils, ntask, orig)
    sret = ntask

    AT = LLVM.ArrayType(T_prjlvalue, 2)
    sret = LLVM.addrspacecast!(B, sret, LLVM.PointerType(T_jlvalue, Derived))
    sret = LLVM.pointercast!(B, sret, LLVM.PointerType(AT, Derived))

    if shadowR != C_NULL
        shadow = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1)]))
        unsafe_store!(shadowR, shadow.ref)
    end

    if normalR != C_NULL
        normal = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]))
        unsafe_store!(normalR, normal.ref)
    end

    return false
end

function newtask_rev(B, orig, gutils, tape)
    return nothing
end

function set_task_tid_fwd(B, orig, gutils, normalR, shadowR)
    ops = collect(operands(orig))[1:end-1]
    if is_constant_value(gutils, ops[1])
        return true
    end

    inv = invert_pointer(gutils, ops[1], B)
    width = get_width(gutils)
    if width == 1
        nops = LLVM.Value[inv, new_from_original(gutils, ops[2])]
        valTys = API.CValueType[API.VT_Shadow, API.VT_Primal]
        cal = call_samefunc_with_inverted_bundles!(B, gutils, orig, nops, valTys, #=lookup=#false)
        debug_from_orig!(gutils, cal, orig)
        callconv!(cal, callconv(orig))
    else
        for idx in 1:width
            nops = LLVM.Value[extract_value(B, inv, idx-1),
                              new_from_original(gutils, ops[2])]
            valTys = API.CValueType[API.VT_Shadow, API.VT_Primal]
            cal = call_samefunc_with_inverted_bundles!(B, gutils, orig, nops, valTys, #=lookup=#false)
    
            debug_from_orig!(gutils, cal, orig)
            callconv!(cal, callconv(orig))
        end
    end

    return false
end

function set_task_tid_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    set_task_tid_fwd(B, orig, gutils, normalR, shadowR)
end

function set_task_tid_rev(B, orig, gutils, tape)
    return nothing
end

function enq_work_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return false
end

function enq_work_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    enq_work_fwd(B, orig, gutils, normalR, shadowR)
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

function enq_work_rev(B, orig, gutils, tape)
    # jl_wait(shadow(t))
    origops = LLVM.operands(orig)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    waitfn = find_match(mod, "jl_wait")
    if waitfn === nothing
        emit_error(B, orig, "Enzyme: could not find jl_wait fn to create shadow of jl_enq_work")
        return nothing
    end
    @assert waitfn !== nothing
    shadowtask = lookup_value(gutils, invert_pointer(gutils, origops[1], B), B)
    cal = LLVM.call!(B, LLVM.function_type(waitfn), waitfn, [shadowtask])
    debug_from_orig!(gutils, cal, orig)
    callconv!(cal, callconv(orig))
    return nothing
end

function wait_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return false
end

function wait_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return false
end

function wait_rev(B, orig, gutils, tape)
    # jl_enq_work(shadow(t))
    origops = LLVM.operands(orig)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    enq_work_fn = find_match(mod, "jl_enq_work")
    if enq_work_fn === nothing
        emit_error(B, orig, "Enzyme: could not find jl_enq_work fn to create shadow of wait")
        return nothing
    end
    @assert enq_work_fn !== nothing
    shadowtask = lookup_value(gutils, invert_pointer(gutils, origops[1], B), B)
    cal = LLVM.call!(B, LLVM.function_type(enq_work_fn), enq_work_fn, [shadowtask])
    debug_from_orig!(gutils, cal, orig)
    callconv!(cal, callconv(orig))
    return nothing
end

function removed_ret_parms(orig::LLVM.CallInst)
    F = LLVM.called_operand(orig)
    if !isa(F, LLVM.Function)
        return false, UInt64[]
    end
    return removed_ret_parms(F)
end

function removed_ret_parms(F::LLVM.Function)
    parmsRemoved = UInt64[]
    parmrem = nothing
    retRemove = false
    for a in collect(function_attributes(F))
        if isa(a, StringAttribute) 
            if kind(a) == "enzyme_parmremove"
                parmrem = a
            end
            if kind(a) == "enzyme_retremove"
                retRemove = true
            end
        end
    end
    if parmrem !== nothing
        str = value(parmrem)
        for v in split(str, ",")
            push!(parmsRemoved, parse(UInt64, v))
        end
    end
    return retRemove, parmsRemoved
end

function enzyme_custom_setup_args(B, orig, gutils, mi, RT, reverse, isKWCall)
    ops = collect(operands(orig))
    called = ops[end]
    ops = ops[1:end-1]
    width = get_width(gutils)
    kwtup = nothing

    args = LLVM.Value[]
    activity = Type[]
    overwritten = Bool[]

    actives = LLVM.Value[]

    uncacheable = get_uncacheable(gutils, orig)
    mode = get_mode(gutils)
    
    retRemoved, parmsRemoved = removed_ret_parms(orig)

    @assert length(parmsRemoved) == 0

    _, sret, returnRoots = get_return_info(RT)
    sret = sret !== nothing
    returnRoots = returnRoots !== nothing

    cv = LLVM.called_operand(orig)
    swiftself = any(any(map(k->kind(k)==kind(EnumAttribute("swiftself")), collect(parameter_attributes(cv, i)))) for i in 1:length(collect(parameters(cv))))
	jlargs = classify_arguments(mi.specTypes, called_type(orig), sret, returnRoots, swiftself, parmsRemoved)

    alloctx = LLVM.IRBuilder()
    position!(alloctx, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))

    for arg in jlargs
        @assert arg.cc != RemovedParam
        if arg.cc == GPUCompiler.GHOST
            @assert isghostty(arg.typ) || Core.Compiler.isconstType(arg.typ)
            if isKWCall && arg.arg_i == 2
                Ty = arg.typ
                kwtup = Ty
                continue
            end
            push!(activity, Const{arg.typ})
            # Don't push overwritten for Core.kwcall
            if !(isKWCall && arg.arg_i == 1)
                push!(overwritten, false)
            end
            if Core.Compiler.isconstType(arg.typ) && !Core.Compiler.isconstType(Const{arg.typ})
                llty = convert(LLVMType, Const{arg.typ})
                al0 = al = emit_allocobj!(B, Const{arg.typ})
                al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(value_type(al))))
                al = addrspacecast!(B, al, LLVM.PointerType(llty, Derived))

                ptr = inbounds_gep!(B, llty, al, [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), 0)])
                val = unsafe_to_llvm(arg.typ.parameters[1])
                store!(B, val, ptr)

                if any_jltypes(llty)
                    emit_writebarrier!(B, get_julia_inner_types(B, al0, val))
                end
                push!(args, al)
            else
                @assert isghostty(Const{arg.typ}) || Core.Compiler.isconstType(Const{arg.typ})
            end
            continue
        end
        @assert !(isghostty(arg.typ) || Core.Compiler.isconstType(arg.typ))

        op = ops[arg.codegen.i]
        # Don't push the keyword args to uncacheable
        if !(isKWCall && arg.arg_i == 2)
            push!(overwritten, uncacheable[arg.codegen.i] != 0)
        end

        val = new_from_original(gutils, op)
        if reverse
            val = lookup_value(gutils, val, B)
        end

        activep = API.EnzymeGradientUtilsGetDiffeType(gutils, op, #=isforeign=#false)

        if isKWCall && arg.arg_i == 2
            Ty = arg.typ

            push!(args, val)

            # Only constant kw arg tuple's are currently supported
            if activep == API.DFT_CONSTANT
                kwtup = Ty
            else
                @assert activep == API.DFT_DUP_ARG
                kwtup = Duplicated{Ty}
            end
            continue
        end

        # TODO type analysis deduce if duplicated vs active
        if activep == API.DFT_CONSTANT
            Ty = Const{arg.typ}
            llty = convert(LLVMType, Ty)
            arty = convert(LLVMType, arg.typ; allow_boxed=true)
            al0 = al = emit_allocobj!(B, Ty)
            al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(value_type(al))))
            al = addrspacecast!(B, al, LLVM.PointerType(llty, Derived))

            ptr = inbounds_gep!(B, llty, al, [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), 0)])
            if value_type(val) != eltype(value_type(ptr))
                val = load!(B, arty, val)
            end
            store!(B, val, ptr)

            if any_jltypes(llty)
                emit_writebarrier!(B, get_julia_inner_types(B, al0, val))
            end

            push!(args, al)

            push!(activity, Ty)

        elseif activep == API.DFT_OUT_DIFF || (mode != API.DEM_ForwardMode && active_reg(arg.typ) )
            Ty = Active{arg.typ}
            llty = convert(LLVMType, Ty)
            arty = convert(LLVMType, arg.typ; allow_boxed=true)
            al0 = al = emit_allocobj!(B, Ty)
            al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(value_type(al))))
            al = addrspacecast!(B, al, LLVM.PointerType(llty, Derived))

            ptr = inbounds_gep!(B, llty, al, [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), 0)])
            if value_type(val) != eltype(value_type(ptr))
                @assert !overwritten[end]
                val = load!(B, arty, val)
            end
            store!(B, val, ptr)

            if any_jltypes(llty)
                emit_writebarrier!(B, get_julia_inner_types(B, al0, val))
            end

            push!(args, al)

            push!(activity, Ty)
            push!(actives, op)
        else
            ival = invert_pointer(gutils, op, B)
            if reverse
                ival = lookup_value(gutils, ival, B)
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
                    Ty = BatchDuplicated{arg.typ, Int(width)}
                else
                    @assert activep == API.DFT_DUP_NONEED
                    Ty = BatchDuplicatedNoNeed{arg.typ, Int(width)}
                end
            end

            llty = convert(LLVMType, Ty)
            arty = convert(LLVMType, arg.typ; allow_boxed=true)
            sarty = LLVM.LLVMType(API.EnzymeGetShadowType(width, arty))
            al0 = al = emit_allocobj!(B, Ty)
            al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(value_type(al))))
            al = addrspacecast!(B, al, LLVM.PointerType(llty, Derived))

            ptr = inbounds_gep!(B, llty, al, [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), 0)])
            if value_type(val) != eltype(value_type(ptr))
                val = load!(B, arty, val)
                ptr_val = ival
                ival = UndefValue(sarty)
                for idx in 1:width
                    ev = (width == 1) ? ptr_val : extract_value!(B, ptr_val, idx-1)
                    ld = load!(B, arty, ev)
                    ival = (width == 1 ) ? ld : insert_value!(B, ival, ld, idx-1)
                end
            end
            store!(B, val, ptr)

            iptr = inbounds_gep!(B, llty, al, [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), 1)])
            store!(B, ival, iptr)

            if any_jltypes(llty)
                emit_writebarrier!(B, get_julia_inner_types(B, al0, val, ival))
            end

            push!(args, al)
            push!(activity, Ty)
        end

    end
    return args, activity, (overwritten...,), actives, kwtup
end

function enzyme_custom_setup_ret(gutils, orig, mi, RealRt)
    width = get_width(gutils)
    mode = get_mode(gutils)

    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)

    activep = API.EnzymeGradientUtilsGetReturnDiffeType(gutils, orig, needsPrimalP, needsShadowP)
    needsPrimal = needsPrimalP[] != 0
    origNeedsPrimal = needsPrimal
    _, sret, _ = get_return_info(RealRt)
    if sret !== nothing
        activep = API.EnzymeGradientUtilsGetDiffeType(gutils, operands(orig)[1], #=isforeign=#false)
        needsPrimal = activep == API.DFT_DUP_ARG || activep == API.DFT_CONSTANT
        needsShadowP[] = false
    end

    if !needsPrimal && activep == API.DFT_DUP_ARG
        activep = API.DFT_DUP_NONEED
    end

    if activep == API.DFT_CONSTANT
        RT = Const{RealRt}

    elseif activep == API.DFT_OUT_DIFF || (mode != API.DEM_ForwardMode && active_reg(RealRt) )
        RT = Active{RealRt}

    elseif activep == API.DFT_DUP_ARG
        if width == 1
            RT = Duplicated{RealRt}
        else
            RT = BatchDuplicated{RealRt, Int(width)}
        end
    else
        @assert activep == API.DFT_DUP_NONEED
        if width == 1
            RT = DuplicatedNoNeed{RealRt}
        else
            RT = BatchDuplicatedNoNeed{RealRt, Int(width)}
        end
    end
    return RT, needsPrimal, needsShadowP[] != 0, origNeedsPrimal
end

function enzyme_custom_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end

    width = get_width(gutils)

    if shadowR != C_NULL
        unsafe_store!(shadowR,UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))).ref)
    end

    # TODO: don't inject the code multiple times for multiple calls

    # 1) extract out the MI from attributes
    mi, RealRt = enzyme_custom_extract_mi(orig)

    kwfunc = nothing

    isKWCall = isKWCallSignature(mi.specTypes)
    if isKWCall
        kwfunc = Core.kwfunc(EnzymeRules.forward)
    end

    # 2) Create activity, and annotate function spec
    args, activity, overwritten, actives, kwtup = enzyme_custom_setup_args(B, orig, gutils, mi, RealRt, #=reverse=#false, isKWCall)
    RT, needsPrimal, needsShadow, origNeedsPrimal = enzyme_custom_setup_ret(gutils, orig, mi, RealRt)

    alloctx = LLVM.IRBuilder()
    position!(alloctx, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))
    mode = get_mode(gutils)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    tt = copy(activity)
    if isKWCall
        popfirst!(tt)
        @assert kwtup !== nothing
        insert!(tt, 1, kwtup)
        insert!(tt, 2, Core.typeof(EnzymeRules.forward))
        insert!(tt, 4, Type{RT})
    else
        @assert kwtup === nothing
        insert!(tt, 2, Type{RT})
    end
    TT = Tuple{tt...}

    if kwtup !== nothing && kwtup <: Duplicated
        @safe_debug "Non-constant keyword argument found for " TT
        emit_error(B, orig, "Enzyme: Non-constant keyword argument found for " * string(TT))
        return false
    end

    # TODO get world
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    world = enzyme_extract_world(fn)
    @safe_debug "Trying to apply custom forward rule" TT isKWCall
    llvmf = nothing
    if isKWCall
        if EnzymeRules.isapplicable(kwfunc, TT; world)
            @safe_debug "Applying custom forward rule (kwcall)" TT
            llvmf = nested_codegen!(mode, mod, kwfunc, TT, world)
            fwd_RT = Core.Compiler.return_type(kwfunc, TT, world)
        end
    else
        if EnzymeRules.isapplicable(EnzymeRules.forward, TT; world)
            @safe_debug "Applying custom forward rule" TT
            llvmf = nested_codegen!(mode, mod, EnzymeRules.forward, TT, world)
            fwd_RT = Core.Compiler.return_type(EnzymeRules.forward, TT, world)
        end
    end

    if llvmf === nothing
        @safe_debug "No custom forward rule is applicable for" TT
        emit_error(B, orig, "Enzyme: No custom rule was appliable for " * string(TT))
        return false
    end
    
    push!(function_attributes(llvmf), EnumAttribute("alwaysinline", 0))

    swiftself = any(any(map(k->kind(k)==kind(EnumAttribute("swiftself")), collect(parameter_attributes(llvmf, i)))) for i in 1:length(collect(parameters(llvmf))))
    if swiftself
        pushfirst!(reinsert_gcmarker!(fn, B))
    end
    _, sret, returnRoots = get_return_info(enzyme_custom_extract_mi(llvmf)[2])
    if sret !== nothing
        sret = alloca!(alloctx, convert(LLVMType, eltype(sret)))
        pushfirst!(args, sret)
        if returnRoots !== nothing
            returnRoots = alloca!(alloctx, convert(LLVMType, eltype(returnRoots)))
            insert!(args, 2, returnRoots)
        else
            returnRoots = nothing
        end
    else
        sret = nothing
    end


    if length(args) != length(parameters(llvmf))
        GPUCompiler.@safe_error "Calling convention mismatch", args, llvmf, orig, isKWCall, kwtup, TT, sret, returnRoots
        return false
    end

    for i in eachindex(args)
        party = value_type(parameters(llvmf)[i])
        if value_type(args[i]) == party
            continue
        end
        GPUCompiler.@safe_error "Calling convention mismatch", party, args[i], i, llvmf, fn, args, sret, returnRoots
        return false
    end

    res = LLVM.call!(B, LLVM.function_type(llvmf), llvmf, args)
    debug_from_orig!(gutils, res, orig)
    callconv!(res, callconv(llvmf))

    hasNoRet = any(map(k->kind(k)==kind(EnumAttribute("noreturn")), collect(function_attributes(llvmf))))

    if hasNoRet
        return false
    end

    if sret !== nothing
        if LLVM.version().major >= 12
            attr = TypeAttribute("sret", eltype(value_type(parameters(llvmf)[1])))
        else
            attr = EnumAttribute("sret")
        end
        LLVM.API.LLVMAddCallSiteAttribute(res, LLVM.API.LLVMAttributeIndex(1), attr)
        res = load!(B, eltype(value_type(parameters(llvmf)[1])), sret)
    end
    if swiftself
        attr = EnumAttribute("swiftself")
        LLVM.API.LLVMAddCallSiteAttribute(res, LLVM.API.LLVMAttributeIndex(1+(sret !== nothing)), attr)
    end

    shadowV = C_NULL
    normalV = C_NULL

    if RT <: Const
        # TODO introduce const-no-need
        if needsPrimal || true
            if RealRt != fwd_RT
                emit_error(B, orig, "Enzyme: incorrect return type of const primal-only forward custom rule - "*(string(RT))*" "*string(activity)*" want just return type "*string(RealRt)*" found "*string(fwd_RT))
                return false
            end
            if get_return_info(RealRt)[2] !== nothing
                val = new_from_original(gutils, operands(orig)[1])
                store!(B, res, val)
            else
                normalV = res.ref
            end
        else
            if Nothing != fwd_RT
                emit_error(B, orig, "Enzyme: incorrect return type of const no-primal forward custom rule - "*(string(RT))*" "*string(activity)*" want just return type Nothing found "*string(fwd_RT))
                return false
            end
        end
    else
        if !needsPrimal
            ST = RealRt
            if width != 1
                ST = NTuple{Int(width), ST}
            end
            if ST != fwd_RT
                emit_error(B, orig, "Enzyme: incorrect return type of shadow-only forward custom rule - "*(string(RT))*" "*string(activity)*" want just shadow type "*string(ST)*" found "*string(fwd_RT))
                return false
            end
            if get_return_info(RealRt)[2] !== nothing
                dval_ptr = invert_pointer(gutils, operands(orig)[1], B)
                for idx in 1:width
                    ev = (width == 1) ? dval : extract_value!(B, dval, idx-1)
                    pev = (width == 1) ? dval_ptr : extract_value!(B, dval_ptr, idx-1)
                    store!(B, res, pev)
                end
            else
                shadowV = res.ref
            end
        else
            ST = if width == 1
                Duplicated{RealRt}
            else
                BatchDuplicated{RealRt, Int(width)}
            end
            if ST != fwd_RT
                emit_error(B, orig, "Enzyme: incorrect return type of prima/shadow forward custom rule - "*(string(RT))*" "*string(activity)*" want just shadow type "*string(ST)*" found "*string(fwd_RT))
                return false
            end
            if get_return_info(RealRt)[2] !== nothing
                val = new_from_original(gutils, operands(orig)[1])
                store!(B, extract_value!(B, res, 0), val)
                
                dval_ptr = invert_pointer(gutils, operands(orig)[1], B)
                dval = extract_value!(B, res, 1)
                for idx in 1:width
                    ev = (width == 1) ? dval : extract_value!(B, dval, idx-1)
                    pev = (width == 1) ? dval_ptr : extract_value!(B, dval_ptr, idx-1)
                    store!(B, ev, pev)
                end
            else
                normalV = extract_value!(B, res, 0).ref
                shadowV = extract_value!(B, res, 1).ref
            end
        end
    end

    if shadowR != C_NULL
        unsafe_store!(shadowR, shadowV)
    end

    # Delete the primal code
    if origNeedsPrimal
        unsafe_store!(normalR, normalV)
    else
        ni = new_from_original(gutils, orig)
        if value_type(ni) != LLVM.VoidType()
            API.EnzymeGradientUtilsReplaceAWithB(gutils, ni, LLVM.UndefValue(value_type(ni)))
        end
        API.EnzymeGradientUtilsErase(gutils, ni)
    end

    return false
end

function enzyme_custom_common_rev(forward::Bool, B, orig::LLVM.CallInst, gutils, normalR, shadowR, tape)::LLVM.API.LLVMValueRef

    ctx = LLVM.context(orig)

    width = get_width(gutils)

    shadowType = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
    if shadowR != C_NULL
        unsafe_store!(shadowR,UndefValue(shadowType).ref)
    end

    # TODO: don't inject the code multiple times for multiple calls

    # 1) extract out the MI from attributes
    mi, RealRt = enzyme_custom_extract_mi(orig)
    isKWCall = isKWCallSignature(mi.specTypes)

    # 2) Create activity, and annotate function spec
    args, activity, overwritten, actives, kwtup = enzyme_custom_setup_args(B, orig, gutils, mi, RealRt, #=reverse=#!forward, isKWCall)
    RT, needsPrimal, needsShadow, origNeedsPrimal = enzyme_custom_setup_ret(gutils, orig, mi, RealRt)

    alloctx = LLVM.IRBuilder()
    position!(alloctx, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))

    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    world = enzyme_extract_world(fn)

    C = EnzymeRules.Config{Bool(needsPrimal), Bool(needsShadow), Int(width), overwritten}
    
    mode = get_mode(gutils)

    ami = nothing

    augprimal_tt = copy(activity)
    if isKWCall
        popfirst!(augprimal_tt)
        @assert kwtup !== nothing
        insert!(augprimal_tt, 1, kwtup)
        insert!(augprimal_tt, 2, Core.typeof(EnzymeRules.augmented_primal))
        insert!(augprimal_tt, 3, C)
        insert!(augprimal_tt, 5, Type{RT})

        augprimal_TT = Tuple{augprimal_tt...}
        kwfunc = Core.kwfunc(EnzymeRules.augmented_primal)
        try
            ami = GPUCompiler.methodinstance(Core.Typeof(kwfunc), augprimal_TT, world)
            @safe_debug "Applying custom augmented_primal rule (kwcall)" TT=augprimal_TT
        catch e
        end
    else
        @assert kwtup === nothing
        insert!(augprimal_tt, 1, C)
        insert!(augprimal_tt, 3, Type{RT})

        augprimal_TT = Tuple{augprimal_tt...}
        try
            ami = GPUCompiler.methodinstance(Core.Typeof(EnzymeRules.augmented_primal), augprimal_TT, world)
            @safe_debug "Applying custom augmented_primal rule" TT=augprimal_TT
        catch e
        end
    end
    
    if ami !== nothing
        target = DefaultCompilerTarget()
        params = PrimalCompilerParams(mode)
        job    = CompilerJob(ami, CompilerConfig(target, params; kernel=false), world)
        interp = GPUCompiler.get_interpreter(job)
        aug_RT = something(Core.Compiler.typeinf_type(interp, ami.def, ami.specTypes, ami.sparam_vals), Any)
    else
        @safe_debug "No custom augmented_primal rule is applicable for" augprimal_TT
        emit_error(B, orig, "Enzyme: No custom augmented_primal rule was appliable for " * string(augprimal_TT))
        return C_NULL
    end

    if kwtup !== nothing && kwtup <: Duplicated
        @safe_debug "Non-constant keyword argument found for " augprimal_TT
        emit_error(B, orig, "Enzyme: Non-constant keyword argument found for " * string(augprimal_TT))
        return C_NULL
    end

    rev_TT = nothing
    rev_RT = nothing

    TapeT = Nothing

    if (aug_RT <: EnzymeRules.AugmentedReturn || aug_RT <: EnzymeRules.AugmentedReturnFlexShadow) && !(aug_RT isa UnionAll) && !(aug_RT isa Union) && !(aug_RT === Union{})
        TapeT = EnzymeRules.tape_type(aug_RT)
    end

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    llvmf = nothing

    if forward
        llvmf = nested_codegen!(mode, mod, ami, world)
        @assert llvmf !== nothing
    else
        tt = copy(activity)
        if isKWCall
            popfirst!(tt)
            @assert kwtup !== nothing
            insert!(tt, 1, kwtup)
            insert!(tt, 2, Core.typeof(EnzymeRules.reverse))
            insert!(tt, 3, C)
            insert!(tt, 5, RT <: Active ? RT : Type{RT})
            insert!(tt, 6, TapeT)
        else
            @assert kwtup === nothing
            insert!(tt, 1, C)
            insert!(tt, 3, RT <: Active ? RT : Type{RT})
            insert!(tt, 4, TapeT)
        end
        rev_TT = Tuple{tt...}

        if isKWCall
            rkwfunc = Core.kwfunc(EnzymeRules.reverse)
            if EnzymeRules.isapplicable(rkwfunc, rev_TT; world)
                @safe_debug "Applying custom reverse rule (kwcall)" TT=rev_TT
                llvmf = nested_codegen!(mode, mod, rkwfunc, rev_TT, world)
                rev_RT = Core.Compiler.return_type(rkwfunc, rev_TT, world)
            end
        else
            if EnzymeRules.isapplicable(EnzymeRules.reverse, rev_TT; world)
                @safe_debug "Applying custom reverse rule" TT=rev_TT
                llvmf = nested_codegen!(mode, mod, EnzymeRules.reverse, rev_TT, world)
                rev_RT = Core.Compiler.return_type(EnzymeRules.reverse, rev_TT, world)
            end
        end

        if llvmf == nothing
            @safe_debug "No custom reverse rule is applicable for" rev_TT
            emit_error(B, orig, "Enzyme: No custom reverse rule was appliable for " * string(rev_TT))
            return C_NULL
        end
    end
    push!(function_attributes(llvmf), EnumAttribute("alwaysinline", 0))

    needsTape = !isghostty(TapeT) && !Core.Compiler.isconstType(TapeT)

    tapeV = C_NULL
    if forward && needsTape
        tapeV = LLVM.UndefValue(convert(LLVMType, TapeT; allow_boxed=true)).ref
    end

    # if !forward
    #     argTys = copy(activity)
    #     if RT <: Active
    #         if width == 1
    #             push!(argTys, RealRt)
    #         else
    #             push!(argTys, NTuple{RealRt, (Int)width})
    #         end
    #     end
    #     push!(argTys, tapeType)
    #     llvmf = nested_codegen!(mode, mod, rev_func, Tuple{argTys...}, world)
    # end

    swiftself = any(any(map(k->kind(k)==kind(EnumAttribute("swiftself")), collect(parameter_attributes(llvmf, i)))) for i in 1:length(collect(parameters(llvmf))))

    if !forward
        if needsTape
            @assert tape != C_NULL
            sret = !isempty(parameters(llvmf)) && any(map(k->kind(k)==kind(EnumAttribute("sret")), collect(parameter_attributes(llvmf, 1))))
            innerTy = value_type(parameters(llvmf)[1+(kwtup!==nothing)+sret+(RT <: Active)+(isKWCall && !isghostty(rev_TT.parameters[4]))])
            if innerTy != value_type(tape)
                llty = convert(LLVMType, TapeT; allow_boxed=true)
                al0 = al = emit_allocobj!(B, TapeT)
                al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(value_type(al))))
                store!(B, tape, al)
                if any_jltypes(llty)
                    emit_writebarrier!(B, get_julia_inner_types(B, al0, tape))
                end
                tape = addrspacecast!(B, al, LLVM.PointerType(llty, Derived))
            end
            insert!(args, 1+(kwtup!==nothing)+(isKWCall && !isghostty(rev_TT.parameters[4])), tape)
        end
        if RT <: Active

            llty = convert(LLVMType, RT)

            if API.EnzymeGradientUtilsGetDiffeType(gutils, orig, #=isforeign=#false) == API.DFT_OUT_DIFF
                val = LLVM.Value(API.EnzymeGradientUtilsDiffe(gutils, orig, B))
            else
                llety = convert(LLVMType, eltype(RT))
                ptr_val = invert_pointer(gutils, operands(orig)[1], B)
                val = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llety)))
                for idx in 1:width
                    ev = (width == 1) ? ptr_val : extract_value!(B, ptr_val, idx-1)
                    ld = load!(B, llety, ev)
                    store!(B, LLVM.null(llety), ev)
                    val = (width == 1 ) ? ld : insert_value!(B, val, ld, idx-1)
                end
            end

            al0 = al = emit_allocobj!(B, RT)
            al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(value_type(al))))
            al = addrspacecast!(B, al, LLVM.PointerType(llty, Derived))

            ptr = inbounds_gep!(B, llty, al, [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), 0)])
            store!(B, val, ptr)

            if any_jltypes(llty)
                emit_writebarrier!(B, get_julia_inner_types(B, al0, val))
            end

            insert!(args, 1+(kwtup!==nothing)+(isKWCall && !isghostty(rev_TT.parameters[4])), al)
        end
    end

    if swiftself
        pushfirst!(reinsert_gcmarker!(fn, B))
    end

    _, sret, returnRoots = get_return_info(enzyme_custom_extract_mi(llvmf)[2])
    if sret !== nothing
        sret = alloca!(alloctx, convert(LLVMType, eltype(sret)))
        pushfirst!(args, sret)
        if returnRoots !== nothing
            returnRoots = alloca!(alloctx, convert(LLVMType, eltype(returnRoots)))
            insert!(args, 2, returnRoots)
        else
            returnRoots = nothing
        end
    else
        sret = nothing
    end

    if length(args) != length(parameters(llvmf))
        GPUCompiler.@safe_error "Calling convention mismatch", args, llvmf, orig, isKWCall, kwtup, augprimal_TT, rev_TT, fn, sret, returnRoots
        return tapeV
    end
    
    for i in 1:length(args)
        party =  value_type(parameters(llvmf)[i])
        if value_type(args[i]) == party
            continue
        end
        GPUCompiler.@safe_error "Calling convention mismatch", party, args[i], i, llvmf, augprimal_TT, rev_TT, fn, args, sret, returnRoots
        return tapeV
    end

    res = LLVM.call!(B, LLVM.function_type(llvmf), llvmf, args)
    ncall = res
    debug_from_orig!(gutils, res, orig)
    callconv!(res, callconv(llvmf))

    hasNoRet = any(map(k->kind(k)==kind(EnumAttribute("noreturn")), collect(function_attributes(llvmf))))

    if hasNoRet
        return tapeV
    end

    if sret !== nothing
        if LLVM.version().major >= 12
            attr = TypeAttribute("sret", eltype(value_type(parameters(llvmf)[1+swiftself])))
        else
            attr = EnumAttribute("sret")
        end
        LLVM.API.LLVMAddCallSiteAttribute(res, LLVM.API.LLVMAttributeIndex(1+swiftself), attr)
        res = load!(B, eltype(value_type(parameters(llvmf)[1+swiftself])), sret)
    end
    if swiftself
        attr = EnumAttribute("swiftself")
        LLVM.API.LLVMAddCallSiteAttribute(res, LLVM.API.LLVMAttributeIndex(1+(sret !== nothing)+(returnRoots !== nothing)), attr)
    end

    shadowV = C_NULL
    normalV = C_NULL


    if forward
        ShadT = RealRt
        if width != 1
            ShadT = NTuple{Int(width), RealRt}
        end
        ST = EnzymeRules.AugmentedReturn{needsPrimal ? RealRt : Nothing, needsShadow ? ShadT : Nothing, TapeT}
        if aug_RT != ST
            if aug_RT <: EnzymeRules.AugmentedReturnFlexShadow
                if convert(LLVMType, EnzymeRules.shadow_type(aug_RT); allow_boxed=true) !=
                    convert(LLVMType, EnzymeRules.shadow_type(ST)    ; allow_boxed=true)
                    emit_error(B, orig, "Enzyme: Augmented forward pass custom rule " * string(augprimal_TT) * " flex shadow ABI return type mismatch, expected "*string(ST)*" found "* string(aug_RT))
                    return tapeV
                end
                ST = EnzymeRules.AugmentedReturnFlexShadow{needsPrimal ? RealRt : Nothing, needsShadow ? EnzymeRules.shadow_type(aug_RT) : Nothing, TapeT}
            end
        end
        if aug_RT != ST
            ST = EnzymeRules.AugmentedReturn{needsPrimal ? RealRt : Nothing, needsShadow ? ShadT : Nothing, Any}
            emit_error(B, orig, "Enzyme: Augmented forward pass custom rule " * string(augprimal_TT) * " return type mismatch, expected "*string(ST)*" found "* string(aug_RT))
            return tapeV
        end

        idx = 0
        if needsPrimal
            @assert !isghostty(RealRt)
            normalV = extract_value!(B, res, idx)
            if get_return_info(RealRt)[2] !== nothing
                val = new_from_original(gutils, operands(orig)[1])
                store!(B, normalV, val)
            else
                @assert value_type(normalV) == value_type(orig)
                normalV = normalV.ref
            end
            idx+=1
        end
        if needsShadow
            @assert !isghostty(RealRt)
            shadowV = extract_value!(B, res, idx)
            if get_return_info(RealRt)[2] !== nothing
                dval = invert_pointer(gutils, operands(orig)[1], B)
                store!(B, shadowV, dval)
                shadowV = C_NULL
            else
                @assert value_type(shadowV) == shadowType
                shadowV = shadowV.ref
            end
            idx+=1
        end
        if needsTape
            tapeV = extract_value!(B, res, idx).ref
            idx+=1
        end
    else
        Tys = (A <: Active ? eltype(A) : Nothing for A in activity[2+isKWCall:end])
        ST = Tuple{Tys...}
        if rev_RT != ST
            emit_error(B, orig, "Enzyme: Reverse pass custom rule " * string(rev_TT) * " return type mismatch, expected "*string(ST)*" found "* string(rev_RT))
            return tapeV
        end
        if length(actives) >= 1 && !isa(value_type(res), LLVM.StructType) && !isa(value_type(res), LLVM.ArrayType)
            GPUCompiler.@safe_error "Shadow arg calling convention mismatch found return ", res
            return tapeV
        end

        idx = 0
        dl = string(LLVM.datalayout(LLVM.parent(LLVM.parent(LLVM.parent(orig)))))
        for (v, Ty) in zip(actives, Tys)
            TT = typetree(Ty, ctx, dl)
            Typ = C_NULL
            ext = extract_value!(B, res, idx)
            shadowVType = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(v)))
            if value_type(ext) != shadowVType
                size = sizeof(Ty)
                align = 0
                premask = C_NULL
                API.EnzymeGradientUtilsAddToInvertedPointerDiffeTT(gutils, orig, C_NULL, TT, size, v,           ext, B, align, premask)
            else
                @assert value_type(ext) == shadowVType
                API.EnzymeGradientUtilsAddToDiffe(gutils, v, ext, B, Typ)
            end
            idx+=1
        end
    end

    if forward
        if shadowR != C_NULL && shadowV != C_NULL
            unsafe_store!(shadowR, shadowV)
        end

        # Delete the primal code
        if origNeedsPrimal
            unsafe_store!(normalR, normalV)
        else
            ni = new_from_original(gutils, orig)
            API.EnzymeGradientUtilsErase(gutils, ni)
        end
    end

    return tapeV
end


function enzyme_custom_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    tape = enzyme_custom_common_rev(#=forward=#true, B, orig, gutils, normalR, shadowR, #=tape=#nothing)
    if tape != C_NULL
        unsafe_store!(tapeR, tape)
    end
    return false
end


function enzyme_custom_rev(B, orig, gutils, tape)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return
    end
    enzyme_custom_common_rev(#=forward=#false, B, orig, gutils, #=normalR=#C_NULL, #=shadowR=#C_NULL, #=tape=#tape)
    return nothing
end

function arraycopy_fwd(B, orig, gutils, normalR, shadowR)
    ctx = LLVM.context(orig)

    if is_constant_value(gutils, orig)
        return true
    end

    origops = LLVM.operands(orig)

    width = get_width(gutils)

    shadowin = invert_pointer(gutils, origops[1], B)

    i8 = LLVM.IntType(8)
    algn = 0

    if width == 1
        shadowres = call_samefunc_with_inverted_bundles!(B, gutils, orig, [shadowin], [API.VT_Shadow], #=lookup=#false)

        # TODO zero based off runtime types, rather than presume floatlike?
        if is_constant_value(gutils, origops[1])
            elSize = get_array_elsz(B, shadowin)
            elSize = LLVM.zext!(B, elSize, LLVM.IntType(8*sizeof(Csize_t)))
            len = get_array_len(B, shadowin)
            length = LLVM.mul!(B, len, elSize)
            isVolatile = LLVM.ConstantInt(LLVM.IntType(1), 0)
            GPUCompiler.@safe_warn "TODO forward zero-set of arraycopy used memset rather than runtime type"
            LLVM.memset!(B, get_array_data(B, shadowres), LLVM.ConstantInt(i8, 0, false), length, algn)
        end
        if API.runtimeActivity()
            prev = new_from_original(gutils, orig)
            shadowres = LLVM.select!(B, LLVM.icmp!(B, LLVM.API.LLVMIntNE, shadowin, new_from_original(gutils, origops[1])), shadowres, prev)
            API.moveBefore(prev, shadowres, B)
        end
    else
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx in 1:width
            ev = extract_value!(B, shadowin, idx-1)
            callv = call_samefunc_with_inverted_bundles!(B, gutils, orig, [ev], [API.VT_Shadow], #=lookup=#false)
            if is_constant_value(gutils, origops[1])
                elSize = get_array_elsz(B, shadowin)
                elSize = LLVM.zext!(B, elSize, LLVM.IntType(8*sizeof(Csize_t)))
                len = get_array_len(B, shadowin)
                length = LLVM.mul!(B, len, elSize)
                isVolatile = LLVM.ConstantInt(LLVM.IntType(1), 0)
                GPUCompiler.@safe_warn "TODO forward zero-set of arraycopy used memset rather than runtime type"
                LLVM.memset!(B, get_array_data(callv), LLVM.ConstantInt(i8, 0, false), length, algn)
            end
            if API.runtimeActivity()
                prev = new_from_original(gutils, orig)
                callv = LLVM.select!(B, LLVM.icmp!(B, LLVM.API.LLVMIntNE, ev, new_from_original(gutils, origops[1])), callv, prev)
                if idx == 1
                    API.moveBefore(prev, callv, B)
                end
            end
            shadowres = insert_value!(B, shadowres, callv, idx-1)
        end
    end

    unsafe_store!(shadowR, shadowres.ref)
	return false
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
    secretty = API.EnzymeConcreteTypeIsFloat(ct)

    off = sizeof(Cstring)
    if true # STORE_ARRAY_LEN
        off += sizeof(Csize_t)
    end
    #jl_array_flags_t
    off += 2

    actualOp = new_from_original(gutils, origArg)
    if fwd
        B0 = B
    elseif typeof(actualOp) <: LLVM.Argument
        B0 = LLVM.IRBuilder()
        position!(B0, first(instructions(new_from_original(gutils, LLVM.entry(LLVM.parent(LLVM.parent(orig)))))))
    else
        B0 = LLVM.IRBuilder()
        nextInst = LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(actualOp))
        while isa(nextInst, LLVM.PHIInst)
            nextInst = LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(nextInst))
        end
        position!(B0, nextInst)
    end

    elSize = get_array_elsz(B0, actualOp)
    elSize = LLVM.zext!(B0, elSize, LLVM.IntType(8*sizeof(Csize_t)))

    len = get_array_len(B0, actualOp)

    length = LLVM.mul!(B0, len, elSize)
    isVolatile = LLVM.ConstantInt(LLVM.IntType(1), 0)

    # forward pass copy already done by underlying call
    allowForward = false
    intrinsic = LLVM.Intrinsic("llvm.memcpy").id

    if !fwd
        shadowdst = lookup_value(gutils, shadowdst, B)
    end
    shadowsrc = invert_pointer(gutils, origArg, B)
    if !fwd
        shadowsrc = lookup_value(gutils, shadowsrc, B)
    end

    width = get_width(gutils)

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
    i8 = LLVM.IntType(8)

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

    addrt = LLVM.PointerType(LLVM.IntType(8), 13)
    shadowsrc0 = load!(B, addrt, bitcast!(B, evsrc, LLVM.PointerType(addrt, LLVM.addrspace(LLVM.value_type(evsrc)))))
    shadowdst0 = load!(B, addrt, bitcast!(B, evdst, LLVM.PointerType(addrt, LLVM.addrspace(LLVM.value_type(evdst)))))

    API.sub_transfer(gutils, fwd ? API.DEM_ReverseModePrimal : API.DEM_ReverseModeGradient, secretty, intrinsic, #=dstAlign=#1, #=srcAlign=#1, #=offset=#0, false, shadowdst0, false, shadowsrc0, length, isVolatile, orig, allowForward, #=shadowsLookedUp=#!fwd)
    end

    end

    return nothing
end

function arraycopy_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    arraycopy_fwd(B, orig, gutils, normalR, shadowR)

    origops = LLVM.operands(orig)

    if !is_constant_value(gutils, origops[1]) && !is_constant_value(gutils, orig)
      shadowres = LLVM.Value(unsafe_load(shadowR))

      arraycopy_common(#=fwd=#true, B, orig, origops[1], gutils, shadowres)
    end

	return false
end

function arraycopy_rev(B, orig, gutils, tape)
    origops = LLVM.operands(orig)
    if !is_constant_value(gutils, origops[1]) && !is_constant_value(gutils, orig)
      arraycopy_common(#=fwd=#false, B, orig, origops[1], gutils, invert_pointer(gutils, orig, B))
    end

    return nothing
end

function arrayreshape_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    origops = LLVM.operands(orig)
    if is_constant_value(gutils, origops[2])
        emit_error(B, orig, "Enzyme: reshape array has active return, but inactive input")
    end

    width = get_width(gutils)

    shadowin = invert_pointer(gutils, origops[2], B)
    if width == 1
        args = LLVM.Value[
                          new_from_original(gutils, origops[1])
                          shadowin
                          new_from_original(gutils, origops[3])
                          ]
        shadowres = call_samefunc_with_inverted_bundles!(B, gutils, orig, args, [API.VT_Primal, API.VT_Shadow, API.VT_Primal], #=lookup=#false)
    else
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx in 1:width
            args = LLVM.Value[new_from_original(gutils, origops[1])
                              extract_value!(B, shadowin, idx-1)
                              new_from_original(gutils, origops[3])
                              ]
            tmp = call_samefunc_with_inverted_bundles!(B, gutils, orig, args, [API.VT_Primal, API.VT_Shadow, API.VT_Primal], #=lookup=#false)
            shadowres = insert_value!(B, shadowres, tmp, idx-1)
        end
    end
    unsafe_store!(shadowR, shadowres.ref)

	return false
end

function arrayreshape_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    arrayreshape_fwd(B, orig, gutils, normalR, shadowR)
end

function arrayreshape_rev(B, orig, gutils, tape)
    return nothing
end

function boxfloat_fwd(B, orig, gutils, normalR, shadowR)
    origops = collect(operands(orig))
    width = get_width(gutils)
    if is_constant_value(gutils, orig)
        return true
    end

    flt = value_type(origops[1])
    shadowsin = LLVM.Value[invert_pointer(gutils, origops[1], B)]
    if width == 1
        shadowres = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), shadowsin)
        callconv!(shadowres, callconv(orig))
    else
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx in 1:width
            args = LLVM.Value[
                              extract_value!(B, s, idx-1) for s in shadowsin
                              ]
            tmp = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), args)
            callconv!(tmp, callconv(orig))
            shadowres = insert_value!(B, shadowres, tmp, idx-1)
        end
    end
    unsafe_store!(shadowR, shadowres.ref)
    return false
end

function boxfloat_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    origops = collect(operands(orig))
    width = get_width(gutils)
    if is_constant_value(gutils, orig)
        return true
    end

    flt = value_type(origops[1])
    TT = tape_type(flt)

    if width == 1
        obj = emit_allocobj!(B, TT)
        o2 = bitcast!(B, obj, LLVM.PointerType(flt, addrspace(value_type(obj))))
        store!(B, ConstantFP(flt, 0.0), o2)
        shadowres = obj
    else
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, flt)))
        for idx in 1:width
            obj = emit_allocobj!(B, TT)
            o2 = bitcast!(B, obj, LLVM.PointerType(flt, addrspace(value_type(obj))))
            store!(B, ConstantFP(flt, 0.0), o2)
            shadowres = insert_value!(B, shadowres, obj, idx-1)
        end
    end
    unsafe_store!(shadowR, shadowres.ref)
    return false
end

function boxfloat_rev(B, orig, gutils, tape)
    origops = collect(operands(orig))
    width = get_width(gutils)
    if !is_constant_value(gutils, orig)
        ip = lookup_value(gutils, invert_pointer(gutils, orig, B), B)
        flt = value_type(origops[1])
        if width == 1
            ipc = bitcast!(B, ip, LLVM.PointerType(flt, addrspace(value_type(orig))))
            ld = load!(B, flt, ipc)
            store!(B, ConstantFP(flt, 0.0), ipc)
            if !is_constant_value(gutils, origops[1])
                API.EnzymeGradientUtilsAddToDiffe(gutils, origops[1], ld, B, flt)
            end
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, flt)))
            for idx in 1:width
                ipc = extract_value!(B, ip, idx-1)
                ipc = bitcast!(B, ipc, LLVM.PointerType(flt, addrspace(value_type(orig))))
                ld = load!(B, flt, ipc)
                store!(B, ConstantFP(flt, 0.0), ipc)
                shadowres = insert_value!(B, shadowres, ld, idx-1)
            end
            if !is_constant_value(gutils, origops[1])
                API.EnzymeGradientUtilsAddToDiffe(gutils, origops[1], shadowret, B, flt)
            end
        end
    end
    return nothing
end

function eqtableget_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig)
        return true
    end

    emit_error(B, orig, "Enzyme: Not yet implemented forward for jl_eqtable_get")

    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return false
end

function eqtableget_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig)
        return true
    end

    emit_error(B, orig, "Enzyme: Not yet implemented augmented forward for jl_eqtable_get")

    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return false
end

function eqtableget_rev(B, orig, gutils, tape)
    return nothing
end

function eqtableput_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    emit_error(B, orig, "Enzyme: Not yet implemented forward for jl_eqtable_put")

    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return false
end

function eqtableput_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    emit_error(B, orig, "Enzyme: Not yet implemented augmented forward for jl_eqtable_put")

    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return false
end

function eqtableput_rev(B, orig, gutils, tape)
    emit_error(B, orig, "Enzyme: Not yet implemented reverse for jl_eqtable_put")
    return nothing
end


function idtablerehash_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    emit_error(B, orig, "Enzyme: Not yet implemented forward for jl_idtable_rehash")

    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return false
end

function idtablerehash_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    emit_error(B, orig, "Enzyme: Not yet implemented augmented forward for jl_idtable_rehash")

    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return false
end

function idtablerehash_rev(B, orig, gutils, tape)
    emit_error(B, orig, "Enzyme: Not yet implemented reverse for jl_idtable_rehash")
    return nothing
end

function jl_array_grow_end_fwd(B, orig, gutils, normalR, shadowR)
    origops = collect(operands(orig))
    if is_constant_value(gutils, origops[1])
        return true
    end

    width = get_width(gutils)

    shadowin = invert_pointer(gutils, origops[1], B)
    if width == 1
        args = LLVM.Value[
                          shadowin
                          new_from_original(gutils, origops[2])
                          ]
        call_samefunc_with_inverted_bundles!(B, gutils, orig, args, [API.VT_Shadow, API.VT_Primal], #=lookup=#false)
    else
        for idx in 1:width
            args = LLVM.Value[
                              extract_value!(B, shadowin, idx-1)
                              new_from_original(gutils, origops[2])
                              ]
            call_samefunc_with_inverted_bundles!(B, gutils, orig, args, [API.VT_Shadow, API.VT_Primal], #=lookup=#false)
        end
    end
    return false
end


function jl_array_grow_end_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    origops = collect(operands(orig))
    if is_constant_value(gutils, origops[1])
        return true
    end

    width = get_width(gutils)

    shadowin = invert_pointer(gutils, origops[1], B)
    ctx = LLVM.context(orig)
    i8 = LLVM.IntType(8)

    inc = new_from_original(gutils, origops[2])

    al = 0

    if width == 1
        anti = shadowin

        idx = get_array_nrows(B, anti)
        elsz = zext!(B, get_array_elsz(B, anti), value_type(idx))
        off = mul!(B, idx, elsz)
        tot = mul!(B, inc, elsz)

        args = LLVM.Value[anti, inc]
        call_samefunc_with_inverted_bundles!(B, gutils, orig, args, [API.VT_Shadow, API.VT_Primal], #=lookup=#false)

        toset = get_array_data(B, anti)
        toset = gep!(B, i8, toset, LLVM.Value[off])
        mcall = LLVM.memset!(B, toset, LLVM.ConstantInt(i8, 0, false), tot, al)
    else
        for idx in 1:width
            anti = extract_value!(B, shadowin, idx-1)

            idx = get_array_nrows(B, anti)
            elsz = zext!(B, get_array_elsz(B, anti), value_type(idx))
            off = mul!(B, idx, elsz)
            tot = mul!(B, inc, elsz)

            args = LLVM.Value[anti, inc]
            call_samefunc_with_inverted_bundles!(B, gutils, orig, args, [API.VT_Shadow, API.VT_Primal], #=lookup=#false)

            toset = get_array_data(B, anti)
            toset = gep!(B, i8, toset, LLVM.Value[off])
            mcall = LLVM.memset!(B, toset, LLVM.ConstantInt(i8, 0, false), tot, al)
        end
    end

    return false
end

function jl_array_grow_end_rev(B, orig, gutils, tape)
    origops = collect(operands(orig))
    if !is_constant_value(gutils, origops[1])

        width = get_width(gutils)

        called_value = origops[end]
        funcT = called_type(orig)
        mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
        delF, fty = get_function!(mod, "jl_array_del_end", funcT)

        shadowin = invert_pointer(gutils, origops[1], B)
        shadowin = lookup_value(gutils, shadowin, B)

        offset = new_from_original(gutils, origops[2])
        offset = lookup_value(gutils, offset, B)

        if width == 1
            args = LLVM.Value[
                              shadowin
                              offset
                              ]
            LLVM.call!(B, fty, delF, args)
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
            for idx in 1:width
                args = LLVM.Value[
                                  extract_value!(B, shadowin, idx-1)
                                  offset
                                  ]
                LLVM.call!(B, fty, delF, args)
            end
        end
    end
    return nothing
end

function jl_array_del_end_fwd(B, orig, gutils, normalR, shadowR)
    jl_array_grow_end_fwd(B, orig, gutils, normalR, shadowR)
end

function jl_array_del_end_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    jl_array_del_end_fwd(B, orig, gutils, normalR, shadowR)
end

function jl_array_del_end_rev(B, orig, gutils, tape)
    origops = collect(operands(orig))
    if !is_constant_value(gutils, origops[1])
        width = get_width(gutils)

        called_value = origops[end]
        funcT = called_type(orig)
        mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
        delF, fty = get_function!(mod, "jl_array_grow_end", funcT)

        shadowin = invert_pointer(gutils, origops[1], B)
        shadowin = lookup_value(gutils, shadowin, B)

        offset = new_from_original(gutils, origops[2])
        offset = lookup_value(gutils, offset, B)

        if width == 1
            args = LLVM.Value[
                              shadowin
                              offset
                              ]
            LLVM.call!(B, fty, delF, args)
        else
            for idx in 1:width
                args = LLVM.Value[
                                  extract_value!(B, shadowin, idx-1)
                                  offset
                                  ]
                LLVM.call!(B, fty, delF, args)
            end
        end

        # GPUCompiler.@safe_warn "Not applying memsetUnknown concrete type" tt=string(tt)
        emit_error(B, orig, "Not applying memset on reverse of jl_array_del_end")
        # memset(data + idx * elsz, 0, inc * elsz);
    end
    return nothing
end

function jl_array_ptr_copy_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_inst(gutils, orig)
        return true
    end
    origops = collect(operands(orig))
    width = get_width(gutils)
    origops = collect(operands(orig))
    width = get_width(gutils)

    args = LLVM.Value[]
    for a in origops[1:end-2]
        v = invert_pointer(gutils, a, B)
        push!(args, v)
    end
    push!(args, new_from_original(gutils, origops[end-1]))
    valTys = API.CValueType[API.VT_Shadow, API.VT_Shadow, API.VT_Shadow, API.VT_Shadow, API.VT_Primal]

    if width == 1
        vargs = args
        cal = call_samefunc_with_inverted_bundles!(B, gutils, orig, vargs, valTys, #=lookup=#false)
        debug_from_orig!(gutils, cal, orig)
        callconv!(cal, callconv(orig))
    else
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx in 1:width
            vargs = LLVM.Value[]
            for a in args[1:end-1]
                push!(vargs, extract_value!(B, a, idx-1))
            end
            push!(vargs, args[end])
            cal = call_samefunc_with_inverted_bundles!(b, gutils, orig, vargs, valTys, #=lookup=#false)
            debug_from_orig!(gutils, cal, orig)
            callconv!(cal, callconv(orig))
        end
    end

    return false
end
function jl_array_ptr_copy_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
  jl_array_ptr_copy_fwd(B, orig, gutils, normalR, shadowR)
end
function jl_array_ptr_copy_rev(B, orig, gutils, tape)
    return nothing
end

function jl_array_sizehint_fwd(B, orig, gutils, normalR, shadowR)
    origops = collect(operands(orig))
    if is_constant_value(gutils, origops[1])
        return true
    end
    width = get_width(gutils)

    shadowin = invert_pointer(gutils, origops[1], B)
    if width == 1
        args = LLVM.Value[
                          shadowin
                          new_from_original(gutils, origops[2])
                          ]
        call_samefunc_with_inverted_bundles!(B, gutils, orig, args, [API.VT_Shadow, API.VT_Primal], #=lookup=#false)
    else
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx in 1:width
            args = LLVM.Value[
                              extract_value!(B, shadowin, idx-1)
                              new_from_original(gutils, origops[2])
                              ]
            call_samefunc_with_inverted_bundles!(B, gutils, orig, args, [API.VT_Shadow, API.VT_Primal], #=lookup=#false)
        end
    end
    return false
end

function jl_array_sizehint_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    jl_array_sizehint_fwd(B, orig, gutils, normalR, shadowR)
end

function jl_array_sizehint_rev(B, orig, gutils, tape)
    return nothing
end

function jl_unhandled_fwd(B, orig, gutils, normalR, shadowR)
    newo = new_from_original(gutils, orig)
    origops = collect(operands(orig))
    err = emit_error(LLVM.IRBuilder(B), orig, "Enzyme: unhandled forward for "*string(origops[end]))
    API.moveBefore(newo, err, C_NULL)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing

    if shadowR != C_NULL && normal !== nothing
        width = get_width(gutils)
        if width == 1
            shadowres = normal
        else
            position!(B, LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(normal)))
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(normal))))
            for idx in 1:width
                shadowres = insert_value!(B, shadowres, normal, idx-1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return false
end
function jl_unhandled_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
  jl_unhandled_fwd(B, orig, gutils, normalR, shadowR)
end
function jl_unhandled_rev(B, orig, gutils, tape)
    return nothing
end

function get_binding_or_error_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig)
        return true
    end
    err = emit_error(B, orig, "Enzyme: unhandled forward for jl_get_binding_or_error")
    newo = new_from_original(gutils, orig)
    API.moveBefore(newo, err, B)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing

    if shadowR != C_NULL && normal !== nothing
        width = get_width(gutils)
        if width == 1
            shadowres = normal
        else
            position!(B, LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(normal)))
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(normal))))
            for idx in 1:width
                shadowres = insert_value!(B, shadowres, normal, idx-1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return false
end

function get_binding_or_error_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig)
        return true
    end
    err = emit_error(B, orig, "Enzyme: unhandled augmented forward for jl_get_binding_or_error")
    newo = new_from_original(gutils, orig)
    API.moveBefore(newo, err, B)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        width = get_width(gutils)
        if width == 1
            shadowres = normal
        else
            position!(B, LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(normal)))
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(normal))))
            for idx in 1:width
                shadowres = insert_value!(B, shadowres, normal, idx-1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return false
end

function get_binding_or_error_rev(B, orig, gutils, tape)
    emit_error(B, orig, "Enzyme: unhandled reverse for jl_get_binding_or_error")
    return nothing
end

function finalizer_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    err = emit_error(B, orig, "Enzyme: unhandled augmented forward for jl_gc_add_finalizer_th or jl_gc_add_ptr_finalizer")
    newo = new_from_original(gutils, orig)
    API.moveBefore(newo, err, B)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return false
end

function finalizer_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    # err = emit_error(B, orig, "Enzyme: unhandled augmented forward for jl_gc_add_finalizer_th")
    # newo = new_from_original(gutils, orig)
    # API.moveBefore(newo, err, B)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    # Delete the primal code
    if normal !== nothing
        unsafe_store!(normalR, C_NULL)
    else
        ni = new_from_original(gutils, orig)
        API.EnzymeGradientUtilsErase(gutils, ni)
    end
    return false
end

function finalizer_rev(B, orig, gutils, tape)
    # emit_error(B, orig, "Enzyme: unhandled reverse for jl_gc_add_finalizer_th")
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

const CheckNan = Ref(false) 
function julia_sanitize(orig::LLVM.API.LLVMValueRef, val::LLVM.API.LLVMValueRef, B::LLVM.API.LLVMBuilderRef, mask::LLVM.API.LLVMValueRef)::LLVM.API.LLVMValueRef
  orig = LLVM.Value(orig)
  val = LLVM.Value(val)
  B = LLVM.IRBuilder(B)
  if CheckNan[]
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    ty = LLVM.value_type(val)
    vt = LLVM.VoidType()
    FT = LLVM.FunctionType(vt, [ty, LLVM.PointerType(LLVM.Int8Type())])

    stringv = "Enzyme: Found nan while computing derivative of "*string(orig)
    if orig !== nothing && isa(orig, LLVM.Instruction)
        bt = GPUCompiler.backtrace(orig)
        function printBT(io)
            print(io,"\nCaused by:")
            Base.show_backtrace(io, bt)
        end
        stringv*=sprint(io->Base.show_backtrace(io, bt))
    end

    fn, _ = get_function!(mod, "julia.sanitize."*string(ty), FT)
    if isempty(blocks(fn))
        let builder = IRBuilder()
            entry = BasicBlock(fn, "entry")
            good = BasicBlock(fn, "good")
            bad = BasicBlock(fn, "bad")
            position!(builder, entry)
            inp, sval = collect(parameters(fn))
            cmp = fcmp!(builder, LLVM.API.LLVMRealUNO, inp, inp)

            br!(builder, cmp, bad, good)

            position!(builder, good)
            ret!(builder)
            # ret!(builder, inp)
            
            position!(builder, bad)
    
            funcT = LLVM.FunctionType(LLVM.VoidType(), LLVMType[LLVM.PointerType(LLVM.Int8Type())])
            ptr = @cfunction(throwerr, Union{}, (Cstring,))
            ptr = convert(UInt, ptr)
            ptr = LLVM.ConstantInt(ptr)
            func = inttoptr!(builder, ptr, LLVM.PointerType(funcT))
            call!(builder, funcT, func, LLVM.Value[sval])
            unreachable!(builder)

            dispose(builder)
        end
    end
    # val = 
    call!(B, fn, LLVM.Value[val, globalstring_ptr!(B, stringv)])
  end
  return val.ref
end

function julia_error(cstr::Cstring, val::LLVM.API.LLVMValueRef, errtype::API.ErrorType, data::Ptr{Cvoid}, data2::LLVM.API.LLVMValueRef, B::LLVM.API.LLVMBuilderRef)::LLVM.API.LLVMValueRef
    msg = Base.unsafe_string(cstr)
    bt = nothing
    ir = nothing
    if val != C_NULL
        val = LLVM.Value(val)
        if isa(val, LLVM.Instruction)
            dbgval = val
            while !haskey(metadata(dbgval), LLVM.MD_dbg)
                dbgval = LLVM.API.LLVMGetNextInstruction(dbgval)
                if dbgval == C_NULL
                    dbgval = nothing
                    break
                else
                    dbgval = LLVM.Instruction(dbgval)
                end
            end
            if dbgval !== nothing
                bt = GPUCompiler.backtrace(dbgval)
            end
        end
        if isa(val, LLVM.ConstantExpr)
            for u in LLVM.uses(val)
                u = LLVM.user(u)
                if isa(u, LLVM.Instruction)
                    bt = GPUCompiler.backtrace(val)
                end
            end
        else
            # Need to convert function to string, since when the error is going to be printed
            # the module might have been destroyed
            ir = string(parent_scope(val))
        end
    end

    if errtype == API.ET_NoDerivative
        exc = NoDerivativeException(msg, ir, bt)
        if B != C_NULL
            B = IRBuilder(B)
            msg2 = sprint() do io
                Base.showerror(io, exc)
            end
            emit_error(B, nothing, msg2)
            return C_NULL
        end
        throw(exc)
    elseif errtype == API.ET_NoShadow
        data = GradientUtils(API.EnzymeGradientUtilsRef(data))
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
        @assert B != C_NULL
        B = IRBuilder(B)
        
        data = API.EnzymeTypeAnalyzerRef(data)
        ip = API.EnzymeTypeAnalyzerToString(data)
        sval = Base.unsafe_string(ip)
        API.EnzymeStringFree(ip)

        msg2 = sprint() do io::IO
            print(io, "Enzyme cannot deduce type\n")
            if ir !== nothing
                print(io, "Current scope: \n")
                print(io, ir)
            end
            print(io, "\n Type analysis state: \n")
            write(io, sval)
            print(io, '\n', msg, '\n')
            if bt !== nothing
                print(io,"\nCaused by:")
                Base.show_backtrace(io, bt)
                println(io)
            end
        end
        emit_error(B, nothing, msg2)
        return C_NULL
    elseif errtype == API.ET_IllegalFirstPointer
        throw(IllegalFirstPointerException(msg, ir, bt))
    elseif errtype == API.ET_InternalError
        throw(EnzymeInternalError(msg, ir, bt))
    elseif errtype == API.ET_TypeDepthExceeded
        msg2 = sprint() do io
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
        GPUCompiler.@safe_warn msg2
        return C_NULL
    elseif errtype == API.ET_IllegalReplaceFicticiousPHIs
        data2 = LLVM.Value(data2)
        msg2 = sprint() do io
            print(io, msg)
            println(io)
            println(io, LLVM.parent(LLVM.parent(data2)))
            println(io, val)
            println(io, data2)
        end
        throw(EnzymeInternalError(msg2, ir, bt))
    elseif errtype == API.ET_MixedActivityError
        data2 = LLVM.Value(data2)
        badval = nothing
        # Ignore mismatched activity if phi/store of ghost
        todo = LLVM.Value[data2]
        seen = Set{LLVM.Value}()
        illegal = false
        while length(todo) != 0
            cur = pop!(todo)
            if cur in seen
                continue
            end
            push!(seen, cur)
            if isa(cur, LLVM.PHIInst)
                for v in LLVM.incoming(cur)
                    push!(todo, cur)
                end
                continue
            end

            if isa(cur, ConstantExpr)
                ce = cur
                while isa(ce, ConstantExpr)
                    if opcode(ce) == LLVM.API.LLVMAddrSpaceCast ||  opcode(ce) == LLVM.API.LLVMIntToPtr
                        ce = operands(ce)[1]
                    else
                        break
                    end
                end
                if isa(ce, ConstantInt)
                    ptr = reinterpret(Ptr{Cvoid}, convert(UInt, ce))
                    typ = Base.unsafe_pointer_to_objref(ptr)
                    if isghostty(Core.Typeof(typ))
                        continue
                    end
                    badval = typ
                    illegal = false
                    break
                end
            end
            if isa(cur, LLVM.PointerNull)
                continue
            end
            if isa(cur, LLVM.UndefValue)
                continue
            end
            if isa(cur, LLVM.ConstantAggregateZero)
                continue
            end
            if isa(cur, LLVM.ConstantAggregate)
                continue
            end
            if isa(cur, LLVM.ConstantDataSequential)
                for v in collect(cur)
                    push!(todo, v)
                end
                continue
            end
            if isa(cur, LLVM.ConstantInt)
                if width(value_type(cur)) <= 8
                    continue
                end
            end
            illegal = true
            break
        end

        if !illegal
            return C_NULL
        end

        if LLVM.API.LLVMIsAReturnInst(val) != C_NULL
            mi, rt = enzyme_custom_extract_mi(LLVM.parent(LLVM.parent(val))::LLVM.Function, #=error=#false)
            if mi !== nothing && isghostty(rt)
                return C_NULL
            end
        end

        gutils = GradientUtils(API.EnzymeGradientUtilsRef(data))
        newb = new_from_original(gutils, val)
        while isa(newb, LLVM.PHIInst)
            newb = LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(newb))
        end
        b = IRBuilder(B)
        msg2 = sprint() do io
            print(io, msg)
            println(io)
            ttval = val
            if isa(ttval, LLVM.StoreInst)
                ttval = operands(ttval)[1]
            end
	        tt = TypeTree(API.EnzymeGradientUtilsAllocAndGetTypeTree(gutils, ttval))
            st = API.EnzymeTypeTreeToString(tt)
            print(io, "Type tree: ")
            println(io, Base.unsafe_string(st))
            API.EnzymeStringFree(st)
            if badval !== nothing
                println(io, " value="*string(badval))
            end
            println(io, "You may be using a constant variable as temporary storage for active memory (https://enzyme.mit.edu/julia/stable/#Activity-of-temporary-storage). If not, please open an issue, and either rewrite this variable to not be conditionally active or use Enzyme.API.runtimeActivity!(true) as a workaround for now")
            if bt !== nothing
                Base.show_backtrace(io, bt)
            end
        end
        emit_error(b, nothing, msg2)
        return C_NULL
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
@inline any_jltypes(::Type{Tuple{}}) = false
@inline any_jltypes(::Type{NTuple{Size, T}}) where {Size, T} = any_jltypes(T)
@inline any_jltypes(::Type{Core.LLVMPtr{T, Addr}}) where {T, Addr} = 10 <= Addr <= 12
@inline any_jltypes(::Type{Any}) = true
@inline any_jltypes(::Type{NamedTuple{A,B}}) where {A,B} = any(any_jltypes(b) for b in B.parameters)
@inline any_jltypes(::Type{T}) where {T<:Tuple} = any(any_jltypes(b) for b in T.parameters)

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

# return result and if contains any
function to_tape_type(Type::LLVM.API.LLVMTypeRef)::Tuple{DataType,Bool}
    tkind = LLVM.API.LLVMGetTypeKind(Type)
    if tkind == LLVM.API.LLVMStructTypeKind
        tys = DataType[]
        nelems = LLVM.API.LLVMCountStructElementTypes(Type)
        containsAny = false
        syms = Symbol[]
        for i in 1:nelems
            e = LLVM.API.LLVMStructGetTypeAtIndex(Type, i-1)
            T, sub = to_tape_type(e)
            containsAny |= sub
            push!(tys, T)
            push!(syms, Symbol(i))
        end
        Tup = Tuple{tys...}
        if containsAny
            res = (syms...,)
            return NamedTuple{res, Tup}, false
        else
            return Tup, false
        end
    end
    if tkind == LLVM.API.LLVMPointerTypeKind
        addrspace = LLVM.API.LLVMGetPointerAddressSpace(Type)
        if 10 <= addrspace <= 12
            return Any, true
        else
            e = LLVM.API.LLVMGetElementType(Type)
            return Core.LLVMPtr{to_tape_type(e)[1], Int(addrspace)}, false
        end
    end
    if tkind == LLVM.API.LLVMArrayTypeKind
        e = LLVM.API.LLVMGetElementType(Type)
        T, sub = to_tape_type(e)
        len = Int(LLVM.API.LLVMGetArrayLength(Type))
        Tup = NTuple{len, T}
        if sub
            return NamedTuple{ntuple(Core.Symbol, Val(len)), Tup}, false
        else
            return Tup, false
        end
    end
    if tkind == LLVM.API.LLVMVectorTypeKind
        e = LLVM.API.LLVMGetElementType(Type)
        T, sub = to_tape_type(e)
        len = Int(LLVM.API.LLVMGetVectorSize(Type))
        Tup = NTuple{len, T}
        if sub
            return NamedTuple{ntuple(Core.Symbol, Val(len)), Tup}, false
        else
            return Tup, false
        end
    end
    if tkind == LLVM.API.LLVMIntegerTypeKind
        N = LLVM.API.LLVMGetIntTypeWidth(Type)
        if N == 1
            return Bool,  false
        elseif N == 8
            return UInt8, false
        elseif N == 16
            return UInt16, false
        elseif N == 32
            return UInt32, false
        elseif N == 64
            return UInt64, false
        elseif N == 128
            return UInt128, false
        else
            error("Can't construct tape type for integer of width $N")
        end
    end
    if tkind == LLVM.API.LLVMHalfTypeKind
        return Float16, false
    end
    if tkind == LLVM.API.LLVMFloatTypeKind
        return Float32, false
    end
    if tkind == LLVM.API.LLVMDoubleTypeKind
        return Float64, false
    end
    if tkind == LLVM.API.LLVMFP128TypeKind
        return Float128, false
    end
    error("Can't construct tape type for $Type")
end

function tape_type(LLVMType::LLVM.LLVMType)
    TT, isAny = to_tape_type(LLVMType.ref)
    if isAny
        return AnonymousStruct(Tuple{Any})
    end
    return TT
end

from_tape_type(::Type{T}) where T<:AbstractFloat = convert(LLVMType, T)
from_tape_type(::Type{T}) where T<:Integer = convert(LLVMType, T)
from_tape_type(::Type{NTuple{Size, T}}) where {Size, T} = LLVM.ArrayType(from_tape_type(T), Size)
from_tape_type(::Type{Core.LLVMPtr{T, Addr}}) where {T, Addr} = LLVM.PointerType(from_tape_type(UInt8), Addr)
# from_tape_type(::Type{Core.LLVMPtr{T, Addr}}, ctx) where {T, Addr} = LLVM.PointerType(from_tape_type(T, ctx), Addr)
from_tape_type(::Type{Any}) = LLVM.PointerType(LLVM.StructType(LLVM.LLVMType[]), Tracked)
function from_tape_type(::Type{NamedTuple{A,B}}) where {A,B}
    from_tape_type(B)
end
function from_tape_type(::Type{B}) where {B<:Tuple}
    ar = LLVM.LLVMType[from_tape_type(b) for b in B.parameters]
    if length(B.parameters) >= 1 && all(ar[1] == b for b in ar)
        return LLVM.ArrayType(ar[1], length(B.parameters))
    else
        return LLVM.StructType(LLVM.LLVMType[from_tape_type(b) for b in B.parameters])
    end
end

# See get_current_task_from_pgcstack (used from 1.7+)
if VERSION >= v"1.9.1"
    current_task_offset() = -(unsafe_load(cglobal(:jl_task_gcstack_offset, Cint)) ÷ sizeof(Ptr{Cvoid}))
elseif VERSION >= v"1.9.0"
    if Sys.WORD_SIZE == 64
        current_task_offset() = -13
    else
        current_task_offset() = -18
    end
else
    if Sys.WORD_SIZE == 64
        current_task_offset() = -12 #1.8/1.7
    else
        current_task_offset() = -17 #1.8/1.7
    end
end

# See get_current_ptls_from_task (used from 1.7+)
if VERSION >= v"1.9.1"
    current_ptls_offset() = unsafe_load(cglobal(:jl_task_ptls_offset, Cint)) ÷ sizeof(Ptr{Cvoid})
elseif VERSION >= v"1.9.0"
    if Sys.WORD_SIZE == 64
        current_ptls_offset() = 15
    else
        current_ptls_offset() = 20
    end
else
    if Sys.WORD_SIZE == 64
        current_ptls_offset() = 14 # 1.8/1.7
    else
        current_ptls_offset() = 19
    end
end

function get_julia_inner_types(B, p, startvals...; added=[])
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    vals = LLVM.Value[p]
    todo = LLVM.Value[startvals...]
    while length(todo) != 0
        cur = popfirst!(todo)
        ty = value_type(cur)
        if isa(ty, LLVM.PointerType)
            if any_jltypes(ty)
                if addrspace(ty) != Tracked
                    cur = addrspacecast!(B, cur, LLVM.PointerType(eltype(ty), Tracked))
                    if isa(cur, LLVM.Instruction)
                        push!(added, cur.ref)
                    end
                end
                if value_type(cur) != T_prjlvalue
                    cur = bitcast!(B, cur, T_prjlvalue)
                    if isa(cur, LLVM.Instruction)
                        push!(added, cur.ref)
                    end
                end
                push!(vals, cur)
            end
            continue
        end
        if isa(ty, LLVM.ArrayType)
            if any_jltypes(ty)
                for i=1:length(ty)
                    ev = extract_value!(B, cur, i-1)
                    if isa(ev, LLVM.Instruction)
                        push!(added, ev.ref)
                    end
                    push!(todo, ev)
                end
            end
            continue
        end
        if isa(ty, LLVM.StructType)
            for (i, t) in enumerate(LLVM.elements(ty))
                if any_jltypes(t)
                    ev = extract_value!(B, cur, i-1)
                    if isa(ev, LLVM.Instruction)
                        push!(added, ev.ref)
                    end
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
    B = LLVM.IRBuilder(B)
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
    if any_jltypes(value_type(v)) && !isa(p, LLVM.AllocaInst)
        ctx = LLVM.context(v)
        T_jlvalue = LLVM.StructType(LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
        pn = bitcast!(B, p, T_prjlvalue)
        if isa(pn, LLVM.Instruction) && p != pn
            push!(added, pn.ref)
        end
        p = pn

        vals = get_julia_inner_types(B, p, v, added=added)
        r = emit_writebarrier!(B, vals)
        @assert isa(r, LLVM.Instruction)
        push!(added, r.ref)
    end
    if R2 != C_NULL
        unsafe_store!(R2, length(added))
        ptr = Base.unsafe_convert(Ptr{LLVM.API.LLVMValueRef}, Libc.malloc(sizeof(LLVM.API.LLVMValueRef)*length(added)))
        for (i, v) in enumerate(added)
            @assert isa(LLVM.Value(v), LLVM.Instruction)
            unsafe_store!(ptr, v, i)
        end
        return ptr
    end
    return C_NULL
end

function julia_default_tape_type(C::LLVM.API.LLVMContextRef)
    ctx = LLVM.Context(C)
    T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    return T_prjlvalue.ref
end
function julia_undef_value_for_type(Ty::LLVM.API.LLVMTypeRef, forceZero::UInt8)::LLVM.API.LLVMValueRef
    ty = LLVM.LLVMType(Ty)
    if !any_jltypes(ty)
        if forceZero != 0
            return LLVM.null(ty).ref
        else
            return UndefValue(ty).ref
        end
    end
    if isa(ty, LLVM.PointerType)
        val = unsafe_to_llvm(nothing)
        if !is_opaque(ty)
            val = const_pointercast(val, LLVM.PointerType(eltype(ty), Tracked))
        end
        if addrspace(ty) != Tracked
            val = const_addrspacecast(val, ty)
        end
        return val.ref
    end
    if isa(ty, LLVM.ArrayType)
        st = LLVM.Constant(julia_undef_value_for_type(eltype(ty).ref, forceZero))
        return ConstantArray(ty, [st for i in 1:length(st)]).ref
    end
    if isa(ty, LLVM.StructType)
        vals = LLVM.Constant[]
        for st in LLVM.elements(ty)
            push!(vals, LLVM.Value(julia_undef_value_for_type(st.ref, forceZero)))
        end
        return ConstantStruct(ty, vals).ref
    end
    @safe_show "Unknown type to val", Ty
    @assert false
end

function julia_allocator(B::LLVM.API.LLVMBuilderRef, LLVMType::LLVM.API.LLVMTypeRef, Count::LLVM.API.LLVMValueRef, AlignedSize::LLVM.API.LLVMValueRef, IsDefault::UInt8, ZI)
    B = LLVM.IRBuilder(B)
    Count = LLVM.Value(Count)
    AlignedSize = LLVM.Value(AlignedSize)
    LLVMType = LLVM.LLVMType(LLVMType)
    return julia_allocator(B, LLVMType, Count, AlignedSize, IsDefault, ZI)
end

function fixup_return(B, retval)
    B = LLVM.IRBuilder(B)

    func = LLVM.parent(position(B))
    mod = LLVM.parent(func)
    T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_prjlvalue_UT = LLVM.PointerType(T_jlvalue)

    retval = LLVM.Value(retval)
    ty = value_type(retval)
    # Special case the union return { {} addr(10)*, i8 }
    #   which can be [ null, 1 ], to not have null in the ptr
    #   field, but nothing
    if isa(ty, LLVM.StructType)
        elems = LLVM.elements(ty)
        if length(elems) == 2 && elems[1] == T_prjlvalue
            fill_val = unsafe_to_llvm(nothing)
            prev = extract_value!(B, retval, 0)
            eq = icmp!(B, LLVM.API.LLVMIntEQ, prev, LLVM.null(T_prjlvalue))
            retval = select!(B, eq, insert_value!(B, retval, fill_val, 0), retval)
        end
    end
    return retval.ref
end

function zero_allocation(B, LLVMType, obj, isTape::UInt8)
    B = LLVM.IRBuilder(B)
    LLVMType = LLVM.LLVMType(LLVMType)
    obj = LLVM.Value(obj)
    jlType = tape_type(LLVMType)
    zeroAll = isTape == 0
    func = LLVM.parent(position(B))
    mod = LLVM.parent(func)
    T_int64 = LLVM.Int64Type()
    zero_single_allocation(B, jlType, LLVMType, obj, zeroAll, LLVM.ConstantInt(T_int64, 0))
    return nothing
end

function zero_single_allocation(builder, jlType, LLVMType, nobj, zeroAll, idx)
    T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_prjlvalue_UT = LLVM.PointerType(T_jlvalue)

    todo = Tuple{Vector{LLVM.Value},LLVM.LLVMType,DataType}[(LLVM.Value[idx], LLVMType, jlType)]

    while length(todo) != 0
        path, ty, jlty = popfirst!(todo)
        if isa(ty, LLVM.PointerType)
            if any_jltypes(ty)
                loc = gep!(builder, LLVMType, nobj, path)
                fill_val = unsafe_to_llvm(nothing)
                loc = bitcast!(builder, loc, LLVM.PointerType(T_prjlvalue, addrspace(value_type(loc))))
                store!(builder, fill_val, loc)
            elseif zeroAll
                loc = gep!(builder, LLVMType, nobj, path)
                store!(builder, LLVM.null(ty), loc)
            end
            continue
        end
        if isa(ty, LLVM.FloatingPointType) || isa(ty, LLVM.IntegerType)
            if zeroAll
                loc = gep!(builder, LLVMType, nobj, path)
                store!(builder, LLVM.null(ty), loc)
            end
            continue
        end
        if isa(ty, LLVM.ArrayType) 
            for i=1:length(ty)
                npath = copy(path)
                push!(npath, LLVM.ConstantInt(LLVM.IntType(32), i-1))
                push!(todo, (npath, eltype(ty), eltype(jlty)))
            end
            continue
        end
        if isa(ty, LLVM.VectorType) 
            for i=1:size(ty)
                npath = copy(path)
                push!(npath, LLVM.ConstantInt(LLVM.IntType(32), i-1))
                push!(todo, (npath, eltype(ty), eltype(jlty)))
            end
            continue
        end
        if isa(ty, LLVM.StructType)
            i = 1
            for ii in 1:fieldcount(jlty)
                jlet = fieldtype(jlty, ii)
                if isghostty(jlet) || Core.Compiler.isconstType(jlet)
                    continue
                end
                t = LLVM.elements(ty)[i]
                npath = copy(path)
                push!(npath, LLVM.ConstantInt(LLVM.IntType(32), i-1))
                push!(todo, (npath, t, jlet))
                i+=1
            end
            @assert i == Int(length(LLVM.elements(ty)))+1
            continue
        end
    end
    return nothing

end


function zero_allocation(B::LLVM.IRBuilder, jlType, LLVMType, obj, AlignedSize, Size, zeroAll::Bool)::LLVM.API.LLVMValueRef
    func = LLVM.parent(position(B))
    mod = LLVM.parent(func)
    T_int8 = LLVM.Int8Type()

    T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_prjlvalue_UT = LLVM.PointerType(T_jlvalue)

    wrapper_f = LLVM.Function(mod, "zeroType", LLVM.FunctionType(LLVM.VoidType(), [value_type(obj), T_int8, value_type(Size)]))
    push!(function_attributes(wrapper_f), StringAttribute("enzyme_math", "enzyme_zerotype"))
    push!(function_attributes(wrapper_f), StringAttribute("enzyme_inactive"))
    push!(function_attributes(wrapper_f), EnumAttribute("alwaysinline", 0))
    push!(function_attributes(wrapper_f), EnumAttribute("nofree", 0))
    push!(function_attributes(wrapper_f), EnumAttribute("argmemonly", 0))
    push!(function_attributes(wrapper_f), EnumAttribute("writeonly", 0))
    push!(function_attributes(wrapper_f), EnumAttribute("willreturn", 0))
    push!(function_attributes(wrapper_f), EnumAttribute("mustprogress", 0))
    push!(parameter_attributes(wrapper_f, 1), EnumAttribute("writeonly", 0))
    push!(parameter_attributes(wrapper_f, 1), EnumAttribute("nocapture", 0))
    linkage!(wrapper_f, LLVM.API.LLVMInternalLinkage)
    let builder = IRBuilder()
        entry = BasicBlock(wrapper_f, "entry")
        loop = BasicBlock(wrapper_f, "loop")
        exit = BasicBlock(wrapper_f, "exit")
        position!(builder, entry)
        nobj, _, nsize = collect(parameters(wrapper_f))
        nobj = pointercast!(builder, nobj, LLVM.PointerType(LLVMType, addrspace(value_type(nobj))))

        LLVM.br!(builder, loop)
        position!(builder, loop)
        idx = LLVM.phi!(builder, value_type(Size))
        inc = add!(builder, idx, LLVM.ConstantInt(value_type(Size), 1))
        append!(LLVM.incoming(idx), [(LLVM.ConstantInt(value_type(Size), 0), entry), (inc, loop)])

        zero_single_allocation(builder, jlType, LLVMType, nobj, zeroAll, idx)

        br!(builder, icmp!(builder, LLVM.API.LLVMIntEQ, inc, LLVM.Value(LLVM.API.LLVMBuildExactUDiv(builder, nsize, AlignedSize, ""))), exit, loop)
        position!(builder, exit)

        ret!(builder)

        dispose(builder)
    end
    return call!(B, LLVM.function_type(wrapper_f), wrapper_f, [obj, LLVM.ConstantInt(T_int8, 0), Size]).ref
end

function julia_allocator(B, LLVMType, Count, AlignedSize, IsDefault, ZI)
    func = LLVM.parent(position(B))
    mod = LLVM.parent(func)

    Size = nuwmul!(B, Count, AlignedSize) # should be nsw, nuw
    T_int8 = LLVM.Int8Type()

    if any_jltypes(LLVMType) || IsDefault != 0
        T_int64 = LLVM.Int64Type()
        T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
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
                GPUCompiler.@safe_error "Size of Enzyme tape is incorrect. Please report this issue" ETT sizeof(ETT) TargetSize = N*convert(Int, AlignedSize) LLVMType
                emit_error(B, nothing, "Enzyme: Tape allocation failed.") # TODO: Pick appropriate orig

                return LLVM.API.LLVMValueRef(LLVM.UndefValue(LLVMType).ref)
            end

            # Obtain tag
            tag = unsafe_to_llvm(ETT)
        else
            if sizeof(Int) == sizeof(Int64)
                boxed_count = emit_box_int64!(B, Count)
            else
                T_size_t = convert(LLVM.LLVMType, Int)
                Count = trunc!(B, Count, T_size_t)
                boxed_count = emit_box_int32!(B, Count)
            end
            tag = emit_apply_type!(B, NTuple, (boxed_count, unsafe_to_llvm(TT)))
        end

        # Check if Julia version has https://github.com/JuliaLang/julia/pull/46914
        # and also https://github.com/JuliaLang/julia/pull/47076
        # and also https://github.com/JuliaLang/julia/pull/48620
        @static if VERSION >= v"1.10.0-DEV.569"
            needs_dynamic_size_workaround = false
        else
            needs_dynamic_size_workaround = !isa(Size, LLVM.ConstantInt) || convert(Int, Size) != 1
        end

        obj = emit_allocobj!(B, tag, Size, needs_dynamic_size_workaround)

        if ZI != C_NULL
            unsafe_store!(ZI, zero_allocation(B, TT, LLVMType, obj, AlignedSize, Size, #=ZeroAll=#false))
        end
        AS = Tracked
    else
        ptr8 = LLVM.PointerType(LLVM.IntType(8))
        mallocF, fty = get_function!(mod, "malloc", LLVM.FunctionType(ptr8, [value_type(Count)]))

        obj = call!(B, fty, mallocF, [Size])
        # if ZI != C_NULL
        #     unsafe_store!(ZI, LLVM.memset!(B, obj,  LLVM.ConstantInt(T_int8, 0),
        #                                           Size,
        #                                          #=align=#0 ).ref)
        # end
        AS = 0
    end

    LLVM.API.LLVMAddCallSiteAttribute(obj, LLVM.API.LLVMAttributeReturnIndex, EnumAttribute("noalias"))
    LLVM.API.LLVMAddCallSiteAttribute(obj, LLVM.API.LLVMAttributeReturnIndex, EnumAttribute("nonnull"))
    if isa(Count, LLVM.ConstantInt)
        val = convert(UInt, AlignedSize)
        val *= convert(UInt, Count)
        LLVM.API.LLVMAddCallSiteAttribute(obj, LLVM.API.LLVMAttributeReturnIndex, EnumAttribute("dereferenceable", val))
        LLVM.API.LLVMAddCallSiteAttribute(obj, LLVM.API.LLVMAttributeReturnIndex, EnumAttribute("dereferenceable_or_null", val))
    end

    mem = pointercast!(B, obj, LLVM.PointerType(LLVMType, AS))
    return LLVM.API.LLVMValueRef(mem.ref)
end

function julia_deallocator(B::LLVM.API.LLVMBuilderRef, Obj::LLVM.API.LLVMValueRef)
    B = LLVM.IRBuilder(B)
    Obj = LLVM.Value(Obj)
    julia_deallocator(B, Obj)
end

function julia_deallocator(B::LLVM.IRBuilder, Obj::LLVM.Value)
    mod = LLVM.parent(LLVM.parent(position(B)))

    T_void = LLVM.VoidType()
    if any_jltypes(LLVM.value_type(Obj))
        return LLVM.API.LLVMValueRef(C_NULL)
    else
        ptr8 = LLVM.PointerType(LLVM.IntType(8))
        freeF, fty = get_function!(mod, "free", LLVM.FunctionType(T_void, [ptr8]))
        callf = call!(B, fty, freeF, [pointercast!(B, Obj, ptr8)])
        LLVM.API.LLVMAddCallSiteAttribute(callf, LLVM.API.LLVMAttributeIndex(1), EnumAttribute("nonnull"))
    end
    return LLVM.API.LLVMValueRef(callf.ref)
end

function emit_inacterror(B, V, orig)
    B = LLVM.IRBuilder(B)
    curent_bb = position(B)
    orig = LLVM.Value(orig)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    bt = GPUCompiler.backtrace(orig)
    bts = sprint(io->Base.show_backtrace(io, bt))
    fmt = globalstring_ptr!(B, "%s:\nBacktrace\n"*bts)

    funcT = LLVM.FunctionType(LLVM.VoidType(), LLVMType[LLVM.PointerType(LLVM.Int8Type())], vararg=true)
    func, _ = get_function!(mod, "jl_errorf", funcT, [EnumAttribute("noreturn")])

    call!(B, funcT, func, LLVM.Value[fmt, LLVM.Value(V)])
    return nothing
end

macro augfunc(f)
   :(@cfunction((B, OrigCI, gutils, normalR, shadowR, tapeR) -> begin
     UInt8($f(LLVM.IRBuilder(B), LLVM.CallInst(OrigCI), GradientUtils(gutils), normalR, shadowR, tapeR)::Bool)
    end, UInt8, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})
    ))
end

macro revfunc(f)
   :(@cfunction((B, OrigCI, gutils, tape) -> begin
     $f(LLVM.IRBuilder(B), LLVM.CallInst(OrigCI), GradientUtils(gutils), tape == C_NULL ? nothing : LLVM.Value(tape))
    end,  Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)
    ))
end

macro fwdfunc(f)
   :(@cfunction((B, OrigCI, gutils, normalR, shadowR) -> begin
     UInt8($f(LLVM.IRBuilder(B), LLVM.CallInst(OrigCI), GradientUtils(gutils), normalR, shadowR)::Bool)
    end, UInt8, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})
    ))
end

function __init__()
    API.EnzymeSetHandler(@cfunction(julia_error, LLVM.API.LLVMValueRef, (Cstring, LLVM.API.LLVMValueRef, API.ErrorType, Ptr{Cvoid}, LLVM.API.LLVMValueRef, LLVM.API.LLVMBuilderRef)))
    API.EnzymeSetSanitizeDerivatives(@cfunction(julia_sanitize, LLVM.API.LLVMValueRef, (LLVM.API.LLVMValueRef, LLVM.API.LLVMValueRef, LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef)));
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
    API.EnzymeSetUndefinedValueForType(@cfunction(
                                            julia_undef_value_for_type, LLVM.API.LLVMValueRef, (LLVM.API.LLVMTypeRef,UInt8)))
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
        @augfunc(jlcall_augfwd),
        @revfunc(jlcall_rev),
        @fwdfunc(jlcall_fwd),
    )
    register_handler!(
        ("julia.call2",),
        @augfunc(jlcall2_augfwd),
        @revfunc(jlcall2_rev),
        @fwdfunc(jlcall2_fwd),
    )
    register_handler!(
        ("jl_apply_generic", "ijl_apply_generic"),
        @augfunc(generic_augfwd),
        @revfunc(generic_rev),
        @fwdfunc(generic_fwd),
    )
    register_handler!(
        ("jl_invoke", "ijl_invoke", "jl_f_invoke"),
        @augfunc(invoke_augfwd),
        @revfunc(invoke_rev),
        @fwdfunc(invoke_fwd),
    )
    register_handler!(
        ("jl_f__apply_latest", "jl_f__call_latest"),
        @augfunc(apply_latest_augfwd),
        @revfunc(apply_latest_rev),
        @fwdfunc(apply_latest_fwd),
    )
    register_handler!(
        ("jl_threadsfor",),
        @augfunc(threadsfor_augfwd),
        @revfunc(threadsfor_rev),
        @fwdfunc(threadsfor_fwd),
    )
    register_handler!(
        ("jl_pmap",),
        @augfunc(pmap_augfwd),
        @revfunc(pmap_rev),
        @fwdfunc(pmap_fwd),
    )
    register_handler!(
        ("jl_new_task", "ijl_new_task"),
        @augfunc(newtask_augfwd),
        @revfunc(newtask_rev),
        @fwdfunc(newtask_fwd),
    )
    register_handler!(
        ("jl_set_task_threadpoolid", "ijl_set_task_threadpoolid"),
        @augfunc(set_task_tid_augfwd),
        @revfunc(set_task_tid_rev),
        @fwdfunc(set_task_tid_fwd),
    )
    register_handler!(
        ("jl_enq_work",),
        @augfunc(enq_work_augfwd),
        @revfunc(enq_work_rev),
        @fwdfunc(enq_work_fwd)
    )
    register_handler!(
        ("enzyme_custom",),
        @augfunc(enzyme_custom_augfwd),
        @revfunc(enzyme_custom_rev),
        @fwdfunc(enzyme_custom_fwd)
    )
    register_handler!(
        ("jl_wait",),
        @augfunc(wait_augfwd),
        @revfunc(wait_rev),
        @fwdfunc(wait_fwd),
    )
    register_handler!(
        ("jl_","jl_breakpoint"),
        @augfunc(noop_augfwd),
        @revfunc(duplicate_rev),
        @fwdfunc(noop_fwd),
    )
    register_handler!(
        ("jl_array_copy","ijl_array_copy"),
        @augfunc(arraycopy_augfwd),
        @revfunc(arraycopy_rev),
        @fwdfunc(arraycopy_fwd),
    )
    register_handler!(
        ("jl_reshape_array","ijl_reshape_array"),
        @augfunc(arrayreshape_augfwd),
        @revfunc(arrayreshape_rev),
        @fwdfunc(arrayreshape_fwd),
    )
    register_handler!(
        ("jl_f_setfield","ijl_f_setfield"),
        @augfunc(setfield_augfwd),
        @revfunc(setfield_rev),
        @fwdfunc(setfield_fwd),
    )
    register_handler!(
        ("jl_box_float32","ijl_box_float32", "jl_box_float64", "ijl_box_float64"),
        @augfunc(boxfloat_augfwd),
        @revfunc(boxfloat_rev),
        @fwdfunc(boxfloat_fwd),
    )
    register_handler!(
        ("jl_f_tuple","ijl_f_tuple"),
        @augfunc(f_tuple_augfwd),
        @revfunc(f_tuple_rev),
        @fwdfunc(f_tuple_fwd),
    )
    register_handler!(
        ("jl_eqtable_get","ijl_eqtable_get"),
        @augfunc(eqtableget_augfwd),
        @revfunc(eqtableget_rev),
        @fwdfunc(eqtableget_fwd),
    )
    register_handler!(
        ("jl_eqtable_put","ijl_eqtable_put"),
        @augfunc(eqtableput_augfwd),
        @revfunc(eqtableput_rev),
        @fwdfunc(eqtableput_fwd),
    )
    register_handler!(
        ("jl_idtable_rehash","ijl_idtable_rehash"),
        @augfunc(idtablerehash_augfwd),
        @revfunc(idtablerehash_rev),
        @fwdfunc(idtablerehash_fwd),
    )
    register_handler!(
        ("jl_f__apply_iterate","ijl_f__apply_iterate"),
        @augfunc(apply_iterate_augfwd),
        @revfunc(apply_iterate_rev),
        @fwdfunc(apply_iterate_fwd),
    )
    register_handler!(
        ("jl_f__svec_ref","ijl_f__svec_ref"),
        @augfunc(f_svec_ref_augfwd),
        @revfunc(f_svec_ref_rev),
        @fwdfunc(f_svec_ref_fwd),
    )
    register_handler!(
        ("jl_new_structv","ijl_new_structv"),
        @augfunc(new_structv_augfwd),
        @revfunc(new_structv_rev),
        @fwdfunc(new_structv_fwd),
    )
    register_handler!(
        ("jl_get_binding_or_error", "ijl_get_binding_or_error"),
        @augfunc(get_binding_or_error_augfwd),
        @revfunc(get_binding_or_error_rev),
        @fwdfunc(get_binding_or_error_fwd),
    )
    register_handler!(
        ("jl_gc_add_finalizer_th","ijl_gc_add_finalizer_th", "jl_gc_add_ptr_finalizer","ijl_gc_add_ptr_finalizer"),
        @augfunc(finalizer_augfwd),
        @revfunc(finalizer_rev),
        @fwdfunc(finalizer_fwd),
    )
    register_handler!(
        ("jl_array_grow_end","ijl_array_grow_end"),
        @augfunc(jl_array_grow_end_augfwd),
        @revfunc(jl_array_grow_end_rev),
        @fwdfunc(jl_array_grow_end_fwd),
    )
    register_handler!(
        ("jl_array_del_end","ijl_array_del_end"),
        @augfunc(jl_array_del_end_augfwd),
        @revfunc(jl_array_del_end_rev),
        @fwdfunc(jl_array_del_end_fwd),
    )
    register_handler!(
        ("jl_f_getfield","ijl_f_getfield"),
        @augfunc(jl_getfield_augfwd),
        @revfunc(jl_getfield_rev),
        @fwdfunc(jl_getfield_fwd),
    )
    register_handler!(
        ("ijl_get_nth_field_checked","jl_get_nth_field_checked"),
        @augfunc(jl_nthfield_augfwd),
        @revfunc(jl_nthfield_rev),
        @fwdfunc(jl_nthfield_fwd),
    )
    register_handler!(
        ("jl_array_sizehint","ijl_array_sizehint"),
        @augfunc(jl_array_sizehint_augfwd),
        @revfunc(jl_array_sizehint_rev),
        @fwdfunc(jl_array_sizehint_fwd),
    )
    register_handler!(
        ("jl_array_ptr_copy","ijl_array_ptr_copy"),
        @augfunc(jl_array_ptr_copy_augfwd),
        @revfunc(jl_array_ptr_copy_rev),
        @fwdfunc(jl_array_ptr_copy_fwd),
    )
    register_handler!(
        ("jl_uv_associate_julia_struct","uv_async_init","cuLaunchHostFunc","uv_timer_init","uv_timer_start","jl_array_del_beg","ijl_array_del_beg","jl_array_grow_beg","ijl_array_grow_beg","cublasDgemm_v2", "cublasDscal_v2", "ijl_call_in_typeinf_world", "jl_call_in_typeinf_world"),
        @augfunc(jl_unhandled_augfwd),
        @revfunc(jl_unhandled_rev),
        @fwdfunc(jl_unhandled_fwd),
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
    TT::Type{<:Tuple}
    mode::API.CDerivativeMode
    width::Int
    rt::Type{<:Annotation{T} where T}
    run_enzyme::Bool
    abiwrap::Bool
    # Whether, in split mode, acessible primal argument data is modified
    # between the call and the split
    modifiedBetween::NTuple{N, Bool} where N
    # Whether to also return the primal
    returnPrimal::Bool
    # Whether to (in aug fwd) += by one
    shadowInit::Bool
    expectedTapeType::Type
    # Whether to use the pointer ABI, default true
    ABI::Type{<:ABI}
end

struct UnknownTapeType end

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
    Interpreter.EnzymeInterpreter(GPUCompiler.ci_cache(job), GPUCompiler.method_table(job), job.world, job.config.params.mode)

include("compiler/passes.jl")
include("compiler/optimize.jl")
include("compiler/interpreter.jl")
include("compiler/validation.jl")

import .Interpreter: isKWCallSignature

"""
Create the methodinstance pair, and lookup the primal return type.
"""
@inline function fspec(@nospecialize(F), @nospecialize(TT), world::Integer)
    # primal function. Inferred here to get return type
    _tt = (TT.parameters...,)

    primal_tt = Tuple{map(eltype, _tt)...}

    primal = GPUCompiler.methodinstance(F, primal_tt, world)

    return primal
end

##
# Enzyme compiler step
##

function annotate!(mod, mode)
    inactive = LLVM.StringAttribute("enzyme_inactive", "")
    active = LLVM.StringAttribute("enzyme_active", "")
    fns = functions(mod)

    for f in fns
        API.EnzymeAttributeKnownFunctions(f.ref)
    end

    for fname in inactivefns
        if haskey(fns, fname)
            fn = fns[fname]
            push!(function_attributes(fn), inactive)
            for u in LLVM.uses(fn)
                c = LLVM.user(u)
                if !isa(c, LLVM.CallInst)
                    continue
                end
                cf = LLVM.called_operand(c)
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
            push!(function_attributes(fn), LLVM.EnumAttribute("nofree", 0))
            for u in LLVM.uses(fn)
                c = LLVM.user(u)
                if !isa(c, LLVM.CallInst)
                    continue
                end
                cf = LLVM.called_operand(c)
                if !isa(cf, LLVM.Function)
                    continue
                end
                if LLVM.name(cf) != "julia.call" && LLVM.name(cf) != "julia.call2"
                    continue
                end
                if operands(c)[1] != fn
                    continue
                end
                LLVM.API.LLVMAddCallSiteAttribute(c, LLVM.API.LLVMAttributeFunctionIndex, LLVM.EnumAttribute("nofree", 0))
            end
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
            push!(function_attributes(fn), LLVM.EnumAttribute("readnone", 0))
            push!(function_attributes(fn), LLVM.StringAttribute("enzyme_shouldrecompute"))
        end
    end

    for fname in ("jl_excstack_state","ijl_excstack_state")
        if haskey(fns, fname)
            fn = fns[fname]
            push!(function_attributes(fn), LLVM.EnumAttribute("readonly", 0))
            push!(function_attributes(fn), LLVM.StringAttribute("inaccessiblememonly"))
        end
    end

    for fname in ("jl_types_equal", "ijl_types_equal")
        if haskey(fns, fname)
            fn = fns[fname]
            push!(function_attributes(fn), LLVM.EnumAttribute("readonly", 0))
            push!(function_attributes(fn), LLVM.StringAttribute("enzyme_shouldrecompute"))
        end
    end

    for fname in ("jl_f_getfield","ijl_f_getfield","jl_get_nth_field_checked","ijl_get_nth_field_checked")
        if haskey(fns, fname)
            fn = fns[fname]
            push!(function_attributes(fn), LLVM.EnumAttribute("readonly", 0))
            for u in LLVM.uses(fn)
                c = LLVM.user(u)
                if !isa(c, LLVM.CallInst)
                    continue
                end
                cf = LLVM.called_operand(c)
                if !isa(cf, LLVM.Function)
                    continue
                end
                if LLVM.name(cf) != "julia.call" && LLVM.name(cf) != "julia.call2"
                    continue
                end
                if operands(c)[1] != fn
                    continue
                end
                LLVM.API.LLVMAddCallSiteAttribute(c, LLVM.API.LLVMAttributeFunctionIndex, LLVM.EnumAttribute("readonly", 0))
            end
        end
    end

    for fname in ("julia.get_pgcstack", "julia.ptls_states", "jl_get_ptls_states")
        if haskey(fns, fname)
            fn = fns[fname]
            # TODO per discussion w keno perhaps this should change to readonly / inaccessiblememonly
            push!(function_attributes(fn), LLVM.EnumAttribute("readnone", 0))
            push!(function_attributes(fn), LLVM.StringAttribute("enzyme_shouldrecompute"))
        end
    end

    for fname in ("julia.pointer_from_objref",)
        if haskey(fns, fname)
            fn = fns[fname]
            push!(function_attributes(fn), LLVM.EnumAttribute("readnone", 0))
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
            push!(return_attributes(fn), LLVM.EnumAttribute("noalias", 0))
            push!(function_attributes(fn), LLVM.EnumAttribute("inaccessiblememonly", 0))
            for u in LLVM.uses(fn)
                c = LLVM.user(u)
                if !isa(c, LLVM.CallInst)
                    continue
                end
                cf = LLVM.called_operand(c)
                if cf == fn
                    LLVM.API.LLVMAddCallSiteAttribute(c, LLVM.API.LLVMAttributeReturnIndex, LLVM.EnumAttribute("noalias", 0))
                    LLVM.API.LLVMAddCallSiteAttribute(c, LLVM.API.LLVMAttributeFunctionIndex, LLVM.EnumAttribute("inaccessiblememonly", 0))
                end
                if !isa(cf, LLVM.Function)
                    continue
                end
                if LLVM.name(cf) != "julia.call" && LLVM.name(cf) != "julia.call2"
                    continue
                end
                if operands(c)[1] != fn
                    continue
                end
                LLVM.API.LLVMAddCallSiteAttribute(c, LLVM.API.LLVMAttributeReturnIndex, LLVM.EnumAttribute("noalias", 0))
                LLVM.API.LLVMAddCallSiteAttribute(c, LLVM.API.LLVMAttributeFunctionIndex, LLVM.EnumAttribute("inaccessiblememonly", 0))
            end
        end
    end

    for gc in ("llvm.julia.gc_preserve_begin", "llvm.julia.gc_preserve_end")
        if haskey(fns, gc)
            fn = fns[gc]
            push!(function_attributes(fn), LLVM.EnumAttribute("inaccessiblememonly", 0))
        end
    end

    for rfn in ("jl_object_id_", "jl_object_id", "ijl_object_id_", "ijl_object_id",
                "jl_eqtable_get", "ijl_eqtable_get")
        if haskey(fns, rfn)
            fn = fns[rfn]
            push!(function_attributes(fn), LLVM.EnumAttribute("readonly", 0))
        end
    end

    for rfn in ("jl_in_threaded_region_", "jl_in_threaded_region")
        if haskey(fns, rfn)
            fn = fns[rfn]
            push!(function_attributes(fn), LLVM.EnumAttribute("readonly", 0))
            push!(function_attributes(fn), LLVM.EnumAttribute("inaccessiblememonly", 0))
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
    ptr = reinterpret(Ptr{Cvoid}, convert(UInt, ce))
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
            ptr = reinterpret(Ptr{Cvoid}, convert(UInt, ce))
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
    ptr = reinterpret(Ptr{Cvoid}, convert(UInt, ce))
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

function enzyme_extract_world(fn::LLVM.Function)::UInt
    for fattr in collect(function_attributes(fn))
        if isa(fattr, LLVM.StringAttribute)
            if kind(fattr) == "enzymejl_world"
                return parse(UInt, LLVM.value(fattr))
            end
        end
    end
    GPUCompiler.@safe_error "Enzyme: Could not find world", fn
end

function enzyme_custom_extract_mi(orig::LLVM.Instruction)
    enzyme_custom_extract_mi(LLVM.called_operand(orig)::LLVM.Function)
end

function enzyme_custom_extract_mi(orig::LLVM.Function, error=true)
    mi = nothing
    RT = nothing
    for fattr in collect(function_attributes(orig))
        if isa(fattr, LLVM.StringAttribute)
            if kind(fattr) == "enzymejl_mi"
                ptr = reinterpret(Ptr{Cvoid}, parse(UInt, LLVM.value(fattr)))
                mi = Base.unsafe_pointer_to_objref(ptr)
            end
            if kind(fattr) == "enzymejl_rt"
                ptr = reinterpret(Ptr{Cvoid}, parse(UInt, LLVM.value(fattr)))
                RT = Base.unsafe_pointer_to_objref(ptr)
            end
        end
    end
    if error && mi === nothing
        GPUCompiler.@safe_error "Enzyme: Custom handler, could not find mi", orig
    end
    return mi, RT
end

function julia_type_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    inst = LLVM.Instruction(val)
    ctx = LLVM.context(inst)

    mi, RT = enzyme_custom_extract_mi(inst)

    ops = collect(operands(inst))[1:end-1]
    called = LLVM.called_operand(inst)


    llRT, sret, returnRoots =  get_return_info(RT)
    retRemoved, parmsRemoved = removed_ret_parms(inst)
    
    dl = string(LLVM.datalayout(LLVM.parent(LLVM.parent(LLVM.parent(inst)))))


    expectLen = (sret !== nothing) + (returnRoots !== nothing)
    for source_typ in mi.specTypes.parameters
        if isghostty(source_typ) || Core.Compiler.isconstType(source_typ)
            continue
        end
        expectLen+=1
    end
    expectLen -= length(parmsRemoved)
    
    # TODO fix the attributor inlining such that this can assert always true
    if expectLen == length(ops)

    cv = LLVM.called_operand(inst)
    swiftself = any(any(map(k->kind(k)==kind(EnumAttribute("swiftself")), collect(parameter_attributes(cv, i)))) for i in 1:length(collect(parameters(cv))))
    jlargs = classify_arguments(mi.specTypes, called_type(inst), sret !== nothing, returnRoots !== nothing, swiftself, parmsRemoved)


    for arg in jlargs
        if arg.cc == GPUCompiler.GHOST || arg.cc == RemovedParam
            continue
        end

        op_idx = arg.codegen.i
        rest = typetree(arg.typ, ctx, dl)
        if arg.cc == GPUCompiler.BITS_REF
            # adjust first path to size of type since if arg.typ is {[-1]:Int}, that doesn't mean the broader
            # object passing this in by ref isnt a {[-1]:Pointer, [-1,-1]:Int}
            # aka the next field after this in the bigger object isn't guaranteed to also be the same.
            if allocatedinline(arg.typ)
                shift!(rest, dl, 0, sizeof(arg.typ), 0)
            end
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

    if sret !== nothing
        idx = 0
        if !in(0, parmsRemoved)
            API.EnzymeMergeTypeTree(unsafe_load(args, idx+1), typetree(sret, ctx, dl))
            idx+=1
        end
        if returnRoots !== nothing
            if !in(1, parmsRemoved)
                allpointer = TypeTree(API.DT_Pointer, -1, ctx)
                API.EnzymeMergeTypeTree(unsafe_load(args, idx+1), typetree(returnRoots, ctx, dl))
            end
        end
    end
    
    end

    if llRT !== nothing && value_type(inst) != LLVM.VoidType()
        @assert !retRemoved
        API.EnzymeMergeTypeTree(ret, typetree(llRT, ctx, dl))
    end

    return UInt8(false)
end

function enzyme!(job, mod, primalf, TT, mode, width, parallel, actualRetType, wrap, modifiedBetween, returnPrimal, jlrules,expectedTapeType)
    world = job.world
    interp = GPUCompiler.get_interpreter(job)
    rt  = job.config.params.rt
    shadow_init = job.config.params.shadowInit
    ctx = context(mod)
    dl  = string(LLVM.datalayout(mod))

    tt = [TT.parameters[2:end]...,]

    args_activity     = API.CDIFFE_TYPE[]
    uncacheable_args  = Bool[]
    args_typeInfo     = TypeTree[]
    args_known_values = API.IntList[]


    @assert length(modifiedBetween) == length(TT.parameters)

    swiftself = any(any(map(k->kind(k)==kind(EnumAttribute("swiftself")), collect(parameter_attributes(primalf, i)))) for i in 1:length(collect(parameters(primalf))))
    if swiftself
        push!(args_activity, API.DFT_CONSTANT)
        push!(args_typeInfo, TypeTree())
        push!(uncacheable_args, false)
        push!(args_known_values, API.IntList())
    end

    for (i, T) in enumerate(TT.parameters)
        source_typ = eltype(T)
        if isghostty(source_typ) || Core.Compiler.isconstType(source_typ)
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
        elseif  T <: Duplicated || T<: BatchDuplicated || T<: BatchDuplicatedFunc
            push!(args_activity, API.DFT_DUP_ARG)
        elseif T <: DuplicatedNoNeed || T<: BatchDuplicatedNoNeed
            push!(args_activity, API.DFT_DUP_NONEED)
        else
            error("illegal annotation type")
        end
        typeTree = typetree(source_typ, ctx, dl)
        if isboxed
            merge!(typeTree, TypeTree(API.DT_Pointer, ctx))
            only!(typeTree, -1)
        end
        push!(args_typeInfo, typeTree)
        push!(uncacheable_args, modifiedBetween[i])
        push!(args_known_values, API.IntList())
    end
    @assert length(uncacheable_args) == length(collect(parameters(primalf)))
    @assert length(args_typeInfo) == length(collect(parameters(primalf)))

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
    elseif rt <: Duplicated || rt <: BatchDuplicated || rt<: BatchDuplicatedFunc
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
        returnUsed = !(isghostty(actualRetType) || Core.Compiler.isconstType(actualRetType))
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
            TapeType = EnzymeTapeToLoad{tape_type(LLVMType(utape))}
            tape = utape
        elseif tape != C_NULL
            TapeType = tape_type(LLVMType(tape))
        else
            TapeType = Cvoid
        end
        if expectedTapeType !== UnknownTapeType
            @assert expectedTapeType === TapeType
        end

        if wrap
            augmented_primalf = create_abi_wrapper(augmented_primalf, TT, rt, actualRetType, API.DEM_ReverseModePrimal, augmented, width, returnUsed, shadow_init, world, interp)
        end

        # TODOs:
        # 1. Handle mutable or !pointerfree arguments by introducing caching
        #     + specifically by setting uncacheable_args[i] = true

        adjointf = LLVM.Function(API.EnzymeCreatePrimalAndGradient(
            logic, primalf, retType, args_activity, TA,
            #=returnValue=#false, #=dretUsed=#false, #=mode=#API.DEM_ReverseModeGradient, width,
            #=additionalArg=#tape, #=forceAnonymousTape=#false, typeInfo,
            uncacheable_args, augmented, #=atomicAdd=# parallel))
        if wrap
            adjointf = create_abi_wrapper(adjointf, TT, rt, actualRetType, API.DEM_ReverseModeGradient, augmented, width, #=returnPrimal=#false, shadow_init, world, interp)
        end
    elseif mode == API.DEM_ReverseModeCombined
        returnUsed = !isghostty(actualRetType)
        returnUsed &= returnPrimal
        adjointf = LLVM.Function(API.EnzymeCreatePrimalAndGradient(
            logic, primalf, retType, args_activity, TA,
            #=returnValue=#returnUsed, #=dretUsed=#false, #=mode=#API.DEM_ReverseModeCombined, width,
            #=additionalArg=#C_NULL, #=forceAnonymousTape=#false, typeInfo,
            uncacheable_args, #=augmented=#C_NULL, #=atomicAdd=# parallel))
        augmented_primalf = nothing
        if wrap
            adjointf = create_abi_wrapper(adjointf, TT, rt, actualRetType, API.DEM_ReverseModeCombined, nothing, width, returnUsed, shadow_init, world, interp)
        end
    elseif mode == API.DEM_ForwardMode
        returnUsed = !(isghostty(actualRetType) || Core.Compiler.isconstType(actualRetType))
        returnUsed &= returnPrimal
        adjointf = LLVM.Function(API.EnzymeCreateForwardDiff(
            logic, primalf, retType, args_activity, TA,
            #=returnValue=#returnUsed, #=mode=#API.DEM_ForwardMode, width,
            #=additionalArg=#C_NULL, typeInfo,
            uncacheable_args))
        augmented_primalf = nothing
        if wrap
          pf = adjointf
          adjointf = create_abi_wrapper(adjointf, TT, rt, actualRetType, API.DEM_ForwardMode, nothing, width, returnUsed, shadow_init, world, interp)
        end
    else
        @assert "Unhandled derivative mode", mode
    end
    API.EnzymeLogicErasePreprocessedFunctions(logic)
    fix_decayaddr!(mod)
    return adjointf, augmented_primalf, TapeType
end

function create_abi_wrapper(enzymefn::LLVM.Function, TT, rettype, actualRetType, Mode::API.CDerivativeMode, augmented, width, returnPrimal, shadow_init, world, interp)
    is_adjoint = Mode == API.DEM_ReverseModeGradient || Mode == API.DEM_ReverseModeCombined
    is_split   = Mode == API.DEM_ReverseModeGradient || Mode == API.DEM_ReverseModePrimal
    needs_tape = Mode == API.DEM_ReverseModeGradient

    mod = LLVM.parent(enzymefn)
    ctx = LLVM.context(mod)

    push!(function_attributes(enzymefn), EnumAttribute("alwaysinline", 0))
    hasNoInline = any(map(k->kind(k)==kind(EnumAttribute("noinline")), collect(function_attributes(enzymefn))))
    if hasNoInline
        LLVM.API.LLVMRemoveEnumAttributeAtIndex(enzymefn, reinterpret(LLVM.API.LLVMAttributeIndex, LLVM.API.LLVMAttributeFunctionIndex), kind(EnumAttribute("noinline")))
    end
    T_void = convert(LLVMType, Nothing)
    ptr8 = LLVM.PointerType(LLVM.IntType(8))
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    # Create Enzyme calling convention
    T_wrapperargs = LLVMType[] # Arguments of the wrapper

    sret_types  = Type[]  # Julia types of all returned variables

    pactualRetType = actualRetType
    sret_union = is_sret_union(actualRetType)
    if sret_union
        actualRetType = Any
    end

    ActiveRetTypes = Type[]
    for (i, T) in enumerate(TT.parameters)
        source_typ = eltype(T)
        if isghostty(source_typ) || Core.Compiler.isconstType(source_typ)
            @assert T <: Const
            if is_adjoint && i != 1
                push!(ActiveRetTypes, Nothing)
            end
            continue
        end

        isboxed = GPUCompiler.deserves_argbox(source_typ)
        llvmT = isboxed ? T_prjlvalue : convert(LLVMType, source_typ)

        push!(T_wrapperargs, llvmT)

        if T <: Const || T <: BatchDuplicatedFunc
            if is_adjoint && i != 1
                push!(ActiveRetTypes, Nothing)
            end
            continue
        end

        if T <: Active
            if is_adjoint && i != 1
                if width == 1
                    push!(ActiveRetTypes, source_typ)
                else
                    push!(ActiveRetTypes, NTuple{width, source_typ})
                end
            end
        elseif T <: Duplicated || T <: DuplicatedNoNeed
            @assert width == 1
            push!(T_wrapperargs, LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmT)))
            if is_adjoint && i != 1
                push!(ActiveRetTypes, Nothing)
            end
        elseif T <: BatchDuplicated || T <: BatchDuplicatedNoNeed
            push!(T_wrapperargs, LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmT)))
            if is_adjoint && i != 1
                push!(ActiveRetTypes, Nothing)
            end
        else
            error("calling convention should be annotated, got $T")
        end
    end

    if is_adjoint
        NT = Tuple{ActiveRetTypes...}
        if any(any_jltypes(convert(LLVM.LLVMType, b; allow_boxed=true)) for b in ActiveRetTypes)
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
        dretTy = LLVM.LLVMType(API.EnzymeGetShadowType(width, convert(LLVMType, actualRetType)))
        push!(T_wrapperargs, dretTy)
    end

    data    = Array{Int64}(undef, 3)
    existed = Array{UInt8}(undef, 3)
    if Mode == API.DEM_ReverseModePrimal
        API.EnzymeExtractReturnInfo(augmented, data, existed)
        # tape -- todo ??? on wrap
        if existed[1] != 0
            tape = API.EnzymeExtractTapeTypeFromAugmentation(augmented)
        end

        tape = API.EnzymeExtractTapeTypeFromAugmentation(augmented)
        utape = API.EnzymeExtractUnderlyingTapeTypeFromAugmentation(augmented)
        if utape != C_NULL
            TapeType = EnzymeTapeToLoad{tape_type(LLVMType(utape))}
        elseif tape != C_NULL
            TapeType = tape_type(LLVMType(tape))
        else
            TapeType = Cvoid
        end
        push!(sret_types, TapeType)

        # primal return
        if existed[2] != 0
            @assert returnPrimal
            push!(sret_types, actualRetType)
        else
            @assert !returnPrimal
            push!(sret_types, Nothing)
        end
        # shadow return
        if existed[3] != 0
            if rettype <: Duplicated || rettype <: DuplicatedNoNeed
                if width == 1
                    push!(sret_types, actualRetType)
                else
                    push!(sret_types, AnonymousStruct(NTuple{width, actualRetType}))
                end
            end
        else
            @assert rettype <: Const || rettype <: Active
            push!(sret_types, Nothing)
        end
    end
    if Mode == API.DEM_ReverseModeCombined
        if returnPrimal
            push!(sret_types, actualRetType)
        end
    end
    if Mode == API.DEM_ForwardMode
        if returnPrimal
            push!(sret_types, actualRetType)
        end
        if rettype <: Duplicated || rettype <: DuplicatedNoNeed
            if width == 1
                push!(sret_types, actualRetType)
            else
                push!(sret_types, AnonymousStruct(NTuple{width, actualRetType}))
            end
        end
    end

    combinedReturn = Tuple{sret_types...}
    if any(any_jltypes(convert(LLVM.LLVMType, T; allow_boxed=true)) for T in sret_types)
        combinedReturn = AnonymousStruct(combinedReturn)
    end

    uses_sret = is_sret(combinedReturn)

    jltype = convert(LLVM.LLVMType, combinedReturn)

    numLLVMReturns = nothing
    if isa(jltype, LLVM.ArrayType)
        numLLVMReturns = length(jltype)
    elseif isa(jltype, LLVM.StructType)
        numLLVMReturns = length(elements(jltype))
    elseif isa(jltype, LLVM.VoidType)
        numLLVMReturns = 0
    else
        @assert false "illegal rt"
    end

    returnRoots = false
    root_ty = nothing
    if uses_sret
    	returnRoots = deserves_rooting(jltype)
		if returnRoots
	        tracked = CountTrackedPointers(jltype)
            root_ty = LLVM.ArrayType(T_prjlvalue, tracked.count)
            pushfirst!(T_wrapperargs, LLVM.PointerType(root_ty))

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
            tape = LLVM.LLVMType(tape)
            jltape = convert(LLVM.LLVMType, tape_type(tape); allow_boxed=true)
            push!(T_wrapperargs, jltape)
        else
            needs_tape = false
        end
    end

    T_ret = returnRoots ? T_void : jltype
    FT = LLVM.FunctionType(T_ret, T_wrapperargs)
    llvm_f = LLVM.Function(mod, safe_name(LLVM.name(enzymefn)*"wrap"), FT)
    API.EnzymeCloneFunctionDISubprogramInto(llvm_f, enzymefn)
    dl = datalayout(mod)

    params = [parameters(llvm_f)...]

    LLVM.IRBuilder() do builder
        entry = BasicBlock(llvm_f, "entry")
        position!(builder, entry)

        realparms = LLVM.Value[]
        i = 1

        if returnRoots
            sret = params[i]
            i+= 1

            attr = if LLVM.version().major >= 12
                TypeAttribute("sret", jltype)
            else
                EnumAttribute("sret")
            end
            push!(parameter_attributes(llvm_f, 1), attr)
            push!(parameter_attributes(llvm_f, 1), EnumAttribute("noalias"))
            push!(parameter_attributes(llvm_f, 2), EnumAttribute("noalias"))
        elseif jltype != T_void
            sret = alloca!(builder, jltype)
        end
        rootRet = nothing
        if returnRoots
            rootRet = params[i]
            i+=1
        end

        activeNum = 0

        for T in TT.parameters
            T′ = eltype(T)

            if isghostty(T′) || Core.Compiler.isconstType(T′)
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
                    ptr = gep!(builder, jltype, sret, [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), activeNum)])
                    cst = pointercast!(builder, ptr, ptr8)
                    push!(realparms, ptr)

                    LLVM.memset!(builder, cst,  LLVM.ConstantInt(LLVM.IntType(8), 0),
                                                LLVM.ConstantInt(LLVM.IntType(64), LLVM.storage_size(dl, Base.eltype(LLVM.value_type(ptr)) )),
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
            elseif T <: BatchDuplicatedFunc
                Func = get_func(T)
                funcspec = GPUCompiler.methodinstance(Func, Tuple{}, world)
                llvmf = nested_codegen!(Mode, mod, funcspec, world)
                push!(function_attributes(llvmf), EnumAttribute("alwaysinline", 0))
                Func_RT = Core.Compiler.typeinf_ext_toplevel(interp, funcspec).rettype
                @assert Func_RT == NTuple{width, T′}
                _, psret, _ = get_return_info(Func_RT)
                args = LLVM.Value[]
                if psret !== nothing
                    psret = alloca!(builder, convert(LLVMType, Func_RT))
                    push!(args, psret)
                end
                res = LLVM.call!(builder, LLVM.function_type(llvmf), llvmf, args)
                if LLVM.get_subprogram(llvmf) !== nothing
                    metadata(res)[LLVM.MD_dbg] = DILocation( 0, 0, LLVM.get_subprogram(llvm_f) )
                end
                if psret !== nothing
                    res = load!(builder, convert(LLVMType, Func_RT), psret)
                end
                push!(realparms, res)
            else
                @assert false
            end
        end

        if is_adjoint && rettype <: Active
            push!(realparms, params[i])
            i += 1
        end

        if needs_tape

            function rectype(val::LLVM.Value, idxs::Vector{Cuint})
                ty = LLVM.value_type(val)
                for i in idxs
                    if isa(ty, LLVM.ArrayType)
                        ty = eltype(ty)
                    else
                        @assert isa(ty, LLVM.StructType)
                        ty = elements(ty)[i+1]
                    end
                end
                return ty
            end
            function typefix(val::LLVM.Value, tape::LLVM.LLVMType, prev::LLVM.Value, lidxs::Vector{Cuint}, ridxs::Vector{Cuint})::LLVM.Value
                ctype = rectype(val, lidxs)
                if ctype == tape
                    if length(lidxs) != 0
                        val = API.e_extract_value!(builder, val, lidxs)
                    end
                    if length(ridxs) == 0
                        return val
                    else
                        return API.e_insert_value!(builder, prev, val, ridxs)
                    end
                end

                if isa(tape, LLVM.StructType)
                    if isa(ctype, LLVM.ArrayType)
                        @assert length(ctype) == length(elements(tape))
                        for (i, ty) in enumerate(elements(tape))
                            ln = copy(lidxs)
                            push!(ln, i-1)
                            rn = copy(ridxs)
                            push!(rn, i-1)
                            prev = typefix(val, ty, prev, ln, rn)
                        end
                        return prev
                    end
                    if isa(ctype, LLVM.StructType)
                        @assert length(elements(ctype)) == length(elements(tape))
                        for (i, ty) in enumerate(elements(tape))
                            ln = copy(lidxs)
                            push!(ln, i-1)
                            rn = copy(ridxs)
                            push!(rn, i-1)
                            prev = typefix(val, ty, prev, ln, rn)
                        end
                        return prev
                    end
                end

                if isa(tape, LLVM.IntegerType) && LLVM.width(tape) == 1 && LLVM.width(ctype) != LLVM.width(tape)
                    if length(lidxs) != 0
                        val = API.e_extract_value!(builder, val, lidxs)
                    end
                    val = trunc!(builder, val, tape)
                    return if length(ridxs) != 0
                        API.e_insert_value!(builder, prev, val, ridxs)
                    else
                        val
                    end
                end
                if isa(tape, LLVM.PointerType) && isa(ctype, LLVM.PointerType) && LLVM.addrspace(tape) == LLVM.addrspace(ctype)
                    if length(lidxs) != 0
                        val = API.e_extract_value!(builder, val, lidxs)
                    end
                    val = pointercast!(builder, val, tape)
                    return if length(ridxs) != 0
                        API.e_insert_value!(builder, prev, val, ridxs)
                    else
                        val
                    end
                end
                if isa(ctype, LLVM.ArrayType) && length(ctype) == 1 && eltype(ctype) == tape
                    lhs_n = copy(lidxs)
                    push!(lhs_n, 0)
                    return typefix(val, tape, prev, lhs_n, ridxs)
                end
                @show ctype, tape, val, prev, idxs, tape_type(tape), convert(LLVM.LLVMType, tape_type(tape); allow_boxed=true)
                @assert false
            end

            # Fix calling convention within julia that Tuple{Float,Float} ->[2 x float] rather than {float, float}
            # and that Bool -> i8, not i1
            tparm = params[i]
            tparm = typefix(tparm, tape, LLVM.UndefValue(tape), Cuint[], Cuint[])
            push!(realparms, tparm)
            i += 1
        end

        val = call!(builder, LLVM.function_type(enzymefn), enzymefn, realparms)
        if LLVM.get_subprogram(llvm_f) !== nothing
            metadata(val)[LLVM.MD_dbg] = DILocation( 0, 0, LLVM.get_subprogram(llvm_f) )
        end

        if Mode == API.DEM_ReverseModePrimal
            returnNum = 0
            for i in 1:3
                if existed[i] != 0
                    eval = val
                    if data[i] != -1
                        eval = extract_value!(builder, val, data[i])
                    end
                    ptr = inbounds_gep!(builder, jltype, sret, [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), returnNum)])
                    ptr = pointercast!(builder, ptr, LLVM.PointerType(value_type(eval)))
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

                        cf = nested_codegen!(Mode, mod, add_one_in_place, Tuple{actualRetType}, world)
                        push!(function_attributes(cf), EnumAttribute("alwaysinline", 0))
                        for shadowv in shadows
                            c = call!(builder, LLVM.function_type(cf), cf, [shadowv])
                            if LLVM.get_subprogram(llvm_f) !== nothing
                                metadata(c)[LLVM.MD_dbg] = DILocation( 0, 0, LLVM.get_subprogram(llvm_f) )
                            end
                        end
                    end
                elseif !isghostty(sret_types[i])
                    @assert !(isghostty(combinedReturn) || Core.Compiler.isconstType(combinedReturn) )
                    @assert Core.Compiler.isconstType(sret_types[i])
                    eval = makeInstanceOf(sret_types[i])
                    ptr = inbounds_gep!(builder, jltype, sret, [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), returnNum)])
                    ptr = pointercast!(builder, ptr, LLVM.PointerType(value_type(eval)))
                    si = store!(builder, eval, ptr)
                    returnNum+=1
                end
            end
            @assert returnNum == numLLVMReturns
        elseif Mode == API.DEM_ForwardMode
            count_Sret = 0
            returnUsed = !isghostty(actualRetType)
            if returnUsed
                if returnPrimal
                    count_Sret += 1
                end
                if !(rettype <: Const)
                    count_Sret += 1
                end
            end
            for returnNum in 0:(count_Sret-1)
                eval = val
                if count_Sret > 1
                    eval = extract_value!(builder, val, returnNum)
                end
                ptr = inbounds_gep!(builder, jltype, sret, [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), returnNum)])
                ptr = pointercast!(builder, ptr, LLVM.PointerType(value_type(eval)))
                si = store!(builder, eval, ptr)
            end
            @assert count_Sret == numLLVMReturns
        else
            activeNum = 0
            returnNum = 0
            if Mode == API.DEM_ReverseModeCombined
                if returnPrimal
                    if !isghostty(actualRetType)
                        eval = extract_value!(builder, val, returnNum)
                        store!(builder, eval, inbounds_gep!(builder, jltype, sret, [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), length(elements(jltype))-1 )]))
                        returnNum+=1
                    end
                end
            end
            for T in TT.parameters[2:end]
                if T <: Active
                    T′ = eltype(T)
                    isboxed = GPUCompiler.deserves_argbox(T′)
                    if !isboxed
                        eval = extract_value!(builder, val, returnNum)
                        store!(builder, eval, inbounds_gep!(builder, jltype, sret, [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), 0), LLVM.ConstantInt(LLVM.IntType(32), activeNum)]))
                        returnNum+=1
                    end
                    activeNum+=1
                end
            end
            @assert (returnNum - activeNum) + (activeNum != 0 ? 1 : 0) == numLLVMReturns
        end

        if returnRoots
            count = 0
            todo = Tuple{Vector{LLVM.Value},LLVM.LLVMType}[([LLVM.ConstantInt(LLVM.IntType(64), 0)], jltype)]
            while length(todo) != 0
                path, ty = popfirst!(todo)
                if isa(ty, LLVM.PointerType)
                    loc = inbounds_gep!(builder, root_ty, rootRet, [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), count)])
                    count+=1
                    outloc = inbounds_gep!(builder, jltype, sret, path)
                    store!(builder, load!(builder, ty, outloc), loc)
                    continue
                end
                if isa(ty, LLVM.ArrayType)
                    if any_jltypes(ty)
                        for i=1:length(ty)
                            npath = copy(path)
                            push!(npath, LLVM.ConstantInt(LLVM.IntType(32), i-1))
                            push!(todo, (npath, eltype(ty)))
                        end
                    end
                    continue
                end
                if isa(ty, LLVM.VectorType)
                    if any_jltypes(ty)
                        for i=1:size(ty)
                            npath = copy(path)
                            push!(npath, LLVM.ConstantInt(LLVM.IntType(32), i-1))
                            push!(todo, (npath, eltype(ty)))
                        end
                    end
                    continue
                end
                if isa(ty, LLVM.StructType)
                    for (i, t) in enumerate(LLVM.elements(ty))
                        if any_jltypes(t)
                            npath = copy(path)
                            push!(npath, LLVM.ConstantInt(LLVM.IntType(32), i-1))
                            push!(todo, (npath, t))
                        end
                    end
                    continue
                end
            end
            @assert count == tracked.count
        end
        if T_ret != T_void
            ret!(builder, load!(builder, T_ret, sret))
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
        if isa(value_type(param), LLVM.PointerType)
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

struct RemovedParam
end

# Modified from GPUCompiler classify_arguments
function classify_arguments(source_sig::Type, codegen_ft::LLVM.FunctionType, has_sret::Bool, has_returnroots::Bool, has_swiftself::Bool, parmsRemoved::Vector{UInt64})
    codegen_types = parameters(codegen_ft)

    args = []
    codegen_i = 1
    orig_i = 1
    if has_sret
        if !in(orig_i-1, parmsRemoved)
            codegen_i += 1
        end
        orig_i += 1
    end
    if has_returnroots
        if !in(orig_i-1, parmsRemoved)
            codegen_i += 1
        end
        orig_i += 1
    end
    if has_swiftself
        if !in(orig_i-1, parmsRemoved)
            codegen_i += 1
        end
        orig_i += 1
    end
    for (source_i, source_typ) in enumerate(source_sig.parameters)
        if isghostty(source_typ) || Core.Compiler.isconstType(source_typ)
            push!(args, (cc=GPUCompiler.GHOST, typ=source_typ, arg_i=source_i))
            continue
        end
        if in(orig_i-1, parmsRemoved)
            push!(args, (cc=RemovedParam, typ=source_typ))
            orig_i += 1
            continue
        end
        codegen_typ = codegen_types[codegen_i]
        if codegen_typ isa LLVM.PointerType && !issized(eltype(codegen_typ))
            push!(args, (cc=GPUCompiler.MUT_REF, typ=source_typ, arg_i=source_i,
                         codegen=(typ=codegen_typ, i=codegen_i)))
        elseif codegen_typ isa LLVM.PointerType && issized(eltype(codegen_typ)) &&
               !(source_typ <: Ptr) && !(source_typ <: Core.LLVMPtr)
            push!(args, (cc=GPUCompiler.BITS_REF, typ=source_typ, arg_i=source_i,
                         codegen=(typ=codegen_typ, i=codegen_i)))
        else
            push!(args, (cc=GPUCompiler.BITS_VALUE, typ=source_typ, arg_i=source_i,
                         codegen=(typ=codegen_typ, i=codegen_i)))
        end
        codegen_i += 1
        orig_i += 1
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
            if LLVM.addrspace(T) != Tracked
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
# returns if all unboxed
function for_each_uniontype_small(f, ty, counter=Ref(0))
    if counter[] > 127
        return false
    end
    if ty isa Union
        allunbox  = for_each_uniontype_small(f, ty.a, counter)
        allunbox &= for_each_uniontype_small(f, ty.b, counter)
        return allunbox
    end
    # https://github.com/JuliaLang/julia/blob/170d6439445c86e640214620dad3423d2bb42337/src/codegen.cpp#L1233
    if Base.isconcretetype(ty) && !ismutabletype(ty) && Base.datatype_pointerfree(ty)
        counter[] += 1
        f(ty)
        return true
    end
    return false
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
function is_sret(jlrettype)
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
        rt = convert(LLVMType, jlrettype )
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

# https://github.com/JuliaLang/julia/blob/0a696a3842750fcedca8832bc0aabe9096c7658f/src/codegen.cpp#L6812
function get_return_info(jlrettype)::Tuple{Union{Nothing, Type}, Union{Nothing, Type}, Union{Nothing, Type}}
    sret = nothing
    returnRoots = nothing
    rt = nothing
    if jlrettype === Union{}
        rt = Nothing
    elseif Base.isstructtype(jlrettype) && Base.issingletontype(jlrettype) &&isa(jlrettype, DataType)
        rt = Nothing
    elseif jlrettype isa Union
        nbytes = 0
        allunbox = for_each_uniontype_small(jlrettype) do jlrettype
            if !(Base.issingletontype(jlrettype) && isa(jlrettype, DataType))
               nbytes = max(nbytes, sizeof(jlrettype))
            end
        end
        if nbytes != 0
            rt = NamedTuple{(Symbol("1"), Symbol("2")),Tuple{Any,UInt8}}
            # Pointer to?, Ptr{NTuple{UInt8, allunbox}
            sret = Ptr{jlrettype}
        elseif allunbox
            rt = UInt8
        else
            rt = Any
        end
    elseif !GPUCompiler.deserves_retbox(jlrettype)
        lRT = convert(LLVMType, jlrettype )
        if !isa(lRT, LLVM.VoidType) && GPUCompiler.deserves_sret(jlrettype, lRT)
            sret = Ptr{jlrettype}
            tracked = CountTrackedPointers(lRT)
            @assert !tracked.derived
            if tracked.count != 0 && !tracked.all
                returnRoots = Ptr{AnyArray(Int(tracked.count))}
            end
        else
            rt = jlrettype
        end
    else
        # retbox
        rt = Ptr{jlrettype}
    end

    return (rt, sret, returnRoots)
end

# Modified from GPUCompiler/src/irgen.jl:365 lower_byval
function lower_convention(functy::Type, mod::LLVM.Module, entry_f::LLVM.Function, actualRetType::Type)
    entry_ft = LLVM.function_type(entry_f)

    RT = LLVM.return_type(entry_ft)

    # generate the wrapper function type & definition
    wrapper_types = LLVM.LLVMType[]
    _, sret, returnRoots = get_return_info(actualRetType)
    sret_union = is_sret_union(actualRetType)

    if sret_union
        T_jlvalue = LLVM.StructType(LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
        RT = T_prjlvalue
    elseif sret !== nothing
        RT = convert(LLVMType, eltype(sret))
    end
    sret = sret !== nothing
    returnRoots = returnRoots !== nothing

    # TODO removed implications
    retRemoved, parmsRemoved = removed_ret_parms(entry_f)
    swiftself = any(any(map(k->kind(k)==kind(EnumAttribute("swiftself")), collect(parameter_attributes(entry_f, i)))) for i in 1:length(collect(parameters(entry_f))))
    @assert !swiftself "Swiftself attribute coming from differentiable context is not supported"
	prargs = classify_arguments(functy, entry_ft, sret, returnRoots, swiftself, parmsRemoved)
    args = copy(prargs)
    filter!(args) do arg
        arg.cc != GPUCompiler.GHOST && arg.cc != RemovedParam
    end


    # @assert length(args) == length(collect(parameters(entry_f))[1+sret+returnRoots:end])


    # if returnRoots
    # 	push!(wrapper_types, value_type(parameters(entry_f)[1+sret]))
    # end
    #

    if swiftself
        push!(wrapper_types, value_type(parameters(entry_f)[1+sret+returnRoots]))
    end

    for arg in args
        typ = if !GPUCompiler.deserves_argbox(arg.typ) && arg.cc == GPUCompiler.BITS_REF
            eltype(arg.codegen.typ)
        else
            arg.codegen.typ
        end
        push!(wrapper_types, typ)
    end
    wrapper_fn = LLVM.name(entry_f)
    LLVM.name!(entry_f, safe_name(wrapper_fn * ".inner"))
    wrapper_ft = LLVM.FunctionType(RT, wrapper_types)
    wrapper_f = LLVM.Function(mod, LLVM.name(entry_f), wrapper_ft)
    callconv!(wrapper_f, callconv(entry_f))
    sfn = LLVM.get_subprogram(entry_f)
    if sfn !== nothing
        LLVM.set_subprogram!(wrapper_f, sfn)
    end

    hasReturnsTwice = any(map(k->kind(k)==kind(EnumAttribute("returns_twice")), collect(function_attributes(entry_f))))
    hasNoInline = any(map(k->kind(k)==kind(EnumAttribute("noinline")), collect(function_attributes(entry_f))))
    if hasNoInline
        LLVM.API.LLVMRemoveEnumAttributeAtIndex(entry_f, reinterpret(LLVM.API.LLVMAttributeIndex, LLVM.API.LLVMAttributeFunctionIndex), kind(EnumAttribute("noinline")))
    end
    push!(function_attributes(wrapper_f), EnumAttribute("returns_twice"))
    push!(function_attributes(entry_f), EnumAttribute("returns_twice"))
    if swiftself
        push!(parameter_attributes(wrapper_f, 1), EnumAttribute("swiftself"))
    end

    # emit IR performing the "conversions"
    let builder = IRBuilder()
        toErase = LLVM.CallInst[]
        for u in LLVM.uses(entry_f)
            ci = LLVM.user(u)
            if !isa(ci, LLVM.CallInst) || called_operand(ci) != entry_f
                continue
            end
            @assert !sret_union
            ops = collect(operands(ci))[1:end-1]
            position!(builder, ci)
            nops = LLVM.Value[]
			if returnRoots
				push!(nops, ops[1+sret])
			end
            if swiftself
                push!(nops, ops[1+sret+returnRoots])
            end
            for arg in args
                parm = ops[arg.codegen.i]
                if !GPUCompiler.deserves_argbox(arg.typ) && arg.cc == GPUCompiler.BITS_REF
                    push!(nops, load!(builder, convert(LLVMType, arg.typ), parm))
                else
                    push!(nops, parm)
                end
            end
            res = call!(builder, LLVM.function_type(wrapper_f), wrapper_f, nops)
            callconv!(res, callconv(wrapper_f))
            if sret
                @assert value_type(res) == eltype(value_type(ops[1]))
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

        entry = BasicBlock(wrapper_f, "entry")
        position!(builder, entry)
        if LLVM.get_subprogram(entry_f) !== nothing
            debuglocation!(builder, DILocation(0, 0, LLVM.get_subprogram(entry_f)))
        end

        wrapper_args = Vector{LLVM.Value}()

        sretPtr = nothing
        if sret
            if !in(0, parmsRemoved)
                sretPtr = alloca!(builder, eltype(value_type(parameters(entry_f)[1])))
                push!(wrapper_args, sretPtr)
            end
            if returnRoots && !in(1, parmsRemoved)
                retRootPtr = alloca!(builder, eltype(value_type(parameters(entry_f)[1+sret])))
                # retRootPtr = alloca!(builder, parameters(wrapper_f)[1])
                push!(wrapper_args, retRootPtr)
            end
        end
        if swiftself
            push!(wrapper_args, parameters(wrapper_f)[1])
        end

        # perform argument conversions
        for arg in args
            parm = parameters(entry_f)[arg.codegen.i]
            wrapparm = parameters(wrapper_f)[arg.codegen.i-sret-returnRoots]
            if !GPUCompiler.deserves_argbox(arg.typ) && arg.cc == GPUCompiler.BITS_REF
                # copy the argument value to a stack slot, and reference it.
                ty = value_type(parm)
                if !isa(ty, LLVM.PointerType)
                    @safe_show entry_f, args, parm, ty
                end
                @assert isa(ty, LLVM.PointerType)
                ptr = alloca!(builder, eltype(ty))
                if LLVM.addrspace(ty) != 0
                    ptr = addrspacecast!(builder, ptr, ty)
                end
                @assert eltype(ty) == value_type(wrapparm)
                store!(builder, wrapparm, ptr)
                push!(wrapper_args, ptr)
            else
                push!(wrapper_args, wrapparm)
                for attr in collect(parameter_attributes(entry_f, arg.codegen.i))
                    push!(parameter_attributes(wrapper_f, arg.codegen.i-sret-returnRoots), attr)
                end
            end
        end
        res = call!(builder, LLVM.function_type(entry_f), entry_f, wrapper_args)

        if LLVM.get_subprogram(entry_f) !== nothing
            metadata(res)[LLVM.MD_dbg] = DILocation( 0, 0, LLVM.get_subprogram(entry_f) )
        end

        callconv!(res, LLVM.callconv(entry_f))
        if swiftself
            attr = EnumAttribute("swiftself")
            LLVM.API.LLVMAddCallSiteAttribute(res, LLVM.API.LLVMAttributeIndex(1+sret+returnRoots), attr)
        end

        # Box union return, from https://github.com/JuliaLang/julia/blob/81813164963f38dcd779d65ecd222fad8d7ed437/src/cgutils.cpp#L3138
        if sret_union
            if retRemoved
                ret!(builder)
            else
                def = BasicBlock(wrapper_f, "defaultBB")
                scase = extract_value!(builder, res, 1)
                sw = switch!(builder, scase, def)
                counter = 1
                T_int8 = LLVM.Int8Type()
                T_int64 = LLVM.Int64Type()
                T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
                T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
                T_prjlvalue_UT = LLVM.PointerType(T_jlvalue)
                function inner(jlrettype)
                    BB = BasicBlock(wrapper_f, "box_union")
                    position!(builder, BB)

                    if isghostty(jlrettype) || Core.Compiler.isconstType(jlrettype)
                        fill_val = unsafe_to_llvm(jlrettype.instance)
                        ret!(builder, fill_val)
                    else
                        obj = emit_allocobj!(builder, jlrettype)
                        if sretPtr !== nothing
                            llty = convert(LLVMType, jlrettype)
                            ld = load!(builder, llty, bitcast!(builder, sretPtr, LLVM.PointerType(llty, addrspace(value_type(sretPtr)))))
                            store!(builder, ld, bitcast!(builder, obj, LLVM.PointerType(llty, addrspace(value_type(obj)))))
                            # memcpy!(builder, bitcast!(builder, obj, LLVM.PointerType(T_int8, addrspace(value_type(obj)))), 0, bitcast!(builder, sretPtr, LLVM.PointerType(T_int8)), 0, LLVM.ConstantInt(T_int64, sizeof(jlrettype)))
                        end
                        ret!(builder, obj)
                    end

                    LLVM.API.LLVMAddCase(sw, LLVM.ConstantInt(value_type(scase), counter), BB)
                    counter+=1
                    return
                end
                for_each_uniontype_small(inner, actualRetType)

                position!(builder, def)
                fill_val = unsafe_to_llvm(nothing)
                ret!(builder, fill_val)
            end
        elseif sret
            if sretPtr === nothing
                ret!(builder)
            else
                ret!(builder, load!(builder, RT, sretPtr))
            end
        elseif LLVM.return_type(entry_ft) == LLVM.VoidType()
            ret!(builder)
        else
            ret!(builder, res)
        end
        dispose(builder)
    end

    # early-inline the original entry function into the wrapper
    push!(function_attributes(entry_f), EnumAttribute("alwaysinline", 0))
    linkage!(entry_f, LLVM.API.LLVMInternalLinkage)

    fixup_metadata!(entry_f)
    
    mi, rt = enzyme_custom_extract_mi(entry_f)
    attributes = function_attributes(wrapper_f)
    push!(attributes, StringAttribute("enzymejl_mi", string(convert(UInt, pointer_from_objref(mi)))))
    push!(attributes, StringAttribute("enzymejl_rt", string(convert(UInt, unsafe_to_pointer(rt)))))

    if LLVM.API.LLVMVerifyFunction(wrapper_f, LLVM.API.LLVMReturnStatusAction) != 0
        @safe_show mod
        @safe_show LLVM.API.LLVMVerifyFunction(wrapper_f, LLVM.API.LLVMPrintMessageAction)
        @safe_show wrapper_f
        @safe_show parmsRemoved, retRemoved, prargs
        flush(stdout)
        throw(LLVM.LLVMException("broken function"))
    end

	ModulePassManager() do pm
        always_inliner!(pm)
        run!(pm, mod)
    end
    if !hasReturnsTwice
        LLVM.API.LLVMRemoveEnumAttributeAtIndex(wrapper_f, reinterpret(LLVM.API.LLVMAttributeIndex, LLVM.API.LLVMAttributeFunctionIndex), kind(EnumAttribute("returns_twice")))
    end
    if hasNoInline
        LLVM.API.LLVMRemoveEnumAttributeAtIndex(wrapper_f, reinterpret(LLVM.API.LLVMAttributeIndex, LLVM.API.LLVMAttributeFunctionIndex), kind(EnumAttribute("alwaysinline")))
        push!(function_attributes(wrapper_f), EnumAttribute("noinline"))
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
                 libraries::Bool=true, deferred_codegen::Bool=true, optimize::Bool=true, toplevel::Bool=true,
                 strip::Bool=false, validate::Bool=true, only_entry::Bool=false, parent_job::Union{Nothing, CompilerJob} = nothing)
    params  = job.config.params
    expectedTapeType = params.expectedTapeType
    mode   = params.mode
    TT = params.TT
    width = params.width
    abiwrap = params.abiwrap
    primal  = job.source
    modifiedBetween = params.modifiedBetween
    returnPrimal = params.returnPrimal

    if !(params.rt <: Const)
        @assert !isghostty(eltype(params.rt))
    end
    if parent_job === nothing
        primal_target = DefaultCompilerTarget()
        primal_params = PrimalCompilerParams(mode)
        primal_job    = CompilerJob(primal, CompilerConfig(primal_target, primal_params; kernel=false), job.world)
    else
        config2 = CompilerConfig(parent_job.config.target, parent_job.config.params; kernel=false, parent_job.config.entry_abi, parent_job.config.name, parent_job.config.always_inline)
        primal_job = CompilerJob(primal, config2, job.world) # TODO EnzymeInterp params, etc
    end


    mod, meta = GPUCompiler.codegen(:llvm, primal_job; optimize=false, toplevel=toplevel, cleanup=false, validate=false, parent_job=parent_job)
 
    prepare_llvm(mod, primal_job, meta)
    LLVM.ModulePassManager() do pm
        API.AddPreserveNVVMPass!(pm, #=Begin=#true)
        run!(pm, mod)
    end

    primalf = meta.entry
    check_ir(job, mod)

    disableFallback = String[]
    # Tablegen BLAS does not support runtime activity, nor forward mode yet
    if !API.runtimeActivity() && mode != API.DEM_ForwardMode
        blas_types = ("s", "d")
        blas_readonly = ("dot",)
        for ty in ("s", "d")
            for func in ("dot",)
                for prefix in ("cblas_")
                #for prefix in ("", "cblas_")
                    for ending in ("", "_", "64_", "_64_")
                        push!(disableFallback, prefix*ty*func*ending)
                    end
                end
            end
        end
    end
    if API.EnzymeBitcodeReplacement(mod, disableFallback) != 0
        ModulePassManager() do pm
            instruction_combining!(pm)
            run!(pm, mod)
        end
        toremove = []
        for f in functions(mod)
            if !any(map(k->kind(k)==kind(EnumAttribute("alwaysinline")), collect(function_attributes(f))))
                continue
            end
            if !any(map(k->kind(k)==kind(EnumAttribute("returns_twice")), collect(function_attributes(f))))
                push!(function_attributes(f), EnumAttribute("returns_twice"))
                push!(toremove, name(f))
            end
            todo = LLVM.CallInst[]
            for u in LLVM.uses(f)
                ci = LLVM.user(u)
                if isa(ci, LLVM.CallInst) && called_operand(ci) == f
                    push!(todo, ci)
                end
            end
            for ci in todo
                b = IRBuilder()
                position!(b, ci)
                args = collect(collect(operands(ci))[1:LLVM.API.LLVMGetNumArgOperands(ci)])
                nc = call!(b, LLVM.function_type(f), f, args)
                replace_uses!(ci, nc)
                LLVM.API.LLVMInstructionEraseFromParent(ci)
            end
        end

        for fname in ("cblas_xerbla",)
            if haskey(functions(mod), fname)
                f = functions(mod)[fname]
                if isempty(LLVM.blocks(f))
                    entry = BasicBlock(f, "entry")
                    b = IRBuilder()
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
                LLVM.API.LLVMRemoveEnumAttributeAtIndex(f, reinterpret(LLVM.API.LLVMAttributeIndex, LLVM.API.LLVMAttributeFunctionIndex), kind(EnumAttribute("returns_twice")))
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

    world = job.world
    interp = GPUCompiler.get_interpreter(job)
    method_table = Core.Compiler.method_table(interp)

    actualRetType = nothing
    lowerConvention = true
    customDerivativeNames = String[]
    for (mi, k) in meta.compiled
        k_name = GPUCompiler.safe_name(k.specfunc)
        has_custom_rule = false

        specTypes = Interpreter.simplify_kw(mi.specTypes)

        caller = mi
        if mode == API.DEM_ForwardMode
            has_custom_rule = EnzymeRules.has_frule_from_sig(specTypes; world, method_table, caller)
            if has_custom_rule
                @safe_debug "Found frule for" mi.specTypes
            end
        else
            has_custom_rule = EnzymeRules.has_rrule_from_sig(specTypes; world, method_table, caller)
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

        function handleCustom(name, attrs=[], setlink=true, noinl=true)
            attributes = function_attributes(llvmfn)
            custom[k_name] = linkage(llvmfn)
            if setlink
              linkage!(llvmfn, LLVM.API.LLVMExternalLinkage)
            end
            for a in attrs
                push!(attributes, a)
            end
            push!(attributes, StringAttribute("enzyme_math", name))
            if noinl
                push!(attributes, EnumAttribute("noinline", 0))
            end
            must_wrap |= llvmfn == primalf
            nothing
        end

        foundTys[k_name] = (LLVM.function_type(llvmfn), mi)
        if has_custom_rule
            handleCustom("enzyme_custom", [StringAttribute("enzyme_preserve_primal", "*")])
            continue
        end

        Base.isbindingresolved(jlmod, name) && isdefined(jlmod, name) || continue
        func = getfield(jlmod, name)

        sparam_vals = mi.specTypes.parameters[2:end] # mi.sparam_vals
        if func == Base.eps || func == Base.nextfloat || func == Base.prevfloat
            handleCustom("jl_inactive_inout", [StringAttribute("enzyme_inactive"),
                                      EnumAttribute("readnone", 0),
                                      EnumAttribute("speculatable", 0),
                                      StringAttribute("enzyme_shouldrecompute")
                                                      ])
            continue
        end
        if func == Base.to_tuple_type
            handleCustom("jl_to_tuple_type",
                   [EnumAttribute("readonly", 0),
                    EnumAttribute("inaccessiblememonly", 0),
                    EnumAttribute("speculatable", 0),
                    StringAttribute("enzyme_shouldrecompute"),
                    StringAttribute("enzyme_inactive"),
                                  ])
            continue
        end
        if func == Base.Threads.threadid || func == Base.Threads.nthreads
            name = (func == Base.Threads.threadid) ? "jl_threadid" : "jl_nthreads"
            handleCustom(name,
                   [EnumAttribute("readonly", 0),
                    EnumAttribute("inaccessiblememonly", 0),
                    EnumAttribute("speculatable", 0),
                    StringAttribute("enzyme_shouldrecompute"),
                    StringAttribute("enzyme_inactive"),
                                  ])
            continue
        end
        # Since this is noreturn and it can't write to any operations in the function
        # in a way accessible by the function. Ideally the attributor should actually
        # handle this and similar not impacting the read/write behavior of the calling
        # fn, but it doesn't presently so for now we will ensure this by hand
        if func == Base.Checked.throw_overflowerr_binaryop
            llvmfn = functions(mod)[k.specfunc]
            handleCustom("enz_noop", [StringAttribute("enzyme_inactive"), EnumAttribute("readonly")])
            continue
        end
        if EnzymeRules.is_inactive_from_sig(mi.specTypes; world, method_table, caller)
            handleCustom("enz_noop", [StringAttribute("enzyme_inactive"), StringAttribute("nofree")])
            continue
        end
        if EnzymeRules.is_inactive_noinl_from_sig(mi.specTypes; world, method_table, caller)
            handleCustom("enz_noop", [StringAttribute("enzyme_inactive"), StringAttribute("nofree")], false, false)
            for bb in blocks(llvmfn)
                for inst in instructions(bb)
                    if isa(inst, LLVM.CallInst)
                        LLVM.API.LLVMAddCallSiteAttribute(inst, LLVM.API.LLVMAttributeFunctionIndex, StringAttribute("enzyme_inactive"))
                    end
                end
            end
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
            push!(function_attributes(llvmfn), StringAttribute("enzymejl_mi", string(convert(UInt, pointer_from_objref(mi)))))
            k_name = LLVM.name(llvmfn)
            if primal
                primalf = llvmfn
                lowerConvention = false
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

        if name == :__fd_sincos_1 || name == :sincospi
          source_sig = Base.signature_type(func, sparam_vals)
          cur = llvmfn == primalf
          llvmfn, _ = lower_convention(source_sig, mod, llvmfn, k.ci.rettype)
          if cur
              primalf = llvmfn
              lowerConvention = false
          end
          k_name = LLVM.name(llvmfn)
        end

        name = string(name)
        name = T == Float32 ? name*"f" : name

        handleCustom(name, [EnumAttribute("readnone", 0),
                    StringAttribute("enzyme_shouldrecompute")])
    end

    @assert actualRetType !== nothing

    if must_wrap
        llvmfn = primalf
        FT = LLVM.function_type(llvmfn)

        wrapper_f = LLVM.Function(mod, safe_name(LLVM.name(llvmfn)*"mustwrap"), FT)

        let builder = IRBuilder()
            entry = BasicBlock(wrapper_f, "entry")
            position!(builder, entry)

            res = call!(builder, LLVM.function_type(llvmfn), llvmfn, collect(parameters(wrapper_f)))

            for idx in length(collect(parameters(llvmfn)))
                for attr in collect(parameter_attributes(llvmfn, idx))
                    if kind(attr) == kind(EnumAttribute("sret"))
                        LLVM.API.LLVMAddCallSiteAttribute(res, LLVM.API.LLVMAttributeIndex(idx), attr)
                    end
                end
            end

            if LLVM.return_type(FT) == LLVM.VoidType()
                ret!(builder)
            else
                ret!(builder, res)
            end

            dispose(builder)
        end
        attributes = function_attributes(wrapper_f)
        push!(attributes, StringAttribute("enzymejl_world", string(job.world)))
        mi, rt = enzyme_custom_extract_mi(primalf)
        push!(attributes, StringAttribute("enzymejl_mi", string(convert(UInt, pointer_from_objref(mi)))))
        push!(attributes, StringAttribute("enzymejl_rt", string(convert(UInt, unsafe_to_pointer(rt)))))
        primalf = wrapper_f
    end

    source_sig = job.source.specTypes
    primalf, returnRoots = lowerConvention ? lower_convention(source_sig, mod, primalf, actualRetType) : (primalf, false)
    push!(function_attributes(primalf), StringAttribute("enzymejl_world", string(job.world)))

    if primal_job.config.target isa GPUCompiler.NativeCompilerTarget
        target_machine = JIT.get_tm()
    else
        target_machine = GPUCompiler.llvm_machine(primal_job.config.target)
    end

    parallel = Threads.nthreads() > 1
    process_module = false
    device_module = false
    if parent_job !== nothing
        if parent_job.config.target isa GPUCompiler.PTXCompilerTarget ||
            parent_job.config.target isa GPUCompiler.GCNCompilerTarget ||
            parent_job.config.target isa GPUCompiler.MetalCompilerTarget
            parallel = true
            device_module = true
        end
        if parent_job.config.target isa GPUCompiler.GCNCompilerTarget ||
            parent_job.config.target isa GPUCompiler.MetalCompilerTarget                      
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
            push!(jlrules, fname)
        end

        adjointf, augmented_primalf, TapeType = enzyme!(job, mod, primalf, TT, mode, width, parallel, actualRetType, abiwrap, modifiedBetween, returnPrimal, jlrules, expectedTapeType)
        toremove = []
        # Inline the wrapper
        for f in functions(mod)
            if !any(map(k->kind(k)==kind(EnumAttribute("alwaysinline")), collect(function_attributes(f))))
                continue
            end
            if !any(map(k->kind(k)==kind(EnumAttribute("returns_twice")), collect(function_attributes(f))))
                push!(function_attributes(f), EnumAttribute("returns_twice"))
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
                LLVM.API.LLVMRemoveEnumAttributeAtIndex(f, reinterpret(LLVM.API.LLVMAttributeIndex, LLVM.API.LLVMAttributeFunctionIndex), kind(EnumAttribute("returns_twice")))
            end
        end
    else
        adjointf = primalf
        augmented_primalf = nothing
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
                if kind(at) == kind(EnumAttribute("noinline"))
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
        adjointf = GPUCompiler.finish_module!(parent_job, mod, adjointf)
    end

    push!(function_attributes(adjointf), EnumAttribute("alwaysinline", 0))
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

# Compiler result
struct CompileResult{AT, PT}
    adjoint::AT
    primal::PT
    TapeType::Type
end

@inline (thunk::CombinedAdjointThunk{PT, FA, RT, TT, Width, ReturnPrimal})(fn::FA, args...) where {PT, FA, Width, RT, TT, ReturnPrimal} =
enzyme_call(Val(false), thunk.adjoint, CombinedAdjointThunk, Width, ReturnPrimal, TT, RT, fn, Cvoid, args...)

@inline (thunk::ForwardModeThunk{PT, FA, RT, TT, Width, ReturnPrimal})(fn::FA, args...) where {PT, FA, Width, RT, TT, ReturnPrimal} =
enzyme_call(Val(false), thunk.adjoint, ForwardModeThunk, Width, ReturnPrimal, TT, RT, fn, Cvoid, args...)

@inline (thunk::AdjointThunk{PT, FA, RT, TT, Width, TapeT})(fn::FA, args...) where {PT, FA, Width, RT, TT, TapeT} =
enzyme_call(Val(false), thunk.adjoint, AdjointThunk, Width, #=ReturnPrimal=#Val(false), TT, RT, fn, TapeT, args...)
@inline raw_enzyme_call(thunk::AdjointThunk{PT, FA, RT, TT, Width, TapeT}, fn::FA, args...) where {PT, FA, Width, RT, TT, TapeT} =
enzyme_call(Val(true), thunk.adjoint, AdjointThunk, Width, #=ReturnPrimal=#Val(false), TT, RT, fn, TapeT, args...)

@inline (thunk::AugmentedForwardThunk{PT, FA, RT, TT, Width, ReturnPrimal, TapeT})(fn::FA, args...) where {PT, FA, Width, RT, TT, ReturnPrimal, TapeT} =
enzyme_call(Val(false), thunk.primal, AugmentedForwardThunk, Width, ReturnPrimal, TT, RT, fn, TapeT, args...)
@inline raw_enzyme_call(thunk::AugmentedForwardThunk{PT, FA, RT, TT, Width, ReturnPrimal, TapeT}, fn::FA, args...) where {PT, FA, Width, RT, TT, ReturnPrimal, TapeT} =
enzyme_call(Val(true), thunk.primal, AugmentedForwardThunk, Width, ReturnPrimal, TT, RT, fn, TapeT, args...)


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

@generated function enzyme_call(::Val{RawCall}, fptr::PT, ::Type{CC}, ::Type{Val{width}}, ::Val{returnPrimal}, tt::Type{T},
        rt::Type{RT}, fn::FA, ::Type{TapeType}, args::Vararg{Any, N}) where {RawCall, PT, FA, T, RT, TapeType, N, CC, width, returnPrimal}

    JuliaContext() do ctx
        F = eltype(FA)
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
            error("Function to differentiate is guaranteed to return an error and doesn't make sense to autodiff. Giving up")
        end
        if !(rettype <: Const) && (isghostty(eltype(rettype)) || Core.Compiler.isconstType(eltype(rettype)) || eltype(rettype) === DataType)
            rrt = eltype(rettype)
            error("Return type `$rrt` not marked Const, but is ghost or const type.")
        end

        sret_types  = []  # Julia types of all returned variables
        # By ref values we create and need to preserve
        ccexprs = Union{Expr, Symbol}[] # The expressions passed to the `llvmcall`

        if !isghostty(F) && !Core.Compiler.isconstType(F)
            isboxed = GPUCompiler.deserves_argbox(F)
            argexpr = :(fn.val)

            if isboxed
                push!(types, Any)
            else
                push!(types, F)
            end

            push!(ccexprs, argexpr)
            if !(FA <: Const)
                argexpr = :(fn.dval)
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
            if isghostty(source_typ) || Core.Compiler.isconstType(source_typ)
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

            if T <: Const || T <: BatchDuplicatedFunc
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
            j_drT = if width == 1
                jlRT
            else
                NTuple{width, jlRT}
            end
            push!(types, j_drT)
            push!(ccexprs, argexprs[i])
            i+=1
        end

        if needs_tape
            if !(isghostty(TapeType) || Core.Compiler.isconstType(TapeType))
                push!(types, TapeType)
                push!(ccexprs, argexprs[i])
            end
            i+=1
        end


        if is_adjoint
            NT = Tuple{ActiveRetTypes...}
            if any(any_jltypes(convert(LLVM.LLVMType, b; allow_boxed=true)) for b in ActiveRetTypes)
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
            if !returnPrimal && CC <: AugmentedForwardThunk
                push!(sret_types, Nothing)
            end
            if rettype <: Duplicated || rettype <: DuplicatedNoNeed
                push!(sret_types, jlRT)
            elseif rettype <: BatchDuplicated || rettype <: BatchDuplicatedNoNeed
                push!(sret_types, AnonymousStruct(NTuple{width, jlRT}))
            elseif CC <: AugmentedForwardThunk
                push!(sret_types, Nothing)
            elseif rettype <: Const
            else
                @show rettype, CC
                @assert false
            end
        end

        # calls fptr
        llvmtys = LLVMType[convert(LLVMType, x; allow_boxed=true) for x in types]

        T_void = convert(LLVMType, Nothing)

        combinedReturn = Tuple{sret_types...}
        if any(any_jltypes(convert(LLVM.LLVMType, T; allow_boxed=true)) for T in sret_types)
            combinedReturn = AnonymousStruct(combinedReturn)
        end
        uses_sret = is_sret(combinedReturn)
        jltype = convert(LLVM.LLVMType, combinedReturn)

        T_jlvalue = LLVM.StructType(LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

        returnRoots = false
        if uses_sret
            returnRoots = deserves_rooting(jltype)
        end

        if !(GPUCompiler.isghosttype(PT) || Core.Compiler.isconstType(PT))
            pushfirst!(llvmtys, convert(LLVMType, PT))
        end

        T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

        T_ret = jltype
        # if returnRoots
        #     T_ret = T_prjlvalue
        # end
        llvm_f, _ = LLVM.Interop.create_function(T_ret, llvmtys)
        push!(function_attributes(llvm_f), EnumAttribute("alwaysinline", 0))

        mod = LLVM.parent(llvm_f)
        i64 = LLVM.IntType(64)
        LLVM.IRBuilder() do builder
            entry = BasicBlock(llvm_f, "entry")
            position!(builder, entry)
            callparams = collect(LLVM.Value, parameters(llvm_f))

            if !(GPUCompiler.isghosttype(PT) || Core.Compiler.isconstType(PT))
                lfn = callparams[1]
                deleteat!(callparams, 1)
            end

            if returnRoots
                tracked = CountTrackedPointers(jltype)
                pushfirst!(callparams, alloca!(builder, LLVM.ArrayType(T_prjlvalue, tracked.count)))
                pushfirst!(callparams, alloca!(builder, jltype))
            end

            if needs_tape && !(isghostty(TapeType) || Core.Compiler.isconstType(TapeType))
                tape = callparams[end]
                if TapeType <: EnzymeTapeToLoad
                    llty = from_tape_type(eltype(TapeType))
                    tape = bitcast!(builder, LLVM.PointerType(llty, LLVM.addrspace(value_type(tape))))
                    tape = load!(builder, llty, tape)
                    API.SetMustCache!(tape)
                    callparams[end] = tape
                else
                    llty = from_tape_type(TapeType)
                    @assert value_type(tape) == llty
                end
            end

            if !(GPUCompiler.isghosttype(PT) || Core.Compiler.isconstType(PT))
                FT = LLVM.FunctionType(returnRoots ? T_void : T_ret, [value_type(x) for x in callparams])
                lfn = inttoptr!(builder, lfn, LLVM.PointerType(FT))
            else
                val_inner(::Type{Val{V}}) where V = V
                submod, subname = val_inner(PT)
                # TODO, consider optimization
                # However, julia will optimize after this, so no need
                submod = parse(LLVM.Module, String(submod))
                LLVM.link!(mod, submod)
                lfn = functions(mod)[String(subname)]
                FT = LLVM.function_type(lfn)
            end

            r = call!(builder, FT, lfn, callparams)
            
            if returnRoots
                attr = if LLVM.version().major >= 12
                    TypeAttribute("sret", jltype)
                else
                    EnumAttribute("sret")
                end
                LLVM.API.LLVMAddCallSiteAttribute(r, LLVM.API.LLVMAttributeIndex(1), attr)
                r = load!(builder, eltype(value_type(callparams[1])), callparams[1])
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

        if !(GPUCompiler.isghosttype(PT) || Core.Compiler.isconstType(PT))
            return quote
                Base.@_inline_meta
                Base.llvmcall(($ir, $fn), $combinedReturn,
                        Tuple{$PT, $(types...)},
                        fptr, $(ccexprs...))
            end
        else
            return quote
                Base.@_inline_meta
                Base.llvmcall(($ir, $fn), $combinedReturn,
                        Tuple{$(types...)},
                        $(ccexprs...))
            end
        end
    end
end

##
# JIT
##

function _link(job, (mod, adjoint_name, primal_name, TapeType))
    # if job.config.params.ABI <: InlineABI
    #     return CompileResult(Val((Symbol(mod), Symbol(adjoint_name))), Val((Symbol(mod), Symbol(primal_name))), TapeType)
    # end

    # Now invoke the JIT
    jitted_mod = JIT.add!(mod)
    #if VERSION >= v"1.9.0-DEV.115"
    #    LLVM.dispose(ctx)
    #else
    #    # we cannot dispose of the global unique context
    #end
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

    return CompileResult(adjoint_ptr, primal_ptr, TapeType)
end

# actual compilation
function _thunk(job, postopt::Bool=true)
    mod, meta = codegen(:llvm, job; optimize=false)
    adjointf, augmented_primalf = meta.adjointf, meta.augmented_primalf

    adjoint_name = name(adjointf)

    if augmented_primalf !== nothing
        primal_name = name(augmented_primalf)
    else
        primal_name = nothing
    end
 
    LLVM.ModulePassManager() do pm
        add!(pm, FunctionPass("ReinsertGCMarker", reinsert_gcmarker_pass!))
        run!(pm, mod)
    end
    
    # Run post optimization pipeline
    if postopt && job.config.params.ABI <: FFIABI
        post_optimze!(mod, JIT.get_tm())
    end
    return (mod, adjoint_name, primal_name, meta.TapeType)
end

const cache = Dict{UInt, CompileResult}()

const cache_lock = ReentrantLock()
@inline function cached_compilation(@nospecialize(job::CompilerJob))::CompileResult
    key = hash(job)

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

@inline remove_innerty(::Type{<:Const}) = Const
@inline remove_innerty(::Type{<:Active}) = Active
@inline remove_innerty(::Type{<:Duplicated}) = Duplicated
@inline remove_innerty(::Type{<:DuplicatedNoNeed}) = DuplicatedNoNeed
@inline remove_innerty(::Type{<:BatchDuplicated}) = Duplicated
@inline remove_innerty(::Type{<:BatchDuplicatedNoNeed}) = DuplicatedNoNeed

@generated function thunk(::Val{World}, ::Type{FA}, ::Type{A}, tt::Type{TT},::Val{Mode}, ::Val{width}, ::Val{ModifiedBetween}, ::Val{ReturnPrimal}, ::Val{ShadowInit}, ::Type{ABI}) where {FA<:Annotation, A<:Annotation, TT, Mode, ModifiedBetween, width, ReturnPrimal, ShadowInit, World, ABI}   
    JuliaContext() do ctx
        mi = fspec(eltype(FA), TT, World)

        target = Compiler.EnzymeTarget()
        params = Compiler.EnzymeCompilerParams(Tuple{FA, TT.parameters...}, Mode, width, remove_innerty(A), true, #=abiwrap=#true, ModifiedBetween, ReturnPrimal, ShadowInit, UnknownTapeType, ABI)
        job    = Compiler.CompilerJob(mi, CompilerConfig(target, params; kernel=false), World)

        sig = Tuple{eltype(FA), map(eltype, TT.parameters)...}

        interp = GPUCompiler.get_interpreter(job)

        # TODO check compile return here, early
        # rrt = Core.Compiler.return_type(f, primal.tt) # nothing
        rrt = something(Core.Compiler.typeinf_type(interp, mi.def, mi.specTypes, mi.sparam_vals), Any)

        if rrt == Union{}
            error("Function to differentiate is guaranteed to return an error and doesn't make sense to autodiff. Giving up")
        end
        
        if !(A <: Const) && (isghostty(rrt) || Core.Compiler.isconstType(rrt) || rrt === DataType)
            error("Return type `$rrt` not marked Const, but is ghost or const type.")
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


        compile_result = cached_compilation(job)
        if Mode == API.DEM_ReverseModePrimal || Mode == API.DEM_ReverseModeGradient
            TapeType = compile_result.TapeType
            AugT = AugmentedForwardThunk{typeof(compile_result.primal), FA, rt, Tuple{params.TT.parameters[2:end]...}, Val{width}, Val(ReturnPrimal), TapeType}
            AdjT = AdjointThunk{typeof(compile_result.adjoint), FA, rt, Tuple{params.TT.parameters[2:end]...}, Val{width}, TapeType}
            return quote
                augmented = $AugT($(compile_result.primal))
                adjoint  = $AdjT($(compile_result.adjoint))
                (augmented, adjoint)
            end
        elseif Mode == API.DEM_ReverseModeCombined
            CAdjT = CombinedAdjointThunk{typeof(compile_result.adjoint), FA, rt, Tuple{params.TT.parameters[2:end]...}, Val{width}, Val(ReturnPrimal)}
            return quote
                $CAdjT($(compile_result.adjoint))
            end
        elseif Mode == API.DEM_ForwardMode
            FMT = ForwardModeThunk{typeof(compile_result.adjoint), FA, rt, Tuple{params.TT.parameters[2:end]...}, Val{width}, Val(ReturnPrimal)}
            return quote
                $FMT($(compile_result.adjoint))
            end
        else
            @assert false
        end
    end
end

import GPUCompiler: deferred_codegen_jobs

@generated function deferred_codegen(::Val{World}, ::Type{FA}, ::Val{tt}, ::Val{rt},::Val{Mode},
        ::Val{width}, ::Val{ModifiedBetween}, ::Val{ReturnPrimal}=Val(false),::Val{ShadowInit}=Val(false),::Type{ExpectedTapeType}=UnknownTapeType) where {World, FA<:Annotation,tt, rt, Mode, width, ModifiedBetween, ReturnPrimal, ShadowInit,ExpectedTapeType}
    JuliaContext() do ctx

        mi = fspec(eltype(FA), tt, World)
        target = EnzymeTarget()
        params = EnzymeCompilerParams(Tuple{FA, tt.parameters...}, Mode, width, remove_innerty(rt), true, #=abiwrap=#true, ModifiedBetween, ReturnPrimal, ShadowInit,ExpectedTapeType, FFIABI)
        job    = Compiler.CompilerJob(mi, CompilerConfig(target, params; kernel=false), World)

        adjoint_addr, primal_addr = get_trampoline(job)
        adjoint_id = Base.reinterpret(Int, pointer(adjoint_addr))
        deferred_codegen_jobs[adjoint_id] = job

        if primal_addr !== nothing
            primal_id = Base.reinterpret(Int, pointer(primal_addr))
            deferred_codegen_jobs[primal_id] = job
        else
            primal_id = 0
        end

        quote
            adjoint = ccall("extern deferred_codegen", llvmcall, Ptr{Cvoid}, (Ptr{Cvoid},), $(reinterpret(Ptr{Cvoid}, adjoint_id)))
            primal = if $(primal_addr !== nothing)
                ccall("extern deferred_codegen", llvmcall, Ptr{Cvoid}, (Ptr{Cvoid},), $(reinterpret(Ptr{Cvoid}, primal_id)))
            else
                nothing
            end
            adjoint, primal
        end
    end
end

include("compiler/reflection.jl")

end
