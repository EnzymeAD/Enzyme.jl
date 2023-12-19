module Compiler

import ..Enzyme
import Enzyme: Const, Active, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed,
               BatchDuplicatedFunc,
               Annotation, guess_activity, eltype,
               API, TypeTree, typetree, only!, shift!, data0!, merge!, to_md,
               TypeAnalysis, FnTypeInfo, Logic, allocatedinline, ismutabletype
using Enzyme

import EnzymeCore
import EnzymeCore: EnzymeRules, ABI, FFIABI, DefaultABI

using LLVM, GPUCompiler, Libdl
import Enzyme_jll

import GPUCompiler: CompilerJob, codegen, safe_name
using LLVM.Interop
import LLVM: Target, TargetMachine

using Printf

using Preferences

bitcode_replacement() = parse(Bool, @load_preference("bitcode_replacement", "true"))
bitcode_replacement!(val) = @set_preferences!("bitcode_replacement" => string(val))

function cpu_name()
    ccall(:jl_get_cpu_name, String, ())
end

function cpu_features()
    if VERSION >= v"1.10.0-beta1"
        return ccall(:jl_get_cpu_features, String, ())
    end

    @static if Sys.ARCH == :x86_64 ||
               Sys.ARCH == :x86
        return "+mmx,+sse,+sse2,+fxsr,+cx8" # mandated by Julia
    else
        return ""
    end
end

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
    "ijl_array_ptr_copy", "jl_array_ptr_copy",
    "ijl_array_copy", "jl_array_copy",
    "ijl_get_nth_field_checked", "ijl_get_nth_field_checked",
    "jl_array_del_end","ijl_array_del_end",
    "jl_get_world_counter", "ijl_get_world_counter",
    "memhash32_seed", "memhash_seed",
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
    "ijl_subtype",
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
    "pcre2_jit_compile_8",
    "memmove",
))

const inactivefns = Set{String}((
    "jl_get_world_counter", "ijl_get_world_counter",
    "memhash32_seed", "memhash_seed",
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
    "ijl_subtype",
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

@enum ActivityState begin
    AnyState = 0
    ActiveState = 1
    DupState = 2
    MixedState = 3
end

@inline function Base.:|(a1::ActivityState, a2::ActivityState)
    ActivityState(Int(a1) | Int(a2))
end

struct Merger{seen,worldT,justActive,UnionSret}
    world::worldT
end

@inline element(::Val{T}) where T = T

@inline function (c::Merger{seen,worldT,justActive,UnionSret})(f::Int) where {seen,worldT,justActive,UnionSret}
    T = element(first(seen))

    reftype = ismutabletype(T) || T isa UnionAll

    if justActive && reftype
        return Val(AnyState)
    end

    subT = fieldtype(T, f)

    if justActive && !allocatedinline(subT)
        return Val(AnyState)
    end

    sub = active_reg_inner(subT, seen, c.world, Val(justActive), Val(UnionSret))

    if sub == AnyState
        Val(AnyState)
    else
        if sub == DupState
            if justActive
                Val(AnyState)
            else
                Val(DupState)
            end
        else
            if reftype
                Val(DupState)
            else
                Val(sub)
            end
        end
    end
end

@inline forcefold(::Val{RT}) where RT = RT

@inline function forcefold(::Val{ty}, ::Val{sty}, C::Vararg{Any, N}) where {ty, sty, N}
    if sty == AnyState || sty == ty
        return forcefold(Val(ty), C...)
    end
    if ty == AnyState
        return forcefold(Val(sty), C...)
    else
        return MixedState
    end
end

@inline ptreltype(::Type{Ptr{T}}) where T = T
@inline ptreltype(::Type{Core.LLVMPtr{T,N}}) where {T,N} = T
@inline ptreltype(::Type{Core.LLVMPtr{T} where N}) where {T} = T
@inline ptreltype(::Type{Base.RefValue{T}}) where T = T
@inline ptreltype(::Type{Array{T,N}}) where {T,N} = T
@inline ptreltype(::Type{Array{T, N} where N}) where {T} = T
@inline ptreltype(::Type{Complex{T}}) where T = T
@inline ptreltype(::Type{Tuple{Vararg{T}}}) where T = T
@inline ptreltype(::Type{IdDict{K, V}}) where {K, V} = V
@inline ptreltype(::Type{IdDict{K, V} where K}) where {V} = V

@inline is_arrayorvararg_ty(::Type) = false
@inline is_arrayorvararg_ty(::Type{Array{T,N}}) where {T,N} = true
@inline is_arrayorvararg_ty(::Type{Array{T, N} where N}) where {T} = true
@inline is_arrayorvararg_ty(::Type{Tuple{Vararg{T2}}}) where T2 = true
@inline is_arrayorvararg_ty(::Type{Ptr{T}}) where T = true
@inline is_arrayorvararg_ty(::Type{Core.LLVMPtr{T,N}}) where {T,N} = true
@inline is_arrayorvararg_ty(::Type{Core.LLVMPtr{T,N} where N}) where {T} = true
@inline is_arrayorvararg_ty(::Type{Base.RefValue{T}}) where T = true
@inline is_arrayorvararg_ty(::Type{IdDict{K, V}}) where {K, V} = true
@inline is_arrayorvararg_ty(::Type{IdDict{K, V} where K}) where {V} = true

@inline function datatype_fieldcount(t::Type{T}) where T
    @static if VERSION < v"1.10.0"
        NT = @static if VERSION < v"1.9.0"
            Base.NamedTuple_typename
        else
            Base._NAMEDTUPLE_NAME
        end
        if t.name === NT
            names, types = t.parameters[1], t.parameters[2]
            if names isa Tuple
                return length(names)
            end
            if types isa DataType && types <: Tuple
                return datatype_fieldcount(types)
            end
            return nothing
        else
            @static if VERSION < v"1.7.0"
                if t.abstract || (t.name === Tuple.name && Base.isvatuple(t))
                    return nothing
                end
            else
                if isabstracttype(t) || (t.name === Tuple.name && Base.isvatuple(t))
                    return nothing
                end
            end
        end
        if isdefined(t, :types)
            return length(t.types)
        end
        return length(t.name.names)
    else
        return Base.datatype_fieldcount(t)
    end
end

@inline function active_reg_inner(::Type{T}, seen::ST, world::Union{Nothing, UInt}, ::Val{justActive}=Val(false), ::Val{UnionSret}=Val(false))::ActivityState where {ST,T, justActive, UnionSret}

    if T === Any
        return DupState
    end

    if T === Union{}
        return AnyState
    end

    if T <: Complex
        return active_reg_inner(ptreltype(T), seen, world, Val(justActive), Val(UnionSret))
    end

    if T <: AbstractFloat
        return ActiveState
    end

    if T <: Ptr || T <: Core.LLVMPtr || T <: Base.RefValue || T <: Array || is_arrayorvararg_ty(T)
        if justActive
            return AnyState
        end

        if is_arrayorvararg_ty(T) && active_reg_inner(ptreltype(T), seen, world, Val(justActive), Val(UnionSret)) == AnyState
            return AnyState
        else
            return DupState
        end
    end

    if T <: Integer
        return AnyState
    end

    if isghostty(T) || Core.Compiler.isconstType(T)
        return AnyState
    end

    inactivety = if typeof(world) === Nothing
        EnzymeCore.EnzymeRules.inactive_type(T)
    else
        inmi = GPUCompiler.methodinstance(typeof(EnzymeCore.EnzymeRules.inactive_type), Tuple{Type{T}}, world)
        args = Any[EnzymeCore.EnzymeRules.inactive_type, T];
        ccall(:jl_invoke, Any, (Any, Ptr{Any}, Cuint, Any), EnzymeCore.EnzymeRules.inactive_type, args, length(args), inmi)
    end

    if inactivety
        return AnyState
    end

    # unknown number of fields
    if T isa UnionAll
        aT = Base.argument_datatype(T)
        if aT === nothing
            return DupState
        end
        if datatype_fieldcount(aT) === nothing
            return DupState
        end
    end

    if T isa Union
        # if sret union, the data is stored in a stack memory location and is therefore
        # not unique'd preventing the boxing of the union in the default case
        if UnionSret && is_sret_union(T)
            @inline function recur(::Type{ST}) where ST
                if ST isa Union
                    return forcefold(Val(recur(ST.a)), Val(recur(ST.b)))
                end
                return active_reg_inner(ST, seen, world, Val(justActive), Val(UnionSret))
            end
            return recur(T)
        else
            if justActive
                return AnyState
            end
            if active_reg_inner(T.a, seen, world, Val(justActive), Val(UnionSret)) != AnyState
                return DupState
            end
            if active_reg_inner(T.b, seen, world, Val(justActive), Val(UnionSret)) != AnyState
                return DupState
            end
        end
        return AnyState
    end

    # if abstract it must be by reference
    if Base.isabstracttype(T)
        return DupState
    end

    if ismutabletype(T)
        # if just looking for active of not
        # we know for a fact this isn't active
        if justActive
            return AnyState
        end
    end

    @inline is_concrete_tuple(x::T2) where T2 = (x <: Tuple) && !(x === Tuple) && !(x isa UnionAll)

    @assert !Base.isabstracttype(T)
    if !(Base.isconcretetype(T) || is_concrete_tuple(T) || T isa UnionAll)
        throw(AssertionError("Type $T is not concrete type or concrete tuple"))
    end

    @static if VERSION < v"1.7.0"
        nT = T
    else
        nT = if is_concrete_tuple(T) && any(T2 isa Core.TypeofVararg for T2 in T.parameters)
            Tuple{((T2 isa Core.TypeofVararg ? Any : T2) for T2 in T.parameters)...,}
        else
            T
        end
    end

    if Val(nT) ∈ seen
        return MixedState
    end

    seen = (Val(nT), seen...)

    fty = Merger{seen,typeof(world),justActive, UnionSret}(world)

    ty = forcefold(Val(AnyState), ntuple(fty, Val(fieldcount(nT)))...)

    return ty
end

@inline @generated function active_reg_nothrow(::Type{T}, ::Val{world}) where {T, world}
    return active_reg_inner(T, (), world)
end

@inline function active_reg(::Type{T}, world::Union{Nothing, UInt}=nothing)::Bool where {T}
    seen = ()

    # check if it could contain an active
    if active_reg_inner(T, seen, world, #=justActive=#Val(true)) == ActiveState
        state = active_reg_inner(T, seen, world, #=justActive=#Val(false))
        if state == ActiveState
            return true
        end
        @assert state == MixedState
        throw(AssertionError(string(T)*" has mixed internal activity types"))
    else
        return false
    end
end

@inline function guaranteed_const(::Type{T}) where T
    rt = active_reg_nothrow(T, Val(nothing))
    res = rt == AnyState
    return res
end

@inline function guaranteed_const_nongen(::Type{T}, world) where T
    rt = active_reg_inner(T, (), world)
    res = rt == AnyState
    return res
end

Enzyme.guess_activity(::Type{T}, mode::Enzyme.Mode) where {T} = guess_activity(T, convert(API.CDerivativeMode, mode))

@inline function Enzyme.guess_activity(::Type{T}, Mode::API.CDerivativeMode) where {T}
    ActReg = active_reg_inner(T, (), nothing)
    if ActReg == AnyState
        return Const{T}
    end
    if Mode == API.DEM_ForwardMode
        return DuplicatedNoNeed{T}
    else
        if ActReg == ActiveState
            return Active{T}
        else
            return Duplicated{T}
        end
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

@inline EnzymeRules.tape_type(::Type{AugmentedForwardThunk{PT, FA, RT, TT, Width, ReturnPrimal, TapeType}}) where {PT, FA, RT, TT, Width, ReturnPrimal, TapeType} = TapeType
@inline EnzymeRules.tape_type(::AugmentedForwardThunk{PT, FA, RT, TT, Width, ReturnPrimal, TapeType}) where {PT, FA, RT, TT, Width, ReturnPrimal, TapeType} = TapeType
@inline EnzymeRules.tape_type(::Type{AdjointThunk{PT, FA, RT, TT, Width, TapeType}}) where {PT, FA, RT, TT, Width, TapeType} = TapeType
@inline EnzymeRules.tape_type(::AdjointThunk{PT, FA, RT, TT, Width, TapeType}) where {PT, FA, RT, TT, Width, TapeType} = TapeType

using .JIT


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

function emit_jl!(B::LLVM.IRBuilder, val::LLVM.Value)::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue])
    fn, _ = get_function!(mod, "jl_", FT)
    call!(B, FT, fn, [val])
end

function emit_box_int32!(B::LLVM.IRBuilder, val::LLVM.Value)::LLVM.Value
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

function emit_box_int64!(B::LLVM.IRBuilder, val::LLVM.Value)::LLVM.Value
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

function emit_apply_generic!(B::LLVM.IRBuilder, args)::LLVM.Value
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

function emit_invoke!(B::LLVM.IRBuilder, args)::LLVM.Value
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

include("absint.jl")

function emit_apply_type!(B::LLVM.IRBuilder, Ty, args)::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    legal = true
    found = []
    for arg in args
        slegal , foundv = absint(arg)
        if slegal
            push!(found, foundv)
        else
            legal = false
            break
        end
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

function emit_tuple!(B, args)::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    legal = true
    found = []
    for arg in args
        slegal , foundv = absint(arg)
        if slegal
            push!(found, foundv)
        else
            legal = false
            break
        end
    end

    if legal
        return unsafe_to_llvm((found...,))
    end

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_pprjlvalue = LLVM.PointerType(T_prjlvalue)
    T_int32 = LLVM.Int32Type()

    generic_FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_pprjlvalue, T_int32])
    f_apply_type, _ = get_function!(mod, "jl_f_tuple", generic_FT)

    @static if VERSION < v"1.9.0-"
        FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_prjlvalue]; vararg=true)
        f_apply_type = bitcast!(B, f_apply_type, LLVM.PointerType(FT))
        # call cc37 nonnull {}* bitcast ({}* ({}*, {}**, i32)* @jl_f_apply_type to {}* ({}*, {}*, {}*, {}*)*)({}* null, {}* inttoptr (i64 140150176657296 to {}*), {}* %4, {}* inttoptr (i64 140149987564368 to {}*))
        tag = call!(B, FT, f_apply_type, LLVM.Value[LLVM.PointerNull(T_prjlvalue), args...])
        LLVM.callconv!(tag, 37)
    else
        # %5 = call nonnull {}* ({}* ({}*, {}**, i32)*, {}*, ...) @julia.call({}* ({}*, {}**, i32)* @jl_f_apply_type, {}* null, {}* inttoptr (i64 139640605802128 to {}*), {}* %4, {}* inttoptr (i64 139640590432896 to {}*))
        julia_call, FT = get_function!(mod, "julia.call",
            LLVM.FunctionType(T_prjlvalue,
                              [LLVM.PointerType(generic_FT), T_prjlvalue]; vararg=true))
        tag = call!(B, FT, julia_call, LLVM.Value[f_apply_type, LLVM.PointerNull(T_prjlvalue), args...])
    end
    return tag
end

function emit_jltypeof!(B::LLVM.IRBuilder, arg::LLVM.Value)::LLVM.Value
    legal, val = abs_typeof(arg)
    if legal
        return unsafe_to_llvm(val)
    end

    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue]; vararg=true)
    fn, _ = get_function!(mod, "jl_typeof", FT)
    call!(B, FT, fn, [arg])
end

function emit_methodinstance!(B::LLVM.IRBuilder, func, args)::LLVM.Value
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
    if isa(array, LLVM.CallInst)
        fn = LLVM.called_operand(array)
        nm = ""
        if isa(fn, LLVM.Function)
            nm = LLVM.name(fn)
        end

        for (fname, num) in (
                             ("jl_alloc_array_1d", 1), ("ijl_alloc_array_1d", 1),
                             ("jl_alloc_array_2d", 2), ("jl_alloc_array_2d", 2),
                             ("jl_alloc_array_2d", 3), ("jl_alloc_array_2d", 3),
                             )
            if nm == fname
                res = operands(array)[2]
                for i in 2:num
                    res = mul!(B, res, operands(array)[1+i])
                end
                return res
            end
        end
    end
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

@inline function EnzymeCore.make_zero(::Type{RT}, seen::IdDict, prev::RT, ::Val{copy_if_inactive}=Val(false))::RT where {copy_if_inactive, RT<:AbstractFloat}
    return RT(0)
end

@inline function EnzymeCore.make_zero(::Type{Complex{RT}}, seen::IdDict, prev::Complex{RT}, ::Val{copy_if_inactive}=Val(false))::Complex{RT} where {copy_if_inactive, RT<:AbstractFloat}
    return RT(0)
end

@inline function EnzymeCore.make_zero(::Type{RT}, seen::IdDict, prev::RT, ::Val{copy_if_inactive}=Val(false))::RT where {copy_if_inactive, RT<:Array}
    if haskey(seen, prev)
        return seen[prev]
    end
    newa = RT(undef, size(prev))
    seen[prev] = newa
    for I in eachindex(prev)
        if isassigned(prev, I)
            pv = prev[I]
            innerty = Core.Typeof(pv)
            @inbounds newa[I] = EnzymeCore.make_zero(innerty, seen, pv, Val(copy_if_inactive))
        end
    end
    return newa
end

@inline function EnzymeCore.make_zero(::Type{RT}, seen::IdDict, prev::RT, ::Val{copy_if_inactive}=Val(false))::RT where {copy_if_inactive, RT<:Tuple}
    return ((EnzymeCore.make_zero(a, seen, prev[i], Val(copy_if_inactive)) for (i, a) in enumerate(RT.parameters))...,)
end


@inline function EnzymeCore.make_zero(::Type{NamedTuple{A,RT}}, seen::IdDict, prev::NamedTuple{A,RT}, ::Val{copy_if_inactive}=Val(false))::NamedTuple{A,RT} where {copy_if_inactive, A,RT}
    return NamedTuple{A,RT}(EnzymeCore.make_zero(RT, seen, RT(prev), Val(copy_if_inactive)))
end

@inline function EnzymeCore.make_zero(::Type{Core.Box}, seen::IdDict, prev::Core.Box, ::Val{copy_if_inactive}=Val(false)) where {copy_if_inactive, RT}
    if haskey(seen, prev)
        return seen[prev]
    end
    prev2 = prev.contents
    res = Core.Box(Base.Ref(EnzymeCore.make_zero(Core.Typeof(prev2), seen, prev2, Val(copy_if_inactive))))
    seen[prev] = res
    return res
end

@inline function EnzymeCore.make_zero(::Type{RT}, seen::IdDict, prev::RT, ::Val{copy_if_inactive}=Val(false))::RT where {copy_if_inactive, RT}
    if guaranteed_const_nongen(RT, nothing)
        return copy_if_inactive ? Base.deepcopy_internal(prev, seen) : prev
    end
    if haskey(seen, prev)
        return seen[prev]
    end
    @assert !Base.isabstracttype(RT)
    @assert Base.isconcretetype(RT)
    nf = fieldcount(RT)
    
    if ismutable(prev)
        y = ccall(:jl_new_struct_uninit, Any, (Any,), RT)
        seen[prev] = y
        for i in 1:nf
            if isdefined(prev, i)
                xi = getfield(prev, i)
                T = Core.Typeof(xi)
                xi = EnzymeCore.make_zero(T, seen, xi, Val(copy_if_inactive))
                ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), y, i-1, xi)
            end
        end
        return y
    end
    
    if nf == 0
        return prev
    end

    flds = Vector{Any}(undef, nf)
    for i in 1:nf
        if isdefined(prev, i)
            xi = getfield(prev, i)
            xi = EnzymeCore.make_zero(Core.Typeof(xi), seen, xi, Val(copy_if_inactive))
            flds[i] = xi
        else
            nf = i - 1 # rest of tail must be undefined values
            break
        end
    end
    y = ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), RT, flds, nf)
    seen[prev] = y
    return y
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
    ct = call!(B, funcT, func, LLVM.Value[globalstring_ptr!(B, string)])
    LLVM.API.LLVMAddCallSiteAttribute(ct, LLVM.API.LLVMAttributeFunctionIndex, EnumAttribute("noreturn"))
    return ct
    # FIXME(@wsmoses): Allow for emission of new BB in this code path
    # unreachable!(B)

    # 3. Change insertion point so that we don't stumble later
    # after_error = BasicBlock(fn, "after_error"; ctx)
    # position!(B, after_error)
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
    if length(ece.sval) != 0
        print(io, "\n Inverted pointers: \n")
        write(io, ece.sval)
    end
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
        sval = ""
        if isa(val, LLVM.Argument)
            fn = parent_scope(val)
            ir = string(LLVM.name(fn))*string(function_type(fn))
        else
            ip = API.EnzymeGradientUtilsInvertedPointersToString(data)
            sval = Base.unsafe_string(ip)
            API.EnzymeStringFree(ip)
        end
        throw(NoShadowException(msg, sval, ir, bt))
    elseif errtype == API.ET_IllegalTypeAnalysis
        data = API.EnzymeTypeAnalyzerRef(data)
        ip = API.EnzymeTypeAnalyzerToString(data)
        sval = Base.unsafe_string(ip)
        API.EnzymeStringFree(ip)
        
        if isa(val, LLVM.Instruction)
            mi, rt = enzyme_custom_extract_mi(LLVM.parent(LLVM.parent(val))::LLVM.Function, #=error=#false)
            if mi !== nothing
                msg *= "\n" * string(mi) * "\n"
            end
        end
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
            if !occursin("Cannot deduce single type of store", msg)
                if ir !== nothing
                    print(io, "Current scope: \n")
                    print(io, ir)
                end
                print(io, "\n Type analysis state: \n")
                write(io, sval)
            end
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
            println(io, string(LLVM.parent(LLVM.parent(data2))))
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

            legal, TT = abs_typeof(cur, true)
            if legal
                world = enzyme_extract_world(LLVM.parent(position(IRBuilder(B))))
                if guaranteed_const_nongen(TT, world)
                    continue
                end
                legal2, obj = absint(cur)
                badval = if legal2
                    string(obj)*" of type"*" "*string(TT)
                else
                    "Unknown object of type"*" "*string(TT)
                end
                illegal = true
                break
            end
            if isa(cur, LLVM.PointerNull)
                continue
            end
            if isa(cur, LLVM.UndefValue)
                continue
            end
            @static if LLVM.version() >= v"12"
            if isa(cur, LLVM.PoisonValue)
                continue
            end
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
                # if storing a constant int as a non-pointer, presume it is not a GC'd var and is safe
                # for activity state to mix
                if isa(val, LLVM.StoreInst) operands(val)[1] == cur && !isa(value_type(operands(val)[1]), LLVM.PointerType)
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
                println(io, " value="*badval)
            end
            println(io, "You may be using a constant variable as temporary storage for active memory (https://enzyme.mit.edu/julia/stable/#Activity-of-temporary-storage). If not, please open an issue, and either rewrite this variable to not be conditionally active or use Enzyme.API.runtimeActivity!(true) as a workaround for now")
            if bt !== nothing
                Base.show_backtrace(io, bt)
            end
        end
        emit_error(b, nothing, msg2)
        return C_NULL
    elseif errtype == API.ET_GetIndexError
        @assert B != C_NULL
        B = IRBuilder(B)
        msg5 = sprint() do io::IO
            print(io, "Enzyme internal error\n")
            print(io,  msg, '\n')
            if bt !== nothing
                print(io,"\nCaused by:")
                Base.show_backtrace(io, bt)
                println(io)
            end
        end
        emit_error(B, nothing, msg5)
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

const WideIntWidths = [256, 512, 1024, 2048]

let
    for n ∈ WideIntWidths
        let T = Symbol(:UInt,n)
            eval(quote primitive type $T  <: Unsigned $n end end)
        end
    end
end
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
        elseif N == 256
            return UInt256, false
        elseif N == 512
            return UInt512, false
        elseif N == 1024
            return UInt1024, false
        elseif N == 2048
            return UInt2048, false
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
                    cur = addrspacecast!(B, cur, LLVM.PointerType(eltype(ty), Tracked), LLVM.name(cur)*".innertracked")
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
        st = LLVM.Value(julia_undef_value_for_type(eltype(ty).ref, forceZero))
        return ConstantArray(eltype(ty), [st for i in 1:length(ty)]).ref
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

function shadow_alloc_rewrite(V::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef)
    V = LLVM.CallInst(V)
    gutils = GradientUtils(gutils)
    mode = get_mode(gutils)
    if mode == API.DEM_ReverseModePrimal || mode == API.DEM_ReverseModeGradient || mode == API.DEM_ReverseModeCombined
        fn = LLVM.parent(LLVM.parent(V))
        world = enzyme_extract_world(fn)
        has, Ty = abs_typeof(V)
        @assert has 
        rt = active_reg_inner(Ty, (), world)
        if rt == ActiveState || rt == MixedState
            operands(V)[3] = unsafe_to_llvm(Base.RefValue{Ty})
        end
    end
    nothing
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
    if LLVM.version().major >= 12
        push!(function_attributes(wrapper_f), EnumAttribute("mustprogress", 0))
    end
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

include("rules/allocrules.jl")
include("rules/llvmrules.jl")

function __init__()
    API.EnzymeSetHandler(@cfunction(julia_error, LLVM.API.LLVMValueRef, (Cstring, LLVM.API.LLVMValueRef, API.ErrorType, Ptr{Cvoid}, LLVM.API.LLVMValueRef, LLVM.API.LLVMBuilderRef)))
    API.EnzymeSetSanitizeDerivatives(@cfunction(julia_sanitize, LLVM.API.LLVMValueRef, (LLVM.API.LLVMValueRef, LLVM.API.LLVMValueRef, LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef)));
    API.EnzymeSetRuntimeInactiveError(@cfunction(emit_inacterror, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, LLVM.API.LLVMValueRef)))
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
    API.EnzymeSetUndefinedValueForType(@cfunction(
                                            julia_undef_value_for_type, LLVM.API.LLVMValueRef, (LLVM.API.LLVMTypeRef,UInt8))) 
    API.EnzymeSetShadowAllocRewrite(@cfunction(
                                               shadow_alloc_rewrite, Cvoid, (LLVM.API.LLVMValueRef,API.EnzymeGradientUtilsRef)))
    register_alloc_rules()
    register_llvm_rules()
end

# Define EnzymeTarget
Base.@kwdef struct EnzymeTarget <: AbstractCompilerTarget
end
GPUCompiler.llvm_triple(::EnzymeTarget) = Sys.MACHINE

# GPUCompiler.llvm_datalayout(::EnzymeTarget) =  nothing

function GPUCompiler.llvm_machine(::EnzymeTarget)
    return JIT.get_tm()
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
                  "jl_array_copy", "ijl_array_copy", "jl_idtable_rehash", "ijl_idtable_rehash",
                  "jl_f_tuple", "ijl_f_tuple", "jl_new_structv", "ijl_new_structv")
        if haskey(fns, boxfn)
            fn = fns[boxfn]
            push!(return_attributes(fn), LLVM.EnumAttribute("noalias", 0))
            if !(boxfn in ("jl_array_copy", "ijl_array_copy", "jl_idtable_rehash", "ijl_idtable_rehash"))
                push!(function_attributes(fn), LLVM.EnumAttribute("inaccessiblememonly", 0))
            end
            for u in LLVM.uses(fn)
                c = LLVM.user(u)
                if !isa(c, LLVM.CallInst)
                    continue
                end
                cf = LLVM.called_operand(c)
                if cf == fn
                    LLVM.API.LLVMAddCallSiteAttribute(c, LLVM.API.LLVMAttributeReturnIndex, LLVM.EnumAttribute("noalias", 0))
                    if !(boxfn in ("jl_array_copy", "ijl_array_copy", "jl_idtable_rehash", "ijl_idtable_rehash"))
                        LLVM.API.LLVMAddCallSiteAttribute(c, LLVM.API.LLVMAttributeFunctionIndex, LLVM.EnumAttribute("inaccessiblememonly", 0))
                    end
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
                if !(boxfn in ("jl_array_copy", "ijl_array_copy", "jl_idtable_rehash", "ijl_idtable_rehash"))
                    LLVM.API.LLVMAddCallSiteAttribute(c, LLVM.API.LLVMAttributeFunctionIndex, LLVM.EnumAttribute("inaccessiblememonly", 0))
                end
            end
        end
    end

    for gc in ("llvm.julia.gc_preserve_begin", "llvm.julia.gc_preserve_end")
        if haskey(fns, gc)
            fn = fns[gc]
            push!(function_attributes(fn), LLVM.EnumAttribute("inaccessiblememonly", 0))
        end
    end

    for rfn in ("jl_object_id_", "jl_object_id", "ijl_object_id_", "ijl_object_id")
        if haskey(fns, rfn)
            fn = fns[rfn]
            push!(function_attributes(fn), LLVM.EnumAttribute("readonly", 0))
        end
    end

    # Key of jl_eqtable_get/put is inactive, definitionally
    for rfn in ("jl_eqtable_get", "ijl_eqtable_get")
        if haskey(fns, rfn)
            fn = fns[rfn]
            push!(parameter_attributes(fn, 2), LLVM.StringAttribute("enzyme_inactive"))
            push!(function_attributes(fn), LLVM.EnumAttribute("readonly", 0))
            push!(function_attributes(fn), LLVM.EnumAttribute("argmemonly", 0))
        end
    end
    # Key of jl_eqtable_get/put is inactive, definitionally
    for rfn in ("jl_eqtable_put", "ijl_eqtable_put")
        if haskey(fns, rfn)
            fn = fns[rfn]
            push!(parameter_attributes(fn, 2), LLVM.StringAttribute("enzyme_inactive"))
            push!(parameter_attributes(fn, 4), LLVM.StringAttribute("enzyme_inactive"))
            push!(parameter_attributes(fn, 4), LLVM.EnumAttribute("writeonly"))
            push!(parameter_attributes(fn, 4), LLVM.EnumAttribute("nocapture"))
            push!(function_attributes(fn), LLVM.EnumAttribute("argmemonly", 0))
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

function enzyme_custom_extract_mi(orig::LLVM.Instruction, error=true)
    operand = LLVM.called_operand(orig)
    if isa(operand, LLVM.Function)
        return enzyme_custom_extract_mi(operand::LLVM.Function, error)
    elseif error
        GPUCompiler.@safe_error "Enzyme: Custom handler, could not find fn", orig
    end
    return nothing, nothing
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


include("rules/typerules.jl")
include("rules/activityrules.jl")

@inline Base.convert(::Type{API.CDIFFE_TYPE}, ::Type{A}) where A <: Const = API.DFT_CONSTANT
@inline Base.convert(::Type{API.CDIFFE_TYPE}, ::Type{A}) where A <: Active = API.DFT_OUT_DIFF
@inline Base.convert(::Type{API.CDIFFE_TYPE}, ::Type{A}) where A <: Duplicated = API.DFT_DUP_ARG
@inline Base.convert(::Type{API.CDIFFE_TYPE}, ::Type{A}) where A <: BatchDuplicated = API.DFT_DUP_ARG
@inline Base.convert(::Type{API.CDIFFE_TYPE}, ::Type{A}) where A <: BatchDuplicatedFunc = API.DFT_DUP_ARG
@inline Base.convert(::Type{API.CDIFFE_TYPE}, ::Type{A}) where A <: DuplicatedNoNeed = API.DFT_DUP_NONEED
@inline Base.convert(::Type{API.CDIFFE_TYPE}, ::Type{A}) where A <: BatchDuplicatedNoNeed = API.DFT_DUP_NONEED

function enzyme!(job, mod, primalf, TT, mode, width, parallel, actualRetType, wrap, modifiedBetween, returnPrimal, jlrules,expectedTapeType, loweredArgs, boxedArgs)
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
        isboxed = i in boxedArgs

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
    retType = convert(API.CDIFFE_TYPE, rt)

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
            augmented_primalf = create_abi_wrapper(augmented_primalf, TT, rt, actualRetType, API.DEM_ReverseModePrimal, augmented, width, returnPrimal, shadow_init, world, interp)
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
            adjointf = create_abi_wrapper(adjointf, TT, rt, actualRetType, API.DEM_ReverseModeCombined, nothing, width, returnPrimal, shadow_init, world, interp)
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
          adjointf = create_abi_wrapper(adjointf, TT, rt, actualRetType, API.DEM_ForwardMode, nothing, width, returnPrimal, shadow_init, world, interp)
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
    literal_rt = eltype(rettype)
    sret_union_rt = is_sret_union(literal_rt)
    @assert sret_union == sret_union_rt
    if sret_union
        actualRetType = Any
        literal_rt = Any
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
        if allocatedinline(actualRetType) != allocatedinline(literal_rt)
            @show actualRetType, literal_rt, rettype
        end
        @assert allocatedinline(actualRetType) == allocatedinline(literal_rt)
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
            push!(sret_types, literal_rt)
        else
            if returnPrimal
                push!(sret_types, literal_rt)
            else
                push!(sret_types, Nothing)
            end
        end
        # shadow return
        if existed[3] != 0
            if rettype <: Duplicated || rettype <: DuplicatedNoNeed || rettype <: BatchDuplicated || rettype <: BatchDuplicatedNoNeed || rettype <: BatchDuplicatedFunc
                if width == 1
                    push!(sret_types, literal_rt)
                else
                    push!(sret_types, AnonymousStruct(NTuple{width, literal_rt}))
                end
            end
        else
            @assert rettype <: Const || rettype <: Active
            push!(sret_types, Nothing)
        end
    end
    if Mode == API.DEM_ReverseModeCombined
        if returnPrimal
            push!(sret_types, literal_rt)
        end
    end
    if Mode == API.DEM_ForwardMode
        if returnPrimal
            push!(sret_types, literal_rt)
        end
        if !(rettype <: Const)
            if width == 1
                push!(sret_types, literal_rt)
            else
                push!(sret_types, AnonymousStruct(NTuple{width, literal_rt}))
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
                    if is_split
                        msg = sprint() do io
                            println(io, "Unimplemented: Had active input arg needing a box in split mode")
                            println(io, T, " at index ", i)
                            println(io, TT)
                        end
                        throw(AssertionError(msg))
                    end
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
            # Fix calling convention within julia that Tuple{Float,Float} ->[2 x float] rather than {float, float}
            # and that Bool -> i8, not i1
            tparm = params[i]
            tparm = calling_conv_fixup(builder, tparm, tape)
            push!(realparms, tparm)
            i += 1
        end

        val = call!(builder, LLVM.function_type(enzymefn), enzymefn, realparms)
        if LLVM.get_subprogram(llvm_f) !== nothing
            metadata(val)[LLVM.MD_dbg] = DILocation( 0, 0, LLVM.get_subprogram(llvm_f) )
        end

        @inline function fixup_abi(index, value)
            valty = sret_types[index]
            # Union becoming part of a tuple needs to be adjusted
            # See https://github.com/JuliaLang/julia/blob/81afdbc36b365fcbf3ae25b7451c6cb5798c0c3d/src/cgutils.cpp#L3795C1-L3801C121
            if valty isa Union
                T_int8 = LLVM.Int8Type()
                if value_type(value) == T_int8
                    value = nuwsub!(builder, value, LLVM.ConstantInt(T_int8, 1))
                end
            end
            return value
        end

        if Mode == API.DEM_ReverseModePrimal

            # if in split mode and the return is a union marked duplicated, upgrade floating point like shadow returns into ref{ty} since otherwise use of the value will create problems.
            # 3 is index of shadow
            if existed[3] != 0 && sret_union && active_reg_inner(pactualRetType, (), world, #=justActive=#Val(true), #=UnionSret=#Val(true)) == ActiveState
                rewrite_union_returns_as_ref(enzymefn, data[3], world, width)
            end
            returnNum = 0
            for i in 1:3
                if existed[i] != 0
                    eval = val
                    if data[i] != -1
                        eval = extract_value!(builder, val, data[i])
                    end
                    eval = fixup_abi(i, eval)
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
                    ty = sret_types[i]
                    # if primal return, we can upgrade to the full known type
                    if i == 2
                        ty = actualRetType
                    end
                    @assert !(isghostty(combinedReturn) || Core.Compiler.isconstType(combinedReturn) )
                    @assert Core.Compiler.isconstType(ty)
                    eval = makeInstanceOf(ty)
                    eval = fixup_abi(i, eval)
                    ptr = inbounds_gep!(builder, jltype, sret, [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), returnNum)])
                    ptr = pointercast!(builder, ptr, LLVM.PointerType(value_type(eval)))
                    si = store!(builder, eval, ptr)
                    returnNum+=1
                end
            end
            @assert returnNum == numLLVMReturns
        elseif Mode == API.DEM_ForwardMode
            count_Sret = 0
            count_llvm_Sret = 0
            if !isghostty(actualRetType)
                if returnPrimal
                    count_llvm_Sret += 1
                end
                if !(rettype <: Const)
                    count_llvm_Sret += 1
                end
            end
            if !isghostty(literal_rt)
                if returnPrimal
                    count_Sret += 1
                end
                if !(rettype <: Const)
                    count_Sret += 1
                end
            end
            for returnNum in 0:(count_Sret-1)
                eval = fixup_abi(returnNum+1, if count_llvm_Sret == 0
                    makeInstanceOf(sret_types[returnNum+1])
                elseif count_llvm_Sret == 1
                    val
                else
                    @assert count_llvm_Sret > 1
                    extract_value!(builder, val, returnNum)
                end)
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
                    if !isghostty(literal_rt)
                        eval = fixup_abi(returnNum+1, if !isghostty(actualRetType)
                            extract_value!(builder, val, returnNum)
                        else
                            makeInstanceOf(sret_types[returnNum+1])
                        end)
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
function lower_convention(functy::Type, mod::LLVM.Module, entry_f::LLVM.Function, actualRetType::Type, RetActivity, TT)
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

    boxedArgs = Set{Int}()
    loweredArgs = Set{Int}()

    for arg in args
        typ = arg.codegen.typ
        if GPUCompiler.deserves_argbox(arg.typ)
            push!(boxedArgs, arg.arg_i)
            push!(wrapper_types, typ)
        elseif arg.cc != GPUCompiler.BITS_REF
            push!(wrapper_types, typ)
        else
            # bits ref, and not boxed
            # if TT.parameters[arg.arg_i] <: Const
            #     push!(boxedArgs, arg.arg_i)
            #     push!(wrapper_types, typ)
            # else
                push!(wrapper_types, eltype(typ))
                push!(loweredArgs, arg.arg_i)
            # end
        end
    end

    if length(loweredArgs) == 0 && !sret && !sret_union
        return entry_f, returnRoots, boxedArgs, loweredArgs
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
                if arg.arg_i in loweredArgs
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
        dl = string(LLVM.datalayout(LLVM.parent(entry_f)))
        if sret
            if !in(0, parmsRemoved)
                sretPtr = alloca!(builder, eltype(value_type(parameters(entry_f)[1])))
                ctx = LLVM.context(entry_f)
                if RetActivity <: Const
                    metadata(sretPtr)["enzyme_inactive"] = MDNode(LLVM.Metadata[])
                end
                metadata(sretPtr)["enzyme_type"] = to_md(typetree(Ptr{actualRetType}, ctx, dl), ctx)
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
            if arg.arg_i in loweredArgs
                # copy the argument value to a stack slot, and reference it.
                ty = value_type(parm)
                if !isa(ty, LLVM.PointerType)
                    @safe_show entry_f, args, parm, ty
                end
                @assert isa(ty, LLVM.PointerType)
                ptr = alloca!(builder, eltype(ty))
                if TT.parameters[arg.arg_i] <: Const
                    metadata(ptr)["enzyme_inactive"] = MDNode(LLVM.Metadata[])
                end
                ctx = LLVM.context(entry_f)
                metadata(ptr)["enzyme_type"] = to_md(typetree(Ptr{arg.typ}, ctx, dl), ctx)
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
                        nobj = if sretPtr !== nothing
                            obj = emit_allocobj!(builder, jlrettype)
                            llty = convert(LLVMType, jlrettype)
                            ld = load!(builder, llty, bitcast!(builder, sretPtr, LLVM.PointerType(llty, addrspace(value_type(sretPtr)))))
                            store!(builder, ld, bitcast!(builder, obj, LLVM.PointerType(llty, addrspace(value_type(obj)))))
                            # memcpy!(builder, bitcast!(builder, obj, LLVM.PointerType(T_int8, addrspace(value_type(obj)))), 0, bitcast!(builder, sretPtr, LLVM.PointerType(T_int8)), 0, LLVM.ConstantInt(T_int64, sizeof(jlrettype)))
                            obj
                        else
                            @assert false
                        end
                        ret!(builder, obj)
                    end

                    LLVM.API.LLVMAddCase(sw, LLVM.ConstantInt(value_type(scase), counter), BB)
                    counter+=1
                    return
                end
                for_each_uniontype_small(inner, actualRetType)

                position!(builder, def)
                ret!(builder, extract_value!(builder, res, 0))
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
    
    # Fix phinodes used exclusively in extractvalue to be separate phi nodes
    phistofix = LLVM.PHIInst[]
    for bb in blocks(wrapper_f)
        for inst in instructions(bb)
            if isa(inst, LLVM.PHIInst)
                if !isa(value_type(inst), LLVM.StructType)
                    continue
                end
                legal = true
                for u in LLVM.uses(inst)
                    u = LLVM.user(u)
                    if !isa(u, LLVM.ExtractValueInst)
                        legal = false
                        break
                    end
                    if LLVM.API.LLVMGetNumIndices(u) != 1
                        legal = false
                        break
                    end
                    for op in operands(u)[2:end]
                        if !isa(op, LLVM.ConstantInt)
                            legal = false
                            break
                        end
                    end
                end
                if legal
                    push!(phistofix, inst)
                end
            end
        end
    end
    for p in phistofix
        nb = IRBuilder()
        position!(nb, p)
        st = value_type(p)::LLVM.StructType
        phis = LLVM.PHIInst[]
        for (i, t) in enumerate(LLVM.elements(st))
            np = phi!(nb, t)
            nvs = Tuple{LLVM.Value, LLVM.BasicBlock}[]
            for (v, b) in LLVM.incoming(p)  
                prevbld = IRBuilder()
                position!(prevbld, terminator(b))
                push!(nvs, (extract_value!(prevbld, v, i-1), b))
            end
            append!(LLVM.incoming(np), nvs)
            push!(phis, np)
        end

        torem = LLVM.Instruction[]
        for u in LLVM.uses(p)
            u = LLVM.user(u)
            @assert isa(u, LLVM.ExtractValueInst)
            @assert LLVM.API.LLVMGetNumIndices(u) == 1
            ind = unsafe_load(LLVM.API.LLVMGetIndices(u))
            replace_uses!(u, phis[ind+1])
            push!(torem, u)
        end
        for u in torem
            LLVM.API.LLVMInstructionEraseFromParent(u)
        end
        LLVM.API.LLVMInstructionEraseFromParent(p)
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
    return wrapper_f, returnRoots, boxedArgs, loweredArgs
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
    @assert length(modifiedBetween) == length(TT.parameters) 
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
    if bitcode_replacement() && API.EnzymeBitcodeReplacement(mod, disableFallback) != 0
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

    loweredArgs = Set{Int}()
    boxedArgs = Set{Int}()
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

        julia_activity_rule(llvmfn)
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
            handleCustom("enz_noop", [StringAttribute("enzyme_inactive"), EnumAttribute("nofree")])
            continue
        end
        if EnzymeRules.is_inactive_noinl_from_sig(mi.specTypes; world, method_table, caller)
            handleCustom("enz_noop", [StringAttribute("enzyme_inactive"), EnumAttribute("nofree")], false, false)
            for bb in blocks(llvmfn)
                for inst in instructions(bb)
                    if isa(inst, LLVM.CallInst)
                        LLVM.API.LLVMAddCallSiteAttribute(inst, LLVM.API.LLVMAttributeFunctionIndex, StringAttribute("enzyme_inactive"))
                        LLVM.API.LLVMAddCallSiteAttribute(inst, LLVM.API.LLVMAttributeFunctionIndex, EnumAttribute("nofree"))
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
            if length(sparam_vals) == 1 && first(sparam_vals) <: Task
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
          llvmfn, _, boxedArgs, loweredArgs = lower_convention(source_sig, mod, llvmfn, k.ci.rettype, Duplicated, (Const, Duplicated))
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


    primalf, returnRoots = primalf, false

    if lowerConvention 
        primalf, returnRoots, boxedArgs, loweredArgs = lower_convention(source_sig, mod, primalf, actualRetType, job.config.params.rt, TT)
    end

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

    for f in functions(mod), bb in blocks(f), inst in instructions(bb)
        if !isa(inst, LLVM.CallInst)
            continue
        end
        fn = LLVM.called_operand(inst)
        if !isa(fn, LLVM.Function)
            continue
        end
        if length(blocks(fn)) != 0
            continue
        end
        ty = value_type(inst)
        if ty == LLVM.VoidType()
            continue
        end

        legal, jTy = abs_typeof(inst, true)
        if !legal
            continue
        end
        if !guaranteed_const_nongen(jTy, world)
            continue
        end        
        LLVM.API.LLVMAddCallSiteAttribute(inst, LLVM.API.LLVMAttributeReturnIndex, StringAttribute("enzyme_inactive"))
    end

    TapeType::Type = Cvoid

    if params.run_enzyme
        # Generate the adjoint
        jlrules = String["enzyme_custom"]
        for (fname, (ftyp, mi)) in foundTys
            haskey(functions(mod), fname) || continue
            push!(jlrules, fname)
        end

        adjointf, augmented_primalf, TapeType = enzyme!(job, mod, primalf, TT, mode, width, parallel, actualRetType, abiwrap, modifiedBetween, returnPrimal, jlrules, expectedTapeType, loweredArgs, boxedArgs)
        toremove = []
        # Inline the wrapper
        for f in functions(mod)
            for b in blocks(f)
                term = terminator(b)
                if isa(term, LLVM.UnreachableInst)
                    shouldemit = true
                    tmp = term
                    while true
                        tmp = LLVM.API.LLVMGetPreviousInstruction(tmp)
                        if tmp == C_NULL
                            break
                        end
                        tmp = LLVM.Instruction(tmp)
                        if isa(tmp, LLVM.CallInst)
                            cf = LLVM.called_operand(tmp)
                            if isa(cf, LLVM.Function)
                                nm = LLVM.name(cf)
                                if nm == "gpu_signal_exception" || nm == "gpu_report_exception"
                                    shouldemit = false
                                    break
                                end
                            end
                        end
                    end

                    if shouldemit
                        b = IRBuilder()
                        position!(b, term)
                        emit_error(b, term, "Enzyme: The original primal code hits this error condition, thus differentiating it does not make sense")
                    end
                end
            end
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
    if parent_job !== nothing
        if parent_job.config.target isa GPUCompiler.PTXCompilerTarget
			arg1 = ("sin",        "cos",     "tan",       "log2",   "exp",    "exp2",
				  "exp10",      "cosh",    "sinh",      "tanh",   "atan",
				  "asin",       "acos",    "log",       "log10",  "log1p",  "acosh",
				  "asinh",      "atanh",   "expm1",    				   "cbrt",
				  "rcbrt",      "j0",      "j1",        "y0",     "y1",   
				  "erf",     "erfinv",    "erfc",   "erfcx",  "erfcinv",
				   "remquo",  "tgamma",
				  "round",      "fdim",    "logb",   "isinf", 
				  "sqrt",        "fabs",   "atan2", )
			# isinf, finite "modf",       "fmod",    "remainder", 
			# "rnorm3d",    "norm4d",  "rnorm4d",   "norm",   "rnorm",
			#   "hypot",  "rhypot",
			# "yn", "jn", "norm3d", "ilogb", powi
		    # "normcdfinv", "normcdf", "lgamma",    "ldexp",  "scalbn", "frexp",
			# arg1 = ("atan2", "fmax", "pow")
			for n in arg1, (T, pf, lpf) in ((LLVM.DoubleType(), "", "f64"), (LLVM.FloatType(), "f", "f32"))
				fname = "__nv_"*n*pf
				if !haskey(functions(mod), fname)
					FT = LLVM.FunctionType(T, [T], vararg=false)
					wrapper_f = LLVM.Function(mod, fname, FT)
					llname = "llvm."*n*"."*lpf
    				push!(function_attributes(wrapper_f), StringAttribute("implements", llname))
				end
			end
		end
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

# @generated function splatnew(::Type{T}, args::NTuple{N,AT}) where {T,N,AT}
#     return quote
#         Base.@_inline_meta
#         $(Expr(:splatnew, :T, :args))
#     end
# end
@inline function splatnew(::Type{T}, args::NTuple{N,AT}) where {T,N,AT}
    ccall(:jl_new_structt, Any, (Any, Any), T, args)::T
end

@inline function recursive_add(x::T, y::T) where T
    if guaranteed_const(T)
        return x
    end
    splatnew(T, ntuple(Val(fieldcount(T))) do i
        Base.@_inline_meta
        prev = getfield(x, i)
        next = getfield(y, i)
        recursive_add(prev, next)
    end)
end

@inline function recursive_add(x::T, y::T) where {T<:AbstractFloat}
    return x + y
end

function add_one_in_place(x)
    ty = typeof(x)
    # ptr = Base.pointer_from_objref(x)
    ptr = unsafe_to_pointer(x)
    if ty <: Base.RefValue || ty == Base.RefValue{Float64}
        x[] = recursive_add(x[], one(eltype(ty)))
    else
        error("Enzyme Mutability Error: Cannot add one in place to immutable value "*string(x))
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
    if job.config.params.ABI <: InlineABI
        return CompileResult(Val((Symbol(mod), Symbol(adjoint_name))), Val((Symbol(mod), Symbol(primal_name))), TapeType)
    end

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

@inline @generated function thunk(::Val{World}, ::Type{FA}, ::Type{A}, tt::Type{TT},::Val{Mode}, ::Val{width}, ::Val{ModifiedBetween}, ::Val{ReturnPrimal}, ::Val{ShadowInit}, ::Type{ABI}) where {FA<:Annotation, A<:Annotation, TT, Mode, ModifiedBetween, width, ReturnPrimal, ShadowInit, World, ABI}   
    JuliaContext() do ctx
        mi = fspec(eltype(FA), TT, World)

        target = Compiler.EnzymeTarget()
        params = Compiler.EnzymeCompilerParams(Tuple{FA, TT.parameters...}, Mode, width, remove_innerty(A), true, #=abiwrap=#true, ModifiedBetween, ReturnPrimal, ShadowInit, UnknownTapeType, ABI)
        tmp_job    = Compiler.CompilerJob(mi, CompilerConfig(target, params; kernel=false), World)

        sig = Tuple{eltype(FA), map(eltype, TT.parameters)...}

        interp = GPUCompiler.get_interpreter(tmp_job)

        # TODO check compile return here, early
        # rrt = Core.Compiler.return_type(f, primal.tt) # nothing
        rrt = something(Core.Compiler.typeinf_type(interp, mi.def, mi.specTypes, mi.sparam_vals), Any)

        if rrt == Union{}
            estr = "Function to differentiate `$mi` is guaranteed to return an error and doesn't make sense to autodiff. Giving up"
			return quote
				error($estr)
			end
        end
        
        if !(A <: Const) && guaranteed_const_nongen(rrt, World)
			estr = "Return type `$rrt` not marked Const, but type is guaranteed to be constant"
            return quote
				error($estr)
			end
        end

        rt2 = if A isa UnionAll
            A{rrt}
        else
            @assert A isa DataType
            # Can we relax this condition?
            # @assert eltype(A) == rrt
            A
        end
       
        params = Compiler.EnzymeCompilerParams(Tuple{FA, TT.parameters...}, Mode, width, rt2, true, #=abiwrap=#true, ModifiedBetween, ReturnPrimal, ShadowInit, UnknownTapeType, ABI)
        job    = Compiler.CompilerJob(mi, CompilerConfig(target, params; kernel=false), World)

        # We need to use primal as the key, to lookup the right method
        # but need to mixin the hash of the adjoint to avoid cache collisions
        # This is counter-intuitive since we would expect the cache to be split
        # by the primal, but we want the generated code to be invalidated by
        # invalidations of the primal, which is managed by GPUCompiler.


        compile_result = cached_compilation(job)
        if Mode == API.DEM_ReverseModePrimal || Mode == API.DEM_ReverseModeGradient
            TapeType = compile_result.TapeType
            AugT = AugmentedForwardThunk{typeof(compile_result.primal), FA, rt2, Tuple{params.TT.parameters[2:end]...}, Val{width}, Val(ReturnPrimal), TapeType}
            AdjT = AdjointThunk{typeof(compile_result.adjoint), FA, rt2, Tuple{params.TT.parameters[2:end]...}, Val{width}, TapeType}
            return quote
                Base.@_inline_meta
                augmented = $AugT($(compile_result.primal))
                adjoint  = $AdjT($(compile_result.adjoint))
                (augmented, adjoint)
            end
        elseif Mode == API.DEM_ReverseModeCombined
            CAdjT = CombinedAdjointThunk{typeof(compile_result.adjoint), FA, rt2, Tuple{params.TT.parameters[2:end]...}, Val{width}, Val(ReturnPrimal)}
            return quote
                Base.@_inline_meta
                $CAdjT($(compile_result.adjoint))
            end
        elseif Mode == API.DEM_ForwardMode
            FMT = ForwardModeThunk{typeof(compile_result.adjoint), FA, rt2, Tuple{params.TT.parameters[2:end]...}, Val{width}, Val(ReturnPrimal)}
            return quote
                Base.@_inline_meta
                $FMT($(compile_result.adjoint))
            end
        else
            @assert false
        end
    end
end

import GPUCompiler: deferred_codegen_jobs

@generated function deferred_codegen(::Val{World}, ::Type{FA}, ::Val{TT}, ::Val{A},::Val{Mode},
        ::Val{width}, ::Val{ModifiedBetween}, ::Val{ReturnPrimal}=Val(false),::Val{ShadowInit}=Val(false),::Type{ExpectedTapeType}=UnknownTapeType) where {World, FA<:Annotation,TT, A, Mode, width, ModifiedBetween, ReturnPrimal, ShadowInit,ExpectedTapeType}
    JuliaContext() do ctx

        mi = fspec(eltype(FA), TT, World)
        target = EnzymeTarget()

        rt2 = if A isa UnionAll 
            params = EnzymeCompilerParams(Tuple{FA, TT.parameters...}, Mode, width, remove_innerty(A), true, #=abiwrap=#true, ModifiedBetween, ReturnPrimal, ShadowInit,ExpectedTapeType, FFIABI)
            tmp_job    = Compiler.CompilerJob(mi, CompilerConfig(target, params; kernel=false), World)
            
            sig = Tuple{eltype(FA), map(eltype, TT.parameters)...}
            interp = GPUCompiler.get_interpreter(tmp_job)

            rrt = something(Core.Compiler.typeinf_type(interp, mi.def, mi.specTypes, mi.sparam_vals), Any)

            # Don't error here but default to nothing return since in cuda context we don't use the device overrides
            if rrt == Union{}
                rrt = Nothing
            end
            
            if !(A <: Const) && guaranteed_const_nongen(rrt, World)
                estr = "Return type `$rrt` not marked Const, but type is guaranteed to be constant"
                return quote
                    error($estr)
                end
            end
            A{rrt}
        else
            @assert A isa DataType
            A
        end
        
        params = EnzymeCompilerParams(Tuple{FA, TT.parameters...}, Mode, width, rt2, true, #=abiwrap=#true, ModifiedBetween, ReturnPrimal, ShadowInit,ExpectedTapeType, FFIABI)
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
            Base.@_inline_meta
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
