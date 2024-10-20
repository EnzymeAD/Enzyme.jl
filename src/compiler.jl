module Compiler

import ..Enzyme
import Enzyme:
    Const,
    Active,
    Duplicated,
    DuplicatedNoNeed,
    BatchDuplicated,
    BatchDuplicatedNoNeed,
    BatchDuplicatedFunc,
    Annotation,
    guess_activity,
    eltype,
    API,
    TypeTree,
    typetree,
    TypeTreeTable,
    only!,
    shift!,
    data0!,
    merge!,
    to_md,
    to_fullmd,
    TypeAnalysis,
    FnTypeInfo,
    Logic,
    allocatedinline,
    ismutabletype
using Enzyme

import EnzymeCore
import EnzymeCore: EnzymeRules, ABI, FFIABI, DefaultABI

using LLVM, GPUCompiler, Libdl
import Enzyme_jll

import GPUCompiler: CompilerJob, codegen, safe_name
using LLVM.Interop
import LLVM: Target, TargetMachine
import SparseArrays
using Printf

using Preferences

bitcode_replacement() = parse(Bool, @load_preference("bitcode_replacement", "true"))
bitcode_replacement!(val) = @set_preferences!("bitcode_replacement" => string(val))

function cpu_name()
    ccall(:jl_get_cpu_name, String, ())
end

function cpu_features()
    return ccall(:jl_get_cpu_features, String, ())
end

import GPUCompiler: @safe_debug, @safe_info, @safe_warn, @safe_error

include("compiler/utils.jl")

include("compiler/orcv2.jl")

include("gradientutils.jl")


# Julia function to LLVM stem and arity
const cmplx_known_ops =
    Dict{DataType,Tuple{Symbol,Int,Union{Nothing,Tuple{Symbol,DataType}}}}(
        typeof(Base.inv) => (:cmplx_inv, 1, nothing),
        typeof(Base.sqrt) => (:cmplx_sqrt, 1, nothing),
    )
const known_ops = Dict{DataType,Tuple{Symbol,Int,Union{Nothing,Tuple{Symbol,DataType}}}}(
    typeof(Base.cbrt) => (:cbrt, 1, nothing),
    typeof(Base.rem2pi) => (:jl_rem2pi, 2, nothing),
    typeof(Base.sqrt) => (:sqrt, 1, nothing),
    typeof(Base.sin) => (:sin, 1, nothing),
    typeof(Base.sinc) => (:sincn, 1, nothing),
    typeof(Base.sincos) => (:__fd_sincos_1, 1, nothing),
    typeof(Base.sincospi) => (:sincospi, 1, nothing),
    typeof(Base.sinpi) => (:sinpi, 1, nothing),
    typeof(Base.cospi) => (:cospi, 1, nothing),
    typeof(Base.:^) => (:pow, 2, nothing),
    typeof(Base.rem) => (:fmod, 2, nothing),
    typeof(Base.cos) => (:cos, 1, nothing),
    typeof(Base.tan) => (:tan, 1, nothing),
    typeof(Base.exp) => (:exp, 1, nothing),
    typeof(Base.exp2) => (:exp2, 1, nothing),
    typeof(Base.expm1) => (:expm1, 1, nothing),
    typeof(Base.exp10) => (:exp10, 1, nothing),
    typeof(Base.FastMath.exp_fast) => (:exp, 1, nothing),
    typeof(Base.log) => (:log, 1, nothing),
    typeof(Base.FastMath.log) => (:log, 1, nothing),
    typeof(Base.log1p) => (:log1p, 1, nothing),
    typeof(Base.log2) => (:log2, 1, nothing),
    typeof(Base.log10) => (:log10, 1, nothing),
    typeof(Base.asin) => (:asin, 1, nothing),
    typeof(Base.acos) => (:acos, 1, nothing),
    typeof(Base.atan) => (:atan, 1, nothing),
    typeof(Base.atan) => (:atan2, 2, nothing),
    typeof(Base.sinh) => (:sinh, 1, nothing),
    typeof(Base.FastMath.sinh_fast) => (:sinh, 1, nothing),
    typeof(Base.cosh) => (:cosh, 1, nothing),
    typeof(Base.FastMath.cosh_fast) => (:cosh, 1, nothing),
    typeof(Base.tanh) => (:tanh, 1, nothing),
    typeof(Base.ldexp) => (:ldexp, 2, nothing),
    typeof(Base.FastMath.tanh_fast) => (:tanh, 1, nothing),
    typeof(Base.fma_emulated) => (:fma, 3, nothing),
)
@inline function find_math_method(@nospecialize(func), sparam_vals)
    if func ∈ keys(known_ops)
        name, arity, toinject = known_ops[func]
        Tys = (Float32, Float64)

        if length(sparam_vals) == arity
            T = first(sparam_vals)
            legal = T ∈ Tys

            if legal
                if name == :ldexp
                    if !(sparam_vals[2] <: Integer)
                        legal = false
                    end
                elseif name == :pow
                    if sparam_vals[2] <: Integer
                        name = :powi
                    elseif sparam_vals[2] != T
                        legal = false
                    end
                elseif name == :jl_rem2pi
                else
                    if !all(==(T), sparam_vals)
                        legal = false
                    end
                end
            end
            if legal
                return name, toinject, T
            end
        end
    end

    if func ∈ keys(cmplx_known_ops)
        name, arity, toinject = cmplx_known_ops[func]
        Tys = (Complex{Float32}, Complex{Float64})
        if length(sparam_vals) == arity
            T = first(sparam_vals)
            legal = T ∈ Tys

            if legal
                if !all(==(T), sparam_vals)
                    legal = false
                end
            end
            if legal
                return name, toinject, T
            end
        end
    end
    return nothing, nothing, nothing
end

const nofreefns = Set{String}((
    "pcre2_match_8",
    "julia.gcroot_flush",
    "pcre2_jit_stack_assign_8",
    "pcre2_match_context_create_8",
    "pcre2_jit_stack_create_8",
    "ijl_gc_enable_finalizers_internal",
    "jl_gc_enable_finalizers_internal",
    "pcre2_match_data_create_from_pattern_8",
    "ijl_gc_run_pending_finalizers",
    "jl_gc_run_pending_finalizers",
    "ijl_typeassert",
    "jl_typeassert",
    "ijl_f_isdefined",
    "jl_f_isdefined",
    "ijl_field_index",
    "jl_field_index",
    "ijl_specializations_get_linfo",
    "jl_specializations_get_linfo",
    "ijl_gf_invoke_lookup_worlds",
    "jl_gf_invoke_lookup_worlds",
    "ijl_gc_get_total_bytes",
    "jl_gc_get_total_bytes",
    "ijl_array_grow_at",
    "jl_array_grow_at",
    "ijl_try_substrtod",
    "jl_try_substrtod",
    "jl_f__apply_iterate",
    "ijl_field_index",
    "jl_field_index",
    "julia.call",
    "julia.call2",
    "ijl_tagged_gensym",
    "jl_tagged_gensym",
    "ijl_array_ptr_copy",
    "jl_array_ptr_copy",
    "ijl_array_copy",
    "jl_array_copy",
    "ijl_genericmemory_copy_slice",
    "jl_genericmemory_copy_slice",
    "ijl_get_nth_field_checked",
    "ijl_get_nth_field_checked",
    "jl_array_del_end",
    "ijl_array_del_end",
    "jl_get_world_counter",
    "ijl_get_world_counter",
    "memhash32_seed",
    "memhash_seed",
    "ijl_module_parent",
    "jl_module_parent",
    "julia.safepoint",
    "ijl_set_task_tid",
    "jl_set_task_tid",
    "ijl_get_task_tid",
    "jl_get_task_tid",
    "julia.get_pgcstack_or_new",
    "ijl_global_event_loop",
    "jl_global_event_loop",
    "ijl_gf_invoke_lookup",
    "jl_gf_invoke_lookup",
    "ijl_f_typeassert",
    "jl_f_typeassert",
    "ijl_type_unionall",
    "jl_type_unionall",
    "jl_gc_queue_root",
    "gpu_report_exception",
    "gpu_signal_exception",
    "julia.ptls_states",
    "julia.write_barrier",
    "julia.typeof",
    "jl_backtrace_from_here",
    "ijl_backtrace_from_here",
    "jl_box_int64",
    "jl_box_int32",
    "ijl_box_int64",
    "ijl_box_int32",
    "jl_box_uint64",
    "jl_box_uint32",
    "ijl_box_uint64",
    "ijl_box_uint32",
    "ijl_box_char",
    "jl_box_char",
    "ijl_subtype",
    "jl_subtype",
    "julia.get_pgcstack",
    "jl_in_threaded_region",
    "jl_object_id_",
    "jl_object_id",
    "ijl_object_id_",
    "ijl_object_id",
    "jl_breakpoint",
    "llvm.julia.gc_preserve_begin",
    "llvm.julia.gc_preserve_end",
    "jl_get_ptls_states",
    "ijl_get_ptls_states",
    "jl_f_fieldtype",
    "jl_symbol_n",
    "jl_stored_inline",
    "ijl_stored_inline",
    "jl_f_apply_type",
    "jl_f_issubtype",
    "jl_isa",
    "ijl_isa",
    "jl_matching_methods",
    "ijl_matching_methods",
    "jl_excstack_state",
    "ijl_excstack_state",
    "jl_current_exception",
    "ijl_current_exception",
    "memhash_seed",
    "jl_f__typevar",
    "ijl_f__typevar",
    "jl_f_isa",
    "ijl_f_isa",
    "jl_set_task_threadpoolid",
    "ijl_set_task_threadpoolid",
    "jl_types_equal",
    "ijl_types_equal",
    "jl_invoke",
    "ijl_invoke",
    "jl_apply_generic",
    "ijl_apply_generic",
    "jl_egal__unboxed",
    "julia.pointer_from_objref",
    "_platform_memcmp",
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
    "pcre2_match_data_create_from_pattern_8",
    "ijl_typeassert",
    "jl_typeassert",
    "ijl_f_isdefined",
    "jl_f_isdefined",
    "ijl_field_index",
    "jl_field_index",
    "ijl_specializations_get_linfo",
    "jl_specializations_get_linfo",
    "ijl_gf_invoke_lookup_worlds",
    "jl_gf_invoke_lookup_worlds",
    "ijl_gc_get_total_bytes",
    "jl_gc_get_total_bytes",
    "ijl_try_substrtod",
    "jl_try_substrtod",
    "ijl_tagged_gensym",
    "jl_tagged_gensym",
    "jl_get_world_counter",
    "ijl_get_world_counter",
    "memhash32_seed",
    "memhash_seed",
    "ijl_module_parent",
    "jl_module_parent",
    "julia.safepoint",
    "ijl_set_task_tid",
    "jl_set_task_tid",
    "ijl_get_task_tid",
    "jl_get_task_tid",
    "julia.get_pgcstack_or_new",
    "ijl_global_event_loop",
    "jl_global_event_loop",
    "ijl_gf_invoke_lookup",
    "jl_gf_invoke_lookup",
    "ijl_f_typeassert",
    "jl_f_typeassert",
    "ijl_type_unionall",
    "jl_type_unionall",
    "jl_gc_queue_root",
    "gpu_report_exception",
    "gpu_signal_exception",
    "julia.ptls_states",
    "julia.write_barrier",
    "julia.typeof",
    "jl_backtrace_from_here",
    "ijl_backtrace_from_here",
    "jl_box_int64",
    "jl_box_int32",
    "ijl_box_int64",
    "ijl_box_int32",
    "jl_box_uint64",
    "jl_box_uint32",
    "ijl_box_uint64",
    "ijl_box_uint32",
    "ijl_box_char",
    "jl_box_char",
    "ijl_subtype",
    "jl_subtype",
    "julia.get_pgcstack",
    "jl_in_threaded_region",
    "jl_object_id_",
    "jl_object_id",
    "ijl_object_id_",
    "ijl_object_id",
    "jl_breakpoint",
    "llvm.julia.gc_preserve_begin",
    "llvm.julia.gc_preserve_end",
    "jl_get_ptls_states",
    "ijl_get_ptls_states",
    "jl_f_fieldtype",
    "jl_symbol_n",
    "jl_stored_inline",
    "ijl_stored_inline",
    "jl_f_apply_type",
    "jl_f_issubtype",
    "jl_isa",
    "ijl_isa",
    "jl_matching_methods",
    "ijl_matching_methods",
    "jl_excstack_state",
    "ijl_excstack_state",
    "jl_current_exception",
    "ijl_current_exception",
    "memhash_seed",
    "jl_f__typevar",
    "ijl_f__typevar",
    "jl_f_isa",
    "ijl_f_isa",
    "jl_set_task_threadpoolid",
    "ijl_set_task_threadpoolid",
    "jl_types_equal",
    "ijl_types_equal",
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
    # "jl_"
))

const activefns = Set{String}(("jl_",))

const inactiveglobs = Set{String}((
    "ijl_boxed_uint8_cache",
    "jl_boxed_uint8_cache",
    "ijl_boxed_int8_cache",
    "jl_boxed_int8_cache",
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

struct Merger{seen,worldT,justActive,UnionSret,AbstractIsMixed}
    world::worldT
end

@inline element(::Val{T}) where {T} = T

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

@inline function (c::Merger{seen,worldT,justActive,UnionSret,AbstractIsMixed})(
    f::Int,
) where {seen,worldT,justActive,UnionSret,AbstractIsMixed}
    T = element(first(seen))

    reftype = ismutabletype(T) || (T isa UnionAll && !AbstractIsMixed)

    if justActive && reftype
        return Val(AnyState)
    end

    subT = typed_fieldtype(T, f)

    if justActive && !allocatedinline(subT)
        return Val(AnyState)
    end

    sub = active_reg_inner(
        subT,
        seen,
        c.world,
        Val(justActive),
        Val(UnionSret),
        Val(AbstractIsMixed),
    )

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

@inline forcefold(::Val{RT}) where {RT} = RT

@inline function forcefold(::Val{ty}, ::Val{sty}, C::Vararg{Any,N}) where {ty,sty,N}
    if sty == AnyState || sty == ty
        return forcefold(Val(ty), C...)
    end
    if ty == AnyState
        return forcefold(Val(sty), C...)
    else
        return MixedState
    end
end

@inline ptreltype(::Type{Ptr{T}}) where {T} = T
@inline ptreltype(::Type{Core.LLVMPtr{T,N}}) where {T,N} = T
@inline ptreltype(::Type{Core.LLVMPtr{T} where N}) where {T} = T
@inline ptreltype(::Type{Base.RefValue{T}}) where {T} = T
@inline ptreltype(::Type{Array{T,N}}) where {T,N} = T
@inline ptreltype(::Type{Array{T,N} where N}) where {T} = T
@inline ptreltype(::Type{Complex{T}}) where {T} = T
@inline ptreltype(::Type{Tuple{Vararg{T}}}) where {T} = T
@inline ptreltype(::Type{IdDict{K,V}}) where {K,V} = V
@inline ptreltype(::Type{IdDict{K,V} where K}) where {V} = V
@inline ptreltype(::Type{SparseArrays.CHOLMOD.Dense{T}}) where T = T
@static if VERSION < v"1.11-"
else
@inline ptreltype(::Type{Memory{T}}) where T = T
end

@inline is_arrayorvararg_ty(::Type) = false
@inline is_arrayorvararg_ty(::Type{Array{T,N}}) where {T,N} = true
@inline is_arrayorvararg_ty(::Type{Array{T,N} where N}) where {T} = true
@inline is_arrayorvararg_ty(::Type{Tuple{Vararg{T2}}}) where {T2} = true
@inline is_arrayorvararg_ty(::Type{Ptr{T}}) where {T} = true
@inline is_arrayorvararg_ty(::Type{Core.LLVMPtr{T,N}}) where {T,N} = true
@inline is_arrayorvararg_ty(::Type{Core.LLVMPtr{T,N} where N}) where {T} = true
@inline is_arrayorvararg_ty(::Type{Base.RefValue{T}}) where {T} = true
@inline is_arrayorvararg_ty(::Type{IdDict{K,V}}) where {K,V} = true
@inline is_arrayorvararg_ty(::Type{IdDict{K,V} where K}) where {V} = true
@inline is_arrayorvararg_ty(::Type{SparseArrays.CHOLMOD.Dense{T}}) where T = true
@static if VERSION < v"1.11-"
else
@inline is_arrayorvararg_ty(::Type{Memory{T}}) where T = true
end

@inline function datatype_fieldcount(t::Type{T}) where {T}
    return Base.datatype_fieldcount(t)
end

@inline function staticInTup(::Val{T}, tup::NTuple{N,Val}) where {T,N}
    any(ntuple(Val(N)) do i
        Base.@_inline_meta
        Val(T) == tup[i]
    end)
end

@inline function active_reg_recur(
    ::Type{ST},
    seen::Seen,
    world,
    ::Val{justActive},
    ::Val{UnionSret},
    ::Val{AbstractIsMixed},
) where {ST,Seen,justActive,UnionSret,AbstractIsMixed}
    if ST isa Union
        return forcefold(
            Val(
                active_reg_recur(
                    ST.a,
                    seen,
                    world,
                    Val(justActive),
                    Val(UnionSret),
                    Val(AbstractIsMixed),
                ),
            ),
            Val(
                active_reg_recur(
                    ST.b,
                    seen,
                    world,
                    Val(justActive),
                    Val(UnionSret),
                    Val(AbstractIsMixed),
                ),
            ),
        )
    end
    return active_reg_inner(
        ST,
        seen,
        world,
        Val(justActive),
        Val(UnionSret),
        Val(AbstractIsMixed),
    )
end

@inline is_vararg_tup(x) = false
@inline is_vararg_tup(::Type{Tuple{Vararg{T2}}}) where {T2} = true

@inline function active_reg_inner(
    ::Type{T},
    seen::ST,
    world::Union{Nothing,UInt},
    ::Val{justActive} = Val(false),
    ::Val{UnionSret} = Val(false),
    ::Val{AbstractIsMixed} = Val(false),
)::ActivityState where {ST,T,justActive,UnionSret,AbstractIsMixed}
    if T === Any
        if AbstractIsMixed
            return MixedState
        else
            return DupState
        end
    end

    if T === Union{}
        return AnyState
    end

    if T <: Complex && !(T isa UnionAll)
        return active_reg_inner(
            ptreltype(T),
            seen,
            world,
            Val(justActive),
            Val(UnionSret),
            Val(AbstractIsMixed),
        )
    end

    if T <: BigFloat
        return DupState
    end

    if T <: AbstractFloat
        return ActiveState
    end

    if T <: Ptr ||
       T <: Core.LLVMPtr ||
       T <: Base.RefValue ||
       T <: Array ||
       is_arrayorvararg_ty(T)
        if justActive
            return AnyState
        end

        if is_arrayorvararg_ty(T) &&
           active_reg_inner(
            ptreltype(T),
            seen,
            world,
            Val(justActive),
            Val(UnionSret),
            Val(AbstractIsMixed),
        ) == AnyState
            return AnyState
        else
            if AbstractIsMixed && is_vararg_tup(T)
                return MixedState
            else
                return DupState
            end
        end
    end

    if T <: Integer
        return AnyState
    end

    if isghostty(T) || Core.Compiler.isconstType(T) || T <: Type
        return AnyState
    end

    inactivety = if typeof(world) === Nothing
        EnzymeCore.EnzymeRules.inactive_type(T)
    else
        inmi = my_methodinstance(
            typeof(EnzymeCore.EnzymeRules.inactive_type),
            Tuple{Type{T}},
            world,
        )
        args = Any[EnzymeCore.EnzymeRules.inactive_type, T]
        GC.@preserve T begin
            ccall(
                :jl_invoke,
                Any,
                (Any, Ptr{Any}, Cuint, Any),
                EnzymeCore.EnzymeRules.inactive_type,
                args,
                length(args),
                inmi,
            )
        end
    end

    if inactivety
        return AnyState
    end

    # unknown number of fields
    if T isa UnionAll
        aT = Base.argument_datatype(T)
        if aT === nothing
            if AbstractIsMixed
                return MixedState
            else
                return DupState
            end
        end
        if datatype_fieldcount(aT) === nothing
            if AbstractIsMixed
                return MixedState
            else
                return DupState
            end
        end
    end

    if T isa Union
        # if sret union, the data is stored in a stack memory location and is therefore
        # not unique'd preventing the boxing of the union in the default case
        if UnionSret && is_sret_union(T)
            return active_reg_recur(
                T,
                seen,
                world,
                Val(justActive),
                Val(UnionSret),
                Val(AbstractIsMixed),
            )
        else
            if justActive
                return AnyState
            end
            if active_reg_inner(T.a, seen, world, Val(justActive), Val(UnionSret)) !=
               AnyState
                if AbstractIsMixed
                    return MixedState
                else
                    return DupState
                end
            end
            if active_reg_inner(T.b, seen, world, Val(justActive), Val(UnionSret)) !=
               AnyState
                if AbstractIsMixed
                    return MixedState
                else
                    return DupState
                end
            end
        end
        return AnyState
    end

    # if abstract it must be by reference
    if Base.isabstracttype(T)
        if AbstractIsMixed
            return MixedState
        else
            return DupState
        end
    end

    if ismutabletype(T)
        # if just looking for active of not
        # we know for a fact this isn't active
        if justActive
            return AnyState
        end
    end

    @assert !Base.isabstracttype(T)
    if !(Base.isconcretetype(T) || (T <: Tuple && T != Tuple) || T isa UnionAll)
        throw(AssertionError("Type $T is not concrete type or concrete tuple"))
    end

    nT = if T <: Tuple && T != Tuple && !(T isa UnionAll)
        Tuple{(
            ntuple(length(T.parameters)) do i
                Base.@_inline_meta
                sT = T.parameters[i]
                if sT isa TypeVar
                    Any
                elseif sT isa Core.TypeofVararg
                    Any
                else
                    sT
                end
            end
        )...}
    else
        T
    end

    if staticInTup(Val(nT), seen)
        return MixedState
    end

    seen2 = (Val(nT), seen...)

    fty = Merger{seen2,typeof(world),justActive,UnionSret,AbstractIsMixed}(world)

    ty = forcefold(Val(AnyState), ntuple(fty, Val(fieldcount(nT)))...)

    return ty
end

@inline @generated function active_reg_nothrow(::Type{T}, ::Val{world}) where {T,world}
    return active_reg_inner(T, (), world)
end

Base.@pure @inline function active_reg(
    ::Type{T},
    world::Union{Nothing,UInt} = nothing,
)::Bool where {T}
    seen = ()

    # check if it could contain an active
    if active_reg_inner(T, seen, world, Val(true)) == ActiveState #=justActive=#
        state = active_reg_inner(T, seen, world, Val(false)) #=justActive=#
        if state == ActiveState
            return true
        end
        @assert state == MixedState
        throw(
            AssertionError(
                string(T) *
                " has mixed internal activity types. See https://enzyme.mit.edu/julia/stable/faq/#Mixed-activity for more information",
            ),
        )
    else
        return false
    end
end

@inline function guaranteed_const(::Type{T}) where {T}
    rt = active_reg_nothrow(T, Val(nothing))
    res = rt == AnyState
    return res
end

@inline function guaranteed_const_nongen(::Type{T}, world) where {T}
    rt = active_reg_inner(T, (), world)
    res = rt == AnyState
    return res
end

# check if a value is guaranteed to be not contain active[register] data
# (aka not either mixed or active)
@inline function guaranteed_nonactive(::Type{T}) where {T}
    rt = Enzyme.Compiler.active_reg_nothrow(T, Val(nothing))
    return rt == Enzyme.Compiler.AnyState || rt == Enzyme.Compiler.DupState
end

"""
    Enzyme.guess_activity(::Type{T}, mode::Enzyme.Mode)

Try to guess the most appropriate [`Annotation`](@ref) for arguments of type `T` passed to [`autodiff`](@ref) with a given `mode`.
"""
@inline Enzyme.guess_activity(::Type{T}, mode::Enzyme.Mode) where {T} =
    guess_activity(T, convert(API.CDerivativeMode, mode))

@inline function Enzyme.guess_activity(::Type{T}, Mode::API.CDerivativeMode) where {T}
    ActReg = active_reg_inner(T, (), nothing)
    if ActReg == AnyState
        return Const{T}
    end
    if Mode == API.DEM_ForwardMode
        return Duplicated{T}
    else
        if ActReg == ActiveState
            return Active{T}
        elseif ActReg == MixedState
            return MixedDuplicated{T}
        else
            return Duplicated{T}
        end
    end
end

# User facing interface
abstract type AbstractThunk{FA,RT,TT,Width} end

struct CombinedAdjointThunk{PT,FA,RT,TT,Width,ReturnPrimal} <: AbstractThunk{FA,RT,TT,Width}
    adjoint::PT
end

struct ForwardModeThunk{PT,FA,RT,TT,Width,ReturnPrimal} <: AbstractThunk{FA,RT,TT,Width}
    adjoint::PT
end

struct AugmentedForwardThunk{PT,FA,RT,TT,Width,ReturnPrimal,TapeType} <:
       AbstractThunk{FA,RT,TT,Width}
    primal::PT
end

struct AdjointThunk{PT,FA,RT,TT,Width,TapeType} <: AbstractThunk{FA,RT,TT,Width}
    adjoint::PT
end

struct PrimalErrorThunk{PT,FA,RT,TT,Width,ReturnPrimal} <: AbstractThunk{FA,RT,TT,Width}
    adjoint::PT
end

@inline return_type(::AbstractThunk{FA,RT}) where {FA,RT} = RT
@inline return_type(
    ::Type{AugmentedForwardThunk{PT,FA,RT,TT,Width,ReturnPrimal,TapeType}},
) where {PT,FA,RT,TT,Width,ReturnPrimal,TapeType} = RT

@inline EnzymeRules.tape_type(
    ::Type{AugmentedForwardThunk{PT,FA,RT,TT,Width,ReturnPrimal,TapeType}},
) where {PT,FA,RT,TT,Width,ReturnPrimal,TapeType} = TapeType
@inline EnzymeRules.tape_type(
    ::AugmentedForwardThunk{PT,FA,RT,TT,Width,ReturnPrimal,TapeType},
) where {PT,FA,RT,TT,Width,ReturnPrimal,TapeType} = TapeType
@inline EnzymeRules.tape_type(
    ::Type{AdjointThunk{PT,FA,RT,TT,Width,TapeType}},
) where {PT,FA,RT,TT,Width,TapeType} = TapeType
@inline EnzymeRules.tape_type(
    ::AdjointThunk{PT,FA,RT,TT,Width,TapeType},
) where {PT,FA,RT,TT,Width,TapeType} = TapeType

using .JIT

include("jlrt.jl")

AnyArray(Length::Int) = NamedTuple{ntuple(i -> Symbol(i), Val(Length)),NTuple{Length,Any}}

struct EnzymeRuntimeException <: Base.Exception
    msg::Cstring
end

function Base.showerror(io::IO, ece::EnzymeRuntimeException)
    print(io, "Enzyme execution failed.\n")
    msg = Base.unsafe_string(ece.msg)
    print(io, msg, '\n')
end

struct EnzymeMutabilityException <: Base.Exception
    msg::Cstring
end

function Base.showerror(io::IO, ece::EnzymeMutabilityException)
    msg = Base.unsafe_string(ece.msg)
    print(io, msg, '\n')
end

struct EnzymeRuntimeActivityError <: Base.Exception
    msg::Cstring
end

function Base.showerror(io::IO, ece::EnzymeRuntimeActivityError)
    println(io, "Constant memory is stored (or returned) to a differentiable variable.")
    println(
        io,
        "As a result, Enzyme cannot provably ensure correctness and throws this error.",
    )
    println(
        io,
        "This might be due to the use of a constant variable as temporary storage for active memory (https://enzyme.mit.edu/julia/stable/faq/#Runtime-Activity).",
    )
    println(
        io,
        "If Enzyme should be able to prove this use non-differentable, open an issue!",
    )
    println(io, "To work around this issue, either:")
    println(
        io,
        " a) rewrite this variable to not be conditionally active (fastest, but requires a code change), or",
    )
    println(
        io,
        " b) set the Enzyme mode to turn on runtime activity (e.g. autodiff(set_runtime_activity(Reverse), ...) ). This will maintain correctness, but may slightly reduce performance.",
    )
    msg = Base.unsafe_string(ece.msg)
    print(io, msg, '\n')
end

struct EnzymeNoTypeError <: Base.Exception
    msg::Cstring
end

function Base.showerror(io::IO, ece::EnzymeNoTypeError)
    print(io, "Enzyme cannot deduce type\n")
    msg = Base.unsafe_string(ece.msg)
    print(io, msg, '\n')
end

struct EnzymeNoShadowError <: Base.Exception
    msg::Cstring
end

function Base.showerror(io::IO, ece::EnzymeNoShadowError)
    print(io, "Enzyme could not find shadow for value\n")
    msg = Base.unsafe_string(ece.msg)
    print(io, msg, '\n')
end

struct EnzymeNoDerivativeError <: Base.Exception
    msg::Cstring
end

function Base.showerror(io::IO, ece::EnzymeNoDerivativeError)
    msg = Base.unsafe_string(ece.msg)
    print(io, msg, '\n')
end

const JuliaEnzymeNameMap = Dict{String,Any}(
    "enz_val_true" => Val(true),
    "enz_val_false" => Val(false),
    "enz_val_1" => Val(1),
    "enz_any_array_1" => AnyArray(1),
    "enz_any_array_2" => AnyArray(2),
    "enz_any_array_3" => AnyArray(3),
    "enz_runtime_exc" => EnzymeRuntimeException,
    "enz_mut_exc" => EnzymeMutabilityException,
    "enz_runtime_activity_exc" => EnzymeRuntimeActivityError,
    "enz_no_type_exc" => EnzymeNoTypeError,
    "enz_no_shadow_exc" => EnzymeNoShadowError,
    "enz_no_derivative_exc" => EnzymeNoDerivativeError,
)

const JuliaGlobalNameMap = Dict{String,Any}(
    "jl_type_type" => Type,
    "jl_any_type" => Any,
    "jl_datatype_type" => DataType,
    "jl_methtable_type" => Core.MethodTable,
    "jl_symbol_type" => Symbol,
    "jl_simplevector_type" => Core.SimpleVector,
    "jl_nothing_type" => Nothing,
    "jl_tvar_type" => TypeVar,
    "jl_typeofbottom_type" => Core.TypeofBottom,
    "jl_bottom_type" => Union{},
    "jl_unionall_type" => UnionAll,
    "jl_uniontype_type" => Union,
    "jl_emptytuple_type" => Tuple{},
    "jl_emptytuple" => (),
    "jl_int8_type" => Int8,
    "jl_uint8_type" => UInt8,
    "jl_int16_type" => Int16,
    "jl_uint16_type" => UInt16,
    "jl_int32_type" => Int32,
    "jl_uint32_type" => UInt32,
    "jl_int64_type" => Int64,
    "jl_uint64_type" => UInt64,
    "jl_float16_type" => Float16,
    "jl_float32_type" => Float32,
    "jl_float64_type" => Float64,
    "jl_ssavalue_type" => Core.SSAValue,
    "jl_slotnumber_type" => Core.SlotNumber,
    "jl_argument_type" => Core.Argument,
    "jl_bool_type" => Bool,
    "jl_char_type" => Char,
    "jl_false" => false,
    "jl_true" => true,
    "jl_abstractstring_type" => AbstractString,
    "jl_string_type" => String,
    "jl_an_empty_string" => "",
    "jl_function_type" => Function,
    "jl_builtin_type" => Core.Builtin,
    "jl_module_type" => Core.Module,
    "jl_globalref_type" => Core.GlobalRef,
    "jl_ref_type" => Ref,
    "jl_pointer_typename" => Ptr,
    "jl_voidpointer_type" => Ptr{Nothing},
    "jl_abstractarray_type" => AbstractArray,
    "jl_densearray_type" => DenseArray,
    "jl_array_type" => Array,
    "jl_array_any_type" => Array{Any,1},
    "jl_array_symbol_type" => Array{Symbol,1},
    "jl_array_uint8_type" => Array{UInt8,1},

    # "jl_array_uint32_type" => Array{UInt32, 1},

    "jl_array_int32_type" => Array{Int32,1},
    "jl_expr_type" => Expr,
    "jl_method_type" => Method,
    "jl_method_instance_type" => Core.MethodInstance,
    "jl_code_instance_type" => Core.CodeInstance,
    "jl_const_type" => Core.Const,
    "jl_llvmpointer_type" => Core.LLVMPtr,
    "jl_namedtuple_type" => NamedTuple,
    "jl_task_type" => Task,
    "jl_uint8pointer_type" => Ptr{UInt8},
    "jl_nothing" => nothing,
    "jl_anytuple_type" => Tuple,
    "jl_vararg_type" => Core.TypeofVararg,
    "jl_opaque_closure_type" => Core.OpaqueClosure,
    "jl_array_uint64_type" => Array{UInt64,1},
    "jl_binding_type" => Core.Binding,
)

include("absint.jl")

# Force sret
struct Return2
    ret1::Any
    ret2::Any
end

function permit_inlining!(f::LLVM.Function)
    for bb in blocks(f), inst in instructions(bb)
        # remove illegal invariant.load and jtbaa_const invariants
        if isa(inst, LLVM.LoadInst)
            md = metadata(inst)
            if haskey(md, LLVM.MD_tbaa)
                modified = LLVM.Metadata(
                    ccall(
                        (:EnzymeMakeNonConstTBAA, API.libEnzyme),
                        LLVM.API.LLVMMetadataRef,
                        (LLVM.API.LLVMMetadataRef,),
                        md[LLVM.MD_tbaa],
                    ),
                )
                setindex!(md, modified, LLVM.MD_tbaa)
            end
            if haskey(md, LLVM.MD_invariant_load)
                delete!(md, LLVM.MD_invariant_load)
            end
        end
    end
end

struct Tape{TapeTy,ShadowTy,ResT}
    internal_tape::TapeTy
    shadow_return::ShadowTy
end

include("make_zero.jl")

function nested_codegen!(mode::API.CDerivativeMode, mod::LLVM.Module, f, tt, world)
    funcspec = my_methodinstance(typeof(f), tt, world)
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
        push!(
            attributes,
            StringAttribute("enzymejl_mi", string(convert(UInt, pointer_from_objref(mi)))),
        )
        push!(
            attributes,
            StringAttribute("enzymejl_rt", string(convert(UInt, unsafe_to_pointer(RT)))),
        )
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

function nested_codegen!(
    mode::API.CDerivativeMode,
    mod::LLVM.Module,
    funcspec::Core.MethodInstance,
    world,
)
    # TODO: Put a cache here index on `mod` and f->tt


    # 3) Use the MI to create the correct augmented fwd/reverse
    # TODO:
    #  - GPU support
    #  - When OrcV2 only use a MaterializationUnit to avoid mutation of the module here

    target = DefaultCompilerTarget()
    params = PrimalCompilerParams(mode)
    job = CompilerJob(funcspec, CompilerConfig(target, params; kernel = false), world)

    # TODO
    parent_job = nothing

    otherMod, meta = GPUCompiler.codegen(
        :llvm,
        job;
        optimize = false,
        cleanup = false,
        validate = false,
        parent_job = parent_job,
    )
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
    ir::Union{Nothing,String}
    bt::Union{Nothing,Vector{StackTraces.StackFrame}}
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

struct IllegalTypeAnalysisException <: CompilationException
    msg::String
    sval::String
    ir::Union{Nothing,String}
    bt::Union{Nothing,Vector{StackTraces.StackFrame}}
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
        print(io, "\nCaused by:")
        Base.show_backtrace(io, ece.bt)
        println(io)
    end
end

struct IllegalFirstPointerException <: CompilationException
    msg::String
    ir::Union{Nothing,String}
    bt::Union{Nothing,Vector{StackTraces.StackFrame}}
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
    ir::Union{Nothing,String}
    bt::Union{Nothing,Vector{StackTraces.StackFrame}}
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

parent_scope(val::LLVM.Function, depth = 0) = depth == 0 ? LLVM.parent(val) : val
parent_scope(val::LLVM.Module, depth = 0) = val
parent_scope(val::LLVM.Value, depth = 0) = parent_scope(LLVM.parent(val), depth + 1)
parent_scope(val::LLVM.Argument, depth = 0) =
    parent_scope(LLVM.Function(LLVM.API.LLVMGetParamParent(val)), depth + 1)

const CheckNan = Ref(false)
function julia_sanitize(
    orig::LLVM.API.LLVMValueRef,
    val::LLVM.API.LLVMValueRef,
    B::LLVM.API.LLVMBuilderRef,
    mask::LLVM.API.LLVMValueRef,
)::LLVM.API.LLVMValueRef
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

        stringv = "Enzyme: Found nan while computing derivative of " * string(orig)
        if orig !== nothing && isa(orig, LLVM.Instruction)
            bt = GPUCompiler.backtrace(orig)
            function printBT(io)
                print(io, "\nCaused by:")
                Base.show_backtrace(io, bt)
            end
            stringv *= sprint(io -> Base.show_backtrace(io, bt))
        end

        fn, _ = get_function!(mod, "julia.sanitize." * string(ty), FT)
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

                position!(builder, bad)

                emit_error(builder, nothing, sval, EnzymeNoDerivativeError)
                unreachable!(builder)
                dispose(builder)
            end
        end
        # val = 
        call!(B, FT, fn, LLVM.Value[val, globalstring_ptr!(B, stringv)])
    end
    return val.ref
end

function julia_error(
    cstr::Cstring,
    val::LLVM.API.LLVMValueRef,
    errtype::API.ErrorType,
    data::Ptr{Cvoid},
    data2::LLVM.API.LLVMValueRef,
    B::LLVM.API.LLVMBuilderRef,
)::LLVM.API.LLVMValueRef
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
        if occursin("No create nofree of empty function", msg) ||
           occursin("No forward mode derivative found for", msg) ||
           occursin("No augmented forward pass", msg) ||
           occursin("No reverse pass found", msg)
            ir = nothing
        end
        if B != C_NULL
            B = IRBuilder(B)
            msg2 = sprint() do io
                if ir !== nothing
                    print(io, "Current scope: \n")
                    print(io, ir)
                end
                print(io, '\n', msg, '\n')
                if bt !== nothing
                    Base.show_backtrace(io, bt)
                    println(io)
                end
            end
            emit_error(B, nothing, msg2, EnzymeNoDerivativeError)
            return C_NULL
        end
        throw(NoDerivativeException(msg, ir, bt))
    elseif errtype == API.ET_NoShadow
        gutils = GradientUtils(API.EnzymeGradientUtilsRef(data))

        msgN = sprint() do io::IO
            if isa(val, LLVM.Argument)
                fn = parent_scope(val)
                ir = string(LLVM.name(fn)) * string(function_type(fn))
                print(io, "Current scope: \n")
                print(io, ir)
            end
            if !isa(val, LLVM.Argument)
                print(io, "\n Inverted pointers: \n")
                ip = API.EnzymeGradientUtilsInvertedPointersToString(gutils)
                sval = Base.unsafe_string(ip)
                write(io, sval)
                API.EnzymeStringFree(ip)
            end
            print(io, '\n', msg, '\n')
            if bt !== nothing
                print(io, "\nCaused by:")
                Base.show_backtrace(io, bt)
                println(io)
            end
        end
        emit_error(IRBuilder(B), nothing, msgN, EnzymeNoShadowError)
        return LLVM.null(get_shadow_type(gutils, value_type(val))).ref
    elseif errtype == API.ET_IllegalTypeAnalysis
        data = API.EnzymeTypeAnalyzerRef(data)
        ip = API.EnzymeTypeAnalyzerToString(data)
        sval = Base.unsafe_string(ip)
        API.EnzymeStringFree(ip)

        if isa(val, LLVM.Instruction)
            mi, rt = enzyme_custom_extract_mi(
                LLVM.parent(LLVM.parent(val))::LLVM.Function,
                false,
            ) #=error=#
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
                print(io, "\nCaused by:")
                Base.show_backtrace(io, bt)
                println(io)
            end
            pscope = parent_scope(val)
            mi, rt = enzyme_custom_extract_mi(pscope, false) #=error=#
            if mi !== nothing
                println(io, "within ", mi)
            end
        end
        emit_error(B, nothing, msg2, EnzymeNoTypeError)
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
        gutils = GradientUtils(API.EnzymeGradientUtilsRef(data))
        # Ignore mismatched activity if phi/store of ghost
        seen = Dict{LLVM.Value,LLVM.Value}()
        illegal = false
        created = LLVM.Instruction[]
        world = enzyme_extract_world(LLVM.parent(position(IRBuilder(B))))
        width = get_width(gutils)
        function make_batched(cur, B)
            if width == 1
                return cur
            else
                shadowres = UndefValue(
                    LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(cur))),
                )
                for idx = 1:width
                    shadowres = insert_value!(B, shadowres, cur, idx - 1)
                    if isa(shadowres, LLVM.Instruction)
                        push!(created, shadowres)
                    end
                end
                return shadowres
            end
        end

        illegalVal = nothing

        function make_replacement(cur::LLVM.Value, prevbb)::LLVM.Value
            ncur = new_from_original(gutils, cur)
            if cur in keys(seen)
                return seen[cur]
            end

            legal, TT, byref = abs_typeof(cur, true)
            if legal
                if guaranteed_const_nongen(TT, world)
                    return make_batched(ncur, prevbb)
                end

                legal2, obj = absint(cur)

                # Only do so for the immediate operand/etc to a phi, since otherwise we will make multiple
                if legal2 &&
                   active_reg_inner(TT, (), world) == ActiveState &&
                   isa(cur, LLVM.ConstantExpr) &&
                   cur == data2
                    if width == 1
                        res = emit_allocobj!(prevbb, Base.RefValue{TT})
                        push!(created, res)
                        return res
                    else
                        shadowres = UndefValue(
                            LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(cur))),
                        )
                        for idx = 1:width
                            res = emit_allocobj!(prevbb, Base.RefValue{TT})
                            shadowres = insert_value!(prevbb, shadowres, res, idx - 1)
                            push!(created, shadowres)
                        end
                        return shadowres
                    end
                end

                badval = if legal2
                    string(obj) * " of type" * " " * string(TT)
                else
                    "Unknown object of type" * " " * string(TT)
                end
                illegalVal = cur
                illegal = true
                return make_batched(ncur, prevbb)
            end

            if isa(cur, LLVM.PointerNull)
                return make_batched(ncur, prevbb)
            end
            if isa(cur, LLVM.UndefValue)
                return make_batched(ncur, prevbb)
            end
            @static if LLVM.version() >= v"12"
                if isa(cur, LLVM.PoisonValue)
                    return make_batched(ncur, prevbb)
                end
            end
            if isa(cur, LLVM.ConstantAggregateZero)
                return make_batched(ncur, prevbb)
            end
            if isa(cur, LLVM.ConstantAggregate)
                return make_batched(ncur, prevbb)
            end
            if isa(cur, LLVM.ConstantInt)
                if convert(UInt64, cur) == 0
                    return make_batched(ncur, prevbb)
                end
            end
            if isa(cur, LLVM.ConstantFP)
                return make_batched(ConstantFP(value_type(cur), 0), prevbb)
            end
            if isa(cur, LLVM.ConstantDataSequential)
                cvals = LLVM.Value[]
                changed = false
                for v in collect(cur)
                    tmp = make_replacement(v, prevbb)
                    if illegal
                        return ncur
                    end
                    if v != tmp
                        changed = true
                    end
                    push!(cvals, tmp)
                end

                cur2 = if changed
                    illegalVal = cur
                    illegal = true
                    # TODO replace with correct insertions/splats
                    ncur
                else
                    make_batched(ncur, prevbb)
                end
                return cur2
            end
            if isa(cur, LLVM.ConstantInt)
                if LLVM.width(value_type(cur)) <= sizeof(Int) * 8
                    return make_batched(ncur, prevbb)
                end
                if LLVM.width(value_type(cur)) == sizeof(Int) * 8 &&
                   abs(convert(Int, cur)) < 10000
                    return make_batched(ncur, prevbb)
                end
                # if storing a constant int as a non-pointer, presume it is not a GC'd var and is safe
                # for activity state to mix
                if isa(val, LLVM.StoreInst)
                    operands(val)[1] == cur &&
                        !isa(value_type(operands(val)[1]), LLVM.PointerType)
                    return make_batched(ncur, prevbb)
                end
            end

            if isa(cur, LLVM.SelectInst)
                lhs = make_replacement(operands(cur)[2], prevbb)
                if illegal
                    return ncur
                end
                rhs = make_replacement(operands(cur)[3], prevbb)
                if illegal
                    return ncur
                end
                if lhs == operands(cur)[2] && rhs == operands(cur)[3]
                    return make_batched(ncur, prevbb)
                end
                if width == 1
                    nv = select!(
                        prevbb,
                        new_from_original(gutils, operands(cur)[1]),
                        lhs,
                        rhs,
                    )
                    push!(created, nv)
                    seen[cur] = nv
                    return nv
                else
                    shadowres = LLVM.UndefValue(value_type(lhs))
                    for idx = 1:width
                        shadowres = insert_value!(
                            prevbb,
                            shadowres,
                            select!(
                                prevbb,
                                new_from_original(gutils, operands(cur)[1]),
                                extract_value!(prevbb, lhs, idx - 1),
                                extract_value!(prevbb, rhs, idx - 1),
                            ),
                            idx - 1,
                        )
                        if isa(shadowres, LLVM.Instruction)
                            push!(created, shadowres)
                        end
                    end
                    return shadowres
                end
            end

            if isa(cur, LLVM.InsertValueInst)
                lhs = make_replacement(operands(cur)[1], prevbb)
                if illegal
                    return ncur
                end
                rhs = make_replacement(operands(cur)[2], prevbb)
                if illegal
                    return ncur
                end
                if lhs == operands(cur)[1] && rhs == operands(cur)[2]
                    return make_batched(ncur, prevbb)
                end
                inds = LLVM.API.LLVMGetIndices(cur.ref)
                ninds = LLVM.API.LLVMGetNumIndices(cur.ref)
                jinds = Cuint[unsafe_load(inds, i) for i = 1:ninds]
                if width == 1
                    nv = API.EnzymeInsertValue(prevbb, lhs, rhs, jinds)
                    push!(created, nv)
                    seen[cur] = nv
                    return nv
                else
                    shadowres = lhs
                    for idx = 1:width
                        jindsv = copy(jinds)
                        pushfirst!(jindsv, idx - 1)
                        shadowres = API.EnzymeInsertValue(
                            prevbb,
                            shadowres,
                            extract_value!(prevbb, rhs, idx - 1),
                            jindsv,
                        )
                        if isa(shadowres, LLVM.Instruction)
                            push!(created, shadowres)
                        end
                    end
                    return shadowres
                end
            end

            if isa(cur, LLVM.PHIInst)
                Bphi = IRBuilder()
                position!(Bphi, ncur)
                shadowty = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(cur)))
                phi2 = phi!(Bphi, shadowty, "tempphi" * LLVM.name(cur))
                seen[cur] = phi2
                changed = false
                recsize = length(created) + 1
                for (v, bb) in LLVM.incoming(cur)
                    B2 = IRBuilder()
                    position!(B2, new_from_original(gutils, last(instructions(bb))))
                    tmp = make_replacement(v, B2)
                    if illegal
                        changed = true
                        break
                    end
                    @assert value_type(tmp) == shadowty
                    if tmp != new_from_original(gutils, v) && v != cur
                        changed = true
                    end
                    push!(LLVM.incoming(phi2), (tmp, new_from_original(gutils, bb)))
                end
                if !changed || illegal
                    LLVM.API.LLVMInstructionEraseFromParent(phi2)
                    seen[cur] = ncur
                    plen = length(created)
                    for i = recsize:plen
                        u = created[i]
                        replace_uses!(u, LLVM.UndefValue(value_type(u)))
                    end
                    for i = recsize:plen
                        u = created[i]
                        LLVM.API.LLVMInstructionEraseFromParent(u)
                    end
                    for i = recsize:plen
                        pop!(created)
                    end
                    return illegal ? ncur : make_batched(ncur, prevbb)
                end
                push!(created, phi2)
                return phi2
            end

            illegal = true
            illegalVal = cur
            return ncur
        end

        b = IRBuilder(B)
        replacement = make_replacement(data2, b)

        if !illegal
            return replacement.ref
        end
        for u in created
            replace_uses!(u, LLVM.UndefValue(value_type(u)))
        end
        for u in created
            LLVM.API.LLVMInstructionEraseFromParent(u)
        end
        if LLVM.API.LLVMIsAReturnInst(val) != C_NULL
            mi, rt = enzyme_custom_extract_mi(
                LLVM.parent(LLVM.parent(val))::LLVM.Function,
                false,
            ) #=error=#
            if mi !== nothing && isghostty(rt)
                return C_NULL
            end
        end
        msg2 = sprint() do io
            print(io, msg)
            println(io)
            if badval !== nothing
                println(io, " value=" * badval)
            else
                ttval = val
                if isa(ttval, LLVM.StoreInst)
                    ttval = operands(ttval)[1]
                end
                tt = TypeTree(API.EnzymeGradientUtilsAllocAndGetTypeTree(gutils, ttval))
                st = API.EnzymeTypeTreeToString(tt)
                print(io, "Type tree: ")
                println(io, Base.unsafe_string(st))
                API.EnzymeStringFree(st)
            end
            if illegalVal !== nothing
                println(io, " llvalue=" * string(illegalVal))
            end
            if bt !== nothing
                Base.show_backtrace(io, bt)
            end
        end
        emit_error(b, nothing, msg2, EnzymeRuntimeActivityError)
        return C_NULL
    elseif errtype == API.ET_GetIndexError
        @assert B != C_NULL
        B = IRBuilder(B)
        msg5 = sprint() do io::IO
            print(io, "Enzyme internal error\n")
            print(io, msg, '\n')
            if bt !== nothing
                print(io, "\nCaused by:")
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
any_jltypes(Type::Union{LLVM.VectorType,LLVM.ArrayType}) = any_jltypes(eltype(Type))
any_jltypes(::LLVM.IntegerType) = false
any_jltypes(::LLVM.FloatingPointType) = false
any_jltypes(::LLVM.VoidType) = false

@inline any_jltypes(::Type{Nothing}) = false
@inline any_jltypes(::Type{T}) where {T<:AbstractFloat} = false
@inline any_jltypes(::Type{T}) where {T<:Integer} = false
@inline any_jltypes(::Type{Complex{T}}) where {T} = any_jltypes(T)
@inline any_jltypes(::Type{Tuple{}}) = false
@inline any_jltypes(::Type{NTuple{Size,T}}) where {Size,T} = any_jltypes(T)
@inline any_jltypes(::Type{Core.LLVMPtr{T,Addr}}) where {T,Addr} = 10 <= Addr <= 12
@inline any_jltypes(::Type{Any}) = true
@inline any_jltypes(::Type{NamedTuple{A,B}}) where {A,B} =
    any(any_jltypes(b) for b in B.parameters)
@inline any_jltypes(::Type{T}) where {T<:Tuple} = any(any_jltypes(b) for b in T.parameters)

nfields(Type::LLVM.StructType) = length(LLVM.elements(Type))
nfields(Type::LLVM.VectorType) = size(Type)
nfields(Type::LLVM.ArrayType) = length(Type)
nfields(Type::LLVM.PointerType) = 1

mutable struct EnzymeTapeToLoad{T}
    data::T
end
Base.eltype(::EnzymeTapeToLoad{T}) where {T} = T

const TapeTypes = Dict{String,DataType}()

base_type(T::UnionAll) = base_type(T.body)
base_type(T::DataType) = T

const WideIntWidths = [256, 512, 1024, 2048]

let
    for n ∈ WideIntWidths
        let T = Symbol(:UInt, n)
            eval(quote
                primitive type $T <: Unsigned $n end
            end)
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
        for i = 1:nelems
            e = LLVM.API.LLVMStructGetTypeAtIndex(Type, i - 1)
            T, sub = to_tape_type(e)
            containsAny |= sub
            push!(tys, T)
            push!(syms, Symbol(i))
        end
        Tup = Tuple{tys...}
        if containsAny
            res = (syms...,)
            return NamedTuple{res,Tup}, false
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
            tkind2 = LLVM.API.LLVMGetTypeKind(e)
            if tkind2 == LLVM.API.LLVMFunctionTypeKind
                return Core.LLVMPtr{Cvoid,Int(addrspace)}, false
            else
                return Core.LLVMPtr{to_tape_type(e)[1],Int(addrspace)}, false
            end
        end
    end
    if tkind == LLVM.API.LLVMArrayTypeKind
        e = LLVM.API.LLVMGetElementType(Type)
        T, sub = to_tape_type(e)
        len = Int(LLVM.API.LLVMGetArrayLength(Type))
        Tup = NTuple{len,T}
        if sub
            return NamedTuple{ntuple(Core.Symbol, Val(len)),Tup}, false
        else
            return Tup, false
        end
    end
    if tkind == LLVM.API.LLVMVectorTypeKind
        e = LLVM.API.LLVMGetElementType(Type)
        T, sub = to_tape_type(e)
        len = Int(LLVM.API.LLVMGetVectorSize(Type))
        Tup = NTuple{len,T}
        if sub
            return NamedTuple{ntuple(Core.Symbol, Val(len)),Tup}, false
        else
            return Tup, false
        end
    end
    if tkind == LLVM.API.LLVMIntegerTypeKind
        N = LLVM.API.LLVMGetIntTypeWidth(Type)
        if N == 1
            return Bool, false
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
    error("Can't construct tape type for $Type $(string(Type)) $tkind")
end

function tape_type(LLVMType::LLVM.LLVMType)
    TT, isAny = to_tape_type(LLVMType.ref)
    if isAny
        return AnonymousStruct(Tuple{Any})
    end
    return TT
end

from_tape_type(::Type{T}) where {T<:AbstractFloat} = convert(LLVMType, T)
from_tape_type(::Type{T}) where {T<:Integer} = convert(LLVMType, T)
from_tape_type(::Type{NTuple{Size,T}}) where {Size,T} =
    LLVM.ArrayType(from_tape_type(T), Size)
from_tape_type(::Type{Core.LLVMPtr{T,Addr}}) where {T,Addr} =
    LLVM.PointerType(from_tape_type(UInt8), Addr)
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
current_task_offset() =
    -(unsafe_load(cglobal(:jl_task_gcstack_offset, Cint)) ÷ sizeof(Ptr{Cvoid}))

# See get_current_ptls_from_task (used from 1.7+)
current_ptls_offset() =
    unsafe_load(cglobal(:jl_task_ptls_offset, Cint)) ÷ sizeof(Ptr{Cvoid})

function store_nonjl_types!(B, startval, p)
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    vals = LLVM.Value[]
    if p != nothing
        push!(vals, p)
    end
    todo = Tuple{Tuple,LLVM.Value}[((), startval)]
    while length(todo) != 0
        path, cur = popfirst!(todo)
        ty = value_type(cur)
        if isa(ty, LLVM.PointerType)
            if any_jltypes(ty)
                continue
            end
        end
        if isa(ty, LLVM.ArrayType)
            if any_jltypes(ty)
                for i = 1:length(ty)
                    ev = extract_value!(B, cur, i - 1)
                    push!(todo, ((path..., i - 1), ev))
                end
                continue
            end
        end
        if isa(ty, LLVM.StructType)
            if any_jltypes(ty)
                for (i, t) in enumerate(LLVM.elements(ty))
                    ev = extract_value!(B, cur, i - 1)
                    push!(todo, ((path..., i - 1), ev))
                end
                continue
            end
        end
        parray = LLVM.Value[LLVM.ConstantInt(LLVM.IntType(64), 0)]
        for v in path
            push!(parray, LLVM.ConstantInt(LLVM.IntType(32), v))
        end
        gptr = gep!(B, value_type(startval), p, parray)
        st = store!(B, cur, gptr)
    end
    return
end

function get_julia_inner_types(B, p, startvals...; added = LLVM.API.LLVMValueRef[])
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    vals = LLVM.Value[]
    if p != nothing
        push!(vals, p)
    end
    todo = LLVM.Value[startvals...]
    while length(todo) != 0
        cur = popfirst!(todo)
        ty = value_type(cur)
        if isa(ty, LLVM.PointerType)
            if any_jltypes(ty)
                if addrspace(ty) != Tracked
                    cur = addrspacecast!(
                        B,
                        cur,
                        LLVM.PointerType(eltype(ty), Tracked),
                        LLVM.name(cur) * ".innertracked",
                    )
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
                for i = 1:length(ty)
                    ev = extract_value!(B, cur, i - 1)
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
                    ev = extract_value!(B, cur, i - 1)
                    if isa(ev, LLVM.Instruction)
                        push!(added, ev.ref)
                    end
                    push!(todo, ev)
                end
            end
            continue
        end
        if isa(ty, LLVM.IntegerType)
            continue
        end
        if isa(ty, LLVM.FloatingPointType)
            continue
        end
        msg = sprint() do io
            println(io, "Enzyme illegal subtype")
            println(io, "ty=", ty)
            println(io, "cur=", cur)
            println(io, "p=", p)
            println(io, "startvals=", startvals)
        end
        throw(AssertionError(msg))
    end
    return vals
end

function julia_post_cache_store(
    SI::LLVM.API.LLVMValueRef,
    B::LLVM.API.LLVMBuilderRef,
    R2,
)::Ptr{LLVM.API.LLVMValueRef}
    B = LLVM.IRBuilder(B)
    SI = LLVM.Instruction(SI)
    v = operands(SI)[1]
    p = operands(SI)[2]
    added = LLVM.API.LLVMValueRef[]
    while true
        if isa(p, LLVM.GetElementPtrInst) ||
           isa(p, LLVM.BitCastInst) ||
           isa(p, LLVM.AddrSpaceCastInst)
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

        vals = get_julia_inner_types(B, p, v, added = added)
        r = emit_writebarrier!(B, vals)
        @assert isa(r, LLVM.Instruction)
        push!(added, r.ref)
    end
    if R2 != C_NULL
        unsafe_store!(R2, length(added))
        ptr = Base.unsafe_convert(
            Ptr{LLVM.API.LLVMValueRef},
            Libc.malloc(sizeof(LLVM.API.LLVMValueRef) * length(added)),
        )
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
function julia_undef_value_for_type(
    mod::LLVM.API.LLVMModuleRef,
    Ty::LLVM.API.LLVMTypeRef,
    forceZero::UInt8,
)::LLVM.API.LLVMValueRef
    ty = LLVM.LLVMType(Ty)
    if !any_jltypes(ty)
        if forceZero != 0
            return LLVM.null(ty).ref
        else
            return UndefValue(ty).ref
        end
    end
    if isa(ty, LLVM.PointerType)
        val = unsafe_nothing_to_llvm(LLVM.Module(mod))
        if !is_opaque(ty)
            val = const_pointercast(val, LLVM.PointerType(eltype(ty), Tracked))
        end
        if addrspace(ty) != Tracked
            val = const_addrspacecast(val, ty)
        end
        return val.ref
    end
    if isa(ty, LLVM.ArrayType)
        st = LLVM.Value(julia_undef_value_for_type(mod, eltype(ty).ref, forceZero))
        return ConstantArray(eltype(ty), [st for i = 1:length(ty)]).ref
    end
    if isa(ty, LLVM.StructType)
        vals = LLVM.Constant[]
        for st in LLVM.elements(ty)
            push!(vals, LLVM.Value(julia_undef_value_for_type(mod, st.ref, forceZero)))
        end
        return ConstantStruct(ty, vals).ref
    end
    throw(AssertionError("Unknown type to val: $(Ty)"))
end

function shadow_alloc_rewrite(V::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef)
    V = LLVM.CallInst(V)
    gutils = GradientUtils(gutils)
    mode = get_mode(gutils)
    if mode == API.DEM_ReverseModePrimal ||
       mode == API.DEM_ReverseModeGradient ||
       mode == API.DEM_ReverseModeCombined
        fn = LLVM.parent(LLVM.parent(V))
        world = enzyme_extract_world(fn)
        has, Ty, byref = abs_typeof(V)
        if !has
            throw(AssertionError("$(string(fn))\n Allocation could not have its type statically determined $(string(V))"))
        end
        rt = active_reg_inner(Ty, (), world)
        if rt == ActiveState || rt == MixedState
            B = LLVM.IRBuilder()
            position!(B, V)
            operands(V)[3] = unsafe_to_llvm(B, Base.RefValue{Ty})
        end
    end
    nothing
end

function julia_allocator(
    B::LLVM.API.LLVMBuilderRef,
    LLVMType::LLVM.API.LLVMTypeRef,
    Count::LLVM.API.LLVMValueRef,
    AlignedSize::LLVM.API.LLVMValueRef,
    IsDefault::UInt8,
    ZI,
)
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
            fill_val = unsafe_to_llvm(B, nothing)
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

    todo = Tuple{Vector{LLVM.Value},LLVM.LLVMType,DataType}[(
        LLVM.Value[idx],
        LLVMType,
        jlType,
    )]

    while length(todo) != 0
        path, ty, jlty = popfirst!(todo)
        if isa(ty, LLVM.PointerType)
            if any_jltypes(ty)
                loc = gep!(builder, LLVMType, nobj, path)
                mod = LLVM.parent(LLVM.parent(Base.position(builder)))
                fill_val = unsafe_nothing_to_llvm(mod)
                loc = bitcast!(
                    builder,
                    loc,
                    LLVM.PointerType(T_prjlvalue, addrspace(value_type(loc))),
                )
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
            for i = 1:length(ty)
                npath = copy(path)
                push!(npath, LLVM.ConstantInt(LLVM.IntType(32), i - 1))
                push!(todo, (npath, eltype(ty), eltype(jlty)))
            end
            continue
        end
        if isa(ty, LLVM.VectorType)
            for i = 1:size(ty)
                npath = copy(path)
                push!(npath, LLVM.ConstantInt(LLVM.IntType(32), i - 1))
                push!(todo, (npath, eltype(ty), eltype(jlty)))
            end
            continue
        end
        if isa(ty, LLVM.StructType)
            i = 1
            for ii = 1:fieldcount(jlty)
                jlet = typed_fieldtype(jlty, ii)
                if isghostty(jlet) || Core.Compiler.isconstType(jlet)
                    continue
                end
                t = LLVM.elements(ty)[i]
                npath = copy(path)
                push!(npath, LLVM.ConstantInt(LLVM.IntType(32), i - 1))
                push!(todo, (npath, t, jlet))
                i += 1
            end
            @assert i == Int(length(LLVM.elements(ty))) + 1
            continue
        end
    end
    return nothing

end


function zero_allocation(
    B::LLVM.IRBuilder,
    jlType,
    LLVMType,
    obj,
    AlignedSize,
    Size,
    zeroAll::Bool,
)::LLVM.API.LLVMValueRef
    func = LLVM.parent(position(B))
    mod = LLVM.parent(func)
    T_int8 = LLVM.Int8Type()

    T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_prjlvalue_UT = LLVM.PointerType(T_jlvalue)

    wrapper_f = LLVM.Function(
        mod,
        "zeroType",
        LLVM.FunctionType(LLVM.VoidType(), [value_type(obj), T_int8, value_type(Size)]),
    )
    push!(function_attributes(wrapper_f), StringAttribute("enzyme_math", "enzyme_zerotype"))
    push!(function_attributes(wrapper_f), StringAttribute("enzyme_inactive"))
    push!(function_attributes(wrapper_f), StringAttribute("enzyme_no_escaping_allocation"))
    push!(function_attributes(wrapper_f), EnumAttribute("alwaysinline", 0))
    push!(function_attributes(wrapper_f), EnumAttribute("nofree", 0))

    if LLVM.version().major <= 15
        push!(function_attributes(wrapper_f), EnumAttribute("argmemonly", 0))
        push!(function_attributes(wrapper_f), EnumAttribute("writeonly", 0))
    else
        push!(function_attributes(wrapper_f), EnumAttribute("memory", WriteOnlyArgMemEffects.data))
    end
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
        nobj = pointercast!(
            builder,
            nobj,
            LLVM.PointerType(LLVMType, addrspace(value_type(nobj))),
        )

        LLVM.br!(builder, loop)
        position!(builder, loop)
        idx = LLVM.phi!(builder, value_type(Size))
        inc = add!(builder, idx, LLVM.ConstantInt(value_type(Size), 1))
        append!(
            LLVM.incoming(idx),
            [(LLVM.ConstantInt(value_type(Size), 0), entry), (inc, loop)],
        )

        zero_single_allocation(builder, jlType, LLVMType, nobj, zeroAll, idx)

        br!(
            builder,
            icmp!(
                builder,
                LLVM.API.LLVMIntEQ,
                inc,
                LLVM.Value(LLVM.API.LLVMBuildExactUDiv(builder, nsize, AlignedSize, "")),
            ),
            exit,
            loop,
        )
        position!(builder, exit)

        ret!(builder)

        dispose(builder)
    end
    return call!(
        B,
        LLVM.function_type(wrapper_f),
        wrapper_f,
        [obj, LLVM.ConstantInt(T_int8, 0), Size],
    ).ref
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
            GPUCompiler.@safe_error "Enzyme aligned size and Julia size disagree" AlignedSize =
                convert(Int, AlignedSize) esizeof(TT) fieldtypes(TT)
            emit_error(B, nothing, "Enzyme: Tape allocation failed.") # TODO: Pick appropriate orig
            return LLVM.API.LLVMValueRef(LLVM.UndefValue(LLVMType).ref)
        end
        @assert esizeof(TT) == convert(Int, AlignedSize)
        if Count isa LLVM.ConstantInt
            N = convert(Int, Count)

            ETT = N == 1 ? TT : NTuple{N,TT}
            if sizeof(ETT) != N * convert(Int, AlignedSize)
                GPUCompiler.@safe_error "Size of Enzyme tape is incorrect. Please report this issue" ETT sizeof(
                    ETT,
                ) TargetSize = N * convert(Int, AlignedSize) LLVMType
                emit_error(B, nothing, "Enzyme: Tape allocation failed.") # TODO: Pick appropriate orig

                return LLVM.API.LLVMValueRef(LLVM.UndefValue(LLVMType).ref)
            end

            # Obtain tag
            tag = unsafe_to_llvm(B, ETT)
        else
            if sizeof(Int) == sizeof(Int64)
                boxed_count = emit_box_int64!(B, Count)
            else
                T_size_t = convert(LLVM.LLVMType, Int)
                Count = trunc!(B, Count, T_size_t)
                boxed_count = emit_box_int32!(B, Count)
            end
            tag = emit_apply_type!(B, NTuple, (boxed_count, unsafe_to_llvm(B, TT)))
        end

        # Check if Julia version has https://github.com/JuliaLang/julia/pull/46914
        # and also https://github.com/JuliaLang/julia/pull/47076
        # and also https://github.com/JuliaLang/julia/pull/48620
        @static if VERSION >= v"1.10.5"
            needs_dynamic_size_workaround = false
        else
            needs_dynamic_size_workaround =
                !isa(Size, LLVM.ConstantInt) || convert(Int, Size) != 1
        end

        T_size_t = convert(LLVM.LLVMType, Int)
        allocSize = if value_type(Size) != T_size_t
            trunc!(B, Size, T_size_t)
        else
            Size
        end

        obj = emit_allocobj!(B, tag, allocSize, needs_dynamic_size_workaround)

        if ZI != C_NULL
            unsafe_store!(
                ZI,
                zero_allocation(B, TT, LLVMType, obj, AlignedSize, Size, false),
            ) #=ZeroAll=#
        end
        AS = Tracked
    else
        ptr8 = LLVM.PointerType(LLVM.IntType(8))
        mallocF, fty =
            get_function!(mod, "malloc", LLVM.FunctionType(ptr8, [value_type(Count)]))

        obj = call!(B, fty, mallocF, [Size])
        # if ZI != C_NULL
        #     unsafe_store!(ZI, LLVM.memset!(B, obj,  LLVM.ConstantInt(T_int8, 0),
        #                                           Size,
        #                                          #=align=#0 ).ref)
        # end
        AS = 0
    end

    LLVM.API.LLVMAddCallSiteAttribute(
        obj,
        LLVM.API.LLVMAttributeReturnIndex,
        EnumAttribute("noalias"),
    )
    LLVM.API.LLVMAddCallSiteAttribute(
        obj,
        LLVM.API.LLVMAttributeReturnIndex,
        EnumAttribute("nonnull"),
    )
    if isa(Count, LLVM.ConstantInt)
        val = convert(UInt, AlignedSize)
        val *= convert(UInt, Count)
        LLVM.API.LLVMAddCallSiteAttribute(
            obj,
            LLVM.API.LLVMAttributeReturnIndex,
            EnumAttribute("dereferenceable", val),
        )
        LLVM.API.LLVMAddCallSiteAttribute(
            obj,
            LLVM.API.LLVMAttributeReturnIndex,
            EnumAttribute("dereferenceable_or_null", val),
        )
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
        LLVM.API.LLVMAddCallSiteAttribute(
            callf,
            LLVM.API.LLVMAttributeIndex(1),
            EnumAttribute("nonnull"),
        )
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
    bts = sprint(io -> Base.show_backtrace(io, bt))
    fmt = globalstring_ptr!(B, "%s:\nBacktrace\n" * bts)

    funcT = LLVM.FunctionType(
        LLVM.VoidType(),
        LLVMType[LLVM.PointerType(LLVM.Int8Type())],
        vararg = true,
    )
    func, _ = get_function!(mod, "jl_errorf", funcT, [EnumAttribute("noreturn")])

    call!(B, funcT, func, LLVM.Value[fmt, LLVM.Value(V)])
    return nothing
end

include("rules/allocrules.jl")
include("rules/llvmrules.jl")

for (k, v) in (
    ("enz_runtime_newtask_fwd", Enzyme.Compiler.runtime_newtask_fwd),
    ("enz_runtime_newtask_augfwd", Enzyme.Compiler.runtime_newtask_augfwd),
    ("enz_runtime_generic_fwd", Enzyme.Compiler.runtime_generic_fwd),
    ("enz_runtime_generic_augfwd", Enzyme.Compiler.runtime_generic_augfwd),
    ("enz_runtime_generic_rev", Enzyme.Compiler.runtime_generic_rev),
    ("enz_runtime_iterate_fwd", Enzyme.Compiler.runtime_iterate_fwd),
    ("enz_runtime_iterate_augfwd", Enzyme.Compiler.runtime_iterate_augfwd),
    ("enz_runtime_iterate_rev", Enzyme.Compiler.runtime_iterate_rev),
    ("enz_runtime_newstruct_augfwd", Enzyme.Compiler.runtime_newstruct_augfwd),
    ("enz_runtime_newstruct_rev", Enzyme.Compiler.runtime_newstruct_rev),
    ("enz_runtime_tuple_augfwd", Enzyme.Compiler.runtime_tuple_augfwd),
    ("enz_runtime_tuple_rev", Enzyme.Compiler.runtime_tuple_rev),
    ("enz_runtime_jl_getfield_aug", Enzyme.Compiler.rt_jl_getfield_aug),
    ("enz_runtime_jl_getfield_rev", Enzyme.Compiler.rt_jl_getfield_rev),
    ("enz_runtime_idx_jl_getfield_aug", Enzyme.Compiler.idx_jl_getfield_aug),
    ("enz_runtime_idx_jl_getfield_rev", Enzyme.Compiler.idx_jl_getfield_aug),
    ("enz_runtime_jl_setfield_aug", Enzyme.Compiler.rt_jl_setfield_aug),
    ("enz_runtime_jl_setfield_rev", Enzyme.Compiler.rt_jl_setfield_rev),
    ("enz_runtime_error_if_differentiable", Enzyme.Compiler.error_if_differentiable),
    ("enz_runtime_error_if_active", Enzyme.Compiler.error_if_active),
)
    JuliaEnzymeNameMap[k] = v
end

function __init__()
    API.memmove_warning!(false)
    API.typeWarning!(false)
    API.EnzymeSetHandler(
        @cfunction(
            julia_error,
            LLVM.API.LLVMValueRef,
            (
                Cstring,
                LLVM.API.LLVMValueRef,
                API.ErrorType,
                Ptr{Cvoid},
                LLVM.API.LLVMValueRef,
                LLVM.API.LLVMBuilderRef,
            )
        )
    )
    API.EnzymeSetSanitizeDerivatives(
        @cfunction(
            julia_sanitize,
            LLVM.API.LLVMValueRef,
            (
                LLVM.API.LLVMValueRef,
                LLVM.API.LLVMValueRef,
                LLVM.API.LLVMBuilderRef,
                LLVM.API.LLVMValueRef,
            )
        )
    )
    API.EnzymeSetRuntimeInactiveError(
        @cfunction(
            emit_inacterror,
            Cvoid,
            (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, LLVM.API.LLVMValueRef)
        )
    )
    API.EnzymeSetDefaultTapeType(
        @cfunction(
            julia_default_tape_type,
            LLVM.API.LLVMTypeRef,
            (LLVM.API.LLVMContextRef,)
        )
    )
    API.EnzymeSetCustomAllocator(
        @cfunction(
            julia_allocator,
            LLVM.API.LLVMValueRef,
            (
                LLVM.API.LLVMBuilderRef,
                LLVM.API.LLVMTypeRef,
                LLVM.API.LLVMValueRef,
                LLVM.API.LLVMValueRef,
                UInt8,
                Ptr{LLVM.API.LLVMValueRef},
            )
        )
    )
    API.EnzymeSetCustomDeallocator(
        @cfunction(
            julia_deallocator,
            LLVM.API.LLVMValueRef,
            (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef)
        )
    )
    API.EnzymeSetPostCacheStore(
        @cfunction(
            julia_post_cache_store,
            Ptr{LLVM.API.LLVMValueRef},
            (LLVM.API.LLVMValueRef, LLVM.API.LLVMBuilderRef, Ptr{UInt64})
        )
    )

    API.EnzymeSetCustomZero(
        @cfunction(
            zero_allocation,
            Cvoid,
            (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMTypeRef, LLVM.API.LLVMValueRef, UInt8)
        )
    )
    API.EnzymeSetFixupReturn(
        @cfunction(
            fixup_return,
            LLVM.API.LLVMValueRef,
            (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef)
        )
    )
    API.EnzymeSetUndefinedValueForType(
        @cfunction(
            julia_undef_value_for_type,
            LLVM.API.LLVMValueRef,
            (LLVM.API.LLVMModuleRef, LLVM.API.LLVMTypeRef, UInt8)
        )
    )
    API.EnzymeSetShadowAllocRewrite(
        @cfunction(
            shadow_alloc_rewrite,
            Cvoid,
            (LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef)
        )
    )
    register_alloc_rules()
    register_llvm_rules()

    # Force compilation of AD stack
    # thunk = Enzyme.Compiler.thunk(Enzyme.Compiler.fspec(typeof(Base.identity), Tuple{Active{Float64}}), Const{typeof(Base.identity)}, Active, Tuple{Active{Float64}}, #=Split=# Val(Enzyme.API.DEM_ReverseModeCombined), #=width=#Val(1), #=ModifiedBetween=#Val((false,false)), Val(#=ReturnPrimal=#false), #=ShadowInit=#Val(false), NonGenABI)
    # thunk(Const(Base.identity), Active(1.0), 1.0)
end

# Define EnzymeTarget
Base.@kwdef struct EnzymeTarget <: AbstractCompilerTarget end

GPUCompiler.llvm_triple(::EnzymeTarget) = LLVM.triple(JIT.get_jit())
GPUCompiler.llvm_datalayout(::EnzymeTarget) = LLVM.datalayout(JIT.get_jit())

function GPUCompiler.llvm_machine(::EnzymeTarget)
    return JIT.get_tm()
end

module Runtime end

abstract type AbstractEnzymeCompilerParams <: AbstractCompilerParams end
struct EnzymeCompilerParams <: AbstractEnzymeCompilerParams
    TT::Type{<:Tuple}
    mode::API.CDerivativeMode
    width::Int
    rt::Type{<:Annotation{T} where {T}}
    run_enzyme::Bool
    abiwrap::Bool
    # Whether, in split mode, acessible primal argument data is modified
    # between the call and the split
    modifiedBetween::NTuple{N,Bool} where {N}
    # Whether to also return the primal
    returnPrimal::Bool
    # Whether to (in aug fwd) += by one
    shadowInit::Bool
    expectedTapeType::Type
    # Whether to use the pointer ABI, default true
    ABI::Type{<:ABI}
    # Whether to error if the function is written to
    err_if_func_written::Bool

    # Whether runtime activity is enabled
    runtimeActivity::Bool
end

struct UnknownTapeType end

struct PrimalCompilerParams <: AbstractEnzymeCompilerParams
    mode::API.CDerivativeMode
end

DefaultCompilerTarget(; kwargs...) =
    GPUCompiler.NativeCompilerTarget(; jlruntime = true, kwargs...)

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
if VERSION >= v"1.11.0-DEV.1552"
    struct EnzymeCacheToken
        target_type::Type
        always_inline::Any
        method_table::Core.MethodTable
        param_type::Type
        is_fwd::Bool
    end

    GPUCompiler.ci_cache_token(job::CompilerJob{<:Any,<:AbstractEnzymeCompilerParams}) =
        EnzymeCacheToken(
            typeof(job.config.target),
            job.config.always_inline,
            GPUCompiler.method_table(job),
            typeof(job.config.params),
            job.config.params.mode == API.DEM_ForwardMode,
        )

    GPUCompiler.get_interpreter(job::CompilerJob{<:Any,<:AbstractEnzymeCompilerParams}) =
        Interpreter.EnzymeInterpreter(
            GPUCompiler.ci_cache_token(job),
            GPUCompiler.method_table(job),
            job.world,
            job.config.params.mode,
        )
else

    # the codeinstance cache to use -- should only be used for the constructor
    # Note that the only way the interpreter modifies codegen is either not inlining a fwd mode
    # rule or not inlining a rev mode rule. Otherwise, all caches can be re-used.
    const GLOBAL_FWD_CACHE = GPUCompiler.CodeCache()
    const GLOBAL_REV_CACHE = GPUCompiler.CodeCache()
    function enzyme_ci_cache(job::CompilerJob{<:Any,<:AbstractEnzymeCompilerParams})
        return if job.config.params.mode == API.DEM_ForwardMode
            GLOBAL_FWD_CACHE
        else
            GLOBAL_REV_CACHE
        end
    end

    GPUCompiler.ci_cache(job::CompilerJob{<:Any,<:AbstractEnzymeCompilerParams}) =
        enzyme_ci_cache(job)

    GPUCompiler.get_interpreter(job::CompilerJob{<:Any,<:AbstractEnzymeCompilerParams}) =
        Interpreter.EnzymeInterpreter(
            enzyme_ci_cache(job),
            GPUCompiler.method_table(job),
            job.world,
            job.config.params.mode,
        )
end

include("compiler/passes.jl")
include("compiler/optimize.jl")
include("compiler/interpreter.jl")
include("compiler/validation.jl")

import .Interpreter: isKWCallSignature

"""
Create the methodinstance pair, and lookup the primal return type.
"""
@inline function fspec(
    @nospecialize(F),
    @nospecialize(TT),
    world::Union{Integer,Nothing} = nothing,
)
    # primal function. Inferred here to get return type
    _tt = (TT.parameters...,)

    primal_tt = Tuple{map(eltype, _tt)...}

    primal = if world isa Nothing
        my_methodinstance(F, primal_tt)
    else
        my_methodinstance(F, primal_tt, world)
    end

    return primal
end

@generated function primal_return_type(
    ::ReverseMode,
    ::Val{world},
    ::Type{FT},
    ::Type{TT},
) where {world,FT,TT}
    mode = Enzyme.API.DEM_ReverseModeCombined

    CT = @static if VERSION >= v"1.11.0-DEV.1552"
        EnzymeCacheToken(
            typeof(DefaultCompilerTarget()),
            false,
            GPUCompiler.GLOBAL_METHOD_TABLE, #=job.config.always_inline=#
            EnzymeCompilerParams,
            false,
        )
    else
        Enzyme.Compiler.GLOBAL_REV_CACHE
    end

    interp = Enzyme.Compiler.Interpreter.EnzymeInterpreter(CT, nothing, world, mode)
    res = Core.Compiler._return_type(interp, Tuple{FT,TT.parameters...})
    return quote
        Base.@_inline_meta
        $res
    end
end

@generated function primal_return_type(
    ::ForwardMode,
    ::Val{world},
    ::Type{FT},
    ::Type{TT},
) where {world,FT,TT}
    mode = Enzyme.API.DEM_ForwardMode

    CT = @static if VERSION >= v"1.11.0-DEV.1552"
        EnzymeCacheToken(
            typeof(DefaultCompilerTarget()),
            false,
            GPUCompiler.GLOBAL_METHOD_TABLE, #=always_inline=#
            EnzymeCompilerParams,
            false,
        )
    else
        Enzyme.Compiler.GLOBAL_FWD_CACHE
    end

    interp = Enzyme.Compiler.Interpreter.EnzymeInterpreter(CT, nothing, world, mode)
    res = Core.Compiler._return_type(interp, Tuple{FT,TT.parameters...})
    return quote
        Base.@_inline_meta
        $res
    end
end

##
# Enzyme compiler step
##

function annotate!(mod, mode)
    inactive = LLVM.StringAttribute("enzyme_inactive", "")
    active = LLVM.StringAttribute("enzyme_active", "")
    no_escaping_alloc = LLVM.StringAttribute("enzyme_no_escaping_allocation")
    fns = functions(mod)

    for f in fns
        API.EnzymeAttributeKnownFunctions(f.ref)
    end

    for gname in inactiveglobs
        globs = LLVM.globals(mod)
        if haskey(globs, gname)
            glob = globs[gname]
            API.SetMD(glob, "enzyme_inactive", LLVM.MDNode(LLVM.Metadata[]))
        end
    end

    for fname in inactivefns
        if haskey(fns, fname)
            fn = fns[fname]
            push!(function_attributes(fn), inactive)
            push!(function_attributes(fn), no_escaping_alloc)
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
                LLVM.API.LLVMAddCallSiteAttribute(
                    c,
                    reinterpret(
                        LLVM.API.LLVMAttributeIndex,
                        LLVM.API.LLVMAttributeFunctionIndex,
                    ),
                    inactive,
                )
                LLVM.API.LLVMAddCallSiteAttribute(
                    c,
                    reinterpret(
                        LLVM.API.LLVMAttributeIndex,
                        LLVM.API.LLVMAttributeFunctionIndex,
                    ),
                    no_escaping_alloc,
                )
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
                LLVM.API.LLVMAddCallSiteAttribute(
                    c,
                    reinterpret(
                        LLVM.API.LLVMAttributeIndex,
                        LLVM.API.LLVMAttributeFunctionIndex,
                    ),
                    LLVM.EnumAttribute("nofree", 0),
                )
            end
        end
    end

    for fname in activefns
        if haskey(fns, fname)
            fn = fns[fname]
            push!(function_attributes(fn), active)
        end
    end

    for fname in
        ("julia.typeof", "jl_object_id_", "jl_object_id", "ijl_object_id_", "ijl_object_id")
        if haskey(fns, fname)
            fn = fns[fname]
            if LLVM.version().major <= 15
                push!(function_attributes(fn), LLVM.EnumAttribute("readnone"))
            else
                push!(function_attributes(fn), EnumAttribute("memory", NoEffects.data))
            end
            push!(function_attributes(fn), LLVM.StringAttribute("enzyme_shouldrecompute"))
        end
    end
    for fname in ("julia.typeof",)
        if haskey(fns, fname)
            fn = fns[fname]
            push!(function_attributes(fn), LLVM.StringAttribute("enzyme_nocache"))
        end
    end

    for fname in
        ("jl_excstack_state", "ijl_excstack_state", "ijl_field_index", "jl_field_index")
        if haskey(fns, fname)
            fn = fns[fname]
            if LLVM.version().major <= 15
                push!(function_attributes(fn), LLVM.EnumAttribute("readonly"))
                push!(function_attributes(fn), LLVM.StringAttribute("inaccessiblememonly"))
            else
                push!(
                    function_attributes(fn),
                    EnumAttribute(
                        "memory",
                        MemoryEffect(
                            (MRI_NoModRef << getLocationPos(ArgMem)) |
                            (MRI_Ref << getLocationPos(InaccessibleMem)) |
                            (MRI_NoModRef << getLocationPos(Other)),
                        ).data,
                    ),
                )
            end
        end
    end

    for fname in ("jl_types_equal", "ijl_types_equal")
        if haskey(fns, fname)
            fn = fns[fname]
            push!(function_attributes(fn), LLVM.StringAttribute("enzyme_shouldrecompute"))
        end
    end

    for fname in (
        "jl_f_getfield",
        "ijl_f_getfield",
        "jl_get_nth_field_checked",
        "ijl_get_nth_field_checked",
        "jl_f__svec_ref",
        "ijl_f__svec_ref",
    )
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
                attr = if LLVM.version().major <= 15
                    LLVM.EnumAttribute("readonly")
                else
                    EnumAttribute(
                        "memory",
                        MemoryEffect(
                            (MRI_Ref << getLocationPos(ArgMem)) |
                            (MRI_NoModRef << getLocationPos(InaccessibleMem)) |
                            (MRI_NoModRef << getLocationPos(Other)),
                        ).data,
                    )
                end
                LLVM.API.LLVMAddCallSiteAttribute(
                    c,
                    reinterpret(
                        LLVM.API.LLVMAttributeIndex,
                        LLVM.API.LLVMAttributeFunctionIndex,
                    ),
                    attr,
                )
            end
        end
    end

    for fname in ("julia.get_pgcstack", "julia.ptls_states", "jl_get_ptls_states")
        if haskey(fns, fname)
            fn = fns[fname]
            # TODO per discussion w keno perhaps this should change to readonly / inaccessiblememonly
            if LLVM.version().major <= 15
                push!(function_attributes(fn), LLVM.EnumAttribute("readnone"))
            else
                push!(function_attributes(fn), EnumAttribute("memory", NoEffects.data))
            end
            push!(function_attributes(fn), LLVM.StringAttribute("enzyme_shouldrecompute"))
        end
    end

    for fname in ("julia.gc_loaded",)
        if haskey(fns, fname)
            fn = fns[fname]
            push!(function_attributes(fn), LLVM.StringAttribute("enzyme_shouldrecompute"))
        end
    end

    for fname in (
        "julia.get_pgcstack",
        "julia.ptls_states",
        "jl_get_ptls_states",
        "julia.safepoint",
        "ijl_throw",
        "julia.pointer_from_objref",
        "ijl_array_grow_end",
        "jl_array_grow_end",
        "ijl_array_del_end",
        "jl_array_del_end",
        "ijl_array_grow_beg",
        "jl_array_grow_beg",
        "ijl_array_del_beg",
        "jl_array_del_beg",
        "ijl_array_grow_at",
        "jl_array_grow_at",
        "ijl_array_del_at",
        "jl_array_del_at",
        "ijl_pop_handler",
        "jl_pop_handler",
        "ijl_push_handler",
        "jl_push_handler",
        "ijl_module_name",
        "jl_module_name",
        "ijl_restore_excstack",
        "jl_restore_excstack",
        "julia.except_enter",
        "ijl_get_nth_field_checked",
        "jl_get_nth_field_checked",
        "jl_egal__unboxed",
        "ijl_reshape_array",
        "jl_reshape_array",
        "ijl_eqtable_get",
        "jl_eqtable_get",
        "jl_gc_run_pending_finalizers",
        "ijl_try_substrtod",
        "jl_try_substrtod",
    )
        if haskey(fns, fname)
            fn = fns[fname]
            push!(function_attributes(fn), no_escaping_alloc)
        end
    end



    for fname in ("julia.pointer_from_objref",)
        if haskey(fns, fname)
            fn = fns[fname]
            if LLVM.version().major <= 15
                push!(function_attributes(fn), LLVM.EnumAttribute("readnone"))
            else
                push!(function_attributes(fn), EnumAttribute("memory", NoEffects.data))
            end
        end
    end

    for boxfn in (
        "julia.gc_alloc_obj",
        "jl_gc_alloc_typed",
        "ijl_gc_alloc_typed",
        "jl_box_float32",
        "jl_box_float64",
        "jl_box_int32",
        "jl_box_int64",
        "ijl_box_float32",
        "ijl_box_float64",
        "ijl_box_int32",
        "ijl_box_int64",
        "jl_alloc_genericmemory",
        "ijl_alloc_genericmemory",
        "jl_alloc_array_1d",
        "jl_alloc_array_2d",
        "jl_alloc_array_3d",
        "ijl_alloc_array_1d",
        "ijl_alloc_array_2d",
        "ijl_alloc_array_3d",
        "jl_array_copy",
        "ijl_array_copy",
        "jl_genericmemory_copy_slice",
        "ijl_genericmemory_copy_slice",
        "jl_alloc_genericmemory",
        "ijl_alloc_genericmemory",
        "jl_idtable_rehash",
        "ijl_idtable_rehash",
        "jl_f_tuple",
        "ijl_f_tuple",
        "jl_new_structv",
        "ijl_new_structv",
        "ijl_new_array",
        "jl_new_array",
    )
        if haskey(fns, boxfn)
            fn = fns[boxfn]
            push!(return_attributes(fn), LLVM.EnumAttribute("noalias", 0))
            push!(function_attributes(fn), no_escaping_alloc)
            accattr = if LLVM.version().major <= 15
                LLVM.EnumAttribute("inaccessiblememonly")
            else
                EnumAttribute(
                    "memory",
                    MemoryEffect(
                        (MRI_NoModRef << getLocationPos(ArgMem)) |
                        (MRI_ModRef << getLocationPos(InaccessibleMem)) |
                        (MRI_NoModRef << getLocationPos(Other)),
                    ).data,
                )
            end
            if !(
                boxfn in (
                    "jl_array_copy",
                    "ijl_array_copy",
                    "jl_genericmemory_copy_slice",
                    "ijl_genericmemory_copy_slice",
                    "jl_idtable_rehash",
                    "ijl_idtable_rehash",
                )
            )
                push!(function_attributes(fn), accattr)
            end
            for u in LLVM.uses(fn)
                c = LLVM.user(u)
                if !isa(c, LLVM.CallInst)
                    continue
                end
                cf = LLVM.called_operand(c)
                if cf == fn
                    LLVM.API.LLVMAddCallSiteAttribute(
                        c,
                        LLVM.API.LLVMAttributeReturnIndex,
                        LLVM.EnumAttribute("noalias", 0),
                    )
                    if !(
                        boxfn in (
                            "jl_array_copy",
                            "ijl_array_copy",
                            "jl_genericmemory_copy_slice",
                            "ijl_genericmemory_copy_slice",
                            "jl_idtable_rehash",
                            "ijl_idtable_rehash",
                        )
                    )
                        LLVM.API.LLVMAddCallSiteAttribute(
                            c,
                            reinterpret(
                                LLVM.API.LLVMAttributeIndex,
                                LLVM.API.LLVMAttributeFunctionIndex,
                            ),
                            accattr,
                        )
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
                LLVM.API.LLVMAddCallSiteAttribute(
                    c,
                    LLVM.API.LLVMAttributeReturnIndex,
                    LLVM.EnumAttribute("noalias", 0),
                )
                LLVM.API.LLVMAddCallSiteAttribute(
                    c,
                    reinterpret(
                        LLVM.API.LLVMAttributeIndex,
                        LLVM.API.LLVMAttributeFunctionIndex,
                    ),
                    no_escaping_alloc,
                )
                if !(
                    boxfn in (
                        "jl_array_copy",
                        "ijl_array_copy",
                        "jl_genericmemory_copy_slice",
                        "ijl_genericmemory_copy_slice",
                        "jl_idtable_rehash",
                        "ijl_idtable_rehash",
                    )
                )
                    attr = if LLVM.version().major <= 15
                        LLVM.EnumAttribute("inaccessiblememonly")
                    else
                        EnumAttribute(
                            "memory",
                            MemoryEffect(
                                (MRI_NoModRef << getLocationPos(ArgMem)) |
                                (MRI_ModRef << getLocationPos(InaccessibleMem)) |
                                (MRI_NoModRef << getLocationPos(Other)),
                            ).data,
                        )
                    end
                    LLVM.API.LLVMAddCallSiteAttribute(
                        c,
                        reinterpret(
                            LLVM.API.LLVMAttributeIndex,
                            LLVM.API.LLVMAttributeFunctionIndex,
                        ),
                        attr,
                    )
                end
            end
        end
    end

    for gc in ("llvm.julia.gc_preserve_begin", "llvm.julia.gc_preserve_end")
        if haskey(fns, gc)
            fn = fns[gc]
            if LLVM.version().major <= 15
                push!(function_attributes(fn), LLVM.EnumAttribute("inaccessiblememonly"))
            else
                push!(
                    function_attributes(fn),
                    EnumAttribute(
                        "memory",
                        MemoryEffect(
                            (MRI_NoModRef << getLocationPos(ArgMem)) |
                            (MRI_ModRef << getLocationPos(InaccessibleMem)) |
                            (MRI_NoModRef << getLocationPos(Other)),
                        ).data,
                    ),
                )
            end
        end
    end

    # Key of jl_eqtable_get/put is inactive, definitionally
    for rfn in ("jl_eqtable_get", "ijl_eqtable_get")
        if haskey(fns, rfn)
            fn = fns[rfn]
            push!(parameter_attributes(fn, 2), LLVM.StringAttribute("enzyme_inactive"))
            if LLVM.version().major <= 15
                push!(function_attributes(fn), LLVM.EnumAttribute("readonly"))
                push!(function_attributes(fn), LLVM.EnumAttribute("argmemonly"))
            else
                push!(
                    function_attributes(fn),
                    EnumAttribute(
                        "memory",
                        MemoryEffect(
                            (MRI_Ref << getLocationPos(ArgMem)) |
                            (MRI_NoModRef << getLocationPos(InaccessibleMem)) |
                            (MRI_NoModRef << getLocationPos(Other)),
                        ).data,
                    ),
                )
            end
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
            if LLVM.version().major <= 15
                push!(function_attributes(fn), LLVM.EnumAttribute("argmemonly"))
            else
                push!(
                    function_attributes(fn),
                    EnumAttribute(
                        "memory",
                        MemoryEffect(
                            (MRI_ModRef << getLocationPos(ArgMem)) |
                            (MRI_NoModRef << getLocationPos(InaccessibleMem)) |
                            (MRI_NoModRef << getLocationPos(Other)),
                        ).data,
                    ),
                )
            end
        end
    end

    for rfn in ("jl_in_threaded_region_", "jl_in_threaded_region")
        if haskey(fns, rfn)
            fn = fns[rfn]
            if LLVM.version().major <= 15
                push!(function_attributes(fn), LLVM.EnumAttribute("readonly"))
                push!(function_attributes(fn), LLVM.EnumAttribute("inaccessiblememonly"))
            else
                push!(
                    function_attributes(fn),
                    EnumAttribute(
                        "memory",
                        MemoryEffect(
                            (MRI_NoModRef << getLocationPos(ArgMem)) |
                            (MRI_Ref << getLocationPos(InaccessibleMem)) |
                            (MRI_NoModRef << getLocationPos(Other)),
                        ).data,
                    ),
                )
            end
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
    throw(AssertionError("Enzyme: could not find world in $(string(fn))"))
end

function enzyme_custom_extract_mi(orig::LLVM.Instruction, error = true)
    operand = LLVM.called_operand(orig)
    if isa(operand, LLVM.Function)
        return enzyme_custom_extract_mi(operand::LLVM.Function, error)
    elseif error
        GPUCompiler.@safe_error "Enzyme: Custom handler, could not find fn", orig
    end
    return nothing, nothing
end

function enzyme_custom_extract_mi(orig::LLVM.Function, error = true)
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

function enzyme_extract_parm_type(fn::LLVM.Function, idx::Int, error = true)
    ty = nothing
    byref = nothing
    for fattr in collect(parameter_attributes(fn, idx))
        if isa(fattr, LLVM.StringAttribute)
            if kind(fattr) == "enzymejl_parmtype"
                ptr = reinterpret(Ptr{Cvoid}, parse(UInt, LLVM.value(fattr)))
                ty = Base.unsafe_pointer_to_objref(ptr)
            end
            if kind(fattr) == "enzymejl_parmtype_ref"
                byref = GPUCompiler.ArgumentCC(parse(UInt, LLVM.value(fattr)))
            end
        end
    end
    if error && (byref === nothing || ty === nothing)
        GPUCompiler.@safe_error "Enzyme: Custom handler, could not find parm type at index",
        idx,
        fn
    end
    return ty, byref
end

include("rules/typerules.jl")
include("rules/activityrules.jl")

@inline Base.convert(::Type{API.CDIFFE_TYPE}, ::Type{A}) where {A<:Const} = API.DFT_CONSTANT
@inline Base.convert(::Type{API.CDIFFE_TYPE}, ::Type{A}) where {A<:Active} =
    API.DFT_OUT_DIFF
@inline Base.convert(::Type{API.CDIFFE_TYPE}, ::Type{A}) where {A<:Duplicated} =
    API.DFT_DUP_ARG
@inline Base.convert(::Type{API.CDIFFE_TYPE}, ::Type{A}) where {A<:BatchDuplicated} =
    API.DFT_DUP_ARG
@inline Base.convert(::Type{API.CDIFFE_TYPE}, ::Type{A}) where {A<:BatchDuplicatedFunc} =
    API.DFT_DUP_ARG
@inline Base.convert(::Type{API.CDIFFE_TYPE}, ::Type{A}) where {A<:DuplicatedNoNeed} =
    API.DFT_DUP_NONEED
@inline Base.convert(::Type{API.CDIFFE_TYPE}, ::Type{A}) where {A<:BatchDuplicatedNoNeed} =
    API.DFT_DUP_NONEED

const DumpPreEnzyme = Ref(false)
const DumpPostWrap = Ref(false)

function enzyme!(
    job,
    mod,
    primalf,
    TT,
    mode,
    width,
    parallel,
    actualRetType,
    wrap,
    modifiedBetween,
    returnPrimal,
    expectedTapeType,
    loweredArgs,
    boxedArgs,
)
    if DumpPreEnzyme[]
        API.EnzymeDumpModuleRef(mod.ref)
    end
    world = job.world
    interp = GPUCompiler.get_interpreter(job)
    rt = job.config.params.rt
    runtimeActivity = job.config.params.runtimeActivity
    @assert eltype(rt) != Union{}

    shadow_init = job.config.params.shadowInit
    ctx = context(mod)
    dl = string(LLVM.datalayout(mod))

    tt = [TT.parameters[2:end]...]

    args_activity = API.CDIFFE_TYPE[]
    uncacheable_args = Bool[]
    args_typeInfo = TypeTree[]
    args_known_values = API.IntList[]


    @assert length(modifiedBetween) == length(TT.parameters)

    swiftself = any(
        any(
            map(
                k -> kind(k) == kind(EnumAttribute("swiftself")),
                collect(parameter_attributes(primalf, i)),
            ),
        ) for i = 1:length(collect(parameters(primalf)))
    )
    if swiftself
        push!(args_activity, API.DFT_CONSTANT)
        push!(args_typeInfo, TypeTree())
        push!(uncacheable_args, false)
        push!(args_known_values, API.IntList())
    end

    seen = TypeTreeTable()
    for (i, T) in enumerate(TT.parameters)
        source_typ = eltype(T)
        if isghostty(source_typ) || Core.Compiler.isconstType(source_typ)
            if !(T <: Const)
                error(
                    "Type of ghost or constant type " *
                    string(T) *
                    " is marked as differentiable.",
                )
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
        elseif T <: Duplicated ||
               T <: BatchDuplicated ||
               T <: BatchDuplicatedFunc ||
               T <: MixedDuplicated ||
               T <: BatchMixedDuplicated
            push!(args_activity, API.DFT_DUP_ARG)
        elseif T <: DuplicatedNoNeed || T <: BatchDuplicatedNoNeed
            push!(args_activity, API.DFT_DUP_NONEED)
        else
            error("illegal annotation type $T")
        end
        typeTree = typetree(source_typ, ctx, dl, seen)
        if isboxed
            typeTree = copy(typeTree)
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
    retType = if rt <: MixedDuplicated || rt <: BatchMixedDuplicated
        API.DFT_OUT_DIFF
    else
        convert(API.CDIFFE_TYPE, rt)
    end

    rules = Dict{String,API.CustomRuleType}(
        "jl_array_copy" => @cfunction(
            inout_rule,
            UInt8,
            (
                Cint,
                API.CTypeTreeRef,
                Ptr{API.CTypeTreeRef},
                Ptr{API.IntList},
                Csize_t,
                LLVM.API.LLVMValueRef,
            )
        ),
        "ijl_array_copy" => @cfunction(
            inout_rule,
            UInt8,
            (
                Cint,
                API.CTypeTreeRef,
                Ptr{API.CTypeTreeRef},
                Ptr{API.IntList},
                Csize_t,
                LLVM.API.LLVMValueRef,
            )
        ),
        "jl_genericmemory_copy_slice" => @cfunction(
            inoutcopyslice_rule,
            UInt8,
            (
                Cint,
                API.CTypeTreeRef,
                Ptr{API.CTypeTreeRef},
                Ptr{API.IntList},
                Csize_t,
                LLVM.API.LLVMValueRef,
            )
        ),
        "ijl_genericmemory_copy_slice" => @cfunction(
            inoutcopyslice_rule,
            UInt8,
            (
                Cint,
                API.CTypeTreeRef,
                Ptr{API.CTypeTreeRef},
                Ptr{API.IntList},
                Csize_t,
                LLVM.API.LLVMValueRef,
            )
        ),
        "julia.gc_loaded" => @cfunction(
            inoutgcloaded_rule,
            UInt8,
            (
                Cint,
                API.CTypeTreeRef,
                Ptr{API.CTypeTreeRef},
                Ptr{API.IntList},
                Csize_t,
                LLVM.API.LLVMValueRef,
            )
        ),
        "julia.pointer_from_objref" => @cfunction(
            inout_rule,
            UInt8,
            (
                Cint,
                API.CTypeTreeRef,
                Ptr{API.CTypeTreeRef},
                Ptr{API.IntList},
                Csize_t,
                LLVM.API.LLVMValueRef,
            )
        ),
        "jl_inactive_inout" => @cfunction(
            inout_rule,
            UInt8,
            (
                Cint,
                API.CTypeTreeRef,
                Ptr{API.CTypeTreeRef},
                Ptr{API.IntList},
                Csize_t,
                LLVM.API.LLVMValueRef,
            )
        ),
        "jl_excstack_state" => @cfunction(
            int_return_rule,
            UInt8,
            (
                Cint,
                API.CTypeTreeRef,
                Ptr{API.CTypeTreeRef},
                Ptr{API.IntList},
                Csize_t,
                LLVM.API.LLVMValueRef,
            )
        ),
        "ijl_excstack_state" => @cfunction(
            int_return_rule,
            UInt8,
            (
                Cint,
                API.CTypeTreeRef,
                Ptr{API.CTypeTreeRef},
                Ptr{API.IntList},
                Csize_t,
                LLVM.API.LLVMValueRef,
            )
        ),
        "julia.except_enter" => @cfunction(
            int_return_rule,
            UInt8,
            (
                Cint,
                API.CTypeTreeRef,
                Ptr{API.CTypeTreeRef},
                Ptr{API.IntList},
                Csize_t,
                LLVM.API.LLVMValueRef,
            )
        ),
    )

    logic = Logic()
    TA = TypeAnalysis(logic, rules)

    retT =
        (!isa(actualRetType, Union) && GPUCompiler.deserves_retbox(actualRetType)) ?
        Ptr{actualRetType} : actualRetType
    retTT =
        (
            !isa(actualRetType, Union) &&
            actualRetType <: Tuple &&
            in(Any, actualRetType.parameters)
        ) ? TypeTree() : typetree(retT, ctx, dl, seen)

    typeInfo = FnTypeInfo(retTT, args_typeInfo, args_known_values)

    TapeType = Cvoid

    if mode == API.DEM_ReverseModePrimal || mode == API.DEM_ReverseModeGradient
        returnUsed = !(isghostty(actualRetType) || Core.Compiler.isconstType(actualRetType))
        shadowReturnUsed =
            returnUsed && (
                retType == API.DFT_DUP_ARG ||
                retType == API.DFT_DUP_NONEED ||
                rt <: MixedDuplicated ||
                rt <: BatchMixedDuplicated
            )
        returnUsed &= returnPrimal
        augmented = API.EnzymeCreateAugmentedPrimal(
            logic,
            primalf,
            retType,
            args_activity,
            TA,
            returnUsed, #=returnUsed=#
            shadowReturnUsed,            #=shadowReturnUsed=#
            typeInfo,
            uncacheable_args,
            false,
            runtimeActivity,
            width,
            parallel,
        ) #=atomicAdd=#

        # 2. get new_primalf and tape
        augmented_primalf =
            LLVM.Function(API.EnzymeExtractFunctionFromAugmentation(augmented))
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
            augmented_primalf = create_abi_wrapper(
                augmented_primalf,
                TT,
                rt,
                actualRetType,
                API.DEM_ReverseModePrimal,
                augmented,
                width,
                returnPrimal,
                shadow_init,
                world,
                interp,
            )
        end

        # TODOs:
        # 1. Handle mutable or !pointerfree arguments by introducing caching
        #     + specifically by setting uncacheable_args[i] = true

        adjointf = LLVM.Function(
            API.EnzymeCreatePrimalAndGradient(
                logic,
                primalf,
                retType,
                args_activity,
                TA,
                false,
                false,
                API.DEM_ReverseModeGradient,
                runtimeActivity,
                width, #=mode=#
                tape,
                false,
                typeInfo, #=forceAnonymousTape=#
                uncacheable_args,
                augmented,
                parallel,
            ),
        ) #=atomicAdd=#
        if wrap
            adjointf = create_abi_wrapper(
                adjointf,
                TT,
                rt,
                actualRetType,
                API.DEM_ReverseModeGradient,
                augmented,
                width,
                false,
                shadow_init,
                world,
                interp,
            ) #=returnPrimal=#
        end
    elseif mode == API.DEM_ReverseModeCombined
        returnUsed = !isghostty(actualRetType)
        returnUsed &= returnPrimal
        adjointf = LLVM.Function(
            API.EnzymeCreatePrimalAndGradient(
                logic,
                primalf,
                retType,
                args_activity,
                TA,
                returnUsed,
                false,
                API.DEM_ReverseModeCombined,
                runtimeActivity,
                width, #=mode=#
                C_NULL,
                false,
                typeInfo, #=forceAnonymousTape=#
                uncacheable_args,
                C_NULL,
                parallel,
            ),
        ) #=atomicAdd=#
        augmented_primalf = nothing
        if wrap
            adjointf = create_abi_wrapper(
                adjointf,
                TT,
                rt,
                actualRetType,
                API.DEM_ReverseModeCombined,
                nothing,
                width,
                returnPrimal,
                shadow_init,
                world,
                interp,
            )
        end
    elseif mode == API.DEM_ForwardMode
        returnUsed = !(isghostty(actualRetType) || Core.Compiler.isconstType(actualRetType))
        returnUsed &= returnPrimal
        adjointf = LLVM.Function(
            API.EnzymeCreateForwardDiff(
                logic,
                primalf,
                retType,
                args_activity,
                TA,
                returnUsed,
                API.DEM_ForwardMode,
                runtimeActivity,
                width, #=mode=#
                C_NULL,
                typeInfo,            #=additionalArg=#
                uncacheable_args,
            ),
        )
        augmented_primalf = nothing
        if wrap
            pf = adjointf
            adjointf = create_abi_wrapper(
                adjointf,
                TT,
                rt,
                actualRetType,
                API.DEM_ForwardMode,
                nothing,
                width,
                returnPrimal,
                shadow_init,
                world,
                interp,
            )
        end
    else
        @assert "Unhandled derivative mode", mode
    end
    if DumpPostWrap[]
        API.EnzymeDumpModuleRef(mod.ref)
    end
    API.EnzymeLogicErasePreprocessedFunctions(logic)
    adjointfname = adjointf == nothing ? nothing : LLVM.name(adjointf)
    augmented_primalfname =
        augmented_primalf == nothing ? nothing : LLVM.name(augmented_primalf)
    for f in collect(functions(mod))
        API.EnzymeFixupBatchedJuliaCallingConvention(f)
    end
    ModulePassManager() do pm
        dce!(pm)
        LLVM.run!(pm, mod)
    end
    fix_decayaddr!(mod)
    adjointf = adjointf == nothing ? nothing : functions(mod)[adjointfname]
    augmented_primalf =
        augmented_primalf == nothing ? nothing : functions(mod)[augmented_primalfname]
    return adjointf, augmented_primalf, TapeType
end

function get_subprogram(f::LLVM.Function)
    @static if isdefined(LLVM, :subprogram)
        LLVM.subprogram(f)
    else
        LLVM.get_subprogram(f)
    end
end

function set_subprogram!(f::LLVM.Function, sp)
    @static if isdefined(LLVM, :subprogram)
        LLVM.subprogram!(f, sp)
    else
        LLVM.set_subprogram!(f, sp)
    end
end

function create_abi_wrapper(
    enzymefn::LLVM.Function,
    TT,
    rettype,
    actualRetType,
    Mode::API.CDerivativeMode,
    augmented,
    width,
    returnPrimal,
    shadow_init,
    world,
    interp,
)
    is_adjoint = Mode == API.DEM_ReverseModeGradient || Mode == API.DEM_ReverseModeCombined
    is_split = Mode == API.DEM_ReverseModeGradient || Mode == API.DEM_ReverseModePrimal
    needs_tape = Mode == API.DEM_ReverseModeGradient

    mod = LLVM.parent(enzymefn)
    ctx = LLVM.context(mod)

    push!(function_attributes(enzymefn), EnumAttribute("alwaysinline", 0))
    hasNoInline = any(
        map(
            k -> kind(k) == kind(EnumAttribute("noinline")),
            collect(function_attributes(enzymefn)),
        ),
    )
    if hasNoInline
        LLVM.API.LLVMRemoveEnumAttributeAtIndex(
            enzymefn,
            reinterpret(LLVM.API.LLVMAttributeIndex, LLVM.API.LLVMAttributeFunctionIndex),
            kind(EnumAttribute("noinline")),
        )
    end
    T_void = convert(LLVMType, Nothing)
    ptr8 = LLVM.PointerType(LLVM.IntType(8))
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    # Create Enzyme calling convention
    T_wrapperargs = LLVMType[] # Arguments of the wrapper

    sret_types = Type[]  # Julia types of all returned variables

    pactualRetType = actualRetType
    sret_union = is_sret_union(actualRetType)
    literal_rt = eltype(rettype)
    @assert literal_rt != Union{}
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
                    push!(ActiveRetTypes, NTuple{width,source_typ})
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
        elseif T <: MixedDuplicated || T <: BatchMixedDuplicated
            push!(T_wrapperargs, LLVM.LLVMType(API.EnzymeGetShadowType(width, T_prjlvalue)))
            if is_adjoint && i != 1
                push!(ActiveRetTypes, Nothing)
            end
        else
            error("calling convention should be annotated, got $T")
        end
    end

    if is_adjoint
        NT = Tuple{ActiveRetTypes...}
        if any(
            any_jltypes(convert(LLVM.LLVMType, b; allow_boxed = true)) for
            b in ActiveRetTypes
        )
            NT = AnonymousStruct(NT)
        end
        push!(sret_types, NT)
    end

    # API.DFT_OUT_DIFF
    if is_adjoint
        if rettype <: Active ||
           rettype <: MixedDuplicated ||
           rettype <: BatchMixedDuplicated
            @assert !sret_union
            if allocatedinline(actualRetType) != allocatedinline(literal_rt)
                msg = sprint() do io
                    println(io, string(enzymefn))
                    println(
                        io,
                        "Base.allocatedinline(actualRetType) != Base.allocatedinline(literal_rt): actualRetType = $(actualRetType), literal_rt = $(literal_rt), rettype = $(rettype), sret_union=$(sret_union), pactualRetType=$(pactualRetType)",
                    )
                end
                throw(AssertionError(msg))
            end
            if rettype <: Active
                if !allocatedinline(actualRetType)
                    throw(
                        AssertionError(
                            "Base.allocatedinline(actualRetType) returns false: actualRetType = $(actualRetType), rettype = $(rettype)",
                        ),
                    )
                end
            end
            dretTy = LLVM.LLVMType(
                API.EnzymeGetShadowType(
                    width,
                    convert(LLVMType, actualRetType; allow_boxed = !(rettype <: Active)),
                ),
            )
            push!(T_wrapperargs, dretTy)
        end
    end

    data = Array{Int64}(undef, 3)
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
            if rettype <: Duplicated ||
               rettype <: DuplicatedNoNeed ||
               rettype <: BatchDuplicated ||
               rettype <: BatchDuplicatedNoNeed ||
               rettype <: BatchDuplicatedFunc
                if width == 1
                    push!(sret_types, literal_rt)
                else
                    push!(sret_types, AnonymousStruct(NTuple{width,literal_rt}))
                end
            elseif rettype <: MixedDuplicated || rettype <: BatchMixedDuplicated
                if width == 1
                    push!(sret_types, Base.RefValue{literal_rt})
                else
                    push!(
                        sret_types,
                        AnonymousStruct(NTuple{width,Base.RefValue{literal_rt}}),
                    )
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
        if !(rettype <: Const)
            if width == 1
                push!(sret_types, literal_rt)
            else
                push!(sret_types, AnonymousStruct(NTuple{width,literal_rt}))
            end
        end
        if returnPrimal
            push!(sret_types, literal_rt)
        end
    end

    combinedReturn =
        if any(
            any_jltypes(convert(LLVM.LLVMType, T; allow_boxed = true)) for T in sret_types
        )
            AnonymousStruct(Tuple{sret_types...})
        else
            Tuple{sret_types...}
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
            jltape = convert(LLVM.LLVMType, tape_type(tape); allow_boxed = true)
            push!(T_wrapperargs, jltape)
        else
            needs_tape = false
        end
    end

    T_ret = returnRoots ? T_void : jltype
    FT = LLVM.FunctionType(T_ret, T_wrapperargs)
    llvm_f = LLVM.Function(mod, safe_name(LLVM.name(enzymefn) * "wrap"), FT)
    API.EnzymeCloneFunctionDISubprogramInto(llvm_f, enzymefn)
    dl = datalayout(mod)

    params = [parameters(llvm_f)...]

    builder = LLVM.IRBuilder()
    entry = BasicBlock(llvm_f, "entry")
    position!(builder, entry)

    realparms = LLVM.Value[]
    i = 1

    for attr in collect(function_attributes(enzymefn))
        if kind(attr) == "enzymejl_world"
            push!(function_attributes(llvm_f), attr)
        end
    end

    if returnRoots
        sret = params[i]
        i += 1

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
        i += 1
    end

    activeNum = 0

    for T in TT.parameters
        T′ = eltype(T)

        if isghostty(T′) || Core.Compiler.isconstType(T′)
            continue
        end

        isboxed = GPUCompiler.deserves_argbox(T′)

        llty = value_type(params[i])

        convty = convert(LLVMType, T′; allow_boxed = true)

        if (T <: MixedDuplicated || T <: BatchMixedDuplicated) && !isboxed # && (isa(llty, LLVM.ArrayType) || isa(llty, LLVM.StructType))
            al0 = al = emit_allocobj!(builder, Base.RefValue{T′}, "mixedparameter")
            al = bitcast!(builder, al, LLVM.PointerType(llty, addrspace(value_type(al))))
            store!(builder, params[i], al)
            emit_writebarrier!(builder, get_julia_inner_types(builder, al0, params[i]))
            al = addrspacecast!(builder, al, LLVM.PointerType(llty, Derived))
            push!(realparms, al)
        else
            push!(realparms, params[i])
        end

        i += 1
        if T <: Const
        elseif T <: Active
            isboxed = GPUCompiler.deserves_argbox(T′)
            if isboxed
                if is_split
                    msg = sprint() do io
                        println(
                            io,
                            "Unimplemented: Had active input arg needing a box in split mode",
                        )
                        println(io, T, " at index ", i)
                        println(io, TT)
                    end
                    throw(AssertionError(msg))
                end
                @assert !is_split
                # TODO replace with better enzyme_zero
                ptr = gep!(
                    builder,
                    jltype,
                    sret,
                    [
                        LLVM.ConstantInt(LLVM.IntType(64), 0),
                        LLVM.ConstantInt(LLVM.IntType(32), activeNum),
                    ],
                )
                cst = pointercast!(builder, ptr, ptr8)
                push!(realparms, ptr)

                LLVM.memset!(
                    builder,
                    cst,
                    LLVM.ConstantInt(LLVM.IntType(8), 0),
                    LLVM.ConstantInt(
                        LLVM.IntType(64),
                        LLVM.storage_size(dl, Base.eltype(LLVM.value_type(ptr))),
                    ),
                    0,
                )                                            #=align=#
            end
            activeNum += 1
        elseif T <: Duplicated || T <: DuplicatedNoNeed
            push!(realparms, params[i])
            i += 1
        elseif T <: MixedDuplicated || T <: BatchMixedDuplicated
            parmsi = params[i]

            if T <: BatchMixedDuplicated
                if GPUCompiler.deserves_argbox(NTuple{width,Base.RefValue{T′}})
                    njlvalue = LLVM.ArrayType(Int(width), T_prjlvalue)
                    parmsi = bitcast!(
                        builder,
                        parmsi,
                        LLVM.PointerType(njlvalue, addrspace(value_type(parmsi))),
                    )
                    parmsi = load!(builder, njlvalue, parmsi)
                end
            end

            isboxed = GPUCompiler.deserves_argbox(T′)

            resty = isboxed ? llty : LLVM.PointerType(llty, Derived)

            ival = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, resty)))
            for idx = 1:width
                pv = (width == 1) ? parmsi : extract_value!(builder, parmsi, idx - 1)
                pv =
                    bitcast!(builder, pv, LLVM.PointerType(llty, addrspace(value_type(pv))))
                pv = addrspacecast!(builder, pv, LLVM.PointerType(llty, Derived))
                if isboxed
                    pv = load!(builder, llty, pv, "mixedboxload")
                end
                ival = (width == 1) ? pv : insert_value!(builder, ival, pv, idx - 1)
            end

            push!(realparms, ival)
            i += 1
        elseif T <: BatchDuplicated || T <: BatchDuplicatedNoNeed
            isboxed = GPUCompiler.deserves_argbox(NTuple{width,T′})
            val = params[i]
            if isboxed
                val = load!(builder, val)
            end
            i += 1
            push!(realparms, val)
        elseif T <: BatchDuplicatedFunc
            Func = get_func(T)
            funcspec = my_methodinstance(Func, Tuple{}, world)
            llvmf = nested_codegen!(Mode, mod, funcspec, world)
            push!(function_attributes(llvmf), EnumAttribute("alwaysinline", 0))
            Func_RT = Core.Compiler.typeinf_ext_toplevel(interp, funcspec).rettype
            @assert Func_RT == NTuple{width,T′}
            _, psret, _ = get_return_info(Func_RT)
            args = LLVM.Value[]
            if psret !== nothing
                psret = alloca!(builder, convert(LLVMType, Func_RT))
                push!(args, psret)
            end
            res = LLVM.call!(builder, LLVM.function_type(llvmf), llvmf, args)
            if get_subprogram(llvmf) !== nothing
                metadata(res)[LLVM.MD_dbg] = DILocation(0, 0, get_subprogram(llvm_f))
            end
            if psret !== nothing
                res = load!(builder, convert(LLVMType, Func_RT), psret)
            end
            push!(realparms, res)
        else
            @assert false
        end
    end

    if is_adjoint &&
       (rettype <: Active || rettype <: MixedDuplicated || rettype <: BatchMixedDuplicated)
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
    if get_subprogram(llvm_f) !== nothing
        metadata(val)[LLVM.MD_dbg] = DILocation(0, 0, get_subprogram(llvm_f))
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
        if existed[3] != 0 &&
           sret_union &&
           active_reg_inner(pactualRetType, (), world, Val(true), Val(true)) == ActiveState #=UnionSret=#
            rewrite_union_returns_as_ref(enzymefn, data[3], world, width)
        end
        returnNum = 0
        for i = 1:3
            if existed[i] != 0
                eval = val
                if data[i] != -1
                    eval = extract_value!(builder, val, data[i])
                end
                if i == 3
                    if rettype <: MixedDuplicated || rettype <: BatchMixedDuplicated
                        ival = UndefValue(
                            LLVM.LLVMType(API.EnzymeGetShadowType(width, T_prjlvalue)),
                        )
                        for idx = 1:width
                            pv =
                                (width == 1) ? eval : extract_value!(builder, eval, idx - 1)
                            al0 =
                                al = emit_allocobj!(
                                    builder,
                                    Base.RefValue{eltype(rettype)},
                                    "batchmixedret",
                                )
                            llty = value_type(pv)
                            al = bitcast!(
                                builder,
                                al,
                                LLVM.PointerType(llty, addrspace(value_type(al))),
                            )
                            store!(builder, pv, al)
                            emit_writebarrier!(
                                builder,
                                get_julia_inner_types(builder, al0, pv),
                            )
                            ival =
                                (width == 1) ? al0 :
                                insert_value!(builder, ival, al0, idx - 1)
                        end
                        eval = ival
                    end
                end
                eval = fixup_abi(i, eval)
                ptr = inbounds_gep!(
                    builder,
                    jltype,
                    sret,
                    [
                        LLVM.ConstantInt(LLVM.IntType(64), 0),
                        LLVM.ConstantInt(LLVM.IntType(32), returnNum),
                    ],
                )
                ptr = pointercast!(builder, ptr, LLVM.PointerType(value_type(eval)))
                si = store!(builder, eval, ptr)
                returnNum += 1
                if i == 3 && shadow_init
                    shadows = LLVM.Value[]
                    if width == 1
                        push!(shadows, eval)
                    else
                        for i = 1:width
                            push!(shadows, extract_value!(builder, eval, i - 1))
                        end
                    end

                    for shadowv in shadows
                        c = emit_apply_generic!(builder, [unsafe_to_llvm(builder, add_one_in_place), shadowv])
                        if get_subprogram(llvm_f) !== nothing
                            metadata(c)[LLVM.MD_dbg] =
                                DILocation(0, 0, get_subprogram(llvm_f))
                        end
                    end
                end
            elseif !isghostty(sret_types[i])
                ty = sret_types[i]
                # if primal return, we can upgrade to the full known type
                if i == 2
                    ty = actualRetType
                end
                @assert !(
                    isghostty(combinedReturn) || Core.Compiler.isconstType(combinedReturn)
                )
                @assert Core.Compiler.isconstType(ty)
                eval = makeInstanceOf(builder, ty)
                eval = fixup_abi(i, eval)
                ptr = inbounds_gep!(
                    builder,
                    jltype,
                    sret,
                    [
                        LLVM.ConstantInt(LLVM.IntType(64), 0),
                        LLVM.ConstantInt(LLVM.IntType(32), returnNum),
                    ],
                )
                ptr = pointercast!(builder, ptr, LLVM.PointerType(value_type(eval)))
                si = store!(builder, eval, ptr)
                returnNum += 1
            end
        end
        @assert returnNum == numLLVMReturns
    elseif Mode == API.DEM_ForwardMode
        count_Sret = 0
        count_llvm_Sret = 0
        if !isghostty(actualRetType)
            if !Core.Compiler.isconstType(actualRetType)
                if returnPrimal
                    count_llvm_Sret += 1
                end
                if !(rettype <: Const)
                    count_llvm_Sret += 1
                end
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
        for returnNum = 0:(count_Sret-1)
            eval = fixup_abi(
                returnNum + 1,
                if count_llvm_Sret == 0
                    makeInstanceOf(builder, actualRetType)
                elseif count_llvm_Sret == 1
                    val
                else
                    @assert count_llvm_Sret > 1
                    extract_value!(builder, val, 1 - returnNum)
                end,
            )
            ptr = inbounds_gep!(
                builder,
                jltype,
                sret,
                [
                    LLVM.ConstantInt(LLVM.IntType(64), 0),
                    LLVM.ConstantInt(LLVM.IntType(32), returnNum),
                ],
            )
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
                    eval = fixup_abi(
                        returnNum + 1,
                        if !isghostty(actualRetType)
                            extract_value!(builder, val, returnNum)
                        else
                            makeInstanceOf(builder, sret_types[returnNum+1])
                        end,
                    )
                    store!(
                        builder,
                        eval,
                        inbounds_gep!(
                            builder,
                            jltype,
                            sret,
                            [
                                LLVM.ConstantInt(LLVM.IntType(64), 0),
                                LLVM.ConstantInt(
                                    LLVM.IntType(32),
                                    length(elements(jltype)) - 1,
                                ),
                            ],
                        ),
                    )
                    returnNum += 1
                end
            end
        end
        for T in TT.parameters[2:end]
            if T <: Active
                T′ = eltype(T)
                isboxed = GPUCompiler.deserves_argbox(T′)
                if !isboxed
                    eval = extract_value!(builder, val, returnNum)
                    store!(
                        builder,
                        eval,
                        inbounds_gep!(
                            builder,
                            jltype,
                            sret,
                            [
                                LLVM.ConstantInt(LLVM.IntType(64), 0),
                                LLVM.ConstantInt(LLVM.IntType(32), 0),
                                LLVM.ConstantInt(LLVM.IntType(32), activeNum),
                            ],
                        ),
                    )
                    returnNum += 1
                end
                activeNum += 1
            end
        end
        @assert (returnNum - activeNum) + (activeNum != 0 ? 1 : 0) == numLLVMReturns
    end

    if returnRoots
        count = 0
        todo = Tuple{Vector{LLVM.Value},LLVM.LLVMType}[(
            [LLVM.ConstantInt(LLVM.IntType(64), 0)],
            jltype,
        )]
        while length(todo) != 0
            path, ty = popfirst!(todo)
            if isa(ty, LLVM.PointerType)
                loc = inbounds_gep!(
                    builder,
                    root_ty,
                    rootRet,
                    [
                        LLVM.ConstantInt(LLVM.IntType(64), 0),
                        LLVM.ConstantInt(LLVM.IntType(32), count),
                    ],
                )
                count += 1
                outloc = inbounds_gep!(builder, jltype, sret, path)
                store!(builder, load!(builder, ty, outloc), loc)
                continue
            end
            if isa(ty, LLVM.ArrayType)
                if any_jltypes(ty)
                    for i = 1:length(ty)
                        npath = copy(path)
                        push!(npath, LLVM.ConstantInt(LLVM.IntType(32), i - 1))
                        push!(todo, (npath, eltype(ty)))
                    end
                end
                continue
            end
            if isa(ty, LLVM.VectorType)
                if any_jltypes(ty)
                    for i = 1:size(ty)
                        npath = copy(path)
                        push!(npath, LLVM.ConstantInt(LLVM.IntType(32), i - 1))
                        push!(todo, (npath, eltype(ty)))
                    end
                end
                continue
            end
            if isa(ty, LLVM.StructType)
                for (i, t) in enumerate(LLVM.elements(ty))
                    if any_jltypes(t)
                        npath = copy(path)
                        push!(npath, LLVM.ConstantInt(LLVM.IntType(32), i - 1))
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

struct RemovedParam end

# Modified from GPUCompiler classify_arguments
function classify_arguments(
    source_sig::Type,
    codegen_ft::LLVM.FunctionType,
    has_sret::Bool,
    has_returnroots::Bool,
    has_swiftself::Bool,
    parmsRemoved::Vector{UInt64},
)
    codegen_types = parameters(codegen_ft)

    args = []
    codegen_i = 1
    orig_i = 1
    if has_sret
        if !in(orig_i - 1, parmsRemoved)
            codegen_i += 1
        end
        orig_i += 1
    end
    if has_returnroots
        if !in(orig_i - 1, parmsRemoved)
            codegen_i += 1
        end
        orig_i += 1
    end
    if has_swiftself
        if !in(orig_i - 1, parmsRemoved)
            codegen_i += 1
        end
        orig_i += 1
    end
    for (source_i, source_typ) in enumerate(source_sig.parameters)
        if isghostty(source_typ) || Core.Compiler.isconstType(source_typ)
            push!(args, (cc = GPUCompiler.GHOST, typ = source_typ, arg_i = source_i))
            continue
        end
        if in(orig_i - 1, parmsRemoved)
            push!(args, (cc = RemovedParam, typ = source_typ))
            orig_i += 1
            continue
        end
        codegen_typ = codegen_types[codegen_i]

        if codegen_typ isa LLVM.PointerType
            llvm_source_typ = convert(LLVMType, source_typ; allow_boxed = true)
            # pointers are used for multiple kinds of arguments
            # - literal pointer values
            if source_typ <: Ptr || source_typ <: Core.LLVMPtr
                @assert llvm_source_typ == codegen_typ
                push!(
                    args,
                    (
                        cc = GPUCompiler.BITS_VALUE,
                        typ = source_typ,
                        arg_i = source_i,
                        codegen = (typ = codegen_typ, i = codegen_i),
                    ),
                )
                # - boxed values
                #   XXX: use `deserves_retbox` instead?
            elseif llvm_source_typ isa LLVM.PointerType
                @assert llvm_source_typ == codegen_typ
                push!(
                    args,
                    (
                        cc = GPUCompiler.MUT_REF,
                        typ = source_typ,
                        arg_i = source_i,
                        codegen = (typ = codegen_typ, i = codegen_i),
                    ),
                )
                # - references to aggregates
            else
                @assert llvm_source_typ != codegen_typ
                push!(
                    args,
                    (
                        cc = GPUCompiler.BITS_REF,
                        typ = source_typ,
                        arg_i = source_i,
                        codegen = (typ = codegen_typ, i = codegen_i),
                    ),
                )
            end
        else
            push!(
                args,
                (
                    cc = GPUCompiler.BITS_VALUE,
                    typ = source_typ,
                    arg_i = source_i,
                    codegen = (typ = codegen_typ, i = codegen_i),
                ),
            )
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
function for_each_uniontype_small(f, ty, counter = Ref(0))
    if counter[] > 127
        return false
    end
    if ty isa Union
        allunbox = for_each_uniontype_small(f, ty.a, counter)
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
        if !(Base.issingletontype(jlrettype) && isa(jlrettype, DataType))
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
    elseif Base.isstructtype(jlrettype) &&
           Base.issingletontype(jlrettype) &&
           isa(jlrettype, DataType)
        # jl_is_structtype(jlrettype) && jl_is_datatype_singleton((jl_datatype_t*)jlrettype)
        return false
    elseif jlrettype isa Union # jl_is_uniontype(jlrettype)
        if union_alloca_type(jlrettype) > 0
            # sret, also a regular return here
            return true
        end
        return false
    elseif !GPUCompiler.deserves_retbox(jlrettype)
        rt = convert(LLVMType, jlrettype)
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
    elseif Base.isstructtype(jlrettype) &&
           Base.issingletontype(jlrettype) &&
           isa(jlrettype, DataType)
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
function get_return_info(
    jlrettype,
)::Tuple{Union{Nothing,Type},Union{Nothing,Type},Union{Nothing,Type}}
    sret = nothing
    returnRoots = nothing
    rt = nothing
    if jlrettype === Union{}
        rt = Nothing
    elseif Base.isstructtype(jlrettype) &&
           Base.issingletontype(jlrettype) &&
           isa(jlrettype, DataType)
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
    elseif jlrettype <: Tuple && in(Any, jlrettype.parameters)
        rt = Any
    elseif !GPUCompiler.deserves_retbox(jlrettype)
        lRT = convert(LLVMType, jlrettype)
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
function lower_convention(
    functy::Type,
    mod::LLVM.Module,
    entry_f::LLVM.Function,
    actualRetType::Type,
    RetActivity,
    TT,
    run_enzyme,
)
    entry_ft = LLVM.function_type(entry_f)

    RT = LLVM.return_type(entry_ft)

    # generate the wrapper function type & definition
    wrapper_types = LLVM.LLVMType[]
    wrapper_attrs = Vector{LLVM.Attribute}[]
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
    swiftself = any(
        any(
            map(
                k -> kind(k) == kind(EnumAttribute("swiftself")),
                collect(parameter_attributes(entry_f, i)),
            ),
        ) for i = 1:length(collect(parameters(entry_f)))
    )
    @assert !swiftself "Swiftself attribute coming from differentiable context is not supported"
    prargs =
        classify_arguments(functy, entry_ft, sret, returnRoots, swiftself, parmsRemoved)
    args = copy(prargs)
    filter!(args) do arg
        Base.@_inline_meta
        arg.cc != GPUCompiler.GHOST && arg.cc != RemovedParam
    end


    # @assert length(args) == length(collect(parameters(entry_f))[1+sret+returnRoots:end])


    # if returnRoots
    # 	push!(wrapper_types, value_type(parameters(entry_f)[1+sret]))
    # end
    #

    if swiftself
        push!(wrapper_types, value_type(parameters(entry_f)[1+sret+returnRoots]))
        push!(wrapper_attrs, LLVM.Attribute[EnumAttribute("swiftself")])
    end

    boxedArgs = Set{Int}()
    loweredArgs = Set{Int}()
    raisedArgs = Set{Int}()

    for arg in args
        typ = arg.codegen.typ
        if GPUCompiler.deserves_argbox(arg.typ)
            push!(boxedArgs, arg.arg_i)
            push!(wrapper_types, typ)
            push!(wrapper_attrs, LLVM.Attribute[])
        elseif arg.cc != GPUCompiler.BITS_REF
            if TT != nothing &&
               (
                   TT.parameters[arg.arg_i] <: MixedDuplicated ||
                   TT.parameters[arg.arg_i] <: BatchMixedDuplicated
               ) &&
               run_enzyme
                push!(boxedArgs, arg.arg_i)
                push!(raisedArgs, arg.arg_i)
                push!(wrapper_types, LLVM.PointerType(typ, Derived))
                push!(wrapper_attrs, LLVM.Attribute[EnumAttribute("noalias")])
            else
                push!(wrapper_types, typ)
                push!(wrapper_attrs, LLVM.Attribute[])
            end
        else
            # bits ref, and not boxed
            if TT != nothing &&
               (
                   TT.parameters[arg.arg_i] <: MixedDuplicated ||
                   TT.parameters[arg.arg_i] <: BatchMixedDuplicated
               ) &&
               run_enzyme
                push!(boxedArgs, arg.arg_i)
                push!(wrapper_types, typ)
                push!(wrapper_attrs, LLVM.Attribute[EnumAttribute("noalias")])
            else
                push!(wrapper_types, eltype(typ))
                push!(wrapper_attrs, LLVM.Attribute[])
                push!(loweredArgs, arg.arg_i)
            end
        end
    end

    if length(loweredArgs) == 0 && length(raisedArgs) == 0 && !sret && !sret_union
        return entry_f, returnRoots, boxedArgs, loweredArgs
    end

    wrapper_fn = LLVM.name(entry_f)
    LLVM.name!(entry_f, safe_name(wrapper_fn * ".inner"))
    wrapper_ft = LLVM.FunctionType(RT, wrapper_types)
    wrapper_f = LLVM.Function(mod, LLVM.name(entry_f), wrapper_ft)
    callconv!(wrapper_f, callconv(entry_f))
    sfn = get_subprogram(entry_f)
    if sfn !== nothing
        set_subprogram!(wrapper_f, sfn)
    end

    hasReturnsTwice = any(
        map(
            k -> kind(k) == kind(EnumAttribute("returns_twice")),
            collect(function_attributes(entry_f)),
        ),
    )
    hasNoInline = any(
        map(
            k -> kind(k) == kind(EnumAttribute("noinline")),
            collect(function_attributes(entry_f)),
        ),
    )
    if hasNoInline
        LLVM.API.LLVMRemoveEnumAttributeAtIndex(
            entry_f,
            reinterpret(LLVM.API.LLVMAttributeIndex, LLVM.API.LLVMAttributeFunctionIndex),
            kind(EnumAttribute("noinline")),
        )
    end
    push!(function_attributes(wrapper_f), EnumAttribute("returns_twice"))
    push!(function_attributes(entry_f), EnumAttribute("returns_twice"))
    for (i, v) in enumerate(wrapper_attrs)
        for attr in v
            push!(parameter_attributes(wrapper_f, i), attr)
        end
    end

    for attr in collect(function_attributes(entry_f))
        if kind(attr) == "enzymejl_world"
            push!(function_attributes(wrapper_f), attr)
        end
    end

    seen = TypeTreeTable()
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
            if swiftself
                push!(nops, ops[1+sret+returnRoots])
            end
            for arg in args
                parm = ops[arg.codegen.i]
                if arg.arg_i in loweredArgs
                    push!(nops, load!(builder, convert(LLVMType, arg.typ), parm))
                elseif arg.arg_i in raisedArgs
                    obj = emit_allocobj!(builder, arg.typ, "raisedArg")
                    bc = bitcast!(
                        builder,
                        obj,
                        LLVM.PointerType(value_type(parm), addrspace(value_type(obj))),
                    )
                    store!(builder, parm, bc)
                    emit_writebarrier!(builder, get_julia_inner_types(builder, obj, parm))
                    addr = addrspacecast!(
                        builder,
                        bc,
                        LLVM.PointerType(value_type(parm), Derived),
                    )
                    push!(nops, addr)
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
                msg = sprint() do io
                    println(io, string(mod))
                    println(io, string(entry_f))
                    println(io, string(e))
                    println(io, "Use after deletion")
                end
                throw(AssertionError(msg))
            end
            LLVM.API.LLVMInstructionEraseFromParent(e)
        end

        entry = BasicBlock(wrapper_f, "entry")
        position!(builder, entry)
        if get_subprogram(entry_f) !== nothing
            debuglocation!(builder, DILocation(0, 0, get_subprogram(entry_f)))
        end

        wrapper_args = Vector{LLVM.Value}()

        sretPtr = nothing
        dl = string(LLVM.datalayout(LLVM.parent(entry_f)))
        if sret
            if !in(0, parmsRemoved)
                sretPtr = alloca!(
                    builder,
                    eltype(value_type(parameters(entry_f)[1])),
                    "innersret",
                )
                ctx = LLVM.context(entry_f)
                if RetActivity <: Const
                    metadata(sretPtr)["enzyme_inactive"] = MDNode(LLVM.Metadata[])
                end
                metadata(sretPtr)["enzyme_type"] =
                    to_md(typetree(Ptr{actualRetType}, ctx, dl, seen), ctx)
                push!(wrapper_args, sretPtr)
            end
            if returnRoots && !in(1, parmsRemoved)
                retRootPtr = alloca!(
                    builder,
                    eltype(value_type(parameters(entry_f)[1+sret])),
                    "innerreturnroots",
                )
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
                    throw(
                        AssertionError(
                            "ty is not a LLVM.PointerType: entry_f = $(entry_f), args = $(args), parm = $(parm), ty = $(ty)",
                        ),
                    )
                end
                ptr = alloca!(builder, eltype(ty), LLVM.name(parm) * ".innerparm")
                if TT !== nothing && TT.parameters[arg.arg_i] <: Const
                    metadata(ptr)["enzyme_inactive"] = MDNode(LLVM.Metadata[])
                end
                ctx = LLVM.context(entry_f)
                metadata(ptr)["enzyme_type"] =
                    to_md(typetree(Ptr{arg.typ}, ctx, dl, seen), ctx)
                if LLVM.addrspace(ty) != 0
                    ptr = addrspacecast!(builder, ptr, ty)
                end
                @assert eltype(ty) == value_type(wrapparm)
                store!(builder, wrapparm, ptr)
                push!(wrapper_args, ptr)
                push!(
                    parameter_attributes(wrapper_f, arg.codegen.i - sret - returnRoots),
                    StringAttribute(
                        "enzyme_type",
                        string(typetree(arg.typ, ctx, dl, seen)),
                    ),
                )
                push!(
                    parameter_attributes(wrapper_f, arg.codegen.i - sret - returnRoots),
                    StringAttribute(
                        "enzymejl_parmtype",
                        string(convert(UInt, unsafe_to_pointer(arg.typ))),
                    ),
                )
                push!(
                    parameter_attributes(wrapper_f, arg.codegen.i - sret - returnRoots),
                    StringAttribute(
                        "enzymejl_parmtype_ref",
                        string(UInt(GPUCompiler.BITS_VALUE)),
                    ),
                )
            elseif arg.arg_i in raisedArgs
                wrapparm = load!(builder, convert(LLVMType, arg.typ), wrapparm)
                ctx = LLVM.context(wrapparm)
                push!(wrapper_args, wrapparm)
                push!(
                    parameter_attributes(wrapper_f, arg.codegen.i - sret - returnRoots),
                    StringAttribute(
                        "enzyme_type",
                        string(typetree(Base.RefValue{arg.typ}, ctx, dl, seen)),
                    ),
                )
                push!(
                    parameter_attributes(wrapper_f, arg.codegen.i - sret - returnRoots),
                    StringAttribute(
                        "enzymejl_parmtype",
                        string(convert(UInt, unsafe_to_pointer(arg.typ))),
                    ),
                )
                push!(
                    parameter_attributes(wrapper_f, arg.codegen.i - sret - returnRoots),
                    StringAttribute(
                        "enzymejl_parmtype_ref",
                        string(UInt(GPUCompiler.BITS_REF)),
                    ),
                )
            else
                push!(wrapper_args, wrapparm)
                for attr in collect(parameter_attributes(entry_f, arg.codegen.i))
                    push!(
                        parameter_attributes(wrapper_f, arg.codegen.i - sret - returnRoots),
                        attr,
                    )
                end
            end
        end
        res = call!(builder, LLVM.function_type(entry_f), entry_f, wrapper_args)

        if get_subprogram(entry_f) !== nothing
            metadata(res)[LLVM.MD_dbg] = DILocation(0, 0, get_subprogram(entry_f))
        end

        callconv!(res, LLVM.callconv(entry_f))
        if swiftself
            attr = EnumAttribute("swiftself")
            LLVM.API.LLVMAddCallSiteAttribute(
                res,
                LLVM.API.LLVMAttributeIndex(1 + sret + returnRoots),
                attr,
            )
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
                        fill_val = unsafe_to_llvm(builder, jlrettype.instance)
                        ret!(builder, fill_val)
                    else
                        nobj = if sretPtr !== nothing
                            obj = emit_allocobj!(builder, jlrettype, "boxunion")
                            llty = convert(LLVMType, jlrettype)
                            ld = load!(
                                builder,
                                llty,
                                bitcast!(
                                    builder,
                                    sretPtr,
                                    LLVM.PointerType(llty, addrspace(value_type(sretPtr))),
                                ),
                            )
                            store!(
                                builder,
                                ld,
                                bitcast!(
                                    builder,
                                    obj,
                                    LLVM.PointerType(llty, addrspace(value_type(obj))),
                                ),
                            )
                            emit_writebarrier!(
                                builder,
                                get_julia_inner_types(builder, obj, ld),
                            )
                            # memcpy!(builder, bitcast!(builder, obj, LLVM.PointerType(T_int8, addrspace(value_type(obj)))), 0, bitcast!(builder, sretPtr, LLVM.PointerType(T_int8)), 0, LLVM.ConstantInt(T_int64, sizeof(jlrettype)))
                            obj
                        else
                            @assert false
                        end
                        ret!(builder, obj)
                    end

                    LLVM.API.LLVMAddCase(
                        sw,
                        LLVM.ConstantInt(value_type(scase), counter),
                        BB,
                    )
                    counter += 1
                    return
                end
                for_each_uniontype_small(inner, actualRetType)

                position!(builder, def)
                ret!(builder, extract_value!(builder, res, 0))

                push!(
                    return_attributes(wrapper_f),
                    StringAttribute(
                        "enzyme_type",
                        string(typetree(actualRetType, ctx, dl, seen)),
                    ),
                )
                push!(
                    return_attributes(wrapper_f),
                    StringAttribute(
                        "enzymejl_parmtype",
                        string(convert(UInt, unsafe_to_pointer(actualRetType))),
                    ),
                )
                push!(
                    return_attributes(wrapper_f),
                    StringAttribute(
                        "enzymejl_parmtype_ref",
                        string(UInt(GPUCompiler.BITS_REF)),
                    ),
                )
            end
        elseif sret
            if sretPtr === nothing
                ret!(builder)
            else
                push!(
                    return_attributes(wrapper_f),
                    StringAttribute(
                        "enzyme_type",
                        string(typetree(actualRetType, ctx, dl, seen)),
                    ),
                )
                push!(
                    return_attributes(wrapper_f),
                    StringAttribute(
                        "enzymejl_parmtype",
                        string(convert(UInt, unsafe_to_pointer(actualRetType))),
                    ),
                )
                push!(
                    return_attributes(wrapper_f),
                    StringAttribute(
                        "enzymejl_parmtype_ref",
                        string(UInt(GPUCompiler.BITS_REF)),
                    ),
                )
                ret!(builder, load!(builder, RT, sretPtr))
            end
        elseif LLVM.return_type(entry_ft) == LLVM.VoidType()
            ret!(builder)
        else
            ctx = LLVM.context(wrapper_f)
            push!(
                return_attributes(wrapper_f),
                StringAttribute(
                    "enzyme_type",
                    string(typetree(actualRetType, ctx, dl, seen)),
                ),
            )
            push!(
                return_attributes(wrapper_f),
                StringAttribute(
                    "enzymejl_parmtype",
                    string(convert(UInt, unsafe_to_pointer(actualRetType))),
                ),
            )
            push!(
                return_attributes(wrapper_f),
                StringAttribute(
                    "enzymejl_parmtype_ref",
                    string(UInt(GPUCompiler.BITS_REF)),
                ),
            )
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
    push!(
        attributes,
        StringAttribute("enzymejl_mi", string(convert(UInt, pointer_from_objref(mi)))),
    )
    push!(
        attributes,
        StringAttribute("enzymejl_rt", string(convert(UInt, unsafe_to_pointer(rt)))),
    )

    for prev in collect(function_attributes(entry_f))
        if kind(prev) == kind(StringAttribute("enzyme_ta_norecur"))
            push!(attributes, prev)
        end
        if kind(prev) == kind(StringAttribute("enzyme_parmremove"))
            push!(attributes, prev)
        end
        if kind(prev) == kind(StringAttribute("enzyme_math"))
            push!(attributes, prev)
        end
        if kind(prev) == kind(StringAttribute("enzyme_shouldrecompute"))
            push!(attributes, prev)
        end
        if LLVM.version().major <= 15
            if kind(prev) == kind(EnumAttribute("readonly"))
                push!(attributes, prev)
            end
            if kind(prev) == kind(EnumAttribute("readnone"))
                push!(attributes, prev)
            end
            if kind(prev) == kind(EnumAttribute("argmemonly"))
                push!(attributes, prev)
            end
            if kind(prev) == kind(EnumAttribute("inaccessiblememonly"))
                push!(attributes, prev)
            end
        end
        if LLVM.version().major > 15
            if kind(prev) == kind(EnumAttribute("memory"))
                old = MemoryEffect(value(attr))
                mem = MemoryEffect(
                    (set_writing(getModRef(old, ArgMem)) << getLocationPos(ArgMem)) |
                    (getModRef(old, InaccessibleMem) << getLocationPos(InaccessibleMem)) |
                    (getModRef(old, Other) << getLocationPos(Other)),
                )
                push!(attributes, EnumAttribute("memory", mem.data))
            end
        end
        if kind(prev) == kind(EnumAttribute("speculatable"))
            push!(attributes, prev)
        end
        if kind(prev) == kind(EnumAttribute("nofree"))
            push!(attributes, prev)
        end
        if kind(prev) == kind(StringAttribute("enzyme_inactive"))
            push!(attributes, prev)
        end
        if kind(prev) == kind(StringAttribute("enzyme_no_escaping_allocation"))
            push!(attributes, prev)
        end
    end

    if LLVM.API.LLVMVerifyFunction(wrapper_f, LLVM.API.LLVMReturnStatusAction) != 0
        msg = sprint() do io
            println(io, string(mod))
            println(
                io,
                LLVM.API.LLVMVerifyFunction(wrapper_f, LLVM.API.LLVMPrintMessageAction),
            )
            println(io, string(wrapper_f))
            println(
                io,
                "parmsRemoved=",
                parmsRemoved,
                " retRemoved=",
                retRemoved,
                " prargs=",
                prargs,
            )
            println(io, "Broken function")
        end
        throw(LLVM.LLVMException(msg))
    end

    ModulePassManager() do pm
        always_inliner!(pm)
        LLVM.run!(pm, mod)
    end
    if !hasReturnsTwice
        LLVM.API.LLVMRemoveEnumAttributeAtIndex(
            wrapper_f,
            reinterpret(LLVM.API.LLVMAttributeIndex, LLVM.API.LLVMAttributeFunctionIndex),
            kind(EnumAttribute("returns_twice")),
        )
    end
    if hasNoInline
        LLVM.API.LLVMRemoveEnumAttributeAtIndex(
            wrapper_f,
            reinterpret(LLVM.API.LLVMAttributeIndex, LLVM.API.LLVMAttributeFunctionIndex),
            kind(EnumAttribute("alwaysinline")),
        )
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
            nvs = Tuple{LLVM.Value,LLVM.BasicBlock}[]
            for (v, b) in LLVM.incoming(p)
                prevbld = IRBuilder()
                position!(prevbld, terminator(b))
                push!(nvs, (extract_value!(prevbld, v, i - 1), b))
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
        LLVM.run!(pm, mod)
    end
    if haskey(globals(mod), "llvm.used")
        eraseInst(mod, globals(mod)["llvm.used"])
        for u in user.(collect(uses(entry_f)))
            if isa(u, LLVM.GlobalVariable) &&
               endswith(LLVM.name(u), "_slot") &&
               startswith(LLVM.name(u), "julia")
                eraseInst(mod, u)
            end
        end
    end

    if LLVM.API.LLVMVerifyFunction(wrapper_f, LLVM.API.LLVMReturnStatusAction) != 0
        msg = sprint() do io
            println(io, string(mod))
            println(
                io,
                LVM.API.LLVMVerifyFunction(wrapper_f, LLVM.API.LLVMPrintMessageAction),
            )
            println(io, string(wrapper_f))
            println(io, "Broken function")
        end
        throw(LLVM.LLVMException(msg))
    end
    return wrapper_f, returnRoots, boxedArgs, loweredArgs
end

using Random
# returns arg, return
function no_type_setting(@nospecialize(specTypes); world = nothing)
    # Even though the julia type here is ptr{int8}, the actual data can be something else
    if specTypes.parameters[1] == typeof(Random.XoshiroSimd.xoshiro_bulk_simd)
        return (true, false)
    end
    if specTypes.parameters[1] == typeof(Random.XoshiroSimd.xoshiro_bulk_nosimd)
        return (true, false)
    end
    return (false, false)
end

const DumpPreOpt = Ref(false)

function GPUCompiler.codegen(
    output::Symbol,
    job::CompilerJob{<:EnzymeTarget};
    libraries::Bool = true,
    deferred_codegen::Bool = true,
    optimize::Bool = true,
    toplevel::Bool = true,
    strip::Bool = false,
    validate::Bool = true,
    only_entry::Bool = false,
    parent_job::Union{Nothing,CompilerJob} = nothing,
)
    params = job.config.params
    if params.run_enzyme
        @assert eltype(params.rt) != Union{}
    end
    expectedTapeType = params.expectedTapeType
    mode = params.mode
    TT = params.TT
    width = params.width
    abiwrap = params.abiwrap
    primal = job.source
    modifiedBetween = params.modifiedBetween
    if length(modifiedBetween) != length(TT.parameters)
        throw(
            AssertionError(
                "length(modifiedBetween) [aka $(length(modifiedBetween))] != length(TT.parameters) [aka $(length(TT.parameters))] at TT=$TT",
            ),
        )
    end
    returnPrimal = params.returnPrimal

    if !(params.rt <: Const)
        @assert !isghostty(eltype(params.rt))
    end
    if parent_job === nothing
        primal_target = DefaultCompilerTarget()
        primal_params = PrimalCompilerParams(mode)
        primal_job = CompilerJob(
            primal,
            CompilerConfig(primal_target, primal_params; kernel = false),
            job.world,
        )
    else
        config2 = CompilerConfig(
            parent_job.config.target,
            parent_job.config.params;
            kernel = false,
            parent_job.config.entry_abi,
            parent_job.config.name,
            parent_job.config.always_inline,
        )
        primal_job = CompilerJob(primal, config2, job.world) # TODO EnzymeInterp params, etc
    end


    mod, meta = GPUCompiler.codegen(
        :llvm,
        primal_job;
        optimize = false,
        toplevel = toplevel,
        cleanup = false,
        validate = false,
        parent_job = parent_job,
    )
    prepare_llvm(mod, primal_job, meta)
    for f in functions(mod)
        permit_inlining!(f)
    end

    LLVM.ModulePassManager() do pm
        API.AddPreserveNVVMPass!(pm, true) #=Begin=#
        LLVM.run!(pm, mod)
    end

    primalf = meta.entry
    check_ir(job, mod)

    disableFallback = String[]

    ForwardModeDerivatives =
        ("nrm2", "dot", "gemm", "gemv", "axpy", "copy", "scal", "symv", "symm", "syrk", "potrf")
    ReverseModeDerivatives = (
        "nrm2",
        "dot",
        "gemm",
        "gemv",
        "axpy",
        "copy",
        "scal",
        "symv",
        "symm",
        "trmv",
        "syrk",
        "trmm",
        "trsm",
        "potrf",
    )
    ForwardModeTypes = ("s", "d", "c", "z")
    ReverseModeTypes = ("s", "d")
    # Tablegen BLAS does not support forward mode yet
    if !(mode == API.DEM_ForwardMode && params.runtimeActivity)
        for ty in (mode == API.DEM_ForwardMode ? ForwardModeTypes : ReverseModeTypes)
            for func in (
                mode == API.DEM_ForwardMode ? ForwardModeDerivatives :
                ReverseModeDerivatives
            )
                for prefix in ("", "cblas_")
                    for ending in ("", "_", "64_", "_64_")
                        push!(disableFallback, prefix * ty * func * ending)
                    end
                end
            end
        end
    end
    found = String[]
    if bitcode_replacement() &&
       API.EnzymeBitcodeReplacement(mod, disableFallback, found) != 0
        ModulePassManager() do pm
            instruction_combining!(pm)
            LLVM.run!(pm, mod)
        end
        toremove = []
        for f in functions(mod)
            if !any(
                map(
                    k -> kind(k) == kind(EnumAttribute("alwaysinline")),
                    collect(function_attributes(f)),
                ),
            )
                continue
            end
            if !any(
                map(
                    k -> kind(k) == kind(EnumAttribute("returns_twice")),
                    collect(function_attributes(f)),
                ),
            )
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
            LLVM.run!(pm, mod)
        end
        for fname in toremove
            if haskey(functions(mod), fname)
                f = functions(mod)[fname]
                LLVM.API.LLVMRemoveEnumAttributeAtIndex(
                    f,
                    reinterpret(
                        LLVM.API.LLVMAttributeIndex,
                        LLVM.API.LLVMAttributeFunctionIndex,
                    ),
                    kind(EnumAttribute("returns_twice")),
                )
            end
        end
        GPUCompiler.@safe_warn "Using fallback BLAS replacements for ($found), performance may be degraded"
        ModulePassManager() do pm
            global_optimizer!(pm)
            LLVM.run!(pm, mod)
        end
    end

    for f in functions(mod)
        mi, RT = enzyme_custom_extract_mi(f, false)
        if mi === nothing
            continue
        end

        llRT, sret, returnRoots = get_return_info(RT)
        retRemoved, parmsRemoved = removed_ret_parms(f)

        dl = string(LLVM.datalayout(LLVM.parent(f)))

        expectLen = (sret !== nothing) + (returnRoots !== nothing)
        for source_typ in mi.specTypes.parameters
            if isghostty(source_typ) || Core.Compiler.isconstType(source_typ)
                continue
            end
            expectLen += 1
        end
        expectLen -= length(parmsRemoved)

        swiftself = any(
            any(
                map(
                    k -> kind(k) == kind(EnumAttribute("swiftself")),
                    collect(parameter_attributes(f, i)),
                ),
            ) for i = 1:length(collect(parameters(f)))
        )

        if swiftself
            expectLen += 1
        end

        # Unsupported calling conv
        # also wouldn't have any type info for this [would for earlier args though]
        if mi.specTypes.parameters[end] === Vararg{Any}
            continue
        end

        world = enzyme_extract_world(f)

        if expectLen != length(parameters(f))
            continue
            throw(
                AssertionError(
                    "Wrong number of parameters $(string(f)) expectLen=$expectLen swiftself=$swiftself sret=$sret returnRoots=$returnRoots spec=$(mi.specTypes.parameters) retRem=$retRemoved parmsRem=$parmsRemoved",
                ),
            )
        end

        jlargs = classify_arguments(
            mi.specTypes,
            function_type(f),
            sret !== nothing,
            returnRoots !== nothing,
            swiftself,
            parmsRemoved,
        )

        ctx = LLVM.context(f)

        push!(function_attributes(f), StringAttribute("enzyme_ta_norecur"))

        if !no_type_setting(mi.specTypes; world)[1]
            for arg in jlargs
                if arg.cc == GPUCompiler.GHOST || arg.cc == RemovedParam
                    continue
                end
                push!(
                    parameter_attributes(f, arg.codegen.i),
                    StringAttribute(
                        "enzymejl_parmtype",
                        string(convert(UInt, unsafe_to_pointer(arg.typ))),
                    ),
                )
                push!(
                    parameter_attributes(f, arg.codegen.i),
                    StringAttribute("enzymejl_parmtype_ref", string(UInt(arg.cc))),
                )

                byref = arg.cc

                rest = typetree(arg.typ, ctx, dl)

                if byref == GPUCompiler.BITS_REF || byref == GPUCompiler.MUT_REF
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
                push!(
                    parameter_attributes(f, arg.codegen.i),
                    StringAttribute("enzyme_type", string(rest)),
                )
            end
        end

        if !no_type_setting(mi.specTypes; world)[2]
            if sret !== nothing
                idx = 0
                if !in(0, parmsRemoved)
                    rest = typetree(sret, ctx, dl)
                    push!(
                        parameter_attributes(f, idx + 1),
                        StringAttribute("enzyme_type", string(rest)),
                    )
                    idx += 1
                end
                if returnRoots !== nothing
                    if !in(1, parmsRemoved)
                        rest = TypeTree(API.DT_Pointer, -1, ctx)
                        push!(
                            parameter_attributes(f, idx + 1),
                            StringAttribute("enzyme_type", string(rest)),
                        )
                    end
                end
            end

            if llRT !== nothing &&
               LLVM.return_type(LLVM.function_type(f)) != LLVM.VoidType()
                @assert !retRemoved
                rest = typetree(llRT, ctx, dl)
                push!(return_attributes(f), StringAttribute("enzyme_type", string(rest)))
            end
        end

    end

    custom = Dict{String,LLVM.API.LLVMLinkage}()
    must_wrap = false

    world = job.world
    interp = GPUCompiler.get_interpreter(job)
    method_table = Core.Compiler.method_table(interp)

    loweredArgs = Set{Int}()
    boxedArgs = Set{Int}()
    actualRetType = nothing
    lowerConvention = true
    customDerivativeNames = String[]
    fnsToInject = Tuple{Symbol,Type}[]
    for (mi, k) in meta.compiled
        k_name = GPUCompiler.safe_name(k.specfunc)
        has_custom_rule = false

        specTypes = Interpreter.simplify_kw(mi.specTypes)

        caller = mi
        if mode == API.DEM_ForwardMode
            has_custom_rule =
                EnzymeRules.has_frule_from_sig(specTypes; world, method_table, caller)
            if has_custom_rule
                @safe_debug "Found frule for" mi.specTypes
            end
        else
            has_custom_rule =
                EnzymeRules.has_rrule_from_sig(specTypes; world, method_table, caller)
            if has_custom_rule
                @safe_debug "Found rrule for" mi.specTypes
            end
        end

        if !haskey(functions(mod), k_name)
            continue
        end

        llvmfn = functions(mod)[k_name]
        if llvmfn == primalf
            actualRetType = k.ci.rettype
        end

        if EnzymeRules.noalias_from_sig(mi.specTypes; world, method_table, caller)
            push!(return_attributes(llvmfn), EnumAttribute("noalias"))
            for u in LLVM.uses(llvmfn)
                c = LLVM.user(u)
                if !isa(c, LLVM.CallInst)
                    continue
                end
                cf = LLVM.called_operand(c)
                if cf == llvmfn
                    LLVM.API.LLVMAddCallSiteAttribute(
                        c,
                        LLVM.API.LLVMAttributeReturnIndex,
                        LLVM.EnumAttribute("noalias", 0),
                    )
                end
            end
        end

        func = mi.specTypes.parameters[1]

        meth = mi.def
        name = meth.name
        jlmod = meth.module

        function handleCustom(llvmfn, name, attrs = [], setlink = true, noinl = true)
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
        if has_custom_rule
            handleCustom(
                llvmfn,
                "enzyme_custom",
                [StringAttribute("enzyme_preserve_primal", "*")],
            )
            continue
        end


        sparam_vals = mi.specTypes.parameters[2:end] # mi.sparam_vals
        if func == typeof(Base.eps) ||
           func == typeof(Base.nextfloat) ||
           func == typeof(Base.prevfloat)
            if LLVM.version().major <= 15
                handleCustom(
                    llvmfn,
                    "jl_inactive_inout",
                    [
                        StringAttribute("enzyme_inactive"),
                        EnumAttribute("readnone"),
                        EnumAttribute("speculatable"),
                        StringAttribute("enzyme_shouldrecompute"),
                    ],
                )
            else
                handleCustom(
                    llvmfn,
                    "jl_inactive_inout",
                    [
                        StringAttribute("enzyme_inactive"),
                        EnumAttribute("memory", NoEffects.data),
                        EnumAttribute("speculatable"),
                        StringAttribute("enzyme_shouldrecompute"),
                    ],
                )
            end
            continue
        end
        if func == typeof(Base.to_tuple_type)
            if LLVM.version().major <= 15
                handleCustom(
                    llvmfn,
                    "jl_to_tuple_type",
                    [
                        EnumAttribute("readonly"),
                        EnumAttribute("inaccessiblememonly", 0),
                        EnumAttribute("speculatable", 0),
                        StringAttribute("enzyme_shouldrecompute"),
                        StringAttribute("enzyme_inactive"),
                    ],
                )
            else
                handleCustom(
                    llvmfn,
                    "jl_to_tuple_type",
                    [
                        EnumAttribute(
                            "memory",
                            MemoryEffect(
                                (MRI_NoModRef << getLocationPos(ArgMem)) |
                                (MRI_Ref << getLocationPos(InaccessibleMem)) |
                                (MRI_NoModRef << getLocationPos(Other)),
                            ).data,
                        ),
                        EnumAttribute("inaccessiblememonly", 0),
                        EnumAttribute("speculatable", 0),
                        StringAttribute("enzyme_shouldrecompute"),
                        StringAttribute("enzyme_inactive"),
                    ],
                )
            end
            continue
        end
        if func == typeof(Base.mightalias)
            if LLVM.version().major <= 15
                handleCustom(
                    llvmfn,
                    "jl_mightalias",
                    [
                        EnumAttribute("readonly"),
                        StringAttribute("enzyme_shouldrecompute"),
                        StringAttribute("enzyme_inactive"),
                        StringAttribute("enzyme_no_escaping_allocation"),
                        EnumAttribute("nofree"),
                        StringAttribute("enzyme_ta_norecur"),
                    ],
                    true,
                    false,
                )
            else
                handleCustom(
                    llvmfn,
                    "jl_mightalias",
                    [
                        EnumAttribute("memory", ReadOnlyEffects.data),
                        StringAttribute("enzyme_shouldrecompute"),
                        StringAttribute("enzyme_inactive"),
                        StringAttribute("enzyme_no_escaping_allocation"),
                        EnumAttribute("nofree"),
                        StringAttribute("enzyme_ta_norecur"),
                    ],
                    true,
                    false,
                )
            end
            continue
        end
        if func == typeof(Base.Threads.threadid) || func == typeof(Base.Threads.nthreads)
            name = (func == typeof(Base.Threads.threadid)) ? "jl_threadid" : "jl_nthreads"
            if LLVM.version().major <= 15
                handleCustom(
                    llvmfn,
                    name,
                    [
                        EnumAttribute("readonly"),
                        EnumAttribute("inaccessiblememonly"),
                        EnumAttribute("speculatable"),
                        StringAttribute("enzyme_shouldrecompute"),
                        StringAttribute("enzyme_inactive"),
                        StringAttribute("enzyme_no_escaping_allocation"),
                    ],
                )
            else
                handleCustom(
                    llvmfn,
                    name,
                    [
                        EnumAttribute(
                            "memory",
                            MemoryEffect(
                                (MRI_NoModRef << getLocationPos(ArgMem)) |
                                (MRI_Ref << getLocationPos(InaccessibleMem)) |
                                (MRI_NoModRef << getLocationPos(Other)),
                            ).data,
                        ),
                        EnumAttribute("speculatable"),
                        StringAttribute("enzyme_shouldrecompute"),
                        StringAttribute("enzyme_inactive"),
                        StringAttribute("enzyme_no_escaping_allocation"),
                    ],
                )
            end
            continue
        end
        # Since this is noreturn and it can't write to any operations in the function
        # in a way accessible by the function. Ideally the attributor should actually
        # handle this and similar not impacting the read/write behavior of the calling
        # fn, but it doesn't presently so for now we will ensure this by hand
        if func == typeof(Base.Checked.throw_overflowerr_binaryop)
            llvmfn = functions(mod)[k.specfunc]
            if LLVM.version().major <= 15
                handleCustom(
                    llvmfn,
                    "enz_noop",
                    [
                        StringAttribute("enzyme_inactive"),
                        EnumAttribute("readonly"),
                        StringAttribute("enzyme_ta_norecur"),
                    ],
                )
            else
                handleCustom(
                    llvmfn,
                    "enz_noop",
                    [
                        StringAttribute("enzyme_inactive"),
                        EnumAttribute("memory", ReadOnlyEffects.data),
                        StringAttribute("enzyme_ta_norecur"),
                    ],
                )
            end
            continue
        end
        if EnzymeRules.is_inactive_from_sig(specTypes; world, method_table, caller) &&
           has_method(
            Tuple{typeof(EnzymeRules.inactive),specTypes.parameters...},
            world,
            method_table,
        )
            handleCustom(
                llvmfn,
                "enz_noop",
                [
                    StringAttribute("enzyme_inactive"),
                    EnumAttribute("nofree"),
                    StringAttribute("enzyme_no_escaping_allocation"),
                    StringAttribute("enzyme_ta_norecur"),
                ],
            )
            continue
        end
        if EnzymeRules.is_inactive_noinl_from_sig(specTypes; world, method_table, caller) &&
           has_method(
            Tuple{typeof(EnzymeRules.inactive_noinl),specTypes.parameters...},
            world,
            method_table,
        )
            handleCustom(
                llvmfn,
                "enz_noop",
                [
                    StringAttribute("enzyme_inactive"),
                    EnumAttribute("nofree"),
                    StringAttribute("enzyme_no_escaping_allocation"),
                    StringAttribute("enzyme_ta_norecur"),
                ],
                false,
                false,
            )
            for bb in blocks(llvmfn)
                for inst in instructions(bb)
                    if isa(inst, LLVM.CallInst)
                        LLVM.API.LLVMAddCallSiteAttribute(
                            inst,
                            reinterpret(
                                LLVM.API.LLVMAttributeIndex,
                                LLVM.API.LLVMAttributeFunctionIndex,
                            ),
                            StringAttribute("no_escaping_allocation"),
                        )
                        LLVM.API.LLVMAddCallSiteAttribute(
                            inst,
                            reinterpret(
                                LLVM.API.LLVMAttributeIndex,
                                LLVM.API.LLVMAttributeFunctionIndex,
                            ),
                            StringAttribute("enzyme_inactive"),
                        )
                        LLVM.API.LLVMAddCallSiteAttribute(
                            inst,
                            reinterpret(
                                LLVM.API.LLVMAttributeIndex,
                                LLVM.API.LLVMAttributeFunctionIndex,
                            ),
                            EnumAttribute("nofree"),
                        )
                    end
                end
            end
            continue
        end
        if func === typeof(Base.match)
            handleCustom(
                llvmfn,
                "base_match",
                [
                    StringAttribute("enzyme_inactive"),
                    EnumAttribute("nofree"),
                    StringAttribute("enzyme_no_escaping_allocation"),
                ],
                false,
                false,
            )
            for bb in blocks(llvmfn)
                for inst in instructions(bb)
                    if isa(inst, LLVM.CallInst)
                        LLVM.API.LLVMAddCallSiteAttribute(
                            inst,
                            reinterpret(
                                LLVM.API.LLVMAttributeIndex,
                                LLVM.API.LLVMAttributeFunctionIndex,
                            ),
                            StringAttribute("no_escaping_allocation"),
                        )
                        LLVM.API.LLVMAddCallSiteAttribute(
                            inst,
                            reinterpret(
                                LLVM.API.LLVMAttributeIndex,
                                LLVM.API.LLVMAttributeFunctionIndex,
                            ),
                            StringAttribute("enzyme_inactive"),
                        )
                        LLVM.API.LLVMAddCallSiteAttribute(
                            inst,
                            reinterpret(
                                LLVM.API.LLVMAttributeIndex,
                                LLVM.API.LLVMAttributeFunctionIndex,
                            ),
                            EnumAttribute("nofree"),
                        )
                    end
                end
            end
            continue
        end
        if func == typeof(Base.enq_work) &&
           length(sparam_vals) == 1 &&
           first(sparam_vals) <: Task
            handleCustom(llvmfn, "jl_enq_work", [StringAttribute("enzyme_ta_norecur")])
            continue
        end
        if func == typeof(Base.wait) || func == typeof(Base._wait)
            if length(sparam_vals) == 1 && first(sparam_vals) <: Task
                handleCustom(llvmfn, "jl_wait", [StringAttribute("enzyme_ta_norecur")])
            end
            continue
        end
        if func == typeof(Base.Threads.threading_run)
            if length(sparam_vals) == 1 || length(sparam_vals) == 2
                handleCustom(llvmfn, "jl_threadsfor")
            end
            continue
        end

        name, toinject, T = find_math_method(func, sparam_vals)
        if name === nothing
            continue
        end

        if toinject !== nothing
            push!(fnsToInject, toinject)
        end

        # If sret, force lower of primitive math fn
        sret = get_return_info(k.ci.rettype)[2] !== nothing
        if sret
            cur = llvmfn == primalf
            llvmfn, _, boxedArgs, loweredArgs = lower_convention(
                mi.specTypes,
                mod,
                llvmfn,
                k.ci.rettype,
                Duplicated,
                nothing,
                params.run_enzyme,
            )
            if cur
                primalf = llvmfn
                lowerConvention = false
            end
            k_name = LLVM.name(llvmfn)
        end

        name = string(name)
        name = T == Float32 ? name * "f" : name

        attrs = if LLVM.version().major <= 15
            [LLVM.EnumAttribute("readnone"), StringAttribute("enzyme_shouldrecompute")]
        else
            [EnumAttribute("memory", NoEffects.data), StringAttribute("enzyme_shouldrecompute")]
        end
        handleCustom(llvmfn, name, attrs)
    end

    @assert actualRetType !== nothing
    if params.run_enzyme
        @assert actualRetType != Union{}
    end

    if must_wrap
        llvmfn = primalf
        FT = LLVM.function_type(llvmfn)

        wrapper_f = LLVM.Function(mod, safe_name(LLVM.name(llvmfn) * "mustwrap"), FT)

        let builder = IRBuilder()
            entry = BasicBlock(wrapper_f, "entry")
            position!(builder, entry)

            res = call!(
                builder,
                LLVM.function_type(llvmfn),
                llvmfn,
                collect(parameters(wrapper_f)),
            )

            sretkind = kind(if LLVM.version().major >= 12
                TypeAttribute("sret", LLVM.Int32Type())
            else
                EnumAttribute("sret")
            end)
            for idx in length(collect(parameters(llvmfn)))
                for attr in collect(parameter_attributes(llvmfn, idx))
                    if kind(attr) == sretkind
                        LLVM.API.LLVMAddCallSiteAttribute(
                            res,
                            LLVM.API.LLVMAttributeIndex(idx),
                            attr,
                        )
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
        push!(
            attributes,
            StringAttribute("enzymejl_mi", string(convert(UInt, pointer_from_objref(mi)))),
        )
        push!(
            attributes,
            StringAttribute("enzymejl_rt", string(convert(UInt, unsafe_to_pointer(rt)))),
        )
        primalf = wrapper_f
    end

    source_sig = job.source.specTypes


    primalf, returnRoots = primalf, false

    if lowerConvention
        primalf, returnRoots, boxedArgs, loweredArgs = lower_convention(
            source_sig,
            mod,
            primalf,
            actualRetType,
            job.config.params.rt,
            TT,
            params.run_enzyme,
        )
    end

    if primal_job.config.target isa GPUCompiler.NativeCompilerTarget
        target_machine = JIT.get_tm()
    else
        target_machine = GPUCompiler.llvm_machine(primal_job.config.target)
    end

    parallel = parent_job === nothing ? Threads.nthreads() > 1 : false
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
    for name in ("gpu_report_exception", "report_exception")
        if haskey(functions(mod), name)
            exc = functions(mod)[name]
            if !isempty(blocks(exc))
                linkage!(exc, LLVM.API.LLVMExternalLinkage)
            end
        end
    end

    if DumpPreOpt[]
        API.EnzymeDumpModuleRef(mod.ref)
    end

    # Run early pipeline
    optimize!(mod, target_machine)

    if process_module
        GPUCompiler.optimize_module!(parent_job, mod)
    end

    for name in ("gpu_report_exception", "report_exception")
        if haskey(functions(mod), name)
            exc = functions(mod)[name]
            if !isempty(blocks(exc))
                linkage!(exc, LLVM.API.LLVMInternalLinkage)
            end
        end
    end

    seen = TypeTreeTable()
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    dl = string(LLVM.datalayout(mod))
    ctx = LLVM.context(mod)
    for f in functions(mod), bb in blocks(f), inst in instructions(bb)
        fn = isa(inst, LLVM.CallInst) ? LLVM.called_operand(inst) : nothing

        if !API.HasFromStack(inst) &&
           isa(inst, LLVM.CallInst) &&
           (!isa(fn, LLVM.Function) || isempty(blocks(fn)))
            legal, source_typ, byref = abs_typeof(inst)
            codegen_typ = value_type(inst)
            if legal
                typ = if codegen_typ isa LLVM.PointerType
                    llvm_source_typ = convert(LLVMType, source_typ; allow_boxed = true)
                    # pointers are used for multiple kinds of arguments
                    # - literal pointer values
                    if source_typ <: Ptr || source_typ <: Core.LLVMPtr
                        source_typ
                    elseif byref == GPUCompiler.MUT_REF || byref == GPUCompiler.BITS_REF
                        Ptr{source_typ}
                    else
                        # println(string(mod))
                        println(string(f))
                        @show legal, source_typ, byref, llvm_source_typ, codegen_typ, string(inst)
                        @show enzyme_custom_extract_mi(f)
                        @assert false
                    end
                else
                    source_typ
                end

                if isa(inst, LLVM.CallInst)
                    LLVM.API.LLVMAddCallSiteAttribute(
                        inst,
                        LLVM.API.LLVMAttributeReturnIndex,
                        StringAttribute(
                            "enzyme_type",
                            string(typetree(typ, ctx, dl, seen)),
                        ),
                    )
                else
                    metadata(inst)["enzyme_type"] = to_md(typetree(typ, ctx, dl, seen), ctx)
                end
            elseif codegen_typ == T_prjlvalue
                if isa(inst, LLVM.CallInst)
                    LLVM.API.LLVMAddCallSiteAttribute(
                        inst,
                        LLVM.API.LLVMAttributeReturnIndex,
                        StringAttribute("enzyme_type", "{[-1]:Pointer}"),
                    )
                else
                    metadata(inst)["enzyme_type"] =
                        to_md(typetree(Ptr{Cvoid}, ctx, dl, seen), ctx)
                end
            end
        end

        if isa(inst, LLVM.CallInst)
            if !isa(fn, LLVM.Function)
                continue
            end
            if length(blocks(fn)) != 0
                continue
            end

            intr = LLVM.API.LLVMGetIntrinsicID(fn)

            if intr == LLVM.Intrinsic("llvm.memcpy").id ||
               intr == LLVM.Intrinsic("llvm.memmove").id ||
               intr == LLVM.Intrinsic("llvm.memset").id
                base, offset, _ = get_base_and_offset(operands(inst)[1])
                legal, jTy, byref = abs_typeof(base)
                sz =
                    if intr == LLVM.Intrinsic("llvm.memcpy").id ||
                       intr == LLVM.Intrinsic("llvm.memmove").id
                        operands(inst)[3]
                    else
                        operands(inst)[3]
                    end
                if legal && Base.isconcretetype(jTy)
                    if !(
                        jTy isa UnionAll ||
                        jTy isa Union ||
                        jTy == Union{} ||
                        jTy === Tuple ||
                        (
                            is_concrete_tuple(jTy) &&
                            any(T2 isa Core.TypeofVararg for T2 in jTy.parameters)
                        )
                    )
                        if offset < sizeof(jTy) && isa(sz, LLVM.ConstantInt) && sizeof(jTy) - offset >= convert(Int, sz)
                            lim = convert(Int, sz)
                            md = to_fullmd(jTy, offset, lim)
                            @assert byref == GPUCompiler.BITS_REF ||
                                    byref == GPUCompiler.MUT_REF
                            metadata(inst)["enzyme_truetype"] = md
                        end
                    end
                end
            end
        end

        ty = value_type(inst)
        if ty == LLVM.VoidType()
            continue
        end

        legal, jTy, byref = abs_typeof(inst, true)
        if !legal
            continue
        end
        if !guaranteed_const_nongen(jTy, world)
            continue
        end
        if isa(inst, LLVM.CallInst)
            LLVM.API.LLVMAddCallSiteAttribute(
                inst,
                LLVM.API.LLVMAttributeReturnIndex,
                StringAttribute("enzyme_inactive"),
            )
        else
            metadata(inst)["enzyme_inactive"] = MDNode(LLVM.Metadata[])
        end
    end


    TapeType::Type = Cvoid

    if params.err_if_func_written
        FT = TT.parameters[1]
        Ty = eltype(FT)
        reg = active_reg_inner(Ty, (), world)
        if reg == DupState || reg == MixedState
            swiftself = any(
                any(
                    map(
                        k -> kind(k) == kind(EnumAttribute("swiftself")),
                        collect(parameter_attributes(primalf, i)),
                    ),
                ) for i = 1:length(collect(parameters(primalf)))
            )
            todo = LLVM.Value[parameters(primalf)[1+swiftself]]
            done = Set{LLVM.Value}()
            doneInst = Set{LLVM.Instruction}()
            while length(todo) != 0
                cur = pop!(todo)
                if cur in done
                    continue
                end
                push!(done, cur)
                for u in LLVM.uses(cur)
                    user = LLVM.user(u)
                    if user in doneInst
                        continue
                    end
                    if LLVM.API.LLVMIsAReturnInst(user) != C_NULL
                        continue
                    end

                    if !mayWriteToMemory(user)
                        slegal, foundv, byref = abs_typeof(user)
                        if slegal
                            reg2 = active_reg_inner(foundv, (), world)
                            if reg2 == ActiveState || reg2 == AnyState
                                continue
                            end
                        end
                        push!(todo, user)
                        continue
                    end

                    if isa(user, LLVM.StoreInst)
                        # we are capturing the variable
                        if operands(user)[1] == cur
                            base = operands(user)[2]
                            while isa(base, LLVM.BitCastInst) ||
                                      isa(base, LLVM.AddrSpaceCastInst) ||
                                      isa(base, LLVM.GetElementPtrInst)
                                base = operands(base)[1]
                            end
                            if isa(base, LLVM.AllocaInst)
                                push!(doneInst, user)
                                push!(todo, base)
                                continue
                            end
                        end
                        # we are storing into the variable
                        if operands(user)[2] == cur
                            slegal, foundv, byref = abs_typeof(operands(user)[1])
                            if slegal
                                reg2 = active_reg_inner(foundv, (), world)
                                if reg2 == AnyState
                                    continue
                                end
                            end
                        end
                    end

                    if isa(user, LLVM.CallInst)
                        called = LLVM.called_operand(user)
                        if isa(called, LLVM.Function)
                            intr = LLVM.API.LLVMGetIntrinsicID(called)
                            if intr == LLVM.Intrinsic("llvm.memset").id
                                if cur != operands(user)[1]
                                    continue
                                end
                            end

                            nm = LLVM.name(called)
                            if nm == "ijl_alloc_array_1d" ||
                               nm == "jl_alloc_array_1d" ||
                               nm == "ijl_alloc_array_2d" ||
                               nm == "jl_alloc_array_2d" ||
                               nm == "ijl_alloc_array_3d" ||
                               nm == "jl_alloc_array_3d" ||
                               nm == "ijl_new_array" ||
                               nm == "jl_new_array" ||
                               nm == "jl_alloc_genericmemory" ||
                               nm == "ijl_alloc_genericmemory"
                                continue
                            end
                            if is_readonly(called)
                                slegal, foundv, byref = abs_typeof(user)
                                if slegal
                                    reg2 = active_reg_inner(foundv, (), world)
                                    if reg2 == ActiveState || reg2 == AnyState
                                        continue
                                    end
                                end
                                push!(todo, user)
                                continue
                            end
                            if !isempty(blocks(called)) &&
                               length(collect(LLVM.uses(called))) == 1
                                for (parm, op) in
                                    zip(LLVM.parameters(called), operands(user)[1:end-1])
                                    if op == cur
                                        push!(todo, parm)
                                    end
                                end
                                slegal, foundv, byref = abs_typeof(user)
                                if slegal
                                    reg2 = active_reg_inner(foundv, (), world)
                                    if reg2 == ActiveState || reg2 == AnyState
                                        continue
                                    end
                                end
                                push!(todo, user)
                                continue
                            end
                        end
                    end

                    builder = LLVM.IRBuilder()
                    position!(builder, user)
                    resstr =
                        "Function argument passed to autodiff cannot be proven readonly.\nIf the the function argument cannot contain derivative data, instead call autodiff(Mode, Const(f), ...)\nSee https://enzyme.mit.edu/index.fcgi/julia/stable/faq/#Activity-of-temporary-storage for more information.\nThe potentially writing call is " *
                        string(user) *
                        ", using " *
                        string(cur)
                    slegal, foundv = absint(cur)
                    if slegal
                        resstr *= "of type " * string(foundv)
                    end
                    emit_error(builder, user, resstr, EnzymeMutabilityException)
                end
            end
        end
    end

    if params.run_enzyme
        # Generate the adjoint
        memcpy_alloca_to_loadstore(mod)

        adjointf, augmented_primalf, TapeType = enzyme!(
            job,
            mod,
            primalf,
            TT,
            mode,
            width,
            parallel,
            actualRetType,
            abiwrap,
            modifiedBetween,
            returnPrimal,
            expectedTapeType,
            loweredArgs,
            boxedArgs,
        )
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
                                if nm == "gpu_signal_exception" ||
                                   nm == "gpu_report_exception" ||
                                   nm == "ijl_throw" ||
                                   nm == "jl_throw"
                                    shouldemit = false
                                    break
                                end
                            end
                        end
                    end

                    if shouldemit
                        b = IRBuilder()
                        position!(b, term)
                        emit_error(
                            b,
                            term,
                            "Enzyme: The original primal code hits this error condition, thus differentiating it does not make sense",
                        )
                    end
                end
            end
            if !any(
                map(
                    k -> kind(k) == kind(EnumAttribute("alwaysinline")),
                    collect(function_attributes(f)),
                ),
            )
                continue
            end
            if !any(
                map(
                    k -> kind(k) == kind(EnumAttribute("returns_twice")),
                    collect(function_attributes(f)),
                ),
            )
                push!(function_attributes(f), EnumAttribute("returns_twice"))
                push!(toremove, name(f))
            end
        end
        ModulePassManager() do pm
            always_inliner!(pm)
            LLVM.run!(pm, mod)
        end
        for fname in toremove
            if haskey(functions(mod), fname)
                f = functions(mod)[fname]
                LLVM.API.LLVMRemoveEnumAttributeAtIndex(
                    f,
                    reinterpret(
                        LLVM.API.LLVMAttributeIndex,
                        LLVM.API.LLVMAttributeFunctionIndex,
                    ),
                    kind(EnumAttribute("returns_twice")),
                )
            end
        end
    else
        adjointf = primalf
        augmented_primalf = nothing
    end

    LLVM.ModulePassManager() do pm
        API.AddPreserveNVVMPass!(pm, false) #=Begin=#
        LLVM.run!(pm, mod)
    end
    if parent_job !== nothing
        if parent_job.config.target isa GPUCompiler.PTXCompilerTarget
            arg1 = (
                "sin",
                "cos",
                "tan",
                "log2",
                "exp",
                "exp2",
                "exp10",
                "cosh",
                "sinh",
                "tanh",
                "atan",
                "asin",
                "acos",
                "log",
                "log10",
                "log1p",
                "acosh",
                "asinh",
                "atanh",
                "expm1",
                "cbrt",
                "rcbrt",
                "j0",
                "j1",
                "y0",
                "y1",
                "erf",
                "erfinv",
                "erfc",
                "erfcx",
                "erfcinv",
                "remquo",
                "tgamma",
                "round",
                "fdim",
                "logb",
                "isinf",
                "sqrt",
                "fabs",
                "atan2",
            )
            # isinf, finite "modf",       "fmod",    "remainder", 
            # "rnorm3d",    "norm4d",  "rnorm4d",   "norm",   "rnorm",
            #   "hypot",  "rhypot",
            # "yn", "jn", "norm3d", "ilogb", powi
            # "normcdfinv", "normcdf", "lgamma",    "ldexp",  "scalbn", "frexp",
            # arg1 = ("atan2", "fmax", "pow")
            for n in arg1,
                (T, pf, lpf) in
                ((LLVM.DoubleType(), "", "f64"), (LLVM.FloatType(), "f", "f32"))

                fname = "__nv_" * n * pf
                if !haskey(functions(mod), fname)
                    FT = LLVM.FunctionType(T, [T], vararg = false)
                    wrapper_f = LLVM.Function(mod, fname, FT)
                    llname = "llvm." * n * "." * lpf
                    push!(
                        function_attributes(wrapper_f),
                        StringAttribute("implements", llname),
                    )
                end
            end
        end
        if parent_job.config.target isa GPUCompiler.GCNCompilerTarget
            arg1 = (
                "acos",
                "acosh",
                "asin",
                "asinh",
                "atan2",
                "atan",
                "atanh",
                "cbrt",
                "ceil",
                "copysign",
                "cos",
                "native_cos",
                "cosh",
                "cospi",
                "i0",
                "i1",
                "erfc",
                "erfcinv",
                "erfcx",
                "erf",
                "erfinv",
                "exp10",
                "native_exp10",
                "exp2",
                "exp",
                "native_exp",
                "expm1",
                "fabs",
                "fdim",
                "floor",
                "fma",
                "fmax",
                "fmin",
                "fmod",
                "frexp",
                "hypot",
                "ilogb",
                "isfinite",
                "isinf",
                "isnan",
                "j0",
                "j1",
                "ldexp",
                "lgamma",
                "log10",
                "native_log10",
                "log1p",
                "log2",
                "log2",
                "logb",
                "log",
                "native_log",
                "modf",
                "nearbyint",
                "nextafter",
                "len3",
                "len4",
                "ncdf",
                "ncdfinv",
                "pow",
                "pown",
                "rcbrt",
                "remainder",
                "remquo",
                "rhypot",
                "rint",
                "rlen3",
                "rlen4",
                "round",
                "rsqrt",
                "scalb",
                "scalbn",
                "signbit",
                "sincos",
                "sincospi",
                "sin",
                "native_sin",
                "sinh",
                "sinpi",
                "sqrt",
                "native_sqrt",
                "tan",
                "tanh",
                "tgamma",
                "trunc",
                "y0",
                "y1",
            )
            for n in arg1,
                (T, pf, lpf) in
                ((LLVM.DoubleType(), "", "f64"), (LLVM.FloatType(), "f", "f32"))

                fname = "__ocml_" * n * "_" * lpf
                if !haskey(functions(mod), fname)
                    FT = LLVM.FunctionType(T, [T], vararg = false)
                    wrapper_f = LLVM.Function(mod, fname, FT)
                    llname = "llvm." * n * "." * lpf
                    push!(
                        function_attributes(wrapper_f),
                        StringAttribute("implements", llname),
                    )
                end
            end
        end
    end
    for (name, fnty) in fnsToInject
        for (T, JT, pf) in
            ((LLVM.DoubleType(), Float64, ""), (LLVM.FloatType(), Float32, "f"))
            fname = String(name) * pf
            if haskey(functions(mod), fname)
                funcspec = my_methodinstance(fnty, Tuple{JT}, world)
                llvmf = nested_codegen!(mode, mod, funcspec, world)
                push!(function_attributes(llvmf), StringAttribute("implements", fname))
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
    for fname in
        ["__enzyme_float", "__enzyme_double", "__enzyme_integer", "__enzyme_pointer"]
        haskey(functions(mod), fname) || continue
        f = functions(mod)[fname]
        for u in uses(f)
            st = LLVM.user(u)
            LLVM.API.LLVMInstructionEraseFromParent(st)
        end
        eraseInst(mod, f)
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
        post_optimze!(mod, target_machine, false) #=machine=#
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

    use_primal = mode == API.DEM_ReverseModePrimal
    entry = use_primal ? augmented_primalf : adjointf
    return mod, (; adjointf, augmented_primalf, entry, compiled = meta.compiled, TapeType)
end

# Compiler result
struct CompileResult{AT,PT}
    adjoint::AT
    primal::PT
    TapeType::Type
end

@inline (thunk::PrimalErrorThunk{PT,FA,RT,TT,Width,ReturnPrimal})(
    fn::FA,
    args...,
) where {PT,FA,RT,TT,Width,ReturnPrimal} = enzyme_call(
    Val(false),
    thunk.adjoint,
    PrimalErrorThunk{PT,FA,RT,TT,Width,ReturnPrimal},
    Val(Width),
    Val(ReturnPrimal),
    TT,
    RT,
    fn,
    Cvoid,
    args...,
)

@inline (thunk::CombinedAdjointThunk{PT,FA,RT,TT,Width,ReturnPrimal})(
    fn::FA,
    args...,
) where {PT,FA,Width,RT,TT,ReturnPrimal} = enzyme_call(
    Val(false),
    thunk.adjoint,
    CombinedAdjointThunk{PT,FA,RT,TT,Width,ReturnPrimal},
    Val(Width),
    Val(ReturnPrimal),
    TT,
    RT,
    fn,
    Cvoid,
    args...,
)

@inline (thunk::ForwardModeThunk{PT,FA,RT,TT,Width,ReturnPrimal})(
    fn::FA,
    args...,
) where {PT,FA,Width,RT,TT,ReturnPrimal} = enzyme_call(
    Val(false),
    thunk.adjoint,
    ForwardModeThunk{PT,FA,RT,TT,Width,ReturnPrimal},
    Val(Width),
    Val(ReturnPrimal),
    TT,
    RT,
    fn,
    Cvoid,
    args...,
)

@inline (thunk::AdjointThunk{PT,FA,RT,TT,Width,TapeT})(
    fn::FA,
    args...,
) where {PT,FA,Width,RT,TT,TapeT} = enzyme_call(
    Val(false),
    thunk.adjoint,
    AdjointThunk{PT,FA,RT,TT,Width,TapeT},
    Val(Width),
    Val(false),
    TT,
    RT,
    fn,
    TapeT,
    args...,
) #=ReturnPrimal=#
@inline raw_enzyme_call(
    thunk::AdjointThunk{PT,FA,RT,TT,Width,TapeT},
    fn::FA,
    args...,
) where {PT,FA,Width,RT,TT,TapeT} = enzyme_call(
    Val(true),
    thunk.adjoint,
    AdjointThunk,
    Val(Width),
    Val(false),
    TT,
    RT,
    fn,
    TapeT,
    args...,
) #=ReturnPrimal=#

@inline (thunk::AugmentedForwardThunk{PT,FA,RT,TT,Width,ReturnPrimal,TapeT})(
    fn::FA,
    args...,
) where {PT,FA,Width,RT,TT,ReturnPrimal,TapeT} = enzyme_call(
    Val(false),
    thunk.primal,
    AugmentedForwardThunk,
    Val(Width),
    Val(ReturnPrimal),
    TT,
    RT,
    fn,
    TapeT,
    args...,
)
@inline raw_enzyme_call(
    thunk::AugmentedForwardThunk{PT,FA,RT,TT,Width,ReturnPrimal,TapeT},
    fn::FA,
    args...,
) where {PT,FA,Width,RT,TT,ReturnPrimal,TapeT} = enzyme_call(
    Val(true),
    thunk.primal,
    AugmentedForwardThunk,
    Val(Width),
    Val(ReturnPrimal),
    TT,
    RT,
    fn,
    TapeT,
    args...,
)


function jl_set_typeof(v::Ptr{Cvoid}, T)
    tag = reinterpret(Ptr{Any}, reinterpret(UInt, v) - 8)
    Base.unsafe_store!(tag, T) # set tag
    return nothing
end

@generated function splatnew(::Type{T}, args::TT) where {T,TT<:Tuple}
    return quote
        Base.@_inline_meta
        $(Expr(:splatnew, :T, :args))
    end
end

# Recursively return x + f(y), where y is active, otherwise x

@inline function recursive_add(
    x::T,
    y::T,
    f::F = identity,
    forcelhs::F2 = guaranteed_const,
) where {T,F,F2}
    if forcelhs(T)
        return x
    end
    splatnew(T, ntuple(Val(fieldcount(T))) do i
        Base.@_inline_meta
        prev = getfield(x, i)
        next = getfield(y, i)
        recursive_add(prev, next, f, forcelhs)
    end)
end

@inline function recursive_add(
    x::T,
    y::T,
    f::F = identity,
    forcelhs::F2 = guaranteed_const,
) where {T<:AbstractFloat,F,F2}
    if forcelhs(T)
        return x
    end
    return x + f(y)
end

@inline function recursive_add(
    x::T,
    y::T,
    f::F = identity,
    forcelhs::F2 = guaranteed_const,
) where {T<:Complex,F,F2}
    if forcelhs(T)
        return x
    end
    return x + f(y)
end

@inline mutable_register(::Type{T}) where {T<:Integer} = true
@inline mutable_register(::Type{T}) where {T<:AbstractFloat} = false
@inline mutable_register(::Type{Complex{T}}) where {T<:AbstractFloat} = false
@inline mutable_register(::Type{T}) where {T<:Tuple} = false
@inline mutable_register(::Type{T}) where {T<:NamedTuple} = false
@inline mutable_register(::Type{Core.Box}) = true
@inline mutable_register(::Type{T}) where {T<:Array} = true
@inline mutable_register(::Type{T}) where {T} = ismutabletype(T)

# Recursively In-place accumulate(aka +=). E.g. generalization of x .+= f(y)
@inline function recursive_accumulate(x::Array{T}, y::Array{T}, f::F = identity) where {T,F}
    if !mutable_register(T)
        for I in eachindex(x)
            prev = x[I]
            @inbounds x[I] = recursive_add(x[I], (@inbounds y[I]), f, mutable_register)
        end
    end
end


# Recursively In-place accumulate(aka +=). E.g. generalization of x .+= f(y)
@inline function recursive_accumulate(x::Core.Box, y::Core.Box, f::F = identity) where {F}
    recursive_accumulate(x.contents, y.contents, seen, f)
end

@inline function recursive_accumulate(x::T, y::T, f::F = identity) where {T,F}
    @assert !Base.isabstracttype(T)
    @assert Base.isconcretetype(T)
    nf = fieldcount(T)

    for i = 1:nf
        if isdefined(x, i)
            xi = getfield(x, i)
            ST = Core.Typeof(xi)
            if !mutable_register(ST)
                @assert ismutable(x)
                yi = getfield(y, i)
                nexti = recursive_add(xi, yi, f, mutable_register)
                setfield!(x, i, nexti)
            end
        end
    end
end

@inline function default_adjoint(T)
    if T == Union{}
        return nothing
    elseif T <: AbstractFloat
        return one(T)
    elseif T <: Complex
        error(
            "Attempted to use automatic pullback (differential return value) deduction on a either a type unstable function returning an active complex number, or autodiff_deferred returning an active complex number. For the first case, please type stabilize your code, e.g. by specifying autodiff(Reverse, f->f(x)::Complex, ...). For the second case, please use regular non-deferred autodiff",
        )
    else
        error(
            "Active return values with automatic pullback (differential return value) deduction only supported for floating-like values and not type $T. If mutable memory, please use Duplicated. Otherwise, you can explicitly specify a pullback by using split mode, e.g. autodiff_thunk(ReverseSplitWithPrimal, ...)",
        )
    end
end

function add_one_in_place(x)
    if x isa Base.RefValue
        x[] = recursive_add(x[], default_adjoint(eltype(Core.Typeof(x))))
    elseif x isa (Array{T,0} where T)
        x[] = recursive_add(x[], default_adjoint(eltype(Core.Typeof(x))))
    else
        error(
            "Enzyme Mutability Error: Cannot add one in place to immutable value " *
            string(x),
        )
    end
    return nothing
end

@generated function enzyme_call(
    ::Val{RawCall},
    fptr::PT,
    ::Type{CC},
    ::Val{width},
    ::Val{returnPrimal},
    tt::Type{T},
    rt::Type{RT},
    fn::FA,
    ::Type{TapeType},
    args::Vararg{Any,N},
) where {RawCall,PT,FA,T,RT,TapeType,N,CC,width,returnPrimal}

    JuliaContext() do ctx
        Base.@_inline_meta
        F = eltype(FA)
        is_forward =
            CC <: AugmentedForwardThunk || CC <: ForwardModeThunk || CC <: PrimalErrorThunk
        is_adjoint = CC <: AdjointThunk || CC <: CombinedAdjointThunk
        is_split = CC <: AdjointThunk || CC <: AugmentedForwardThunk
        needs_tape = CC <: AdjointThunk

        argtt = tt.parameters[1]
        rettype = rt.parameters[1]
        argtypes = DataType[argtt.parameters...]
        argexprs = Union{Expr,Symbol}[:(args[$i]) for i = 1:N]

        if false && CC <: PrimalErrorThunk
            primargs = [
                quote
                    convert($(eltype(T)), $(argexprs[i]).val)
                end for (i, T) in enumerate(argtypes)
            ]
            return quote
                fn.val($(primargs...))
                error(
                    "Function to differentiate is guaranteed to return an error and doesn't make sense to autodiff. Giving up",
                )
            end
        end

        if !RawCall && !(CC <: PrimalErrorThunk)
            if rettype <: Active ||
               rettype <: MixedDuplicated ||
               rettype <: BatchMixedDuplicated
                if length(argtypes) + is_adjoint + needs_tape != length(argexprs)
                    return quote
                        throw(MethodError($CC(fptr), (fn, args...)))
                    end
                end
            elseif rettype <: Const
                if length(argtypes) + needs_tape != length(argexprs)
                    return quote
                        throw(MethodError($CC(fptr), (fn, args...)))
                    end
                end
            else
                if length(argtypes) + needs_tape != length(argexprs)
                    return quote
                        throw(MethodError($CC(fptr), (fn, args...)))
                    end
                end
            end
        end

        types = DataType[]

        if !(rettype <: Const) && (
            isghostty(eltype(rettype)) ||
            Core.Compiler.isconstType(eltype(rettype)) ||
            eltype(rettype) === DataType
        )
            rrt = eltype(rettype)
            error("Return type `$rrt` not marked Const, but is ghost or const type.")
        end

        sret_types = []  # Julia types of all returned variables
        # By ref values we create and need to preserve
        ccexprs = Union{Expr,Symbol}[] # The expressions passed to the `llvmcall`

        if !isghostty(F) && !Core.Compiler.isconstType(F)
            isboxed = GPUCompiler.deserves_argbox(F)
            argexpr = :(fn.val)

            if isboxed
                push!(types, Any)
            else
                push!(types, F)
            end

            push!(ccexprs, argexpr)
            if (FA <: Active)
                return quote
                    error("Cannot have function with Active annotation, $FA")
                end
            elseif !(FA <: Const)
                argexpr = :(fn.dval)
                if isboxed
                    push!(types, Any)
                elseif width == 1
                    push!(types, F)
                else
                    push!(types, NTuple{width,F})
                end
                push!(ccexprs, argexpr)
            end
        end

        i = 1
        ActiveRetTypes = Type[]

        for T in argtypes
            source_typ = eltype(T)

            expr = argexprs[i]
            i += 1
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
            if CC <: PrimalErrorThunk
                continue
            end
            if T <: Active
                if is_adjoint
                    if width == 1
                        push!(ActiveRetTypes, source_typ)
                    else
                        push!(ActiveRetTypes, NTuple{width,source_typ})
                    end
                end
            elseif T <: Duplicated || T <: DuplicatedNoNeed
                if RawCall
                    argexpr = argexprs[i]
                    i += 1
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
                    i += 1
                else
                    argexpr = Expr(:., expr, QuoteNode(:dval))
                end
                isboxedvec = GPUCompiler.deserves_argbox(NTuple{width,source_typ})
                if isboxedvec
                    push!(types, Any)
                else
                    push!(types, NTuple{width,source_typ})
                end
                if is_adjoint
                    push!(ActiveRetTypes, Nothing)
                end
                push!(ccexprs, argexpr)
            elseif T <: MixedDuplicated
                if RawCall
                    argexpr = argexprs[i]
                    i += 1
                else
                    argexpr = Expr(:., expr, QuoteNode(:dval))
                end
                push!(types, Any)
                if is_adjoint
                    push!(ActiveRetTypes, Nothing)
                end
                push!(ccexprs, argexpr)
            elseif T <: BatchMixedDuplicated
                if RawCall
                    argexpr = argexprs[i]
                    i += 1
                else
                    argexpr = Expr(:., expr, QuoteNode(:dval))
                end
                isboxedvec =
                    GPUCompiler.deserves_argbox(NTuple{width,Base.RefValue{source_typ}})
                if isboxedvec
                    push!(types, Any)
                else
                    push!(types, NTuple{width,Base.RefValue{source_typ}})
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
        if is_adjoint
            if rettype <: Active ||
               rettype <: MixedDuplicated ||
               rettype <: BatchMixedDuplicated
                # TODO handle batch width
                if rettype <: Active
                    @assert allocatedinline(jlRT)
                end
                j_drT = if width == 1
                    jlRT
                else
                    NTuple{width,jlRT}
                end
                push!(types, j_drT)
                push!(ccexprs, argexprs[i])
                i += 1
            end
        end

        if needs_tape
            if !(isghostty(TapeType) || Core.Compiler.isconstType(TapeType))
                push!(types, TapeType)
                push!(ccexprs, argexprs[i])
            end
            i += 1
        end


        if is_adjoint
            NT = Tuple{ActiveRetTypes...}
            if any(
                any_jltypes(convert(LLVM.LLVMType, b; allow_boxed = true)) for
                b in ActiveRetTypes
            )
                NT = AnonymousStruct(NT)
            end
            push!(sret_types, NT)
        end

        if !(CC <: PrimalErrorThunk)
            @assert i == length(argexprs) + 1
        end

        # Tape
        if CC <: AugmentedForwardThunk
            push!(sret_types, TapeType)
        end

        if returnPrimal && !(CC <: ForwardModeThunk)
            push!(sret_types, jlRT)
        end
        if is_forward
            if !returnPrimal && CC <: AugmentedForwardThunk
                push!(sret_types, Nothing)
            end
            if rettype <: Duplicated || rettype <: DuplicatedNoNeed
                push!(sret_types, jlRT)
            elseif rettype <: MixedDuplicated
                push!(sret_types, Base.RefValue{jlRT})
            elseif rettype <: BatchDuplicated || rettype <: BatchDuplicatedNoNeed
                push!(sret_types, AnonymousStruct(NTuple{width,jlRT}))
            elseif rettype <: BatchMixedDuplicated
                push!(sret_types, AnonymousStruct(NTuple{width,Base.RefValue{jlRT}}))
            elseif CC <: AugmentedForwardThunk
                push!(sret_types, Nothing)
            elseif rettype <: Const
            else
                @show rettype, CC
                @assert false
            end
        end

        if returnPrimal && (CC <: ForwardModeThunk)
            push!(sret_types, jlRT)
        end

        # calls fptr
        llvmtys = LLVMType[convert(LLVMType, x; allow_boxed = true) for x in types]

        T_void = convert(LLVMType, Nothing)

        combinedReturn =
            (CC <: PrimalErrorThunk && eltype(rettype) == Union{}) ? Union{} :
            Tuple{sret_types...}
        if any(
            any_jltypes(convert(LLVM.LLVMType, T; allow_boxed = true)) for T in sret_types
        )
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

        builder = LLVM.IRBuilder()
        entry = BasicBlock(llvm_f, "entry")
        position!(builder, entry)
        callparams = collect(LLVM.Value, parameters(llvm_f))

        if !(GPUCompiler.isghosttype(PT) || Core.Compiler.isconstType(PT))
            lfn = callparams[1]
            deleteat!(callparams, 1)
        end

        if returnRoots
            tracked = CountTrackedPointers(jltype)
            pushfirst!(
                callparams,
                alloca!(builder, LLVM.ArrayType(T_prjlvalue, tracked.count)),
            )
            pushfirst!(callparams, alloca!(builder, jltype))
        end

        if needs_tape && !(isghostty(TapeType) || Core.Compiler.isconstType(TapeType))
            tape = callparams[end]
            if TapeType <: EnzymeTapeToLoad
                llty = from_tape_type(eltype(TapeType))
                tape = bitcast!(
                    builder,
                    tape,
                    LLVM.PointerType(llty, LLVM.addrspace(value_type(tape))),
                )
                tape = load!(builder, llty, tape)
                API.SetMustCache!(tape)
                callparams[end] = tape
            else
                llty = from_tape_type(TapeType)
                @assert value_type(tape) == llty
            end
        end

        if !(GPUCompiler.isghosttype(PT) || Core.Compiler.isconstType(PT))
            FT = LLVM.FunctionType(
                returnRoots ? T_void : T_ret,
                [value_type(x) for x in callparams],
            )
            lfn = inttoptr!(builder, lfn, LLVM.PointerType(FT))
        else
            val_inner(::Type{Val{V}}) where {V} = V
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
        reinsert_gcmarker!(llvm_f)

        ir = string(mod)
        fn = LLVM.name(llvm_f)

        @assert length(types) == length(ccexprs)

        if !(GPUCompiler.isghosttype(PT) || Core.Compiler.isconstType(PT))
            return quote
                Base.@_inline_meta
                Base.llvmcall(
                    ($ir, $fn),
                    $combinedReturn,
                    Tuple{$PT,$(types...)},
                    fptr,
                    $(ccexprs...),
                )
            end
        else
            return quote
                Base.@_inline_meta
                Base.llvmcall(
                    ($ir, $fn),
                    $combinedReturn,
                    Tuple{$(types...)},
                    $(ccexprs...),
                )
            end
        end
    end
end

##
# JIT
##

function _link(job, (mod, adjoint_name, primal_name, TapeType))
    if job.config.params.ABI <: InlineABI
        return CompileResult(
            Val((Symbol(mod), Symbol(adjoint_name))),
            Val((Symbol(mod), Symbol(primal_name))),
            TapeType,
        )
    end

    # Now invoke the JIT
    jitted_mod = JIT.add!(mod)
    adjoint_addr = JIT.lookup(jitted_mod, adjoint_name)

    adjoint_ptr = pointer(adjoint_addr)
    if adjoint_ptr === C_NULL
        throw(
            GPUCompiler.InternalCompilerError(
                job,
                "Failed to compile Enzyme thunk, adjoint not found",
            ),
        )
    end
    if primal_name === nothing
        primal_ptr = C_NULL
    else
        primal_addr = JIT.lookup(jitted_mod, primal_name)
        primal_ptr = pointer(primal_addr)
        if primal_ptr === C_NULL
            throw(
                GPUCompiler.InternalCompilerError(
                    job,
                    "Failed to compile Enzyme thunk, primal not found",
                ),
            )
        end
    end

    return CompileResult(adjoint_ptr, primal_ptr, TapeType)
end

const DumpPostOpt = Ref(false)

# actual compilation
function _thunk(job, postopt::Bool = true)
    mod, meta = codegen(:llvm, job; optimize = false)
    adjointf, augmented_primalf = meta.adjointf, meta.augmented_primalf

    adjoint_name = name(adjointf)

    if augmented_primalf !== nothing
        primal_name = name(augmented_primalf)
    else
        primal_name = nothing
    end

    LLVM.ModulePassManager() do pm
        add!(pm, FunctionPass("ReinsertGCMarker", reinsert_gcmarker_pass!))
        LLVM.run!(pm, mod)
    end

    # Run post optimization pipeline
    if postopt
        if job.config.params.ABI <: FFIABI || job.config.params.ABI <: NonGenABI
            post_optimze!(mod, JIT.get_tm())
            if DumpPostOpt[]
                API.EnzymeDumpModuleRef(mod.ref)
            end
        else
            propagate_returned!(mod)
        end
    end
    return (mod, adjoint_name, primal_name, meta.TapeType)
end

const cache = Dict{UInt,CompileResult}()

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
@inline remove_innerty(::Type{<:MixedDuplicated}) = MixedDuplicated
@inline remove_innerty(::Type{<:BatchMixedDuplicated}) = MixedDuplicated

@inline function thunkbase(
    ctx,
    mi::Core.MethodInstance,
    ::Val{World},
    ::Type{FA},
    ::Type{A},
    tt::Type{TT},
    ::Val{Mode},
    ::Val{width},
    ::Val{ModifiedBetween},
    ::Val{ReturnPrimal},
    ::Val{ShadowInit},
    ::Type{ABI},
    ::Val{ErrIfFuncWritten},
    ::Val{RuntimeActivity},
) where {
    FA<:Annotation,
    A<:Annotation,
    TT,
    Mode,
    ModifiedBetween,
    width,
    ReturnPrimal,
    ShadowInit,
    World,
    ABI,
    ErrIfFuncWritten,
    RuntimeActivity,
}
    target = Compiler.EnzymeTarget()
    params = Compiler.EnzymeCompilerParams(
        Tuple{FA,TT.parameters...},
        Mode,
        width,
        remove_innerty(A),
        true,
        true,
        ModifiedBetween,
        ReturnPrimal,
        ShadowInit,
        UnknownTapeType,
        ABI,
        ErrIfFuncWritten,
        RuntimeActivity,
    ) #=abiwrap=#
    tmp_job = if World isa Nothing
        Compiler.CompilerJob(mi, CompilerConfig(target, params; kernel = false))
    else
        Compiler.CompilerJob(mi, CompilerConfig(target, params; kernel = false), World)
    end

    interp = GPUCompiler.get_interpreter(tmp_job)

    # TODO check compile return here, early
    # rrt = Core.Compiler.return_type(f, primal.tt) # nothing
    rrt = something(
        Core.Compiler.typeinf_type(interp, mi.def, mi.specTypes, mi.sparam_vals),
        Any,
    )
    rrt = Core.Compiler.typeinf_ext_toplevel(interp, mi).rettype

    run_enzyme = true

    A2 = if rrt == Union{}
        run_enzyme = false
        Const
    else
        A
    end

    if run_enzyme && !(A2 <: Const) && guaranteed_const_nongen(rrt, World)
        estr = "Return type `$rrt` not marked Const, but type is guaranteed to be constant"
        return error(estr)
    end

    rt2 = if !run_enzyme
        Const{rrt}
    elseif A2 isa UnionAll
        A2{rrt}
    else
        @assert A isa DataType
        # Can we relax this condition?
        # @assert eltype(A) == rrt
        A2
    end

    params = Compiler.EnzymeCompilerParams(
        Tuple{FA,TT.parameters...},
        Mode,
        width,
        rt2,
        run_enzyme,
        true,
        ModifiedBetween,
        ReturnPrimal,
        ShadowInit,
        UnknownTapeType,
        ABI,
        ErrIfFuncWritten,
        RuntimeActivity,
    ) #=abiwrap=#
    job = if World isa Nothing
        Compiler.CompilerJob(mi, CompilerConfig(target, params; kernel = false))
    else
        Compiler.CompilerJob(mi, CompilerConfig(target, params; kernel = false), World)
    end
    # We need to use primal as the key, to lookup the right method
    # but need to mixin the hash of the adjoint to avoid cache collisions
    # This is counter-intuitive since we would expect the cache to be split
    # by the primal, but we want the generated code to be invalidated by
    # invalidations of the primal, which is managed by GPUCompiler.


    compile_result = cached_compilation(job)
    if !run_enzyme
        ErrT = PrimalErrorThunk{typeof(compile_result.adjoint),FA,rt2,TT,width,ReturnPrimal}
        if Mode == API.DEM_ReverseModePrimal || Mode == API.DEM_ReverseModeGradient
            return (ErrT(compile_result.adjoint), ErrT(compile_result.adjoint))
        else
            return ErrT(compile_result.adjoint)
        end
    elseif Mode == API.DEM_ReverseModePrimal || Mode == API.DEM_ReverseModeGradient
        TapeType = compile_result.TapeType
        AugT = AugmentedForwardThunk{
            typeof(compile_result.primal),
            FA,
            rt2,
            Tuple{params.TT.parameters[2:end]...},
            width,
            ReturnPrimal,
            TapeType,
        }
        AdjT = AdjointThunk{
            typeof(compile_result.adjoint),
            FA,
            rt2,
            Tuple{params.TT.parameters[2:end]...},
            width,
            TapeType,
        }
        return (AugT(compile_result.primal), AdjT(compile_result.adjoint))
    elseif Mode == API.DEM_ReverseModeCombined
        CAdjT = CombinedAdjointThunk{
            typeof(compile_result.adjoint),
            FA,
            rt2,
            Tuple{params.TT.parameters[2:end]...},
            width,
            ReturnPrimal,
        }
        return CAdjT(compile_result.adjoint)
    elseif Mode == API.DEM_ForwardMode
        FMT = ForwardModeThunk{
            typeof(compile_result.adjoint),
            FA,
            rt2,
            Tuple{params.TT.parameters[2:end]...},
            width,
            ReturnPrimal,
        }
        return FMT(compile_result.adjoint)
    else
        @assert false
    end
end

@inline function thunk(
    mi::Core.MethodInstance,
    ::Type{FA},
    ::Type{A},
    tt::Type{TT},
    ::Val{Mode},
    ::Val{width},
    ::Val{ModifiedBetween},
    ::Val{ReturnPrimal},
    ::Val{ShadowInit},
    ::Type{ABI},
    ::Val{ErrIfFuncWritten},
    ::Val{RuntimeActivity},
) where {
    FA<:Annotation,
    A<:Annotation,
    TT,
    Mode,
    ModifiedBetween,
    width,
    ReturnPrimal,
    ShadowInit,
    ABI,
    ErrIfFuncWritten,
    RuntimeActivity,
}
    ts_ctx = JuliaContext()
    ctx = context(ts_ctx)
    activate(ctx)
    try
        return thunkbase(
            ctx,
            mi,
            Val(nothing),
            FA,
            A,
            TT,
            Val(Mode),
            Val(width),
            Val(ModifiedBetween),
            Val(ReturnPrimal),
            Val(ShadowInit),
            ABI,
            Val(ErrIfFuncWritten),
            Val(RuntimeActivity),
        ) #=World=#
    finally
        deactivate(ctx)
        dispose(ts_ctx)
    end
end

@inline @generated function thunk(
    ::Val{World},
    ::Type{FA},
    ::Type{A},
    tt::Type{TT},
    ::Val{Mode},
    ::Val{width},
    ::Val{ModifiedBetween},
    ::Val{ReturnPrimal},
    ::Val{ShadowInit},
    ::Type{ABI},
    ::Val{ErrIfFuncWritten},
    ::Val{RuntimeActivity},
) where {
    FA<:Annotation,
    A<:Annotation,
    TT,
    Mode,
    ModifiedBetween,
    width,
    ReturnPrimal,
    ShadowInit,
    World,
    ABI,
    ErrIfFuncWritten,
    RuntimeActivity,
}
    mi = fspec(eltype(FA), TT, World)
    ts_ctx = JuliaContext()
    ctx = context(ts_ctx)
    activate(ctx)
    res = try
        thunkbase(
            ctx,
            mi,
            Val(World),
            FA,
            A,
            TT,
            Val(Mode),
            Val(width),
            Val(ModifiedBetween),
            Val(ReturnPrimal),
            Val(ShadowInit),
            ABI,
            Val(ErrIfFuncWritten),
            Val(RuntimeActivity),
        )
    finally
        deactivate(ctx)
        dispose(ts_ctx)
    end
    return quote
        Base.@_inline_meta
        return $(res)
    end
end

import GPUCompiler: deferred_codegen_jobs

@generated function deferred_codegen(
    ::Val{World},
    ::Type{FA},
    ::Val{TT},
    ::Val{A},
    ::Val{Mode},
    ::Val{width},
    ::Val{ModifiedBetween},
    ::Val{ReturnPrimal},
    ::Val{ShadowInit},
    ::Type{ExpectedTapeType},
    ::Val{ErrIfFuncWritten},
    ::Val{RuntimeActivity},
) where {
    World,
    FA<:Annotation,
    TT,
    A,
    Mode,
    width,
    ModifiedBetween,
    ReturnPrimal,
    ShadowInit,
    ExpectedTapeType,
    ErrIfFuncWritten,
    RuntimeActivity,
}
    JuliaContext() do ctx
        Base.@_inline_meta
        mi = fspec(eltype(FA), TT, World)
        target = EnzymeTarget()

        rt2 = if A isa UnionAll
            params = EnzymeCompilerParams(
                Tuple{FA,TT.parameters...},
                Mode,
                width,
                remove_innerty(A),
                true,
                true,
                ModifiedBetween,
                ReturnPrimal,
                ShadowInit,
                ExpectedTapeType,
                FFIABI,
                ErrIfFuncWritten,
                RuntimeActivity,
            ) #=abiwrap=#
            tmp_job = Compiler.CompilerJob(
                mi,
                CompilerConfig(target, params; kernel = false),
                World,
            )

            interp = GPUCompiler.get_interpreter(tmp_job)

            rrt = something(
                Core.Compiler.typeinf_type(interp, mi.def, mi.specTypes, mi.sparam_vals),
                Any,
            )

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

        params = EnzymeCompilerParams(
            Tuple{FA,TT.parameters...},
            Mode,
            width,
            rt2,
            true,
            true,
            ModifiedBetween,
            ReturnPrimal,
            ShadowInit,
            ExpectedTapeType,
            FFIABI,
            ErrIfFuncWritten,
            RuntimeActivity,
        ) #=abiwrap=#
        job =
            Compiler.CompilerJob(mi, CompilerConfig(target, params; kernel = false), World)

        addr = get_trampoline(job)
        id = Base.reinterpret(Int, pointer(addr))
        deferred_codegen_jobs[id] = job

        quote
            Base.@_inline_meta
            ccall(
                "extern deferred_codegen",
                llvmcall,
                Ptr{Cvoid},
                (Ptr{Cvoid},),
                $(reinterpret(Ptr{Cvoid}, id)),
            )
        end
    end
end

include("compiler/reflection.jl")

@generated function onehot_internal(fn::F, x::T, startv::Int, lengthv::Int) where {F, T<:Array}
    ir = JuliaContext() do ctx
        Base.@_inline_meta

        target = Compiler.DefaultCompilerTarget()
        params = Compiler.PrimalCompilerParams(API.DEM_ForwardMode)
        mi = my_methodinstance(fn, Tuple{T, Int})
        job = CompilerJob(mi, CompilerConfig(target, params; kernel = false))
        mod, meta = GPUCompiler.codegen(
            :llvm,
            job;
            optimize = false,
            cleanup = false,
            validate = false,
        )
        copysetfn = meta.entry
        blk = first(blocks(copysetfn))
        for inst in collect(instructions(blk))
            if isa(inst, LLVM.FenceInst)
                eraseInst(blk, inst)
            end
            if isa(inst, LLVM.CallInst)
                fn = LLVM.called_operand(inst)
                if isa(fn, LLVM.Function)
                    if LLVM.name(fn) == "julia.safepoint"
                        eraseInst(blk, inst)
                    end
                end     
            end
        end
        hasNoRet = any(
            map(
                k -> kind(k) == kind(EnumAttribute("noreturn")),
                collect(function_attributes(copysetfn)),
            ),
        )
        @assert !hasNoRet
        if !hasNoRet
            push!(function_attributes(copysetfn), EnumAttribute("alwaysinline", 0))
        end
        ity = convert(LLVMType, Int)
        jlvaluet = convert(LLVMType, T; allow_boxed=true)

        FT = LLVM.FunctionType(jlvaluet,  LLVMType[jlvaluet, ity, ity])
        llvm_f = LLVM.Function(mod, "f", FT)
        push!(function_attributes(llvm_f), EnumAttribute("alwaysinline", 0))

        # Check if Julia version has https://github.com/JuliaLang/julia/pull/46914
        # and also https://github.com/JuliaLang/julia/pull/47076
        # and also https://github.com/JuliaLang/julia/pull/48620
        needs_dynamic_size_workaround = !(VERSION >= v"1.10.5")

        builder = LLVM.IRBuilder()
        entry = BasicBlock(llvm_f, "entry")
        position!(builder, entry)
        inp, lstart, len = collect(LLVM.Value, parameters(llvm_f))

        boxed_count = if sizeof(Int) == sizeof(Int64)
            emit_box_int64!(builder, len)
        else
            emit_box_int32!(builder, len)
        end

        tag = emit_apply_type!(builder, NTuple, (boxed_count, unsafe_to_llvm(builder, T)))

        fullsize = nuwmul!(builder, len, LLVM.ConstantInt(sizeof(Int)))
        obj = emit_allocobj!(builder, tag, fullsize, needs_dynamic_size_workaround)

        T_int8 = LLVM.Int8Type()
        LLVM.memset!(builder, obj,  LLVM.ConstantInt(T_int8, 0), fullsize, 0)

        alloc = pointercast!(builder, obj, LLVM.PointerType(jlvaluet, Tracked))
        alloc = pointercast!(builder, alloc, LLVM.PointerType(jlvaluet, 11))

        loop = BasicBlock(llvm_f, "loop")
        exit = BasicBlock(llvm_f, "exit")

        br!(builder, icmp!(builder, LLVM.API.LLVMIntEQ, LLVM.ConstantInt(0), len), exit, loop)

        position!(builder, loop)
        idx = phi!(builder, ity)

        push!(LLVM.incoming(idx), (LLVM.ConstantInt(0), entry))
        inc = add!(builder, idx, LLVM.ConstantInt(1))
        push!(LLVM.incoming(idx), (inc, loop))
        rval = add!(builder, inc, lstart)
        res = call!(builder, LLVM.function_type(copysetfn), copysetfn, [inp, rval])
        if !hasNoRet
            gidx = gep!(builder, jlvaluet, alloc, [idx])
            store!(builder, res, gidx)
            emit_writebarrier!(builder, get_julia_inner_types(builder, obj, res))
        end

        br!(builder, icmp!(builder, LLVM.API.LLVMIntEQ, inc, len), exit, loop)


        T_int32 = LLVM.Int32Type()

        reinsert_gcmarker!(llvm_f)

        position!(builder, exit)
        ret!(builder, obj)

        string(mod)
    end
    return quote
        Base.@_inline_meta
        Base.llvmcall(
            ($ir, "f"),
            Tuple{Vararg{T}},
            Tuple{T, Int, Int},
            x,
            startv,
            lengthv
        )
    end
end


end
