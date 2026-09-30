module Compiler

function funcwrapper_rewrite end

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
    EnzymeContext,
    ENZYME_CONTEXT,
    enzyme_context,
    enzyme_world,
    enzyme_world_if_active,
    TypeTree,
    typetree,
    typetree_in_world,
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
    ismutabletype,
    create_fresh_codeinfo,
    add_edge!

using Enzyme
using ScopedValues: @with

import EnzymeCore
import EnzymeCore: EnzymeRules, ABI, FFIABI, DefaultABI

using LLVM, LLVM.IR, LLVM.Build, LLVM.Passes, GPUCompiler, Libdl
import Enzyme_jll

import GPUCompiler: CompilerJob, compile, safe_name
using LLVM.Interop
import LLVM: Target, TargetMachine
import SparseArrays
using Printf

using Preferences

bitcode_replacement() = parse(Bool, @load_preference("bitcode_replacement", "true"))
bitcode_replacement!(val) = @set_preferences!("bitcode_replacement" => string(val))

function llvm_compatible_cpu_name(name::String)
    if Base.libllvm_version < v"19"
        if Sys.isapple() && Sys.ARCH === :aarch64
            m = match(r"^apple-m(\d+)$", name)
            if m !== nothing
                v = parse(Int, m.captures[1])
                if v >= 4
                    return "apple-m3"
                end
            end
        end
        m = match(r"^znver(\d+)$", name)
        if m !== nothing
            v = parse(Int, m.captures[1])
            if v >= 5
                return "znver4"
            end
        end
    end
    return name
end

function cpu_name()
    name = ccall(:jl_get_cpu_name, String, ())
    return llvm_compatible_cpu_name(name)
end

function cpu_features()
    return ccall(:jl_get_cpu_features, String, ())
end

# Define EnzymeTarget
# Base.@kwdef 
struct EnzymeTarget{Target<:AbstractCompilerTarget} <: AbstractCompilerTarget
    target::Target
end

GPUCompiler.llvm_triple(target::EnzymeTarget) = GPUCompiler.llvm_triple(target.target)
GPUCompiler.llvm_datalayout(target::EnzymeTarget) = GPUCompiler.llvm_datalayout(target.target)
GPUCompiler.llvm_machine(target::EnzymeTarget) = GPUCompiler.llvm_machine(target.target)
if isdefined(GPUCompiler, :llvm_targetinfo)
    GPUCompiler.llvm_targetinfo(target::EnzymeTarget) = GPUCompiler.llvm_targetinfo(target.target)
end
GPUCompiler.nest_target(::EnzymeTarget, other::AbstractCompilerTarget) = EnzymeTarget(other)
GPUCompiler.have_fma(target::EnzymeTarget, T::Type) = GPUCompiler.have_fma(target.target, T)
GPUCompiler.dwarf_version(target::EnzymeTarget) = GPUCompiler.dwarf_version(target.target)

module Runtime end

abstract type AbstractEnzymeCompilerParams <: AbstractCompilerParams end
struct EnzymeCompilerParams{Params<:AbstractCompilerParams} <: AbstractEnzymeCompilerParams
    params::Params

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

    # Whether to enforce that a zero derivative propagates as a zero (and never a nan)
    strongZero::Bool
end

# FIXME: Should this take something like PTXCompilerParams/CUDAParams?
struct PrimalCompilerParams <: AbstractEnzymeCompilerParams
    mode::API.CDerivativeMode
end

function EnzymeCompilerParams(TT, mode, width, rt, run_enzyme, abiwrap,
                              modifiedBetween, returnPrimal, shadowInit,
                              expectedTapeType, ABI,
                              err_if_func_written, runtimeActivity, strongZero)
    params = PrimalCompilerParams(mode)
    EnzymeCompilerParams(
        params,
        TT,
        mode,
        width,
        rt,
        run_enzyme,
        abiwrap,
        modifiedBetween,
        returnPrimal,
        shadowInit,
        expectedTapeType,
        ABI,
        err_if_func_written,
        runtimeActivity,
        strongZero
    )
end

DefaultCompilerTarget(; kwargs...) =
    GPUCompiler.NativeCompilerTarget(; jlruntime = true, kwargs...)

# TODO: Audit uses
function EnzymeTarget()
    EnzymeTarget(DefaultCompilerTarget())
end

# TODO: We shouldn't blanket opt-out
GPUCompiler.check_invocation(job::CompilerJob{EnzymeTarget}, entry::LLVM.Function) = nothing

GPUCompiler.runtime_module(::CompilerJob{<:Any,<:AbstractEnzymeCompilerParams}) = Runtime
# GPUCompiler.isintrinsic(::CompilerJob{EnzymeTarget}, fn::String) = true
# GPUCompiler.can_throw(::CompilerJob{EnzymeTarget}) = true

# GPUCompiler 2.x moved the code-instance cache onto CompilerCaching.jl and renamed the hooks:
# `ci_cache_token` became `cache_owner`, `ci_cache` (Julia 1.10) became `get_code_cache`,
# `runtime_slug` and `codegen` are gone, and `compile_method_instance` drives inference
# through `drive_inference!`, which it only provides for its own interpreter. Enzyme supports
# both majors; every version split below keys on this one probe.
const HAS_GPUCOMPILER_2 = isdefined(GPUCompiler, :cache_owner)

# GPUCompiler 1.x names the runtime library per back-end; 2.x derives it from the cache owner.
@static if !HAS_GPUCOMPILER_2
    # TODO: encode debug build or not in the compiler job
    #       https://github.com/JuliaGPU/CUDAnative.jl/issues/368
    GPUCompiler.runtime_slug(job::CompilerJob{EnzymeTarget}) = "enzyme"
end

# The method tables a method table view looks up methods in, in order of priority. Unlike
# the view, which is bound to a world, this identifies the lookup across worlds and
# sessions, as required for a cache token. Back-ends can stack method tables in their
# `GPUCompiler.method_table_view`, so the view has to be considered and not just
# `GPUCompiler.method_table`: jobs with the same main table but different stacks would
# otherwise share inference results.
method_tables(::Core.Compiler.InternalMethodTable) = ()
method_tables(view::Core.Compiler.OverlayMethodTable) = (view.mt,)
method_tables(view::GPUCompiler.StackedMethodTable) = (view.mt, method_tables(view.parent)...)

# GPUCompiler 2.11 and later give a stacked view to the rules lookup (`is_inactive`).
@inline function Enzyme.has_method(@nospecialize(sig::Type), world::UInt, view::GPUCompiler.StackedMethodTable)
    return Enzyme.has_method(sig, view.world, view.mt) || Enzyme.has_method(sig, view.world, view.parent)
end
@static if isdefined(Core.Compiler, :CachedMethodTable)
    method_tables(view::Core.Compiler.CachedMethodTable) = method_tables(view.table)
end
# other views are specific to a world, so only share inference results within that world
method_tables(@nospecialize(view::Core.Compiler.MethodTableView)) = (view,)

# provide a specific interpreter to use.
if VERSION >= v"1.11.0-DEV.1552"
    # The owner of the CodeInstances produced by an `EnzymeInterpreter`, compared with
    # `jl_egal`. It only carries the inputs that change what the interpreter infers: the
    # method tables it looks up methods in (see `method_tables`), and the set of rules
    # visible in the world for the mode in question.
    # The compiler target and params types deliberately are not part of it: one `autodiff`
    # call builds interpreters from several differently-typed jobs (the `EnzymeTarget`
    # thunk job, the unwrapped primal job, `primal_interp_world`) that all infer the same
    # way, and keying on those types made each of them re-infer the whole call graph.
    struct EnzymeCacheToken
        method_tables::Tuple
        last_fwd_rule_world::Union{Nothing, Tuple}
        last_rev_rule_world::Union{Nothing, Tuple}
        last_ina_rule_world::Union{Nothing, Tuple}
    end

    @inline EnzymeCacheToken(method_tables::Tuple, world::UInt, is_forward::Bool, is_reverse::Bool, inactive_rule::Bool) =
        EnzymeCacheToken(
        method_tables,
            is_forward ? (Enzyme.Compiler.Interpreter.get_rule_signatures(EnzymeRules.forward, Tuple{<:EnzymeCore.EnzymeRules.FwdConfig, <:Annotation, Type{<:Annotation}, Vararg{Annotation}}, world)...,) : nothing,
            is_reverse ? (Enzyme.Compiler.Interpreter.get_rule_signatures(EnzymeRules.augmented_primal, Tuple{<:EnzymeCore.EnzymeRules.RevConfig, <:Annotation, Type{<:Annotation}, Vararg{Annotation}}, world)...,) : nothing,
            inactive_rule ? (Enzyme.Compiler.Interpreter.get_rule_signatures(EnzymeRules.inactive, Tuple{Vararg{Any}}, world)...,) : nothing
        )

    # The owner of the code instances Enzyme infers for `job`. GPUCompiler asks for it through
    # a hook whose name differs between its majors.
    enzyme_cache_owner(job::CompilerJob{<:Any, <:AbstractEnzymeCompilerParams}) =
        EnzymeCacheToken(
        method_tables(GPUCompiler.method_table_view(job)),
            job.world,
            job.config.params.mode == API.DEM_ForwardMode,
            job.config.params.mode != API.DEM_ForwardMode,
            true
        )

    @static if HAS_GPUCOMPILER_2
        GPUCompiler.cache_owner(job::CompilerJob{<:Any, <:AbstractEnzymeCompilerParams}) =
            enzyme_cache_owner(job)
    else
        GPUCompiler.ci_cache_token(job::CompilerJob{<:Any, <:AbstractEnzymeCompilerParams}) =
            enzyme_cache_owner(job)
    end

    GPUCompiler.get_interpreter(job::CompilerJob{<:Any,<:AbstractEnzymeCompilerParams}) =
        Interpreter.EnzymeInterpreter(
        enzyme_cache_owner(job),
            GPUCompiler.method_table_view(job),
            job.world,
            job.config.params.mode,
            true
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

    @static if HAS_GPUCOMPILER_2
        GPUCompiler.get_code_cache(job::CompilerJob{<:Any, <:AbstractEnzymeCompilerParams}) =
            enzyme_ci_cache(job)
    else
        GPUCompiler.ci_cache(job::CompilerJob{<:Any, <:AbstractEnzymeCompilerParams}) =
            enzyme_ci_cache(job)
    end

    GPUCompiler.get_interpreter(job::CompilerJob{<:Any,<:AbstractEnzymeCompilerParams}) =
        Interpreter.EnzymeInterpreter(
            enzyme_ci_cache(job),
            GPUCompiler.method_table_view(job),
            job.world,
            job.config.params.mode,
            true
        )
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
    typeof(Base.FastMath.exp2_fast) => (:exp2, 1, nothing),
    typeof(Base.FastMath.exp10_fast) => (:exp10, 1, nothing),
    # the implementations the FastMath versions forward to manipulate the float's bits,
    # which Enzyme cannot differentiate; in GPU kernels only these may be left after inlining
    typeof(Base.Math.exp_fast) => (:exp, 1, nothing),
    typeof(Base.Math.exp2_fast) => (:exp2, 1, nothing),
    typeof(Base.Math.exp10_fast) => (:exp10, 1, nothing),
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

@inline function find_math_method(@nospecialize(func::Type), sparam_vals::Core.SimpleVector)
    if func ∈ keys(known_ops)
        name, arity, toinject = known_ops[func]
        Tys = (Float32, Float64)

        if length(sparam_vals) == arity
            T = first(sparam_vals)
            if (T isa Type)
                T = T::Type
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
    end

    if func ∈ keys(cmplx_known_ops)
        name, arity, toinject = cmplx_known_ops[func]
        Tys = (Complex{Float32}, Complex{Float64})
        if length(sparam_vals) == arity
            if name == :cmplx_in || name == :cmplx_jn || name == :cmplx_yn
                if (sparam_vals[2] ∈ Tys) && sparam_vals[2].parameters[1] == sparam_vals[1]
                    return name, toinject, sparam_vals[2]
                end
            end
            T = first(sparam_vals)
            if (T isa Type)
                T = T::Type
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
    end
    return nothing, nothing, nothing
end

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

include("llvm/attrkinds.jl")
include("llvm/attributes.jl")

include("typeutils/conversion.jl")
include("typeutils/jltypes.jl")
include("typeutils/lltypes.jl")

include("analyses/activity.jl")

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

@inline fn_type(::Type{<:CombinedAdjointThunk{<:Any,FA}}) where FA = FA
@inline fn_type(::Type{<:ForwardModeThunk{<:Any,FA}}) where FA = FA
@inline fn_type(::Type{<:AugmentedForwardThunk{<:Any,FA}}) where FA = FA
@inline fn_type(::Type{<:AdjointThunk{<:Any,FA}}) where FA = FA
@inline fn_type(::Type{<:PrimalErrorThunk{<:Any,FA}}) where FA = FA

using .JIT

include("jlrt.jl")
include("errors.jl")



AnyArray(Length::Int) = NamedTuple{ntuple(Symbol, Val(Length)),NTuple{Length,Any}}

const JuliaEnzymeNameMap = Dict{String,Any}(
    "enz_val_true" => Val(true),
    "enz_val_false" => Val(false),
    "enz_val_1" => Val(1),
    "enz_any_array_1" => AnyArray(1),
    "enz_any_array_2" => AnyArray(2),
    "enz_any_array_3" => AnyArray(3),
    "enz_runtime_exc" => EnzymeRuntimeException,
    "enz_runtime_mi_exc" => EnzymeRuntimeExceptionMI,
    "enz_mut_exc" => EnzymeMutabilityException,
    "enz_runtime_activity_exc" => EnzymeRuntimeActivityError{Cstring, Nothing, Nothing},
    "enz_runtime_activity_str_exc" => EnzymeRuntimeActivityError{String, Nothing, Nothing},
    "enz_runtime_activity_mi_exc" => EnzymeRuntimeActivityError{Cstring, Core.MethodInstance, UInt},
    "enz_no_type_exc" => EnzymeNoTypeError{Nothing, Nothing},
    "enz_no_type_mi_exc" => EnzymeNoTypeError{Core.MethodInstance, UInt},
    "enz_no_shadow_exc" => EnzymeNoShadowError,
    "enz_no_derivative_exc" => EnzymeNoDerivativeError{Nothing, Nothing},
    "enz_no_derivative_mi_exc" => EnzymeNoDerivativeError{Core.MethodInstance, UInt},
    "enz_non_const_kwarg_exc" => NonConstantKeywordArgException,
    "enz_callconv_mismatch_exc"=> CallingConventionMismatchError{Cstring},
    "enz_illegal_ta_exc" => IllegalTypeAnalysisException,
    "enz_illegal_first_pointer_exc" => IllegalFirstPointerException,
    "enz_internal_exc" => EnzymeInternalError,
    "enz_non_scalar_return_exc" => EnzymeNonScalarReturnException,
)

include("absint.jl")
include("llvm/transforms.jl")
include("llvm/passes.jl")
include("typeutils/make_zero.jl")

function nested_codegen!(mode::API.CDerivativeMode, mod::LLVM.Module, @nospecialize(f), @nospecialize(tt::Type))
    funcspec = my_methodinstance(mode == API.DEM_ForwardMode ? Forward : Reverse, typeof(f), tt, enzyme_world())
    return nested_codegen!(mode, mod, funcspec)
end


function prepare_llvm(interp, mod::LLVM.Module, job, meta, enzyme_ctx::EnzymeContext)
    for (mi, k) in meta.compiled
        k_name = GPUCompiler.safe_name(k.specfunc)
        if !haskey(mod.functions, k_name)
            continue
        end
        llvmfn = mod.functions[k_name]

        RT = return_type(interp, mi)

        _, _, returnRoots0 = get_return_info(RT)
        returnRoots = returnRoots0 !== nothing

        attributes = llvmfn.function_attributes
        # A function that already carries `enzymejl_mi` is not this emission's
        # output: a derivative embedded for nested differentiation may reuse
        # the name of a fresh function, since Julia's function name counters
        # restart per compilation.
        fresh = !haskey(llvmfn.function_attributes, "enzymejl_mi")
        push!(
            attributes,
            StringAttribute("enzymejl_mi", string(convert(UInt, pointer_from_objref(mi)))),
        )
        push!(
            attributes,
            StringAttribute("enzymejl_rt", string(convert(UInt, unsafe_to_pointer(RT)))),
        )
        if cached_has_easy_rule(Interpreter.simplify_kw(mi.specTypes), job.world)
            push!(attributes, LLVM.StringAttribute("enzyme_LocalReadOnlyOrThrow"))
            # The rule accesses no more than the body does, so Enzyme may keep
            # inferring the function's full parameter attributes from the body.
            push!(attributes, LLVM.StringAttribute("enzyme_custom_full_attributes"))
        end

        if startswith(llvmfn.name, "japi3") || startswith(llvmfn.name, "japi1") || startswith(llvmfn.name, "jlcapi")
	   continue
	end

        if fresh
            check_emitted_specsig(mod, llvmfn, mi, RT)
        end

        if is_sret_union(RT)
            attr = StringAttribute("enzymejl_sret_union_bytes", string(union_alloca_type(RT)))
            push!(llvmfn.parameter_attributes[1], attr)
            for u in llvmfn.uses
                u = u.user
                @assert isa(u, LLVM.CallInst)
                push!(u.argument_attributes[1], attr)
            end
        end

        if returnRoots
            attr = StringAttribute("enzymejl_returnRoots", string(length(eltype(returnRoots0).parameters[1])))
            push!(llvmfn.parameter_attributes[2], attr)
            for u in llvmfn.uses
                u = u.user
                @assert isa(u, LLVM.CallInst)
                push!(u.argument_attributes[2], attr)
            end
        end

        fixup_1p12_sret!(llvmfn)
    end

    # We explicitly save the type of alloca's before they get lowered
    for f in mod.functions
        for bb in f.blocks, inst in bb.instructions
            if !isa(inst, LLVM.CallInst)
                continue
            end
            fn = inst.called_operand
            if !isa(fn, LLVM.Function)
                continue
            end
            if fn.name == "julia.gc_alloc_obj"
                legal, RT, _ = abs_typeof(inst, enzyme_ctx)
                if legal
                    inst.metadata["enzymejl_gc_alloc_rt"] = MDNode(LLVM.Metadata[MDString(string(convert(UInt, unsafe_to_pointer(RT))))])
                end
            end
        end
    end

    return rewrite_abi_converter_calls!(mod)
end

const FRULE_CACHE = Dict{Tuple{Type, UInt, Any}, Bool}()
const RRULE_CACHE = Dict{Tuple{Type, UInt, Any}, Bool}()
const INACTIVE_CACHE = Dict{Tuple{Type, UInt, Any}, Bool}()
const EASY_RULE_CACHE = Dict{Tuple{Type, UInt}, Bool}()
const NOALIAS_CACHE = Dict{Tuple{Type, UInt, Any}, Bool}()

function cached_has_easy_rule(specTypes::Type, world::UInt)
    key = (specTypes, world)
    val = get(EASY_RULE_CACHE, key, nothing)
    if val !== nothing
        return val::Bool
    end
    func = specTypes.parameters[1]
    if func isa Core.Builtin
        EASY_RULE_CACHE[key] = false
        return false
    end
    res = EnzymeRules.has_easy_rule_from_sig(specTypes; world)
    EASY_RULE_CACHE[key] = res
    return res
end

function cached_has_frule(specTypes::Type, world::UInt, method_table)
    key = (specTypes, world, method_table)
    val = get(FRULE_CACHE, key, nothing)
    if val !== nothing
        return val::Bool
    end
    func = specTypes.parameters[1]
    if func isa Core.Builtin
        FRULE_CACHE[key] = false
        return false
    end
    res = EnzymeRules.has_frule_from_sig(specTypes; world, method_table)
    FRULE_CACHE[key] = res
    return res
end

function cached_has_rrule(specTypes::Type, world::UInt, method_table)
    key = (specTypes, world, method_table)
    val = get(RRULE_CACHE, key, nothing)
    if val !== nothing
        return val::Bool
    end
    func = specTypes.parameters[1]
    if func isa Core.Builtin
        RRULE_CACHE[key] = false
        return false
    end
    res = EnzymeRules.has_rrule_from_sig(specTypes; world, method_table)
    RRULE_CACHE[key] = res
    return res
end

function cached_is_inactive(specTypes::Type, world::UInt, method_table)
    key = (specTypes, world, method_table)
    val = get(INACTIVE_CACHE, key, nothing)
    if val !== nothing
        return val::Bool
    end
    func = specTypes.parameters[1]
    if func isa Core.Builtin
        INACTIVE_CACHE[key] = false
        return false
    end
    res = EnzymeRules.is_inactive_from_sig(specTypes; world, method_table)
    INACTIVE_CACHE[key] = res
    return res
end

function cached_noalias(specTypes::Type, world::UInt, method_table)
    key = (specTypes, world, method_table)
    val = get(NOALIAS_CACHE, key, nothing)
    if val !== nothing
        return val::Bool
    end
    func = specTypes.parameters[1]
    if func isa Core.Builtin
        NOALIAS_CACHE[key] = false
        return false
    end
    res = EnzymeRules.noalias_from_sig(specTypes; world, method_table)
    NOALIAS_CACHE[key] = res
    return res
end

include("compiler/optimize.jl")
include("compiler/interpreter.jl")
include("compiler/callconv.jl")
include("compiler/validation.jl")
include("typeutils/inference.jl")

import .Interpreter: isKWCallSignature

"""
    record_julia_values!(ctx, job, meta)

Make the Julia value each global slot of a freshly emitted module refers to known to `ctx`.

That is the authoritative answer to "which object is this global?", and unlike an address
decoded from an initializer it does not depend on the address having been written into the IR,
which no back-end has before Enzyme is done (see [`emit_unresolved_llvm`](@ref) and
[`make_slots_symbolic!`](@ref)). GPUCompiler 1.x reports the address of the object behind each
slot as `gv_to_value`, keyed by the name of the slot, which is copied into the tables of `ctx`.
GPUCompiler 2.x reports a relocation record per slot, keyed the same way, whose target is the
value: `ctx` keeps the records themselves, and whether the back-end of `job` lets host addresses
into the module (see [`bakes_julia_values`](@ref)). Either way `absint` and `abs_typeof` read
the value through [`slot_value`](@ref), [`resolve_slots!`](@ref) writes the address back in
from [`slot_address`](@ref), and the context keeps the values rooted for the duration of the
compilation.
"""
function record_julia_values!(ctx::EnzymeContext, @nospecialize(job::CompilerJob), meta)
    @static if HAS_GPUCOMPILER_2
        push!(ctx.relocations, meta.relocations)
        ctx.bakes = bakes_julia_values(job)
    else
        for (name, ptr) in meta.gv_to_value
            # A slot whose initializer GPUCompiler could not match to an object.
            ptr == C_NULL && continue
            record_julia_value!(ctx, name, Base.unsafe_pointer_to_objref(ptr), ptr)
        end
    end
    return nothing
end

# The relocation record of the slot called `name` in the modules the compilation emitted, if it
# has one and it refers to a Julia value (GPUCompiler 2.x).
function slot_record(ctx::EnzymeContext, name::String)
    @static if HAS_GPUCOMPILER_2
        for relocs in ctx.relocations
            rec = GPUCompiler.find_relocation(relocs, name)
            rec === nothing && continue
            rec.kind === GPUCompiler.SlotSite && rec.target isa GPUCompiler.JuliaValueRef && return rec
        end
    end
    return nothing
end

"""
    slot_value(ctx, name)::Union{Some{Any}, Nothing}

The Julia value the slot called `name` refers to, as the compilation `ctx` knows it: from its
tables (GPUCompiler 1.x's `gv_to_value`, or a module compiled earlier, see
[`merge_julia_value_table!`](@ref)), or from the relocation records of the modules it emitted
(GPUCompiler 2.x); `nothing` if neither knows the slot.
"""
function slot_value(ctx::EnzymeContext, name::String)::Union{Some{Any}, Nothing}
    haskey(ctx.julia_values, name) && return Some{Any}(ctx.julia_values[name])
    rec = slot_record(ctx, name)
    rec === nothing && return nothing
    return Some{Any}(rec.target.value)
end

"""
    slot_address(ctx, name)::Union{Ptr{Cvoid}, Nothing}

The address [`resolve_slots!`](@ref) writes into the slot called `name`, or `nothing` if the
compilation `ctx` has none: it does not know the slot, or the back-end keeps it symbolic
(GPUCompiler 2.x's `:patch` and `:table`), in which case a host address must not go into the
module. On GPUCompiler 2.x the address is that of the instance GPUCompiler roots, which gives an
`isbits` value a box whose address stays valid.
"""
function slot_address(ctx::EnzymeContext, name::String)::Union{Ptr{Cvoid}, Nothing}
    ptr = get(ctx.julia_slot_addrs, name, nothing)
    ptr === nothing || return ptr
    @static if HAS_GPUCOMPILER_2
        ctx.bakes || return nothing
        rec = slot_record(ctx, name)
        rec === nothing && return nothing
        return Ptr{Cvoid}(GPUCompiler.resolve_relocation_target(rec.target))
    else
        return nothing
    end
end

# The type a small type tag stands for: codegen refers to some types by their tag in Julia's
# table rather than by their address, and a box's header holds the tag of such a type.
function small_typeof(tag::UInt)::Union{Some{Any}, Nothing}
    tag < UInt(64 << 4) || return nothing   # jl_max_tags << 4
    table = cglobal(:jl_small_typeof, Ptr{Cvoid})
    ptr = unsafe_load(table, tag ÷ sizeof(Ptr{Cvoid}) + 1)
    ptr == C_NULL && return nothing
    return Some{Any}(Base.unsafe_pointer_to_objref(ptr))
end

# The bytes of the constant `c`, an array of `i8`.
function constant_bytes(c::LLVM.Constant)::Union{Vector{UInt8}, Nothing}
    T = c.value_type
    isa(T, LLVM.ArrayType) && T.element_type == LLVM.Int8Type() || return nothing
    isa(c, LLVM.ConstantAggregateZero) && return zeros(UInt8, T.length)
    isa(c, LLVM.ConstantDataArray) || return nothing
    bytes = Vector{UInt8}(undef, T.length)
    for i in 1:T.length
        b = LLVM.Value(LLVM.API.LLVMGetElementAsConstant(c, i - 1))
        isa(b, LLVM.ConstantInt) || return nothing
        bytes[i] = convert(UInt8, convert(UInt, b))
    end
    return bytes
end

"""
    materialized_box_value(ctx, gv)::Union{Some{Any}, Nothing}

The value of the box the slot `gv` points into, if GPUCompiler 2.x materialized one for it: in
device code it gives an `isbits` value a box in the module instead of the address of a host
one, `{[padding,] header, bytes}`, and points the initializer of the slot at its bytes. The
type is the header's small type tag, or, for a type that has none, the target of the relocation
record of the header in the modules `ctx` emitted. The value only lives in the module: analysis
can read it, but no host address stands for it, so a load of the slot is never folded.
"""
function materialized_box_value(ctx::EnzymeContext, gv::LLVM.GlobalVariable)::Union{Some{Any}, Nothing}
    init = gv.initializer
    init === nothing && return nothing
    while isa(init, LLVM.ConstantExpr) &&
            init.opcode in (LLVM.API.LLVMAddrSpaceCast, LLVM.API.LLVMBitCast)
        init = init.operands[1]
    end
    isa(init, LLVM.ConstantExpr) && init.opcode == LLVM.API.LLVMGetElementPtr || return nothing
    ops = init.operands
    length(ops) == 3 || return nothing
    box = ops[1]
    isa(box, LLVM.GlobalVariable) && isa(ops[3], LLVM.ConstantInt) || return nothing
    boxinit = box.initializer
    isa(boxinit, LLVM.ConstantStruct) || return nothing
    fields = boxinit.operands
    payload_idx = convert(Int, ops[3])
    payload_idx >= 1 && payload_idx < length(fields) || return nothing
    header = fields[payload_idx]
    isa(header, LLVM.ConstantInt) || return nothing
    tag = convert(UInt, header)
    T = if tag != 0
        small_typeof(tag)
    else
        found = nothing
        @static if HAS_GPUCOMPILER_2
            # The box has one record, for its header. It is matched by the name of the box
            # alone: the offset of a field is numbered from 0 or from 1 depending on the major
            # of LLVM.jl that GPUCompiler pairs with.
            boxname = box.name
            for relocs in ctx.relocations, rec in relocs.records
                rec.name == boxname && rec.kind === GPUCompiler.InteriorSite || continue
                rec.target isa GPUCompiler.JuliaValueRef || continue
                found = Some{Any}(rec.target.value)
                break
            end
        end
        found
    end
    T === nothing && return nothing
    T = something(T)
    isa(T, DataType) && isbitstype(T) || return nothing
    bytes = constant_bytes(fields[payload_idx + 1])
    bytes === nothing && return nothing
    length(bytes) == sizeof(T) || return nothing
    return Some{Any}(reinterpret(T, bytes)[1])
end

"""
    bakes_julia_values(job)

Whether the back-end of `job` lets the host addresses of Julia values into the code it
compiles. GPUCompiler 1.x always does; 2.x asks the back-end's `relocation_lowering`, which keeps
them symbolic on `:patch` and `:table`.

The strategy is asked of a `kernel = true` flavour of the job: a deferred derivative is
linked into the kernel that requested it, and a back-end whose strategy depends on
`kernel` (Metal answers `:table` for kernels only) would otherwise report `:bake` for the
non-kernel primal job Enzyme holds and let host addresses into a persisted kernel.
"""
function bakes_julia_values(@nospecialize(job::CompilerJob))::Bool
    @static if HAS_GPUCOMPILER_2
        kernel_job = CompilerJob(job; config = CompilerConfig(job.config; kernel = true))
        return GPUCompiler.relocation_lowering(kernel_job) === :bake
    else
        return true
    end
end

function record_julia_value!(ctx::EnzymeContext, name::String, @nospecialize(val), ptr::Ptr{Cvoid})
    ctx.julia_values[name] = val
    ctx.julia_slot_addrs[name] = ptr
    return nothing
end

"""
    JuliaValueTable

The Julia values one module refers to by name, the part of a compilation's tables that module
needs once the compilation is over:

- `slots`: the value, and its address, behind each symbolic `julia.constgv` slot, keyed by the
  name of the slot (see [`make_slots_symbolic!`](@ref));
- `inserted`: the value each `ejl_<key>` global Enzyme inserted stands for, keyed by `key`
  (see `insert_julia_value!`).

[`julia_value_table`](@ref) takes it out of the context. `_thunk` resolves the module it
compiled with it, and `autodiff_cache` keeps it next to the bitcode of the thunk, so that a
later compilation importing the bitcode ([`import_cached_autodiff!`](@ref)) can take the values
up into its own tables. Holding the values keeps them rooted for as long as the bitcode refers
to them by name.
"""
struct JuliaValueTable
    slots::Dict{String, Tuple{Any, Ptr{Cvoid}}}
    inserted::Dict{String, Any}
end
JuliaValueTable() = JuliaValueTable(Dict{String, Tuple{Any, Ptr{Cvoid}}}(), Dict{String, Any}())

"""
    julia_value_table(ctx, mod)::JuliaValueTable

The entries of the tables of `ctx` for what `mod` refers to by name: its slots that are still
symbolic, and the `ejl_` globals Enzyme inserted into it.
"""
function julia_value_table(ctx::EnzymeContext, mod::LLVM.Module)::JuliaValueTable
    table = JuliaValueTable()
    for gv in mod.globals
        name = gv.name
        if haskey(gv.metadata, "julia.constgv")
            LLVM.isdeclaration(gv) || continue
            ptr = slot_address(ctx, name)
            ptr === nothing && continue
            table.slots[name] = (something(slot_value(ctx, name)), ptr)
        elseif startswith(name, "ejl_")
            key = name[(ncodeunits("ejl_") + 1):end]
            haskey(ctx.inserted_values, key) || continue
            table.inserted[key] = ctx.inserted_values[key]
        end
    end
    return table
end

"""
    merge_julia_value_table!(ctx, table::JuliaValueTable)

Take the values of a module compiled earlier, which is being linked into this compilation's,
up into the tables of `ctx`.
"""
function merge_julia_value_table!(ctx::EnzymeContext, table::JuliaValueTable)
    for (name, (val, ptr)) in table.slots
        record_julia_value!(ctx, name, val, ptr)
    end
    Base.merge!(ctx.inserted_values, table.inserted)
    return nothing
end

"""
    make_slots_symbolic!(mod, ctx)

Drop the address GPUCompiler 1.x writes into each `julia.constgv` slot of `mod` whose value is
recorded in the table of `ctx`, leaving the slot a declaration: the shape GPUCompiler 2.x
emits for a job compiled on behalf of another, with the value known only by name. Enzyme works
on the module through the table of Julia values, and [`resolve_slots!`](@ref) writes the
addresses back in once the module is linked into what runs: late, as GPUCompiler 2.x resolves
its relocations.
"""
function make_slots_symbolic!(mod::LLVM.Module, ctx::EnzymeContext)
    for gv in mod.globals
        haskey(gv.metadata, "julia.constgv") || continue
        name = gv.name
        slot_value(ctx, name) === nothing && continue
        if gv.initializer !== nothing
            # Only an address the table has can be written back in.
            slot_address(ctx, name) === nothing && continue
            gv.linkage = LLVM.API.LLVMExternalLinkage
            gv.initializer = nothing
        end
        # GPUCompiler 2.x emits the slot as a declaration to begin with.
        mark_symbolic_slot!(gv)
    end
    return nothing
end

"""
    mark_symbolic_slot!(gv)

Tell Enzyme what the initializer of the `julia.constgv` slot `gv` said before it became a
declaration: the slot never changes (`constant`), and as it only ever holds the address of a
Julia value, it is inactive. Activity analysis inferred that from a constant initializer; a
declaration gives it nothing to go by, and an active-looking slot would need a shadow global.
Type analysis is kept from accumulating information on the slot, which every load of it shares
(as for the globals `unsafe_to_llvm` inserts); a load LLVM folded into the address
gave it none either.
"""
function mark_symbolic_slot!(gv::LLVM.GlobalVariable)
    gv.constant = true
    gv.metadata["enzyme_inactive"] = MDNode(LLVM.Metadata[])
    gv.metadata["enzyme_ta_norecur"] = MDNode(LLVM.Metadata[])
    return nothing
end

"""
    resolve_slots!(mod, table::JuliaValueTable)

Write the address of its value back into every `julia.constgv` slot of `mod` that
[`make_slots_symbolic!`](@ref) left a declaration, from the module's table (see
[`julia_value_table`](@ref)). This is the resolver for GPUCompiler 1.x, which resolves nothing itself:
it runs where the module is linked into code that runs, so that everything before sees the
slots as names.
"""
function resolve_slots!(mod::LLVM.Module, table::JuliaValueTable)
    slots = table.slots
    for gv in mod.globals
        haskey(gv.metadata, "julia.constgv") || continue
        LLVM.isdeclaration(gv) || continue
        entry = get(slots, gv.name, nothing)
        entry === nothing && continue
        addr = LLVM.ConstantInt(reinterpret(UInt, entry[2]))
        gv.initializer = LLVM.const_inttoptr(addr, global_value_type(gv))
        gv.linkage = LLVM.API.LLVMPrivateLinkage
        gv.constant = true
    end
    return nothing
end

"""
    ejl_value(key, inserted)::Union{Some{Any}, Nothing}

The Julia value the global `ejl_<key>` stands for: a well-known Julia global
(`JuliaGlobalNameMap`), one Enzyme knows when it loads (`JuliaEnzymeNameMap`), or one the
compilation inserted (`inserted`, see `insert_julia_value!`); `nothing` if none. A load folded
through a binding (Julia 1.10) stands for the binding's value.
"""
function ejl_value(key::AbstractString, inserted::Dict{String, Any})::Union{Some{Any}, Nothing}
    val = if haskey(JuliaGlobalNameMap, key)
        JuliaGlobalNameMap[key]
    elseif haskey(JuliaEnzymeNameMap, key)
        JuliaEnzymeNameMap[key]
    elseif haskey(inserted, key)
        inserted[key]
    else
        return nothing
    end
    return Some{Any}(unbind(val))
end

# Replace the global `g`, which stands for the Julia value `val`, with the address of the value.
# `unsafe_to_ptr` roots the value for good, as the code holds its address.
function bake_julia_value_global!(g::LLVM.GlobalVariable, @nospecialize(val))
    T_pjlvalue = LLVM.PointerType(LLVM.StructType(LLVM.LLVMType[]))
    addr = LLVM.ConstantInt(reinterpret(UInt, unsafe_to_ptr(val)))
    replace_uses!(g, LLVM.const_addrspacecast(LLVM.const_inttoptr(addr, T_pjlvalue), g.value_type))
    LLVM.erase!(g)
    return nothing
end

"""
    bake_julia_value_globals!(mod, inserted)

Replace each `ejl_<key>` global of the device module `mod`, which stands for the Julia value
`JuliaGlobalNameMap[key]`, `JuliaEnzymeNameMap[key]` or `inserted[key]` (the values the
compilation inserted, see `unsafe_to_llvm`), with the address of that value. On the host the JIT
resolves the well-known names, and the module the values were inserted into gets their addresses
when it is linked ([`bake_inserted_values!`](@ref)); GPUCompiler 1.x resolves nothing in device
code, so the address is written in when the derivative is handed to the kernel that requested
it.
"""
function bake_julia_value_globals!(mod::LLVM.Module, inserted::Dict{String, Any})
    for g in collect(mod.globals)
        name = g.name
        startswith(name, "ejl_") || continue
        found = ejl_value(name[(ncodeunits("ejl_") + 1):end], inserted)
        found === nothing && continue
        bake_julia_value_global!(g, something(found))
    end
    return nothing
end

"""
    bake_inserted_values!(mod, inserted)

Write the address of each Julia value the compilation inserted (`inserted`, see
`insert_julia_value!`) into `mod`, which refers to it as `ejl_<key>`, as the module is linked into
code that runs. The names stay local to the module and its table until then, so there is nothing
to keep of them in a global; the well-known names (`JuliaGlobalNameMap`, `JuliaEnzymeNameMap`)
the JIT resolves.
"""
function bake_inserted_values!(mod::LLVM.Module, inserted::Dict{String, Any})
    for (key, val) in inserted
        haskey(mod.globals, "ejl_" * key) || continue
        bake_julia_value_global!(mod.globals["ejl_" * key], unbind(val))
    end
    return nothing
end

# GPUCompiler 2.x's `compile_method_instance` drives inference through this hook, which
# GPUCompiler only defines for its own `GPUInterpreter`.
@static if HAS_GPUCOMPILER_2
    @static if VERSION >= v"1.11.0-DEV.1552"
        GPUCompiler.drive_inference!(interp::Interpreter.EnzymeInterpreter, mi::Core.MethodInstance) =
            GPUCompiler.CompilerCaching.typeinf!(interp, mi)
    else
        # Mirrors GPUCompiler's own `CodeCache`-based `drive_inference!` (jlgen.jl), which
        # is typed on `GPUInterpreter` and therefore cannot be reused for ours.
        function GPUCompiler.drive_inference!(interp::Interpreter.EnzymeInterpreter, mi::Core.MethodInstance)
            src = Core.Compiler.typeinf_ext_toplevel(interp, mi)
            @assert src !== nothing "Inference of $mi failed"

            # For const-return CIs the inference result wasn't recorded; set it from the
            # returned source so callers re-using the CI don't need to re-infer.
            wvc = Core.Compiler.code_cache(interp)
            if Core.Compiler.haskey(wvc, mi)
                ci = Core.Compiler.getindex(wvc, mi)
                if ci.inferred === nothing
                    @atomic ci.inferred = src
                end
            end
            return nothing
        end
    end
end

# Whether the constant `c` is, or is a constant expression over, the global `g`.
function refers_to(@nospecialize(c::LLVM.Value), g::LLVM.GlobalVariable)::Bool
    c == g && return true
    isa(c, LLVM.ConstantExpr) || return false
    return any(Base.Fix2(refers_to, g), c.operands)
end

# Rebuild the constant `c` as instructions emitted by `B`, with `val` in place of `g`.
function rebuild_without!(B::LLVM.IRBuilder, @nospecialize(c::LLVM.Value), g::LLVM.GlobalVariable, val::LLVM.Value)::LLVM.Value
    c == g && return val
    refers_to(c, g) || return c
    ops = LLVM.Value[rebuild_without!(B, op, g, val) for op in c.operands]
    op = c.opcode
    if op in (LLVM.API.LLVMBitCast, LLVM.API.LLVMAddrSpaceCast, LLVM.API.LLVMPtrToInt, LLVM.API.LLVMIntToPtr)
        return LLVM.Value(LLVM.API.LLVMBuildCast(B, op, ops[1], c.value_type, ""))
    elseif op == LLVM.API.LLVMGetElementPtr
        srcty = LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(c))
        if LLVM.API.LLVMIsInBounds(c) != 0
            return inbounds_gep!(B, srcty, ops[1], ops[2:end])
        end
        return gep!(B, srcty, ops[1], ops[2:end])
    elseif op == LLVM.API.LLVMICmp
        return icmp!(B, LLVM.API.LLVMGetICmpPredicate(c), ops[1], ops[2])
    end
    error("Enzyme internal error: cannot rebuild the constant expression $(string(c)) over $(g.name) as instructions")
end

"""
    relocate_julia_value_globals!(mod, relocs, inserted)

Hand the Julia values Enzyme refers to in the device module `mod` to GPUCompiler 2.x's
relocation machinery.

Enzyme refers to a Julia value through a global `ejl_<key>` whose address is the value
(`JuliaGlobalNameMap[key]`, `JuliaEnzymeNameMap[key]`, or `inserted[key]` for the values the
compilation inserted, see `unsafe_to_llvm`). On the host the JIT resolves the well-known ones and
the module gets the addresses of the inserted ones when it is linked (`bake_inserted_values!`);
nothing does in device code. Replace the global
with what codegen emits for a Julia value: a load of a word-sized slot, here
`ejl_slot_<key>`, with a relocation record for the value. The job that requested the
derivative then lowers it with its own strategy, as it does Julia's own slots: bakes the
address in, or leaves it for its loader.
"""
function relocate_julia_value_globals!(mod::LLVM.Module, relocs, inserted::Dict{String, Any})
    T_slot = LLVM.PointerType(LLVM.StructType(LLVM.LLVMType[]))
    for g in collect(mod.globals)
        name = g.name
        startswith(name, "ejl_") || continue
        key = name[(ncodeunits("ejl_") + 1):end]
        found = ejl_value(key, inserted)
        found === nothing && continue
        val = something(found)

        # The instructions using `g`, directly or through constant expressions.
        users = Set{LLVM.Instruction}()
        todo = LLVM.Value[g]
        while !isempty(todo)
            v = pop!(todo)
            for u in v.uses
                user = u.user
                if isa(user, LLVM.Instruction)
                    push!(users, user)
                elseif isa(user, LLVM.ConstantExpr)
                    push!(todo, user)
                else
                    error("Enzyme internal error: cannot relocate $name, used by $(string(user))")
                end
            end
        end
        isempty(users) && continue

        slot_name = "ejl_slot_" * GPUCompiler.safe_name(key)
        slot = if haskey(mod.globals, slot_name)
            mod.globals[slot_name]
        else
            GlobalVariable(mod, T_slot, slot_name)
        end
        GPUCompiler.add_relocation!(relocs, GPUCompiler.SlotSite, slot_name, 0, GPUCompiler.JuliaValueRef(val))

        @dispose B = IRBuilder() begin
            for inst in users
                for (i, op) in enumerate(inst.operands)
                    refers_to(op, g) || continue
                    # A phi needs the value at the end of the incoming block.
                    at = if isa(inst, LLVM.PHIInst)
                        inst.incoming[i][2].terminator
                    else
                        inst
                    end
                    position!(B, LLVM.before(at))
                    word = load!(B, T_slot, slot)
                    loaded = addrspacecast!(B, word, g.value_type)
                    inst.operands[i] = rebuild_without!(B, op, g, loaded)
                end
            end
        end
        # What still uses `g` are constant expressions nothing uses any more.
        replace_uses!(g, LLVM.UndefValue(g.value_type))
        LLVM.erase!(g)
    end
    return nothing
end

"""
    emit_unresolved_llvm(job)

`GPUCompiler.emit_llvm(job)`, without resolving the references to Julia values on GPUCompiler
2.x: what it returns for a job compiled on behalf of another, also for a toplevel one, so that
nothing is baked into the module before Enzyme is done with it. The relocation records say
what each slot holds; [`link_julia_values!`](@ref) resolves them once the module is linked.
GPUCompiler 1.x always resolves; [`make_slots_symbolic!`](@ref) undoes it.
"""
function emit_unresolved_llvm(@nospecialize(job::CompilerJob))
    @static if HAS_GPUCOMPILER_2
        return GPUCompiler.emit_llvm(job; resolve_relocations = false)
    else
        return GPUCompiler.emit_llvm(job)
    end
end

"""
    record_symbolic_slots!(mod, relocs, ctx)

Give every slot of `mod` that is still a declaration, and whose value the table of `ctx`
knows, a relocation record in `relocs`, unless it has one. The slots of the primal module come
with records; those of a module linked in later (`nested_codegen!`, an imported thunk) do not,
and whoever resolves the records must see them too.
"""
function record_symbolic_slots!(mod::LLVM.Module, relocs, ctx::EnzymeContext)
    @static if HAS_GPUCOMPILER_2
        named = Set{String}(rec.name for rec in relocs.records)
        for gv in mod.globals
            haskey(gv.metadata, "julia.constgv") || continue
            LLVM.isdeclaration(gv) || continue
            name = gv.name
            name in named && continue
            found = slot_value(ctx, name)
            found === nothing && continue
            GPUCompiler.add_relocation!(relocs, GPUCompiler.SlotSite, name, 0, GPUCompiler.JuliaValueRef(something(found)))
        end
    end
    return nothing
end

"""
    link_julia_values!(mod, meta)

Resolve the Julia values the toplevel module `mod` refers to, as it is linked into what runs:
on GPUCompiler 2.x by baking its relocation records (`meta.relocations`), which also cover
what is not a slot of a Julia value, and from the module's table (`meta.value_table`): its
slots ([`resolve_slots!`](@ref)) and the values the compilation inserted
([`bake_inserted_values!`](@ref)).
"""
function link_julia_values!(mod::LLVM.Module, meta)
    @static if HAS_GPUCOMPILER_2
        relocs = meta.relocations
        if !isempty(relocs)
            GPUCompiler.prune_dead_relocations!(mod, relocs)
            GPUCompiler.bake_relocations!(mod, relocs)
        end
    end
    resolve_slots!(mod, meta.value_table)
    bake_inserted_values!(mod, meta.value_table.inserted)
    return nothing
end


mutable struct HandlerState
    primalf::Union{Nothing, LLVM.Function}
    must_wrap::Bool
    actualRetType::Union{Nothing, Type}
    lowerConvention::Bool
    loweredArgs::Set{Int}
    boxedArgs::Set{Int}
    removedRoots::Set{Int}
    fnsToInject::Vector{Tuple{Symbol,Type}}
end


function handleCustom(state::HandlerState, custom, k_name::String, llvmfn::LLVM.Function, name::String, attrs::Vector{LLVM.Attribute} = LLVM.Attribute[], setlink::Bool = true, noinl::Bool = true)
    attributes = llvmfn.function_attributes
    custom[k_name] = llvmfn.linkage
    if setlink
        llvmfn.linkage = LLVM.Linkage.External
    end
    for a in attrs
        push!(attributes, a)
    end
    push!(attributes, StringAttribute("enzyme_math", name))
    if noinl
        push!(attributes, EnumAttribute(:noinline))
    end
    state.must_wrap |= llvmfn == state.primalf
    nothing
end

function handle_compiled(state::HandlerState, edges::Vector, run_enzyme::Bool, mode::API.CDerivativeMode, world::UInt, method_table, custom::Dict{String, LLVM.Linkage.T}, mod::LLVM.Module, mi::Core.MethodInstance, k_name::String, @nospecialize(rettype::Type), enzyme_ctx::EnzymeContext)::Nothing
    has_custom_rule = false

    specTypes = Interpreter.simplify_kw(mi.specTypes)

    if mode == API.DEM_ForwardMode
        has_custom_rule = cached_has_frule(specTypes, world, method_table)
        if has_custom_rule
            @safe_debug "Found frule for" mi.specTypes
        end
    else
        has_custom_rule = cached_has_rrule(specTypes, world, method_table)
        if has_custom_rule
            @safe_debug "Found rrule for" mi.specTypes
        end
    end

    if !haskey(mod.functions, k_name)
        return
    end

    llvmfn = mod.functions[k_name]
    if llvmfn == state.primalf
        state.actualRetType = rettype
    end

    if cached_noalias(specTypes, world, method_table)
        push!(edges, mi)
        push!(llvmfn.return_attributes, EnumAttribute(:noalias))
        for u in llvmfn.uses
            c = u.user
            if !isa(c, LLVM.CallInst)
                continue
            end
            cf = c.called_operand
            if cf == llvmfn
                push!(c.return_attributes, LLVM.EnumAttribute(:noalias))
            end
        end
    end

    func = mi.specTypes.parameters[1]

@static if VERSION < v"1.11-"
else
    if func == typeof(Core.memoryref)
            attributes = llvmfn.function_attributes
            push!(attributes, EnumAttribute(:alwaysinline))
    end
end

    meth = mi.def
    name = meth.name
    jlmod = meth.module

    julia_activity_rule(llvmfn, method_table)
    if has_custom_rule
        handleCustom(
            state,
            custom,
            k_name,
            llvmfn,
            "enzyme_custom",
            LLVM.Attribute[StringAttribute(PRESERVEPRIMAL_ATTR_KIND, "*")],
        )
        return
    end


    sparam_vals = mi.specTypes.parameters[2:end] # mi.sparam_vals
    if func == typeof(Base.eps) ||
       func == typeof(Base.nextfloat) ||
       func == typeof(Base.prevfloat)
        if LLVM.version().major <= 15
            handleCustom(
                state,
                custom,
                k_name,
                llvmfn,
                "jl_inactive_inout",
                LLVM.Attribute[
                    StringAttribute("enzyme_inactive"),
                    EnumAttribute(:readnone),
                    EnumAttribute(:speculatable),
                    EnumAttribute(:willreturn),
                    EnumAttribute(:nosync),
                    EnumAttribute(:nofree),
                    EnumAttribute(:nounwind),
                    StringAttribute("enzyme_shouldrecompute"),
                ],
            )
        else
            handleCustom(
                state,
                custom,
                k_name,
                llvmfn,
                "jl_inactive_inout",
                LLVM.Attribute[
                    StringAttribute("enzyme_inactive"),
                    EnumAttribute(:memory, NoEffects.data),
                    EnumAttribute(:speculatable),
                    EnumAttribute(:willreturn),
                    EnumAttribute(:nosync),
                    EnumAttribute(:nofree),
                    EnumAttribute(:nounwind),
                    StringAttribute("enzyme_shouldrecompute"),
                ],
            )
        end
        return
    end
    if func == typeof(Base.to_tuple_type)
        if LLVM.version().major <= 15
            handleCustom(
                state,
                custom,
                k_name,
                llvmfn,
                "jl_to_tuple_type",
                LLVM.Attribute[
                    EnumAttribute(:readonly),
                    EnumAttribute(:inaccessiblememonly),
                    EnumAttribute(:speculatable),
                    EnumAttribute(:willreturn),
                    EnumAttribute(:nosync),
                    EnumAttribute(:nofree),
                    StringAttribute("enzyme_shouldrecompute"),
                    StringAttribute("enzyme_inactive"),
                ],
            )
        else
            handleCustom(
                state,
                custom,
                k_name,
                llvmfn,
                "jl_to_tuple_type",
                LLVM.Attribute[
                    EnumAttribute(
                        "memory",
                        MemoryEffect(
                            (MRI_NoModRef << getLocationPos(ArgMem)) |
                            (MRI_Ref << getLocationPos(InaccessibleMem)) |
                            (MRI_NoModRef << getLocationPos(Other)),
                        ).data,
                    ),
                    EnumAttribute(:willreturn),
                    EnumAttribute(:nosync),
                    EnumAttribute(:nofree),
                    EnumAttribute(:speculatable),
                    StringAttribute("enzyme_shouldrecompute"),
                    StringAttribute("enzyme_inactive"),
                ],
            )
        end
        return
    end
    if func == typeof(Base.mightalias)
        if LLVM.version().major <= 15
            handleCustom(
                state,
                custom,
                k_name,
                llvmfn,
                "jl_mightalias",
                LLVM.Attribute[
                    EnumAttribute(:readonly),
                    StringAttribute("enzyme_shouldrecompute"),
                    StringAttribute("enzyme_inactive"),
                    StringAttribute("enzyme_no_escaping_allocation"),
                    EnumAttribute(:willreturn),
                    EnumAttribute(:nosync),
                    EnumAttribute(:nofree),
                    StringAttribute("enzyme_ta_norecur"),
                ],
                true,
                false,
            )
        else
            handleCustom(
                state,
                custom,
                k_name,
                llvmfn,
                "jl_mightalias",
                LLVM.Attribute[
                    EnumAttribute(:memory, ReadOnlyEffects.data),
                    StringAttribute("enzyme_shouldrecompute"),
                    StringAttribute("enzyme_inactive"),
                    StringAttribute("enzyme_no_escaping_allocation"),
                    EnumAttribute(:willreturn),
                    EnumAttribute(:nosync),
                    EnumAttribute(:nofree),
                    StringAttribute("enzyme_ta_norecur"),
                ],
                true,
                false,
            )
        end
        return
    end
    if func == typeof(Base.Threads.threadid) || func == typeof(Base.Threads.nthreads)
        name = (func == typeof(Base.Threads.threadid)) ? "jl_threadid" : "jl_nthreads"
        if LLVM.version().major <= 15
            handleCustom(
                state,
                custom,
                k_name,
                llvmfn,
                name,
                LLVM.Attribute[
                    EnumAttribute(:readonly),
                    EnumAttribute(:inaccessiblememonly),
                    EnumAttribute(:speculatable),
                    EnumAttribute(:willreturn),
                    EnumAttribute(:nosync),
                    EnumAttribute(:nofree),
                    EnumAttribute(:nounwind),
                    StringAttribute("enzyme_shouldrecompute"),
                    StringAttribute("enzyme_inactive"),
                    StringAttribute("enzyme_no_escaping_allocation"),
                ],
            )
        else
            handleCustom(
                state,
                custom,
                k_name,
                llvmfn,
                name,
                LLVM.Attribute[
                    EnumAttribute(
                        "memory",
                        MemoryEffect(
                            (MRI_NoModRef << getLocationPos(ArgMem)) |
                            (MRI_Ref << getLocationPos(InaccessibleMem)) |
                            (MRI_NoModRef << getLocationPos(Other)),
                        ).data,
                    ),
                    EnumAttribute(:speculatable),
                    EnumAttribute(:willreturn),
                    EnumAttribute(:nosync),
                    EnumAttribute(:nofree),
                    EnumAttribute(:nounwind),
                    StringAttribute("enzyme_shouldrecompute"),
                    StringAttribute("enzyme_inactive"),
                    StringAttribute("enzyme_no_escaping_allocation"),
                ],
            )
        end
        return
    end
    # Since this is noreturn and it can't write to any operations in the function
    # in a way accessible by the function. Ideally the attributor should actually
    # handle this and similar not impacting the read/write behavior of the calling
    # fn, but it doesn't presently so for now we will ensure this by hand
    if func == typeof(Base.Checked.throw_overflowerr_binaryop)
        if LLVM.version().major <= 15
            handleCustom(
                state,
                custom,
                k_name,
                llvmfn,
                "enz_noop",
                LLVM.Attribute[
                    StringAttribute("enzyme_inactive"),
                    EnumAttribute(:readonly),
                    StringAttribute("enzyme_ta_norecur"),
                ],
            )
        else
            handleCustom(
                state,
                custom,
                k_name,
                llvmfn,
                "enz_noop",
                LLVM.Attribute[
                    StringAttribute("enzyme_inactive"),
                    EnumAttribute(:memory, ReadOnlyEffects.data),
                    StringAttribute("enzyme_ta_norecur"),
                ],
            )
        end
        return
    end
    if EnzymeRules.is_inactive_from_sig(specTypes; world, method_table)
        push!(edges, mi)
        handleCustom(
            state,
            custom,
            k_name,
            llvmfn,
            "enz_noop",
            LLVM.Attribute[
                StringAttribute("enzyme_inactive"),
                EnumAttribute(:nofree),
                StringAttribute("enzyme_no_escaping_allocation"),
                StringAttribute("enzyme_ta_norecur"),
            ],
        )
        return
    end
    if EnzymeRules.is_inactive_noinl_from_sig(specTypes; world, method_table)
        push!(edges, mi)
        handleCustom(
            state,
            custom,
            k_name,
            llvmfn,
            "enz_noop",
            LLVM.Attribute[
                StringAttribute("enzyme_inactive"),
                EnumAttribute(:nofree),
                StringAttribute("enzyme_no_escaping_allocation"),
                StringAttribute("enzyme_ta_norecur"),
            ],
            false,
            false,
        )
        for bb in llvmfn.blocks
            for inst in bb.instructions
                if isa(inst, LLVM.CallInst)
                    push!(inst.function_attributes, StringAttribute("no_escaping_allocation"))
                    push!(inst.function_attributes, StringAttribute("enzyme_inactive"))
                    push!(inst.function_attributes, EnumAttribute(:nofree))
                end
            end
        end
        return
    end
    if func === typeof(Base.match)
        handleCustom(
            state,
            custom,
            k_name,
            llvmfn,
            "base_match",
            LLVM.Attribute[
                StringAttribute("enzyme_inactive"),
                EnumAttribute(:nofree),
                StringAttribute("enzyme_no_escaping_allocation"),
            ],
            false,
            false,
        )
        for bb in llvmfn.blocks
            for inst in bb.instructions
                if isa(inst, LLVM.CallInst)
                    push!(inst.function_attributes, StringAttribute("no_escaping_allocation"))
                    push!(inst.function_attributes, StringAttribute("enzyme_inactive"))
                    push!(inst.function_attributes, EnumAttribute(:nofree))
                end
            end
        end
        return
    end
    if func == typeof(Base.enq_work) &&
       length(sparam_vals) == 1 &&
       first(sparam_vals) <: Task
        handleCustom(state, custom, k_name, llvmfn, "jl_enq_work", LLVM.Attribute[StringAttribute("enzyme_ta_norecur")])
        return
    end
    if func == typeof(Base.wait) || func == typeof(Base._wait)
        if length(sparam_vals) == 1 && first(sparam_vals) <: Task
            handleCustom(state, custom, k_name, llvmfn, "jl_wait", LLVM.Attribute[StringAttribute("enzyme_ta_norecur")])
        end
        return
    end
    if func == typeof(Base.Threads.threading_run)
        if length(sparam_vals) == 1 || length(sparam_vals) == 2
            handleCustom(state, custom, k_name, llvmfn, "jl_threadsfor")
        end
        return
    end

    name, toinject, T = find_math_method(func, sparam_vals)
    if name === nothing
        return
    end

    if toinject !== nothing
        push!(state.fnsToInject, toinject)
    end

    # If sret, force lower of primitive math fn
    sret = get_return_info(rettype)[2] !== nothing
    if sret
        cur = llvmfn == state.primalf
        llvmfn, _, state.boxedArgs, state.loweredArgs, state.removedRoots = lower_convention(
            mi.specTypes,
            mod,
            llvmfn,
            rettype,
            Duplicated,
            nothing,
            run_enzyme,
            world,
            mi,
            enzyme_ctx,
        )
        if cur
            state.primalf = llvmfn
            state.lowerConvention = false
        end
        k_name = llvmfn.name
        if !haskey(llvmfn.function_attributes, :nofree)
            push!(llvmfn.function_attributes, EnumAttribute(:nofree))
        end
    end

    name = string(name)
    name = T == Float32 ? name * "f" : name

    attrs = if LLVM.version().major <= 15
        LLVM.Attribute[
            LLVM.EnumAttribute(:readnone), StringAttribute("enzyme_shouldrecompute"),
            EnumAttribute(:willreturn),
            EnumAttribute(:nosync),
            EnumAttribute(:nofree),
           	    StringAttribute("enzyme_preserve_primal", "*"),
		      ]
    else
        LLVM.Attribute[
            EnumAttribute(:memory, NoEffects.data), StringAttribute("enzyme_shouldrecompute"),
            EnumAttribute(:willreturn),
            EnumAttribute(:nosync),
            EnumAttribute(:nofree),
           	    StringAttribute("enzyme_preserve_primal", "*"),
		    ]
    end
    handleCustom(state, custom, k_name, llvmfn, name, attrs)
    return
end

function set_module_types!(interp, mod::LLVM.Module, primalf::Union{Nothing, LLVM.Function}, job, edges, run_enzyme, mode::API.CDerivativeMode, enzyme_ctx::EnzymeContext)::Tuple{Dict{String, LLVM.Linkage.T}, HandlerState}
    # One memo table for the whole module: the argument and return types of
    # its functions overlap heavily (the same model / array types recur in
    # every kernel), and building a TypeTree walks the type's fields through
    # the C API each time.
    seen = TypeTreeTable()

    for f in mod.functions
        if startswith(f.name, "japi3") || startswith(f.name, "japi1") || startswith(f.name, "jlcapi")
            continue
        end
        mi, RT = enzyme_custom_extract_mi(f, false)
        if mi === nothing
            continue
        end

        if haskey(f.function_attributes, LOWERED_CONVENTION_ATTR_KIND)
            continue
        end

        llRT, sret, returnRoots = get_return_info(RT)
        retRemoved, parmsRemoved = removed_ret_parms(f)

        dl = string(f.parent.datalayout)

        ftype = f.function_type

        swiftself = has_swiftself(f)

        # Unsupported calling conv
        # also wouldn't have any type info for this [would for earlier args though]
        if Base.isvarargtype(mi.specTypes.parameters[end])
            continue
        end

        world = enzyme_world()

        jlargs = classify_arguments(
            mi.specTypes,
            ftype,
            sret !== nothing,
            returnRoots !== nothing,
            swiftself,
            parmsRemoved,
            mi,
            world,
        )

        ctx = f.context

        push!(f.function_attributes, StringAttribute("enzyme_ta_norecur"))

        if !no_type_setting(mi.specTypes; world)[1]
            for arg in jlargs
                if arg.cc == GPUCompiler.GHOST || arg.cc == RemovedParam
                    continue
                end
                push!(
                    f.parameter_attributes[arg.codegen.i],
                    StringAttribute(
                        "enzymejl_parmtype",
                        string(convert(UInt, unsafe_to_pointer(arg.typ))),
                    ),
                )
                if EmitTypeNames[]
                    push!(
                        f.parameter_attributes[arg.codegen.i],
                        StringAttribute(
                            "enzymejl_parmtype_str",
                            string(arg.typ),
                        ),
                    )
                end
                push!(
                    f.parameter_attributes[arg.codegen.i],
                    StringAttribute("enzymejl_parmtype_ref", string(UInt(arg.cc))),
                )
		if arg.rooted_typ !== nothing
			push!(
                        f.parameter_attributes[arg.codegen.i],
			    StringAttribute("enzymejl_rooted_typ", string(convert(UInt, unsafe_to_pointer(arg.rooted_typ))))
			)
		end

                byref = arg.cc

                rest = copy(typetree_in_world(world, arg.typ, ctx, dl, seen))

                if byref == GPUCompiler.BITS_REF || byref == GPUCompiler.MUT_REF
                    # adjust first path to size of type since if arg.typ is {[-1]:Int}, that doesn't mean the broader
                    # object passing this in by ref isnt a {[-1]:Pointer, [-1,-1]:Int}
                    # aka the next field after this in the bigger object isn't guaranteed to also be the same.
                    if allocatedinline(arg.typ)
                        # An aggregate with inline roots is passed as a pointer to its
                        # data half only, whose buffer since Julia 1.13.1 omits the
                        # trailing tracked slots (`split_value_size`). Bound the type
                        # tree by that buffer, not by the full layout.
                        sz = if byref == GPUCompiler.BITS_REF && inline_roots_type(arg.typ) != 0
                            split_value_size(LLVM.DataLayout(dl), convert(LLVMType, arg.typ))
                        else
                            sizeof(arg.typ)
                        end
                        shift!(rest, dl, 0, sz, 0)
                    end
                    merge!(rest, TypeTree(API.DT_Pointer, ctx))
                    only!(rest, -1)
                else
                    # canonicalize wrt size
                end
                push!(
                    f.parameter_attributes[arg.codegen.i],
                    StringAttribute("enzyme_type", string(rest)),
                )
            end
        end

        if !no_type_setting(mi.specTypes; world)[2]
            if sret !== nothing
                idx = 0
                if !in(0, parmsRemoved)
                    @assert sret <: Ptr
                    sret_et = eltype(sret)
                    rest = copy(typetree_in_world(world, sret_et, ctx, dl, seen))
                    shift!(rest, dl, 0, LLVM.storage_size(LLVM.DataLayout(dl), sret_ty(f, 1)), 0)
                    merge!(rest, TypeTree(API.DT_Pointer, ctx))
                    only!(rest, -1)
                    push!(
                        f.parameter_attributes[idx + 1],
                        StringAttribute("enzyme_type", string(rest)),
                    )
                    idx += 1
                end
                if returnRoots !== nothing
                    if !in(1, parmsRemoved)
                        rest = TypeTree(API.DT_Pointer, -1, ctx)
                        push!(
                            f.parameter_attributes[idx + 1],
                            StringAttribute("enzyme_type", string(rest)),
                        )
                    end
                end
            end

            if llRT !== nothing &&
                    f.function_type.return_type != LLVM.VoidType()
                @assert !retRemoved
                rest = if llRT == Ptr{RT}
                    typeTree = copy(typetree_in_world(world, RT, ctx, dl, seen))
                    merge!(typeTree, TypeTree(API.DT_Pointer, ctx))
                    only!(typeTree, -1)
                    typeTree
                else
                    typetree_in_world(world, RT, ctx, dl)
                end
                push!(f.return_attributes, StringAttribute("enzyme_type", string(rest)))
            end
        end

    end

    custom = Dict{String, LLVM.Linkage.T}()

    world = job.world
    method_table = Core.Compiler.method_table(interp)

    state = HandlerState(
        primalf,
        #=mustwrap=#false,
        #=actualRetType=#nothing,
        #=lowerConvention=#true,
        #=loweredArgs=#Set{Int}(),
        #=boxedArgs=#Set{Int}(),
	#=removedRoots=#Set{Int}(),
        #=fnsToInject=#Tuple{Symbol,Type}[],
    )

    for fname in [fn.name for fn in mod.functions]
        if !haskey(mod.functions, fname)
            continue
        end
        fn = mod.functions[fname]
        if haskey(fn.function_attributes, LOWERED_CONVENTION_ATTR_KIND)
            continue
        end
        attributes = fn.function_attributes
        mi = nothing
        RT = nothing
        for fattr in collect(attributes)
            if isa(fattr, LLVM.StringAttribute)
                if fattr.kind == "enzymejl_mi"
                    ptr = reinterpret(Ptr{Cvoid}, parse(UInt, fattr.value))
                    mi = Base.unsafe_pointer_to_objref(ptr)
                end
            end
            if fattr.kind == "enzymejl_rt"
                ptr = reinterpret(Ptr{Cvoid}, parse(UInt, fattr.value))
                RT = Base.unsafe_pointer_to_objref(ptr)
            end
        end
        if mi !== nothing && RT !== nothing
            handle_compiled(state, edges, run_enzyme, mode, world, method_table, custom, mod, mi, fname, RT, enzyme_ctx)
        end
    end

    return custom, state
end

const DumpPreNestedCheck = Ref(false)
const DumpPreNestedOpt = Ref(false)
const DumpPostNestedOpt = Ref(false)

function nested_codegen!(
    mode::API.CDerivativeMode,
    mod::LLVM.Module,
    funcspec::Core.MethodInstance,
    alwaysinline::Bool=false,
)
    enzyme_ctx = enzyme_context()
    world = enzyme_ctx.world
    cache_key = funcspec
    if haskey(enzyme_ctx.nested_cache, cache_key)
        fname = enzyme_ctx.nested_cache[cache_key]
        if haskey(mod.functions, fname)
            return mod.functions[fname]
        end
        for m in enzyme_ctx.modules_to_link
            if haskey(m.functions, fname)
                return m.functions[fname]
            end
        end
        error("Cached function $fname not found in any module!")
    end

    # 3) Use the MI to create the correct augmented fwd/reverse
    # TODO:
    #  - GPU support
    #  - When OrcV2 only use a MaterializationUnit to avoid mutation of the module here

    target = DefaultCompilerTarget()
    params = PrimalCompilerParams(mode)
    job = CompilerJob(funcspec, CompilerConfig(target, params; kernel = false, libraries = true, toplevel = true, optimize = false, cleanup = false, only_entry = false, validate = false, entry_abi = :specfunc), world)

    GPUCompiler.prepare_job!(job)
    otherMod, meta = emit_unresolved_llvm(job)
    record_julia_values!(enzyme_ctx, job, meta)
    make_slots_symbolic!(otherMod, enzyme_ctx)
    
    interp = GPUCompiler.get_interpreter(job)
    prepare_llvm(interp, otherMod, job, meta, enzyme_ctx)

    entry = meta.entry.name

    for f in otherMod.functions
        permit_inlining!(f)
    end

    edges = enzyme_ctx.edges
    push!(edges, funcspec)

    LLVM.@dispose pb = LLVM.PassBuilder() begin
        registerEnzymeAndPassPipeline!(pb)
        LLVM.add!(pb, LLVM.ModulePassManager()) do mpm
            LLVM.add!(mpm, PreserveNVVMPass())
        end
        LLVM.run!(pb, mod)
    end
    
    if DumpPreNestedCheck[]
	API.EnzymeDumpModuleRef(otherMod.ref)
    end

    check_ir(interp, job, otherMod, enzyme_ctx)

    if DumpPreNestedOpt[]
	API.EnzymeDumpModuleRef(otherMod.ref)
    end

    # Skipped inline of blas

    run_enzyme = false
    set_module_types!(interp, otherMod, nothing, job, edges, run_enzyme, mode, enzyme_ctx)

    # Apply first stage of optimization's so that this module is at the same stage as `mod`
    optimize!(otherMod, JIT.get_tm(), enzyme_ctx)
    
    if DumpPostNestedOpt[]
	API.EnzymeDumpModuleRef(otherMod.ref)
    end
    
    # 4) Record module to link
    push!(enzyme_ctx.modules_to_link, otherMod)

    # Declare the function in mod so it can be called
    lfn = otherMod.functions[entry]
    if alwaysinline
        # A method declared `@noinline` carries that attribute. Remove it: this
        # path always inlines, and `noinline` conflicts with `alwaysinline`.
        delete!(lfn.function_attributes, :noinline)
        push!(lfn.function_attributes, EnumAttribute(:alwaysinline))
    end

    lfn.linkage = LLVM.Linkage.External
    FT = lfn.function_type
    decl = LLVM.Function(mod, entry, FT)

    # Copy function attributes
    for attr in collect(lfn.function_attributes)
        push!(decl.function_attributes, attr)
    end

    # Copy parameter attributes
    for idx in 1:length(lfn.parameters)
        for attr in collect(lfn.parameter_attributes[idx])
            push!(decl.parameter_attributes[idx], attr)
        end
    end

    # Copy return attributes
    for attr in collect(lfn.return_attributes)
        push!(decl.return_attributes, attr)
    end

    enzyme_ctx.nested_cache[cache_key] = decl.name
    return decl
end

function removed_ret_parms(orig::LLVM.CallInst)
    F = orig.called_operand
    if !isa(F, LLVM.Function)
        return false, UInt64[]
    end
    return removed_ret_parms(F)
end

function removed_ret_parms(F::LLVM.Function)
    parmsRemoved = UInt64[]
    parmrem = nothing
    retRemove = false
    for a in collect(F.function_attributes)
        if isa(a, StringAttribute)
            if a.kind == "enzyme_parmremove"
                parmrem = a
            end
            if a.kind == "enzyme_retremove"
                retRemove = true
            end
        end
    end
    if parmrem !== nothing
        str = parmrem.value
        for v in eachsplit(str, ",")
            push!(parmsRemoved, parse(UInt64, v))
        end
    end
    return retRemove, parmsRemoved
end

"""
    CheckNan::Ref{Bool}

If `Enzyme.Compiler.CheckNan[] == true`, Enzyme will error at the first encounter of a `NaN`
during differentiation. Useful as a debugging tool to help locate the call whose derivative
is the source of unexpected `NaN`s. Off by default.
"""
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
        curent_bb = B.insert_block
        fn = curent_bb.parent
        mod = fn.parent
        ty = val.value_type
        vt = LLVM.VoidType()
        FT = LLVM.FunctionType(vt, [ty, LLVM.PointerType(LLVM.Int8Type())])

        stringv = "Enzyme: Found nan while computing derivative of " * string(orig)
        if orig !== nothing && isa(orig, LLVM.Instruction)
            bt = GPUCompiler.backtrace(orig)
            stringv *= sprint(Base.Fix2(Base.show_backtrace, bt))
        end

        fn, _ = get_function!(mod, "julia.sanitize." * string(ty), FT)
        if isempty(fn.blocks)
            let builder = IRBuilder()
                entry = BasicBlock(fn, "entry")
                good = BasicBlock(fn, "good")
                bad = BasicBlock(fn, "bad")
                position!(builder, LLVM.at_end(entry))
                inp, sval = collect(fn.parameters)
                cmp = fcmp!(builder, LLVM.RealPredicate.UNO, inp, inp)

                br!(builder, cmp, bad, good)

                position!(builder, LLVM.at_end(good))
                ret!(builder)

                position!(builder, LLVM.at_end(bad))

                emit_error(builder, nothing, sval, EnzymeNoDerivativeError{Nothing, Nothing})
                unreachable!(builder)
                dispose(builder)
            end
        end
        # val = 
        call!(B, FT, fn, LLVM.Value[val, globalstring_ptr!(B, stringv)])
    end
    return val.ref
end

mutable struct EnzymeTapeToLoad{T}
    data::T
end
Base.eltype(::EnzymeTapeToLoad{T}) where {T} = T

# See get_current_task_from_pgcstack (used from 1.7+)
current_task_offset() =
    -(unsafe_load(cglobal(:jl_task_gcstack_offset, Cint)) ÷ sizeof(Ptr{Cvoid}))

# See get_current_ptls_from_task (used from 1.7+)
current_ptls_offset() =
    unsafe_load(cglobal(:jl_task_ptls_offset, Cint)) ÷ sizeof(Ptr{Cvoid})

function julia_post_cache_store(
    SI::LLVM.API.LLVMValueRef,
    B::LLVM.API.LLVMBuilderRef,
    R2::Ptr{UInt64},
)::Ptr{LLVM.API.LLVMValueRef}
    B = LLVM.IRBuilder(B)
    SI = LLVM.Instruction(SI)
    v = SI.operands[1]
    p = SI.operands[2]
    added = LLVM.API.LLVMValueRef[]
    while true
        if isa(p, LLVM.GetElementPtrInst) ||
           isa(p, LLVM.BitCastInst) ||
           isa(p, LLVM.AddrSpaceCastInst)
            p = p.operands[1]
            continue
        end
        break
    end
    if any_jltypes(v.value_type) && !isa(p, LLVM.AllocaInst)
        ctx = v.context
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
        if !isopaque(ty)
            val = const_pointercast(val, LLVM.PointerType(ty.element_type, Tracked))
        end
        if ty.addrspace != Tracked
            val = const_addrspacecast(val, ty)
        end
        return val.ref
    end
    if isa(ty, LLVM.ArrayType)
        st = LLVM.Value(julia_undef_value_for_type(mod, ty.element_type.ref, forceZero))
        return ConstantArray(ty.element_type, [st for i in 1:ty.length]).ref
    end
    if isa(ty, LLVM.StructType)
        vals = LLVM.Constant[]
        for st in ty.elements
            push!(vals, LLVM.Value(julia_undef_value_for_type(mod, st.ref, forceZero)))
        end
        return ConstantStruct(ty, vals).ref
    end
    throw(AssertionError("Unknown type to val: $(Ty)"))
end

# If count is nothing, it represents that we have an allocation of one of `Ty`. If it is a tuple LLVM values, it represents {the total size in bytes, the aligned size of each element}
function create_recursive_stores(B::LLVM.IRBuilder, @nospecialize(Ty::DataType), @nospecialize(prev::LLVM.Value), @nospecialize(count::Union{Nothing, Tuple{LLVM.Value, LLVM.ConstantInt}}))::Nothing
    if Base.datatype_pointerfree(Ty)
        return
    end

    isboxed_ref = Ref{Bool}()
    LLVMType = LLVM.LLVMType(ccall(:jl_type_to_llvm, LLVM.API.LLVMTypeRef,
                (Any, LLVM.Context, Ptr{Bool}), Ty, LLVM.context(), isboxed_ref))

    if !isboxed_ref[]
        zeroAll = false
        prev = bitcast!(B, prev, LLVM.PointerType(LLVMType, prev.value_type.addrspace))
        prev = addrspacecast!(B, prev, LLVM.PointerType(LLVMType, Derived))
	atomic = true
	if count === nothing
	    T_int64 = LLVM.Int64Type()
            zero_single_allocation(B, Ty, LLVMType, prev, zeroAll, LLVM.ConstantInt(T_int64, 0); atomic)
	    nothing
	else
	    (Size, AlignedSize) = count
	    zero_allocation(B, Ty, LLVMType, prev, AlignedSize, Size, zeroAll, atomic)
	    nothing
	end
    else
	if Ty == Core.SimpleVector
	   @assert count === nothing
	   @assert isa(prev, LLVM.CallInst)
            @assert (prev.called_operand::LLVM.Function).name == "julia.gc_alloc_obj"
            sz = prev.operands[2]
	   sz = sub!(B, sz, LLVM.ConstantInt(Int(sizeof(Ptr{Cvoid}))))
           T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
           T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
	   prev = addrspacecast!(B, prev, LLVM.PointerType(T_jlvalue, Derived))
	   prev = bitcast!(B, prev, LLVM.PointerType(T_prjlvalue, Derived))
	   gep = LLVM.gep!(B, T_prjlvalue, prev, LLVM.Value[LLVM.ConstantInt(Int64(1))])
	   zeroAll = false
	   atomic = true
	   zero_allocation(B, Any, T_prjlvalue, prev, LLVM.ConstantInt(sizeof(Ptr{Cvoid})), sz, zeroAll, atomic)
	   return
        end

        if fieldcount(Ty) == 0
            error("Error handling recursive stores for $Ty which has a fieldcount of 0")
        end

        T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

        T_int8 = LLVM.Int8Type()
        T_int64 = LLVM.Int64Type()
        
        T_pint8 = LLVM.PointerType(T_int8)

        prev2 = bitcast!(B, prev, LLVM.PointerType(T_int8, prev.value_type.addrspace))
        typedesc = Base.DataTypeFieldDesc(Ty)

	needs_fullzero = false
	if count !== nothing
		for i in 1:fieldcount(Ty)
		    Ty2 = fieldtype(Ty, i)
		    off = fieldoffset(Ty, i)

		    if typedesc[i].isptr || !(off == 0 && Base.aligned_sizeof(Ty) == Base.aligned_sizeof(Ty2))
			needs_fullzero = true
			break
		    end
		end
	end
        
	if needs_fullzero
		zeroAll = false
            prev = bitcast!(B, prev, LLVM.PointerType(LLVMType, prev.value_type.addrspace))
		prev = addrspacecast!(B, prev, LLVM.PointerType(LLVMType, Derived))
		atomic = true
	    (Size, AlignedSize) = count
	    zero_allocation(B, Ty, LLVMType, prev, AlignedSize, Size, zeroAll, atomic)
	    nothing
	else
		for i in 1:fieldcount(Ty)
		    Ty2 = fieldtype(Ty, i)
		    off = fieldoffset(Ty, i)

		    prev3 = inbounds_gep!(
			B,
			T_int8,
			prev2,
			LLVM.Value[LLVM.ConstantInt(Int64(off))],
		    )
		
		    if typedesc[i].isptr
			@assert count === nothing
			Ty2 = Any
			zeroAll = false
                    prev3 = bitcast!(B, prev3, LLVM.PointerType(T_prjlvalue, prev3.value_type.addrspace))
                    if prev3.value_type.addrspace != Derived
			  prev3 = addrspacecast!(B, prev3, LLVM.PointerType(T_prjlvalue, Derived))
			end
			zero_single_allocation(B, Ty2, T_prjlvalue, prev3, zeroAll, LLVM.ConstantInt(T_int64, 0); atomic=true)
		    elseif !typedesc[i].isptr && Ty2 isa Union
			# Inline union type e.g. Union{Int, Nothing} -> struct { ..., i64, i8 }
			continue
		    else
			if count !== nothing
			   @assert off == 0
			   @assert Base.aligned_sizeof(Ty) == Base.aligned_sizeof(Ty2)
			end
			create_recursive_stores(B, Ty2, prev3, count)
		    end
		end
		nothing
	end
    end
end

function shadow_alloc_rewrite(V::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, Orig::LLVM.API.LLVMValueRef, idx::UInt64, prev::API.LLVMValueRef, used::UInt8)
    enzyme_ctx = enzyme_context()
    used = used != 0
    V = LLVM.CallInst(V)
    gutils = GradientUtils(gutils)
    mode = get_mode(gutils)
    has, Ty, byref = abs_typeof(V, enzyme_ctx)
    partial = false
    count = nothing
    if !has
        arg = V
	if isa(arg, LLVM.CallInst)
            fn = arg.called_operand
		nm = ""
		if isa(fn, LLVM.Function)
                nm = fn.name
		end

		# Type tag is arg 3
		if nm == "julia.gc_alloc_obj" ||
			nm == "jl_gc_alloc_typed" ||
			nm == "ijl_gc_alloc_typed"
                totalsize = arg.operands[2]

                @assert totalsize.value_type isa LLVM.IntegerType
		   
                arg = arg.operands[3]

                ntuple = abs_ntuple_type(arg, enzyme_ctx)
                if ntuple !== nothing
                    Ty = ntuple[2]
                    # count should represent {the total size in bytes, the aligned size of each element}
                    alignsize = LLVM.ConstantInt(totalsize.value_type, Base.aligned_sizeof(Ty))
                    count = (totalsize, alignsize)
                    has = true
                elseif isa(arg, LLVM.CallInst)
                    fn = arg.called_operand
			nm = ""
			if isa(fn, LLVM.Function)
                        nm = fn.name
			end
                    if arg.callconv == 37 || nm == "julia.call"
			    index = 1
                        if arg.callconv != 37
                            fn = first(arg.operands)
                            nm = fn.name
				index += 1
			    end
			    if nm == "jl_f_apply_type" || nm == "ijl_f_apply_type"
				index += 1
				found = Any[]
                            legal, Ty = absint(arg.operands[index], enzyme_ctx, partial)
				Ty = unbind(Ty)
				if legal && Ty == NTuple
                                legal, Ty = absint(arg.operands[index + 2], enzyme_ctx)
				   Ty = unbind(Ty)
				   if legal
					# count should represent {the total size in bytes, the aligned size of each element}
					B = LLVM.IRBuilder()
                                    position!(B, LLVM.before(V))
                                    alignsize = LLVM.ConstantInt(totalsize.value_type, Base.aligned_sizeof(Ty))
					count = (totalsize, alignsize)
					has = true
				end
			    end
			end
	            end
		end
            end
        end


	if !has
            fn = V.parent.parent
	    throw(AssertionError("$(string(fn))\n Allocation could not have its type statically determined $(string(V))"))
	end
    end

    if mode == API.DEM_ReverseModePrimal ||
       mode == API.DEM_ReverseModeGradient ||
       mode == API.DEM_ReverseModeCombined
        if !guaranteed_nonactive(Ty, enzyme_world())
            B = LLVM.IRBuilder()
            position!(B, LLVM.before(V))
            V.operands[3] = unsafe_to_llvm(B, Base.RefValue{Ty})
        end
    end
  
    if Base.datatype_pointerfree(Ty)
	return
    end
    @static if VERSION >= v"1.11"
	if Ty <: GenericMemory
	    # TODO throw(AssertionError("What the heck is happening, why are we gc.alloca'ing memory, $(string(V)) $Ty"))
	    return
	end
    end

    if mode == API.DEM_ForwardMode && (used || idx != 0)
        # Zero any jlvalue_t inner elements of preceeding allocation.

        # Specifically in forward mode, you will first run the original allocation,
        # then all shadow allocations. These allocations will thus all run before
        # any value may store into them. For example, as follows:
        #   %orig = julia.gc_alloc(...)
        #   %"orig'" = julia.gcalloc(...)
        #   store orig[0] = jlvaluet
        #   store "orig'"[0] = jlvaluet'
        # As a result, by the time of the subsequent GC allocation, the memory in the preceeding
        # allocation might be undefined, and trigger a GC error. To avoid this,
        # we will explicitly zero the GC'd fields of the previous allocation.

        # Reverse mode will do similarly, except doing the shadow first
        prev = LLVM.Instruction(prev)
        B = LLVM.IRBuilder()
        position!(B, LLVM.after(prev))

	create_recursive_stores(B, Ty, prev, count)
    end
    if (mode == API.DEM_ReverseModePrimal || mode == API.DEM_ReverseModeCombined) && used
        # Zero any jlvalue_t inner elements of preceeding allocation.

        # Specifically in reverse mode, you will run the original allocation,
        # after all shadow allocations. The shadow allocations will thus all run before any value may store into them. For example, as follows:
        #   %"orig'" = julia.gcalloc(...)
        #   %orig = julia.gc_alloc(...)
        #   store "orig'"[0] = jlvaluet'
        #   store orig[0] = jlvaluet
        #
        # Normally this is fine, since we will memset right after the shadow
        # however we will do this memset non atomically and if you have a case like the following, there will be an issue

        #   %"orig'" = julia.gcalloc(...)
        #   memset("orig'")
        #   %orig = julia.gc_alloc(...)
        #   store "orig'"[0] = jlvaluet'
        #   store orig[0] = jlvaluet
        #
        # Julia could decide to dead store eliminate the memset (not being read before the store of jlvaluet'), resulting in an error
        B = LLVM.IRBuilder()
        position!(B, LLVM.after(V))
	
	create_recursive_stores(B, Ty, V, count)
    end

    nothing
end

function julia_allocator(
    B::LLVM.API.LLVMBuilderRef,
    LLVMType::LLVM.API.LLVMTypeRef,
    Count::LLVM.API.LLVMValueRef,
    AlignedSize::LLVM.API.LLVMValueRef,
    IsDefault::UInt8,
    ZI::Ptr{LLVM.API.LLVMValueRef},
)
    B = LLVM.IRBuilder(B)
    Count = LLVM.Value(Count)
    AlignedSize = LLVM.Value(AlignedSize)
    LLVMType = LLVM.LLVMType(LLVMType)
    return julia_allocator(B, LLVMType, Count, AlignedSize, IsDefault, ZI)
end

function fixup_return(B::LLVM.API.LLVMBuilderRef, retval::LLVM.API.LLVMValueRef)
    B = LLVM.IRBuilder(B)

    func = B.insert_block.parent
    mod = func.parent
    T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_prjlvalue_UT = LLVM.PointerType(T_jlvalue)

    retval = LLVM.Value(retval)
    ty = retval.value_type
    # Special case the union return { {} addr(10)*, i8 }
    #   which can be [ null, 1 ], to not have null in the ptr
    #   field, but nothing
    if isa(ty, LLVM.StructType)
        elems = ty.elements
        if length(elems) == 2 && elems[1] == T_prjlvalue
            fill_val = unsafe_to_llvm(B, nothing)
            prev = extract_value!(B, retval, 0)
            eq = icmp!(B, LLVM.IntPredicate.EQ, prev, LLVM.null(T_prjlvalue))
            retval = select!(B, eq, insert_value!(B, retval, fill_val, 0), retval)
        end
    end
    return retval.ref
end

function zero_allocation(B::LLVM.API.LLVMBuilderRef, LLVMType::LLVM.API.LLVMTypeRef, obj::LLVM.API.LLVMValueRef, isTape::UInt8)
    B = LLVM.IRBuilder(B)
    LLVMType = LLVM.LLVMType(LLVMType)
    obj = LLVM.Value(obj)
    jlType = Compiler.tape_type(LLVMType)
    zeroAll = isTape == 0
    func = B.insert_block.parent
    mod = func.parent
    T_int64 = LLVM.Int64Type()
    zero_single_allocation(B, jlType, LLVMType, obj, zeroAll, LLVM.ConstantInt(T_int64, 0))
    return nothing
end

function zero_single_allocation(builder::LLVM.IRBuilder, @nospecialize(jlType::DataType), @nospecialize(LLVMType::LLVM.LLVMType), @nospecialize(nobj::LLVM.Value), zeroAll::Bool, @nospecialize(idx::LLVM.Value); write_barrier=false, atomic=false)
    T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_prjlvalue_UT = LLVM.PointerType(T_jlvalue)

    todo = Tuple{Vector{LLVM.Value},LLVM.LLVMType,Type}[(
        LLVM.Value[idx],
        LLVMType,
        jlType,
    )]
	    
    addedvals = LLVM.Value[]
    while length(todo) != 0
        path, ty, jlty = popfirst!(todo)
        if isa(ty, LLVM.PointerType)
            if any_jltypes(ty)
                loc = gep!(builder, LLVMType, nobj, path)
                mod = builder.insert_block.parent.parent
                fill_val = unsafe_nothing_to_llvm(mod)
                push!(addedvals, fill_val)
                loc = bitcast!(
                    builder,
                    loc,
                    LLVM.PointerType(T_prjlvalue, loc.value_type.addrspace),
                )
                st = store!(builder, fill_val, loc)
                if atomic
                    st.ordering = LLVM.AtomicOrdering.Release
                    st.syncscope = LLVM.SyncScope("singlethread")
                    st.metadata["enzymejl_atomicgc"] = LLVM.MDNode(LLVM.Metadata[])
                end
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
            for i in 1:ty.length
                subTy = if jlty isa DataType
                    typed_fieldtype(jlty, i)
                elseif !(jlty isa DataType)
                    if ty.element_type isa LLVM.PointerType && ty.element_type.addrspace == 10
                       Any
                    else
                       throw(AssertionError("jlty=$jlty ty=$ty"))
                    end
                end
                npath = copy(path)
                push!(npath, LLVM.ConstantInt(LLVM.IntType(32), i - 1))
                push!(todo, (npath, ty.element_type, subTy))
            end
            continue
        end
        if isa(ty, LLVM.VectorType)
	    @assert jlty isa DataType
            for i in 1:ty.length
                npath = copy(path)
                push!(npath, LLVM.ConstantInt(LLVM.IntType(32), i - 1))
                push!(todo, (npath, ty.element_type, eltype(jlty)))
            end
            continue
        end
        if isa(ty, LLVM.StructType)
            i = 1
            if !(jlty isa DataType)
                throw(AssertionError("Could not handle non datatype $jlty in zero_single_allocation $ty"))
            end
            desc = Base.DataTypeFieldDesc(jlty)
            for ii = 1:fieldcount(jlty)
                jlet = typed_fieldtype(jlty, ii)
                if isghostty(jlet) || Core.Compiler.isconstType(jlet)
                    continue
                end

                t = ty.elements[i]
                npath = copy(path)
                push!(npath, LLVM.ConstantInt(LLVM.IntType(32), i - 1))
                push!(todo, (npath, t, jlet))
                i += 1

                # Extra i8 at the end of an inline union type
                if !desc[ii].isptr && jlet isa Union
                    i += 1
                end
            end
            if i != Int(length(ty.elements)) + 1
                throw(AssertionError("Number of non-ghost elements of julia type $jlty ($i) did not match number number of elements of llvmtype $(string(ty)) ($(length(ty.elements))) "))
            end
            continue
        end
    end
    if length(addedvals) != 0 && write_barrier
        pushfirst!(addedvals, get_base_and_offset(nobj; offsetAllowed=false, inttoptr=false)[1])
        emit_writebarrier!(builder, addedvals)
    end
    return nothing

end


function zero_allocation(
    B::LLVM.IRBuilder,
    @nospecialize(jlType::DataType),
    @nospecialize(LLVMType::LLVM.LLVMType),
    @nospecialize(obj::LLVM.Value),
    @nospecialize(AlignedSize::LLVM.Value),
    @nospecialize(Size::LLVM.Value),
    zeroAll::Bool,
    atomic::Bool=false
)::LLVM.API.LLVMValueRef
    func = B.insert_block.parent
    mod = func.parent
    T_int8 = LLVM.Int8Type()

    T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_prjlvalue_UT = LLVM.PointerType(T_jlvalue)

    name = "zeroType." * string(jlType)
    if atomic
	name = name * ".atomic"
    end

    wrapper_f = LLVM.Function(
        mod,
        name,
        LLVM.FunctionType(LLVM.VoidType(), [obj.value_type, T_int8, Size.value_type]),
    )
    push!(wrapper_f.function_attributes, StringAttribute("enzyme_math", "enzyme_zerotype"))
    push!(wrapper_f.function_attributes, StringAttribute("enzyme_inactive"))
    push!(wrapper_f.function_attributes, StringAttribute("enzyme_no_escaping_allocation"))
    push!(wrapper_f.function_attributes, EnumAttribute(:alwaysinline))
    push!(wrapper_f.function_attributes, EnumAttribute(:nofree))

    if LLVM.version().major <= 15
        push!(wrapper_f.function_attributes, EnumAttribute(:argmemonly))
        push!(wrapper_f.function_attributes, EnumAttribute(:writeonly))
    else
        push!(wrapper_f.function_attributes, EnumAttribute(:memory, WriteOnlyArgMemEffects.data))
    end
    push!(wrapper_f.function_attributes, EnumAttribute(:willreturn))
    push!(wrapper_f.function_attributes, EnumAttribute(:mustprogress))
    push!(wrapper_f.parameter_attributes[1], EnumAttribute(:writeonly))
    push!(wrapper_f.parameter_attributes[1], EnumAttribute(:nocapture))
    wrapper_f.linkage = LLVM.Linkage.Internal
    let builder = IRBuilder()
        entry = BasicBlock(wrapper_f, "entry")
        loop = BasicBlock(wrapper_f, "loop")
        exit = BasicBlock(wrapper_f, "exit")
        position!(builder, LLVM.at_end(entry))
        nobj, _, nsize = collect(wrapper_f.parameters)
        nobj = pointercast!(
            builder,
            nobj,
            LLVM.PointerType(LLVMType, nobj.value_type.addrspace),
        )

        cond = icmp!(builder, LLVM.IntPredicate.EQ, nsize, LLVM.ConstantInt(nsize.value_type, 0))
        br!(builder, cond, exit, loop)
        position!(builder, LLVM.at_end(loop))
        idx = LLVM.phi!(builder, Size.value_type, "zero_alloc_idx")
        inc = add!(builder, idx, LLVM.ConstantInt(Size.value_type, 1))
        append!(
            idx.incoming,
            [(LLVM.ConstantInt(Size.value_type, 0), entry), (inc, loop)],
        )

        zero_single_allocation(builder, jlType, LLVMType, nobj, zeroAll, idx; atomic)

        br!(
            builder,
            icmp!(
                builder,
                LLVM.IntPredicate.EQ,
                inc,
                exactudiv!(builder, nsize, AlignedSize),
            ),
            exit,
            loop,
        )
        position!(builder, LLVM.at_end(exit))

        ret!(builder)

        dispose(builder)
    end
    return call!(
        B,
        wrapper_f.function_type,
        wrapper_f,
        [obj, LLVM.ConstantInt(T_int8, 0), Size],
    ).ref
end

function julia_allocator(B::LLVM.IRBuilder, @nospecialize(LLVMType::LLVM.LLVMType), @nospecialize(Count::LLVM.Value), @nospecialize(AlignedSize::LLVM.Value), IsDefault::UInt8, ZI::Ptr{LLVM.API.LLVMValueRef})
    func = B.insert_block.parent
    mod = func.parent

    Size = nuwmul!(B, Count, AlignedSize) # should be nsw, nuw
    T_int8 = LLVM.Int8Type()

    if any_jltypes(LLVMType) || IsDefault != 0
        T_int64 = LLVM.Int64Type()
        T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
        T_pint8 = LLVM.PointerType(T_int8)
        T_ppint8 = LLVM.PointerType(T_pint8)

        esizeof(X) = X == Any ? sizeof(Int) : sizeof(X)

        TT = Compiler.tape_type(LLVMType)
        if esizeof(TT) != convert(Int, AlignedSize)
            GPUCompiler.@safe_error "Enzyme aligned size and Julia size disagree" AlignedSize =
                convert(Int, AlignedSize) esizeof(TT) fieldtypes(TT) LLVMType=strip(string(LLVMType))
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
            T_size_t = convert(LLVM.LLVMType, Int)
            if Count.value_type != T_size_t
                Count = trunc!(B, Count, T_size_t)
            end
            tag = emit_ntuple_type!(B, Count, TT)
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
        allocSize = if Size.value_type != T_size_t
            trunc!(B, Size, T_size_t)
        else
            Size
        end

        obj = emit_allocobj!(B, tag, allocSize, needs_dynamic_size_workaround)
        if CountTrackedPointers(LLVMType).all
            push!(
                obj.return_attributes, StringAttribute(
                    "enzyme_type",
                    "{[-1]:Pointer, [-1,-1]:Pointer}",
                )
            )
        end

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
            get_function!(mod, "malloc", LLVM.FunctionType(ptr8, [Count.value_type]))

        obj = call!(B, fty, mallocF, [Size])
        # if ZI != C_NULL
        #     unsafe_store!(ZI, LLVM.memset!(B, obj,  LLVM.ConstantInt(T_int8, 0),
        #                                           Size,
        #                                          #=align=#0 ).ref)
        # end
        AS = 0
    end

    push!(obj.return_attributes, EnumAttribute(:noalias))
    push!(obj.return_attributes, EnumAttribute(:nonnull))
    if isa(Count, LLVM.ConstantInt)
        val = convert(UInt, AlignedSize)
        val *= convert(UInt, Count)
        push!(obj.return_attributes, EnumAttribute(:dereferenceable, val))
        push!(obj.return_attributes, EnumAttribute(:dereferenceable_or_null, val))
    end

    mem = pointercast!(B, obj, LLVM.PointerType(LLVMType, AS))
    return LLVM.API.LLVMValueRef(mem.ref)
end

function julia_deallocator(B::LLVM.API.LLVMBuilderRef, Obj::LLVM.API.LLVMValueRef)
    B = LLVM.IRBuilder(B)
    Obj = LLVM.Value(Obj)
    julia_deallocator(B, Obj)
end

function julia_deallocator(B::LLVM.IRBuilder, @nospecialize(Obj::LLVM.Value))
    mod = B.insert_block.parent.parent

    T_void = LLVM.VoidType()
    if any_jltypes(Obj.value_type)
        return LLVM.API.LLVMValueRef(C_NULL)
    else
        ptr8 = LLVM.PointerType(LLVM.IntType(8))
        freeF, fty = get_function!(mod, "free", LLVM.FunctionType(T_void, [ptr8]))
        callf = call!(B, fty, freeF, [pointercast!(B, Obj, ptr8)])
        push!(callf.argument_attributes[1], EnumAttribute(:nonnull))
    end
    return LLVM.API.LLVMValueRef(callf.ref)
end

function emit_inacterror(B::LLVM.API.LLVMBuilderRef, V::LLVM.API.LLVMValueRef, orig::LLVM.API.LLVMValueRef)
    B = LLVM.IRBuilder(B)
    curent_bb = B.insert_block
    orig = LLVM.Value(orig)
    fn = curent_bb.parent
    mod = fn.parent

    bt = GPUCompiler.backtrace(orig)
    bts = sprint(Base.Fix2(Base.show_backtrace, bt))
    fmt = globalstring_ptr!(B, "%s:\nBacktrace\n" * bts)

    funcT = LLVM.FunctionType(
        LLVM.VoidType(),
        LLVMType[LLVM.PointerType(LLVM.Int8Type())],
        vararg = true,
    )
    func, _ = get_function!(mod, "jl_errorf", funcT, LLVM.Attribute[EnumAttribute(:noreturn)])

    call!(B, funcT, func, LLVM.Value[fmt, LLVM.Value(V)])
    return nothing
end

include("rules/allocrules.jl")
include("rules/llvmrules.jl")

function add_one_in_place(x)
    if x isa Base.RefValue
        x[] = recursive_add(x[], default_adjoint(eltype(Core.Typeof(x))))
    elseif x isa (Array{T,0} where T)
        x[] = recursive_add(x[], default_adjoint(eltype(Core.Typeof(x))))
    else
        throw(EnzymeNonScalarReturnException(x, ""))
    end
    return nothing
end

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
    ("enz_runtime_idx_jl_getfield_rev", Enzyme.Compiler.idx_jl_getfield_rev),
    ("enz_runtime_jl_setfield_aug", Enzyme.Compiler.rt_jl_setfield_aug),
    ("enz_runtime_jl_setfield_rev", Enzyme.Compiler.rt_jl_setfield_rev),
    ("enz_runtime_error_if_differentiable", Enzyme.Compiler.error_if_differentiable),
    ("enz_runtime_error_if_active", Enzyme.Compiler.error_if_active),
    ("enz_add_one_in_place", Enzyme.Compiler.add_one_in_place),
)
    JuliaEnzymeNameMap[k] = v
end

function __init__()
    API.memmove_warning!(false)
    API.typeWarning!(false)
    API.EnzymeNonPower2Cache!(false)
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
            (LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef, UInt64, LLVM.API.LLVMValueRef, UInt8)
        )
    )
    register_alloc_rules()
    register_llvm_rules()
end

# FIXME: Use params.parent in more places where we rely on the behavior of the underlying 
function GPUCompiler.nest_params(params::AbstractEnzymeCompilerParams, parent::AbstractCompilerParams)
    EnzymeCompilerParams(
        parent,
        params.TT,
        params.mode,
        params.width,
        params.rt,
        params.run_enzyme,
        params.abiwrap,
        params.modifiedBetween,
        params.returnPrimal,
        params.shadowInit,
        params.expectedTapeType,
        params.ABI,
        params.err_if_func_written,
        params.runtimeActivity,
        params.strongZero,
    )
end

# Backends define `method_table` and `method_table_view` for their own
# `CompilerJob{Target, Params}` (e.g. CUDA.jl for
# `CompilerJob{PTXCompilerTarget, CUDACompilerParams}`). An Enzyme job wraps both the
# target and the params, so those definitions would not apply and the job would silently
# fall back to the global method table, while the primal code emitted for it (see
# `codegen`) is compiled under the backend's overlay table(s). Unwrap the job so both agree.
function unwrap_enzyme_job(@nospecialize(job::CompilerJob{<:EnzymeTarget, <:EnzymeCompilerParams}))
    primal_config = CompilerConfig(job.config; target = job.config.target.target, params = job.config.params.params)
    return CompilerJob(job.source, primal_config, job.world)
end
GPUCompiler.method_table(@nospecialize(job::CompilerJob{<:EnzymeTarget, <:EnzymeCompilerParams})) =
    GPUCompiler.method_table(unwrap_enzyme_job(job))
GPUCompiler.method_table_view(@nospecialize(job::CompilerJob{<:EnzymeTarget, <:EnzymeCompilerParams})) =
    GPUCompiler.method_table_view(unwrap_enzyme_job(job))

struct UnknownTapeType end


##
# Enzyme compiler step
##

function enzyme_custom_extract_mi(orig::LLVM.CallInst, error::Bool = true)
    operand = orig.called_operand
    if isa(operand, LLVM.Function)
        return enzyme_custom_extract_mi(operand::LLVM.Function, error)
    elseif error
        GPUCompiler.@safe_error "Enzyme: Custom handler, could not find fn", orig
    end
    return nothing, nothing
end

function enzyme_custom_extract_mi(orig::LLVM.Function, error::Bool = true)
    mi = nothing
    RT = nothing
    for fattr in collect(orig.function_attributes)
        if isa(fattr, LLVM.StringAttribute)
            if fattr.kind == "enzymejl_mi"
                ptr = reinterpret(Ptr{Cvoid}, parse(UInt, fattr.value))
                mi = Base.unsafe_pointer_to_objref(ptr)
            end
            if fattr.kind == "enzymejl_rt"
                ptr = reinterpret(Ptr{Cvoid}, parse(UInt, fattr.value))
                RT = Base.unsafe_pointer_to_objref(ptr)
            end
        end
    end
    if error && mi === nothing
        GPUCompiler.@safe_error "Enzyme: Custom handler, could not find mi", orig
    end
    return mi, RT
end

function enzyme_extract_parm_type(fn::LLVM.Function, idx::Int, error::Bool = true)
    ty = nothing
    byref = nothing
    for fattr in collect(fn.parameter_attributes[idx])
        if isa(fattr, LLVM.StringAttribute)
            if fattr.kind == "enzymejl_parmtype"
                ptr = reinterpret(Ptr{Cvoid}, parse(UInt, fattr.value))
                ty = Base.unsafe_pointer_to_objref(ptr)
            end
            if fattr.kind == "enzymejl_parmtype_ref"
                byref = GPUCompiler.ArgumentCC(parse(UInt, fattr.value))
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

include("rules/activityrules.jl")

const DumpPreEnzyme = Ref(false)
const DumpPostEnzyme = Ref(false)
const DumpPostWrap = Ref(false)

function enzyme!(
    job::CompilerJob,
    interp,
    mod::LLVM.Module,
    primalf::LLVM.Function,
    @nospecialize(TT::Type),
    mode::API.CDerivativeMode,
    width::Int,
    parallel::Bool,
    @nospecialize(actualRetType::Type),
    wrap::Bool,
    @nospecialize(modifiedBetween::NTuple{N, Bool} where N),
    returnPrimal::Bool,
    @nospecialize(expectedTapeType::Type),
    loweredArgs::Set{Int},
    boxedArgs::Set{Int},
    removedRoots::Set{Int},
        enzyme_ctx::EnzymeContext,
)
    if DumpPreEnzyme[]
        API.EnzymeDumpModuleRef(mod.ref)
    end
    rt = job.config.params.rt
    runtimeActivity = job.config.params.runtimeActivity
    strongZero = job.config.params.strongZero
    @assert eltype(rt) != Union{}

    shadow_init = job.config.params.shadowInit
    ctx = mod.context
    dl = string(mod.datalayout)

    tt = [TT.parameters[2:end]...]

    args_activity = API.CDIFFE_TYPE[]
    uncacheable_args = Bool[]
    args_typeInfo = TypeTree[]
    args_known_values = API.IntList[]


    @assert length(modifiedBetween) == length(TT.parameters)

    swiftself = has_swiftself(primalf)
    if swiftself
        push!(args_activity, API.DFT_CONSTANT)
        push!(args_typeInfo, TypeTree())
        push!(uncacheable_args, false)
        push!(args_known_values, API.IntList())
    end

    seen = TypeTreeTable()
    
    seen_roots = 0

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

        isboxed = (i + seen_roots) in boxedArgs
        inline_root = false
        

        if inline_roots_type(eltype(T)) != 0
            # TODO(wmoses,vchuravy) if a parameter is removed, it is not possible to know if it was byref or not, we will assume it is byref for now.
            # Note that getting this wrong can result in segfaults, invalid IR, or worse.

            # This is already after lower_convention
            seen_roots += 1
            if false
                inline_root = true
            end
        end

        if T <: Const
            push!(args_activity, API.DFT_CONSTANT)
	    if inline_root
               push!(args_activity, API.DFT_CONSTANT)
	    end
        elseif T <: Active
            if isboxed
	    	@assert !inline_root
                push!(args_activity, API.DFT_DUP_ARG)
            else
                push!(args_activity, API.DFT_OUT_DIFF)
	        if inline_root
                   push!(args_activity, API.DFT_CONSTANT)
	        end
            end
        elseif T <: Duplicated ||
               T <: BatchDuplicated ||
               T <: BatchDuplicatedFunc ||
               T <: MixedDuplicated ||
               T <: BatchMixedDuplicated
            push!(args_activity, API.DFT_DUP_ARG)
	    if inline_root
               push!(args_activity, API.DFT_DUP_ARG)
	    end
        elseif T <: DuplicatedNoNeed || T <: BatchDuplicatedNoNeed
            push!(args_activity, API.DFT_DUP_NONEED)
	    if inline_root
               push!(args_activity, API.DFT_DUP_ARG)
	    end
        else
            error("illegal annotation type $T")
        end
        typeTree = typetree_in_world(job.world, source_typ, ctx, dl, seen)
        if isboxed
            typeTree = copy(typeTree)
            merge!(typeTree, TypeTree(API.DT_Pointer, ctx))
            only!(typeTree, -1)
        end
        push!(args_typeInfo, typeTree)
        push!(uncacheable_args, modifiedBetween[i])
        push!(args_known_values, API.IntList())
	if inline_root
            typeTree = typetree_in_world(job.world, Any, ctx, dl, seen)
           push!(args_typeInfo, typeTree)
           push!(uncacheable_args, modifiedBetween[i])
           push!(args_known_values, API.IntList())
	end
    end
    if length(uncacheable_args) != length(collect(primalf.parameters))
                msg = sprint() do io
		    println(io, "length(uncacheable_args) != length(collect(parameters(primalf))) ")
		    println(io, "TT=", TT)
                    println(io, "modifiedBetween=", modifiedBetween)
		    println(io, "uncacheable_args=", uncacheable_args)
		    println(io, "primal", string(primalf))
                end
                throw(AssertionError(msg))
    end
    @assert length(args_typeInfo) == length(collect(primalf.parameters))

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

    LLVM.@dispose logic = Logic() begin

    TA = TypeAnalysis(logic)

    retTT = if !isa(actualRetType, Union) &&
            actualRetType <: Tuple &&
            in(Any, actualRetType.parameters)
        TypeTree()
    else
            typeTree = typetree_in_world(job.world, actualRetType, ctx, dl, seen)
        if !isa(actualRetType, Union) && GPUCompiler.deserves_retbox(actualRetType)
            typeTree = copy(typeTree)
            merge!(typeTree, TypeTree(API.DT_Pointer, ctx))
            only!(typeTree, -1)
        end
        typeTree
    end

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
        nowrite_shadows = zeros(UInt8, length(uncacheable_args))
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
            nowrite_shadows,
            false,
            runtimeActivity,
            strongZero,
            width,
            parallel,
        ) #=atomicAdd=#

        # 2. get new_primalf and tape
        augmented_primalf =
            LLVM.Function(API.EnzymeExtractFunctionFromAugmentation(augmented))
        tape = API.EnzymeExtractTapeTypeFromAugmentation(augmented)
        utape = API.EnzymeExtractUnderlyingTapeTypeFromAugmentation(augmented)
        if utape != C_NULL
            TapeType = EnzymeTapeToLoad{Compiler.tape_type(LLVMType(utape))}
            tape = utape
        elseif tape != C_NULL
            TapeType = Compiler.tape_type(LLVMType(tape))
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
                interp,
                runtimeActivity,
                    enzyme_ctx,
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
                strongZero,
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
                interp,
                    runtimeActivity,
                    enzyme_ctx,
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
                strongZero,
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
                interp,
                    runtimeActivity,
                    enzyme_ctx,
            )
        end
    elseif mode == API.DEM_ForwardMode
        returnUsed = !(isghostty(actualRetType) || Core.Compiler.isconstType(actualRetType))

        literal_rt = eltype(rt)

        if !isghostty(literal_rt) && runtimeActivity && GPUCompiler.deserves_argbox(actualRetType) && !GPUCompiler.deserves_argbox(literal_rt)
        else
            returnUsed &= returnPrimal        
        end

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
                strongZero,
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
                interp,
                    runtimeActivity,
                    enzyme_ctx,
            )
        end
    else
        @assert "Unhandled derivative mode", mode
    end
    if DumpPostWrap[]
        API.EnzymeDumpModuleRef(mod.ref)
    end

    # Rewrite enzyme_ignore_derivatives functions to the identity of their first argument.
    to_delete = LLVM.Function[]
        for fn in mod.functions
            if startswith(fn.name, "__enzyme_ignore_derivatives")
            push!(to_delete, fn)
            to_delete_inst = LLVM.CallInst[]
                for u in fn.uses
                    ci = u.user
                @assert isa(ci, LLVM.CallInst)
                    LLVM.replace_uses!(ci, ci.operands[1])
                push!(to_delete_inst, ci)
            end
            for ci in to_delete_inst
                LLVM.erase!(ci)
            end
        end
    end
    for fn in to_delete
        LLVM.erase!(fn)
    end
    LLVM.verify(mod)

    API.EnzymeLogicErasePreprocessedFunctions(logic)
        adjointfname = adjointf == nothing ? nothing : adjointf.name
    augmented_primalfname =
            augmented_primalf == nothing ? nothing : augmented_primalf.name
        @dispose pb = PassBuilder() begin
        registerEnzymeAndPassPipeline!(pb)
        add!(pb, "enzyme-fixup-batched-julia")
        run!(pb, mod)
    end
    run!(DCEPass(), mod)
    fix_decayaddr!(mod)
        adjointf = adjointf == nothing ? nothing : mod.functions[adjointfname]
    augmented_primalf =
            augmented_primalf == nothing ? nothing : mod.functions[augmented_primalfname]
    if DumpPostEnzyme[]
        API.EnzymeDumpModuleRef(mod.ref)
    end

    return adjointf, augmented_primalf, TapeType
    end # @dispose logic
end

function create_abi_wrapper(
    enzymefn::LLVM.Function,
    @nospecialize(TT::Type),
    @nospecialize(rettype::Type),
    @nospecialize(actualRetType::Type),
    Mode::API.CDerivativeMode,
    augmented,
    width::Int,
    returnPrimal::Bool,
    shadow_init::Bool,
    interp,
        runtime_activity::Bool,
        enzyme_ctx::EnzymeContext,
)
    world = enzyme_world()
    is_adjoint = Mode == API.DEM_ReverseModeGradient || Mode == API.DEM_ReverseModeCombined
    is_split = Mode == API.DEM_ReverseModeGradient || Mode == API.DEM_ReverseModePrimal
    needs_tape = Mode == API.DEM_ReverseModeGradient

    mod = enzymefn.parent
    ctx = mod.context

    # TODO
    arg_rooting = false # true

    push!(enzymefn.function_attributes, EnumAttribute(:alwaysinline))
    hasNoInline = haskey(enzymefn.function_attributes, :noinline)
    if hasNoInline
        delete!(enzymefn.function_attributes, :noinline)
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
        arg_roots = isboxed ? inline_roots_type(source_typ) : 0
        if arg_rooting && arg_roots != 0
           push!(T_wrapperargs, convert(LLVMType, AnyArray(arg_roots)))
        end

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
        elseif T <: Duplicated || T <: DuplicatedNoNeed || T <: BatchDuplicated || T <: BatchDuplicatedNoNeed
            push!(T_wrapperargs, LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmT)))
            if inline_roots_type(source_typ) != 0
                @assert isboxed == GPUCompiler.deserves_argbox(T)
            end
            if arg_rooting && arg_roots != 0
               push!(T_wrapperargs, convert(LLVMType, AnyArray(width * arg_roots)))
            end
            if is_adjoint && i != 1
                push!(ActiveRetTypes, Nothing)
            end
        elseif T <: MixedDuplicated || T <: BatchMixedDuplicated
            push!(T_wrapperargs, LLVM.LLVMType(API.EnzymeGetShadowType(width, T_prjlvalue)))
            if inline_roots_type(source_typ) != 0
                @assert isboxed == GPUCompiler.deserves_argbox(T)
            end
            if arg_rooting && arg_roots != 0
               push!(T_wrapperargs, convert(LLVMType, AnyArray(width * arg_roots)))
            end
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
                throw(NonInferredActiveReturn(actualRetType, rettype))
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
            # TODO(wmoses,vchuravy) if a parameter is removed, it is not possible to know if it was byref or not, we will assume it is byref for now.
            # Note that getting this wrong can result in segfaults, invalid IR, or worse.
            arg_roots = inline_roots_type(actualRetType)
            if arg_rooting && arg_roots != 0
               push!(T_wrapperargs, convert(LLVMType, AnyArray(width * arg_roots)))
            end
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
            TapeType = EnzymeTapeToLoad{Compiler.tape_type(LLVMType(utape))}
        elseif tape != C_NULL
            TapeType = Compiler.tape_type(LLVMType(tape))
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
            # A non-batch activity describes a single shadow, so it is only legal at
            # width one; a batch activity must agree with the width it was built for.
            if rettype <: Duplicated ||
                    rettype <: DuplicatedNoNeed ||
                    rettype <: MixedDuplicated
                @assert width == 1
            elseif rettype <: BatchDuplicated ||
                    rettype <: BatchDuplicatedNoNeed ||
                    rettype <: BatchDuplicatedFunc ||
                    rettype <: BatchMixedDuplicated
                @assert width == batch_size(rettype)
            end
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
                rty = if Base.isconcretetype(literal_rt)
                    Base.RefValue{literal_rt}
                else
                    (Base.RefValue{T} where T <: literal_rt)
                end
                if width == 1
                    push!(sret_types, rty)
                else
                    push!(
                        sret_types,
                        AnonymousStruct(NTuple{width,rty}),
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
        numLLVMReturns = jltype.length
    elseif isa(jltype, LLVM.StructType)
        numLLVMReturns = length(jltype.elements)
    elseif isa(jltype, LLVM.VoidType)
        numLLVMReturns = 0
    else
        @assert false "illegal rt"
    end

    returnRoots = false
    root_ty = nothing
    tracked = nothing
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
            jltape = convert(LLVM.LLVMType, Compiler.tape_type(tape); allow_boxed = true)
            push!(T_wrapperargs, jltape)
            arg_roots = inline_roots_type(tape)
            if arg_rooting && arg_roots != 0
               push!(T_wrapperargs, convert(LLVMType, AnyArray(arg_roots)))
            end
        else
            needs_tape = false
        end
    end

    T_ret = returnRoots ? T_void : jltype
    FT = LLVM.FunctionType(T_ret, T_wrapperargs)
    llvm_f = LLVM.Function(mod, safe_name(enzymefn.name * "wrap"), FT)
    API.EnzymeCloneFunctionDISubprogramInto(llvm_f, enzymefn)
    dl = mod.datalayout

    params = [llvm_f.parameters...]

    builder = LLVM.IRBuilder()
    entry = BasicBlock(llvm_f, "entry")
    position!(builder, LLVM.at_end(entry))

    realparms = LLVM.Value[]
    i = 1

    if returnRoots
        sret = params[i]
        i += 1

        attr = TypeAttribute(:sret, jltype)
        push!(llvm_f.parameter_attributes[1], attr)
        push!(llvm_f.parameter_attributes[1], EnumAttribute(:noalias))
        push!(llvm_f.parameter_attributes[2], StringAttribute("enzymejl_returnRoots", string(Int(tracked.count))))
        push!(llvm_f.parameter_attributes[2], EnumAttribute(:noalias))
    elseif jltype != T_void
        sret = alloca!(builder, jltype, "abi_wrapper_sret")
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

        llty = params[i].value_type

        convty = convert(LLVMType, T′; allow_boxed = true)

        arg_roots = isboxed ? inline_roots_type(T′) : 0

        if (T <: MixedDuplicated || T <: BatchMixedDuplicated) && !isboxed # && (isa(llty, LLVM.ArrayType) || isa(llty, LLVM.StructType))
            @assert Base.isconcretetype(T′)
            al0 = al = emit_allocobj!(builder, Base.RefValue{T′}, "mixedparameter")
            parm = params[i]
            if arg_rooting && arg_roots != 0
                parm = recombine_value!(builder, parm, params[i+1])
                i += 1
            end
            al = bitcast!(builder, al, LLVM.PointerType(llty, al.value_type.addrspace))
            store!(builder, parm, al)
            emit_writebarrier!(builder, get_julia_inner_types(builder, al0, parm))
            al = addrspacecast!(builder, al, LLVM.PointerType(llty, Derived))
            push!(realparms, al)
        else
            push!(realparms, params[i])
        end

        i += 1
        if T <: Const
	    if arg_rooting && arg_roots != 0
		 push(realparms, params[i])
		 i += 1
	    end
        elseif T <: Active
            isboxed = GPUCompiler.deserves_argbox(T′)
            if isboxed
		@assert arg_roots == 0
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
                        LLVM.storage_size(dl, ptr.value_type.element_type),
                    ),
                    0,
                )                                            #=align=#
            end
	    if arg_rooting &&arg_roots != 0
		 push(realparms, params[i])
		 i += 1
	    end
            activeNum += 1
        elseif T <: Duplicated || T <: DuplicatedNoNeed || T <: BatchDuplicated || T <: BatchDuplicatedNoNeed
	    # Enzyme expects, arg, darg, root, droot
	    # Julia expects   arg, root, darg, droot
	    # We already pushed arg
	    # now params[i] refers to root
	    isboxed = (T <: BatchDuplicated || T <: BatchDuplicatedNoNeed) && GPUCompiler.deserves_argbox(NTuple{width,T′})
	    darg = nothing
	    root = nothing
	    droot = nothing
	    if arg_rooting && arg_roots != 0
		 root = params[i]
		 darg = params[i+1]
		 droot = params[i+2]
		 i += 3
	    else
		 darg = params[i]
		 i += 1
	    end

	    if isboxed
	        darg = load!(builder, convert(LLVMType, NTuple{width,T′}), darg)
	    end
	    push!(realparms, darg)
	    if arg_rooting && arg_roots != 0
		push!(realparms, root)
		push!(realparms, droot)
	    end
        elseif T <: MixedDuplicated || T <: BatchMixedDuplicated
	    # Enzyme expects, arg, [w x darg], root, droot
	    # Julia expects   arg, root, darg, droot
	    # We already pushed arg
	    # now params[i] referrs to root
	    darg = nothing
	    root = nothing
	    droot = nothing
	    if arg_rooting && arg_roots != 0
		 root = params[i]
		 darg = params[i+1]
		 droot = params[i+2]
		 i += 3
	    else
		 darg = params[i]
		 i += 1
	    end

            if T <: BatchMixedDuplicated
                @assert Base.isconcretetype(T′)
                if GPUCompiler.deserves_argbox(NTuple{width,Base.RefValue{T′}})
                    njlvalue = LLVM.ArrayType(Int(width), T_prjlvalue)
                    parmsi = bitcast!(
                        builder,
                        darg,
                        LLVM.PointerType(njlvalue, darg.value_type.addrspace),
                    )
                    darg = load!(builder, njlvalue, darg)
                end
            end

            isboxed = GPUCompiler.deserves_argbox(T′)

            resty = isboxed ? llty : LLVM.PointerType(llty, Derived)

            ival = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, resty)))
            for idx = 1:width
                pv = (width == 1) ? darg : extract_value!(builder, darg, idx - 1)
                pv =
                    bitcast!(builder, pv, LLVM.PointerType(llty, pv.value_type.addrspace))
                pv = addrspacecast!(builder, pv, LLVM.PointerType(llty, Derived))
                if isboxed
                    pv = load!(builder, llty, pv, "mixedboxload")
                end
                ival = (width == 1) ? pv : insert_value!(builder, ival, pv, idx - 1)
            end

            push!(realparms, ival)
	    
	    if arg_rooting && arg_roots != 0
		push!(realparms, root)
		push!(realparms, droot)
	    end
        elseif T <: BatchDuplicatedFunc
	    # TODO handle this
	    if arg_rooting
		 @assert arg_roots == 0
	    end
            Func = get_func(T)
            funcspec = my_methodinstance(Mode == API.DEM_ForwardMode ? Forward : Reverse, Func, Tuple{}, world)
            llvmf = nested_codegen!(Mode, mod, funcspec)
            push!(llvmf.function_attributes, EnumAttribute(:alwaysinline))
            Func_RT = return_type(interp, funcspec)
            @assert Func_RT == NTuple{width,T′}
            _, psret, _ = get_return_info(Func_RT)
            args = LLVM.Value[]
            if psret !== nothing
                psret = alloca!(builder, convert(LLVMType, Func_RT), "psret")
                push!(args, psret)
            end
            res = LLVM.call!(builder, llvmf.function_type, llvmf, args)
            if psret !== nothing
                attr = TypeAttribute(:sret, convert(LLVMType, Func_RT))
                push!(res.argument_attributes[1], attr)
            end
            if llvm_f.subprogram !== nothing
                res.debug_location = DILocation(0, 0, llvm_f.subprogram)
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

    val = call!(builder, enzymefn.function_type, enzymefn, realparms)
    if llvm_f.subprogram !== nothing
        val.debug_location = DILocation(0, 0, llvm_f.subprogram)
    end

    @inline function fixup_abi(index::Int, @nospecialize(value::LLVM.Value))
        valty = sret_types[index]

        # Union becoming part of a tuple needs to be adjusted
        # See https://github.com/JuliaLang/julia/blob/81afdbc36b365fcbf3ae25b7451c6cb5798c0c3d/src/cgutils.cpp#L3795C1-L3801C121
        if valty isa Union
            T_int8 = LLVM.Int8Type()
            if value.value_type == T_int8
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
           active_reg(pactualRetType, world; justActive=true, UnionSret=true) == ActiveState
            rewrite_union_returns_as_ref(enzymefn, data[3], world, width, enzyme_ctx)
        end
        returnNum = 0
        for i = 1:3
            if existed[i] != 0
                eval = val
                if data[i] != -1
                    eval = extract_value!(builder, val, data[i], "revprimal_extract_$(i)")
                end
                if i == 2 && actualRetType != literal_rt
                    if Base.isconcretetype(literal_rt) && !Base.isconcretetype(actualRetType)
                        eval = addrspacecast!(builder, eval, LLVM.PointerType(LLVM.StructType(LLVM.LLVMType[]), Derived))
                        lvalty = convert(LLVM.LLVMType, literal_rt)
                        eval = bitcast!(builder, eval, LLVM.PointerType(lvalty, Derived))
                        eval = load!(builder, lvalty, eval)
                    else
			emit_error(builder, nothing, "Unexpected type inference from LLVM codegen. \nActual return type from GPUCompiler: $(actualRetType)\n Inferred return type: $(literal_rt)\n rettype=$(rettype)\n Mode=$Mode\n TT=$TT")
                    end 
                end
                if i == 3
                    if rettype <: MixedDuplicated || rettype <: BatchMixedDuplicated
                        ival = UndefValue(
                            LLVM.LLVMType(API.EnzymeGetShadowType(width, T_prjlvalue)),
                        )
                        for idx = 1:width
                            pv =
                                (width == 1) ? eval : extract_value!(builder, eval, idx - 1)
                            irt = eltype(rettype)
                            ires = if Base.isconcretetype(irt)
                                al = emit_allocobj!(
                                    builder,
                                    Base.RefValue{eltype(rettype)},
                                    "batchmixedret",
                                )
                                al0 = al
                                llty = pv.value_type
                                al = bitcast!(
                                    builder,
                                    al,
                                    LLVM.PointerType(llty, al.value_type.addrspace),
                                )
                                store!(builder, pv, al)
                                emit_writebarrier!(
                                    builder,
                                    get_julia_inner_types(builder, al0, pv),
                                )
                                al0
                            else
                                # emit_allocobj!(
                                #     builder,
                                #     emit_apply_type!(builder, Base.RefValue, [emit_jltypeof!(builder, pv)]),
                                #     "batchmixedret",
                                # )
                                pv
                            end
                            ival =
                                (width == 1) ? ires :
                                insert_value!(builder, ival, ires, idx - 1)
                        end
                        eval = ival
                    elseif actualRetType != literal_rt
                        if Base.isconcretetype(literal_rt) && !Base.isconcretetype(actualRetType)
                            lvalty = convert(LLVM.LLVMType, literal_rt)
                            ival = UndefValue(
                                LLVM.LLVMType(API.EnzymeGetShadowType(width, lvalty)),
                            )
                            for idx = 1:width
                                pv =
                                    (width == 1) ? eval : extract_value!(builder, eval, idx - 1)
                                eval = addrspacecast!(builder, eval, LLVM.PointerType(LLVM.StructType(LLVM.LLVMType[]), Derived))
                                eval = bitcast!(builder, eval, LLVM.PointerType(lvalty, Derived))
                                eval = load!(builder, lvalty, eval)

                                ival =
                                    (width == 1) ? eval :
                                    insert_value!(builder, ival, eval, idx - 1)
                            end
                            eval = ival
                        else
			    emit_error(builder, nothing, "Unexpected type inference from LLVM codegen. \nActual return type from GPUCompiler: $(actualRetType)\n Inferred return type: $(literal_rt)\n rettype=$(rettype)\n Mode=$Mode\n TT=$TT")
                        end 
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
                    "revprimal_1_wrap_sret_gep_$returnNum"
                )
                ptr = pointercast!(
                    builder, ptr, LLVM.PointerType(eval.value_type),
                    "revprimal_1_wrap_sret_cast_$returnNum")
                extract_struct_into!(builder, ptr, eval, "revprimal_1_wrap_sret_extract_$returnNum")
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
                        c = emit_apply_generic!(builder, LLVM.Value[unsafe_to_llvm(builder, add_one_in_place), shadowv])
                        if llvm_f.subprogram !== nothing
                            c.debug_location =
                                DILocation(0, 0, llvm_f.subprogram)
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
                    "revprimal_2_wrap_sret_gep_$returnNum"
                )
                ptr = pointercast!(builder, ptr, LLVM.PointerType(eval.value_type), "revprimal_1_wrap_sret_cast_$returnNum")
        		extract_struct_into!(builder, ptr, eval, "revprimal_2_wrap_sret_extract_$returnNum")
                returnNum += 1
            end
        end
        @assert returnNum == numLLVMReturns
    elseif Mode == API.DEM_ForwardMode
        count_Sret = 0
        count_llvm_Sret = 0
        if !isghostty(actualRetType)
            if !Core.Compiler.isconstType(actualRetType)
                if returnPrimal || (!isghostty(literal_rt) && runtime_activity && GPUCompiler.deserves_argbox(actualRetType) && !GPUCompiler.deserves_argbox(literal_rt))
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
            eval = if count_llvm_Sret == 0
                makeInstanceOf(builder, actualRetType)
            elseif count_llvm_Sret == 1
                val
            else
                @assert count_llvm_Sret > 1
                if !returnPrimal && (runtime_activity && GPUCompiler.deserves_argbox(actualRetType) && !GPUCompiler.deserves_argbox(literal_rt))
                    extract_value!(builder, val, 1)
                else
                    extract_value!(builder, val, 1 - returnNum)
                end
            end

            if count_llvm_Sret != 0 && GPUCompiler.deserves_argbox(actualRetType) && !GPUCompiler.deserves_argbox(literal_rt)
                twidth = if width == 1
                    1
                else
                    if (rettype <: Const) && returnNum == 0
                        1
                    else
                        width
                    end
                end

                SPT0 = convert(LLVMType, literal_rt)

                compare = nothing

                # only compare for derivative (aka returnNum == 0), when runtime activity is on and required checking
                if returnNum == 0 && runtime_activity && GPUCompiler.deserves_argbox(actualRetType) && !GPUCompiler.deserves_argbox(literal_rt)
                    compare = extract_value!(builder, val, 0)
                end

                if twidth == 1
                    eval0 = eval
                    SPT = LLVM.PointerType(SPT0, eval.value_type.addrspace)
                    eval = bitcast!(builder, eval, SPT)
                    eval = addrspacecast!(builder, eval, LLVM.PointerType(SPT0, Derived))
                    eval = load!(builder, SPT0, eval)
                    if !(compare isa Nothing)
                        is_inactive = icmp!(builder, LLVM.IntPredicate.EQ, eval0, compare)
                        eval = select!(builder, is_inactive, LLVM.null(SPT0), eval)
                    end
                else
                    ival = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(twidth, SPT0)))
                    for idx in 1:twidth
                        pv = extract_value!(builder, eval, idx - 1)
                        pv0 = pv
                        pv = bitcast!(builder, pv, LLVM.PointerType(SPT0, pv.value_type.addrspace))
                        pv = addrspacecast!(builder, pv, LLVM.PointerType(SPT0, Derived))
                        pv = load!(builder, SPT0, pv)
                        if !(compare isa Nothing)
                            is_inactive = icmp!(builder, LLVM.IntPredicate.EQ, pv0, compare)
                            pv = select!(builder, is_inactive, LLVM.null(SPT0), pv)
                        end
                        ival = insert_value!(builder, ival, pv, idx - 1)
                    end
                    eval = ival
                end

            end

            eval = fixup_abi(returnNum + 1, eval)
            ptr = inbounds_gep!(
                builder,
                jltype,
                sret,
                [
                    LLVM.ConstantInt(LLVM.IntType(64), 0),
                    LLVM.ConstantInt(LLVM.IntType(32), returnNum),
                ],
                "fwd_wrap_sret_gep_$returnNum"
            )
            ptr = pointercast!(builder, ptr, LLVM.PointerType(eval.value_type), "fwd_wrap_sret_cast_$returnNum")
    	    extract_struct_into!(builder, ptr, eval, "fwd_wrap_sret_extract_$returnNum")
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
                    ptr = inbounds_gep!(
                        builder,
                        jltype,
                        sret,
                        [
                            LLVM.ConstantInt(LLVM.IntType(64), 0),
                            LLVM.ConstantInt(
                                LLVM.IntType(32),
                                length(jltype.elements) - 1,
                            ),
                        ],
                        "revcombined_wrap_sret_gep_$returnNum"
                    )
	    	    extract_struct_into!(
                        builder,
                        ptr,
                        eval,
                        "revcombined_wrap_sret_extract_$returnNum"
                    )
                    returnNum += 1
                end
            end
        end
        for (i, T) in enumerate(TT.parameters[2:end])
            if T <: Active
                T′ = eltype(T)
                isboxed = GPUCompiler.deserves_argbox(T′)
                if !isboxed
                    eval = extract_value!(builder, val, returnNum)
                    ptr = inbounds_gep!(
                        builder,
                        jltype,
                        sret,
                        [
                            LLVM.ConstantInt(LLVM.IntType(64), 0),
                            LLVM.ConstantInt(LLVM.IntType(32), 0),
                            LLVM.ConstantInt(LLVM.IntType(32), activeNum),
                        ],
                        "revcombined_wrap_sret_gep_active_$(i)_$(T′)"
                    )
	    	    extract_struct_into!(
                        builder,
                        ptr,
                        eval,
                        "revcombined_wrap_sret_extract_active_$(i)_$(T′)"
                    )
                    returnNum += 1
                end
                activeNum += 1
            end
        end
        @assert (returnNum - activeNum) + (activeNum != 0 ? 1 : 0) == numLLVMReturns
    end

    if returnRoots
       move_sret_tofrom_roots!(builder, jltype, sret, root_ty, pointercast!(builder, rootRet, LLVM.PointerType(T_prjlvalue)), SRetPointerToRootPointer)
    end
    if T_ret != T_void
        ret!(builder, load!(builder, T_ret, sret))
    else
        ret!(builder)
    end

    reinsert_gcmarker!(llvm_f)
    verifier_msg = verification_error(llvm_f)
    if verifier_msg !== nothing
        msg = sprint() do io
            println(io, string(mod))
            println(io, verifier_msg)
            println(io, string(llvm_f))
            println(
                io,
                "TT=",
                TT
            )
            println(io, "Broken create_abi_wrapper function")
        end
        throw(LLVM.LLVMException(msg))
    end

    return llvm_f
end

function fixup_metadata!(f::LLVM.Function)
    for param in f.parameters
        if isa(param.value_type, LLVM.PointerType)
            # collect all uses of the pointer
            worklist = Vector{LLVM.Instruction}(collect(param.users))
            while !isempty(worklist)
                value = popfirst!(worklist)

                # remove the invariant.load attribute
                md = value.metadata
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
                    append!(worklist, collect(value.users))
                end

                # IMPORTANT NOTE: if we ever want to inline functions at the LLVM level,
                # we need to recurse into call instructions here, and strip metadata from
                # called functions (see CUDAnative.jl#238).
            end
        end
    end
end

@enum(SRetRootMovement,
    SRetPointerToRootPointer = 0,
    SRetValueToRootPointer = 1,
    RootPointerToSRetValue = 2,
    RootPointerToSRetPointer = 3,
    NullifySRetValue = 4,
    RootAndSRetPointerToValue = 5,
    ValueToSRetAndRootPointers = 6,
   )

function to_llvm(lst::Vector{Cuint})
    vals = LLVM.Value[]
    push!(vals, LLVM.ConstantInt(LLVM.IntType(64), 0))
    for i in lst
       push!(vals, LLVM.ConstantInt(LLVM.IntType(32), i))
    end
    return vals
end

function initialize_roots_to_null!(builder::LLVM.IRBuilder, al::LLVM.Value, count::Int)
    T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    for i in 1:count
        gep = inbounds_gep!(builder, T_prjlvalue, al, [LLVM.ConstantInt(LLVM.IntType(sizeof(Int)*8), i-1)])
        store!(builder, LLVM.null(T_prjlvalue), gep)
    end
end

function create_rooted_array(builder::LLVM.IRBuilder, count::Int, name::String="")
    T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    count_val = LLVM.ConstantInt(LLVM.IntType(sizeof(Int)*8), count)
    al = array_alloca!(builder, T_prjlvalue, count_val, name)
    initialize_roots_to_null!(builder, al, count)
    return al
end

function create_rooted_array(builder::LLVM.IRBuilder, array_ty::LLVM.ArrayType, name::String="")
    T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    @assert array_ty.element_type == T_prjlvalue "create_rooted_array: ArrayType element type must be T_prjlvalue"
    return create_rooted_array(builder, array_ty.length, name)
end
    
function move_sret_tofrom_roots!(builder::LLVM.IRBuilder, jltype::LLVM.LLVMType, sret::LLVM.Value, root_ty::LLVM.LLVMType, rootRet::Union{LLVM.Value, Nothing}, direction::SRetRootMovement; must_cache::Bool = false, dst::Union{LLVM.Value, Nothing} = nothing)
        # For `ValueToSRetAndRootPointers`, `sret` is the value and `dst` the buffer.
        @assert (dst !== nothing) == (direction == ValueToSRetAndRootPointers)
        count = 0
        todo = Tuple{Vector{Cuint},LLVM.LLVMType}[(
	    Cuint[],
            jltype,
        )]

	extracted = LLVM.Value[]

	val = sret
	if direction == RootAndSRetPointerToValue
	    val = LLVM.UndefValue(jltype)
	end

	# TODO check that we perform this in the same order that extraction happens within julia
	# aka bfs/etc
        while length(todo) != 0
            path, ty = popfirst!(todo)
            if !any_jltypes(ty) && direction != RootAndSRetPointerToValue && direction != ValueToSRetAndRootPointers
                continue
            end

            if isa(ty, LLVM.PointerType) && any_jltypes(ty)

        		if direction == SRetPointerToRootPointer || direction == SRetValueToRootPointer || direction == RootPointerToSRetPointer || direction == RootPointerToSRetValue || direction == RootAndSRetPointerToValue || direction == ValueToSRetAndRootPointers
                          T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
                          T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
                          loc = inbounds_gep!(
                              builder,
                              T_prjlvalue,
                              rootRet,
        		      [LLVM.ConstantInt(LLVM.IntType(sizeof(Int)*8), count)],
        		     )
        		end
                        
        		if direction == SRetPointerToRootPointer
        		    outloc = inbounds_gep!(builder, jltype, sret, to_llvm(path))
        		    outloc = load!(builder, ty, outloc)
			    if must_cache
		                API.SetMustCache!(outloc)
			    end
                            store!(builder, outloc, loc)
        		elseif direction == SRetValueToRootPointer || direction == ValueToSRetAndRootPointers
                outloc = extract_value!(builder, sret, path)
                            store!(builder, outloc, loc)
        		elseif direction == RootPointerToSRetValue || direction == RootAndSRetPointerToValue
        		    loc = load!(builder, ty, loc)
			    if must_cache
		                API.SetMustCache!(loc)
			    end
                val = insert_value!(builder, val, loc, path)
			elseif direction == NullifySRetValue
			    loc = unsafe_to_llvm(builder, nothing)
                val = insert_value!(builder, val, loc, path)
        		elseif direction == RootPointerToSRetPointer
        		    outloc = inbounds_gep!(builder, jltype, sret, to_llvm(path))
        		    loc = load!(builder, ty, loc)
        		    push!(extracted, loc)
                            store!(builder, loc, outloc)
        		else
        		    @assert false "Unhandled direction"
        		end
                        
        		count += 1
                continue
            end
            if isa(ty, LLVM.ArrayType)
            for i in reverse(1:ty.length)
                    npath = copy(path)
		    push!(npath, i - 1)
                pushfirst!(todo, (npath, ty.element_type))
                end
                continue
            end
            if isa(ty, LLVM.VectorType)
            for i in reverse(1:ty.length)
                    npath = copy(path)
		    push!(npath, i - 1)
                pushfirst!(todo, (npath, ty.element_type))
                end
                continue
            end
            if isa(ty, LLVM.StructType)
            for (i, t) in reverse(collect(enumerate(ty.elements)))
                        npath = copy(path)
			push!(npath, i - 1)
                        pushfirst!(todo, (npath, t))
                end
                continue
            end
        
	    if direction == RootAndSRetPointerToValue
		    outloc = inbounds_gep!(builder, jltype, sret, to_llvm(path))
		    outloc = load!(builder, ty, outloc)
		    if must_cache
			API.SetMustCache!(outloc)
		    end
            val = insert_value!(builder, val, outloc, path)
	    elseif direction == ValueToSRetAndRootPointers
		    outloc = inbounds_gep!(builder, jltype, dst, to_llvm(path))
            store!(builder, extract_value!(builder, sret, path), outloc)
	    end
        end

	if direction == RootPointerToSRetPointer	        
	    obj = get_base_and_offset(sret)[1]
	    @assert length(extracted) > 0
	    emit_writebarrier!(builder, LLVM.Value[obj, extracted...])
	end
        tracked = CountTrackedPointers(jltype)
        @assert count == tracked.count
	return val
end

"""
    nullify_rooted_values!(builder, sret)

Return the value `sret` with every GC-tracked field replaced by a null reference.

Used where only the inline data of a value is wanted, and the tracked fields are
either held elsewhere (in a `returnRoots` array, see [`extract_roots_from_value!`](@ref))
or known not to be needed, so that leaving the original pointers in place would
root objects that must not be kept alive.
"""
function nullify_rooted_values!(builder::LLVM.IRBuilder, sret::LLVM.Value)
    jltype = sret.value_type
   tracked = CountTrackedPointers(jltype)
   @assert tracked.count > 0
   @assert !tracked.all
   root_ty = convert(LLVMType, AnyArray(Int(tracked.count)))
   move_sret_tofrom_roots!(builder, jltype, sret, root_ty, nothing, NullifySRetValue)
end

"""
    recombine_value!(builder, sret, roots; must_cache=false)

Rebuild a whole return value from the two halves the `sret`/`returnRoots` calling
convention splits it into.

A callee returning a type with both GC-tracked and inline fields writes the tracked
pointers into the `returnRoots` array and the remaining data into the `sret` buffer,
leaving the tracked slots of `sret` undefined. `sret` here is the *value* already
loaded out of that buffer and `roots` the pointer to the root array; the tracked
fields are loaded from `roots` and inserted back into their slots, and the completed
value is returned. This is the inverse of [`extract_roots_from_value!`](@ref); see
[`recombine_value_ptr!`](@ref) for the variant that takes `sret` as a pointer.

`must_cache` marks the loads from `roots` as must-cache, for a caller that needs the
recombined value to survive into the reverse pass.
"""
function recombine_value!(builder::LLVM.IRBuilder, sret::LLVM.Value, roots::LLVM.Value; must_cache::Bool=false)::LLVM.Value
    jltype = sret.value_type
   tracked = CountTrackedPointers(jltype)
   @assert tracked.count > 0
   @assert !tracked.all "Not tracked.all, jltype ($(string(jltype)))"
   root_ty = convert(LLVMType, AnyArray(Int(tracked.count)))
   move_sret_tofrom_roots!(builder, jltype, sret, root_ty, roots, RootPointerToSRetValue; must_cache)
end

"""
    roots_follow(args, ai, removedRoots) -> Bool

Whether the classified argument after `args[ai]` carries the inline roots of
`args[ai]` and is folded back into it (`removedRoots`), so that `args[ai]` is
handled through its data pointer rather than loaded whole.
"""
function roots_follow(args, ai::Int, removedRoots)::Bool
    ai < length(args) || return false
    nxt = args[ai+1]
    return nxt.rooted_typ !== nothing && nxt.rooted_arg_i == args[ai].arg_i && nxt.arg_i in removedRoots
end

"""
    split_value_into!(builder, val, sret, roots)

Store the value `val` through the `sret`/`returnRoots` convention: its GC-tracked
fields into the `roots` array and every other field into the `sret` buffer, leaving
the tracked slots of the buffer alone, as a caller reads them from `roots` only.
The inverse of [`recombine_value_ptr!`](@ref).
"""
function split_value_into!(builder::LLVM.IRBuilder, val::LLVM.Value, sret::LLVM.Value, roots::LLVM.Value)
    jltype = val.value_type
   tracked = CountTrackedPointers(jltype)
   @assert tracked.count > 0
   @assert !tracked.all "Not tracked.all, jltype ($(string(jltype)))"
   root_ty = convert(LLVMType, AnyArray(Int(tracked.count)))
   move_sret_tofrom_roots!(builder, jltype, val, root_ty, roots, ValueToSRetAndRootPointers; dst=sret)
   return nothing
end

"""
    recombine_value_ptr!(builder, jltype, sret, roots; must_cache=false)

Like [`recombine_value!`](@ref), but loads the inline half out of the `sret` buffer
rather than taking it as an already-loaded value.

Both `sret` and `roots` are pointers; a fresh `jltype` value is built by loading the
untracked fields from `sret` and the tracked ones from `roots`.
"""
function recombine_value_ptr!(builder::LLVM.IRBuilder, jltype::LLVM.LLVMType, sret::LLVM.Value, roots::LLVM.Value; must_cache::Bool=false)::LLVM.Value
   tracked = CountTrackedPointers(jltype)
   @assert tracked.count > 0
   @assert !tracked.all "Not tracked.all, jltype ($(string(jltype)))"
   root_ty = convert(LLVMType, AnyArray(Int(tracked.count)))
   move_sret_tofrom_roots!(builder, jltype, sret, root_ty, roots, RootAndSRetPointerToValue; must_cache)
end

"""
    extract_roots_from_value!(builder, sret, roots)

Store the GC-tracked fields of the value `sret` into the `returnRoots` array `roots`,
which must have room for `CountTrackedPointers(value_type(sret)).count` entries.

This is the split [`recombine_value!`](@ref) undoes: a caller that needs to hand a
value on through the `sret`/`returnRoots` convention writes the tracked pointers here
and the inline data into the `sret` buffer separately.
"""
function extract_roots_from_value!(builder::LLVM.IRBuilder, sret::LLVM.Value, roots::LLVM.Value)
    jltype = sret.value_type
   tracked = CountTrackedPointers(jltype)
   @assert tracked.count > 0
   @assert !tracked.all "Not tracked.all, jltype ($(string(jltype)))"
   root_ty = convert(LLVMType, AnyArray(Int(tracked.count)))
   move_sret_tofrom_roots!(builder, jltype, sret, root_ty, roots, SRetValueToRootPointer)
end

function copy_floats_into!(builder::LLVM.IRBuilder, jltype::LLVM.LLVMType, dst::LLVM.Value, src::LLVM.Value)
    count = 0
    todo = Tuple{Vector{Cuint},LLVM.LLVMType}[(
	    Cuint[],
        jltype,
    )]

	extracted = LLVM.Value[]

    while length(todo) != 0
            path, ty = popfirst!(todo)

            if isa(ty, LLVM.PointerType) || isa(ty, LLVM.IntegerType)
                continue
            end

            if isa(ty, LLVM.FloatingPointType)
		dstloc = inbounds_gep!(builder, jltype, dst, to_llvm(path), "dstloccf")
		srcloc = inbounds_gep!(builder, jltype, src, to_llvm(path), "srcloccf")
                val = load!(builder, ty, srcloc)
                st = store!(builder, val, dstloc)
                continue
            end

            if isa(ty, LLVM.ArrayType)
            for i in 1:ty.length
                    npath = copy(path)
                    push!(npath, i - 1)
                push!(todo, (npath, ty.element_type))
                end
                continue
            end

            if isa(ty, LLVM.VectorType)
            for i in 1:ty.length
                    npath = copy(path)
                    push!(npath, i - 1)
                push!(todo, (npath, ty.element_type))
                end
                continue
            end

            if isa(ty, LLVM.StructType)
            for (i, t) in enumerate(ty.elements)
                    npath = copy(path)
                    push!(npath, i - 1)
                    push!(todo, (npath, t))
                end
                continue
            end
        end

	return nothing
end

function extract_nonjlvalues_into!(builder::LLVM.IRBuilder, jltype::LLVM.LLVMType, dst::LLVM.Value, src::LLVM.Value)
    count = 0
    todo = Tuple{Vector{Cuint},LLVM.LLVMType}[(
	    Cuint[],
        jltype,
    )]

    extracted = LLVM.Value[]
	
    if dst.value_type.addrspace == 10
        PT2 = if LLVM.isopaque(dst.value_type)
	   LLVM.PointerType(11)
       else
            LLVM.PointerType(dst.value_type.element_type, 11)
       end
       dst = addrspacecast!(builder, PT2, dst)
    end

    while length(todo) != 0
            path, ty = popfirst!(todo)

            if isa(ty, LLVM.PointerType)
                if any_jltypes(ty)
			continue
		end
            end

            if isa(ty, LLVM.ArrayType) && any_jltypes(ty)
            for i in 1:ty.length
                    npath = copy(path)
                    push!(npath, i - 1)
                push!(todo, (npath, ty.element_type))
                end
                continue
            end

            if isa(ty, LLVM.VectorType) && any_jltypes(ty)
            for i in 1:ty.length
                    npath = copy(path)
                    push!(npath, i - 1)
                push!(todo, (npath, ty.element_type))
                end
                continue
            end

            if isa(ty, LLVM.StructType) && any_jltypes(ty)
            for (i, t) in enumerate(ty.elements)
                    npath = copy(path)
                    push!(npath, i - 1)
                    push!(todo, (npath, t))
                end
                continue
            end
		
	    dstloc = inbounds_gep!(builder, jltype, dst, to_llvm(path), "dstlocnjl")
        val = extract_value!(builder, src, path)
	    st = store!(builder, val, dstloc)
        end

	return nothing
end

function extract_struct_into!(builder::LLVM.IRBuilder, dst::LLVM.Value, src::LLVM.Value, name::String)
    count = 0
    jltype = src.value_type
    todo = Tuple{Vector{Cuint},LLVM.LLVMType}[(
	    Cuint[],
        jltype,
    )]

    extracted = LLVM.Value[]
	
    if dst.value_type.addrspace == 10
        PT2 = if LLVM.isopaque(dst.value_type)
	   LLVM.PointerType(11)
       else
            LLVM.PointerType(dst.value_type.element_type, 11)
       end
       dst = addrspacecast!(builder, PT2, dst)
    end

    while length(todo) != 0
            path, ty = popfirst!(todo)

            if isa(ty, LLVM.ArrayType) && any_jltypes(ty)
            for i in 1:ty.length
                    npath = copy(path)
                    push!(npath, i - 1)
                push!(todo, (npath, ty.element_type))
                end
                continue
            end

            if isa(ty, LLVM.VectorType) && any_jltypes(ty)
            for i in 1:ty.length
                    npath = copy(path)
                    push!(npath, i - 1)
                push!(todo, (npath, ty.element_type))
                end
                continue
            end

            if isa(ty, LLVM.StructType) && any_jltypes(ty)
            for (i, t) in enumerate(ty.elements)
                    npath = copy(path)
                    push!(npath, i - 1)
                    push!(todo, (npath, t))
                end
                continue
            end
		
	    dstloc = inbounds_gep!(builder, jltype, dst, to_llvm(path), "dstlocsi_$(name)_$(join(path, ","))")
        val = length(path) == 0 ? src : extract_value!(builder, src, path, "srclocei_$(name)_$(join(path, ","))")
	    st = store!(builder, val, dstloc)
        end

	return nothing
end

function copy_struct_into!(builder::LLVM.IRBuilder, jltype::LLVM.LLVMType, dst::LLVM.Value, src::LLVM.Value, copy_jlvalues::Bool)
    count = 0
    todo = Tuple{Vector{Cuint},LLVM.LLVMType}[(
        Cuint[],
        jltype,
    )]

    extracted = LLVM.Value[]
	
    if dst.value_type.addrspace == 10
        PT2 = if LLVM.isopaque(dst.value_type)
	   LLVM.PointerType(11)
       else
            LLVM.PointerType(dst.value_type.element_type, 11)
       end
       dst = addrspacecast!(builder, PT2, dst)
    end
    
    if src.value_type.addrspace == 10
        PT2 = if LLVM.isopaque(src.value_type)
	   LLVM.PointerType(11)
       else
            LLVM.PointerType(src.value_type.element_type, 11)
       end
       src = addrspacecast!(builder, src, PT2)
    end

    while length(todo) != 0
            path, ty = popfirst!(todo)
            
	    if isa(ty, LLVM.PointerType) && any_jltypes(ty) && !copy_jlvalues
		    continue
	    end

            if isa(ty, LLVM.ArrayType) && any_jltypes(ty)
            for i in 1:ty.length
                    npath = copy(path)
                    push!(npath, i - 1)
                push!(todo, (npath, ty.element_type))
                end
                continue
            end

            if isa(ty, LLVM.VectorType) && any_jltypes(ty)
            for i in 1:ty.length
                    npath = copy(path)
                    push!(npath, i - 1)
                push!(todo, (npath, ty.element_type))
                end
                continue
            end

            if isa(ty, LLVM.StructType) && any_jltypes(ty)
            for (i, t) in enumerate(ty.elements)
                    npath = copy(path)
                    push!(npath, i - 1)
                    push!(todo, (npath, t))
                end
                continue
            end
        
        dstloc = inbounds_gep!(builder, jltype, dst, to_llvm(path), "dstloccs")
        srcloc = inbounds_gep!(builder, jltype, src, to_llvm(path), "srcloccs")
        val = load!(builder, ty, srcloc)
        st = store!(builder, val, dstloc)
        end
    return nothing
end

# The operand of an `!enzyme_inactive` tag records why Enzyme.jl made the
# instruction inactive. Enzyme only checks that the tag is present.
#
# - `INACTIVE_GUARANTEED_CONST`: the Julia type of the value cannot hold
#   derivative data. This is true at every derivative order.
# - `INACTIVE_FROM_ACTIVITY`: `lower_convention` restates a `Const` annotation
#   of the autodiff call being compiled on the stack slot it creates for a
#   by-value argument or for the sret. This is only valid for the current compilation.
#   Enzyme leaves the tag on the derivative it emits, and
#   nested differentiation (see `autodiff_cache`) differentiates that
#   derivative again with its own activities. If the tag stays, the outer differentiation
#   treats every load of the slot as constant and silently zeroes the gradient
#   (EnzymeAD/Enzyme.jl#3617). `strip_activity_inactive_md!` removes these tags
#   after `enzyme!` has consumed them.
#
# A tag without an operand (for example one set by Enzyme itself) is treated
# like `INACTIVE_GUARANTEED_CONST` and stays.
const INACTIVE_GUARANTEED_CONST = "guaranteed_const"
const INACTIVE_FROM_ACTIVITY = "activity_derived"

inactive_md(reason::String) = MDNode(LLVM.Metadata[MDString(reason)])

function inactive_reason(md::LLVM.Metadata)
    isa(md, MDNode) || return nothing
    ops = md.operands
    (length(ops) == 1 && isa(ops[1], MDString)) || return nothing
    return convert(String, ops[1])
end

function strip_activity_inactive_md!(mod::LLVM.Module)
    # This also finds copies Enzyme made of a tag (for example onto the heap
    # allocation that replaces a stack slot, see `enzyme_fromstack`).
    for f in mod.functions, bb in f.blocks, inst in bb.instructions
        md = inst.metadata
        haskey(md, "enzyme_inactive") || continue
        if inactive_reason(md["enzyme_inactive"]) == INACTIVE_FROM_ACTIVITY
            delete!(md, "enzyme_inactive")
        end
    end
    return
end

# Modified from GPUCompiler/src/irgen.jl:365 lower_byval
function lower_convention(
    @nospecialize(functy::Type),
    mod::LLVM.Module,
    entry_f::LLVM.Function,
    @nospecialize(actualRetType::Type),
    @nospecialize(RetActivity::Type),
    @nospecialize(TT::Union{Type, Nothing}),
    run_enzyme::Bool,
    world::UInt,
    mi::Core.MethodInstance,
        enzyme_ctx::EnzymeContext,
)
    entry_ft = entry_f.function_type

    RT = entry_ft.return_type


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

    loweredReturn = RetActivity <: Active && !allocatedinline(actualRetType)
    if (RetActivity <: Active || RetActivity <: MixedDuplicated ||  RetActivity <: BatchMixedDuplicated) && (allocatedinline(actualRetType) != allocatedinline(eltype(RetActivity)))
	  @assert !allocatedinline(actualRetType)
	  loweredReturn = true
    end
 
    expected_RT = Nothing
    if loweredReturn
        @assert !sret
        @assert !returnRoots
        expected_RT = eltype(RetActivity)
        if expected_RT === Any
            expected_RT = Float64
        end
        RT = convert(LLVMType, expected_RT)
    end

    # TODO removed implications
    retRemoved, parmsRemoved = removed_ret_parms(entry_f)
    swiftself = has_swiftself(entry_f)
    @assert !swiftself "Swiftself attribute coming from differentiable context is not supported"
    prargs =
        classify_arguments(functy, entry_ft, sret, returnRoots, swiftself, parmsRemoved, mi, world)
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
        push!(wrapper_types, entry_f.parameters[1 + sret + returnRoots].value_type)
        push!(wrapper_attrs, LLVM.Attribute[EnumAttribute(:swiftself)])
    end

    boxedArgs = Set{Int}()
    loweredArgs = Set{Int}()
    raisedArgs = Set{Int}()
    removedRoots = Set{Int}()

    function is_mixed(idx::Int)
	if TT === nothing
	   return false
	end
	if idx > length(TT.parameters)
	   throw(AssertionError("TT=$TT, args=$args idx=$idx"))
	end
	return (
                   TT.parameters[idx] <: MixedDuplicated ||
                   TT.parameters[idx] <: BatchMixedDuplicated
               ) &&
               run_enzyme
    end

    for arg in args
        typ = arg.codegen.typ
	
	if arg.rooted_typ !== nothing

	   # There cannot exist a root arg if the original arg was boxed
	   @assert !GPUCompiler.deserves_argbox(arg.rooted_typ)
	   
	   # There only can exist a rooting if the original argument was a bits_ref
	   @assert arg.rooted_cc == GPUCompiler.BITS_REF
	   
	   # If the original arg exists and was lowered to be a bits_ref, we will destroy
	   # the extra rooted arg and recombine with the bits_ref
	   if (arg.arg_i - 1) in loweredArgs
	        push!(removedRoots, arg.arg_i)
		continue
	   end
	   
	   # If we are raising an argument to mixed, we will still destroy the extra rooted
	   # arg and recombine with the bits ref
	   if (arg.arg_i - 1) in boxedArgs
		@assert is_mixed(arg.arg_jl_i)
	        push!(removedRoots, arg.arg_i)
		continue
	   end

	   @assert false "Unhandled rooted arg condition"
	end

	if GPUCompiler.deserves_argbox(arg.typ)
            push!(boxedArgs, arg.arg_i)
            push!(wrapper_types, typ)
            push!(wrapper_attrs, LLVM.Attribute[])
        elseif arg.cc != GPUCompiler.BITS_REF
	    if is_mixed(arg.arg_jl_i)
                push!(boxedArgs, arg.arg_i)
                push!(raisedArgs, arg.arg_i)
                push!(wrapper_types, LLVM.PointerType(typ, Derived))
                push!(wrapper_attrs, LLVM.Attribute[EnumAttribute(:noalias)])
            else
                push!(wrapper_types, typ)
                push!(wrapper_attrs, LLVM.Attribute[])
            end
        else
            # bits ref, and not boxed
	    if is_mixed(arg.arg_jl_i)
                push!(boxedArgs, arg.arg_i)
                push!(wrapper_types, typ)
                push!(wrapper_attrs, LLVM.Attribute[EnumAttribute(:noalias)])
            else

                elty = convert(LLVMType, arg.typ)
                if !LLVM.isopaque(typ)
                    @assert elty == typ.element_type
                end

                push!(wrapper_types, elty)
                push!(wrapper_attrs, LLVM.Attribute[])
                push!(loweredArgs, arg.arg_i)
            end
        end
    end

    if length(loweredArgs) == 0 && length(raisedArgs) == 0 && length(removedRoots) == 0 && !sret && !sret_union && !loweredReturn
        return entry_f, returnRoots, boxedArgs, loweredArgs, removedRoots, actualRetType
    end

    wrapper_fn = entry_f.name
    entry_f.name = safe_name(wrapper_fn * ".inner")
    wrapper_ft = LLVM.FunctionType(RT, wrapper_types)
    wrapper_f = LLVM.Function(mod, entry_f.name, wrapper_ft)
    wrapper_f.callconv = entry_f.callconv
    sfn = entry_f.subprogram
    if sfn !== nothing
        wrapper_f.subprogram = sfn
    end

    hasReturnsTwice = haskey(entry_f.function_attributes, :returns_twice)
    hasNoInline = haskey(entry_f.function_attributes, :noinline)
    if hasNoInline
        delete!(entry_f.function_attributes, :noinline)
    end
    push!(wrapper_f.function_attributes, EnumAttribute(:returns_twice))
    push!(entry_f.function_attributes, EnumAttribute(:returns_twice))
    for (i, v) in enumerate(wrapper_attrs)
        for attr in v
            push!(wrapper_f.parameter_attributes[i], attr)
        end
    end

    seen = TypeTreeTable()
    # emit IR performing the "conversions"
    let builder = IRBuilder()
        toErase = LLVM.CallInst[]
        for u in entry_f.uses
            ci = u.user
            if !isa(ci, LLVM.CallInst) || ci.called_operand != entry_f
                continue
            end
            @assert !sret_union
            ops = ci.arguments
            position!(builder, LLVM.before(ci))
            nops = LLVM.Value[]
            if swiftself
                push!(nops, ops[1+sret+returnRoots])
            end
            for (ai, arg) in enumerate(args)
                parm = ops[arg.codegen.i]
		if arg.arg_i in removedRoots
		    if arg.rooted_arg_i in loweredArgs
		        # `nops[end]` is the pointer to the data half of the argument
		        # (see `roots_follow` below); rebuild the value from it and
		        # the roots `parm` field by field, which never reads the
		        # tracked slots a Julia 1.13.1+ data buffer omits.
		        nops[end] = recombine_value_ptr!(builder, convert(LLVMType, arg.rooted_typ), nops[end], parm)
		    elseif arg.rooted_arg_i in raisedArgs
                jltype = convert(LLVMType, arg.rooted_typ)
                tracked = CountTrackedPointers(jltype)
                @assert tracked.count > 0
                @assert !tracked.all
                root_ty = convert(LLVMType, AnyArray(Int(tracked.count)))
                move_sret_tofrom_roots!(builder, jltype, nops[end], root_ty, parm, RootPointerToSRetPointer)
            else
                @assert false
		    end
		elseif (arg.arg_i) in removedRoots && (arg.rooted_arg_i in loweredArgs || arg)
		    continue
		elseif arg.arg_i in loweredArgs
		    if roots_follow(args, ai, removedRoots)
		        push!(nops, parm)
		    else
                        push!(nops, load!(builder, convert(LLVMType, arg.typ), parm))
		    end
                elseif arg.arg_i in raisedArgs
                    obj = emit_allocobj!(builder, arg.typ, "raisedArg")
                    bc = bitcast!(
                        builder,
                        obj,
                        LLVM.PointerType(parm.value_type, obj.value_type.addrspace),
                    )
                    store!(builder, parm, bc)
		    if !(arg.arg_i in removedRoots)
                        emit_writebarrier!(builder, get_julia_inner_types(builder, obj, parm))
		    end
		    addr = addrspacecast!(
                        builder,
                        bc,
                        LLVM.PointerType(parm.value_type, Derived),
                    )
                    push!(nops, addr)
                else
                    push!(nops, parm)
                end
            end
            res = call!(builder, wrapper_f.function_type, wrapper_f, nops)
            res.callconv = wrapper_f.callconv
            if sret
                if !LLVM.isopaque(ops[1].value_type)
                    @assert res.value_type == ops[1].value_type.element_type
                end
                if returnRoots && VERSION >= v"1.12"
                    # Since Julia 1.12 the caller reads the tracked pointers from
                    # its `returnRoots` array and only the remaining data from the
                    # `sret` buffer; a plain store of the whole value would leave
                    # the array unset. Before 1.12 the buffer holds the whole value.
                    split_value_into!(builder, res, ops[1], ops[2])
                else
                    store!(builder, res, ops[1])
                end
            else
                LLVM.replace_uses!(ci, res)
            end
            push!(toErase, ci)
        end
        for e in toErase
            if !isempty(collect(e.uses))
                msg = sprint() do io
                    println(io, string(mod))
                    println(io, string(entry_f))
                    println(io, string(e))
                    println(io, "Use after deletion")
                end
                throw(AssertionError(msg))
            end
            erase!(e)
        end

        entry = BasicBlock(wrapper_f, "entry")
        position!(builder, LLVM.at_end(entry))
        if entry_f.subprogram !== nothing
            builder.debug_location = DILocation(0, 0, entry_f.subprogram)
        end

        wrapper_args = Vector{LLVM.Value}()

        sretPtr = nothing
	retRootPtr = nothing
        dl = string(entry_f.parent.datalayout)
        if sret
            if !in(0, parmsRemoved)
                sretPtr = alloca!(
                    builder,
                    sret_ty(entry_f, 1),
                    "innersret",
                )
                ctx = entry_f.context
                if RetActivity <: Const
                    sretPtr.metadata["enzyme_inactive"] = inactive_md(INACTIVE_FROM_ACTIVITY)
                end
        
                typeTree = copy(typetree_in_world(world, actualRetType, ctx, dl, seen))
                merge!(typeTree, TypeTree(API.DT_Pointer, ctx))
                only!(typeTree, -1)
                sretPtr.metadata["enzyme_type"] = to_md(typeTree, ctx)
                push!(wrapper_args, sretPtr)
            end
            if returnRoots && !in(1, parmsRemoved)
                retRootPtr = alloca!(
                    builder,
                    sret_ty(entry_f, 1+sret),
                    "innerreturnroots",
                )
                # retRootPtr = alloca!(builder, parameters(wrapper_f)[1])
                push!(wrapper_args, retRootPtr)
            end
        end
        if swiftself
            push!(wrapper_args, wrapper_f.parameters[1])
        end

        # perform argument conversions
	wrapper_idx = 1
        for arg in args
            parm = entry_f.parameters[arg.codegen.i]
	    if arg.arg_i in removedRoots
                wrapparm = wrapper_f.parameters[wrapper_idx - 1]
		root_ty = convert(LLVMType, arg.typ)
                ptr = create_rooted_array(builder, root_ty, parm.name * ".innerparm")
                if TT !== nothing && TT.parameters[arg.arg_jl_i] <: Const
                    ptr.metadata["enzyme_inactive"] = inactive_md(INACTIVE_FROM_ACTIVITY)
                end
                
                ctx = entry_f.context
                typeTree = copy(typetree_in_world(world, arg.typ, ctx, dl, seen))
                merge!(typeTree, TypeTree(API.DT_Pointer, ctx))
                only!(typeTree, -1)
                ptr.metadata["enzyme_type"] = to_md(typeTree, ctx)
	
		if arg.arg_i-1 in loweredArgs
		   extract_roots_from_value!(builder, wrapparm, ptr)
		else
	           @assert (arg.arg_i - 1) in boxedArgs
		   @assert is_mixed(arg.arg_jl_i) 
		   jltype = convert(LLVMType, arg.rooted_typ)
		   move_sret_tofrom_roots!(builder, jltype, wrapparm, root_ty, ptr, SRetPointerToRootPointer)
	        end

                push!(wrapper_args, ptr)
		continue
	    end

            wrapparm = wrapper_f.parameters[wrapper_idx]
	    wrapper_idx += 1
	    if arg.arg_i in loweredArgs
                # copy the argument value to a stack slot, and reference it.
                ty = parm.value_type
                if !isa(ty, LLVM.PointerType)
                    throw(
                        AssertionError(
                            "ty is not a LLVM.PointerType: entry_f = $(entry_f), args = $(args), parm = $(parm), ty = $(ty)",
                        ),
                    )
                end

                elty = convert(LLVMType, arg.typ)
                if !LLVM.isopaque(ty)
                    @assert elty == ty.element_type
                end

                elty_foralloca = if VERSION >= v"1.12" && arg.rooted_typ !== nothing
                    strip_tracked_pointers(elty)
                else
                    elty
                end

                ptr = alloca!(builder, elty_foralloca, parm.name * ".innerparm")
                if TT !== nothing && TT.parameters[arg.arg_jl_i] <: Const
                    ptr.metadata["enzyme_inactive"] = inactive_md(INACTIVE_FROM_ACTIVITY)
                end
                ctx = entry_f.context
        
                typeTree = copy(typetree_in_world(world, arg.typ, ctx, dl, seen))
                merge!(typeTree, TypeTree(API.DT_Pointer, ctx))
                only!(typeTree, -1)
                ptr.metadata["enzyme_type"] = to_md(typeTree, ctx)
                if ty.addrspace != 0
                    ptr = addrspacecast!(builder, ptr, ty)
                end
                @assert elty == wrapparm.value_type
                store!(builder, wrapparm, ptr)
                push!(wrapper_args, ptr)
                push!(
                    wrapper_f.parameter_attributes[wrapper_idx - 1],
                    StringAttribute(
                        "enzyme_type",
                        string(typetree_in_world(world, arg.typ, ctx, dl, seen)),
                    ),
                )
                push!(
                    wrapper_f.parameter_attributes[wrapper_idx - 1],
                    StringAttribute(
                        "enzymejl_parmtype",
                        string(convert(UInt, unsafe_to_pointer(arg.typ))),
                    ),
                )
                push!(
                    wrapper_f.parameter_attributes[wrapper_idx - 1],
                    StringAttribute(
                        "enzymejl_parmtype_ref",
                        string(UInt(GPUCompiler.BITS_VALUE)),
                    ),
                )
		if arg.rooted_typ !== nothing
                push!(
                        wrapper_f.parameter_attributes[wrapper_idx - 1],
                    StringAttribute(
                        "enzymejl_rooted_typ",
                        string(convert(UInt, unsafe_to_pointer(arg.rooted_typ))),
                    ),
                )
	end
            elseif arg.arg_i in raisedArgs
                wrapparm = load!(builder, convert(LLVMType, arg.typ), wrapparm)
                ctx = wrapparm.context
                push!(wrapper_args, wrapparm)
                typeTree = copy(typetree_in_world(world, arg.typ, ctx, dl, seen))
                merge!(typeTree, TypeTree(API.DT_Pointer, ctx))
                only!(typeTree, -1)
                push!(
                    wrapper_f.parameter_attributes[wrapper_idx - 1],
                    StringAttribute(
                        "enzyme_type",
                        string(typeTree),
                    ),
                )
                push!(
                    wrapper_f.parameter_attributes[wrapper_idx - 1],
                    StringAttribute(
                        "enzymejl_parmtype",
                        string(convert(UInt, unsafe_to_pointer(arg.typ))),
                    ),
                )
                push!(
                    wrapper_f.parameter_attributes[wrapper_idx - 1],
                    StringAttribute(
                        "enzymejl_parmtype_ref",
                        string(UInt(GPUCompiler.BITS_REF)),
                    ),
                )
		if arg.rooted_typ !== nothing
                push!(
                        wrapper_f.parameter_attributes[wrapper_idx - 1],
                    StringAttribute(
                        "enzymejl_rooted_typ",
			string(convert(UInt, unsafe_to_pointer(arg.rooted_typ)))
                    ),
                )
	end
            else
                push!(wrapper_args, wrapparm)
                for attr in collect(entry_f.parameter_attributes[arg.codegen.i])
                    push!(
                        wrapper_f.parameter_attributes[wrapper_idx - 1],
                        attr,
                    )
                end
            end
        end
        res = call!(builder, entry_f.function_type, entry_f, wrapper_args)

        if entry_f.subprogram !== nothing
            res.debug_location = DILocation(0, 0, entry_f.subprogram)
        end

        res.callconv = entry_f.callconv
        if swiftself
            attr = EnumAttribute(:swiftself)
            push!(res.argument_attributes[1 + sret + returnRoots], attr)
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
                function inner(@nospecialize(jlrettype::Type))
                    BB = BasicBlock(wrapper_f, "box_union")
                    position!(builder, LLVM.at_end(BB))

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
                                    LLVM.PointerType(llty, sretPtr.value_type.addrspace),
                                ),
                            )
                            store!(
                                builder,
                                ld,
                                bitcast!(
                                    builder,
                                    obj,
                                    LLVM.PointerType(llty, obj.value_type.addrspace),
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

                    push!(sw.cases, (LLVM.ConstantInt(scase.value_type, counter), BB))
                    counter += 1
                    return
                end
                for_each_uniontype_small(inner, actualRetType)

                position!(builder, LLVM.at_end(def))
                ret!(builder, extract_value!(builder, res, 0))

                ret_tt0 = typetree_in_world(world, actualRetType, ctx, dl, seen)

                push!(
                    wrapper_f.return_attributes,
                    StringAttribute(
                        "enzyme_type",
            			string(ret_tt0)
                    ),
                )
                push!(
                    wrapper_f.return_attributes,
                    StringAttribute(
                        "enzymejl_parmtype",
                        string(convert(UInt, unsafe_to_pointer(actualRetType))),
                    ),
                )
                if EmitTypeNames[]
                    push!(
                        wrapper_f.return_attributes,
                        StringAttribute(
                            "enzymejl_parmtype_str",
                            string(actualRetType),
                        ),
                    )
                end
                push!(
                    wrapper_f.return_attributes,
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
                    wrapper_f.return_attributes,
                    StringAttribute(
                        "enzyme_type",
                        string(typetree_in_world(world, actualRetType, ctx, dl, seen)),
                    ),
                )
                push!(
                    wrapper_f.return_attributes,
                    StringAttribute(
                        "enzymejl_parmtype",
                        string(convert(UInt, unsafe_to_pointer(actualRetType))),
                    ),
                )
                if EmitTypeNames[]
                    push!(
                        wrapper_f.return_attributes,
                        StringAttribute(
                            "enzymejl_parmtype_str",
                            string(actualRetType),
                        ),
                    )
                end
                push!(
                    wrapper_f.return_attributes,
                    StringAttribute(
                        "enzymejl_parmtype_ref",
                        string(UInt(GPUCompiler.BITS_REF)),
                    ),
                )
		res = load!(builder, RT, sretPtr)
		@static if VERSION >= v"1.12"
            	   if returnRoots
		     res = recombine_value!(builder, res, retRootPtr)
		   end
		end
		ret!(builder, res)
            end
        elseif entry_ft.return_type == LLVM.VoidType()
            ret!(builder)
        else
            ctx = wrapper_f.context

            if loweredReturn
                push!(
                    wrapper_f.return_attributes,
                    StringAttribute(
                        "enzyme_type",
                        string(typetree_in_world(world, eltype(RetActivity), ctx, dl, seen)),
                    ),
                )
                push!(
                    wrapper_f.return_attributes,
                    StringAttribute(
                        "enzymejl_parmtype",
                        string(convert(UInt, unsafe_to_pointer(expected_RT))),
                    ),
                )
                if EmitTypeNames[]
                    push!(
                        wrapper_f.return_attributes,
                        StringAttribute(
                            "enzymejl_parmtype_str",
                            string(expected_RT),
                        ),
                    )
                end
                push!(
                    wrapper_f.return_attributes,
                    StringAttribute(
                        "enzymejl_parmtype_ref",
                        string(UInt(GPUCompiler.BITS_VALUE)),
                    ),
                )
                ty = emit_jltypeof!(builder, res, enzyme_ctx)
                cmp = icmp!(builder, LLVM.IntPredicate.EQ, ty, unsafe_to_llvm(builder, expected_RT))
                cmpret = BasicBlock(wrapper_f, "ret")
                failure = BasicBlock(wrapper_f, "fail")
                br!(builder, cmp, cmpret, failure)

                position!(builder, LLVM.at_end(cmpret))
                res = bitcast!(builder, res, LLVM.PointerType(RT, res.value_type.addrspace))
                res = addrspacecast!(builder, res, LLVM.PointerType(RT, Derived))
                res = load!(builder, RT, res)
                ret!(builder, res)

                position!(builder, LLVM.at_end(failure))

                emit_error(builder, nothing, "Expected return type of primal to be "*string(expected_RT)*" but did not find a value of that type")
                unreachable!(builder)
            else
                llactualRetType = get_return_info(actualRetType)[1]
                ret_tt0 = typetree_in_world(world, actualRetType, ctx, dl, seen)
                ret_tt = if llactualRetType == Ptr{actualRetType}
                    typeTree = copy(ret_tt0)
                    merge!(typeTree, TypeTree(API.DT_Pointer, ctx))
                    only!(typeTree, -1)
                    typeTree
                else
                    ret_tt0
                end

                push!(
                    wrapper_f.return_attributes,
                    StringAttribute(
                        "enzyme_type",
                        string(ret_tt),
                    ),
                )
                push!(
                    wrapper_f.return_attributes,
                    StringAttribute(
                        "enzymejl_parmtype",
                        string(convert(UInt, unsafe_to_pointer(actualRetType))),
                    ),
                )
                push!(
                    wrapper_f.return_attributes,
                    StringAttribute(
                        "enzymejl_parmtype_ref",
                        string(UInt(GPUCompiler.BITS_REF)),
                    ),
                )
                ret!(builder, res)
            end
        end
        dispose(builder)
    end

    # early-inline the original entry function into the wrapper
    push!(entry_f.function_attributes, EnumAttribute(:alwaysinline))
    entry_f.linkage = LLVM.Linkage.Internal

    fixup_metadata!(entry_f)

    mi, rt = enzyme_custom_extract_mi(entry_f)
    attributes = wrapper_f.function_attributes
    push!(attributes, StringAttribute(LOWERED_CONVENTION_ATTR_KIND))
    push!(
        attributes,
        StringAttribute("enzymejl_mi", string(convert(UInt, pointer_from_objref(mi)))),
    )
    push!(
        attributes,
        StringAttribute("enzymejl_rt", string(convert(UInt, unsafe_to_pointer(rt)))),
    )
    if cached_has_easy_rule(Interpreter.simplify_kw(mi.specTypes), world)
        push!(attributes, LLVM.StringAttribute("enzyme_LocalReadOnlyOrThrow"))
        push!(attributes, LLVM.StringAttribute("enzyme_custom_full_attributes"))
    end
    for prev in collect(entry_f.function_attributes)
        if prev.kind == "enzyme_ta_norecur"
            push!(attributes, prev)
        end
        if prev.kind == "enzyme_parmremove"
            push!(attributes, prev)
        end
        if prev.kind == "enzyme_math"
            push!(attributes, prev)
        end
        if prev.kind == "enzyme_shouldrecompute"
            push!(attributes, prev)
        end
        if LLVM.version().major <= 15
            if prev.kind == :readonly
                push!(attributes, prev)
            end
            if prev.kind == :readnone
                push!(attributes, prev)
            end
            if prev.kind == :argmemonly
                push!(attributes, prev)
            end
            if prev.kind == :inaccessiblememonly
                push!(attributes, prev)
            end
        end
        if LLVM.version().major > 15
            if prev.kind == :memory
                old = MemoryEffect(attr.value)
                mem = MemoryEffect(
                    (set_writing(getModRef(old, ArgMem)) << getLocationPos(ArgMem)) |
                    (getModRef(old, InaccessibleMem) << getLocationPos(InaccessibleMem)) |
                    (getModRef(old, Other) << getLocationPos(Other)),
                )
                push!(attributes, EnumAttribute(:memory, mem.data))
            end
        end
        if prev.kind == :speculatable
            push!(attributes, prev)
        end
        if prev.kind == :nofree
            push!(attributes, prev)
        end
        if prev.kind == "enzyme_inactive"
            push!(attributes, prev)
        end
        if prev.kind == "enzyme_no_escaping_allocation"
            push!(attributes, prev)
        end
    end

    verifier_msg = verification_error(wrapper_f)
    if verifier_msg !== nothing
        msg = sprint() do io
            println(io, string(mod))
            println(io, verifier_msg)
            println(io, string(wrapper_f))
            println(
                io,
		"TT=$TT\n",
                "parmsRemoved=",
                parmsRemoved,
                "\nretRemoved=",
                retRemoved,
                "\nprargs=",
                prargs,
		"\nreturnRoots=",
		returnRoots,
		"\nboxedArgs=",
		boxedArgs,
		"\nloweredArgs=",
		loweredArgs,
		"\nraisedArgs=",
		raisedArgs,
		"\nremovedRoots=",
		removedRoots,
		"\nloweredReturn=",
		loweredReturn
            )
            println(io, "Broken lower convention")
        end
        throw(LLVM.LLVMException(msg))
    end



    remove_alwaysinline_roots!(mod)
    run!(AlwaysInlinerPass(), mod)
    if !hasReturnsTwice
        delete!(wrapper_f.function_attributes, :returns_twice)
    end
    if hasNoInline
        delete!(wrapper_f.function_attributes, :alwaysinline)
        push!(wrapper_f.function_attributes, EnumAttribute(:noinline))
    end

    # Fix phinodes used exclusively in extractvalue to be separate phi nodes
    phistofix = LLVM.PHIInst[]
    for bb in wrapper_f.blocks
        for inst in bb.instructions
            if isa(inst, LLVM.PHIInst)
                if !isa(inst.value_type, LLVM.StructType)
                    continue
                end
                legal = true
                for u in inst.uses
                    u = u.user
                    if !isa(u, LLVM.ExtractValueInst)
                        legal = false
                        break
                    end
                    if length(u.indices) != 1
                        legal = false
                        break
                    end
                    for op in u.operands[2:end]
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
        position!(nb, LLVM.before(p))
        st = p.value_type::LLVM.StructType
        phis = LLVM.PHIInst[]
        for (i, t) in enumerate(st.elements)
            np = phi!(nb, t, "wrap.fixphi")
            nvs = Tuple{LLVM.Value,LLVM.BasicBlock}[]
            for (v, b) in p.incoming
                prevbld = IRBuilder()
                position!(prevbld, LLVM.before(b.terminator))
                push!(nvs, (extract_value!(prevbld, v, i - 1), b))
            end
            append!(np.incoming, nvs)
            push!(phis, np)
        end

        torem = LLVM.Instruction[]
        for u in p.uses
            u = u.user
            @assert isa(u, LLVM.ExtractValueInst)
            @assert length(u.indices) == 1
            ind = u.indices[1]
            replace_uses!(u, phis[ind+1])
            push!(torem, u)
        end
        for u in torem
            erase!(u)
        end
        erase!(p)
    end

    LLVM.@dispose pb = PassBuilder() begin
        add!(pb, ModulePassManager()) do mpm
            # Kill the temporary staging function
	    add!(mpm, GlobalDCEPass())
	    add!(mpm, GlobalOptPass())
        end
        LLVM.run!(pb, mod)
    end

    if haskey(mod.globals, "llvm.used")
        eraseInst(mod, mod.globals["llvm.used"])
        for u in collect(entry_f.users)
            if isa(u, LLVM.GlobalVariable) &&
                    endswith(u.name, "_slot") &&
                    startswith(u.name, "julia")
                eraseInst(mod, u)
            end
        end
    end

    verifier_msg = verification_error(wrapper_f)
    if verifier_msg !== nothing
        msg = sprint() do io
            println(io, string(mod))
            println(io, verifier_msg)
            println(io, string(wrapper_f))
            println(io, "Broken function")
        end
        throw(LLVM.LLVMException(msg))
    end
    return wrapper_f, returnRoots, boxedArgs, loweredArgs, removedRoots, loweredReturn ? expected_RT : actualRetType
end

using Random
# returns arg, return
function no_type_setting(@nospecialize(specTypes::Type{<:Tuple}); world = nothing)
    # Even though the julia type here is ptr{int8}, the actual data can be something else
    if specTypes.parameters[1] == typeof(Random.XoshiroSimd.xoshiro_bulk_simd)
        return (true, false)
    end
    if specTypes.parameters[1] == typeof(Random.XoshiroSimd.xoshiro_bulk_nosimd)
        return (true, false)
    end
    if specTypes.parameters[1] == typeof(Base.hash)
        return (true, false)
    end
    return (false, false)
end

const DumpPreCheck = Ref(false)
const DumpPostCheck = Ref(false)
const DumpPreOpt = Ref(false)

"""
    link_split_existing!(mod::LLVM.Module, newmod::LLVM.Module)

Link `newmod` into `mod` like `LLVM.link!(mod, newmod)`, but set `LLVMInternalLinkage` on
any function defined in both modules before linking. This allows LLVM's linker to natively
internalize and resolve duplicate definitions without string comparisons or linker collisions.
"""
function link_split_existing!(mod::LLVM.Module, newmod::LLVM.Module)
    modfns = mod.functions
    newfns = newmod.functions
    has_collided = false
    for f in collect(newfns)
        isdeclaration(f) && continue
        fname = f.name
        haskey(modfns, fname) || continue
        isdeclaration(modfns[fname]) && continue
        f.linkage = LLVM.Linkage.Internal
        has_collided = true
    end
    LLVM.link!(mod, newmod)
    if has_collided
        LLVM.@dispose pb = LLVM.PassBuilder() begin
            mpm = LLVM.ModulePassManager()
            LLVM.add!(mpm, LLVM.MergeFunctionsPass())
            LLVM.add!(pb, mpm)
            LLVM.run!(pb, mod)
        end
    end
    return nothing
end

function GPUCompiler.compile_unhooked(output::Symbol, job::CompilerJob{<:EnzymeTarget})
    # Every compilation runs under its own context; a nested compilation gets
    # its own and hands the outer one back when it returns.
    mod, meta = @with ENZYME_CONTEXT => EnzymeContext(job.world) begin
        compile_unhooked_impl(output, job)
    end
    # A host derivative compiled on behalf of another job is linked into that job's module:
    # the module of the compilation in progress, for `autodiff_deferred` in the function it
    # differentiates, which takes up the values this compilation inserted. Without one, nothing
    # knows them where the module goes: write their addresses in.
    inserted = meta.value_table.inserted
    if !job.config.toplevel && !isempty(inserted)
        if isassigned(ENZYME_CONTEXT)
            Base.merge!(ENZYME_CONTEXT[].inserted_values, inserted)
        else
            bake_inserted_values!(mod, inserted)
        end
    end
    return mod, meta
end

"""
    memtransfer_truetype(world, ptr, sz, enzyme_ctx)

The `enzyme_truetype` metadata for a memcpy/memmove/memset of `sz` bytes at
`ptr`, or `nothing` if `ptr` cannot be traced back to a Julia object of known
concrete type. The type is read off the Julia layout, so it is not limited to the
offsets Enzyme's type analysis keeps.
"""
function memtransfer_truetype(world::UInt, @nospecialize(ptr::LLVM.Value), @nospecialize(sz::LLVM.Value), enzyme_ctx::EnzymeContext)
    base, offset = get_base_and_offset(ptr)
    legal, jTy, byref = abs_typeof(base, enzyme_ctx)
    legal || return nothing
    if byref == GPUCompiler.BITS_VALUE && jTy <: Ptr
        ET = eltype(jTy)
        if Base.isconcretetype(ET)
            sz_et = actual_size(ET)
            if sz_et > 0
                jTy = ET
                byref = GPUCompiler.MUT_REF
                offset = Base.mod(offset, sz_et)
            end
        end
    end
    Base.isconcretetype(jTy) || return nothing
    if jTy isa UnionAll ||
            jTy isa Union ||
            jTy == Union{} ||
            jTy === Tuple ||
            (is_concrete_tuple(jTy) && any(T2 isa Core.TypeofVararg for T2 in jTy.parameters))
        return nothing
    end
    size = Compiler.datatype_layoutsize(jTy)
    @assert offset >= 0
    if offset < size && isa(sz, LLVM.ConstantInt) && size - offset >= convert(Int, sz)
        @assert byref == GPUCompiler.BITS_REF || byref == GPUCompiler.MUT_REF
        return to_fullmd(world, jTy, offset, convert(Int, sz))
    elseif byref == GPUCompiler.BITS_VALUE && jTy <: Ptr && eltype(jTy) == Any
        # Todo generalize this
        return to_fullmd(world, jTy, 0, sizeof(Ptr{Cvoid}))
    end
    return nothing
end

function compile_unhooked_impl(output::Symbol, job::CompilerJob{<:EnzymeTarget})
    @assert output == :llvm
    
    config = job.config

    params = config.params

    enzyme_ctx = enzyme_context()

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

    primal_target = (job.config.target::EnzymeTarget).target
    primal_params = (job.config.params::EnzymeCompilerParams).params
    if primal_target isa GPUCompiler.NativeCompilerTarget
        if !(primal_params isa PrimalCompilerParams)
            # XXX: This means mode is not propagated and rules are not applied for GPU code.
            @safe_debug "NativeCompilerTarget without primal compiler params" primal_params
        end
    else
        # XXX: This means mode is not propagated and rules are not applied for GPU code.
    end
    primal_config = CompilerConfig(
        primal_target,
        primal_params;
        toplevel = config.toplevel,
        always_inline = config.always_inline,
        kernel = false,
        libraries = true,
        optimize = false,
        cleanup = false,
        only_entry = false,
        validate = false,
        entry_abi = :specfunc,
    )
    primal_job = CompilerJob(primal, primal_config, job.world)
    @safe_debug "Emit LLVM with" primal_job
    GPUCompiler.prepare_job!(primal_job)
    mod, meta = emit_unresolved_llvm(primal_job)
    # `emit_llvm` is not concretely inferred, so without this assertion every
    # subsequent use of `mod` (e.g. `LLVM.context(mod)`) is a dynamic dispatch
    # through jl_apply_generic, which forces boxing and GC-rooting across it.
    mod = mod::LLVM.Module
    record_julia_values!(enzyme_ctx, primal_job, meta)
    make_slots_symbolic!(mod, enzyme_ctx)
    edges = enzyme_ctx.edges

    primal_interp = GPUCompiler.get_interpreter(primal_job)
    prepare_llvm(primal_interp, mod, primal_job, meta, enzyme_ctx)
    for f in mod.functions
        permit_inlining!(f)
    end

    # A derivative linked into this module as a deferred job calls its rules
    # natively. Differentiating it again needs their bodies.
    materialize_native_invokes!(mode, mod)

    LLVM.@dispose pb = LLVM.PassBuilder() begin
        registerEnzymeAndPassPipeline!(pb)
        LLVM.add!(pb, LLVM.ModulePassManager()) do mpm
            LLVM.add!(mpm, PreserveNVVMPass())
        end
        LLVM.run!(pb, mod)
    end

    primalf = meta.entry
    if DumpPreCheck[]
        API.EnzymeDumpModuleRef(mod.ref)
    end
    interp = GPUCompiler.get_interpreter(job)
    check_ir(interp, job, mod, enzyme_ctx)
    if DumpPostCheck[]
        API.EnzymeDumpModuleRef(mod.ref)
    end
    # `check_ir` links in the derivatives that `enzyme_call` embeds. Their
    # natively called rules need bodies too.
    materialize_native_invokes!(mode, mod)

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
        # "trsm", Not actually implemented yet
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
        run!(InstCombinePass(), mod)
        toremove = String[]
        for f in mod.functions
            if !haskey(f.function_attributes, :alwaysinline)
                continue
            end
            if !haskey(f.function_attributes, :returns_twice)
                push!(f.function_attributes, EnumAttribute(:returns_twice))
                push!(toremove, f.name)
            end
            todo = LLVM.CallInst[]
            for u in f.uses
                ci = u.user
                if isa(ci, LLVM.CallInst) && ci.called_operand == f
                    push!(todo, ci)
                end
            end
            for ci in todo
                b = IRBuilder()
                position!(b, LLVM.before(ci))
                args = collect(LLVM.Value, ci.arguments)
                nc = call!(b, f.function_type, f, args)
                replace_uses!(ci, nc)
                erase!(ci)
            end
        end

        for fname in ("cblas_xerbla",)
            if haskey(mod.functions, fname)
                f = mod.functions[fname]
                if isempty(f.blocks)
                    entry = BasicBlock(f, "entry")
                    b = IRBuilder()
                    position!(b, LLVM.at_end(entry))
                    emit_error(b, nothing, "BLAS Error")
                    ret!(b)
                end
            end
        end

        remove_alwaysinline_roots!(mod)
        run!(AlwaysInlinerPass(), mod)
        for fname in toremove
            if haskey(mod.functions, fname)
                f = mod.functions[fname]
                delete!(f.function_attributes, :returns_twice)
            end
        end
        GPUCompiler.@safe_warn "Using fallback BLAS replacements for ($found), performance may be degraded"
	run!(GlobalOptPass(), mod)
    end

    custom, state = set_module_types!(interp, mod, primalf, job, edges, params.run_enzyme, mode, enzyme_ctx)

    primalf = state.primalf
    must_wrap = state.must_wrap
    actualRetType = state.actualRetType
    loweredArgs = state.loweredArgs
    boxedArgs = state.boxedArgs
    removedRoots = state.removedRoots

    @assert actualRetType !== nothing
    if params.run_enzyme
        @assert actualRetType != Union{}
    end

    if must_wrap
        llvmfn = primalf
        FT = llvmfn.function_type

        wrapper_f = LLVM.Function(mod, safe_name(llvmfn.name * "mustwrap"), FT)

        for idx in 1:length(collect(llvmfn.parameters))
            for attr in collect(llvmfn.parameter_attributes[idx])
                push!(wrapper_f.parameter_attributes[idx], attr)
            end
        end

        for attr in collect(llvmfn.function_attributes)
            push!(wrapper_f.function_attributes, attr)
        end

        for attr in collect(llvmfn.return_attributes)
            push!(wrapper_f.return_attributes, attr)
        end

        mi, rt = enzyme_custom_extract_mi(primalf)

        let builder = IRBuilder()
            entry = BasicBlock(wrapper_f, "entry")
            position!(builder, LLVM.at_end(entry))

            res = call!(
                builder,
                llvmfn.function_type,
                llvmfn,
                collect(wrapper_f.parameters),
            )

            sretkind = :sret
            for idx in 1:length(collect(llvmfn.parameters))
                for attr in collect(llvmfn.parameter_attributes[idx])
                    if attr.kind == sretkind
                        push!(res.argument_attributes[idx], attr)
                    end
                end
            end

            _, _, returnRoots0 = get_return_info(rt)
            returnRoots = returnRoots0 !== nothing
            if returnRoots
                attr = StringAttribute("enzymejl_returnRoots", string(length(eltype(returnRoots0).parameters[1])))
                push!(wrapper_f.parameter_attributes[2], attr)
                push!(res.argument_attributes[2], attr)
            end

            if FT.return_type == LLVM.VoidType()
                ret!(builder)
            else
                ret!(builder, res)
            end

            dispose(builder)
        end
        attributes = wrapper_f.function_attributes
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


    returnRoots = false

    if state.lowerConvention
        primalf, returnRoots, boxedArgs, loweredArgs, removedRoots, actualRetType = lower_convention(
            source_sig,
            mod,
            primalf,
            actualRetType,
            job.config.params.rt,
            TT,
            params.run_enzyme,
            job.world,
            job.source,
            enzyme_ctx,
        )
    end

    # if primal_job.config.target isa GPUCompiler.NativeCompilerTarget
    #     target_machine = JIT.get_tm()
    # else
    target_machine = GPUCompiler.llvm_machine(job.config.target)
    target_info = if isdefined(GPUCompiler, :llvm_targetinfo)
        GPUCompiler.llvm_targetinfo(job.config.target)
    else
        nothing
    end

    parallel = false
    process_module = false
    device_module = false
    if primal_target isa GPUCompiler.NativeCompilerTarget 
        parallel = Base.Threads.nthreads() > 1 
    else
        # All other targets are GPU targets
        parallel = true
        device_module = true
        
        if primal_target isa GPUCompiler.GCNCompilerTarget ||
           primal_target isa GPUCompiler.MetalCompilerTarget
            process_module = true
        end
    end

    # annotate
    replace_builtin_fptr!(mod, enzyme_ctx)
    annotate!(mod)
    for name in ("gpu_report_exception", "report_exception")
        if haskey(mod.functions, name)
            exc = mod.functions[name]
            if !isempty(exc.blocks)
                exc.linkage = LLVM.Linkage.External
            end
        end
    end

    if DumpPreOpt[]
        API.EnzymeDumpModuleRef(mod.ref)
    end

    # Run early pipeline
    optimize!(mod, target_machine, enzyme_ctx, target_info)

    if process_module
        GPUCompiler.optimize_module!(primal_job, mod)
    end

    # The Julia pipeline above folds a phi of loaded roots into a load through a
    # phi of the root arrays; undo that before Enzyme promotes the allocas among
    # them (see `unfold_root_phi_loads!`).
    for f in mod.functions
        isempty(f.blocks) && continue
        unfold_root_phi_loads!(f)
    end

    for name in ("gpu_report_exception", "report_exception")
        if haskey(mod.functions, name)
            exc = mod.functions[name]
            if !isempty(exc.blocks)
                exc.linkage = LLVM.Linkage.Internal
            end
        end
    end
    replace_nothing_loads!(mod)

    seen = TypeTreeTable()
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    DL = mod.datalayout
    dl = string(DL)
    ctx = mod.context
                        
    sretkind = :sret

    for f in mod.functions
        _, RT = enzyme_custom_extract_mi(f, false)
        valid_type = RT !== nothing && Base.isconcretetype(RT) && !(
            RT isa UnionAll ||
            RT isa Union ||
            RT == Union{} ||
            RT === Tuple ||
            (
                is_concrete_tuple(RT) &&
                any(T2 isa Core.TypeofVararg for T2 in RT.parameters)
            )
        )

        if valid_type
            size = Compiler.datatype_layoutsize(RT)
            md = to_fullmd(job.world, RT, 0, size)
            for bb in f.blocks
                term = bb.terminator
                if term !== nothing && term isa LLVM.RetInst && !isempty(term.operands)
                    cur = term.operands[1]
                    while cur isa LLVM.InsertValueInst
                        cur.metadata["enzyme_truetype"] = md
                        cur = cur.operands[1]
                    end
                end
            end
        end
    end

    for f in mod.functions, bb in f.blocks, inst in bb.instructions
        fn = isa(inst, LLVM.CallInst) ? inst.called_operand : nothing
       
        if !API.HasFromStack(inst) && isa(inst, LLVM.AllocaInst)
            calluse = LLVM.CallInst[]
            is_returnroots = false
            for u in inst.uses
                u = u.user
                if isa(u, LLVM.CallInst)
                    for i in 1:2
                        if i >= length(u.operands) || u.operands[i] != inst
                            continue
                        end
                        hassret = false
                        llvmfn = u.called_operand
                        # the callee can be called with more arguments than it has
                        # parameters (e.g., when its type differs from the call's)
                        if llvmfn isa LLVM.Function && i <= length(llvmfn.parameters)
                            for attr in collect(llvmfn.parameter_attributes[i])
                                if attr.kind == sretkind
                                    hassret = true
                                    break
                                end
                                if attr.kind == "enzymejl_returnRoots"
                                    hassret = true
                                    is_returnroots = true
                                    break
                                end
                            end
                        end
                        if hassret
                            push!(calluse, u)
                        end
                    end
                end
            end
            if length(calluse) > 0
                RTs = Union{Nothing, Type}[]
                for cu in calluse
                    _, RT = enzyme_custom_extract_mi(cu, false)
                    push!(RTs, RT)
                end
                @assert all(RTs[1] == RT for RT in RTs)
                RT = RTs[1]
                if RT !== nothing
                    llrt, sret, returnRoots = get_return_info(RT)
                    at = inst.allocated_type
                    if !(sret isa Nothing) && !is_sret_union(RT)
                        if is_returnroots
                            @assert returnRoots !== nothing
                            RT = equivalent_rooted_type(RT)
                        end
                        lRT = convert(LLVMType, RT)
                        if LLVM.storage_size(DL, lRT) == LLVM.storage_size(DL, at)
                            inst.metadata["enzymejl_allocart"] = MDNode(LLVM.Metadata[MDString(string(convert(UInt, unsafe_to_pointer(RT))))])
                            if EmitTypeNames[]
                                inst.metadata["enzymejl_allocart_name"] = MDNode(LLVM.Metadata[MDString(string(RT))])
                            end
                        end
                    end
                end
            end
        end

        if !API.HasFromStack(inst) &&
           ((isa(inst, LLVM.CallInst) &&
             (!isa(fn, LLVM.Function) || isempty(fn.blocks)) ) || isa(inst, LLVM.LoadInst) || isa(inst, LLVM.AllocaInst) || isa(inst, LLVM.ExtractValueInst))
            legal, source_typ, byref = abs_typeof(inst, enzyme_ctx)
            codegen_typ = inst.value_type
            if legal
                if codegen_typ isa LLVM.PointerType || codegen_typ isa LLVM.IntegerType
                else
                    if byref != GPUCompiler.BITS_VALUE
                        throw(AssertionError("Expected cc to be bits_value, found $byref, ty=$source_typ, cg_typ=$codegen_typ, inst=$(string(inst))\n\n$(string(fn))\n\n$fn\n\n$(string(inst.parent.parent))"))
		    end
                    source_typ
                end

                ec = typetree_in_world(job.world, source_typ, ctx, dl, seen)
                if byref == GPUCompiler.MUT_REF || byref == GPUCompiler.BITS_REF
                    ec = copy(ec)
                    merge!(ec, TypeTree(API.DT_Pointer, ctx))
                    only!(ec, -1)
                end
                if isa(inst, LLVM.CallInst)
                    push!(
                        inst.return_attributes, StringAttribute(
                            "enzyme_type",
                            string(ec),
                        )
                    )
                else
                    inst.metadata["enzyme_type"] = to_md(ec, ctx)
                    if EmitTypeNames[]
                        inst.metadata["enzymejl_source_type_$(source_typ)"] = MDNode(LLVM.Metadata[])
                    end
                    inst.metadata["enzymejl_byref_$(byref)"] = MDNode(LLVM.Metadata[])
                    if isa(inst, LLVM.LoadInst)
                        mark_load_dereferenceable!(inst, source_typ, byref)
                    end
            
@static if VERSION < v"1.11-"
else    
                        legal2, obj = absint(inst, enzyme_ctx)
		    obj = unbind(obj)
		    if legal2 && is_memory_instance(obj)
                            inst.metadata["nonnull"] = MDNode(LLVM.Metadata[])
                    end
end


                end
            elseif codegen_typ == T_prjlvalue
                if isa(inst, LLVM.CallInst)
                    push!(inst.return_attributes, StringAttribute("enzyme_type", "{[-1]:Pointer}"))
                else
                    inst.metadata["enzyme_type"] =
                        to_md(typetree_in_world(job.world, Ptr{Cvoid}, ctx, dl, seen), ctx)
                end
            end
        end

        if isa(inst, LLVM.CallInst)
            if !isa(fn, LLVM.Function)
                continue
            end
            if length(fn.blocks) != 0
                continue
            end

            intr = fn.intrinsic

            if intr == LLVM.Intrinsic("llvm.memcpy") ||
                    intr == LLVM.Intrinsic("llvm.memmove") ||
                    intr == LLVM.Intrinsic("llvm.memset")
                sz = inst.operands[3]
                md = memtransfer_truetype(job.world, inst.operands[1], sz, enzyme_ctx)
                if md === nothing && intr != LLVM.Intrinsic("llvm.memset")
                    # The destination is often a fresh stack slot with no Julia
                    # type of its own, being filled piecewise from an object that
                    # does have one; the source can then still tell us the type.
                    md = memtransfer_truetype(job.world, inst.operands[2], sz, enzyme_ctx)
                end
                if md !== nothing
                    inst.metadata["enzyme_truetype"] = md
                end
            end
        end

        ty = inst.value_type
        if ty == LLVM.VoidType()
            continue
        end

        legal, jTy, byref = abs_typeof(inst, enzyme_ctx, true)
        if !legal
            continue
        end

        if !guaranteed_const_nongen(jTy, job.world)
            continue
        end
        if isa(inst, LLVM.CallInst)
            push!(inst.return_attributes, StringAttribute("enzyme_inactive"))
        else
            inst.metadata["enzyme_inactive"] = inactive_md(INACTIVE_GUARANTEED_CONST)
        end
    end


    TapeType::Type = Cvoid

    if params.err_if_func_written
        FT = TT.parameters[1]
        Ty = eltype(FT)
        reg = active_reg(Ty, job.world)
        if reg == DupState || reg == MixedState
            swiftself = has_swiftself(primalf)
            todo = LLVM.Value[primalf.parameters[1 + swiftself]]
            done = Set{LLVM.Value}()
            doneInst = Set{LLVM.Instruction}()
            while length(todo) != 0
                cur = pop!(todo)
                if cur in done
                    continue
                end
                push!(done, cur)
                for u in cur.uses
                    user = u.user
                    if user in doneInst
                        continue
                    end
                    if user isa LLVM.RetInst
                        continue
                    end

                    if !mayWriteToMemory(user)
                        slegal, foundv, byref = abs_typeof(user, enzyme_ctx)
                        if slegal
                            reg2 = active_reg(foundv, job.world)
                            if reg2 == ActiveState || reg2 == AnyState
                                continue
                            end
                        end
                        push!(todo, user)
                        continue
                    end

                    if isa(user, LLVM.StoreInst)
                        # we are capturing the variable
                        if user.operands[1] == cur
                            base = user.operands[2]
                            while isa(base, LLVM.BitCastInst) ||
                                      isa(base, LLVM.AddrSpaceCastInst) ||
                                      isa(base, LLVM.GetElementPtrInst)
                                base = base.operands[1]
                            end
                            if isa(base, LLVM.AllocaInst)
                                push!(doneInst, user)
                                push!(todo, base)
                                continue
                            end
                        end
                        # we are storing into the variable
                        if user.operands[2] == cur
                            slegal, foundv, byref = abs_typeof(user.operands[1], enzyme_ctx)
                            if slegal
                                reg2 = active_reg(foundv, job.world)
                                if reg2 == AnyState
                                    continue
                                end
                            end
                        end
                    end

                    if isa(user, LLVM.CallInst)
                        called = user.called_operand
                        if isa(called, LLVM.Function)
                            intr = called.intrinsic
                            if intr == LLVM.Intrinsic("llvm.memset")
                                if cur != user.operands[1]
                                    continue
                                end
                            end

                            nm = called.name
                            if nm == "ijl_alloc_array_1d" ||
                               nm == "jl_alloc_array_1d" ||
                               nm == "ijl_alloc_array_2d" ||
                               nm == "jl_alloc_array_2d" ||
                               nm == "ijl_alloc_array_3d" ||
                               nm == "jl_alloc_array_3d" ||
                               nm == "ijl_new_array" ||
                               nm == "jl_new_array" ||
                               nm == "jl_alloc_genericmemory" ||
                               nm == "ijl_alloc_genericmemory" ||
			       nm == "jl_alloc_genericmemory_unchecked" ||
			       nm == "ijl_alloc_genericmemory_unchecked"
                                continue
                            end
                            if is_readonly(called)
                                slegal, foundv, byref = abs_typeof(user, enzyme_ctx)
                                if slegal
                                    reg2 = active_reg(foundv, job.world)
                                    if reg2 == ActiveState || reg2 == AnyState
                                        continue
                                    end
                                end
                                push!(todo, user)
                                continue
                            end
                            if !isempty(called.blocks) &&
                                    length(collect(called.uses)) == 1
                                for (parm, op) in
                                    zip(called.parameters, user.arguments)
                                    if op == cur
                                        push!(todo, parm)
                                    end
                                end
                                slegal, foundv, byref = abs_typeof(user, enzyme_ctx)
                                if slegal
                                    reg2 = active_reg(foundv, job.world)
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
                    position!(builder, LLVM.before(user))
                    resstr =
                        "Function argument passed to autodiff cannot be proven readonly.\nIf the the function argument cannot contain derivative data, instead call autodiff(Mode, Const(f), ...)\nSee https://enzyme.mit.edu/index.fcgi/julia/stable/faq/#Activity-of-temporary-storage for more information.\nThe potentially writing call is " *
                        string(user) *
                        ", using " *
                        string(cur)
                    slegal, foundv = absint(cur, enzyme_ctx)
                    if slegal
		    	foundv = unbind(foundv)
                        resstr *= "of type " * string(foundv)
                    end
                    emit_error(builder, user, resstr, EnzymeMutabilityException)
                end
            end
        end
    end

    if params.run_enzyme
        # Generate the adjoint
        erase_memcpy_from_undef!(mod)
        memcpy_alloca_to_loadstore(mod, job.world, enzyme_ctx)
        force_recompute!(mod)
        API.EnzymeDetectReadonlyOrThrow(mod)

        adjointf, augmented_primalf, TapeType = enzyme!(
            job,
	    interp,
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
	    removedRoots,
            enzyme_ctx,
        )
        # The activity hints that must not reach an outer differentiation of the result.
        strip_activity_inactive_md!(mod)

        # Link deferred modules
        for otherMod in enzyme_ctx.modules_to_link
            link_split_existing!(mod, otherMod)
        end
        empty!(enzyme_ctx.modules_to_link)
        toremove = String[]
        # Inline the wrapper
        for f in mod.functions
            for b in f.blocks
                term = b.terminator
                if isa(term, LLVM.UnreachableInst)
                    shouldemit = true
                    tmp = term
                    while true
                        tmp = tmp.prev
                        if tmp === nothing
                            break
                        end
                        if isa(tmp, LLVM.CallInst)
                            cf = tmp.called_operand
                            if isa(cf, LLVM.Function)
                                nm = cf.name
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
                        position!(b, LLVM.before(term))
                        emit_error(
                            b,
                            term,
                            "Enzyme: The original primal code hits this error condition, thus differentiating it does not make sense",
                        )
                    end
                end
            end
            if !haskey(f.function_attributes, :alwaysinline)
                continue
            end
            if !haskey(f.function_attributes, :returns_twice)
                push!(f.function_attributes, EnumAttribute(:returns_twice))
                push!(toremove, f.name)
            end       
        end
        remove_alwaysinline_roots!(mod)
        run!(AlwaysInlinerPass(), mod)
        for fname in toremove
            if haskey(mod.functions, fname)
                f = mod.functions[fname]
                delete!(f.function_attributes, :returns_twice)
            end
        end
    else
        adjointf = primalf
        augmented_primalf = nothing
    end

    LLVM.@dispose pb = LLVM.PassBuilder() begin
        registerEnzymeAndPassPipeline!(pb)
        LLVM.add!(pb, LLVM.ModulePassManager()) do mpm
            LLVM.add!(mpm, PreserveNVVMEndPass())
        end
        LLVM.run!(pb, mod)
    end

    if !(primal_target isa GPUCompiler.NativeCompilerTarget)
        mark_gpu_intrinsics!(primal_target, mod)
    end

    for (name, fnty) in state.fnsToInject
        for (T, JT, pf) in
            ((LLVM.DoubleType(), Float64, ""), (LLVM.FloatType(), Float32, "f"))
            fname = String(name) * pf
            if haskey(mod.functions, fname)
                funcspec = my_methodinstance(Mode == API.DEM_ForwardMode ? Forward : Reverse, fnty, Tuple{JT}, job.world)
                llvmf = nested_codegen!(mode, mod, funcspec)

                llvmf = llvmf.name

                # Link deferred modules generated by fnsToInject
                for otherMod in enzyme_ctx.modules_to_link
                    link_split_existing!(mod, otherMod)
                end
                empty!(enzyme_ctx.modules_to_link)

                llvmf = mod.functions[llvmf]

                push!(llvmf.function_attributes, StringAttribute("implements", fname))
            end
        end
    end

    API.EnzymeReplaceFunctionImplementation(mod)

    # Every deferred module is linked by now, so nothing declares the entry of
    # an imported cached thunk any more.
    internalize_imported_thunks!(mod)

    for (fname, lnk) in custom
        haskey(mod.functions, fname) || continue
        f = mod.functions[fname]
        f.linkage = lnk
        delete!(f.function_attributes, :noinline)
    end
    for fname in
        ["__enzyme_float", "__enzyme_double", "__enzyme_integer", "__enzyme_pointer"]
        haskey(mod.functions, fname) || continue
        f = mod.functions[fname]
        for u in f.uses
            st = u.user
            erase!(st)
        end
        eraseInst(mod, f)
    end

    adjointf.linkage = LLVM.Linkage.External
    adjointf_name = adjointf.name

    if augmented_primalf !== nothing
        augmented_primalf.linkage = LLVM.Linkage.External
        augmented_primalf_name = augmented_primalf.name
    end

    if !device_module
        # Don't restore pointers when we are doing GPU compilation. The
        # declarations of natively called rules stay symbolic until `_thunk`
        # compiles the module, so that an outer differentiation can still
        # recognize them (see `materialize_native_invokes!`).
        restore_lookups(mod; native_invokes = false)
    end

    if !(primal_target isa GPUCompiler.NativeCompilerTarget)
        reinsert_gcmarker!(adjointf)
        augmented_primalf !== nothing && reinsert_gcmarker!(augmented_primalf)
        post_optimize!(mod, target_machine, false; tti=target_info) #=machine=#
    end

    adjointf = mod.functions[adjointf_name]

    # API.EnzymeRemoveTrivialAtomicIncrements(adjointf)

    push!(adjointf.function_attributes, EnumAttribute(:alwaysinline))
    if augmented_primalf !== nothing
        augmented_primalf = mod.functions[augmented_primalf_name]
    end

    for fn in mod.functions
        fn == adjointf && continue
        augmented_primalf !== nothing && fn === augmented_primalf && continue
        isempty(fn.blocks) && continue
        fn.linkage = LLVM.Linkage.LinkerPrivate
    end
    


    use_primal = mode == API.DEM_ReverseModePrimal
    entry = use_primal ? augmented_primalf : adjointf
    # A derivative compiled on behalf of another job is linked into it here. A toplevel one is
    # linked by `_thunk`, once it has kept the symbolic bitcode. A host derivative's inserted
    # values are handed on with the module (`compile_unhooked`).
    @static if !HAS_GPUCOMPILER_2
        # GPUCompiler 1.x resolves nothing: resolve the slots, and write the addresses of Julia
        # values into device code, where nothing resolves their names.
        if !job.config.toplevel
            resolve_slots!(mod, julia_value_table(enzyme_ctx, mod))
            device_module && bake_julia_value_globals!(mod, enzyme_ctx.inserted_values)
        end
    end
    # What is still symbolic leaves the compilation with the module.
    value_table = julia_value_table(enzyme_ctx, mod)
    # GPUCompiler 2.x hands every Julia value a derivative compiled on behalf of another job
    # refers to on as a relocation, which the requesting job lowers with its own strategy:
    # `link_relocatable!` reads the records off this tuple.
    relocations = @static if HAS_GPUCOMPILER_2
        relocs = meta.relocations
        record_symbolic_slots!(mod, relocs, enzyme_ctx)
        # Nothing resolves the names of Julia values in device code: they become relocations.
        device_module && relocate_julia_value_globals!(mod, relocs, enzyme_ctx.inserted_values)
        relocs
    else
        nothing
    end
    return mod, (; adjointf, augmented_primalf, entry, compiled = meta.compiled, TapeType, edges, value_table, relocations)
end

# Compiler result
struct CompileResult{AT,PT}
    adjoint::AT
    primal::PT
    TapeType::Type
    edges::Vector{Any}
end

@inline (thunk::PrimalErrorThunk{PT,FA,RT,TT,Width,ReturnPrimal})(
    fn,
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
    fn,
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
    fn,
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
    fn,
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
    AdjointThunk{PT,FA,RT,TT,Width,TapeT},
    Val(Width),
    Val(false),
    TT,
    RT,
    fn,
    TapeT,
    args...,
) #=ReturnPrimal=#

@inline (thunk::AugmentedForwardThunk{PT,FA,RT,TT,Width,ReturnPrimal,TapeT})(
    fn,
    args...,
) where {PT,FA,Width,RT,TT,ReturnPrimal,TapeT} = enzyme_call(
    Val(false),
    thunk.primal,
    AugmentedForwardThunk{PT,FA,RT,TT,Width,ReturnPrimal,TapeT},
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
    AugmentedForwardThunk{PT,FA,RT,TT,Width,ReturnPrimal,TapeT},
    Val(Width),
    Val(ReturnPrimal),
    TT,
    RT,
    fn,
    TapeT,
    args...,
)

include("typeutils/recursive_add.jl")

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
@inline default_adjoint(::Type{T}, ::Val{1}) where {T} = default_adjoint(T)
@inline default_adjoint(::Type{T}, ::Val{W}) where {T,W} = ntuple(Returns(default_adjoint(T)), Val(W))

const DumpLLVMCall = Ref(false)

@generated function enzyme_call(
    ::Val{RawCall},
    fptr::PT,
    ::Type{CC},
    ::Val{width},
    ::Val{returnPrimal},
    tt::Type{T},
    rt::Type{RT},
    fn,
    ::Type{TapeType},
    args::Vararg{Any,N},
) where {RawCall,PT,T,RT,TapeType,N,CC,width,returnPrimal}
        FA = fn_type(CC)
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
            argtys = copy(argtypes)

            pushfirst!(argtys, FA)

            hint = "Arguments to the thunk should be the activities of the function and arguments"

            if is_adjoint
                if rettype <: Active ||
                   rettype <: MixedDuplicated ||
                   rettype <: BatchMixedDuplicated

                    push!(argtys,
                        if width == 1
                            eltype(rettype)
                        else
                            NTuple{width,eltype(rettype)}
                        end)
                    if width == 1
                        hint *=", then the seed of the active return"
                    else
                        hint *=", then an NTuple of width $width for the seeds of the batched active return"
                    end
                end

            end

            if needs_tape
                push!(argtys, TapeType)
                hint *=", then the tape from the forward pass"
            end

            truety = Tuple{argtys...}
            if length(argtys) != length(args) + 1
                return quote
                    throw(ThunkCallError($CC, $fn, $args, $truety, $hint))
                end
            end

            # An argument whose static type is wider than expected (for example
            # a tape stored as `Any` by a rule and then invoked through its
            # specialized signature) may still hold the right value. If every
            # argument is either narrower or wider than expected, assert the
            # expected types and dispatch again on the narrowed ones, so that
            # only a genuinely mismatched call is rejected.
            mismatched = false
            narrowable = true
            for (expected, found) in zip(argtys, (fn, args...))
                if !(found <: expected)
                    mismatched = true
                    if !(expected <: found)
                        narrowable = false
                    end
                end
            end
            if mismatched
                if !narrowable
                    return quote
                        throw(ThunkCallError($CC, $fn, $args, $truety, $hint))
                    end
                end
                narrowed = Expr[]
                for i in 1:length(args)
                    push!(narrowed, :(args[$i]::$(argtys[i + 1])))
                end
                return quote
                    Base.@_inline_meta
                    enzyme_call(
                        Val($RawCall), fptr, $CC, Val($width), Val($returnPrimal),
                        tt, rt, fn::$(argtys[1]), $TapeType, $(narrowed...),
                    )
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

	needs_rooting = false

        sret_types = Type[]  # Julia types of all returned variables
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
                F_ABI = F
                if width == 1
                    if (FA <: MixedDuplicated)
                        push!(types, Any)
                    else
                        push!(types, F_ABI)
                    end
                else
                    if F_ABI <: BatchMixedDuplicated
                        F_ABI = Base.RefValue{F_ABI}
                    end
                    F_ABI = NTuple{width, F_ABI}
                    isboxedvec = GPUCompiler.deserves_argbox(F_ABI)
                    if isboxedvec
                        push!(types, Any)
                    else
                        push!(types, F_ABI)
                    end
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

    ts_ctx = JuliaContext()
    ctx = context(ts_ctx)
    activate(ctx)
    (ir, fn, combinedReturn) = try

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
                @assert width == 1
                push!(sret_types, jlRT)
            elseif rettype <: MixedDuplicated
                @assert width == 1
                rty = if Base.isconcretetype(jlRT)
                    Base.RefValue{jlRT}
                else
                    (Base.RefValue{T} where T <: jlRT)
                end
                push!(sret_types, rty)
            elseif rettype <: BatchDuplicated || rettype <: BatchDuplicatedNoNeed
                @assert width == batch_size(rettype)
                push!(sret_types, AnonymousStruct(NTuple{width,jlRT}))
            elseif rettype <: BatchMixedDuplicated
                @assert width == batch_size(rettype)
                rty = if Base.isconcretetype(jlRT)
                    Base.RefValue{jlRT}
                else
                    (Base.RefValue{T} where T <: jlRT)
                end
                push!(sret_types, AnonymousStruct(NTuple{width,rty}))
            elseif CC <: AugmentedForwardThunk
                push!(sret_types, Nothing)
            elseif rettype <: Const
            else
                msg = sprint() do io
                    println(io, "rettype=", rettype)
                    println(io, "CC=", CC)
                end
                throw(AssertionError(msg))
            end
        end

        if returnPrimal && (CC <: ForwardModeThunk)
            push!(sret_types, jlRT)
        end

        # calls fptr
        llvmtys = LLVMType[]
        for x in types
            push!(llvmtys, convert(LLVMType, x; allow_boxed = true))
            arg_roots = inline_roots_type(x)
            if needs_rooting && arg_roots != 0
                push!(llvmtys, convert(LLVMType, AnyArray(3)))
            end
        end

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

        # With `InlineABI` the code of the thunk is linked in below, and Julia inlines it into
        # the caller. That code needs a pgcstack, so take the one of the caller as an argument.
        inline_abi = isghosttype(PT) || Core.Compiler.isconstType(PT)
        if inline_abi
            pushfirst!(llvmtys, convert(LLVMType, Ptr{Cvoid}))
        else
            pushfirst!(llvmtys, convert(LLVMType, PT))
        end

        T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

        T_ret = jltype
        # if returnRoots
        #     T_ret = T_prjlvalue
        # end
        mod = LLVM.Module("llvmcall")
        llvm_f = LLVM.Function(mod, "entry", LLVM.FunctionType(T_ret, llvmtys))
        push!(llvm_f.function_attributes, EnumAttribute(:alwaysinline))
        i64 = LLVM.IntType(64)

        builder = LLVM.IRBuilder()
        entry = BasicBlock(llvm_f, "entry")
        position!(builder, LLVM.at_end(entry))
        callparams = collect(LLVM.Value, llvm_f.parameters)

        if inline_abi
            # The pgcstack, which `use_gcstack_arg!` hands to the code of the thunk.
            pgcstack = popfirst!(callparams)
        else
            lfn = popfirst!(callparams)
        end

        if returnRoots
            tracked = CountTrackedPointers(jltype)
            pushfirst!(
                callparams,
                alloca!(builder, LLVM.ArrayType(T_prjlvalue, tracked.count), "enzyme_call.return_roots")
            )
	    jltype_foralloca = if VERSION >= v"1.12"
	       strip_tracked_pointers(jltype)
	    else
	       jltype
	    end
            pushfirst!(callparams, alloca!(builder, jltype_foralloca, "enzyme_call.sret"))
        end

        if needs_tape && !(isghostty(TapeType) || Core.Compiler.isconstType(TapeType))
            tape = callparams[end]
            if TapeType <: EnzymeTapeToLoad
                llty = Compiler.from_tape_type(eltype(TapeType))
	        
		        arg_roots = inline_roots_type(llty)
                if needs_rooting && arg_roots != 0
                    throw(AssertionError("Should check about rooted tape calling conv"))
                end

                tape = bitcast!(
                    builder,
                    tape,
                    LLVM.PointerType(llty, tape.value_type.addrspace),
                )
                tape = load!(builder, llty, tape)
                API.SetMustCache!(tape)
                callparams[end] = tape

            else
                llty = Compiler.from_tape_type(TapeType)
                arg_roots = inline_roots_type(llty)
                if needs_rooting && arg_roots != 0
                    tape = callparams[end-1]
                end
                if tape.value_type != llty
                    throw(AssertionError("MisMatched Tape type, expected $(string(tape.value_type)) found $(string(llty)) from $TapeType arg_roots=$arg_roots"))
		end
            end
        end

        if !inline_abi
            FT = LLVM.FunctionType(
                returnRoots ? T_void : T_ret,
                [x.value_type for x in callparams],
            )
            lfn = inttoptr!(builder, lfn, LLVM.PointerType(FT))
        else
            val_inner(::Type{Val{V}}) where {V} = V
            submod, subname = val_inner(PT)
            # TODO, consider optimization
            # However, julia will optimize after this, so no need
            submod = parse(LLVM.Module, String(submod))
            # Julia's llvmcall links this module as is into every caller, so any external
            # definition in it (e.g. the `ccalllib_*` library handle cache that Julia's
            # codegen emits for a `ccall`) would be defined once per caller, and the JIT
            # aborts with a duplicate symbol. Only this function uses the module, so all its
            # definitions can be local to the caller.
            for gv in Iterators.flatten((submod.globals, submod.functions))
                LLVM.isdeclaration(gv) && continue
                gv.name == String(subname) && continue
                if !(gv.linkage in (LLVM.API.LLVMInternalLinkage, LLVM.API.LLVMPrivateLinkage))
                    gv.linkage = LLVM.API.LLVMInternalLinkage
                end
            end
            LLVM.link!(mod, submod)
            lfn = mod.functions[String(subname)]
            # Only this function calls the thunk, so the inliner can drop it afterwards.
            lfn.linkage = LLVM.Linkage.Internal
            FT = lfn.function_type
        end

        r = call!(builder, FT, lfn, callparams)

        if returnRoots
            attr = TypeAttribute(:sret, jltype)
            push!(r.argument_attributes[1], attr)
            if !LLVM.isopaque(callparams[1].value_type)
                @assert callparams[1].value_type.element_type == jltype
            end
	    r = @static if VERSION >= v"1.12"
	        recombine_value_ptr!(builder, jltype, callparams[1], callparams[2])
	    else
                load!(builder, jltype, callparams[1])
	    end
        end

        if T_ret != T_void
            ret!(builder, r)
        else
            ret!(builder)
        end
        # Julia inlines this function into its caller. A `julia.get_pgcstack` call in it
        # would delay the push of the caller's GC frame on 1.13 (see `use_gcstack_arg!`).
        # Without `InlineABI` this function needs no pgcstack, since the thunk gets its own.
        if inline_abi
            use_gcstack_arg!(llvm_f, pgcstack)
        end

	Enzyme.Compiler.JIT.prepare!(mod)
	if DumpLLVMCall[]
	   API.EnzymeDumpModuleRef(mod.ref)
	end

        ir = string(mod)
        fn = llvm_f.name
        (ir, fn, combinedReturn)
    finally
        deactivate(ctx)
        dispose(ts_ctx)
    end

    @assert length(types) == length(ccexprs)


    if !(isghosttype(PT) || Core.Compiler.isconstType(PT))
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
                Tuple{Ptr{Cvoid}, $(types...)},
                current_pgcstack(),
                $(ccexprs...),
            )
        end
    end
end

##
# JIT
##

function _link(@nospecialize(job::CompilerJob{<:EnzymeTarget}), mod::LLVM.Module, edges::Vector{Any}, adjoint_name::String, @nospecialize(primal_name::Union{String, Nothing}), @nospecialize(TapeType), prepost::String)
    if job.config.params.ABI <: InlineABI
        return CompileResult(
            Val((Symbol(mod), Symbol(adjoint_name))),
            Val((Symbol(mod), Symbol(primal_name))),
            TapeType,
            edges
        )
    end

    # Now invoke the JIT
    jit_dylib = JIT.add!(mod)
    adjoint_addr = JIT.lookup(jit_dylib, adjoint_name)

    adjoint_ptr = pointer(adjoint_addr)
    if adjoint_ptr === C_NULL
        throw(
            GPUCompiler.InternalCompilerError(
                job,
                "Failed to compile Enzyme thunk, adjoint not found",
            ),
        )
    end
    if primal_name isa Nothing
        primal_ptr = C_NULL
    else
        primal_addr = JIT.lookup(jit_dylib, primal_name)
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

    return CompileResult(adjoint_ptr, primal_ptr, TapeType, edges)
end

const DumpPrePostOpt = Ref(false)
const DumpPostOpt = Ref(false)

# actual compilation
function _thunk(job, postopt::Bool = true)::Tuple{LLVM.Module, Vector{Any}, String, Union{String, Nothing}, Type, String, JuliaValueTable}
    config = CompilerConfig(job.config; optimize=false)
    job = CompilerJob(job.source, config, job.world)
    mod, meta = compile(:llvm, job)
    adjointf, augmented_primalf = meta.adjointf, meta.augmented_primalf
    value_table = meta.value_table


    adjoint_name = adjointf.name

    if augmented_primalf !== nothing
        primal_name = augmented_primalf.name
    else
        primal_name = nothing
    end

    LLVM.@dispose pb = PassBuilder() begin
        register!(pb, ReinsertGCMarkerPass())
        fpm = FunctionPassManager()
        add!(fpm, ReinsertGCMarkerPass())
        add!(pb, fpm)
        LLVM.run!(pb, mod)
    end

    # Run post optimization pipeline
    prepost = if postopt
        mstr = if job.config.params.ABI <: InlineABI
            ""
        else
            fixup_callconv!(mod, JIT.get_tm())
            for f in mod.functions
                for i in 1:length(f.parameters)
                    for a in collect(f.parameter_attributes[i])
                        if a.kind == "enzyme_sret"
                           API.EnzymeDumpValueRef(f)
                       end
                        @assert a.kind != "enzyme_sret"
                        @assert a.kind != "enzyme_sret_v"
                    end
                end
            end
            # Kept for nested differentiation (see autodiff_cache); bitcode
            # is far cheaper to write than textual IR and parses faster.
            String(convert(Vector{UInt8}, mod))
        end
        # The bitcode keeps its slots symbolic for a differentiation that imports it, which
        # takes `value_table` along; the code that runs gets the addresses, before it is
        # optimized.
        link_julia_values!(mod, meta)
        if job.config.params.ABI <: FFIABI || job.config.params.ABI <: NonGenABI
            if DumpPrePostOpt[]
                API.EnzymeDumpModuleRef(mod.ref)
            end
            post_optimize!(mod, JIT.get_tm(); callconv=false)
            if DumpPostOpt[]
                API.EnzymeDumpModuleRef(mod.ref)
            end
        else
            define_ntuple_type!(mod)
            propagate_returned!(mod)
            Compiler.JIT.prepare!(mod)
        end
        mstr
    else
        link_julia_values!(mod, meta)
        ""
    end
    # The module string above keeps the rule declarations symbolic for nested
    # differentiation; the compiled module binds them to their addresses.
    restore_native_invokes!(mod)
    return (mod, meta.edges, adjoint_name, primal_name, meta.TapeType, prepost, value_table)
end

const cache = Dict{UInt,CompileResult}()

"""
    CachedThunk

What `autodiff_cache` keeps of a thunk for a later compilation to import (see
[`import_cached_autodiff!`](@ref)): the name of the function to call (`entry`), the bitcode
of the module before post-optimization (`bitcode`), which refers to Julia values by name, and
those values (`value_table`, see [`JuliaValueTable`](@ref)).
"""
struct CachedThunk
    entry::String
    bitcode::String
    value_table::JuliaValueTable
end

# adjoint/primal pointer => the thunk's IR, for nested differentiation
const autodiff_cache = Dict{Ptr{Cvoid}, CachedThunk}()

const cache_lock = ReentrantLock()
@inline function cached_compilation(@nospecialize(job::CompilerJob))::CompileResult
    key = hash(job)

    # NOTE: no use of lock(::Function)/@lock/get! to keep stack traces clean
    lock(cache_lock)
    try
        obj = get(cache, key, nothing)
        if obj === nothing
            mod, edges, adjoint_name, primal_name, TapeType, prepost, value_table = _thunk(job)
            obj = _link(job, mod, edges, adjoint_name, primal_name, TapeType, prepost)
            if obj.adjoint isa Ptr{Nothing}
                autodiff_cache[obj.adjoint] = CachedThunk(adjoint_name, prepost, value_table)
            end
            if obj.primal isa Ptr{Nothing} && primal_name isa String
                autodiff_cache[obj.primal] = CachedThunk(primal_name, prepost, value_table)
            end
            cache[key] = obj
        end
        obj
    finally
        unlock(cache_lock)
    end
end

"""
    clear_caches!()

Empty every cache Enzyme keeps in a global, dropping what this session compiled, looked up
and rooted along with them.

Nearly all of it means something only to the session that filled it. A `CompileResult` holds
the address the JIT gave a thunk, `captured_constants` roots objects because their addresses
were written into that code, the rule and activity memos are keyed on world ages, and the
`jl_load_and_lookup` handles are ones this process opened. Anything outliving the session
must not carry them, which is why Enzyme's precompile workload ends with this call: what it
left behind would otherwise be serialized into Enzyme's package image and inherited, dead,
by every session that loads it.

This is meant for the end of precompilation and not for a live session. It hands back the
thunks the JIT compiled and unroots the objects their code refers to by address, so a thunk
still held anywhere is left pointing at objects that may now be collected.

Caches filled by `__init__` rather than by compiling are left alone: they are rebuilt per
session and so never reach an image.
"""
function clear_caches!()
    # Thunks, held as the addresses the JIT gave them, and the objects rooted because those
    # addresses were written into their code.
    empty!(cache)
    empty!(autodiff_cache)
    empty!(Enzyme.tape_cache)
    empty!(Enzyme.captured_constants)

    # Which rules apply, memoized against the world the methods were read in.
    empty!(FRULE_CACHE)
    empty!(RRULE_CACHE)
    empty!(INACTIVE_CACHE)
    empty!(EASY_RULE_CACHE)
    empty!(NOALIAS_CACHE)
    empty!(Interpreter.SigCache)
    Interpreter.LastFwdWorld[] = Base.IdSet{Type}()
    Interpreter.LastRevWorld[] = Base.IdSet{Type}()
    Interpreter.LastInaWorld[] = Base.IdSet{Type}()

    # Activity, and the world its `inactive_type` methods were last checked in. Left set, it
    # tells a later session its own worlds need no check.
    empty!(ActivityCache)
    empty!(ActivityMethodCache)
    ActivityWorldCache[] = 0

    # The library handles `ejlstr$` and `ejlptr$` symbols were resolved through.
    empty!(JIT.hnd_string_map)
    empty!(JIT.hnd_int_map)

    @static if VERSION < v"1.11.0-DEV.1552"
        # Inference results, kept by Enzyme itself on versions where Julia's own cache does
        # not hold them. The `CodeInstance`s carry `invoke` and `specptr` addresses.
        empty!(GLOBAL_FWD_CACHE.dict)
        empty!(GLOBAL_REV_CACHE.dict)
    end
    return nothing
end

"""
    instantiate_annotation(A, rt, width)

Fill in the free parameters of a (possibly partially applied) activity annotation
`A` with element type `rt` and batch width `width`.

The batch annotations take a second parameter carrying the batch width. Applying
only `A{rt}` to them leaves that parameter free, and a subsequent `A{rt}` binds the
*element type* to it, yielding an annotation whose `batch_size` is a type rather
than the width. Filling both explicitly keeps `batch_size(A) == width`, which the
shadow-return ABI in `create_abi_wrapper` and `enzyme_call` asserts.
"""
@inline function instantiate_annotation(
        @nospecialize(A::Type{<:Annotation}),
        @nospecialize(rt::Type),
        width::Int,
    )
    A isa UnionAll || return A
    return if A <: BatchDuplicated
        BatchDuplicated{rt, width}
    elseif A <: BatchDuplicatedNoNeed
        BatchDuplicatedNoNeed{rt, width}
    elseif A <: BatchDuplicatedFunc
        BatchDuplicatedFunc{rt, width}
    elseif A <: BatchMixedDuplicated
        BatchMixedDuplicated{rt, width}
    else
        A{rt}
    end
end

@inline function thunkbase(
    mi::Core.MethodInstance,
    World::Union{UInt, Nothing},
    @nospecialize(FA::Type{<:Annotation}),
    @nospecialize(A::Type{<:Annotation}),
    @nospecialize(TT::Type),
    Mode::API.CDerivativeMode,
    width::Int,
    @nospecialize(ModifiedBetween::(NTuple{N, Bool} where N)),
    ReturnPrimal::Bool,
    ShadowInit::Bool,
    @nospecialize(ABI::Type),
    ErrIfFuncWritten::Bool,
    RuntimeActivity::Bool,
    StrongZero::Bool,
    edges::Union{Nothing, Vector{Any}}
)
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
        StrongZero
    ) #=abiwrap=#
    tmp_job = if World isa Nothing
        jb = Compiler.CompilerJob(mi, CompilerConfig(target, params; kernel = false))
        check_activity_cache_invalidations(jb.world)
        jb
    else
        Compiler.CompilerJob(mi, CompilerConfig(target, params; kernel = false), World)
    end

    interp = GPUCompiler.get_interpreter(tmp_job)

    # TODO check compile return here, early
    rrt = return_type(interp, mi)

    run_enzyme = true

    A2 = if rrt == Union{}
        run_enzyme = false
        Const
    else
        A
    end

    if run_enzyme && !(A2 <: Const) && (World isa Nothing ? guaranteed_const(rrt) : guaranteed_const_nongen(rrt, World))
        estr = "Return type `$rrt` not marked Const, but type is guaranteed to be constant"
        return error(estr)
    end

    rt2 = if !run_enzyme
        Const{rrt}
    elseif A2 isa UnionAll
        instantiate_annotation(A2, rrt, width)
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
        StrongZero
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
    if edges !== nothing
        for e in compile_result.edges
            push!(edges, e)
        end
    end
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
    ::Val{StrongZero}
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
    StrongZero
}
    ts_ctx = JuliaContext()
    ctx = context(ts_ctx)
    activate(ctx)
    try
        return thunkbase(
            mi,
            nothing,
            FA,
            A,
            TT,
            Mode,
            width,
            ModifiedBetween,
            ReturnPrimal,
            ShadowInit,
            ABI,
            ErrIfFuncWritten,
            RuntimeActivity,
            StrongZero,
            nothing
        )
    finally
        deactivate(ctx)
        dispose(ts_ctx)
    end
end

function thunk end

function thunk_generator(world::UInt, source::Union{Method, LineNumberNode}, @nospecialize(FA::Type), @nospecialize(A::Type), @nospecialize(TT::Type), Mode::Enzyme.API.CDerivativeMode, Width::Int, @nospecialize(ModifiedBetween::(NTuple{N, Bool} where N)), ReturnPrimal::Bool, ShadowInit::Bool, @nospecialize(ABI::Type), ErrIfFuncWritten::Bool, RuntimeActivity::Bool, StrongZero::Bool, @nospecialize(self), @nospecialize(fakeworld), @nospecialize(fa::Type), @nospecialize(a::Type), @nospecialize(tt::Type), @nospecialize(mode::Type), @nospecialize(width::Type), @nospecialize(modifiedbetween::Type), @nospecialize(returnprimal::Type), @nospecialize(shadowinit::Type), @nospecialize(abi::Type), @nospecialize(erriffuncwritten::Type), @nospecialize(runtimeactivity::Type), @nospecialize(strongzero::Type))
    @nospecialize
    
    slotnames = Core.svec(Symbol("#self#"), 
                    :fakeworld, :fa, :a, :tt, :mode, :width,
                    :modifiedbetween, :returnprimal, :shadowinit,
                    :abi, :erriffuncwritten, :runtimeactivity, :strongzero)
    stub = Core.GeneratedFunctionStub(identity, slotnames, Core.svec())

    ft = eltype(FA)
    primal_tt = Tuple{map(eltype, TT.parameters)...}
    # look up the method match
    
    min_world = Ref{UInt}(typemin(UInt))
    max_world = Ref{UInt}(typemax(UInt))
    
    mi = my_methodinstance(Mode == API.DEM_ForwardMode ? Forward : Reverse, ft, primal_tt, world, min_world, max_world)
    
    mi === nothing && return stub(world, source, :(throw(MethodError($ft, $primal_tt, $world))))
 
    check_activity_cache_invalidations(world)

    edges = Any[]
    add_edge!(edges, mi)
    
    ts_ctx = JuliaContext()
    ctx = context(ts_ctx)
    activate(ctx)
    result = try
        thunkbase(
            mi,
            world,
            FA,
            A,
            TT,
            Mode,
            Width,
            ModifiedBetween,
            ReturnPrimal,
            ShadowInit,
            ABI,
            ErrIfFuncWritten,
            RuntimeActivity,
            StrongZero,
            edges
        )
    finally
        deactivate(ctx)
        dispose(ts_ctx)
    end

    code = Any[Core.Compiler.ReturnNode(result)]
    ci = create_fresh_codeinfo(thunk, source, world, slotnames, code)



    if Mode == API.DEM_ForwardMode
        fwd_sig = Tuple{typeof(EnzymeRules.forward), <:EnzymeRules.FwdConfig, <:Enzyme.EnzymeCore.Annotation, Type{<:Enzyme.EnzymeCore.Annotation},Vararg{Enzyme.EnzymeCore.Annotation}}
        add_edge!(edges, fwd_sig)
    else
        rev_sig = Tuple{typeof(EnzymeRules.augmented_primal), <:EnzymeRules.RevConfig, <:Enzyme.EnzymeCore.Annotation, Type{<:Enzyme.EnzymeCore.Annotation},Vararg{Enzyme.EnzymeCore.Annotation}}
        add_edge!(edges, rev_sig)
        
        rev_sig = Tuple{typeof(EnzymeRules.reverse), <:EnzymeRules.RevConfig, <:Enzyme.EnzymeCore.Annotation, Union{Type{<:Enzyme.EnzymeCore.Annotation}, Enzyme.EnzymeCore.Active}, Any, Vararg{Enzyme.EnzymeCore.Annotation}}
        add_edge!(edges, rev_sig)
    end
    
    for gen_sig in (
        Tuple{typeof(EnzymeRules.inactive), Vararg{Any}},
        Tuple{typeof(EnzymeRules.inactive_noinl), Vararg{Any}},
        Tuple{typeof(EnzymeRules.inactive_arg), Vararg{Any}},
        Tuple{typeof(EnzymeRules.inactive_kwarg), Vararg{Any}},
        Tuple{typeof(EnzymeRules.noalias), Vararg{Any}},
        Tuple{typeof(EnzymeRules.inactive_type), Type},
    )
        add_edge!(edges, gen_sig)
    end

    ci.edges = edges
    return ci
end

# The generated wrapper `thunk` is defined in src/late_generated.jl; see the note there.

import GPUCompiler: deferred_codegen_jobs

function deferred_id_codegen end

function deferred_id_generator(world::UInt, source::Union{Method, LineNumberNode}, @nospecialize(FA::Type), @nospecialize(A::Type), @nospecialize(TT::Type), Mode::Enzyme.API.CDerivativeMode, Width::Int, @nospecialize(ModifiedBetween::(NTuple{N, Bool} where N)), ReturnPrimal::Bool, ShadowInit::Bool, @nospecialize(ExpectedTapeType::Type), ErrIfFuncWritten::Bool, RuntimeActivity::Bool, StrongZero::Bool, @nospecialize(self), @nospecialize(fa::Type), @nospecialize(a::Type), @nospecialize(tt::Type), @nospecialize(mode::Type), @nospecialize(width::Type), @nospecialize(modifiedbetween::Type), @nospecialize(returnprimal::Type), @nospecialize(shadowinit::Type), @nospecialize(expectedtapetype::Type), @nospecialize(erriffuncwritten::Type), @nospecialize(runtimeactivity::Type), @nospecialize(strongzero::Type))
    @nospecialize
    
    slotnames = Core.svec(Symbol("#self#"),
                          :fa, :a, :tt, :mode, :width, :modifiedbetween,
                          :returnprimal, :shadowinit, :expectedtapetype,
                          :erriffuncwritten, :runtimeactivity, :strongzero)

    stub = Core.GeneratedFunctionStub(identity, slotnames, Core.svec())

    ft = eltype(FA)
    primal_tt = Tuple{map(eltype, TT.parameters)...}
    # look up the method match
    
    min_world = Ref{UInt}(typemin(UInt))
    max_world = Ref{UInt}(typemax(UInt))
 
    mi = my_methodinstance(Mode == API.DEM_ForwardMode ? Forward : Reverse, ft, primal_tt, world, min_world, max_world)
    
    mi === nothing && return stub(world, source, :(throw(MethodError($ft, $primal_tt, $world))))
    
    target = EnzymeTarget()
    rt2 = if A isa UnionAll
        rrt = primal_return_type_world(Mode == API.DEM_ForwardMode ? Forward : Reverse, world, mi)

        # Don't error here but default to nothing return since in cuda context we don't use the device overrides
        if rrt == Union{}
            rrt = Nothing
        end

        if !(A <: Const) && guaranteed_const_nongen(rrt, world)
            estr = "Return type `$rrt` not marked Const, but type is guaranteed to be constant"
            return quote
                error($estr)
            end
        end
        instantiate_annotation(A, rrt, Width)
    else
        @assert A isa DataType
        A
    end

    params = EnzymeCompilerParams(
        PrimalCompilerParams(Mode),
        Tuple{FA,TT.parameters...},
        Mode,
        Width,
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
        StrongZero
    ) #=abiwrap=#
    job =
        Compiler.CompilerJob(mi, CompilerConfig(target, params; kernel = false), world)

    addr = get_trampoline(job)
    id = Base.reinterpret(Int, pointer(addr))
    deferred_codegen_jobs[id] = job

    code = Any[Core.Compiler.ReturnNode(reinterpret(UInt, id))]
    ci = create_fresh_codeinfo(deferred_id_codegen, source, world, slotnames, code)

    ci.edges = Any[mi]

    return ci
end

# The generated wrapper `deferred_id_codegen` is defined in src/late_generated.jl; see the note there.

@inline function deferred_codegen(
    @nospecialize(fa::Type),
    @nospecialize(a::Type),
    @nospecialize(tt::Type),
    @nospecialize(mode::Val),
    @nospecialize(width::Val),
    @nospecialize(modifiedbetween::Val),
    @nospecialize(returnprimal::Val),
    @nospecialize(shadowinit::Val),
    @nospecialize(expectedtapetype::Type),
    @nospecialize(erriffuncwritten::Val),
    @nospecialize(runtimeactivity::Val),
    @nospecialize(strongzero::Val)
)
    id = deferred_id_codegen(fa, a, tt, mode, width, modifiedbetween, returnprimal, shadowinit, expectedtapetype, erriffuncwritten, runtimeactivity, strongzero)
    return _deferred_codegen_call(Val(id))
end

# `@generated` shell so the `ccall("extern deferred_codegen", …)` body
# isn't a static method body that AOT despecialization
# (sysimage `compile=all`, juliac, PrecompileTools) trips on — fixes
# EnzymeAD/Enzyme.jl#3091. Same pattern as
# `GPUCompiler.deferred_codegen(::Val{ft}, ::Val{tt})`.
@generated function _deferred_codegen_call(::Val{id}) where {id}
    id_lit = reinterpret(UInt, id)
    return quote
        Base.@_inline_meta
        ccall("extern deferred_codegen", llvmcall, Ptr{Cvoid}, (UInt,), $id_lit)
    end
end

include("compiler/reflection.jl")

end
