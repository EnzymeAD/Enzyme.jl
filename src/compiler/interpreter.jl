module Interpreter
import Enzyme: API
using Core.Compiler:
    AbstractInterpreter,
    InferenceResult,
    InferenceParams,
    InferenceState,
    OptimizationParams,
    MethodInstance
using GPUCompiler: @safe_debug
if VERSION < v"1.11.0-DEV.1552"
    using GPUCompiler: CodeCache, WorldView, @safe_debug
end
const HAS_INTEGRATED_CACHE = VERSION >= v"1.11.0-DEV.1552"

import ..Enzyme
import ..EnzymeRules

@static if VERSION ≥ v"1.11.0-DEV.1498"
    import Core.Compiler: get_inference_world
    using Base: get_world_counter
else
    import Core.Compiler: get_world_counter, get_world_counter as get_inference_world
end

struct EnzymeInterpreter <: AbstractInterpreter
    @static if HAS_INTEGRATED_CACHE
        token::Any
    else
        code_cache::CodeCache
    end
    method_table::Union{Nothing,Core.MethodTable}

    # Cache of inference results for this particular interpreter
    local_cache::Vector{InferenceResult}
    # The world age we're working inside of
    world::UInt

    # Parameters for inference and optimization
    inf_params::InferenceParams
    opt_params::OptimizationParams

    mode::API.CDerivativeMode
end

function EnzymeInterpreter(
    cache_or_token,
    mt::Union{Nothing,Core.MethodTable},
    world::UInt,
    mode::API.CDerivativeMode,
)
    @assert world <= Base.get_world_counter()

    parms = @static if VERSION < v"1.12"
        InferenceParams(unoptimize_throw_blocks = false)
    else
        InferenceParams()
    end

    return EnzymeInterpreter(
        cache_or_token,
        mt,

        # Initially empty cache
        Vector{InferenceResult}(),

        # world age counter
        world,

        # parameters for inference and optimization
        parms,
        OptimizationParams(),
        mode,
    )
end

Core.Compiler.InferenceParams(interp::EnzymeInterpreter) = interp.inf_params
Core.Compiler.OptimizationParams(interp::EnzymeInterpreter) = interp.opt_params
get_inference_world(interp::EnzymeInterpreter) = interp.world
Core.Compiler.get_inference_cache(interp::EnzymeInterpreter) = interp.local_cache
@static if HAS_INTEGRATED_CACHE
    Core.Compiler.cache_owner(interp::EnzymeInterpreter) = interp.token
else
    Core.Compiler.code_cache(interp::EnzymeInterpreter) =
        WorldView(interp.code_cache, interp.world)
end

# No need to do any locking since we're not putting our results into the runtime cache
Core.Compiler.lock_mi_inference(::EnzymeInterpreter, ::MethodInstance) = nothing
Core.Compiler.unlock_mi_inference(::EnzymeInterpreter, ::MethodInstance) = nothing

Core.Compiler.may_optimize(::EnzymeInterpreter) = true
Core.Compiler.may_compress(::EnzymeInterpreter) = true
# From @aviatesk:
#     `may_discard_trees = true`` means a complicated (in terms of inlineability) source will be discarded,
#      but as far as I understand Enzyme wants "always inlining, except special cased functions",
#      so I guess we really don't want to discard sources?
Core.Compiler.may_discard_trees(::EnzymeInterpreter) = false
Core.Compiler.verbose_stmt_info(::EnzymeInterpreter) = false

if isdefined(Base.Experimental, Symbol("@overlay"))
    Core.Compiler.method_table(interp::EnzymeInterpreter, sv::InferenceState) =
        Core.Compiler.OverlayMethodTable(interp.world, interp.method_table)
else

    # On 1.6- CUDA.jl will poison the method table at the end of the world
    # using GPUCompiler: WorldOverlayMethodTable
    # Core.Compiler.method_table(interp::EnzymeInterpreter, sv::InferenceState) =
    #     WorldOverlayMethodTable(interp.world)
end

function is_alwaysinline_func(@nospecialize(TT))
    isa(TT, DataType) || return false
    return false
end

function is_primitive_func(@nospecialize(TT))
    isa(TT, DataType) || return false
    ft = TT.parameters[1]
    if ft == typeof(Enzyme.pmap)
        return true
    end
    match = Enzyme.Compiler.find_math_method(ft, TT.parameters[2:end])[1]
    if match !== nothing
        return true
    end

    # FIXME(@wsmoses): For which types should we not inline?
    if ft === typeof(Base.wait) ||
       ft === typeof(Base._wait) ||
       ft === typeof(Base.enq_work) ||
       ft === typeof(Base.Threads.threadid) ||
       ft == typeof(Base.Threads.nthreads) ||
       ft === typeof(Base.Threads.threading_run)
        return true
    end
    return false
end

function isKWCallSignature(@nospecialize(TT))
    return TT <: Tuple{typeof(Core.kwcall),Any,Any,Vararg}
end

function simplify_kw(@nospecialize specTypes)
    if isKWCallSignature(specTypes)
        return Base.tuple_type_tail(Base.tuple_type_tail(specTypes))
    else
        return specTypes
    end
end

import Core.Compiler: CallInfo
struct NoInlineCallInfo <: CallInfo
    info::CallInfo # wrapped call
    tt::Any # ::Type
    kind::Symbol
    NoInlineCallInfo(@nospecialize(info::CallInfo), @nospecialize(tt), kind::Symbol) =
        new(info, tt, kind)
end
Core.Compiler.nsplit_impl(info::NoInlineCallInfo) = Core.Compiler.nsplit(info.info)
Core.Compiler.getsplit_impl(info::NoInlineCallInfo, idx::Int) =
    Core.Compiler.getsplit(info.info, idx)
Core.Compiler.getresult_impl(info::NoInlineCallInfo, idx::Int) =
    Core.Compiler.getresult(info.info, idx)
struct AlwaysInlineCallInfo <: CallInfo
    info::CallInfo # wrapped call
    tt::Any # ::Type
    AlwaysInlineCallInfo(@nospecialize(info::CallInfo), @nospecialize(tt)) = new(info, tt)
end
Core.Compiler.nsplit_impl(info::AlwaysInlineCallInfo) = Core.Compiler.nsplit(info.info)
Core.Compiler.getsplit_impl(info::AlwaysInlineCallInfo, idx::Int) =
    Core.Compiler.getsplit(info.info, idx)
Core.Compiler.getresult_impl(info::AlwaysInlineCallInfo, idx::Int) =
    Core.Compiler.getresult(info.info, idx)

using Core.Compiler: ArgInfo, StmtInfo, AbsIntState
function Core.Compiler.abstract_call_gf_by_type(
    interp::EnzymeInterpreter,
    @nospecialize(f),
    arginfo::ArgInfo,
    si::StmtInfo,
    @nospecialize(atype),
    sv::AbsIntState,
    max_methods::Int,
)
    ret = @invoke Core.Compiler.abstract_call_gf_by_type(
        interp::AbstractInterpreter,
        f::Any,
        arginfo::ArgInfo,
        si::StmtInfo,
        atype::Any,
        sv::AbsIntState,
        max_methods::Int,
    )
    callinfo = ret.info
    method_table = Core.Compiler.method_table(interp)
    specTypes = simplify_kw(atype)
    if is_primitive_func(specTypes)
        callinfo = NoInlineCallInfo(callinfo, atype, :primitive)
    elseif is_alwaysinline_func(specTypes)
        callinfo = AlwaysInlineCallInfo(callinfo, atype)
    elseif EnzymeRules.is_inactive_from_sig(specTypes; world = interp.world, method_table)
        callinfo = NoInlineCallInfo(callinfo, atype, :inactive)
    elseif interp.mode == API.DEM_ForwardMode
        if EnzymeRules.has_frule_from_sig(specTypes; world = interp.world, method_table)
            callinfo = NoInlineCallInfo(callinfo, atype, :frule)
        end
    elseif EnzymeRules.has_rrule_from_sig(specTypes; world = interp.world, method_table)
        callinfo = NoInlineCallInfo(callinfo, atype, :rrule)
    end
    @static if VERSION ≥ v"1.11-"
        return Core.Compiler.CallMeta(ret.rt, ret.exct, ret.effects, callinfo)
    else
        return Core.Compiler.CallMeta(ret.rt, ret.effects, callinfo)
    end
end

let # overload `inlining_policy`
    @static if VERSION ≥ v"1.11.0-DEV.879"
        sigs_ex = :(
            interp::EnzymeInterpreter,
            @nospecialize(src),
            @nospecialize(info::Core.Compiler.CallInfo),
            stmt_flag::UInt32,
        )
        args_ex = :(
            interp::AbstractInterpreter,
            src::Any,
            info::Core.Compiler.CallInfo,
            stmt_flag::UInt32,
        )
    else
        sigs_ex = :(
            interp::EnzymeInterpreter,
            @nospecialize(src),
            @nospecialize(info::Core.Compiler.CallInfo),
            stmt_flag::UInt8,
            mi::MethodInstance,
            argtypes::Vector{Any},
        )
        args_ex = :(
            interp::AbstractInterpreter,
            src::Any,
            info::Core.Compiler.CallInfo,
            stmt_flag::UInt8,
            mi::MethodInstance,
            argtypes::Vector{Any},
        )
    end
    @static if isdefined(Core.Compiler, :inlining_policy)
    @eval function Core.Compiler.inlining_policy($(sigs_ex.args...))
        if info isa NoInlineCallInfo
            if info.kind === :primitive
                @safe_debug "Blocking inlining for primitive func" info.tt
            elseif info.kind === :inactive
                @safe_debug "Blocking inlining due to inactive rule" info.tt
            elseif info.kind === :frule
                @safe_debug "Blocking inlining due to frule" info.tt
            else
                @assert info.kind === :rrule
                @safe_debug "Blocking inlining due to rrule" info.tt
            end
            return nothing
        elseif info isa AlwaysInlineCallInfo
            @safe_debug "Forcing inlining for primitive func" info.tt
            return src
        end
        return @invoke Core.Compiler.inlining_policy($(args_ex.args...))
    end
    else
    @eval function Core.Compiler.src_inlining_policy($(sigs_ex.args...))
        if info isa NoInlineCallInfo
            if info.kind === :primitive
                @safe_debug "Blocking inlining for primitive func" info.tt
            elseif info.kind === :inactive
                @safe_debug "Blocking inlining due to inactive rule" info.tt
            elseif info.kind === :frule
                @safe_debug "Blocking inlining due to frule" info.tt
            else
                @assert info.kind === :rrule
                @safe_debug "Blocking inlining due to rrule" info.tt
            end
            return nothing
        elseif info isa AlwaysInlineCallInfo
            @safe_debug "Forcing inlining for primitive func" info.tt
            return src
        end
        return @invoke Core.Compiler.src_inlining_policy($(args_ex.args...))
    end
    end
end

import Core.Compiler:
    abstract_call,
    abstract_call_known,
    ArgInfo,
    StmtInfo,
    AbsIntState,
    get_max_methods,
    CallMeta,
    Effects,
    NoCallInfo,
    widenconst,
    mapany,
    MethodResultPure

struct AutodiffCallInfo <: CallInfo
    # ...
    info::CallInfo
end

@static if VERSION < v"1.11.0-"
else
    @inline function myunsafe_copyto!(dest::MemoryRef{T}, src::MemoryRef{T}, n) where {T}
        Base.@_terminates_globally_notaskstate_meta
        @boundscheck memoryref(dest, n), memoryref(src, n)
        t1 = Base.@_gc_preserve_begin dest
        t2 = Base.@_gc_preserve_begin src
        Base.memmove(pointer(dest), pointer(src), n * Base.aligned_sizeof(T))
        Base.@_gc_preserve_end t2
        Base.@_gc_preserve_end t1
        return dest
    end
end


function abstract_call_known(
    interp::EnzymeInterpreter,
    @nospecialize(f),
    arginfo::ArgInfo,
    si::StmtInfo,
    sv::AbsIntState,
    max_methods::Int = get_max_methods(interp, f, sv),
)

    (; fargs, argtypes) = arginfo

    if f === Enzyme.within_autodiff
        if length(argtypes) != 1
            @static if VERSION < v"1.11.0-"
                return CallMeta(Union{}, Effects(), NoCallInfo())
            else
                return CallMeta(Union{}, Union{}, Effects(), NoCallInfo())
            end
        end
        @static if VERSION < v"1.11.0-"
            return CallMeta(
                Core.Const(true),
                Core.Compiler.EFFECTS_TOTAL,
                MethodResultPure(),
            )
        else
            return CallMeta(
                Core.Const(true),
                Union{},
                Core.Compiler.EFFECTS_TOTAL,
                MethodResultPure(),
            )
        end
    end

    @static if VERSION < v"1.11.0-"
    else
        if f === Base.unsafe_copyto! && length(argtypes) == 4 &&
            widenconst(argtypes[2]) <: Base.MemoryRef &&
            widenconst(argtypes[3]) == widenconst(argtypes[2]) && 
            Base.allocatedinline(eltype(widenconst(argtypes[2]))) && Base.isbitstype(eltype(widenconst(argtypes[2])))

            arginfo2 = ArgInfo(
                fargs isa Nothing ? nothing :
                [:(Enzyme.Compiler.Interpreter.myunsafe_copyto!), fargs[2:end]...],
                [Core.Const(Enzyme.Compiler.Interpreter.myunsafe_copyto!), argtypes[2:end]...],
            )
            return abstract_call_known(
                interp,
                Enzyme.Compiler.Interpreter.myunsafe_copyto!,
                arginfo2,
                si,
                sv,
                max_methods,
            )
        end
    end

    if f === Enzyme.autodiff && length(argtypes) >= 4
        if widenconst(argtypes[2]) <: Enzyme.Mode &&
           widenconst(argtypes[3]) <: Enzyme.Annotation &&
           widenconst(argtypes[4]) <: Type{<:Enzyme.Annotation}
            arginfo2 = ArgInfo(
                fargs isa Nothing ? nothing :
                [:(Enzyme.autodiff_deferred), fargs[2:end]...],
                [Core.Const(Enzyme.autodiff_deferred), argtypes[2:end]...],
            )
            return abstract_call_known(
                interp,
                Enzyme.autodiff_deferred,
                arginfo2,
                si,
                sv,
                max_methods,
            )
        end
    end
    return Base.@invoke abstract_call_known(
        interp::AbstractInterpreter,
        f,
        arginfo::ArgInfo,
        si::StmtInfo,
        sv::AbsIntState,
        max_methods::Int,
    )
end

end
