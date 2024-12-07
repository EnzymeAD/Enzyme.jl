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

struct EnzymeInterpreter{T} <: AbstractInterpreter
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

    rules_cache::IdDict{Any, Bool}

    forward_rules::Bool
    reverse_rules::Bool
    broadcast_rewrite::Bool
    handler::T
end

function EnzymeInterpreter(
    cache_or_token,
    mt::Union{Nothing,Core.MethodTable},
    world::UInt,
    forward_rules::Bool,
    reverse_rules::Bool,
    broadcast_rewrite::Bool = true,
    handler = nothing
)
    @assert world <= Base.get_world_counter()

    parms = @static if VERSION >= v"1.12.0-DEV.1017"
        InferenceParams()
    else
        InferenceParams(; unoptimize_throw_blocks=false)
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
        IdDict{Any, Bool}(),
        forward_rules,
        reverse_rules,
        broadcast_rewrite,
        handler
    )
end

EnzymeInterpreter(
    cache_or_token,
    mt::Union{Nothing,Core.MethodTable},
    world::UInt,
    mode::API.CDerivativeMode,
    broadcast_rewrite::Bool = true,
    handler = nothing
) = EnzymeInterpreter(cache_or_token, mt, world, mode == API.DEM_ForwardMode, mode == API.DEM_ReverseModeCombined || mode == API.DEM_ReverseModePrimal || mode == API.DEM_ReverseModeGradient, broadcast_rewrite, handler)

Core.Compiler.InferenceParams(@nospecialize(interp::EnzymeInterpreter)) = interp.inf_params
Core.Compiler.OptimizationParams(@nospecialize(interp::EnzymeInterpreter)) = interp.opt_params
get_inference_world(@nospecialize(interp::EnzymeInterpreter)) = interp.world
Core.Compiler.get_inference_cache(@nospecialize(interp::EnzymeInterpreter)) = interp.local_cache
@static if HAS_INTEGRATED_CACHE
    Core.Compiler.cache_owner(@nospecialize(interp::EnzymeInterpreter)) = interp.token
else
    Core.Compiler.code_cache(@nospecialize(interp::EnzymeInterpreter)) =
        WorldView(interp.code_cache, interp.world)
end

# No need to do any locking since we're not putting our results into the runtime cache
Core.Compiler.lock_mi_inference(@nospecialize(::EnzymeInterpreter), ::MethodInstance) = nothing
Core.Compiler.unlock_mi_inference(@nospecialize(::EnzymeInterpreter), ::MethodInstance) = nothing

Core.Compiler.may_optimize(@nospecialize(::EnzymeInterpreter)) = true
Core.Compiler.may_compress(@nospecialize(::EnzymeInterpreter)) = true
# From @aviatesk:
#     `may_discard_trees = true`` means a complicated (in terms of inlineability) source will be discarded,
#      but as far as I understand Enzyme wants "always inlining, except special cased functions",
#      so I guess we really don't want to discard sources?
Core.Compiler.may_discard_trees(@nospecialize(::EnzymeInterpreter)) = false
Core.Compiler.verbose_stmt_info(@nospecialize(::EnzymeInterpreter)) = false

Core.Compiler.method_table(@nospecialize(interp::EnzymeInterpreter), sv::InferenceState) =
    Core.Compiler.OverlayMethodTable(interp.world, interp.method_table)

function is_alwaysinline_func(@nospecialize(TT))::Bool
    isa(TT, DataType) || return false
    @static if VERSION ≥ v"1.11-"
    if TT.parameters[1] == typeof(Core.memoryref)
        return true
    end
    end
    return false
end

function is_primitive_func(@nospecialize(TT))::Bool
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

function isKWCallSignature(@nospecialize(TT))::Bool
    return TT <: Tuple{typeof(Core.kwcall),Any,Any,Vararg}
end

function simplify_kw(@nospecialize(specTypes))
    if isKWCallSignature(specTypes)
        return Base.tuple_type_tail(Base.tuple_type_tail(specTypes))
    else
        return specTypes
    end
end

include("tfunc.jl")

struct EnzymeCache
    inactive::Bool
    has_rule::Bool
end

if VERSION >= v"1.11.0-"
function CC.ipo_dataflow_analysis!(interp::EnzymeInterpreter, ir::Core.Compiler.IRCode,
                                   caller::Core.Compiler.InferenceResult)
    mi = caller.linfo
    specTypes = simplify_kw(mi.specTypes)
    inactive = false
    has_rule = false
    if is_inactive_from_sig(interp, specTypes, mi)
        inactive = true
    else
        # 2. Check if rule is defined
        if interp.forward_rules && has_frule_from_sig(interp, specTypes, mi)
            has_rule = true
        elseif interp.reverse_rules && has_rrule_from_sig(interp, specTypes, mi)
            has_rule = true
        end
    end
    CC.stack_analysis_result!(caller, EnzymeCache(inactive, has_rule))
    @invoke CC.ipo_dataflow_analysis!(interp::Core.Compiler.AbstractInterpreter, ir::Core.Compiler.IRCode,
                                      caller::Core.Compiler.InferenceResult)
end

else # v1.10
# 1.10 doesn't have stack_analysis_result or ipo_dataflow_analysis
function Core.Compiler.finish(interp::EnzymeInterpreter, opt::Core.Compiler.OptimizationState, ir::Core.Compiler.IRCode,
                   caller::Core.Compiler.InferenceResult)
    (; src, linfo) = opt
    specTypes = simplify_kw(linfo.specTypes)
    inactive = false
    has_rule = false
    if is_inactive_from_sig(interp, specTypes, linfo)
        inactive = true
    else
        # 2. Check if rule is defined
        if interp.forward_rules && has_frule_from_sig(interp, specTypes, linfo)
            has_rule = true
        elseif interp.reverse_rules && has_rrule_from_sig(interp, specTypes, linfo)
            has_rule = true
        end
    end
    @invoke Core.Compiler.finish(interp::Core.Compiler.AbstractInterpreter, opt::Core.Compiler.OptimizationState,
                      ir::Core.Compiler.IRCode, caller::Core.Compiler.InferenceResult)
    # Must happen afterwards
    if inactive || has_rule
        Core.Compiler.set_inlineable!(src, false)
    end
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

import .EnzymeRules: FwdConfig, RevConfig, Annotation
using Core.Compiler: ArgInfo, StmtInfo, AbsIntState
function Core.Compiler.abstract_call_gf_by_type(
    @nospecialize(interp::EnzymeInterpreter),
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
    specTypes = simplify_kw(atype)

    if is_primitive_func(specTypes)
        callinfo = NoInlineCallInfo(callinfo, atype, :primitive)
    elseif is_alwaysinline_func(specTypes)
        callinfo = AlwaysInlineCallInfo(callinfo, atype)
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
            @nospecialize(interp::EnzymeInterpreter),
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
            @nospecialize(interp::EnzymeInterpreter),
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
        # if dest.length < n
        #     throw(BoundsError(dest, 1:n))
        # end
        # if src.length < n
        #     throw(BoundsError(src, 1:n))
        # end
        t1 = Base.@_gc_preserve_begin dest
        t2 = Base.@_gc_preserve_begin src
        Base.memmove(pointer(dest), pointer(src), n * Base.aligned_sizeof(T))
        Base.@_gc_preserve_end t2
        Base.@_gc_preserve_end t1
        return dest
    end
end


# julia> @btime Base.copyto!(dst, src);
#   668.438 ns (0 allocations: 0 bytes)

# inp = rand(2,3,4,5);
# src = Base.Broadcast.preprocess(inp, convert(Base.Broadcast.Broadcasted{Nothing}, Base.Broadcast.instantiate(Base.broadcasted(Main.sin, inp))));
# 
# idx = Base.eachindex(src);
# 
# src2 = sin.(inp);
# 
# dst = zero(inp);
# lindex_v1(idx, dst, src);
# @assert dst == sin.(inp)
# 
# dst = zero(inp);
# lindex_v1(idx, dst, src2);
# @assert dst == sin.(inp)
# 
# @btime lindex_v1(idx, dst, src)
# # 619.140 ns (0 allocations: 0 bytes)
# 
# @btime lindex_v1(idx, dst, src2)
# # 153.258 ns (0 allocations: 0 bytes)

@generated function lindex_v1(idx::BC2, dest, src) where BC2
    if BC2 <: Base.CartesianIndices
        nloops = BC2.parameters[1]
        exprs = Expr[]
        tot = :true
        idxs = Symbol[]
        lims = Symbol[]
        for i in 1:nloops
            sym = Symbol("lim_$i")
            push!(lims, sym)
            sidx = Symbol("idx_$i")
            push!(idxs, sidx)
            push!(exprs, quote
                $sym = idx.indices[$i].stop
            end)
            if tot == :true
                tot = quote $sym != 0 end
            else
                tot = quote $tot && ($sym != 0) end
            end
        end

        loops = quote
            @inbounds dest[$(idxs...)] = @inbounds Base.Broadcast._broadcast_getindex(src, Base.CartesianIndex($(idxs...)))
        end

        # for (sidx, lim) in zip(reverse(idxs), reverse(lims))
        for (sidx, lim) in zip(idxs, lims)
            loops = quote
                let $sidx = 0
                    @inbounds while true
                        $sidx += 1
                        $loops
                        if $sidx == $lim
                            break
                        end
                        $(Expr(:loopinfo, Symbol("julia.simdloop"), nothing))  # Mark loop as SIMD loop
                    end
                end
            end
        end

        return quote
            Base.@_inline_meta
            $(exprs...)
            if $tot
                $loops
            end
        end
    else
        return quote
            Base.@_inline_meta
            @inbounds @simd for I in idx
                dest[I] = src[I]
            end
        end
    end
end

# inp = rand(2,3,4,5);
# # inp = [2.0 3.0; 4.0 5.0; 7.0 9.0]
# src = Base.Broadcast.preprocess(inp, convert(Base.Broadcast.Broadcasted{Nothing}, Base.Broadcast.instantiate(Base.broadcasted(Main.sin, inp))));
# 
# idx = Base.eachindex(src);
# 
# src2 = sin.(inp);
# 
# dst = zero(inp);
# lindex_v2(idx, dst, src);
# @assert dst == sin.(inp)
# 
# dst = zero(inp);
# lindex_v2(idx, dst, src2);
# @assert dst == sin.(inp)
# 
# @btime lindex_v2(idx, dst, src)
# # 1.634 μs (0 allocations: 0 bytes)
# 
# @btime lindex_v2(idx, dst, src2)
# # 1.617 μs (0 allocations: 0 bytes)
@generated function lindex_v2(idx::BC2, dest, src, ::Val{Checked}=Val(true)) where {BC2, Checked}
    if BC2 <: Base.CartesianIndices
        nloops = BC2.parameters[1]
        exprs = Union{Expr,Symbol}[]
        tot = :true
        idxs = Symbol[]
        lims = Symbol[]

        total = :1
        for i in 1:nloops
            sym = Symbol("lim_$i")
            push!(lims, sym)
            sidx = Symbol("idx_$i")
            push!(idxs, sidx)
            push!(exprs, quote
                $sym = idx.indices[$i].stop
            end)
            if tot == :true
                tot = quote $sym != 0 end
                total = sym
            else
                tot = quote $tot && ($sym != 0) end
                total = quote $total * $sym end
            end
        end

        push!(exprs, quote total = $total end)

        lexprs = Expr[]

        if Checked
            for (lidx, lim) in zip(idxs, lims)
                push!(lexprs, quote
                    $lidx = Base.urem_int(tmp, $lim) + 1
                    tmp = Base.udiv_int(tmp, $lim)
                end)
            end
        else
            idxs = [quote I+1 end]
        end

        return quote
            Base.@_inline_meta
            $(exprs...)
            if $tot
                let I = 0
                    @inbounds while true
                        let tmp = I
                            $(lexprs...)
                            @inbounds dest[I+1] = @inbounds Base.Broadcast._broadcast_getindex(src, Base.CartesianIndex($(idxs...)))
                        end
                        I += 1
                        if I == total
                            break
                        end
                        $(Expr(:loopinfo, Symbol("julia.simdloop"), nothing))  # Mark loop as SIMD loop
                    end
                end
            end
        end
    else
        return quote
            Base.@_inline_meta
            @inbounds @simd for I in idx
                dest[I] = src[I]
            end
        end
    end
end


# inp = rand(2,3,4,5);
# src = Base.Broadcast.preprocess(inp, convert(Base.Broadcast.Broadcasted{Nothing}, Base.Broadcast.instantiate(Base.broadcasted(Main.sin, inp))));
# 
# idx = Base.eachindex(src);
# 
# src2 = sin.(inp);
# 
# dst = zero(inp);
# lindex_v3(idx, dst, src);
# @assert dst == sin.(inp)
# 
# dst = zero(inp);
# lindex_v3(idx, dst, src2);
# @assert dst == sin.(inp)
# 
# @btime lindex_v3(idx, dst, src)
# # 568.065 ns (0 allocations: 0 bytes)

# @btime lindex_v3(idx, dst, src2)
# # 23.906 ns (0 allocations: 0 bytes)
@generated function lindex_v3(idx::BC2, dest, src) where BC2
    if BC2 <: Base.CartesianIndices
        nloops = BC2.parameters[1]
        exprs = Union{Expr,Symbol}[]
        tot = :true
        idxs = Symbol[]
        lims = Symbol[]

        condition = :true
        todo = Tuple{Type, Tuple}[(src, ())]

        function index(x, ::Tuple{})
            return x
        end

        function index(x, path)
            if path[1] isa Symbol
                return quote
                    $(index(x, Base.tail(path))).$(path[1])
                end
            else
                return quote getindex($(index(x, Base.tail(path))), $(path[1])) end
            end
        end

        legal = true
        while length(todo) != 0
            cur, path = pop!(todo)
            if cur <: AbstractArray
                if condition == :true
                    condition = quote idx.indices == axes($(index(:src, path))) end
                else                
                    condition = quote $condition && idx.indices == axes($(index(:src, path))) end
                end
                continue
            end
            if cur <: Base.Broadcast.Extruded
                if condition == :true
                    condition = quote all(($(index(:src, path))).keeps) end
                else                
                    condition = quote $condition && all(($(index(:src, path))).keeps) end
                end
                push!(todo, (cur.parameters[1], (:x, path...)))
                continue
            end
            if cur == src && cur <: Base.Broadcast.Broadcasted
                for (i, v) in enumerate(cur.parameters[4].parameters)
                    push!(todo, (v, (i, :args, path...)))
                end
                continue
            end
            if cur <: AbstractFloat
                continue
            end
            legal = false
        end

        if legal
            return quote
                Base.@_inline_meta
                if $condition
                    lindex_v2(idx, dest, src, Val(false))
                else
                    lindex_v1(idx, dest, src)
                end
            end
        else
            return quote
                Base.@_inline_meta
                lindex_v1(idx, dest, src)
            end
        end
    else
        return quote
            Base.@_inline_meta
            @inbounds @simd for I in idx
                dest[I] = src[I]
            end
        end
    end
end

# Override Base.copyto!(dest::AbstractArray, bc::Broadcasted{Nothing}) with
#  a form which provides better analysis of loop indices
@inline function override_bc_copyto!(dest::AbstractArray, bc::Base.Broadcast.Broadcasted{Nothing})
	axdest = Base.axes(dest)
	axbc = Base.axes(bc)
    axdest == axbc || Base.Broadcast.throwdm(axdest, axbc)

    if bc.args isa Tuple{AbstractArray}
        A = bc.args[1]
        if axdest == Base.axes(A)
            if bc.f === Base.identity
                Base.copyto!(dest, A)
                return dest
            end
        end
    end

    # The existing code is rather slow for broadcast in practice: https://github.com/EnzymeAD/Enzyme.jl/issues/1434
    src = Base.Broadcast.preprocess(dest, bc)
    idx = Base.eachindex(src)
    @inline Enzyme.Compiler.Interpreter.lindex_v3(idx, dest, src)
    return dest
end

@generated function same_sized(x::Tuple)
    result = :true
    prev = nothing
    for i in 1:length(x.parameters)
        if x.parameters[i] <: Number
            continue
        end
        if prev == nothing
            prev = quote
                sz = size(x[$i])
            end
            continue
        end
        if result == :true
            result = quote
                sz == size(x[$i])
            end
        else
            result = quote
                $result && sz == size(x[$i])
            end
        end
    end
    return quote
        Base.@_inline_meta
        $prev
        return $result
    end
end


Base.@propagate_inbounds overload_broadcast_getindex(A::Union{Ref,AbstractArray{<:Any,0},Number}, I) = A[] # Scalar-likes can just ignore all indices
Base.@propagate_inbounds overload_broadcast_getindex(::Ref{Type{T}}, I) where {T} = T
# Tuples are statically known to be singleton or vector-like
Base.@propagate_inbounds overload_broadcast_getindex(A::Tuple{Any}, I) = A[1]
Base.@propagate_inbounds overload_broadcast_getindex(A::Tuple, I) = error("unhandled") # A[I[1]]
Base.@propagate_inbounds overload_broadcast_getindex(A, I) = A[I]

@inline function override_bc_materialize(bc)
    if bc.args isa Tuple{AbstractArray} && bc.f === Base.identity
        return copy(bc.args[1])
    end
    ElType = Base.Broadcast.combine_eltypes(bc.f, bc.args)
    dest = similar(bc, ElType)
    if all(isa_array_or_number, bc.args) && same_sized(bc.args)
        @inbounds @simd for I in 1:length(bc)
            val = Base.Broadcast._broadcast_getindex_evalf(bc.f, map(Base.Fix2(overload_broadcast_getindex, I), bc.args)...)
            dest[I] = val
        end
    else
        Base.copyto!(dest, bc)
    end
    return dest 
end

struct MultiOp{Position, NumUsed, F1, F2}
    f1::F1
    f2::F2
end

@generated function (m::MultiOp{Position, NumUsed})(args::Vararg{Any, N}) where {N, Position, NumUsed}
    f2args = Union{Symbol, Expr}[]
    for i in Position:(Position+NumUsed)
        push!(f2args, :(args[$i]))
    end
    f1args = Union{Symbol, Expr}[]
    for i in 1:Position
        push!(f1args, :(args[$i]))
    end
    push!(f1args, quote
        f2($(f2args...))
    end)
    for i in (Position+NumUsed):N
        push!(f1args, :(args[$i]))
    end
    return quote
        Base.@_inline_meta
        f1($(f1args...))
    end
end

@inline function array_or_number(@nospecialize(Ty))::Bool
    return Ty <: AbstractArray || Ty <: Number
end

@inline function isa_array_or_number(@nospecialize(x))::Bool
    return x isa AbstractArray || x isa Number
end

@inline function num_or_eltype(@nospecialize(Ty))::Type
    if Ty <: AbstractArray
        eltype(Ty)
    else
        return Ty
    end
end

function abstract_call_known(
    interp::EnzymeInterpreter{Handler},
    @nospecialize(f),
    arginfo::ArgInfo,
    si::StmtInfo,
    sv::AbsIntState,
    max_methods::Int = get_max_methods(interp, f, sv),
) where Handler

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
    
    if interp.broadcast_rewrite
        if f === Base.materialize && length(argtypes) == 2
            bcty = widenconst(argtypes[2])
            if Base.isconcretetype(bcty) && bcty <: Base.Broadcast.Broadcasted{<:Base.Broadcast.DefaultArrayStyle, Nothing} && all(array_or_number, bcty.parameters[4].parameters) && any(Base.Fix2(Base.:<:, AbstractArray), bcty.parameters[4].parameters)
                    arginfo2 = ArgInfo(
                        fargs isa Nothing ? nothing :
                        [:(Enzyme.Compiler.Interpreter.override_bc_materialize), fargs[2:end]...],
                        [Core.Const(Enzyme.Compiler.Interpreter.override_bc_materialize), argtypes[2:end]...],
                    )
                    return Base.@invoke abstract_call_known(
                        interp::AbstractInterpreter,
                        Enzyme.Compiler.Interpreter.override_bc_materialize::Any,
                        arginfo2::ArgInfo,
                        si::StmtInfo,
                        sv::AbsIntState,
                        max_methods::Int,
                    )
            end
        end

        if f === Base.copyto! && length(argtypes) == 3
            # Ideally we just override uses of the AbstractArray base class, but
            # I don't know how to override the method in base, without accidentally overridding
            # it for say CuArray or other users. For safety, we only override for Array
            if widenconst(argtypes[2]) <: Array &&
               widenconst(argtypes[3]) <: Base.Broadcast.Broadcasted{Nothing}
            
                arginfo2 = ArgInfo(
                    fargs isa Nothing ? nothing :
                    [:(Enzyme.Compiler.Interpreter.override_bc_copyto!), fargs[2:end]...],
                    [Core.Const(Enzyme.Compiler.Interpreter.override_bc_copyto!), argtypes[2:end]...],
                )
                return Base.@invoke abstract_call_known(
                    interp::AbstractInterpreter,
                    Enzyme.Compiler.Interpreter.override_bc_copyto!::Any,
                    arginfo2::ArgInfo,
                    si::StmtInfo,
                    sv::AbsIntState,
                    max_methods::Int,
                )
            end
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
            return Base.@invoke abstract_call_known(
                interp::AbstractInterpreter,
                Enzyme.Compiler.Interpreter.myunsafe_copyto!::Any,
                arginfo2::ArgInfo,
                si::StmtInfo,
                sv::AbsIntState,
                max_methods::Int,
            )
        end
    end

    if interp.handler != nothing
        return interp.handler(interp, f, arginfo, si, sv, max_methods)
    end
    return Base.@invoke abstract_call_known(
        interp::AbstractInterpreter,
        f::Any,
        arginfo::ArgInfo,
        si::StmtInfo,
        sv::AbsIntState,
        max_methods::Int,
    )
end

end
