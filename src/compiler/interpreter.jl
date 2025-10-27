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
using GPUCompiler
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

function rule_backedge_holder end

function rule_backedge_holder_generator(world::UInt, source, self, ft::Type)
    @nospecialize

    code = Any[Core.Compiler.ReturnNode(world)]
    ci = Core.Compiler.create_fresh_codeinfo(rule_backedge_holder, source, world, Core.svec(Symbol("#self#"), :ft), code)

    edges = Any[]

    if ft == typeof(EnzymeRules.augmented_primal)
        sig = Tuple{typeof(EnzymeRules.augmented_primal), <:RevConfig, <:Annotation, Type{<:Annotation},Vararg{Annotation}}
    elseif ft == typeof(EnzymeRules.forward)
        sig = Tuple{typeof(EnzymeRules.forward), <:FwdConfig, <:Annotation, Type{<:Annotation},Vararg{Annotation}}
    else
        sig = Tuple{typeof(EnzymeRules.inactive), Vararg{Annotation}}
    end
    add_edge!(edges, sig)

    ci.edges = edges

    return ci
end

@eval Base.@assume_effects :removable :foldable :nothrow @inline function rule_backedge_holder(ft)
    $(Expr(:meta, :generated_only))
    $(Expr(:meta, :generated, rule_backedge_holder_generator))
end

struct EnzymeInterpreter{T} <: AbstractInterpreter
    @static if HAS_INTEGRATED_CACHE
        token::Any
    else
        code_cache::CodeCache
    end
    method_table::Core.Compiler.MethodTableView

    # Cache of inference results for this particular interpreter
    local_cache::Vector{InferenceResult}
    # The world age we're working inside of
    world::UInt

    # Parameters for inference and optimization
    inf_params::InferenceParams
    opt_params::OptimizationParams

    forward_rules::Bool
    reverse_rules::Bool
    inactive_rules::Bool
    broadcast_rewrite::Bool

    # When false, leave the check for within_autodiff to the handler.
    within_autodiff_rewrite::Bool

    handler::T
end

const SigCache = Dict{Tuple, Dict{UInt, Base.IdSet{Type}}}()
function get_rule_signatures(f, TT, world)
    subdict = if haskey(SigCache, (f, TT))
       SigCache[(f, TT)]
    else
       tmp = Dict{UInt, Base.IdSet{Type}}()
       SigCache[(f, TT)] = tmp
       tmp
    end
    if haskey(subdict, world)
       return subdict[world]
    end
    fwdrules_meths = Base._methods(f, TT, -1, world)::Vector
    sigs = Type[]
    for rule in fwdrules_meths
        push!(sigs, (rule::Core.MethodMatch).method.sig)
    end
    result = Base.IdSet{Type}(sigs)
    subdict[world] = result
    return result
end

function rule_sigs_equal(a, b)
    if length(a) != length(b)
        return false
    end
    for v in a
        if v in b
            continue
        end
        return false
    end
    return true
end

const LastFwdWorld = Ref(Base.IdSet{Type}())
const LastRevWorld = Ref(Base.IdSet{Type}())
const LastInaWorld = Ref(Base.IdSet{Type}())

function EnzymeInterpreter(
    cache_or_token,
    mt::Union{Nothing,Core.MethodTable},
    world::UInt,
    forward_rules::Bool,
    reverse_rules::Bool,
    inactive_rules::Bool,
    broadcast_rewrite::Bool = true,
    within_autodiff_rewrite::Bool = true,
    handler = nothing
)
    @assert world <= Base.get_world_counter()

    parms = @static if VERSION >= v"1.12.0-DEV.1017"
        InferenceParams()
    else
        InferenceParams(; unoptimize_throw_blocks=false)
    end
    
    @static if HAS_INTEGRATED_CACHE

    else
        cache_or_token = cache_or_token::CodeCache
        invalid = false
        if forward_rules
            fwdrules = get_rule_signatures(EnzymeRules.forward, Tuple{<:FwdConfig, <:Annotation, Type{<:Annotation}, Vararg{Annotation}}, world)
            if !rule_sigs_equal(fwdrules, LastFwdWorld[])
                LastFwdWorld[] = fwdrules
                invalid = true
            end
        end
        if reverse_rules
            revrules = get_rule_signatures(EnzymeRules.augmented_primal, Tuple{<:RevConfig, <:Annotation, Type{<:Annotation}, Vararg{Annotation}}, world)
            if !rule_sigs_equal(revrules, LastRevWorld[])
                LastRevWorld[] = revrules
                invalid = true
            end
        end

        if inactive_rules
            inarules = get_rule_signatures(EnzymeRules.inactive, Tuple{Vararg{Any}}, world)
            if !rule_sigs_equal(inarules, LastInaWorld[])
                LastInaWorld[] = inarules
                invalid = true
            end
        end
        
        if invalid
            Base.empty!(cache_or_token)
        end
    end

    return EnzymeInterpreter(
        cache_or_token,
	mt == nothing ? Core.Compiler.InternalMethodTable(world) : Core.Compiler.OverlayMethodTable(world, mt),

        # Initially empty cache
        Vector{InferenceResult}(),

        # world age counter
        world,

        # parameters for inference and optimization
        parms,
        OptimizationParams(),
        forward_rules::Bool,
        reverse_rules::Bool,
        inactive_rules::Bool,
        broadcast_rewrite::Bool,
        within_autodiff_rewrite::Bool,
        handler
    )
end

EnzymeInterpreter(
    cache_or_token,
    mt::Union{Nothing,Core.MethodTable},
    world::UInt,
    mode::API.CDerivativeMode,
    inactive_rules::Bool,
    broadcast_rewrite::Bool = true,
    within_autodiff_rewrite::Bool = true,
    handler = nothing
) = EnzymeInterpreter(cache_or_token, mt, world, mode == API.DEM_ForwardMode, mode == API.DEM_ReverseModeCombined || mode == API.DEM_ReverseModePrimal || mode == API.DEM_ReverseModeGradient, inactive_rules, broadcast_rewrite, within_autodiff_rewrite, handler)

function EnzymeInterpreter(interp::EnzymeInterpreter;
    cache_or_token = (@static if HAS_INTEGRATED_CACHE
        interp.token
    else
        interp.code_cache
    end),
    mt = interp.method_table,
    local_cache = interp.local_cache,
    world = interp.world,
    inf_params = interp.inf_params,
    opt_params = interp.opt_params,
    forward_rules = interp.forward_rules,
    reverse_rules = interp.reverse_rules,
    inactive_rules = interp.inactive_rules,
    broadcast_rewrite = interp.broadcast_rewrite,
    within_autodiff_rewrite = interp.within_autodiff_rewrite,
    handler = interp.handler)
    return EnzymeInterpreter(
        cache_or_token,
        mt,
        local_cache,
        world,
        inf_params,
        opt_params,
        forward_rules,
        reverse_rules,
        inactive_rules,
        broadcast_rewrite,
        within_autodiff_rewrite,
        handler
    )
end

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
if isdefined(Core.Compiler, :verbose_stmt_inf)
    Core.Compiler.verbose_stmt_info(@nospecialize(::EnzymeInterpreter)) = false
end

Core.Compiler.method_table(@nospecialize(interp::EnzymeInterpreter)) = interp.method_table

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
if VERSION >= v"1.12.0-DEV.1531"
    Core.Compiler.add_edges_impl(edges::Vector{Any}, info::NoInlineCallInfo) =
        Core.Compiler.add_edges!(edges, info.info)
end

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
if VERSION >= v"1.12.0-DEV.1531"
    Core.Compiler.add_edges_impl(edges::Vector{Any}, info::AlwaysInlineCallInfo) =
        Core.Compiler.add_edges!(edges, info.info)
end


import .EnzymeRules: FwdConfig, RevConfig, Annotation
using Core.Compiler: ArgInfo, StmtInfo, AbsIntState

struct FutureCallinfoByType
    atype::Any
end

@inline function (closure::FutureCallinfoByType)(ret::Core.Compiler.CallMeta, @nospecialize(interp::AbstractInterpreter), sv::AbsIntState)
    atype = closure.atype
    callinfo = ret.info
    specTypes = simplify_kw(atype)

    if is_primitive_func(specTypes)
        callinfo = NoInlineCallInfo(callinfo, atype, :primitive)
    elseif is_alwaysinline_func(specTypes)
        callinfo = AlwaysInlineCallInfo(callinfo, atype)
    else
        method_table = Core.Compiler.method_table(interp)
        if interp.inactive_rules && EnzymeRules.is_inactive_from_sig(specTypes; world = interp.world, method_table)
            callinfo = NoInlineCallInfo(callinfo, atype, :inactive)
        elseif interp.forward_rules && EnzymeRules.has_frule_from_sig(specTypes; world = interp.world, method_table)
            callinfo = NoInlineCallInfo(callinfo, atype, :frule)
        elseif interp.reverse_rules && EnzymeRules.has_rrule_from_sig(specTypes; world = interp.world, method_table)
            callinfo = NoInlineCallInfo(callinfo, atype, :rrule)
        end
    end
    @static if VERSION ≥ v"1.11-"
        return Core.Compiler.CallMeta(ret.rt, ret.exct, ret.effects, callinfo)
    else
        return Core.Compiler.CallMeta(ret.rt, ret.effects, callinfo)
    end
end

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

    if isdefined(Core.Compiler, :Future) # if stackless inference
        return Core.Compiler.Future{Core.Compiler.CallMeta}(FutureCallinfoByType(atype), ret, interp, sv)
    end

    return FutureCallinfoByType(atype)(ret, interp, sv)
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

            return false
        elseif info isa AlwaysInlineCallInfo
            @safe_debug "Forcing inlining for primitive func" info.tt

            return true
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
    MethodResultPure

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
    todo = Tuple{Expr, Type}[]
    for i in 1:length(x.parameters)
	push!(todo, (:(x[$i]), x.parameters[i]))
    end
    while length(todo) != 0
	expr, ty = pop!(todo)
        if ty <: Number || ty <: Base.RefValue
            continue
        end
	if ty <: Base.Broadcast.Broadcasted{<:Base.Broadcast.DefaultArrayStyle, Nothing}
	    for i in 1:length(ty.parameters[4].parameters)
	       push!(todo, (:($expr.args[$i]), ty.parameters[4].parameters[i]))
	    end
	    continue
	end
	@assert ty <: AbstractArray
        if prev == nothing
            prev = quote
                sz = size($expr)
            end
            continue
        end
        if result == :true
            result = quote
                sz == size($expr)
            end
        else
            result = quote
                $result && sz == size($expr)
            end
        end
    end
    if result == :true
	return quote
	   Base.@_inline_meta
   	   true
  	end
    end
    return quote
        Base.@_inline_meta
        $prev
        return $result
    end
end

@generated function first_array(x::Tuple)
    result = :true
    prev = nothing
    todo = Tuple{Expr, Type}[]
    for i in 1:length(x.parameters)
	push!(todo, (:(x[$i]), x.parameters[i]))
    end
    while length(todo) != 0
	expr, ty = pop!(todo)
        if ty <: Number || ty <: Base.RefValue
            continue
        end
	if ty <: Base.Broadcast.Broadcasted{<:Base.Broadcast.DefaultArrayStyle, Nothing}
	    for i in 1:length(ty.parameters[4].parameters)
	       push!(todo, (:($expr.args[$i]), ty.parameters[4].parameters[i]))
	    end
	    continue
	end
	@assert ty <: AbstractArray
	return quote
	    Base.@_inline_meta
	    $expr
	end
    end
    return quote
        Base.@_inline_meta
	throw(AssertionError("No array"))
    end
end


Base.@propagate_inbounds @inline overload_broadcast_getindex(A::Union{Ref,AbstractArray{<:Any,0},Number}, I) = A[] # Scalar-likes can just ignore all indices
Base.@propagate_inbounds @inline overload_broadcast_getindex(::Ref{Type{T}}, I) where {T} = T
# Tuples are statically known to be singleton or vector-like
Base.@propagate_inbounds @inline overload_broadcast_getindex(A::Tuple{Any}, I) = A[1]
Base.@propagate_inbounds @inline overload_broadcast_getindex(A::Tuple, I) = error("unhandled") # A[I[1]]
Base.@propagate_inbounds @generated function overload_broadcast_getindex(bc::Base.Broadcast.Broadcasted, I)
   args = Expr[]
   for i in 1:length(bc.parameters[4].parameters)
      push!(args, Expr(:call, overload_broadcast_getindex, :(bc.args[$i]), :I))
   end
   expr = Expr(:call, Base.Broadcast._broadcast_getindex_evalf, :(bc.f), args...)
   return quote
      Base.@_inline_meta
      $expr
   end
end

Base.@propagate_inbounds @inline overload_broadcast_getindex(A, I) = @inbounds A[I]

struct OverrideBCMaterialize{ElType}
end

@inline function (::OverrideBCMaterialize{ElType})(bc) where ElType
    if bc.args isa Tuple{AbstractArray} && bc.f === Base.identity
        return copy(bc.args[1])
    end
    dest = @inline similar(bc, ElType)
    if same_sized(bc.args)
        # dest = @inline similar(first_array(bc.args), ElType)
	@inbounds @simd for I in 1:length(bc)
	    val = overload_broadcast_getindex(bc, I)
            dest[I] = val
        end
	return dest
    else
       # The existing code is rather slow for broadcast in practice: https://github.com/EnzymeAD/Enzyme.jl/issues/1434
       src = @inline Base.Broadcast.preprocess(nothing, bc)
       idx = Base.eachindex(src)
       @inline Enzyme.Compiler.Interpreter.lindex_v3(idx, dest, src)
       return dest
    end
end

@inline function override_bc_foldl(op, init, itr)
    # Unroll the while loop once; if init is known, the call to op may
    # be evaluated at compile time
    y = iterate(itr)
    y === nothing && return init
    v = op(init, y[1])
   
    if same_sized(itr.args)
	@inbounds @simd for I in 2:length(itr)
	    val = overload_broadcast_getindex(itr, I)
            v = op(v, val)
        end
    else
	while true
	    y = iterate(itr, y[2])
	    y === nothing && break
	    v = op(v, y[1])
	end
    end
    return v
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

@inline function bc_or_array_or_number_ty(@nospecialize(Ty::Type), midnothing::Bool=true)::Bool
    if ( midnothing && Ty <: Base.Broadcast.Broadcasted{<:Base.Broadcast.DefaultArrayStyle, Nothing}) ||
       (!midnothing && Ty <: Base.Broadcast.Broadcasted{<:Base.Broadcast.DefaultArrayStyle})
        return all(Base.Fix2(bc_or_array_or_number_ty, midnothing), Ty.parameters[4].parameters)
    else
	return Ty <: AbstractArray || Ty <: Number || Ty <: Base.RefValue
    end
end

@inline function has_array(@nospecialize(Ty::Type), midnothing::Bool=true)::Bool
    if ( midnothing && Ty <: Base.Broadcast.Broadcasted{<:Base.Broadcast.DefaultArrayStyle, Nothing}) ||
       (!midnothing && Ty <: Base.Broadcast.Broadcasted{<:Base.Broadcast.DefaultArrayStyle})
        return any(Base.Fix2(has_array, midnothing), Ty.parameters[4].parameters)
    else
	return Ty <: AbstractArray
    end
end

@generated function isa_bc_or_array_or_number(x)::Bool
    res = bc_or_array_or_number_ty(x)
    return quote
       Base.@_inline_meta
       $res
    end
end

@inline function num_or_eltype(@nospecialize(Ty))::Type
    if Ty <: AbstractArray
        eltype(Ty)
    else
        return Ty
    end
end


## Computation of inferred result type, for empty and concretely inferred cases only
ty_broadcast_getindex_eltype(interp, bc::Type{<:Base.Broadcast.Broadcasted}) = ty_combine_eltypes(interp, bc.parameters[3], (bc.parameters[4].parameters...,))
ty_broadcast_getindex_eltype(interp, A) = eltype(A)  # Tuple, Array, etc.

ty_eltypes(interp, ::Tuple{}) = Tuple{}
ty_eltypes(interp, t::Tuple{Any}) = Iterators.TupleOrBottom(ty_broadcast_getindex_eltype(interp, t[1]))
ty_eltypes(interp, t::Tuple{Any,Any}) = Iterators.TupleOrBottom(ty_broadcast_getindex_eltype(interp, t[1]), ty_broadcast_getindex_eltype(interp, t[2]))
ty_eltypes(interp, t::Tuple) = (TT = ty_eltypes(interp, Base.tail(t)); TT === Union{} ? Union{} : Iterators.TupleOrBottom(ty_broadcast_getindex_eltype(interp, t[1]), TT.parameters...))
# eltypes(t::Tuple) = Iterators.TupleOrBottom(ntuple(i -> _broadcast_getindex_eltype(t[i]), Val(length(t)))...)

# Inferred eltype of result of broadcast(f, args...)
function ty_combine_eltypes(interp, f, args::Tuple)
    argT = ty_eltypes(interp, args)
    argT === Union{} && return Union{}
    preprom = Core.Compiler._return_type(interp, Tuple{f, argT.parameters...})
    return Base.promote_typejoin_union(preprom)
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

    if interp.within_autodiff_rewrite && f === Enzyme.within_autodiff
        if length(argtypes) != 1
            @static if VERSION < v"1.11.0-"
                return CallMeta(Union{}, Effects(), NoCallInfo())
            elseif VERSION < v"1.12.0-"
                return CallMeta(Union{}, Union{}, Effects(), NoCallInfo())
            else
                return Core.Compiler.Future{Core.Compiler.CallMeta}(CallMeta(Union{}, Union{}, Effects(), NoCallInfo()))
            end
        end
        @static if VERSION < v"1.11.0-"
            return CallMeta(
                Core.Const(true),
                Core.Compiler.EFFECTS_TOTAL,
                MethodResultPure(),
            )
        elseif VERSION < v"1.12.0-"
            return CallMeta(
                Core.Const(true),
                Union{},
                Core.Compiler.EFFECTS_TOTAL,
                MethodResultPure(),
            )
        else
            return Core.Compiler.Future{Core.Compiler.CallMeta}(CallMeta(
                Core.Const(true),
                Union{},
                Core.Compiler.EFFECTS_TOTAL,
                MethodResultPure(),
            ))
        end
    end
    
    if interp.broadcast_rewrite
        if f === Base.materialize && length(argtypes) == 2
            bcty = widenconst(argtypes[2])
    	    if Base.isconcretetype(bcty) && bcty <: Base.Broadcast.Broadcasted{<:Base.Broadcast.DefaultArrayStyle, Nothing} && bc_or_array_or_number_ty(bcty) && has_array(bcty)
        		ElType = ty_broadcast_getindex_eltype(interp, bcty)
        		if ElType !== Union{} && Base.isconcretetype(ElType)
        		    fn2 = Enzyme.Compiler.Interpreter.OverrideBCMaterialize{ElType}()
                    arginfo2 = ArgInfo(
                        fargs isa Nothing ? nothing : [:(fn2), fargs[2:end]...],
        	           [Core.Const(fn2), argtypes[2:end]...],
                    )

                    return Base.@invoke abstract_call_known(
                        interp::AbstractInterpreter,
                        fn2::Any,
                        arginfo2::ArgInfo,
                        si::StmtInfo,
                        sv::AbsIntState,
                        max_methods::Int,
                    )
                end
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
       
	if f === Base._foldl_impl &&  length(argtypes) == 4
	    
	    bcty = widenconst(argtypes[4])


            if widenconst(argtypes[3]) <: Base._InitialValue &&
	       bcty <: Base.Broadcast.Broadcasted{<:Base.Broadcast.DefaultArrayStyle} && ndims(bcty) >= 2 &&
	       bc_or_array_or_number_ty(bcty, false) && has_array(bcty, false)
           
                arginfo2 = ArgInfo(
                    fargs isa Nothing ? nothing :
                    [:(Enzyme.Compiler.Interpreter.override_bc_foldl), fargs[2:end]...],
                    [Core.Const(Enzyme.Compiler.Interpreter.override_bc_foldl), argtypes[2:end]...],
                )

                return Base.@invoke abstract_call_known(
                    interp::AbstractInterpreter,
                    Enzyme.Compiler.Interpreter.override_bc_foldl::Any,
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
