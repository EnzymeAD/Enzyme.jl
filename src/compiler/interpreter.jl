module Interpreter
import Enzyme: API
import ..Enzyme
import ..EnzymeRules

@static if VERSION ≥ v"1.11.0-DEV.1498"
    import Core.Compiler: get_inference_world
    using Base: get_world_counter
else
    import Core.Compiler: get_world_counter, get_world_counter as get_inference_world
end
struct EnzymeMeta
    mode::API.CDerivativeMode
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

import GPUCompiler: GPUInterpreter, NoInlineCallInfo, AlwaysInlineCallInfo
function inlining_handler(meta::EnzymeMeta, interp::GPUInterpreter, @nospecialize(atype), callinfo)
    method_table = Core.Compiler.method_table(interp)
    world = get_inference_world(interp)
    
    specTypes = simplify_kw(atype)
    if is_primitive_func(specTypes)
        return NoInlineCallInfo(callinfo, atype, :primitive)
    elseif is_alwaysinline_func(specTypes)
        return AlwaysInlineCallInfo(callinfo, atype)
    elseif EnzymeRules.is_inactive_from_sig(specTypes; world, method_table)
        return NoInlineCallInfo(callinfo, atype, :inactive)
    elseif meta.mode == API.DEM_ForwardMode
        if EnzymeRules.has_frule_from_sig(specTypes; world, method_table)
            return NoInlineCallInfo(callinfo, atype, :frule)
        end
    elseif meta.mode == API.DEM_ReverseModeCombined || 
           meta.mode == API.DEM_ReverseModePrimal ||
           meta.mode == API.DEM_ReverseModeGradient 
        if EnzymeRules.has_rrule_from_sig(specTypes; world, method_table)
            return NoInlineCallInfo(callinfo, atype, :rrule)
        end
    end
    return nothing
end

struct AutodiffCallInfo <: CC.CallInfo
    # ...
    info::CC.CallInfo
end

import GPUCompiler: abstract_call_known
import CC: CallMeta, Effects, NoCallInfo
function abstract_call_known(meta::EnzymeMeta, interp::GPUInterpreter, @nospecialize(f),
                             arginfo::ArgInfo, si::StmtInfo, sv::AbsIntState, max_methods::Int)
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
            return CallMeta(Core.Const(true), CC.EFFECTS_TOTAL, MethodResultPure())
        else
            return CallMeta(Core.Const(true), Union{}, CC.EFFECTS_TOTAL, MethodResultPure(),)
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
            # FIXME: Use AutodiffCallInfo and a custom inlining handler
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
    return nothing
end

end
