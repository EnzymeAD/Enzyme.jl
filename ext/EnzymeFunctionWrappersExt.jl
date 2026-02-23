module EnzymeFunctionWrappersExt

using FunctionWrappers: FunctionWrapper
using EnzymeCore
using EnzymeCore.EnzymeRules
using Enzyme

# Helper to extract the raw function from a FunctionWrapper
@inline unwrap_fw(fw::FunctionWrapper) = fw.obj[]

# Helper to reconstruct an annotation with a cached primal value
@inline _reconstruct_arg(arg::Const, cached, overwritten::Bool) = arg
@inline function _reconstruct_arg(arg::Duplicated, cached, overwritten::Bool)
    overwritten && cached !== nothing ? Duplicated(cached, arg.dval) : arg
end
@inline function _reconstruct_arg(arg::BatchDuplicated, cached, overwritten::Bool)
    overwritten && cached !== nothing ? BatchDuplicated(cached, arg.dval) : arg
end
@inline _reconstruct_arg(arg::Active, cached, overwritten::Bool) = arg

# Helper for type-stable reverse return values
@inline _reverse_val(::Active{T}, grad, dret_val) where {T} = (grad * dret_val)::T
@inline _reverse_val(::Const, grad, dret_val) = nothing
@inline _reverse_val(::Duplicated, grad, dret_val) = nothing
@inline _reverse_val(::BatchDuplicated, grad, dret_val) = nothing

# ---------------------------------------------------------------------------
# Forward mode rule
# ---------------------------------------------------------------------------
# Single rule for both IIP (Nothing return) and OOP FunctionWrappers.
# Extracts the wrapped function and delegates to autodiff_deferred.
function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    func::Const{<:FunctionWrapper},
    RT::Type{<:Annotation},
    args::Annotation...,
)
    raw_f = unwrap_fw(func.val)

    # For IIP functions (Const{Nothing} return), needs_shadow is false but we
    # still must propagate tangents into argument shadow arrays via AD.
    if RT <: Const
        # IIP or inactive return — run AD for tangent propagation into arg shadows
        Enzyme.autodiff_deferred(Forward, Const(raw_f), Const{eltype(RT)}, args...)
        if EnzymeRules.needs_primal(config)
            return raw_f(map(x -> x.val, args)...)
        else
            return nothing
        end
    end

    # OOP: shadow is needed. Always use Duplicated for autodiff_deferred
    # (it rejects DuplicatedNoNeed).
    RealRt = eltype(RT)
    if EnzymeRules.needs_primal(config)
        res = Enzyme.autodiff_deferred(ForwardWithPrimal, Const(raw_f), Duplicated, args...)
        # autodiff ForwardWithPrimal returns (derivs, primal)
        if EnzymeRules.width(config) == 1
            return Duplicated(res[2]::RealRt, res[1]::RealRt)
        else
            return BatchDuplicated(res[2]::RealRt, res[1]::NTuple{EnzymeRules.width(config),RealRt})
        end
    else
        res = Enzyme.autodiff_deferred(Forward, Const(raw_f), Duplicated, args...)
        # autodiff Forward returns (derivs,)
        if EnzymeRules.width(config) == 1
            return res[1]::RealRt
        else
            return res[1]::NTuple{EnzymeRules.width(config),RealRt}
        end
    end
end

# ---------------------------------------------------------------------------
# Reverse mode rules
# ---------------------------------------------------------------------------

# augmented_primal: execute the forward pass, cache data for reverse
function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    func::Const{<:FunctionWrapper{Ret}},
    RT::Type{<:Annotation},
    args::Annotation...,
) where {Ret}
    raw_f = unwrap_fw(func.val)
    ow = EnzymeRules.overwritten(config)
    nargs = length(args)

    # Cache copies of overwritten mutable args (needed for reverse pass)
    cached_args = ntuple(Val(nargs)) do i
        Base.@_inline_meta
        # ow[1] is the function itself, ow[i+1] is the i-th argument
        if ow[i + 1] && !(args[i] isa Const)
            deepcopy(args[i].val)
        else
            nothing
        end
    end

    # Execute the primal
    primal_result = raw_f(map(x -> x.val, args)...)

    primal = if EnzymeRules.needs_primal(config)
        primal_result
    else
        nothing
    end

    shadow = if EnzymeRules.needs_shadow(config)
        if Ret === Nothing
            nothing
        else
            if EnzymeRules.width(config) == 1
                Enzyme.make_zero(primal_result)
            else
                ntuple(Val(EnzymeRules.width(config))) do j
                    Base.@_inline_meta
                    Enzyme.make_zero(primal_result)
                end
            end
        end
    else
        nothing
    end

    tape = (raw_f, cached_args)
    return EnzymeRules.AugmentedReturn(primal, shadow, tape)
end

# reverse for IIP (Nothing return): accumulate gradients into dval arrays
function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    func::Const{<:FunctionWrapper{Nothing}},
    ::Type{<:Const{Nothing}},
    tape,
    args::Annotation...,
)
    raw_f, cached_args = tape
    ow = EnzymeRules.overwritten(config)
    nargs = length(args)

    new_args = ntuple(Val(nargs)) do i
        Base.@_inline_meta
        _reconstruct_arg(args[i], cached_args[i], ow[i + 1])
    end

    Enzyme.autodiff_deferred(Reverse, Const(raw_f), Const{Nothing}, new_args...)

    return ntuple(Val(nargs)) do i
        Base.@_inline_meta
        nothing
    end
end

# reverse for OOP with Active return: return scaled per-arg gradients
function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    func::Const{<:FunctionWrapper{Ret}},
    dret::Active,
    tape,
    args::Annotation...,
) where {Ret}
    raw_f, cached_args = tape
    ow = EnzymeRules.overwritten(config)
    nargs = length(args)

    new_args = ntuple(Val(nargs)) do i
        Base.@_inline_meta
        _reconstruct_arg(args[i], cached_args[i], ow[i + 1])
    end

    # autodiff_deferred(Reverse, ..., Active, args...) returns ((grad1, grad2, ...),)
    res = Enzyme.autodiff_deferred(Reverse, Const(raw_f), Active, new_args...)
    grads = res[1]

    return ntuple(Val(nargs)) do i
        Base.@_inline_meta
        _reverse_val(args[i], grads[i], dret.val)
    end
end

# reverse for OOP with Duplicated/Const return type (non-Active)
function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    func::Const{<:FunctionWrapper{Ret}},
    dret::Type{<:Annotation},
    tape,
    args::Annotation...,
) where {Ret}
    if !(dret <: Const)
        raw_f, cached_args = tape
        ow = EnzymeRules.overwritten(config)
        nargs = length(args)

        new_args = ntuple(Val(nargs)) do i
            Base.@_inline_meta
            _reconstruct_arg(args[i], cached_args[i], ow[i + 1])
        end

        Enzyme.autodiff_deferred(Reverse, Const(raw_f), dret, new_args...)
    end

    return ntuple(Val(length(args))) do i
        Base.@_inline_meta
        nothing
    end
end

end # module
