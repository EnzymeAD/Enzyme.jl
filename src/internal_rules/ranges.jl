# Ranges
# Float64 ranges in Julia use bitwise `&` with higher precision
# to correct for numerical error, thus we put rules over the
# operations as this is not directly differentiable
function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{Colon},
        RT::Type{
            <:Union{Const, DuplicatedNoNeed, Duplicated, BatchDuplicated, BatchDuplicatedNoNeed},
        },
        start::Annotation{<:AbstractFloat},
        step::Annotation{<:AbstractFloat},
        stop::Annotation{<:AbstractFloat},
    )
    ret = func.val(start.val, step.val, stop.val)
    dstart = if start isa Const
        zero(eltype(ret))
    elseif start isa Duplicated || start isa DuplicatedNoNeed
        start.dval
    elseif start isa BatchDuplicated || start isa BatchDuplicatedNoNeed
        ntuple(i -> start.dval[i], Val(EnzymeRules.width(config)))
    else
        error(
            "Annotation type $(typeof(start)) not supported for range start. Please open an issue",
        )
    end

    dstep = if step isa Const
        zero(eltype(ret))
    elseif step isa Duplicated || step isa DuplicatedNoNeed
        step.dval
    elseif step isa BatchDuplicated || step isa BatchDuplicatedNoNeed
        ntuple(i -> step.dval[i], Val(EnzymeRules.width(config)))
    else
        error(
            "Annotation type $(typeof(start)) not supported for range step. Please open an issue",
        )
    end

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            return Duplicated(ret, range(dstart; step = dstep, length = length(ret)))
        else
            return BatchDuplicated(
                ret,
                ntuple(
                    i -> range(
                        dstart isa Number ? dstart : dstart[i];
                        step = dstep isa Number ? dstep : dstep[i],
                        length = length(ret),
                    ),
                    Val(EnzymeRules.width(config)),
                ),
            )
        end
    elseif EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            return range(dstart; step = dstep, length = length(ret))
        else
            return ntuple(
                i -> range(
                    dstart isa Number ? dstart : dstart[i];
                    step = dstep isa Number ? dstep : dstep[i],
                    length = length(ret),
                ),
                Val(EnzymeRules.width(config)),
            )
        end
    elseif EnzymeRules.needs_primal(config)
        return ret
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{Colon},
        ::Type{RT},
        start::Annotation{<:AbstractFloat},
        step::Annotation{<:AbstractFloat},
        stop::Annotation{<:AbstractFloat},
    ) where {RT <: Union{Active, Const}}

    if EnzymeRules.needs_primal(config)
        primal = func.val(start.val, step.val, stop.val)
    else
        primal = nothing
    end
    return EnzymeRules.AugmentedReturn{
        EnzymeRules.primal_type(config, RT),
        Nothing,
        Nothing,
    }(primal, nothing, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{Colon},
        dret::Const,
        tape::Nothing,
        start::Annotation{T1},
        step::Annotation{T2},
        stop::Annotation{T3},
    ) where {T1 <: AbstractFloat, T2 <: AbstractFloat, T3 <: AbstractFloat}
    dstart = if start isa Const
        nothing
    elseif EnzymeRules.width(config) == 1
        zero(T1)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            zero(T1)
        end
    end

    dstep = if step isa Const
        nothing
    elseif EnzymeRules.width(config) == 1
        zero(T2)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            zero(T2)
        end
    end

    dstop = if stop isa Const
        nothing
    elseif EnzymeRules.width(config) == 1
        zero(T3)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            zero(T3)
        end
    end

    return (dstart, dstep, dstop)
end


function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{Colon},
        dret,
        tape::Nothing,
        start::Annotation{T1},
        step::Annotation{T2},
        stop::Annotation{T3},
    ) where {T1 <: AbstractFloat, T2 <: AbstractFloat, T3 <: AbstractFloat}

    dstart = if start isa Const
        nothing
    elseif EnzymeRules.width(config) == 1
        T1(dret.val.ref.hi)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            T1(dret.val[i].ref.hi)
        end
    end

    dstep = if step isa Const
        nothing
    elseif EnzymeRules.width(config) == 1
        T2(dret.val.step.hi)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            T2(dret.val[i].step.hi)
        end
    end

    dstop = if stop isa Const
        nothing
    elseif EnzymeRules.width(config) == 1
        zero(T3)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            zero(T3)
        end
    end

    return (dstart, dstep, dstop)
end

# The StepRangeLen returned by `range_start_stop_length` stores its value at
# index `offset` (not necessarily the first element) in `ref`. Indexing into the
# shadow reuses the integer `offset` of the primal, so the tangent of `ref` must be
# taken at that same index.
function _range_sslen_shadow(primal::StepRangeLen, dstart, dstop)
    len = primal.len
    off = primal.offset
    if len < 2
        dref = dstart
        dstep = zero(dstart)
    else
        dstep = (dstop - dstart) / (len - 1)
        dref = dstart + (off - 1) * dstep
    end
    R = typeof(primal.ref)
    S = typeof(primal.step)
    return typeof(primal)(R(dref), S(dstep), len, off)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(Base.range_start_stop_length)},
        RT,
        start::Annotation{T},
        stop::Annotation{T},
        len::Annotation{<:Integer},
    ) where {T <: Base.IEEEFloat}
    primal = func.val(start.val, stop.val, len.val)
    if EnzymeRules.needs_shadow(config)
        shadow = if EnzymeRules.width(config) == 1
            _range_sslen_shadow(
                primal,
                start isa Const ? zero(start.val) : start.dval,
                stop isa Const ? zero(stop.val) : stop.dval,
            )
        else
            ntuple(
                i -> _range_sslen_shadow(
                    primal,
                    start isa Const ? zero(start.val) : start.dval[i],
                    stop isa Const ? zero(stop.val) : stop.dval[i],
                ),
                Val(EnzymeRules.width(config)),
            )
        end
        if EnzymeRules.needs_primal(config)
            if EnzymeRules.width(config) == 1
                return Duplicated(primal, shadow)
            else
                return BatchDuplicated(primal, shadow)
            end
        else
            return shadow
        end
    elseif EnzymeRules.needs_primal(config)
        return primal
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(Base.range_start_stop_length)},
        ::Type{RT},
        start::Annotation{T},
        stop::Annotation{T},
        len::Annotation{<:Base.Integer},
    ) where {RT, T <: Base.IEEEFloat}
    r = func.val(start.val, stop.val, len.val)
    primal = EnzymeRules.needs_primal(config) ? r : nothing
    # the reverse pass needs to know at which index `ref` is stored
    return EnzymeRules.AugmentedReturn(primal, nothing, r.offset)
end

# Float16/Float32 ranges store `ref`/`step` as plain Float64, Float64 ones as TwicePrecision
_range_hi(x::Base.TwicePrecision) = x.hi
_range_hi(x::AbstractFloat) = x

# Adjoint of `ref = start + (offset - 1) * step` and `step = (stop - start) / (len - 1)`.
function _range_sslen_pullback(::Type{T}, dr::StepRangeLen, len, off) where {T}
    dref = T(_range_hi(dr.ref))
    if len < 2
        return dref, zero(T)
    end
    dstep = T(_range_hi(dr.step)) + (off - 1) * dref
    dstop = dstep / (len - 1)
    return dref - dstop, dstop
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(Base.range_start_stop_length)},
        dret,
        tape,
        start::Annotation{T},
        stop::Annotation{T},
        len::Annotation{T3},
    ) where {T <: Base.IEEEFloat, T3 <: Integer}
    off = tape
    grads = if EnzymeRules.width(config) == 1
        _range_sslen_pullback(T, dret.val, len.val, off)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            _range_sslen_pullback(T, dret.val[i], len.val, off)
        end
    end

    dstart = if start isa Const
        nothing
    elseif EnzymeRules.width(config) == 1
        grads[1]
    else
        ntuple(i -> grads[i][1], Val(EnzymeRules.width(config)))
    end

    dstop = if stop isa Const
        nothing
    elseif EnzymeRules.width(config) == 1
        grads[2]
    else
        ntuple(i -> grads[i][2], Val(EnzymeRules.width(config)))
    end

    return (dstart, dstop, nothing)
end
