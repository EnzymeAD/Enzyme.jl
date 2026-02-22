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

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(Base.range_start_stop_length)},
        RT,
        start::Annotation{T},
        stop::Annotation{T},
        len::Annotation{<:Integer},
    ) where {T <: Base.IEEEFloat}
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            return Duplicated(
                func.val(start.val, stop.val, len.val),
                func.val(
                    start isa Const ? zero(start.val) : -start.dval,
                    stop isa Const ? zero(stop.val) : stop.dval,
                    len.val
                )
            )
        else
            return BatchDuplicated(
                func.val(start.val, stop.val, len.val),
                ntuple(
                    i -> func.val(
                        start isa Const ? zero(start.val) : -start.dval[i],
                        stop isa Const ? zero(stop.val) : stop.dval[i],
                        len.val,
                    ),
                    Val(EnzymeRules.width(config)),
                ),
            )
        end
    elseif EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            return func.val(
                start isa Const ? zero(start.val) : -start.dval,
                stop isa Const ? zero(stop.val) : stop.dval,
                len.val
            )
        else
            return ntuple(
                i -> func.val(
                    start isa Const ? zero(start.val) : -start.dval[i],
                    stop isa Const ? zero(stop.val) : stop.dval[i],
                    len.val,
                ),
                Val(EnzymeRules.width(config)),
            )
        end
    elseif EnzymeRules.needs_primal(config)
        return func.val(start.val, stop.val, len.val)
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
    if EnzymeRules.needs_primal(config)
        primal = func.val(start.val, stop.val, len.val)
    else
        primal = nothing
    end
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
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
    dstart = if start isa Const
        nothing
    elseif EnzymeRules.width(config) == 1
        T(dret.val.ref.hi) - T(dret.val.step.hi) / (len.val - 1)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            T(dret.val[i].ref.hi) - T(dret.val[i].step.hi) / (len.val - 1)
        end
    end

    dstop = if stop isa Const
        nothing
    elseif EnzymeRules.width(config) == 1
        T(dret.val.step.hi) / (len.val - 1)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            T(dret.val[i].step.hi) / (len.val - 1)
        end
    end

    return (dstart, dstop, nothing)
end
