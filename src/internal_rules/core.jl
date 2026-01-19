# Note all of these forward mode definitions do not support runtime activity as
# they do not keep the primal if shadow(x.y) == primal(x.y)
function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    ::Const{typeof(Base.deepcopy)},
    ::Type{<:DuplicatedNoNeed},
    x::Duplicated,
)
    return deepcopy(x.dval)
end

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    ::Const{typeof(Base.deepcopy)},
    ::Type{<:BatchDuplicatedNoNeed},
    x::BatchDuplicated{T,N},
) where {T,N}
    ntuple(Val(N)) do _
        deepcopy(x.dval)
    end
end

# Deepcopy preserving the primal if runtime inactive
@inline function deepcopy_rtact(
    copied::RT,
    primal::RT,
    seen::Union{IdDict,Nothing},
    shadow::RT,
)::RT where RT
    rt = Enzyme.Compiler.active_reg_nothrow(RT)
    if rt == Enzyme.Compiler.ActiveState || rt == Enzyme.Compiler.AnyState
        if seen === nothing
            return Base.deepcopy(shadow)
        else
            return Base.deepcopy_internal(shadow, seen)
        end
    else
        if seen !== nothing && haskey(seen, shadow)
            return seen[shadow]
        end
        if primal === shadow
            if seen !== nothing
                seen[shadow] = copied
            end
            return copied
        end

        if RT <: Array
            newa = similar(primal, size(shadow))
            if seen === nothing
                seen = IdDict()
            end
            seen[shadow] = newa
            for i in eachindex(shadow)
                @inbounds newa[i] = deepcopy_rtact(copied[i], primal[i], seen, shadow[i])
            end
            return newa
        end
        throw(AssertionError("Unimplemented deepcopy with runtime activity for type $RT"))
    end
end

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    func::Const{typeof(Base.deepcopy)},
    ::Type{<:Duplicated},
    x::Duplicated,
)
    primal = func.val(x.val)
    return Duplicated(primal, deepcopy_rtact(primal, x.val, nothing, x.dval))
end

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    func::Const{typeof(Base.deepcopy)},
    ::Type{<:BatchDuplicated},
    x::BatchDuplicated{T,N},
) where {T,N}
    primal = func.val(x.val)
    return BatchDuplicated(primal, ntuple(Val(N)) do i
        deepcopy_rtact(primal, x.val, nothing, x.dval[i])
    end)
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    func::Const{typeof(Base.deepcopy)},
    ::Type{RT},
    x::Annotation{Ty},
) where {RT,Ty}
    primal = if EnzymeRules.needs_primal(config)
        func.val(x.val)
    else
        nothing
    end

    source = if EnzymeRules.needs_primal(config)
        primal
    else
        x.val
    end

    shadow = if EnzymeRules.needs_shadow(config)
        @assert !(x isa Active)
        if EnzymeRules.width(config) == 1
            Enzyme.make_zero(
                source,
                Val(!EnzymeRules.needs_primal(config)),                #=copy_if_inactive=#
            )
        else
            ntuple(Val(EnzymeRules.width(config))) do _
                Base.@_inline_meta
                Enzyme.make_zero(
                    source,
                    Val(!EnzymeRules.needs_primal(config)),                    #=copy_if_inactive=#
                )
            end
        end
    else
        nothing
    end

    return EnzymeRules.AugmentedReturn(primal, shadow, shadow)
end


@inline function accumulate_into(
    into::RT,
    seen::IdDict,
    from::RT,
)::Tuple{RT,RT} where {RT<:Array}
    if Enzyme.Compiler.guaranteed_const(RT)
        return (into, from)
    end
    if !haskey(seen, into)
        seen[into] = (into, from)
        for i in eachindex(from)
            tup = accumulate_into(into[i], seen, from[i])
            @inbounds into[i] = tup[1]
            @inbounds from[i] = tup[2]
        end
    end
    return seen[into]
end

@inline function accumulate_into(
    into::RT,
    seen::IdDict,
    from::RT,
)::Tuple{RT,RT} where {RT<:AbstractFloat}
    if !haskey(seen, into)
        seen[into] = (into + from, RT(0))
    end
    return seen[into]
end

@inline function accumulate_into(into::RT, seen::IdDict, from::RT)::Tuple{RT,RT} where {RT}
    if Enzyme.Compiler.guaranteed_const(RT)
        return (into, from)
    end
    if !haskey(seen, into)
        throw(AssertionError("Unknown type to accumulate into: $RT"))
    end
    return seen[into]
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    func::Const{typeof(Base.deepcopy)},
    ::Type{RT},
    shadow,
    x::Annotation{Ty},
) where {RT,Ty}
    @assert !(x isa Active)

    if EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            accumulate_into(x.dval, IdDict(), shadow)
        else
            for i = 1:EnzymeRules.width(config)
                accumulate_into(x.dval[i], IdDict(), shadow[i])
            end
        end
    end

    return (nothing,)
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    func::Const{typeof(Base.deepcopy)},
    dret::Active,
    shadow,
    x::Annotation,
)
    return (dret.val,)
end


@inline function pmap_fwd(
    idx,
    tapes::Vector,
    thunk::ThunkTy,
    f::F,
    fargs::Vararg{Annotation,N},
) where {ThunkTy,F,N}
    @inbounds tapes[idx] = thunk(f, Const(idx), fargs...)[1]
end

@inline function pmap_fwd(
    idx,
    tapes::Ptr,
    thunk::ThunkTy,
    f::F,
    fargs::Vararg{Annotation,N},
) where {ThunkTy,F,N}
    unsafe_store!(tapes, thunk(f, Const(idx), fargs...)[1], idx)
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    func::Const{typeof(Enzyme.pmap)},
    ::Type{Const{Nothing}},
    body::BodyTy,
    count,
    args::Vararg{Annotation,N},
) where {BodyTy,N}

    config2 = ReverseModeSplit{
        false,
        false,
        EnzymeRules.runtime_activity(config),
        EnzymeRules.strong_zero(config),
        EnzymeRules.width(config),
        EnzymeRules.overwritten(config)[2:end],
        InlineABI,
        false,
        false,
        false
    }()
    fwd_thunk, rev_thunk =
        autodiff_thunk(config2, BodyTy, Const, typeof(count), map(typeof, args)...)

    TapeType = EnzymeRules.tape_type(fwd_thunk)

    tapes = if Enzyme.Compiler.any_jltypes(TapeType)
        Vector{TapeType}(undef, count.val)
    else
        Base.unsafe_convert(Ptr{TapeType}, Libc.malloc(sizeof(TapeType) * count.val))
    end

    Enzyme.pmap(pmap_fwd, count.val, tapes, fwd_thunk, body, args...)
    return EnzymeRules.AugmentedReturn(nothing, nothing, tapes)
end

@inline function pmap_rev(
    idx,
    tapes::Vector,
    thunk::ThunkTy,
    f::F,
    fargs::Vararg{Annotation,N},
) where {ThunkTy,F,N}
    thunk(f, Const(idx), fargs..., @inbounds tapes[idx])
end

@inline function pmap_rev(
    idx,
    tapes::Ptr,
    thunk::ThunkTy,
    f::F,
    fargs::Vararg{Annotation,N},
) where {ThunkTy,F,N}
    thunk(f, Const(idx), fargs..., unsafe_load(tapes, idx))
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    func::Const{typeof(Enzyme.pmap)},
    ::Type{Const{Nothing}},
    tapes,
    body::BodyTy,
    count,
    args::Vararg{Annotation,N},
) where {BodyTy,N}

    config2 = ReverseModeSplit{
        false,
        false,
        EnzymeRules.runtime_activity(config),
        EnzymeRules.strong_zero(config),
        EnzymeRules.width(config),
        EnzymeRules.overwritten(config)[2:end],
        InlineABI,
        false,
        false,
        false
    }()
    fwd_thunk, rev_thunk =
        autodiff_thunk(config2, BodyTy, Const, typeof(count), map(typeof, args)...)

    Enzyme.pmap(pmap_rev, count.val, tapes, rev_thunk, body, args...)

    TapeType = EnzymeRules.tape_type(fwd_thunk)

    if !Enzyme.Compiler.any_jltypes(TapeType)
        Libc.free(tapes)
    end

    return ntuple(Returns(nothing), Val(2 + length(args)))
end

# Force a rule around hvcat_fill as it is type unstable if the tuple is not of the same type (e.g., int, float, int, float)
function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    func::Const{typeof(Base.hvcat_fill!)},
    ::Type{RT},
    out::Annotation{AT},
    inp::Annotation{BT},
) where {RT,AT<:Array,BT<:Tuple}
    primal = if EnzymeRules.needs_primal(config)
        out.val
    else
        nothing
    end
    shadow = if EnzymeRules.needs_shadow(config)
        out.dval
    else
        nothing
    end
    func.val(out.val, inp.val)
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    func::Const{typeof(Base.hvcat_fill!)},
    ::Type{RT},
    _,
    out::Annotation{AT},
    inp::Annotation{BT},
) where {RT,AT<:Array,BT<:Tuple}
    nr, nc = size(out.val, 1), size(out.val, 2)
    for b = 1:EnzymeRules.width(config)
        da = if EnzymeRules.width(config) == 1
            out.dval
        else
            out.dval[b]
        end
        i = 1
        j = 1
        if (typeof(inp) <: Active)
            dinp = ntuple(Val(length(inp.val))) do k
                Base.@_inline_meta
                res = da[i, j]
                da[i, j] = 0
                j += 1
                if j == nc + 1
                    i += 1
                    j = 1
                end
                T = BT.parameters[k]
                if T <: AbstractFloat
                    T(res)
                else
                    T(0)
                end
            end
            return (nothing, dinp)::Tuple{Nothing,BT}
        end
    end
    return (nothing, nothing)
end

function EnzymeRules.forward(config, ::Const{typeof(Base.finalizer)}, _, f::Const, o)
    f = f.val
    Base.finalizer(f, o.val)
    if EnzymeRules.width(config) == 1
        Base.finalizer(f, o.dval)
    else
        foreach(o.dval) do dv
            Base.finalizer(f, dv)
        end
    end

    if EnzymeRules.needs_primal(config)
        return o
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(config, ::Const{typeof(Base.finalizer)}, _, f::Const, o)
    @assert !(o isa Active)
    f = f.val
    Base.finalizer(f, o.val)
    if EnzymeRules.width(config) == 1
        Base.finalizer(f, o.dval)
    else
        foreach(o.dval) do dv
            Base.finalizer(f, dv)
        end
    end

    primal = EnzymeRules.needs_primal(config) ? o.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? o.dval : nothing

    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

function EnzymeRules.reverse(config, ::Const{typeof(Base.finalizer)}, dret, tape, f::Const, o)
    # No-op
    return (nothing, nothing)
end
