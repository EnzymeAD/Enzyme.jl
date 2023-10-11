using Random

function EnzymeRules.inactive(::typeof(Base.CoreLogging.logmsg_code), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.CoreLogging.shouldlog), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.CoreLogging.current_logger), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.CoreLogging.current_logger_for_env), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.fixup_stdlib_path), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.CoreLogging.handle_message), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.CoreLogging.logging_error), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.to_tuple_type), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.println), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.print), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.show), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.flush), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.string), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.repr), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.print_to_string), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.Threads.threadid), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.Threads.nthreads), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.eps), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.nextfloat), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.prevfloat), args...)
    return nothing
end
function EnzymeRules.inactive(::Type{Base.Val}, args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Core.kwfunc), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Random.rand), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Random.rand!), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Random.randn), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Random.default_rng), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Random.seed!), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.thisind), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.nextind), args...)
    return nothing
end
function EnzymeRules.inactive_noinl(::typeof(Core.Compiler.return_type), args...)
    return nothing
end
function EnzymeRules.inactive_noinl(::typeof(Base.Broadcast.combine_eltypes), args...)
    return nothing
end
function EnzymeRules.inactive_noinl(::typeof(Base.typejoin), args...)
    return nothing
end
function EnzymeRules.inactive_noinl(::typeof(Base.size), args...)
    return nothing
end

@inline EnzymeRules.inactive_type(v::Type{Nothing}) = true
@inline EnzymeRules.inactive_type(v::Type{Union{}}) = true
@inline EnzymeRules.inactive_type(v::Type{T}) where {T<:Integer} = true
@inline EnzymeRules.inactive_type(v::Type{Function}) = true
@inline EnzymeRules.inactive_type(v::Type{T}) where {T<:DataType} = true
@inline EnzymeRules.inactive_type(v::Type{T}) where {T<:Module} = true
@inline EnzymeRules.inactive_type(v::Type{T}) where {T<:AbstractString} = true

# Note all of these forward mode definitions do not support runtime activity as
# the do not keep the primal if shadow(x.y) == primal(x.y)
function EnzymeRules.forward(::Const{typeof(Base.deepcopy)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated)
    return deepcopy(x.dval)
end

function EnzymeRules.forward(::Const{typeof(Base.deepcopy)}, ::Type{<:BatchDuplicatedNoNeed}, x::BatchDuplicated{T, N}) where {T, N}
    ntuple(Val(N)) do _
        deepcopy(x.dval)
    end
end

# Deepcopy preserving the primal if runtime inactive
@inline function deepcopy_rtact(copied::RT, primal::RT, seen::IdDict, shadow::RT) where {RT <: Integer}
    return Base.deepcopy_internal(shadow, seen)
end
@inline function deepcopy_rtact(copied::RT, primal::RT, seen::IdDict, shadow::RT) where {RT <: AbstractFloat}
    return Base.deepcopy_internal(shadow, seen)
end
@inline function deepcopy_rtact(copied::RT, primal::RT, seen::IdDict, shadow::RT) where {RT <: Array}
    if !haskey(seen, shadow)
        if primal === shadow
            return seen[shadow] = copied
        end
        newa = RT(undef, size(shadow))
        seen[shadow] = newa
        for i in eachindex(shadow)
            @inbounds newa[i] = deepcopy_rtact(copied[i], primal[i], seen, shadow[i])
        end
    end
    return seen[shadow]
end

function EnzymeRules.forward(func::Const{typeof(Base.deepcopy)}, ::Type{<:Duplicated}, x::Duplicated)
    primal = func.val(x.val)
    return Duplicated(primal, deepcopy_rtact(primal, x.val, IdDict(), x.dval))
end

function EnzymeRules.forward(func::Const{typeof(Base.deepcopy)}, ::Type{<:BatchDuplicated}, x::BatchDuplicated{T, N}) where {T,N}
    primal = func.val(x.val)
    return BatchDuplicated(primal, ntuple(Val(N)) do i
        deepcopy_rtact(primal, x.val, IdDict(), x.dval[i])
    end)
end

function EnzymeRules.augmented_primal(config, func::Const{typeof(Base.deepcopy)}, ::Type{RT}, x::Annotation{Ty}) where {RT, Ty}
    primal = if EnzymeRules.needs_primal(config)
        func.val(x.val)
    else
        nothing
    end

    @assert !(typeof(x) <: Active)

    source = if EnzymeRules.needs_primal(config)
        primal
    else
        x.val
    end

    shadow = ntuple(Val(EnzymeRules.width(config))) do _
        Base.@_inline_meta
        Enzyme.Compiler.make_zero(Core.Typeof(source), IdDict(), source,
            #=copy_if_inactive=#Val(!EnzymeRules.needs_primal(config))
        )
    end

    if EnzymeRules.width(config) == 1
        shadow = shadow[1]
    end

    return EnzymeRules.AugmentedReturn(primal, shadow, shadow)
end


@inline function accumulate_into(into::RT, seen::IdDict, from::RT)::Tuple{RT,RT} where {RT<:Array}
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

@inline function accumulate_into(into::RT, seen::IdDict, from::RT)::Tuple{RT,RT} where {RT<:AbstractFloat}
    if !haskey(seen, into)
        seen[into] = (into+from, RT(0))
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

function EnzymeRules.reverse(config, func::Const{typeof(Base.deepcopy)}, ::Type{RT}, shadow, x::Annotation{Ty}) where {RT, Ty}
    if EnzymeRules.width(config) == 1
        accumulate_into(x.dval, IdDict(), shadow)
    else
        for i in 1:EnzymeRules.width(config)
            accumulate_into(x.dval[i], IdDict(), shadow[i])
        end
    end

    return (nothing,)
end

@inline function pmap_fwd(idx, tapes, thunk::ThunkTy, f::F, fargs::Vararg{Annotation, N}) where {ThunkTy, F, N}
    @inbounds tapes[idx] = thunk(f, Const(idx), fargs...)
end

function EnzymeRules.augmented_primal(config, func::Const{typeof(Enzyme.pmap)}, ::Type{Const{Nothing}}, body::BodyTy, count, args::Vararg{Annotation, N}) where {BodyTy, N}

    config2 = ReverseModeSplit{false, false, EnzymeRules.width(config), EnzymeRules.overwritten(config)[2:end],InlineABI}()
    fwd_thunk, rev_thunk = autodiff_thunk(config2, BodyTy, Const, typeof(count), map(typeof, args)...)

    TapeType = EnzymeRules.tape_type(fwd_thunk)

    tapes = Vector{TapeType}(undef, count.val)

    Enzyme.pmap(count, pmap_fwd, tapes, fwd_thunk, body, args...)
    return AugmentedReturn(nothing, nothing, tapes)
end

@inline function pmap_rev(idx, tapes, thunk::ThunkTy, f::F, fargs::Vararg{Annotation, N}) where {ThunkTy, F, N}
    st = @inbounds tapes[idx]
    thunk(f, Const(idx), fargs..., st)
end

function EnzymeRules.reverse(config, func::Const{typeof(Enzyme.pmap)}, ::Type{Const{Nothing}}, tapes, body::BodyTy, count, args::Vararg{Annotation, N}) where {BodyTy, N}

    config2 = ReverseModeSplit{false, false, EnzymeRules.width(config), EnzymeRules.overwritten(config)[2:end],InlineABI}()
    fwd_thunk, rev_thunk =  autodiff_thunk(config2, BodyTy, Const, typeof(count), map(typeof, args)...)

    Enzyme.pmap(count, pmap_rev, tapes, rev_thunk, body, args...)

    return ntuple(Val(2+length(args))) do _
        Base.@_inline_meta
        nothing
    end
end