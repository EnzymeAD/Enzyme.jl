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
function EnzymeRules.inactive(::typeof(Random.rand), ::Random.AbstractRNG, ::Random.Sampler)
    return nothing
end
function EnzymeRules.inactive(::typeof(Random.rand!), ::Random.AbstractRNG, ::Random.Sampler, ::AbstractArray)
    return nothing
end
function EnzymeRules.inactive(::typeof(Random.randn), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Random.randn!), args...)
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
function EnzymeRules.inactive_noinl(
    ::typeof(Base.setindex!), ::IdDict{K,V}, ::K, ::V
) where {K,V<:Integer}
    return nothing
end

if VERSION >= v"1.9"
    Enzyme.EnzymeRules.inactive_noinl(::typeof(Core._compute_sparams), args...) = nothing
end

@inline EnzymeRules.inactive_type(v::Type{Nothing}) = true
@inline EnzymeRules.inactive_type(v::Type{Union{}}) = true
@inline EnzymeRules.inactive_type(v::Type{Char}) = true
@inline EnzymeRules.inactive_type(v::Type{T}) where {T<:Integer} = true
@inline EnzymeRules.inactive_type(v::Type{Function}) = true
@inline EnzymeRules.inactive_type(v::Type{T}) where {T<:DataType} = true
@inline EnzymeRules.inactive_type(v::Type{T}) where {T<:Module} = true
@inline EnzymeRules.inactive_type(v::Type{T}) where {T<:AbstractString} = true

@inline width(::Duplicated) = 1
@inline width(::BatchDuplicated{T,N}) where {T,N} = N
@inline width(::DuplicatedNoNeed) = 1
@inline width(::BatchDuplicatedNoNeed{T,N}) where {T,N} = N

@inline width(::Type{Duplicated{T}}) where {T} = 1
@inline width(::Type{BatchDuplicated{T,N}}) where {T,N} = N
@inline width(::Type{DuplicatedNoNeed{T}}) where {T} = 1
@inline width(::Type{BatchDuplicatedNoNeed{T,N}}) where {T,N} = N

# Note all of these forward mode definitions do not support runtime activity as
# the do not keep the primal if shadow(x.y) == primal(x.y)
function EnzymeRules.forward(
    ::Const{typeof(Base.deepcopy)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated
)
    return deepcopy(x.dval)
end

function EnzymeRules.forward(
    ::Const{typeof(Base.deepcopy)}, ::Type{<:BatchDuplicatedNoNeed}, x::BatchDuplicated{T,N}
) where {T,N}
    ntuple(Val(N)) do _
        deepcopy(x.dval)
    end
end

# Deepcopy preserving the primal if runtime inactive
@inline function deepcopy_rtact(
    copied::RT, primal::RT, seen::IdDict, shadow::RT
) where {RT<:Union{Integer,Char}}
    return Base.deepcopy_internal(shadow, seen)
end
@inline function deepcopy_rtact(
    copied::RT, primal::RT, seen::IdDict, shadow::RT
) where {RT<:AbstractFloat}
    return Base.deepcopy_internal(shadow, seen)
end
@inline function deepcopy_rtact(
    copied::RT, primal::RT, seen::IdDict, shadow::RT
) where {RT<:Array}
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

function EnzymeRules.forward(
    func::Const{typeof(Base.deepcopy)}, ::Type{<:Duplicated}, x::Duplicated
)
    primal = func.val(x.val)
    return Duplicated(primal, deepcopy_rtact(primal, x.val, IdDict(), x.dval))
end

function EnzymeRules.forward(
    func::Const{typeof(Base.deepcopy)}, ::Type{<:BatchDuplicated}, x::BatchDuplicated{T,N}
) where {T,N}
    primal = func.val(x.val)
    return BatchDuplicated(
        primal,
        ntuple(Val(N)) do i
            deepcopy_rtact(primal, x.val, IdDict(), x.dval[i])
        end,
    )
end

function EnzymeRules.augmented_primal(
    config, func::Const{typeof(Base.deepcopy)}, ::Type{RT}, x::Annotation{Ty}
) where {RT,Ty}
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
        Enzyme.make_zero(
            source,
            Val(!EnzymeRules.needs_primal(config)),            #=copy_if_inactive=#
        )
    end

    if EnzymeRules.width(config) == 1
        shadow = shadow[1]
    end

    return EnzymeRules.AugmentedReturn(primal, shadow, shadow)
end

@inline function accumulate_into(
    into::RT, seen::IdDict, from::RT
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
    into::RT, seen::IdDict, from::RT
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
    config, func::Const{typeof(Base.deepcopy)}, ::Type{RT}, shadow, x::Annotation{Ty}
) where {RT,Ty}
    if EnzymeRules.width(config) == 1
        accumulate_into(x.dval, IdDict(), shadow)
    else
        for i in 1:EnzymeRules.width(config)
            accumulate_into(x.dval[i], IdDict(), shadow[i])
        end
    end

    return (nothing,)
end

@inline function pmap_fwd(
    idx, tapes::Vector, thunk::ThunkTy, f::F, fargs::Vararg{Annotation,N}
) where {ThunkTy,F,N}
    @inbounds tapes[idx] = thunk(f, Const(idx), fargs...)[1]
end

@inline function pmap_fwd(
    idx, tapes::Ptr, thunk::ThunkTy, f::F, fargs::Vararg{Annotation,N}
) where {ThunkTy,F,N}
    return unsafe_store!(tapes, thunk(f, Const(idx), fargs...)[1], idx)
end

function EnzymeRules.augmented_primal(
    config,
    func::Const{typeof(Enzyme.pmap)},
    ::Type{Const{Nothing}},
    body::BodyTy,
    count,
    args::Vararg{Annotation,N},
) where {BodyTy,N}
    config2 = ReverseModeSplit{
        false,
        false,
        EnzymeRules.width(config),
        EnzymeRules.overwritten(config)[2:end],
        InlineABI,
    }()
    fwd_thunk, rev_thunk = autodiff_thunk(
        config2, BodyTy, Const, typeof(count), map(typeof, args)...
    )

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
    idx, tapes::Vector, thunk::ThunkTy, f::F, fargs::Vararg{Annotation,N}
) where {ThunkTy,F,N}
    return thunk(f, Const(idx), fargs..., @inbounds tapes[idx])
end

@inline function pmap_rev(
    idx, tapes::Ptr, thunk::ThunkTy, f::F, fargs::Vararg{Annotation,N}
) where {ThunkTy,F,N}
    return thunk(f, Const(idx), fargs..., unsafe_load(tapes, idx))
end

function EnzymeRules.reverse(
    config,
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
        EnzymeRules.width(config),
        EnzymeRules.overwritten(config)[2:end],
        InlineABI,
    }()
    fwd_thunk, rev_thunk = autodiff_thunk(
        config2, BodyTy, Const, typeof(count), map(typeof, args)...
    )

    Enzyme.pmap(pmap_rev, count.val, tapes, rev_thunk, body, args...)

    TapeType = EnzymeRules.tape_type(fwd_thunk)

    if !Enzyme.Compiler.any_jltypes(TapeType)
        Libc.free(tapes)
    end

    return ntuple(Val(2 + length(args))) do _
        Base.@_inline_meta
        nothing
    end
end

# From LinearAlgebra ~/.julia/juliaup/julia-1.10.0-beta3+0.x64.apple.darwin14/share/julia/stdlib/v1.10/LinearAlgebra/src/generic.jl:1110
@inline function compute_lu_cache(cache_A::AT, b::BT) where {AT,BT}
    LinearAlgebra.require_one_based_indexing(cache_A, b)
    m, n = size(cache_A)

    if m == n
        if LinearAlgebra.istril(cache_A)
            if LinearAlgebra.istriu(cache_A)
                return LinearAlgebra.Diagonal(cache_A)
            else
                return LinearAlgebra.LowerTriangular(cache_A)
            end
        elseif LinearAlgebra.istriu(cache_A)
            return LinearAlgebra.UpperTriangular(cache_A)
        else
            return LinearAlgebra.lu(cache_A)
        end
    end
    return LinearAlgebra.qr(cache_A, ColumnNorm())
end

# y=inv(A) B
#   dA −= z y^T
#   dB += z, where  z = inv(A^T) dy
function EnzymeRules.augmented_primal(
    config, func::Const{typeof(\)}, ::Type{RT}, A::Annotation{AT}, b::Annotation{BT}
) where {RT,AT<:Array,BT<:Array}
    cache_A = if EnzymeRules.overwritten(config)[2]
        copy(A.val)
    else
        A.val
    end

    cache_A = compute_lu_cache(cache_A, b.val)

    res = (cache_A \ b.val)::eltype(RT)

    dres = if EnzymeRules.width(config) == 1
        zero(res)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            zero(res)
        end
    end

    retres = if EnzymeRules.needs_primal(config)
        res
    else
        nothing
    end

    cache_res = if EnzymeRules.needs_primal(config)
        copy(res)
    else
        res
    end

    cache_b = if EnzymeRules.overwritten(config)[3]
        copy(b.val)
    else
        nothing
    end

    @static if VERSION < v"1.8.0"
        UT = Union{
            LinearAlgebra.Diagonal{eltype(AT),BT},
            LinearAlgebra.LowerTriangular{eltype(AT),AT},
            LinearAlgebra.UpperTriangular{eltype(AT),AT},
            LinearAlgebra.LU{eltype(AT),AT},
            LinearAlgebra.QRCompactWY{eltype(AT),AT},
        }
    else
        UT = Union{
            LinearAlgebra.Diagonal{eltype(AT),BT},
            LinearAlgebra.LowerTriangular{eltype(AT),AT},
            LinearAlgebra.UpperTriangular{eltype(AT),AT},
            LinearAlgebra.LU{eltype(AT),AT,Vector{Int}},
            LinearAlgebra.QRPivoted{eltype(AT),AT,BT,Vector{Int}},
        }
    end

    cache = NamedTuple{
        (Symbol("1"), Symbol("2"), Symbol("3"), Symbol("4")),
        Tuple{typeof(res),typeof(dres),UT,typeof(cache_b)},
    }((cache_res, dres, cache_A, cache_b))

    return EnzymeRules.AugmentedReturn{typeof(retres),typeof(dres),typeof(cache)}(
        retres, dres, cache
    )
end

function EnzymeRules.reverse(
    config,
    func::Const{typeof(\)},
    ::Type{RT},
    cache,
    A::Annotation{<:Array},
    b::Annotation{<:Array},
) where {RT}
    y, dys, cache_A, cache_b = cache

    if !EnzymeRules.overwritten(config)[3]
        cache_b = b.val
    end

    if EnzymeRules.width(config) == 1
        dys = (dys,)
    end

    dAs = if EnzymeRules.width(config) == 1
        if typeof(A) <: Const
            (nothing,)
        else
            (A.dval,)
        end
    else
        if typeof(A) <: Const
            ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                nothing
            end
        else
            A.dval
        end
    end

    dbs = if EnzymeRules.width(config) == 1
        if typeof(b) <: Const
            (nothing,)
        else
            (b.dval,)
        end
    else
        if typeof(b) <: Const
            ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                nothing
            end
        else
            b.dval
        end
    end

    for (dA, db, dy) in zip(dAs, dbs, dys)
        z = transpose(cache_A) \ dy
        if !(typeof(A) <: Const)
            dA .-= z * transpose(y)
        end
        if !(typeof(b) <: Const)
            db .+= z
        end
        dy .= eltype(dy)(0)
    end

    return (nothing, nothing)
end

const EnzymeTriangulars = Union{
    UpperTriangular,LowerTriangular,UnitUpperTriangular,UnitLowerTriangular
}

function EnzymeRules.augmented_primal(
    config,
    func::Const{typeof(ldiv!)},
    ::Type{RT},
    Y::Annotation{YT},
    A::Annotation{AT},
    B::Annotation{BT},
) where {RT,YT<:Array,AT<:EnzymeTriangulars,BT<:Array}
    cache_Y = EnzymeRules.overwritten(config)[1] ? copy(Y.val) : Y.val
    cache_A = EnzymeRules.overwritten(config)[2] ? copy(A.val) : A.val
    cache_A = compute_lu_cache(cache_A, B.val)
    cache_B = EnzymeRules.overwritten(config)[3] ? copy(B.val) : nothing
    primal = EnzymeRules.needs_primal(config) ? Y.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? Y.dval : nothing
    func.val(Y.val, A.val, B.val)
    return EnzymeRules.AugmentedReturn{typeof(primal),typeof(shadow),Any}(
        primal, shadow, (cache_Y, cache_A, cache_B)
    )
end

function EnzymeRules.reverse(
    config,
    func::Const{typeof(ldiv!)},
    ::Type{RT},
    cache,
    Y::Annotation{YT},
    A::Annotation{AT},
    B::Annotation{BT},
) where {YT<:Array,RT,AT<:EnzymeTriangulars,BT<:Array}
    if !isa(Y, Const)
        (cache_Yout, cache_A, cache_B) = cache
        for b in 1:EnzymeRules.width(config)
            dY = EnzymeRules.width(config) == 1 ? Y.dval : Y.dval[b]
            z = adjoint(cache_A) \ dY
            if !isa(B, Const)
                dB = EnzymeRules.width(config) == 1 ? B.dval : B.dval[b]
                dB .+= z
            end
            if !isa(A, Const)
                dA = EnzymeRules.width(config) == 1 ? A.dval : A.dval[b]
                dA.data .-= _zero_unused_elements!(z * adjoint(cache_Yout), A.val)
            end
            dY .= zero(eltype(dY))
        end
    end
    return (nothing, nothing, nothing)
end

_zero_unused_elements!(X, ::UpperTriangular) = triu!(X)
_zero_unused_elements!(X, ::LowerTriangular) = tril!(X)
_zero_unused_elements!(X, ::UnitUpperTriangular) = triu!(X, 1)
_zero_unused_elements!(X, ::UnitLowerTriangular) = tril!(X, -1)

@static if VERSION >= v"1.7-"
    # Force a rule around hvcat_fill as it is type unstable if the tuple is not of the same type (e.g., int, float, int, float)
    function EnzymeRules.augmented_primal(
        config,
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
        config,
        func::Const{typeof(Base.hvcat_fill!)},
        ::Type{RT},
        _,
        out::Annotation{AT},
        inp::Annotation{BT},
    ) where {RT,AT<:Array,BT<:Tuple}
        nr, nc = size(out.val, 1), size(out.val, 2)
        for b in 1:EnzymeRules.width(config)
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
end

function EnzymeRules.forward(
    ::Const{typeof(sort!)},
    RT::Type{<:Union{Const,DuplicatedNoNeed,Duplicated}},
    xs::Duplicated{T};
    kwargs...,
) where {T<:AbstractArray{<:AbstractFloat}}
    inds = sortperm(xs.val; kwargs...)
    xs.val .= xs.val[inds]
    xs.dval .= xs.dval[inds]
    if RT <: Const
        return xs.val
    elseif RT <: DuplicatedNoNeed
        return xs.dval
    else
        return xs
    end
end

function EnzymeRules.forward(
    ::Const{typeof(sort!)},
    RT::Type{<:Union{Const,BatchDuplicatedNoNeed,BatchDuplicated}},
    xs::BatchDuplicated{T,N};
    kwargs...,
) where {T<:AbstractArray{<:AbstractFloat},N}
    inds = sortperm(xs.val; kwargs...)
    xs.val .= xs.val[inds]
    for i in 1:N
        xs.dval[i] .= xs.dval[i][inds]
    end
    if RT <: Const
        return xs.val
    elseif RT <: BatchDuplicatedNoNeed
        return xs.dval
    else
        return xs
    end
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.ConfigWidth{1},
    ::Const{typeof(sort!)},
    RT::Type{<:Union{Const,DuplicatedNoNeed,Duplicated}},
    xs::Duplicated{T};
    kwargs...,
) where {T<:AbstractArray{<:AbstractFloat}}
    inds = sortperm(xs.val; kwargs...)
    xs.val .= xs.val[inds]
    xs.dval .= xs.dval[inds]
    if EnzymeRules.needs_primal(config)
        primal = xs.val
    else
        primal = nothing
    end
    if RT <: Const
        shadow = nothing
    else
        shadow = xs.dval
    end
    return EnzymeRules.AugmentedReturn(primal, shadow, inds)
end

function EnzymeRules.reverse(
    config::EnzymeRules.ConfigWidth{1},
    ::Const{typeof(sort!)},
    RT::Type{<:Union{Const,DuplicatedNoNeed,Duplicated}},
    tape,
    xs::Duplicated{T};
    kwargs...,
) where {T<:AbstractArray{<:AbstractFloat}}
    inds = tape
    back_inds = sortperm(inds)
    xs.dval .= xs.dval[back_inds]
    return (nothing,)
end

function EnzymeRules.forward(
        ::Const{typeof(partialsort!)},
        RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated}},
        xs::Duplicated{T},
        k::Const{<:Union{Integer, OrdinalRange}};
        kwargs...
    ) where {T <: AbstractArray{<:AbstractFloat}}
    kv = k.val
    inds = collect(eachindex(xs.val))
    partialsortperm!(inds, xs.val, kv; kwargs...)
    xs.val .= xs.val[inds]
    xs.dval .= xs.dval[inds]
    if RT <: Const
        return kv isa Integer ? xs.val[kv] : view(xs.val, kv)
    elseif RT <: DuplicatedNoNeed
        return kv isa Integer ? xs.dval[kv] : view(xs.dval, kv)
    else
        if kv isa Integer
            return Duplicated(xs.val[kv], xs.dval[kv])
        else
            return Duplicated(view(xs.val, kv), view(xs.dval, kv))
        end
    end
end

function EnzymeRules.forward(
        ::Const{typeof(partialsort!)},
        RT::Type{<:Union{Const, BatchDuplicatedNoNeed, BatchDuplicated}},
        xs::BatchDuplicated{T, N},
        k::Const{<:Union{Integer, OrdinalRange}};
        kwargs...
    ) where {T <: AbstractArray{<:AbstractFloat}, N}
    kv = k.val
    inds = collect(eachindex(xs.val))
    partialsortperm!(inds, xs.val, kv; kwargs...)
    xs.val .= xs.val[inds]
    for i in 1:N
        xs.dval[i] .= xs.dval[i][inds]
    end
    if RT <: Const
        return kv isa Integer ? xs.val[kv] : view(xs.val, kv)
    elseif RT <: BatchDuplicatedNoNeed
        if kv isa Integer
            return ntuple(i -> xs.dval[i][kv], N)
        else
            return ntuple(i -> view(xs.dval[i], kv), N)
        end
    else
        if kv isa Integer
            return BatchDuplicated(xs.val[kv], ntuple(i -> xs.dval[i][kv], N))
        else
            return BatchDuplicated(view(xs.val, kv), ntuple(i -> view(xs.dval[i], kv), N))
        end
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.ConfigWidth{1},
        ::Const{typeof(partialsort!)},
        RT::Type{<:Union{Const, Active, DuplicatedNoNeed, Duplicated}},
        xs::Duplicated{T},
        k::Const{<:Union{Integer, OrdinalRange}};
        kwargs...
    ) where {T <: AbstractArray{<:AbstractFloat}}
    kv = k.val
    inds = collect(eachindex(xs.val))
    partialsortperm!(inds, xs.val, kv; kwargs...)
    xs.val .= xs.val[inds]
    xs.dval .= xs.dval[inds]
    if EnzymeRules.needs_primal(config)
        primal = kv isa Integer ? xs.val[kv] : view(xs.val, kv)
    else
        primal = nothing
    end
    if RT <: Const || RT <: Active
        shadow = nothing
    else
        shadow = kv isa Integer ? xs.dval[kv] : view(xs.dval, kv)
    end
    return EnzymeRules.AugmentedReturn(primal, shadow, inds)
end

function EnzymeRules.reverse(
        config::EnzymeRules.ConfigWidth{1},
        ::Const{typeof(partialsort!)},
        dret::Union{Active, Type{<:Union{Const, Active, DuplicatedNoNeed, Duplicated}}},
        tape,
        xs::Duplicated{T},
        k::Const{<:Union{Integer, OrdinalRange}};
        kwargs...,
    ) where {T <: AbstractArray{<:AbstractFloat}}
    inds = tape
    kv = k.val
    if dret isa Active
        if kv isa Integer
            xs.dval[kv] += dret.val
        else
            xs.dval[kv] .+= dret.val
        end
    end
    back_inds = sortperm(inds)
    xs.dval .= xs.dval[back_inds]
    return (nothing, nothing)
end

function EnzymeRules.forward(::Const{typeof(cholesky)}, RT::Type, A; kwargs...)
    fact = cholesky(A.val; kwargs...)
    if RT <: Const
        return fact
    else
        N = width(RT)

        dA = if isa(A, Const)
            ntuple(Val(N)) do i
                Base.@_inline_meta
                zero(A.val)
            end
        else
            N == 1 ? (A.dval,) : A.dval
        end

        dfact = ntuple(Val(N)) do i
            Base.@_inline_meta
            return _cholesky_forward(fact, dA[i])
        end

        if (RT <: DuplicatedNoNeed) || (RT <: BatchDuplicatedNoNeed)
            return dfact
        elseif RT <: Duplicated
            return Duplicated(fact, dfact[1])
        else
            return BatchDuplicated(fact, dfact)
        end
    end
end

function _cholesky_forward(C::Cholesky, Ȧ)
    # Computes the cholesky forward mode update rule
    # C.f. eq. 8 in https://arxiv.org/pdf/1602.07527.pdf
    if C.uplo === 'U'
        U = C.U
        U̇ = Ȧ / U
        ldiv!(U', U̇)
        idx = diagind(U̇)
        U̇[idx] ./= 2
        triu!(U̇)
        rmul!(U̇, U)
        return Cholesky(U̇, 'U', C.info)
    else
        L = C.L
        L̇ = L \ Ȧ
        rdiv!(L̇, L')
        idx = diagind(L̇)
        L̇[idx] ./= 2
        tril!(L̇)
        lmul!(L, L̇)
        return Cholesky(L̇, 'L', C.info)
    end
end

# y = inv(A) B
# dY = inv(A) [ dB - dA y ]
# ->
# B(out) = inv(A) B(in)
# dB(out) = inv(A) [ dB(in) - dA B(out) ]
function EnzymeRules.forward(
    func::Const{typeof(ldiv!)}, RT::Type, fact::Annotation{<:Cholesky}, B; kwargs...
)
    if isa(B, Const)
        @assert (RT <: Const)
        return func.val(fact.val, B.val; kwargs...)
    else
        N = width(B)

        @assert !isa(B, Const)

        retval =
            if !isa(fact, Const) ||
                (RT <: Const) ||
                (RT <: Duplicated) ||
                (RT <: BatchDuplicated)
                func.val(fact.val, B.val; kwargs...)
            else
                nothing
            end

        dretvals = ntuple(Val(N)) do b
            Base.@_inline_meta

            dB = if N == 1
                B.dval
            else
                B.dval[b]
            end

            if !isa(fact, Const)
                dfact = if N == 1
                    fact.dval
                else
                    fact.dval[b]
                end

                tmp = dfact.U * retval
                mul!(dB, dfact.L, tmp, -1, 1)
            end

            func.val(fact.val, dB; kwargs...)
        end

        if RT <: Const
            return retval
        elseif RT <: DuplicatedNoNeed
            return dretvals[1]
        elseif RT <: Duplicated
            return Duplicated(retval, dretvals[1])
        elseif RT <: BatchDuplicatedNoNeed
            return dretvals
        else
            return BatchDuplicated(retval, dretvals)
        end
    end
end

function EnzymeRules.augmented_primal(
    config,
    func::Const{typeof(cholesky)},
    RT::Type,
    A::Annotation{<:Union{Matrix,LinearAlgebra.RealHermSym{<:Real,<:Matrix}}};
    kwargs...,
)
    fact = if EnzymeRules.needs_primal(config) || !(RT <: Const)
        cholesky(A.val; kwargs...)
    else
        nothing
    end

    fact_returned = EnzymeRules.needs_primal(config) ? fact : nothing

    # dfact would be a dense matrix, prepare buffer
    dfact = if RT <: Const
        nothing
    else
        if EnzymeRules.width(config) == 1
            Enzyme.make_zero(fact)
        else
            ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                Enzyme.make_zero(fact)
            end
        end
    end

    cache = isa(A, Const) ? nothing : (fact, dfact)
    return EnzymeRules.AugmentedReturn(fact_returned, dfact, cache)
end

function EnzymeRules.reverse(
    config,
    ::Const{typeof(cholesky)},
    RT::Type,
    cache,
    A::Annotation{<:Union{Matrix,LinearAlgebra.RealHermSym{<:Real,<:Matrix}}};
    kwargs...,
)
    if !(RT <: Const) && !isa(A, Const)
        fact, dfact = cache
        dAs = EnzymeRules.width(config) == 1 ? (A.dval,) : A.dval
        dfacts = EnzymeRules.width(config) == 1 ? (dfact,) : dfact

        for (dA, dfact) in zip(dAs, dfacts)
            _dA = dA isa LinearAlgebra.RealHermSym ? dA.data : dA
            if _dA !== dfact.factors
                Ā = _cholesky_pullback_shared_code(fact, dfact)
                idx = diagind(Ā)
                @views Ā[idx] .= real.(Ā[idx]) ./ 2
                _dA .+= UpperTriangular(Ā) .+ UpperTriangular(tril!(dfact.factors, -1)')
                dfact.factors .= 0
            end
        end
    end
    return (nothing,)
end

# Adapted from ChainRules.jl
# MIT "Expat" License
# Copyright (c) 2018: Jarrett Revels.
# https://github.com/JuliaDiff/ChainRules.jl/blob/9f1817a22404259113e230bef149a54d379a660b/src/rulesets/LinearAlgebra/factorization.jl#L507-L528
function _cholesky_pullback_shared_code(C, ΔC)
    Ā = similar(C.factors)
    if C.uplo === 'U'
        U = C.U
        Ū = ΔC.U
        mul!(Ā, Ū, U')
        LinearAlgebra.copytri!(Ā, 'U', true)
        eltype(Ā) <: Real || _realifydiag!(Ā)
        ldiv!(U, Ā)
        rdiv!(Ā, U')
    else  # C.uplo === 'L'
        L = C.L
        L̄ = ΔC.L
        mul!(Ā, L', L̄)
        LinearAlgebra.copytri!(Ā, 'L', true)
        eltype(Ā) <: Real || _realifydiag!(Ā)
        rdiv!(Ā, L)
        ldiv!(L', Ā)
    end
    return Ā
end

function _realifydiag!(A)
    for i in diagind(A)
        @inbounds A[i] = real(A[i])
    end
    return A
end

function EnzymeRules.augmented_primal(
    config,
    func::Const{typeof(ldiv!)},
    RT::Type{
        <:Union{Const,DuplicatedNoNeed,Duplicated,BatchDuplicatedNoNeed,BatchDuplicated}
    },
    A::Annotation{<:Cholesky},
    B::Union{Const,DuplicatedNoNeed,Duplicated,BatchDuplicatedNoNeed,BatchDuplicated};
    kwargs...,
)
    cache_B = if !isa(A, Const) && !isa(B, Const)
        EnzymeRules.overwritten(config)[3] ? copy(B.val) : B.val
    else
        nothing
    end

    cache_A = if !isa(B, Const)
        EnzymeRules.overwritten(config)[2] ? copy(A.val) : A.val
    else
        nothing
    end

    primal = EnzymeRules.needs_primal(config) ? B.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? B.dval : nothing
    func.val(A.val, B.val; kwargs...)
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_B))
end

function EnzymeRules.reverse(
    config,
    func::Const{typeof(ldiv!)},
    dret,
    cache,
    A::Annotation{<:Cholesky},
    B::Union{Const,DuplicatedNoNeed,Duplicated,BatchDuplicatedNoNeed,BatchDuplicated};
    kwargs...,
)
    if !isa(B, Const)
        (cache_A, cache_B) = cache
        Y = B.val
        U = cache_A.U
        Z = isa(A, Const) ? nothing : U' \ cache_B
        for b in 1:EnzymeRules.width(config)
            dB = EnzymeRules.width(config) == 1 ? B.dval : B.dval[b]
            dZ = U' \ dB
            func.val(cache_A, dB; kwargs...)
            if !isa(A, Const)
                ∂B = U \ dZ
                Ā = -dZ * Y' - Z * ∂B'
                dA = EnzymeRules.width(config) == 1 ? A.dval : A.dval[b]
                dA.factors .+= UpperTriangular(Ā)
            end
        end
    end
    return (nothing, nothing)
end
