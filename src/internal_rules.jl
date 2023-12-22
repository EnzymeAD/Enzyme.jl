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
function EnzymeRules.inactive_noinl(::typeof(Base.setindex!), ::IdDict{K, V}, ::K, ::V) where {K, V <:Integer}
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
        Enzyme.make_zero(source,
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

@inline function pmap_fwd(idx, tapes::Vector, thunk::ThunkTy, f::F, fargs::Vararg{Annotation, N}) where {ThunkTy, F, N}
    @inbounds tapes[idx] = thunk(f, Const(idx), fargs...)[1]
end

@inline function pmap_fwd(idx, tapes::Ptr, thunk::ThunkTy, f::F, fargs::Vararg{Annotation, N}) where {ThunkTy, F, N}
    unsafe_store!(tapes, thunk(f, Const(idx), fargs...)[1], idx)
end

function EnzymeRules.augmented_primal(config, func::Const{typeof(Enzyme.pmap)}, ::Type{Const{Nothing}}, body::BodyTy, count, args::Vararg{Annotation, N}) where {BodyTy, N}

    config2 = ReverseModeSplit{false, false, EnzymeRules.width(config), EnzymeRules.overwritten(config)[2:end],InlineABI}()
    fwd_thunk, rev_thunk = autodiff_thunk(config2, BodyTy, Const, typeof(count), map(typeof, args)...)

    TapeType = EnzymeRules.tape_type(fwd_thunk)

    tapes = if Enzyme.Compiler.any_jltypes(TapeType)
        Vector{TapeType}(undef, count.val)
    else
        Base.unsafe_convert(Ptr{TapeType}, Libc.malloc(sizeof(TapeType)*count.val))
    end

    Enzyme.pmap(pmap_fwd, count.val, tapes, fwd_thunk, body, args...)
    return EnzymeRules.AugmentedReturn(nothing, nothing, tapes)
end

@inline function pmap_rev(idx, tapes::Vector, thunk::ThunkTy, f::F, fargs::Vararg{Annotation, N}) where {ThunkTy, F, N}
    thunk(f, Const(idx), fargs..., @inbounds tapes[idx])
end

@inline function pmap_rev(idx, tapes::Ptr, thunk::ThunkTy, f::F, fargs::Vararg{Annotation, N}) where {ThunkTy, F, N}
    thunk(f, Const(idx), fargs..., unsafe_load(tapes, idx))
end

function EnzymeRules.reverse(config, func::Const{typeof(Enzyme.pmap)}, ::Type{Const{Nothing}}, tapes, body::BodyTy, count, args::Vararg{Annotation, N}) where {BodyTy, N}

    config2 = ReverseModeSplit{false, false, EnzymeRules.width(config), EnzymeRules.overwritten(config)[2:end],InlineABI}()
    fwd_thunk, rev_thunk =  autodiff_thunk(config2, BodyTy, Const, typeof(count), map(typeof, args)...)

    Enzyme.pmap(pmap_rev, count.val, tapes, rev_thunk, body, args...)

    TapeType = EnzymeRules.tape_type(fwd_thunk)

    if !Enzyme.Compiler.any_jltypes(TapeType)
        Libc.free(tapes)
    end

    return ntuple(Val(2+length(args))) do _
        Base.@_inline_meta
        nothing
    end
end



# From LinearAlgebra ~/.julia/juliaup/julia-1.10.0-beta3+0.x64.apple.darwin14/share/julia/stdlib/v1.10/LinearAlgebra/src/generic.jl:1110
@inline function compute_lu_cache(cache_A::AT, b::BT) where {AT, BT}
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
function EnzymeRules.augmented_primal(config, func::Const{typeof(\)}, ::Type{RT}, A::Annotation{AT}, b::Annotation{BT}) where {RT, AT <: Array, BT <: Array}

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
        LinearAlgebra.Diagonal{eltype(AT), BT},
        LinearAlgebra.LowerTriangular{eltype(AT), AT},
        LinearAlgebra.UpperTriangular{eltype(AT), AT},
        LinearAlgebra.LU{eltype(AT), AT},
        LinearAlgebra.QRCompactWY{eltype(AT), AT}
    }
else
    UT = Union{
        LinearAlgebra.Diagonal{eltype(AT), BT},
        LinearAlgebra.LowerTriangular{eltype(AT), AT},
        LinearAlgebra.UpperTriangular{eltype(AT), AT},
        LinearAlgebra.LU{eltype(AT), AT, Vector{Int}},
        LinearAlgebra.QRPivoted{eltype(AT), AT, BT, Vector{Int}}
    }
end

    cache = NamedTuple{(Symbol("1"),Symbol("2"), Symbol("3"), Symbol("4")), Tuple{typeof(res), typeof(dres), UT, typeof(cache_b)}}(
        (cache_res, dres, cache_A, cache_b)
    )

    return EnzymeRules.AugmentedReturn{typeof(retres), typeof(dres), typeof(cache)}(retres, dres, cache)
end

function EnzymeRules.reverse(config, func::Const{typeof(\)}, ::Type{RT}, cache, A::Annotation{<:Array}, b::Annotation{<:Array}) where RT

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

    return (nothing,nothing)
end

@static if VERSION >= v"1.7-"
# Force a rule around hvcat_fill as it is type unstable if the tuple is not of the same type (e.g., int, float, int, float)
function EnzymeRules.augmented_primal(config, func::Const{typeof(Base.hvcat_fill!)}, ::Type{RT}, out::Annotation{AT}, inp::Annotation{BT}) where {RT, AT <: Array, BT <: Tuple}
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

function EnzymeRules.reverse(config, func::Const{typeof(Base.hvcat_fill!)}, ::Type{RT}, _, out::Annotation{AT}, inp::Annotation{BT}) where {RT, AT <: Array, BT <: Tuple}
    nr, nc = size(out.val,1), size(out.val,2)
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
                if j == nc+1
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
            return (nothing, dinp)::Tuple{Nothing, BT}
        end
    end
    return (nothing, nothing)
end
end

function EnzymeRules.forward(
        ::Const{typeof(sort!)},
        RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated}},
        xs::Duplicated{T};
        kwargs...
    ) where {T <: AbstractArray{<:AbstractFloat}}
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
        RT::Type{<:Union{Const, BatchDuplicatedNoNeed, BatchDuplicated}},
        xs::BatchDuplicated{T, N};
        kwargs...
    ) where {T <: AbstractArray{<:AbstractFloat}, N}
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
        RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated}},
        xs::Duplicated{T};
        kwargs...
    ) where {T <: AbstractArray{<:AbstractFloat}}
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
        RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated}},
        tape,
        xs::Duplicated{T};
        kwargs...,
    ) where {T <: AbstractArray{<:AbstractFloat}}
    inds = tape
    back_inds = sortperm(inds)
    xs.dval .= xs.dval[back_inds]
    return (nothing,)
end

function EnzymeRules.forward(
    ::Const{typeof(cholesky)},
    RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated}},
    A::Union{Const, Duplicated};
    kwargs...
)
    fact = cholesky(A.val; kwargs...)
    if RT <: Const
        return fact
    end
    # TODO: This will be a problem for sparse matrices as invL and dL are dense
    invL = inv(fact.L)
    # TODO: dL is dense even when L was sparse
    dL = Matrix(fact.L * LowerTriangular(invL * A.dval * invL' * 0.5 * I))
    # TODO: Stored as Cholesky, although it isn't a Cholesky factorization
    dfact = Cholesky(dL, 'L', 0)
    if RT <: DuplicatedNoNeed
        return dfact
    else
        return Duplicated(fact, dfact)
    end
end

function EnzymeRules.forward(
    ::Const{typeof(cholesky)},
    RT::Type{<:Union{BatchDuplicatedNoNeed, BatchDuplicated}},
    A::Union{BatchDuplicatedNoNeed{T,N}, BatchDuplicated{T,N}};
    kwargs...
) where {T,N}
    fact = cholesky(A.val; kwargs...)
    if RT <: Const
        return fact
    end
    invL = inv(fact.L)
    dfact = ntuple(
        i-> Cholesky(
            Matrix(fact.L * LowerTriangular(invL * A.dval[i] * invL' * 0.5 * I)), 'L', 0
        ), Val(N)
    )
    if RT <: BatchDuplicatedNoNeed
        return dfact
    else
        return BatchDuplicated(fact, dfact)
    end
end

function EnzymeRules.forward(
        func::Const{typeof(\)},
        RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated}},
        fact::Annotation{<:Cholesky}, B::Union{Const, Duplicated};
        kwargs...
)
    retval = copy(B.val)
    ldiv!(fact.val, retval)
    retdval = []
    if isa(fact, Duplicated) && isa(B, Duplicated)
        retdval = zeros(length(retval))
        mul!(retdval, fact.dval.U, retval)
        mul!(retdval, fact.dval.L, retdval)
        retdval .= B.dval .- retdval
        ldiv!(fact.val, retdval)
    elseif isa(fact, Duplicated) && isa(B, Const)
        retdval = zeros(length(retval))
        mul!(retdval, A, B.val)
        mul!(retdval, -1, retdval)
        ldiv!(fact.val, retdval)
    elseif isa(fact, Const) && isa(B, Duplicated)
        retdval = copy(B.dval)
        ldiv!(fact.val, retdval)
    elseif isa(fact, Const) && isa(B, Const)
        nothing
    else
        error("Error in forward $(typeof(func)) Enzyme rule $(typeof(fact)) $(typeof(B)).")
    end
    if RT <: Const
        return retval
    elseif (RT <: DuplicatedNoNeed) || (RT <: BatchDuplicatedNoNeed)
        return retdval
    else
        return Duplicated(retval, retdval)
    end
end

function EnzymeRules.forward(
        func::Const{typeof(\)},
        RT::Type{<:Union{Const, BatchDuplicatedNoNeed, BatchDuplicated}},
        fact::Annotation{<:Cholesky}, B::Union{Const, BatchDuplicated{T,N}};
        kwargs...
) where {T,N}
    retval = copy(B.val)
    ldiv!(fact.val, retval)
    bretdval = ntuple(Val(N)) do b
        if isa(fact, BatchDuplicated) && isa(B, BatchDuplicated)
            retdval = zeros(length(retval))
            mul!(retdval, fact.dval[b].U, retval)
            mul!(retdval, fact.dval[b].L, retdval)
            retdval .= B.dval[b] .- retdval
            ldiv!(fact.val, retdval)
        elseif isa(fact, BatchDuplicated) && isa(B, Const)
            retdval = zeros(length(retval))
            mul!(retdval, A, B.val)
            mul!(retdval, -1, retdval)
            ldiv!(fact.val, retdval)
        elseif isa(fact, Const) && isa(B, BatchDuplicated)
            retdval = copy(B.dval)
            ldiv!(fact.val, retdval)
        elseif isa(fact, Const) && isa(B, Const)
            nothing
        else
            error("Error in forward \\ Enzyme rule $(typeof(fact)) $(typeof(B)).")
        end
        retdval
    end
    if RT <: Const
        return retval
    elseif (RT <: DuplicatedNoNeed) || (RT <: BatchDuplicatedNoNeed)
        return retdval
    else
        return BatchDuplicated(retval, ntuple(i -> bretdval[i], Val(N)))
    end
end

function EnzymeRules.forward(
        func::Const{typeof(ldiv!)},
        RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated}},
        fact::Annotation{<:Cholesky}, B::Union{Const, Duplicated};
        kwargs...
)
    retval = B.val
    ldiv!(fact.val, retval)
    retdval = []
    if isa(fact, Duplicated) && isa(B, Duplicated)
        retdval = zeros(length(retval))
        mul!(retdval, fact.dval.U, retval)
        mul!(retdval, fact.dval.L, retdval)
        retdval .= B.dval .- retdval
        ldiv!(fact.val, retdval)
        B.val  .= retval
        B.dval .= retdval
    elseif isa(fact, Duplicated) && isa(B, Const)
        retdval = zeros(length(retval))
        mul!(retdval, A, B.val)
        mul!(retdval, -1, retdval)
        ldiv!(fact.val, retdval)
        B.val  .= retval
        B.dval .= retdval
    elseif isa(fact, Const) && isa(B, Duplicated)
        retdval = copy(B.dval)
        ldiv!(fact.val, retdval)
        B.val  .= retval
        B.dval .= retdval
    elseif isa(fact, Const) && isa(B, Const)
        nothing
    else
        error("Error in forward ldiv! Enzyme rule $(typeof(fact)) $(typeof(B)).")
    end
    if RT <: Const
        return retval
    elseif (RT <: DuplicatedNoNeed)
        return retdval
    else
        return Duplicated(retval, retdval)
    end
end

function EnzymeRules.forward(
        func::Const{typeof(ldiv!)},
        RT::Type{<:Union{Const, BatchDuplicatedNoNeed, BatchDuplicated}},
        fact::Annotation{<:Cholesky}, B::Union{Const, BatchDuplicated{T,N}};
        kwargs...
) where {T,N}
    retval = B.val
    retval = ldiv!(fact.val, retval)
    for b in 1:N
        if isa(fact, BatchDuplicated) && isa(B, BatchDuplicated)
            retdval = zeros(length(retval))
            mul!(retdval, fact.dval[b].U, retval)
            mul!(retdval, fact.dval[b].L, retdval)
            retdval .= B.dval[b] .- retdval
            ldiv!(fact.val, retdval)
            B.val     .= retval
            B.dval[b] .= retdval
        elseif isa(fact, BatchDuplicated) && isa(B, Const)
            retdval = zeros(length(retval))
            mul!(retdval, A, B.val)
            mul!(retdval, -1, retdval)
            ldiv!(fact.val, retdval)
            B.val     .= retval
            B.dval[b] .= retdval
        elseif isa(fact, Const) && isa(B, BatchDuplicated)
            retdval = copy(B.dval)
            ldiv!(fact.val, retdval)
            B.val     .= retval
            B.dval[b] .= retdval
        elseif isa(fact, Const) && isa(B, Const)
            nothing
        else
            error("Error in forward ldiv! Enzyme rule $(typeof(fact)) $(typeof(B)).")
        end
        retdval
    end
    if RT <: Const
        return retval
    elseif (RT <: DuplicatedNoNeed) || (RT <: BatchDuplicatedNoNeed)
        return retdval
    else
        return BatchDuplicated(retval, ntuple(i -> bretdval[i], Val(N)))
    end
end

function EnzymeRules.augmented_primal(
    config,
    func::Const{typeof(cholesky)},
    RT::Type{<:Union{Const, Duplicated, BatchDuplicatedNoNeed, BatchDuplicated}},
    A::Union{Const, Duplicated, BatchDuplicated};
    kwargs...
)
    fact = if EnzymeRules.needs_primal(config)
        cholesky(A.val; kwargs...)
    else
        nothing
    end
    # dfact would be a dense matrix, prepare buffer
    dfact = if EnzymeRules.width(config) == 1
        Cholesky(Matrix(fact), 'L', 0)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            Cholesky(Matrix(fact), 'L', 0)
        end
    end
    return EnzymeRules.AugmentedReturn(fact, dfact, (dfact,))
end

function EnzymeRules.reverse(
    config,
    ::Const{typeof(cholesky)},
    dret,
    cache,
    A;
    kwargs...
)
    (_dfact,) = cache
    for b in 1:EnzymeRules.width(config)
        dA = if EnzymeRules.width(config) == 1
            A.dval
        else
            A.dval[b]
        end
        dfact = if EnzymeRules.width(config) == 1
            _dfact
        else
            _dfact[b]
        end
        dA .+= dfact.factors
    end
    return (nothing,)
end

function EnzymeRules.augmented_primal(
        config,
        func::Const{typeof(\)},
        RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated, BatchDuplicatedNoNeed, BatchDuplicated}},

        fact::Annotation{<:Cholesky},
        B::Union{Const, DuplicatedNoNeed, Duplicated, BatchDuplicatedNoNeed, BatchDuplicated};
        kwargs...
)
    x = copy(B.val)
    primal = if EnzymeRules.needs_primal(config)
        x
    else
        nothing
    end
    ldiv!(fact.val, x)
    nobatched = EnzymeRules.width(config) == 1 ? true : false
    shadow = nobatched ? zeros(size(B.val)) :
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            zeros(size(B.val))
        end
    buffer = nobatched ? zeros(size(B.val)) :
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            zeros(size(B.val))
        end
    return EnzymeRules.AugmentedReturn(primal, shadow, (x, shadow, buffer))
end

function EnzymeRules.reverse(
    config,
    func::Union{Const{typeof(\)}},
    dret,
    cache,
    _fact::Annotation{<:Cholesky},
    _B::Annotation;
    kwargs...
)

    (x, _dx, _buffer) = cache
    for b in 1:EnzymeRules.width(config)
        dx = if EnzymeRules.width(config) == 1
            _dx
        else
            _dx[b]
        end
        buffer = if EnzymeRules.width(config) == 1
            _buffer
        else
            _buffer[b]
        end
        dfact = if EnzymeRules.width(config) == 1
            _fact.dval
        else
            _fact.dval[b]
        end
        dB = if EnzymeRules.width(config) == 1
            _B.dval
        else
            _B.dval[b]
        end
        buffer .= _fact.val\dx
        dB .+= buffer
        buffer = reshape(buffer, size(buffer,2), size(buffer,1))
        dfact.factors .+= -x .* buffer
    end
    return (nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config,
        func::Const{typeof(ldiv!)},
        RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated, BatchDuplicatedNoNeed, BatchDuplicated}},

        fact::Annotation{<:Cholesky},
        B::Union{Const, DuplicatedNoNeed, Duplicated, BatchDuplicatedNoNeed, BatchDuplicated};
        kwargs...
)
    x = B.val
    primal = if EnzymeRules.needs_primal(config)
        x
    else
        nothing
    end
    ldiv!(fact.val, x)
    nobatched = EnzymeRules.width(config) == 1 ? true : false
    shadow = nothing
    buffer = nobatched ? zeros(size(B.val)) :
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            zeros(size(B.val))
        end
    return EnzymeRules.AugmentedReturn(primal, shadow, (x, shadow, buffer))
end

function EnzymeRules.reverse(
    config,
    func::Const{typeof(ldiv!)},
    dret,
    cache,
    _fact::Annotation{<:Cholesky},
    _B::Annotation;
    kwargs...
)

    (x, _dx, _buffer) = cache
    for b in 1:EnzymeRules.width(config)
        dx = if EnzymeRules.width(config) == 1
            _B.dval
        else
            _B.dval[b]
        end
        buffer = if EnzymeRules.width(config) == 1
            _buffer
        else
            _buffer[b]
        end
        dfact = if EnzymeRules.width(config) == 1
            _fact.dval
        else
            _fact.dval[b]
        end
        dB = if EnzymeRules.width(config) == 1
            _B.dval
        else
            _B.dval[b]
        end
        buffer .= _fact.val\dx
        dB .+= buffer
        buffer = reshape(buffer, size(buffer,2), size(buffer,1))
        dfact.factors .+= -x .* buffer
    end
    return (nothing, nothing)
end
