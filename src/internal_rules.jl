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
function EnzymeRules.inactive(::typeof(Base.CoreLogging.handle_message), args...; kwargs...)
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
function EnzymeRules.inactive(
    ::typeof(Random.rand!),
    ::Random.AbstractRNG,
    ::Random.Sampler,
    ::AbstractArray,
)
    return nothing
end
function EnzymeRules.inactive(::typeof(Random.randn!), ::Random.AbstractRNG, ::AbstractArray)
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
    ::typeof(Base.setindex!),
    ::IdDict{K,V},
    ::K,
    ::V,
) where {K,V<:Integer}
    return nothing
end

function EnzymeRules.inactive_noinl(::typeof(Base.hasproperty), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.startswith), ::AbstractString, args...)
    return nothing
end

Enzyme.EnzymeRules.inactive_noinl(::typeof(Core._compute_sparams), args...) = nothing

@inline EnzymeRules.inactive_type(v::Type{Nothing}) = true
@inline EnzymeRules.inactive_type(v::Type{Union{}}) = true
@inline EnzymeRules.inactive_type(v::Type{Char}) = true
@inline EnzymeRules.inactive_type(v::Type{T}) where {T<:Integer} = true
@inline EnzymeRules.inactive_type(v::Type{Function}) = true
@inline EnzymeRules.inactive_type(v::Type{T}) where {T<:DataType} = true
@inline EnzymeRules.inactive_type(v::Type{T}) where {T<:Module} = true
@inline EnzymeRules.inactive_type(v::Type{T}) where {T<:AbstractString} = true
@inline EnzymeRules.inactive_type(v::Type{Core.MethodMatch}) = true
@inline EnzymeRules.inactive_type(v::Type{Core.Compiler.WorldRange}) = true
@inline EnzymeRules.inactive_type(v::Type{Core.MethodInstance}) = true

# Note all of these forward mode definitions do not support runtime activity as
# the do not keep the primal if shadow(x.y) == primal(x.y)
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
    seen::IdDict,
    shadow::RT,
) where {RT<:Union{Integer,Char}}
    return Base.deepcopy_internal(shadow, seen)
end
@inline function deepcopy_rtact(
    copied::RT,
    primal::RT,
    seen::IdDict,
    shadow::RT,
) where {RT<:AbstractFloat}
    return Base.deepcopy_internal(shadow, seen)
end
@inline function deepcopy_rtact(
    copied::RT,
    primal::RT,
    seen::IdDict,
    shadow::RT,
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
    config::EnzymeRules.FwdConfig,
    func::Const{typeof(Base.deepcopy)},
    ::Type{<:Duplicated},
    x::Duplicated,
)
    primal = func.val(x.val)
    return Duplicated(primal, deepcopy_rtact(primal, x.val, IdDict(), x.dval))
end

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    func::Const{typeof(Base.deepcopy)},
    ::Type{<:BatchDuplicated},
    x::BatchDuplicated{T,N},
) where {T,N}
    primal = func.val(x.val)
    return BatchDuplicated(primal, ntuple(Val(N)) do i
        deepcopy_rtact(primal, x.val, IdDict(), x.dval[i])
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

    @assert !(typeof(x) <: Active)

    source = if EnzymeRules.needs_primal(config)
        primal
    else
        x.val
    end

    shadow = if EnzymeRules.needs_shadow(config)
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

@inline onedimensionalize(::Type{T}) where {T<:Array} = Vector{eltype(T)}

# y=inv(A) B
#   dA −= z y^T
#   dB += z, where  z = inv(A^T) dy
function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    func::Const{typeof(\)},
    ::Type{RT},
    A::Annotation{AT},
    b::Annotation{BT},
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

    UT = Union{
        LinearAlgebra.Diagonal{eltype(AT),onedimensionalize(BT)},
        LinearAlgebra.LowerTriangular{eltype(AT),AT},
        LinearAlgebra.UpperTriangular{eltype(AT),AT},
        LinearAlgebra.LU{eltype(AT),AT,Vector{Int}},
        LinearAlgebra.QRPivoted{eltype(AT),AT,onedimensionalize(BT),Vector{Int}},
    }

    cache = NamedTuple{
        (Symbol("1"), Symbol("2"), Symbol("3"), Symbol("4")),
        Tuple{
            eltype(RT),
            EnzymeRules.needs_shadow(config) ?
            (
                EnzymeRules.width(config) == 1 ? eltype(RT) :
                NTuple{EnzymeRules.width(config),eltype(RT)}
            ) : Nothing,
            UT,
            typeof(cache_b),
        },
    }((cache_res, dres, cache_A, cache_b))

    return EnzymeRules.AugmentedReturn{
        EnzymeRules.primal_type(config, RT),
        EnzymeRules.shadow_type(config, RT),
        typeof(cache),
    }(
        retres,
        dres,
        cache,
    )
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
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
    UpperTriangular{<:Complex},
    LowerTriangular{<:Complex},
    UnitUpperTriangular{<:Complex},
    UnitLowerTriangular{<:Complex},
}

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
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
    return EnzymeRules.AugmentedReturn{
        EnzymeRules.primal_type(config, RT),
        EnzymeRules.shadow_type(config, RT),
        Tuple{typeof(cache_Y),typeof(cache_A),typeof(cache_B)},
    }(
        primal,
        shadow,
        (cache_Y, cache_A, cache_B),
    )
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    func::Const{typeof(ldiv!)},
    ::Type{RT},
    cache,
    Y::Annotation{YT},
    A::Annotation{AT},
    B::Annotation{BT},
) where {YT<:Array,RT,AT<:EnzymeTriangulars,BT<:Array}
    if !isa(Y, Const)
        (cache_Yout, cache_A, cache_B) = cache
        for b = 1:EnzymeRules.width(config)
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


function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, 
                                      func::Const{typeof(LinearAlgebra.mul!)},
                                      ::Type{RT}, 
                                      C::Annotation{<:StridedVecOrMat},
                                      A::Const{<:SparseArrays.SparseMatrixCSCUnion},
                                      B::Annotation{<:StridedVecOrMat},
                                      α::Annotation{<:Number},
                                      β::Annotation{<:Number}
                                    ) where {RT}

    cache_C = !(isa(β, Const)) ? copy(C.val) : nothing
    # Always need to do forward pass otherwise primal may not be correct
    func.val(C.val, A.val, B.val, α.val, β.val)

    primal = if EnzymeRules.needs_primal(config)
        C.val
    else
        nothing
    end

    shadow = if EnzymeRules.needs_shadow(config)
        C.dval
    else
        nothing
    end

    # Check if A is overwritten and B is active (and thus required)
    cache_A = ( EnzymeRules.overwritten(config)[5]
                && !(typeof(B) <: Const)
                && !(typeof(C) <: Const)
                ) ? copy(A.val) : nothing
    
    # cache_B = ( EnzymeRules.overwritten(config)[6]) ? copy(B.val) : nothing

    if !isa(α, Const)
        cache_α = A.val*B.val
    else
        cache_α = nothing
    end
    
    cache = (cache_C, cache_A, cache_α)

    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfig,
                             func::Const{typeof(LinearAlgebra.mul!)},
                             ::Type{RT}, cache,
                             C::Annotation{<:StridedVecOrMat},
                             A::Const{<:SparseArrays.SparseMatrixCSCUnion},
                             B::Annotation{<:StridedVecOrMat},
                             α::Annotation{<:Number},
                             β::Annotation{<:Number}
                             ) where {RT}

    cache_C, cache_A, cache_α = cache
    Cval = !isnothing(cache_C) ? cache_C : C.val
    Aval = !isnothing(cache_A) ? cache_A : A.val
    # Bval = !isnothing(cache_B) ? cache_B : B.val

    N = EnzymeRules.width(config)
    if !isa(C, Const)
        dCs = C.dval
        dBs  = isa(B, Const) ? dCs : B.dval

        dα = if !isa(α, Const)
                if N == 1
                    LinearAlgebra.dot(C.dval, cache_α)
                else
                    ntuple(Val(N)) do i
                        Base.@_inline_meta
                        LinearAlgebra.dot(C.dval[i], cache_α)
                    end
                end
        else
            nothing
        end

        dβ = if !isa(β, Const)
                if N == 1
                    LinearAlgebra.dot(C.dval, Cval)
                else
                    ntuple(Val(N)) do i
                        Base.@_inline_meta
                        LinearAlgebra.dot(C.dval[i], Cval)
                    end
                end
        else
            nothing
        end

        for i in 1:N
            # This rule is incorrect since you need to project dA to have the same
            # sparsity pattern as A.
            # if !isa(A, Const)
            #     dA = EnzymeRules.width(config) == 1 ? A.dval : A.dval[b]
            #     #dA .+= α*dC*B'
            #     mul!(dA, dC, Bval', α.val, true)
            # end

            if !isa(B, Const)
                #dB .+= α*A'*dC
                if N ==1
                    func.val(dBs, Aval', dCs, α.val, true)
                else
                    func.val(dBs[i], Aval', dCs[i], α.val, true)
                end
            end

            if N==1
                dCs .*= β.val
            else
                dCs[i] .*= β.val
            end
        end
    end
   
    return (nothing, nothing, nothing, dα, dβ)
end







function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    ::Const{typeof(sort!)},
    RT::Type{<:Union{Const,DuplicatedNoNeed,Duplicated}},
    xs::Duplicated{T};
    kwargs...,
) where {T<:AbstractArray{<:AbstractFloat}}
    inds = sortperm(xs.val; kwargs...)
    xs.val .= xs.val[inds]
    xs.dval .= xs.dval[inds]
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return xs
    elseif EnzymeRules.needs_shadow(config)
        return xs.dval
    elseif EnzymeRules.needs_primal(config)
        return xs.val
    else
        return nothing
    end
end

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    ::Const{typeof(sort!)},
    RT::Type{<:Union{Const,BatchDuplicatedNoNeed,BatchDuplicated}},
    xs::BatchDuplicated{T,N};
    kwargs...,
) where {T<:AbstractArray{<:AbstractFloat},N}
    inds = sortperm(xs.val; kwargs...)
    xs.val .= xs.val[inds]
    for i = 1:N
        xs.dval[i] .= xs.dval[i][inds]
    end
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return xs
    elseif EnzymeRules.needs_shadow(config)
        return xs.dval
    elseif EnzymeRules.needs_primal(config)
        return xs.val
    else
        return nothing
    end
end


function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
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
    config::EnzymeRules.RevConfigWidth{1},
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
    config::EnzymeRules.FwdConfig,
    ::Const{typeof(partialsort!)},
    RT::Type{<:Union{Const,DuplicatedNoNeed,Duplicated}},
    xs::Duplicated{T},
    k::Const{<:Union{Integer,OrdinalRange}};
    kwargs...,
) where {T<:AbstractArray{<:AbstractFloat}}
    kv = k.val
    inds = collect(eachindex(xs.val))
    partialsortperm!(inds, xs.val, kv; kwargs...)
    xs.val .= xs.val[inds]
    xs.dval .= xs.dval[inds]

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        if kv isa Integer
            return Duplicated(xs.val[kv], xs.dval[kv])
        else
            return Duplicated(view(xs.val, kv), view(xs.dval, kv))
        end
    elseif EnzymeRules.needs_shadow(config)
        return kv isa Integer ? xs.dval[kv] : view(xs.dval, kv)
    elseif EnzymeRules.needs_primal(config)
        return kv isa Integer ? xs.val[kv] : view(xs.val, kv)
    else
        return nothing
    end
end

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    ::Const{typeof(partialsort!)},
    RT::Type{<:Union{Const,BatchDuplicatedNoNeed,BatchDuplicated}},
    xs::BatchDuplicated{T,N},
    k::Const{<:Union{Integer,OrdinalRange}};
    kwargs...,
) where {T<:AbstractArray{<:AbstractFloat},N}
    kv = k.val
    inds = collect(eachindex(xs.val))
    partialsortperm!(inds, xs.val, kv; kwargs...)
    xs.val .= xs.val[inds]
    for i = 1:N
        xs.dval[i] .= xs.dval[i][inds]
    end

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        if kv isa Integer
            return BatchDuplicated(xs.val[kv], ntuple(i -> xs.dval[i][kv], N))
        else
            return BatchDuplicated(view(xs.val, kv), ntuple(i -> view(xs.dval[i], kv), N))
        end
    elseif EnzymeRules.needs_shadow(config)
        if kv isa Integer
            return ntuple(i -> xs.dval[i][kv], N)
        else
            return ntuple(i -> view(xs.dval[i], kv), N)
        end
    elseif EnzymeRules.needs_primal(config)
        return kv isa Integer ? xs.val[kv] : view(xs.val, kv)
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(partialsort!)},
    RT::Type{<:Union{Const,Active,DuplicatedNoNeed,Duplicated}},
    xs::Duplicated{T},
    k::Const{<:Union{Integer,OrdinalRange}};
    kwargs...,
) where {T<:AbstractArray{<:AbstractFloat}}
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
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(partialsort!)},
    dret::Union{Active,Type{<:Union{Const,Active,DuplicatedNoNeed,Duplicated}}},
    tape,
    xs::Duplicated{T},
    k::Const{<:Union{Integer,OrdinalRange}};
    kwargs...,
) where {T<:AbstractArray{<:AbstractFloat}}
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

# y = inv(A) B
# dY = inv(A) [ dB - dA y ]
# ->
# B(out) = inv(A) B(in)
# dB(out) = inv(A) [ dB(in) - dA B(out) ]
function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    func::Const{typeof(ldiv!)},
    RT::Type{<:Union{Const,Duplicated,BatchDuplicated}},
    fact::Annotation{<:Cholesky},
    B::Annotation{<:AbstractVecOrMat};
    kwargs...,
)
    if B isa Const
        retval = func.val(fact.val, B.val; kwargs...)
        if EnzymeRules.needs_primal(config)
            retval
        else
            return nothing
        end
    else
        N = EnzymeRules.width(config)
        retval = B.val

        L = fact.val.L
        U = fact.val.U

        ldiv!(L, B.val)
        ntuple(Val(N)) do b
            Base.@_inline_meta
            dB = N == 1 ? B.dval : B.dval[b]
            if !(fact isa Const)
                dL = N == 1 ? fact.dval.L : fact.dval[b].L
                mul!(dB, dL, B.val, -1, 1)
            end
            ldiv!(L, dB)
        end

        ldiv!(U, B.val)
        dretvals = ntuple(Val(N)) do b
            Base.@_inline_meta
            dB = N == 1 ? B.dval : B.dval[b]
            if !(fact isa Const)
                dU = N == 1 ? fact.dval.U : fact.dval[b].U
                mul!(dB, dU, B.val, -1, 1)
            end
            ldiv!(U, dB)
            return dB
        end


        if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
            if EnzymeRules.width(config) == 1
                return Duplicated(retval, dretvals[1])
            else
                return BatchDuplicated(retval, dretvals)
            end
        elseif EnzymeRules.needs_shadow(config)
            if EnzymeRules.width(config) == 1
                return dretvals[1]
            else
                return dretvals
            end
        elseif EnzymeRules.needs_primal(config)
            return retval
        else
            return nothing
        end
    end
end

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    func::Const{typeof(Base.range_start_stop_length)},
    RT,
    start::Annotation{T},
    stop::Annotation{T},
    len::Annotation{<:Integer},
) where T <: Base.IEEEFloat
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            return Duplicated(
                func.val(start.val, stop.val, len.val),
                func.val(
                        start isa Const ? zero(start.val) : -start.dval,
                        stop isa Const ? zero(stop.val) : stop.dval,
                        len.val)
                )
        else
            return BatchDuplicated(
                func.val(start.val, stop.val, len.val),
                ntuple(
                    i -> func.val(
                        start isa Const ? zero(start.val) : -start.dval[i],
                        stop isa Const ? zero(stop.val)  : stop.dval[i],
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
                        len.val)
        else
            return ntuple(
                i -> func.val(
                    start isa Const ? zero(start.val) : -start.dval[i],
                    stop isa Const ? zero(stop.val)  : stop.dval[i],
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
) where {T <: Base.IEEEFloat, T3<:Integer}
    dstart = if start isa Const
        nothing
    elseif EnzymeRules.width(config) == 1
        T(dret.val.ref.hi) - T(dret.val.step.hi) / (len.val - 1)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            T(dret.val[i].ref.hi)  - T(dret.val[i].step.hi) / (len.val - 1)
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


# Ranges
# Float64 ranges in Julia use bitwise `&` with higher precision
# to correct for numerical error, thus we put rules over the
# operations as this is not directly differentiable
function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    func::Const{Colon},
    RT::Type{
        <:Union{Const,DuplicatedNoNeed,Duplicated,BatchDuplicated,BatchDuplicatedNoNeed},
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
    ::Type{<:Active},
    start::Annotation{<:AbstractFloat},
    step::Annotation{<:AbstractFloat},
    stop::Annotation{<:AbstractFloat},
)

    if EnzymeRules.needs_primal(config)
        primal = func.val(start.val, step.val, stop.val)
    else
        primal = nothing
    end
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    func::Const{Colon},
    dret,
    tape::Nothing,
    start::Annotation{T1},
    step::Annotation{T2},
    stop::Annotation{T3},
) where {T1<:AbstractFloat,T2<:AbstractFloat,T3<:AbstractFloat}

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
    Ty::Const{Type{BigFloat}},
    RT::Type{<:Union{DuplicatedNoNeed,Duplicated,BatchDuplicated,BatchDuplicatedNoNeed}};
    kwargs...,
)

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            return RT(Ty.val(; kwargs...), Ty.val(; kwargs...))
        else
            tup = ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                Ty.val(; kwargs...)
            end
            return RT(Ty.val(; kwargs...), tup)
        end
    elseif EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            return Ty.val(; kwargs...)
        else
            return ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                Ty.val(; kwargs...)
            end
        end
    elseif EnzymeRules.needs_primal(config)
        return Ty.val(; kwargs...)
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    Ty::Const{Type{BigFloat}},
    RT::Type{<:Union{DuplicatedNoNeed,Duplicated,BatchDuplicated,BatchDuplicatedNoNeed}},
    kwargs...,
)
    primal = if EnzymeRules.needs_primal(config)
        Ty.val(; kwargs...)
    else
        nothing
    end
    shadow = if RT <: Const
        shadow = nothing
    else
        if EnzymeRules.width(config) == 1
            Ty.val(; kwargs...)
        else
            ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                Ty.val(; kwargs...)
            end
        end
    end
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    Ty::Const{Type{BigFloat}},
    RT::Type{<:Union{DuplicatedNoNeed,Duplicated,BatchDuplicated,BatchDuplicatedNoNeed}},
    tape,
    kwargs...,
)
    return ()
end

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    Ty::Const{typeof(Random.rand!)},
    RT::Type,
    rng::Annotation{rngty},
    dst::Annotation{<:Array{FT}},
    smpl::Annotation{<:Random.SamplerTrivial{Random.CloseOpen01{FT}}},
) where {rngty<:Union{TaskLocalRNG,Xoshiro},FT<:Union{Float32,Float64}}
    Ty.val(rng.val, dst.val, smpl.val)

    if !(dst isa Const)
        if EnzymeRules.width(config) == 1
            fill!(dst.dval, 0)
        else
            ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                fill!(dst.dval[i], 0)
                nothing
            end
        end
    end

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        dst
    elseif EnzymeRules.needs_shadow(config)
        dst.dval
    elseif EnzymeRules.needs_primal(config)
        dst.val
    else
        nothing
    end
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    Ty::Const{typeof(Random.rand!)},
    RT::Type,
    rng::Annotation{rngty},
    dst::Annotation{<:Array{FT}},
    smpl::Annotation{<:Random.SamplerTrivial{Random.CloseOpen01{FT}}},
) where {rngty<:Union{TaskLocalRNG,Xoshiro},FT<:Union{Float32,Float64}}
    Ty.val(rng.val, dst.val, smpl.val)
    if RT <: Duplicated || RT <: DuplicatedNoNeed
        fill!(dst.dval, 0)
        dst.dval
    elseif RT <: BatchDuplicated || RT <: BatchDuplicatedNoNeed
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            fill!(dst.dval[i], 0)
            nothing
        end
    end
    return EnzymeRules.AugmentedReturn(
        EnzymeRules.needs_primal(config) ? dst.val : nothing,
        EnzymeRules.needs_shadow(config) ? dst.dval : nothing,
        nothing,
    )
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    Ty::Const{typeof(Random.rand!)},
    RT::Type,
    tape,
    rng::Annotation{rngty},
    dst::Annotation{<:Array{FT}},
    smpl::Annotation{<:Random.SamplerTrivial{Random.CloseOpen01{FT}}},
) where {rngty<:Union{TaskLocalRNG,Xoshiro},FT<:Union{Float32,Float64}}
    return (nothing, nothing, nothing)
end

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    Ty::Const{typeof(Random.randn!)},
    RT::Type,
    rng::Annotation{<:Random.AbstractRNG},
    dst::Annotation{<:AbstractArray})

    Ty.val(rng.val, dst.val)

    if !(dst isa Const)
        if EnzymeRules.width(config) == 1
            make_zero!(dst.dval)
        else
            ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                make_zero!(dst.dval[i])
                nothing
            end
        end
    end

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        dst
    elseif EnzymeRules.needs_shadow(config)
        dst.dval
    elseif EnzymeRules.needs_primal(config)
        dst.val
    else
        nothing
    end
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    Ty::Const{typeof(Random.randn!)},
    RT::Type,
    rng::Annotation{<:Random.AbstractRNG},
    dst::Annotation{<:AbstractArray}
)
    Ty.val(rng.val, dst.val)
    if RT <: Duplicated || RT <: DuplicatedNoNeed
        make_zero!(dst.dval)
        dst.dval
    elseif RT <: BatchDuplicated || RT <: BatchDuplicatedNoNeed
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            make_zero!(dst.dval[i])
            nothing
        end
    end
    return EnzymeRules.AugmentedReturn(
        EnzymeRules.needs_primal(config) ? dst.val : nothing,
        EnzymeRules.needs_shadow(config) ? dst.dval : nothing,
        nothing,
    )
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    Ty::Const{typeof(Random.randn!)},
    RT::Type,
    tape,
    rng::Annotation{<:Random.AbstractRNG},
    dst::Annotation{<:AbstractArray})
    return (nothing, nothing)
end
