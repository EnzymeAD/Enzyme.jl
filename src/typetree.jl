# TODO:
# - type tags?
# - recursive types

import LLVM: refcheck
LLVM.@checked struct TypeTree
    ref::API.CTypeTreeRef
end
Base.unsafe_convert(::Type{API.CTypeTreeRef}, tt::TypeTree) = tt.ref
LLVM.dispose(tt::TypeTree) = API.EnzymeFreeTypeTree(tt)

TypeTree() = TypeTree(API.EnzymeNewTypeTree())
TypeTree(CT, ctx) = TypeTree(API.EnzymeNewTypeTreeCT(CT, ctx))
function TypeTree(CT, idx, ctx)
    tt = TypeTree(CT, ctx)
    only!(tt, idx)
    return tt
end
Base.copy(tt::TypeTree) = TypeTree(API.EnzymeNewTypeTreeTR(tt))
Base.copy!(dst::TypeTree, src::TypeTree) = API.EnzymeSetTypeTree(dst, src)

function Base.string(tt::TypeTree)
    raw = API.EnzymeTypeTreeToString(tt)
    str = Base.unsafe_string(raw)
    API.EnzymeStringFree(raw)
    return str
end

# function Base.show(io::IO, ::MIME"text/plain", tt::TypeTree)
#     print(io, "TypeTree: ")
#     for data in tt.data
#         println(io)
#         print(io, "- ")
#         show(io, data)
#     end
# end

function only!(tt::TypeTree, offset::Integer)
    API.EnzymeTypeTreeOnlyEq(tt, offset)
end

function data0!(tt::TypeTree)
    API.EnzymeTypeTreeData0Eq(tt)
end

function shift!(tt::TypeTree, dl, offset, maxSize, addOffset)
    API.EnzymeTypeTreeShiftIndiciesEq(tt, dl, offset, maxSize, addOffset)
end

function merge!(dst::TypeTree, src::TypeTree; consume=true)
    API.EnzymeMergeTypeTree(dst, src)
    LLVM.dispose(src)
    return nothing
end

function typetree(::Type{T}, ctx, dl, seen=nothing) where T <: Integer
    tt = TypeTree()
    for i in 1:sizeof(T)
        merge!(tt, TypeTree(API.DT_Integer, i-1, ctx))
    end
    return tt
end

function typetree(::Type{Float16}, ctx, dl, seen=nothing)
    return TypeTree(API.DT_Half, -1, ctx)
end

function typetree(::Type{Float32}, ctx, dl, seen=nothing)
    return TypeTree(API.DT_Float, -1, ctx)
end

function typetree(::Type{Float64}, ctx, dl, seen=nothing)
    return TypeTree(API.DT_Double, -1, ctx)
end

function typetree(::Type{T}, ctx, dl, seen=nothing) where T<:AbstractFloat
    @warn "Unknown floating point type" T
    return TypeTree()
end

function typetree(::Type{<:DataType}, ctx, dl, seen=nothing)
    return TypeTree()
end

function typetree(::Type{Any}, ctx, dl, seen=nothing)
    return TypeTree()
end

function typetree(::Type{Symbol}, ctx, dl, seen=nothing)
    return TypeTree()
end

function typetree(::Type{Core.SimpleVector}, ctx, dl, seen=nothing)
    tt = TypeTree()
    for i in 0:(sizeof(Csize_t)-1)
        merge!(tt, TypeTree(API.DT_Integer, i, ctx))
    end
    return tt
end

function typetree(::Type{<:AbstractString}, ctx, dl, seen=nothing)
    return TypeTree()
end

function typetree(::Type{<:Union{Ptr{T}, Core.LLVMPtr{T}}}, ctx, dl, seen=nothing) where T
    tt = typetree(T, ctx, dl, seen)
    merge!(tt, TypeTree(API.DT_Pointer, ctx))
    only!(tt, -1)
    return tt
end

function typetree(::Type{<:Array{T}}, ctx, dl, seen=nothing) where T
    offset = 0

    tt = typetree(T, ctx, dl, seen)
    if !allocatedinline(T)
        merge!(tt, TypeTree(API.DT_Pointer, ctx))
        only!(tt, 0)
    end
    merge!(tt, TypeTree(API.DT_Pointer, ctx))
    only!(tt, offset)

    offset += sizeof(Ptr{Cvoid})

    sizeofstruct = offset + 2 + 2 + 4 + 2 * sizeof(Csize_t)
    if true # STORE_ARRAY_LEN
        sizeofstruct += sizeof(Csize_t)
    end

    for i in offset:sizeofstruct
        merge!(tt, TypeTree(API.DT_Integer, i, ctx))
    end
    return tt
end

if VERSION >= v"1.7.0-DEV.204"
    import Base: ismutabletype
else
    ismutabletype(T) = isa(T, DataType) && T.mutable
end

function typetree(@nospecialize(T), ctx, dl, seen=nothing)
    if T isa UnionAll || T isa Union || T == Union{} || Base.isabstracttype(T)
        return TypeTree()
    end

    if seen !== nothing && T ∈ seen
        @warn "Recursive type" T
        return TypeTree()
    end
    if seen === nothing
        seen = Set{DataType}()
    else
        seen = copy(seen) # need to copy otherwise we'll count siblings as recursive
    end
    push!(seen, T)

    try
        fieldcount(T)
    catch
        @warn "Type does not have a definite number of fields" T
        return TypeTree()
    end

    if fieldcount(T) == 0
        if T <: Function
            return TypeTree()
        end
        if isa(T, DataType) && !ismutabletype(T) # singleton
            return TypeTree()
        end
        if T <: Module
            return TypeTree()
        end
        error("$T is unknown leaf")
    end

    tt = TypeTree()
    for f in 1:fieldcount(T)
        offset  = fieldoffset(T, f)
        subT    = fieldtype(T, f)
        subtree = typetree(subT, ctx, dl, seen)

        if subT isa UnionAll || subT isa Union || subT == Union{}
            continue
        end

        # Allocated inline so adjust first path
        if allocatedinline(subT)
            shift!(subtree, dl, 0, sizeof(subT), offset)
        else
            merge!(subtree, TypeTree(API.DT_Pointer, ctx))
            only!(subtree, offset)
        end

        merge!(tt, subtree)
    end
    return tt
end

struct FnTypeInfo
    rTT::TypeTree
    argTTs::Vector{TypeTree}
    known_values::Vector{API.IntList}
end
Base.cconvert(::Type{API.CFnTypeInfo}, fnti::FnTypeInfo) = fnti
function Base.unsafe_convert(::Type{API.CFnTypeInfo}, fnti::FnTypeInfo)
    args_kv = Base.unsafe_convert(Ptr{API.IntList}, Base.cconvert(Ptr{API.IntList}, fnti.known_values))
    rTT = Base.unsafe_convert(API.CTypeTreeRef, Base.cconvert(API.CTypeTreeRef, fnti.rTT))

    tts = API.CTypeTreeRef[]
    for tt in fnti.argTTs
        raw_tt = Base.unsafe_convert(API.CTypeTreeRef, Base.cconvert(API.CTypeTreeRef, tt))
        push!(tts, raw_tt)
    end
    argTTs = Base.unsafe_convert(Ptr{API.CTypeTreeRef}, Base.cconvert(Ptr{API.CTypeTreeRef}, tts))
    return API.CFnTypeInfo(argTTs, rTT, args_kv)
end
