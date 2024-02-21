# TODO:
# - type tags?
# - recursive types

import LLVM: refcheck
import GPUCompiler
LLVM.@checked struct TypeTree
    ref::API.CTypeTreeRef
end
Base.unsafe_convert(::Type{API.CTypeTreeRef}, tt::TypeTree) = tt.ref
LLVM.dispose(tt::TypeTree) = API.EnzymeFreeTypeTree(tt)

TypeTree() = TypeTree(API.EnzymeNewTypeTree())
TypeTree(CT, ctx) = TypeTree(API.EnzymeNewTypeTreeCT(CT, ctx))
# function typetree_inner(CT, idx, ctx)
#     tt = TypeTree(CT, ctx)
#     only!(tt, idx)
#     return tt
# end
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

function canonicalize!(tt::TypeTree, size, dl)
    API.EnzymeTypeTreeCanonicalizeInPlace(tt, size, dl)
end
function shift!(tt::TypeTree, dl, offset, maxSize, addOffset)
    API.EnzymeTypeTreeShiftIndiciesEq(tt, dl, offset, maxSize, addOffset)
end

function merge!(dst::TypeTree, src::TypeTree; consume=true)
    API.EnzymeMergeTypeTree(dst, src)
    if consume
        LLVM.dispose(src)
    end
    return nothing
end

function to_md(tt::TypeTree, ctx)
    return LLVM.Metadata(LLVM.MetadataAsValue(ccall((:EnzymeTypeTreeToMD, API.libEnzyme), LLVM.API.LLVMValueRef, (API.CTypeTreeRef,LLVM.API.LLVMContextRef), tt, ctx)))
end

const TypeTreeTable = IdDict{DataType, Union{Nothing, TypeTree}}

function typetree(@nospecialize(T), ctx, dl, seen=TypeTreeTable())
    if haskey(seen, T)
        tree = seen[T]
        if tree === nothing
            return TypeTree() # stop recursion, but don't cache
        else
            return tree::TypeTree
        end
    else
        seen[T] = nothing # place recursion marker
        tree = typetree_inner(T, ctx, dl, seen)
        seen[T] = tree 
    end
end

function typetree_inner(::Type{T}, ctx, dl, seen::TypeTreeTable) where T <: Integer
    return TypeTree(API.DT_Integer, -1, ctx)
end

function typetree_inner(::Type{Char}, ctx, dl, seen::TypeTreeTable)
    return TypeTree(API.DT_Integer, -1, ctx)
end

function typetree_inner(::Type{Float16}, ctx, dl, seen::TypeTreeTable)
    return TypeTree(API.DT_Half, -1, ctx)
end

function typetree_inner(::Type{Float32}, ctx, dl, seen::TypeTreeTable)
    return TypeTree(API.DT_Float, -1, ctx)
end

function typetree_inner(::Type{Float64}, ctx, dl, seen::TypeTreeTable)
    return TypeTree(API.DT_Double, -1, ctx)
end

function typetree_inner(::Type{T}, ctx, dl, seen::TypeTreeTable) where T<:AbstractFloat
    GPUCompiler.@safe_warn "Unknown floating point type" T
    return TypeTree()
end

function typetree_inner(::Type{<:DataType}, ctx, dl, seen::TypeTreeTable)
    return TypeTree()
end

function typetree_inner(::Type{Any}, ctx, dl, seen::TypeTreeTable)
    return TypeTree()
end

function typetree_inner(::Type{Symbol}, ctx, dl, seen::TypeTreeTable)
    return TypeTree()
end

function typetree_inner(::Type{Core.SimpleVector}, ctx, dl, seen::TypeTreeTable)
    tt = TypeTree()
    for i in 0:(sizeof(Csize_t)-1)
        merge!(tt, TypeTree(API.DT_Integer, i, ctx))
    end
    return tt
end

function typetree_inner(::Type{Union{}}, ctx, dl, seen::TypeTreeTable)
    return TypeTree()
end

function typetree_inner(::Type{<:AbstractString}, ctx, dl, seen::TypeTreeTable)
    return TypeTree()
end

function typetree_inner(::Type{<:Union{Ptr{T}, Core.LLVMPtr{T}}}, ctx, dl, seen::TypeTreeTable) where T
    tt = copy(typetree(T, ctx, dl, seen))
    merge!(tt, TypeTree(API.DT_Pointer, ctx))
    only!(tt, -1)
    return tt
end

function typetree_inner(::Type{<:Array{T}}, ctx, dl, seen::TypeTreeTable) where T
    offset = 0

    tt = copy(typetree(T, ctx, dl, seen))
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

    for i in offset:(sizeofstruct-1)
        merge!(tt, TypeTree(API.DT_Integer, i, ctx))
    end
    return tt
end

if VERSION >= v"1.7.0-DEV.204"
    import Base: ismutabletype
else
    ismutabletype(T) = isa(T, DataType) && T.mutable
end

function typetree_inner(@nospecialize(T), ctx, dl, seen::TypeTreeTable)
    if T isa UnionAll || T isa Union || T == Union{} || Base.isabstracttype(T)
        return TypeTree()
    end

    if T === Tuple
        return TypeTree()
    end

    try
        fieldcount(T)
    catch
        GPUCompiler.@safe_warn "Type does not have a definite number of fields" T
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
    end

    if !Base.isconcretetype(T)
        return TypeTree(API.DT_Pointer, -1, ctx)
    end

    tt = TypeTree()
    for f in 1:fieldcount(T)
        offset  = fieldoffset(T, f)
        subT    = fieldtype(T, f)
        subtree = copy(typetree(subT, ctx, dl, seen))

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
    canonicalize!(tt, sizeof(T), dl)
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
