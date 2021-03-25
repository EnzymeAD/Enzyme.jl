# TODO:
# - type tags?
# - eltype of Ptr and Array is only `first Element`
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

function shift!(tt::TypeTree, dl, offset, maxSize, addOffset)
    API.EnzymeTypeTreeShiftIndiciesEq(tt, dl, offset, maxSize, addOffset)
end

function merge!(dst::TypeTree, src::TypeTree; consume=true)
    API.EnzymeMergeTypeTree(dst, src)
    LLVM.dispose(src)
    return nothing
end

function typetree(::Type{Nothing}, ctx, dl)
    TypeTree()
end

function typetree(::Type{T}, ctx, dl) where T <: Integer
    tt = TypeTree()
    for i in 1:sizeof(T)
        merge!(tt, TypeTree(API.DT_Integer, i-1, ctx))
    end
    return tt
end

function typetree(::Type{Float16}, ctx, dl)
    return TypeTree(API.DT_Half, 0, ctx)
end

function typetree(::Type{Float32}, ctx, dl)
    return TypeTree(API.DT_Float, 0, ctx)
end

function typetree(::Type{Float64}, ctx, dl)
    return TypeTree(API.DT_Double, 0, ctx)
end

function typetree(::Type{<:Union{Ptr{T}, Core.LLVMPtr{T}}}, ctx, dl) where T
    tt = typetree(T, ctx, dl)
    merge!(tt, TypeTree(API.DT_Pointer, ctx))
    only!(tt, 0)
    return tt
end

function typetree(::Type{<:Array{T}}, ctx, dl) where T
    offset = 0

    tt = typetree(T, ctx, dl)
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


function typetree(@nospecialize(T), ctx, dl)
    if fieldcount(T) == 0
        error("$T is unknown leaf")
    end

    tt = TypeTree()
    for f in 1:fieldcount(T)
        offset  = fieldoffset(T, f)
        subT    = fieldtype(T, f)
        subtree = typetree(subT, ctx, dl)

        # Allocated inline so adjust first path
        if subT.isinlinealloc
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
