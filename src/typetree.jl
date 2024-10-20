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

function canonicalize!(tt::TypeTree, size, dl)
    API.EnzymeTypeTreeCanonicalizeInPlace(tt, size, dl)
end
function shift!(tt::TypeTree, dl, offset, maxSize, addOffset)
    API.EnzymeTypeTreeShiftIndiciesEq(tt, dl, offset, maxSize, addOffset)
end

function merge!(dst::TypeTree, src::TypeTree; consume = true)
    API.EnzymeMergeTypeTree(dst, src)
    if consume
        LLVM.dispose(src)
    end
    return nothing
end

@inline function typetree_primitive(t)
    return nothing
end
@inline function typetree_primitive(::Type{T}) where {T<:Integer}
    return API.DT_Integer
end
@inline function typetree_primitive(::Type{Char})
    return API.DT_Integer
end
@inline function typetree_primitive(::Type{Float16})
    return API.DT_Half
end
@inline function typetree_primitive(::Type{Float32})
    return API.DT_Float
end
@inline function typetree_primitive(::Type{Float64})
    return API.DT_Double
end


@static if VERSION >= v"1.11-"
    const TypeTreePrimitives = (Char, Float16, Float32, Float64, Core.BFloat16)
else
    const TypeTreePrimitives = (Char, Float16, Float32, Float64)
end

const TypeTreeEmptyPointers = (BigFloat, Any, Symbol, Union{})

function get_offsets(@nospecialize(T::Type))
    for sT in (Integer, TypeTreePrimitives...)
        if T <: sT
            return ((typetree_primitive(T), 0),)
        end
    end
    for sT in (DataType, AbstractString)
        if T <: sT
            return ((API.DT_Pointer, 0),)
        end
    end
    for sT in TypeTreeEmptyPointers
        if T == sT
            return ((API.DT_Pointer, 0),)
        end
    end
    @static if VERSION < v"1.11-"
        TypeTreePtrs = (Core.SimpleVector, Ptr, Core.LLVMPtr, Array)
    else
        TypeTreePtrs = (Core.SimpleVector, Ptr, Core.LLVMPtr, Array, GenericMemory)
    end
    for sT in TypeTreePtrs
        if T <: sT
            return ((API.DT_Pointer, 0),)
        end
    end

    @assert !(T <: AbstractFloat)

    if fieldcount(T) == 0
        return ()
    end

    results = Tuple{API.CConcreteType,Int}[]
    for f = 1:fieldcount(T)
        offset = fieldoffset(T, f)
        subT = typed_fieldtype(T, f)

        if !allocatedinline(subT) || subT isa UnionAll || subT isa Union || subT == Union{}
            push!(results, (API.DT_Pointer, offset))
            continue
        end

        for (sT, sO) in get_offsets(subT)
            push!(results, (sT, sO + offset))
        end
    end
    return results
end

function to_fullmd(@nospecialize(T::Type), offset::Int, lim::Int)
    mds = LLVM.Metadata[]
    offs = get_offsets(T)

    minoff = -1
    for (sT, sO) in offs
        if sO >= offset
            if sO == offset
                minoff = sO
            end
        else
            minoff = max(minoff, sO)
        end
    end

    for (sT, sO) in offs
        if sO != minoff && (sO < offset)
            continue
        end
        if sO >= lim + offset
            continue
        end
        if sT == API.DT_Pointer
            push!(mds, LLVM.MDString("Pointer"))
        elseif sT == API.DT_Integer
            push!(mds, LLVM.MDString("Integer"))
        elseif sT == API.DT_Half
            push!(mds, LLVM.MDString("Float@half"))
        elseif sT == API.DT_Float
            push!(mds, LLVM.MDString("Float@float"))
        elseif sT == API.DT_BFloat16
            push!(mds, LLVM.MDString("Float@bfloat16"))
        elseif sT == API.DT_Double
            push!(mds, LLVM.MDString("Float@double"))
        else
            @assert false
        end
        push!(mds, LLVM.Metadata(LLVM.ConstantInt(max(0, sO - offset))))
    end
    return LLVM.MDNode(mds)
end

function to_md(tt::TypeTree, ctx)
    return LLVM.Metadata(
        LLVM.MetadataAsValue(
            ccall(
                (:EnzymeTypeTreeToMD, API.libEnzyme),
                LLVM.API.LLVMValueRef,
                (API.CTypeTreeRef, LLVM.API.LLVMContextRef),
                tt,
                ctx,
            ),
        ),
    )
end

const TypeTreeTable = IdDict{Any,Union{Nothing,TypeTree}}

"""
    function typetree(T, ctx, dl, seen=TypeTreeTable())

Construct a Enzyme typetree from a Julia type.

!!! warning
    When using a memoized lookup by providing `seen` across multiple calls to typtree
    the user must call `copy` on the returned value before mutating it.
"""
function typetree(@nospecialize(T::Type), ctx, dl, seen = TypeTreeTable())
    if haskey(seen, T)
        tree = seen[T]
        if tree === nothing
            return TypeTree() # stop recursion, but don't cache
        end
    else
        seen[T] = nothing # place recursion marker
        tree = typetree_inner(T, ctx, dl, seen)
        seen[T] = tree
    end
    return tree::TypeTree
end

function typetree_inner(::Type{<:Integer}, ctx, dl, seen::TypeTreeTable)
    return TypeTree(API.DT_Integer, -1, ctx)
end
for sT in TypeTreePrimitives
    @eval function typetree_inner(::Type{$sT}, ctx, dl, seen::TypeTreeTable)
        return TypeTree($(typetree_primitive(sT)), -1, ctx)
    end
end

function typetree_inner(::Type{<:DataType}, ctx, dl, seen::TypeTreeTable)
    return TypeTree()
end
function typetree_inner(::Type{<:AbstractString}, ctx, dl, seen::TypeTreeTable)
    return TypeTree()
end
for sT in TypeTreeEmptyPointers
    @eval function typetree_inner(::Type{$sT}, ctx, dl, seen::TypeTreeTable)
        return TypeTree()
    end
end


function typetree_inner(::Type{Core.SimpleVector}, ctx, dl, seen::TypeTreeTable)
    tt = TypeTree()
    for i = 0:(sizeof(Csize_t)-1)
        merge!(tt, TypeTree(API.DT_Integer, i, ctx))
    end
    return tt
end

function typetree_inner(
    ::Type{<:Union{Ptr{T},Core.LLVMPtr{T}}},
    ctx,
    dl,
    seen::TypeTreeTable,
) where {T}
    tt = copy(typetree(T, ctx, dl, seen))
    merge!(tt, TypeTree(API.DT_Pointer, ctx))
    only!(tt, -1)
    return tt
end

@static if VERSION < v"1.11-"
    function typetree_inner(::Type{<:Array{T}}, ctx, dl, seen::TypeTreeTable) where {T}
        offset = 0

        tt = copy(typetree(T, ctx, dl, seen))
        if !allocatedinline(T) && Base.isconcretetype(T)
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

        for i = offset:(sizeofstruct-1)
            merge!(tt, TypeTree(API.DT_Integer, i, ctx))
        end
        return tt
    end
else
    function typetree_inner(
        ::Type{<:GenericMemory{kind,T}},
        ctx,
        dl,
        seen::TypeTreeTable,
    ) where {kind,T}
        offset = 0
        tt = copy(typetree(T, ctx, dl, seen))
        if !allocatedinline(T) && Base.isconcretetype(T)
            merge!(tt, TypeTree(API.DT_Pointer, ctx))
            only!(tt, 0)
        end
        merge!(tt, TypeTree(API.DT_Pointer, ctx))
        only!(tt, sizeof(Csize_t))

        for i = 0:(sizeof(Csize_t)-1)
            merge!(tt, TypeTree(API.DT_Integer, i, ctx))
        end
        return tt
    end

    function typetree_inner(
        AT::Type{<:GenericMemoryRef{kind,T}},
        ctx,
        dl,
        seen::TypeTreeTable,
    ) where {kind,T}
        offset = 0
        tt = copy(typetree(T, ctx, dl, seen))
        if !allocatedinline(T) && Base.isconcretetype(T)
            Enzyme.merge!(tt, TypeTree(API.DT_Pointer, ctx))
            only!(tt, 0)
        end
        Enzyme.merge!(tt, TypeTree(API.DT_Pointer, ctx))
        only!(tt, 0)

        for f = 2:fieldcount(AT)
            offset = fieldoffset(AT, f)
            subT = typed_fieldtype(AT, f)
            
            subtree = copy(typetree(subT, ctx, dl, seen))

            # Allocated inline so adjust first path
            if allocatedinline(subT)
                shift!(subtree, dl, 0, sizeof(subT), offset)
            else
                Enzyme.merge!(subtree, TypeTree(API.DT_Pointer, ctx))
                only!(subtree, offset)
            end

            Enzyme.merge!(tt, subtree)
        end
        canonicalize!(tt, sizeof(AT), dl)

        return tt
    end
end

import Base: ismutabletype

function typetree_inner(@nospecialize(T::Type), ctx, dl, seen::TypeTreeTable)
    if T isa UnionAll || T isa Union || T == Union{} || Base.isabstracttype(T)
        return TypeTree()
    end

    if T === Tuple
        return TypeTree()
    end

    if is_concrete_tuple(T) && any(T2 isa Core.TypeofVararg for T2 in T.parameters)
        return TypeTree()
    end

    if T <: AbstractFloat
        throw(AssertionError("Unknown floating point type $T"))
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
    for f = 1:fieldcount(T)
        offset = fieldoffset(T, f)
        subT = typed_fieldtype(T, f)

        if subT isa UnionAll || subT isa Union || subT == Union{}
            if !allocatedinline(subT)
                subtree = TypeTree(API.DT_Pointer, offset, ctx)
                merge!(tt, subtree)
            end
            # FIXME: Handle union
            continue
        end
        
        subtree = copy(typetree(subT, ctx, dl, seen))

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
    args_kv = Base.unsafe_convert(
        Ptr{API.IntList},
        Base.cconvert(Ptr{API.IntList}, fnti.known_values),
    )
    rTT = Base.unsafe_convert(API.CTypeTreeRef, Base.cconvert(API.CTypeTreeRef, fnti.rTT))

    tts = API.CTypeTreeRef[]
    for tt in fnti.argTTs
        raw_tt = Base.unsafe_convert(API.CTypeTreeRef, Base.cconvert(API.CTypeTreeRef, tt))
        push!(tts, raw_tt)
    end
    argTTs = Base.unsafe_convert(
        Ptr{API.CTypeTreeRef},
        Base.cconvert(Ptr{API.CTypeTreeRef}, tts),
    )
    return API.CFnTypeInfo(argTTs, rTT, args_kv)
end
