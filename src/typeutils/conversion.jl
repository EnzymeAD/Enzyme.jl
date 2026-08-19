# The tape ABI assumes that the Julia lowering of the tape type is
# memory-layout-identical to Enzyme's own LLVM tape type: the augmented forward
# stores the tape using Enzyme's layout into a slot typed by the Julia tape type,
# and the reverse pass reads it back.  Julia only keeps a struct-typed field
# inline in an enclosing object when `Base.allocatedinline` holds for it, and
# that fails as soon as the type has pointers *and* any of its own fields is
# larger than 32767 bytes -- `jl_fielddesc16_t` only has a 15-bit size field, so
# the layout falls back to `fielddesc_type == 2`, which `jl_datatype_isinlinealloc`
# rejects.  Such a field is boxed by Julia, so the two layouts diverge and the
# tape can neither be stored nor read back (see `calling_conv_fixup`).
#
# Model that explicitly: any nested aggregate Julia would not store inline is
# mapped to `Any`, and `julia_conv_fixup` / `calling_conv_fixup` box and unbox it
# in the ABI wrappers.  The Julia type of the box's contents is recovered by
# calling `tape_type` on the corresponding LLVM type again, which is a top-level
# call and hence not boxed itself.
@inline function tape_type_stored_inline(@nospecialize(TT::Type))::Bool
    return Base.isconcretetype(TT) && Base.allocatedinline(TT)
end

# return result and if contains any
function to_tape_type(
    Type::LLVM.API.LLVMTypeRef,
    nested::Bool = false,
    boxnested::Bool = true,
)::Tuple{DataType,Bool}
    tkind = LLVM.API.LLVMGetTypeKind(Type)
    if tkind == LLVM.API.LLVMStructTypeKind
        tys = DataType[]
        nelems = LLVM.API.LLVMCountStructElementTypes(Type)
        containsAny = false
        syms = Symbol[]
        for i = 1:nelems
            e = LLVM.API.LLVMStructGetTypeAtIndex(Type, i - 1)
            T, sub = to_tape_type(e, true, boxnested)
            containsAny |= sub
            push!(tys, T)
            push!(syms, Symbol(i))
        end
        Tup = Tuple{tys...}
        res = if containsAny
            NamedTuple{(syms...,),Tup}
        else
            Tup
        end
        if nested && boxnested && !tape_type_stored_inline(res)
            return Any, true
        end
        return res, false
    end
    if tkind == LLVM.API.LLVMPointerTypeKind
        addrspace = LLVM.API.LLVMGetPointerAddressSpace(Type)
        if 10 <= addrspace <= 12
            return Any, true
        elseif LLVM.is_opaque(LLVM.PointerType(Type))
            return Core.LLVMPtr{Cvoid,Int(addrspace)}, false
        else
            e = LLVM.API.LLVMGetElementType(Type)
            tkind2 = LLVM.API.LLVMGetTypeKind(e)
            if tkind2 == LLVM.API.LLVMFunctionTypeKind
                return Core.LLVMPtr{Cvoid,Int(addrspace)}, false
            else
                return Core.LLVMPtr{to_tape_type(e)[1],Int(addrspace)}, false
            end
        end
    end
    if tkind == LLVM.API.LLVMArrayTypeKind
        e = LLVM.API.LLVMGetElementType(Type)
        T, sub = to_tape_type(e, true, boxnested)
        len = Int(LLVM.API.LLVMGetArrayLength(Type))
        Tup = NTuple{len,T}
        res = if sub
            NamedTuple{ntuple(Core.Symbol, Val(len)),Tup}
        else
            Tup
        end
        if nested && boxnested && !tape_type_stored_inline(res)
            return Any, true
        end
        return res, false
    end
    if tkind == LLVM.API.LLVMVectorTypeKind
        e = LLVM.API.LLVMGetElementType(Type)
        T, sub = to_tape_type(e, true, boxnested)
        len = Int(LLVM.API.LLVMGetVectorSize(Type))
        Tup = NTuple{len,Core.VecElement{T}}
        res = if sub
            NamedTuple{ntuple(Core.Symbol, Val(len)),Tup}
        else
            Tup
        end
        if nested && boxnested && !tape_type_stored_inline(res)
            return Any, true
        end
        return res, false
    end
    if tkind == LLVM.API.LLVMIntegerTypeKind
        N = LLVM.API.LLVMGetIntTypeWidth(Type)
        if N == 1
            return Bool, false
        elseif N == 8
            return UInt8, false
        elseif N == 16
            return UInt16, false
        elseif N == 32
            return UInt32, false
        elseif N == 64
            return UInt64, false
        elseif N == 128
            return UInt128, false
        elseif N == 256
            return UInt256, false
        elseif N == 512
            return UInt512, false
        elseif N == 1024
            return UInt1024, false
        elseif N == 2048
            return UInt2048, false
        else
            error("Can't construct tape type for integer of width $N")
        end
    end
    if tkind == LLVM.API.LLVMHalfTypeKind
        return Float16, false
    end
    @static if isdefined(Core, :BFloat16)
        if tkind == LLVM.API.LLVMBFloatTypeKind
            return Core.BFloat16, false
        end
    end
    if tkind == LLVM.API.LLVMFloatTypeKind
        return Float32, false
    end
    if tkind == LLVM.API.LLVMDoubleTypeKind
        return Float64, false
    end
    if tkind == LLVM.API.LLVMFP128TypeKind
        return Float128, false
    end
    error("Can't construct tape type for $Type $(string(Type)) $tkind")
end

# `boxnested = false` returns the type as it was computed before nested boxing
# existed: faithful to Enzyme's LLVM layout, but possibly not something Julia can
# store inline.  Only correct where the type is used to size/zero raw tape
# storage rather than to describe a Julia field.
function tape_type(@nospecialize(LLVMType::LLVM.LLVMType); boxnested::Bool = true)
    TT, isAny = to_tape_type(LLVMType.ref, false, boxnested)
    if isAny
        return AnonymousStruct(Tuple{Any})
    end
    return TT
end

from_tape_type(::Type{T}) where {T<:AbstractFloat} = convert(LLVMType, T)
from_tape_type(::Type{T}) where {T<:Integer} = convert(LLVMType, T)
# `Tuple{}` matches `NTuple{Size,T}` with `T` unbound, so it needs its own method
from_tape_type(::Type{Tuple{}}) = LLVM.StructType(LLVM.LLVMType[])
from_tape_type(::Type{NTuple{Size,T}}) where {Size,T} =
    LLVM.ArrayType(from_tape_type(T), Size)
from_tape_type(::Type{Core.VecElement{T}}) where {T} = from_tape_type(T)
from_tape_type(::Type{NTuple{Size,Core.VecElement{T}}}) where {Size,T} =
    LLVM.VectorType(from_tape_type(T), Size)
from_tape_type(::Type{Core.LLVMPtr{T,Addr}}) where {T,Addr} =
    LLVM.PointerType(from_tape_type(UInt8), Addr)
# from_tape_type(::Type{Core.LLVMPtr{T, Addr}}, ctx) where {T, Addr} = LLVM.PointerType(from_tape_type(T, ctx), Addr)
from_tape_type(::Type{Any}) = LLVM.PointerType(LLVM.StructType(LLVM.LLVMType[]), Tracked)
function from_tape_type(::Type{NamedTuple{A,B}}) where {A,B}
    from_tape_type(B)
end
function from_tape_type(::Type{B}) where {B<:Tuple}
    ar = LLVM.LLVMType[from_tape_type(b) for b in B.parameters]
    if length(B.parameters) >= 1 && all(ar[1] == b for b in ar)
        return LLVM.ArrayType(ar[1], length(B.parameters))
    else
        return LLVM.StructType(LLVM.LLVMType[from_tape_type(b) for b in B.parameters])
    end
end


# Enzyme's heap-allocated tapes and caches are raw storage: Enzyme writes them in
# its own LLVM layout, and the only thing Julia contributes is an allocation of
# the right size whose type lets the GC find the tracked pointers inside.  The
# nested type from `to_tape_type` cannot always serve that role -- Julia boxes any
# field of 32768 bytes or more that contains pointers, so the type ends up
# describing far less memory than the allocation holds (`sizeof` 248 instead of
# 34144, say).
#
# `raw_storage_type` sidesteps the nesting entirely: it flattens the LLVM type to
# its leaves and rebuilds a flat NamedTuple, using explicit `NTuple{n, UInt8}`
# padding so every leaf lands on its original byte offset.  Pointer leaves stay
# `Any`, so the GC still traces them.  Padding runs are split below the 32767-byte
# field limit so the result also stays inline-allocatable where it can be, which
# is what `NTuple{N, TT}` needs when Enzyme allocates more than one element.
const JL_MAX_INLINE_FIELD_SIZE = 32767

function push_padding!(tys::Vector{DataType}, nbytes::Int)
    while nbytes > 0
        chunk = min(nbytes, JL_MAX_INLINE_FIELD_SIZE)
        push!(tys, NTuple{chunk,UInt8})
        nbytes -= chunk
    end
    return nothing
end

function collect_leaf_types!(
    leaves::Vector{Tuple{Int,DataType}},
    dl::LLVM.DataLayout,
    @nospecialize(ty::LLVM.LLVMType),
    offset::Int,
)
    if isa(ty, LLVM.StructType) && !LLVM.ispacked(ty)
        for (i, e) in enumerate(LLVM.elements(ty))
            collect_leaf_types!(leaves, dl, e, offset + Int(LLVM.offsetof(dl, ty, i - 1)))
        end
        return nothing
    end
    if isa(ty, LLVM.ArrayType)
        e = eltype(ty)
        stride = Int(LLVM.abi_size(dl, e))
        for i = 1:length(ty)
            collect_leaf_types!(leaves, dl, e, offset + (i - 1) * stride)
        end
        return nothing
    end
    # Primitives, pointers, and anything we do not split (packed structs,
    # vectors) become leaves -- but only when Julia gives them the same size,
    # otherwise the offsets computed above would be fiction.
    T = to_tape_type(ty.ref, false, false)[1]
    esz = T === Any ? sizeof(Ptr{Cvoid}) : sizeof(T)
    esz == Int(LLVM.abi_size(dl, ty)) ||
        throw(ArgumentError("no faithful Julia type for $(string(ty))"))
    push!(leaves, (offset, T))
    return nothing
end

# Returns `nothing` when no faithful type could be built, so callers can keep
# their existing diagnostics rather than allocating something mislaid out.
function raw_storage_type(dl::LLVM.DataLayout, @nospecialize(ty::LLVM.LLVMType))
    total = Int(LLVM.abi_size(dl, ty))
    leaves = Tuple{Int,DataType}[]
    try
        collect_leaf_types!(leaves, dl, ty, 0)
    catch
        return nothing
    end
    tys = DataType[]
    placed = Tuple{Int,Int}[]   # (field index, intended byte offset)
    pos = 0
    for (off, T) in leaves
        off < pos && return nothing
        off > pos && push_padding!(tys, off - pos)
        push!(tys, T)
        push!(placed, (length(tys), off))
        pos = off + (T === Any ? sizeof(Ptr{Cvoid}) : sizeof(T))
    end
    pos > total && return nothing
    pos < total && push_padding!(tys, total - pos)
    res = NamedTuple{ntuple(Core.Symbol, length(tys)),Tuple{tys...}}
    # Only usable if Julia agrees on the size and put every leaf exactly where
    # Enzyme's layout has it -- otherwise the GC would trace the wrong words.
    sizeof(res) == total || return nothing
    for (idx, off) in placed
        Int(fieldoffset(res, idx)) == off || return nothing
    end
    return res
end
