# return result and if contains any
function to_tape_type(@nospecialize(Type::LLVM.LLVMType))::Tuple{DataType, Bool}
    if Type isa LLVM.StructType
        tys = DataType[]
        containsAny = false
        syms = Symbol[]
        for (i, e) in enumerate(Type.elements)
            T, sub = to_tape_type(e)
            containsAny |= sub
            push!(tys, T)
            push!(syms, Symbol(i))
        end
        Tup = Tuple{tys...}
        if containsAny
            res = (syms...,)
            return NamedTuple{res,Tup}, false
        else
            return Tup, false
        end
    end
    if Type isa LLVM.PointerType
        addrspace = Type.addrspace
        if 10 <= addrspace <= 12
            return Any, true
        elseif LLVM.isopaque(Type)
            return Core.LLVMPtr{Cvoid,Int(addrspace)}, false
        else
            e = Type.element_type
            if e isa LLVM.FunctionType
                return Core.LLVMPtr{Cvoid,Int(addrspace)}, false
            else
                return Core.LLVMPtr{to_tape_type(e)[1],Int(addrspace)}, false
            end
        end
    end
    if Type isa LLVM.ArrayType
        T, sub = to_tape_type(Type.element_type)
        len = Type.length
        Tup = NTuple{len,T}
        if sub
            return NamedTuple{ntuple(Core.Symbol, Val(len)),Tup}, false
        else
            return Tup, false
        end
    end
    if Type isa LLVM.VectorType
        T, sub = to_tape_type(Type.element_type)
        len = Type.length
        Tup = NTuple{len,Core.VecElement{T}}
        if sub
            return NamedTuple{ntuple(Core.Symbol, Val(len)),Tup}, false
        else
            return Tup, false
        end
    end
    if Type isa LLVM.IntegerType
        N = Type.width
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
    # Workaround: LLVM.jl has no public Julia types for the floating-point types
    # (`LLVM.HalfType` & co. are constructors), so compare their type kind.
    tkind = LLVM.API.LLVMGetTypeKind(Type)
    if tkind == LLVM.TypeKind.Half
        return Float16, false
    end
    @static if isdefined(Core, :BFloat16)
        if tkind == LLVM.TypeKind.BFloat
            return Core.BFloat16, false
        end
    end
    if tkind == LLVM.TypeKind.Float
        return Float32, false
    end
    if tkind == LLVM.TypeKind.Double
        return Float64, false
    end
    if tkind == LLVM.TypeKind.FP128
        return Float128, false
    end
    error("Can't construct tape type for $(string(Type))")
end

function tape_type(@nospecialize(LLVMType::LLVM.LLVMType))
    TT, isAny = to_tape_type(LLVMType)
    if isAny
        return AnonymousStruct(Tuple{Any})
    end
    return TT
end

from_tape_type(::Type{T}) where {T<:AbstractFloat} = convert(LLVMType, T)
from_tape_type(::Type{T}) where {T<:Integer} = convert(LLVMType, T)
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

