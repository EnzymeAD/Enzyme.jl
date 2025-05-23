@enum ActivityState begin
    AnyState = 0
    ActiveState = 1
    DupState = 2
    MixedState = 3
end

@inline function Base.:|(a1::ActivityState, a2::ActivityState)
    ActivityState(Int(a1) | Int(a2))
end

struct Merger{seen,worldT,justActive,UnionSret,AbstractIsMixed}
    world::worldT
end

@inline element(::Val{T}) where {T} = T

@inline function (c::Merger{seen,worldT,justActive,UnionSret,AbstractIsMixed})(
    f::Int,
) where {seen,worldT,justActive,UnionSret,AbstractIsMixed}
    T = element(first(seen))

    reftype = ismutabletype(T) || (T isa UnionAll && !AbstractIsMixed)

    if justActive && reftype
        return Val(AnyState)
    end

    subT = typed_fieldtype(T, f)

    if justActive && ismutabletype(subT)
        return Val(AnyState)
    end

    sub = active_reg_inner(
        subT,
        seen,
        c.world,
        Val(justActive),
        Val(UnionSret),
        Val(AbstractIsMixed),
    )

    if sub == AnyState
        Val(AnyState)
    else
        if sub == DupState
            if justActive
                Val(AnyState)
            else
                Val(DupState)
            end
        else
            if reftype
                Val(DupState)
            else
                Val(sub)
            end
        end
    end
end

@inline forcefold(::Val{RT}) where {RT} = RT

@inline function forcefold(::Val{ty}, ::Val{sty}, C::Vararg{Any,N})::ActivityState where {ty,sty,N}
    if sty == AnyState || sty == ty
        return forcefold(Val(ty), C...)
    end
    if ty == AnyState
        return forcefold(Val(sty), C...)
    else
        return MixedState
    end
end

@inline numbereltype(::Type{<:EnzymeCore.RNumber{T}}) where {T} = T
@inline ptreltype(::Type{<:AbstractArray{T}}) where {T} = T
@inline ptreltype(::Type{Ptr{T}}) where {T} = T
@inline ptreltype(::Type{Core.LLVMPtr{T,N}}) where {T,N} = T
@inline ptreltype(::Type{Core.LLVMPtr{T} where N}) where {T} = T
@inline ptreltype(::Type{Base.RefValue{T}}) where {T} = T
@inline ptreltype(::Type{Array{T,N}}) where {T,N} = T
@inline ptreltype(::Type{Array{T,N} where N}) where {T} = T
@inline ptreltype(::Type{Complex{T}}) where {T} = T
@inline ptreltype(::Type{Tuple{Vararg{T}}}) where {T} = T
@inline ptreltype(::Type{IdDict{K,V}}) where {K,V} = V
@inline ptreltype(::Type{IdDict{K,V} where K}) where {V} = V
@inline ptreltype(::Type{SparseArrays.CHOLMOD.Dense{T}}) where T = T
@static if VERSION < v"1.11-"
else
@inline ptreltype(::Type{Memory{T}}) where T = T
end

@inline is_arrayorvararg_ty(::Type) = false
@inline is_arrayorvararg_ty(::Type{Array{T,N}}) where {T,N} = true
@inline is_arrayorvararg_ty(::Type{Array{T,N} where N}) where {T} = true
@inline is_arrayorvararg_ty(::Type{Tuple{Vararg{T2}}}) where {T2} = true
@inline is_arrayorvararg_ty(::Type{Ptr{T}}) where {T} = true
@inline is_arrayorvararg_ty(::Type{Core.LLVMPtr{T,N}}) where {T,N} = true
@inline is_arrayorvararg_ty(::Type{Core.LLVMPtr{T,N} where N}) where {T} = true
@inline is_arrayorvararg_ty(::Type{Base.RefValue{T}}) where {T} = true
@inline is_arrayorvararg_ty(::Type{IdDict{K,V}}) where {K,V} = true
@inline is_arrayorvararg_ty(::Type{IdDict{K,V} where K}) where {V} = true
@inline is_arrayorvararg_ty(::Type{SparseArrays.CHOLMOD.Dense{T}}) where T = true
@static if VERSION < v"1.11-"
else
@inline is_arrayorvararg_ty(::Type{Memory{T}}) where T = true
end

Base.@assume_effects :removable :foldable :nothrow @inline function staticInTup(::Val{T}, tup::NTuple{N,Val})::Bool where {T,N}
    any(ntuple(Val(N)) do i
        Base.@_inline_meta
        Val(T) == tup[i]
    end)
end

@inline function active_reg_recur(
    ::Type{ST},
    seen::Seen,
    world,
    ::Val{justActive},
    ::Val{UnionSret},
    ::Val{AbstractIsMixed},
)::ActivityState where {ST,Seen,justActive,UnionSret,AbstractIsMixed}
    if ST isa Union
        return forcefold(
            Val(
                active_reg_recur(
                    ST.a,
                    seen,
                    world,
                    Val(justActive),
                    Val(UnionSret),
                    Val(AbstractIsMixed),
                ),
            ),
            Val(
                active_reg_recur(
                    ST.b,
                    seen,
                    world,
                    Val(justActive),
                    Val(UnionSret),
                    Val(AbstractIsMixed),
                ),
            ),
        )
    end
    return active_reg_inner(
        ST,
        seen,
        world,
        Val(justActive),
        Val(UnionSret),
        Val(AbstractIsMixed),
    )
end

@inline is_vararg_tup(x) = false
@inline is_vararg_tup(::Type{Tuple{Vararg{T2}}}) where {T2} = true

@inline function active_reg_inner(
    ::Type{T},
    seen::ST,
    world::Union{Nothing,UInt},
    ::Val{justActive} = Val(false),
    ::Val{UnionSret} = Val(false),
    ::Val{AbstractIsMixed} = Val(false),
)::ActivityState where {ST,T,justActive,UnionSret,AbstractIsMixed}
    if T === Any
        if AbstractIsMixed
            return MixedState
        else
            return DupState
        end
    end

    if T === Union{}
        return AnyState
    end

    if T <: Complex && !(T isa UnionAll)
        return active_reg_inner(
            ptreltype(T),
            seen,
            world,
            Val(justActive),
            Val(UnionSret),
            Val(AbstractIsMixed),
        )
    end

    if T <: BigFloat
        return DupState
    end

    if T <: AbstractFloat
        return ActiveState
    end

    if T <: EnzymeCore.RNumber
        return active_reg_inner(
            numbereltype(T),
            seen,
            world,
            Val(justActive),
            Val(UnionSret),
            Val(AbstractIsMixed),
        )
    end

    if T <: Ptr ||
       T <: Core.LLVMPtr ||
       T <: Base.RefValue ||
       EnzymeCore.is_mutable_array(T) || is_arrayorvararg_ty(T)
        if justActive
            return AnyState
        end

        if (EnzymeCore.is_mutable_array(T) || is_arrayorvararg_ty(T)) &&
           active_reg_inner(
            ptreltype(T),
            seen,
            world,
            Val(justActive),
            Val(UnionSret),
            Val(AbstractIsMixed),
        ) == AnyState
            return AnyState
        else
            if AbstractIsMixed && is_vararg_tup(T)
                return MixedState
            else
                return DupState
            end
        end
    end

    if T <: Integer
        return AnyState
    end

    if isghostty(T) || Core.Compiler.isconstType(T) || T <: Type
        return AnyState
    end

    inactivety = if typeof(world) === Nothing
        EnzymeCore.EnzymeRules.inactive_type(T)
    else
        inmi = my_methodinstance(
            nothing,
            typeof(EnzymeCore.EnzymeRules.inactive_type),
            Tuple{Type{T}},
            world,
        )
        args = Any[EnzymeCore.EnzymeRules.inactive_type, T]
        GC.@preserve T begin
            ccall(
                :jl_invoke,
                Any,
                (Any, Ptr{Any}, Cuint, Any),
                EnzymeCore.EnzymeRules.inactive_type,
                args,
                length(args),
                inmi,
            )
        end
    end

    if inactivety
        return AnyState
    end

    # unknown number of fields
    if T isa UnionAll
        aT = Base.argument_datatype(T)
        if aT === nothing
            if AbstractIsMixed
                return MixedState
            else
                return DupState
            end
        end
        if Base.datatype_fieldcount(aT) === nothing
            if AbstractIsMixed
                return MixedState
            else
                return DupState
            end
        end
    end

    if T isa Union
        # if sret union, the data is stored in a stack memory location and is therefore
        # not unique'd preventing the boxing of the union in the default case
        if UnionSret && is_sret_union(T)
            return active_reg_recur(
                T,
                seen,
                world,
                Val(justActive),
                Val(UnionSret),
                Val(AbstractIsMixed),
            )
        else
            if justActive
                return AnyState
            end
            if active_reg_inner(T.a, seen, world, Val(justActive), Val(UnionSret)) !=
               AnyState
                if AbstractIsMixed
                    return MixedState
                else
                    return DupState
                end
            end
            if active_reg_inner(T.b, seen, world, Val(justActive), Val(UnionSret)) !=
               AnyState
                if AbstractIsMixed
                    return MixedState
                else
                    return DupState
                end
            end
        end
        return AnyState
    end

    # if abstract it must be by reference
    if Base.isabstracttype(T) || T == Tuple
        if AbstractIsMixed
            return MixedState
        else
            return DupState
        end
    end

    if ismutabletype(T)
        # if just looking for active of not
        # we know for a fact this isn't active
        if justActive
            return AnyState
        end
    end

    @assert !Base.isabstracttype(T)
    if !(Base.isconcretetype(T) || T <: Tuple || T isa UnionAll)
        throw(AssertionError("Type $T is not concrete type or concrete tuple"))
    end

    nT = if T <: Tuple && !(T isa UnionAll)
        Tuple{(
            ntuple(length(T.parameters)) do i
                Base.@_inline_meta
                sT = T.parameters[i]
                if sT isa TypeVar
                    Any
                elseif sT isa Core.TypeofVararg
                    Any
                else
                    sT
                end
            end
        )...}
    else
        T
    end

    if staticInTup(Val(nT), seen)
        return MixedState
    end

    seen2 = (Val(nT), seen...)

    fty = Merger{seen2,typeof(world),justActive,UnionSret,AbstractIsMixed}(world)

    ty = forcefold(Val(AnyState), ntuple(fty, Val(fieldcount(nT)))...)

    return ty
end

Base.@assume_effects :removable :foldable @inline @generated function active_reg_nothrow(::Type{T}, ::Val{world})::ActivityState where {T,world}
    return active_reg_inner(T, (), world)
end

Base.@assume_effects :removable :foldable @inline function active_reg(
    ::Type{T},
    world::Union{Nothing,UInt} = nothing,
)::Bool where {T}
    seen = ()

    # check if it could contain an active
    if active_reg_inner(T, seen, world, Val(true)) == ActiveState #=justActive=#
        state = active_reg_inner(T, seen, world, Val(false)) #=justActive=#
        if state == ActiveState
            return true
        end
        @assert state == MixedState
        throw(
            AssertionError(
                string(T) *
                " has mixed internal activity types. See https://enzyme.mit.edu/julia/stable/faq/#Mixed-activity for more information",
            ),
        )
    else
        return false
    end
end

Base.@assume_effects :removable :foldable :nothrow @inline function guaranteed_const(::Type{T})::Bool where {T}
    rt = active_reg_nothrow(T, Val(nothing))
    res = rt == AnyState
    return res
end

Base.@assume_effects :removable :foldable :nothrow @inline function guaranteed_const_nongen(::Type{T}, world)::Bool where {T}
    rt = active_reg_inner(T, (), world)
    res = rt == AnyState
    return res
end

# check if a value is guaranteed to be not contain active[register] data
# (aka not either mixed or active)
Base.@assume_effects :removable :foldable :nothrow @inline function guaranteed_nonactive(::Type{T})::Bool where {T}
    rt = Enzyme.Compiler.active_reg_nothrow(T, Val(nothing))
    return rt == Enzyme.Compiler.AnyState || rt == Enzyme.Compiler.DupState
end

"""
    Enzyme.guess_activity(::Type{T}, mode::Enzyme.Mode)

Try to guess the most appropriate [`Annotation`](@ref) for arguments of type `T` passed to [`autodiff`](@ref) with a given `mode`.
"""
Base.@assume_effects :removable :foldable :nothrow @inline Enzyme.guess_activity(::Type{T}, mode::Enzyme.Mode) where {T} =
    guess_activity(T, convert(API.CDerivativeMode, mode))

Base.@assume_effects :removable :foldable :nothrow @inline function Enzyme.guess_activity(::Type{T}, Mode::API.CDerivativeMode) where {T}
    ActReg = active_reg_nothrow(T, Val(nothing))
    if ActReg == AnyState
        return Const{T}
    end
    if Mode == API.DEM_ForwardMode
        return Duplicated{T}
    else
        if ActReg == ActiveState
            return Active{T}
        elseif ActReg == MixedState
            return MixedDuplicated{T}
        else
            return Duplicated{T}
        end
    end
end
