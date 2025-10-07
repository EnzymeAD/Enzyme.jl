@enum ActivityState begin
    AnyState = 0
    ActiveState = 1
    DupState = 2
    MixedState = 3
end

@inline function Base.:|(a1::ActivityState, a2::ActivityState)
    ActivityState(Int(a1) | Int(a2))
end

@inline element(::Val{T}) where {T} = T

@inline ptreltype(::Type{Ptr{T}}) where {T} = T
@inline ptreltype(::Type{Core.LLVMPtr{T,N}}) where {T,N} = T
@inline ptreltype(::Type{Core.LLVMPtr{T} where N}) where {T} = T
@inline ptreltype(::Type{Base.RefValue{T}}) where {T} = T
@inline ptreltype(::Type{Complex{T}}) where {T} = T
@inline ptreltype(::Type{Tuple{Vararg{T}}}) where {T} = T
@inline ptreltype(::Type{IdDict{K,V}}) where {K,V} = V
@inline ptreltype(::Type{IdDict{K,V} where K}) where {V} = V
@static if Base.USE_GPL_LIBS
    @inline ptreltype(::Type{SparseArrays.CHOLMOD.Dense{T}}) where T = T
end
@static if VERSION < v"1.11-"
else
@inline ptreltype(::Type{Memory{T}}) where T = T
end

@inline is_arrayorvararg_ty(::Type) = false
@inline is_arrayorvararg_ty(::Type{Tuple{Vararg{T2}}}) where {T2} = true
@inline is_arrayorvararg_ty(::Type{Ptr{T}}) where {T} = true
@inline is_arrayorvararg_ty(::Type{Core.LLVMPtr{T,N}}) where {T,N} = true
@inline is_arrayorvararg_ty(::Type{Core.LLVMPtr{T,N} where N}) where {T} = true
@inline is_arrayorvararg_ty(::Type{Base.RefValue{T}}) where {T} = true
@inline is_arrayorvararg_ty(::Type{IdDict{K,V}}) where {K,V} = true
@inline is_arrayorvararg_ty(::Type{IdDict{K,V} where K}) where {V} = true

@static if Base.USE_GPL_LIBS
    @inline is_arrayorvararg_ty(::Type{SparseArrays.CHOLMOD.Dense{T}}) where T = true
end
@static if VERSION < v"1.11-"
else
@inline is_arrayorvararg_ty(::Type{Memory{T}}) where T = true
end

Base.@nospecializeinfer function active_reg_recur(
    @nospecialize(ST::Type),
    seen::Base.IdSet{Type},
    world::UInt,
    justActive::Bool,
    UnionSret::Bool,
    AbstractIsMixed::Bool,
)::ActivityState
    if ST isa Union
        return (
            active_reg_recur(
                ST.a,
                seen,
                world,
                justActive,
                UnionSret,
                AbstractIsMixed,
            )
            |
            active_reg_recur(
                ST.b,
                seen,
                world,
                justActive,
                UnionSret,
                AbstractIsMixed,
            )
        )
    end
    return active_reg_inner(
        ST,
        seen,
        world,
        justActive,
        UnionSret,
        AbstractIsMixed,
    )
end

@inline is_vararg_tup(x) = false
@inline is_vararg_tup(::Type{Tuple{Vararg{T2}}}) where {T2} = true

Base.@nospecializeinfer @inline function is_mutable_array(@nospecialize(T::Type))
    if T <: Array
        return true
    end
    while T isa UnionAll
        T = T.body
    end
    if T isa DataType
        if hasproperty(T, :name) && hasproperty(T.name, :module)
            mod = T.name.module
            if string(mod) == "Reactant" && (T.name.name == :ConcretePJRTArray || T.name.name == :ConcreteIFRTArray || T.name.name == :TracedRArray)
                return true
            end
        end
    end
    return false
end

Base.@nospecializeinfer @inline function is_wrapped_number(@nospecialize(T::Type))
    if T isa UnionAll
        return is_wrapped_number(T.body)
    end
    while T isa UnionAll
        T = T.body
    end
    if T isa DataType && hasproperty(T, :name) && hasproperty(T.name, :module)
        mod = T.name.module
        if string(mod) == "Reactant" && (T.name.name == :ConcretePJRTNumber || T.name.name == :ConcreteIFRTNumber || T.name.name == :TracedRNumber)
            return true
        end
    end
    return false
end

Base.@nospecializeinfer @inline function unwrapped_number_type(@nospecialize(T::Type))
    while T isa UnionAll
        T = T.body
    end
    return T.parameters[1]
end

Base.@nospecializeinfer @inline function active_reg_inner(
    @nospecialize(T::Type),
    seen::Base.IdSet{Type},
    world::UInt,
    justActive::Bool,
    UnionSret::Bool,
    AbstractIsMixed::Bool,
)::ActivityState
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
            justActive,
            UnionSret,
            AbstractIsMixed,
        )
    end

    if T <: BigFloat
        return DupState
    end

    if T <: AbstractFloat
        return ActiveState
    end

    if is_wrapped_number(T)
        if justActive
            return AnyState
        end

        if active_reg_inner(
            unwrapped_number_type(T),
            seen,
            world,
            justActive,
            UnionSret,
            AbstractIsMixed,
        ) == AnyState
            return AnyState
        else
            if AbstractIsMixed
                return MixedState
            else
                return DupState
            end
        end
    end

    if is_mutable_array(T)
        if justActive
            return AnyState
        end

        if active_reg_inner(
            eltype(T),
            seen,
            world,
            justActive,
            UnionSret,
            AbstractIsMixed,
        ) == AnyState
            return AnyState
        else
            if AbstractIsMixed
                return MixedState
            else
                return DupState
            end
        end
    end

    if T <: Ptr ||
       T <: Core.LLVMPtr ||
       T <: Base.RefValue || is_arrayorvararg_ty(T)
        if justActive && !AbstractIsMixed
            return AnyState
        end

        if is_arrayorvararg_ty(T) &&
           active_reg_inner(
            ptreltype(T),
            seen,
            world,
            justActive,
            UnionSret,
            AbstractIsMixed,
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

    # Use _call_in_world_total to perform a concrete eval.
    inactivety = Core._call_in_world_total(world, EnzymeCore.EnzymeRules.inactive_type, T)

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
                justActive,
                UnionSret,
                AbstractIsMixed,
            )
        else
            if justActive
                return AnyState
            end
            if active_reg_inner(T.a, seen, world, justActive, UnionSret, false) !=
               AnyState
                if AbstractIsMixed
                    return MixedState
                else
                    return DupState
                end
            end
            if active_reg_inner(T.b, seen, world, justActive, UnionSret, false) !=
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

    if nT in seen
        return MixedState
    end

    reftype = ismutabletype(nT) || (nT isa UnionAll && !AbstractIsMixed)

    if justActive && reftype
        return AnyState
    end

    seen2 = copy(seen)
    push!(seen2, nT)

    ty = AnyState

    for f in 1:typed_fieldcount(nT)
        subT = typed_fieldtype(nT, f)

        if justActive && ismutabletype(subT)
            # AnyState
            continue
        end

        sub = active_reg_inner(
            subT,
            seen2,
            world,
            justActive,
            UnionSret,
            AbstractIsMixed,
        )

        if sub == AnyState
            continue
        end

        if sub == DupState && justActive
            continue
        end

        if reftype
            sub = DupState
        end

        ty |= sub
    end

    return ty
end

const ActivityCache = Dict{Tuple{Type, Bool, Bool, Bool}, ActivityState}()

const ActivityWorldCache = Ref(0)

const ActivityMethodCache = Core.MethodMatch[]
# given the current worldage of compilation, check if there are any methods 
# of inactive_type which may invalidate the cache, and if so clear it. 
function check_activity_cache_invalidations(world::UInt)
    # We've already guaranteed that this world doesn't have any stale caches
    if world <= ActivityWorldCache[]
        return
    end

    invalid = true

    tt = Tuple{typeof(EnzymeRules.inactive_type), Type}

    methods = Core.MethodMatch[]
    matches = Base._methods_by_ftype(tt, -1, world)
    if matches === nothing
        @assert ActivityCache.size() == 0
        return
    end

    methods = Core.MethodMatch[]
    for match in matches::Vector
        push!(methods, match::Core.MethodMatch)
    end

    if methods == ActivityMethodCache
        return
    end

    empty!(ActivityCache)
    empty!(ActivityMethodCache)
    for match in matches::Vector
        push!(ActivityMethodCache, match::Core.MethodMatch)
    end

    ActivityWorldCache[] = world

end

Base.@nospecializeinfer @inline function active_reg(@nospecialize(ST::Type), world::UInt; justActive=false, UnionSret = false, AbstractIsMixed = false)
    key = (ST, justActive, UnionSret, AbstractIsMixed)
    if haskey(ActivityCache, key)
        return ActivityCache[key]
    end
    set = Base.IdSet{Type}()
    result = active_reg_inner(ST, set, world, justActive, UnionSret, AbstractIsMixed)
    ActivityCache[key] = result
    return result
end

function active_reg_nothrow_generator(world::UInt, source::LineNumberNode, T, self, _)
    @nospecialize
    result = active_reg(T, world)

    # create an empty CodeInfo to return the result
    ci = ccall(:jl_new_code_info_uninit, Ref{Core.CodeInfo}, ())
    
    @static if isdefined(Core, :DebugInfo)
        ci.debuginfo = Core.DebugInfo(:none)
    else
        ci.codelocs = Int32[]
        ci.linetable = [
            Core.Compiler.LineInfoNode(@__MODULE__, :active_reg_nothrow, source.file, Int32(source.line), Int32(0))
        ]
    end
    check_activity_cache_invalidations(world)
    ci.min_world = world
    ci.max_world = typemax(UInt)

    edges = Any[]
    # Create the edge for the "query"
    # TODO: Check if we can use `Tuple{typeof(EnzymeRules.inactive_type), T}` directly
    inactive_type_sig = Tuple{typeof(EnzymeRules.inactive_type), Type}
    push!(edges, ccall(:jl_method_table_for, Any, (Any,), inactive_type_sig)::Core.MethodTable)
    push!(edges, inactive_type_sig)

    ci.edges = edges

    # prepare the slots
    ci.slotnames = Symbol[Symbol("#self#"), :t]
    ci.slotflags = UInt8[0x00 for i = 1:2]

    # return the result
    ci.code = Any[Core.Compiler.ReturnNode(result)]
    ci.ssaflags = UInt32[0x00]   # Julia's native compilation pipeline (and its verifier) expects `ssaflags` to be the same length as `code`
    @static if isdefined(Core, :DebugInfo)
    else
        push!(ci.codelocs, 1)
    end

    ci.ssavaluetypes = 1

    return ci
end

@eval Base.@assume_effects :removable :foldable :nothrow @inline function active_reg_nothrow(::Type{T})::ActivityState where {T}
    $(Expr(:meta, :generated_only))
    $(Expr(:meta, :generated, active_reg_nothrow_generator))
end

Base.@assume_effects :removable :foldable :nothrow @inline function guaranteed_const(::Type{T})::Bool where {T}
    rt = active_reg_nothrow(T)
    res = rt == AnyState
    return res
end

Base.@assume_effects :removable :foldable :nothrow @inline function guaranteed_const_nongen(::Type{T}, world::UInt)::Bool where {T}
    rt = active_reg(T, world)
    res = rt == AnyState
    return res
end

# check if a value is guaranteed to be not contain active[register] data
# (aka not either mixed or active)
Base.@assume_effects :removable :foldable :nothrow @inline function guaranteed_nonactive(::Type{T})::Bool where {T}
    rt = active_reg_nothrow(T)
    return rt == Enzyme.Compiler.AnyState || rt == Enzyme.Compiler.DupState
end

# check if a value is guaranteed to be not contain active[register] data
# (aka not either mixed or active)
@inline function guaranteed_nonactive(@nospecialize(T::Type), world::UInt; AbstractIsMixed=false)::Bool
    rt = active_reg(T, world; justActive=true, AbstractIsMixed)
    return rt == Enzyme.Compiler.AnyState || rt == Enzyme.Compiler.DupState
end

"""
    Enzyme.guess_activity(::Type{T}, mode::Enzyme.Mode)

Try to guess the most appropriate [`Annotation`](@ref) for arguments of type `T` passed to [`autodiff`](@ref) with a given `mode`.
"""
Base.@assume_effects :removable :foldable :nothrow @inline Enzyme.guess_activity(::Type{T}, mode::Enzyme.Mode) where {T} =
    guess_activity(T, convert(API.CDerivativeMode, mode))

Base.@assume_effects :removable :foldable :nothrow @inline function Enzyme.guess_activity(::Type{T}, Mode::API.CDerivativeMode) where {T}
    ActReg = active_reg_nothrow(T)
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
