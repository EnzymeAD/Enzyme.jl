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
@inline ptreltype(::Type{Core.SimpleVector}) = Any
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
@inline is_arrayorvararg_ty(::Type{Core.SimpleVector}) = true
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

@inline function record_dep!(deps::Vector{Any}, @nospecialize(sig::Type))
    sig in deps || push!(deps, sig)
    return nothing
end

"""
    @record_dep deps f(args...)

Call `f(args...)` and record the signature of the call in `deps`, so that code built from
the result gets an edge to the method the call dispatches to (`add_activity_edges!`). For
the calls to the hooks packages add methods to (`is_mutable_array`, `inactive_type`,
`eltype`), whose answer the result depends on.
"""
macro record_dep(deps, call)
    Meta.isexpr(call, :call) || error("@record_dep expects a call, got $call")
    f = gensym(:f)
    args = Symbol[]
    bindings = Expr[:($f = $(esc(call.args[1])))]
    sigparts = Any[:(Core.Typeof($f))]
    for arg in call.args[2:end]
        a = gensym(:arg)
        push!(args, a)
        push!(bindings, :($a = $(esc(arg))))
        push!(sigparts, :(Core.Typeof($a)))
    end
    return quote
        $(bindings...)
        record_dep!($(esc(deps)), Tuple{$(sigparts...)})
        $f($(args...))
    end
end

Base.@nospecializeinfer function active_reg_recur(
    @nospecialize(ST::Type),
    seen::Base.IdSet{Type},
        deps::Vector{Any},
    justActive::Bool,
    UnionSret::Bool,
    AbstractIsMixed::Bool,
)::ActivityState
    if ST isa Union
        return (
            active_reg_recur(
                ST.a,
                seen,
                deps,
                justActive,
                UnionSret,
                AbstractIsMixed,
            )
            |
            active_reg_recur(
                ST.b,
                seen,
                deps,
                justActive,
                UnionSret,
                AbstractIsMixed,
            )
        )
    end
    return active_reg_inner(
        ST,
        seen,
        deps,
        justActive,
        UnionSret,
        AbstractIsMixed,
    )
end

@inline is_vararg_tup(x) = false
@inline is_vararg_tup(::Type{Tuple{Vararg{T2}}}) where {T2} = true

"""
    is_mutable_array(T::Type)::Bool

Return `true` if `T` refers to mutable memory of elements of type `eltype(T)`, as an `Array`
does. The activity of `T` then comes from `eltype(T)`. A package can add methods for its
array or pointer types, for example in a package extension. Enzyme calls this function in
the world of the compilation.
"""
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
            if nameof(mod) === :Reactant && (T.name.name == :ConcretePJRTArray || T.name.name == :ConcreteIFRTArray || T.name.name == :TracedRArray)
                return true
            end
        end
    end
    return false
end

Base.@nospecializeinfer @inline function is_wrapped_number(@nospecialize(T::Type))
    while T isa UnionAll
        T = T.body
    end
    if T isa DataType && hasproperty(T, :name) && hasproperty(T.name, :module)
        mod = T.name.module
        if nameof(mod) === :Reactant && T.name.name in (
                :ConcretePJRTNumber, :ConcreteIFRTNumber, :TracedRNumber,
                :TracedRInteger, :TracedRFloat, :TracedRComplex,
                :ConcretePJRTInteger, :ConcretePJRTFloat, :ConcretePJRTComplex,
                :ConcreteIFRTInteger, :ConcreteIFRTFloat, :ConcreteIFRTComplex,
            )
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

# The activity of `T`.
#
# World age: this runs in the world of the compilation, which `active_reg` enters with
# `Core._call_in_world_total` before calling it. So plain dispatch here, including the calls
# to the hooks packages add methods to (`is_mutable_array`, `inactive_type`, `eltype`), sees
# the methods of that world, and not of the (older) world a generator runs in. Do not call
# this directly from a generator; call `active_reg`.
#
# `deps` collects the signatures of the dispatches the result depends on (the hooks, and
# `eltype` of the mutable array types), so that code that bakes the result in gets an edge
# to each of them (see `add_activity_edges!`).
Base.@nospecializeinfer @inline function active_reg_inner(
    @nospecialize(T::Type),
    seen::Base.IdSet{Type},
        deps::Vector{Any},
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
            deps,
            justActive,
            UnionSret,
            AbstractIsMixed,
        )
    end

    # Wrapped numbers are mutable references to their underlying value; they
    # may subtype AbstractFloat (or Integer) and must be classified before the
    # raw float/integer fast paths.
    if is_wrapped_number(T)
        if justActive
            return AnyState
        end

        if active_reg_inner(
            unwrapped_number_type(T),
            seen,
                deps,
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

    if T <: BigFloat
        return DupState
    end

    if T <: AbstractFloat
        return ActiveState
    end

    if @record_dep deps is_mutable_array(T)
        if justActive
            return AnyState
        end

        if active_reg_inner(
                (@record_dep deps eltype(T)),
            seen,
                deps,
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
                deps,
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

    if T <: Integer || T <: Enum
        return AnyState
    end

    if isghostty(T) || Core.Compiler.isconstType(T) || T <: Type
        return AnyState
    end

    if @record_dep deps EnzymeCore.EnzymeRules.inactive_type(T)
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
                deps,
                justActive,
                UnionSret,
                AbstractIsMixed,
            )
        else
            if justActive
                return AnyState
            end
            if active_reg_inner(T.a, seen, deps, justActive, UnionSret, false) !=
               AnyState
                if AbstractIsMixed
                    return MixedState
                else
                    return DupState
                end
            end
            if active_reg_inner(T.b, seen, deps, justActive, UnionSret, false) !=
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
            deps,
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

# The signatures of the hooks packages add methods to, and that change the result of
# `active_reg`. Code that cannot name the types its activity results were computed for
# (a compiled thunk) takes an edge on each of these; see `add_compilation_edges!`.
const ActivityHookSigs = (
    Tuple{typeof(EnzymeRules.inactive_type), Type},
    Tuple{typeof(is_mutable_array), Type},
)


"""
    active_reg_with_deps(T, world, justActive, UnionSret, AbstractIsMixed)

The `ActivityState` of `T` in `world`, with the signatures of the dispatches the
result depends on: the hooks packages add methods to, and `eltype` of the mutable array
types. This is the uncached analysis, for the generators; everything else goes through
[`active_reg`](@ref), which caches.

The analysis runs in `world` through `Core._call_in_world_total`, which also changes the
world inside a generator (where `Base.invoke_in_world` does not), so the methods packages
add to the hooks are seen whichever world the caller runs in.
"""
Base.@nospecializeinfer function active_reg_with_deps(@nospecialize(ST::Type), world::UInt, justActive::Bool, UnionSret::Bool, AbstractIsMixed::Bool)::Tuple{ActivityState, Vector{Any}}
    deps = Any[]
    state = Core._call_in_world_total(world, active_reg_inner, ST, Base.IdSet{Type}(), deps, justActive, UnionSret, AbstractIsMixed)::ActivityState
    return (state, deps)
end

"""
    add_activity_edges!(edges, deps, world)

Add the edges that keep code built from a result of [`active_reg_with_deps`](@ref)
current: one on each dispatch in `deps`, resolved in `world`. These are method instance
edges, which Julia closes when a method takes a dispatch over, and verifies against the
loaded methods when a package image loads (unlike edges on a bare signature, which it
only registers).
"""
function add_activity_edges!(edges::Vector{Any}, deps::Vector{Any}, world::UInt)
    for sig in deps
        add_dispatch_edge!(edges, sig::Type, world)
    end
    return edges
end

# The code of the generated functions below: the result as a constant, with the edges.
function activity_codeinfo(fn, source::Union{Method, LineNumberNode}, world::UInt, slotnames::Core.SimpleVector, @nospecialize(T::Type), justActive::Bool, UnionSret::Bool, AbstractIsMixed::Bool)
    state, deps = active_reg_with_deps(T, world, justActive, UnionSret, AbstractIsMixed)
    code = Any[Core.Compiler.ReturnNode(state)]
    ci = create_fresh_codeinfo(fn, source, world, slotnames, code)
    ci.edges = add_activity_edges!(Any[], deps, world)
    return ci
end

function active_reg_nothrow end

function active_reg_nothrow_generator(world::UInt, source::Union{Method, LineNumberNode}, T, self, _)
    @nospecialize
    return activity_codeinfo(active_reg_nothrow, source, world, Core.svec(Symbol("#self#"), :T), T, false, false, false)
end

# The activity of `T`, as the generated code of a `:foldable` function.
#
# This is also the cache of the analysis. Julia keeps the `CodeInstance` of each
# specialization with the world range it holds in: open when it is made, and closed by the
# invalidation of one of its edges when a method is added or deleted that changes a
# dispatch the result depends on. A call in a world the instance covers returns the
# constant; a call in a world it does not runs the generator again, in that world. The
# instances precompile, and are verified against the loaded methods when the image loads.
@eval Base.@assume_effects :removable :foldable :nothrow @inline function active_reg_nothrow(::Type{T})::ActivityState where {T}
    $(Expr(:meta, :generated_only))
    $(Expr(:meta, :generated, active_reg_nothrow_generator))
end

function active_reg_flags end

function active_reg_flags_generator(world::UInt, source::Union{Method, LineNumberNode}, T, justActive, UnionSret, AbstractIsMixed, self, _, _, _, _)
    @nospecialize
    slotnames = Core.svec(Symbol("#self#"), :T, :justActive, :UnionSret, :AbstractIsMixed)
    return activity_codeinfo(active_reg_flags, source, world, slotnames, T, justActive::Bool, UnionSret::Bool, AbstractIsMixed::Bool)
end

# `active_reg_nothrow` with the flags of `active_reg`; see there.
@eval Base.@assume_effects :removable :foldable :nothrow @inline function active_reg_flags(::Type{T}, ::Val{justActive}, ::Val{UnionSret}, ::Val{AbstractIsMixed})::ActivityState where {T, justActive, UnionSret, AbstractIsMixed}
    $(Expr(:meta, :generated_only))
    $(Expr(:meta, :generated, active_reg_flags_generator))
end

"""
    active_reg(T, world; justActive=false, UnionSret=false, AbstractIsMixed=false)

The `ActivityState` of `T` in `world`, usually the world of the compilation.

Calls `active_reg_flags` in `world`, so the answer comes from the `CodeInstance`
Julia keeps for `T` and the flags in that world, or from the analysis if there is none yet.
`Core._call_in_world_total` changes the world also inside a generator, where this is
called while a thunk compiles.
"""
Base.@nospecializeinfer @inline function active_reg(@nospecialize(ST::Type), world::UInt; justActive = false, UnionSret = false, AbstractIsMixed = false)::ActivityState
    return Core._call_in_world_total(world, active_reg_flags, ST, Val(justActive::Bool), Val(UnionSret::Bool), Val(AbstractIsMixed::Bool))::ActivityState
end

Base.@assume_effects :removable :foldable :nothrow @inline function guaranteed_const(@nospecialize(T::Type))::Bool
    rt = active_reg_nothrow(T)
    res = rt == AnyState
    return res
end

Base.@assume_effects :removable :foldable :nothrow @inline function guaranteed_const_nongen(@nospecialize(T::Type), world::UInt)::Bool
    rt = active_reg(T, world)
    res = rt == AnyState
    return res
end

# check if a value is guaranteed to be not contain active[register] data
# (aka not either mixed or active)
Base.@assume_effects :removable :foldable :nothrow @inline function guaranteed_nonactive(@nospecialize(T::Type))::Bool
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
