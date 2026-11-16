using Enzyme, Test

struct Ints{A, B}
    v::B
    q::Int
end

mutable struct MInts{A, B}
    v::B
    q::Int
end

@testset "Activity Tests" begin
    @static if VERSION < v"1.11-"
    else
        @test Enzyme.Compiler.active_reg(Memory{Float64}, Base.get_world_counter()) == Enzyme.Compiler.DupState
    end
    @test Enzyme.Compiler.active_reg(Type{Array}, Base.get_world_counter()) == Enzyme.Compiler.AnyState
    @test Enzyme.Compiler.active_reg(Core.SimpleVector, Base.get_world_counter()) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(Ints{<:Any, Integer}, Base.get_world_counter()) == Enzyme.Compiler.AnyState
    @test Enzyme.Compiler.active_reg(Ints{<:Any, Float64}, Base.get_world_counter()) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(Ints{Integer, <:Any}, Base.get_world_counter()) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(Ints{Integer, <:Integer}, Base.get_world_counter()) == Enzyme.Compiler.AnyState
    @test Enzyme.Compiler.active_reg(Ints{Integer, <:AbstractFloat}, Base.get_world_counter()) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(Ints{Integer, Float64}, Base.get_world_counter()) == Enzyme.Compiler.ActiveState
    @test Enzyme.Compiler.active_reg(MInts{Integer, Float64}, Base.get_world_counter()) == Enzyme.Compiler.DupState

    @test Enzyme.Compiler.active_reg(Tuple{Float32, Float32, Int}, Base.get_world_counter()) == Enzyme.Compiler.ActiveState
    @test Enzyme.Compiler.active_reg(Tuple{NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}}, Base.get_world_counter()) == Enzyme.Compiler.AnyState
    @test Enzyme.Compiler.active_reg(Base.RefValue{Float32}, Base.get_world_counter()) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(Ptr, Base.get_world_counter()) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(Base.RefValue{Float32}, Base.get_world_counter()) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(Colon, Base.get_world_counter()) == Enzyme.Compiler.AnyState
    @test Enzyme.Compiler.active_reg(Symbol, Base.get_world_counter()) == Enzyme.Compiler.AnyState
    @test Enzyme.Compiler.active_reg(String, Base.get_world_counter()) == Enzyme.Compiler.AnyState
    @test Enzyme.Compiler.active_reg(Tuple{Any, Int64}, Base.get_world_counter()) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(Tuple{S, Int64} where {S}, Base.get_world_counter()) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(Union{Float64, Nothing}, Base.get_world_counter()) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(Union{Float64, Nothing}, Base.get_world_counter(), UnionSret = true) == Enzyme.Compiler.ActiveState
    @test Enzyme.Compiler.active_reg(Tuple, Base.get_world_counter()) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(Tuple, Base.get_world_counter(); AbstractIsMixed = true) == Enzyme.Compiler.MixedState
    @test Enzyme.Compiler.active_reg(Tuple{A, A} where {A}, Base.get_world_counter(), AbstractIsMixed = true) == Enzyme.Compiler.MixedState

    @test Enzyme.Compiler.active_reg(Tuple, Base.get_world_counter(), AbstractIsMixed = true, justActive = true) == Enzyme.Compiler.MixedState
end

# Reactant number and array wrappers are recognized by module/type name in
# `is_wrapped_number`/`is_mutable_array`; mock the names so classification is
# testable without a Reactant dependency.
module Reactant
    mutable struct TracedRNumber{T} <: Number
        v::T
    end
    mutable struct TracedRInteger{T} <: Integer
        v::T
    end
    mutable struct TracedRFloat{T} <: AbstractFloat
        v::T
    end
    mutable struct TracedRComplex{T} <: Number
        v::T
    end
    mutable struct ConcretePJRTFloat{T, D} <: AbstractFloat
        v::T
    end
    mutable struct ConcreteIFRTInteger{T} <: Integer
        v::T
    end
    mutable struct TracedRArray{T, N} <: AbstractArray{T, N} end
end

module NotReactant
    mutable struct TracedRFloat{T} <: AbstractFloat
        v::T
    end
end

@testset "Wrapped number activity" begin
    @test Enzyme.Compiler.is_wrapped_number(Reactant.TracedRFloat{Float64})
    @test Enzyme.Compiler.is_wrapped_number(Reactant.ConcretePJRTFloat{Float64, 1})
    @test !Enzyme.Compiler.is_wrapped_number(NotReactant.TracedRFloat{Float64})
    @test !Enzyme.Compiler.is_wrapped_number(Float64)

    # wrapped numbers subtyping AbstractFloat/Integer must classify as mutable
    # wrappers of their unwrapped type, not hit the raw float/integer fast paths
    @test Enzyme.Compiler.active_reg(Reactant.TracedRFloat{Float64}, Base.get_world_counter()) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(Reactant.TracedRFloat{Float64}, Base.get_world_counter(), justActive = true) == Enzyme.Compiler.AnyState
    @test Enzyme.Compiler.active_reg(Reactant.TracedRInteger{Int}, Base.get_world_counter()) == Enzyme.Compiler.AnyState
    @test Enzyme.Compiler.active_reg(Reactant.TracedRComplex{ComplexF64}, Base.get_world_counter()) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(Reactant.TracedRNumber{Float64}, Base.get_world_counter()) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(Reactant.ConcretePJRTFloat{Float64, 1}, Base.get_world_counter()) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(Reactant.ConcreteIFRTInteger{Int}, Base.get_world_counter()) == Enzyme.Compiler.AnyState
    @test Enzyme.Compiler.active_reg(Tuple{Reactant.TracedRFloat{Float64}, Float64}, Base.get_world_counter()) == Enzyme.Compiler.MixedState

    # same-named types outside the Reactant module use the plain rules
    @test Enzyme.Compiler.active_reg(NotReactant.TracedRFloat{Float64}, Base.get_world_counter()) == Enzyme.Compiler.ActiveState

    @test Enzyme.Compiler.is_mutable_array(Reactant.TracedRArray{Float64, 1})
    @test Enzyme.Compiler.active_reg(Reactant.TracedRArray{Float64, 1}, Base.get_world_counter()) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(Reactant.TracedRArray{Int, 1}, Base.get_world_counter()) == Enzyme.Compiler.AnyState
end

# A pointer type from a package, with a method for `is_mutable_array` (for example, in a
# package extension), as CUDA's `CuPtr` when Enzyme is loaded before CUDA. These methods,
# and the package's `eltype`, are newer than the generator of `active_reg_nothrow`
# (EnzymeAD/Enzyme.jl#3806).
primitive type ExtDevPtr{T} 64 end
Base.eltype(::Type{<:ExtDevPtr{T}}) where {T} = T
Enzyme.Compiler.is_mutable_array(::Type{<:ExtDevPtr}) = true

# The device memory of an array: an immutable holder of a pointer to nothing.
struct ExtDevMem
    ptr::ExtDevPtr{Nothing}
    len::Int
end
mutable struct ExtDevArr
    mem::ExtDevMem
end

@testset "is_mutable_array methods from a newer world" begin
    # The generated functions see the package's `eltype`, not the `Any` fallback.
    @test Enzyme.Compiler.guaranteed_const(ExtDevMem)
    @test Enzyme.Compiler.active_reg_nothrow(ExtDevPtr{Nothing}) == Enzyme.Compiler.AnyState
    @test Enzyme.Compiler.active_reg_nothrow(ExtDevPtr{Float64}) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(ExtDevPtr{Nothing}, Base.get_world_counter()) == Enzyme.Compiler.AnyState
    @test Enzyme.Compiler.active_reg(ExtDevMem, Base.get_world_counter()) == Enzyme.Compiler.AnyState

    # So the pointer is left alone when the holder is zeroed in place.
    ptr = reinterpret(ExtDevPtr{Nothing}, UInt64(0x1234))
    arr = ExtDevArr(ExtDevMem(ptr, 4))
    Enzyme.make_zero!(arr)
    @test arr.mem.ptr === ptr
    Enzyme.remake_zero!(arr)
    @test arr.mem.ptr === ptr

    # And `gradient` takes the holder as constant, in the world it is compiled in.
    @test Enzyme.gradient(Reverse, (x, m) -> x * m.len, 2.0, ExtDevMem(ptr, 3)) == (3.0, nothing)
end

# A method for `eltype` defined after the activity of the type was compiled in. The
# generated code has an edge to the `eltype` dispatch it was built from.
primitive type LateEltypePtr 64 end
Enzyme.Compiler.is_mutable_array(::Type{LateEltypePtr}) = true

late_eltype_activity() = Enzyme.Compiler.active_reg_nothrow(LateEltypePtr)
late_eltype_const() = Enzyme.Compiler.guaranteed_const(LateEltypePtr)

@testset "eltype methods invalidate the compiled activity" begin
    # Without an `eltype` method, the elements are `Any`.
    @test late_eltype_activity() == Enzyme.Compiler.DupState
    @test !late_eltype_const()
    @test Enzyme.Compiler.active_reg(LateEltypePtr, Base.get_world_counter()) == Enzyme.Compiler.DupState

    @eval Base.eltype(::Type{LateEltypePtr}) = Nothing
    @test late_eltype_activity() == Enzyme.Compiler.AnyState
    @test late_eltype_const()
    # The cached entry's `eltype` dispatch no longer resolves as it did, so it is recomputed.
    @test Enzyme.Compiler.active_reg(LateEltypePtr, Base.get_world_counter()) == Enzyme.Compiler.AnyState
end

# The results are cached as the `CodeInstance`s of `active_reg_flags`, each with the world
# range it holds in: open when made, and closed when a method is added that changes a
# dispatch the result depends on. Worlds in the range are answered from the instance.
struct CachedAcrossWorlds
    x::Float64
end
cache_worlds_filler_1() = nothing

function code_instances(mi::Core.MethodInstance)
    cis = Core.CodeInstance[]
    isdefined(mi, :cache) || return cis
    ci = mi.cache
    while ci !== nothing
        # `owner` exists since Julia 1.11; before, every instance is the native one.
        (!hasfield(Core.CodeInstance, :owner) || ci.owner === nothing) && push!(cis, ci)
        ci = isdefined(ci, :next) ? ci.next : nothing
    end
    return cis
end

function activity_instance(T::Type)
    C = Enzyme.Compiler
    sig = Tuple{typeof(C.active_reg_flags), Type{T}, Val{false}, Val{false}, Val{false}}
    for mi in Base.specializations(only(methods(C.active_reg_flags)))
        mi.specTypes == sig && return mi
    end
    return nothing
end

@testset "activity results are cached by world range" begin
    C = Enzyme.Compiler
    w1 = Base.get_world_counter()
    @test C.active_reg(CachedAcrossWorlds, w1) == C.ActiveState
    mi = activity_instance(CachedAcrossWorlds)
    @test mi !== nothing
    ci1 = only(code_instances(mi))
    @test ci1.min_world <= w1
    @test ci1.max_world == typemax(UInt)

    # A method that is not a hook moves the world but not the dispatches the result depends
    # on: the instance stays open and answers the newer world.
    @eval cache_worlds_filler_2() = nothing
    w2 = Base.get_world_counter()
    @test w2 > w1
    @test C.active_reg(CachedAcrossWorlds, w2) == C.ActiveState
    @test only(code_instances(mi)) === ci1
    @test ci1.max_world == typemax(UInt)
    @test C.active_reg(CachedAcrossWorlds, w1) == C.ActiveState

    # A hook method for the type closes the instance's range, and the newer world gets a
    # new instance with the new result; the older world keeps its answer.
    @eval Enzyme.EnzymeRules.inactive_type(::Type{CachedAcrossWorlds}) = true
    w3 = Base.get_world_counter()
    @test C.active_reg(CachedAcrossWorlds, w3) == C.AnyState
    @test ci1.max_world < w3
    @test length(code_instances(mi)) == 2
    @test C.active_reg(CachedAcrossWorlds, w2) == C.ActiveState
    @test ci1.max_world == w2
end

# A hook method for an unrelated type does not touch the instance: the edges are to the
# dispatches for the types the analysis visited, not to the hooks as a whole.
struct UnrelatedToTheCache
    x::Float64
end
struct NotInAnyActivityResult end

@testset "activity instances have precise edges" begin
    C = Enzyme.Compiler
    @test C.active_reg(UnrelatedToTheCache, Base.get_world_counter()) == C.ActiveState
    ci = only(code_instances(activity_instance(UnrelatedToTheCache)))
    @eval Enzyme.EnzymeRules.inactive_type(::Type{NotInAnyActivityResult}) = true
    @test ci.max_world == typemax(UInt)
    # `is_mutable_array` does not specialize on its argument; the edge is still to the
    # dispatch for this type, and not to the method as a whole.
    @eval Enzyme.Compiler.is_mutable_array(::Type{NotInAnyActivityResult}) = true
    @test ci.max_world == typemax(UInt)
    @test C.active_reg(UnrelatedToTheCache, Base.get_world_counter()) == C.ActiveState
    @test only(code_instances(activity_instance(UnrelatedToTheCache))) === ci
end

# An edge only catches the methods defined after the result is cached. A result first asked
# for in an older world, when a hook method for the type already exists in a newer one, is
# valid up to that method only.
struct OldWorldFirst
    x::Float64
end
struct OldWorldFirstGradient
    x::Float64
end
old_world_first_nothrow() = Enzyme.Compiler.active_reg_nothrow(OldWorldFirst)
old_world_first_const() = Enzyme.Compiler.guaranteed_const(OldWorldFirst)
old_world_first_f(x, s) = x * s.x
old_world_first_gradient() = Enzyme.gradient(Reverse, old_world_first_f, 2.0, OldWorldFirstGradient(3.0))

@testset "activity first asked for in an older world" begin
    C = Enzyme.Compiler
    w1 = Base.get_world_counter()
    @eval Enzyme.EnzymeRules.inactive_type(::Type{OldWorldFirst}) = true
    @eval Enzyme.EnzymeRules.inactive_type(::Type{OldWorldFirstGradient}) = true
    w2 = Base.get_world_counter()

    @test C.active_reg(OldWorldFirst, w1) == C.ActiveState
    ci = only(code_instances(activity_instance(OldWorldFirst)))
    @test ci.max_world < w2
    @test C.active_reg(OldWorldFirst, w2) == C.AnyState
    @test C.active_reg(OldWorldFirst, w1) == C.ActiveState

    @test Base.invoke_in_world(w1, old_world_first_nothrow) == C.ActiveState
    @test Base.invoke_in_world(w1, old_world_first_const) == false
    @test old_world_first_nothrow() == C.AnyState
    @test old_world_first_const()

    @test Base.invoke_in_world(w1, old_world_first_gradient) == (3.0, OldWorldFirstGradient(2.0))
    @test old_world_first_gradient() == (3.0, nothing)
end

# A compiled thunk is only as current as the activity results its compilation used,
# including those that depend on `eltype` of a mutable array type.
mutable struct ThunkEltypeBox
    x::Float64
end
Enzyme.Compiler.is_mutable_array(::Type{ThunkEltypeBox}) = true
Base.eltype(::Type{ThunkEltypeBox}) = Float64
thunk_eltype_f(b::ThunkEltypeBox) = b.x * b.x
function thunk_eltype_grad()
    b = ThunkEltypeBox(3.0)
    db = ThunkEltypeBox(0.0)
    Enzyme.autodiff(Reverse, thunk_eltype_f, Active, Duplicated(b, db))
    return db.x
end

function thunk_instances(f)
    cis = Core.CodeInstance[]
    for m in methods(Enzyme.Compiler.thunk), mi in Base.specializations(m)
        FA = Base.unwrap_unionall(mi.specTypes).parameters[3]
        FA == Type{Const{typeof(f)}} || continue
        append!(cis, code_instances(mi))
    end
    return cis
end

@testset "thunks depend on the activity results of their compilation" begin
    C = Enzyme.Compiler
    @test thunk_eltype_grad() == 6.0
    cis = thunk_instances(thunk_eltype_f)
    @test !isempty(cis)
    @test all(ci -> ci.max_world == typemax(UInt), cis)
    # The thunks built from the result are invalidated along with it.
    @eval Base.eltype(::Type{ThunkEltypeBox}) = Nothing
    @test all(ci -> ci.max_world < Base.get_world_counter(), cis)
    @test C.active_reg(ThunkEltypeBox, Base.get_world_counter()) == C.AnyState
end
