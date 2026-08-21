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

    @test Enzyme.Compiler.active_reg(Tuple{Float32,Float32,Int}, Base.get_world_counter()) == Enzyme.Compiler.ActiveState
    @test Enzyme.Compiler.active_reg(Tuple{NamedTuple{(), Tuple{}}, NamedTuple{(), Tuple{}}}, Base.get_world_counter()) == Enzyme.Compiler.AnyState
    @test Enzyme.Compiler.active_reg(Base.RefValue{Float32}, Base.get_world_counter()) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(Ptr, Base.get_world_counter()) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(Base.RefValue{Float32}, Base.get_world_counter()) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(Colon, Base.get_world_counter()) == Enzyme.Compiler.AnyState
    @test Enzyme.Compiler.active_reg(Symbol, Base.get_world_counter()) == Enzyme.Compiler.AnyState
    @test Enzyme.Compiler.active_reg(String, Base.get_world_counter()) == Enzyme.Compiler.AnyState
    @test Enzyme.Compiler.active_reg(Tuple{Any,Int64}, Base.get_world_counter()) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(Tuple{S,Int64} where S, Base.get_world_counter()) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(Union{Float64,Nothing}, Base.get_world_counter()) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(Union{Float64,Nothing}, Base.get_world_counter(), UnionSret=true) == Enzyme.Compiler.ActiveState
    @test Enzyme.Compiler.active_reg(Tuple, Base.get_world_counter()) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(Tuple, Base.get_world_counter(); AbstractIsMixed=true) == Enzyme.Compiler.MixedState
    @test Enzyme.Compiler.active_reg(Tuple{A,A} where A, Base.get_world_counter(), AbstractIsMixed=true) == Enzyme.Compiler.MixedState

    @test Enzyme.Compiler.active_reg(Tuple, Base.get_world_counter(), AbstractIsMixed=true, justActive=true) == Enzyme.Compiler.MixedState
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
