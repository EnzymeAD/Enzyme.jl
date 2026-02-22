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
