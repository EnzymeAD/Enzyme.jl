using Enzyme
using LLVM
using Test

import Enzyme: typetree, TypeTree, API

const ctx = LLVM.Context()
const dl = string(LLVM.DataLayout(LLVM.JITTargetMachine()))

tt(T) = string(typetree(T, ctx, dl))

struct Atom
  x::Float32
  y::Float32
  z::Float32
  type::Int32
end

struct Composite
    k::Atom
    y::Atom
end

@testset "TypeTree" begin
    @test tt(Float16) == "{[-1]:Float@half}"
    @test tt(Float32) == "{[-1]:Float@float}"
    @test tt(Float64) == "{[-1]:Float@double}"
    @test tt(Symbol) == "{}"
    @test tt(String) ==  "{}"
    @test tt(AbstractChannel) == "{}"
    if sizeof(Int) == sizeof(Int64)
        @test tt(Base.ImmutableDict{Symbol, Any}) == "{[0]:Pointer, [8]:Pointer, [16]:Pointer}"
    else
        @test tt(Base.ImmutableDict{Symbol, Any}) == "{[0]:Pointer, [4]:Pointer, [8]:Pointer}" 
    end
    @test tt(Atom) == "{[0]:Float@float, [4]:Float@float, [8]:Float@float, [12]:Integer, [13]:Integer, [14]:Integer, [15]:Integer}"
    @test tt(Composite) == "{[0]:Float@float, [4]:Float@float, [8]:Float@float, [12]:Integer, [13]:Integer, [14]:Integer, [15]:Integer, [16]:Float@float, [20]:Float@float, [24]:Float@float, [28]:Integer, [29]:Integer, [30]:Integer, [31]:Integer}"
    @test tt(Tuple{Any,Any}) ==  "{[-1]:Pointer}"
end
