using Enzyme
using LLVM
using Test

import Enzyme: typetree, TypeTree, API, make_zero

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

struct LList2{T}
  next::Union{Nothing, LList2{T}}
  v::T
end

struct Sibling
  a::LList2{Float64}
  b::LList2{Float64}
end

@testset "TypeTree" begin
    @test tt(Float16) == "{[-1]:Float@half}"
    @test tt(Float32) == "{[-1]:Float@float}"
    @test tt(Float64) == "{[-1]:Float@double}"
    @test tt(Symbol) == "{}"
    @test tt(String) ==  "{}"
    @test tt(AbstractChannel) == "{}"
    @test tt(Base.ImmutableDict{Symbol, Any}) == "{[-1]:Pointer}"
    @test tt(Atom) == "{[0]:Float@float, [4]:Float@float, [8]:Float@float, [12]:Integer, [13]:Integer, [14]:Integer, [15]:Integer}"
    @test tt(Composite) == "{[0]:Float@float, [4]:Float@float, [8]:Float@float, [12]:Integer, [13]:Integer, [14]:Integer, [15]:Integer, [16]:Float@float, [20]:Float@float, [24]:Float@float, [28]:Integer, [29]:Integer, [30]:Integer, [31]:Integer}"
    @test tt(Tuple{Any,Any}) ==  "{[-1]:Pointer}"
    at = Atom(1.0, 2.0, 3.0, 4)
    at2 = make_zero(at)
    @test at2.x == 0.0
    @test at2.y == 0.0
    @test at2.z == 0.0
    @test at2.type == 4

    @test tt(LList2{Float64}) == "{[8]:Float@double}"
    @test tt(Sibling) == "{[-1]:Pointer, [-1,8]:Float@double}"
end
