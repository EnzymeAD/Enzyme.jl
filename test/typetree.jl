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
    next::Union{Nothing,LList2{T}}
    v::T
end

struct Sibling{T}
    a::T
    b::T
end

struct Sibling2{T}
    a::T
    something::Bool
    b::T
end

struct UnionMember
    a::Float32
    b::Union{Function, Number}
    c::Bool
end

struct UnionStruct1{T}
    x::Union{T,Nothing}
    y::Any
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


    if Sys.WORD_SIZE == 64
        @test tt(UnionMember) == "{[0]:Float@float, [8]:Pointer, [16]:Integer}"
        @test tt(LList2{Float64}) == "{[0]:Pointer, [8]:Float@double}"
        @test tt(Sibling{LList2{Float64}}) == "{[-1]:Pointer, [-1,0]:Pointer, [-1,8]:Float@double}"
        @test tt(Sibling2{LList2{Float64}}) ==
              "{[0]:Pointer, [0,0]:Pointer, [0,8]:Float@double, [8]:Integer, [16]:Pointer, [16,0]:Pointer, [16,8]:Float@double}"
        @test tt(Sibling{Tuple{Int,Float64}}) ==
              "{[0]:Integer, [1]:Integer, [2]:Integer, [3]:Integer, [4]:Integer, [5]:Integer, [6]:Integer, [7]:Integer, [8]:Float@double, [16]:Integer, [17]:Integer, [18]:Integer, [19]:Integer, [20]:Integer, [21]:Integer, [22]:Integer, [23]:Integer, [24]:Float@double}"
        @test tt(Sibling{LList2{Tuple{Int,Float64}}}) ==
              "{[-1]:Pointer, [-1,0]:Pointer, [-1,8]:Integer, [-1,9]:Integer, [-1,10]:Integer, [-1,11]:Integer, [-1,12]:Integer, [-1,13]:Integer, [-1,14]:Integer, [-1,15]:Integer, [-1,16]:Float@double}"
        @test tt(Sibling2{Sibling2{LList2{Tuple{Float32,Float64}}}}) ==
              "{[0]:Pointer, [0,0]:Pointer, [0,8]:Float@float, [0,16]:Float@double, [8]:Integer, [16]:Pointer, [16,0]:Pointer, [16,8]:Float@float, [16,16]:Float@double, [24]:Integer, [32]:Pointer, [32,0]:Pointer, [32,8]:Float@float, [32,16]:Float@double, [40]:Integer, [48]:Pointer, [48,0]:Pointer, [48,8]:Float@float, [48,16]:Float@double}"

        @static if VERSION < v"1.11-"
            @test tt(Vector{Vector{Float32}}) == "{[0]:Pointer, [0,0]:Pointer, [0,0,0]:Pointer, [0,0,0,-1]:Float@float, [0,0,8]:Integer, [0,0,9]:Integer, [0,0,10]:Integer, [0,0,11]:Integer, [0,0,12]:Integer, [0,0,13]:Integer, [0,0,14]:Integer, [0,0,15]:Integer, [0,0,16]:Integer, [0,0,17]:Integer, [0,0,18]:Integer, [0,0,19]:Integer, [0,0,20]:Integer, [0,0,21]:Integer, [0,0,22]:Integer, [0,0,23]:Integer, [0,0,24]:Integer, [0,0,25]:Integer, [0,0,26]:Integer, [0,0,27]:Integer, [0,0,28]:Integer, [0,0,29]:Integer, [0,0,30]:Integer, [0,0,31]:Integer, [0,0,32]:Integer, [0,0,33]:Integer, [0,0,34]:Integer, [0,0,35]:Integer, [0,0,36]:Integer, [0,0,37]:Integer, [0,0,38]:Integer, [0,0,39]:Integer, [8]:Integer, [9]:Integer, [10]:Integer, [11]:Integer, [12]:Integer, [13]:Integer, [14]:Integer, [15]:Integer, [16]:Integer, [17]:Integer, [18]:Integer, [19]:Integer, [20]:Integer, [21]:Integer, [22]:Integer, [23]:Integer, [24]:Integer, [25]:Integer, [26]:Integer, [27]:Integer, [28]:Integer, [29]:Integer, [30]:Integer, [31]:Integer, [32]:Integer, [33]:Integer, [34]:Integer, [35]:Integer, [36]:Integer, [37]:Integer, [38]:Integer, [39]:Integer}"
        else
            @test tt(MemoryRef{Float32}) == "{[-1]:Pointer, [0,-1]:Float@float, [8,0]:Integer, [8,1]:Integer, [8,2]:Integer, [8,3]:Integer, [8,4]:Integer, [8,5]:Integer, [8,6]:Integer, [8,7]:Integer, [8,8]:Pointer, [8,8,-1]:Float@float}"
        end
    else
        @test tt(UnionMember) == "{[0]:Float@float, [4]:Pointer, [8]:Integer}"
        @test tt(LList2{Float64}) == "{[0]:Pointer, [4]:Float@double}"
        @test tt(Sibling{LList2{Float64}}) == "{[-1]:Pointer, [-1,0]:Pointer, [-1,4]:Float@double}"
        @test tt(Sibling2{LList2{Float64}}) ==
              "{[0]:Pointer, [0,0]:Pointer, [0,4]:Float@double, [4]:Integer, [8]:Pointer, [8,0]:Pointer, [8,4]:Float@double}"
        @test tt(Sibling{Tuple{Int,Float64}}) ==
              "{[0]:Integer, [1]:Integer, [2]:Integer, [3]:Integer, [4]:Float@double, [12]:Integer, [13]:Integer, [14]:Integer, [15]:Integer, [16]:Float@double}"
        @test tt(Sibling{LList2{Tuple{Int,Float64}}}) ==
              "{[-1]:Pointer, [-1,0]:Pointer, [-1,4]:Integer, [-1,5]:Integer, [-1,6]:Integer, [-1,7]:Integer, [-1,8]:Float@double}"
        @test tt(Sibling2{Sibling2{LList2{Tuple{Float32,Float64}}}}) ==
              "{[0]:Pointer, [0,0]:Pointer, [0,4]:Float@float, [0,8]:Float@double, [4]:Integer, [8]:Pointer, [8,0]:Pointer, [8,4]:Float@float, [8,8]:Float@double, [12]:Integer, [16]:Pointer, [16,0]:Pointer, [16,4]:Float@float, [16,8]:Float@double, [20]:Integer, [24]:Pointer, [24,0]:Pointer, [24,4]:Float@float, [24,8]:Float@double}"
    end
    @test tt(UnionStruct1{Float32}) == "{[0]:Float@float, [4]:Integer, [8]:Pointer}"
    @test tt(UnionStruct1{Float64}) == "{[0]:Float@double, [8]:Integer, [16]:Pointer}"
end

@testset "GetOffsets" begin
    @test Enzyme.get_offsets(Float16) == ((Enzyme.API.DT_Half,0),)
    @test Enzyme.get_offsets(Float32) == ((Enzyme.API.DT_Float,0),)
    @test Enzyme.get_offsets(Float64) == ((Enzyme.API.DT_Double,0),)
    @test Enzyme.get_offsets(Int) == ((Enzyme.API.DT_Integer,0),)
    @test Enzyme.get_offsets(Char) == ((Enzyme.API.DT_Integer,0),)
    @test Enzyme.get_offsets(Ptr) == ((Enzyme.API.DT_Pointer,0),)
    @test Enzyme.get_offsets(Ptr{Char}) == ((Enzyme.API.DT_Pointer,0),)
    @test Enzyme.get_offsets(Ptr{Float32}) == ((Enzyme.API.DT_Pointer,0),)
    @test Enzyme.get_offsets(Vector{Float32}) == ((Enzyme.API.DT_Pointer,0),)
    @test Enzyme.get_offsets(Tuple{Float64, Int}) == [(Enzyme.API.DT_Double,0),(Enzyme.API.DT_Integer, 8)]

    if Sys.WORD_SIZE == 64
        @test Enzyme.get_offsets(UnionMember) == [(Enzyme.API.DT_Float,0),(Enzyme.API.DT_Pointer, 8), (Enzyme.API.DT_Integer, 16)]
    else
        @test Enzyme.get_offsets(UnionMember) == [(Enzyme.API.DT_Float, 0), (Enzyme.API.DT_Pointer, 4), (Enzyme.API.DT_Integer, 8)]
    end
end
