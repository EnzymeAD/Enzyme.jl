using Test
using Adapt
using EnzymeCore

# Changes the type of what it adapts, like moving an array to a device.
struct ToFloat32 end
Adapt.adapt_storage(::ToFloat32, x::Array{Float64}) = Float32.(x)

# Replaces a `Base.RefValue` with its own ref type, like CUDA's `KernelAdaptor`.
struct OtherRef{T} <: Ref{T}
    x::T
end
struct ToOtherRef end
Adapt.adapt_structure(to::ToOtherRef, r::Base.RefValue) = OtherRef(adapt(to, r[]))

@testset "MixedDuplicated" begin
    r = Ref(([1.0, 2.0], 0.5))
    x = MixedDuplicated(([3.0, 4.0], 2.0), r)

    y = adapt(ToFloat32(), x)
    @test y isa MixedDuplicated{Tuple{Vector{Float32}, Float64}}
    @test y.val == (Float32[3, 4], 2.0)
    @test y.dval isa Base.RefValue
    @test y.dval[] == (Float32[1, 2], 0.5)
    @test r[] == ([1.0, 2.0], 0.5)

    z = adapt(ToOtherRef(), x)
    @test z isa MixedDuplicated{Tuple{Vector{Float64}, Float64}}
    @test z.dval isa Base.RefValue
    @test z.dval[] == r[]
end

@testset "BatchMixedDuplicated" begin
    r1 = Ref(([1.0, 2.0], 0.5))
    r2 = Ref(([5.0, 6.0], 1.5))
    x = BatchMixedDuplicated(([3.0, 4.0], 2.0), (r1, r2))

    y = adapt(ToFloat32(), x)
    @test y isa BatchMixedDuplicated{Tuple{Vector{Float32}, Float64}, 2}
    @test y.val == (Float32[3, 4], 2.0)
    @test all(d -> d isa Base.RefValue, y.dval)
    @test y.dval[1][] == (Float32[1, 2], 0.5)
    @test y.dval[2][] == (Float32[5, 6], 1.5)

    z = adapt(ToOtherRef(), x)
    @test z isa BatchMixedDuplicated{Tuple{Vector{Float64}, Float64}, 2}
    @test all(d -> d isa Base.RefValue, z.dval)
    @test z.dval[1][] == r1[]
    @test z.dval[2][] == r2[]
end
