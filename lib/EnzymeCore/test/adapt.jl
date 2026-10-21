using Test
using Adapt
using EnzymeCore

# Changes the type of what it adapts, like moving an array to a device.
struct ToFloat32 end
Adapt.adapt_storage(::ToFloat32, x::Array{Float64}) = Float32.(x)

const MixedAdaptError = Base.get_extension(EnzymeCore, :AdaptExt).MixedAdaptError

@testset "MixedDuplicated" begin
    x = MixedDuplicated(([3.0, 4.0], 2.0), Ref(([1.0, 2.0], 0.5)))
    @test_throws MixedAdaptError adapt(ToFloat32(), x)
    @test occursin("MixedDuplicated", sprint(showerror, MixedAdaptError(MixedDuplicated)))
end

@testset "BatchMixedDuplicated" begin
    r1 = Ref(([1.0, 2.0], 0.5))
    r2 = Ref(([5.0, 6.0], 1.5))
    x = BatchMixedDuplicated(([3.0, 4.0], 2.0), (r1, r2))
    @test_throws MixedAdaptError adapt(ToFloat32(), x)
end
