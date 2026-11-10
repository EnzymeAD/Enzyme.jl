using Enzyme, Test
using Enzyme: EnzymeRules

# An immutable struct with GC pointers whose layout needs 32-bit field descriptors
# (here: a field of more than 32 KiB) is not `allocatedinline`, and Julia's codegen
# passes and returns it boxed. `GPUCompiler.deserves_argbox` misses this case, which
# shifted the derived calling convention by a roots parameter.
const BigPad = NTuple{4100, Float64}

struct BigStruct
    pad::BigPad
    x::Vector{Float64}
end

function make_pad(v)
    buf = fill(v, 4100)
    return GC.@preserve buf unsafe_load(Ptr{BigPad}(pointer(buf)))
end

@testset "Layout of a struct that Julia does not store inline" begin
    @test !Base.allocatedinline(BigStruct)
    @test Enzyme.Compiler.deserves_argbox(BigStruct)
end

big_const_arg(b::BigStruct, y) = b.pad[3] * y[1]^2

@testset "Boxed struct as a Const argument" begin
    y = [1.5]
    dy = zero(y)
    autodiff(Reverse, big_const_arg, Active, Const(BigStruct(make_pad(2.0), [0.0])), Duplicated(y, dy))
    @test dy[1] ≈ 2 * 2.0 * 1.5
end

big_dup_arg(b::BigStruct) = b.pad[3] * b.x[1]^2

@testset "Boxed struct as a Duplicated argument" begin
    b = BigStruct(make_pad(2.0), [1.5])
    db = BigStruct(make_pad(0.0), [0.0])
    autodiff(Reverse, big_dup_arg, Active, Duplicated(b, db))
    @test db.x[1] ≈ 2 * 2.0 * 1.5
end

@noinline big_return(y) = BigStruct(ntuple(_ -> 0.0, Val(4100)), [y[1]^2])
# the inference barrier keeps `big_return` a real call returning the boxed struct
big_return_caller(y) = Base.inferencebarrier(nothing) === nothing ? 2 * big_return(y).x[1] : 0.0

@testset "Boxed struct returned from a call" begin
    y = [1.5]
    dy = zero(y)
    autodiff(Reverse, big_return_caller, Active, Duplicated(y, dy))
    @test dy[1] ≈ 4 * 1.5
end
