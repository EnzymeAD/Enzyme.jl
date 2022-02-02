using Enzyme
using LLVM
using Test

import Enzyme: typetree, TypeTree, API

const ctx = LLVM.Context()
const dl = string(LLVM.DataLayout(LLVM.JITTargetMachine()))

tt(T) = string(typetree(T, ctx, dl))

@testset "TypeTree" begin
    @test tt(Float16) == "{[0]:Float@half}" # TODO(VC) inconsistent
    @test tt(Float32) == "{[-1]:Float@float}"
    @test tt(Float64) == "{[-1]:Float@double}"
    @test tt(Symbol) == "{}"
    @test tt(String) ==  "{}"
    @test tt(AbstractChannel) == "{}"
end
