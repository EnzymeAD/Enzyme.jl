using Enzyme
using Test

mutable struct SimpleMut
    a::Float64
    b::Int
end

mutable struct NestedMut
    a::Float64
    inner::SimpleMut
end

mutable struct ArrayMut
    a::Float64
    arr::Vector{Float64}
end

@testset "recursive_accumulate" begin
    @testset "atomic=false" begin
        # Test SimpleMut
        x = SimpleMut(1.0, 2)
        y = SimpleMut(3.0, 4)
        Enzyme.Compiler.recursive_accumulate(x, y)
        @test x.a ≈ 4.0
        @test x.b == 2
        
        # Test NestedMut
        x = NestedMut(1.0, SimpleMut(2.0, 3))
        y = NestedMut(4.0, SimpleMut(5.0, 6))
        Enzyme.Compiler.recursive_accumulate(x, y)
        @test x.a ≈ 5.0
        @test x.inner.a ≈ 7.0
        @test x.inner.b == 3
        
        # Test ArrayMut
        x = ArrayMut(1.0, [2.0, 3.0])
        y = ArrayMut(4.0, [5.0, 6.0])
        Enzyme.Compiler.recursive_accumulate(x, y)
        @test x.a ≈ 5.0
        @test x.arr ≈ [7.0, 9.0]
        
        # Test Core.Box
        b1 = Core.Box(SimpleMut(1.0, 2))
        b2 = Core.Box(SimpleMut(3.0, 4))
        Enzyme.Compiler.recursive_accumulate(b1, b2)
        @test b1.contents.a ≈ 4.0
        @test b1.contents.b == 2
        
        # Test RefValue
        r1 = Ref(1.0)
        r2 = Ref(2.0)
        Enzyme.Compiler.recursive_accumulate(r1, r2)
        @test r1[] ≈ 3.0
        
        # Test Ref{Tuple{Tuple{Float64, Float32}}}
        r1 = Ref(((1.0, 2.0f0),))
        r2 = Ref(((3.0, 4.0f0),))
        Enzyme.Compiler.recursive_accumulate(r1, r2)
        @test r1[][1][1] ≈ 4.0
        @test r1[][1][2] ≈ 6.0f0
        
        # Test with custom function f
        x = ArrayMut(1.0, [2.0, 3.0])
        y = ArrayMut(4.0, [5.0, 6.0])
        Enzyme.Compiler.recursive_accumulate(x, y, Val(false), x -> x * 2)
        @test x.a ≈ 9.0
        @test x.arr ≈ [12.0, 15.0]
    end

    @testset "atomic=true" begin
        # Test SimpleMut
        x = SimpleMut(1.0, 2)
        y = SimpleMut(3.0, 4)
        Enzyme.Compiler.recursive_accumulate(x, y, Val(true))
        @test x.a ≈ 4.0
        @test x.b == 2
        
        # Test Array directly
        arr1 = [2.0, 3.0]
        arr2 = [5.0, 6.0]
        Enzyme.Compiler.recursive_accumulate(arr1, arr2, Val(true))
        @test arr1 ≈ [7.0, 9.0]
        
        # Test RefValue
        r1 = Ref(1.0)
        r2 = Ref(2.0)
        Enzyme.Compiler.recursive_accumulate(r1, r2, Val(true))
        @test r1[] ≈ 3.0
        
        # Test Ref{Tuple{Tuple{Float64, Float32}}}
        r1 = Ref(((1.0, 2.0f0),))
        r2 = Ref(((3.0, 4.0f0),))
        Enzyme.Compiler.recursive_accumulate(r1, r2, Val(true))
        @test r1[][1][1] ≈ 4.0
        @test r1[][1][2] ≈ 6.0f0
        
        # Test with custom function f on Array
        arr1 = [2.0, 3.0]
        arr2 = [5.0, 6.0]
        Enzyme.Compiler.recursive_accumulate(arr1, arr2, Val(true), x -> x * 2)
        @test arr1 ≈ [12.0, 15.0]
    end
end
