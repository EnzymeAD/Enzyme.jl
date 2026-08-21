using Test
using EnzymeCore

@testset "Const" begin
    c = Const(1.0)
    @test c isa Const{Float64}
    @test c isa EnzymeCore.Annotation{Float64}
    @test c.val === 1.0
    @test eltype(typeof(c)) === Float64

    # `Const` of a type should wrap the type, not collapse to `Const{DataType}`
    ct = Const(Float64)
    @test ct isa Const{Type{Float64}}
    @test ct.val === Float64
end

@testset "Active" begin
    a = Active(1.0)
    @test a isa Active{Float64}
    @test a isa EnzymeCore.Annotation{Float64}
    @test a.val === 1.0

    # plain integers are promoted to floating point
    ai = Active(1)
    @test ai isa Active{Float64}
    @test ai.val === 1.0

    # complex integers are promoted as well
    ac = Active(1 + 2im)
    @test ac isa Active{ComplexF64}
    @test ac.val === ComplexF64(1, 2)

    # arrays are not supported as Active
    @test_throws ErrorException Active([1.0])
end

@testset "Duplicated" begin
    x = [1.0]
    dx = [0.0]
    d = Duplicated(x, dx)
    @test d isa Duplicated{Vector{Float64}}
    @test d isa EnzymeCore.Annotation{Vector{Float64}}
    @test d.val === x
    @test d.dval === dx
    @test eltype(typeof(d)) === Vector{Float64}

    # explicit parametric constructor agrees with the outer constructor
    @test Duplicated{Vector{Float64}}(x, dx) isa Duplicated{Vector{Float64}}

    # val and dval must have the same type
    @test_throws MethodError Duplicated(1.0, 1)

    # SubArray: matching views are accepted
    a = [1.0, 2.0, 3.0]
    da = zero(a)
    v = view(a, 1:2)
    dv = view(da, 1:2)
    @test Duplicated(v, dv) isa Duplicated{<:SubArray}

    # SubArray: mismatched views are rejected unless `check=false`
    dv_bad = view(da, 2:3)
    @test_throws AssertionError Duplicated(v, dv_bad)
    @test Duplicated(v, dv_bad, false) isa Duplicated{<:SubArray}
end

@testset "Duplicated{Any}" begin
    # When the type parameter is abstract (e.g. `Any`), the constructor still
    # enforces that the primal and shadow share the same concrete type.
    d = Duplicated{Any}(1.0, 2.0)
    @test d isa Duplicated{Any}
    @test d.val === 1.0
    @test d.dval === 2.0

    @test Duplicated{Any}([1.0], [2.0]) isa Duplicated{Any}

    # mismatched concrete types are rejected
    @test_throws AssertionError Duplicated{Any}(1.0, 2)
    @test_throws AssertionError Duplicated{Any}([1.0], [1.0f0])
end

@testset "DuplicatedNoNeed" begin
    x = [1.0]
    dx = [0.0]
    d = DuplicatedNoNeed(x, dx)
    @test d isa DuplicatedNoNeed{Vector{Float64}}
    @test d isa EnzymeCore.Annotation{Vector{Float64}}
    @test d.val === x
    @test d.dval === dx

    @test DuplicatedNoNeed{Vector{Float64}}(x, dx) isa DuplicatedNoNeed{Vector{Float64}}

    # SubArray handling mirrors `Duplicated`
    a = [1.0, 2.0, 3.0]
    da = zero(a)
    v = view(a, 1:2)
    dv = view(da, 1:2)
    @test DuplicatedNoNeed(v, dv) isa DuplicatedNoNeed{<:SubArray}
    dv_bad = view(da, 2:3)
    @test_throws AssertionError DuplicatedNoNeed(v, dv_bad)
    @test DuplicatedNoNeed(v, dv_bad, false) isa DuplicatedNoNeed{<:SubArray}

    # abstract type parameter still enforces matching concrete types
    @test DuplicatedNoNeed{Any}(1.0, 2.0) isa DuplicatedNoNeed{Any}
    @test_throws AssertionError DuplicatedNoNeed{Any}(1.0, 2)
end

@testset "BatchDuplicated" begin
    x = [1.0]
    dx1 = [0.0]
    dx2 = [0.0]
    d = BatchDuplicated(x, (dx1, dx2))
    @test d isa BatchDuplicated{Vector{Float64}, 2}
    @test d isa EnzymeCore.Annotation{Vector{Float64}}
    @test d.val === x
    @test d.dval === (dx1, dx2)
    @test EnzymeCore.batch_size(d) == 2
    @test EnzymeCore.batch_size(typeof(d)) == 2

    @test BatchDuplicated{Vector{Float64}, 2}(x, (dx1, dx2)) isa BatchDuplicated{Vector{Float64}, 2}

    # SubArray handling
    a = [1.0, 2.0, 3.0]
    da = zero(a)
    v = view(a, 1:2)
    dv = view(da, 1:2)
    @test BatchDuplicated(v, (dv, dv)) isa BatchDuplicated{<:SubArray, 2}
    dv_bad = view(da, 2:3)
    @test_throws AssertionError BatchDuplicated(v, (dv, dv_bad))
    @test BatchDuplicated(v, (dv, dv_bad), false) isa BatchDuplicated{<:SubArray, 2}

    # abstract type parameter still enforces matching concrete types per shadow
    @test BatchDuplicated{Any, 2}(1.0, (2.0, 3.0)) isa BatchDuplicated{Any, 2}
    @test_throws AssertionError BatchDuplicated{Any, 2}(1.0, (2.0, 3))
end

@testset "BatchDuplicatedNoNeed" begin
    x = [1.0]
    dx1 = [0.0]
    dx2 = [0.0]
    d = BatchDuplicatedNoNeed(x, (dx1, dx2))
    @test d isa BatchDuplicatedNoNeed{Vector{Float64}, 2}
    @test d isa EnzymeCore.Annotation{Vector{Float64}}
    @test d.val === x
    @test d.dval === (dx1, dx2)
    @test EnzymeCore.batch_size(d) == 2
    @test EnzymeCore.batch_size(typeof(d)) == 2

    @test BatchDuplicatedNoNeed{Vector{Float64}, 2}(x, (dx1, dx2)) isa BatchDuplicatedNoNeed{Vector{Float64}, 2}

    a = [1.0, 2.0, 3.0]
    da = zero(a)
    v = view(a, 1:2)
    dv = view(da, 1:2)
    @test BatchDuplicatedNoNeed(v, (dv, dv)) isa BatchDuplicatedNoNeed{<:SubArray, 2}
    dv_bad = view(da, 2:3)
    @test_throws AssertionError BatchDuplicatedNoNeed(v, (dv, dv_bad))
    @test BatchDuplicatedNoNeed(v, (dv, dv_bad), false) isa BatchDuplicatedNoNeed{<:SubArray, 2}

    @test BatchDuplicatedNoNeed{Any, 2}(1.0, (2.0, 3.0)) isa BatchDuplicatedNoNeed{Any, 2}
    @test_throws AssertionError BatchDuplicatedNoNeed{Any, 2}(1.0, (2.0, 3))
end

@testset "MixedDuplicated" begin
    r = Ref(0.0)
    d = MixedDuplicated(1.0, r)
    @test d isa MixedDuplicated{Float64}
    @test d isa EnzymeCore.Annotation{Float64}
    @test d.val === 1.0
    @test d.dval === r

    @test MixedDuplicated{Float64}(1.0, r) isa MixedDuplicated{Float64}

    # abstract type parameter still enforces matching concrete types
    @test MixedDuplicated{Any}(1.0, Ref{Any}(2.0)) isa MixedDuplicated{Any}
    @test_throws AssertionError MixedDuplicated{Any}(1.0, Ref{Any}(2))
end

@testset "BatchMixedDuplicated" begin
    r1 = Ref(0.0)
    r2 = Ref(0.0)
    d = BatchMixedDuplicated(1.0, (r1, r2))
    @test d isa BatchMixedDuplicated{Float64, 2}
    @test d isa EnzymeCore.Annotation{Float64}
    @test d.val === 1.0
    @test d.dval === (r1, r2)
    @test EnzymeCore.batch_size(d) == 2
    @test EnzymeCore.batch_size(typeof(d)) == 2

    @test BatchMixedDuplicated{Float64, 2}(1.0, (r1, r2)) isa BatchMixedDuplicated{Float64, 2}

    @test BatchMixedDuplicated{Any, 2}(1.0, (Ref{Any}(2.0), Ref{Any}(3.0))) isa BatchMixedDuplicated{Any, 2}
    @test_throws AssertionError BatchMixedDuplicated{Any, 2}(1.0, (Ref{Any}(2.0), Ref{Any}(3)))
end
