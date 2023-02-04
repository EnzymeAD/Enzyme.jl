module ForwardRules

using Enzyme
using Enzyme: EnzymeRules
using Test

import .EnzymeRules: forward, Annotation, has_frule, has_frule_from_sig

f(x) = x^2

function f_ip(x)
    x[1] *= x[1]
    return nothing
end

function forward(::Const{typeof(f)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated)
    return 10+2*x.val*x.dval
end

function forward(::Const{typeof(f)}, ::Type{<:BatchDuplicatedNoNeed}, x::BatchDuplicated{T, N}) where {T, N}
    return NTuple{N, T}(1000+2*x.val*dv for dv in x.dval)
end

function forward(func::Const{typeof(f)}, ::Type{<:Duplicated}, x::Duplicated)
    return Duplicated(func.val(x.val), 100+2*x.val*x.dval)
end

function forward(func::Const{typeof(f)}, ::Type{<:BatchDuplicated}, x::BatchDuplicated{T, N}) where {T,N}
    return BatchDuplicated(func.val(x.val), NTuple{N, T}(10000+2*x.val*dv for dv in x.dval))
end

function forward(::Const{Core.typeof(f_ip)}, ::Type{<:Const}, x::Duplicated)
    ld = x.val[1]
    x.val[1] *= ld
    x.dval[1] *= 2 * ld + 10
    return nothing
end

@testset "has_frule" begin
    @test has_frule_from_sig(Base.signature_type(f, Tuple{Float64}))
    @test has_frule_from_sig(Base.signature_type(f_ip, Tuple{Vector{Float64}}))

    @test has_frule(f, Duplicated, Tuple{<:Duplicated})
    @test has_frule(f, DuplicatedNoNeed, Tuple{<:Duplicated})
    @test has_frule(f, BatchDuplicated, Tuple{<:BatchDuplicated})
    @test has_frule(f, BatchDuplicatedNoNeed, Tuple{<:BatchDuplicated})

    @test !has_frule(f, Duplicated, Tuple{<:BatchDuplicated})
    @test !has_frule(f, DuplicatedNoNeed, Tuple{<:BatchDuplicated})
    @test !has_frule(f, BatchDuplicated, Tuple{<:Duplicated})
    @test !has_frule(f, BatchDuplicatedNoNeed, Tuple{<:Duplicated})
end

@testset "autodiff(Forward, ...) custom rules" begin
    @test autodiff(Forward, f, Duplicated(2.0, 1.0))[1] ≈ 14.0
    @test autodiff(Forward, x->f(x)^2, Duplicated(2.0, 1.0))[1] ≈ 832.0

    res = autodiff(Forward, f, BatchDuplicatedNoNeed, BatchDuplicated(2.0, (1.0, 3.0)))[1] 
    @test res[1] ≈ 1004.0
    @test res[2] ≈ 1012.0

    res = Enzyme.autodiff(Forward, x->f(x)^2, BatchDuplicatedNoNeed, BatchDuplicated(2.0, (1.0, 3.0)))[1]

    @test res[1] ≈ 80032.0
    @test res[2] ≈ 80096.0
end

@testset "In place" begin
    vec = [2.0]
    dvec = [1.0]

    Enzyme.autodiff(Forward, f_ip, Duplicated(vec, dvec))

    @test vec ≈ [4.0]
    @test dvec ≈ [14.0]
end

# TODO: Test error for no frule applicable despite frule on Function.


end # module ForwardRules
