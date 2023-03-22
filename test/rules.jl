module ForwardRules

using Enzyme
using Enzyme: EnzymeRules
using Test

import .EnzymeRules: forward, Annotation, has_frule_from_sig

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

function has_frule(f, @nospecialize(RT), @nospecialize(TT::Type{<:Tuple}); world=Base.get_world_counter())
    TT = Base.unwrap_unionall(TT)
    TT = Tuple{<:Annotation{Core.typeof(f)}, Type{<:RT}, TT.parameters...}
    EnzymeRules.isapplicable(forward, TT; world)
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

g(x) = x ^ 2
function forward(func::Const{typeof(g)}, ::Type{<:Const}, x::Const)
    return Const(g(x.val))
end

@testset "Registry" begin
    @test_throws Enzyme.Compiler.EnzymeRuntimeException Enzyme.autodiff(Forward, g, Duplicated(1.0, 1.0))

    rh(cond, x) = cond ? g(x) : x
    @test Enzyme.autodiff(Forward, rh, Const(false), Duplicated(1.0, 1.0)) == (1.0,)
    @test_throws Enzyme.Compiler.EnzymeRuntimeException Enzyme.autodiff(Forward, rh, Const(true), Duplicated(1.0, 1.0))
end

function alloc_sq(x)
    return Ref(x*x)
end

function h(x)
    alloc_sq(x)[]
end

function h2(x)
    y = alloc_sq(x)[]
    y * y
end

function forward(func::Const{typeof(alloc_sq)}, ::Type{<:Duplicated}, x::Duplicated)
    return Duplicated(Ref(x.val*x.val), Ref(10*2*x.val*x.dval))
end

function forward(func::Const{typeof(alloc_sq)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated)
    return Ref(1000*2*x.val*x.dval)
end

function alloc_sq2(x)
    return Ref(x*x)
end

function h3(x)
    alloc_sq2(x)[]
end

function forward(func::Const{typeof(alloc_sq2)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated)
    return Duplicated(Ref(0.0), Ref(1000*2*x.val*x.dval))
end

@testset "Shadow" begin
    @test Enzyme.autodiff(Forward, h, Duplicated(3.0, 1.0)) == (6000.0,)
    @test Enzyme.autodiff(Forward, h, Duplicated, Duplicated(3.0, 1.0))  == (9.0, 60.0)
    @test Enzyme.autodiff(Forward, h2, Duplicated(3.0, 1.0))  == (1080.0,)
    @test_throws Enzyme.Compiler.EnzymeRuntimeException Enzyme.autodiff(Forward, h3, Duplicated(3.0, 1.0)) 
end

end # module ForwardRules
