module ForwardRules

using Enzyme
using Enzyme: EnzymeRules
using Test

import .EnzymeRules: forward, Annotation, has_frule_from_sig, FwdConfig

f(x) = x^2

function f_ip(x)
    x[1] *= x[1]
    return nothing
end

function forward(config, ::Const{typeof(f)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated)
    return 10 + 2 * x.val * x.dval
end

function forward(config, ::Const{typeof(f)}, ::Type{<:BatchDuplicatedNoNeed}, x::BatchDuplicated{T, N}) where {T, N}
    return NTuple{N, T}(1000 + 2 * x.val * dv for dv in x.dval)
end

function forward(config, func::Const{typeof(f)}, ::Type{<:Duplicated}, x::Duplicated)
    return Duplicated(func.val(x.val), 100 + 2 * x.val * x.dval)
end

function forward(config, func::Const{typeof(f)}, ::Type{<:BatchDuplicated}, x::BatchDuplicated{T, N}) where {T, N}
    return BatchDuplicated(func.val(x.val), NTuple{N, T}(10000 + 2 * x.val * dv for dv in x.dval))
end

function forward(config, ::Const{Core.typeof(f_ip)}, ::Type{<:Const}, x::Duplicated)
    ld = x.val[1]
    x.val[1] *= ld
    x.dval[1] *= 2 * ld + 10
    return nothing
end

function has_frule(f, @nospecialize(RT), @nospecialize(TT::Type{<:Tuple}); world = Base.get_world_counter())
    TT = Base.unwrap_unionall(TT)
    TT = Tuple{<:FwdConfig, <:Annotation{Core.typeof(f)}, Type{<:RT}, TT.parameters...}
    return EnzymeRules.isapplicable(forward, TT; world)
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
    @test autodiff(Forward, x -> f(x)^2, Duplicated(2.0, 1.0))[1] ≈ 832.0

    res = autodiff(Forward, f, BatchDuplicated, BatchDuplicated(2.0, (1.0, 3.0)))[1]
    @test res[1] ≈ 1004.0
    @test res[2] ≈ 1012.0

    res = Enzyme.autodiff(Forward, x -> f(x)^2, BatchDuplicated, BatchDuplicated(2.0, (1.0, 3.0)))[1]

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

g(x) = x^2
function forward(config, func::Const{typeof(g)}, ::Type{<:Const}, x::Const)
    return Const(g(x.val))
end

@testset "Registry" begin
    @test_throws MethodError Enzyme.autodiff(Forward, g, Duplicated(1.0, 1.0))

    rh(cond, x) = cond ? g(x) : x
    @test Enzyme.autodiff(Forward, rh, Const(false), Duplicated(1.0, 1.0)) == (1.0,)
    @test_throws MethodError Enzyme.autodiff(Forward, rh, Const(true), Duplicated(1.0, 1.0))
end

function alloc_sq(x)
    return Ref(x * x)
end

function h(x)
    return alloc_sq(x)[]
end

function h2(x)
    y = alloc_sq(x)[]
    return y * y
end

function forward(config, func::Const{typeof(alloc_sq)}, ::Type{<:Duplicated}, x::Duplicated)
    return Duplicated(Ref(x.val * x.val), Ref(10 * 2 * x.val * x.dval))
end

function forward(config, func::Const{typeof(alloc_sq)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated)
    return Ref(1000 * 2 * x.val * x.dval)
end

function alloc_sq2(x)
    return Ref(x * x)
end

function h3(x)
    return alloc_sq2(x)[]
end

function forward(config, func::Const{typeof(alloc_sq2)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated)
    return Duplicated(Ref(0.0), Ref(1000 * 2 * x.val * x.dval))
end

@testset "Shadow" begin
    @test Enzyme.autodiff(Forward, h, Duplicated(3.0, 1.0)) == (6000.0,)
    @test Enzyme.autodiff(ForwardWithPrimal, h, Duplicated(3.0, 1.0)) == (60.0, 9.0)
    @test Enzyme.autodiff(Forward, h2, Duplicated(3.0, 1.0)) == (1080.0,)
    @test_throws Enzyme.Compiler.ForwardRuleReturnError Enzyme.autodiff(Forward, h3, Duplicated(3.0, 1.0))
end

foo(x) = 2x;

function EnzymeRules.forward(
        config,
        func::Const{typeof(foo)},
        RT::Type{<:Union{Duplicated, BatchDuplicated}},
        x::Union{Duplicated, BatchDuplicated},
    )
    if RT <: BatchDuplicated
        return BatchDuplicated(func.val(x.val), map(func.val, x.dval))
    else
        return Duplicated(func.val(x.val), func.val(x.dval))
    end
end

@testset "Batch complex" begin
    res = autodiff(ForwardWithPrimal, foo, BatchDuplicated(0.1 + 0im, (0.2 + 0im, 0.3 + 0im)))
    @test res[2] ≈ 0.2 + 0.0im
    @test res[1][1] ≈ 0.4 + 0.0im
    @test res[1][2] ≈ 0.6 + 0.0im
end

struct TestSpace
    dim::Int64
    flag::Bool
end

struct TestTensor
    space::TestSpace
    data::Vector{Float64}
end

function test_trace!(dst::TestTensor, src::TestTensor, beta::Float64; kw = 1)
    dst.data .= dst.data .* beta
    return nothing
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(test_trace!)},
        RT::Type{<:Const},
        dst::Annotation{TestTensor},
        src::Annotation{TestTensor},
        beta::Duplicated{Float64};
        kw = 1
    )
    test_trace!(dst.val, src.val, beta.val; kw = kw)
    return nothing
end

struct TestCallWithKWargs{K <: NamedTuple}
    kwargs::K
end

function (f::TestCallWithKWargs)(func, args...)
    return func(args...; f.kwargs...)
end

@testset "Rooted constant with active non-rooted arg" begin
    dst = TestTensor(TestSpace(2, true), [1.0, 2.0])
    src = TestTensor(TestSpace(2, true), [3.0, 4.0])
    beta = 2.0
    dbeta = 1.0

    # `test_trace!` has a custom rule, which may read the whole argument, so since
    # EnzymeAD/Enzyme#3352 nothing infers the data half of the split convention
    # `readnone` from its body on any Julia version: runtime activity is needed.
    @test_throws Enzyme.Compiler.EnzymeRuntimeActivityError autodiff(
        Forward,
        TestCallWithKWargs((;)),
        Const,
        Const(test_trace!),
        Const(dst),
        Const(src),
        Duplicated(beta, dbeta)
    )

    dst = TestTensor(TestSpace(2, true), [1.0, 2.0])

    autodiff(
        set_runtime_activity(Forward),
        TestCallWithKWargs((;)),
        Const,
        Const(test_trace!),
        Const(dst),
        Const(src),
        Duplicated(beta, dbeta)
    )
    @test dst.data == [2.0, 4.0]
end

function foo_3313(idx::Tuple{NTuple{N1, Int}, NTuple{N2, Int}}, x::Vararg{Vector{Float64}, N}) where {N1, N2, N}
    return sum(sum.(x))
end

function EnzymeRules.forward(
        config,
        ::Const{typeof(foo_3313)},
        ::Type{RT},
        idx::Const,
        x::Vararg{Union{Duplicated{Vector{Float64}}, BatchDuplicated{Vector{Float64}}}, N},
    ) where {RT, N}
    primal = foo_3313(idx.val, map(xi -> xi.val, x)...)
    if EnzymeRules.width(config) == 1
        dval = sum(sum(xi.dval) for xi in x)
    else
        w = EnzymeRules.width(config)
        dval = ntuple(b -> sum(sum(xi.dval[b]) for xi in x), Val(w))
    end
    if EnzymeRules.needs_primal(config)
        return EnzymeRules.needs_shadow(config) ? RT(primal, dval) : primal
    else
        return EnzymeRules.needs_shadow(config) ? dval : nothing
    end
end

function EnzymeRules.augmented_primal(
        config,
        ::Const{typeof(foo_3313)},
        ::Type{<:Active},
        idx::Const,
        x::Vararg{Union{Duplicated{Vector{Float64}}, BatchDuplicated{Vector{Float64}}}, N},
    ) where {N}
    primal = EnzymeRules.needs_primal(config) ? foo_3313(idx.val, map(xi -> xi.val, x)...) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(
        config,
        ::Const{typeof(foo_3313)},
        dret,
        tape,
        idx::Const,
        x::Vararg{Union{Duplicated{Vector{Float64}}, BatchDuplicated{Vector{Float64}}}, N},
    ) where {N}
    if EnzymeRules.width(config) == 1
        for xi in x
            xi.dval .+= dret.val
        end
    else
        for b in 1:EnzymeRules.width(config)
            for xi in x
                xi.dval[b] .+= dret[b].val
            end
        end
    end
    return (nothing, ntuple(_ -> nothing, Val(N))...)
end

bar_3313(idx, x) = foo_3313(idx, x...)
splat_3313(args...) = foo_3313(args...)

@testset "BITS_REF guaranteed-inactive arg in custom rule (#3313)" begin
    idx = (ntuple(identity, 2), ntuple(identity, 6))
    x = ([1.0, 2.0], [3.0, 4.0])

    dx = ([0.0, 0.0], [0.0, 0.0])
    autodiff(Reverse, bar_3313, Active, Const(idx), Duplicated(x, dx))
    @test dx[1] ≈ [1.0, 1.0]
    @test dx[2] ≈ [1.0, 1.0]

    dx1 = ([0.0, 0.0], [0.0, 0.0])
    dx2 = ([0.0, 0.0], [0.0, 0.0])
    autodiff(Reverse, bar_3313, Active, Const(idx), BatchDuplicated(x, (dx1, dx2)))
    @test dx1[1] ≈ [1.0, 1.0]
    @test dx1[2] ≈ [1.0, 1.0]
    @test dx2[1] ≈ [1.0, 1.0]
    @test dx2[2] ≈ [1.0, 1.0]

    vx1 = ([1.0, 2.0], [3.0, 4.0])
    vx2 = ([5.0, 6.0], [7.0, 8.0])
    @test autodiff(Forward, bar_3313, Duplicated, Const(idx), Duplicated(x, vx1))[1] ≈ 10.0
    @test Tuple(autodiff(Forward, bar_3313, BatchDuplicated, Const(idx), BatchDuplicated(x, (vx1, vx2)))[1]) == (10.0, 26.0)

    s_dx1 = [0.0, 0.0]
    s_dx2 = [0.0, 0.0]
    autodiff(Reverse, splat_3313, Active, Const(idx), Duplicated(x[1], s_dx1), Duplicated(x[2], s_dx2))
    @test s_dx1 ≈ [1.0, 1.0]
    @test s_dx2 ≈ [1.0, 1.0]
end

end # module ForwardRules
