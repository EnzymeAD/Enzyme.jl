using Enzyme, Test
using Enzyme: EnzymeRules

const M_test = [1.0 0.2 0.0; 0.0 1.0 0.1; 0.3 0.0 1.0]
inner_test(t) = sum((M_test * t) .^ 2)
g_test(p) = sum(Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), inner_test, p)[1])

@testset "Nested BLAS AD calling convention / GC preserve inlining" begin
    dp = [0.0, 0.0, 0.0]
    Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(g_test), Enzyme.Active, Enzyme.Duplicated([1.0, 2.0, 3.0], dp))
    @test dp ≈ [3.18, 2.68, 2.82]
end

abstract type AbstractDomainCallConv end

mutable struct InnerPlanCallConv
    b::Float64
end

struct MyPlanCallConv
    a::Int64
    b::Float64
    c::Int32
    d::Int32
    plan::InnerPlanCallConv
    phases::Vector{Float64}
    indices::Tuple{Vector{Int64}, Vector{Int64}}
    h::Int64
end

struct MyDomainCallConv <: AbstractDomainCallConv
    plan::MyPlanCallConv
end

@noinline forward_plan_callconv(g::AbstractDomainCallConv) = getfield(g, :plan)
EnzymeRules.inactive(::typeof(forward_plan_callconv), args...) = nothing

@noinline getplan_callconv(p::MyPlanCallConv) = getfield(p, :plan)
EnzymeRules.inactive(::typeof(getplan_callconv), args...) = nothing

@noinline getindices_callconv(p::MyPlanCallConv) = getfield(p, :indices)
EnzymeRules.inactive(::typeof(getindices_callconv), args...) = nothing

@noinline function my_nuft_callconv!(out, A, b)
    out .= b .* A.b
    return nothing
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth,
    func::Const{typeof(my_nuft_callconv!)},
    ::Type{<:Const},
    out::EnzymeRules.Annotation,
    A::EnzymeRules.Annotation,
    b::EnzymeRules.Annotation,
)
    primal = EnzymeRules.needs_primal(config) ? out.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? out.dval : nothing
    func.val(out.val, A.val, b.val)
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfigWidth,
    ::Const{typeof(my_nuft_callconv!)},
    ::Type{RT},
    tape,
    out::EnzymeRules.Annotation,
    A::EnzymeRules.Annotation,
    b::EnzymeRules.Annotation,
) where {RT}
    b.dval .+= out.dval .* A.val.b
    fill!(out.dval, 0)
    return (nothing, nothing, nothing)
end

function applyphases_callconv!(vis, phases)
    for i in eachindex(vis, phases)
        vis[i] *= phases[i]
    end
    return vis
end

@inline function applyft_callconv(p, img)
    vis = similar(img)
    plan = getplan_callconv(p)
    iminds, visinds = getindices_callconv(p)
    for i in eachindex(iminds, visinds)
        imind = iminds[i]
        visind = visinds[i]
        vis_view = @view(vis[visind:visind])
        img_view = @view(img[imind:imind])
        my_nuft_callconv!(vis_view, plan, img_view)
    end
    applyphases_callconv!(vis, p.phases)
    return vis
end

@noinline function visibilitymap_numeric_callconv(grid::AbstractDomainCallConv, img::Vector{Float64})
    return applyft_callconv(forward_plan_callconv(grid), img)
end

@noinline function foo_callconv(grid::AbstractDomainCallConv, img)
    return sum(visibilitymap_numeric_callconv(grid, img))
end

@testset "Custom rule calling conv rewrite" begin
    inner = InnerPlanCallConv(2.0)
    plan = MyPlanCallConv(1, 2.0, 3, 4, inner, [2.0, 3.0, 4.0], ([1, 2, 3], [1, 2, 3]), 7)
    grid = MyDomainCallConv(plan)
    img = [1.0, 2.0, 3.0]
    dimg = zeros(3)

    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        foo_callconv,
        Active,
        Const(grid),
        Duplicated(img, dimg),
    )

    @test dimg == [4.0, 6.0, 8.0]
end

struct MyBufferCallConv{A}
    data::A
    count::Int32
    datatype::Float64
end

mutable struct MyRequestCallConv
    buffer::Any
end

@inline function my_isend_callconv(x, req)
    buf = MyBufferCallConv(x, Int32(1), 1.0)
    req.buffer = buf
    return req
end

function f_my_request_callconv(x)
    req = MyRequestCallConv(nothing)
    my_isend_callconv(x, req)
    buf = req.buffer::MyBufferCallConv{Vector{Float64}}
    return buf.data[1] * buf.data[1]
end

@testset "Typed Alloca restore_alloca_type! with Any field" begin
    x = [3.0]
    dx = zeros(1)
    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        f_my_request_callconv,
        Active,
        Duplicated(x, dx),
    )
    @test dx ≈ [6.0]
end

mutable struct MutableUnion
    u::Vector{Float64}
    conv::Union{Nothing,Bool}     # Union{Nothing,Int} triggers it too; a plain Bool does NOT
end
@noinline dispatch(x)::MutableUnion = Base.inferencebarrier(MutableUnion([x], nothing))   # runtime dispatch required

f_mutunion(x) = (m = dispatch(x); m.u[1]^2)

@testset "Typed Alloca restore_alloca_type! with Any field" begin
    @test Enzyme.gradient(Enzyme.Reverse, f_mutunion, 3.0)[1] ≈ 6.0
end
