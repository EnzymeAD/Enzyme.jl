using Enzyme, Test
using Enzyme: EnzymeRules, Annotation

# A region that holds no differentiable data. Its shadow carries data from the reverse
# pass of `region_end` to the reverse pass of `region_begin`.
mutable struct ActiveTypeRegion
    id::Int
    reverse_id::Int
end
EnzymeRules.active_type(::Type{ActiveTypeRegion}) = true

struct ActiveTypeWrapper
    region::ActiveTypeRegion
end

mutable struct ActiveTypeBoth
    id::Int
end
EnzymeRules.active_type(::Type{ActiveTypeBoth}) = true
EnzymeRules.inactive_type(::Type{ActiveTypeBoth}) = true

mutable struct ActiveTypeNone
    id::Int
end

const ACTIVE_TYPE_LOG = Any[]
const ACTIVE_TYPE_COUNTER = Ref(0)

@noinline function region_begin()
    ACTIVE_TYPE_COUNTER[] += 1
    return ActiveTypeRegion(ACTIVE_TYPE_COUNTER[], 0)
end
@noinline function region_end(r::ActiveTypeRegion)
    push!(ACTIVE_TYPE_LOG, (:end, r.id))
    return nothing
end

function EnzymeRules.augmented_primal(
        config, ::Const{typeof(region_begin)},
        RT::Type{<:Annotation}
    )
    r = region_begin()
    primal = EnzymeRules.needs_primal(config) ? r : nothing
    if !EnzymeRules.needs_shadow(config)
        return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
    end
    shadow = ActiveTypeRegion(r.id, 0)
    width = EnzymeRules.width(config)
    shadows = width == 1 ? shadow : ntuple(Returns(shadow), Val(width))
    return EnzymeRules.AugmentedReturn(primal, shadows, shadow)
end

function EnzymeRules.reverse(
        config, ::Const{typeof(region_begin)},
        RT::Type{<:Annotation}, shadow
    )
    push!(ACTIVE_TYPE_LOG, (:reverse_begin, shadow === nothing ? 0 : shadow.reverse_id))
    return ()
end

function EnzymeRules.augmented_primal(
        config, ::Const{typeof(region_end)},
        ::Type{<:Const}, r::Annotation{ActiveTypeRegion}
    )
    region_end(r.val)
    return EnzymeRules.AugmentedReturn(nothing, nothing, nothing)
end

function EnzymeRules.reverse(
        config, ::Const{typeof(region_end)}, ::Type{<:Const},
        tape, r::Annotation{ActiveTypeRegion}
    )
    shadow = r isa Duplicated ? r.dval : r isa BatchDuplicated ? first(r.dval) : nothing
    if shadow !== nothing
        shadow.reverse_id = 10 * r.val.id
    end
    push!(ACTIVE_TYPE_LOG, (:reverse_end, typeof(r)))
    return (nothing,)
end

function active_type_region(x)
    r = region_begin()
    y = x^2
    region_end(r)
    return y
end

function active_type_region!(y, x)
    r = region_begin()
    y[1] = x[1]^2
    region_end(r)
    return nothing
end

@testset "active_type" begin
    world = Base.get_world_counter()
    @test Enzyme.Compiler.active_reg(ActiveTypeNone, world) == Enzyme.Compiler.AnyState
    @test Enzyme.Compiler.active_reg(ActiveTypeRegion, world) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(ActiveTypeWrapper, world) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg(ActiveTypeBoth, world) == Enzyme.Compiler.AnyState
    @test Enzyme.guess_activity(ActiveTypeRegion, Reverse) == Duplicated{ActiveTypeRegion}
    @test Enzyme.guess_activity(ActiveTypeNone, Reverse) == Const{ActiveTypeNone}

    # The shadow carries data from the reverse of `region_end` to the reverse of
    # `region_begin`
    empty!(ACTIVE_TYPE_LOG)
    ACTIVE_TYPE_COUNTER[] = 0
    @test autodiff(Reverse, active_type_region, Active, Active(3.0))[1][1] == 6.0
    @test ACTIVE_TYPE_LOG == Any[
        (:end, 1), (:reverse_end, Duplicated{ActiveTypeRegion}),
        (:reverse_begin, 10),
    ]

    # Batch mode
    empty!(ACTIVE_TYPE_LOG)
    ACTIVE_TYPE_COUNTER[] = 0
    x = [3.0]
    dxs = ([0.0], [0.0])
    autodiff(
        Reverse, active_type_region!, Const,
        BatchDuplicated([0.0], ([1.0], [2.0])), BatchDuplicated(x, dxs)
    )
    @test dxs == ([6.0], [12.0])
    @test ACTIVE_TYPE_LOG == Any[
        (:end, 1),
        (:reverse_end, BatchDuplicated{ActiveTypeRegion, 2}), (:reverse_begin, 10),
    ]
end
