using Enzyme
using Enzyme.EnzymeRules
using Test

# Primal function
function my_func_union(x)
    return x * x
end

# Custom augmented primal returning a Union of AugmentedReturn types
@inline function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig, ::Const{typeof(my_func_union)}, ::Type{<:Active}, x
)
    primal = EnzymeRules.needs_primal(config) ? my_func_union(x.val) : nothing
    if x.val > 0.0
        return EnzymeRules.AugmentedReturn(primal, nothing, 1)
    else
        return EnzymeRules.AugmentedReturn(primal, nothing, 2.0)
    end
end

# Custom reverse rule consuming the union tape
@inline function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig, ::Const{typeof(my_func_union)}, dret::Active, tape, x
)
    # tape is a Union{Int64, Float64}
    dx = if tape isa Int64
        2.0 * x.val * tape
    else
        2.0 * x.val * tape
    end
    return (dx,)
end

@testset "Union sret return rule" begin
    grads = Enzyme.gradient(Reverse, my_func_union, 1.0)
    @test grads[1] ≈ 2.0
end
