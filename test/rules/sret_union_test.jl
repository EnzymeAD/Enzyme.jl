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

# https://github.com/EnzymeAD/Enzyme.jl/issues/3925
# Functions using `sret_union` return `{ {} addrspace(10)*, i8 }`, where element 0
# is `null` whenever the union value is returned unboxed in the caller's `sret`
# buffer. Caching that struct into an old-gen loop tape must not pass `null` as a
# child pointer to `@julia.write_barrier`.
struct SretWriteBarrierElem
    val::UInt8
end

@noinline sret_wb_step(k::Int, n::Int)::Union{Nothing, Tuple{SretWriteBarrierElem, Int}} =
    k <= n ? (SretWriteBarrierElem(UInt8(k)), k + 1) : nothing

@noinline function sret_wb_map(c::Float64, b::SretWriteBarrierElem)
    b.val == 0x06 && GC.gc(true)
    return c^Int(b.val)
end

@noinline function sret_wb_loop_sum(n::Int, c::Float64)
    s = 0.0
    for k in 1:n
        u = sret_wb_step(k, n)
        u === nothing && break
        s += sret_wb_map(c, u[1])
    end
    return s
end

function sret_wb_outer!(x::Vector{Float64}, n::Int, c::Float64)
    x[1] *= sret_wb_loop_sum(n, c)
    return nothing
end

@testset "sret_union null box write barrier in old-gen tape (#3925)" begin
    x = [3.0]
    dx = [1.0]
    c = 2.0
    res = autodiff(
        set_runtime_activity(Reverse),
        sret_wb_outer!,
        Const,
        Duplicated(x, dx),
        Const(16),
        Active(c),
    )
    @test x[1] ≈ 3.0 * sum(c^k for k in 1:16)
    @test res[1][3] ≈ 3.0 * sum(k * c^(k - 1) for k in 1:16)
end
