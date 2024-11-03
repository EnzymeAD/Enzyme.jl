using Enzyme, LinearAlgebra, Test
using Random, Statistics

function gcloaded_fixup(dest, src)
    N = size(src)
    dat = src.data
    len = N[1]

    i = 1
    while true
        j = 1
        while true
            ld = @inbounds if i <= j
                dat[(i-1) * 2 + j]
            else
                dat[(j-1) * 2 + i]
            end
            @inbounds dest[(i-1) * 2 + j] = ld
            if j == len
                break
            end
            j += 1
        end
        if i == len
            break
        end
        i += 1
    end
    return nothing
end

@testset "GCLoaded fixup" begin
	H = Hermitian(Matrix([4.0 1.0; 2.0 5.0]))
	dest = Matrix{Float64}(undef, 2, 2)

	Enzyme.autodiff(
	    ForwardWithPrimal,
	    gcloaded_fixup,
	    Const,
	    Const(dest),
	    Const(H),
	)[1]
    @test dest ≈ [4.0 2.0; 2.0 5.0]
    dest = Matrix{Float64}(undef, 2, 2)
    gcloaded_fixup(dest, H)
    @test dest ≈ [4.0 2.0; 2.0 5.0]
end

struct MyNormal
    sigma::Float64
    off::Float64
end

struct MvLocationScale{
    S, D, L
}
    location ::L
    scale    ::S
    dist     ::D
end

@noinline function law(dist, flat::AbstractVector)
    ccall(:jl_, Cvoid, (Any,), flat)
    n_dims = div(length(flat), 2)
    data = first(flat, n_dims)
    scale = Diagonal(data)
    return MvLocationScale(nothing, scale, dist)
end

function destructure(q::MvLocationScale)
    return diag(q.scale)
end


myxval(d::MyNormal, z::Real) = muladd(d.sigma, z, d.off)

function myrand!(rng::AbstractRNG, d::MyNormal, A::AbstractArray{<:Real})
    # randn!(rng, A)
    map!(Base.Fix1(myxval, d), A, A)
    return A
end

function man(q::MvLocationScale)
    dist = MyNormal(1.0, 0.0)
    
    out = ones(2,3) # Array{Float64}(undef, (2,3))
    @inbounds myrand!(Random.default_rng(), dist, out)

    return q.scale[1] * out
end

function estimate_repgradelbo_ad_forward(params, dist)
    q = law(dist, params)
    samples = man(q)
    mean(samples)
end

@testset "Removed undef arguments" begin
    T = Float64
	d = 2
    dist = MyNormal(1.0, 0.0)
    q = MvLocationScale(zeros(T, d), Diagonal(ones(T, d)), dist)
    params = destructure(q)
    
    ∇x = zero(params)
    fill!(∇x, zero(eltype(∇x)))
    
    estimate_repgradelbo_ad_forward(params, dist)

    Enzyme.autodiff(
        set_runtime_activity(Enzyme.ReverseWithPrimal),
        estimate_repgradelbo_ad_forward,
        Enzyme.Active,
        Enzyme.Duplicated(params, ∇x),
        Enzyme.Const(dist)
    )
end
