using Enzyme, LinearAlgebra, Test

# `iterate(::SretUnionIter)` returns an isbits union, `Union{Nothing, Tuple{Int8, Int}}`, which
# (not inlined) is returned through a stack buffer. The pointer to the payload inside that
# buffer is a decayed pointer, and in `coef` it is a loop-carried phi that Enzyme caches.
# `nodecayed_phis!` used to root that phi on the stack buffer itself, disguised as a GC
# object, so the tape held a dangling stack pointer after the augmented forward pass and the
# next full collection crashed while marking it.
struct SretUnionIter
    a::Int8
end
@noinline function Base.iterate(p::SretUnionIter, st::Int = 0)
    st >= (p.a == 3 ? 4 : 1) && return nothing
    return (Int8(st), st + 1)
end
Base.IteratorSize(::Type{SretUnionIter}) = Base.SizeUnknown()

@noinline function sret_union_mat(a::Int8, b::Int8)
    n = a == 3 && b == 3 ? 2 : 1
    M = zeros(ComplexF64, n, n)
    M[1, 1] = a == b ? -1 : 1
    return M
end
sret_union_coef(a::Int8) = sum((b + 1) / (a + 1) * tr(sret_union_mat(a, b)) for b in SretUnionIter(a))

function sret_union_scale!(x::Vector{ComplexF64}, labels::Vector{Int8})
    for i in eachindex(x, labels)
        x[i] *= sret_union_coef(labels[i])
    end
    return x
end

@testset "Cached pointer into a union sret buffer" begin
    labels = Int8[0, 3, 1, 3, 2, 3, 3, 3]
    fwd, rev = autodiff_thunk(
        ReverseSplitWithPrimal, Const{typeof(sret_union_scale!)}, Duplicated,
        Duplicated{Vector{ComplexF64}}, Const{Vector{Int8}}
    )
    x = ones(ComplexF64, length(labels))
    dx = zeros(ComplexF64, length(labels))
    tape, y, dy = fwd(Const(sret_union_scale!), Duplicated(x, dx), Const(labels))
    # The tape must only reference valid objects: a full collection marks all of it.
    GC.gc(true)
    GC.gc(true)
    @test y ≈ [sret_union_coef(l) for l in labels]
    dy .= 1
    rev(Const(sret_union_scale!), Duplicated(x, dx), Const(labels), tape)
    @test dx ≈ [conj(sret_union_coef(l)) for l in labels]
end
