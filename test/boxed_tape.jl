using Enzyme, Test
using Enzyme: EnzymeRules

const BigBitsTuple = NTuple{4100, Float64}

@noinline big_tape_inner(t::BigBitsTuple, v, i) = t[i] * v[1]^2

function big_tape_outer(x, r, r2, i)
    t = r[]
    r[] = r2[] # overwrites `t`, so the tape must keep it for the reverse pass
    v = [x[1]] # and a GC-tracked value next to it
    return big_tape_inner(t, v, i)
end

@testset "Split mode tape that Julia does not store inline" begin
    # The tape holds a `BigBitsTuple` (> 32 KiB) and a GC pointer. Julia needs 32-bit field
    # descriptors for such a type and keeps it boxed inside the returned
    # `(tape, primal, shadow)`, so the augmented forward pass has to box the tape too;
    # writing it inline overran the return buffer ("Illegal GC setup in which rerooting
    # is required").
    buf = collect(range(0.5, 2.0; length = 4100))
    r = Ref(GC.@preserve buf unsafe_load(Ptr{BigBitsTuple}(pointer(buf))))
    r2 = Ref(r[])
    x = [1.5]
    dx = zero(x)
    fwd, rev = autodiff_thunk(
        ReverseSplitWithPrimal, Const{typeof(big_tape_outer)}, Active{Float64},
        Duplicated{Vector{Float64}}, Const{typeof(r)}, Const{typeof(r)}, Const{Int}
    )
    TapeType = EnzymeRules.tape_type(typeof(fwd))
    @test !Base.allocatedinline(TapeType)
    tape, primal, _ = fwd(Const(big_tape_outer), Duplicated(x, dx), Const(r), Const(r2), Const(3))
    GC.gc()
    rev(Const(big_tape_outer), Duplicated(x, dx), Const(r), Const(r2), Const(3), 1.0, tape)
    @test primal ≈ buf[3] * x[1]^2
    @test dx[1] ≈ 2 * buf[3] * x[1]
end
