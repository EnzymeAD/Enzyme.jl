using Enzyme
using Test

const THUNK_CACHE = Enzyme.Compiler.THUNK_CACHE

session_sq(x) = x * x

@testset "Session-scoped thunk cache" begin
    Enzyme.Compiler.reset_session!()
    epoch = Enzyme.Compiler.SESSION_EPOCH[]
    @test isodd(epoch)
    @test isempty(THUNK_CACHE.thunks)
    @test isempty(THUNK_CACHE.session_links)
    @test isempty(THUNK_CACHE.tapes)

    @test autodiff(Reverse, session_sq, Active(3.0))[1][1] == 6.0
    n = length(THUNK_CACHE.thunks)
    @test n >= 1
    # A compiled thunk is linked into this session's JIT.
    for r in values(THUNK_CACHE.thunks)
        l = Enzyme.Compiler.current_link(r.adjoint)
        @test l !== nothing && l.ptr != C_NULL && l.epoch == epoch
    end

    # Same job, same session: no new entry.
    @test autodiff(Reverse, session_sq, Active(4.0))[1][1] == 8.0
    @test length(THUNK_CACHE.thunks) == n

    # A new session drops everything; the already-seen signature is compiled and linked
    # again on its next call (its thunk handle has no link in the new session), and a new
    # signature lands in the fresh cache too.
    Enzyme.Compiler.reset_session!()
    @test Enzyme.Compiler.SESSION_EPOCH[] != epoch
    @test isempty(THUNK_CACHE.thunks)
    @test autodiff(Reverse, session_sq, Active(5.0))[1][1] == 10.0
    @test length(THUNK_CACHE.thunks) == 1
    @test autodiff(Reverse, session_sq, Active(5.0f0))[1][1] == 10.0f0
    @test length(THUNK_CACHE.thunks) == 2
end

# Kernel-style signature, as KernelAbstractions' `EnzymeCore08Ext` uses `tape_type(job, …)`.
function session_kernel!(y, x)
    y[1] = 3 * x[1]
    return nothing
end

@testset "Session-scoped tape cache" begin
    Enzyme.Compiler.reset_session!()
    args = (Const{typeof(session_kernel!)}, Const{Nothing}, Duplicated{Vector{Float64}}, Duplicated{Vector{Float64}})
    TT = Enzyme.tape_type(nothing, ReverseSplitWithPrimal, args...)
    @test TT isa Type
    @test length(THUNK_CACHE.tapes) == 1
    @test only(values(THUNK_CACHE.tapes)) === TT
    @test Enzyme.tape_type(nothing, ReverseSplitWithPrimal, args...) === TT
    @test length(THUNK_CACHE.tapes) == 1
    Enzyme.Compiler.reset_session!()
    @test isempty(THUNK_CACHE.tapes)
end
