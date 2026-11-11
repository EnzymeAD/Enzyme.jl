using Enzyme, Test

@noinline function subtape_step!(buf::Vector{Float64}, scale::Float64, m::Int)
    for j in 1:m
        buf[1] = sin(buf[1] * scale + 0.1 * j)
    end
    return nothing
end

function subtape_outer(x::Vector{Float64}, n::Int, m::Int)
    buf = copy(x)
    for i in 1:n
        subtape_step!(buf, x[2] + 0.05 * i, m)
    end
    return buf[1]^2
end

function grad_via_split!(g::Vector{Float64}, x::Vector{Float64}, n::Int, m::Int)
    fill!(g, 0.0)
    fwd_f, rev_f = Enzyme.autodiff_thunk(
        ReverseSplitNoPrimal,
        Const{typeof(subtape_outer)},
        Active,
        Duplicated{Vector{Float64}},
        Const{Int},
        Const{Int},
    )
    tape = fwd_f(Const(subtape_outer), Duplicated(x, g), Const(n), Const(m))[1]
    rev_f(Const(subtape_outer), Duplicated(x, g), Const(n), Const(m), 1.0, tape)
    return nothing
end

struct MockTapeWithLLVMPtr
    subtape::Core.LLVMPtr{Cvoid, 0}
    raw::Core.LLVMPtr{UInt8, 0}
    val::Float64
end

@testset "Untyped Core.LLVMPtr sub-tape activity and 0-field helpers" begin
    @test Enzyme.Compiler.active_reg_nothrow(Core.LLVMPtr{Cvoid, 0}) == Enzyme.Compiler.DupState
    @test Enzyme.Compiler.active_reg_nothrow(Core.LLVMPtr{UInt8, 0}) == Enzyme.Compiler.DupState
    @test !Enzyme.Compiler.guaranteed_const(Core.LLVMPtr{Cvoid, 0})
    @test !Enzyme.Compiler.guaranteed_const(Core.LLVMPtr{UInt8, 0})

    p_void = reinterpret(Core.LLVMPtr{Cvoid, 0}, UInt(0))
    p_u8 = reinterpret(Core.LLVMPtr{UInt8, 0}, UInt(0))
    @test Enzyme.make_zero(p_void) === p_void
    @test Enzyme.make_zero(p_u8) === p_u8
    @test Enzyme.Compiler.recursive_add(p_void, p_void, identity, Enzyme.Compiler.guaranteed_nonactive) === p_void
    @test Enzyme.Compiler.recursive_add(p_u8, p_u8, identity, Enzyme.Compiler.guaranteed_nonactive) === p_u8

    mt = MockTapeWithLLVMPtr(p_void, p_u8, 3.5)
    mt_zero = Enzyme.make_zero(mt)
    @test mt_zero.subtape === p_void
    @test mt_zero.raw === p_u8
    @test mt_zero.val == 0.0

    mt_dest = [MockTapeWithLLVMPtr(p_void, p_u8, 9.0)]
    Enzyme.make_zero!(mt_dest, Base.IdSet{Any}())
    @test mt_dest[1].val == 0.0

    vmt = [mt]
    dvmt = [MockTapeWithLLVMPtr(p_void, p_u8, 2.5)]
    Enzyme.autodiff(
        Reverse,
        v -> Enzyme.make_zero(v)[1].val,
        Active,
        Duplicated(vmt, dvmt),
    )
    @test dvmt[1].val == 2.5

    # Forward-over-Reverse through ReverseSplitNoPrimal whose tape holds Core.LLVMPtr{Cvoid, 0} sub-tapes
    x = [0.7, 1.3]
    v = [0.4, -0.6]
    g = zeros(2)
    hv = zeros(2)
    Enzyme.autodiff(
        Forward,
        grad_via_split!,
        Const,
        Duplicated(g, hv),
        Duplicated(copy(x), copy(v)),
        Const(3),
        Const(2),
    )

    eps = 1.0e-6
    g_p = zeros(2)
    g_m = zeros(2)
    grad_via_split!(g_p, x .+ eps .* v, 3, 2)
    grad_via_split!(g_m, x .- eps .* v, 3, 2)
    fd = (g_p .- g_m) ./ (2 * eps)
    @test hv ≈ fd rtol = 1.0e-5
end
