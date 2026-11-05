using Test
using LinearAlgebra
using EnzymeCore
using EnzymeTestUtils
using EnzymeTestUtils: j′vp
using FiniteDifferences

@testset "j′vp with empty inputs" begin
    # we can make f_vec here identity since it cannot be
    # reached if x is itself empty
    @test isempty(j′vp(FiniteDifferences.central_fdm(5, 1), identity, Float32[], Float32[]))
    # test also the real case
    @test isempty(j′vp(FiniteDifferences.central_fdm(5, 1), identity, Float32[0.26], Float32[]))
    # test also the complex case
    @test isempty(j′vp(FiniteDifferences.central_fdm(5, 1), identity, Float32[0.26, 0.14], Float32[]))
end

@testset "_fd_reverse with Const return aliasing Duplicated argument" begin
    f_alias!(y, x) = (y .+= 2 .* x; y)
    for T in (Float64, ComplexF64)
        y = randn(T, 2, 3)
        x = randn(T, 2, 3)
        dy = randn(T, 2, 3)
        dx = randn(T, 2, 3)
        y_expected = copy(dy)
        x_expected = dx .+ 2 .* dy
        res = EnzymeTestUtils._fd_reverse(
            FiniteDifferences.central_fdm(5, 1),
            (f, args...) -> f(args...),
            EnzymeTestUtils.zero_tangent(y),
            (Const(f_alias!), Duplicated(copy(y), copy(dy)), Duplicated(copy(x), copy(dx))),
            false,
        )
        @test res[2] ≈ y_expected
        @test res[3] ≈ x_expected
    end
end

@testset "test_reverse and zero_tangent allocations (#3314)" begin
    C = randn(Float64, 16, 16)
    A = randn(Float64, 16, 16)
    B = randn(Float64, 16, 16)
    test_reverse(
        LinearAlgebra.mul!,
        Duplicated,
        (C, Duplicated),
        (A, Duplicated),
        (B, Duplicated),
        (1.2, Const),
        (0.3, Const),
    )

    fdm = FiniteDifferences.central_fdm(5, 1)
    ȳ = randn(Float64, 4096)
    y_buf = zeros(Float64, 4096)
    x_fd = randn(Float64, 64)
    f_lin(v) = (copyto!(y_buf, 1, v, 1, length(v)); y_buf)
    j′vp(fdm, f_lin, ȳ, x_fd)
    alloc_jvp = @allocated j′vp(fdm, f_lin, ȳ, x_fd)
    @test alloc_jvp < 256 * 1024

    @static if VERSION >= v"1.11-"
        Cc = randn(ComplexF64, 32, 32)
        EnzymeTestUtils.zero_tangent(Cc.ref.mem)
        alloc_mem = @allocated EnzymeTestUtils.zero_tangent(Cc.ref.mem)
        @test alloc_mem < 128 * 1024
    end
end
