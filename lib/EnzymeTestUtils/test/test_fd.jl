using Test
using EnzymeTestUtils
using EnzymeTestUtils: j′vp
using FiniteDifferences
using JLArrays

@testset "j′vp with empty inputs" begin
    # we can make f_vec here identity since it cannot be
    # reached if x is itself empty
    @test isempty(j′vp(FiniteDifferences.central_fdm(5, 1), identity, Float32[], Float32[]))
    # test also the real case
    @test isempty(j′vp(FiniteDifferences.central_fdm(5, 1), identity, Float32[0.26], Float32[]))
    # test also the complex case
    @test isempty(j′vp(FiniteDifferences.central_fdm(5, 1), identity, Float32[0.26, 0.14], Float32[]))
end

@testset "j′vp with device-backed vectors" begin
    # passing `output_tangent` to `test_reverse` is the only way the cotangent reaches
    # here already on the device. Both it and `x` are read one element at a time, which
    # GPU arrays disallow, so they have to be pulled to the host first. `JLArray` reaches
    # that path without a GPU. Note the end-to-end `test_reverse` cannot be used here:
    # Enzyme cannot differentiate `JLArray` kernels, as KernelAbstractions' Enzyme
    # extension has no `mkcontext` for `JLBackend`.
    fdm = FiniteDifferences.central_fdm(5, 1)
    x = JLArray(randn(3))
    ȳ = JLArray(randn(3))
    # mirrors the real `f_vec`, which is handed a host vector and returns a device vector
    f_vec(v) = 2 .* JLArray(collect(v))

    res = j′vp(fdm, f_vec, ȳ, x)
    # the jacobian of `2 .* x` is `2I`, so the pullback is just `2ȳ`
    @test res isa JLVector{Float64}
    @test Array(res) ≈ 2 .* Array(ȳ)

    # a host cotangent against device inputs must work too, since which side is
    # device-backed depends on the activities under test
    res_host = j′vp(fdm, f_vec, Array(ȳ), x)
    @test Array(res_host) ≈ 2 .* Array(ȳ)
end
