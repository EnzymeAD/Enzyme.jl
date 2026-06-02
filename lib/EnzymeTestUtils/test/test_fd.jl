using Test
using EnzymeTestUtils
using EnzymeTestUtils: j′vp
using FiniteDifferences

@testset "j′vp with empty inputs" begin
    # we can make f_vec here identity since it cannot be
    # reached if x is itself empty
    @test isempty(j′vp(FiniteDifferences.central_fdm(5, 1), identity, Float32[], Float32[]))
end
