using Enzyme
using EnzymeTestUtils
using FiniteDifferences
using Test

@testset "BigFloat +/-" begin
    a = rand(BigFloat)
    b = rand(BigFloat)

    # doesn't work because of https://github.com/EnzymeAD/Enzyme.jl/issues/2888
    #test_reverse(+, Const, (a, Const), (b, Const))
    #test_reverse(+, Active, (a, Active), (b, Active))
    #test_reverse(-, Const, (a, Const), (b, Const))
    #test_reverse(-, Active, (a, Active), (b, Active))

    test_forward(+, Const, (a, Const), (b, Const))
    test_forward(+, Duplicated, (a, Duplicated), (b, Duplicated))
    test_forward(-, Const, (a, Const), (b, Const))
    test_forward(-, Duplicated, (a, Duplicated), (b, Duplicated))
end
