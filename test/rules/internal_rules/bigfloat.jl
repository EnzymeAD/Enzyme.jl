using Enzyme
using EnzymeTestUtils
using FiniteDifferences
using Test

@testset "BigFloat arithmetic" begin
    a = rand(BigFloat)
    b = rand(BigFloat)
    b_int = rand(Int)

    # doesn't work because of https://github.com/EnzymeAD/Enzyme.jl/issues/2888
    #test_reverse(+, Const, (a, Const), (b, Const))
    #test_reverse(+, Active, (a, Active), (b, Active))
    #test_reverse(-, Const, (a, Const), (b, Const))
    #test_reverse(-, Active, (a, Active), (b, Active))

    test_forward(+, Const, (a, Const), (b, Const))
    test_forward(+, Duplicated, (a, Duplicated), (b, Duplicated))
    test_forward(-, Const, (a, Const), (b, Const))
    test_forward(-, Duplicated, (a, Duplicated), (b, Duplicated))
    test_forward(/, Const, (a, Const), (b, Const))
    test_forward(/, Duplicated, (a, Duplicated), (b, Const))
    test_forward(/, Duplicated, (a, Const), (b, Duplicated))
    test_forward(/, Duplicated, (a, Duplicated), (b, Duplicated))
    test_forward(/, Const, (a, Const), (b_int, Const))
    test_forward(/, Duplicated, (a, Duplicated), (b_int, Const))
end
