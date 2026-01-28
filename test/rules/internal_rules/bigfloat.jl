using Enzyme
using EnzymeTestUtils
using FiniteDifferences
using Test

@testset "BigFloat arithmetic" begin
    a = rand(BigFloat)
    b = rand(BigFloat)
    bf64 = rand(Float64) # for testing mixed methods

    # doesn't work because of https://github.com/EnzymeAD/Enzyme.jl/issues/2888
    #test_reverse(+, Const, (a, Const), (b, Const))
    #test_reverse(+, Active, (a, Active), (b, Active))
    #test_reverse(-, Const, (a, Const), (b, Const))
    #test_reverse(-, Active, (a, Active), (b, Active))

    for TR in (Const, Duplicated), TA in (Const, Duplicated), TB in (Const, Duplicated)
        test_forward(+, TR, (a, TA), (b, TB))
        test_forward(-, TR, (a, TA), (b, TB))
        test_forward(*, TR, (a, TA), (b, TB))
        test_forward(/, TR, (a, TA), (b, TB))
        test_forward(+, TR, (a, TA), (bf64, TB))
        test_forward(-, TR, (a, TA), (bf64, TB))
        test_forward(*, TR, (a, TA), (bf64, TB))
        test_forward(/, TR, (a, TA), (bf64, TB))
    end
    for TR in (Const, Duplicated), TA in (Const, Duplicated)
        test_forward(inv, TR, (a, TA))
    end
end
