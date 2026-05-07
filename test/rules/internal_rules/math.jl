using Enzyme
using EnzymeTestUtils
import Random
using Test

@testset "hypot rules" begin
    @testset "forward" begin
        @testset for RT in (Const, DuplicatedNoNeed, Duplicated),
                Tx in (Const, Duplicated),
                Ty in (Const, Duplicated),
                Tz in (Const, Duplicated),
                Txs in (Const, Duplicated)

            x, y, z, xs = 2.0, 3.0, 5.0, 17.0
            test_forward(hypot, RT, (x, Tx), (y, Ty))
            test_forward(hypot, RT, (x, Tx), (y, Ty), (z, Tz))
            test_forward(hypot, RT, (x, Tx), (y, Ty), (z, Tz), (xs, Txs))

            x, y, z, xs = 2.0 + 7.0im, 3.0 + 11.0im, 5.0 + 13.0im, 17.0 + 19.0im
            test_forward(hypot, RT, (x, Tx), (y, Ty))
            test_forward(hypot, RT, (x, Tx), (y, Ty), (z, Tz))
            test_forward(hypot, RT, (x, Tx), (y, Ty), (z, Tz), (xs, Txs))
        end
    end
    @testset "reverse" begin
        @testset for RT in (Active,),
                Tx in (Const, Active),
                Ty in (Const, Active),
                Tz in (Const, Active),
                Txs in (Const, Active)

            x, y, z, xs = 2.0, 3.0, 5.0, 17.0
            test_reverse(hypot, RT, (x, Tx), (y, Ty))
            test_reverse(hypot, RT, (x, Tx), (y, Ty), (z, Tz))
            test_reverse(hypot, RT, (x, Tx), (y, Ty), (z, Tz), (xs, Txs))

            x, y, z, xs = 2.0 + 7.0im, 3.0 + 11.0im, 5.0 + 13.0im, 17.0 + 19.0im
            test_reverse(hypot, RT, (x, Tx), (y, Ty))
            test_reverse(hypot, RT, (x, Tx), (y, Ty), (z, Tz))
            test_reverse(hypot, RT, (x, Tx), (y, Ty), (z, Tz), (xs, Txs))
        end
    end
end
