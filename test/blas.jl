using Enzyme
using EnzymeTestUtils
using LinearAlgebra
using Test

@testset "BLAS rules" begin
    RTs = (Float32, Float64)
    RCs = (ComplexF32, ComplexF64)
    n = 10

    @testset for fun in (BLAS.dot, BLAS.dotu, BLAS.dotc)

        # Wrap `test_reverse` and `test_forward` in `Ref` containers to be able to capture
        # and test the warnings issued by `@generated` functions.  Also, prepare the
        # expected warning message to test.
        testrev = Ref{Any}(test_reverse)
        testfwd = Ref{Any}(test_forward)
        warn_msg = fun === BLAS.dot ? "" : r"Using fallback BLAS replacements"

        @testset "forward" begin
            @testset for Tret in (
                    Const,
                    Duplicated,
                    DuplicatedNoNeed,
                    BatchDuplicated,
                    BatchDuplicatedNoNeed,
                ),
                Tx in (Const, Duplicated, BatchDuplicated),
                Ty in (Const, Duplicated, BatchDuplicated),
                T in (fun == BLAS.dot ? RTs : RCs),
                (sz, inc) in ((10, 1), ((2, 20), -2))

                are_activities_compatible(Tret, Tx, Ty) || continue

                x = randn(T, sz)
                y = randn(T, sz)
                atol = rtol = sqrt(eps(real(T)))
                @test_warn warn_msg testfwd[](fun, Tret, n, (x, Tx), inc, (y, Ty), inc; atol, rtol)
            end
        end

        @testset "reverse" begin
            @testset for Tret in (Const, Active),
                Tx in (Const, Duplicated, BatchDuplicated),
                Ty in (Const, Duplicated, BatchDuplicated),
                T in (fun == BLAS.dot ? RTs : RCs),
                (sz, inc) in ((10, 1), ((2, 20), -2))

                are_activities_compatible(Tret, Tx, Ty) || continue

                x = randn(T, sz)
                y = randn(T, sz)
                atol = rtol = sqrt(eps(real(T)))
                @test_warn warn_msg testrev[](fun, Tret, n, (x, Tx), inc, (y, Ty), inc; atol, rtol)
            end
        end
    end
end

@noinline function my_dotc_test(y)
    n = length(y)
    result = Ref{ComplexF64}()
    ccall((:cblas_zdotc_sub64_, LinearAlgebra.BLAS.libblastrampoline), Nothing,
          (Int64, Ptr{ComplexF64}, Int64, Ptr{ComplexF64}, Int64, Ptr{ComplexF64}),
          n, pointer(y), 1, pointer(y), 1, result)
    return result[]
end

function CallWithKWargs_test(xs)
    s = my_dotc_test(xs)
    return nothing
end

@testset "fallback cblas_zdotc_sub64_ calling convention regression" begin
    n = 3
    T = ComplexF64
    y = randn(T, n)
    cwk = CallWithKWargs_test

    @test_warn r"Using fallback BLAS replacements" autodiff(Forward, Const(cwk), Const, Const(y))
end

