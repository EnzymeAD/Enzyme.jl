using Enzyme
using Enzyme: EnzymeRules
using Test

@testset "Custom Forward Rules" begin

    rule_f(x) = x^2

    function rule_fip(x)
       x[1] *= x[1]
       return nothing
    end

    function Enzyme.EnzymeRules.forward(::Type{Tuple{typeof(rule_f), Float64}}, RT, Args)
        @assert Args[1] <: Const
        if RT <: DuplicatedNoNeed && Args[2] <: Duplicated
            tmp1(func, x) = 10+2*x.val*x.dval
            return tmp1
        end
        if RT <: Enzyme.BatchDuplicatedNoNeed && Args[2] <: Enzyme.BatchDuplicated
            function tmp1b(func, x::Enzyme.BatchDuplicated{T, N}) where {T, N}
                return NTuple{N, T}(1000+2*x.val*dv for dv in x.dval)
            end
            return tmp1b
        end
        if RT <: Duplicated && Args[2] <: Duplicated
            tmp2(func, x) = Enzyme.Duplicated(func.val(x.val), 100+2*x.val*x.dval)
            return tmp2
        end
        if RT <: Enzyme.BatchDuplicated && Args[2] <: Enzyme.BatchDuplicated
            function tmp2b(func, x::Enzyme.BatchDuplicated{T, N}) where {T, N}
                return Enzyme.BatchDuplicated(func.val(x.val), NTuple{N, T}(10000+2*x.val*dv for dv in x.dval))
            end
            return tmp2b
        end
        return nothing
    end

    function Enzyme.EnzymeRules.forward(::Type{Tuple{typeof(rule_fip), T}}, RT, Args) where {T}
        @assert Args[1] <: Const
        @assert RT <: Const
        if Args[2] <: Duplicated
            function tmp1(func, x)
                ld = x.val[1]
                x.val[1] *= ld
                x.dval[1] *= 2 * ld + 10
                nothing
            end
            return tmp1
        end
        return nothing
    end

    @test Enzyme.autodiff(Enzyme.Forward, rule_f, Duplicated(2.0, 1.0))[1] ≈ 14.0
    @test Enzyme.autodiff(Enzyme.Forward, x->rule_f(x)^2, Duplicated(2.0, 1.0))[1] ≈ 832.0

    res = Enzyme.autodiff(Enzyme.Forward, rule_f, BatchDuplicatedNoNeed, BatchDuplicated(2.0, (1.0, 3.0)))[1] 
    @test res[1] ≈ 1004.0
    @test res[2] ≈ 1012.0

    res = Enzyme.autodiff(Enzyme.Forward, x->rule_f(x)^2, BatchDuplicatedNoNeed, BatchDuplicated(2.0, (1.0, 3.0)))[1]

    @test res[1] ≈ 80032.0
    @test res[2] ≈ 80096.0

    vec = [2.0]
    dvec = [1.0]

    Enzyme.fwddiff(rule_fip, Duplicated(vec, dvec))

    @test vec ≈ [4.0]
    @test dvec ≈ [14.0]

end
