using Enzyme
using Enzyme: EnzymeRules
using Test

import EnzymeRules: forward

@testset "Custom Forward Rules" begin

    rule_f(x) = x^2

    function rule_fip(x)
       x[1] *= x[1]
       return nothing
    end

    function forward(::Type{Tuple{typeof(rule_f), Float64}}, rt, args)
        @assert args[1] <: Const

        if rt <: DuplicatedNoNeed && args[2] <: Duplicated
            tmp1(func, x) = 10+2*x.val*x.dval
            return tmp1
        end

        if rt <: BatchDuplicatedNoNeed && args[2] <: BatchDuplicated
            function tmp1b(func, x::BatchDuplicated{T, N}) where {T, N}
                return NTuple{N, T}(1000+2*x.val*dv for dv in x.dval)
            end
            return tmp1b
        end
        if rt <: Duplicated && args[2] <: Duplicated
            tmp2(func, x) = Duplicated(func.val(x.val), 100+2*x.val*x.dval)
            return tmp2
        end
        if rt <: BatchDuplicated && Args[2] <: BatchDuplicated
            function tmp2b(func, x::BatchDuplicated{T, N}) where {T, N}
                return BatchDuplicated(func.val(x.val), NTuple{N, T}(10000+2*x.val*dv for dv in x.dval))
            end
            return tmp2b
        end
        return nothing
    end

    function forward(::Type{Tuple{typeof(rule_fip), T}}, rt, args) where {T}
        @assert args[1] <: Const
        @assert rt <: Const
        if args[2] <: Duplicated
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

    @test autodiff(Forward, rule_f, Duplicated(2.0, 1.0))[1] ≈ 14.0
    @test autodiff(Forward, x->rule_f(x)^2, Duplicated(2.0, 1.0))[1] ≈ 832.0

    res = autodiff(Forward, rule_f, BatchDuplicatedNoNeed, BatchDuplicated(2.0, (1.0, 3.0)))[1] 
    @test res[1] ≈ 1004.0
    @test res[2] ≈ 1012.0

    res = Enzyme.autodiff(Forward, x->rule_f(x)^2, BatchDuplicatedNoNeed, BatchDuplicated(2.0, (1.0, 3.0)))[1]

    @test res[1] ≈ 80032.0
    @test res[2] ≈ 80096.0

    vec = [2.0]
    dvec = [1.0]

    Enzyme.fwddiff(rule_fip, Duplicated(vec, dvec))

    @test vec ≈ [4.0]
    @test dvec ≈ [14.0]
end
