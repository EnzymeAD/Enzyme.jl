using Enzyme
using Enzyme: EnzymeRules
using Test

@testset "Custom Reverse Rules" begin
    rule_f(x) = x^2

    function rule_fip(x)
       x[1] *= x[1]
       return nothing
    end

    function Enzyme.EnzymeRules.reverse(::Type{Tuple{typeof(rule_f), Float64}}, RT, Args, needsPrimal, needsShadow, width, overwritten)
        if width != 1
            return nothing
        end
        @assert Args[1] <: Const
        @assert !needsShadow
        if RT <: Active && Args[2] <: Active && needsPrimal
            tmp1_aug(func, x) = (func.val(x.val),nothing)
            tmp1_rev(func, x, dret, tape) = (10+2*x.val*dret,)
            return tmp1_aug, tmp1_rev, typeof(nothing)
        end
        if RT <: Active && Args[2] <: Active && !needsPrimal
            tmp2_aug(func, x) = (nothing,)
            tmp2_rev(func, x, dret, tape) = (100+2*x.val*dret,)
            return tmp2_aug, tmp2_rev, typeof(nothing)
        end
        return nothing
    end

    function Enzyme.EnzymeRules.reverse(::Type{Tuple{typeof(rule_fip), T}}, RT, Args, needsPrimal, needsShadow, width, overwritten) where {T}
        if width != 1
            return nothing
        end
        @assert Args[1] <: Const
        @assert !needsPrimal
        @assert !needsShadow
        if RT <: Const && Args[2] <: Duplicated
            function tmp1_aug(func, x) 
                v = x.val[1]
                x.val[1] *= v
                return (v,)
            end
            function tmp1_rev(func, x, tape)
                x.dval[1] = 100 + x.dval[1] * tape
                return ()
            end
            return tmp1_aug, tmp1_rev, eltype(T)
        end
        return nothing
    end
    
    @test Enzyme.autodiff(Enzyme.Reverse, rule_f, Active(2.0))[1] ≈ 104.0
    @test Enzyme.autodiff(Enzyme.Reverse, x->rule_f(x)^2, Active(2.0))[1] ≈ 42.0

    x = [2.0]
    dx = [1.0]
    
    Enzyme.autodiff(Enzyme.Reverse, rule_fip, Duplicated(x, dx))
    
    @test x ≈ [4.0]
    @test dx ≈ [102.0]
end
