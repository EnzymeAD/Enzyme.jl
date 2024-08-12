module RuleInvalidation

using Test
using Enzyme
import .EnzymeRules: forward, inactive

issue696(x) = x^2
call_issue696(args...) = issue696(args...)

@test autodiff(Forward, issue696, Duplicated(1.0, 1.0))[1] ≈ 2.0
@test autodiff(Forward, call_issue696, Duplicated(1.0, 1.0))[1] ≈ 2.0

# should invalidate cache for the previous result
forward(::Const{typeof(issue696)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated) =
    10+2*x.val*x.dval
forward(func::Const{typeof(issue696)}, ::Type{<:Duplicated}, x::Duplicated) =
    Duplicated(func.val(x.val), 10+2*x.val*x.dval)

@test autodiff(Forward, issue696, Duplicated(1.0, 1.0))[1] ≈ 12.0
@test autodiff(Forward, call_issue696, Duplicated(1.0, 1.0))[1] ≈ 12.0

# should invalidate cache for the previous result again
forward(::Const{typeof(issue696)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated) =
    20+2*x.val*x.dval
forward(func::Const{typeof(issue696)}, ::Type{<:Duplicated}, x::Duplicated) =
    Duplicated(func.val(x.val), 20+2*x.val*x.dval)

@test autodiff(Forward, issue696, Duplicated(1.0, 1.0))[1] ≈ 22.0
@test autodiff(Forward, call_issue696, Duplicated(1.0, 1.0))[1] ≈ 22.0

# check that `Base.delete_method` works as expected
for m in methods(forward, Tuple{Const{typeof(issue696)},Vararg{Any}})
    Base.delete_method(m)
end
@test autodiff(Forward, issue696, Duplicated(1.0, 1.0))[1] ≈ 2.0
@test autodiff(Forward, call_issue696, Duplicated(1.0, 1.0))[1] ≈ 2.0

# now test invalidation for `inactive`
inactive(::typeof(issue696), args...) = nothing
@test autodiff(Forward, issue696, Duplicated(1.0, 1.0))[1] ≈ 0.0
@test autodiff(Forward, call_issue696, Duplicated(1.0, 1.0))[1] ≈ 0.0

end # module
