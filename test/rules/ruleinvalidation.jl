module RuleInvalidation

using Test
using Enzyme
import .EnzymeRules: forward, inactive

issue696(x) = x^2
call_issue696(args...) = issue696(args...)

@test autodiff(Forward, issue696, Duplicated(1.0, 1.0))[1] ≈ 2.0
@test autodiff(Forward, call_issue696, Duplicated(1.0, 1.0))[1] ≈ 2.0

# should invalidate cache for the previous result
forward(config, ::Const{typeof(issue696)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated) =
    10 + 2 * x.val * x.dval
forward(config, func::Const{typeof(issue696)}, ::Type{<:Duplicated}, x::Duplicated) =
    Duplicated(func.val(x.val), 10 + 2 * x.val * x.dval)

@test autodiff(Forward, issue696, Duplicated(1.0, 1.0))[1] ≈ 12.0
@test autodiff(Forward, call_issue696, Duplicated(1.0, 1.0))[1] ≈ 12.0

# should invalidate cache for the previous result again
forward(config, ::Const{typeof(issue696)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated) =
    20 + 2 * x.val * x.dval
forward(config, func::Const{typeof(issue696)}, ::Type{<:Duplicated}, x::Duplicated) =
    Duplicated(func.val(x.val), 20 + 2 * x.val * x.dval)

@test autodiff(Forward, issue696, Duplicated(1.0, 1.0))[1] ≈ 22.0
@test autodiff(Forward, call_issue696, Duplicated(1.0, 1.0))[1] ≈ 22.0

# check that `Base.delete_method` works as expected
# Loop required because on 1.12 delete_method uncovers the previous method
while !isempty(methods(forward, Tuple{Any, Const{typeof(issue696)}, Vararg{Any}}))
    for m in methods(forward, Tuple{Any, Const{typeof(issue696)}, Vararg{Any}})
        Base.delete_method(m)
    end
end
@test autodiff(Forward, issue696, Duplicated(1.0, 1.0))[1] ≈ 2.0
@test autodiff(Forward, call_issue696, Duplicated(1.0, 1.0))[1] ≈ 2.0

# now test invalidation for `inactive`
inactive(::typeof(issue696), args...) = nothing
@test autodiff(Forward, issue696, Duplicated(1.0, 1.0))[1] ≈ 0.0
@test autodiff(Forward, call_issue696, Duplicated(1.0, 1.0))[1] ≈ 0.0

# check that `Base.delete_method` works as expected
for m in methods(inactive, Tuple{typeof(issue696), Vararg{Any}})
    Base.delete_method(m)
end

@test_broken autodiff(Forward, issue696, Duplicated(1.0, 1.0))[1] ≈ 2.0
@test_broken autodiff(Forward, call_issue696, Duplicated(1.0, 1.0))[1] ≈ 2.0

end # module
