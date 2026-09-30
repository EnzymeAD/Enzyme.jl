using Enzyme, LinearAlgebra, Test
import Enzyme.EnzymeRules

# An aggregate with inline roots is passed to its callee as a pointer to its data half
# plus its roots. `crir_norm` only uses the pointer field, so its body never reads the
# data half, and attribute inference marked that argument `readnone`. The `DSEPass` of
# `middle_optimize!` then removed the caller's store of the bits field into the buffer
# before AD (on LLVM 20, Julia 1.13), and the rule, which reads the whole argument, saw
# uninitialized memory. See also `unread_const_arg.jl`.
struct CRIRTensor
    data::Vector{Float64}
    n::Int
end
crir_norm(A::CRIRTensor) = norm(A.data)

const CRIR_SEEN = Int[]

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1}, ::Const{typeof(crir_norm)}, ::Type{RT},
        A::Annotation{CRIRTensor}
    ) where {RT}
    push!(CRIR_SEEN, A.val.n)
    isa(A, Const) || push!(CRIR_SEEN, A.dval.n)
    ret = crir_norm(A.val)
    primal = EnzymeRules.needs_primal(config) ? ret : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, ret)
end
function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1}, ::Const{typeof(crir_norm)}, dret::Active, n,
        A::Annotation{CRIRTensor}
    )
    push!(CRIR_SEEN, A.val.n)
    if !isa(A, Const)
        push!(CRIR_SEEN, A.dval.n)
        A.dval.data .+= (dret.val / n) .* A.val.data
    end
    return (nothing,)
end
function EnzymeRules.forward(
        config::EnzymeRules.FwdConfigWidth{1}, ::Const{typeof(crir_norm)}, ::Type{RT},
        A::Annotation{CRIRTensor}
    ) where {RT}
    push!(CRIR_SEEN, A.val.n)
    isa(A, Const) || push!(CRIR_SEEN, A.dval.n)
    n = crir_norm(A.val)
    dn = isa(A, Const) ? 0.0 : dot(A.val.data, A.dval.data) / n
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated(n, dn)
    elseif EnzymeRules.needs_primal(config)
        return n
    elseif EnzymeRules.needs_shadow(config)
        return dn
    end
    return nothing
end

crir_outer(A) = crir_norm(A)
crir_scaled(A, x) = crir_norm(A) * x

@testset "Custom rule arguments with inline roots" begin
    A = CRIRTensor([1.0, 2.0, 3.0], 7)
    nA = norm(A.data)

    empty!(CRIR_SEEN)
    dA = CRIRTensor(zeros(3), 7)
    autodiff(Reverse, crir_outer, Active, Duplicated(A, dA))
    @test dA.data ≈ A.data ./ nA
    @test !isempty(CRIR_SEEN) && all(==(7), CRIR_SEEN)

    empty!(CRIR_SEEN)
    @test autodiff(Forward, crir_outer, Duplicated(A, CRIRTensor([1.0, 0.0, 0.0], 7)))[1] ≈ 1 / nA
    @test !isempty(CRIR_SEEN) && all(==(7), CRIR_SEEN)

    empty!(CRIR_SEEN)
    @test autodiff(Forward, crir_norm, Duplicated(A, CRIRTensor([1.0, 0.0, 0.0], 7)))[1] ≈ 1 / nA
    @test !isempty(CRIR_SEEN) && all(==(7), CRIR_SEEN)

    empty!(CRIR_SEEN)
    @test autodiff(Forward, crir_scaled, Const(A), Duplicated(2.0, 1.0))[1] ≈ nA
    @test !isempty(CRIR_SEEN) && all(==(7), CRIR_SEEN)
end
