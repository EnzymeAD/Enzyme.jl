using Test
using EnzymeCore

import EnzymeCore.EnzymeRules: forward, has_frule_from_sig

g(x) = x ^ 2
function forward(::Const{typeof(g)}, ::Type{<:Const}, x::Const)
    return Const(g(x.val))
end

@test has_frule_from_sig(Base.signature_type(g, Tuple{Float64}))

f(;kwargs) = 1.0

function forward(::Const{typeof(f)}, ::Type{<:Const}; kwargs...)
    return Const(f(; kwargs...))
end

@test has_frule_from_sig(Base.signature_type(f, Tuple{}))

# Check loading of extension
if isdefined(Base, :get_extension)
    @test Base.get_extension(EnzymeCore, :EnzymeCoreAdaptExt) === nothing
end
using Adapt
EnzymeCoreAdaptExt = if isdefined(Base, :get_extension)
    Base.get_extension(EnzymeCore, :EnzymeCoreAdaptExt)
else
    EnzymeCore.EnzymeCoreAdaptExt
end
@test EnzymeCoreAdaptExt isa Module

# Test `adapt` with example from Adapt docs
struct IntegerLessAdaptor end
Adapt.adapt_storage(::IntegerLessAdaptor, x::Int) = Float64(x)

@test adapt(IntegerLessAdaptor(), Const(1)) === Const(1.0)
@test adapt(IntegerLessAdaptor(), Active(2)) === Active(2.0)
@test adapt(IntegerLessAdaptor(), Duplicated(3, 4)) === Duplicated(3.0, 4.0)
@test adapt(IntegerLessAdaptor(), DuplicatedNoNeed(5, 6)) === DuplicatedNoNeed(5.0, 6.0)
@test adapt(IntegerLessAdaptor(), BatchDuplicated(7, (8, 9))) === BatchDuplicated(7.0, (8.0, 9.0))
@test adapt(IntegerLessAdaptor(), BatchDuplicatedNoNeed(10, (11, 12))) === BatchDuplicatedNoNeed(10.0, (11.0, 12.0))
