module EnzymeMultiFloatsExt

# A `MultiFloat{T,N}` stores a number as the unevaluated sum of `N` limbs of type `T`.
# Differentiating its arithmetic limb by limb gives wrong derivatives (the renormalization
# steps double-count cotangents), so every operation gets a rule that differentiates the
# operation on the represented number, computed in `MultiFloat` arithmetic.

using MultiFloats: MultiFloat
using Enzyme
using Enzyme: EnzymeRules
using Enzyme.EnzymeRules: @easy_rule

function Enzyme.typetree_inner(
        ::Type{MultiFloat{T, N}}, ctx, dl, seen::Enzyme.Compiler.TypeTreeTable
    ) where {T, N}
    return copy(Enzyme.typetree(NTuple{N, T}, ctx, dl, seen))
end
Enzyme.get_offsets(::Type{MultiFloat{T, N}}) where {T, N} = Enzyme.get_offsets(NTuple{N, T})

# The type annotations are interpolated because `@easy_rule` resolves them in `EnzymeRules`.

# Arithmetic
@eval @easy_rule(+(a::$MultiFloat, b::$MultiFloat), (1, 1))
@eval @easy_rule(+(a::$MultiFloat, b::$Real), (1, 1))
@eval @easy_rule(+(a::$Real, b::$MultiFloat), (1, 1))
@eval @easy_rule(-(a::$MultiFloat, b::$MultiFloat), (1, -1))
@eval @easy_rule(-(a::$MultiFloat, b::$Real), (1, -1))
@eval @easy_rule(-(a::$Real, b::$MultiFloat), (1, -1))
@eval @easy_rule(-(a::$MultiFloat), (-1,))
@eval @easy_rule(+(a::$MultiFloat), (1,))
@eval @easy_rule(*(a::$MultiFloat, b::$MultiFloat), (b, a))
@eval @easy_rule(*(a::$MultiFloat, b::$Real), (b, a))
@eval @easy_rule(*(a::$Real, b::$MultiFloat), (b, a))
@eval @easy_rule(/(a::$MultiFloat, b::$MultiFloat), (inv(b), -(Ω / b)))
@eval @easy_rule(/(a::$MultiFloat, b::$Real), (inv(oftype(a, b)), -(Ω / b)))
@eval @easy_rule(/(a::$Real, b::$MultiFloat), (inv(b), -(Ω / b)))
@eval @easy_rule(Base.inv(a::$MultiFloat), (-(Ω * Ω),))
@eval @easy_rule(Base.muladd(a::$MultiFloat, b::$MultiFloat, c::$MultiFloat), (b, a, 1))
@eval @easy_rule(Base.abs(a::$MultiFloat), (sign(a),))
@eval @easy_rule(Base.abs2(a::$MultiFloat), (2a,))
@eval @easy_rule(Base.:^(a::$MultiFloat, p::$Integer), (p * a^(p - 1), @Constant))
@eval @easy_rule(Base.sqrt(a::$MultiFloat), (inv(2Ω),))
@eval @easy_rule(Base.cbrt(a::$MultiFloat), (Ω / (3a),))

# Exponentials and logarithms (implemented natively by MultiFloats)
@eval @easy_rule(Base.exp(a::$MultiFloat), (Ω,))
@eval @easy_rule(Base.exp2(a::$MultiFloat), (Ω * log(oftype(a, 2)),))
@eval @easy_rule(Base.exp10(a::$MultiFloat), (Ω * log(oftype(a, 10)),))
@eval @easy_rule(Base.log(a::$MultiFloat), (inv(a),))
@eval @easy_rule(Base.log2(a::$MultiFloat), (inv(a * log(oftype(a, 2))),))
@eval @easy_rule(Base.log10(a::$MultiFloat), (inv(a * log(oftype(a, 10))),))

# Functions MultiFloats evaluates with BigFloat after `use_bigfloat_transcendentals()`
@eval @easy_rule(Base.expm1(a::$MultiFloat), (Ω + 1,))
@eval @easy_rule(Base.log1p(a::$MultiFloat), (inv(a + 1),))
@eval @easy_rule(Base.sin(a::$MultiFloat), (cos(a),))
@eval @easy_rule(Base.cos(a::$MultiFloat), (-sin(a),))
@eval @easy_rule(Base.tan(a::$MultiFloat), (1 + Ω * Ω,))
@eval @easy_rule(Base.sinh(a::$MultiFloat), (cosh(a),))
@eval @easy_rule(Base.cosh(a::$MultiFloat), (sinh(a),))
@eval @easy_rule(Base.tanh(a::$MultiFloat), (1 - Ω * Ω,))
@eval @easy_rule(Base.asin(a::$MultiFloat), (inv(sqrt(1 - a * a)),))
@eval @easy_rule(Base.acos(a::$MultiFloat), (-inv(sqrt(1 - a * a)),))
@eval @easy_rule(Base.atan(a::$MultiFloat), (inv(1 + a * a),))

# Conversions between a limb and a MultiFloat need no rules. `MultiFloat{T,N}(x::T)` puts
# `x` into the first limb and `T(x::MultiFloat)` returns the first limb; differentiated limb
# by limb, both are exact to the precision of `T`.

end
