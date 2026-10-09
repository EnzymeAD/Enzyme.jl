module EnzymeDoubleFloatsExt

# A `DoubleFloat{T}` stores a number as the unevaluated sum of two values `hi` and `lo` of
# type `T`.
# Differentiating its arithmetic limb by limb gives wrong derivatives (the renormalization
# steps double-count cotangents), so every operation gets a rule that differentiates the
# operation on the represented number, computed in `DoubleFloat` arithmetic.

using DoubleFloats: DoubleFloats, DoubleFloat
using Enzyme
using Enzyme: EnzymeRules
using Enzyme.EnzymeRules: @easy_rule

function Enzyme.typetree_inner(
        ::Type{DoubleFloat{T}}, ctx, dl, seen::Enzyme.Compiler.TypeTreeTable
    ) where {T}
    return copy(Enzyme.typetree(NTuple{2, T}, ctx, dl, seen))
end
Enzyme.get_offsets(::Type{DoubleFloat{T}}) where {T} = Enzyme.get_offsets(NTuple{2, T})

# DoubleFloats evaluates some functions (e.g. the trigonometric ones) with Quadmath's
# `Float128`, which has no Enzyme type. These calls only occur inside the rules below and
# are never differentiated, so leave their type unknown.
Enzyme.typetree_inner(::Type{DoubleFloats.Float128}, ctx, dl, seen::Enzyme.Compiler.TypeTreeTable) = Enzyme.TypeTree()
Enzyme.get_offsets(::Type{DoubleFloats.Float128}) = ()

# The type annotations are interpolated because `@easy_rule` resolves them in `EnzymeRules`.

# Arithmetic
@eval @easy_rule(+(a::$DoubleFloat, b::$DoubleFloat), (1, 1))
@eval @easy_rule(+(a::$DoubleFloat, b::$Real), (1, 1))
@eval @easy_rule(+(a::$Real, b::$DoubleFloat), (1, 1))
@eval @easy_rule(-(a::$DoubleFloat, b::$DoubleFloat), (1, -1))
@eval @easy_rule(-(a::$DoubleFloat, b::$Real), (1, -1))
@eval @easy_rule(-(a::$Real, b::$DoubleFloat), (1, -1))
@eval @easy_rule(-(a::$DoubleFloat), (-1,))
@eval @easy_rule(+(a::$DoubleFloat), (1,))
@eval @easy_rule(*(a::$DoubleFloat, b::$DoubleFloat), (b, a))
@eval @easy_rule(*(a::$DoubleFloat, b::$Real), (b, a))
@eval @easy_rule(*(a::$Real, b::$DoubleFloat), (b, a))
@eval @easy_rule(/(a::$DoubleFloat, b::$DoubleFloat), (inv(b), -(Ω / b)))
@eval @easy_rule(/(a::$DoubleFloat, b::$Real), (inv(oftype(a, b)), -(Ω / b)))
@eval @easy_rule(/(a::$Real, b::$DoubleFloat), (inv(b), -(Ω / b)))
@eval @easy_rule(Base.inv(a::$DoubleFloat), (-(Ω * Ω),))
@eval @easy_rule(Base.muladd(a::$DoubleFloat, b::$DoubleFloat, c::$DoubleFloat), (b, a, 1))
@eval @easy_rule(Base.abs(a::$DoubleFloat), (sign(a),))
@eval @easy_rule(Base.abs2(a::$DoubleFloat), (2a,))
@eval @easy_rule(Base.:^(a::$DoubleFloat, p::$Integer), (p * a^(p - 1), @Constant))
@eval @easy_rule(Base.sqrt(a::$DoubleFloat), (inv(2Ω),))
@eval @easy_rule(Base.cbrt(a::$DoubleFloat), (Ω / (3a),))

# Exponentials and logarithms
@eval @easy_rule(Base.exp(a::$DoubleFloat), (Ω,))
@eval @easy_rule(Base.exp2(a::$DoubleFloat), (Ω * log(oftype(a, 2)),))
@eval @easy_rule(Base.exp10(a::$DoubleFloat), (Ω * log(oftype(a, 10)),))
@eval @easy_rule(Base.log(a::$DoubleFloat), (inv(a),))
@eval @easy_rule(Base.log2(a::$DoubleFloat), (inv(a * log(oftype(a, 2))),))
@eval @easy_rule(Base.log10(a::$DoubleFloat), (inv(a * log(oftype(a, 10))),))

# Trigonometric and hyperbolic functions
@eval @easy_rule(Base.expm1(a::$DoubleFloat), (Ω + 1,))
@eval @easy_rule(Base.log1p(a::$DoubleFloat), (inv(a + 1),))
@eval @easy_rule(Base.sin(a::$DoubleFloat), (cos(a),))
@eval @easy_rule(Base.cos(a::$DoubleFloat), (-sin(a),))
@eval @easy_rule(Base.tan(a::$DoubleFloat), (1 + Ω * Ω,))
@eval @easy_rule(Base.sinh(a::$DoubleFloat), (cosh(a),))
@eval @easy_rule(Base.cosh(a::$DoubleFloat), (sinh(a),))
@eval @easy_rule(Base.tanh(a::$DoubleFloat), (1 - Ω * Ω,))
@eval @easy_rule(Base.asin(a::$DoubleFloat), (inv(sqrt(1 - a * a)),))
@eval @easy_rule(Base.acos(a::$DoubleFloat), (-inv(sqrt(1 - a * a)),))
@eval @easy_rule(Base.atan(a::$DoubleFloat), (inv(1 + a * a),))

# Conversions between `T` and `DoubleFloat{T}` need no rules. `DoubleFloat{T}(x::T)` sets
# `hi = x, lo = 0` and `T(x::DoubleFloat{T})` returns `hi`; differentiated field by field,
# both are exact to the precision of `T`.

end
