module EnzymeSpecialFunctionsExt

using SpecialFunctions
using Enzyme

function __init__()
    Enzyme.Compiler.known_ops[typeof(SpecialFunctions._logabsgamma)] = (:logabsgamma, 1, (:digamma, typeof(SpecialFunctions.digamma)))
    Enzyme.Compiler.cmplx_known_ops[typeof(SpecialFunctions.bessely)] = (:cmplx_yn, 2, nothing)
    Enzyme.Compiler.cmplx_known_ops[typeof(SpecialFunctions.besseli)] = (:cmplx_in, 2, nothing)
    Enzyme.Compiler.cmplx_known_ops[typeof(SpecialFunctions.besselj)] = (:cmplx_jn, 2, nothing)
    Enzyme.Compiler.cmplx_known_ops[typeof(SpecialFunctions.besselk)] = (:cmplx_kn, 2, nothing)
end

# libEnzyme's tgamma rule calls an external digamma and omits the gamma factor.
# Use Julia's digamma so the derivative also works without a loggamma call nearby.
EnzymeRules.@easy_rule(
    SpecialFunctions.gamma(x::Real),
    (Ω * SpecialFunctions.digamma(x),),
)

# The reflection formulas for digamma and polygamma contain inv(tanpi(x)).
# At half-integers tanpi(x) is infinite, so differentiating that intermediate
# gives NaN although these functions and all their derivatives are finite.
# Keep nested differentiation on the analytic polygamma identities as well.
EnzymeRules.@easy_rule(SpecialFunctions.digamma(x::Real), (SpecialFunctions.trigamma(x),))
EnzymeRules.@easy_rule(SpecialFunctions.trigamma(x::Real), (SpecialFunctions.polygamma(2, x),))
EnzymeRules.@easy_rule(
    SpecialFunctions.polygamma(n::Integer, x::Real),
    @setup(n < 0 && SpecialFunctions.polygamma(n, x)),
    (@Constant, SpecialFunctions.polygamma(n + 1, x)),
)

# Exponentially scaled Bessel functions (x/ref: https://github.com/EnzymeAD/Enzyme.jl/issues/2880)
# besselix(nu, x) = besseli(nu, x) * exp(-abs(x)) for real x, hence the extra -sign(x)*Ω term.
EnzymeRules.@easy_rule(
    SpecialFunctions.besselix(nu::Real, x::Real),
    (@Constant, (SpecialFunctions.besselix(nu - 1, x) + SpecialFunctions.besselix(nu + 1, x)) / 2 - sign(x) * Ω),
)

# besseljx(nu, x) = besselj(nu, x) * exp(-abs(imag(x))), so the scaling is constant for real x.
EnzymeRules.@easy_rule(
    SpecialFunctions.besseljx(nu::Real, x::Real),
    (@Constant, (SpecialFunctions.besseljx(nu - 1, x) - SpecialFunctions.besseljx(nu + 1, x)) / 2),
)

# besselyx(nu, x) = bessely(nu, x) * exp(-abs(imag(x))), so the scaling is constant for real x.
EnzymeRules.@easy_rule(
    SpecialFunctions.besselyx(nu::Real, x::Real),
    (@Constant, (SpecialFunctions.besselyx(nu - 1, x) - SpecialFunctions.besselyx(nu + 1, x)) / 2),
)

# besselkx(nu, x) = besselk(nu, x) * exp(x), hence the extra +Ω term.
EnzymeRules.@easy_rule(
    SpecialFunctions.besselkx(nu::Real, x::Real),
    (@Constant, Ω - (SpecialFunctions.besselkx(nu - 1, x) + SpecialFunctions.besselkx(nu + 1, x)) / 2),
)

## Scaled complementary error function: derivatives of every order
#
# erfcx(x) = exp(x^2) erfc(x), and its derivatives y_k = erfcx^(k)(x) satisfy
#
#     y_1 = 2x y_0 - 2/sqrt(pi),    y_{k+1} = 2x y_k + 2k y_{k-1}  (k >= 1).
#
# libEnzyme's built-in rule for `Faddeeva_erfcx_re` is the first identity, and a higher
# order differentiates it again. For x > 0 its two terms nearly cancel, because
# x erfcx(x) -> 1/sqrt(pi): the first derivative loses about log10(2x^2) digits, and every
# further order loses as many again. At x = exp(20) not one digit of the first derivative
# is right, and the second is wrong by a factor of 1e17.
#
# Where every term of the recurrence has one sign (x <= 0), or 2x^2 is small (x <= 1),
# it loses at most a few ulps and is used as it stands. Elsewhere the ratios
# p_k = -y_k / (2 y_{k-1}) satisfy p_k = (k/2) / (x + p_{k+1}), with every term
# positive: p_n is Laplace's continued fraction, an S-fraction that converges for every
# x > 0, and y_n = erfcx(x) (-2 p_1) ... (-2 p_n) involves no subtraction at all.
#
# y_n falls like x^-(n+1), so it leaves Float64's range long before the product it is
# needed for does. In log(erfcx(exp(x))) the partial erfcx'(exp(x)) is below the smallest
# subnormal from x = 372 on, while the cotangent 1/erfcx(exp(x)) that multiplies it is
# just as large as the partial is small, and their product is about -exp(-x). A rule that
# returns y_n for Enzyme to multiply by the seed therefore gives a zero gradient there, and
# Hessians and third derivatives that are off by O(1) or NaN much earlier (measured from
# x = 237 and x = 170). So the rules never form y_n alone: their partial is a
# `_ErfcxTimes`, and multiplying it by a seed (which `@easy_rule` does through
# `multiply_fwd_into` and `multiply_rev_into`) evaluates s 2^e y_n with `_erfcx_seeded`,
# which keeps every factor's binary exponent apart from its significand until the end. The seeds a nested mode multiplies in are folded into
# s and e the same way, by `_scale2`, so a product of several large seeds cannot overflow.
# `_erfcx_seeded` has a rule of its own, giving order n + 1, so a nested mode of any
# order reaches this evaluation rather than differentiating it.

# Widen, compute, narrow, as `erfcx` itself does for Float16 and Float32.
@inline _erfcx_work(x::Union{Float16, Float32}) = Float64(x)
@inline _erfcx_work(x::AbstractFloat) = x

# erfcx(x) as (m, e) with erfcx(x) = m 2^e. erfcx(x) is below floatmin only for
# x > 2.5e307, where SpecialFunctions returns 1 / (sqrt(pi) x) rounded to a subnormal;
# that formula is exact to Float64 precision there, so it is evaluated without the
# rounding.
@inline function _erfcx_frexp(x::T, y0::T) where {T <: AbstractFloat}
    if x > 0 && isfinite(x) && y0 < floatmin(T)
        (mx, ex) = frexp(x)
        (m, e) = frexp(inv(sqrt(T(π))) / mx)
        return (m, e - ex)
    end
    return frexp(y0)
end

@inline _isnormal(v::AbstractFloat) = isfinite(v) && abs(v) >= floatmin(v)

# (m, e) times a / b, with a / b formed from b's significand.
@inline function _frexp_times_ratio(m::T, e::Int, a::T, b::T) where {T <: AbstractFloat}
    (mb, eb) = frexp(b)
    (m2, e2) = frexp(m * (a / mb))
    return (m2, e + e2 - eb)
end

# erfcx^(n)(x) as (m, e) with erfcx^(n)(x) = m 2^e.
function _erfcx_derivative_frexp(n::Int, x::T) where {T <: AbstractFloat}
    n >= 0 || throw(DomainError(n, "the order of a derivative of erfcx must be nonnegative"))
    y0 = SpecialFunctions.erfcx(x)
    (n == 0 || isnan(x)) && return _erfcx_frexp(x, y0)
    # The first step's relative error is about (2x y_0 + 2/sqrt(pi)) / |y_1| ulps, 20 at
    # x = 2, so a first derivative takes the recurrence that far; each later step can
    # multiply the error by 2x^2 / (k + 1), so higher orders stop at x = 1. Every y_k is
    # within Float64's range there, unless erfcx(x) itself overflows (x < -26.6).
    if x <= (n == 1 ? T(2) : one(T))
        two_x = 2x
        prev = y0
        y = muladd(two_x, y0, -2 / sqrt(T(π)))
        for k in 1:(n - 1)
            (prev, y) = (y, muladd(two_x, y, 2k * prev))
        end
        return frexp(y)
    end
    isinf(x) && return (zero(T), 0)
    # p_n = (n/2) / f with f = x + a_1 / (x + a_2 / (x + ...)), a_j = (n + j)/2. The tail
    # changes f by less than a_1 / x^2 relative to x, under eps/8 once x^2 eps > 4(n + 1),
    # and there f is x. Below that, the modified Lentz method: its tail contracts by about
    # exp(-2 sqrt(2) x (sqrt(j) - sqrt(n))) after j terms, so at most a few hundred are
    # needed for Float64 at x > 1. The cap allows four times that bound and only guards
    # against a defect.
    if x * x * eps(T) > 4 * (n + 1)
        f = x
    else
        maxiter = 100 + 4 * ceil(Int, (sqrt(T(n)) + (precision(T) * log(T(2)) + 5) / (2 * sqrt(T(2)) * x))^2)
        f = x
        C = x
        D = zero(T)
        j = 0
        while true
            j += 1
            j > maxiter && error("erfcx derivative of order $n at $x: continued fraction did not converge in $maxiter terms")
            a = T(n + j) / 2
            D = inv(x + a * D)
            C = x + a / C
            Δ = C * D
            f *= Δ
            abs(Δ - one(T)) <= eps(T) && break
        end
    end
    # y_n = y_0 (-n / f) (-(n - 1) / (x + p_n)) ... (-1 / (x + p_2)), as -2 p_k = -k / (x + p_{k+1}).
    # The factors shrink with k, so the partial products rise and then fall: where y_0 and
    # y_n are normal numbers, so is every partial product, and the plain product has the
    # same bits as the scaled one below.
    y = y0 * (-T(n) / f)
    p = T(n) / 2 / f
    for k in (n - 1):-1:1
        den = x + p
        p = T(k) / 2 / den
        y *= -T(k) / den
    end
    _isnormal(y0) && _isnormal(y) && return frexp(y)
    (m, e) = _erfcx_frexp(x, y0)
    (m, e) = _frexp_times_ratio(m, e, -T(n), f)
    p = T(n) / 2 / f
    for k in (n - 1):-1:1
        den = x + p
        p = T(k) / 2 / den
        (m, e) = _frexp_times_ratio(m, e, -T(k), den)
    end
    return (m, e)
end

# s 2^e erfcx^(n)(x), rounded once at the end: the result is exact to Float64 precision
# whenever it is a normal number, however far outside the range y_n and s 2^e are.
function _erfcx_seeded(x::Real, n::Int, s::Real, e::Int)
    T = float(promote_type(typeof(x), typeof(s)))
    return T(_erfcx_seeded_work(_erfcx_work(T(x)), n, _erfcx_work(T(s)), e))
end

function _erfcx_seeded_work(x::T, n::Int, s::T, e::Int) where {T <: AbstractFloat}
    (m, k) = _erfcx_derivative_frexp(n, x)
    (ms, ks) = frexp(s)
    return ldexp(m * ms, k + ks + e)
end

# v 2^k, with a rule that applies 2^k to the seed rather than forming it (2^k overflows
# for the k that normalizes a subnormal seed).
@inline _scale2(v::Real, k::Int) = ldexp(float(v), k)

# The binary exponent that `_scale2` normalizes a seed with; 0 where there is none.
@inline _exponent2(v::Real) = (iszero(v) || !isfinite(v)) ? 0 : exponent(v)
EnzymeRules.inactive(::typeof(_exponent2), args...) = true

# The partial of s 2^e erfcx^(n)(x) with respect to x, so s 2^e erfcx^(n+1)(x); for
# erfcx(x) itself n = 0, s = 1 and e = 0.
struct _ErfcxTimes{T, S}
    x::T
    n::Int
    s::S
    e::Int
end

# The partial of s 2^e erfcx^(n)(x) with respect to s, so 2^e erfcx^(n)(x).
struct _ErfcxScaled{T}
    x::T
    n::Int
    e::Int
end

# The partial of v 2^k with respect to v.
struct _Scale2Times
    k::Int
end

# A seed whose product with another such seed is a normal number (or zero).
@inline _moderate(v::AbstractFloat) = iszero(v) || (sqrt(floatmin(v)) <= abs(v) <= sqrt(floatmax(v)))
@inline _moderate(v::Real) = _moderate(float(v))

# Each partial times a seed d. Seeds of moderate size multiply as they are; otherwise the
# significands of s and d multiply to a number in [1, 4), and their exponents go to e.
@inline function _times_seed(p::_ErfcxTimes, d::Real)
    _moderate(p.s) && _moderate(d) && return _erfcx_seeded(p.x, p.n + 1, p.s * d, p.e)
    ks = _exponent2(p.s)
    kd = _exponent2(d)
    s = _scale2(p.s, -ks) * _scale2(d, -kd)
    return _erfcx_seeded(p.x, p.n + 1, s, p.e + ks + kd)
end
@inline _times_seed(p::_ErfcxScaled, d::Real) = _erfcx_seeded(p.x, p.n, d, p.e)
@inline _times_seed(p::_Scale2Times, d::Real) = _scale2(d, p.k)

# `@easy_rule` starts each sum from the type of its result.
@inline _add_into(prev::Type, v) = prev(v)
@inline _add_into(prev, v) = prev + v

for P in (:_ErfcxTimes, :_ErfcxScaled, :_Scale2Times)
    @eval begin
        @inline EnzymeRules.multiply_fwd_into(prev, p::$P, d::Real) = _add_into(prev, _times_seed(p, d))
        @inline EnzymeRules.multiply_rev_into(prev, p::$P, d::Real) = _add_into(prev, _times_seed(p, d))
    end
end

EnzymeRules.@easy_rule(
    SpecialFunctions.erfcx(x::Real),
    (_ErfcxTimes(x, 0, one(x), 0),),
)

EnzymeRules.@easy_rule(
    _erfcx_seeded(x::Real, n::Int, s::Real, e::Int),
    (_ErfcxTimes(x, n, s, e), @Constant, _ErfcxScaled(x, n, e), @Constant),
)

EnzymeRules.@easy_rule(
    _scale2(v::Real, k::Int),
    (_Scale2Times(k), @Constant),
)

# x/ref: https://github.com/JuliaMath/SpecialFunctions.jl/pull/506
## Incomplete beta derivatives via Boik & Robinson-Cox
#
# Reference
#   R. J. Boik and J. F. Robinson-Cox (1999).
#   "Derivatives of the incomplete beta function."
#   Journal of Statistical Software, 3(1).
#   URL: https://www.jstatsoft.org/article/view/v003i01
#
# The following implementation computes the regularized incomplete beta
# I_x(a,b) together with its partial derivatives with respect to a, b, and x
# using a continued-fraction representation of ₂F₁ and differentiating through it.
# This is an independent implementation adapted from https://github.com/arzwa/IncBetaDer.jl.

# Generic-typed helpers used by the continued-fraction evaluation of I_x(a,b)
# and its partial derivatives. These implement the scalar prefactor K(x;p,q),
# the auxiliary variable f, the continued-fraction coefficients a_n, b_n, and
# their partial derivatives w.r.t. p (≡ a) and q (≡ b). See Boik & Robinson-Cox (1999).

function _Kfun(x::T, p::T, q::T) where {T<:AbstractFloat}
    # K(x;p,q) = x^p (1-x)^{q-1} / (p * B(p,q)) computed in log-space for stability
    return exp(p * log(x) + (q - 1) * log1p(-x) - log(p) - logbeta(p, q))
end

function _ffun(x::T, p::T, q::T) where {T<:AbstractFloat}
    # f = q x / (p (1-x)) — convenience variable appearing in CF coefficients
    return q * x / (p * (1 - x))
end

function _a1fun(p::T, q::T, f::T) where {T<:AbstractFloat} 
    # a₁ coefficient of the continued fraction for ₂F₁ representation
    return p * f * (q - 1) / (q * (p + 1))
end

function _anfun(p::T, q::T, f::T, n::Int) where {T<:AbstractFloat}
    # a_n coefficient (n ≥ 1) of the continued fraction for ₂F₁ in terms of p=a, q=b, f.
    # For n=1, falls back to a₁; for n≥2 uses the closed-form product from the Gauss CF.
    n == 1 && return _a1fun(p, q, f)
    return p^2 * f^2 * (n - 1) * (p + q + n - 2) * (p + n - 1) * (q - n) /
           (q^2 * (p + 2*n - 3) * (p + 2*n - 2)^2 * (p + 2*n - 1))
end

function _bnfun(p::T, q::T, f::T, n::Int) where {T<:AbstractFloat}
    # b_n coefficient (n ≥ 1) of the continued fraction. Derived for the same CF.
    x = 2 * (p * f + 2 * q) * n^2 + 2 * (p * f + 2 * q) * (p - 1) * n + p * q * (p - 2 - p * f)
    y = q * (p + 2*n - 2) * (p + 2*n)
    return x / y
end

function _dK_dp(x::T, p::T, q::T, K::T, ψpq::T, ψp::T) where {T<:AbstractFloat} 
    # ∂K/∂p using digamma identities: d/dp log B(p,q) = ψ(p) - ψ(p+q)
    return K * (log(x) - inv(p) + ψpq - ψp)
end

function _dK_dq(x::T, p::T, q::T, K::T, ψpq::T, ψq::T) where {T<:AbstractFloat} 
    # ∂K/∂q using identical pattern
    K * (log1p(-x) + ψpq - ψq)
end

function _dK_dpdq(x::T, p::T, q::T) where {T<:AbstractFloat}
    # Convenience: compute (∂K/∂p, ∂K/∂q) together with shared ψ(p+q)
    ψ = digamma(p + q)
    Kf = _Kfun(x, p, q)
    dKdp = _dK_dp(x, p, q, Kf, ψ, digamma(p))
    dKdq = _dK_dq(x, p, q, Kf, ψ, digamma(q))
    return dKdp, dKdq
end

function _da1_dp(p::T, q::T, f::T) where {T<:AbstractFloat}
    # ∂a₁/∂p from the closed form of a₁
    return - _a1fun(p, q, f) / (p + 1)
end

function _dan_dp(p::T, q::T, f::T, n::Int) where {T<:AbstractFloat}
    # ∂a_n/∂p via log-derivative: d a_n = a_n * d log a_n; for n=1, uses ∂a₁/∂p
    if n == 1
        return _da1_dp(p, q, f)
    end
    an = _anfun(p, q, f, n)
    dlog = inv(p + q + n - 2) + inv(p + n - 1) - inv(p + 2*n - 3) - 2 * inv(p + 2*n - 2) - inv(p + 2*n - 1)
    return an * dlog
end

function _da1_dq(p::T, q::T, f::T) where {T<:AbstractFloat}
    # ∂a₁/∂q avoiding the removable singularity at q == 1, where a₁ vanishes
    # and the naive a₁/(q-1) is 0/0. Writing a₁ = (x/(1-x)) * (q-1)/(p+1) with
    # (x/(1-x)) = p*f/q leaves (q-1) as the only q-dependent factor.
    # x/ref: https://github.com/EnzymeAD/Enzyme.jl/issues/3581
    pfq = (p * f) / q
    return pfq / (p + 1)
end


function _dan_dq(p::T, q::T, f::T, n::Int) where {T<:AbstractFloat}
    # ∂a_n/∂q avoiding the removable singularity at q ≈ n for integer q.
    # For n=1, defer to the specific a₁ derivative.
    if n == 1
        return _da1_dq(p, q, f)
    end
    # Use the simplified closed-form of a_n that eliminates explicit q^2 via f:
    #   a_n = (x/(1-x))^2 * (n-1) * (p+n-1) * (p+q+n-2) * (q-n) / D(p,n)
    # where D(p,n) = (p+2n-3)*(p+2n-2)^2*(p+2n-1) and (x/(1-x)) = p*f/q.
    # Differentiate only the q-dependent factor G(q) = (p+q+n-2)*(q-n):
    #   dG/dq = (q-n) + (p+q+n-2) = p + 2q - 2.

    # This is equivalent to  
    #   return _anfun(p,q,f,n) * (inv(p+q+n-2) + inv(q-n))
    # but more precise.

    pfq = (p * f) / q
    C   = (pfq * pfq) * (n - 1) * (p + n - 1) /
          ((p + 2*n - 3) * (p + 2*n - 2)^2 * (p + 2*n - 1))
    return C * (p + 2*q - 2)
end

function _dbn_dp(p::T, q::T, f::T, n::Int) where {T<:AbstractFloat}
    # ∂b_n/∂p via quotient rule on b_n = N/D.
    # Note the internal dependence f(p,q)=q x/(p(1-x)) — terms cancel in N as per derivation.
    g = p * f + 2 * q
    A = 2 * n^2 + 2 * (p - 1) * n
    N1 = g * A
    N2 = p * q * (p - 2 - p * f)
    N = N1 + N2
    D = q * (p + 2*n - 2) * (p + 2*n)
    dN1_dp = 2 * n * g
    dN2_dp = q * (2 * p - 2) - p * q * f
    dN_dp = dN1_dp + dN2_dp
    dD_dp = q * (2 * p + 4 * n - 2)
    return (dN_dp * D - N * dD_dp) / (D^2)
end

function _dbn_dq(p::T, q::T, f::T, n::Int) where {T<:AbstractFloat}
    # ∂b_n/∂q similarly via quotient rule
    g = p * f + 2 * q
    A = 2 * n^2 + 2 * (p - 1) * n
    N1 = g * A
    N2 = p * q * (p - 2 - p * f)
    N = N1 + N2
    D = q * (p + 2*n - 2) * (p + 2*n)
    g_q = p * (f / q) + 2
    dN1_dq = g_q * A
    dN2_dq = p * (p - 2 - p * f) - p^2 * f
    dN_dq = dN1_dq + dN2_dq
    dD_dq = (p + 2*n - 2) * (p + 2*n)
    return (dN_dq * D - N * dD_dq) / (D^2)
end

function _nextapp(f::T, p::T, q::T, n::Int, App::T, Ap::T, Bpp::T, Bp::T) where {T<:AbstractFloat}
    # One step of the continuant recurrences:
    #   A_n = a_n A_{n-2} + b_n A_{n-1}
    #   B_n = a_n B_{n-2} + b_n B_{n-1}
    an = _anfun(p, q, f, n)
    bn = _bnfun(p, q, f, n)
    An = an * App + bn * Ap
    Bn = an * Bpp + bn * Bp
    return An, Bn, an, bn
end

function _dnextapp(an::T, bn::T, dan::T, dbn::T, Xpp::T, Xp::T, dXpp::T, dXp::T) where {T<:AbstractFloat}
    # Derivative propagation for the same recurrences (X∈{A,B})
    return dan * Xpp + an * dXpp + dbn * Xp + bn * dXp
end

function _beta_inc_grad(a, b, x; maxapp::Int=200, minapp::Int=3)
    T = promote_type(float(typeof(a)), float(typeof(b)), float(typeof(x)));
    err::T=eps(T)*T(1e4)
    a = T(a)
    b = T(b)
    x = T(x)
    # Compute I_x(a,b) and partial derivatives (∂I/∂a, ∂I/∂b, ∂I/∂x)
    # using a differentiated continued fraction with convergence control.
    oneT = one(T)
    zeroT = zero(T)

    # 1) Boundary cases for x
    x == oneT && return oneT, zeroT, zeroT, zeroT
    x == zeroT && return zeroT, zeroT, zeroT, zeroT

    # 2) Clamp iteration/tolerance parameters to robust defaults
    ϵ = min(err, T(1e-14))
    maxapp = max(1000, maxapp)
    minapp = max(5, minapp)

    # 3) Non-boundary path: precompute ∂I/∂x at original (a,b,x) via stable log form
    dx = exp((a - oneT) * log(x) + (b - oneT) * log1p(-x) - logbeta(a,b))

    # 4) Optional tail-swap for symmetry and improved CF convergence:
    #    if x > a/(a+b), evaluate at (p,q,x₀) = (b,a,1-x) and swap back at the end.
    p    = a
    q    = b
    x₀   = x
    swap = false
    if x > a / (a + b)
        x₀   = oneT - x
        p    = b
        q    = a
        swap = true
    end

    # 5) Initialize CF state and derivatives
    K                    = _Kfun(x₀, p, q)
    dK_dp_val, dK_dq_val = _dK_dpdq(x₀, p, q)
    f                    = _ffun(x₀, p, q)
    App                  = oneT
    Ap                   = oneT
    Bpp                  = zeroT
    Bp                   = oneT
    dApp_dp              = zeroT
    dBpp_dp              = zeroT
    dAp_dp               = zeroT
    dBp_dp               = zeroT
    dApp_dq              = zeroT
    dBpp_dq              = zeroT
    dAp_dq               = zeroT
    dBp_dq               = zeroT
    dI_dp                = T(NaN)
    dI_dq                = T(NaN)
    Ixpq                 = T(NaN)
    Ixpqn                = T(NaN)
    dI_dp_prev           = T(NaN)
    dI_dq_prev           = T(NaN)

    # 6) Main CF loop (n from 1): update continuants, scale, form current approximant Cn=A_n/B_n
    #    and its derivatives to update I and ∂I/∂(p,q). Stop on relative convergence of all.
    for n=1:maxapp

        # Update continuants. 
        An, Bn, an, bn = _nextapp(f, p, q, n, App, Ap, Bpp, Bp)
        dan            = _dan_dp(p, q, f, n)
        dbn            = _dbn_dp(p, q, f, n)
        dAn_dp         = _dnextapp(an, bn, dan, dbn, App, Ap, dApp_dp, dAp_dp)
        dBn_dp         = _dnextapp(an, bn, dan, dbn, Bpp, Bp, dBpp_dp, dBp_dp)
        dan            = _dan_dq(p, q, f, n)
        dbn            = _dbn_dq(p, q, f, n)
        dAn_dq         = _dnextapp(an, bn, dan, dbn, App, Ap, dApp_dq, dAp_dq)
        dBn_dq         = _dnextapp(an, bn, dan, dbn, Bpp, Bp, dBpp_dq, dBp_dq)

        # Normalize states to control growth/underflow (scale-invariant transform)
        s = maximum((abs(An), abs(Bn), abs(Ap), abs(Bp), abs(App), abs(Bpp)))
        if isfinite(s) && s > zeroT
            invs     = inv(s)
            An      *= invs
            Bn      *= invs
            Ap      *= invs
            Bp      *= invs
            App     *= invs
            Bpp     *= invs
            dAn_dp  *= invs
            dBn_dp  *= invs
            dAn_dq  *= invs
            dBn_dq  *= invs
            dAp_dp  *= invs
            dBp_dp  *= invs
            dApp_dp *= invs
            dBpp_dp *= invs
            dAp_dq  *= invs
            dBp_dq  *= invs
            dApp_dq *= invs
            dBpp_dq *= invs
        end

        # Form current approximant Cn=A_n/B_n and its derivatives.
        # Guard against tiny/zero Bn to avoid NaNs/Inf in divisions.
        tiny   = sqrt(eps(T))
        absBn  = abs(Bn)
        sgnBn  = ifelse(Bn >= zeroT, oneT, -oneT)
        invBn  = absBn > tiny && isfinite(absBn) ? inv(Bn) : inv(sgnBn * tiny)
        Cn     = An * invBn
        invBn2 = invBn * invBn
        dI_dp  = dK_dp_val * Cn + K * (invBn * dAn_dp - (An * invBn2) * dBn_dp)
        dI_dq  = dK_dq_val * Cn + K * (invBn * dAn_dq - (An * invBn2) * dBn_dq)
        Ixpqn  = K * Cn

        # Decide convergence: 
        if n >= minapp
            # Relative convergence for I, ∂I/∂p, ∂I/∂q (guards against tiny denominators)
            denomI = max(abs(Ixpqn), abs(Ixpq), eps(T))
            denomp = max(abs(dI_dp), abs(dI_dp_prev), eps(T))
            denomq = max(abs(dI_dq), abs(dI_dq_prev), eps(T))
            rI     = abs(Ixpqn - Ixpq) / denomI
            rp     = abs(dI_dp - dI_dp_prev) / denomp
            rq     = abs(dI_dq - dI_dq_prev) / denomq
            if max(rI, rp, rq) < ϵ
                break
            end
        end
        Ixpq       = Ixpqn
        dI_dp_prev = dI_dp
        dI_dq_prev = dI_dq

        # Shift CF state for next iteration
        App        = Ap
        Bpp        = Bp
        Ap         = An
        Bp         = Bn
        dApp_dp    = dAp_dp
        dApp_dq    = dAp_dq
        dBpp_dp    = dBp_dp
        dBpp_dq    = dBp_dq
        dAp_dp     = dAn_dp
        dAp_dq     = dAn_dq
        dBp_dp     = dBn_dp
        dBp_dq     = dBn_dq
    end

    # 7) Undo tail-swap if applied; ∂I/∂x is the pdf at original (a,b,x)
    if swap
        return oneT - Ixpqn, -dI_dq, -dI_dp, dx
    else
        return Ixpqn, dI_dp, dI_dq, dx
    end
end

EnzymeRules.@easy_rule(
    SpecialFunctions.beta_inc(a, b, x),
    @setup(
    (_, dIa, dIb, dIx) = _beta_inc_grad(a, b, x)
    ),
    (dIa, dIb, dIx),
    (-dIa, -dIb, -dIx),
)

Enzyme.EnzymeRules.@easy_rule(
    SpecialFunctions.beta_inc(a, b, x, y),
    @setup(
    (_, dIa, dIb, dIx) = _beta_inc_grad(a, b, x)
    ),
    (dIa, dIb, dIx, -dIx),
    (-dIa, -dIb, -dIx, dIx),
)

Enzyme.EnzymeRules.@easy_rule(
    SpecialFunctions.beta_inc_inv(a, b, p),
    @setup(

    (x, y) = Ω,

    # Implicit differentiation at solved x: I_x(a,b) = p
    (_, dIa, dIb, _) = _beta_inc_grad(a, b, x),

    # ∂I/∂x at solved x via stable log-space expression
    dIx_acc = exp(muladd(a - one(a), log(x), muladd(b - one(b), log1p(-x), -logbeta(a, b)))),
    inv_dIx = inv(dIx_acc),
    dx_da = -dIa * inv_dIx,
    dx_db = -dIb * inv_dIx,
    dx_dp = inv_dIx,
    ),
    (dx_da, dx_db, dx_dp),
    (-dx_da, -dx_db, -dx_dp)
)

end
