module EnzymeSpecialFunctionsExt

using SpecialFunctions
using Enzyme

function __init__()
    Enzyme.Compiler.known_ops[typeof(SpecialFunctions._logabsgamma)] = (:logabsgamma, 1, (:digamma, typeof(SpecialFunctions.digamma)))
    Enzyme.Compiler.cmplx_known_ops[typeof(SpecialFunctions.bessely)] = (:cmplx_yn, 2, nothing)
    Enzyme.Compiler.cmplx_known_ops[typeof(SpecialFunctions.besseli)] = (:cmplx_jn, 2, nothing)
    Enzyme.Compiler.cmplx_known_ops[typeof(SpecialFunctions.besselj)] = (:cmplx_jn, 2, nothing)
    Enzyme.Compiler.cmplx_known_ops[typeof(SpecialFunctions.besselk)] = (:cmplx_kn, 2, nothing)
end

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
    # ∂a₁/∂q
    return _a1fun(p, q, f) / (q - 1)
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
