module EnzymeDistributionsExt

using Distributions: Distributions, Product, logpdf
using Enzyme: Enzyme

# Distributions sums the log density of a `Product` as
# `sum(n -> logpdf(dists[n], x[n]), 1:length(d))`. The closure captures both the
# distributions, which hold the active parameters, and the data `x`, which is
# constant, and Enzyme can only differentiate a struct mixing active and
# constant pointers with runtime activity on. When Enzyme compiles the method it
# instead finds this one, which adds the same terms in the same order (Base's
# pairwise summation: blocks of at most 1024 terms left to right, longer ranges
# split in half) without the closure.
Base.Experimental.@overlay Enzyme.Compiler.GPUCompiler.GLOBAL_METHOD_TABLE function Distributions._logpdf(
        d::Product, x::AbstractVector{<:Real}
    )
    dists = d.v
    if isempty(dists)
        # What `sum(map(logpdf, dists, x))` gives, without zipping `dists` with `x`.
        return zero(Base.promote_op(logpdf, eltype(dists), eltype(x)))
    end
    return product_logpdf(dists, x, 1, length(d))
end

function product_logpdf(dists, x, lo::Int, hi::Int)
    if hi - lo < 1024
        s = logpdf(dists[lo], x[lo])
        for n in (lo + 1):hi
            s += logpdf(dists[n], x[n])
        end
        return s
    end
    mid = lo + (hi - lo) >> 1
    return product_logpdf(dists, x, lo, mid) + product_logpdf(dists, x, mid + 1, hi)
end

end
