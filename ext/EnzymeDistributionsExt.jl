module EnzymeDistributionsExt

using Distributions: Distributions, Product, logpdf
using Enzyme: Enzyme

# Distributions sums the log density of a `Product` as
# `sum(n -> logpdf(dists[n], x[n]), 1:length(d))`. The closure captures both the
# distributions, which hold the active parameters, and the data `x`, which is
# constant, and Enzyme can only differentiate a struct mixing active and
# constant pointers with runtime activity on. When Enzyme compiles the method it
# instead finds this one, which adds the terms in a plain loop. The sum is
# sequential where Base's is pairwise, so the last bits may differ from the
# value computed outside Enzyme.
Base.Experimental.@overlay Enzyme.Compiler.GPUCompiler.GLOBAL_METHOD_TABLE function Distributions._logpdf(
        d::Product, x::AbstractVector{<:Real}
    )
    dists = d.v
    if isempty(dists)
        # What `sum(map(logpdf, dists, x))` gives, without zipping `dists` with `x`.
        return zero(Base.promote_op(logpdf, eltype(dists), eltype(x)))
    end
    s = logpdf(dists[1], x[1])
    for n in 2:length(d)
        s += logpdf(dists[n], x[n])
    end
    return s
end

end
