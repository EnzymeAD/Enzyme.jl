module AdaptExt

using Adapt
using EnzymeCore

Adapt.adapt_structure(to, x::Const) = Const(adapt(to, x.val))
Adapt.adapt_structure(to, x::Active) = Active(adapt(to, x.val))
Adapt.adapt_structure(to, x::Duplicated) = Duplicated(adapt(to, x.val), adapt(to, x.dval))
function Adapt.adapt_structure(to, x::DuplicatedNoNeed)
    return DuplicatedNoNeed(adapt(to, x.val), adapt(to, x.dval))
end
function Adapt.adapt_structure(to, x::BatchDuplicated)
    return BatchDuplicated(adapt(to, x.val), adapt(to, x.dval))
end
function Adapt.adapt_structure(to, x::BatchDuplicatedNoNeed)
    return BatchDuplicatedNoNeed(adapt(to, x.val), adapt(to, x.dval))
end
# The shadow of a mixed annotation is a `Base.RefValue` that the caller reads the active
# parts of the gradient back from. Adapting it means building a new `Ref`, and writes to
# that never reach the caller's, so the gradient would be silently dropped. Throw instead.
# See https://github.com/EnzymeAD/Enzyme.jl/issues/3451.
struct MixedAdaptError <: Exception
    T::Type
end
function Base.showerror(io::IO, e::MixedAdaptError)
    return print(
        io, "MixedAdaptError: cannot `adapt` a `", nameof(e.T), "`. Its shadow is a ",
        "`Base.RefValue`, and adapting it would create a new `Ref`, so gradients written ",
        "to it would not reach the original shadow. ",
        "See https://github.com/EnzymeAD/Enzyme.jl/issues/3451."
    )
end
Adapt.adapt_structure(to, ::MixedDuplicated) = throw(MixedAdaptError(MixedDuplicated))
function Adapt.adapt_structure(to, ::BatchMixedDuplicated)
    throw(MixedAdaptError(BatchMixedDuplicated))
end

end #module
