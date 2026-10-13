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
# The shadow of a mixed annotation must be a `Base.RefValue`, but adaptors either leave
# a `Base.RefValue` alone (so its contents are not adapted) or replace it with their own
# ref type (CUDA's `KernelAdaptor` gives a `KernelRefValue`). Adapt the contents and wrap
# them in a new `Ref` instead. Writes to the new `Ref` do not reach the old one.
function Adapt.adapt_structure(to, x::MixedDuplicated)
    return MixedDuplicated(adapt(to, x.val), Ref(adapt(to, x.dval[])))
end
function Adapt.adapt_structure(to, x::BatchMixedDuplicated)
    dval = map(r -> Ref(adapt(to, r[])), x.dval)
    return BatchMixedDuplicated(adapt(to, x.val), dval)
end

end #module
