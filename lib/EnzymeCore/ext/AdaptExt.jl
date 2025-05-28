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
function Adapt.adapt_structure(to, x::StackedBatchDuplicated)
    return StackedBatchDuplicated(adapt(to, x.val), adapt(to, x.dval))
end
function Adapt.adapt_structure(to, x::StackedBatchDuplicatedNoNeed)
    return StackedBatchDuplicatedNoNeed(adapt(to, x.val), adapt(to, x.dval))
end
function Adapt.adapt_structure(to, x::MixedDuplicated)
    return MixedDuplicated(adapt(to, x.val), adapt(to, x.dval))
end
function Adapt.adapt_structure(to, x::BatchMixedDuplicated)
    return BatchMixedDuplicated(adapt(to, x.val), adapt(to, x.dval))
end

end #module
