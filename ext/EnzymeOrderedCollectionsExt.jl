module EnzymeOrderedCollectionsExt

using OrderedCollections
using Enzyme
import Enzyme.EnzymeCore.EnzymeRules: inactive_noinl

function inactive_noinl(::typeof(OrderedCollections.ht_keyindex2), args...)
    return nothing
end

end
