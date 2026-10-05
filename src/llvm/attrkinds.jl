const RETURNED_ATTR_KIND = :returned
const HAS_CAPTURES_ATTR = LLVM.version() >= v"21"
# LLVM 21 replaced `nocapture` with `captures(none)`.
nocapture_attr() = HAS_CAPTURES_ATTR ? EnumAttribute(:captures, 0) : EnumAttribute(:nocapture)
function is_nocapture_attr(attr)
    attr isa LLVM.EnumAttribute || return false
    if HAS_CAPTURES_ATTR
        return attr.kind == :captures && attr.value == 0
    else
        return attr.kind == :nocapture
    end
end
const READONLY_ATTR_KIND = :readonly
const READNONE_ATTR_KIND = :readnone
const WRITEONLY_ATTR_KIND = :writeonly
const PRESERVEPRIMAL_ATTR_KIND = "enzyme_preserve_primal"
const LOWERED_CONVENTION_ATTR_KIND = "enzyme_lowered_convention"
