function get_attr_kind_from_name(str::String)
    return LLVM.API.LLVMGetEnumAttributeKindForName(str, length(str))
end

const RETURNED_ATTR_KIND = get_attr_kind_from_name("returned")
const NOCAPTURE_ATTR_KIND = get_attr_kind_from_name("nocapture")
const READONLY_ATTR_KIND = get_attr_kind_from_name("readonly")
const READNONE_ATTR_KIND = get_attr_kind_from_name("readnone")
const WRITEONLY_ATTR_KIND = get_attr_kind_from_name("writeonly")
const PRESERVEPRIMAL_ATTR_KIND = "enzyme_preserve_primal"
