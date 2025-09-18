module EnzymeBFloat16sExt

using BFloat16s
using Enzyme

if !(isdefined(Core, :BFloat16) && Core.BFloat16 === BFloat16)
function Enzyme.typetree_inner(::Type{BFloat16}, ctx, dl, seen::Enzyme.Compiler.TypeTreeTable)
    return Enzyme.TypeTree(Enzyme.API.DT_BFloat16, -1, ctx)
end
end

end
