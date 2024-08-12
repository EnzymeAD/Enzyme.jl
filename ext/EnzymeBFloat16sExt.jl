module EnzymeBFloat16sExt

using BFloat16s
using Enzyme

function Enzyme.typetree_inner(::Type{BFloat16}, ctx, dl, seen::Enzyme.Compiler.TypeTreeTable)
    return Enzyme.TypeTree(Enzyme.API.DT_BFloat16, -1, ctx)
end

end
