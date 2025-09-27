import LLVM: refcheck

LLVM.@checked struct Logic
    ref::API.EnzymeLogicRef
    ctx::EnzymeContext
    function Logic(ctx::EnzymeContext)
        ref = API.CreateLogic()
        API.LogicSetExternalContext(ref, ctx)
        new(ref, ctx)
    end
end
Base.unsafe_convert(::Type{API.EnzymeLogicRef}, logic::Logic) = logic.ref
LLVM.dispose(logic::Logic) = API.FreeLogic(logic)

function enzyme_context(logic::Logic)
    return logic.ctx::EnzymeContext
end

function enzyme_context(logic::API.EnzymeLogicRef)
    ptr = API.LogicGetExternalContext(logic)
    @assert ptr != C_NULL
    return unsafe_pointer_to_objref(ptr)::EnzymeContext
end

# typedef bool (*CustomRuleType)(int /*direction*/, CTypeTree * /*return*/,
#                                CTypeTree * /*args*/, size_t /*numArgs*/,
#                                LLVMValueRef)=T
