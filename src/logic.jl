import LLVM: refcheck

LLVM.@checked struct Logic
    ref::API.EnzymeLogicRef
    Logic() = new(API.CreateLogic())
end
Base.unsafe_convert(::Type{API.EnzymeLogicRef}, logic::Logic) = logic.ref
LLVM.dispose(logic::Logic) = API.FreeLogic(logic)

# typedef bool (*CustomRuleType)(int /*direction*/, CTypeTree * /*return*/,
#                                CTypeTree * /*args*/, size_t /*numArgs*/,
#                                LLVMValueRef)=T
