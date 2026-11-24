import LLVM: refcheck

LLVM.@checked struct Logic
    ref::API.EnzymeLogicRef
    Logic() = new(API.CreateEnzymeLogic(false))
end
Base.unsafe_convert(::Type{API.EnzymeLogicRef}, logic::Logic) = logic.ref
LLVM.dispose(logic::Logic) = API.FreeEnzymeLogic(logic)

# typedef bool (*CustomRuleType)(int /*direction*/, CTypeTree * /*return*/,
#                                CTypeTree * /*args*/, size_t /*numArgs*/,
#                                LLVMValueRef)=T
