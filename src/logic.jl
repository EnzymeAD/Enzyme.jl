import LLVM: refcheck

LLVM.@checked struct Logic
    ref::API.EnzymeLogicRef
    ctx::EnzymeContext
    function Logic(ctx::EnzymeContext)
        ref = API.CreateLogic()
        GC.@preserve ctx begin
            API.LogicSetExternalContext(ref, Base.pointer_from_objref(ctx))
            return new(ref, ctx)
        end
    end
    # Wrap a borrowed logic ref (e.g. obtained from a `GradientUtils`/`TypeAnalyzer`); the
    # external context was set when the owning logic was created, so recover it from there.
    function Logic(ref::API.EnzymeLogicRef)
        ptr = API.LogicGetExternalContext(ref)
        @assert ptr != C_NULL
        return new(ref, unsafe_pointer_to_objref(ptr)::EnzymeContext)
    end
end
Base.unsafe_convert(::Type{API.EnzymeLogicRef}, logic::Logic) = logic.ref
LLVM.dispose(logic::Logic) = API.FreeLogic(logic)

function enzyme_context(logic::Logic)
    return logic.ctx::EnzymeContext
end

LLVM.@checked struct TypeAnalyzer
    ref::API.EnzymeTypeAnalyzerRef
end
Base.unsafe_convert(::Type{API.EnzymeTypeAnalyzerRef}, ta::TypeAnalyzer) = ta.ref
LLVM.dispose(ta::TypeAnalyzer) = throw("Cannot free type analyzer")

get_logic(ta::TypeAnalyzer) = Logic(API.EnzymeTypeAnalyzerGetLogic(ta))

enzyme_context(ta::TypeAnalyzer) = enzyme_context(get_logic(ta))

# typedef bool (*CustomRuleType)(int /*direction*/, CTypeTree * /*return*/,
#                                CTypeTree * /*args*/, size_t /*numArgs*/,
#                                LLVMValueRef)=T
