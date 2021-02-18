import .API: CustomRuleType
import LLVM: refcheck

LLVM.@checked struct TypeAnalysis
    ref::API.EnzymeTypeAnalysisRef
end
Base.unsafe_convert(::Type{API.EnzymeTypeAnalysisRef}, ta::TypeAnalysis) = ta.ref
LLVM.dispose(ta::TypeAnalysis) = API.FreeTypeAnalysis(ta)

function TypeAnalysis(triple, typerules::Dict{String, CustomRuleType}=Dict{String,CustomRuleType}())
    rulenames = String[]
    rules = CustomRuleType[]
    for (rulename, rule) in typerules
        push!(rulenames, rulename)
        push!(rules, rule)
    end
    ref = API.CreateTypeAnalysis(triple, rulenames, rules)
    TypeAnalysis(ref)
end

# typedef uint8_t (*CustomRuleType)(int /*direction*/, CTypeTreeRef /*return*/,
#                                   CTypeTreeRef * /*args*/,
#                                   struct IntList * /*knownValues*/,
#                                   size_t /*numArgs*/, LLVMValueRef);
