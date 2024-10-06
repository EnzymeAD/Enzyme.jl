import .API: CustomRuleType
import LLVM: refcheck

LLVM.@checked struct TypeAnalysis
    ref::API.EnzymeTypeAnalysisRef
end
Base.unsafe_convert(::Type{API.EnzymeTypeAnalysisRef}, ta::TypeAnalysis) = ta.ref
LLVM.dispose(ta::TypeAnalysis) = API.FreeTypeAnalysis(ta)

function TypeAnalysis(
    logic,
    typerules::Dict{String,CustomRuleType} = Dict{String,CustomRuleType}(),
)
    rulenames = String[]
    rules = CustomRuleType[]
    for (rulename, rule) in typerules
        push!(rulenames, rulename)
        push!(rules, rule)
    end
    ref = API.CreateTypeAnalysis(logic, rulenames, rules)
    TypeAnalysis(ref)
end

# typedef bool (*CustomRuleType)(int /*direction*/, CTypeTree * /*return*/,
#                                CTypeTree * /*args*/, size_t /*numArgs*/,
#                                LLVMValueRef)=T
