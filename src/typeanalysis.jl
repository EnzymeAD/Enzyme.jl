import .API: CustomRuleType
import LLVM: refcheck

LLVM.@checked struct TypeAnalysis
    ref::API.EnzymeTypeAnalysisRef
end
Base.unsafe_convert(::Type{API.EnzymeTypeAnalysisRef}, ta::TypeAnalysis) = ta.ref
LLVM.dispose(ta::TypeAnalysis) = API.FreeTypeAnalysis(ta)

function TypeAnalysis(
    logic,
    typerules::Union{Dict{String,CustomRuleType}, Nothing} = nothing,
)
    if typerules isa Nothing
        ref = API.CreateTypeAnalysis(logic, (), ())
    else
        rulenames = String[]
        rules = CustomRuleType[]
        for (rulename, rule) in typerules
            push!(rulenames, rulename)
            push!(rules, rule)
        end
        ref = API.CreateTypeAnalysis(logic, rulenames, rules)
    end
    TypeAnalysis(ref)
end

# typedef bool (*CustomRuleType)(int /*direction*/, CTypeTree * /*return*/,
#                                CTypeTree * /*args*/, size_t /*numArgs*/,
#                                LLVMValueRef)=T
