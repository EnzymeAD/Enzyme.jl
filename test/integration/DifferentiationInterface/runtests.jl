using ADTypes: AutoEnzyme
using DifferentiationInterface: DifferentiationInterface
using DifferentiationInterfaceTest:
    default_scenarios,
    sparse_scenarios,
    static_scenarios,
    test_differentiation,
    function_place,
    operator_place,
    FIRST_ORDER,
    SECOND_ORDER,
    Scenario
using Enzyme: Enzyme
using EnzymeCore: Forward, Reverse, Const, Duplicated
using StaticArrays: StaticArrays
using Test

logging = get(ENV, "CI", "false") == "false"

function remove_matrices(scens::Vector{<:Scenario})  # TODO: remove
    return filter(s -> s.x isa Union{Number, AbstractVector} && s.y isa Union{Number, AbstractVector}, scens)
end

backends = [
    AutoEnzyme(; function_annotation = Const),
    AutoEnzyme(; mode = Forward),
    AutoEnzyme(; mode = Reverse),
]

duplicated_backends = [
    AutoEnzyme(; mode = Forward, function_annotation = Duplicated),
    AutoEnzyme(; mode = Reverse, function_annotation = Duplicated),
]

@testset verbose = true "DifferentiationInterface integration" begin
    test_differentiation(
        backends,
        default_scenarios(; include_constantified = true);
        excluded = SECOND_ORDER,
        logging,
        testset_name = "Generic first order",
    )

    test_differentiation(
        backends[1],
        remove_matrices(default_scenarios(; include_constantified = true));
        excluded = FIRST_ORDER,
        logging,
        testset_name = "Generic second order",
    )

    test_differentiation(
        backends[2],
        remove_matrices(
            default_scenarios(;
                include_normal = false,
                include_cachified = true,
                include_constantorcachified = true,
                use_tuples = true,
            )
        );
        excluded = SECOND_ORDER,
        logging,
        testset_name = "Caches",
    )

    test_differentiation(
        duplicated_backends,
        remove_matrices(default_scenarios(; include_normal = false, include_closurified = true));
        excluded = SECOND_ORDER,
        logging,
        testset_name = "Closures",
    )

    filtered_static_scenarios = filter(remove_matrices(static_scenarios())) do s
        operator_place(s) == :out && function_place(s) == :out
    end

    test_differentiation(
        backends[2:3],
        filtered_static_scenarios;
        excluded = SECOND_ORDER,
        logging,
        testset_name = "Static arrays",
    )
end
