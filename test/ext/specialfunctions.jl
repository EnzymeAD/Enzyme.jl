using SpecialFunctions

# From https://github.com/JuliaDiff/ChainRules.jl/blob/02e7857e34b5c01067a288262f69cfcb9fce069b/test/rulesets/packages/SpecialFunctions.jl#L1

Enzyme.Compiler.DumpPreEnzyme[] = true

x = 1.5 + 0.7im
# for x in (1, -1, 0, 0.5, 10, -17.1, 1.5 + 0.7im)
	test_scalar(SpecialFunctions.besselj0, x)
