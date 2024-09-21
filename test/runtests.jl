using Test

using InteractiveUtils
using Enzyme
Enzyme.API.printall!(true)
Enzyme.Compiler.DumpPostOpt[] = true

f1(x) = 1.0 + x
f2(x) = x*x
T = Float16
cmp = if T == Float64
    T(0.41997434161402606939)
else
    T(0.41997434161402606939f0)
end
@show @code_llvm autodiff(Forward, tanh, Duplicated(T(1), T(1)))
res = autodiff(Forward, tanh, Duplicated(T(1), T(1)))[1]
@test res isa T
@test res ≈ cmp
