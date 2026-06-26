using Enzyme
using Test

Enzyme.API.printall!(true)
Enzyme.Compiler.DumpPostOpt[] = true

include("test_isolated.jl")
