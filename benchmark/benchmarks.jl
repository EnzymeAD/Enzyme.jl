# To run:
# using PkgBenchmark, Enzyme
# result = benchmarkpkg(KernelAbstractions)
# export_markdown("benchmark/perf.md", result)

# Note: if you change this file you will need to delete an regenerate tune.json
# Your "v1.x" environment needs to have BenchmarkTools and PkgBenchmark installed.

using BenchmarkTools
using Enzyme

const SUITE = BenchmarkGroup()

SUITE["basics"] = BenchmarkGroup()

SUITE["basics"]["overhead"] = @benchmarkable Enzyme.autodiff(Forward, identity, Const(1.0))