# To run:
# using PkgBenchmark, Enzyme
# result = benchmarkpkg(Enzyme)
# export_markdown("benchmark/perf.md", result)

# Note: if you change this file you will need to delete an regenerate tune.json
# Your "v1.x" environment needs to have BenchmarkTools and PkgBenchmark installed.

using BenchmarkTools
using Enzyme

const SUITE = BenchmarkGroup()

SUITE["basics"] = BenchmarkGroup()

SUITE["basics"]["overhead"] = @benchmarkable Enzyme.autodiff(Forward, identity, Const(1.0))

SUITE["basics"]["make_zero"] = BenchmarkGroup()
SUITE["basics"]["remake_zero!"] = BenchmarkGroup()

p = (; x = 1.0, y = zeros(3))

SUITE["basics"]["make_zero"]["namedtuple"] = @benchmarkable Enzyme.make_zero($p)
SUITE["basics"]["remake_zero!"]["namedtuple"] = @benchmarkable Enzyme.remake_zero!(dp) setup = (dp = Enzyme.make_zero(p))

struct MyStruct
    x::Float64
    y::Vector{Float64}
end

x = MyStruct(1.0, zeros(3))

SUITE["basics"]["make_zero"]["struct"] = @benchmarkable Enzyme.make_zero($x)
SUITE["basics"]["remake_zero!"]["struct"] = @benchmarkable Enzyme.remake_zero!(dx) setup = (dx = Enzyme.make_zero(x))
