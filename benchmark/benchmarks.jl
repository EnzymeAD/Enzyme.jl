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


@noinline function sass(σ, x)
    z = σ / x
    return z
end

@noinline function multidim_sum_bcast0(dist, x, res)
    broadcasted = Broadcast.broadcasted(sass, dist, x)
    lp = sum(Broadcast.instantiate(broadcasted))
    res[] = lp
    return nothing
end

function multidim_sum_bcast(dist, y)
    tmp = Ref{Float64}(0.0)
    multidim_sum_bcast0(dist, y, tmp)
    return tmp[]::Float64
end


y1d = rand(40);
dist1d = fill(10.0, 40);

SUITE["fold_broadcast"]["multidim_sum_bcast"]["1D"] = @benchmarkable Enzyme.gradient(set_runtime_activity(Reverse), multidim_sum_bcast, Const($dist1d), $y1d)

y2d = rand(10, 4);
dist2d = fill(10.0, 10, 4);

SUITE["fold_broadcast"]["multidim_sum_bcast"]["2D"] = @benchmarkable Enzyme.gradient(set_runtime_activity(Reverse), multidim_sum_bcast, Const($dist2d), $y2d)


# With `InlineABI` Julia inlines the derivative into its caller. A call must then cost no more
# than the derivative: no lookup of the pgcstack, and nothing opaque that stops the
# optimization of a loop around it.
inline_abi_f(x) = x[1] * x[1]
inline_abi_f_alloc(x) = sum(abs2, x .* 2.0)

@noinline function inline_abi_call(x, dx)
    Enzyme.autodiff(set_abi(Reverse, InlineABI), inline_abi_f, Active, Duplicated(x, dx))
    return nothing
end

@noinline function inline_abi_call_alloc(x, dx)
    Enzyme.autodiff(set_abi(Reverse, InlineABI), inline_abi_f_alloc, Active, Duplicated(x, dx))
    return nothing
end

function inline_abi_loop(x, dx, n)
    for _ in 1:n
        Enzyme.autodiff(set_abi(Reverse, InlineABI), inline_abi_f, Active, Duplicated(x, dx))
    end
    return nothing
end

x_inline_abi = [3.0, 1.0]

SUITE["inline_abi"] = BenchmarkGroup()
SUITE["inline_abi"]["call"] = @benchmarkable inline_abi_call($x_inline_abi, dx) setup = (dx = zero(x_inline_abi))
SUITE["inline_abi"]["call_alloc"] = @benchmarkable inline_abi_call_alloc($x_inline_abi, dx) setup = (dx = zero(x_inline_abi))
# 1000 calls per evaluation
SUITE["inline_abi"]["loop"] = @benchmarkable inline_abi_loop($x_inline_abi, dx, 1000) setup = (dx = zero(x_inline_abi))
