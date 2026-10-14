using Enzyme, Test
using Enzyme: EnzymeRules

# Derivative-free callees called natively instead of being emitted (Julia 1.10 and later).

struct NCStructure
    idx::Vector{Int}
    n::Int
end
@noinline nc_structure(n::Int) = NCStructure(collect(1:n), n)

function nc_loss(x::Vector{Float64})
    s = nc_structure(length(x))
    acc = 0.0
    for i in s.idx
        acc += x[i]^2
    end
    return acc
end

# `_sortperm` on integers takes and returns only boxed values, so Julia compiles it with
# the boxed `jl_fptr_args` ABI and not a specialized entry: it must stay emitted.
nc_sortperm(x) = (p = sortperm([3, 1, 2]); x * p[1])

# A read-only function whose callee calls an effect-free native callee. On Julia 1.10 and
# 1.11 that call goes through `jl_invoke`; unless it is marked read-only-or-throw, Enzyme
# no longer finds `nc_shared` read-only-or-throw and clones it, and on Julia 1.10 (typed
# pointers) that clone failed LLVM verification.
struct NCView
    parent::Vector{Float64}
    size::NTuple{5, Int}
    strides::NTuple{5, Int}
    offset::Int
end
@noinline function nc_strides(dims::NTuple{5, Int})
    s2 = dims[1]
    s3 = s2 * dims[2]
    s4 = s3 * dims[3]
    return (1, s2, s3, s4, s4 * dims[4])
end
@noinline function nc_view(x::Vector{Float64})
    dims = (length(x) ÷ 72, 3, 4, 3, 2)
    return NCView(x, dims, nc_strides(dims), 0)
end
@noinline function nc_shared(x::Vector{Float64}, p::NTuple{3, Int})
    t = nc_view(x)
    return t.size[p[1]] * t.strides[p[2]] == t.size[p[3]]
end
nc_branch(x, p) = nc_shared(x, p) ? sum(x) : 2 * sum(x)

# A callee with few boxed parameters and a boxed return: Julia gives it a specialized entry
# point all the same, so it is called natively.
@noinline nc_structure2(idx::Vector{Int}) = NCStructure(idx, length(idx))
function nc_loss2(x::Vector{Float64})
    s = nc_structure2(collect(1:length(x)))
    acc = 0.0
    for i in s.idx
        acc += x[i]^2
    end
    return acc
end

# Keyword calls: `@noinline` keeps the `Core.kwcall` wrapper out of line, so it is a
# candidate for a native call. Whether its function has a rule is asked of the signature
# without the wrapper, as every other decision point asks it.
@noinline function nc_kw(x::Int; iters::Int = 3)
    acc = 0
    for i in 1:iters
        acc += (x * i) % 7 + i ÷ 2
    end
    return acc
end
@noinline function nc_kw_ruled(x::Int; iters::Int = 3)
    acc = 0
    for i in 1:iters
        acc += (x * i) % 7 + i ÷ 2
    end
    return acc
end
function EnzymeRules.forward(config, ::Const{typeof(nc_kw_ruled)}, ::Type{<:Const}, x::Const{Int}; iters::Int = 3)
    return Const(nc_kw_ruled(x.val; iters))
end
nc_kw_user(x) = x * nc_kw(2; iters = 3)

kwcall_methodinstance(f, nt, args...) =
    Enzyme.my_methodinstance(nothing, typeof(Core.kwcall), Tuple{nt, typeof(f), args...})

# The job `onehot` emits for `llvmcall` outside any `EnzymeContext` is one of a host
# compilation whose parameters turn native callees off.
function primal_job(mi::Core.MethodInstance, params)
    GPUCompiler = Enzyme.Compiler.GPUCompiler
    config = GPUCompiler.CompilerConfig(Enzyme.Compiler.DefaultCompilerTarget(), params; kernel = false)
    return GPUCompiler.CompilerJob(mi, config)
end

@static if VERSION < v"1.12-" || v"1.12-beta3" <= VERSION
    @testset "Native derivative-free callees" begin
        x = [1.0, 2.0, 3.0]
        dx = zero(x)
        autodiff(Reverse, nc_loss, Active, Duplicated(x, dx))
        @test dx ≈ 2 .* x
        @test autodiff(Forward, nc_loss, Duplicated(x, ones(3)))[1] ≈ 12.0

        @test autodiff(Reverse, nc_sortperm, Active, Active(2.0))[1][1] ≈ 2.0
        @test autodiff(Forward, nc_sortperm, Duplicated(2.0, 1.0))[1] ≈ 2.0

        world = Base.get_world_counter()
        interp = Enzyme.Compiler.primal_interp_world(Forward, world)
        nt = NamedTuple{(:iters,), Tuple{Int}}
        mi_kw = kwcall_methodinstance(nc_kw, nt, Int)
        mi_kw_ruled = kwcall_methodinstance(nc_kw_ruled, nt, Int)
        @test Enzyme.Compiler.native_entry(interp, mi_kw) !== nothing
        @test Enzyme.Compiler.native_entry(interp, mi_kw_ruled) === nothing
        # Memoized per world and rules.
        key = Enzyme.Compiler.native_entry_key(interp, mi_kw)
        @test haskey(Enzyme.Compiler.NATIVE_ENTRY_CACHE, key)
        @test Enzyme.Compiler.NATIVE_ENTRY_CACHE[key] === Enzyme.Compiler.native_entry(interp, mi_kw)
        @test autodiff(Forward, nc_kw_user, Duplicated(2.0, 1.0))[1] ≈ nc_kw(2; iters = 3)
        @test autodiff(Reverse, nc_kw_user, Active, Active(2.0))[1][1] ≈ nc_kw(2; iters = 3)

        mi2 = Enzyme.my_methodinstance(nothing, typeof(nc_structure2), Tuple{Vector{Int}})
        @test Enzyme.Compiler.native_entry(interp, mi2) !== nothing
        x = [1.0, 2.0, 3.0]
        dx = zero(x)
        autodiff(Reverse, nc_loss2, Active, Duplicated(x, dx))
        @test dx ≈ 2 .* x

        mi_zero = Enzyme.my_methodinstance(nothing, typeof(Enzyme.zerosetfn), Tuple{Vector{Float64}, Int})
        GPUCompiler = Enzyme.Compiler.GPUCompiler
        params = Enzyme.Compiler.PrimalCompilerParams(Enzyme.API.DEM_ForwardMode)
        @test params.native_callees
        @test GPUCompiler.get_interpreter(primal_job(mi_zero, params)).native_callees
        params = Enzyme.Compiler.PrimalCompilerParams(Enzyme.API.DEM_ForwardMode; native_callees = false)
        @test !GPUCompiler.get_interpreter(primal_job(mi_zero, params)).native_callees
        @test Enzyme.onehot([1.0, 2.0]) == ([1.0, 0.0], [0.0, 1.0])

        # Julia 1.10: Enzyme keeps its inference results in caches of its own, and a job
        # that does not call natively must not find a caller inferred by one that does.
        @static if VERSION < v"1.11.0-DEV.1552"
            CC = Core.Compiler
            mode = Enzyme.API.DEM_ReverseModeCombined
            native_cache = Enzyme.Compiler.enzyme_ci_cache(mode, true)
            cache = Enzyme.Compiler.enzyme_ci_cache(mode, false)
            @test native_cache !== cache
            mi_loss = Enzyme.my_methodinstance(nothing, typeof(nc_loss), Tuple{Vector{Float64}})
            mi_callee = Enzyme.my_methodinstance(nothing, typeof(nc_structure), Tuple{Int})
            @test CC.get(CC.WorldView(native_cache, world, world), mi_loss, nothing) !== nothing
            @test CC.get(CC.WorldView(native_cache, world, world), mi_callee, nothing) === nothing
            interp = Enzyme.Compiler.Interpreter.EnzymeInterpreter(cache, nothing, world, mode, true; native_callees = false)
            CC.typeinf_ext_toplevel(interp, mi_loss)
            @test CC.get(CC.code_cache(interp), mi_callee, nothing) !== nothing
        end

        # TODO: with code coverage `nc_shared` writes its counters, so Enzyme clones it
        # anyway, and on Julia 1.10 the clone fails verification until Enzyme_jll
        # carries EnzymeAD/Enzyme#3491.
        @static if VERSION >= v"1.11-" || Base.JLOptions().code_coverage == 0
            x = ones(144)
            for p in ((1, 3, 2), (2, 1, 2))
                dx = zero(x)
                autodiff(Reverse, nc_branch, Active, Duplicated(x, dx), Const(p))
                @test dx == fill(nc_shared(x, p) ? 1.0 : 2.0, 144)
            end
        end
    end
end
