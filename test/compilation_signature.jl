using Enzyme, Test

const CC = Core.Compiler
const LLVM = Enzyme.Compiler.LLVM
const GPUCompiler = Enzyme.Compiler.GPUCompiler

# For a varargs call the signature that is compiled and invoked differs from the signature
# seen at the call site: `csig_vararg(x, 2.0, 3.0)` is inferred at
# `Tuple{typeof(csig_vararg), Float64, Float64, Float64}` but compiled as
# `Tuple{typeof(csig_vararg), Float64, Float64, Vararg{Float64}}`.
const csig_sink = Ref(0.0)

# `@noinline` plus the store keep the call from being inlined or dropped, which is what
# makes inference bother with the compilation signature in the first place.
@noinline function csig_vararg(x::Float64, ys::Float64...)
    csig_sink[] = x
    return x * prod(ys)
end

csig_caller(x::Float64) = csig_vararg(x, 2.0, 3.0)

csig_vararg_ad(x::Float64, ys::Float64...) = x * prod(ys)
csig_caller_ad(x::Float64) = csig_vararg_ad(x, 2.0, 3.0)

# The `CodeInstance` for the compilation signature of `csig_vararg`, as cached by `interp`.
function compilation_signature_ci(interp)
    meth = which(csig_vararg, Tuple{Float64, Float64, Float64})
    call_sig = Tuple{typeof(csig_vararg), Float64, Float64, Float64}
    csig = CC.get_compileable_sig(meth, call_sig, Core.svec())
    @assert csig !== nothing && csig !== call_sig
    cmi = CC.specialize_method(meth, csig, Core.svec())
    return CC.get(CC.code_cache(interp), cmi, nothing)
end

@testset "infer_compilation_signature" begin
    world = Base.get_world_counter()

    for mode in (Reverse, Forward)
        @test CC.infer_compilation_signature(Enzyme.Compiler.primal_interp_world(mode, world))
    end

    interp = Enzyme.Compiler.primal_interp_world(Reverse, world)
    # Nothing has inferred `csig_vararg` under the Enzyme interpreter yet.
    @test compilation_signature_ci(interp) === nothing

    mi = Enzyme.Compiler.my_methodinstance(Reverse, typeof(csig_caller), Tuple{Float64}, world)
    @test Enzyme.Compiler.return_type(interp, mi) === Float64

    # Inferring the caller inferred the varargs compilation signature as well, so codegen
    # finds it in the cache instead of having to re-infer it.
    @test compilation_signature_ci(interp) isa Core.CodeInstance

    # Differentiating through a varargs call keeps working.
    @test autodiff(Reverse, csig_caller_ad, Active, Active(1.5))[1][1] ≈ 6.0
end

# Julia has no specsig entry point for a varargs compilation signature, so it compiles
# one with the jlcall ABI (`japi1_*`) and the call site becomes a `julia.call` of that
# wrapper. Enzyme has no derivative for that convention and instead differentiates such
# a site as a dynamic call, so check that the resulting derivatives are right — and that
# the site is emitted in the first place, since a `@noinline` that stopped mattering
# would leave the path untested.
@noinline csig_sumsq(f, args...) = sum(abs2, f(args...))
csig_scale(x::Vector{Float64}, y::Vector{Float64}) = x .* y
function csig_byref(out, fn, args...)
    out[] = fn(args...)
    return nothing
end

function has_jlcall_abi_site(mod)
    for f in LLVM.functions(mod), bb in LLVM.blocks(f), inst in LLVM.instructions(bb)
        isa(inst, LLVM.CallInst) || continue
        callee = LLVM.called_operand(inst)
        (isa(callee, LLVM.Function) && LLVM.name(callee) == "julia.call") || continue
        target = LLVM.operands(inst)[1]
        if isa(target, LLVM.Function) && startswith(LLVM.name(target), "japi1")
            return true
        end
    end
    return false
end

@testset "jlcall ABI call sites" begin
    x, y = [2.0, 3.0], [5.0, 7.0]
    primal = sum(abs2, x .* y)
    grad = 2 .* (x .* y) .* y

    TT = Tuple{
        BatchDuplicatedNoNeed{Base.RefValue{Float64}, 2}, Const{typeof(csig_sumsq)},
        Const{typeof(csig_scale)}, BatchDuplicated{Vector{Float64}, 2},
        Const{Vector{Float64}},
    }
    job = Enzyme.Compiler.get_job(csig_byref, Const, TT; width = 2, run_enzyme = false, optimize = false)
    GPUCompiler.JuliaContext() do ctx
        mod, _ = GPUCompiler.codegen(:llvm, job)
        @test has_jlcall_abi_site(mod)
    end

    out, dout = Ref(0.0), Ref(0.0)
    dx = [1.0, 0.0]
    autodiff(
        Forward, csig_byref, Const, Duplicated(out, dout),
        Const(csig_sumsq), Const(csig_scale), Duplicated(x, dx), Const(y),
    )
    @test out[] ≈ primal
    @test dout[] ≈ sum(grad .* dx)

    out, dout = Ref(0.0), Ref(1.0)
    dx = [0.0, 0.0]
    autodiff(
        Reverse, csig_byref, Const, DuplicatedNoNeed(out, dout),
        Const(csig_sumsq), Const(csig_scale), Duplicated(x, dx), Const(y),
    )
    @test dx ≈ grad

    out, dout, dout2 = Ref(0.0), Ref(1.0), Ref(3.0)
    dx, dx2 = [0.0, 0.0], [0.0, 0.0]
    autodiff(
        Reverse, csig_byref, Const, BatchDuplicatedNoNeed(out, (dout, dout2)),
        Const(csig_sumsq), Const(csig_scale), BatchDuplicated(x, (dx, dx2)), Const(y),
    )
    @test dx ≈ grad
    @test dx2 ≈ 3 .* grad
end
