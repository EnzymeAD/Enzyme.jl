using Enzyme, Test

const CC = Core.Compiler

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
