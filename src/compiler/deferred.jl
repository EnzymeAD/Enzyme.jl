# Deferred compilation is Enzyme's device-side entry point: a kernel calls `autodiff_deferred`,
# which requests the differentiated function with `GPUCompiler.deferred_codegen(Val(spec))`.
# `spec` is a `DeferredSpec` type, which describes the Enzyme job entirely at the type level.
# GPUCompiler registers the request when it generates code for the kernel, and compiles the
# job that `GPUCompiler.deferred_job` (below) creates for it as part of the kernel's
# compilation. The kernel's inferred (and possibly cached) code only refers to the
# specification, never to an id or address of a session.
#
# There is no host runtime path (and none is needed: higher-order differentiation goes
# through plain `autodiff`, whose nested calls Enzyme recognizes by itself). Called natively
# on the host, the request returns `C_NULL`, and `deferred_codegen` raises
# `DeferredOnHostError`. In device code the request resolves to the function's address, so
# the check folds away.

struct DeferredSpec{
        FA, A, TT, Mode, Width, ModifiedBetween, ReturnPrimal, ShadowInit, ExpectedTapeType,
        ErrIfFuncWritten, RuntimeActivity, StrongZero,
    } end

@inline function deferred_spec(
        ::Type{FA},
        ::Type{A},
        ::Type{TT},
        ::Val{Mode},
        ::Val{Width},
        ::Val{ModifiedBetween},
        ::Val{ReturnPrimal},
        ::Val{ShadowInit},
        ::Type{ExpectedTapeType},
        ::Val{ErrIfFuncWritten},
        ::Val{RuntimeActivity},
        ::Val{StrongZero},
    ) where {
        FA, A, TT, Mode, Width, ModifiedBetween, ReturnPrimal, ShadowInit, ExpectedTapeType,
        ErrIfFuncWritten, RuntimeActivity, StrongZero,
    }
    return Val{
        DeferredSpec{
            FA, A, TT, Mode, Width, ModifiedBetween, ReturnPrimal, ShadowInit,
            ExpectedTapeType, ErrIfFuncWritten, RuntimeActivity, StrongZero,
        },
    }()
end

struct DeferredOnHostError <: EnzymeError end

function Base.showerror(io::IO, ::DeferredOnHostError)
    return print(
        io,
        "autodiff_deferred (and autodiff_deferred_thunk) can only be used inside GPU kernels. ",
        "Use autodiff on the host, also for higher-order differentiation.",
    )
end

# The compiler job for the specification at `world`; `nothing` if the primal has no method
# for the signature, or a `String` describing why the request is invalid.
function deferred_job(
        ::Type{
            DeferredSpec{
                FA, A, TT, Mode, Width, ModifiedBetween, ReturnPrimal, ShadowInit,
                ExpectedTapeType, ErrIfFuncWritten, RuntimeActivity, StrongZero,
            },
        },
        world::UInt,
    ) where {
        FA, A, TT, Mode, Width, ModifiedBetween, ReturnPrimal, ShadowInit, ExpectedTapeType,
        ErrIfFuncWritten, RuntimeActivity, StrongZero,
    }
    ft = eltype(FA)
    primal_tt = Tuple{map(eltype, TT.parameters)...}
    sugar = Mode == API.DEM_ForwardMode ? Forward : Reverse

    min_world = Ref{UInt}(typemin(UInt))
    max_world = Ref{UInt}(typemax(UInt))
    mi = my_methodinstance(sugar, ft, primal_tt, world, min_world, max_world)
    mi === nothing && return nothing

    rt2 = if A isa UnionAll
        rrt = primal_return_type_world(sugar, world, mi)

        # Don't error here but default to nothing return since in cuda context we don't use the device overrides
        if rrt == Union{}
            rrt = Nothing
        end

        if !(A <: Const) && guaranteed_const_nongen(rrt, world)
            return "Return type `$rrt` not marked Const, but type is guaranteed to be constant"
        end
        instantiate_annotation(A, rrt, Width)
    else
        @assert A isa DataType
        A
    end

    params = EnzymeCompilerParams(
        PrimalCompilerParams(Mode),
        Tuple{FA, TT.parameters...},
        Mode,
        Width,
        rt2,
        true,
        true,
        ModifiedBetween,
        ReturnPrimal,
        ShadowInit,
        ExpectedTapeType,
        FFIABI,
        ErrIfFuncWritten,
        RuntimeActivity,
        StrongZero
    ) #=abiwrap=#
    return CompilerJob(mi, CompilerConfig(EnzymeTarget(), params; kernel = false), world)
end

# The job GPUCompiler compiles for a request from `deferred_codegen`, as part of the
# compilation of `job`. `deferred_check` already rejects invalid requests in the requesting
# code, so this only fails for a request that is invalid in the world of `job` but was valid
# where the requesting code was inferred.
function GPUCompiler.deferred_job(::Type{S}, job::CompilerJob) where {S <: DeferredSpec}
    dyn_job = deferred_job(S, job.world)
    if dyn_job === nothing
        error("The function to differentiate has no method for the deferred request $S")
    elseif dyn_job isa String
        error(dyn_job)
    end
    return dyn_job
end

# `deferred_check` (a generated function, defined in src/late_generated.jl) checks a request
# while the requesting code is inferred. It throws the errors of invalid requests there (and
# so also on the host, before `DeferredOnHostError`), and gives the requesting code an edge to
# the primal's method instance, so that it is invalidated when the primal is redefined.
# Otherwise it returns `nothing`.
function deferred_check end

function deferred_check_generator(world::UInt, source::Union{Method, LineNumberNode}, @nospecialize(S::Type), @nospecialize(self), @nospecialize(spec::Type))
    @nospecialize

    slotnames = Core.svec(Symbol("#self#"), :spec)
    stub = Core.GeneratedFunctionStub(identity, slotnames, Core.svec())

    job = deferred_job(S, world)
    if job === nothing
        ft = eltype(S.parameters[1])
        primal_tt = Tuple{map(eltype, S.parameters[3].parameters)...}
        return stub(world, source, :(throw(MethodError($ft, $primal_tt, $world))))
    elseif job isa String
        return stub(world, source, :(error($job)))
    end

    code = Any[Core.Compiler.ReturnNode(nothing)]
    ci = create_fresh_codeinfo(deferred_check, source, world, slotnames, code)
    ci.edges = Any[job.source]
    return ci
end

@inline function deferred_codegen(
        @nospecialize(fa::Type),
        @nospecialize(a::Type),
        @nospecialize(tt::Type),
        @nospecialize(mode::Val),
        @nospecialize(width::Val),
        @nospecialize(modifiedbetween::Val),
        @nospecialize(returnprimal::Val),
        @nospecialize(shadowinit::Val),
        @nospecialize(expectedtapetype::Type),
        @nospecialize(erriffuncwritten::Val),
        @nospecialize(runtimeactivity::Val),
        @nospecialize(strongzero::Val)
    )
    spec = deferred_spec(fa, a, tt, mode, width, modifiedbetween, returnprimal, shadowinit, expectedtapetype, erriffuncwritten, runtimeactivity, strongzero)
    deferred_check(spec)
    ptr = GPUCompiler.deferred_codegen(spec)
    # `C_NULL` only outside of a compilation by GPUCompiler, i.e. when running natively
    if ptr == C_NULL
        throw(DeferredOnHostError())
    end
    return ptr
end
