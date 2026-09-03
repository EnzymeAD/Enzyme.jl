# Deferred compilation is Enzyme's device-side entry point: a kernel calls `autodiff_deferred`,
# and GPUCompiler's `deferred_codegen` pass resolves the marker call emitted by
# `_deferred_codegen_call` to the differentiated function, compiled from the job registered
# under the marker's id and linked into the kernel module. There is no host runtime path (and
# none is needed: higher-order differentiation goes through plain `autodiff`, whose nested
# calls Enzyme recognizes by itself). Outside of a GPUCompiler compilation the marker resolves
# to GPUCompiler's host stub, which hands the id back unchanged as a "pointer";
# `deferred_codegen` detects that by emitting the marker twice, under the id and its twin
# (`deferred_twin_id`), and raising `DeferredOnHostError` when the two results differ.
#
# The id is derived from the job's specification, which is entirely type-level, so it is the
# same in every session. `deferred_codegen_jobs` is process-local; `rebuild_deferred_registry!`
# recreates its entries from the specializations of `deferred_id_codegen`.

struct DeferredSpec
    FA::Type
    A::Type
    TT::Type
    mode::API.CDerivativeMode
    width::Int
    modified_between::Tuple{Vararg{Bool}}
    return_primal::Bool
    shadow_init::Bool
    expected_tape_type::Type
    err_if_func_written::Bool
    runtime_activity::Bool
    strong_zero::Bool
end

function Base.hash(spec::DeferredSpec, h::UInt)
    h = hash(spec.FA, h)
    h = hash(spec.A, h)
    h = hash(spec.TT, h)
    h = hash(Int(spec.mode), h)
    h = hash(spec.width, h)
    h = hash(spec.modified_between, h)
    h = hash(spec.return_primal, h)
    h = hash(spec.shadow_init, h)
    h = hash(spec.expected_tape_type, h)
    h = hash(spec.err_if_func_written, h)
    h = hash(spec.runtime_activity, h)
    return hash(spec.strong_zero, h)
end

# Set in every Enzyme deferred id, so ids never collide with the small counters GPUCompiler
# hands out for its own deferred jobs.
const DEFERRED_ID_BIT = UInt(1) << (8 * sizeof(UInt) - 2)

# Non-negative, so it round-trips through the `Int`-keyed `deferred_codegen_jobs`.
deferred_id(spec::DeferredSpec) = Int((hash(spec) | DEFERRED_ID_BIT) & ~(UInt(1) << (8 * sizeof(UInt) - 1)))

# The second id every job is registered under; see `deferred_codegen`.
deferred_twin_id(id::Int) = id ⊻ 1
deferred_twin_id(id::UInt) = id ⊻ UInt(1)

const DEFERRED_LOCK = ReentrantLock()
const DEFERRED_SPECS = Dict{Int, DeferredSpec}()

struct DeferredOnHostError <: EnzymeError end

function Base.showerror(io::IO, ::DeferredOnHostError)
    return print(
        io,
        "autodiff_deferred (and autodiff_deferred_thunk) can only be used inside GPU kernels. ",
        "Use autodiff on the host, also for higher-order differentiation.",
    )
end

# The compiler job for `spec` at `world`; `nothing` if the primal has no method for the
# signature, or a `String` describing why the request is invalid.
function deferred_job(spec::DeferredSpec, world::UInt)
    ft = eltype(spec.FA)
    primal_tt = Tuple{map(eltype, spec.TT.parameters)...}
    sugar = spec.mode == API.DEM_ForwardMode ? Forward : Reverse

    min_world = Ref{UInt}(typemin(UInt))
    max_world = Ref{UInt}(typemax(UInt))
    mi = my_methodinstance(sugar, ft, primal_tt, world, min_world, max_world)
    mi === nothing && return nothing

    A = spec.A
    rt2 = if A isa UnionAll
        rrt = primal_return_type_world(sugar, world, mi)

        # Don't error here but default to nothing return since in cuda context we don't use the device overrides
        if rrt == Union{}
            rrt = Nothing
        end

        if !(A <: Const) && guaranteed_const_nongen(rrt, world)
            return "Return type `$rrt` not marked Const, but type is guaranteed to be constant"
        end
        instantiate_annotation(A, rrt, spec.width)
    else
        @assert A isa DataType
        A
    end

    params = EnzymeCompilerParams(
        PrimalCompilerParams(spec.mode),
        Tuple{spec.FA, spec.TT.parameters...},
        spec.mode,
        spec.width,
        rt2,
        true,
        true,
        spec.modified_between,
        spec.return_primal,
        spec.shadow_init,
        spec.expected_tape_type,
        FFIABI,
        spec.err_if_func_written,
        spec.runtime_activity,
        spec.strong_zero
    ) #=abiwrap=#
    return CompilerJob(mi, CompilerConfig(EnzymeTarget(), params; kernel = false), world)
end

# Register `job` for the marker id of `spec` and return the id.
function register_deferred!(spec::DeferredSpec, job::CompilerJob)
    id = deferred_id(spec)
    lock(DEFERRED_LOCK)
    try
        prev = get(DEFERRED_SPECS, id, nothing)
        if prev !== nothing && prev != spec
            error("deferred codegen id $id is claimed by two specifications: $prev and $spec")
        end
        DEFERRED_SPECS[id] = spec
        # The same job object under both ids, so the deferred codegen pass compiles it once
        # and resolves both markers to one function.
        deferred_codegen_jobs[id] = job
        deferred_codegen_jobs[deferred_twin_id(id)] = job
    finally
        unlock(DEFERRED_LOCK)
    end
    return id
end

# The specification a `deferred_id_codegen` specialization was created for, or `nothing` when
# the specialization is not fully concrete.
function deferred_spec(@nospecialize(specTypes::Type))
    ps = specTypes.parameters
    length(ps) == 13 || return nothing
    vals = Vector{Any}(undef, 12)
    for i in 2:13
        p = ps[i]
        p isa DataType || return nothing
        if i in (2, 3, 4, 10)
            p <: Type || return nothing
            isempty(p.parameters) && return nothing
            vals[i - 1] = p.parameters[1]
        else
            p <: Val || return nothing
            isempty(p.parameters) && return nothing
            vals[i - 1] = p.parameters[1]
        end
    end
    return DeferredSpec(vals...)
end

"""
    rebuild_deferred_registry!(world = Base.get_world_counter()) -> Int

Register a compiler job for every `deferred_id_codegen` specialization that exists in this
process, e.g. after the registry was cleared, and return how many were registered. A kernel
whose code was loaded from a package image refers to its deferred ids without having run the
generator that registers them in this session.
"""
function rebuild_deferred_registry!(world::UInt = Base.get_world_counter())
    n = 0
    for mi in Base.specializations(only(methods(deferred_id_codegen)))
        spec = deferred_spec(mi.specTypes)
        spec === nothing && continue
        job = deferred_job(spec, world)
        job isa CompilerJob || continue
        register_deferred!(spec, job)
        n += 1
    end
    return n
end
