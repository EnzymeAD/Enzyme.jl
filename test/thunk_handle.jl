using Enzyme, Test

# FFIABI thunks hold their entry-point `CodeInstance` (a `ThunkHandle` on Julia 1.10 and
# 1.11) rather than a function pointer. A call links (or compiles) the thunk in the current
# session when its entry is not published, so a thunk object embedded in a package image, or
# kept across a session reset, still works. Differentiating a call of a thunk binds it
# statically, as before.

const C = Enzyme.Compiler
const THUNK_CACHE = C.THUNK_CACHE

th_sq(x) = x * x
th_cube(x) = x * x * x

@testset "entry points link lazily and survive a session reset" begin
    C.reset_session!()
    thunk = C.thunk(Val(0), Const{typeof(th_sq)}, Active, Tuple{Active{Float64}}, Val(Enzyme.API.DEM_ReverseModeCombined), Val(1), Val((false, false)), Val(false), Val(false), DefaultABI, Val(false), Val(false), Val(false))
    e = thunk.adjoint
    if C.HAS_CI_THUNKS
        @test e isa Core.CodeInstance
        @test e.owner isa C.ThunkEntryOwner
        @test C.thunk_key_of(e)[2] === :adjoint
    else
        @test e isa C.ThunkHandle
        @test e.which === :adjoint
    end
    @test C.thunk_primal_type(e) === typeof(th_sq)
    l = C.thunk_link(e)
    @test l !== nothing && l.ptr != C_NULL && l.epoch == C.SESSION_EPOCH[]
    @test C.thunk_fptr(e) == l.ptr
    @test thunk(Const(th_sq), Active(3.0), 1.0) == ((6.0,),)

    # A new session: the link is gone, the next call recompiles and relinks the same entry.
    C.reset_session!()
    @test isempty(THUNK_CACHE.thunks)
    @test C.thunk_link(e) === nothing
    C.HAS_CI_THUNKS && @test e.specptr == C_NULL
    @test thunk(Const(th_sq), Active(3.0), 1.0) == ((6.0,),)
    l2 = C.thunk_link(e)
    @test l2 !== nothing && l2.ptr != C_NULL
    @test length(THUNK_CACHE.thunks) == 1
    @test C.thunk_fptr(e) == l2.ptr

    # Split mode gives a primal and an adjoint entry.
    fwd, rev = C.thunk(Val(0), Const{typeof(th_sq)}, Active, Tuple{Active{Float64}}, Val(Enzyme.API.DEM_ReverseModeGradient), Val(1), Val((false, false)), Val(false), Val(false), DefaultABI, Val(false), Val(false), Val(false))
    which(x) = C.HAS_CI_THUNKS ? C.thunk_key_of(x)[2] : x.which
    @test which(fwd.primal) === :primal
    @test which(rev.adjoint) === :adjoint
    tape, = fwd(Const(th_sq), Active(3.0))
    @test rev(Const(th_sq), Active(3.0), 1.0, tape) == ((6.0,),)
end

@testset "primal error thunks" begin
    th_err(x) = error("boom $x")
    @test_throws ErrorException autodiff(Reverse, th_err, Active(1.0))
end

# Nested differentiation: a differentiated function that itself calls a thunk sees the
# thunk's code, not an opaque pointer.
th_grad(x) = autodiff(Reverse, th_cube, Active(x))[1][1]

@testset "nested differentiation stays static" begin
    C.reset_session!()
    @test autodiff(Forward, th_grad, Duplicated(2.0, 1.0))[1] ≈ 12.0
    outer = String[]
    for e in C.thunk_entries(th_grad)
        l = C.thunk_link(e)
        l === nothing || push!(outer, l.modstr)
    end
    @test length(outer) == 1
    ir = only(outer)
    @test !occursin("thunk_pointer", ir)
    @test !occursin("thunk_link", ir)
    # No call through a literal address, and the inner thunk's code is present.
    @test !occursin(r"call[^\n]*inttoptr \(i64 \d{9,} to (ptr|[^\n]*\*)\)\(", ir)
    @test occursin("th_cube", ir)
end

# The scenario of issue #1549: a thunk compiled while a package image is generated is
# embedded in that image; a fresh session must compile it anew rather than use a dead pointer.
@testset "thunks embedded in a package image" begin
    load_path = mktempdir()
    depot = mktempdir()
    pkg = "EnzymeThunkImageTest"
    write(
        joinpath(load_path, "$pkg.jl"),
        """
        module $pkg
        using Enzyme
        sq(x) = x * x
        grad(x) = Enzyme.autodiff(Reverse, sq, Active(x))[1][1]
        # Run during precompilation, so the generated thunk lands in the image.
        const PRECOMPILED = grad(2.0)
        end
        """
    )
    code = """
    pushfirst!(LOAD_PATH, $(repr(load_path)))
    using $pkg
    print($pkg.grad(3.0), " ", $pkg.PRECOMPILED)
    """
    # A fresh first depot receives the package image; the others stay readable (a trailing
    # colon alone would not keep the user depot on 1.12).
    cmd = addenv(
        `$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $code`,
        "JULIA_DEPOT_PATH" => join([depot; DEPOT_PATH], Sys.iswindows() ? ";" : ":"),
    )
    # First run precompiles the package (and runs the workload), the second loads the image.
    first = read(cmd, String)
    @test endswith(first, "6.0 4.0")
    second = read(cmd, String)
    @test endswith(second, "6.0 4.0")
    # The image was written to the fresh depot (a package without a UUID caches as a flat file).
    compiled = joinpath(depot, "compiled", "v$(VERSION.major).$(VERSION.minor)")
    @test any(startswith(pkg), readdir(compiled))
end

# On Julia 1.12+ a compiled thunk's entry point is stored in the `specptr` of a
# `CodeInstance` of `enzyme_thunk_entry`, whose method instance is derived from the job, the
# same shape Julia gives a natively compiled function.
@testset "entry points live in CodeInstances" begin
    C.HAS_CI_THUNKS || return
    C.reset_session!()
    @test autodiff(Reverse, th_sq, Active(3.0))[1][1] == 6.0
    e = only(C.thunk_entries(th_sq))
    l = C.thunk_link(e)
    @test l.ci === e
    @test e.specptr == l.ptr != C_NULL
    @test e.invoke != C_NULL
    # The entry is Enzyme-owned: Julia never dispatches to it through `enzyme_thunk_entry`.
    @test e.owner isa C.ThunkEntryOwner
    @test C.enzyme_thunk_entry(Int, Val(:adjoint)) === nothing
    # The entry is published the way Julia publishes one, so the runtime's own reader
    # returns it instead of waiting forever for the flag that says `invoke` and `specptr`
    # agree (it spins on that flag whenever both are set).
    flags = Ref{UInt8}(0)
    invoke = Ref{Ptr{Cvoid}}(C_NULL)
    specptr = Ref{Ptr{Cvoid}}(C_NULL)
    @ccall jl_read_codeinst_invoke(
        e::Any, flags::Ptr{UInt8}, invoke::Ptr{Ptr{Cvoid}}, specptr::Ptr{Ptr{Cvoid}}, 0::Cint
    )::Cvoid
    @test specptr[] == l.ptr
    @test flags[] & C.CI_INVOKE_MATCHES_SPECPTR != 0
    # `invoke` is a real boxed-ABI wrapper: invoking the instance runs the thunk on boxed
    # arguments and gives what a call of the thunk gives.
    K, which = C.thunk_key_of(e)
    @test which === :adjoint
    expected = autodiff(Reverse, th_sq, Active(3.0))
    @test Core.invoke(C.enzyme_thunk_entry, e, K, Val(which), Const(th_sq), Active(3.0), 1.0) == expected

    # The instance is derived from the job: the same thunk finds the same instance, and it
    # is valid from the job's world on.
    job, _ = C.thunk_job(e)
    @test C.thunk_entry_ci(job, :adjoint) === e
    @test C.thunk_entry_ci(job, :primal) === nothing
    @test e.min_world <= Base.get_world_counter()
    # The instance carries the derivative's edges, the differentiated method among them.
    @test any(x -> x isa Core.MethodInstance && x.specTypes.parameters[1] === typeof(th_sq), e.edges)

    # A new session reuses the instance and refreshes its entry point.
    C.reset_session!()
    @test e.specptr == C_NULL
    @test autodiff(Reverse, th_sq, Active(5.0))[1][1] == 10.0
    @test only(C.thunk_entries(th_sq)) === e
    @test e.specptr == C.thunk_link(e).ptr != C_NULL
    @test Core.invoke(C.enzyme_thunk_entry, e, K, Val(which), Const(th_sq), Active(5.0), 1.0) ==
        autodiff(Reverse, th_sq, Active(5.0))
end
