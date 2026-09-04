using Enzyme, Test

# FFIABI thunks hold a `ThunkHandle` rather than a function pointer. The handle names the
# job; `thunk_pointer` links (or compiles) it in the current session, so a thunk object
# embedded in a package image, or kept across a session reset, still works. Differentiating
# a call of a thunk binds it statically, as before.

const C = Enzyme.Compiler
const THUNK_CACHE = C.THUNK_CACHE

th_sq(x) = x * x
th_cube(x) = x * x * x

@testset "handles link lazily and survive a session reset" begin
    C.reset_session!()
    thunk = C.thunk(Val(0), Const{typeof(th_sq)}, Active, Tuple{Active{Float64}}, Val(Enzyme.API.DEM_ReverseModeCombined), Val(1), Val((false, false)), Val(false), Val(false), DefaultABI, Val(false), Val(false), Val(false))
    h = thunk.adjoint
    @test h isa C.ThunkHandle
    @test h.which === :adjoint
    @test h.mi.specTypes.parameters[1] === typeof(th_sq)
    l = C.current_link(h)
    @test l !== nothing && l.ptr != C_NULL && l.epoch == C.SESSION_EPOCH[]
    @test C.thunk_pointer(h) == l.ptr
    @test thunk(Const(th_sq), Active(3.0), 1.0) == ((6.0,),)

    # A new session: the link is stale, the next call recompiles and relinks the same handle.
    C.reset_session!()
    @test isempty(THUNK_CACHE.thunks)
    @test C.current_link(h) === nothing
    @test thunk(Const(th_sq), Active(3.0), 1.0) == ((6.0,),)
    l2 = C.current_link(h)
    @test l2 !== nothing && l2.epoch == C.SESSION_EPOCH[] && l2.epoch != l.epoch
    @test length(THUNK_CACHE.thunks) == 1
    @test C.thunk_pointer(h) == l2.ptr

    # Split mode gives a primal and an adjoint handle.
    fwd, rev = C.thunk(Val(0), Const{typeof(th_sq)}, Active, Tuple{Active{Float64}}, Val(Enzyme.API.DEM_ReverseModeGradient), Val(1), Val((false, false)), Val(false), Val(false), DefaultABI, Val(false), Val(false), Val(false))
    @test fwd.primal isa C.ThunkHandle && fwd.primal.which === :primal
    @test rev.adjoint isa C.ThunkHandle && rev.adjoint.which === :adjoint
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
    for r in values(THUNK_CACHE.thunks), h in (r.adjoint, r.primal)
        h isa C.ThunkHandle || continue
        h.mi.specTypes.parameters[1] === typeof(th_grad) || continue
        l = C.current_link(h)
        l === nothing || push!(outer, l.modstr)
    end
    @test length(outer) == 1
    ir = only(outer)
    @test !occursin("thunk_pointer", ir)
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
