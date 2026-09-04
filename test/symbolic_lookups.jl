using Enzyme, LinearAlgebra, Test

# The module Enzyme hands to its JIT must not bind C callees to addresses of this session:
# a call through a literal pointer that `check_ir` could attribute to a symbol in a
# library becomes an `ejlstr$name$library` declaration, resolved when the module is
# linked, and calls into Julia's runtime or the C library keep their plain names for the
# JIT to resolve from the process.

const THUNK_CACHE = Enzyme.Compiler.THUNK_CACHE

# The modules of the thunks compiled for `f` in this session, as kept for nested differentiation.
function thunk_modules(f)
    irs = String[]
    for r in values(THUNK_CACHE.thunks), h in (r.adjoint, r.primal)
        h isa Enzyme.Compiler.ThunkHandle || continue
        h.mi.specTypes.parameters[1] === typeof(f) || continue
        l = Enzyme.Compiler.current_link(h)
        l === nothing || push!(irs, l.modstr)
    end
    return unique(irs)
end

# The IR of the thunk compiled for `f`.
function last_thunk_ir(f, args...)
    autodiff(Reverse, f, args...)
    return only(thunk_modules(f))
end

# A call whose callee is a literal address (opaque or typed pointers).
const LITERAL_CALLEE = r"call[^\n]*inttoptr \(i64 \d{9,} to (ptr|[^\n]*\*)\)\("

literal_callees(ir) = [m.match for m in eachmatch(LITERAL_CALLEE, ir)]
symbolic_decls(ir) = [m.captures[1] for m in eachmatch(r"declare[^\n]*@\"?(ejlstr\$[^\"( ]+)", ir)]
plain_decls(ir) = [m.captures[1] for m in eachmatch(r"declare[^\n]*@(i?jl_[A-Za-z0-9_]+|malloc|free|memcpy|memmove|memset|memcmp)\b", ir)]

sl_memcmp(x) = (a = [x[1], x[2]]; b = [x[2], x[1]]; ccall(:memcmp, Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t), a, b, 16) == 0 ? x[1] : x[2])
sl_blas(x) = dot(x, x)
sl_alloc(x) = sum(exp.(x) ./ (1 .+ x .^ 2))
function sl_malloc(x)
    p = Libc.malloc(8)
    Base.unsafe_store!(Ptr{Float64}(p), x[1] * 2)
    v = Base.unsafe_load(Ptr{Float64}(p))
    Libc.free(p)
    return v
end

x = [1.0, 2.0]

@testset "C symbols are declared, not bound" begin
    # libc is loaded globally, so the JIT resolves `memcmp` by its plain name.
    ir = last_thunk_ir(sl_memcmp, Duplicated(x, zero(x)))
    @test isempty(literal_callees(ir))
    @test "memcmp" in plain_decls(ir)
    @test !occursin(Enzyme.Compiler.NONRELOCATABLE_FLAG, ir)

    # libblastrampoline is not, so its symbols are declared with their library.
    ir = last_thunk_ir(sl_blas, Duplicated(x, zero(x)))
    @test isempty(literal_callees(ir))
    @test any(d -> occursin("blastrampoline", d), symbolic_decls(ir))
    @test occursin(r"\"enzyme_math\"=\"[a-z_0-9]*(dot|copy|axpy)", ir)
    @test !occursin(Enzyme.Compiler.NONRELOCATABLE_FLAG, ir)
    @test autodiff(Reverse, sl_blas, Duplicated(x, zero(x))) isa Tuple
end

@testset "runtime and libc symbols keep their names" begin
    ir = last_thunk_ir(sl_alloc, Duplicated(x, zero(x)))
    @test isempty(literal_callees(ir))
    @test !isempty(plain_decls(ir))

    # Type analysis of the malloc'd store/load pair is not supported on 1.10.
    @static if VERSION >= v"1.11"
        ir = last_thunk_ir(sl_malloc, Duplicated(x, zero(x)))
        @test isempty(literal_callees(ir))
        @test any(d -> d in ("malloc", "free"), plain_decls(ir))
        @test !occursin(Enzyme.Compiler.NONRELOCATABLE_FLAG, ir)
    end
end

@testset "results are unchanged" begin
    dx = zero(x)
    autodiff(Reverse, sl_alloc, Duplicated(x, dx))
    @test dx ≈ [(exp(1.0) * (1 + 1) - exp(1.0) * 2) / 4, (exp(2.0) * 5 - exp(2.0) * 4) / 25]
    dx = zero(x)
    autodiff(Reverse, sl_blas, Duplicated(x, dx))
    @test dx ≈ 2 .* x
    @static if VERSION >= v"1.11"
        dx = zero(x)
        autodiff(Reverse, sl_malloc, Duplicated(x, dx))
        @test dx ≈ [2.0, 0.0]
    end
end
