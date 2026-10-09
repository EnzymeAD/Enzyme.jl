# A reverse rule returning the cotangent of an `Active` struct argument that is
# passed by reference. Accumulating it into the shadow must use the alignment
# of the Julia type, not the ABI alignment of the vector type Enzyme combines
# the fields into (`<4 x double>`, align 32), or AVX targets fault on an
# aligned store into the 8-aligned shadow slot.
# https://github.com/EnzymeAD/Enzyme.jl/issues/3775
module ActiveStructAlign

using Enzyme
using Enzyme.EnzymeCore: EnzymeRules
using Test

struct S4
    a::Float64
    b::Float64
    c::Float64
    d::Float64
end

nrm(s::S4) = sqrt(s.a^2 + s.b^2 + s.c^2 + s.d^2)

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(nrm)}, ::Type{<:Active}, s::Active{S4}
    )
    h = func.val(s.val)
    return EnzymeRules.AugmentedReturn(EnzymeRules.needs_primal(config) ? h : nothing, nothing, h)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(nrm)}, dret::Active, h, s::Active{S4}
    )
    x, d = s.val, dret.val / h
    return (S4(x.a * d, x.b * d, x.c * d, x.d * d),)
end

@noinline lin(s::S4) = s.a - 2s.b + 3s.c - 4s.d
f(s) = lin(s) + nrm(s)

# Whether the shadow slot happens to be 32-aligned depends on the stack, so a
# run alone does not catch this reliably. Check the IR instead: no vector load
# or store (the combined fields) may claim more alignment than the memory can
# have. That bound is not `datatype_alignment(S4)`: LLVM may legitimately
# claim more when it knows the underlying allocation is more aligned (on i686
# `S4` is 4-aligned, yet the accesses carry `align 8`). Julia never aligns an
# allocation beyond 16 bytes, so the 32 of `<4 x double>` is never justified.
const MAX_ALIGN = max(Base.datatype_alignment(S4), 16)

function overaligned_accesses(ir)
    pat = r"(?:load|store) <\d+ x [^\n]*, align (\d+)"
    return [m.match for m in eachmatch(pat, ir) if parse(Int, m[1]) > MAX_ALIGN]
end

@testset "Active struct cotangent alignment" begin
    ir = sprint(io -> Enzyme.Compiler.enzyme_code_llvm(io, f, Active, Tuple{Active{S4}}; dump_module = true))
    @test isempty(overaligned_accesses(ir))

    x = S4(0.3, -0.7, 1.1, 0.4)
    n = nrm(x)
    g = Enzyme.gradient(Reverse, f, x)[1]
    @test g.a ≈ 1 + x.a / n
    @test g.b ≈ -2 + x.b / n
    @test g.c ≈ 3 + x.c / n
    @test g.d ≈ -4 + x.d / n
end

end # module ActiveStructAlign
