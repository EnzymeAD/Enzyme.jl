using Enzyme, LinearAlgebra, SHA, TOML, Test, Libdl

const EXPECTED_BACKEND = ENV["ENZYME_EXPECT_BACKEND"]
const jll = Enzyme.Compiler.Enzyme_jll

function summed_running_max(x)
    m = -Inf
    s = 0.0
    for i in eachindex(x)
        m = ifelse(x[i] > m, x[i], m)
        s += m
    end
    return s
end

gemm_overwrite!(C, A, B) = (BLAS.gemm!('N', 'N', 1.0, A, B, 0.0, C); nothing)

function cells(x, s, I)
    total = 0.0
    for i in I
        n = length(i:i)
        out = Vector{Float64}(undef, n)
        for j in 1:n
            @inbounds out[j] = x[i + j - 1] * s + 1.0
        end
        total += only(out)
    end
    return total
end

@noinline function three_way(x::Float64, y::Float64)
    x <= 0 && return y * y
    s = sqrt(x)
    c = max(s, y)
    if 1.0e-3 * c >= min(s, 2.0)
        return y * exp(s)
    else
        return s * y
    end
end
three_way_vector(v) = three_way(v[1], v[2])
quadratic(v) = sum(abs2, v)

println("JULIA ", VERSION, " LLVM ", Base.libllvm_version, " MACHINE ", Sys.MACHINE)
println("ENZYME ", pkgversion(Enzyme), " SOURCE ", pkgdir(Enzyme))
println("LIBENZYME ", jll.libEnzyme_path)
println("LIBENZYMEBCLOAD ", jll.libEnzymeBCLoad_path)

@testset "aarch64 dogfood provenance and canaries" begin
    # The native runner fixes this to aarch64; x86_64 is only for recipe smoke tests.
    @test Sys.ARCH == Symbol(get(ENV, "ENZYME_EXPECT_ARCH", "aarch64"))
    decl = TOML.parsefile(joinpath(pkgdir(Enzyme), "deps", "dogfood_libenzyme.toml"))
    @test decl["commit"] == EXPECTED_BACKEND
    build_root = get(ENV, "KB_LIBENZYME_DOGFOOD_ROOT", joinpath(homedir(), ".local", "state", "kb-agents", "libenzyme-dogfood"))
    build_dir = joinpath(build_root, EXPECTED_BACKEND, "llvm$(Base.libllvm_version.major)")
    @test realpath(jll.libEnzyme_path) == realpath(joinpath(build_dir, "libEnzyme-$(Base.libllvm_version.major).$(Libdl.dlext)"))
    @test realpath(jll.libEnzymeBCLoad_path) == realpath(joinpath(build_dir, "libEnzymeBCLoad-$(Base.libllvm_version.major).$(Libdl.dlext)"))
    checksums = Dict{String, String}()
    for line in eachline(joinpath(build_dir, "SHA256SUMS"))
        fields = split(line; limit = 2)
        length(fields) == 2 || error("Invalid checksum row: $line")
        checksums[lstrip(strip(fields[2]), '*')] = fields[1]
    end
    for library in (jll.libEnzyme_path, jll.libEnzymeBCLoad_path)
        digest = bytes2hex(open(sha256, library))
        @test checksums[basename(library)] == digest
        println("LIBRARY_SHA256 ", digest, " ", library)
    end

    control = Enzyme.gradient(Reverse, quadratic, [1.0, 2.0, 3.0])[1]
    @test control == [2.0, 4.0, 6.0]
    println("CANARY quadratic ", control)

    x = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 0.5, 7.0]
    x_before = copy(x)
    dx = zero(x)
    Enzyme.autodiff(Reverse, summed_running_max, Active, Duplicated(x, dx))
    @test dx == [1.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0]
    @test x == x_before
    println("CANARY running_max ", dx)

    v = [-1.0, 0.5]
    dv = zero(v)
    Enzyme.autodiff(Reverse, three_way_vector, Active, Duplicated(v, dv))
    @test dv == [0.0, 1.0]
    @test v == [-1.0, 0.5]
    println("CANARY three_way ", dv)

    x = [0.3, -1.2, 2.0, 0.7]
    dx = zero(x)
    Enzyme.autodiff(set_runtime_activity(Reverse), cells, Active, Duplicated(x, dx), Const(1.7), Const([2, 3]))
    @test dx == [0.0, 1.7, 1.7, 0.0]
    @test x == [0.3, -1.2, 2.0, 0.7]
    println("CANARY only_runtime_activity ", dx)

    A = [sin(i + 2j) for i in 1:3, j in 1:3]
    B = [cos(i - j) for i in 1:3, j in 1:3]
    C = zeros(3, 3)
    dC = fill(NaN, 3, 3)
    Enzyme.autodiff(Forward, Const(gemm_overwrite!), Duplicated(C, dC), Const(A), Const(B))
    @test dC == zeros(3, 3)
    @test C ≈ A * B
    println("CANARY gemm_nan ", dC)
end
println("CANARIES_COMPLETE")
