using Enzyme, Test

@noinline function force_stup(A)
    A11 = A[];
    return (A11, 0.0)
end

@noinline function mul2x2(y, x)
    Aelements = force_stup(x)

    A11 = Aelements[1]

    unsafe_store!(y, A11*A11)

    nothing
end

function f_exc(x)
    y = Base.reinterpret(Ptr{Float64}, Libc.malloc(8))

    mul2x2(y, x)

    ld = unsafe_load(y)

    return ld * ld
end

@testset "No JLValueT Calling Conv" begin
	y = Ref(1.0)
	f_x = make_zero(y)
	Enzyme.autodiff(Reverse, f_exc, Duplicated(y, f_x))

	@test f_x[] ≈ 4.0
end

struct Inner
    a::Float64
    b::Float64
end

struct Outer
    inner::Inner
    c::Float64
end

@noinline function force_stup_multi(A)
    return (A.inner, 0.0)
end

@noinline function process_multi(x)
    Aelements = force_stup_multi(x)
    val = Aelements[1].a
    return val * val
end

function wrapper_multi(x)
    return process_multi(x)
end

@testset "Multi-index ExtractValue" begin
    x = Outer(Inner(2.0, 3.0), 4.0)
    dx = Enzyme.gradient(Reverse, wrapper_multi, x)
    @test dx[1].inner.a ≈ 4.0
end

struct StructWithPtr
    p::Vector{Float64}
    x::Float64
end

@noinline function f_unboxed_rooted(x, y)
    return x.p[1] * x.x * y
end

@testset "Unboxed aggregate with rooted param" begin
    x = StructWithPtr([2.0], 3.0)
    y = 4.0
    dx, dy = Enzyme.gradient(Reverse, f_unboxed_rooted, x, y)
    @test dx.x ≈ 8.0
    @test dy ≈ 6.0
end

struct MyFill{T,A}
    val::T
    axes::A
end
struct MyTrnc
    tp::Float64
    logtp::Float64
end
@noinline function trnc(l::Float64)
    # Call a C function from libc that takes a double and returns a double to block constprop
	lcdf = ccall("extern sin", llvmcall, Float64, (Float64,), l)
    MyTrnc(lcdf, lcdf)
end
@noinline function lpdf(dists::MyFill, x::Vector{Float64})
    sz = length(dists.axes)
    return @inbounds x[1] + sz
end
function f_sret_nested(x)
    dists = MyFill(trnc(0.0), 1:2)
    return lpdf(dists, x)
end

# This test is required because the sret alloca from trnc is actually an alloca of the MyFill, and just
# fills the first sizeof(MyTrnc) bytes rather than exclusively being an alloca for MyTrnc. As a result,
# we need to make sure the type propagation up from the sret return doesn't assume [-1, -1]:Pointer.
@testset "Sret Nested Struct Type Analysis" begin
    x = [0.5, 0.3]
    dx = Enzyme.gradient(Reverse, f_sret_nested, x)
    @test dx[1] == [1.0, 0.0]
end

@noinline function rec(x, ps)
    if length(ps) == 0
        return (x,), x, 1
    end
    rest, lj2, idx2 = rec(x, Base.tail(ps))
    return (x, rest...), x, idx2
end

function objective(y, ps)
    res = rec(y, ps)
    return res[2]
end

@testset "Recursive Struct Sret" begin
    y0 = [1.0]
    ps = ((1.0, 1.0), (2.0, 2.0))
    dx = Enzyme.gradient(Reverse, objective, y0[1], ps)
    @test dx == (1.0, ((0.0, 0.0), (0.0, 0.0)))
end
