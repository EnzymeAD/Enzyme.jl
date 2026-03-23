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
