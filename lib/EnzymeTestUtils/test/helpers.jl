using LinearAlgebra

struct TestStruct{X, A}
    x::X
    a::A
end

struct TestStruct2
    x::Any
    a::Any
    TestStruct2(x) = new(x)
end

mutable struct MutableTestStruct
    x::Any
    a::Any
    MutableTestStruct() = new()
end

struct MutatedCallable{T}
    x::T
end
function (c::MutatedCallable)(y)
    s = c.x'y
    c.x ./= s
    return s
end

f_array(x) = sum(abs2, x)
f_multiarg(x::AbstractArray, a) = abs2.(a .* x)

function f_structured_array(x::Hermitian)
    y = x * 3
    # mutate the unused triangle, which ensures that our Jacobian differs from FiniteDifferences
    if y.uplo == 'U'
        LowerTriangular(y.data) .*= 2
    else
        UpperTriangular(y.data) .*= 2
    end
    return y
end

function f_structured_nan(x::Hermitian)
    new = Matrix{Float32}(undef, 2, 2)
    new[1, 1] = parent(x)[1, 1]
    new[1, 2] = parent(x)[1, 2]
    new[2, 1] = NaN
    new[2, 2] = parent(x)[2, 2]
    return Hermitian(new)
end
