using LinearAlgebra

struct TestStruct{X,A}
    x::X
    a::A
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
