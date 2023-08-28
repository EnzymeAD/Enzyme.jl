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
