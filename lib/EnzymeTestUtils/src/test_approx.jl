test_approx(x, y; kwargs...) = test_approx(x, y, ""; kwargs...)
function test_approx(x::Number, y::Number, msg; kwargs...)
    @test_msg msg isapprox(x, y; kwargs...)
    return nothing
end
function test_approx(x::Array{<:Number}, y::Array{<:Number}, msg; kwargs...)
    @test_msg msg isapprox(x, y; kwargs...)
    return nothing
end
@static if VERSION < v"1.11-"
else
using LinearAlgebra
function test_approx(x::LinearAlgebra.HermOrSym{<:Number}, y::LinearAlgebra.HermOrSym{<:Number}, msg; kwargs...)
    # on 1.11+ similar of a hermitian or symmetric leaves the other part of the array undefined
    # this mean that when we try to compare that data, we compare undefined memory -- which may be different
    # here let's explicitly zero it to make the test more meaningful.
    x2 = zero(parent(x))
    y2 = zero(parent(y))
    copyto!(x2, x)
    copyto!(y2, y)
    test_approx(x2, y2, msg; kwargs...)
    return nothing
end
end
function test_approx(x::AbstractArray{<:Number}, y::AbstractArray{<:Number}, msg; kwargs...)
    @test_msg msg isapprox(x, y; kwargs...)
    # for custom array types, fields should also match
    _test_fields_approx(x, y, msg; kwargs...)
    return nothing
end
function test_approx(x::AbstractArray, y::AbstractArray, msg; kwargs...)
    @test_msg "$msg: indices must match" eachindex(x) == eachindex(y)
    for i in eachindex(x)
        msg_new = "$msg: ::$(typeof(x))[$i]"
        test_approx(x[i], y[i], msg_new; kwargs...)
    end
    return nothing
end
function test_approx(x::Tuple, y::Tuple, msg; kwargs...)
    @test_msg "$msg: lengths must match" length(x) == length(y)
    for i in eachindex(x)
        msg_new = "$msg: ::$(typeof(x))[$i]"
        test_approx(x[i], y[i], msg_new; kwargs...)
    end
    return nothing
end
function test_approx(x::Dict, y::Dict, msg; kwargs...)
    @test_msg "$msg: keys must match" issetequal(keys(x), keys(y))
    for k in keys(x)
        msg_new = "$msg: ::$(typeof(x))[$k]"
        test_approx(x[k], y[k], msg_new; kwargs...)
    end
    return nothing
end
function test_approx(x::Type, y::Type, msg; kwargs...)
    @test_msg "$msg: types must match" x === y
    return nothing
end
test_approx(x, y, msg; kwargs...) = _test_fields_approx(x, y, msg; kwargs...)

function _test_fields_approx(x, y, msg; kwargs...)
    @test_msg "$msg: types must match" typeof(x) == typeof(y)
    names = fieldnames(typeof(x))
    if isempty(names)
        @test_msg msg x == y
    else
        for k in names
            if k isa Symbol && hasproperty(x, k)
                msg_new = "$msg: ::$(typeof(x)).$k"
            else
                msg_new = "$msg: getfield(::$(typeof(x)), $k)"
            end
            test_approx(getfield(x, k), getfield(y, k), msg_new; kwargs...)
        end
    end
    return nothing
end
