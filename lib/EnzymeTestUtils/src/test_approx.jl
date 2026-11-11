test_approx(x, y; kwargs...) = _test_approx(x, y, "", IdDict(); kwargs...)
test_approx(x, y, msg; kwargs...) = _test_approx(x, y, msg, IdDict(); kwargs...)

function _test_approx(x::Number, y::Number, msg, seen::IdDict; kwargs...)
    @test_msg msg isapprox(x, y; kwargs...)
    return nothing
end
function _test_approx(x::Array{<:Number}, y::Array{<:Number}, msg, seen::IdDict; kwargs...)
    @test_msg msg isapprox(x, y; kwargs...)
    return nothing
end
using LinearAlgebra
function zero_copy(x)
    y = zero(parent(x))
    copyto!(x.uplo == 'U' ? UpperTriangular(y) : LowerTriangular(y), x.uplo == 'U' ? UpperTriangular(x) : LowerTriangular(x))
    y
end
function _test_approx(x::LinearAlgebra.HermOrSym{<:Number}, y::LinearAlgebra.HermOrSym{<:Number}, msg, seen::IdDict; kwargs...)
    # on 1.11+ similar of a hermitian or symmetric leaves the other part of the array undefined
    # this mean that when we try to compare that data, we compare undefined memory -- which may be different
    # here let's explicitly zero it to make the test more meaningful.
    x2 = zero_copy(x)
    y2 = zero_copy(y)
    _test_approx(x2, y2, msg, seen; kwargs...)
    return nothing
end
function _test_approx(x::AbstractArray{<:Number}, y::AbstractArray{<:Number}, msg, seen::IdDict; kwargs...)
    @test_msg msg isapprox(x, y; kwargs...)
    # for custom array types, fields should also match
    _test_fields_approx(x, y, msg, seen; kwargs...)
    return nothing
end
function _test_approx(x::AbstractArray, y::AbstractArray, msg, seen::IdDict; kwargs...)
    if ismutable(x) && ismutable(y)
        haskey(seen, (x, y)) && return nothing
        seen[(x, y)] = nothing
    end
    @test_msg "$msg: indices must match" eachindex(x) == eachindex(y)
    for i in eachindex(x)
        msg_new = "$msg: ::$(typeof(x))[$i]"
        @test_msg "$msg_new: assignedness must match" isassigned(x, i) == isassigned(y, i)
        if isassigned(x, i) && isassigned(y, i)
            _test_approx(x[i], y[i], msg_new, seen; kwargs...)
        end
    end
    return nothing
end
function _test_approx(x::Tuple, y::Tuple, msg, seen::IdDict; kwargs...)
    @test_msg "$msg: lengths must match" length(x) == length(y)
    for i in eachindex(x)
        msg_new = "$msg: ::$(typeof(x))[$i]"
        _test_approx(x[i], y[i], msg_new, seen; kwargs...)
    end
    return nothing
end
function _test_approx(x::Dict, y::Dict, msg, seen::IdDict; kwargs...)
    if ismutable(x) && ismutable(y)
        haskey(seen, (x, y)) && return nothing
        seen[(x, y)] = nothing
    end
    @test_msg "$msg: keys must match" issetequal(keys(x), keys(y))
    for k in keys(x)
        msg_new = "$msg: ::$(typeof(x))[$k]"
        _test_approx(x[k], y[k], msg_new, seen; kwargs...)
    end
    return nothing
end
function _test_approx(x::Type, y::Type, msg, seen::IdDict; kwargs...)
    @test_msg "$msg: types must match" x === y
    return nothing
end
_test_approx(x, y, msg, seen::IdDict; kwargs...) = _test_fields_approx(x, y, msg, seen; kwargs...)

function _test_fields_approx(x, y, msg, seen::IdDict = IdDict(); kwargs...)
    @test_msg "$msg: types must match" typeof(x) == typeof(y)
    if ismutable(x) && ismutable(y)
        haskey(seen, (x, y)) && return nothing
        seen[(x, y)] = nothing
    end
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
            @test_msg "$msg_new: definedness must match" isdefined(x, k) == isdefined(y, k)
            if isdefined(x, k) && isdefined(y, k)
                _test_approx(getfield(x, k), getfield(y, k), msg_new, seen; kwargs...)
            end
        end
    end
    return nothing
end
