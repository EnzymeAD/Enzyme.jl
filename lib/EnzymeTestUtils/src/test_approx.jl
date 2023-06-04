function test_approx(x::Number, y::Number; kwargs...)
    @test isapprox(x, y; kwargs...)
    return nothing
end
function test_approx(x::AbstractArray{<:Number}, y::AbstractArray{<:Number}; kwargs...)
    @test isapprox(x, y; kwargs...)
    return nothing
end
function test_approx(x::AbstractArray, y::AbstractArray; kwargs...)
    test_approx.(x, y; kwargs...)
    return nothing
end
function test_approx(x, y; kwargs...)
    names = fieldnames(typeof(x))
    if isempty(names)
        @test x === y || x == y
    else
        for k in names
            test_approx(getfield(x, k), getfield(y, k); kwargs...)
        end
    end
    return nothing
end
