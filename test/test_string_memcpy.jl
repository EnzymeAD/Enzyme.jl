using Enzyme
using Test

@testset "String memcpy sizeof crash" begin
    function my_copy(p1::Ptr{String}, p2::Ptr{String})
        unsafe_copyto!(p1, p2, 10)
        return nothing
    end

    p1 = Libc.malloc(10 * sizeof(Ptr{Cvoid}))
    p2 = Libc.malloc(10 * sizeof(Ptr{Cvoid}))
    try
        @test_nowarn autodiff(Reverse, my_copy, Const(convert(Ptr{String}, p1)), Const(convert(Ptr{String}, p2)))
    finally
        Libc.free(p1)
        Libc.free(p2)
    end
end
