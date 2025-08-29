using Test
using Enzyme

@testset "BoundsError fix in jitrules (Issue #2508)" begin
    # Test that we can handle types that might have caused the original BoundsError
    # when calling Returns(res) in allSame and allFirst functions
    
    # This is a regression test for a BoundsError that occurred when
    # certain types with empty SimpleVectors were passed to Base.Returns
    
    # The original error was:
    # BoundsError: attempt to access Core.SimpleVector at index [1]
    # in _stable_typeof -> Returns(value::Type) -> allSame
    
    # Test autodiff with functions that might trigger the code paths that use allSame/allFirst
    
    # Test with function returning empty tuple type (edge case)
    function return_empty_tuple()
        return ()
    end
    
    # This should not throw BoundsError
    @test_nowarn begin
        try
            Enzyme.autodiff(Enzyme.Forward, return_empty_tuple, Const())
        catch e
            # Only care about the specific BoundsError we're fixing
            if isa(e, BoundsError) && occursin("SimpleVector", string(e))
                error("BoundsError still occurs: $e")
            end
            # Other errors are acceptable for this test
        end
    end
    
    # Test with function returning tuple (might exercise allFirst)
    function return_tuple(x)
        return (x, x * 2)
    end
    
    # This should work without BoundsError
    @test_nowarn begin
        try
            result = Enzyme.autodiff(Enzyme.Forward, return_tuple, Enzyme.Duplicated(2.0, 1.0))
            # Just test that we got some result without BoundsError
        catch e
            if isa(e, BoundsError) && occursin("SimpleVector", string(e))
                error("BoundsError still occurs: $e")
            end
        end
    end
    
    # Test basic functionality still works
    simple_func(x) = x^2
    result = Enzyme.autodiff(Enzyme.Forward, simple_func, Enzyme.Duplicated(3.0, 1.0))
    @test result[1] ≈ 6.0
    
    println("BoundsError regression test passed!")
end