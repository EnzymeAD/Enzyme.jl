using Test

function precompile_test_harness(@nospecialize(f), testset::String)
    @testset "$testset" begin
        precompile_test_harness(f, true)
    end
end
function precompile_test_harness(@nospecialize(f), separate::Bool)
    load_path = mktempdir()
    load_cache_path = separate ? mktempdir() : load_path
    try
        pushfirst!(LOAD_PATH, load_path)
        pushfirst!(DEPOT_PATH, load_cache_path)
        f(load_path)
    finally
        try
            rm(load_path, force=true, recursive=true)
        catch err
            @show err
        end
        if separate
            try
                rm(load_cache_path, force=true, recursive=true)
            catch err
                @show err
            end
        end
        filter!((≠)(load_path), LOAD_PATH)
        separate && filter!((≠)(load_cache_path), DEPOT_PATH)
    end
    nothing
end

precompile_test_harness("Inference caching") do load_path
    write(joinpath(load_path, "InferenceCaching.jl"), :(module InferenceCaching
        using Enzyme
        using PrecompileTools

        function mul(x, y)
            return x * y
        end

        @setup_workload begin
            @compile_workload begin
                autodiff(Reverse, mul, Active, Active(1.0), Active(2.0))
                autodiff(Forward, mul, Duplicated, Duplicated(1.0, 1.0), Const(2.0))
            end
        end
    end) |> string)

    Base.compilecache(Base.PkgId("InferenceCaching"))
    @eval let
        using InferenceCaching
        using Enzyme

        autodiff(Reverse, InferenceCaching.mul, Active, Active(1.0), Active(2.0))
        autodiff(Forward, InferenceCaching.mul, Duplicated, Duplicated(1.0, 1.0), Const(2.0))
    end
end