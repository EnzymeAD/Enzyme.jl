using Distributed, Test

## entry point

function runtests(f, name)
    old_print_setting = Test.TESTSET_PRINT_ENABLE[]
    if VERSION < v"1.13.0-DEV.1044"
        Test.TESTSET_PRINT_ENABLE[] = false
    else
        Test.TESTSET_PRINT_ENABLE[] => false
    end

    try
        # generate a temporary module to execute the tests in
        mod_name = Symbol("Test", rand(1:100), "Main_", replace(name, '/' => '_'))
        mod = @eval(Main, module $mod_name end)
        @eval(mod, using Test, Random)

        let id = myid()
            wait(@spawnat 1 print_testworker_started(name, id))
        end

        ex = quote
            GC.gc(true)
            Random.seed!(1)

            res = @timed @testset $name begin
                $f()
            end
            res..., 0, 0, 0
        end
        data = Core.eval(mod, ex)
        #data[1] is the testset

        # process results
        rss = Sys.maxrss()
        if VERSION >= v"1.11.0-DEV.1529"
            tc = Test.get_test_counts(data[1])
            passes,fails,error,broken,c_passes,c_fails,c_errors,c_broken =
                tc.passes, tc.fails, tc.errors, tc.broken, tc.cumulative_passes,
                tc.cumulative_fails, tc.cumulative_errors, tc.cumulative_broken
        else
            passes,fails,errors,broken,c_passes,c_fails,c_errors,c_broken =
                Test.get_test_counts(data[1])
        end
        if data[1].anynonpass == false
            data = ((passes+c_passes,broken+c_broken),
                    data[2],
                    data[3],
                    data[4],
                    data[5],
                    data[6],
                    data[7],
                    data[8])
        end
        res = vcat(collect(data), rss)

        GC.gc(true)
        res
    finally
        if VERSION < v"1.13.0-DEV.1044"
            Test.TESTSET_PRINT_ENABLE[] = old_print_setting
        else
            Test.TESTSET_PRINT_ENABLE[] => old_print_setting
        end
    end
end


## auxiliary stuff

# NOTE: based on test/pkg.jl::capture_stdout, but doesn't discard exceptions
macro grab_output(ex)
    quote
        mktemp() do fname, fout
            ret = nothing
            open(fname, "w") do fout
                redirect_stdout(fout) do
                    ret = $(esc(ex))

                    # NOTE: CUDA requires a 'proper' sync to flush its printf buffer
                    synchronize(context())
                end
            end
            ret, read(fname, String)
        end
    end
end

function julia_exec(args::Cmd, env...)
    # FIXME: this doesn't work when the compute mode is set to exclusive
    cmd = Base.julia_cmd()
    cmd = `$cmd --project=$(Base.active_project()) --color=no $args`

    out = Pipe()
    err = Pipe()
    proc = run(pipeline(addenv(cmd, env...), stdout=out, stderr=err), wait=false)
    close(out.in)
    close(err.in)
    wait(proc)
    proc, read(out, String), read(err, String)
end

nothing # File is loaded via a remotecall to "include". Ensure it returns "nothing".
