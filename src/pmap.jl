function pmap(body::Body, count, args::Vararg{Any,N}) where {Body,N}
    ccall(:jl_enter_threaded_region, Cvoid, ())
    n_threads = Base.Threads.nthreads()
    n_gen = min(n_threads, count)
    tasks = Vector{Task}(undef, n_gen)
    cnt = (count + n_gen - 1) ÷ n_gen
    for i = 0:(n_gen-1)
        let start = i * cnt, endv = min(count, (i + 1) * cnt) - 1
            t = Task() do
                for j = start:endv
                    body(j + 1, args...)
                end
                nothing
            end
            t.sticky = true
            ccall(:jl_set_task_tid, Cint, (Any, Cint), t, i)
            @inbounds tasks[i+1] = t
            schedule(t)
        end
    end
    try
        for t in tasks
            wait(t)
        end
    finally
        ccall(:jl_exit_threaded_region, Cvoid, ())
    end
end

macro parallel(args...)
    captured = args[1:end-1]
    ex = args[end]
    if !(isa(ex, Expr) && ex.head === :for)
        throw(ArgumentError("@parallel requires a `for` loop expression"))
    end
    if !(ex.args[1] isa Expr && ex.args[1].head === :(=))
        throw(ArgumentError("nested outer loops are not currently supported by @parallel"))
    end
    iter = ex.args[1]
    lidx = iter.args[1]         # index
    range = iter.args[2]
    body = ex.args[2]
    esc(quote
        let range = $(range)
            function bodyf(idx, iter, $(captured...))
                local $(lidx) = @inbounds iter[idx]
                $(body)
                nothing
            end
            lenr = length(range)
            $pmap(bodyf, lenr, range, $(captured...))
        end
    end)
end
