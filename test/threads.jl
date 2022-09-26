using Enzyme
Enzyme.API.printall!(true)

    function foo(y)
        Threads.@threads for i in 1:3
            y[i] *= 2
        end
        nothing
    end

    x = [1.0, 2.0, 3.0]
    dx = [1.0, 1.0, 1.0]
    Enzyme.autodiff(Reverse, foo, Duplicated(x, dx))

    @show length(x)
    flush(stdout)
    @show length(dx)
    flush(stdout)
    
    @show x[1]
    flush(stdout)

    @show dx[1]
    flush(stdout)

    @show x
    flush(stdout)

    @show dx
    flush(stdout)

