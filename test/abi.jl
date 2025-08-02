using Enzyme
Enzyme.API.printall!(true)

    f(x) = x

    res = autodiff_deferred(Reverse, Const(f), Const, Const(nothing))
