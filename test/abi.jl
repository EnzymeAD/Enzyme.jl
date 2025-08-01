using Enzyme


    f(x) = x

    res = autodiff_deferred(Reverse, Const(f), Const, Const(nothing))
