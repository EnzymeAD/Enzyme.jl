struct CallWithKWargs{KW}
    kwargs::KW
end

function (c::CallWithKWargs)(f, xs...)
    return f(xs...; c.kwargs...)
end

struct CallWithCopyKWargs{KW}
    kwargs::KW
end

function (c::CallWithCopyKWargs)(f, xs...)
    return deepcopy(f)(deepcopy(xs)...; deepcopy(c.kwargs)...)
end

@inline function get_primal(x::Annotation)
    return x.val
end
