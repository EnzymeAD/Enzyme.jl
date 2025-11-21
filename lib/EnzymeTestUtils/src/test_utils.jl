
struct CallWithKWargs{KW}
    kwargs::KW
end

function (c::CallWithKWargs)(f, xs...)
    f(xs...; c.kwargs...)
end

struct CallWithCopyKWargs{KW}
    kwargs::KW
end

function (c::CallWithCopyKWargs)(f, xs...)
    deepcopy(f)(deepcopy(xs)...; deepcopy(c.kwargs)...)
end

@inline function get_primal(x::Annotation)
    x.val
end