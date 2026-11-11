struct CallWithKWargs{KW}
    kwargs::KW
end

@inline function (c::CallWithKWargs)(f::FT, xs...) where {FT}
    return f(xs...; c.kwargs...)
end

# Force evaluation to avoid problem of a tuple being created but not being SROA'd
# Can cause some tests to unnecessarily fail without runtime activity
for N in 1:30
    argexprs = [Symbol(:arg, Symbol(i)) for i in 1:N]
    eval(
        quote
            function (c::CallWithKWargs{KW})(f::FT, $(argexprs...)) where {KW, FT}
                Base.@_inline_meta
                return @inline f($(argexprs...); c.kwargs...)
            end
        end,
    )
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
