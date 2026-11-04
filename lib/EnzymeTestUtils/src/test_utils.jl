@inline call_with_kwargs(fkwargs::NT, f::FT, xs...) where {NT, FT} = f(xs...; fkwargs...)

# test_forward and test_reverse call `f` through this under autodiff.
# Force evaluation to avoid problem of a tuple being created but not being SROA'd
# Can cause some tests to unnecessarily fail without runtime activity
for N in 1:30
    argexprs = [Symbol(:arg, Symbol(i)) for i in 1:N]
    eval(
        quote
            function call_with_kwargs(fkwargs::NT, f::FT, $(argexprs...)) where {NT, FT}
                Base.@_inline_meta
                return @inline f($(argexprs...); fkwargs...)
            end
        end
    )
end

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
