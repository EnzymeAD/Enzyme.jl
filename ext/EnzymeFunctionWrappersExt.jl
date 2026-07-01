module EnzymeFunctionWrappersExt

using Enzyme
import FunctionWrappers: FunctionWrapper

@inline function Enzyme.Compiler.funcwrapper_rewrite(f::FunctionWrapper{Ret, Args}, args...) where {Ret, Args}
    closure = f.obj[]
    val = Core.invoke(closure, Tuple{Args.parameters...}, args...)
    return val::Ret
end

@inline function Enzyme.Compiler.funcwrapper_rewrite(::typeof(FunctionWrappers.do_ccall), f::FunctionWrapper{Ret, Args}, args::Tuple) where {Ret, Args}
    closure = f.obj[]
    val = Core.invoke(closure, Tuple{Args.parameters...}, args...)
    return val::Ret
end

end # module
