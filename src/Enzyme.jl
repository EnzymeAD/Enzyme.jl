module Enzyme

export autodiff
export Const, Active, Duplicated

using Cassette

abstract type Annotation{T} end
struct Const{T} <: Annotation{T}
    val::T
end
# To deal with Const(Int) and prevent it to go to `Const{DataType}(T)`
Const(::Type{T}) where T = Const{Type{T}}(T)
struct Active{T} <: Annotation{T}
    val::T
end
Active(i::Integer) = Active(float(i))
struct Duplicated{T} <: Annotation{T}
    val::T
    dval::T
end
struct DuplicatedNoNeed{T} <: Annotation{T}
    val::T
    dval::T
end

Base.eltype(::Type{<:Annotation{T}}) where T = T

import LLVM

include("api.jl")
include("logic.jl")
include("typeanalysis.jl")
include("typetree.jl")
include("utils.jl")
include("compiler.jl")

# @inline annotate() = ()
# @inline annotate(arg::A, args::Vararg{Any, N}) where {A<:Annotation, N} = (arg, annotate(args...)...)
# @inline annotate(arg, args::Vararg{Any, N}) where N = (Const(arg), annotate(args...)...)

@inline function annotate(args::Vararg{Any, N}) where N
    ntuple(Val(N)) do i
        Base.@_inline_meta
        arg = @inbounds args[i]
        if arg isa Annotation
            return arg
        else
            return Const(arg)
        end
    end
end

prepare_cc() = ()
prepare_cc(arg::Duplicated, args...) = (arg.val, arg.dval, prepare_cc(args...)...)
prepare_cc(arg::DuplicatedNoNeed, args...) = (arg.val, arg.dval, prepare_cc(args...)...)
prepare_cc(arg::Annotation, args...) = (arg.val, prepare_cc(args...)...)

@inline function autodiff(f::F, args...) where F
    args′ = annotate(args...)
    tt′   = Tuple{map(Core.Typeof, args′)...}
    ptr   = Compiler.deferred_codegen(Val(f), Val(tt′), Val(true))
    tt    = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
    rt    = Core.Compiler.return_type(f, tt)
    thunk = Compiler.CombinedAdjointThunk{F, rt, tt′}(ptr)
    thunk(args′...)
end

@inline function autodiff_no_cassette(f::F, args...) where F
    args′ = annotate(args...)
    tt′   = Tuple{map(Core.Typeof, args′)...}
    ptr   = Compiler.deferred_codegen(Val(f), Val(tt′), Val(false))
    tt    = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
    rt    = Core.Compiler.return_type(f, tt)
    thunk = Compiler.CombinedAdjointThunk{F, rt, tt′}(ptr)
    thunk(args′...)
end

import .Compiler: EnzymeCtx
# Ops that have intrinsics
for op in (sin, cos, tan, exp, log)
    for (T, suffix) in ((Float32, "f32"), (Float64, "f64"))
        llvmf = "llvm.$(nameof(op)).$suffix"
        @eval begin
            @inline function Cassette.overdub(::EnzymeCtx, ::typeof($op), x::$T)
                ccall($llvmf, llvmcall, $T, ($T,), x)
            end
        end
    end
end

for op in (copysign,)
    for (T, suffix) in ((Float32, "f32"), (Float64, "f64"))
        llvmf = "llvm.$(nameof(op)).$suffix"
        @eval begin
            @inline function Cassette.overdub(::EnzymeCtx, ::typeof($op), x::$T, y::$T)
                ccall($llvmf, llvmcall, $T, ($T, $T), x, y)
            end
        end
    end
end

for op in (asin,tanh)
    for (T, llvm_t, suffix) in ((Float32, "float", "f"), (Float64, "double", ""))
        mod = """
                declare $llvm_t @$(nameof(op))$suffix($llvm_t)
               
                define $llvm_t @entry($llvm_t) #0 {
                    %val = call $llvm_t @$op$suffix($llvm_t %0)
                    ret $llvm_t %val
                }
                attributes #0 = { alwaysinline }
               """
       @eval begin
            @inline function Cassette.overdub(::EnzymeCtx, ::typeof($op), x::$T)
                Base.llvmcall(($mod, "entry"), $T, Tuple{$T}, x)
            end
        end
    end
end

@inline function pack(args...)
    ntuple(Val(length(args))) do i
        Base.@_inline_meta
        arg = args[i]
        @assert arg isa AbstractFloat
        return Duplicated(Ref(args[i]), Ref(zero(args[i])))
    end
end

@inline unpack() = ()
@inline unpack(arg) = (arg[],)
@inline unpack(arg, args...) = (arg[], unpack(args...)...)

@inline ∇unpack() = ()
@inline ∇unpack(arg::Duplicated) = (arg.dval[],)
@inline ∇unpack(arg::Duplicated, args...) = (arg.dval[], ∇unpack(args...)...)

function gradient(f, args...)
    ∇args = pack(args...)
    f′ = function (args...)
        Base.@_inline_meta
        f(unpack(args...)...)
    end
    autodiff(f′, ∇args...)
    return ∇unpack(∇args...)
end

function pullback(f, args...)
    return (c) -> begin
        ∇vals = gradient(f, args...)
        return ntuple(Val(length(∇vals))) do i
            Base.@_inline_meta
            return c*∇vals[i]
        end
    end
end

using Adapt
Adapt.adapt_structure(to, x::Duplicated) = Duplicated(adapt(to, x.val), adapt(to, x.dval))
Adapt.adapt_structure(to, x::DuplicatedNoNeed) = DuplicatedNoNeed(adapt(to, x.val), adapt(to, x.dval))
Adapt.adapt_structure(to, x::Const) = Const(adapt(to, x.val))
Adapt.adapt_structure(to, x::Active) = Active(adapt(to, x.val))

# WIP
# @inline Cassette.overdub(::EnzymeCtx, ::typeof(asin), x::Float64) = ccall(:asin, Float64, (Float64,), x)
# @inline Cassette.overdub(::EnzymeCtx, ::typeof(asin), x::Float32) = ccall(:asinf, Float32, (Float32,), x)
end # module
