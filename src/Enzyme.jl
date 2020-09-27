module Enzyme

export autodiff
export Const, Active, Duplicated

using Cassette

abstract type Annotation{T} end 
struct Const{T} <: Annotation{T}
    val::T
end
struct Active{T<:AbstractFloat} <: Annotation{T}
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

include("utils.jl")
include("compiler.jl")

annotate() = ()
annotate(arg::Annotation, args...) = (arg, annotate(args...)...)
annotate(arg, args...) = (Const(arg), annotate(args...)...)

prepare_cc() = ()
prepare_cc(arg::Duplicated, args...) = (arg.val, arg.dval, prepare_cc(args...)...)
prepare_cc(arg::DuplicatedNoNeed, args...) = (arg.val, arg.dval, prepare_cc(args...)...)
prepare_cc(arg::Annotation, args...) = (arg.val, prepare_cc(args...)...)

function autodiff(f, args...)
    args′ =  annotate(args...)
    thunk = Compiler.thunk(f, Tuple{map(Core.Typeof, args′)...})
    thunk(prepare_cc(args′...)...)
end

import .Compiler: EnzymeCtx
# Ops that have intrinsics
for op in (sin, cos, exp)
    for (T, suffix) in ((Float32, "f32"), (Float64, "f64"))
        llvmf = "llvm.$(nameof(op)).$suffix"
        @eval begin
            @inline function Cassette.overdub(::EnzymeCtx, ::typeof($op), x::$T)
                ccall($llvmf, llvmcall, $T, ($T,), x)
            end
        end
    end
end

for op in (asin,)
    for (T, llvm_t) in ((Float32, "float"), (Float64, "double"))
        decl = "declare double @$(nameof(op))($llvm_t)"
        func = """
               %val = call $llvm_t @asin($llvm_t %0)
               ret $llvm_t %val
               """
       @eval begin
            @inline function Cassette.overdub(::EnzymeCtx, ::typeof($op), x::$T)
                Base.llvmcall(($decl,$func), $T, Tuple{$T}, x)
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

# WIP
# @inline Cassette.overdub(::EnzymeCtx, ::typeof(asin), x::Float64) = ccall(:asin, Float64, (Float64,), x)
# @inline Cassette.overdub(::EnzymeCtx, ::typeof(asin), x::Float32) = ccall(:asin, Float32, (Float32,), x)
end # module
