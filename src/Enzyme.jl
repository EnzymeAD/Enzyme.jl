module Enzyme

export autodiff, autodiff_deferred
export Const, Active, Duplicated, DuplicatedNoNeed

"""
    abstract type Annotation{T}

Abstract type for [`autodiff`](@ref) function argument wrappers like
[`Const`](@ref), [`Active`](@ref) and [`Duplicated`](@ref).
"""
abstract type Annotation{T} end

"""
    struct Const{T} <: Annotation{T}

Constructor: `Const(x)`

Mark a function argument `x` of [`autodiff`](@ref) as constant,
Enzyme will not auto-differentiate in respect `Const` arguments.
"""
struct Const{T} <: Annotation{T}
    val::T
end

# To deal with Const(Int) and prevent it to go to `Const{DataType}(T)`
Const(::Type{T}) where T = Const{Type{T}}(T)

"""
    struct Active{T} <: Annotation{T}

Constructor: `Active(x)`

Mark a function argument `x` of [`autodiff`](@ref) as active,
Enzyme will auto-differentiate in respect `Active` arguments.

!!! note

    Enzyme gradients with respect to integer values are zero.
    [`Active`](@ref) will automatically convert plain integers to floating
    point values, but cannot do so for integer values in tuples and structs.
"""
struct Active{T} <: Annotation{T}
    val::T
end

Active(i::Integer) = Active(float(i))


"""
    struct Duplicated{T} <: Annotation{T}

Constructor: `Duplicated(x, ∂f_∂x)`

Mark a function argument `x` of [`autodiff`](@ref) as duplicated, Enzyme will
auto-differentiate in respect to such arguments, with `dx` acting as an
accumulator for gradients (so ``\\partial f / \\partial x`` will be *added to*)
`∂f_∂x`.
"""
struct Duplicated{T} <: Annotation{T}
    val::T
    dval::T
end


struct DuplicatedNoNeed{T} <: Annotation{T}
    val::T
    dval::T
end

Base.eltype(::Type{<:Annotation{T}}) where T = T

function guess_activity(T)
    if T <: AbstractFloat || T <: Complex{<:AbstractFloat}
        return Active{T}
    elseif T <: AbstractArray
        return Duplicated{T}
    else
        return Const{T}
    end
end

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

"""
    autodiff(f, Activity, args...)

Auto-differentiate function `f` at arguments `args`.

Limitations:

* `f` may only return a `Real` (of a built-in/primitive type) or `nothing`,
  not an array, struct, `BigFloat`, etc. To handle vector-valued return
  types, use a mutating `f!` that returns `nothing` and stores it's return
  value in one of the arguments, which must be wrapped in a
  [`Duplicated`](@ref).

* `f` may not allocate memory, this restriction is likely to be removed in
  future versions. Technically it can currently allocate memory directly, but not in a function called by `f`.

`args` may be numbers, arrays, structs of numbers, structs of arrays and so
on. Enzyme will only differentiate in respect to arguments that are wrapped
in an [`Active`](@ref) (for immutable arguments like primitive types and
structs thereof) or [`Duplicated`](@ref) (for mutable arguments like arrays,
`Ref`s and structs thereof). Non-annotated arguments will automatically be
treated as [`Const`](@ref).

`Activity` is the Activity of the return value, it may be `Const` or `Active`.

Example:

```jldoctest
using Enzyme

a = 4.2
b = [2.2, 3.3]; ∂f_∂b = zero(b)
c = 55; d = 9

f(a, b, c, d) = a * √(b[1]^2 + b[2]^2) + c^2 * d^2
∂f_∂a, ∂f_∂d = autodiff(f, Active, Active(a), Duplicated(b, ∂f_∂b), c, Active(d))

# output

(3.966106403010388, 54450.0)
```

here, `autodiff` returns a tuple
``(\\partial f/\\partial a, \\partial f/\\partial d)``,
while ``\\partial f/\\partial b`` will be *added to* `∂f_∂b` (but not returned).
`c` will be treated as `Const(c)`.

!!! note

    Enzyme gradients with respect to integer values are zero.
    [`Active`](@ref) will automatically convert plain integers to floating
    point values, but cannot do so for integer values in tuples and structs.
"""
@inline function autodiff(f::F, ::Type{A}, args...) where {F, A<:Annotation}
    args′  = annotate(args...)
    tt′    = Tuple{map(Core.Typeof, args′)...}
    thunk = Enzyme.Compiler.thunk(f, A, tt′, #=Split=# Val(false))
    if A <: Active
        rt = eltype(Compiler.return_type(thunk))
        args′ = (args′..., one(rt))
    end
    thunk(args′...)
end

"""
    autodiff(f, args...)

Like [`autodiff`](@ref) but will try to guess the activity of the return value.
"""
@inline function autodiff(f::F, args...) where {F}
    args′ = annotate(args...)
    tt′   = Tuple{map(Core.Typeof, args′)...}
    tt    = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
    rt    = Core.Compiler.return_type(f, tt)
    A     = guess_activity(rt)
    autodiff(f, A, args′...)
end


"""
    autodiff_deferred(f, Activity, args...)

Same as [`autodiff`](@ref) but uses deferred compilation to support usage in GPU
code, as well as high-order differentiation.
"""
@inline function autodiff_deferred(f::F, ::Type{A}, args...) where {F, A<:Annotation}
    args′ = annotate(args...)
    tt′   = Tuple{map(Core.Typeof, args′)...}
    if A isa UnionAll
        tt = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
        rt = Core.Compiler.return_type(f, tt)
        rt = A{rt}
    else
        @assert A isa DataType
        rt = A
    end

    ptr   = Compiler.deferred_codegen(Val(f), Val(tt′), Val(rt))
    thunk = Compiler.CombinedAdjointThunk{F, rt, tt′}(f, ptr)
    if rt <: Active
        args′ = (args′..., one(eltype(rt)))
    end
    thunk(args′...)
end

"""
    autodiff_deferred(f, args...)

Like [`autodiff_deferred`](@ref) but will try to guess the activity of the return value.
"""
@inline function autodiff_deferred(f::F, args...) where {F}
    args′ = annotate(args...)
    tt′   = Tuple{map(Core.Typeof, args′)...}
    tt    = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
    rt    = Core.Compiler.return_type(f, tt)
    rt    = guess_activity(rt)
    ptr   = Compiler.deferred_codegen(Val(f), Val(tt′), Val(rt))
    thunk = Compiler.CombinedAdjointThunk{F, rt, tt′}(f, ptr)
    if rt <: Active
        args′ = (args′..., one(eltype(rt)))
    end
    thunk(args′...)
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

# TODO: Remove these functions
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

end # module
