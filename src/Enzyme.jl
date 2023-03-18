module Enzyme

import EnzymeCore: Forward, Reverse, ReverseWithPrimal, ReverseSplitNoPrimal, ReverseSplitWithPrimal, ReverseSplitModified, ReverseSplitWidth
export Forward, Reverse, ReverseWithPrimal, ReverseSplitNoPrimal, ReverseSplitWithPrimal, ReverseSplitModified, ReverseSplitWidth

import EnzymeCore: Const, Active, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed
export Const, Active, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed

import EnzymeCore: batch_size
export batch_size

import EnzymeCore: autodiff, autodiff_deferred, autodiff_thunk
export autodiff, autodiff_deferred, autodiff_thunk

export jacobian, gradient, gradient!
export markType, batch_size, onehot, chunkedonehot

using LinearAlgebra
import EnzymeCore: ReverseMode, ReverseModeSplit, ForwardMode, Annotation, Mode

import EnzymeCore: EnzymeRules
export EnzymeRules

# Independent code, must be loaded before "compiler.jl"
include("pmap.jl")

import LLVM
include("api.jl")

Base.convert(::Type{API.CDerivativeMode}, ::ReverseMode) = API.DEM_ReverseModeCombined
Base.convert(::Type{API.CDerivativeMode}, ::ReverseModeSplit) = API.DEM_ReverseModeGradient
Base.convert(::Type{API.CDerivativeMode}, ::ForwardMode) = API.DEM_ForwardMode

function guess_activity end

include("logic.jl")
include("typeanalysis.jl")
include("typetree.jl")
include("utils.jl")
include("compiler.jl")
include("internal_rules.jl")

import .Compiler: CompilationException

# @inline annotate() = ()
# @inline annotate(arg::A, args::Vararg{Any, N}) where {A<:Annotation, N} = (arg, annotate(args...)...)
# @inline annotate(arg, args::Vararg{Any, N}) where N = (Const(arg), annotate(args...)...)

@inline function falses_from_args(::Val{add}, args::Vararg{Any, N}) where {add,N}
    ntuple(Val(add+N)) do i
        Base.@_inline_meta
        false
    end
end

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

@inline function any_active(args::Vararg{Any, N}) where N
    any(ntuple(Val(N)) do i
        Base.@_inline_meta
        arg = @inbounds args[i]
        if arg isa Active
            return true
        else
            return false
        end
    end)
end

@inline function same_or_one_helper(current, next)
    if current == -1
        return next
    elseif current == next
        return next
    else
        error("Multiple distinct batch sizes")
    end
end

@inline same_or_one_rec(current) = current
@inline same_or_one_rec(current, arg::BatchDuplicated{T, N}, args...) where {T,N} =
   same_or_one_rec(same_or_one_helper(current, N), args...)
@inline same_or_one_rec(current, arg::Type{BatchDuplicated{T, N}}, args...) where {T,N} =
   same_or_one_rec(same_or_one_helper(current, N), args...)
@inline same_or_one_rec(current, arg::BatchDuplicatedNoNeed{T, N}, args...) where {T,N} =
   same_or_one_rec(same_or_one_helper(current, N), args...)
@inline same_or_one_rec(current, arg::Type{BatchDuplicatedNoNeed{T, N}}, args...) where {T,N} =
   same_or_one_rec(same_or_one_helper(current, N), args...)
@inline same_or_one_rec(current, arg, args...) = same_or_one_rec(current, args...)

@inline function same_or_one(args...)
    res = same_or_one_rec(-1, args...)
    if res == -1
        return 1
    else
        return res
    end
end

"""
    autodiff(::ReverseMode, f, Activity, args...)

Auto-differentiate function `f` at arguments `args` using reverse mode.

Limitations:

* `f` may only return a `Real` (of a built-in/primitive type) or `nothing`,
  not an array, struct, `BigFloat`, etc. To handle vector-valued return
  types, use a mutating `f!` that returns `nothing` and stores it's return
  value in one of the arguments, which must be wrapped in a
  [`Duplicated`](@ref).

`args` may be numbers, arrays, structs of numbers, structs of arrays and so
on. Enzyme will only differentiate in respect to arguments that are wrapped
in an [`Active`](@ref) (for immutable arguments like primitive types and
structs thereof) or [`Duplicated`](@ref) (for mutable arguments like arrays,
`Ref`s and structs thereof). Non-annotated arguments will automatically be
treated as [`Const`](@ref).

`Activity` is the Activity of the return value, it may be `Const` or `Active`.

Example:

```jldoctest
a = 4.2
b = [2.2, 3.3]; ∂f_∂b = zero(b)
c = 55; d = 9

f(a, b, c, d) = a * √(b[1]^2 + b[2]^2) + c^2 * d^2
∂f_∂a, _, _, ∂f_∂d = autodiff(Reverse, f, Active, Active(a), Duplicated(b, ∂f_∂b), c, Active(d))[1]

# output

(3.966106403010388, nothing, nothing, 54450.0)
```

here, `autodiff` returns a tuple
``(\\partial f/\\partial a, \\partial f/\\partial d)``,
while ``\\partial f/\\partial b`` will be *added to* `∂f_∂b` (but not returned).
`c` will be treated as `Const(c)`.

One can also request the original returned value of the computation.

Example:

```jldoctest
Enzyme.autodiff(ReverseWithPrimal, x->x*x, Active(3.0))

# output

((6.0,), 9.0)
```

!!! note

    Enzyme gradients with respect to integer values are zero.
    [`Active`](@ref) will automatically convert plain integers to floating
    point values, but cannot do so for integer values in tuples and structs.
"""
@inline function autodiff(::ReverseMode{ReturnPrimal}, f::FA, ::Type{A}, args...) where {FA<:Annotation, A<:Annotation, ReturnPrimal}
    args′  = annotate(args...)
    tt′    = Tuple{map(Core.Typeof, args′)...}
    width = same_or_one(args...)
    if width == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end

    ModifiedBetween = Val(falses_from_args(Val(1), args...))

    if A <: Active
        tt    = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
        rt = Core.Compiler.return_type(f.val, tt)
        if !allocatedinline(rt) || rt isa Union
            forward, adjoint = Enzyme.Compiler.thunk(FA, Duplicated{rt}, tt′, #=Split=# Val(API.DEM_ReverseModeGradient), Val(width), ModifiedBetween, #=ReturnPrimal=#Val(ReturnPrimal), #=ShadowInit=#Val(true))
            res = forward(f, args′...)
            tape = res[1]
            if ReturnPrimal
                return (adjoint(args′..., tape)[1], res[2])
            else
                return adjoint(f, args′..., tape)
            end
        end
    elseif A <: Duplicated || A<: DuplicatedNoNeed || A <: BatchDuplicated || A<: BatchDuplicatedNoNeed
        throw(ErrorException("Duplicated Returns not yet handled"))
    end
    thunk = Enzyme.Compiler.thunk(FA, A, tt′, #=Split=# Val(API.DEM_ReverseModeCombined), Val(width), ModifiedBetween, Val(ReturnPrimal))
    if A <: Active
        tt    = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
        rt = eltype(Compiler.return_type(thunk))
        args′ = (args′..., one(rt))
    end
    thunk(f, args′...)
end

"""
    autodiff(mode::Mode, f, ::Type{A}, args...)

Like [`autodiff`](@ref) but will try to extend f to an annotation, if needed.
"""
@inline function autodiff(mode::CMode, f::F, args...) where {F, CMode<:Mode}
    autodiff(mode, Const(f), args...)
end

"""
    autodiff(mode::Mode, f, args...)

Like [`autodiff`](@ref) but will try to guess the activity of the return value.
"""
@inline function autodiff(mode::CMode, f::FA, args...) where {FA<:Annotation, CMode<:Mode}
    args′ = annotate(args...)
    tt′   = Tuple{map(Core.Typeof, args′)...}
    tt    = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
    rt    = Core.Compiler.return_type(f.val, tt)
    A     = guess_activity(rt, mode)
    autodiff(mode, f, A, args′...)
end

"""
    autodiff(::ForwardMode, f, Activity, args...)

Auto-differentiate function `f` at arguments `args` using forward mode.

`args` may be numbers, arrays, structs of numbers, structs of arrays and so
on. Enzyme will only differentiate in respect to arguments that are wrapped
in a [`Duplicated`](@ref) or similar argument. Non-annotated arguments will
automatically be treated as [`Const`](@ref). Unlike reverse mode in
[`autodiff`](@ref), [`Active`](@ref) arguments are not allowed here, since
all 

`Activity` is the Activity of the return value, it may be:
* `Const` if the return is not to be differentiated with respect to
* `Duplicated`, if the return is being differentiated with respect to and
  both the original value and the derivative return are desired
* `DuplicatedNoNeed`, if the return is being differentiated with respect to
  and only the derivative return is desired.

Example returning both original return and derivative:

```jldoctest
a = 4.2
b = [2.2, 3.3]; ∂f_∂b = zero(b)
c = 55; d = 9

f(x) = x*x
res, ∂f_∂x = autodiff(Forward, f, Duplicated, Duplicated(3.14, 1.0))

# output

(9.8596, 6.28)
```

Example returning just the derivative:

```jldoctest
a = 4.2
b = [2.2, 3.3]; ∂f_∂b = zero(b)
c = 55; d = 9

f(x) = x*x
∂f_∂x = autodiff(Forward, f, DuplicatedNoNeed, Duplicated(3.14, 1.0))

# output

(6.28,)
```
"""
@inline function autodiff(::ForwardMode, f::FA, ::Type{A}, args...) where {FA<:Annotation, A<:Annotation}
    args′  = annotate(args...)
    if any_active(args′...)
        throw(ErrorException("Active arguments not allowed in forward mode"))
    end
    tt′    = Tuple{map(Core.Typeof, args′)...}
    width = same_or_one(args′...)
    if width == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end
    if A <: Active
        throw(ErrorException("Active Returns not allowed in forward mode"))
    end
    ReturnPrimal = Val(A <: Duplicated || A <: BatchDuplicated)
    RT = if A <: Duplicated && width != 1
        if A isa UnionAll
            BatchDuplicated{T, width} where T
        else
            BatchDuplicated{eltype(A), width}
        end
    elseif A <: DuplicatedNoNeed && width != 1
        if A isa UnionAll
            BatchDuplicatedNoNeed{T, width} where T
        else
            BatchDuplicatedNoNeed{eltype(A), width}
        end
    else
        A
    end
    
    ModifiedBetween = Val(falses_from_args(Val(1), args...))

    thunk = Enzyme.Compiler.thunk(FA, RT, tt′, #=Mode=# Val(API.DEM_ForwardMode), Val(width),
                                     ModifiedBetween, ReturnPrimal)
    thunk(f, args′...)
end

# F as first arg for `do` syntax
@inline autodiff(dupf::Duplicated{F}, mode::Mode, ::Type{A}, args...) where {F,A<:Annotation} = autodiff(mode, dupf, A, args...)
@inline autodiff(f::F, mode::Mode, ::Type{A}, args...) where {F,A<:Annotation} = autodiff(mode, f, A, args...)
@inline autodiff(dupf::Duplicated{F}, mode::Mode, args...) where {F} = autodiff(mode, dupf, args...)
@inline autodiff(f::F, mode::Mode, args...) where {F} = autodiff(mode, f, args...)

"""
    autodiff_deferred(::ReverseMode, f, Activity, args...)

Same as [`autodiff`](@ref) but uses deferred compilation to support usage in GPU
code, as well as high-order differentiation.
"""
@inline function autodiff_deferred(::ReverseMode{ReturnPrimal}, f::FA, ::Type{A}, args...) where {FA<:Annotation, A<:Annotation, ReturnPrimal}
    args′ = annotate(args...)
    tt′   = Tuple{map(Core.Typeof, args′)...}
    width = same_or_one(args...)
    if width == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end
    if A isa UnionAll
        tt = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
        rt = Core.Compiler.return_type(f.val, tt)
        rt = A{rt}
    else
        @assert A isa DataType
        rt = A
    end

    if eltype(rt) == Union{}
        error("Return type inferred to be Union{}. Giving up.")
    end

    ModifiedBetween = Val(falses_from_args(Val(1), args...))
    
    adjoint_ptr, primal_ptr = Compiler.deferred_codegen(FA, Val(tt′), Val(rt), Val(API.DEM_ReverseModeCombined), Val(width), ModifiedBetween, Val(ReturnPrimal))
    @assert primal_ptr === nothing
    thunk = Compiler.CombinedAdjointThunk{FA, rt, tt′, typeof(Val(width)), Val(ReturnPrimal)}(adjoint_ptr)
    if rt <: Active
        args′ = (args′..., one(eltype(rt)))
    elseif A <: Duplicated || A<: DuplicatedNoNeed || A <: BatchDuplicated || A<: BatchDuplicatedNoNeed
        throw(ErrorException("Duplicated Returns not yet handled"))
    end
    thunk(f, args′...)
end

"""
    autodiff_deferred(::ForwardMode, f, Activity, args...)

Same as `autodiff(::ForwardMode, ...)` but uses deferred compilation to support usage in GPU
code, as well as high-order differentiation.
"""
@inline function autodiff_deferred(::ForwardMode, f::FA, ::Type{A}, args...) where {FA<:Annotation, A<:Annotation}
    args′ = annotate(args...)
    if any_active(args′...)
        throw(ErrorException("Active arguments not allowed in forward mode"))
    end
    tt′   = Tuple{map(Core.Typeof, args′)...}
    width = same_or_one(args...)
    if width == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end
    RT = if A <: Duplicated && width != 1
        if A isa UnionAll
            BatchDuplicated{T, width} where T
        else
            BatchDuplicated{eltype(A), width}
        end
    elseif A <: DuplicatedNoNeed && width != 1
        if A isa UnionAll
            BatchDuplicatedNoNeed{T, width} where T
        else
            BatchDuplicatedNoNeed{eltype(A), width}
        end
    else
        A
    end
    if RT isa UnionAll
        tt = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
        rt = Core.Compiler.return_type(f.val, tt)
        rt = RT{rt}
    else
        @assert RT isa DataType
        rt = RT
    end

    if eltype(rt) == Union{}
        error("Return type inferred to be Union{}. Giving up.")
    end

    if RT <: Active
        throw(ErrorException("Active Returns not allowed in forward mode"))
    end

    ReturnPrimal = Val(RT <: Duplicated || RT <: BatchDuplicated)
    ModifiedBetween = Val(falses_from_args(Val(1), args...))

    adjoint_ptr, primal_ptr = Compiler.deferred_codegen(FA, Val(tt′), Val(rt), Val(API.DEM_ForwardMode), Val(width), ModifiedBetween, ReturnPrimal)
    @assert primal_ptr === nothing
    thunk = Compiler.ForwardModeThunk{FA, rt, tt′, typeof(Val(width)), ReturnPrimal}(adjoint_ptr)
    thunk(f, args′...)
end

"""
    autodiff_deferred(mode::Mode, f, ::Type{A}, args...)

Like [`autodiff_deferred`](@ref) but will try to extend f to an annotation, if needed.
"""
@inline function autodiff_deferred(mode::CMode, f::F, args...) where {F, CMode<:Mode}
    autodiff_deferred(mode, Const(f), args...)
end
"""
    autodiff_deferred(mode, f, args...)

Like [`autodiff_deferred`](@ref) but will try to guess the activity of the return value.
"""

@inline function autodiff_deferred(mode::SMode, f::FA, args...) where {FA<:Annotation, SMode<:Mode}
    args′ = annotate(args...)
    tt    = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
    rt    = Core.Compiler.return_type(f.val, tt)
    rt    = guess_activity(rt, mode)
    autodiff_deferred(mode, f, rt, args′...)
end

"""
    autodiff_thunk(::ReverseModeSplit, ftype, Activity, argtypes...)

Provide the split forward and reverse pass functions for annotated function type
ftype when called with args of type `argtypes` when using reverse mode.

`Activity` is the Activity of the return value, it may be `Const`, `Active`,
or `Duplicated` (or its variants `DuplicatedNoNeed`, `BatchDuplicated`, and
`BatchDuplicatedNoNeed`).

The forward function will return a tape, the primal (or nothing if not requested),
and the shadow (or nothing if not a `Duplicated` variant), and tapes the corresponding
type arguements provided.

The reverse function will return the derivative of `Active` arguments, updating the `Duplicated`
arguments in place. The same arguments to the forward pass should be provided, followed by
the adjoint of the return (if the return is active), and finally the tape from the forward pass.

Example:

```jldoctest

A = [2.2]; ∂A = zero(A)
v = 3.3

function f(A, v)
    res = A[1] * v
    A[1] = 0
    res
end

forward, reverse = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f)}, Active, Duplicated{typeof(A)}, Active{typeof(v)})

tape, result, shadow_result  = forward(Const(f), Duplicated(A, ∂A), Active(v))
_, ∂v = reverse(Const(f), Duplicated(A, ∂A), Active(v), 1.0, tape)[1]

result, ∂v, ∂A 

# output

(7.26, 2.2, [3.3])
```
"""
@inline function autodiff_thunk(::ReverseModeSplit{ReturnPrimal,ReturnShadow,Width,ModifiedBetweenT}, ::Type{FA}, ::Type{A}, args...) where {FA<:Annotation, A<:Annotation, ReturnPrimal,ReturnShadow,Width,ModifiedBetweenT}
    # args′  = annotate(args...)
    width = if Width == 0
        w = same_or_one(args...)
        if w == 0
            throw(ErrorException("Cannot differentiate with a batch size of 0"))
        end
        w
    else
        Width
    end

    if ModifiedBetweenT === true
        ModifiedBetween = Val(falses_from_args(Val(1), args...))
    else
        ModifiedBetween = Val(ModifiedBetweenT)
    end

    @assert ReturnShadow
    Enzyme.Compiler.thunk(FA, A, Tuple{args...}, #=Split=# Val(API.DEM_ReverseModeGradient), Val(width), ModifiedBetween, #=ReturnPrimal=#Val(ReturnPrimal))
end

# White lie, should be `Core.LLVMPtr{Cvoid, 0}` but that's not supported by ccallable
Base.@ccallable function __enzyme_float(x::Ptr{Cvoid})::Cvoid
    return nothing
end

Base.@ccallable function __enzyme_double(x::Ptr{Cvoid})::Cvoid
    return nothing
end

@inline function markType(::Type{T}, ptr::Ptr{Cvoid}) where T
    markType(Base.unsafe_convert(Ptr{T}, ptr))
end

@inline function markType(data::Array{T}) where T
    GC.@preserve data markType(pointer(data))
end

# TODO(WM): We record the type of a single index here, we could give it a range
@inline function markType(data::SubArray)
    GC.@preserve data markType(pointer(data))
end

@inline function markType(data::Ptr{Float32})
    Base.llvmcall(("declare void @__enzyme_float(i8* nocapture) nounwind define void @c(i64 %q) nounwind alwaysinline { %p = inttoptr i64 %q to i8* call void @__enzyme_float(i8* %p) ret void }", "c"), Cvoid, Tuple{Ptr{Float32}}, data)
    nothing
end

@inline function markType(data::Ptr{Float64})
    Base.llvmcall(("declare void @__enzyme_double(i8* nocapture) nounwind define void @c(i64 %q) nounwind alwaysinline { %p = inttoptr i64 %q to i8* call void @__enzyme_double(i8* %p) ret void }", "c"), Cvoid, Tuple{Ptr{Float64}}, data)
    nothing
end

@inline function onehot(x, start=1, endl=length(x))
    ntuple(Val(endl-start+1)) do i
        Base.@_inline_meta
        res = similar(x)
        for idx in 1:length(x)
            @inbounds res[idx] = (i + start - 1== idx) ? 1.0 : 0.0
        end
        return res
    end
end

@inline function onehot(x::NTuple{N, T}, start=1, endl=N) where {T, N}
    ntuple(Val(endl-start+1)) do i
        Base.@_inline_meta
        ntuple(N) do idx
            Base.@_inline_meta
            return (i + start - 1 == idx) ? 1.0 : 0.0
        end
    end
end

"""
    gradient(::ReverseMode, f, x)

Compute the gradient of an array-input function `f` using reverse mode.
This will allocate and return new array with the gradient result.

Example:

```jldoctest
f(x) = x[1]*x[2]

grad = gradient(Reverse, f, [2.0, 3.0])

# output

2-element Vector{Float64}:
 3.0
 2.0
```
"""
@inline function gradient(::ReverseMode, f, x)
    dx = zero(x)
    autodiff(Reverse, f, Duplicated(x, dx))
    dx
end


"""
    gradient!(::ReverseMode, dx, f, x)

Compute the gradient of an array-input function `f` using reverse mode,
storing the derivative result in an existing array `dx`.

Example:

```jldoctest
f(x) = x[1]*x[2]

dx = [0.0, 0.0]
gradient!(Reverse, dx, f, [2.0, 3.0])

# output

2-element Vector{Float64}:
 3.0
 2.0
```
"""
@inline function gradient!(::ReverseMode, dx, f, x)
    dx .= 0
    autodiff(Reverse, f, Duplicated(x, dx))
    dx
end

"""
    gradient(::ForwardMode, f, x; shadow=onehot(x))

Compute the gradient of an array-input function `f` using forward mode. The
optional keyword argument `shadow` is a vector of one-hot vectors of type `x`
which are used to forward-propagate into the return. For performance reasons,
this should be computed once, outside the call to `gradient`, rather than
within this call.

Example:

```jldoctest
f(x) = x[1]*x[2]

grad = gradient(Forward, f, [2.0, 3.0])

# output

(3.0, 2.0)
```
"""
@inline function gradient(::ForwardMode, f, x; shadow=onehot(x))
    if length(x) == 0
        return ()
    end
    values(only(autodiff(Forward, f, BatchDuplicatedNoNeed, BatchDuplicated(x, shadow))))
end

@inline function chunkedonehot(x, ::Val{chunk}) where chunk
    sz = length(x)
    num = ((sz + chunk - 1) ÷ chunk)
    ntuple(Val(num)) do i
        Base.@_inline_meta
        onehot(x, (i-1)*chunk+1, i == num ? sz : (i*chunk) )
    end
end

@inline tupleconcat(x) = x
@inline tupleconcat(x, y) = (x..., y...)
@inline tupleconcat(x, y, z...) = (x..., tupleconcat(y, z...)...)

"""
    gradient(::ForwardMode, f, x, ::Val{chunk}; shadow=onehot(x))

Compute the gradient of an array-input function `f` using vector forward mode.
Like [`gradient`](@ref), except it uses a chunk size of `chunk` to compute
`chunk` derivatives in a single call.

Example:

```jldoctest
f(x) = x[1]*x[2]

grad = gradient(Forward, f, [2.0, 3.0], Val(2))

# output

(3.0, 2.0)
```
"""
@inline function gradient(::ForwardMode, f::F, x::X, ::Val{chunk}; shadow=chunkedonehot(x, Val(chunk))) where {F, X, chunk}
    if chunk == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end
    tmp = ntuple(length(shadow)) do i
        values(autodiff(Forward, f, BatchDuplicatedNoNeed, BatchDuplicated(x, shadow[i]))[1])
    end
    tupleconcat(tmp...)
end

@inline function gradient(::ForwardMode, f::F, x::X, ::Val{1}; shadow=onehot(x)) where {F,X}
    ntuple(length(shadow)) do i
        autodiff(Forward, f, DuplicatedNoNeed, Duplicated(x, shadow[i]))[1]
    end
end

"""
    jacobian(::ForwardMode, f, x; shadow=onehot(x))
    jacobian(::ForwardMode, f, x, ::Val{chunk}; shadow=onehot(x))

Compute the jacobian of an array-input function `f` using (potentially vector)
forward mode. This is a simple rename of the [`gradient`](@ref) function,
and all relevant arguments apply here.

Example:

```jldoctest
f(x) = [x[1]*x[2], x[2]]

grad = jacobian(Forward, f, [2.0, 3.0])

# output

2×2 Matrix{Float64}:
 3.0  2.0
 0.0  1.0
```
"""
@inline function jacobian(::ForwardMode, f, x; shadow=onehot(x))
    cols = if length(x) == 0
        return ()
    else
        values(only(autodiff(Forward, f, BatchDuplicatedNoNeed, BatchDuplicated(x, shadow))))
    end
    reduce(hcat, cols)
end

@inline function jacobian(::ForwardMode, f::F, x::X, ::Val{chunk}; shadow=chunkedonehot(x, Val(chunk))) where {F, X, chunk}
    if chunk == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end
    tmp = ntuple(length(shadow)) do i
        values(autodiff(Forward, f, BatchDuplicatedNoNeed, BatchDuplicated(x, shadow[i]))[1])
    end
    cols = tupleconcat(tmp...)
    reduce(hcat, cols)
end

@inline function jacobian(::ForwardMode, f::F, x::X, ::Val{1}; shadow=onehot(x)) where {F,X}
    cols = ntuple(length(shadow)) do i
        autodiff(Forward, f, DuplicatedNoNeed, Duplicated(x, shadow[i]))[1]
    end
    reduce(hcat, cols)
end

"""
    jacobian(::ReverseMode, f, x, ::Val{num_outs}, ::Val{chunk})

Compute the jacobian of an array-input function `f` using (potentially vector)
reverse mode. The `chunk` argument denotes the chunk size to use and `num_outs`
denotes the number of outputs `f` will return in an array.

Example:

```jldoctest
f(x) = [x[1]*x[2], x[2]]

grad = jacobian(Reverse, f, [2.0, 3.0], Val(2))

# output

2×2 Matrix{Float64}:
 3.0  2.0
 0.0  1.0
```
"""
@inline function jacobian(::ReverseMode, f::F, x::X, n_outs::Val{n_out_val}, ::Val{chunk}) where {F, X, chunk, n_out_val}
    num = ((n_out_val + chunk - 1) ÷ chunk)
    
    if chunk == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end

    tt′   = Tuple{BatchDuplicated{Core.Typeof(x), chunk}}
    tt    = Tuple{Core.Typeof(x)}
    rt = Core.Compiler.return_type(f, tt)
    ModifiedBetween = Val((false, false))
    FA = Const{Core.Typeof(f)}
    primal, adjoint = Enzyme.Compiler.thunk(FA, BatchDuplicatedNoNeed{rt}, tt′, #=Split=# Val(API.DEM_ReverseModeGradient), #=width=#Val(chunk), ModifiedBetween)
    
    if num * chunk == n_out_val
        last_size = chunk
        primal2, adjoint2 = primal, adjoint
    else
        last_size = n_out_val - (num-1)*chunk
        tt′ = Tuple{BatchDuplicated{Core.Typeof(x), last_size}}
        primal2, adjoint2 = Enzyme.Compiler.thunk(FA, BatchDuplicatedNoNeed{rt}, tt′, #=Split=# Val(API.DEM_ReverseModeGradient), #=width=#Val(last_size), ModifiedBetween)
    end

    tmp = ntuple(num) do i
        Base.@_inline_meta
        dx = ntuple(i == num ? last_size : chunk) do idx
            Base.@_inline_meta
            zero(x)
        end
        res = (i == num ? primal2 : primal)(Const(f), BatchDuplicated(x, dx))
        tape = res[1]
        j = 0
        for shadow in res[3]
            j += 1
            @inbounds shadow[(i-1)*chunk+j] += one(eltype(typeof(shadow)))
        end
        (i == num ? adjoint2 : adjoint)(Const(f), BatchDuplicated(x, dx), tape)
        return dx
    end
    rows = tupleconcat(tmp...)
    mapreduce(LinearAlgebra.adjoint, vcat, rows)
end

@inline function jacobian(::ReverseMode, f::F, x::X, n_outs::Val{n_out_val}, ::Val{1} = Val(1)) where {F, X, n_out_val}
    tt′   = Tuple{Duplicated{Core.Typeof(x)}}
    tt    = Tuple{Core.Typeof(x)}
    rt = Core.Compiler.return_type(f, tt)
    ModifiedBetween = Val((false, false))
    FA = Const{Core.Typeof(f)}
    primal, adjoint = Enzyme.Compiler.thunk(FA, DuplicatedNoNeed{rt}, tt′, #=Split=# Val(API.DEM_ReverseModeGradient), #=width=#Val(1), ModifiedBetween)
    rows = ntuple(n_outs) do i
        Base.@_inline_meta
        dx = zero(x)
        res = primal(Const(f), Duplicated(x, dx))
        tape = res[1]
        @inbounds res[3][i] += one(eltype(typeof(res[3])))
        adjoint(Const(f), Duplicated(x, dx), tape)
        return dx
    end
    mapreduce(LinearAlgebra.adjoint, vcat, rows)
end


end # module
