module Enzyme

import EnzymeCore: Forward, Reverse, ReverseWithPrimal
export Forward, Reverse, ReverseWithPrimal

import EnzymeCore: Const, Active, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed
export Const, Active, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed

import EnzymeCore: batch_size
export batch_size

import EnzymeCore: autodiff, autodiff_deferred
export autodiff, autodiff_deferred

export jacobian, gradient, gradient!
export markType, batch_size, onehot, chunkedonehot

using LinearAlgebra
import EnzymeCore: ReverseMode, ForwardMode, Annotation, Mode

import EnzymeCore: EnzymeRules
export EnzymeRules

# Independent code, must be loaded before "compiler.jl"
include("pmap.jl")

import LLVM
include("api.jl")

Base.convert(::Type{API.CDerivativeMode}, ::ReverseMode{<:Any, false}) = API.DEM_ReverseModeCombined
Base.convert(::Type{API.CDerivativeMode}, ::ReverseMode{<:Any, true}) = API.DEM_ReverseModeGradient
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
@inline same_or_one_rec(current, arg::BatchDuplicatedNoNeed{T, N}, args...) where {T,N} =
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
@inline function autodiff(::ReverseMode{ReturnPrimal}, f::F, ::Type{A}, args...) where {F, A<:Annotation, ReturnPrimal}
    args′  = annotate(args...)
    tt′    = Tuple{map(Core.Typeof, args′)...}
    width = same_or_one(args...)
    if width == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end
    if A <: Active
        tt    = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
        rt = Core.Compiler.return_type(f, tt)
        if !allocatedinline(rt) || rt isa Union
            forward, adjoint = Enzyme.Compiler.thunk(f, #=df=#nothing, Duplicated{rt}, tt′, #=Split=# Val(API.DEM_ReverseModeGradient), Val(width), #=ModifiedBetween=#Val(false), #=ReturnPrimal=#Val(ReturnPrimal), #=ShadowInit=#Val(true))
            res = forward(args′...)
            tape = res[1]
            if ReturnPrimal
                return (adjoint(args′..., tape)[1], res[2])
            else
                return adjoint(args′..., tape)
            end
        end
    elseif A <: Duplicated || A<: DuplicatedNoNeed || A <: BatchDuplicated || A<: BatchDuplicatedNoNeed
        throw(ErrorException("Duplicated Returns not yet handled"))
    end
    thunk = Enzyme.Compiler.thunk(f, #=df=#nothing, A, tt′, #=Split=# Val(API.DEM_ReverseModeCombined), Val(width), #=ModifiedBetween=#Val(false), Val(ReturnPrimal))
    if A <: Active
        tt    = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
        rt = eltype(Compiler.return_type(thunk))
        args′ = (args′..., one(rt))
    end
    thunk(args′...)
end

@inline function autodiff(::ReverseMode{ReturnPrimal}, dupf::Duplicated{F}, ::Type{A}, args...) where {F, A<:Annotation, ReturnPrimal}
    args′  = annotate(args...)
    tt′    = Tuple{map(Core.Typeof, args′)...}
    width = same_or_one(args...)
    if width == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end
    thunk = Enzyme.Compiler.thunk(#=f=#dupf.val, #=df=#dupf.dval, A, tt′, #=Split=# Val(API.DEM_ReverseModeCombined), Val(width), #=ModifiedBetween=#Val(false), Val(ReturnPrimal))
    if A <: Active
        rt = eltype(Compiler.return_type(thunk))
        args′ = (args′..., one(rt))
    end
    thunk(args′...)
end


"""
    autodiff(mode::Mode, f, args...)
    autodiff(f, mode::Mode, args...)

Like [`autodiff`](@ref) but will try to guess the activity of the return value.
"""
@inline function autodiff(mode::Mode, f::F, args...) where {F}
    args′ = annotate(args...)
    tt′   = Tuple{map(Core.Typeof, args′)...}
    tt    = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
    rt    = Core.Compiler.return_type(f, tt)
    A     = guess_activity(rt, mode)
    autodiff(mode, f, A, args′...)
end

@inline function autodiff(mode::Mode, dupf::Duplicated{F}, args...) where {F}
    args′ = annotate(args...)
    tt′   = Tuple{map(Core.Typeof, args′)...}
    tt    = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
    rt    = Core.Compiler.return_type(dupf.val, tt)
    A     = guess_activity(rt, mode)
    autodiff(mode, dupf, A, args′...)
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
@inline function autodiff(::ForwardMode, f::F, ::Type{A}, args...) where {F, A<:Annotation}
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
        BatchDuplicated{eltype(A), width}
    elseif A <: DuplicatedNoNeed && width != 1
        BatchDuplicatedNoNeed{eltype(A), width}
    else
        A
    end

    thunk = Enzyme.Compiler.thunk(f, #=df=#nothing, RT, tt′, #=Mode=# Val(API.DEM_ForwardMode), Val(width),
                                     #=ModifiedBetween=#Val(false), ReturnPrimal)
    thunk(args′...)
end

@inline function autodiff(::ForwardMode, dupf::Duplicated{F}, ::Type{A}, args...) where {F, A<:Annotation}
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
        BatchDuplicated{eltype(A), width}
    elseif A <: DuplicatedNoNeed && width != 1
        BatchDuplicatedNoNeed{eltype(A), width}
    else
        A
    end
    thunk = Enzyme.Compiler.thunk(#=f=#dupf.val, #=df=#dupf.dval, RT, tt′, #=Mode=# Val(API.DEM_ForwardMode), Val(width), #=ModifiedBetween=#Val(false), ReturnPrimal)
    thunk(args′...)
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
@inline function autodiff_deferred(::ReverseMode{ReturnPrimal}, f::F, ::Type{A}, args...) where {F, A<:Annotation, ReturnPrimal}
    args′ = annotate(args...)
    tt′   = Tuple{map(Core.Typeof, args′)...}
    width = same_or_one(args...)
    if width == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end
    if A isa UnionAll
        tt = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
        rt = Core.Compiler.return_type(f, tt)
        rt = A{rt}
    else
        @assert A isa DataType
        rt = A
    end

    if eltype(rt) == Union{}
        error("Return type inferred to be Union{}. Giving up.")
    end

    ptr   = Compiler.deferred_codegen(f, Val(tt′), Val(rt), #=dupClosure=#Val(false), Val(API.DEM_ReverseModeCombined), Val(width), #=ModifiedBetween=#Val(false), Val(ReturnPrimal))
    thunk = Compiler.CombinedAdjointThunk{F, rt, tt′, typeof(Val(width)), Nothing, Val(ReturnPrimal)}(f, ptr, #=df=#nothing)
    if rt <: Active
        args′ = (args′..., one(eltype(rt)))
    elseif A <: Duplicated || A<: DuplicatedNoNeed || A <: BatchDuplicated || A<: BatchDuplicatedNoNeed
        throw(ErrorException("Duplicated Returns not yet handled"))
    end
    thunk(args′...)
end

"""
    autodiff_deferred(::ForwardMode, f, Activity, args...)

Same as `autodiff(::ForwardMode, ...)` but uses deferred compilation to support usage in GPU
code, as well as high-order differentiation.
"""
@inline function autodiff_deferred(::ForwardMode, f::F, ::Type{A}, args...) where {F, A<:Annotation}
    args′ = annotate(args...)
    tt′   = Tuple{map(Core.Typeof, args′)...}
    width = same_or_one(args...)
    if width == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end
    if A isa UnionAll
        tt = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
        rt = Core.Compiler.return_type(f, tt)
        rt = A{rt}
    else
        @assert A isa DataType
        rt = A
    end

    if eltype(rt) == Union{}
        error("Return type inferred to be Union{}. Giving up.")
    end

    if A <: Active
        throw(ErrorException("Active Returns not allowed in forward mode"))
    end

    ReturnPrimal = Val(A <: Duplicated || A <: BatchDuplicated)
    ptr   = Compiler.deferred_codegen(f, Val(tt′), Val(rt), #=dupClosure=#Val(false), Val(API.DEM_ForwardMode), Val(width), #=ModifiedBetween=#Val(false), ReturnPrimal)
    thunk = Compiler.ForwardModeThunk{F, rt, tt′, typeof(Val(width)), Nothing, ReturnPrimal}(f, ptr, #=df=#nothing)
    thunk(args′...)
end

"""
    autodiff_deferred(mode, f, args...)

Like [`autodiff_deferred`](@ref) but will try to guess the activity of the return value.
"""
@inline function autodiff_deferred(mode::SMode, f::F, args...) where {F, SMode<:Mode}
    args′ = annotate(args...)
    tt    = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
    rt    = Core.Compiler.return_type(f, tt)
    rt    = guess_activity(rt, mode)
    autodiff_deferred(mode, f, rt, args′...)
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
    primal, adjoint = Enzyme.Compiler.thunk(f, #=df=#nothing, BatchDuplicatedNoNeed{rt}, tt′, #=Split=# Val(API.DEM_ReverseModeGradient), #=width=#Val(chunk), #=ModifiedBetween=#Val(false))
    
    if num * chunk == n_out_val
        last_size = chunk
        primal2, adjoint2 = primal, adjoint
    else
        last_size = n_out_val - (num-1)*chunk
        tt′ = Tuple{BatchDuplicated{Core.Typeof(x), last_size}}
        primal2, adjoint2 = Enzyme.Compiler.thunk(f, #=df=#nothing, BatchDuplicatedNoNeed{rt}, tt′, #=Split=# Val(API.DEM_ReverseModeGradient), #=width=#Val(last_size), #=ModifiedBetween=#Val(false))
    end

    tmp = ntuple(num) do i
        Base.@_inline_meta
        dx = ntuple(i == num ? last_size : chunk) do idx
            Base.@_inline_meta
            zero(x)
        end
        res = (i == num ? primal2 : primal)(BatchDuplicated(x, dx))
        tape = res[1]
        j = 0
        for shadow in res[2]
            j += 1
            @inbounds shadow[(i-1)*chunk+j] += one(eltype(typeof(shadow)))
        end
        (i == num ? adjoint2 : adjoint)(BatchDuplicated(x, dx), tape)
        return dx
    end
    rows = tupleconcat(tmp...)
    mapreduce(LinearAlgebra.adjoint, vcat, rows)
end

@inline function jacobian(::ReverseMode, f::F, x::X, n_outs::Val{n_out_val}, ::Val{1} = Val(1)) where {F, X, n_out_val}
    tt′   = Tuple{Duplicated{Core.Typeof(x)}}
    tt    = Tuple{Core.Typeof(x)}
    rt = Core.Compiler.return_type(f, tt)
    primal, adjoint = Enzyme.Compiler.thunk(f, #=df=#nothing, DuplicatedNoNeed{rt}, tt′, #=Split=# Val(API.DEM_ReverseModeGradient), #=width=#Val(1), #=ModifiedBetween=#Val(false))
    rows = ntuple(n_outs) do i
        Base.@_inline_meta
        dx = zero(x)
        res = primal(Duplicated(x, dx))
        tape = res[1]
        @inbounds res[2][i] += one(eltype(typeof(res[2])))
        adjoint(Duplicated(x, dx), tape)
        return dx
    end
    mapreduce(LinearAlgebra.adjoint, vcat, rows)
end


end # module
