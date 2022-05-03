module Enzyme

export autodiff, autodiff_deferred, fwddiff, fwddiff_deferred, markType
export Const, Active, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed, batch_size
export parallel, pmap

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


"""
    struct BatchDuplicated{T} <: Annotation{T}

Constructor: `BatchDuplicated(x, ∂f_∂x)`

Mark a function argument `x` of [`autodiff`](@ref) as duplicated, Enzyme will
auto-differentiate in respect to such arguments, with `dx` acting as an
accumulator for gradients (so ``\\partial f / \\partial x`` will be *added to*)
`∂f_∂x`.
"""
struct BatchDuplicated{T,N} <: Annotation{T}
    val::T
    dval::NTuple{N,T}
end
batch_size(::BatchDuplicated{T,N}) where {T,N} = N
struct BatchDuplicatedNoNeed{T,N} <: Annotation{T}
    val::T
    dval::NTuple{N,T}
end
batch_size(::BatchDuplicatedNoNeed{T,N}) where {T,N} = N

Base.eltype(::Type{<:Annotation{T}}) where T = T

@inline function guess_activity(::Type{T}, Mode=API.DEM_ReverseModeCombined) where {T}
    return Const{T}
end
@inline function guess_activity(::Type{T}, Mode=API.DEM_ReverseModeCombined) where {T<:AbstractFloat}
    if Mode == API.DEM_ForwardMode
        return DuplicatedNoNeed{T}
    else
        return Active{T}
    end
end
@inline function guess_activity(::Type{T}, Mode=API.DEM_ReverseModeCombined) where {T<:Complex{<:AbstractFloat}}
    if Mode == API.DEM_ForwardMode
        return DuplicatedNoNeed{T}
    else
        return Active{T}
    end
end

@inline function guess_activity(::Type{T}, Mode=API.DEM_ReverseModeCombined) where {T<:AbstractArray}
    if Mode == API.DEM_ForwardMode
        return DuplicatedNoNeed{T}
    else
        return Duplicated{T}
    end
end

import LLVM

include("api.jl")
include("logic.jl")
include("typeanalysis.jl")
include("typetree.jl")
include("utils.jl")
include("compiler.jl")

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

@inline function same_or_one(args::Vararg{Any, N}) where N
    current = -1
    for arg in args
        if arg isa BatchDuplicated
            if current == -1
                current = batch_size(arg)
            else
                @assert current == batch_size(arg)
            end
        elseif arg isa BatchDuplicatedNoNeed
            if current == -1
                current = batch_size(arg)
            else
                @assert current == batch_size(arg)
            end
        end
    end

    if current == -1
        current = 1
    end

    return current
end

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
    width = Val(same_or_one(args...))
    if A <: Active
        tt    = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
        rt = Core.Compiler.return_type(f, tt)
        if !allocatedinline(rt)
            forward, adjoint = Enzyme.Compiler.thunk(f, #=df=#nothing, Duplicated{rt}, tt′, #=Split=# Val(API.DEM_ReverseModeGradient), width, #=ModifiedBetween=#Val(false))
            res = forward(args′...)
            tape = res[1]
            if res[3] isa Base.RefValue
                res[3][] += one(eltype(typeof(res[3])))
            else
                res[3] += one(eltype(typeof(res[3])))
            end
            return adjoint(args′..., tape)
        end
    elseif A <: Duplicated || A<: DuplicatedNoNeed || A <: BatchDuplicated || A<: BatchDuplicatedNoNeed
        throw(ErrorException("Duplicated Returns not yet handled"))
    end
    thunk = Enzyme.Compiler.thunk(f, #=df=#nothing, A, tt′, #=Split=# Val(API.DEM_ReverseModeCombined), width)
    if A <: Active
        tt    = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
        rt = eltype(Compiler.return_type(thunk))
        args′ = (args′..., one(rt))
    end
    thunk(args′...)
end

@inline function autodiff(dupf::Duplicated{F}, ::Type{A}, args...) where {F, A<:Annotation}
    args′  = annotate(args...)
    tt′    = Tuple{map(Core.Typeof, args′)...}
    width = Val(same_or_one(args...))
    thunk = Enzyme.Compiler.thunk(#=f=#dupf.val, #=df=#dupf.dval, A, tt′, #=Split=# Val(API.DEM_ReverseModeCombined), width)
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

@inline function autodiff(dupf::Duplicated{F}, args...) where {F}
    args′ = annotate(args...)
    tt′   = Tuple{map(Core.Typeof, args′)...}
    tt    = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
    rt    = Core.Compiler.return_type(dupf.val, tt)
    A     = guess_activity(rt)
    autodiff(dupf, A, args′...)
end


"""
    autodiff_deferred(f, Activity, args...)

Same as [`autodiff`](@ref) but uses deferred compilation to support usage in GPU
code, as well as high-order differentiation.
"""
@inline function autodiff_deferred(f::F, ::Type{A}, args...) where {F, A<:Annotation}
    args′ = annotate(args...)
    tt′   = Tuple{map(Core.Typeof, args′)...}
    width = Val(same_or_one(args...))
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

    ptr   = Compiler.deferred_codegen(f, Val(tt′), Val(rt), #=dupClosure=#Val(false), Val(API.DEM_ReverseModeCombined), width)
    thunk = Compiler.CombinedAdjointThunk{F, rt, tt′, typeof(width), Nothing}(f, ptr, #=df=#nothing)
    if rt <: Active
        args′ = (args′..., one(eltype(rt)))
    elseif A <: Duplicated || A<: DuplicatedNoNeed || A <: BatchDuplicated || A<: BatchDuplicatedNoNeed
        throw(ErrorException("Duplicated Returns not yet handled"))
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
    autodiff_deferred(f, rt, args′...) 
end

using Adapt
Adapt.adapt_structure(to, x::Duplicated) = Duplicated(adapt(to, x.val), adapt(to, x.dval))
Adapt.adapt_structure(to, x::DuplicatedNoNeed) = DuplicatedNoNeed(adapt(to, x.val), adapt(to, x.dval))
Adapt.adapt_structure(to, x::BatchDuplicated) = BatchDuplicated(adapt(to, x.val), adapt(to, x.dval))
Adapt.adapt_structure(to, x::BatchDuplicatedNoNeed) = BatchDuplicatedNoNeed(adapt(to, x.val), adapt(to, x.dval))
Adapt.adapt_structure(to, x::Const) = Const(adapt(to, x.val))
Adapt.adapt_structure(to, x::Active) = Active(adapt(to, x.val))

@inline function fwddiff(f::F, ::Type{A}, args...) where {F, A<:Annotation}
    args′  = annotate(args...)
    tt′    = Tuple{map(Core.Typeof, args′)...}
    width = Val(same_or_one(args...))
    if A <: Active
        throw(ErrorException("Active Returns not allowed in forward mode"))
    end
    thunk = Enzyme.Compiler.thunk(f, #=df=#nothing, A, tt′, #=Mode=# Val(API.DEM_ForwardMode), width)
    thunk(args′...)
end

@inline function fwddiff(dupf::Duplicated{F}, ::Type{A}, args...) where {F, A<:Annotation}
    args′  = annotate(args...)
    tt′    = Tuple{map(Core.Typeof, args′)...}
    width = Val(same_or_one(args...))
    if A <: Active
        throw(ErrorException("Active Returns not allowed in forward mode"))
    end
    thunk = Enzyme.Compiler.thunk(#=f=#dupf.val, #=df=#dupf.dval, A, tt′, #=Mode=# Val(API.DEM_ForwardMode), width)
    thunk(args′...)
end

"""
    fwddiff(f, args...)

Like [`fwddiff`](@ref) but will try to guess the activity of the return value.
"""
@inline function fwddiff(f::F, args...) where {F}
    args′ = annotate(args...)
    tt    = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
    rt    = Core.Compiler.return_type(f, tt)
    A     = guess_activity(rt, API.DEM_ForwardMode)
    fwddiff(f, A, args′...)
end

@inline function fwddiff(dupf::Duplicated{F}, args...) where {F}
    args′ = annotate(args...)
    tt    = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
    rt    = Core.Compiler.return_type(dupf.val, tt)
    A     = guess_activity(rt, API.DEM_ForwardMode)
    fwddiff(dupf, A, args′...)
end

"""
    fwddiff_deferred(f, Activity, args...)

Same as [`fwddiff`](@ref) but uses deferred compilation to support usage in GPU
code, as well as high-order differentiation.
"""
@inline function fwddiff_deferred(f::F, ::Type{A}, args...) where {F, A<:Annotation}
    args′ = annotate(args...)
    tt′   = Tuple{map(Core.Typeof, args′)...}
    width = Val(same_or_one(args...))
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

    ptr   = Compiler.deferred_codegen(f, Val(tt′), Val(rt), #=dupClosure=#Val(false), Val(API.DEM_ForwardMode), width)
    thunk = Compiler.ForwardModeThunk{F, rt, tt′, typeof(width), Nothing}(f, ptr, #=df=#nothing)
    thunk(args′...)
end

"""
    fwddiff_deferred(f, args...)

Like [`fwddiff_deferred`](@ref) but will try to guess the activity of the return value.
"""
@inline function fwddiff_deferred(f::F, args...) where {F}
    args′ = annotate(args...)
    tt    = Tuple{map(T->eltype(Core.Typeof(T)), args′)...}
    rt    = Core.Compiler.return_type(f, tt)
    rt    = guess_activity(rt, API.DEM_ForwardMode)
    fwddiff_deferred(f, rt, args′...)
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

function pmap(count, body::Body, args::Vararg{Any,N}) where {Body,N}
    ccall(:jl_enter_threaded_region, Cvoid, ())
    n_threads = Base.Threads.nthreads()
    n_gen = min(n_threads, count)
    tasks = Vector{Task}(undef, n_gen)
    cnt = (count + n_gen - 1) ÷ n_gen
    for i = 0:(n_gen-1)
        let start = i * cnt, endv = min(count, (i+1) * cnt)-1
        t = Task() do
           for j in start:endv
              body(j+1, args...)
           end
           nothing
        end
        t.sticky = true
        ccall(:jl_set_task_tid, Cint, (Any, Cint), t, i)
        @inbounds tasks[i+1] = t
        schedule(t)
        end
    end
    try
        for t in tasks
            wait(t)
        end
    finally
        ccall(:jl_exit_threaded_region, Cvoid, ())
    end
end

function pmap_(count, body::Body, args::Vararg{Any,N}) where {Body,N}
  for i in 1:count
    body(i, args...)
  end
  nothing
end

macro parallel(args...)
  captured = args[1:end-1]
  ex = args[end]
  if !(isa(ex, Expr) && ex.head === :for)
    throw(ArgumentError("@parallel requires a `for` loop expression"))
  end
  if !(ex.args[1] isa Expr && ex.args[1].head === :(=))
        throw(ArgumentError("nested outer loops are not currently supported by @parallel"))
   end
   iter = ex.args[1]
   lidx = iter.args[1]         # index
   range = iter.args[2]
   body = ex.args[2]
   esc(quote
     let range = $(range)
       function bodyf(idx, iter, $(captured...))
         local $(lidx) = @inbounds iter[idx]
         $(body)
         nothing
       end
       lenr = length(range)
       pmap(lenr, bodyf, range, $(captured...))
     end
   end)
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

@inline function revgradient(f, x)
    dx = zeros(x)
    fwddiff(f, Duplicated(x, dx))
    dx
end

# Like ForwardDiff.gradient(f, x, ForwardDiff.GradientConfig(sum, x, ForwardDiff.Chunk{length(x)}()))
@inline function fwdgradient(f, x; shadow=onehot(x))
    fwddiff(f, BatchDuplicatedNoNeed, BatchDuplicated(x, shadow))
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

@inline function fwdgradient(f, x, ::Val{chunk}; shadow=chunkedonehot(x, Val(chunk))) where chunk
    tmp = ntuple(length(shadow)) do i
        fwddiff(f, BatchDuplicatedNoNeed, BatchDuplicated(x, shadow[i]))[1]
    end
    tupleconcat(tmp...)
end

@inline function fwdgradient(f, x, ::Val{1}; shadow=onehot(x))
    ntuple(length(shadow)) do i
        fwddiff(f, DuplicatedNoNeed, Duplicated(x, shadow[i]))[1]
    end
end

@inline function fwdjacobian(args...; kwargs...)
    fwdgradient(args...; kwargs...)
end

@inline function revjacobian(f, x, ::Val{chunk}; n_outs::Val{n_out_val}) where {chunk, n_out_val}
    num = ((n_out_val + chunk - 1) ÷ chunk)

    tt′    = Tuple{BatchDuplicated{Core.Typeof(x), chunk}}
    tt    = Tuple{Core.Typeof(x)}
    rt = Core.Compiler.return_type(f, tt)
    forward, adjoint = Enzyme.Compiler.thunk(f, #=df=#nothing, BatchDuplicatedNoNeed{rt}, tt′, #=Split=# Val(API.DEM_ReverseModeGradient), #=width=#Val(chunk), #=ModifiedBetween=#Val(false))
    
    if num * chunk == n_out_val
        last_size = chunk
        forward2, adjoint2 = forward, adjoint
    else
        last_size = n_out_val - (num-1)*chunk
        tt′    = Tuple{BatchDuplicated{Core.Typeof(x), last_size}}
        forward2, adjoint2 = Enzyme.Compiler.thunk(f, #=df=#nothing, BatchDuplicatedNoNeed{rt}, tt′, #=Split=# Val(API.DEM_ReverseModeGradient), #=width=#Val(last_size), #=ModifiedBetween=#Val(false))
    end

    tmp = ntuple(num) do i
        Base.@_inline_meta
        dx = ntuple(i == num ? last_size : chunk) do idx
            Base.@_inline_meta
            zero(x)
        end
        res = (i == num ? forward2 : forward)(BatchDuplicated(x, dx))
        tape = res[1]
        res[2][i] += one(eltype(typeof(res[2])))
        (i == num ? adjoint2 : adjoint)(BatchDuplicated(x, dx), tape)
        return dx
    end
    tupleconcat(tmp...)
end

@inline function revjacobian(f, x, ::Val{1}; n_outs::Val{n_out_val}) where {n_out_val}
    tt′    = Tuple{Duplicated{Core.Typeof(x)}}
    tt    = Tuple{Core.Typeof(x)}
    rt = Core.Compiler.return_type(f, tt)
    forward, adjoint = Enzyme.Compiler.thunk(f, #=df=#nothing, DuplicatedNoNeed{rt}, tt′, #=Split=# Val(API.DEM_ReverseModeGradient), #=width=#Val(1), #=ModifiedBetween=#Val(false))
    ntuple(n_outs) do i
        Base.@_inline_meta
        dx = zero(x)
        res = forward(Duplicated(x, dx))
        tape = res[1]
        res[2][i] += one(eltype(typeof(res[2])))
        adjoint(Duplicated(x, dx), tape)
        return dx
    end
end

end # module
