module Enzyme

export autodiff, autodiff_deferred, fwddiff, fwddiff_deferred, markType
export Const, Active, Duplicated, DuplicatedNoNeed
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

Base.eltype(::Type{<:Annotation{T}}) where T = T

function guess_activity(T, Mode=API.DEM_ReverseModeCombined)
    if T <: AbstractFloat || T <: Complex{<:AbstractFloat}
        if Mode == API.DEM_ForwardMode
            return DuplicatedNoNeed{T}
        else
            return Active{T}
        end
    elseif T <: AbstractArray
        if Mode == API.DEM_ForwardMode
            return DuplicatedNoNeed{T}
        else
            return Duplicated{T}
        end
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

import .Compiler: CompilationException

include("JET.jl")

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

# annotated args to argtypes
getargtypes(args′) = Tuple{map(@nospecialize(t)->eltype(Core.Typeof(t)), args′)...}

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
    if A <: Active
        rt = Core.Compiler.return_type(f, getargtypes(args′))
        if !allocatedinline(rt)
            forward, adjoint = Enzyme.Compiler.thunk(f, #=df=#nothing, Duplicated{rt}, tt′, #=Split=# Val(API.DEM_ReverseModeGradient))
            res = forward(args′...)
            tape = res[1]
            res3 = res[3]
            if res3 isa Base.RefValue
                res3[] += one(eltype(res3))
            else
                res3 += one(eltype(res3))
            end
            return adjoint(args′..., tape)
        end
    elseif A <: Duplicated
        throw(ErrorException("Duplicated Returns not yet handled"))
    end
    thunk = Enzyme.Compiler.thunk(f, #=df=#nothing, A, tt′, #=Split=# Val(API.DEM_ReverseModeCombined))
    rt = eltype(Compiler.return_type(thunk))
    if A <: Active
        args′ = (args′..., one(rt))
    end
    thunk(args′...)
end

@inline function autodiff(dupf::Duplicated{F}, ::Type{A}, args...) where {F, A<:Annotation}
    args′  = annotate(args...)
    tt′    = Tuple{map(Core.Typeof, args′)...}
    thunk = Enzyme.Compiler.thunk(#=f=#dupf.val, #=df=#dupf.dval, A, tt′, #=Split=# Val(API.DEM_ReverseModeCombined))
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
    rt    = Core.Compiler.return_type(f, getargtypes(args′))
    A     = guess_activity(rt)
    autodiff(f, A, args′...)
end

@inline function autodiff(dupf::Duplicated{F}, args...) where {F}
    args′ = annotate(args...)
    rt    = Core.Compiler.return_type(dupf.val, getargtypes(args′))
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

    ptr   = Compiler.deferred_codegen(Val(f), Val(tt′), Val(rt))
    thunk = Compiler.CombinedAdjointThunk{F, rt, tt′, Nothing}(f, ptr, #=df=#nothing)
    if rt <: Active
        args′ = (args′..., one(eltype(rt)))
    elseif A <: Duplicated
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
Adapt.adapt_structure(to, x::Const) = Const(adapt(to, x.val))
Adapt.adapt_structure(to, x::Active) = Active(adapt(to, x.val))


@inline function fwddiff(f::F, ::Type{A}, args...) where {F, A<:Annotation}
    args′  = annotate(args...)
    tt′    = Tuple{map(Core.Typeof, args′)...}
    if A <: Active
        throw(ErrorException("Active Returns not allowed in forward mode"))
    end
    thunk = Enzyme.Compiler.thunk(f, #=df=#nothing, A, tt′, #=Mode=# Val(API.DEM_ForwardMode))
    thunk(args′...)
end

@inline function fwddiff(dupf::Duplicated{F}, ::Type{A}, args...) where {F, A<:Annotation}
    args′  = annotate(args...)
    tt′    = Tuple{map(Core.Typeof, args′)...}
    if A <: Active
        throw(ErrorException("Active Returns not allowed in forward mode"))
    end
    thunk = Enzyme.Compiler.thunk(#=f=#dupf.val, #=df=#dupf.dval, A, tt′, #=Mode=# Val(API.DEM_ForwardMode))
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

    ptr   = Compiler.deferred_codegen(Val(f), Val(tt′), Val(rt), #=dupClosure=#Val(false), Val(API.DEM_ForwardMode))
    thunk = Compiler.ForwardModeThunk{F, rt, tt′, Nothing}(f, ptr, #=df=#nothing)
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

end # module
