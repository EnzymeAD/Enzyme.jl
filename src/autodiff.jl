
@inline function falses_from_args(N)
    ntuple(Val(N)) do i
        Base.@_inline_meta
        false
    end
end

@inline function any_active(args::Vararg{Annotation, N}) where N
    any(ntuple(Val(N)) do i
        Base.@_inline_meta
        arg = @inbounds args[i]
        if arg isa Active
            return true
        elseif arg isa MixedDuplicated
            return true
        elseif arg isa BatchMixedDuplicated
            return true
        else
            return false
        end
    end)
end

@inline function vaTypeof(args::Vararg{Any, N}) where N
    return Tuple{(ntuple(Val(N)) do i
        Base.@_inline_meta
        Core.Typeof(args[i])
    end)...}
end

@inline function vaEltypes(args::Type{Ty}) where {Ty <: Tuple}
    return Tuple{(ntuple(Val(length(Ty.parameters))) do i
        Base.@_inline_meta
        eltype(Ty.parameters[i])
    end)...}
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
@inline same_or_one_rec(current, arg::BatchMixedDuplicated{T, N}, args...) where {T,N} =
   same_or_one_rec(same_or_one_helper(current, N), args...)
@inline same_or_one_rec(current, arg::Type{BatchMixedDuplicated{T, N}}, args...) where {T,N} =
   same_or_one_rec(same_or_one_helper(current, N), args...)
@inline same_or_one_rec(current, arg::BatchDuplicatedFunc{T, N}, args...) where {T,N} =
   same_or_one_rec(same_or_one_helper(current, N), args...)
@inline same_or_one_rec(current, arg::Type{BatchDuplicatedFunc{T, N}}, args...) where {T,N} =
   same_or_one_rec(same_or_one_helper(current, N), args...)
@inline same_or_one_rec(current, arg::BatchDuplicated{T, N}, args...) where {T,N} =
   same_or_one_rec(same_or_one_helper(current, N), args...)
@inline same_or_one_rec(current, arg::Type{BatchDuplicated{T, N}}, args...) where {T,N} =
   same_or_one_rec(same_or_one_helper(current, N), args...)
@inline same_or_one_rec(current, arg::BatchDuplicatedNoNeed{T, N}, args...) where {T,N} =
   same_or_one_rec(same_or_one_helper(current, N), args...)
@inline same_or_one_rec(current, arg::Type{BatchDuplicatedNoNeed{T, N}}, args...) where {T,N} =
   same_or_one_rec(same_or_one_helper(current, N), args...)
@inline same_or_one_rec(current, arg, args...) = same_or_one_rec(current, args...)

@inline function same_or_one(defaultVal, args...)
    local_soo_res = same_or_one_rec(-1, args...)
    if local_soo_res == -1
        defaultVal
    else
        local_soo_res
    end
end


@inline function refn_seed(x::T) where T
    if T <: Complex
        return conj(x) / 2
    else
        return x
    end
end

@inline function imfn_seed(x::T) where T
    if T <: Complex
        return im * conj(x) / 2
    else
        return T(0)
    end
end

@inline function seed_complex_args(seen, seen2, args::Vararg{Annotation, Nargs}) where {Nargs}
    return ntuple(Val(Nargs)) do i
        Base.@_inline_meta
        arg = args[i]
        if arg isa Const || arg isa Active
            arg
        elseif arg isa Duplicated || arg isa DuplicatedNoNeed
            RT = eltype(Core.Typeof(arg))
            BatchDuplicated(arg.val, (arg.dval, make_zero(RT, seen, arg.dval), make_zero(RT, seen2, arg.dval)))
        else
            throw(ErrorException("Active Complex return does not yet support batching in combined reverse mode"))
        end
    end
end

@inline function fuse_complex_results(results, args::Vararg{Annotation, Nargs}) where {Nargs}
    ntuple(Val(Nargs)) do i
        Base.@_inline_meta
        if args[i] isa Active
            Compiler.recursive_add(Compiler.recursive_add(results[1][i][1], results[1][i][2], refn_seed), results[1][i][3], imfn_seed)
        else
            results[1][i]
        end
    end
end

"""
    autodiff(::ReverseMode, f, Activity, args::Vararg{<:Annotation, Nargs})

Auto-differentiate function `f` at arguments `args` using reverse mode.

Limitations:

* `f` may only return a `Real` (of a built-in/primitive type) or `nothing`,
  not an array, struct, `BigFloat`, etc. To handle vector-valued return
  types, use a mutating `f!` that returns `nothing` and stores it's return
  value in one of the arguments, which must be wrapped in a
  [`Duplicated`](@ref).

`args` may be numbers, arrays, structs of numbers, structs of arrays and so
on. Enzyme will only differentiate in respect to arguments that are wrapped
in an [`Active`](@ref) (for arguments whose derivative result must be returned
rather than mutated in place, such as primitive types and structs thereof)
or [`Duplicated`](@ref) (for mutable arguments like arrays, `Ref`s and structs
thereof).

`Activity` is the Activity of the return value, it may be `Const` or `Active`.

Example:

```jldoctest
a = 4.2
b = [2.2, 3.3]; ∂f_∂b = zero(b)
c = 55; d = 9

f(a, b, c, d) = a * √(b[1]^2 + b[2]^2) + c^2 * d^2
∂f_∂a, _, _, ∂f_∂d = autodiff(Reverse, f, Active, Active(a), Duplicated(b, ∂f_∂b), Const(c), Active(d))[1]

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
@inline function autodiff(rmode::ReverseMode{ReturnPrimal, RABI,Holomorphic, ErrIfFuncWritten}, f::FA, ::Type{A}, args::Vararg{Annotation, Nargs}) where {FA<:Annotation, A<:Annotation, ReturnPrimal, RABI<:ABI,Holomorphic, Nargs, ErrIfFuncWritten}
    tt′   = vaTypeof(args...)
    width = same_or_one(1, args...)
    if width == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end

    ModifiedBetween = Val(falses_from_args(Nargs+1))

    tt    = Tuple{map(T->eltype(Core.Typeof(T)), args)...}

    FTy = Core.Typeof(f.val)

    opt_mi = if RABI <: NonGenABI
        Compiler.fspec(eltype(FA), tt′)
    else
        Val(codegen_world_age(FTy, tt))
    end

    rt = if A isa UnionAll
        @static if VERSION >= v"1.8.0"
            Compiler.primal_return_type(rmode, Val(codegen_world_age(FTy, tt)), FTy, tt)
        else
            Core.Compiler.return_type(f.val, tt)
        end
    else
        eltype(A)    
    end

    if A <: Active
        if (!allocatedinline(rt) || rt isa Union) && rt != Union{}
            forward, adjoint = Enzyme.Compiler.thunk(opt_mi, FA, Duplicated{rt}, tt′, #=Split=# Val(API.DEM_ReverseModeGradient), Val(width), ModifiedBetween, #=ReturnPrimal=#Val(ReturnPrimal), #=ShadowInit=#Val(true), RABI, Val(ErrIfFuncWritten))
            res = forward(f, args...)
            tape = res[1]
            if ReturnPrimal
                return (adjoint(f, args..., tape)[1], res[2])
            else
                return adjoint(f, args..., tape)
            end
        end
    elseif A <: Duplicated || A<: DuplicatedNoNeed || A <: BatchDuplicated || A<: BatchDuplicatedNoNeed || A <: BatchDuplicatedFunc
        throw(ErrorException("Duplicated Returns not yet handled"))
    end

    if (A <: Active && rt <: Complex) && rt != Union{}
        if Holomorphic
            seen = IdDict()
            seen2 = IdDict()

            f = if f isa Const || f isa Active
                f
            elseif f isa Duplicated || f isa DuplicatedNoNeed
                BatchDuplicated(f.val, (f.dval, make_zero(typeof(f), seen, f.dval), make_zero(typeof(f), seen2, f.dval)))
            else
                throw(ErrorException("Active Complex return does not yet support batching in combined reverse mode"))
            end

            width = same_or_one(3, args...)
            args = seed_complex_args(seen, seen2, args...)
            tt′   = vaTypeof(args...)

            thunk = Enzyme.Compiler.thunk(opt_mi, typeof(f), A, tt′, #=Split=# Val(API.DEM_ReverseModeCombined), Val(width), ModifiedBetween, #=ReturnPrimal=#Val(ReturnPrimal), #=ShadowInit=#Val(false), RABI, Val(ErrIfFuncWritten))

            results = thunk(f, args..., (rt(0), rt(1), rt(im)))

            # compute the correct complex derivative in reverse mode by propagating the conjugate return values
            # then subtracting twice the imaginary component to get the correct result

            for (k, v) in seen
                Compiler.recursive_accumulate(k, v, refn_seed)
            end
            for (k, v) in seen2
                Compiler.recursive_accumulate(k, v, imfn_seed)
            end

            fused = fuse_complex_results(results, args...)

            return (fused, results[2:end]...)
        end

        throw(ErrorException("Reverse-mode Active Complex return is ambiguous and requires more information to specify the desired result. See https://enzyme.mit.edu/julia/stable/faq/#Complex-numbers for more details."))
    end

    thunk = Enzyme.Compiler.thunk(opt_mi, FA, A, tt′, #=Split=# Val(API.DEM_ReverseModeCombined), Val(width), ModifiedBetween, Val(ReturnPrimal), #=ShadowInit=#Val(false), RABI, Val(ErrIfFuncWritten))

    if A <: Active
        args = (args..., Compiler.default_adjoint(rt))
    end
    thunk(f, args...)
end

"""
    autodiff(mode::Mode, f, ::Type{A}, args::Vararg{Annotation, Nargs})

Like [`autodiff`](@ref) but will try to extend f to an annotation, if needed.
"""
@inline function autodiff(mode::CMode, f::F, args::Vararg{Annotation, Nargs}) where {F, CMode<:Mode, Nargs}
    autodiff(EnzymeCore.set_err_if_func_written(mode), Const(f), args...)
end
@inline function autodiff(mode::CMode, f::F, ::Type{RT}, args::Vararg{Annotation, Nargs}) where {F, RT<:Annotation, CMode<:Mode, Nargs}
    autodiff(EnzymeCore.set_err_if_func_written(mode), Const(f), RT, args...)
end

"""
    autodiff(mode::Mode, f, args...)

Like [`autodiff`](@ref) but will try to guess the activity of the return value.
"""
@inline function autodiff(mode::CMode, f::FA, args::Vararg{Annotation, Nargs}) where {FA<:Annotation, CMode<:Mode, Nargs}
    tt    = Tuple{map(T->eltype(Core.Typeof(T)), args)...}
    rt    = if mode isa ReverseMode && VERSION >= v"1.8.0"
        Compiler.primal_return_type(mode, Val(codegen_world_age(eltype(FA), tt)), eltype(FA), tt)
    else
        Core.Compiler.return_type(f.val, tt)
    end
    A     = guess_activity(rt, mode)
    autodiff(mode, f, A, args...)
end

"""
    autodiff(::ForwardMode, f, Activity, args::Vararg{<:Annotation, Nargs})

Auto-differentiate function `f` at arguments `args` using forward mode.

`args` may be numbers, arrays, structs of numbers, structs of arrays and so
on. Enzyme will only differentiate in respect to arguments that are wrapped
in a [`Duplicated`](@ref) or similar argument. Unlike reverse mode in
[`autodiff`](@ref), [`Active`](@ref) arguments are not allowed here, since
all derivative results of immutable objects will be returned and should
instead use [`Duplicated`](@ref) or variants like [`DuplicatedNoNeed`](@ref).

`Activity` is the Activity of the return value, it may be:
* `Const` if the return is not to be differentiated with respect to
* `Duplicated`, if the return is being differentiated with respect to and
  both the original value and the derivative return are desired
* `DuplicatedNoNeed`, if the return is being differentiated with respect to
  and only the derivative return is desired.
* `BatchDuplicated`, like `Duplicated`, but computing multiple derivatives
  at once. All batch sizes must be the same for all arguments.
* `BatchDuplicatedNoNeed`, like `DuplicatedNoNeed`, but computing multiple
  derivatives at one. All batch sizes must be the same for all arguments.

Example returning both original return and derivative:

```jldoctest
f(x) = x*x
res, ∂f_∂x = autodiff(Forward, f, Duplicated, Duplicated(3.14, 1.0))

# output

(9.8596, 6.28)
```

Example returning just the derivative:

```jldoctest
f(x) = x*x
∂f_∂x = autodiff(Forward, f, DuplicatedNoNeed, Duplicated(3.14, 1.0))

# output

(6.28,)
```
"""
@inline function autodiff(::ForwardMode{RABI, ErrIfFuncWritten}, f::FA, ::Type{A}, args::Vararg{Annotation, Nargs}) where {FA<:Annotation, A<:Annotation} where {RABI <: ABI, Nargs, ErrIfFuncWritten}
    if any_active(args...)
        throw(ErrorException("Active arguments not allowed in forward mode"))
    end
    tt′   = vaTypeof(args...)
    width = same_or_one(1, args...)
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
    
    ModifiedBetween = Val(falses_from_args(Nargs+1))
    
    tt    = Tuple{map(T->eltype(Core.Typeof(T)), args)...}

    opt_mi = if RABI <: NonGenABI
        Compiler.fspec(eltype(FA), tt′)
    else
        Val(codegen_world_age(Core.Typeof(f.val), tt))
    end

    thunk = Enzyme.Compiler.thunk(opt_mi, FA, RT, tt′, #=Mode=# Val(API.DEM_ForwardMode), Val(width),
                                     ModifiedBetween, ReturnPrimal, #=ShadowInit=#Val(false), RABI, Val(ErrIfFuncWritten))
    thunk(f, args...)
end

"""
    autodiff_deferred(::ReverseMode, f, Activity, args::Vararg{<:Annotation, Nargs})

Same as [`autodiff`](@ref) but uses deferred compilation to support usage in GPU
code, as well as high-order differentiation.
"""
@inline function autodiff_deferred(::ReverseMode{ReturnPrimal, ABI,Holomorphic,ErrIfFuncWritten}, f::FA, ::Type{A}, args::Vararg{Annotation, Nargs}) where {FA<:Annotation, A<:Annotation, ReturnPrimal, Nargs, ABI,Holomorphic,ErrIfFuncWritten}
    tt′   = vaTypeof(args...)
    width = same_or_one(1, args...)
    if width == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end
    tt = Tuple{map(T->eltype(Core.Typeof(T)), args)...}
        
    world = codegen_world_age(Core.Typeof(f.val), tt)
    
    if A isa UnionAll
        rt = Core.Compiler.return_type(f.val, tt)
        rt = A{rt}
    else
        @assert A isa DataType
        rt = A
    end

    if eltype(rt) == Union{}
        error("Return type inferred to be Union{}. Giving up.")
    end

    ModifiedBetween = Val(falses_from_args(Nargs+1))

    adjoint_ptr = Compiler.deferred_codegen(Val(world), FA, Val(tt′), Val(rt), Val(API.DEM_ReverseModeCombined), Val(width), ModifiedBetween, Val(ReturnPrimal), #=ShadowInit=#Val(false), UnknownTapeType, Val(ErrIfFuncWritten))

    thunk = Compiler.CombinedAdjointThunk{Ptr{Cvoid}, FA, rt, tt′, width, ReturnPrimal}(adjoint_ptr)
    if rt <: Active
        args = (args..., Compiler.default_adjoint(eltype(rt)))
    elseif A <: Duplicated || A<: DuplicatedNoNeed || A <: BatchDuplicated || A<: BatchDuplicatedNoNeed
        throw(ErrorException("Duplicated Returns not yet handled"))
    end
    thunk(f, args...)
end

"""
    autodiff_deferred(::ForwardMode, f, Activity, args::Vararg{<:Annotation, Nargs})

Same as `autodiff(::ForwardMode, f, Activity, args)` but uses deferred compilation to support usage in GPU
code, as well as high-order differentiation.
"""
@inline function autodiff_deferred(::ForwardMode{ABI, ErrIfFuncWritten}, f::FA, ::Type{A}, args::Vararg{Annotation, Nargs}) where {FA<:Annotation, A<:Annotation, Nargs, ABI, ErrIfFuncWritten}
    if any_active(args...)
        throw(ErrorException("Active arguments not allowed in forward mode"))
    end
    tt′   = vaTypeof(args...)
    width = same_or_one(1, args...)
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
    tt = Tuple{map(T->eltype(Core.Typeof(T)), args)...}
    
    world = codegen_world_age(Core.Typeof(f.val), tt)
    
    if RT isa UnionAll
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

    ReturnPrimal = RT <: Duplicated || RT <: BatchDuplicated
    ModifiedBetween = Val(falses_from_args(Nargs+1))
    
    adjoint_ptr = Compiler.deferred_codegen(Val(world), FA, Val(tt′), Val(rt), Val(API.DEM_ForwardMode), Val(width), ModifiedBetween, Val(ReturnPrimal), #=ShadowInit=#Val(false), UnknownTapeType, Val(ErrIfFuncWritten))
    thunk = Compiler.ForwardModeThunk{Ptr{Cvoid}, FA, rt, tt′, width, ReturnPrimal}(adjoint_ptr)
    thunk(f, args...)
end

"""
    autodiff_deferred(mode::Mode, f, ::Type{A}, args)

Like [`autodiff_deferred`](@ref) but will try to extend f to an annotation, if needed.
"""
@inline function autodiff_deferred(mode::CMode, f::F, args::Vararg{Annotation, Nargs}) where {F, CMode<:Mode, Nargs}
    autodiff_deferred(EnzymeCore.set_err_if_func_written(mode), Const(f), args...)
end
@inline function autodiff_deferred(mode::CMode, f::F, ::Type{RT}, args::Vararg{Annotation, Nargs}) where {F, RT<:Annotation, CMode<:Mode, Nargs}
    autodiff_deferred(EnzymeCore.set_err_if_func_written(mode), Const(f), RT, args...)
end

"""
    autodiff_deferred(mode, f, args...)

Like [`autodiff_deferred`](@ref) but will try to guess the activity of the return value.
"""

@inline function autodiff_deferred(mode::M, f::FA, args::Vararg{Annotation, Nargs}) where {FA<:Annotation, M<:Mode, Nargs}
    tt    = Tuple{map(T->eltype(Core.Typeof(T)), args)...}
    rt    = if mode isa ReverseMode && VERSION >= v"1.8.0"
        Compiler.primal_return_type(mode, Val(codegen_world_age(eltype(FA), tt)), eltype(FA), tt)
    else
        Core.Compiler.return_type(f.val, tt)
    end

    if rt === Union{}
        error("return type is Union{}, giving up.")
    end
    rt    = guess_activity(rt, mode)
    autodiff_deferred(mode, f, rt, args...)
end

"""
    autodiff_thunk(::ReverseModeSplit, ftype, Activity, argtypes::Vararg{Type{<:Annotation, Nargs})

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
@inline function autodiff_thunk(::ReverseModeSplit{ReturnPrimal,ReturnShadow,Width,ModifiedBetweenT,RABI, ErrIfFuncWritten}, ::Type{FA}, ::Type{A}, args::Vararg{Type{<:Annotation}, Nargs}) where {FA<:Annotation, A<:Annotation, ReturnPrimal,ReturnShadow,Width,ModifiedBetweenT,RABI<:ABI, Nargs, ErrIfFuncWritten}
    width = if Width == 0
        w = same_or_one(1, args...)
        if w == 0
            throw(ErrorException("Cannot differentiate with a batch size of 0"))
        end
        w
    else
        Width
    end

    if ModifiedBetweenT === true
        ModifiedBetween = Val(falses_from_args(Nargs+1))
    else
        ModifiedBetween = Val(ModifiedBetweenT)
    end

    tt    = Tuple{map(eltype, args)...}
    
    if !(A <: Const)
        @assert ReturnShadow
    end
    tt′ = Tuple{args...}
    opt_mi = if RABI <: NonGenABI
        Compiler.fspec(eltype(FA), tt′)
    else
        Val(codegen_world_age(eltype(FA), tt))
    end
    Enzyme.Compiler.thunk(opt_mi, FA, A, tt′, #=Split=# Val(API.DEM_ReverseModeGradient), Val(width), ModifiedBetween, #=ReturnPrimal=#Val(ReturnPrimal), #=ShadowInit=#Val(false), RABI, Val(ErrIfFuncWritten))
end

"""
    autodiff_thunk(::ForwardMode, ftype, Activity, argtypes::Vararg{Type{<:Annotation}, Nargs})

Provide the thunk forward mode function for annotated function type
ftype when called with args of type `argtypes`.

`Activity` is the Activity of the return value, it may be `Const` or `Duplicated`
(or its variants `DuplicatedNoNeed`, `BatchDuplicated`, and`BatchDuplicatedNoNeed`).

The forward function will return the primal (if requested) and the shadow
(or nothing if not a `Duplicated` variant).

Example returning both original return and derivative:

```jldoctest
a = 4.2
b = [2.2, 3.3]; ∂f_∂b = zero(b)
c = 55; d = 9

f(x) = x*x
forward = autodiff_thunk(Forward, Const{typeof(f)}, Duplicated, Duplicated{Float64})
res, ∂f_∂x = forward(Const(f), Duplicated(3.14, 1.0))

# output

(9.8596, 6.28)
```

Example returning just the derivative:

```jldoctest
a = 4.2
b = [2.2, 3.3]; ∂f_∂b = zero(b)
c = 55; d = 9

f(x) = x*x
forward = autodiff_thunk(Forward, Const{typeof(f)}, DuplicatedNoNeed, Duplicated{Float64})
∂f_∂x = forward(Const(f), Duplicated(3.14, 1.0))

# output

(6.28,)
```
"""
@inline function autodiff_thunk(::ForwardMode{RABI, ErrIfFuncWritten}, ::Type{FA}, ::Type{A}, args::Vararg{Type{<:Annotation}, Nargs}) where {FA<:Annotation, A<:Annotation, RABI<:ABI, Nargs, ErrIfFuncWritten}
    width = same_or_one(1, A, args...)
    if width == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end
    if A <: Active
        throw(ErrorException("Active Returns not allowed in forward mode"))
    end
    ReturnPrimal = Val(A <: Duplicated || A <: BatchDuplicated)
    ModifiedBetween = Val(falses_from_args(Nargs+1))

    tt    = Tuple{map(eltype, args)...}
    
    tt′ = Tuple{args...}
    opt_mi = if RABI <: NonGenABI
        Compiler.fspec(eltype(FA), tt′)
    else
        Val(codegen_world_age(eltype(FA), tt))
    end
    Enzyme.Compiler.thunk(opt_mi, FA, A, tt′, #=Mode=# Val(API.DEM_ForwardMode), Val(width), ModifiedBetween, ReturnPrimal, #=ShadowInit=#Val(false), RABI, Val(ErrIfFuncWritten))
end

@inline function tape_type(::ReverseModeSplit{ReturnPrimal,ReturnShadow,Width,ModifiedBetweenT, RABI, ErrIfFuncWritten}, ::Type{FA}, ::Type{A}, args::Vararg{Type{<:Annotation}, Nargs}) where {FA<:Annotation, A<:Annotation, ReturnPrimal,ReturnShadow,Width,ModifiedBetweenT, RABI<:ABI, Nargs, ErrIfFuncWritten}
    width = if Width == 0
        w = same_or_one(1, args...)
        if w == 0
            throw(ErrorException("Cannot differentiate with a batch size of 0"))
        end
        w
    else
        Width
    end

    if ModifiedBetweenT === true
        ModifiedBetween = Val(falses_from_args(Nargs+1))
    else
        ModifiedBetween = Val(ModifiedBetweenT)
    end

    @assert ReturnShadow
    TT = Tuple{args...}
   
    primal_tt = Tuple{map(eltype, args)...}
    opt_mi = if RABI <: NonGenABI
        Compiler.fspec(eltype(FA), TT)
    else
        Val(codegen_world_age(eltype(FA), primal_tt))
    end
    nondef = Enzyme.Compiler.thunk(opt_mi, FA, A, TT, #=Split=# Val(API.DEM_ReverseModeGradient), Val(width), ModifiedBetween, #=ReturnPrimal=#Val(ReturnPrimal), #=ShadowInit=#Val(false), RABI, Val(ErrIfFuncWritten))
    if nondef[1] isa Enzyme.Compiler.PrimalErrorThunk
        return Nothing
    else
        TapeType = EnzymeRules.tape_type(nondef[1])
        return TapeType
    end
end

const tape_cache = Dict{UInt, Type}()

const tape_cache_lock = ReentrantLock()

import .Compiler: fspec, remove_innerty, UnknownTapeType

@inline function tape_type(
    parent_job::Union{GPUCompiler.CompilerJob,Nothing}, ::ReverseModeSplit{ReturnPrimal,ReturnShadow,Width,ModifiedBetweenT, RABI},
    ::Type{FA}, ::Type{A}, args::Vararg{Type{<:Annotation}, Nargs}
) where {FA<:Annotation, A<:Annotation, ReturnPrimal,ReturnShadow,Width,ModifiedBetweenT, RABI<:ABI, Nargs}
    width = if Width == 0
        w = same_or_one(1, args...)
        if w == 0
            throw(ErrorException("Cannot differentiate with a batch size of 0"))
        end
        w
    else
        Width
    end

    if ModifiedBetweenT === true
        ModifiedBetween = falses_from_args(Val(1), args...)
    else
        ModifiedBetween = ModifiedBetweenT
    end

    @assert ReturnShadow
    TT = Tuple{args...}

    primal_tt = Tuple{map(eltype, args)...}

    world = codegen_world_age(eltype(FA), primal_tt)

    mi = Compiler.fspec(eltype(FA), TT, world)

    target = Compiler.EnzymeTarget()
    params = Compiler.EnzymeCompilerParams(
        Tuple{FA, TT.parameters...}, API.DEM_ReverseModeGradient, width,
        Compiler.remove_innerty(A), true, #=abiwrap=#false, ModifiedBetweenT,
        ReturnPrimal, #=ShadowInit=#false, Compiler.UnknownTapeType, RABI, #=errifwritte=#false
    )
    job    = Compiler.CompilerJob(mi, Compiler.CompilerConfig(target, params; kernel=false))


    key = hash(parent_job, hash(job))

    # NOTE: no use of lock(::Function)/@lock/get! to keep stack traces clean
    lock(tape_cache_lock)

    try
        obj = get(tape_cache, key, nothing)
        if obj === nothing

            Compiler.JuliaContext() do ctx
                _, meta = Compiler.codegen(:llvm, job; optimize=false, parent_job)
                obj = meta.TapeType
                tape_cache[key] = obj
            end
        end
        obj
    finally
        unlock(tape_cache_lock)
    end
end

"""
    autodiff_deferred_thunk(::ReverseModeSplit, ftype, Activity, argtypes::Vararg{Type{<:Annotation}, Nargs})

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

TapeType = tape_type(ReverseSplitWithPrimal, Const{typeof(f)}, Active, Duplicated{typeof(A)}, Active{typeof(v)})
forward, reverse = autodiff_deferred_thunk(ReverseSplitWithPrimal, TapeType, Const{typeof(f)}, Active{Float64}, Duplicated{typeof(A)}, Active{typeof(v)})

tape, result, shadow_result  = forward(Const(f), Duplicated(A, ∂A), Active(v))
_, ∂v = reverse(Const(f), Duplicated(A, ∂A), Active(v), 1.0, tape)[1]

result, ∂v, ∂A 

# output

(7.26, 2.2, [3.3])
```
"""
@inline function autodiff_deferred_thunk(mode::ReverseModeSplit{ReturnPrimal,ReturnShadow,Width,ModifiedBetweenT, RABI, ErrIfFuncWritten}, tt::Type{TapeType}, fa::Type{FA}, a2::Type{A2}, args::Vararg{Type{<:Annotation}, Nargs}) where {FA<:Annotation, A2<:Annotation, TapeType, ReturnPrimal,ReturnShadow,Width,ModifiedBetweenT, RABI<:ABI, Nargs, ErrIfFuncWritten}
    @assert RABI == FFIABI
    width = if Width == 0
        w = same_or_one(1, args...)
        if w == 0
            throw(ErrorException("Cannot differentiate with a batch size of 0"))
        end
        w
    else
        Width
    end

    if ModifiedBetweenT === true
        ModifiedBetween = Val(falses_from_args(Nargs+1))
    else
        ModifiedBetween = Val(ModifiedBetweenT)
    end

    @assert ReturnShadow
    TT = Tuple{args...}

    primal_tt = Tuple{map(eltype, args)...}
    world = codegen_world_age(eltype(FA), primal_tt)

    primal_ptr = Compiler.deferred_codegen(Val(world), FA, Val(TT), Val(Compiler.remove_innerty(A2)), Val(API.DEM_ReverseModePrimal), Val(width), ModifiedBetween, Val(ReturnPrimal), #=ShadowInit=#Val(false), TapeType, Val(ErrIfFuncWritten))
    adjoint_ptr = Compiler.deferred_codegen(Val(world), FA, Val(TT), Val(Compiler.remove_innerty(A2)), Val(API.DEM_ReverseModeGradient), Val(width), ModifiedBetween, Val(ReturnPrimal), #=ShadowInit=#Val(false), TapeType, Val(ErrIfFuncWritten))

    RT = if A2 <: Duplicated && width != 1
        if A2 isa UnionAll
            BatchDuplicated{T, width} where T
        else
            BatchDuplicated{eltype(A2), width}
        end
    elseif A2 <: DuplicatedNoNeed && width != 1
        if A2 isa UnionAll
            BatchDuplicatedNoNeed{T, width} where T
        else
            BatchDuplicatedNoNeed{eltype(A2), width}
        end
    elseif A2 <: MixedDuplicated && width != 1
        if A2 isa UnionAll
            BatchMixedDuplicated{T, width} where T
        else
            BatchMixedDuplicated{eltype(A2), width}
        end
    else
        A2
    end
    
    rt = if RT isa UnionAll
        @static if VERSION < v"1.8-"
            throw(MethodError(autodiff_deferred_thunk, (mode, tt, fa, a2, args...)))
        else
            RT{Core.Compiler.return_type(Tuple{eltype(FA), map(eltype, args)...})}
        end
    else
        @assert RT isa DataType
        RT
    end

    aug_thunk = Compiler.AugmentedForwardThunk{Ptr{Cvoid}, FA, rt, TT, width, ReturnPrimal, TapeType}(primal_ptr)
    adj_thunk = Compiler.AdjointThunk{Ptr{Cvoid}, FA, rt, TT, width, TapeType}(adjoint_ptr)
    aug_thunk, adj_thunk
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
@static if sizeof(Int) == sizeof(Int64)
    Base.llvmcall(("declare void @__enzyme_float(i8* nocapture) nounwind define void @c(i64 %q) nounwind alwaysinline { %p = inttoptr i64 %q to i8* call void @__enzyme_float(i8* %p) ret void }", "c"), Cvoid, Tuple{Ptr{Float32}}, data)
else
    Base.llvmcall(("declare void @__enzyme_float(i8* nocapture) nounwind define void @c(i32 %q) nounwind alwaysinline { %p = inttoptr i32 %q to i8* call void @__enzyme_float(i8* %p) ret void }", "c"), Cvoid, Tuple{Ptr{Float32}}, data)
end
    nothing
end

@inline function markType(data::Ptr{Float64})
@static if sizeof(Int) == sizeof(Int64)
    Base.llvmcall(("declare void @__enzyme_double(i8* nocapture) nounwind define void @c(i64 %q) nounwind alwaysinline { %p = inttoptr i64 %q to i8* call void @__enzyme_double(i8* %p) ret void }", "c"), Cvoid, Tuple{Ptr{Float64}}, data)
else
    Base.llvmcall(("declare void @__enzyme_double(i8* nocapture) nounwind define void @c(i32 %q) nounwind alwaysinline { %p = inttoptr i32 %q to i8* call void @__enzyme_double(i8* %p) ret void }", "c"), Cvoid, Tuple{Ptr{Float64}}, data)
end
    nothing
end

