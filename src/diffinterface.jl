
@inline function onehot(x)
    N = length(x)
    ntuple(Val(N)) do i
        Base.@_inline_meta
        res = similar(x)
        for idx in 1:N
            @inbounds res[idx] = (i == idx) ? 1.0 : 0.0
        end
        return res
    end
end
@inline function onehot(x, start, endl)
    ntuple(Val(endl-start+1)) do i
        Base.@_inline_meta
        res = similar(x)
        for idx in 1:length(x)
            @inbounds res[idx] = (i + start - 1== idx) ? 1.0 : 0.0
        end
        return res
    end
end

@inline function onehot(::Type{NTuple{N, T}}) where {T, N}
    ntuple(Val(N)) do i
        Base.@_inline_meta
        ntuple(Val(N)) do idx
            Base.@_inline_meta
            return (i == idx) ? 1.0 : 0.0
        end
    end
end
@inline onehot(x::NTuple{N, T}) where {T, N} = onehot(NTuple{N, T})
@inline function onehot(x::NTuple{N, T}, start, endl) where {T, N}
    ntuple(Val(endl-start+1)) do i
        Base.@_inline_meta
        ntuple(Val(N)) do idx
            Base.@_inline_meta
            return (i + start - 1 == idx) ? 1.0 : 0.0
        end
    end
end

@inline onehot(x::AbstractFloat) = (one(x),)

@inline function chunkedonehot(x, ::Val{chunk}) where chunk
    sz = length(x)
    num = ((sz + chunk - 1) ÷ chunk)
    ntuple(Val(num)) do i
        Base.@_inline_meta
        onehot(x, (i-1)*chunk+1, i == num ? sz : (i*chunk) )
    end
end

@inline chunkedonehot(x::AbstractFloat, ::Val{chunk}) where chunk = ((one(x),),)

@inline tupleconcat(x) = x
@inline tupleconcat(x, y) = (x..., y...)
@inline tupleconcat(x, y, z...) = (x..., tupleconcat(y, z...)...)

@inline function derivative(mode::ForwardMode, f, x; shadow=onehot(x))
    values(only(autodiff(mode, f, BatchDuplicatedNoNeed, BatchDuplicated(x, shadow))))
end
@inline function derivative_deferred(mode::ForwardMode, f, x; shadow=onehot(x))
    values(only(autodiff_deferred(mode, f, BatchDuplicatedNoNeed, BatchDuplicated(x, shadow))))
end

@inline _chunkcheck(::Val{0}) = throw(ArgumentError("Cannot differentiate with a batch size of 0"))
@inline _chunkcheck(::Val) = nothing

@inline function derivative(mode::ForwardMode, f::F, x::X, ::Val{chunk};
                            shadow=chunkedonehot(x, Val(chunk))) where {F,X,chunk}
    _chunkcheck(Val(chunk))
    tmp = ntuple(length(shadow)) do i
        values(autodiff(mode, f, BatchDuplicatedNoNeed, BatchDuplicated(x, shadow[i]))[1])
    end
    tupleconcat(tmp...)
end

@inline gradient_output_size(df1, x) = (size(df1)..., size(x)...)

@inline gradient_output_forward(df, df1, x) = df
@inline gradient_output_forward(df, df1::Number, x::Number) = df1
@inline gradient_output_forward(df, df1, x::AbstractArray) = copyto!(similar(x, eltype(df)), df)
@inline gradient_output_forward(df, df1::AbstractArray, x::Number) = reduce(hcat, df)
@inline function gradient_output_forward(df, df1::AbstractArray, x::AbstractArray)
    reshape(reduce(hcat, df), gradient_output_size(df1, x))
end

# this should handle general mutable array types
@inline gradient_output_forward(df, x::AbstractArray) = copyto!(similar(x), df)

@inline function gradient(mode::ForwardMode, f::F, x::X; shadow=onehot(x)) where {F,X}
    df = derivative(mode, f, x; shadow)
    gradient_output_forward(df, df[1], x)
end

@inline function gradient(mode::ForwardMode, f::F, x::X, ::Val{chunk};
                          shadow=chunkedonehot(x, Val(chunk))) where {F,X,chunk}
    df = derivative(mode, f, x, Val(chunk); shadow)
    gradient_output_forward(df, df[1], x)
end

@inline function gradient(mode::ReverseMode, f::F, x::X) where {F,X}
    if Compiler.active_reg_inner(X, #=seen=#(), #=world=#nothing, #=justActive=#Val(true)) == Compiler.ActiveState
        dx = Ref(make_zero(x))
        autodiff(mode, f, Active, MixedDuplicated(x, dx))
        return only(dx)
    else
        dx = make_zero(x)
        autodiff(mode, f, Active, Duplicated(x, dx))
        return dx
    end
end

"""
    gradient!(::ReverseMode, dx, f, x)

Compute the gradient of an array-input function `f` using reverse mode,
storing the derivative result in an existing array `dx`.
Both `x` and `dx` must be `Array`s of the same type.

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
@inline function gradient!(::ReverseMode, dx::X, f::F, x::X) where {X<:Array, F}
    make_zero!(dx)
    autodiff(Reverse, f, Active, Duplicated(x, dx))
    dx
end

"""
    gradient_deferred!(::ReverseMode, f, x)

Like [`gradient!`](@ref), except it using deferred mode.
"""
@inline function gradient_deferred!(::ReverseMode, dx::X, f::F, x::X) where {X<:Array, F}
    make_zero!(dx)
    autodiff_deferred(Reverse, f, Active, Duplicated(x, dx))
    dx
end

@inline function jacobian(mode::ForwardMode, f, x; shadow=onehot(x))
    gradient(mode, f, x; shadow)
end

@inline function jacobian(mode::ForwardMode, f::F, x::X, ::Val{chunk};
                          shadow=chunkedonehot(x, Val(chunk))) where {F,X,chunk}
    gradient(mode, f, x, Val(chunk); shadow)
end

@inline gradient_output_reverse(df, df1, x) = df

@inline gradient_output_reverse(df, df1::AbstractArray, x::Number) = reduce(vcat, df)

@inline function gradient_output_reverse(df, df1::AbstractArray{T1,n1}, x::AbstractArray{T2,n2}) where {T1,T2,n1,n2}
    # gives us \partial_{ij}y_{kl}
    tmp = reshape(reduce(hcat, df), gradient_output_size(x, df1))
    dims = ntuple(Val(n1+n2)) do j
        j ≤ n1 ? n2+j : j-n1
    end
    permutedims(tmp, dims)  # we must permute to get \partial_{kl}y_{ij}
end

"""
    jacobian(::ReverseMode, f, x, ::Val{num_outs}, ::Val{chunk}=Val(1))
    jacobian(::ReverseMode, f, x)

Compute the jacobian of an array-output function `f` using (potentially vector)
reverse mode. The `chunk` argument denotes the chunk size to use and `num_outs`
denotes the number of outputs `f` will return in an array.

Example:

```jldoctest
f(x) = [ x[1] * x[2], x[2] + x[3] ]

grad = jacobian(Reverse, f, [2.0, 3.0, 4.0], Val(2))

# output

2×3 transpose(::Matrix{Float64}) with eltype Float64:
 3.0  2.0  0.0
 0.0  1.0  1.0
```

For functions which return an AbstractArray, this function will return an array
whose shape is `(size(output)..., size(input)...)`

For functions who return other types, this function will retun an array or tuple
of shape `size(output)` of values of the input type. 
```
"""
@inline function jacobian(::ReverseMode{false,RABI,ErrIfFuncWritten}, f::F, x::X,
                          n_outs::Val{n_out_val},
                          ::Val{chunk}) where {F, X, chunk, n_out_val, RABI<:ABI, ErrIfFuncWritten}
    _chunkcheck(Val(chunk))
    num = ((n_out_val + chunk - 1) ÷ chunk)
    XT = Core.Typeof(x) 
    MD = Compiler.active_reg_inner(XT, #=seen=#(), #=world=#nothing, #=justActive=#Val(true)) == Compiler.ActiveState
    tt′   = MD ? Tuple{BatchMixedDuplicated{XT, chunk}} : Tuple{BatchDuplicated{XT, chunk}}
    tt    = Tuple{XT}
    rt = Core.Compiler.return_type(f, tt)
    ModifiedBetween = Val((false, false))
    FA = Const{Core.Typeof(f)}
    opt_mi = if RABI <: NonGenABI
        Compiler.fspec(eltype(FA), tt′)
    else
        Val(codegen_world_age(Core.Typeof(f), tt))
    end
    primal, adjoint = Enzyme.Compiler.thunk(opt_mi, FA, BatchDuplicatedNoNeed{rt}, tt′,
                                            #=Split=# Val(API.DEM_ReverseModeGradient),
                                            #=width=#Val(chunk), ModifiedBetween,
                                            #=ReturnPrimal=#Val(false), #=ShadowInit=#Val(false),
                                            RABI, Val(ErrIfFuncWritten))
    if num * chunk == n_out_val
        last_size = chunk
        primal2, adjoint2 = primal, adjoint
    else
        last_size = n_out_val - (num-1)*chunk
        tt′ = Tuple{BatchDuplicated{Core.Typeof(x), last_size}}
        primal2, adjoint2 = Enzyme.Compiler.thunk(opt_mi, FA, BatchDuplicatedNoNeed{rt}, tt′,
                                                  #=Split=# Val(API.DEM_ReverseModeGradient),
                                                  #=width=#Val(last_size), ModifiedBetween,
                                                  #=ReturnPrimal=#Val(false), #=ShadowInit=#Val(false),
                                                  RABI, Val(ErrIfFuncWritten))
    end
    #TODO: this is broken for static arrays
    tmp = ntuple(num) do i
        Base.@_inline_meta
        dx = ntuple(Val(i == num ? last_size : chunk)) do idx
            Base.@_inline_meta
            z = make_zero(x)
            MD ? Ref(z) : z
        end
        res = (i == num ? primal2 : primal)(Const(f), MD ? BatchMixedDuplicated(x, dx) : BatchDuplicated(x, dx))
        tape = res[1]
        j = 0
        for shadow in res[3]
            j += 1
            @inbounds shadow[(i-1)*chunk+j] += Compiler.default_adjoint(eltype(typeof(shadow)))
        end
        (i == num ? adjoint2 : adjoint)(Const(f), MD ? BatchMixedDuplicated(x, dx) : BatchDuplicated(x, dx), tape)
        return MD ? (ntuple(Val(i == num ? last_size : chunk)) do idx
            Base.@_inline_meta
            dx[idx][]
        end) : dx, (i == 1 ? res[3][1] : nothing)
    end
    df1 = tmp[1][2]
    df = tupleconcat(map(first, tmp)...)
    gradient_output_reverse(df, df1, x)
end

@inline function jacobian(mode::ReverseMode{false,RABI,ErrIfFuncWritten}, f::F, x::X,
                          n_outs::Val{n_out_val},
                          chunks::Val{1}) where {RABI<:ABI,ErrIfFuncWritten,F,X,n_out_val}
    XT = Core.Typeof(x) 
    MD = Compiler.active_reg_inner(XT, #=seen=#(), #=world=#nothing, #=justActive=#Val(true)) == Compiler.ActiveState
    tt′   = MD ? Tuple{MixedDuplicated{XT}} : Tuple{Duplicated{XT}}
    tt    = Tuple{XT}
    rt = Core.Compiler.return_type(f, tt)
    ModifiedBetween = Val((false, false))
    FA = Const{Core.Typeof(f)}
    opt_mi = if RABI <: NonGenABI
        Compiler.fspec(eltype(FA), tt′)
    else
        Val(codegen_world_age(Core.Typeof(f), tt))
    end
    primal, adjoint = Enzyme.Compiler.thunk(opt_mi, FA, DuplicatedNoNeed{rt}, tt′,
                                            #=Split=# Val(API.DEM_ReverseModeGradient),
                                            #=width=#Val(1), ModifiedBetween, #=ReturnPrimal=#Val(false),
                                            #=ShadowInit=#Val(false), RABI, Val(ErrIfFuncWritten))
    #TODO: this is broken for static arrays
    tmp = ntuple(n_outs) do i
        Base.@_inline_meta
        z = make_zero(x)
        dx = MD ? Ref(z) : z
        res = primal(Const(f), MD ? MixedDuplicated(x, dx) : Duplicated(x, dx))
        tape = res[1]
        @inbounds res[3][i] += Compiler.default_adjoint(eltype(typeof(res[3])))
        adjoint(Const(f), MD ? MixedDuplicated(x, dx) : Duplicated(x, dx), tape)
        (MD ? dx[] : dx, (i == 1 ? res[3] : nothing))
    end
    df1 = tmp[1][2]
    df = map(first, tmp)
    gradient_output_reverse(df, df1, x)
end

# resolves ambiguity
@inline function jacobian(mode::ReverseMode{false}, f::F, x::X, n_outs::Val) where {F,X}
    jacobian(mode, f, x, n_outs, Val(1))
end

@inline function _jacobian(::ReverseMode{ReturnPrimal,RABI,ErrIfFuncWritten,T},
                           f::F, x::X) where {ReturnPrimal,F,X,T,RABI<:ABI, ErrIfFuncWritten}
    y = f(x)
    df = if y isa AbstractArray
        jacobian(ReverseMode{false,RABI,ErrIfFuncWritten,T}(), f, x, Val(length(y)))
    elseif y isa AbstractFloat
        gradient(ReverseMode{false,RABI,ErrIfFuncWritten,T}(), f, x)
    else
        throw(AssertionError("Unsupported return type of function for reverse-mode jacobian, $(Core.Typeof(res))"))
    end
    (y, df)
end

@inline jacobian(mode::ReverseMode{false}, f::F, x::X) where {F,X} = _jacobian(mode, f, x)[2]
@inline jacobian(mode::ReverseMode{true}, f::F, x::X) where {F,X} = _jacobian(mode, f, x)

"""
    hvp(f::F, x::X, v::X) where {F, X}

Compute the Hessian-vector product of an array-input scalar-output function `f`, as evaluated at `x` times the vector `v`.

In other words, compute hessian(f)(x) * v

See [`hvp!`](@ref) for a version which stores the result in an existing buffer and also [`hvp_and_gradient!`](@ref) for a function to compute both the hvp and the gradient in a single call.

Example:

```jldoctest hvp; filter = r"([0-9]+\\.[0-9]{8})[0-9]+" => s"\\1***"
f(x) = sin(x[1] * x[2])

hvp(f, [2.0, 3.0], [5.0, 2.7])

# output
2-element Vector{Float64}:
 19.6926882637302
 16.201003759768003
```
"""
@inline function hvp(f::F, x::X, v::X) where {F, X}
    res = make_zero(x)
    hvp!(res, f, x, v)
    return res
end

"""
    hvp!(res::X, f::F, x::X, v::X) where {F, X}

Compute an in-place Hessian-vector product of an array-input scalar-output function `f`, as evaluated at `x` times the vector `v`.
The result will be stored into `res`. The function still allocates and zero's a buffer to store the intermediate gradient, which is
not returned to the user.

In other words, compute res .= hessian(f)(x) * v

See [`hvp_and_gradient!`](@ref) for a function to compute both the hvp and the gradient in a single call.

Example:

```jldoctest hvpip; filter = r"([0-9]+\\.[0-9]{8})[0-9]+" => s"\\1***"
f(x) = sin(x[1] * x[2])

res = Vector{Float64}(undef, 2)
hvp!(res, f, [2.0, 3.0], [5.0, 2.7])

res
# output
2-element Vector{Float64}:
 19.6926882637302
 16.201003759768003
```
"""
@inline function hvp!(res::X, f::F, x::X, v::X) where {F, X}
    grad = make_zero(x)
    Enzyme.autodiff(Forward, gradient_deferred!, Const(Reverse), DuplicatedNoNeed(grad, res), Const(f), Duplicated(x, v))
    return nothing
end

"""
    hvp_and_gradient!(res::X, grad::X, f::F, x::X, v::X) where {F, X}

Compute an in-place Hessian-vector product of an array-input scalar-output function `f`, as evaluated at `x` times the vector `v` as well as
the gradient, storing the gradient into `grad`. Both the hessian vector product and the gradient can be computed together more efficiently
than computing them separately.

The result will be stored into `res`. The gradient will be stored into `grad`.

In other words, compute res .= hessian(f)(x) * v  and grad .= gradient(Reverse, f)(x)

Example:

```jldoctest hvp_and_gradient; filter = r"([0-9]+\\.[0-9]{8})[0-9]+" => s"\\1***"
f(x) = sin(x[1] * x[2])

res = Vector{Float64}(undef, 2)
grad = Vector{Float64}(undef, 2)
hvp_and_gradient!(res, grad, f, [2.0, 3.0], [5.0, 2.7])

res
grad
# output
2-element Vector{Float64}:
 2.880510859951098
 1.920340573300732
```
"""
@inline function hvp_and_gradient!(res::X, grad::X, f::F, x::X, v::X) where {F, X}
    Enzyme.autodiff(Forward, gradient_deferred!, Const(Reverse),  Duplicated(grad, res), Const(f), Duplicated(x, v))
    return nothing
end

