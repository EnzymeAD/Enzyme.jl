
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

@inline function derivative(mode::ForwardMode, f::F, x::X, ::Val{chunk};
                            shadow=chunkedonehot(x, Val(chunk))) where {F,X,chunk}
    if chunk == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end
    tmp = ntuple(length(shadow)) do i
        values(autodiff(mode, f, BatchDuplicatedNoNeed, BatchDuplicated(x, shadow[i]))[1])
    end
    tupleconcat(tmp...)
end

@inline function _split_tuple_of_tuples(tpl::Tuple)
    t1 = ntuple(i -> tpl[i][1], length(tpl))
    t2 = ntuple(i -> values(tpl[i][2]), length(tpl))        
    (t1, tupleconcat(t2...))
end

# note that we don't provide valderivative for batched mode because it causes it to re-evaluate

@inline gradient_output(df, x) = df

# this should handle general mutable array types
@inline gradient_output(df, x::AbstractArray) = copyto!(similar(x), df)

@inline function gradient(mode::ForwardMode, f::F, x::X; shadow=onehot(x)) where {F,X}
    df = derivative(mode, f, x; shadow)
    gradient_output(df, x)
end

@inline function gradient(mode::ForwardMode, f::F, x::X, ::Val{chunk};
                          shadow=chunkedonehot(x, Val(chunk))) where {F,X,chunk}
    df = derivative(mode, f, x, Val(chunk); shadow)
    gradient_output(df, x)
end

#WARN: presumably this is done in this particular way for efficiency, but it's not super clear that
# it'll always give consistent results...
# Before this PR, this would return type X even though forward mode gradients always returned tuple
# I think it's consistent now?
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
    gradient_deferred(::ReverseMode, f, x)

Like [`gradient`](@ref), except it using deferred mode.
"""
@inline function gradient_deferred(rm::ReverseMode, f::F, x::X) where {F, X}
    if Compiler.active_reg_inner(X, #=seen=#(), #=world=#nothing, #=justActive=#Val(true)) == Compiler.ActiveState
        dx = Ref(make_zero(x))
        autodiff_deferred(rm, f, Active, MixedDuplicated(x, dx))
        return only(dx)
    else
        dx = make_zero(x)
        autodiff_deferred(rm, f, Active, Duplicated(x, dx))
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

@inline jacobian_output(df, df1, x) = df

#TODO: are you really sure this always works?
@inline jacobian_output(df, df1, x::Number) = df1

@inline jacobian_output(df, df1::Number, x) = gradient_output(df, x)

# resolves method ambiguity
@inline jacobian_output(df, df1::Number, x::Number) = df1

# static array packages can overload this
@inline jacsize(dfshape, xshape) = (xshape..., dfshape...)

@inline function jacobian_output(df, df1::AbstractArray, x::AbstractArray)
    reshape(reduce(hcat, df), jacsize(size(df1), size(x)))
end


@inline function jacobian(mode::ForwardMode, f, x; shadow=onehot(x))
    df = derivative(mode, f, x; shadow)
    jacobian_output(df, df[1], x)
end

@inline function jacobian(mode::ForwardMode, f::F, x::X, ::Val{chunk};
                          shadow=chunkedonehot(x, Val(chunk))) where {F,X,chunk}
    df = derivative(mode, f, x, Val(chunk); shadow)
    jacobian_output(df, df[1], x)
end

#TODO: it's very confusing from an API point of view that Reverse has the ReturnPrimal type
#paramter, but Forward has to specify whether to do this a completely different way


#WARN: obviously all the reverse mode stuff below is completely FUBAR right now

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
@inline function _OLD_jacobian(::ReverseMode{#=ReturnPrimal=#false,RABI, ErrIfFuncWritten}, f::F, x::X,
                          n_outs::Val{n_out_val},
                          ::Val{chunk}) where {F, X, chunk, n_out_val, RABI<:ABI, ErrIfFuncWritten}
    num = ((n_out_val + chunk - 1) ÷ chunk)
    
    if chunk == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end

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

    #WARN: need to re-do this stuff

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
        end) : dx, (i == 1 ? size(res[3][1]) : nothing)
    end
    rows = tupleconcat(map(first, tmp)...)
    outshape = tmp[1][2]
    if x isa AbstractArray
        inshape = size(x)

        st = @static if VERSION >= v"1.9"
            Base.stack(rows)
        else
            reshape(cat(rows..., dims=length(inshape)), (inshape..., outshape...))
        end

        st2 = if length(outshape) == 1 || VERSION < v"1.9"
            st
        else
            reshape(st, (inshape..., outshape...))
        end

        st3 = if length(outshape) == 1 && length(inshape) == 1
            transpose(st2)
        else
            transp = ( ((length(inshape)+1):(length(inshape)+length(outshape)))... , (1:length(inshape))...  )
            PermutedDimsArray(st2, transp)
        end

        st3
    else
        reshape(collect(rows), outshape)
    end
end

#WARN: these are most definitely not done
@inline function reverse_jacobian_out(shape, df, x::AbstractArray)
    reshape(cat(df..., dims=length(size(x))), jacsize(shape, size(x)))
end
@inline function reverse_jacobian_out(shape, df, x)
    reshape(collect(df), shape)    
end

function jacobian(mode::ReverseMode{false,RABI,ErrIfFuncWritten}, f::F, x::X, n_outs::Val,
                  chunks::Val{1}=Val(1)) where {RABI,ErrIfFuncWritten,F,X}
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
    tmp = ntuple(n_outs) do i
        Base.@_inline_meta
        z = make_zero(x)
        dx = MD ? Ref(z) : z
        res = primal(Const(f), MD ? MixedDuplicated(x, dx) : Duplicated(x, dx))
        tape = res[1]
        @inbounds res[3][i] += Compiler.default_adjoint(eltype(typeof(res[3])))
        adjoint(Const(f), MD ? MixedDuplicated(x, dx) : Duplicated(x, dx), tape)
        return MD ? dx[] : dx, (i == 1 ? size(res[3]) : nothing)
    end
    shape = tmp[1][2]
    df = map(first, tmp)
    reverse_jacobian_out(shape, df, x)
end

@inline function _OLD_jacobian(::ReverseMode{#=ReturnPrimal=#false,RABI, ErrIfFuncWritten}, f::F, x::X,
                          n_outs::Val{n_out_val}, ::Val{1}=Val(1)) where {F, X, n_out_val,RABI<:ABI, ErrIfFuncWritten}
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
    tmp = ntuple(n_outs) do i
        Base.@_inline_meta
        z = make_zero(x)
        dx = MD ? Ref(z) : z
        res = primal(Const(f), MD ? MixedDuplicated(x, dx) : Duplicated(x, dx))
        tape = res[1]
        @inbounds res[3][i] += Compiler.default_adjoint(eltype(typeof(res[3])))
        adjoint(Const(f), MD ? MixedDuplicated(x, dx) : Duplicated(x, dx), tape)
        return MD ? dx[] : dx, (i == 1 ? size(res[3]) : nothing)
    end
    df = map(first, tmp)
    jacobian_output_rev(df, df[1], x)
end

@inline function _jacobian(::ReverseMode{ReturnPrimal,RABI,ErrIfFuncWritten,T},
                           f::F, x::X) where {ReturnPrimal,F,X,T,RABI<:ABI, ErrIfFuncWritten}
    y = f(x)
    df = if y isa AbstractArray
        jacobian(ReverseMode{false,RABI,ErrIfFuncWritten,T}(), f, x, Val(length(y)))
    elseif res isa AbstractFloat
        gradient(ReverseMode{false,RABI,ErrIfFuncWritten,T}(), f, x)
    else
        throw(AssertionError("Unsupported return type of function for reverse-mode jacobian, $(Core.Typeof(res))"))
    end
    (y, df)
end

@inline jacobian(mode::ReverseMode{false}, f::F, x::X) where {F,X} = _jacobian(mode, f, x)[2]
@inline jacobian(mode::ReverseMode{true}, f::F, x::X) where {F,X} = _jacobian(mode, f, x)
