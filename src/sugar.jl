# Syntactic sugar over autodiff calls (e.g. Enzyme.gradient and Enzyme.jacobian)


function zerosetfn(x, i::Int)
    res = zero(x)
    @inbounds res[i] = 1
    return res
end

@generated function onehot_internal(fn::F, x::T, startv::Int, lengthv::Int) where {F, T<:Array}
    ir = GPUCompiler.JuliaContext() do ctx
        Base.@_inline_meta

        target = Compiler.DefaultCompilerTarget()
        params = Compiler.PrimalCompilerParams(API.DEM_ForwardMode)
        mi = my_methodinstance(nothing, fn, Tuple{T, Int})
        job = GPUCompiler.CompilerJob(mi, GPUCompiler.CompilerConfig(target, params; kernel = false))

        GPUCompiler.prepare_job!(job)
        mod, meta = GPUCompiler.emit_llvm(job; libraries=true, toplevel=true, optimize=false, cleanup=false, only_entry=false, validate=false)
        
        copysetfn = meta.entry
        blk = first(LLVM.blocks(copysetfn))
        iter = LLVM.API.LLVMGetFirstInstruction(blk)
        while iter != C_NULL
            inst = LLVM.Instruction(iter)
            iter = LLVM.API.LLVMGetNextInstruction(iter)
            if isa(inst, LLVM.FenceInst)
                Compiler.eraseInst(blk, inst)
            end
            if isa(inst, LLVM.CallInst)
                fn = LLVM.called_operand(inst)
                if isa(fn, LLVM.Function)
                    if LLVM.name(fn) == "julia.safepoint"
                        Compiler.eraseInst(blk, inst)
                    end
                end     
            end
        end
        hasNoRet = Compiler.has_fn_attr(copysetfn, LLVM.EnumAttribute("noreturn"))
        @assert !hasNoRet
        if !hasNoRet
            push!(LLVM.function_attributes(copysetfn), LLVM.EnumAttribute("alwaysinline", 0))
        end
        ity = convert(LLVM.LLVMType, Int)
        jlvaluet = convert(LLVM.LLVMType, T; allow_boxed=true)

        FT = LLVM.FunctionType(jlvaluet,  LLVM.LLVMType[jlvaluet, ity, ity])
        llvm_f = LLVM.Function(mod, "f", FT)
        push!(LLVM.function_attributes(llvm_f), LLVM.EnumAttribute("alwaysinline", 0))

        # Check if Julia version has https://github.com/JuliaLang/julia/pull/46914
        # and also https://github.com/JuliaLang/julia/pull/47076
        # and also https://github.com/JuliaLang/julia/pull/48620
        needs_dynamic_size_workaround = !(VERSION >= v"1.10.5")

        builder = LLVM.IRBuilder()
        entry = LLVM.BasicBlock(llvm_f, "entry")
        LLVM.position!(builder, entry)
        inp, lstart, len = collect(LLVM.Value, LLVM.parameters(llvm_f))

        boxed_count = if sizeof(Int) == sizeof(Int64)
            Compiler.emit_box_int64!(builder, len)
        else
            Compiler.emit_box_int32!(builder, len)
        end

        tag = Compiler.emit_apply_type!(builder, NTuple, LLVM.Value[boxed_count, unsafe_to_llvm(builder, T)])

        fullsize = LLVM.nuwmul!(builder, len, LLVM.ConstantInt(sizeof(Int)))
        obj = Compiler.emit_allocobj!(builder, tag, fullsize, needs_dynamic_size_workaround)

        T_int8 = LLVM.Int8Type()
        LLVM.memset!(builder, obj,  LLVM.ConstantInt(T_int8, 0), fullsize, 0)

        alloc = LLVM.pointercast!(builder, obj, LLVM.PointerType(jlvaluet, Tracked))
        alloc = LLVM.pointercast!(builder, alloc, LLVM.PointerType(jlvaluet, 11))

        loop = LLVM.BasicBlock(llvm_f, "loop")
        exit = LLVM.BasicBlock(llvm_f, "exit")

        LLVM.br!(builder, LLVM.icmp!(builder, LLVM.API.LLVMIntEQ, LLVM.ConstantInt(0), len), exit, loop)

        LLVM.position!(builder, loop)
        idx = LLVM.phi!(builder, ity)

        push!(LLVM.incoming(idx), (LLVM.ConstantInt(0), entry))
        inc = LLVM.add!(builder, idx, LLVM.ConstantInt(1))
        push!(LLVM.incoming(idx), (inc, loop))
        rval = LLVM.add!(builder, inc, lstart)
        res = LLVM.call!(builder, LLVM.function_type(copysetfn), copysetfn, [inp, rval])
        if !hasNoRet
            gidx = LLVM.gep!(builder, jlvaluet, alloc, [idx])
            LLVM.store!(builder, res, gidx)
            Compiler.emit_writebarrier!(builder, Compiler.get_julia_inner_types(builder, obj, res))
        end

        LLVM.br!(builder, LLVM.icmp!(builder, LLVM.API.LLVMIntEQ, inc, len), exit, loop)


        T_int32 = LLVM.Int32Type()

        Compiler.reinsert_gcmarker!(llvm_f)

        LLVM.position!(builder, exit)
        LLVM.ret!(builder, obj)

        string(mod)
    end
    return quote
        Base.@_inline_meta
        Base.llvmcall(
            ($ir, "f"),
            Tuple{Vararg{T}},
            Tuple{T, Int, Int},
            x,
            startv,
            lengthv
        )
    end
end

@inline function onehot(x::Array)
    onehot_internal(zerosetfn, x, 0, length(x))
end

@inline function onehot(x::Array, start::Int, endl::Int)
    onehot_internal(zerosetfn, x, start-1, endl-start+1)
end

@inline function onehot(x::AbstractArray)
    N = length(x)
    ntuple(Val(N)) do i
        Base.@_inline_meta
        res = similar(x)
        for idx = 1:N
            @inbounds res[idx] = (i == idx) ? 1.0 : 0.0
        end
        return res
    end
end
@inline function onehot(x::AbstractArray, start::Int, endl::Int)
    ntuple(Val(endl - start + 1)) do i
        Base.@_inline_meta
        res = similar(x)
        for idx = 1:length(x)
            @inbounds res[idx] = (i + start - 1 == idx) ? 1.0 : 0.0
        end
        return res
    end
end

@inline function onehot(::Type{NTuple{N,T}}) where {T,N}
    ntuple(Val(N)) do i
        Base.@_inline_meta
        ntuple(Val(N)) do idx
            Base.@_inline_meta
            return (i == idx) ? T(1) : T(0)
        end
    end
end
@inline onehot(x::Tuple{}) = ()
@inline function onehot(x::NTuple{N,T}) where {T,N}
    onehot(NTuple{N,T})
end
@inline function onehot(x::NTuple{N,T}, start, endl) where {T,N}
    ntuple(Val(endl - start + 1)) do i
        Base.@_inline_meta
        ntuple(Val(N)) do idx
            Base.@_inline_meta
            return (i + start - 1 == idx) ? T(1) : T(0)
        end
    end
end

@inline function onehot(x::AbstractFloat)
    return (one(x),)
end

"""
    gradient(::ReverseMode, f, args...)

Compute the gradient of a real-valued function `f` using reverse mode.
For each differentiable argument, this function will allocate and return new derivative object, returning
a tuple of derivatives for each argument. If an argument is not differentiable, the element of the returned
tuple with be nothing.

In reverse mode (here), the derivatives will be the same type as the original argument.

This is a structure gradient. For a struct `x` it returns another instance of the same type,
whose fields contain the components of the gradient.
In the result, `grad.a` contains `∂f/∂x.a` for any differential `x.a`,
while `grad.c == x.c` for other types.

Examples:

```jldoctest gradient
f(x) = x[1]*x[2]

grad = gradient(Reverse, f, [2.0, 3.0])

# output
([3.0, 2.0],)
```

```jldoctest gradient
grad = gradient(Reverse, only ∘ f, (a = 2.0, b = [3.0], c = "str"))

# output

((a = 3.0, b = [2.0], c = "str"),)
```

```jldoctest gradient
mul(x, y) = x[1]*y[1]

grad = gradient(Reverse, mul, [2.0], [3.0])

# output
([3.0], [2.0])
```

```jldoctest gradient

grad = gradient(Reverse, mul, [2.0], Const([3.0]))

# output
([3.0], nothing)
```

If passing a mode that returns the primal (e.g. ReverseWithPrimal), the return type will instead be
a tuple where the first element contains the derivatives, and the second element contains the result of the original computation.

```jldoctest gradient

grad = gradient(ReverseWithPrimal, f, [2.0, 3.0])

# output
(derivs = ([3.0, 2.0],), val = 6.0)
```
```jldoctest gradient

grad = gradient(ReverseWithPrimal, mul, [2.0], [3.0])

# output
(derivs = ([3.0], [2.0]), val = 6.0)
```

```jldoctest gradient
grad = gradient(ReverseWithPrimal, mul, [2.0], Const([3.0]))

# output
(derivs = ([3.0], nothing), val = 6.0)
```

"""
# TODO eventually add an invalidation edge here from inactive_type
@generated function gradient(
    rm::ReverseMode{ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten},
    f::F,
    x::ty_0,
    args::Vararg{Any,N},
) where {F,ty_0,ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten,N}
    rargs = Union{Symbol,Expr}[:x]
    gentys = Type[x]
    acts = Symbol[Symbol("act_0")]

    for i = 1:N
        argidx = quote
            args[$i]
        end
        push!(rargs, argidx)
        sym = Symbol("act_$i")
        push!(acts, sym)
        push!(gentys, args[i])
    end

    toemit = Expr[]
    states = Compiler.ActivityState[]

    for (argidx, act, genty) in zip(rargs, acts, gentys)
        if genty <: Enzyme.Const
            push!(
                toemit,
                quote
                    $act = false
                end
            )
            push!(states, Compiler.AnyState)
        else
            state = Compiler.active_reg_inner(genty, (), nothing)
            push!(states, state)
        end
    end

    idx = 0
    enz_args = Union{Expr,Symbol}[]
    resargs = Union{Expr,Symbol}[]
    for (i, (arg, act, state, genty)) in enumerate(zip(rargs, acts, states, gentys))
        shad = Symbol("shad_$i")
        if genty <: Enzyme.Const
            push!(enz_args, arg)
            push!(resargs, :nothing)
        elseif state == Compiler.MixedState
            push!(toemit, quote
                $shad = Ref(make_zero($arg))
            end)
            push!(enz_args, quote
                MixedDuplicated($arg, $shad)
            end)
            push!(resargs, quote
                $shad[]
            end)
        elseif state == Compiler.DupState
            push!(toemit, quote
                $shad = make_zero($arg)
            end)
            push!(enz_args, quote
                Duplicated($arg, $shad)
            end)
            push!(resargs, quote
                $shad
            end)
        elseif state == Compiler.ActiveState
            push!(enz_args, quote
                Active($arg)
            end)
            push!(resargs, quote
                res[1][$i]
            end)
        else
            @assert state == Compiler.AnyState
            push!(enz_args, quote
                Const($arg)
            end)
            push!(resargs, :nothing)
        end
        idx += 1
    end
    push!(toemit, quote
        res = autodiff(rm, f, Active, $(enz_args...))
    end)

    if ReturnPrimal
        return quote
            Base.@_inline_meta
            $(toemit...)
            (; derivs = ($(resargs...),), val = res[2])
        end
    else
        return quote
            Base.@_inline_meta
            $(toemit...)
            ($(resargs...),)
        end
    end
end

"""
    gradient!(::ReverseMode, dx, f, x)

Compute the gradient of an array-input function `f` using reverse mode,
storing the derivative result in an existing array `dx`.
Both `x` and `dx` must be `Array`s of the same type.

Example:

```jldoctest gradip
f(x) = x[1]*x[2]

dx = [0.0, 0.0]
gradient!(Reverse, dx, f, [2.0, 3.0])

# output
([3.0, 2.0],)
```

```jldoctest gradip
dx = [0.0, 0.0]
gradient!(ReverseWithPrimal, dx, f, [2.0, 3.0])

# output
(derivs = ([3.0, 2.0],), val = 6.0)
```
"""
@inline function gradient!(
    rm::ReverseMode{ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten},
    dx::X,
    f::F,
    x::X,
) where {X<:Array,F,ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten}
    make_zero!(dx)
    res = autodiff(rm, f, Active, Duplicated(x, dx))
    return if ReturnPrimal
        (; derivs = (dx,), val = res[2])
    else
        (dx,)
    end
end

@inline function chunkedonehot(x, ::Val{chunk}) where {chunk}
    sz = length(x)
    num = ((sz + chunk - 1) ÷ chunk)
    ntuple(Val(num)) do i
        Base.@_inline_meta
        onehot(x, (i - 1) * chunk + 1, i == num ? sz : (i * chunk))
    end
end

@inline function chunkedonehot(x::AbstractFloat, ::Val{chunk}) where {chunk}
    return ((one(x),),)
end

@inline tupleconcat(x) = x
@inline tupleconcat(x, y) = (x..., y...)
@inline tupleconcat(x, y, z...) = (x..., tupleconcat(y, z...)...)

@generated function create_shadows(chunk::ChunkTy, x::X, vargs::Vararg{Any,N}) where {ChunkTy, X, N}
    args =  Union{Symbol,Expr}[:x]
    tys =  Type[X]
    for i in 1:N
        push!(args, :(vargs[$i]))
        push!(tys, vargs[i])
    end

    exprs = Union{Symbol,Expr}[]
    for (arg, ty) in zip(args, tys)
        if ty <: Enzyme.Const
            push!(exprs, :(nothing))
        elseif ty <: AbstractFloat
            push!(exprs, :(nothing))
        elseif ChunkTy == Nothing || ChunkTy == Val{1}
            push!(exprs, :(onehot($arg)))
        else
            push!(exprs, :(chunkedonehot($arg, chunk)))
        end
    end
    return quote
        Base.@_inline_meta
        ($(exprs...),)
    end
end

struct TupleArray{T,Shape,Length,N} <: AbstractArray{T,N}
    data::NTuple{Length,T}
end
TupleArray(data::NTuple{Length,T}, Shape) where {Length,T} =
    TupleArray{T,Shape,Length,length(Shape)}(data)

@inline Base.eltype(::TupleArray{T}) where {T} = T
@inline Base.eltype(::Type{<:TupleArray{T}}) where {T} = T
@inline Base.size(::TupleArray{<:Any,Shape}) where {Shape} = Shape
@inline Base.ndims(::TupleArray{<:Any,<:Any,<:Any,N}) where {N} = N

function Base.convert(
    ::Type{Array{T,N}},
    X::TupleArray{T,Shape,Length,N},
) where {T,Shape,Length,N}
    vals = Array{T,N}(undef, Shape...)
    for i = 1:Length
        @inbounds val[i] = X.data[i]
    end
    return vals
end

function Base.getindex(a::TupleArray, args::Vararg{Int,N}) where {N}
    start = 0
    for i = 1:N
        start *= size(a, N - i + 1)
        start += (args[N-i+1] - 1)
    end
    start += 1
    return a.data[start]
end

@inline function tupstack(data::Tuple{Vararg{Array{T}}}, outshape::Tuple{Vararg{Int}}, inshape::Tuple{Vararg{Int}}) where {T}
	num = prod(outshape)
	res = Array{T}(undef, outshape..., inshape...)
	for (i, val) in enumerate(data)
		Base.unsafe_copyto!(res, num*(i-1)+1, val, 1, Base.reinterpret(UInt, num))
	end
	res
end

@inline function tupstack(x, outshape::Tuple{Vararg{Int}}, inshape::Tuple{Vararg{Int}})
    st = Base.stack(x)
    if length(outshape) == 1
        st
    else
        reshape(st, (outshape..., inshape...))
    end
end

@inline specialize_output(output, input) = output

"""
    gradient(::ForwardMode, f, x; shadows=onehot(x), chunk=nothing)

Compute the gradient of an array-input function `f` using forward mode. The
optional keyword argument `shadow` is a vector of one-hot vectors of type `x`
which are used to forward-propagate into the return. For performance reasons,
this should be computed once, outside the call to `gradient`, rather than
within this call.

Example:

```jldoctest gradfwd
f(x) = x[1]*x[2]

gradient(Forward, f, [2.0, 3.0])

# output

([3.0, 2.0],)
```

```jldoctest gradfwd
gradient(ForwardWithPrimal, f, [2.0, 3.0])

# output
(derivs = ([3.0, 2.0],), val = 6.0)
```

```jldoctest gradfwd
gradient(Forward, f, [2.0, 3.0]; chunk=Val(1))

# output

([3.0, 2.0],)
```

```jldoctest gradfwd
gradient(ForwardWithPrimal, f, [2.0, 3.0]; chunk=Val(1))

# output
(derivs = ([3.0, 2.0],), val = 6.0)
```

For functions which return an AbstractArray or scalar, this function will return an AbstractArray
whose shape is `(size(output)..., size(input)...)`. No guarantees are presently made
about the type of the AbstractArray returned by this function (which may or may not be the same
as the input AbstractArray if provided).

For functions who return other types, this function will retun an AbstractArray
of shape `size(input)` of values of the output type. 
```jldoctest
f(x) = [ x[1] * x[2], x[2] + x[3] ]

grad = gradient(Forward, f, [2.0, 3.0, 4.0])

# output
([3.0 2.0 0.0; 0.0 1.0 1.0],)
```

This function supports multiple arguments and computes the gradient with respect to each

```jldoctest gradfwd2
mul(x, y) = x[1]*y[2] + x[2]*y[1]

gradient(Forward, mul, [2.0, 3.0], [2.7, 3.1])

# output

([3.1, 2.7], [3.0, 2.0])
```

This includes the ability to mark some arguments as `Const` if its derivative is not needed, returning nothing in the corresponding derivative map.

```jldoctest gradfwd2
gradient(Forward, mul, [2.0, 3.0], Const([2.7, 3.1]))

# output

([3.1, 2.7], nothing)
```
"""
@generated function gradient(
    fm::ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity},
    f::F,
    x::ty_0,
    args::Vararg{Any,N};
    chunk::CS = nothing,
    shadows::ST = create_shadows(chunk, x, args...),
) where {F, ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,CS,ST, ty_0, N}

    syms = Union{Symbol,Expr}[:x]
    shads = Union{Symbol,Expr}[:(shadows[1])]
    tys = Type[ty_0]
    for i in 1:N
        push!(syms, :(args[$i]))
        push!(tys, args[i])
        push!(shads, :(shadows[1+$i]))
    end
    fval = if F <: Annotation
        :(f.val)
    else
        :f
    end

    vals = Union{Symbol,Expr}[]
    consts = Union{Symbol,Expr}[]
    for (arg, ty) in zip(syms, tys)
        if ty <: Const
            push!(vals, :($arg.val))
            push!(consts, arg)
        else
            push!(vals, arg)
            push!(consts, :(Const($arg)))
        end
    end

    if CS == Val{0}
        return quote
            Base.@_inline_meta
            throw(ErrorException("Cannot differentiate with a batch size of 0"))
        end
    end

    exprs = Union{Symbol,Expr}[]
    primal = nothing
    derivatives = Union{Symbol,Expr}[]

    primmode = :(fm)
    for (i, (arg, ty)) in enumerate(zip(syms, tys))
        if ty <: Const
            push!(derivatives, :(nothing))
            continue
        end

        argnum = length(ST.parameters[i].parameters)

        argderivative = if ty <: AbstractFloat
            dargs = Union{Symbol,Expr}[]
            for (j, arg2) in enumerate(syms)
                if i == j
                    push!(dargs, :(Duplicated($arg, one($arg))))
                else
                    push!(dargs, consts[j])
                end
            end

            resp = Symbol("resp_$i")
            push!(exprs, quote
                $resp = autodiff($primmode, f, Duplicated, $(dargs...))
            end)
            if ReturnPrimal && primal == nothing
                primal = :($resp[2])
                primmode = NoPrimal(fm())
            end

            :($resp[1])
        elseif argnum == 0
            vals[i]
        elseif CS == Nothing
            dargs = Union{Symbol,Expr}[]
            for (j, arg2) in enumerate(syms)
                if i == j
                    push!(dargs, :(BatchDuplicated($arg, $(shads[i]))))
                else
                    push!(dargs, consts[j])
                end
            end

            df = :f
            if F <: Enzyme.Duplicated
                zeros = Expr[]
                for i in 1:argnum
                    push!(zeros, :(f.dval))
                end
                df = :(BatchDuplicated(f.val, ($(zeros...),) ))
            end

            resp = Symbol("resp_$i")
            push!(exprs, quote
                $resp = autodiff($primmode, $df, BatchDuplicated, $(dargs...))
            end)
            if ReturnPrimal && primal == nothing
                primal = :($resp[2])
                primmode = NoPrimal(fm())
            end

            :(values($resp[1]))
        elseif CS == Val{1}
            subderivatives = Union{Symbol,Expr}[]
            for an in 1:argnum
                dargs = Union{Symbol,Expr}[]
                for (j, arg2) in enumerate(syms)
                    if i == j
                        push!(dargs, :(Duplicated($arg, $(shads[i])[$an])))
                    else
                        push!(dargs, consts[j])
                    end
                end

                resp = Symbol("resp_$i"*"_"*string(an))
                push!(exprs, quote
                    $resp = autodiff($primmode, f, Duplicated, $(dargs...))
                end)
                if ReturnPrimal && primal == nothing
                    primal = :($resp[2])
                    primmode = NoPrimal(fm())
                end

                push!(subderivatives, :(values($resp[1])))
            end
            :(($(subderivatives...),))
        else
            subderivatives = Union{Symbol,Expr}[]
            for an in 1:argnum
                dargs = Union{Symbol,Expr}[]
                for (j, arg2) in enumerate(syms)
                    if i == j
                        push!(dargs, :(BatchDuplicated($arg, $(shads[i])[$an])))
                    else
                        push!(dargs, consts[j])
                    end
                end

                resp = Symbol("resp_$i"*"_"*string(an))
                push!(exprs, quote
                    $resp = autodiff($primmode, f, BatchDuplicated, $(dargs...))
                end)
                if ReturnPrimal && primal == nothing
                    primal = :($resp[2])
                    primmode = NoPrimal(fm())
                end

                push!(subderivatives, :(values($resp[1])))
            end
            :(tupleconcat($(subderivatives...)))
        end

        deriv = if ty <: AbstractFloat
            argderivative
        else
            tmp = Symbol("tmp_$i")
            push!(exprs, :($tmp = $argderivative))
            if ty <: AbstractArray
                if argnum > 0
                    quote
                        if $tmp[1] isa AbstractArray
                            inshape = size($(vals[1]))
                            outshape = size($tmp[1])
                            # st : outshape x total inputs
                            tupstack($tmp, outshape, inshape)
                        else
                            specialize_output(TupleArray($tmp, size($arg)), $(vals[1]))
                        end
                    end
                else
                    tmp
                end
            else
                tmp
            end
        end
        push!(derivatives, deriv)
    end

    # We weirdly asked for no derivatives
    if ReturnPrimal && primal == nothing
        primal = :($fval($(vals...)))
    end

    result = if ReturnPrimal
        :((; derivs = ($(derivatives...),), val = $primal))
    else
        :(($(derivatives...),))
    end

    return quote
        Base.@_inline_meta
        $(exprs...)
        $result
    end
end

"""
    jacobian(::ForwardMode, args...; kwargs...)

Equivalent to gradient(::ForwardMode, args...; kwargs...)
"""
@inline function jacobian(fm::ForwardMode, args...; kwargs...)
    gradient(fm, args...; kwargs...)
end

"""
    jacobian(::ReverseMode, f, x; n_outs=nothing, chunk=nothing)
    jacobian(::ReverseMode, f, x)

Compute the jacobian of a array-output function `f` using (potentially vector)
reverse mode. The `chunk` argument optionally denotes the chunk size to use and
`n_outs` optionally denotes the shape of the array returned by `f` (e.g `size(f(x))`).

Example:

```jldoctest
f(x) = [ x[1] * x[2], x[2] + x[3] ]

jacobian(Reverse, f, [2.0, 3.0, 4.0])

# output
([3.0 2.0 0.0; 0.0 1.0 1.0],)
```

```jldoctest
f(x) = [ x[1] * x[2], x[2] + x[3] ]

grad = jacobian(ReverseWithPrimal, f, [2.0, 3.0, 4.0])

# output
(derivs = ([3.0 2.0 0.0; 0.0 1.0 1.0],), val = [6.0, 7.0])
```

```jldoctest
f(x) = [ x[1] * x[2], x[2] + x[3] ]

grad = jacobian(Reverse, f, [2.0, 3.0, 4.0], n_outs=Val((2,)))

# output
([3.0 2.0 0.0; 0.0 1.0 1.0],)
```

```jldoctest
f(x) = [ x[1] * x[2], x[2] + x[3] ]

grad = jacobian(ReverseWithPrimal, f, [2.0, 3.0, 4.0], n_outs=Val((2,)))

# output
(derivs = ([3.0 2.0 0.0; 0.0 1.0 1.0],), val = [6.0, 7.0])
```

This function will return an AbstractArray whose shape is `(size(output)..., size(input)...)`.
No guarantees are presently made about the type of the AbstractArray returned by this function
(which may or may not be the same as the input AbstractArray if provided).

In the future, when this function is extended to handle non-array return types, 
this function will retun an AbstractArray of shape `size(output)` of values of the input type. 
```
"""
@inline function jacobian(
    mode::ReverseMode{ReturnPrimal,RuntimeActivity,RABI,Holomorphic,ErrIfFuncWritten},
    f::F,
    x::X;
    n_outs::OutType = nothing,
    chunk::CT = nothing,
) where {ReturnPrimal,F,X,RABI<:ABI,ErrIfFuncWritten,RuntimeActivity,OutType,CT,Holomorphic}

    if n_outs == nothing
        res = if f isa Const
            f.val(x)
        else
            f(x)
        end
        jac = if res isa AbstractArray
            jacobian(
                ReverseMode{false,RuntimeActivity,RABI,Holomorphic,ErrIfFuncWritten}(),
                f,
                x;
                n_outs = Val(size(res)),
                chunk,
            )
        elseif res isa AbstractFloat
            gradient(
                ReverseMode{false,RuntimeActivity,RABI,Holomorphic,ErrIfFuncWritten}(),
                f,
                x,
            )
        else
            throw(
                AssertionError(
                    "Unsupported return type of function for reverse-mode jacobian, $(Core.Typeof(res))",
                ),
            )
        end

        return if ReturnPrimal
            (; derivs = jac, val = res)
        else
            jac
        end
    else
        n_out_val = if length(Compiler.element(n_outs)) == 0
            0
        else
            prod(Compiler.element(n_outs))
        end

        if chunk == Val(0)
            throw(ErrorException("Cannot differentiate with a batch size of 0"))
        end

        XT = Core.Typeof(x)
        MD = Compiler.active_reg_inner(XT, (), nothing, Val(true)) == Compiler.ActiveState #=justActive=#
        tt = Tuple{XT}
        FRT = if f isa Const
            Core.Typeof(f.val)
        else
            Core.Typeof(f)
        end

        rt = Compiler.primal_return_type(Reverse, FRT, tt)

        ModifiedBetweenT = (false, false)
        FA = Const{FRT}

        if chunk == Val(1) || chunk == nothing
            primal, adjoint = autodiff_thunk(
                ReverseModeSplit{
                    #=ReturnPrimal=#false,
                    #=ReturnShadow=#true,
                    RuntimeActivity,
                    #=width=#1,
                    ModifiedBetweenT,
                    RABI,
                    Holomorphic,
                    ErrIfFuncWritten,
                    #=ShadowInit=#false
                }(),
                FA,
                DuplicatedNoNeed{rt},
                MD ? MixedDuplicated{XT} : Duplicated{XT}
            )
            tmp = ntuple(Val(n_out_val)) do i
                Base.@_inline_meta
                z = make_zero(x)
                dx = MD ? Ref(z) : z
                res = primal(Const(f), MD ? MixedDuplicated(x, dx) : Duplicated(x, dx))
                tape = res[1]
                @inbounds res[3][i] += Compiler.default_adjoint(eltype(typeof(res[3])))
                adjoint(Const(f), MD ? MixedDuplicated(x, dx) : Duplicated(x, dx), tape)
                return MD ? dx[] : dx, (i == 1 ? size(res[3]) : nothing)
            end
            rows = map(first, tmp)
            outshape = tmp[1][2]
            rows, outshape
        else
            chunksize = Compiler.element(chunk)
            primal, adjoint = autodiff_thunk(
                ReverseModeSplit{
                    #=ReturnPrimal=#false,
                    #=ReturnShadow=#true,
                    RuntimeActivity,
                    chunksize,
                    ModifiedBetweenT,
                    RABI,
                    Holomorphic,
                    ErrIfFuncWritten,
                    #=ShadowInit=#false
                }(),
                FA,
                BatchDuplicatedNoNeed{rt, chunksize},
                MD ? BatchMixedDuplicated{XT, chunksize} : BatchDuplicated{XT, chunksize}
            )

            num = ((n_out_val + chunksize - 1) ÷ chunksize)

            if num * chunksize == n_out_val
                last_size = chunksize
                primal2, adjoint2 = primal, adjoint
            else
                last_size = n_out_val - (num - 1) * chunksize
                tt′ = Tuple{BatchDuplicated{Core.Typeof(x),last_size}}
                primal2, adjoint2 = autodiff_thunk(
                    ReverseModeSplit{
                        #=ReturnPrimal=#false,
                        #=ReturnShadow=#true,
                        RuntimeActivity,
                        last_size,
                        ModifiedBetweenT,
                        RABI,
                        Holomorphic,
                        ErrIfFuncWritten,
                        #=ShadowInit=#false
                    }(),
                    FA,
                    BatchDuplicatedNoNeed{rt, last_size},
                    MD ? BatchMixedDuplicated{XT, last_size} : BatchDuplicated{XT, last_size}
                )
            end

            tmp = ntuple(num) do i
                Base.@_inline_meta
                dx = ntuple(Val(i == num ? last_size : chunksize)) do idx
                    Base.@_inline_meta
                    z = make_zero(x)
                    MD ? Ref(z) : z
                end
                res = (i == num ? primal2 : primal)(
                    Const(f),
                    MD ? BatchMixedDuplicated(x, dx) : BatchDuplicated(x, dx),
                )
                tape = res[1]
                j = 0
                for shadow in res[3]
                    j += 1
                    @inbounds shadow[(i-1)*chunksize+j] +=
                        Compiler.default_adjoint(eltype(typeof(shadow)))
                end
                (i == num ? adjoint2 : adjoint)(
                    Const(f),
                    MD ? BatchMixedDuplicated(x, dx) : BatchDuplicated(x, dx),
                    tape,
                )
                return MD ? (
                    ntuple(Val(i == num ? last_size : chunksize)) do idx
                        Base.@_inline_meta
                        dx[idx][]
                    end
                ) : dx,
                (i == 1 ? size(res[3][1]) : nothing)
            end
            rows = tupleconcat(map(first, tmp)...)
            outshape = tmp[1][2]
            rows, outshape
        end
        res = if x isa AbstractArray
            inshape = size(x)
            st2 = tupstack(rows, inshape, outshape)

            st3 = if length(outshape) == 1 && length(inshape) == 1
                transpose(st2)
            else
                transp = (
                    ((length(inshape)+1):(length(inshape)+length(outshape)))...,
                    (1:length(inshape))...,
                )
                PermutedDimsArray(st2, transp)
            end

            st3
        else
            reshape(collect(rows), outshape)
        end
        if ReturnPrimal
            # TODO optimize away redundant fwd pass
            (; derivs = (res,), val = if f isa Enzyme.Const
                f.val(x)
            else
                f(x)
            end)
        else
            (res,)
        end
    end
end

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
@inline function hvp(f::F, x::X, v::X) where {F,X}
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
@inline function hvp!(res::X, f::F, x::X, v::X) where {F,X}
    grad = make_zero(x)
    Enzyme.autodiff(
        Forward,
        gradient!,
        Const(Reverse),
        DuplicatedNoNeed(grad, res),
        Const(f),
        Duplicated(x, v),
    )
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
@inline function hvp_and_gradient!(res::X, grad::X, f::F, x::X, v::X) where {F,X}
    Enzyme.autodiff(
        Forward,
        gradient!,
        Const(Reverse),
        Duplicated(grad, res),
        Const(f),
        Duplicated(x, v),
    )
    return nothing
end

