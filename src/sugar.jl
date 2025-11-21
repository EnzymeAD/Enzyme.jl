# Syntactic sugar over autodiff calls (e.g. Enzyme.gradient and Enzyme.jacobian)


function zerosetfn(x, i::Int)
    res = zero(x)
    @inbounds res[i] = 1
    return res
end

function zerosetfn!(x, i::Int, val)
    @inbounds x[i] += val
    nothing
end

@generated function onehot_internal(fn::F, x::T, startv::Int, lengthv::Int) where {F, T<:AbstractArray}
    ir = GPUCompiler.JuliaContext() do ctx
        Base.@_inline_meta

        target = Compiler.DefaultCompilerTarget()
        params = Compiler.PrimalCompilerParams(API.DEM_ForwardMode)
        mi = my_methodinstance(nothing, fn, Tuple{T, Int})
        job = GPUCompiler.CompilerJob(mi, GPUCompiler.CompilerConfig(target, params; kernel = false, libraries = true, toplevel = true, optimize = false, cleanup = false, only_entry = false, validate = false))

        GPUCompiler.prepare_job!(job)
        mod, meta = GPUCompiler.emit_llvm(job)
        
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
        idx = LLVM.phi!(builder, ity, "onehot.idx")

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
@inline function onehot(x::NTuple{N,T}, start::Int, endl::Int) where {T,N}
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

@inline function onehot(x::Tuple{Vararg{<:AbstractFloat}})
    ntuple(Val(length(x))) do i
        Base.@_inline_meta
        ntuple(Val(length(x))) do idx
            Base.@_inline_meta
            T = typeof(x[idx])
            return (i == idx) ? T(1) : T(0)
        end
    end
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
@generated function gradient(
    rm::ReverseMode{ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten},
    f::F,
    x::ty_0,
    args::Vararg{Any,N},
) where {F,ty_0,ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten,N}
    # TODO eventually add an invalidation edge here from inactive_type
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
            # TODO we need to make this call the worldage one
            state = Compiler.active_reg_nothrow(genty)
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
    rm::ReverseMode{ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten},
    dx::X,
    f::F,
    x::X,
) where {X<:AbstractArray,F,ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten}
    make_zero!(dx)
    res = autodiff(rm, f, Active, Duplicated(x, dx))
    return if ReturnPrimal
        (; derivs = (dx,), val = res[2])
    else
        (dx,)
    end
end

const ExtendedChunkStrategy = Union{ChunkStrategy, Nothing, Val}

# eats and returns a type because generated functions work on argument types
get_strategy(chunk::Type{CS}) where {CS<:ChunkStrategy} = chunk

function get_strategy(::Type{Nothing})
    Base.depwarn(
        "The `chunk=nothing` configuration will be deprecated in a future release. Please use `chunk=SmallestChunk()` instead.",
        :get_strategy,
    )
    return SmallestChunk
end

function get_strategy(::Type{Val{C}}) where {C}
    Base.depwarn(
        "The `chunk=Val(C)` configuration will be deprecated in a future release. Please use `chunk=FixedChunk{C}()` instead.",
        :get_strategy,
    )
    return FixedChunk{C}
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

@inline function chunkedonehot(x, strategy::ChunkStrategy)
    return chunkedonehot(x, pick_chunksize(strategy, x))
end

@inline tupleconcat(x) = x
@inline tupleconcat(x, y) = (x..., y...)
@inline tupleconcat(x, y, z...) = (x..., tupleconcat(y, z...)...)

@generated function create_shadows(chunk::ExtendedChunkStrategy, x::X, vargs::Vararg{Any,N}) where {X, N}
    chunk_strategy = get_strategy(chunk)
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
        elseif chunk_strategy == SmallestChunk || chunk_strategy == FixedChunk{1}
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

@inline specialize_output(output, input) = output

"""
    gradient(::ForwardMode, f, x, args...; chunk=SmallestChunk(), shadows=create_shadows(chunk, x, args...))

Compute the gradient of an array-input function `f` using forward mode.
The optional keyword argument `chunk` denotes the chunk size to use: it can be any instance of [`EnzymeCore.ChunkStrategy`](@ref EnzymeCore.ChunkStrategy).
The optional keyword argument `shadow` is a vector of one-hot vectors of type `x`
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
gradient(Forward, f, [2.0, 3.0]; chunk=FixedChunk{1}())

# output

([3.0, 2.0],)
```

```jldoctest gradfwd
gradient(ForwardWithPrimal, f, [2.0, 3.0]; chunk=FixedChunk{1}())

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
    fm::ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero},
    f::F,
    x::ty_0,
    args::Vararg{Any,N};
    chunk::ExtendedChunkStrategy = SmallestChunk(),
    shadows::ST = create_shadows(chunk, x, args...),
) where {F, ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero,ST, ty_0, N}

    chunk_strategy = get_strategy(chunk)

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

    if chunk_strategy == FixedChunk{0}
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
        elseif chunk_strategy == SmallestChunk
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
        elseif chunk_strategy == FixedChunk{1}
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
                            inshape = size($(vals[i]))
                            outshape = size($tmp[1])
                            num = prod(outshape)

                            # st : outshape x total inputs
                            tupstack($tmp, outshape, inshape)
                        else
                            specialize_output(TupleArray($tmp, size($arg)), $(vals[i]))
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

Equivalent to `gradient(::ForwardMode, args...; kwargs...)`.
"""
@inline function jacobian(fm::ForwardMode, args...; kwargs...)
    gradient(fm, args...; kwargs...)
end

@generated function jacobian_helper(
    mode::ReverseMode{ReturnPrimal},
    RT::RType,
    n_outs::OutType,
    chunk::ExtendedChunkStrategy,
    f::F,
    xs::Vararg{Any, Nargs}
) where {ReturnPrimal,RType, F,Nargs,OutType}
    chunk_strategy = get_strategy(chunk)
    fty = if f <: Enzyme.Annotation
        f.parameters[1]
    else
        f
    end

    primval = if f <: Enzyme.Annotation
        :(f.val)
    else
        :f
    end

    constargs = []
    consttys = []
    for (i, T) in enumerate(xs)
        if T <: Enzyme.Annotation
            push!(consttys, T.parameters[1])
            push!(constargs, :(xs[$i].val))
        else
            push!(consttys, T)
            push!(constargs, :(xs[$i]))
        end
    end

    callprim = Expr(:call, primval, constargs...)

    if length(xs) == 0
        if ReturnPrimal
            return quote
                Base.@_inline_meta
                (; derivs = (), val = $callprim)
            end
        else
            return quote
                Base.@_inline_meta
                ()
            end
        end
    end

    noutsexpr = :n_outs

    exprs = Expr[]
    if n_outs == Nothing
        if RT <: AbstractFloat
            return quote
                Base.@_inline_meta
                gradient(
                    mode,
                    f,
                    xs...,
                )
            end
        end

        return quote
            Base.@_inline_meta
            res = $callprim
            jac = if res isa AbstractArray
                jacobian_helper(
                    NoPrimal(mode),
                    RT,
                    Val(size(res)),
                    chunk,
                    f,
                    xs...
                )
            elseif res isa AbstractFloat
                gradient(
                    NoPrimal(mode),
                    f,
                    xs...,
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
        end
    end

    if chunk_strategy == FixedChunk{0}
        return quote
            throw(ErrorException("Cannot differentiate with a batch size of 0"))
        end
    end

    @assert n_outs <: Val
    nouts2 = n_outs.parameters[1]

    n_out_val = if length(nouts2) == 0
        0
    else
        prod(nouts2)
    end

    exprs = Expr[]
    XTs = Symbol[]
    MDs = Symbol[]
    MDTys = Union{Expr,Symbol}[]
    MDTysLast = Union{Expr,Symbol}[]

    chunksize_val = pick_chunksize(chunk_strategy(), n_out_val)
    chunksize = typeof(chunksize_val).parameters[1]

    num = ((n_out_val + chunksize - 1) ÷ chunksize)

    last_size = if num * chunksize == n_out_val
        chunksize
    else
        n_out_val - (num - 1) * chunksize
    end

    for i in 1:length(xs)
        xti = Symbol("XT_", i)
        push!(XTs, xti)
        mdi = Symbol("MD_", i)
        push!(MDs, mdi)

        push!(exprs, Expr(:(=), xti, :(Core.Typeof(xs[$i]))))

        if xs[i] <: Const
            push!(exprs, Expr(:(=), mdi, false))
            push!(MDTys, xti)
            push!(MDTysLast, xti)
        else
            push!(exprs, Expr(:(=), mdi, :(Compiler.active_reg_nothrow($xti) == Compiler.ActiveState || Compiler.active_reg_nothrow($xti) == Compiler.MixedState)))

            if chunk_strategy == SmallestChunk || chunk_strategy == FixedChunk{1}
                push!(MDTys, :($mdi ? MixedDuplicated{$xti} : Duplicated{$xti}))
            else
                push!(MDTys, :($mdi ? BatchMixedDuplicated{$xti, $chunksize} : BatchDuplicated{$xti, $chunksize}))
                if last_size == 1
                    push!(MDTysLast, :($mdi ? MixedDuplicated{$xti} : Duplicated{$xti}))
                else
                    push!(MDTysLast, :($mdi ? BatchMixedDuplicated{$xti, $last_size} : BatchDuplicated{$xti, $last_size}))
                end
            end
        end
    end

    ModifiedBetween = ntuple(Returns(false), length(xs)+1)

    cst_fn = if f <: Enzyme.Annotation
        :f
    else
        :(Const(f))
    end

    postexprs = Expr[]

    torows = Matrix{Union{Expr, Nothing}}(undef, n_out_val, length(xs))


    curidx = 1
    for i in 1:num

        batchcnt = if i == num
            last_size
        else
            chunksize
        end

        args = []
        dxs = Symbol[]
        for j in 1:length(xs)
            if xs[j] <: Enzyme.Const
                push!(args, :(xs[$j]))
                push!(dxs, :undefined)
                continue
            end

            dx = Symbol("dx_", curidx, "_", j)

            e0 = :(make_zero(xs[$j]))

            if batchcnt == 1
                zsym = Symbol("zero_", curidx, "_", j)
                push!(postexprs, Expr(:(=), zsym, e0))
                push!(postexprs, Expr(:(=), dx, :($(MDs[j]) ? Ref($zsym) : $zsym)))
            else
                eexprs = Expr[]
                for b in 1:batchcnt
                    zsym = Symbol("zero_", curidx, "_", j, "_", b)
                    push!(postexprs, Expr(:(=), zsym, e0))
                    push!(eexprs, :($(MDs[j]) ? Ref($zsym) : $zsym))
                end
                tup = Expr(:tuple, eexprs...)
                push!(postexprs, Expr(:(=), dx, tup))
            end

            push!(dxs, dx)
            if batchcnt == 1
                push!(args, :($(MDs[j]) ? MixedDuplicated(xs[$j], $dx) : Duplicated(xs[$j], $dx)))
            else
                push!(args, :($(MDs[j]) ? BatchMixedDuplicated(xs[$j], $dx) : BatchDuplicated(xs[$j], $dx)))
            end
        end

        ressym = Symbol("res_", curidx)
        push!(postexprs, Expr(:(=), ressym, Expr(:call, (chunksize != 1 && i == num) ? :primal2 : :primal, cst_fn, args...)))

        if batchcnt == 1
            push!(postexprs, :(zerosetfn!($ressym[3], $curidx, Compiler.default_adjoint(eltype(typeof($ressym[3]))))))
        else
            for k in 1:batchcnt
                push!(postexprs, :(zerosetfn!($ressym[3][$k], $(curidx+k-1), Compiler.default_adjoint(eltype(typeof($ressym[3][$k]))))))
            end
        end

        push!(postexprs, Expr(:call, (chunksize != 1 && i == num) ? :adjoint2 : :adjoint, cst_fn, args..., :($ressym[1])))

        if curidx == 1
            if batchcnt == 1
                push!(postexprs,
                    Expr(:(=), :outshape, :(size($ressym[3])))
                )
            else
                push!(postexprs,
                    Expr(:(=), :outshape, :(size($ressym[3][1])))
                )
            end
        end

        for j in 1:length(xs)
            if batchcnt == 1
                if xs[j] <: Enzyme.Const
                    torows[curidx, j] = nothing
                else
                    torows[curidx, j] = :($(MDs[j]) ? $(dxs[j])[] : $(dxs[j]))
                end
            else
                for k in 1:batchcnt
                    if xs[j] <: Enzyme.Const
                        torows[curidx+k-1, j] = nothing
                    else
                        torows[curidx+k-1, j] = :($(MDs[j]) ? $(dxs[j])[$k][] : $(dxs[j])[$k])
                    end
                end
            end
        end

        curidx += batchcnt
    end

    results = []


    for j in 1:length(xs)
        if xs[j] <: Enzyme.Const
            push!(results, nothing)
            continue
        end
        if xs[j] <: AbstractArray
            inshape = Symbol("inshape_", j)
            push!(postexprs, Expr(:(=), inshape, :(size(xs[$j]))))

            resj = Symbol("tempres_", j)
            push!(postexprs, Expr(:(=), resj, :(Array{$(eltype(xs[j]))}(undef, $(inshape)..., outshape...))))

            numv = Symbol("num_", j)
            push!(postexprs, Expr(:(=), numv, :(prod($inshape))))

            for i in 1:n_out_val
                push!(postexprs, Expr(:call, :(Base.unsafe_copyto!), resj, :($numv*($i-1)+1), torows[i, j], 1, :(Base.reinterpret(UInt, $numv))))
            end

            push!(results, quote
                if length(outshape) == 1 && length($inshape) == 1
                    transpose($resj)
                else
                    transp = (
                        ((length($inshape)+1):(length($inshape)+length(outshape)))...,
                        (1:length($inshape))...,
                    )
                    PermutedDimsArray($resj, transp)
                end
            end)
        else
            push!(results, :(reshape($(xs[j])[$(torows[:, j]...)], outshape)))
        end
    end


    if ReturnPrimal
        # TODO optimize away redundant fwd pass
        push!(postexprs, quote
            derivs = ($(results...),)
            return (; derivs = derivs, val = $callprim)
        end)
    else
        push!(postexprs, quote
            return ($(results...),)
        end)
    end

    prim2 = if chunksize == 1
        quote end
    else
        if num * chunksize == n_out_val
            quote
                primal2, adjoint2 = primal, adjoint
            end    
        else
            BNN2 = if last_size == 1
                :(DuplicatedNoNeed{RT})
            else
                :(BatchDuplicatedNoNeed{RT, $last_size})
            end 
            quote
                primal2, adjoint2 = autodiff_thunk(
                    EnzymeCore.Split(EnzymeCore.NoPrimal(mode),
                        #=ReturnShadow=#Val(true),
                        #=Width=#Val($last_size),
                        #=ModifiedBetween=#Val(ModifiedBetweenT),
                        #=ShadowInit=#Val(false)
                    ),
                    FA,
                    $BNN2,
                    $(MDTysLast...)
                )
            end
        end
    end

    DRT = if chunksize == 1
        :(DuplicatedNoNeed{RT})
    else
        :(BatchDuplicatedNoNeed{RT, $chunksize})
    end

    return quote
        Base.@_inline_meta
        $(exprs...)

        ModifiedBetweenT = $ModifiedBetween
        FA = Core.Typeof($cst_fn)

        primal, adjoint = autodiff_thunk(
            EnzymeCore.Split(EnzymeCore.NoPrimal(mode),
                #=ReturnShadow=#Val(true),
                #=Width=#Val($chunksize),
                #=ModifiedBetween=#Val(ModifiedBetweenT),
                #=ShadowInit=#Val(false)
            ),
            FA,
            $DRT,
            $(MDTys...)
        )

        $prim2

        $(postexprs...)
    end
end

"""
    jacobian(::ReverseMode, f, x; n_outs=nothing, chunk=SmallestChunk())
    jacobian(::ReverseMode, f, x)

Compute the jacobian of a array-output function `f` using (potentially vector) reverse mode.
The optional keyword argument `chunk` denotes the chunk size to use: it can be any instance of [`EnzymeCore.ChunkStrategy`](@ref EnzymeCore.ChunkStrategy).
The optional keyword argument `n_outs` denotes the shape of the array returned by `f` (e.g `size(f(x))`).

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
@generated function jacobian(
    mode::ReverseMode,
    f::F,
    xs::Vararg{Any, Nargs};
    n_outs::OutType = nothing,
    chunk::ExtendedChunkStrategy = SmallestChunk(),
) where {F,Nargs, OutType}

    fty = if f <: Enzyme.Annotation
        f.parameters[1]
    else
        f
    end

    consttys = []
    for (i, T) in enumerate(xs)
        if T <: Enzyme.Annotation
            push!(consttys, T.parameters[1])
        else
            push!(consttys, T)
        end
    end

    return quote
        Base.@_inline_meta
        RT = Compiler.primal_return_type(Reverse, $fty, $(Tuple{consttys...}))
        return @inline jacobian_helper(mode, RT, n_outs, chunk, f, xs...)
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
