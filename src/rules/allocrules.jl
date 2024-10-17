
function array_inner(::Type{<:Array{T}}) where {T}
    return T
end
function array_shadow_handler(
    B::LLVM.API.LLVMBuilderRef,
    OrigCI::LLVM.API.LLVMValueRef,
    numArgs::Csize_t,
    Args::Ptr{LLVM.API.LLVMValueRef},
    gutils::API.EnzymeGradientUtilsRef,
)::LLVM.API.LLVMValueRef
    inst = LLVM.Instruction(OrigCI)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(inst)))
    ctx = LLVM.context(LLVM.Value(OrigCI))
    gutils = GradientUtils(gutils)

    legal, typ, byref = abs_typeof(inst)
    if !legal
        throw(
            AssertionError(
                "Could not statically ahead-of-time determine allocation element type of " *
                string(inst),
            ),
        )
    end

    typ = eltype(typ)

    b = LLVM.IRBuilder(B)
    orig = LLVM.Value(OrigCI)::LLVM.CallInst

    nm = LLVM.name(LLVM.called_operand(orig)::LLVM.Function)

    memory = nm == "jl_alloc_genericmemory" || nm == "ijl_alloc_genericmemory"

    vals = LLVM.Value[]
    valTys = API.CValueType[]
    for i = 1:numArgs
        push!(vals, LLVM.Value(unsafe_load(Args, i)))
        push!(valTys, API.VT_Primal)
    end

    anti = call_samefunc_with_inverted_bundles!(b, gutils, orig, vals, valTys, false) #=lookup=#

    prod = if memory
        get_memory_len(b, anti)
    else
        get_array_len(b, anti)
    end

    isunboxed, elsz, al = Base.uniontype_layout(typ)

    isunion = typ isa Union

    LLT_ALIGN(x, sz) = (((x) + (sz) - 1) & ~((sz) - 1))

    if !isunboxed
        elsz = sizeof(Ptr{Cvoid})
        al = elsz
    else
        elsz = LLT_ALIGN(elsz, al)
    end

    tot = prod
    tot = LLVM.mul!(b, tot, LLVM.ConstantInt(LLVM.value_type(tot), elsz, false))

    if elsz == 1 && !isunion
        # extra byte for all julia allocated byte arrays
        tot = LLVM.add!(b, tot, LLVM.ConstantInt(LLVM.value_type(tot), 1, false))
    end
    if (isunion)
        # an extra byte for each isbits union array element, stored after a->maxsize
        tot = LLVM.add!(b, tot, prod)
    end

    i8 = LLVM.IntType(8)
    toset = if memory
        get_memory_data(b, anti)
    else
        get_array_data(b, anti)
    end

    mcall = LLVM.memset!(b, toset, LLVM.ConstantInt(i8, 0, false), tot, al)

    ref::LLVM.API.LLVMValueRef = Base.unsafe_convert(LLVM.API.LLVMValueRef, anti)
    return ref
end

function null_free_handler(
    B::LLVM.API.LLVMBuilderRef,
    ToFree::LLVM.API.LLVMValueRef,
    Fn::LLVM.API.LLVMValueRef,
)::LLVM.API.LLVMValueRef
    return C_NULL
end

function register_alloc_handler!(variants, alloc_handler, free_handler)
    for variant in variants
        API.EnzymeRegisterAllocationHandler(variant, alloc_handler, free_handler)
    end
end

@inline function register_alloc_rules()
    register_alloc_handler!(
        (
         "jl_alloc_array_1d", "ijl_alloc_array_1d",
         "jl_alloc_array_2d", "ijl_alloc_array_2d",
         "jl_alloc_array_3d", "ijl_alloc_array_3d",
         "jl_new_array", "ijl_new_array",
         "jl_alloc_genericmemory", "ijl_alloc_genericmemory",
        ),
        @cfunction(
            array_shadow_handler,
            LLVM.API.LLVMValueRef,
            (
                LLVM.API.LLVMBuilderRef,
                LLVM.API.LLVMValueRef,
                Csize_t,
                Ptr{LLVM.API.LLVMValueRef},
                API.EnzymeGradientUtilsRef,
            )
        ),
        @cfunction(
            null_free_handler,
            LLVM.API.LLVMValueRef,
            (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, LLVM.API.LLVMValueRef)
        )
    )
end
