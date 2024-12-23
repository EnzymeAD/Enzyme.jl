
struct RemovedParam end

# Modified from GPUCompiler classify_arguments
function classify_arguments(
    @nospecialize(source_sig::Type),
    codegen_ft::LLVM.FunctionType,
    has_sret::Bool,
    has_returnroots::Bool,
    has_swiftself::Bool,
    parmsRemoved::Vector{UInt64},
)
    codegen_types = parameters(codegen_ft)

    args = []
    codegen_i = 1
    orig_i = 1
    if has_sret
        if !in(orig_i - 1, parmsRemoved)
            codegen_i += 1
        end
        orig_i += 1
    end
    if has_returnroots
        if !in(orig_i - 1, parmsRemoved)
            codegen_i += 1
        end
        orig_i += 1
    end
    if has_swiftself
        if !in(orig_i - 1, parmsRemoved)
            codegen_i += 1
        end
        orig_i += 1
    end
    for (source_i, source_typ) in enumerate(source_sig.parameters)
        if isghostty(source_typ) || Core.Compiler.isconstType(source_typ)
            push!(args, (cc = GPUCompiler.GHOST, typ = source_typ, arg_i = source_i))
            continue
        end
        if in(orig_i - 1, parmsRemoved)
            push!(args, (cc = RemovedParam, typ = source_typ))
            orig_i += 1
            continue
        end
        codegen_typ = codegen_types[codegen_i]

        if codegen_typ isa LLVM.PointerType
            llvm_source_typ = convert(LLVMType, source_typ; allow_boxed = true)
            # pointers are used for multiple kinds of arguments
            # - literal pointer values
            if source_typ <: Ptr || source_typ <: Core.LLVMPtr
                @assert llvm_source_typ == codegen_typ
                push!(
                    args,
                    (
                        cc = GPUCompiler.BITS_VALUE,
                        typ = source_typ,
                        arg_i = source_i,
                        codegen = (typ = codegen_typ, i = codegen_i),
                    ),
                )
                # - boxed values
                #   XXX: use `deserves_retbox` instead?
            elseif llvm_source_typ isa LLVM.PointerType
                if llvm_source_typ != codegen_typ
                    throw(AssertionError("Mismatch codegen type llvm_source_typ=$(string(llvm_source_typ)) codegen_typ=$(string(codegen_typ)) source_i=$source_i source_sig=$source_sig, source_typ=$source_typ, codegen_i=$codegen_i, codegen_types=$(string(codegen_ft))"))
                end
                push!(
                    args,
                    (
                        cc = GPUCompiler.MUT_REF,
                        typ = source_typ,
                        arg_i = source_i,
                        codegen = (typ = codegen_typ, i = codegen_i),
                    ),
                )
                # - references to aggregates
            else
                @assert llvm_source_typ != codegen_typ
                push!(
                    args,
                    (
                        cc = GPUCompiler.BITS_REF,
                        typ = source_typ,
                        arg_i = source_i,
                        codegen = (typ = codegen_typ, i = codegen_i),
                    ),
                )
            end
        else
            push!(
                args,
                (
                    cc = GPUCompiler.BITS_VALUE,
                    typ = source_typ,
                    arg_i = source_i,
                    codegen = (typ = codegen_typ, i = codegen_i),
                ),
            )
        end

        codegen_i += 1
        orig_i += 1
    end

    return args
end

# https://github.com/JuliaLang/julia/blob/64378db18b512677fc6d3b012e6d1f02077af191/src/cgutils.cpp#L823
# returns if all unboxed
function for_each_uniontype_small(@nospecialize(f), @nospecialize(ty::Type), counter::Base.RefValue{Int} = Ref(0))
    if counter[] > 127
        return false
    end
    if ty isa Union
        allunbox = for_each_uniontype_small(f, ty.a, counter)
        allunbox &= for_each_uniontype_small(f, ty.b, counter)
        return allunbox
    end
    # https://github.com/JuliaLang/julia/blob/170d6439445c86e640214620dad3423d2bb42337/src/codegen.cpp#L1233
    if Base.isconcretetype(ty) && !ismutabletype(ty) && Base.datatype_pointerfree(ty)
        counter[] += 1
        f(ty)
        return true
    end
    return false
end

# From https://github.com/JuliaLang/julia/blob/038d31463f0ef744c8308bdbe87339b9c3f0b890/src/cgutils.cpp#L3108
function union_alloca_type(@nospecialize(UT::Type))
    nbytes = 0
    function inner(@nospecialize(jlrettype::Type))
        if !(Base.issingletontype(jlrettype) && isa(jlrettype, DataType))
            nbytes = max(nbytes, sizeof(jlrettype))
        end
    end
    for_each_uniontype_small(inner, UT)
    return nbytes
end

# From https://github.com/JuliaLang/julia/blob/e6bf81f39a202eedc7bd4f310c1ab60b5b86c251/src/codegen.cpp#L6447
function is_sret(@nospecialize(jlrettype::Type))
    if jlrettype === Union{}
        # jlrettype == (jl_value_t*)jl_bottom_type
        return false
    elseif Base.isstructtype(jlrettype) &&
           Base.issingletontype(jlrettype) &&
           isa(jlrettype, DataType)
        # jl_is_structtype(jlrettype) && jl_is_datatype_singleton((jl_datatype_t*)jlrettype)
        return false
    elseif jlrettype isa Union # jl_is_uniontype(jlrettype)
        if union_alloca_type(jlrettype) > 0
            # sret, also a regular return here
            return true
        end
        return false
    elseif !GPUCompiler.deserves_retbox(jlrettype)
        rt = convert(LLVMType, jlrettype)
        if !isa(rt, LLVM.VoidType) && GPUCompiler.deserves_sret(jlrettype, rt)
            return true
        end
    end
    return false
end
function is_sret_union(@nospecialize(jlrettype::Type))
    if jlrettype === Union{}
        # jlrettype == (jl_value_t*)jl_bottom_type
        return false
    elseif Base.isstructtype(jlrettype) &&
           Base.issingletontype(jlrettype) &&
           isa(jlrettype, DataType)
        # jl_is_structtype(jlrettype) && jl_is_datatype_singleton((jl_datatype_t*)jlrettype)
        return false
    elseif jlrettype isa Union # jl_is_uniontype(jlrettype)
        if union_alloca_type(jlrettype) > 0
            # sret, also a regular return here
            return true
        end
    end
    return false
end

# https://github.com/JuliaLang/julia/blob/0a696a3842750fcedca8832bc0aabe9096c7658f/src/codegen.cpp#L6812
function get_return_info(
    @nospecialize(jlrettype::Type),
)::Tuple{Union{Nothing,Type},Union{Nothing,Type},Union{Nothing,Type}}
    sret = nothing
    returnRoots = nothing
    rt = nothing
    if jlrettype === Union{}
        rt = Nothing
    elseif Base.isstructtype(jlrettype) &&
           Base.issingletontype(jlrettype) &&
           isa(jlrettype, DataType)
        rt = Nothing
    elseif jlrettype isa Union
        nbytes = 0
        allunbox = for_each_uniontype_small(jlrettype) do jlrettype
            if !(Base.issingletontype(jlrettype) && isa(jlrettype, DataType))
                nbytes = max(nbytes, sizeof(jlrettype))
            end
        end
        if nbytes != 0
            rt = NamedTuple{(Symbol("1"), Symbol("2")),Tuple{Any,UInt8}}
            # Pointer to?, Ptr{NTuple{UInt8, allunbox}
            sret = Ptr{jlrettype}
        elseif allunbox
            rt = UInt8
        else
            rt = Any
        end
    elseif jlrettype <: Tuple && in(Any, jlrettype.parameters)
        rt = Any
    elseif !GPUCompiler.deserves_retbox(jlrettype)
        lRT = convert(LLVMType, jlrettype)
        if !isa(lRT, LLVM.VoidType) && GPUCompiler.deserves_sret(jlrettype, lRT)
            sret = Ptr{jlrettype}
            tracked = CountTrackedPointers(lRT)
            @assert !tracked.derived
            if tracked.count != 0 && !tracked.all
                returnRoots = Ptr{AnyArray(Int(tracked.count))}
            end
        else
            rt = jlrettype
        end
    else
        # retbox
        rt = Ptr{jlrettype}
    end

    return (rt, sret, returnRoots)
end

# From https://github.com/JuliaLang/julia/blob/81813164963f38dcd779d65ecd222fad8d7ed437/src/cgutils.cpp#L570
@inline function isghostty(@nospecialize(ty))
    if ty === Union{}
        return true
    end
    if Base.isconcretetype(ty) && !ismutabletype(ty)
        if sizeof(ty) == 0
            return true
        end
        # TODO consider struct_to_llvm ?
    end
    return false
end

struct Tape{TapeTy,ShadowTy,ResT}
    internal_tape::TapeTy
    shadow_return::ShadowTy
end


@inline any_jltypes(::Type{Nothing}) = false
@inline any_jltypes(::Type{T}) where {T<:AbstractFloat} = false
@inline any_jltypes(::Type{T}) where {T<:Integer} = false
@inline any_jltypes(::Type{Complex{T}}) where {T} = any_jltypes(T)
@inline any_jltypes(::Type{Tuple{}}) = false
@inline any_jltypes(::Type{NTuple{Size,T}}) where {Size,T} = any_jltypes(T)
@inline any_jltypes(::Type{Core.LLVMPtr{T,Addr}}) where {T,Addr} = 10 <= Addr <= 12
@inline any_jltypes(::Type{Any}) = true
@inline any_jltypes(::Type{NamedTuple{A,B}}) where {A,B} =
    any(any_jltypes(b) for b in B.parameters)
@inline any_jltypes(::Type{T}) where {T<:Tuple} = any(any_jltypes(b) for b in T.parameters)

const WideIntWidths = [256, 512, 1024, 2048]

let
    for n ∈ WideIntWidths
        let T = Symbol(:UInt, n)
            eval(quote
                primitive type $T <: Unsigned $n end
            end)
        end
    end
end

function jl_set_typeof(v::Ptr{Cvoid}, @nospecialize(T::Type))
    tag = reinterpret(Ptr{Any}, reinterpret(UInt, v) - 8)
    Base.unsafe_store!(tag, T) # set tag
    return nothing
end

@generated function splatnew(::Type{T}, args::TT) where {T,TT<:Tuple}
    return quote
        Base.@_inline_meta
        $(Expr(:splatnew, :T, :args))
    end
end

@inline remove_innerty(::Type{<:Const}) = Const
@inline remove_innerty(::Type{<:Active}) = Active
@inline remove_innerty(::Type{<:Duplicated}) = Duplicated
@inline remove_innerty(::Type{<:DuplicatedNoNeed}) = DuplicatedNoNeed
@inline remove_innerty(::Type{<:BatchDuplicated}) = Duplicated
@inline remove_innerty(::Type{<:BatchDuplicatedNoNeed}) = DuplicatedNoNeed
@inline remove_innerty(::Type{<:MixedDuplicated}) = MixedDuplicated
@inline remove_innerty(::Type{<:BatchMixedDuplicated}) = MixedDuplicated
