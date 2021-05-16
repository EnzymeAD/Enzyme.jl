using Cassette

Cassette.@context EnzymeCtx

###
# Cassette fixes
###
@inline Cassette.overdub(::EnzymeCtx, ::typeof(Core.kwfunc), f) = return Core.kwfunc(f)
@inline Cassette.overdub(::EnzymeCtx, ::typeof(Core.apply_type), args...) = return Core.apply_type(args...)

# TODO the following should be handled and tested, but as is don't function entirely @vchuravy
# @inline Cassette.overdub(::EnzymeCtx, ::typeof(Base.OverflowError), args...) = return Base.OverflowError(args...)
# @inline Cassette.overdub(::EnzymeCtx, ::typeof(Base.DomainError), args...) = return Base.DomainError(args...)
# @inline Cassette.overdub(::EnzymeCtx, ::typeof(StaticArrays.Size), x::Type{<:AbstractArray{<:Any, N}}) where {N} = return StaticArrays.Size(x)

# @inline function Cassette.overdub(::EnzymeCtx, ::typeof(Base.Math.nan_dom_err), out, x)
#     isnan(out) & !isnan(x) ? error("NaN result for non-NaN input.") : out
# end

@inline function Cassette.overdub(::EnzymeCtx, ::typeof(Base.factorial_lookup), n, table, lim)
    n < 0 && error("n must not be negative")
    n > lim && error("n is to large")
    n == 0 && return one(n)
    @inbounds f = table[n]
    return oftype(n, f)
end

@inline function Cassette.overdub(::EnzymeCtx, ::typeof(Base.Checked.throw_overflowerr_binaryop), op, x, y)
    Base.@_inline_meta
    error("overflowed for type")
    # throw(OverflowError(Base.invokelatest(string, x, " ", op, " ", y, " overflowed for type ", typeof(x)))))
end

@inline function Cassette.overdub(::EnzymeCtx, ::typeof(Base.Checked.checked_add), x, y)
    Base.@_inline_meta
    z, b = Base.Checked.add_with_overflow(x, y)
    b && error("overflowed + for type")
    z
end

@inline function Cassette.overdub(::EnzymeCtx, ::typeof(Base.Checked.checked_sub), x, y)
    Base.@_inline_meta
    z, b = Base.Checked.sub_with_overflow(x, y)
    b && error("overflowed - for type")
    z
end

@inline function Cassette.overdub(::EnzymeCtx, ::typeof(Base.Checked.checked_mul), x, y)
    Base.@_inline_meta
    z, b = Base.Checked.mul_with_overflow(x, y)
    b && error("overflowed * for type")
    z
end

function ir_element(x, code::Vector)
    while isa(x, Core.SSAValue)
        x = code[x.id]
    end
    return x
end

@inline function Base.copy(a::Array{T,1}) where {T}
    len = UInt64(Base.length(a))
    typesize = UInt64(Base.aligned_sizeof(T))
    bitsunion = Base.isbitsunion(T)
  
    mod = "
  declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)
  declare {} addrspace(10)* @jl_alloc_array_1d({} addrspace(10)*, i64)
  define internal i8* @jl_array_ptr({} addrspace(10)* %a) #0 {
  top:
    %bc = bitcast {} addrspace(10)* %a to i8* addrspace(10)*
    %ld = load i8*, i8* addrspace(10)* %bc
    ret i8* %ld
  }
  
  declare i8* @jl_array_typetagdata({} addrspace(10)*)
  define {} addrspace(10)* @entry({} addrspace(10)* %typ, {} addrspace(10)* %a, i64 %len, i64 %tsize, i8 %bitsunion) #0 {
  top:
    %data = call nonnull {} addrspace(10)* @jl_alloc_array_1d({} addrspace(10)* %typ, i64 %len)
    %src = call i8* @jl_array_ptr({} addrspace(10)* %a)
    %dst = call i8* @jl_array_ptr({} addrspace(10)* %data)
    %size = mul i64 %len, %tsize
    tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 %size, i1 false)
    %cmp = icmp eq i8 %bitsunion, 0
    br i1 %cmp, label %end, label %bu
  
  bu:
    %dst2 = call i8* @jl_array_typetagdata({} addrspace(10)* %data)
    %src2 = call i8* @jl_array_typetagdata({} addrspace(10)* %a)
    tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst2, i8* %src2, i64 %size, i1 false)
    br label %end
  
  end:
    ret {} addrspace(10)* %data
    }
    attributes #0 = { alwaysinline }"

    Base.llvmcall((mod, "entry"), Array{T,1}, Tuple{Any, Any, UInt64, UInt64, Bool}, Array{T, 1}, a, len, typesize, bitsunion)
end

##
# Forces inlining on everything that is not marked `@noinline`
# avoids overdubbing of pure functions
# avoids overdubbing of IntrinsicFunctions and Builtins 
##
function transform(ctx, ref)
    CI = ref.code_info
    noinline = any(@nospecialize(x) ->
                       Core.Compiler.isexpr(x, :meta) &&
                       x.args[1] == :noinline,
                   CI.code)
    CI.inlineable = !noinline

    # don't overdub pure functions
    if CI.pure
        n_method_args = Int(ref.method.nargs)
        if ref.method.isva
            Cassette.insert_statements!(CI.code, CI.codelocs,
                (x, i) -> i == 1 ?  3 : nothing,
                (x, i) -> i == 1 ? [
                    # this could run into troubles when the function is @pure f(x...) since then n_method_args==2, but this seems to work sofar.
                    Expr(:call, Expr(:nooverdub, GlobalRef(Core, :tuple)), (Core.SlotNumber(i) for i in 2:(n_method_args-1))...),
                    Expr(:call, Expr(:nooverdub, GlobalRef(Core, :_apply)), Core.SlotNumber(1), Core.SSAValue(i), Core.SlotNumber(n_method_args)),
                    ReturnNode(Core.SSAValue(i+1))] : nothing)
        else
            Cassette.insert_statements!(CI.code, CI.codelocs,
                (x, i) -> i == 1 ?  2 : nothing,
                (x, i) -> i == 1 ? [
                    Expr(:call, Expr(:nooverdub, Core.SlotNumber(1)), (Core.SlotNumber(i) for i in 2:n_method_args)...)
                    ReturnNode(Core.SSAValue(i))] : nothing)
        end
        CI.ssavaluetypes = length(CI.code)
        return CI
    end

    # overdubbing IntrinsicFunctions removes our ability to profile code
    newstmt = (x, i) -> begin
        isassign = Base.Meta.isexpr(x, :(=))
        stmt = isassign ? x.args[2] : x
        if Base.Meta.isexpr(stmt, :call)
            applycall = Cassette.is_ir_element(stmt.args[1], GlobalRef(Core, :_apply), CI.code) 
            applyitercall = Cassette.is_ir_element(stmt.args[1], GlobalRef(Core, :_apply_iterate), CI.code) 
            if applycall
                fidx = 2
            elseif applyitercall
                fidx = 3
            else
                fidx = 1
            end
            f = stmt.args[fidx]
            f = ir_element(f, CI.code)
            if f isa GlobalRef
                mod = f.mod
                name = f.name
                if Base.isbindingresolved(mod, name) && Base.isdefined(mod, name)
                    ff = getfield(f.mod, f.name)
                    if ff isa Core.IntrinsicFunction || ff isa Core.Builtin
                        stmt.args[fidx] = Expr(:nooverdub, f)
                    end
                end
            end
        end
        return [x]
    end

    Cassette.insert_statements!(CI.code, CI.codelocs, (x, i) -> 1, newstmt)
    CI.ssavaluetypes = length(CI.code)
    # Core.Compiler.validate_code(CI)
    return CI
end

const CompilerPass = Cassette.@pass transform
const CTX = Cassette.disablehooks(EnzymeCtx(pass = CompilerPass))
