function isSpecialPtr(@nospecialize(Ty::LLVM.LLVMType))
    if !isa(Ty, LLVM.PointerType)
        return false
    end
    AS = LLVM.addrspace(Ty)
    return 10 <= AS && AS <= 13
end

mutable struct CountTrackedPointers
    count::UInt
    all::Bool
    derived::Bool
end

function CountTrackedPointers(@nospecialize(T::LLVM.LLVMType))
    res = CountTrackedPointers(0, true, false)

    if isa(T, LLVM.PointerType)
        if isSpecialPtr(T)
            res.count += 1
            if LLVM.addrspace(T) != Tracked
                res.derived = true
            end
        end
    elseif isa(T, LLVM.StructType)
        for ElT in elements(T)
            sub = CountTrackedPointers(ElT)
            res.count += sub.count
            res.all &= sub.all
            res.derived |= sub.derived
        end
    elseif isa(T, LLVM.ArrayType) || isa(T, LLVM.VectorType)
        sub = CountTrackedPointers(eltype(T))
        res.count += sub.count
        res.all &= sub.all
        res.derived |= sub.derived
        res.count *= length(T)
    end
    if res.count == 0
        res.all = false
    end
    return res
end

# must deserve sret
function deserves_rooting(@nospecialize(T::LLVM.LLVMType))
    tracked = CountTrackedPointers(T)
    @assert !tracked.derived
    if tracked.count != 0 && !tracked.all
        return true # tracked.count;
    end
    return false
end


function any_jltypes(Type::LLVM.PointerType)
    if 10 <= LLVM.addrspace(Type) <= 12
        return true
    else
        # do we care about {} addrspace(11)**
        return false
    end
end

any_jltypes(Type::LLVM.StructType) = any(any_jltypes, LLVM.elements(Type))
any_jltypes(Type::Union{LLVM.VectorType,LLVM.ArrayType}) = any_jltypes(eltype(Type))
any_jltypes(::LLVM.IntegerType) = false
any_jltypes(::LLVM.FloatingPointType) = false
any_jltypes(::LLVM.VoidType) = false

nfields(Type::LLVM.StructType) = length(LLVM.elements(Type))
nfields(Type::LLVM.VectorType) = size(Type)
nfields(Type::LLVM.ArrayType) = length(Type)
nfields(Type::LLVM.PointerType) = 1

function store_nonjl_types!(B::LLVM.IRBuilder, @nospecialize(startval::LLVM.Value), @nospecialize(p::LLVM.Value))
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    vals = LLVM.Value[]
    if p != nothing
        push!(vals, p)
    end
    todo = Tuple{Tuple,LLVM.Value}[((), startval)]
    while length(todo) != 0
        path, cur = popfirst!(todo)
        ty = value_type(cur)
        if isa(ty, LLVM.PointerType)
            if any_jltypes(ty)
                continue
            end
        end
        if isa(ty, LLVM.ArrayType)
            if any_jltypes(ty)
                for i = 1:length(ty)
                    ev = extract_value!(B, cur, i - 1)
                    push!(todo, ((path..., i - 1), ev))
                end
                continue
            end
        end
        if isa(ty, LLVM.StructType)
            if any_jltypes(ty)
                for (i, t) in enumerate(LLVM.elements(ty))
                    ev = extract_value!(B, cur, i - 1)
                    push!(todo, ((path..., i - 1), ev))
                end
                continue
            end
        end
        parray = LLVM.Value[LLVM.ConstantInt(LLVM.IntType(64), 0)]
        for v in path
            push!(parray, LLVM.ConstantInt(LLVM.IntType(32), v))
        end
        gptr = gep!(B, value_type(startval), p, parray)
        st = store!(B, cur, gptr)
    end
    return
end

function get_julia_inner_types(B::LLVM.IRBuilder, @nospecialize(p::Union{Nothing, LLVM.Value}), @nospecialize(startvals::Vararg{LLVM.Value}); added = LLVM.API.LLVMValueRef[])
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    vals = LLVM.Value[]
    if p != nothing
        push!(vals, p)
    end
    todo = LLVM.Value[startvals...]
    while length(todo) != 0
        cur = popfirst!(todo)
        ty = value_type(cur)
        if isa(ty, LLVM.PointerType)
            if any_jltypes(ty)
                if addrspace(ty) != Tracked
                    cur = addrspacecast!(
                        B,
                        cur,
                        LLVM.PointerType(eltype(ty), Tracked),
                        LLVM.name(cur) * ".innertracked",
                    )
                    if isa(cur, LLVM.Instruction)
                        push!(added, cur.ref)
                    end
                end
                if value_type(cur) != T_prjlvalue
                    cur = bitcast!(B, cur, T_prjlvalue)
                    if isa(cur, LLVM.Instruction)
                        push!(added, cur.ref)
                    end
                end
                push!(vals, cur)
            end
            continue
        end
        if isa(ty, LLVM.ArrayType)
            if any_jltypes(ty)
                for i = 1:length(ty)
                    ev = extract_value!(B, cur, i - 1)
                    if isa(ev, LLVM.Instruction)
                        push!(added, ev.ref)
                    end
                    push!(todo, ev)
                end
            end
            continue
        end
        if isa(ty, LLVM.StructType)
            for (i, t) in enumerate(LLVM.elements(ty))
                if any_jltypes(t)
                    ev = extract_value!(B, cur, i - 1)
                    if isa(ev, LLVM.Instruction)
                        push!(added, ev.ref)
                    end
                    push!(todo, ev)
                end
            end
            continue
        end
        if isa(ty, LLVM.IntegerType)
            continue
        end
        if isa(ty, LLVM.FloatingPointType)
            continue
        end
        msg = sprint() do io
            println(io, "Enzyme illegal subtype")
            println(io, "ty=", ty)
            println(io, "cur=", cur)
            println(io, "p=", p)
            println(io, "startvals=", startvals)
        end
        throw(AssertionError(msg))
    end
    return vals
end
