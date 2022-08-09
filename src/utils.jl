function hasfieldcount(@nospecialize(dt))
    try
        fieldcount(dt)
    catch
        return false
    end
    return true
end

if VERSION <= v"1.6"
    allocatedinline(@nospecialize(T)) = T.isinlinealloc
else
    import Base: allocatedinline
end

AnonymousStruct(::Type{U}) where U<:Tuple = NamedTuple{ntuple(i->Symbol(i), Val(length(U.parameters))), U}
