"""
    pick_batchsize(totalsize, mode, ftype, return_activity, argtypes...)

Return a reasonable batch size for batched differentiation.

!!! warning
    This function is experimental, and not part of the public API.
"""
function pick_batchsize(totalsize::Integer,
                        mode::Mode,
                        ftype::Type,
                        return_activity, ::Type{<:Annotation},
                        argtypes::Vararg{Type{<:Annotation},Nargs}) where {Nargs}
    return min(totalsize, 16)
end
