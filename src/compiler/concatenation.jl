# Visiting tuple operands through first/tail keeps each operand's activity
# separate. Iterating the tuple merges constant and differentiable arrays.
@inline copy_hcat!(out, offset, ::Tuple{}) = out

@inline function copy_hcat!(out, offset, arrays::Tuple)
    a = first(arrays)
    n = length(a)
    copy_vector_block!(out, offset, a)
    return copy_hcat!(out, offset + n, Base.tail(arrays))
end

concat_rows(a) = size(a, 1)
concat_cols(a) = size(a, 2)

# An `Array` operand fills an `Array` output as contiguous runs. A view, reshape,
# transpose or adjoint of one takes the block assignment Base's own
# concatenation uses for it. The output is fresh either way.
@inline copy_vector_block!(out::Array, offset, a::Array) = copyto!(out, offset, a, 1, length(a))

@inline function copy_vector_block!(out, offset, a)
    out[offset:(offset + length(a) - 1)] = a
    return out
end

@inline copy_block!(out::Array, row, col, a::Array) = copy_columns!(out, row, col, a)

@inline function copy_block!(out, row, col, a)
    out[row:(row + concat_rows(a) - 1), col:(col + concat_cols(a) - 1)] = a
    return out
end

# A full-height `Array` block is a single run.
@inline copy_hcat_block!(out::Array, col, a::Array) =
    copyto!(out, (col - 1) * size(out, 1) + 1, a, 1, length(a))
@inline copy_hcat_block!(out, col, a) = copy_block!(out, 1, col, a)

@inline copy_hcat_columns!(out, col, ::Tuple{}) = out

@inline function copy_hcat_columns!(out, col, arrays::Tuple)
    a = first(arrays)
    copy_hcat_block!(out, col, a)
    return copy_hcat_columns!(out, col + concat_cols(a), Base.tail(arrays))
end

@inline function override_hcat(arrays::Vector{T}...) where {T}
    height = length(first(arrays))
    all(==(height), map(length, arrays)) ||
        throw(DimensionMismatch("vectors must have same lengths"))
    return copy_hcat!(Matrix{T}(undef, height, length(arrays)), 1, arrays)
end

@inline function override_typed_hcat(::Type{T}, arrays::Tuple) where {T}
    heights = map(concat_rows, arrays)
    height = first(heights)
    all(==(height), heights) || throw(
        DimensionMismatch(
            "number of rows of each array must match (got $heights)",
        )
    )
    out = similar(first(arrays), T, height, concat_sum(concat_cols, arrays))
    return copy_hcat_columns!(out, 1, arrays)
end

@inline copy_matrix_vcat!(out, row, ::Tuple{}) = out

@inline function copy_columns!(out, row, col, a)
    height = size(a, 1)
    for j in 1:size(a, 2)
        copyto!(out, (col + j - 2) * size(out, 1) + row, a, (j - 1) * height + 1, height)
    end
    return out
end

@inline function copy_matrix_vcat!(out, row, arrays::Tuple)
    a = first(arrays)
    copy_block!(out, row, 1, a)
    return copy_matrix_vcat!(out, row + size(a, 1), Base.tail(arrays))
end

@inline function override_typed_vcat(::Type{T}, arrays::Tuple{Vararg{AbstractVector}}) where {T}
    out = similar(first(arrays), T, concat_sum(length, arrays))
    return copy_hcat!(out, 1, arrays)
end

@inline function override_typed_vcat(::Type{T}, arrays::Tuple) where {T}
    widths = map(concat_cols, arrays)
    width = first(widths)
    all(==(width), widths) || throw(
        DimensionMismatch(
            "number of columns of each array must match (got $widths)",
        )
    )
    out = similar(first(arrays), T, concat_sum(concat_rows, arrays), width)
    return copy_matrix_vcat!(out, 1, arrays)
end

const DENSE_HCAT_METHOD = which(Base.hcat, Tuple{Vararg{Vector{T}}} where {T})
const DENSE_TYPED_HCAT_METHOD = which(Base._typed_hcat, Tuple{Type{Float64}, Tuple{Matrix{Float64}, Vector{Float64}}})
const DENSE_TYPED_VCAT_METHOD = which(Base._typed_vcat, Tuple{Type{Float64}, Tuple{Matrix{Float64}, Vector{Float64}}})
const DENSE_TYPED_VECTOR_VCAT_METHOD = which(Base._typed_vcat, Tuple{Type{Float64}, Tuple{Vector{Float64}, Vector{Float64}}})

import LinearAlgebra: Adjoint, Transpose

# The overrides take `Array`s and the views, reshapes, transposes and adjoints
# of them. Any other array type (GPU, sparse, static, offset or user-defined
# arrays) keeps Base's concatenation.
concat_operand(::Type) = false
concat_operand(::Type{<:Array}) = true
concat_operand(::Type{<:SubArray{<:Any, <:Any, P}}) where {P} = concat_operand(P)
concat_operand(::Type{<:Base.ReshapedArray{<:Any, <:Any, P}}) where {P} = concat_operand(P)
concat_operand(::Type{<:Transpose{<:Any, P}}) where {P} = concat_operand(P)
concat_operand(::Type{<:Adjoint{<:Any, P}}) where {P} = concat_operand(P)

concat_operands(types) = all(concat_operand, types)

concat_replacement(f, sig) = nothing

function concat_replacement(::typeof(Base.vcat), sig)
    sig <: (Tuple{typeof(Base.vcat), Vararg{Vector{T}}} where {T}) || return nothing
    return DENSE_VCAT_METHOD, override_vcat
end

function concat_replacement(::typeof(Base.hcat), sig)
    sig <: (Tuple{typeof(Base.hcat), Vararg{Vector{T}}} where {T}) || return nothing
    return DENSE_HCAT_METHOD, override_hcat
end

function concat_replacement(::typeof(Base._typed_hcat), sig)
    sig <: Tuple{typeof(Base._typed_hcat), Type, Tuple{Vararg{AbstractVecOrMat}}} || return nothing
    concat_operands(fieldtypes(fieldtypes(sig)[3])) || return nothing
    return DENSE_TYPED_HCAT_METHOD, override_typed_hcat
end

function concat_replacement(::typeof(Base._typed_vcat), sig)
    sig <: Tuple{typeof(Base._typed_vcat), Type, Tuple{Vararg{AbstractVecOrMat}}} || return nothing
    concat_operands(fieldtypes(fieldtypes(sig)[3])) || return nothing
    method = sig <: Tuple{typeof(Base._typed_vcat), Type, Tuple{Vararg{AbstractVector}}} ?
        DENSE_TYPED_VECTOR_VCAT_METHOD : DENSE_TYPED_VCAT_METHOD
    return method, override_typed_vcat
end

function hvcat_layout(rows, heights, widths)
    @static if VERSION >= v"1.12-"
        concat_sum(identity, rows) == length(heights) || throw(
            ArgumentError(
                "argument count does not match specified block shape",
            )
        )
    end
    width = 0
    for k in 1:rows[1]
        width += widths[k]
    end
    height = 0
    a = 1
    for n in rows
        height += heights[a]
        a += n
    end
    offsets = Vector{Tuple{Int, Int}}(undef, length(heights))
    a, r = 1, 1
    for i in eachindex(rows)
        c = 1
        h = heights[a]
        for j in 1:rows[i]
            k = a + j - 1
            heights[k] == h || throw(
                DimensionMismatch(
                    "mismatched height in block row $i (expected $h, got $(heights[k]))",
                )
            )
            c - 1 + widths[k] <= width || throw(
                DimensionMismatch(
                    "block row $i has mismatched number of columns (expected $width, got $(c - 1 + widths[k]))",
                )
            )
            offsets[k] = (r, c)
            c += widths[k]
        end
        c == width + 1 || throw(
            DimensionMismatch(
                "block row $i has mismatched number of columns (expected $width, got $(c - 1))",
            )
        )
        r += h
        a += rows[i]
    end
    resize!(offsets, max(0, concat_sum(identity, rows)))
    return height, width, offsets
end

@inline copy_hvcat!(out, offsets, i, ::Tuple{}) = out

@inline function copy_hvcat!(out, offsets, i, arrays::Tuple)
    i > length(offsets) && return out
    a = first(arrays)
    row, col = offsets[i]
    copy_block!(out, row, col, a)
    return copy_hvcat!(out, offsets, i + 1, Base.tail(arrays))
end

@inline function override_typed_hvcat(::Type{T}, rows::Tuple, arrays...) where {T}
    height, width, offsets = hvcat_layout(rows, map(concat_rows, arrays), map(concat_cols, arrays))
    out = similar(first(arrays), T, height, width)
    return copy_hvcat!(out, offsets, 1, arrays)
end

@inline copy_stack!(out, offset, ax, ::Tuple{}) = out

@inline function copy_stack!(out, offset, ax, arrays::Tuple)
    a = first(arrays)
    Base._stack_size_check(a, ax)
    copyto!(out, offset, a)
    return copy_stack!(out, offset + length(a), ax, Base.tail(arrays))
end

@inline function override_typed_stack(::Colon, ::Type{T}, ::Type{S}, arrays::Tuple, ax) where {T, S}
    isempty(arrays) && return Base._empty_stack(:, T, S, arrays)
    a = first(arrays)
    out = similar(a, T, axes(a)..., ax...)
    return copy_stack!(out, firstindex(out), axes(a), arrays)
end

@inline function override_stack(dims, arrays::Tuple)
    T = promote_type(map(eltype, arrays)...)
    S = Base.promote_typejoin_union(eltype(arrays))
    if dims isa Colon
        return override_typed_stack(dims, T, S, arrays, Base._iterator_axes(arrays))
    end
    return override_dim_stack(dims, T, S, arrays)
end

@inline override_stack_default(arrays::Tuple) = override_stack(:, arrays)
@inline override_kw_stack(kws, ::typeof(Base.stack), arrays::Tuple) = override_stack(kws.dims, arrays)

Base.@constprop :aggressive @inline override_cat(dims, arrays...) =
    override_cat_t(dims, Base.promote_eltypeof(arrays...), arrays...)

@inline copy_dim_stack!(out, dim, i, ax, ::Tuple{}) = out

@inline function copy_dim_stack!(out, dim, i, ax, arrays::Tuple)
    a = first(arrays)
    Base._stack_size_check(a, ax)
    copy_stack_block!(dim, out, i, a)
    return copy_dim_stack!(out, dim, i + 1, ax, Base.tail(arrays))
end

@inline function override_dim_stack!(::Val{D}, out, a, rest::Tuple) where {D}
    i = firstindex(out, D)
    copy_stack_block!(Val(D), out, i, a)
    copy_dim_stack!(out, Val(D), i + 1, axes(a), rest)
    return nothing
end

@generated function copy_stack_block!(::Val{D}, out::Array{T, N}, i, a::Array{S, M}) where {D, T, N, S, M}
    # The output is fresh, so copy directly instead of using a view's alias check.
    dest = :1
    stride = :1
    for d in 1:N
        source_dim = d - (d > D)
        coordinate = if d == D
            :(i - 1)
        elseif D == 1
            :(I[$source_dim] - 1)
        elseif source_dim > 1
            :(I[$(source_dim - 1)] - 1)
        else
            :0
        end
        dest = :($dest + $coordinate * $stride)
        stride = :($stride * size(out, $d))
    end
    outer_shape = D == 1 ? :(size(a)) : :(Base.tail(size(a)))
    height = D == 1 ? :1 : :(size(a, 1))
    return quote
        Base.@_inline_meta
        height = $height
        height == 0 && return out
        source = 1
        for I in CartesianIndices($outer_shape)
            copyto!(out, $dest, a, source, height)
            source += height
        end
        return out
    end
end

# Any other slice or output takes a block assignment into the output. Base's
# `_dim_stack!` copies into a view of it instead, and Enzyme differentiates
# `copyto!` from a view of Const data into a view wrongly: Reverse adds the
# gradient into the Const data and Forward returns it as the tangent.
@inline function copy_stack_block!(::Val{D}, out, i, a) where {D}
    before = ntuple(Returns(Colon()), Val(D - 1))
    after = ntuple(Returns(Colon()), Val(ndims(out) - D))
    out[before..., i, after...] = a
    return out
end

struct StackAxis{A, I, D}
    axes::A
    newaxis::I
    dim::D
end

@inline (s::StackAxis)(i) = i == s.dim ? s.newaxis : s.axes[i - (i > s.dim)]

@inline function override_dim_stack(dims, ::Type{T}, ::Type{S}, arrays::Tuple) where {T, S}
    isempty(arrays) && return Base._empty_stack(dims, T, S, arrays)
    a = first(arrays)
    n = ndims(a) + 1
    dims in 1:n || throw(
        ArgumentError(
            LazyString("cannot stack slices ndims(x) = ", n - 1, " along dims = ", dims),
        )
    )
    ax = ntuple(StackAxis(axes(a), Base.OneTo(length(arrays)), dims), Val(n))
    out = similar(a, T, ax...)
    override_dim_stack!(Val(dims), out, a, Base.tail(arrays))
    return out
end

@inline copy_cat!(out, shape, catdims, offsets, ::Tuple{}) = out

@generated function copy_cat_block!(out::Array{T, N}, offsets::NTuple{N, Int}, a::Array{S, M}) where {T, N, S, M}
    # Copy contiguous first-axis runs, with dimension arithmetic specialized on rank.
    dest = :1
    stride = :1
    for d in 1:N
        coordinate = 1 < d <= M ? :(I[$(d - 1)] - 1) : :0
        dest = :($dest + (offsets[$d] + $coordinate) * $stride)
        stride = :($stride * size(out, $d))
    end
    outer_shape = M == 0 ? :(()) : :(Base.tail(size(a)))
    return quote
        Base.@_inline_meta
        height = size(a, 1)
        height == 0 && return out
        source = 1
        for I in CartesianIndices($outer_shape)
            copyto!(out, $dest, a, source, height)
            source += height
        end
        return out
    end
end

@generated function cat_copy_offsets(catdims, offsets::NTuple{N, Int}, a) where {N}
    parts = Any[]
    for d in 1:N
        push!(parts, :($d <= length(catdims) && catdims[$d] ? offsets[$d] + size(a, $d) : offsets[$d]))
    end
    return Expr(:tuple, parts...)
end

@inline function copy_cat!(out, shape, catdims, offsets, arrays::Tuple)
    a = first(arrays)
    copy_cat_block!(out, offsets, a)
    next = cat_copy_offsets(catdims, offsets, a)
    return copy_cat!(out, shape, catdims, next, Base.tail(arrays))
end

@inline cat_copy_shape(catdims, shape, ::Tuple{}) = shape

@inline function cat_copy_shape(catdims, shape, sizes::Tuple)
    next = Base._cshp(1, catdims, shape, first(sizes))
    return cat_copy_shape(catdims, next, Base.tail(sizes))
end

Base.@constprop :aggressive @inline function override_cat_t(dims, ::Type{T}, arrays...) where {T}
    catdims = Base.dims2cat(dims)
    shape = cat_copy_shape(catdims, (), map(size, arrays))
    out = Base.cat_similar(first(arrays), T, shape)
    if count(!iszero, catdims)::Int > 1
        fill!(out, zero(T))
    end
    return copy_cat!(out, shape, catdims, ntuple(zero, length(shape)), arrays)
end

const DENSE_TYPED_HVCAT_METHOD = which(Base.typed_hvcat, Tuple{Type{Float64}, Tuple{Int, Int}, Vector{Float64}, Vector{Float64}})
const DENSE_STACK_METHOD = which(Base.stack, Tuple{Tuple{Vector{Float64}, Vector{Float64}}})
const DENSE_KW_STACK_METHOD = which(Core.kwcall, Tuple{NamedTuple{(:dims,), Tuple{Int}}, typeof(Base.stack), Tuple{Vector{Float64}, Vector{Float64}}})
const DENSE_CAT_METHOD = which(Base._cat_t, Tuple{Int, Type{Float64}, Vector{Float64}, Vector{Float64}})
const DENSE_PROMOTED_CAT_METHOD = which(Base._cat, Tuple{Int, Array{Float64, 3}, Array{Float64, 3}})
const DENSE_NUMERIC_CAT_METHOD = which(Base._cat, Tuple{Int, Vector{Float64}, Vector{Float64}})

function concat_replacement(::typeof(Base.typed_hvcat), sig)
    sig <: Tuple{typeof(Base.typed_hvcat), Type, Tuple{Vararg{Int}}, AbstractVecOrMat, Vararg{AbstractVecOrMat}} || return nothing
    concat_operands(fieldtypes(sig)[4:end]) || return nothing
    return DENSE_TYPED_HVCAT_METHOD, override_typed_hvcat
end

function concat_replacement(::typeof(Base.stack), sig)
    sig <: Tuple{typeof(Base.stack), Tuple{AbstractArray, Vararg{AbstractArray}}} || return nothing
    concat_operands(fieldtypes(fieldtypes(sig)[2])) || return nothing
    return DENSE_STACK_METHOD, override_stack_default
end

function concat_replacement(::typeof(Core.kwcall), sig)
    sig <: Tuple{typeof(Core.kwcall), NamedTuple{(:dims,), <:Tuple{Union{Integer, Colon}}}, typeof(Base.stack), Tuple{AbstractArray, Vararg{AbstractArray}}} || return nothing
    concat_operands(fieldtypes(fieldtypes(sig)[4])) || return nothing
    return DENSE_KW_STACK_METHOD, override_kw_stack
end

function concat_replacement(::typeof(Base._cat_t), sig)
    sig <: Tuple{typeof(Base._cat_t), Any, Type, Array, Vararg{Array}} || return nothing
    return DENSE_CAT_METHOD, override_cat_t
end

function concat_replacement(::typeof(Base._cat), sig)
    sig <: Tuple{typeof(Base._cat), Any, Array, Vararg{Array}} || return nothing
    method = sig <: Tuple{typeof(Base._cat), Any, VecOrMat{<:Number}, Vararg{VecOrMat{<:Number}}} ?
        DENSE_NUMERIC_CAT_METHOD : DENSE_PROMOTED_CAT_METHOD
    return method, override_cat
end

const DenseConcatFunction = Union{
    typeof(Base.vcat), typeof(Base.hcat), typeof(Base._typed_hcat), typeof(Base._typed_vcat),
    typeof(Base.typed_hvcat), typeof(Base.stack), typeof(Core.kwcall), typeof(Base._cat), typeof(Base._cat_t),
}
