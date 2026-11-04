using Test, Enzyme

# The buffer-ownership walk and slice reuse, expressed using Base only.
struct BufferGroup
    slots::Vector{Base.RefValue{Any}}
end
function group_references(result, group::BufferGroup)
    for slot in group.slots
        held = slot[]
        held === nothing || !may_reference(result, held) || return true
    end
    return false
end
may_reference(result, @nospecialize(buffer)) = true
may_reference(::Number, @nospecialize(buffer)) = false
function may_reference(result::AbstractArray, @nospecialize(buffer))
    buffer isa BufferGroup && return group_references(result, buffer)
    mightalias(result, buffer) && return true
    isbitstype(eltype(result)) && return false
    for item in result
        may_reference(item, buffer) && return true
    end
    return false
end
may_reference(result::NamedTuple, @nospecialize(buffer)) = may_reference(values(result), buffer)
may_reference(::Tuple{}, @nospecialize(buffer)) = false
@inline may_reference(result::Tuple, @nospecialize(buffer)) =
    may_reference(first(result), buffer) || may_reference(Base.tail(result), buffer)
@inline mightalias(result::A, @nospecialize(buffer)) where {A <: AbstractArray} =
    buffer isa A ? Base.mightalias(result, buffer::A) :
    buffer isa Array ? Base.mightalias(result, buffer::Array) :
    buffer isa AbstractArray && Base.mightalias(result, buffer)

@inline function release!(result, slot)
    buffer = slot[]
    buffer === nothing || !may_reference(result, buffer) || (slot[] = nothing)
    return result
end
@inline function release_direct!(result, slot)
    buffer = slot[]
    buffer === nothing || !Base.mightalias(result.tail, buffer) || (slot[] = nothing)
    return result
end
@inline function slice!(slot, x)
    old = slot[]
    indices = 2:lastindex(x)
    return if old isa Vector{eltype(x)} && axes(old) == (Base.OneTo(length(indices)),)
        copyto!(old, view(x, indices))
    else
        slot[] = x[indices]
    end
end

function batch(x, release)
    slot = Base.RefValue{Any}(nothing)
    tails = similar(x, size(x, 1) - 1, size(x, 2))
    totals = similar(x, size(x, 2))
    for j in axes(x, 2)
        column = view(x, :, j)
        result = (; tail = slice!(slot, column), total = sum(abs2, column))
        release(result, slot)
        tails[:, j] = result.tail
        totals[j] = result.total
    end
    return (; tail = tails, total = totals)
end
loss(x, release) = sum(batch(x, release).total)

@testset "Recursive buffer ownership" begin
    for release in (release!, release_direct!), width in (1, 4), columns in (0, 1, 4)
        x = reshape(collect(1.0:(width * columns)), width, columns)
        @test batch(x, release) == (; tail = x[2:end, :], total = vec(sum(abs2, x; dims = 1)))
        dx = zero(x)
        _, primal = Enzyme.autodiff(
            Enzyme.ReverseWithPrimal, loss, Enzyme.Active,
            Enzyme.Duplicated(x, dx), Enzyme.Const(release)
        )
        @test primal == sum(abs2, x)
        @test dx == 2x
    end

    x = [1.0, 2.0, 3.0]
    slot = Base.RefValue{Any}(nothing)
    tail = slice!(slot, x)
    @test tail == x[2:end]
    # A non-aliasing result preserves the buffer for reuse.
    @test release!((; total = sum(x)), slot) == (; total = sum(x))
    @test slot[] === tail
    @test slice!(slot, 2x) === tail
    @test tail == 2x[2:end]
    release!((; tail = view(tail, :)), slot)
    @test slot[] === nothing

    # The generic recursive arm must remain when a group can really occur.
    group = BufferGroup([Base.RefValue{Any}(nothing), Base.RefValue{Any}(x)])
    nested = BufferGroup([Base.RefValue{Any}(group)])
    @test may_reference((; tail = view(x, 2:3)), nested)
    @test !may_reference((; tail = copy(x)), nested)
    @test may_reference(Any[view(x, :)], nested)
    @test !may_reference((1, 2.0), nested)
    group_slot = Base.RefValue{Any}(nested)
    release!((; tail = copy(x)), group_slot)
    @test group_slot[] === nested
    release!((; tail = x), group_slot)
    @test group_slot[] === nothing
end
