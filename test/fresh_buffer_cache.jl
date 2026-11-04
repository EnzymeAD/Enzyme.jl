using Enzyme, Test

@noinline function fresh_buffer_distance(x, m)
    total = zero(m)
    for i in eachindex(x)
        total += (x[i] - m)^2
    end
    return total
end

function fresh_buffer_loss(x, m)
    distance = fresh_buffer_distance(x, m)
    scratch = zeros(length(x))
    for i in eachindex(x)
        scratch[i] = x[i] + m
    end
    return distance + sum(scratch)
end

function overwritten_buffer_loss(x, m)
    distance = fresh_buffer_distance(x, m)
    for i in eachindex(x)
        x[i] += 2.0
    end
    return distance + sum(x)
end

fresh_buffer_ir(f, argtypes) = sprint() do io
    Enzyme.Compiler.enzyme_code_llvm(
        io, f, Active, argtypes;
        dump_module = true, second_stage = false, debuginfo = :none
    )
end

# On Julia 1.10 a split callee rereads x in reverse when nothing after the call
# can overwrite x, so its augmented forward pass has no tape to return.
const tape_free_distance = r"define[^\n]*void @augmented_julia_fresh_buffer_distance_"

# `vec` of a fresh matrix after the call: Julia 1.10 fills a dims box and calls
# jl_reshape_array, which writes only the header it creates.
function reshaped_buffer_loss(x, m)
    distance = fresh_buffer_distance(x, m)
    scratch = zeros(length(x), 1)
    for i in eachindex(x)
        scratch[i, 1] = x[i] + m
    end
    return distance + sum(vec(scratch))
end

# The same with x carried by a structured Const operand, which reaches the
# compiled function as a field of an aggregate argument.
function structured_buffer_loss(data, m)
    distance = fresh_buffer_distance(data.x, m)
    scratch = zeros(length(data.x), 1)
    for i in eachindex(data.x)
        scratch[i, 1] = data.x[i] + m
    end
    return distance + sum(vec(scratch))
end

# A broadcast after the call that allocates its output on each of its branches
# and merges the headers before writing through them.
function broadcast_buffer_loss(x, m)
    distance = fresh_buffer_distance(x, m)
    a = zeros(length(x), 1)
    b = ones(length(x), 1)
    for i in eachindex(x)
        a[i, 1] = x[i] + m
    end
    return distance + sum(vec(a) .- vec(b))
end

# A genuine overwrite of the structured operand after the call.
function structured_overwrite_loss(data, m)
    distance = fresh_buffer_distance(data.x, m)
    for i in eachindex(data.x)
        data.x[i] += 2.0
    end
    return distance + sum(vec(reshape(data.x, :, 1)))
end

@testset "Fresh buffer writes preserve earlier reads" begin
    for x in (Float64[], [0.4], [0.4, -0.7, 1.3])
        m = 0.2
        expected = 2sum(m .- x)
        @test autodiff(Reverse, fresh_buffer_loss, Active, Const(x), Active(m))[1][2] ≈ expected + length(x)
        # A genuine overwrite must retain the old values for the reverse pass.
        @test autodiff(Reverse, overwritten_buffer_loss, Active, Const(copy(x)), Active(m))[1][2] ≈ expected
    end

    io = IOBuffer()
    Enzyme.Compiler.enzyme_code_llvm(
        io, fresh_buffer_loss, Active,
        Tuple{Const{Vector{Float64}}, Active{Float64}};
        dump_module = true, second_stage = false, debuginfo = :none
    )
    ir = String(take!(io))
    # On Julia 1.10 the split callee rereads x in reverse, so its augmented
    # forward pass has no tape to return. Later Julia versions merge fresh
    # Memory with a shared empty singleton before loading its data pointer;
    # that control-dependent case still conservatively retains the tape.
    if VERSION < v"1.11-"
        @test occursin(r"define[^\n]*void @augmented_julia_fresh_buffer_distance_", ir)
    else
        @test_broken occursin(r"define[^\n]*void @augmented_julia_fresh_buffer_distance_", ir)
    end
end

@testset "Fresh boxes and reshapes after the call preserve earlier reads" begin
    for x in (Float64[], [0.4], [0.4, -0.7, 1.3])
        m = 0.2
        expected = 2sum(m .- x) + length(x)
        @test autodiff(Reverse, reshaped_buffer_loss, Active, Const(x), Active(m))[1][2] ≈ expected
        data = (; x)
        @test autodiff(Reverse, structured_buffer_loss, Active, Const(data), Active(m))[1][2] ≈ expected
        @test autodiff(Reverse, broadcast_buffer_loss, Active, Const(x), Active(m))[1][2] ≈ expected
        # A genuine overwrite must retain the old values for the reverse pass.
        overwritten = (; x = copy(x))
        @test autodiff(Reverse, structured_overwrite_loss, Active, Const(overwritten), Active(m))[1][2] ≈
            expected - length(x)
        @test overwritten.x == x .+ 2.0
    end

    reshaped = fresh_buffer_ir(reshaped_buffer_loss, Tuple{Const{Vector{Float64}}, Active{Float64}})
    merged = fresh_buffer_ir(broadcast_buffer_loss, Tuple{Const{Vector{Float64}}, Active{Float64}})
    structured = fresh_buffer_ir(
        structured_buffer_loss,
        Tuple{Const{@NamedTuple{x::Vector{Float64}}}, Active{Float64}}
    )
    overwrite = fresh_buffer_ir(
        structured_overwrite_loss,
        Tuple{Const{@NamedTuple{x::Vector{Float64}}}, Active{Float64}}
    )
    @test !occursin(tape_free_distance, overwrite)
    if VERSION < v"1.11-"
        @test occursin(tape_free_distance, reshaped)
        @test occursin(tape_free_distance, structured)
        @test occursin(tape_free_distance, merged)
    else
        # Later Julia versions merge fresh Memory with a shared empty
        # singleton before loading its data pointer (testset above).
        @test_broken occursin(tape_free_distance, reshaped)
        @test_broken occursin(tape_free_distance, structured)
        @test_broken occursin(tape_free_distance, merged)
    end
end

# Calls after the differentiated call that Julia does not inline, as a model
# kernel makes them: they read an earlier result and write only arrays they
# allocate, or the array an earlier such call returned.
@noinline fresh_buffer_doses(field, level) = exp.(field .+ permutedims(level))

@noinline function fresh_buffer_readout(field, channel)
    out = similar(field, size(field, 1))
    for i in eachindex(out)
        out[i] = field[i, channel[i]]
    end
    return out
end

@noinline function fresh_buffer_double!(y)
    for i in eachindex(y)
        y[i] *= 2
    end
    return y
end

function fresh_buffer_channels(x, m)
    field = zeros(length(x), 2)
    for i in eachindex(x)
        field[i, 1] = x[i] + m
        field[i, 2] = x[i] - m
    end
    return field
end

# A broadcast that allocates its own output (on Julia 1.10, on either branch).
function doses_buffer_loss(x, m)
    distance = fresh_buffer_distance(x, m)
    return distance + sum(fresh_buffer_doses(fresh_buffer_channels(x, m), [0.1, -0.2]))
end

# A readout of chosen columns into an array from `similar`.
function readout_buffer_loss(x, m)
    distance = fresh_buffer_distance(x, m)
    channel = [isodd(i) ? 1 : 2 for i in eachindex(x)]
    return distance + sum(fresh_buffer_readout(fresh_buffer_channels(x, m), channel))
end

# A reduction over dims: on Julia 1.10 reducedim_initarray splats tuples
# through Core._apply_iterate, and _mapreducedim! fills the array it returned.
function rowsum_buffer_loss(x, m)
    distance = fresh_buffer_distance(x, m)
    return distance + sum(abs2, vec(sum(fresh_buffer_channels(x, m); dims = 2)))
end

# A later call that writes the operand itself.
function doubled_operand_loss(x, m)
    distance = fresh_buffer_distance(x, m)
    return distance + sum(fresh_buffer_double!(x))
end

# A later call that writes an array allocated before the call.
function doubled_prior_loss(x, m)
    prior = x .+ m
    distance = fresh_buffer_distance(x, m)
    return distance + sum(fresh_buffer_double!(prior))
end

@testset "Calls that write only new arrays after the call preserve earlier reads" begin
    for x in (Float64[], [0.4], [0.4, -0.7, 1.3])
        m = 0.2
        d = 2sum(m .- x)
        doses = sum(exp.(x .+ m .+ 0.1) .- exp.(x .- m .- 0.2))
        readout = sum(isodd(i) ? 1.0 : -1.0 for i in eachindex(x); init = 0.0)
        @test autodiff(Reverse, doses_buffer_loss, Active, Const(x), Active(m))[1][2] ≈ d + doses
        @test autodiff(Reverse, readout_buffer_loss, Active, Const(x), Active(m))[1][2] ≈ d + readout
        # (x + m) + (x - m) does not depend on m.
        @test autodiff(Reverse, rowsum_buffer_loss, Active, Const(x), Active(m))[1][2] ≈ d
        @test autodiff(Reverse, doubled_prior_loss, Active, Const(x), Active(m))[1][2] ≈ d + 2length(x)
        # A genuine overwrite must retain the old values for the reverse pass.
        overwritten = copy(x)
        @test autodiff(Reverse, doubled_operand_loss, Active, Const(overwritten), Active(m))[1][2] ≈ d
        @test overwritten == 2 .* x
    end

    argtypes = Tuple{Const{Vector{Float64}}, Active{Float64}}
    @test !occursin(tape_free_distance, fresh_buffer_ir(doubled_operand_loss, argtypes))
    @test !occursin(tape_free_distance, fresh_buffer_ir(doubled_prior_loss, argtypes))
    for loss in (doses_buffer_loss, readout_buffer_loss, rowsum_buffer_loss)
        ir = fresh_buffer_ir(loss, argtypes)
        if VERSION < v"1.11-"
            @test occursin(tape_free_distance, ir)
        else
            # Fresh Memory merged with the shared empty singleton (above).
            @test_broken occursin(tape_free_distance, ir)
        end
    end
end

# Calls after the differentiated call that a model kernel makes for its other
# fields and readouts: they write only arrays they allocate, but through calls
# that write the arrays they are given (a matrix product filling the result
# through gemv! and BLAS), or by filling an array with arrays they compute
# (collect). The product takes x itself: on Julia 1.12 with bounds checks,
# `reshape` of a Const array needs runtime activity whatever follows it.
@noinline fresh_buffer_mix(x) = fill(0.5, length(x), length(x)) * x

@noinline fresh_buffer_groups(y, groups) = [y[g] for g in groups]

# A second field evaluated after the first, as a kernel evaluates several.
function second_field_loss(x, m)
    distance = fresh_buffer_distance(x, m)
    return distance + sum(fresh_buffer_mix(x))
end

# The same with x carried by a structured Const operand, as a kernel receives
# its data arrays.
function structured_field_loss(data, m)
    distance = fresh_buffer_distance(data.x, m)
    return distance + sum(fresh_buffer_mix(data.x))
end

# A readout that groups an earlier result before summing each group.
function grouped_buffer_loss(x, m, groups)
    distance = fresh_buffer_distance(x, m)
    return distance + sum(sum, fresh_buffer_groups(exp.(x .+ m), groups); init = 0.0)
end

@testset "Calls that fill only new arrays through other calls preserve earlier reads" begin
    for x in (Float64[], [0.4], [0.4, -0.7, 1.3], [0.4, -0.7, 1.3, 0.2, -1.1])
        m = 0.2
        d = 2sum(m .- x)
        @test autodiff(Reverse, second_field_loss, Active, Const(x), Active(m))[1][2] ≈ d
        @test autodiff(Reverse, structured_field_loss, Active, Const((; x)), Active(m))[1][2] ≈ d
        groups = [collect(i:2:length(x)) for i in 1:min(2, length(x))]
        grouped = sum(exp(x[i] + m) for g in groups for i in g; init = 0.0)
        @test autodiff(Reverse, grouped_buffer_loss, Active, Const(x), Active(m), Const(groups))[1][2] ≈
            d + grouped
    end

    argtypes = Tuple{Const{Vector{Float64}}, Active{Float64}}
    structured = Tuple{Const{@NamedTuple{x::Vector{Float64}}}, Active{Float64}}
    grouped = Tuple{Const{Vector{Float64}}, Active{Float64}, Const{Vector{Vector{Int}}}}
    for ir in (
            fresh_buffer_ir(second_field_loss, argtypes),
            fresh_buffer_ir(structured_field_loss, structured),
            fresh_buffer_ir(grouped_buffer_loss, grouped),
        )
        if VERSION < v"1.11-"
            @test occursin(tape_free_distance, ir)
        else
            # Fresh Memory merged with the shared empty singleton (above).
            @test_broken occursin(tape_free_distance, ir)
        end
    end
end
