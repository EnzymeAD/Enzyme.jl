module Testx86TapeRooting
    using Enzyme
    using Test
    using LinearAlgebra

    const V3 = NTuple{3, Float64}
    Base.zero(::Type{V3}) = (0.0, 0.0, 0.0)
    Base.zero(::V3) = (0.0, 0.0, 0.0)
    @inline Base.:-(a::V3, b::V3) = a .- b
    @inline Base.:+(a::V3, b::V3) = a .+ b
    @inline Base.:*(a::V3, b::Float64) = a .* b
    @inline Base.:/(a::V3, b::Float64) = a ./ b

    function myforces!(fs, fs_nounits, coords, pairwise_inters, neighbors, σ,
                       ::Val{needs_vir}, step_n::Integer = 0) where {needs_vir}
        fill!(fs_nounits, zero(eltype(fs_nounits)))
        @inbounds for ni in eachindex(neighbors)
            i, j, _ = neighbors[ni]
            f = σ .* (coords[j] .- coords[i])
            fs_nounits[i] -= f
            fs_nounits[j] += f
        end
        fs .= fs_nounits
        return nothing
    end

    function loss(σ, coords, velocities, boundary, pairwise_inters,
                  dt, n_steps, n_threads, n_atoms, atom_mass, neighbors, ::Val{T}) where {T}
        masses = fill(atom_mass, n_atoms)

        nsteps_sim = 200
        needs_vir_steps = length(coords) + nsteps_sim
        forces_t, forces_t_dt = zero(coords), zero(coords)
        fs_nounits = zero(coords)
        myforces!(forces_t, fs_nounits, coords, pairwise_inters, neighbors, σ, Val(false), 0)
        accels_t = forces_t ./ masses
        accels_t_dt = zero(accels_t)
        dt_sq_div2 = dt^2 / 2

        for step_n in 1:nsteps_sim
            needs_vir = (step_n % needs_vir_steps == 0)
            coords .+= accels_t .* dt_sq_div2
            myforces!(forces_t_dt, fs_nounits, coords, pairwise_inters, neighbors, σ, Val(needs_vir), step_n)
            accels_t_dt .= forces_t_dt ./ masses
            accels_t .= accels_t_dt
            GC.gc(false)
        end
        return sum(c -> sum(abs2, c), coords)
    end

    function run_case()
        σ = 0.4
        n_atoms = 50
        n_steps = 200
        n_threads = 1
        atom_mass = 10.0
        boundary = 3.0
        dt = 0.001
        faux_inters = 1.0

        coords = [(0.375 + 0.75 * (idx % 4),
                   0.375 + 0.75 * ((idx ÷ 4) % 4),
                   0.375 + 0.75 * (idx ÷ 16)) for idx in 0:(n_atoms - 1)]
        velocities = [(0.0, 0.0, 0.0) for _ in 1:n_atoms]

        neighbors = Tuple{Int32, Int32, Bool}[]
        for i in 1:n_atoms, j in (i + 1):n_atoms
            if sum(abs2, coords[i] - coords[j]) <= 1.5^2
                push!(neighbors, (Int32(i), Int32(j), false))
            end
        end

        const_args = [
            Const(boundary), Const(faux_inters), Const(dt),
            Const(n_steps), Const(n_threads), Const(n_atoms), Const(atom_mass),
            Const(neighbors), Const(Val(Float64)),
        ]
        grad_enzyme = autodiff(
            set_runtime_activity(Reverse), loss, Active,
            Active(σ), Duplicated(copy(coords), zero(coords)),
            Duplicated(copy(velocities), zero(velocities)), const_args...,
        )[1][1]
        return grad_enzyme
    end
end

using Test
@testset "x86_64 tape GC rooting segfault" begin
    for i in 1:10
        grad = Testx86TapeRooting.run_case()
        @test grad ≈ 0.01019299862508171 atol=1e-6
    end
end
