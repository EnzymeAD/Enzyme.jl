using Molly
using Molly: to_device
using Enzyme
using FiniteDifferences
using LinearAlgebra
using Random
using Statistics
using Test

const data_dir = joinpath(dirname(pathof(Molly)), "..", "data")
const ff_dir = joinpath(data_dir, "force_fields")
const run_parallel_tests = (Threads.nthreads() > 1)

@testset "Gradients" begin
    inter = LennardJones()
    boundary = CubicBoundary(5.0)
    a1, a2 = Atom(σ = 0.3, ϵ = 0.5), Atom(σ = 0.3, ϵ = 0.5)

    function force_direct(dist)
        c1 = SVector(1.0, 1.0, 1.0)
        c2 = SVector(dist + 1.0, 1.0, 1.0)
        vec = vector(c1, c2, boundary)
        F = force(inter, vec, a1, a2, NoUnits)
        return F[1]
    end

    function pe(dist)
        c1 = SVector(1.0, 1.0, 1.0)
        c2 = SVector(dist + 1.0, 1.0, 1.0)
        vec = vector(c1, c2, boundary)
        potential_energy(inter, vec, a1, a2, NoUnits)
    end

    function force_grad(dist)
        grads = autodiff(
            Reverse,
            pe,
            Active,
            Active(dist),
        )
        return -grads[1][1]
    end

    dists = collect(0.2:0.01:1.2)
    forces_direct = force_direct.(dists)
    forces_grad = force_grad.(dists)
    @test all(forces_direct .≈ forces_grad)
end

@testset "Differentiable PME" begin
    T = Float64
    AT = Array
    ff = MolecularForceField(
        T,
        joinpath.(ff_dir, ["ff99SBildn.xml", "tip3p_standard.xml"])...,
        units = false,
    )
    sys = System(
        joinpath(data_dir, "6mrr_equil.pdb"),
        ff;
        units = false,
        array_type = AT,
        nonbonded_method = :pme,
        grad_safe = true,
    )

    pme = sys.general_inters[1]
    Fs = zero(sys.coords)
    d_sys = zero(sys)
    d_pme = zero(pme)

    pe = Molly.ewald_pe_forces!(Fs, nothing, sys, pme, Val(false))
    Fs_ad = zero(sys.coords)

    pe_ad = autodiff(
        ReverseWithPrimal,
        Molly.ewald_pe_forces!,
        Active,
        Const(Fs_ad),
        Const(nothing),
        Duplicated(sys, d_sys),
        Duplicated(pme, d_pme),
        Const(Val(false)),
    )[2]

    @test pe_ad ≈ pe atol = 1.0e-6
    @test Fs_ad ≈ Fs atol = 1.0e-10
    @test -d_sys.coords ≈ Fs atol = 1.0e-10

    function coord_fdm(c)
        coords_mod = copy(sys.coords)
        coords_mod[1] = SVector(c, coords_mod[1][2], coords_mod[1][3])
        sys_mod = System(deepcopy(sys); coords = coords_mod)
        return Molly.ewald_pe_forces!(Fs, nothing, sys_mod, pme, Val(false))
    end

    c = sys.coords[1][1]
    coord_fdm(c)
    coord_grad = central_fdm(5, 1)(coord_fdm, c)
    @test d_sys.coords[1][1] ≈ coord_grad atol = 1.0e-6

    function charge_fdm(ch)
        atoms_mod = copy(sys.atoms)
        at = sys.atoms[1]
        atoms_mod[1] = Atom(mass = at.mass, charge = ch, σ = at.σ, ϵ = at.σ)
        sys_mod = System(deepcopy(sys); atoms = atoms_mod)
        return Molly.ewald_pe_forces!(Fs, nothing, sys_mod, pme, Val(false))
    end

    at = sys.atoms[1]
    charge_fdm(charge(at))
    charge_grad = central_fdm(5, 1)(charge_fdm, charge(at))
    @test charge(d_sys.atoms[1]) ≈ charge_grad atol = 1.0e-6
end

@testset "Differentiable simulation" begin
    runs = [ #               gpu    par    fwd    f32    obc2   gbn2   tol_σ   tol_r0
        ("CPU",              Array, false, false, false, false, false, 1.0e-4, 1.0e-4),
        ("CPU forward",      Array, false, true,  false, false, false, 0.5,    0.1),
        ("CPU f32",          Array, false, false, true,  false, false, 0.01,   5.0e-4),
        ("CPU obc2",         Array, false, false, false, true,  false, 1.0e-4, 1.0e-4),
        ("CPU gbn2",         Array, false, false, false, false, true,  1.0e-4, 1.0e-4),
        ("CPU gbn2 forward", Array, false, true,  false, false, true,  0.5,    0.1),
    ]
    if run_parallel_tests #                  gpu    par    fwd    f32    obc2   gbn2   tol_σ   tol_r0
        push!(runs, ("CPU parallel",         Array, true,  false, false, false, false, 1.0e-4, 1.0e-4))
        push!(runs, ("CPU parallel forward", Array, true,  true,  false, false, false, 0.5,    0.1))
        push!(runs, ("CPU parallel f32",     Array, true,  false, true,  false, false, 0.01,   5.0e-4))
    end

    function mean_min_separation(coords, boundary, ::Val{T}) where {T}
        min_seps = T[]
        for i in eachindex(coords)
            min_sq_sep = T(100.0)
            for j in eachindex(coords)
                if i != j
                    sq_dist = sum(abs2, vector(coords[i], coords[j], boundary))
                    min_sq_sep = min(sq_dist, min_sq_sep)
                end
            end
            push!(min_seps, sqrt(min_sq_sep))
        end
        return mean(min_seps)
    end

    function loss(
            σ, r0, coords, velocities, boundary, pairwise_inters, general_inters,
            neighbor_finder, simulator, n_steps, n_threads, n_atoms, atom_mass, bond_dists,
            bond_is, bond_js, angles, torsions, rng, ::Val{T}, ::Val{AT}
        ) where {T, AT}
        atoms = [Atom(i, 1, atom_mass, (i % 2 == 0 ? T(-0.02) : T(0.02)), σ, T(0.2)) for i in 1:n_atoms]
        bonds_inner = HarmonicBond{T, T}[]
        for i in 1:(n_atoms ÷ 2)
            push!(bonds_inner, HarmonicBond(T(100.0), bond_dists[i] * r0))
        end
        bonds = InteractionList2Atoms(
            bond_is,
            bond_js,
            to_device(bonds_inner, AT),
        )

        sys = System(
            atoms = to_device(atoms, AT),
            coords = to_device(coords, AT),
            boundary = boundary,
            velocities = to_device(velocities, AT),
            pairwise_inters = pairwise_inters,
            specific_inter_lists = (bonds, angles, torsions),
            general_inters = general_inters,
            neighbor_finder = neighbor_finder,
            force_units = NoUnits,
            energy_units = NoUnits,
        )

        simulate!(sys, simulator, n_steps; n_threads = n_threads, rng = rng)

        return mean_min_separation(sys.coords, boundary, Val(T))
    end

    for (name, AT, parallel, forward, f32, obc2, gbn2, tol_σ, tol_r0) in runs
        T = (f32 ? Float32 : Float64)
        σ = T(0.4)
        r0 = T(1.0)
        n_atoms = 50
        n_steps = 100
        atom_mass = T(10.0)
        boundary = CubicBoundary(T(3.0))
        temp = T(1.0)
        simulator = VelocityVerlet(
            dt = T(0.001),
            coupling = (ImmediateThermostat(temp),),
        )
        rng = Xoshiro(1000) # Same system every time, not required but increases stability
        coords = place_atoms(n_atoms, boundary; min_dist = T(0.6), max_attempts = 500, rng = rng)
        velocities = [random_velocity(atom_mass, temp; rng = rng) for i in 1:n_atoms]
        nb_cutoff = T(1.2)
        lj = LennardJones(cutoff = DistanceCutoff(nb_cutoff), use_neighbors = true)
        crf = CoulombReactionField(
            dist_cutoff = nb_cutoff,
            solvent_dielectric = T(Molly.crf_solvent_dielectric),
            use_neighbors = true,
            coulomb_const = T(ustrip(Molly.coulomb_const)),
        )
        pairwise_inters = (lj, crf)
        bond_is = to_device(Int32.(collect(1:(n_atoms ÷ 2))), AT)
        bond_js = to_device(Int32.(collect((1 + n_atoms ÷ 2):n_atoms)), AT)
        bond_dists = [
            norm(vector(coords[i], coords[i + n_atoms ÷ 2], boundary))
                for i in 1:(n_atoms ÷ 2)
        ]
        angles_inner = [HarmonicAngle(k = T(10.0), θ0 = T(2.0)) for i in 1:15]
        angles = InteractionList3Atoms(
            to_device(Int32.(collect(1:15)), AT),
            to_device(Int32.(collect(16:30)), AT),
            to_device(Int32.(collect(31:45)), AT),
            to_device(angles_inner, AT),
        )
        torsions_inner = [
            PeriodicTorsion(
                    periodicities = [1, 2, 3],
                    phases = T[1.0, 0.0, -1.0],
                    ks = T[10.0, 5.0, 8.0],
                    n_terms = 6,
                ) for i in 1:10
        ]
        torsions = InteractionList4Atoms(
            to_device(Int32.(collect(1:10)), AT),
            to_device(Int32.(collect(11:20)), AT),
            to_device(Int32.(collect(21:30)), AT),
            to_device(Int32.(collect(31:40)), AT),
            to_device(torsions_inner, AT),
        )
        atoms_setup = [Atom(charge = zero(T), σ = zero(T)) for i in 1:n_atoms]
        if obc2
            imp_obc2 = ImplicitSolventOBC(
                to_device(atoms_setup, AT),
                [AtomData(element = "O") for i in 1:n_atoms],
                InteractionList2Atoms(bond_is, bond_js, nothing);
                kappa = T(0.7),
                use_OBC2 = true,
            )
            general_inters = (imp_obc2,)
        elseif gbn2
            imp_gbn2 = ImplicitSolventGBN2(
                to_device(atoms_setup, AT),
                [AtomData(element = "O") for i in 1:n_atoms],
                InteractionList2Atoms(bond_is, bond_js, nothing);
                kappa = T(0.7),
            )
            general_inters = (imp_gbn2,)
        else
            general_inters = ()
        end
        neighbor_finder = DistanceNeighborFinder(
            eligible = to_device(trues(n_atoms, n_atoms), AT),
            n_steps = 10,
            dist_cutoff = T(1.5),
        )
        n_threads = (parallel ? Threads.nthreads() : 1)

        const_args = [
            Const(boundary), Const(pairwise_inters),
            Const(general_inters), Const(neighbor_finder), Const(simulator),
            Const(n_steps), Const(n_threads), Const(n_atoms), Const(atom_mass),
            Const(bond_dists), Const(bond_is), Const(bond_js), Const(angles),
            Const(torsions), Const(rng), Const(Val(T)), Const(Val(AT)),
        ]
        if forward
            grad_enzyme = (
                autodiff(
                    set_runtime_activity(Forward), loss, Duplicated,
                    Duplicated(σ, one(T)), Const(r0), Duplicated(copy(coords), zero(coords)),
                    Duplicated(copy(velocities), zero(velocities)), const_args...,
                )[1],
                autodiff(
                    set_runtime_activity(Forward), loss, Duplicated,
                    Const(σ), Duplicated(r0, one(T)), Duplicated(copy(coords), zero(coords)),
                    Duplicated(copy(velocities), zero(velocities)), const_args...,
                )[1],
            )
        else
            grad_enzyme = autodiff(
                set_runtime_activity(Reverse), loss, Active,
                Active(σ), Active(r0), Duplicated(copy(coords), zero(coords)),
                Duplicated(copy(velocities), zero(velocities)), const_args...,
            )[1][1:2]
        end

        grad_fd = (
            central_fdm(6, 1)(
                σ -> loss(
                    σ, r0, copy(coords), copy(velocities), boundary, pairwise_inters, general_inters,
                    neighbor_finder, simulator, n_steps, n_threads, n_atoms, atom_mass, bond_dists,
                    bond_is, bond_js, angles, torsions, rng, Val(T), Val(AT),
                ),
                σ,
            ),
            central_fdm(6, 1)(
                r0 -> loss(
                    σ, r0, copy(coords), copy(velocities), boundary, pairwise_inters, general_inters,
                    neighbor_finder, simulator, n_steps, n_threads, n_atoms, atom_mass, bond_dists,
                    bond_is, bond_js, angles, torsions, rng, Val(T), Val(AT),
                ),
                r0,
            ),
        )
        for (prefix, genz, gfd, tol) in zip(("σ", "r0"), grad_enzyme, grad_fd, (tol_σ, tol_r0))
            if abs(gfd) < 1.0e-13
                @info "$(rpad(name, 20)) - $(rpad(prefix, 2)) - FD $gfd, Enzyme $genz"
                ztol = (contains(name, "f32") ? 1.0e-8 : 1.0e-10)
                @test isnothing(genz) || abs(genz) < ztol
            elseif isnothing(genz)
                @info "$(rpad(name, 20)) - $(rpad(prefix, 2)) - FD $gfd, Enzyme $genz"
                @test !isnothing(genz)
            else
                frac_diff = abs(genz - gfd) / abs(gfd)
                @info "$(rpad(name, 20)) - $(rpad(prefix, 2)) - FD $gfd, Enzyme $genz, fractional difference $frac_diff"
                @test frac_diff < tol
            end
        end
    end
end

@testset "Differentiable protein" begin
    function create_sys(AT)
        ff = MolecularForceField(joinpath.(ff_dir, ["ff99SBildn.xml"])...; units = false)
        return System(
            joinpath(data_dir, "6mrr_nowater.pdb"),
            ff;
            units = false,
            array_type = AT,
            nonbonded_method = :cutoff,
            implicit_solvent = :gbn2,
            kappa = 0.7,
        )
    end

    EnzymeRules.inactive(::typeof(create_sys), args...) = nothing

    function test_energy_grad(params_dic, sys_ref, coords, neighbor_finder, n_threads)
        atoms, pis, sis, gis = Molly.inject_gradients(sys_ref, params_dic)

        sys = System(
            atoms = atoms,
            coords = coords,
            boundary = sys_ref.boundary,
            pairwise_inters = pis,
            specific_inter_lists = sis,
            general_inters = gis,
            neighbor_finder = neighbor_finder,
            force_units = NoUnits,
            energy_units = NoUnits,
        )

        return potential_energy(sys; n_threads = n_threads)
    end

    function test_forces_grad(params_dic, sys_ref, coords, neighbor_finder, n_threads)
        atoms, pis, sis, gis = Molly.inject_gradients(sys_ref, params_dic)

        sys = System(
            atoms = atoms,
            coords = coords,
            boundary = sys_ref.boundary,
            pairwise_inters = pis,
            specific_inter_lists = sis,
            general_inters = gis,
            neighbor_finder = neighbor_finder,
            force_units = NoUnits,
            energy_units = NoUnits,
        )

        fs = forces(sys; n_threads = n_threads)
        return sum(sum.(abs, fs))
    end

    params_dic = Dict(
        "atom_C8_σ" => 0.33996695084235345,
        "atom_C8_ϵ" => 0.4577296,
        "atom_C9_σ" => 0.33996695084235345,
        "atom_C9_ϵ" => 0.4577296,
        "atom_CA_σ" => 0.33996695084235345,
        "atom_CA_ϵ" => 0.359824,
        "atom_CT_σ" => 0.33996695084235345,
        "atom_CT_ϵ" => 0.4577296,
        "atom_C_σ" => 0.33996695084235345,
        "atom_C_ϵ" => 0.359824,
        "atom_N3_σ" => 0.32499985237759577,
        "atom_N3_ϵ" => 0.71128,
        "atom_N_σ" => 0.32499985237759577,
        "atom_N_ϵ" => 0.71128,
        "atom_O2_σ" => 0.2959921901149463,
        "atom_O2_ϵ" => 0.87864,
        "atom_OH_σ" => 0.30664733878390477,
        "atom_OH_ϵ" => 0.8803136,
        "atom_O_σ" => 0.2959921901149463,
        "atom_O_ϵ" => 0.87864,
        "inter_CO_weight_14" => 0.8333,
        "inter_GB_neck_cut" => 0.68,
        "inter_GB_neck_scale" => 0.826836,
        "inter_GB_offset" => 0.0195141,
        "inter_GB_params_C_α" => 0.733756,
        "inter_GB_params_C_β" => 0.506378,
        "inter_GB_params_C_γ" => 0.205844,
        "inter_GB_params_N_α" => 0.503364,
        "inter_GB_params_N_β" => 0.316828,
        "inter_GB_params_N_γ" => 0.192915,
        "inter_GB_params_O_α" => 0.867814,
        "inter_GB_params_O_β" => 0.876635,
        "inter_GB_params_O_γ" => 0.387882,
        "inter_GB_probe_radius" => 0.14,
        "inter_GB_radius_C" => 0.17,
        "inter_GB_radius_N" => 0.155,
        "inter_GB_radius_O" => 0.15,
        "inter_GB_radius_O_CAR" => 0.14,
        "inter_GB_sa_factor" => 28.3919551,
        "inter_GB_screen_C" => 1.058554,
        "inter_GB_screen_N" => 0.733599,
        "inter_GB_screen_O" => 1.061039,
        "inter_LJ_weight_14" => 0.5,
        "inter_PT_-/C/CT/-_k_1" => 0.0,
        "inter_PT_-/C/N/-_k_1" => -10.46,
        "inter_PT_-/CA/CA/-_k_1" => -15.167,
        "inter_PT_-/CA/CT/-_k_1" => 0.0,
        "inter_PT_-/CT/C8/-_k_1" => 0.64852,
        "inter_PT_-/CT/C9/-_k_1" => 0.64852,
        "inter_PT_-/CT/CT/-_k_1" => 0.6508444444444447,
        "inter_PT_-/CT/N/-_k_1" => 0.0,
        "inter_PT_-/CT/N3/-_k_1" => 0.6508444444444447,
        "inter_PT_C/N/CT/C_k_1" => -0.142256,
        "inter_PT_C/N/CT/C_k_2" => 1.40164,
        "inter_PT_C/N/CT/C_k_3" => 2.276096,
        "inter_PT_C/N/CT/C_k_4" => 0.33472,
        "inter_PT_C/N/CT/C_k_5" => 1.6736,
        "inter_PT_CT/CT/C/N_k_1" => 0.8368,
        "inter_PT_CT/CT/C/N_k_2" => 0.8368,
        "inter_PT_CT/CT/C/N_k_3" => 1.6736,
        "inter_PT_CT/CT/N/C_k_1" => 8.368,
        "inter_PT_CT/CT/N/C_k_2" => 8.368,
        "inter_PT_CT/CT/N/C_k_3" => 1.6736,
        "inter_PT_H/N/C/O_k_1" => 8.368,
        "inter_PT_H/N/C/O_k_2" => -10.46,
        "inter_PT_H1/CT/C/O_k_1" => 3.3472,
        "inter_PT_H1/CT/C/O_k_2" => -0.33472,
        "inter_PT_HC/CT/C4/CT_k_1" => 0.66944,
        "inter_PT_N/CT/C/N_k_1" => 2.7196,
        "inter_PT_N/CT/C/N_k_10" => 0.1046,
        "inter_PT_N/CT/C/N_k_11" => -0.046024,
        "inter_PT_N/CT/C/N_k_2" => -0.824248,
        "inter_PT_N/CT/C/N_k_3" => 6.04588,
        "inter_PT_N/CT/C/N_k_4" => 2.004136,
        "inter_PT_N/CT/C/N_k_5" => -0.0799144,
        "inter_PT_N/CT/C/N_k_6" => -0.016736,
        "inter_PT_N/CT/C/N_k_7" => -1.06692,
        "inter_PT_N/CT/C/N_k_8" => 0.3138,
        "inter_PT_N/CT/C/N_k_9" => 0.238488,
    )

    platform_runs = [("CPU", Array, false)]
    if run_parallel_tests
        push!(platform_runs, ("CPU parallel", Array, true))
    end
    test_runs = [
        ("Energy", test_energy_grad, 1.0e-8),
        ("Force", test_forces_grad, 1.0e-8),
    ]
    params_to_test = (
        "atom_N_ϵ",
        "inter_PT_C/N/CT/C_k_1",
        "inter_GB_screen_O",
    )

    for (test_name, test_fn, test_tol) in test_runs
        for (platform, AT, parallel) in platform_runs
            sys_ref = create_sys(AT)
            n_threads = (parallel ? Threads.nthreads() : 1)
            grads_enzyme = Dict(k => 0.0 for k in keys(params_dic))
            autodiff(
                set_runtime_activity(Reverse), test_fn, Active,
                Duplicated(params_dic, grads_enzyme), Const(sys_ref),
                Duplicated(copy(sys_ref.coords), zero(sys_ref.coords)),
                Duplicated(sys_ref.neighbor_finder, sys_ref.neighbor_finder),
                Const(n_threads),
            )
            for param in params_to_test
                genz = grads_enzyme[param]
                gfd = central_fdm(6, 1)(params_dic[param]) do val
                    dic = copy(params_dic)
                    dic[param] = val
                    test_fn(dic, sys_ref, copy(sys_ref.coords), sys_ref.neighbor_finder, n_threads)
                end
                frac_diff = abs(genz - gfd) / abs(gfd)
                @info "$(rpad(test_name, 6)) - $(rpad(platform, 12)) - $(rpad(param, 21)) - " *
                    "FD $gfd, Enzyme $genz, fractional difference $frac_diff"
                tol = (test_name == "Force" && param == "atom_N_ϵ" ? 2.0e-3 : test_tol)
                @test frac_diff < tol
            end
        end
    end
end
