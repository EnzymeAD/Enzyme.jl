using Comrade
using Pyehtim
using Enzyme
using Distributions
using FiniteDifferences
using VLBISkyModels
using LinearAlgebra

using Test


const ComradePATH = joinpath(dirname(pathof(Comrade)), "..", "examples", "Data")
const dataurl = "https://de.cyverse.org/anon-files/iplant/home/shared/commons_repo/curated/EHTC_M87pol2017_Nov2023/hops_data/April11/SR2_M87_2017_101_lo_hops_ALMArot.uvfits"
const arrayf = joinpath(ComradePATH, "array.txt")
const dataf = Base.download(dataurl)

function FiniteDifferences.to_vec(k::IntensityMap)
    v, b = to_vec(DD.data(k))
    back(x) = DD.rebuild(k, b(x))
    return v, back
end

function FiniteDifferences.to_vec(k::UnstructuredMap)
    v, b = to_vec(parent(k))
    back(x) = UnstructuredMap(b(x), axisdims(k))
    return v, back
end

function testgrad(f, x, g; atol = 1.0e-8, rtol = 1.0e-5)
    dx = Enzyme.make_zero(x)
    autodiff(set_runtime_activity(Enzyme.Reverse), Const(f), Active, Duplicated(x, dx), Const(g))
    fdm = central_fdm(5, 1)

    gf = grad(fdm, Base.Fix2(f, g), x)[begin]
    return @test isapprox(dx, gf; atol, rtol)
end


@testset "Model Gradients" begin

    u = randn(10) * 0.5
    v = randn(10) * 0.5
    t = sort(rand(10) * 0.5)
    f = fill(230.0e9, 10)
    g = UnstructuredDomain((U = u, V = v, Ti = t, Fr = f))


    @testset "Gaussian" begin
        foo(x, g) = sum(abs2, VLBISkyModels.visibilitymap_analytic(x[1] * Gaussian(), g))
        testgrad(foo, [1.0], g)
    end

    @testset "Disk" begin
        foo(x, g) = sum(abs2, VLBISkyModels.visibilitymap_analytic(x[1] * Disk(), g))
        testgrad(foo, [0.5], g)
    end

    @testset "SlashedDisk" begin
        foo(x, g) = sum(abs2, VLBISkyModels.visibilitymap_analytic(SlashedDisk(x[1]), g))
        testgrad(foo, [0.5], g)
    end

    @testset "MRing" begin
        foo(x, g) = sum(abs2, VLBISkyModels.visibilitymap_analytic(MRing(x[1], x[2]), g))
        testgrad(foo, [0.5, 0.2], g)
    end

    @testset "MRing2" begin
        foo(x, g) = sum(
            abs2,
            VLBISkyModels.visibilitymap_analytic(
                MRing((x[1], x[2]), (x[3], x[4])),
                g
            )
        )
        x = rand(4)
        testgrad(foo, x, g)
    end

    @testset "Multicomponent" begin
        foo(x, g) = sum(
            abs2,
            VLBISkyModels.visibilitymap_analytic(
                MultiComponentModel(
                    Gaussian(),
                    @view(x[:, 1]),
                    @view(x[:, 2]),
                    @view(x[:, 3])
                ),
                g
            )
        )

        x = randn(10, 4)
        testgrad(foo, x, g)

    end

    @testset "Modifiers" begin
        ma = Gaussian()
        @testset "Shifted" begin
            foo(x, g) = sum(abs2, VLBISkyModels.visibilitymap_analytic(shifted(ma, x[1], x[2]), g))
            x = rand(2)
            testgrad(foo, x, g)
        end

        @testset "Stretched" begin
            foo(x, g) = sum(abs2, VLBISkyModels.visibilitymap_analytic(stretched(ma, x[1], x[2]), g))
            x = rand(2)
            testgrad(foo, x, g)
        end

        @testset "Rotated" begin
            foo(x, g) = sum(abs2, VLBISkyModels.visibilitymap_analytic(rotated(ma, x[1]), g))
            x = rand(1)
            testgrad(foo, x, g)
        end

        @testset "All mods" begin
            foo(x, g) = sum(
                abs2,
                VLBISkyModels.visibilitymap_analytic(
                    modify(
                        ma, Shift(x[1], x[2]),
                        Stretch(x[3], x[4]),
                        Rotate(x[5]),
                        Renormalize(x[6])
                    ), g
                )
            )
            x = rand(6)
            testgrad(foo, x, g)
        end
    end

    @testset "Composite models" begin
        m1 = Disk()
        m2 = GaussianRing(2.0)

        @testset "Sum" begin
            foo(x, g) = sum(
                abs2,
                VLBISkyModels.visibilitymap_analytic(
                    x[1] *
                        stretched(
                        Gaussian(), x[2],
                        x[3]
                    ) +
                        shifted(
                        MRing(x[4], x[4]), x[5],
                        x[6]
                    ), g
                )
            )
            x = rand(6)
            foo(x, g)
            testgrad(foo, x, g)
        end

        @testset "Convolved" begin
            foo(x, g) = sum(
                abs2,
                VLBISkyModels.visibilitymap_analytic(
                    convolved(
                        x[1] *
                            stretched(
                            Disk(), x[2],
                            x[3]
                        ),
                        stretched(
                            Ring(), x[4],
                            x[4]
                        )
                    ), g
                )
            )
            x = rand(4)
            testgrad(foo, x, g)

        end

        @testset "Polarized Model" begin
            foo(x, g) = sum(
                norm,
                VLBISkyModels.visibilitymap_analytic(
                    rotated(
                        PolarizedModel(
                            Gaussian(),
                            x[1] *
                                Gaussian(),
                            x[2] *
                                Gaussian(),
                            x[3] *
                                Gaussian()
                        ),
                        x[4]
                    ), g
                )
            )
            testgrad(foo, rand(4), g)

        end
    end

    @testset "M87 model test" begin
        function model(θ)
            rad = θ[1]
            wid = θ[2]
            a = θ[3]
            b = θ[4]
            f = θ[5]
            sig = θ[6]
            asy = θ[7]
            pa = θ[8]
            x = θ[9]
            y = θ[10]
            ring = f * smoothed(
                stretched(MRing((a,), (b,)), μas2rad(rad), μas2rad(rad)),
                μas2rad(wid)
            )
            g = (1 - f) *
                shifted(
                rotated(
                    stretched(Gaussian(), μas2rad(sig) * asy, μas2rad(sig)),
                    pa
                ), μas2rad(x), μas2rad(y)
            )
            return ring + g
        end

        foo(θ, g) = sum(abs2, VLBISkyModels.visibilitymap_analytic(model(θ), g))
        x = [40.0, 5.0, 0.7, 0.5, 0.3, 10.0, 1.2, pi / 4, 5.0, -3.0]
        foo(x, g)
        testgrad(foo, [40.0, 5.0, 0.7, 0.5, 0.3, 10.0, 1.2, pi / 4, 5.0, -3.0], g)
    end

    @testset "ContinuousImage" begin
        gim = imagepixels(5.0, 5.0, 32, 32)
        gfour = FourierDualDomain(gim, g, NFFTAlg())
        foo(x, gfour) = sum(
            abs2,
            VLBISkyModels.visibilitymap(
                modify(
                    ContinuousImage(
                        IntensityMap(
                            reshape(
                                @view(x[1:(end - 1)]),
                                size(VLBISkyModels.imgdomain(gfour))
                            ),
                            VLBISkyModels.imgdomain(gfour)
                        ),
                        BSplinePulse{3}()
                    ),
                    Shift(x[end], -x[end])
                ), gfour
            )
        )
        x = rand(prod(size(VLBISkyModels.imgdomain(gfour))) + 1)
        foo(x, gfour)
        testgrad(foo, x, gfour)
    end
end


@testset "Inference Tests" begin

    obs = ehtim.obsdata.load_uvfits(dataf)
    obsavg = scan_average(obs)

    obspol = Pyehtim.load_uvfits_and_array(
        dataf,
        arrayf;
        polrep = "circ"
    )
    obsavgpol = scan_average(obspol)


    vis = extract_table(obsavg, Visibilities())
    amp = extract_table(obsavg, VisibilityAmplitudes())
    lcamp = extract_table(obsavg, LogClosureAmplitudes())
    cphase = extract_table(obsavg, ClosurePhases())
    dcoh = extract_table(obsavgpol, Coherencies())


    @testset "Geometric Closures" begin
        g = imagepixels(μas2rad(150.0), μas2rad(150.0), 256, 256)
        function closuregeom(θ, meta)
            m1 = θ.f1 * rotated(stretched(Gaussian(), θ.σ1 * θ.τ1, θ.σ1), θ.ξ1)
            return m1
        end

        prior = (
            f1 = Uniform(0.8, 1.2),
            σ1 = Uniform(μas2rad(1.0), μas2rad(40.0)),
            τ1 = Uniform(0.35, 0.65),
            ξ1 = Uniform(-π / 2, π / 2),
        )

        skym = SkyModel(closuregeom, prior, g)
        post = VLBIPosterior(skym, lcamp, cphase)
        tpost = asflat(post)
        x = prior_sample(tpost)
        dx = Enzyme.make_zero(x)

        autodiff(set_runtime_activity(ReverseWithPrimal), logdensityof, Const(tpost), Duplicated(x, dx))

        fdm = central_fdm(5, 1)
        gf = grad(fdm, tpost, x)
        @test isapprox(dx, gf; atol = 1.0e-8, rtol = 1.0e-5)
    end

end
