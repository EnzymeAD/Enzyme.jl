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

    @inferred f(x, g)

    dx = Enzyme.make_zero(x)
    autodiff(set_runtime_activity(Enzyme.Reverse), Const(f), Active, Duplicated(x, dx), Const(g))
    fdm = central_fdm(5, 1)

    gf = grad(fdm, Base.Fix2(f, g), x)[begin]
    return @test isapprox(dx, gf; atol, rtol)
end


# If this function is defined in `@testset` some weird scoping issue arises
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
        stretched(MRing((a,), (b,)), rad, rad),
        wid
    )
    # Why can't this be g?
    g = (1 - f) *
        shifted(
        rotated(
            stretched(Gaussian(), sig * asy, sig),
            pa
        ), x, y
    )
    return ring + g
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
        foo1(θ, g) = sum(abs2, VLBISkyModels.visibilitymap_analytic(model(θ), g))
        x = [40.0, 5.0, 0.7, 0.5, 0.3, 10.0, 1.2, pi / 4, 5.0, -3.0]
        @inferred foo1(x, g)
        testgrad(foo1, x, g)
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


    @testset "multidomain" begin
        fov = 1.0
        x = range(-fov / 2, fov; length = 8)
        y = range(-fov, fov; length = 8)
        Ti = [1.0, 2.0]
        Fr = [86.0e9, 230.0e9]
        g = RectiGrid((; X = x, Y = y, Ti, Fr))
        function foo4D(x, p)
            cimg = ContinuousImage(IntensityMap(x, VLBISkyModels.imgdomain(p)), DeltaPulse())
            vis = VLBISkyModels.visibilitymap(cimg, p)
            return sum(abs2, vis)
        end

        x = rand(size(g)...)
        U = randn(20) * 0.5
        V = randn(20) * 0.5
        uT = vcat(fill(Ti[1], 10), fill(Ti[2], 10))
        uFr = vcat(fill(Fr[1], 5), fill(Fr[2], 5), fill(Fr[1], 5), fill(Fr[2], 5))
        guv = UnstructuredDomain((U = U, V = V, Ti = uT, Fr = uFr))
        gfour = FourierDualDomain(g, guv, NFFTAlg())
        testgrad(foo4D, x, gfour)
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
            m2 = θ.f2 * rotated(stretched(Gaussian(), θ.σ2 * θ.τ2, θ.σ2), θ.ξ2)
            return m1 + shifted(m2, θ.x, θ.y)
        end

        prior = (
            f1 = Uniform(0.8, 1.2),
            σ1 = Uniform(μas2rad(1.0), μas2rad(40.0)),
            τ1 = Uniform(0.35, 0.65),
            ξ1 = Uniform(-π / 2, π / 2),
            f2 = Uniform(0.3, 0.7),
            σ2 = Uniform(μas2rad(1.0), μas2rad(40.0)),
            τ2 = Uniform(0.35, 0.65),
            ξ2 = Uniform(-π / 2, π / 2),
            x = Uniform(-μas2rad(40.0), μas2rad(40.0)),
            y = Uniform(-μas2rad(40.0), μas2rad(40.0)),
        )

        skym = SkyModel(closuregeom, prior, g)
        post = VLBIPosterior(skym, lcamp, cphase)
        tpost = asflat(post)
        x = prior_sample(tpost)
        dx = Enzyme.make_zero(x)

        autodiff(set_runtime_activity(Reverse), logdensityof, Const(tpost), Duplicated(x, dx))

        fdm = central_fdm(5, 1)
        gf = first(grad(fdm, tpost, x))
        @test isapprox(dx, gf; atol = 1.0e-8, rtol = 1.0e-5)
    end

    # This is currently broken on Julia 1.11 #2750
    if VERSION < v"1.11"
        @testset "Polarized Imaging" begin
            function polarizedsky(θ, metadata)
                (; σs, as, ain, aout, r) = θ
                (; grid, ftot) = metadata
                δs = ntuple(Val(4)) do i
                    σs[i] * as[i].params
                end

                pmap = VLBISkyModels.PolExp2Map!(δs..., grid)

                mmodel = modify(RingTemplate(RadialDblPower(ain, aout), AzimuthalUniform()), Stretch(r))
                mimg = intensitymap(mmodel, grid)

                ft = zero(eltype(mimg))
                for i in eachindex(pmap, mimg)
                    pmap[i] *= mimg[i]
                    ft += pmap[i].I
                end

                pmap .= ftot .* pmap ./ ft
                x0, y0 = centroid(pmap)
                m = ContinuousImage(pmap, BSplinePulse{3}())
                return shifted(m, -x0, -y0)
            end

            fovx = μas2rad(60.0)
            fovy = μas2rad(60.0)
            nx = ny = 8
            grid = imagepixels(fovx, fovy, nx, ny)
            skymeta = (; grid, ftot = 0.6)

            cprior = corr_image_prior(grid, dcoh; order = 2)
            skyprior = (
                σs = ntuple(Returns(truncated(Normal(0.0, 0.5); lower = 0.0)), 4),
                as = ntuple(Returns(cprior), 4),
                ain = Uniform(0.0, 5.0),
                aout = Uniform(0.0, 5.0),
                r = Uniform(μas2rad(1.0), μas2rad(30.0)),
            )
            skym = SkyModel(polarizedsky, skyprior, grid; metadata = skymeta)

            function fgain(x)
                gR = exp(x.lgR + 1im * x.gpR)
                gL = gR * exp(x.lgrat + 1im * x.gprat)
                return gR, gL
            end
            G = JonesG(fgain)

            function fdterms(x)
                dR = complex(x.dRx, x.dRy)
                dL = complex(x.dLx, x.dLy)
                return dR, dL
            end
            D = JonesD(fdterms)
            R = JonesR(; add_fr = true)

            js(g, d, r) = adjoint(r) * g * d * r
            J = JonesSandwich(js, G, D, R)

            intprior = (
                lgR = ArrayPrior(IIDSitePrior(ScanSeg(), Normal(0.0, 0.2)); LM = IIDSitePrior(ScanSeg(), Normal(0.0, 1.0))),
                lgrat = ArrayPrior(IIDSitePrior(ScanSeg(), Normal(0.0, 0.1))),
                gpR = ArrayPrior(IIDSitePrior(ScanSeg(), DiagonalVonMises(0.0, inv(π^2))); refant = SEFDReference(0.0), phase = true),
                gprat = ArrayPrior(IIDSitePrior(ScanSeg(), DiagonalVonMises(0.0, inv(0.1^2))); refant = SingleReference(:AA, 0.0), phase = true),
                dRx = ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
                dRy = ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
                dLx = ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
                dLy = ArrayPrior(IIDSitePrior(TrackSeg(), Normal(0.0, 0.2))),
            )
            intmodel = InstrumentModel(J, intprior)
            # Only do this for a small chunk to keep test time reasonable
            post = VLBIPosterior(skym, intmodel, filter(x -> 1.0 < x.baseline.Ti < 3.0, dcoh))
            tpost = asflat(post)

            x = prior_sample(tpost)
            logdensityof(tpost, x)
            dx = Enzyme.make_zero(x)
            autodiff(set_runtime_activity(Reverse), logdensityof, Const(tpost), Duplicated(x, dx))


            fdm = central_fdm(5, 1)
            gf = first(grad(fdm, tpost, x))
            @test isapprox(dx, gf; atol = 1.0e-8, rtol = 1.0e-5)
        end
    end

end
