using Enzyme, Test

@testset "Generic Active Union Return" begin

    function generic_union_ret(A)
            if 0 < length(A)
                @inbounds A[1]
            else
                nothing
                Base._InitialValue()
            end
    end

    function outergeneric(weights::Vector{Float64})::Float64
        v = generic_union_ret(Base.inferencebarrier(weights))
        return v::Float64
    end

    weights = [0.2]
    dweights = [0.0]

    Enzyme.autodiff(Enzyme.Reverse, outergeneric, Enzyme.Duplicated(weights, dweights))

    @test dweights[1] ≈ 1.
end

function Valuation1(z,Ls1)
    @inbounds Ls1[1] = sum(Base.inferencebarrier(z))
    return nothing
end
@testset "Active setindex!" begin
    v=ones(5)
    dv=zero(v)

    DV1=Float32[0]
    DV2=Float32[1]

    Enzyme.autodiff(Reverse,Valuation1,Duplicated(v,dv),Duplicated(DV1,DV2))
    @test dv[1] ≈ 1.
    
    DV1=Float32[0]
    DV2=Float32[1]
    v=ones(5)
    dv=zero(v)
    dv[1] = 1.    
    Enzyme.autodiff(Forward,Valuation1,Duplicated(v,dv),Duplicated(DV1,DV2))
    @test DV2[1] ≈ 1.
end


mutable struct RTGData
	x
end

@noinline function rtg_sub(V, cv)
	return cv
end

@noinline function rtg_cast(cv)
	return cv
end

function rtg_f(V,@nospecialize(cv))
	s = rtg_sub(V, Base.inferencebarrier(cv))::RTGData
	s = rtg_cast(Base.inferencebarrier(s.x))::Float64
	return s
end

@testset "RuntimeActivity generic call" begin
    res = autodiff(set_runtime_activity(ForwardWithPrimal), rtg_f, Duplicated, Duplicated([0.2], [1.0]), Const(RTGData(3.14)))
    @test 3.14 ≈ res[2]
    @test 0.0 ≈ res[1]
end

@testset "Batched inactive" begin
    augres = Enzyme.Compiler.runtime_generic_augfwd(Val{(false, false, false)}, Val(false), Val(2), Val((true, true, true)),
                                                    Val(Enzyme.Compiler.AnyArray(2+Int(2))),
                                ==, nothing, nothing,
                                :foo, nothing, nothing,
                                :bar, nothing, nothing)

    Enzyme.Compiler.runtime_generic_rev(Val{(false, false, false)}, Val(false), Val(2), Val((true, true, true)), augres[end],
                                ==, nothing, nothing,
                                :foo, nothing, nothing,
                                :bar, nothing, nothing)
end

@testset "Static activity" begin

    struct Test2{T}
        obs::T
    end

    function test(t, x)
        o = t.obs
        y = (x .- o)
        yv = @inbounds y[1]
        return yv*yv
    end

    obs = [1.0]
    t = Test2(obs)

    x0 = [0.0]
    dx0 = [0.0]

    autodiff(Reverse, test, Const(t), Duplicated(x0, dx0))

    @test obs[1] ≈ 1.0
    @test x0[1] ≈ 0.0
    @test dx0[1] ≈ -2.0

end

@testset "Const Activity through intermediate" begin
    struct RHS_terms
        eta1::Vector{Float64}
        u_t::Vector{Float64}
        eta_t::Vector{Float64}
    end

    @noinline function comp_u_v_eta_t(rhs)
        Base.unsafe_copyto!(rhs.eta_t, 1, rhs.u_t, 1, 1)
        return nothing
    end

    function advance(eta, rhs)

        @inbounds rhs.eta1[1] = @inbounds eta[1]

        comp_u_v_eta_t(rhs)

        @inbounds eta[1] = @inbounds rhs.eta_t[1]

        return nothing

    end

    rhs_terms = RHS_terms(zeros(1), zeros(1), zeros(1))

    u_v_eta = Float64[NaN]
    ad_eta = zeros(1)

    autodiff(Reverse, advance,
        Duplicated(u_v_eta, ad_eta),
        Const(rhs_terms),
    )
    @test ad_eta[1] ≈ 0.0
end

