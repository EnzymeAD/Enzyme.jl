using Enzyme, Test
using Statistics


@testset "GC" begin
    function gc_alloc(x)  # Basically g(x) = x^2
        a = Array{Float64, 1}(undef, 10)
        for n in 1:length(a)
            a[n] = x^2
        end
        return mean(a)
    end
    @test autodiff(Reverse, gc_alloc, Active, Active(5.0))[1][1] ≈ 10
    @test autodiff(Forward, gc_alloc, Duplicated(5.0, 1.0))[1] ≈ 10

    A = Float64[2.0, 3.0]
    B = Float64[4.0, 5.0]
    dB = Float64[0.0, 0.0]
    f = (X, Y) -> sum(X .* Y)
    Enzyme.autodiff(Reverse, f, Active, Const(A), Duplicated(B, dB))

    function gc_copy(x)  # Basically g(x) = x^2
        a = x * ones(10)
        for n in 1:length(a)
            a[n] = x^2
        end
        return mean(a)
    end

    @test Enzyme.autodiff(Reverse, gc_copy, Active, Active(5.0))[1][1] ≈ 10
    @test Enzyme.autodiff(Forward, gc_copy, Duplicated(5.0, 1.0))[1] ≈ 10
end

@testset "GCPreserve" begin
    function f(x, y)
        GC.@preserve x y begin
            ccall(:memcpy, Cvoid,
                (Ptr{Float64},Ptr{Float64},Csize_t), x, y, 8)
        end
        nothing
    end
    autodiff(Reverse, f, Duplicated([1.0], [0.0]), Duplicated([1.0], [0.0]))
    autodiff(Forward, f, Duplicated([1.0], [0.0]), Duplicated([1.0], [0.0]))
end

@testset "Copy" begin
    function advance(u_v_eta)
        eta = copy(u_v_eta)
        return @inbounds eta[1]
    end

    u_v_eta = [0.0]
    ad_struct = [1.0]

    autodiff(Reverse, advance, Active, Duplicated(u_v_eta, ad_struct))
    @test ad_struct[1] ≈ 2.0

    function advance2(u_v_eta)
        eta = copy(u_v_eta)
        return @inbounds eta[1][]
    end

    u_v_eta = [Ref(0.0)]
    ad_struct = [Ref(1.0)]

    autodiff(Reverse, advance2, Active, Duplicated(u_v_eta, ad_struct))
    @test ad_struct[1][] ≈ 2.0


    function incopy(u_v_eta, val, i)
        eta = copy(u_v_eta)
        eta[1] = val
        return @inbounds eta[i]
    end

    u_v_eta = [0.0]

    v = autodiff(Reverse, incopy, Active, Const(u_v_eta), Active(3.14), Const(1))[1][2]
    @test v ≈ 1.0
    @test u_v_eta[1] ≈ 0.0

    function incopy2(val, i)
        eta = Float64[2.3]
        eta[1] = val
        return @inbounds eta[i]
    end

    v = autodiff(Reverse, incopy2, Active, Active(3.14), Const(1))[1][1]
    @test v ≈ 1.0
end


@testset "GCPreserve2" begin
    function f!(a_out, a_in)
           a_out[1:end-1] .= a_in[2:end]
           return nothing
    end
    a_in = rand(4)
    a_out = a_in

    shadow_a_out = ones(4)
    shadow_a_in = shadow_a_out

    autodiff(Reverse, f!, Const, Duplicated(a_out, shadow_a_out), Duplicated(a_in, shadow_a_in))

    @test shadow_a_in ≈ Float64[0.0, 1.0, 1.0, 2.0]
    @test shadow_a_out ≈ Float64[0.0, 1.0, 1.0, 2.0]

    autodiff(Forward, f!, Const, Duplicated(a_out, shadow_a_out), Duplicated(a_in, shadow_a_in))

    @test shadow_a_in ≈ Float64[1.0, 1.0, 2.0, 2.0]
    @test shadow_a_out ≈ Float64[1.0, 1.0, 2.0, 2.0]
end

@testset "Return GC error" begin
	t = 0.0

	function tobedifferentiated(cond, a)::Float64
		if cond
			t + t
		else
			0.0
		end
	end

    @test 0.0 ≈ autodiff(Reverse, tobedifferentiated, Const(true), Active(2.1))[1][2]
	@test 0.0 ≈ autodiff(Forward, tobedifferentiated, Const(true), Duplicated(2.1, 1.0))[1]

	function tobedifferentiated2(cond, a)::Float64
		if cond
			a + t
		else
			0.0
		end
	end

    @test 1.0 ≈ autodiff(Reverse, tobedifferentiated2, Const(true), Active(2.1))[1][2]
	@test 1.0 ≈ autodiff(Forward, tobedifferentiated2, Const(true), Duplicated(2.1, 1.0))[1]

    @noinline function copy(dest, p1, cond)
        bc = convert(Broadcast.Broadcasted{Nothing}, Broadcast.instantiate(p1))

        if cond
            return nothing
        end

        bc2 = Broadcast.preprocess(dest, bc)
        @inbounds    dest[1] = bc2[1]

        nothing
    end

    function mer(F, F_H, cond)
        p1 = Base.broadcasted(Base.identity, F_H)
        copy(F, p1, cond)

        # Force an apply generic
        flush(stdout)
        nothing
    end

    L_H = Array{Float64, 1}(undef, 2)
    L = Array{Float64, 1}(undef, 2)

    F_H = [1.0, 0.0]
    F = [1.0, 0.0]

    autodiff(Reverse, mer, Duplicated(F, L), Duplicated(F_H, L_H), Const(true))
    autodiff(Forward, mer, Duplicated(F, L), Duplicated(F_H, L_H), Const(true))
end

@testset "GC Sret" begin
    @noinline function _get_batch_statistics(x)
        batchmean = @inbounds x[1]
        return (x, x)
    end

    @noinline function _normalization_impl(x)
        _stats = _get_batch_statistics(x)
        return x
    end

    function gcloss(x)
        _normalization_impl(x)[1]
        return nothing
    end

    x = randn(10)
    dx = zero(x)

    Enzyme.autodiff(Reverse, gcloss, Duplicated(x, dx))
end

typeunknownvec = Float64[]

@testset "GC Sret 2" begin

    struct AGriddedInterpolation{K<:Tuple{Vararg{AbstractVector}}} <: AbstractArray{Float64, 1}
        knots::K
        v::Int
    end

    function AGriddedInterpolation(A::AbstractArray{Float64, 1})
        knots = (A,)
        use(A)
        AGriddedInterpolation{typeof(knots)}(knots, 2)
    end

    function ainterpolate(A::AbstractArray{Float64,1})
        AGriddedInterpolation(A)
    end

    function cost(C::Vector{Float64})
        zs = typeunknownvec
        ainterpolate(zs)
        return nothing
    end

    A = Float64[]
    dA = Float64[]
    @test_throws Base.UndefVarError autodiff(Reverse, cost, Const, Duplicated(A, dA))
end

@testset "No Decayed / GC" begin
    @noinline function deduplicate_knots!(knots)
        last_knot = first(knots)
        for i = eachindex(knots)
            if i == 1
                continue
            end
            if knots[i] == last_knot
                @warn knots[i]
                @inbounds knots[i] *= knots[i]
            else
                last_knot = @inbounds knots[i]
            end
        end
    end

    function cost(C::Vector{Float64})
        deduplicate_knots!(C)
        @inbounds C[1] = 0
        return nothing
    end
    A = Float64[1, 3, 3, 7]
    dA = Float64[1, 1, 1, 1]
    autodiff(Reverse, cost, Const, Duplicated(A, dA))
    @test dA ≈ [0.0, 1.0, 6.0, 1.0]
end

@testset "Split GC" begin
    @noinline function bmat(x)
        data = [x]
        return data
    end

    function f(x::Float64)
        @inbounds return bmat(x)[1]
    end
    @test 1.0 ≈ autodiff(Reverse, f, Active(0.1))[1][1]
    @test 1.0 ≈ autodiff(Forward, f, Duplicated(0.1, 1.0))[1]
end

@testset "Recursive GC" begin
    function modf!(a)
        as = [zero(a) for _ in 1:2]
        a .+= sum(as)
        return nothing
    end

    a = rand(5)
    da = zero(a)
    autodiff(Reverse, modf!, Duplicated(a, da))
end

const CUmemoryPool2 = Ptr{Float64} 

struct CUmemPoolProps2
    reserved::NTuple{31,Char}
end

mutable struct CuMemoryPool2
    handle::CUmemoryPool2
end

function ccall_macro_lower(func, rettype, types, args, nreq)
    # instead of re-using ccall or Expr(:foreigncall) to perform argument conversion,
    # we need to do so ourselves in order to insert a jl_gc_safe_enter|leave
    # just around the inner ccall

    cconvert_exprs = []
    cconvert_args = []
    for (typ, arg) in zip(types, args)
        var = gensym("$(func)_cconvert")
        push!(cconvert_args, var)
        push!(cconvert_exprs, quote
            $var = Base.cconvert($(esc(typ)), $(esc(arg)))
        end)
    end

    unsafe_convert_exprs = []
    unsafe_convert_args = []
    for (typ, arg) in zip(types, cconvert_args)
        var = gensym("$(func)_unsafe_convert")
        push!(unsafe_convert_args, var)
        push!(unsafe_convert_exprs, quote
            $var = Base.unsafe_convert($(esc(typ)), $arg)
        end)
    end

    quote
        $(cconvert_exprs...)

        $(unsafe_convert_exprs...)

        ret = ccall($(esc(func)), $(esc(rettype)), $(Expr(:tuple, map(esc, types)...)),
                    $(unsafe_convert_args...))
    end
end

macro gcsafe_ccall(expr)
    ccall_macro_lower(Base.ccall_macro_parse(expr)...)
end

function cuMemPoolCreate2(pool, poolProps)
    # CUDA.initialize_context()
    #CUDA.
    gc_state = @ccall(jl_gc_safe_enter()::Int8)
    @gcsafe_ccall cuMemPoolCreate(pool::Ptr{CUmemoryPool2},
                                          poolProps::Ptr{CUmemPoolProps2})::Cvoid
    @ccall(jl_gc_safe_leave(gc_state::Int8)::Cvoid)
end

function cual()
        props = Ref(CUmemPoolProps2( 
            ntuple(i->Char(0), 31)
        ))
        handle_ref = Ref{CUmemoryPool2}()
        cuMemPoolCreate2(handle_ref, props)

        CuMemoryPool2(handle_ref[])
end

@testset "Unused shadow phi rev" begin
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(cual)}, Duplicated)
end

