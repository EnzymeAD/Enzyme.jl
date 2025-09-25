@testset "deferred" begin

    @testset "Deferred and deferred thunk" begin
        function dot(A)
            return A[1] * A[1] + A[2] * A[2]
        end
        dA = zeros(2)
        A = [3.0, 5.0]
        thunk_dA, def_dA = copy(dA), copy(dA)
        def_A, thunk_A = copy(A), copy(A)
        primal = Enzyme.autodiff(ReverseWithPrimal, dot, Active, Duplicated(A, dA))[2]
        @test primal == 34.0
        primal = Enzyme.autodiff_deferred(ReverseWithPrimal, Const(dot), Active, Duplicated(def_A, def_dA))[2]
        @test primal == 34.0

        dup = Duplicated(thunk_A, thunk_dA)
        TapeType = Enzyme.EnzymeCore.tape_type(
            ReverseSplitWithPrimal,
            Const{typeof(dot)}, Active, Duplicated{typeof(thunk_A)}
        )
        @test Tuple{Float64, Float64} === TapeType
        Ret = Active
        fwd, rev = Enzyme.autodiff_deferred_thunk(
            ReverseSplitWithPrimal,
            TapeType,
            Const{typeof(dot)},
            Ret,
            Duplicated{typeof(thunk_A)}
        )
        tape, primal, _ = fwd(Const(dot), dup)
        @test isa(tape, Tuple{Float64, Float64})
        rev(Const(dot), dup, 1.0, tape)
        @test all(primal == 34)
        @test all(dA .== [6.0, 10.0])
        @test all(dA .== def_dA)
        @test all(dA .== thunk_dA)

        function kernel(len, A)
            for i in 1:len
                A[i] *= A[i]
            end
        end

        A = Array{Float64}(undef, 64)
        dA = Array{Float64}(undef, 64)

        A .= (1:1:64)
        dA .= 1

        function aug_fwd(ctx, f::FT, ::Val{ModifiedBetween}, args...) where {ModifiedBetween, FT}
            TapeType = Enzyme.tape_type(ReverseSplitModified(ReverseSplitWithPrimal, Val(ModifiedBetween)), Const{Core.Typeof(f)}, Const, Const{Core.Typeof(ctx)}, map(Core.Typeof, args)...)
            forward, reverse = Enzyme.autodiff_deferred_thunk(ReverseSplitModified(ReverseSplitWithPrimal, Val(ModifiedBetween)), TapeType, Const{Core.Typeof(f)}, Const, Const{Core.Typeof(ctx)}, map(Core.Typeof, args)...)
            forward(Const(f), Const(ctx), args...)[1]
            return nothing
        end

        ModifiedBetween = Val((false, false, true))

        aug_fwd(64, kernel, ModifiedBetween, Duplicated(A, dA))

    end

    @testset "Deferred upgrade" begin
        function gradsin(x)
            return gradient(Reverse, sin, x)[1]
        end
        res = Enzyme.gradient(Reverse, gradsin, 3.1)[1]
        @test res ≈ -sin(3.1)
    end

end # testset "deferred"
