using Enzyme, Test

jit_cube(x) = x * x * x

# Thunks of one function for different jobs. Since JuliaLang/julia#60031, Julia's codegen gives
# all of them the same entry name.
jit_sum_of_thunks(x) = autodiff(Forward, jit_cube, Duplicated(x, 1.0))[1] +
    autodiff(Forward, jit_cube, Duplicated(Float32(x), 1.0f0))[1] +
    autodiff(Reverse, jit_cube, Active(x))[1][1]

@testset "Thunks with the same entry name" begin
    @test autodiff(Reverse, jit_cube, Active(2.0))[1][1] ≈ 12.0
    @test autodiff(Reverse, jit_cube, Active(2.0f0))[1][1] ≈ 12.0f0
    @test autodiff(Reverse, jit_cube, Const(2.0))[1][1] === nothing
    @test jit_sum_of_thunks(2.0) ≈ 36.0

    # One compilation imports the three thunks.
    @test autodiff(Reverse, jit_sum_of_thunks, Active(2.0))[1][1] ≈ 36.0
    @test autodiff(Forward, jit_sum_of_thunks, Duplicated(2.0, 1.0))[1] ≈ 36.0
end

jit_unstable(x) = Base.inferencebarrier(x)[1]^2

@testset "InlineABI resolves the globals of Enzyme" begin
    # Julia's JIT compiles this thunk. The thunk refers to Julia values that Enzyme defines
    # in the JIT, like `ejl_enz_runtime_generic_rev`.
    x = [2.0]
    dx = [0.0]
    autodiff(Enzyme.set_abi(Reverse, Enzyme.InlineABI), jit_unstable, Active, Duplicated(x, dx))
    @test dx ≈ [4.0]
end

@static if Enzyme.Compiler.HAS_GPUCOMPILER_2
    using GPUCompiler: CompilerJob, CompilerConfig, relocation_namespace

    namespace_f(x) = sin(x) * x

    function namespace_of(mode, rt, args; runtimeActivity = false, modifiedBetween = (false, false))
        mi = Enzyme.Compiler.my_methodinstance(nothing, typeof(namespace_f), Tuple{Float64})
        params = Enzyme.Compiler.EnzymeCompilerParams(
            Tuple{Const{typeof(namespace_f)}, args}, mode, 1, rt, true, true,
            modifiedBetween, false, false, Nothing, Enzyme.FFIABI, false, runtimeActivity, false
        )
        config = CompilerConfig(Enzyme.Compiler.EnzymeTarget(), params; kernel = false)
        return relocation_namespace(CompilerJob(mi, config, Base.get_world_counter()))
    end

    @testset "relocation_namespace has the Enzyme options" begin
        API = Enzyme.API
        rev = namespace_of(API.DEM_ReverseModeCombined, Active{Float64}, Active{Float64})
        fwd = namespace_of(API.DEM_ForwardMode, Duplicated{Float64}, Duplicated{Float64})
        ra = namespace_of(
            API.DEM_ReverseModeCombined, Active{Float64}, Active{Float64}; runtimeActivity = true
        )
        @test rev == "_Z11namespace_f7Float64_rev_w1_rA_aCA_FFIABI_aw_o3"
        @test fwd == "_Z11namespace_f7Float64_fwd_w1_rD_aCD_FFIABI_aw_o3"
        @test ra == "_Z11namespace_f7Float64_rev_w1_ra_rA_aCA_FFIABI_aw_o3"
        overwritten = namespace_of(
            API.DEM_ReverseModeCombined, Active{Float64}, Active{Float64};
            modifiedBetween = (true, false)
        )
        @test overwritten == "_Z11namespace_f7Float64_rev_w1_rA_aCA_FFIABI_aw_o2"
        @test Enzyme.Compiler.hex_mask((false, false, false, false, true)) == "01"
        @test Enzyme.Compiler.hex_mask((true, true)) == "3"
        @test Enzyme.Compiler.hex_mask(()) == ""
    end
end
