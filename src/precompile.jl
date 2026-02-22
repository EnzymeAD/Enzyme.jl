using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    precompile_module = @eval module $(gensym())
    f(x) = x^2
    end

    Compiler.JIT.setup_globals()

    @compile_workload begin
        Enzyme.autodiff(Reverse, precompile_module.f, Active(2.0))
    end
end
