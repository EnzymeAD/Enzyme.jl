using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    precompile_module = @eval module $(gensym())
        f(x) = x^2
    end
   
    Compiler.JIT.setup_globals()

    @compile_workload begin
        Enzyme.autodiff(Reverse, precompile_module.f, Active(2.0))
    end

    # Everything the workload cached belongs to this process: thunks are held as JIT
    # addresses, the rule and activity memos are keyed on worlds that mean nothing once the
    # image is loaded elsewhere. Serializing them hands every session that loads Enzyme a
    # cache it must not use, so drop them and leave the image with none.
    Compiler.clear_caches!()
end
