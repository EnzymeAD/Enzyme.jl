using Enzyme, Test
using Enzyme.Compiler: LLVM

# Use known Julia globals as leaves, so this fixture exercises the same
# abstract interpreter as compiled Julia IR without dereferencing fake objects.
function type_test_fixture(; first = "@jl_nothing", second = "@jl_true", target = UInt(0x00012345), predicate = "eq", swapped = false, bits = Sys.WORD_SIZE, chain = 0)
    tracked = LLVM.supports_typed_pointers(LLVM.context()) ? "{} addrspace(10)*" : "ptr addrspace(10)"
    casts = IOBuffer()
    source = "%selected"
    for i in 1:chain
        println(casts, "%chain$i = select i1 %choose, $tracked $source, $tracked $first")
        source = "%chain$i"
    end
    lhs, rhs = swapped ? (string(target), "%tagint") : ("%tagint", string(target))
    return parse(
        LLVM.Module, """
        @jl_nothing = external addrspace(10) global {}
        @jl_true = external addrspace(10) global {}
        @jl_false = external addrspace(10) global {}
        declare $tracked @julia.typeof($tracked)
        define i1 @test(i1 %choose, i1 %again, $tracked %unknown) {
        entry:
            br label %loop
        loop:
            %held = phi $tracked [ $first, %entry ], [ %selected, %loop ]
            %selected = select i1 %choose, $tracked %held, $tracked $second
            br i1 %again, label %loop, label %exit
        exit:
            $(String(take!(casts)))
            %tag = call $tracked @julia.typeof($tracked $source)
            %tagint = ptrtoint $tracked %tag to i$bits
            %test = icmp $predicate i$bits $lhs, $rhs
            ret i1 %test
        }
        """
    )
end

function folded_type_test_result(mod)
    for bb in LLVM.blocks(LLVM.functions(mod)["test"]), inst in LLVM.instructions(bb)
        inst isa LLVM.RetInst || continue
        value = LLVM.operands(inst)[1]
        return value isa LLVM.ConstantInt ? convert(Bool, value) : nothing
    end
    return nothing
end

@testset "Exact type tests through PHI/select cycles" begin
    LLVM.Context() do ctx
        # These tags come directly from the object headers, independently of
        # the compiler helper under test. Julia's small tags are not addresses.
        nothing_tag = unsafe_load(unsafe_load(cglobal(:jl_nothing, Ptr{UInt})) - sizeof(UInt)) & ~UInt(15)
        bool_tag = unsafe_load(unsafe_load(cglobal(:jl_true, Ptr{UInt})) - sizeof(UInt)) & ~UInt(15)
        @test Enzyme.Compiler.type_test_tag(Nothing) == nothing_tag
        @test Enzyme.Compiler.type_test_tag(Bool) == bool_tag
        @test Enzyme.Compiler.type_test_tag(Vector{Float64}) == UInt(Enzyme.unsafe_to_pointer(Vector{Float64}))

        for predicate in ("eq", "ne"), swapped in (false, true)
            # No input can have this address; it must never be dereferenced.
            mod = type_test_fixture(; predicate, swapped)
            @test Enzyme.Compiler.fold_type_tests!(mod, nothing) == 1
            @test folded_type_test_result(mod) === (predicate == "ne")
            LLVM.verify(mod)

            mod = type_test_fixture(; first = "@jl_false", second = "@jl_true", target = bool_tag, predicate, swapped)
            @test Enzyme.Compiler.fold_type_tests!(mod, nothing) == 1
            @test folded_type_test_result(mod) === (predicate == "eq")
            LLVM.verify(mod)
        end

        # A genuine mixed answer and an unknown leaf must retain the check.
        for options in ((; target = nothing_tag), (; second = "%unknown"), (; predicate = "ult"), (; bits = 8, target = UInt(3)), (; chain = 130))
            mod = type_test_fixture(; options...)
            @test Enzyme.Compiler.fold_type_tests!(mod, nothing) == 0
            @test folded_type_test_result(mod) === nothing
            LLVM.verify(mod)
        end
    end
end
