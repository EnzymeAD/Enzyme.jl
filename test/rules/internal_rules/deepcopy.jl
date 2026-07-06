using Enzyme
using Enzyme.EnzymeRules
using Test

@testset "core fallback traversal" begin
    @testset "Dict metadata" begin
        mutable struct CustomStateFallback
            values::Vector{Float64}
            metadata::Dict{Symbol, Any}
        end

        function loss_fallback(p)
            s = CustomStateFallback(p .* 2.0, deepcopy(Dict{Symbol, Any}(:k => "v")))
            return sum(s.values)
        end

        p = [1.0, 2.0]
        grad = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), loss_fallback, p)
        @test only(grad) ≈ [2.0, 2.0]
    end

    @testset "circular reference" begin
        mutable struct CyclicNode
            val::Vector{Float64}
            next::Any # Can be nothing or CyclicNode
        end

        function loss_cyclic(p)
            # Create a cycle: n1 -> n2 -> n1
            n1 = CyclicNode(p .* 3.0, nothing)
            n2 = CyclicNode(p .* 5.0, n1)
            n1.next = n2
            
            n1_copied = deepcopy(n1)
            # Both nodes are visited via recursion if cycle not memoized
            return sum(n1_copied.val) + sum(n1_copied.next.val)
        end

        p = [1.0, 1.0]
        grad = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), loss_cyclic, p)
        @test only(grad) ≈ [8.0, 8.0]
    end

    @testset "multiple nested structs" begin
        mutable struct Leaf
            x::Vector{Float64}
        end
        mutable struct Branch
            leaf::Leaf
            coef::Float64
        end
        mutable struct Tree
            branch::Branch
        end

        function loss_tree(p)
            t = Tree(Branch(Leaf(p .* 10.0), 3.0))
            t_copied = deepcopy(t)
            return sum(t_copied.branch.leaf.x) * t_copied.branch.coef
        end

        p = [1.0, 2.0]
        grad = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), loss_tree, p)
        # sum(10p) * 3 = 30 * sum(p) -> gradient is [30.0, 30.0]
        @test only(grad) ≈ [30.0, 30.0]
    end

    @testset "tuple and immutable accumulation" begin
        mutable struct PTupleAccum
            tspan::Tuple{Float64, Float64}
            u0::Vector{Float64}
        end

        f_tuple(x) = (q = deepcopy(PTupleAccum((0.0, 1.0), [x, 2x])); sum(q.u0))
        grad = Enzyme.gradient(Enzyme.Reverse, f_tuple, 2.0)
        @test grad[1] ≈ 3.0

        struct ImmutableSub
            val::Float64
        end
        mutable struct PImmutableAccum
            sub::ImmutableSub
            u0::Vector{Float64}
        end
        f_immutable(x) = (q = deepcopy(PImmutableAccum(ImmutableSub(5.0), [x, 2x])); sum(q.u0) + q.sub.val)
        grad2 = Enzyme.gradient(Enzyme.Reverse, f_immutable, 2.0)
        @test grad2[1] ≈ 3.0
    end

    @testset "runtime activity forward-mode deepcopy" begin
        struct Wrap
            v::Vector{Float64}
            tag::Int
        end

        f(w) = Base.deepcopy(w)
        w  = Wrap([1.0, 2.0], 7)
        dw = Wrap([1.0, 0.0], 0)
        
        res = Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Forward), Enzyme.Const(f), Enzyme.Duplicated, Enzyme.Duplicated(w, dw))
        @test res[1].v ≈ [1.0, 0.0]
    end

    @testset "forward-mode deepcopy with Const argument" begin
        f(x) = Base.deepcopy(x)
        x = [1.0, 2.0]

        # A Const argument has no shadow, but a Duplicated result may still be
        # requested (runtime-activity widening). The shadow of an inactive input
        # is zero. This exercises the issue MWE through the public API.
        for mode in (Enzyme.Forward, Enzyme.set_runtime_activity(Enzyme.Forward))
            res = Enzyme.autodiff(mode, Enzyme.Const(f), Enzyme.Duplicated, Enzyme.Const(x))
            @test res[1] ≈ [0.0, 0.0]
        end

        # Drive the rule methods directly to cover the Batch and NoNeed variants
        # the public API collapses to width-1 Duplicated when all args are Const.
        cfg1 = EnzymeRules.FwdConfig{true, true, 1, false, false}()
        cfg2 = EnzymeRules.FwdConfig{true, true, 2, false, false}()
        fc = Enzyme.Const(Base.deepcopy)
        xc = Enzyme.Const(x)

        dup = EnzymeRules.forward(cfg1, fc, Enzyme.Duplicated, xc)
        @test dup.val ≈ x
        @test dup.dval ≈ [0.0, 0.0]
        @test dup.val !== x

        dupnn = EnzymeRules.forward(cfg1, fc, Enzyme.DuplicatedNoNeed, xc)
        @test dupnn ≈ [0.0, 0.0]

        bdup = EnzymeRules.forward(cfg2, fc, Enzyme.BatchDuplicated, xc)
        @test bdup.val ≈ x
        @test length(bdup.dval) == 2
        @test all(d -> d ≈ [0.0, 0.0], bdup.dval)

        bdupnn = EnzymeRules.forward(cfg2, fc, Enzyme.BatchDuplicatedNoNeed, xc)
        @test length(bdupnn) == 2
        @test all(d -> d ≈ [0.0, 0.0], bdupnn)
    end
end
