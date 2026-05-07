using Enzyme
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
end
