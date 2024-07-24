using Test, Enzyme, DynamicExpressions

operators = OperatorEnum(; binary_operators=(+, -, *, /), unary_operators=(cos, sin))

tree = Node(; op=1, l=Node{Float64}(; feature=1), r=Node(; op=1, l=Node{Float64}(; feature=2)))
# == x1 + cos(x2)

X = randn(3, 100);
dX = zero(X)

function f(tree, X, operators, output)
    output[] = sum(eval_tree_array(tree, X, operators)[1])
    return nothing
end

output = [0.0]
doutput = [1.0]

autodiff(
    Reverse,
    f,
    Const(tree),
    Duplicated(X, dX),
    Const(operators),
    Duplicated(output, doutput),
)

true_dX = cat(ones(100), -sin.(X[2, :]), zeros(100); dims=2)'

@test true_dX ≈ dX
