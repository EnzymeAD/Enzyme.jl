using Enzyme, Test, JLArrays

function jlres(x)
    return 2 * collect(x)
end

@testset "JLArrays" begin
    # TODO fix activity of jlarray
    # Enzyme.jacobian(Forward, jlres, JLArray([3.0, 5.0]))
    # Enzyme.jacobian(Reverse, jlres, JLArray([3.0, 5.0]))
end
