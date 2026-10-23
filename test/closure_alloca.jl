using Enzyme, Test

# Julia >= 1.12 emits the stack slot of an immutable struct (here the closure built in
# the loop) as an untyped `[N x i64]` alloca. Copying a struct with padding into it
# (the `Bool` of `ClosureFlag` is followed by 7 padding bytes) left Enzyme unable to
# type those words (EnzymeAD/Enzyme.jl#3757).
struct ClosureFlag
    static::Bool
end
struct ClosureWrapped{A}
    flag::ClosureFlag
    data::A
end

@noinline function closure_step!(w)
    d = w.data
    @inbounds for j in eachindex(d)
        d[j] = sin(d[j])
    end
    return nothing
end
@noinline closure_callit(f) = (f(); nothing)

function closure_cost(w, n)
    J = 0.0
    for _ in 1:n
        closure_callit(() -> closure_step!(w))
        J += sum(w.data)
    end
    return J
end

function closure_cost_ref(w, n)
    J = 0.0
    for _ in 1:n
        closure_step!(w)
        J += sum(w.data)
    end
    return J
end

@testset "Closure capturing a struct with padding, built in a loop" begin
    make() = ClosureWrapped(ClosureFlag(true), view(collect(1.0:4.0), 1:4))
    w = make()
    dw = make_zero(w)
    autodiff(set_runtime_activity(Reverse), closure_cost, Active, Duplicated(w, dw), Const(2))
    wref = make()
    dwref = make_zero(wref)
    autodiff(set_runtime_activity(Reverse), closure_cost_ref, Active, Duplicated(wref, dwref), Const(2))
    @test any(!iszero, dw.data)
    @test dw.data ≈ dwref.data
end
