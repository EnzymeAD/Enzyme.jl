module EnzymeJuMPExt

using Enzyme
using JuMP

function jump_operator(f)
    @inline function f!(y, x...)
        y[1] = f(x...)
    end
    function gradient!(g::AbstractVector{T}, x::Vararg{T,N}) where {T,N}
        y = zeros(1)
        ry = ones(1)
        rx = ntuple(N) do i
            Active(x[i])
        end
        g .= autodiff(ReverseWithPrimal, f!, Const, Duplicated(y,ry), rx...)[1][2:end]
        return nothing
    end

    function gradient_deferred!(g, y, ry, rx...)
        g .= autodiff_deferred(ReverseWithPrimal, f!, Const, Duplicated(y,ry), rx...)[1][2:end]
        return nothing
    end

    function hessian!(H::AbstractMatrix{T}, x::Vararg{T,N}) where {T,N}
        y = zeros(1)
        dg = zeros(N)
        y[1] = 0.0
        dy = ones(1)
        ry = ones(1)
        dry = zeros(1)
        g = zeros(N)
        dg = zeros(N)

        rx = ntuple(N) do i
            Active(x[i])
        end

        for j in 1:N
            y[1] = 0.0
            dy[1] = 1.0
            ry[1] = 1.0
            dry[1] = 0.0
            drx = ntuple(N) do i
                if i == j
                    Active(one(T))
                else
                    Active(zero(T))
                end
            end
            tdrx= ntuple(N) do i
                Duplicated(rx[i], drx[i])
            end
            fill!(dg, 0.0)
            fill!(g, 0.0)
            autodiff(Forward, gradient_deferred!, Const, Duplicated(g,dg), Duplicated(y,dy), Duplicated(ry, dry), tdrx...)
            for i in 1:N
                if i <= j
                    H[j,i] = dg[i]
                end
            end
        end

        return nothing
    end
    return gradient!, hessian!
end

function JuMP.add_nonlinear_operator(
    model::GenericModel,
    dim::Int,
    f::Function;
    name::Symbol = Symbol(f),
)
    gradient, hessian = jump_operator(f)
    @show tuple(f, gradient, hessian)
    MOI.set(model, MOI.UserDefinedFunction(name, dim), tuple(f, gradient, hessian))
    return NonlinearOperator(f, name)
end
end
