using Test, MatrixAlgebraKit, Random, LinearAlgebra

function call_and_zero!(f!, A, alg)
    F′ = f!(A, alg)
    MatrixAlgebraKit.zero!(A)
    return F′
end

structured_randn!(A::AbstractMatrix) = randn!(A)
structured_randn!(A::Diagonal) = (randn!(diagview(A)); return A)

instantiate_matrix(::Type{T}, size) where {T <: Number} = randn(rng, T, size)
instantiate_matrix(::Type{AT}, size) where {AT <: Diagonal} = Diagonal(randn(rng, eltype(AT), size))

@testset "lq" for T in (Float64, ComplexF64), sz in ((19, 17), (19, 19), (19, 23)) 
    A = instantiate_matrix(T, sz)
    alg = MatrixAlgebraKit.select_algorithm(lq_compact, A)
    LQ = lq_compact(A)
    ΔLQ = structured_randn!.(similar.(LQ))
    MatrixAlgebraKit.remove_lq_gauge_dependence!(ΔLQ..., A, LQ...)
    test_reverse(lq_compact, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔLQ)
    test_reverse(call_and_zero!, RT, (lq_compact!, Const), (A, TA), (alg, Const); atol, rtol, output_tangent = ΔLQ)

    if sz[1] == sz[2]
        A = instantiate_matrix(Diagonal{T}, sz)
        alg = MatrixAlgebraKit.select_algorithm(lq_compact, A)
        LQ = lq_compact(A)
        ΔLQ = structured_randn!.(similar.(LQ))
        MatrixAlgebraKit.remove_lq_gauge_dependence!(ΔLQ..., A, LQ...)
        test_reverse(lq_compact, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔLQ)
        test_reverse(call_and_zero!, RT, (lq_compact!, Const), (A, TA), (alg, Const); atol, rtol, output_tangent = ΔLQ)
    end
end

@testset "qr" for T in (Float64, ComplexF64), sz in ((19, 17), (19, 19), (19, 23)) 
    A = instantiate_matrix(T, sz)
    alg = MatrixAlgebraKit.select_algorithm(qr_compact, A)
    QR = qr_compact(A)
    ΔQR = structured_randn!.(similar.(QR))
    MatrixAlgebraKit.remove_lq_gauge_dependence!(ΔQR..., A, QR...)
    test_reverse(qr_compact, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔQR)
    test_reverse(call_and_zero!, RT, (qr_compact!, Const), (A, TA), (alg, Const); atol, rtol, output_tangent = ΔQR)
    
    if sz[1] == sz[2]
        A = instantiate_matrix(Diagonal{T}, sz)
        alg = MatrixAlgebraKit.select_algorithm(qr_compact, A)
        QR = qr_compact(A)
        ΔQR = structured_randn!.(similar.(QR))
        MatrixAlgebraKit.remove_qr_gauge_dependence!(ΔQR..., A, QR...)
        test_reverse(qr_compact, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔQR)
        test_reverse(call_and_zero!, RT, (qr_compact!, Const), (A, TA), (alg, Const); atol, rtol, output_tangent = ΔQR)
    end
end

@testset "svd" for T in (Float64, ComplexF64), sz in ((19, 17), (19, 19), (19, 23)) 
    A = instantiate_matrix(T, sz)
    alg = MatrixAlgebraKit.select_algorithm(svd_compact, A)
    USVᴴ = svd_compact(A)
    ΔU, ΔS, ΔVᴴ = structured_randn!.(similar.((U, S, Vᴴ)))
    ΔU, ΔVᴴ = MatrixAlgebraKit.remove_svd_gauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ)
    test_reverse(svd_compact, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔUSVᴴ)
    test_reverse(call_and_zero!, RT, (svd_compact!, Const), (A, TA), (alg, Const); atol, rtol, output_tangent = ΔUSVᴴ)

    if sz[1] == sz[2]
        A = instantiate_matrix(Diagonal{T}, sz)
        alg = MatrixAlgebraKit.select_algorithm(svd_compact, A)
        USVᴴ = svd_compact(A)
        ΔU, ΔS, ΔVᴴ = structured_randn!.(similar.((U, S, Vᴴ)))
        ΔU, ΔVᴴ = MatrixAlgebraKit.remove_svd_gauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ)
        test_reverse(svd_compact, RT, (A, TA), (alg, Const); atol, rtol, output_tangent = ΔUSVᴴ)
        test_reverse(call_and_zero!, RT, (svd_compact!, Const), (A, TA), (alg, Const); atol, rtol, output_tangent = ΔUSVᴴ)
    end
end
