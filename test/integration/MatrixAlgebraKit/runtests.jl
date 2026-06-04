using Test, MatrixAlgebraKit, Random, LinearAlgebra, Enzyme, EnzymeTestUtils

function call_and_zero!(f!, A, alg)
    F′ = f!(A, alg)
    MatrixAlgebraKit.zero!(A)
    return F′
end

precision(::Type{T}) where {T <: Number} = sqrt(eps(real(T)))
precision(::Type{T}) where {T} = precision(eltype(T))

rng = Random.default_rng()

structured_randn!(A::AbstractMatrix) = randn!(A)
structured_randn!(A::Diagonal) = (randn!(diagview(A)); return A)

instantiate_matrix(::Type{T}, size) where {T <: Number} = randn(rng, T, size)
instantiate_matrix(::Type{AT}, size) where {AT <: Diagonal} = Diagonal(randn(rng, eltype(AT), size))

@testset "lq" for T in (Float64, ComplexF64), sz in ((19, 17), (19, 19), (19, 23)) 
    A = instantiate_matrix(T, sz)
    m, n = sz
    atol = m * n * precision(T)
    rtol = m * n * precision(T)
    alg = MatrixAlgebraKit.select_algorithm(lq_compact, A)
    LQ = lq_compact(A)
    ΔLQ = structured_randn!.(similar.(LQ))
    MatrixAlgebraKit.remove_lq_gauge_dependence!(ΔLQ..., A, LQ...)
    test_reverse(lq_compact, Duplicated, (A, Duplicated), (alg, Const); atol, rtol, output_tangent = ΔLQ)
    test_reverse(call_and_zero!, Duplicated, (lq_compact!, Const), (A, Duplicated), (alg, Const); atol, rtol, output_tangent = ΔLQ)

    if sz[1] == sz[2]
        A = instantiate_matrix(Diagonal{T}, sz)
        alg = MatrixAlgebraKit.select_algorithm(lq_compact, A)
        LQ = lq_compact(A)
        ΔLQ = structured_randn!.(similar.(LQ))
        MatrixAlgebraKit.remove_lq_gauge_dependence!(ΔLQ..., A, LQ...)
        test_reverse(lq_compact, Duplicated, (A, Duplicated), (alg, Const); atol, rtol, output_tangent = ΔLQ)
        test_reverse(call_and_zero!, Duplicated, (lq_compact!, Const), (A, Duplicated), (alg, Const); atol, rtol, output_tangent = ΔLQ)
    end
end

@testset "qr" for T in (Float64, ComplexF64), sz in ((19, 17), (19, 19), (19, 23)) 
    A = instantiate_matrix(T, sz)
    alg = MatrixAlgebraKit.select_algorithm(qr_compact, A)
    m, n = sz
    atol = m * n * precision(T)
    rtol = m * n * precision(T)
    QR = qr_compact(A)
    ΔQR = structured_randn!.(similar.(QR))
    MatrixAlgebraKit.remove_lq_gauge_dependence!(ΔQR..., A, QR...)
    test_reverse(qr_compact, Duplicated, (A, Duplicated), (alg, Const); atol, rtol, output_tangent = ΔQR)
    test_reverse(call_and_zero!, Duplicated, (qr_compact!, Const), (A, Duplicated), (alg, Const); atol, rtol, output_tangent = ΔQR)
    
    if sz[1] == sz[2]
        A = instantiate_matrix(Diagonal{T}, sz)
        alg = MatrixAlgebraKit.select_algorithm(qr_compact, A)
        QR = qr_compact(A)
        ΔQR = structured_randn!.(similar.(QR))
        MatrixAlgebraKit.remove_qr_gauge_dependence!(ΔQR..., A, QR...)
        test_reverse(qr_compact, Duplicated, (A, Duplicated), (alg, Const); atol, rtol, output_tangent = ΔQR)
        test_reverse(call_and_zero!, Duplicated, (qr_compact!, Const), (A, Duplicated), (alg, Const); atol, rtol, output_tangent = ΔQR)
    end
end

@testset "svd" for T in (Float64, ComplexF64), sz in ((19, 17), (19, 19), (19, 23)) 
    A = instantiate_matrix(T, sz)
    m, n = sz
    atol = m * n * precision(T)
    rtol = m * n * precision(T)
    alg = MatrixAlgebraKit.select_algorithm(svd_compact, A)
    USVᴴ = svd_compact(A)
    U, S, Vᴴ = USVᴴ
    ΔU, ΔS, ΔVᴴ = structured_randn!.(similar.((U, S, Vᴴ)))
    ΔU, ΔVᴴ = MatrixAlgebraKit.remove_svd_gauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ)
    ΔUSVᴴ = (ΔU, ΔS, ΔVᴴ)
    test_reverse(svd_compact, Duplicated, (A, Duplicated), (alg, Const); atol, rtol, output_tangent = ΔUSVᴴ)
    test_reverse(call_and_zero!, Duplicated, (svd_compact!, Const), (A, Duplicated), (alg, Const); atol, rtol, output_tangent = ΔUSVᴴ)

    #=if sz[1] == sz[2]
        A = instantiate_matrix(Diagonal{T}, sz)
        alg = MatrixAlgebraKit.select_algorithm(svd_compact, A)
        USVᴴ = svd_compact(A)
        ΔU, ΔS, ΔVᴴ = structured_randn!.(similar.((U, S, Vᴴ)))
        ΔU, ΔVᴴ = MatrixAlgebraKit.remove_svd_gauge_dependence!(ΔU, ΔVᴴ, U, S, Vᴴ)
        test_reverse(svd_compact, Duplicated, (A, Duplicated), (alg, Const); atol, rtol, output_tangent = ΔUSVᴴ)
        test_reverse(call_and_zero!, Duplicated, (svd_compact!, Const), (A, Duplicated), (alg, Const); atol, rtol, output_tangent = ΔUSVᴴ)
    end=# # currently broken, waiting on fix in MAK 0.6.9
end
