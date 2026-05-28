using Test, Pkg

Pkg.develop("MatrixAlgebraKit")
Pkg.test("MatrixAlgebraKit"; test_args = ["enzyme"])
