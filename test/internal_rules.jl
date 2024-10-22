
using Enzyme
using Enzyme.EnzymeRules
using LinearAlgebra
using SparseArrays
using Test
import Random

Enzyme.Compiler.DumpPostOpt[] = true

function chol_lower0(x)
  c = copy(x)
  C, info = LinearAlgebra.LAPACK.potrf!('L', c)
  return @inbounds c[2,1]
end

Enzyme.API.printall!(true)

x = reshape([1.0, -0.10541615131279458, 0.6219810761363638, 0.293343219811946, -0.10541615131279458, 1.0, -0.05258941747718969, 0.34629296878264443, 0.6219810761363638, -0.05258941747718969, 1.0, 0.4692436399208845, 0.293343219811946, 0.34629296878264443, 0.4692436399208845, 1.0], 4, 4)
 dL = zero(x)
 dL[2, 1] = 1.0

 @test Enzyme.autodiff(Forward, chol_lower0, Duplicated(x, dL))[1] ≈ 0.05270807565639164
