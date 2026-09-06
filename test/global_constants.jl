using Enzyme
using Libdl
using Test

const LLVM_IR = raw"""
; ModuleID = '<stdin>'
source_filename = "<string>"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-conda-linux-gnu"


@A = private unnamed_addr constant [3 x double] [double 1.000000e+00, double 2.000000e+00, double 3.000000e+00], align 8

define double @func(double %x, double %y, i64 %n) {
entry:
  %ptr  = getelementptr inbounds [3 x double], [3 x double]* @A, i64 0, i64 %n
  %aval = load double, double* %ptr, align 8
  %prod = fmul double %x, %aval
  %sum  = fadd double %prod, %y
  ret double %sum
}
"""

tmp_dir = tempdir()
tmp_so_file = joinpath(tmp_dir, "func.so")

run(
    pipeline(
        `clang -x ir - -Xclang -no-opaque-pointers -O3 -fPIC -fembed-bitcode -shared -o $(tmp_so_file)`;
        stdin=IOBuffer(LLVM_IR)
    )
);
lib = Libdl.dlopen(tmp_so_file);
const fptr = Libdl.dlsym(lib, :func);


function func_llvm(x::Float64, y::Float64, n::Int)
    n >= 0 && n <= 2 || throw("0 ≤ n ≤ 2")
    Base.llvmcall((LLVM_IR, "func"), Cdouble,
        Tuple{Cdouble,Cdouble,Clong},
        x, y, n
    )
end;


function func_ccall(x::Float64, y::Float64, n::Int)
    n >= 0 && n <= 2 || throw("0 ≤ n ≤ 2")
    ccall(fptr, Cdouble,
        (Cdouble, Cdouble, Clong),
        x, y, n
    )
end;

@testset "Rename external global constants ccall" begin

    x = 2.0
    y = 1.0
    n = 2
    A = [1.0, 2.0, 3.0]

    @test func_llvm(x, y, n) == func_ccall(x, y, n)
    @test func_llvm(x, y, n) == x * A[n+1] + y
    @test func_ccall(x, y, n) == x * A[n+1] + y



    @test gradient(Reverse, func_llvm, Const(x), y, Const(n)) == (nothing, 1.0, nothing)
    @test gradient(Reverse, func_llvm, x, Const(y), Const(n)) == (3.0, nothing, nothing)

    @test gradient(Reverse, func_ccall, Const(x), y, Const(n)) == (nothing, 1.0, nothing)
    @test gradient(Reverse, func_ccall, x, Const(y), Const(n)) == (3.0, nothing, nothing)
end
