using Enzyme
using Clang_jll
using Libdl
using Test

const FUNC_LLVM_IR = """
    declare double @llvm.rint.f64(double) #1

    define i32 @func(double* noalias nocapture writeonly %retptr, { i8*, i32, i8*, i8*, i32 }** noalias nocapture readnone %excinfo, double %arg.t, i8* nocapture readnone %arg.arr.0, i8* nocapture readnone %arg.arr.1, i64 %arg.arr.2, i64 %arg.arr.3, double* %arg.arr.4, i64 %arg.arr.5.0, i64 %arg.arr.6.0) local_unnamed_addr #0 {
    common.ret:
      %.27 = fdiv double %arg.t, 1.000000e-02
      %.28 = tail call double @llvm.rint.f64(double %.27)
      %.29 = fptosi double %.28 to i64
      %.42 = icmp slt i64 %.29, 0
      %.43 = select i1 %.42, i64 %arg.arr.5.0, i64 0
      %.44 = add i64 %.43, %.29
      %.55 = mul i64 %.44, %arg.arr.6.0
      %.56 = ptrtoint double* %arg.arr.4 to i64
      %.57 = add i64 %.55, %.56
      %.58 = inttoptr i64 %.57 to double*
      %.59 = load double, double* %.58, align 8
      store double %.59, double* %retptr, align 8
      ret i32 0
    }

    define double @func_wrap({ i8*, i32, i8*, i8*, i32 }** %excinfo, double %arg.t, i8* %arg.arr.0, i8* %arg.arr.1, i64 %arg.arr.2, i64 %arg.arr.3, double* %arg.arr.4, i64 %arg.arr.5.0, i64 %arg.arr.6.0) {
    entry:
        %tmp = alloca double, align 8
        %st  = call i32 @func(double* %tmp, { i8*, i32, i8*, i8*, i32 }** %excinfo, double %arg.t, i8* %arg.arr.0, i8* %arg.arr.1, i64 %arg.arr.2, i64 %arg.arr.3, double* %arg.arr.4, i64 %arg.arr.5.0, i64 %arg.arr.6.0)
        %val = load double, double* %tmp, align 8
        ret double %val
    }


    attributes #0 = { mustprogress nofree nosync nounwind willreturn }
    attributes #1 = { mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn }
    attributes #2 = { noinline }
"""


tmp_dir = tempdir()
tmp_so_file = joinpath(tmp_dir, "func.so")
run(
    pipeline(
        `$(clang()) -x ir - -Xclang -no-opaque-pointers -O3 -fPIC -fembed-bitcode -shared -o $(tmp_so_file)`;
        stdin=IOBuffer(FUNC_LLVM_IR)
    )
)

lib = Libdl.dlopen(tmp_so_file)
const fptr = Libdl.dlsym(lib, :func_wrap)


function func_ccall(t::Float64, arr::AbstractVector{Float64})
    nitems = length(arr)
    bitsize = Base.elsize(arr)
    GC.@preserve arr begin
        excinfo = Ptr{Ptr{Cvoid}}(C_NULL)
        base::Ptr{Cdouble} = pointer(arr)

        ccall(fptr, Cdouble,
            (Ptr{Ptr{Cvoid}}, Cdouble, Ptr{Cvoid}, Ptr{Cvoid},
                Clong, Clong, Ptr{Cdouble}, Clong, Clong),
            excinfo, t, C_NULL, C_NULL, nitems, bitsize,
            base, nitems, nitems * bitsize)
    end
end

@testset "Broken Function ccall + @view" begin
    a = rand(10)
    expected_grad_a = (nothing, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    grad_a = gradient(Reverse, func_ccall, Const(0.0), a)
    @test expected_grad_a == grad_a


    errstream = joinpath(tempdir(), "stdout.txt")
    err_llvmir = nothing
    b = @view a[1:5]

    redirect_stdio(stdout=errstream, stderr=errstream, stdin=devnull) do
        try
            gradient(Reverse, func_ccall, Const(0.0), b)
        catch e
            err_llvmir = e
            #    finally
            #        redirect_stdout(old_stdout)
        end

        @test err_llvmir !== nothing
        @test occursin("Broken function", err_llvmir.info)
    end

    errtxt = read(errstream, String)
    @test occursin("Called function is not the same type as the call!", errtxt)
end
