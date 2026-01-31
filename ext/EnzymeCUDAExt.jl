module EnzymeCUDAExt

using CUDA, Enzyme

@inline Enzyme.Compiler.is_arrayorvararg_ty(::Type{CUDA.CuRefValue{T}}) where T = true
@inline Enzyme.Compiler.ptreltype(::Type{CUDA.CuRefValue{T}}) where T = T

@inline Enzyme.Compiler.is_arrayorvararg_ty(::Type{CUDA.CuPtr{T}}) where T = true
@inline Enzyme.Compiler.ptreltype(::Type{CUDA.CuPtr{T}}) where T = T
end # module
