# Invoke with
# `julia --project=deps deps/build_local.jl [path-to-enzyme]`

# the pre-built Enzyme_jll might not be loadable on this platform
Enzyme_jll = Base.UUID("7cc45869-7501-5eee-bdea-0790c847d4ef")

using Pkg, Scratch, Preferences, Libdl

# 1. Get a scratch directory
scratch_dir = get_scratch!(Enzyme_jll, "build")
isdir(scratch_dir) && rm(scratch_dir; recursive=true)

source_dir = nothing
branch = nothing
if length(ARGS) == 2 
    @assert ARGS[1] == "--branch"
    branch = ARGS[2]
    source_dir = nothing
elseif length(ARGS) == 1
    source_dir = ARGS[1]
end

if branch === nothing
    branch = "main"
end

if source_dir === nothing
    scratch_src_dir = get_scratch!(Enzyme_jll, "src")
    cd(scratch_src_dir) do
        if !isdir("Enzyme")
            run(`git clone https://github.com/wsmoses/Enzyme`)
        end
        run(`git -C Enzyme fetch`)
        run(`git -C Enzyme checkout origin/$(branch)`)
    end
    source_dir = joinpath(scratch_src_dir, "Enzyme", "enzyme")
end

# 2. Ensure that an appropriate LLVM_full_jll is installed
Pkg.activate(; temp=true)
llvm_assertions = try
    cglobal((:_ZN4llvm24DisableABIBreakingChecksE, Base.libllvm_path()), Cvoid)
    false
catch
    true
end
LLVM = if llvm_assertions
    Pkg.add(name="LLVM_full_assert_jll", version=Base.libllvm_version)
    using LLVM_full_assert_jll
    LLVM_full_assert_jll
else
    Pkg.add(name="LLVM_full_jll", version=Base.libllvm_version)
    using LLVM_full_jll
    LLVM_full_jll
end
LLVM_DIR = joinpath(LLVM.artifact_dir, "lib", "cmake", "llvm")
LLVM_VER_MAJOR = Base.libllvm_version.major

# Build!
@info "Building" source_dir scratch_dir LLVM_DIR
run(`cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR=$(LLVM_DIR) -DENZYME_EXTERNAL_SHARED_LIB=ON -B$(scratch_dir) -S$(source_dir)`)
run(`cmake --build $(scratch_dir) --parallel $(Sys.CPU_THREADS) -t Enzyme-$(LLVM_VER_MAJOR) EnzymeBCLoad-$(LLVM_VER_MAJOR)`)

# Discover built libraries
built_libs = filter(readdir(joinpath(scratch_dir, "Enzyme"))) do file
    endswith(file, ".$(Libdl.dlext)") && startswith(file, "lib")
end

lib_path = joinpath(scratch_dir, "Enzyme", only(built_libs))
isfile(lib_path) || error("Could not find library $lib_path in build directory")

built_libs = filter(readdir(joinpath(scratch_dir, "BCLoad"))) do file
    endswith(file, ".$(Libdl.dlext)") && startswith(file, "lib")
end

libBC_path = joinpath(scratch_dir, "BCLoad", only(built_libs))
isfile(libBC_path) || error("Could not find library $libBC_path in build directory")

# Tell Enzyme_jll to load our library instead of the default artifact one
set_preferences!(
    joinpath(dirname(@__DIR__), "LocalPreferences.toml"),
    "Enzyme_jll",
    "libEnzyme_path" => lib_path,
    "libEnzymeBCLoad_path" => libBC_path;
    force=true,
)
