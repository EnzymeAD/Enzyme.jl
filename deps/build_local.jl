# Invoke with
# `julia --project=deps deps/build_local.jl [path-to-enzyme]`

# the pre-built Enzyme_jll might not be loadable on this platform
Enzyme_jll = Base.UUID("7cc45869-7501-5eee-bdea-0790c847d4ef")

using Pkg, Scratch, Preferences, Libdl

BUILD_TYPE = "RelWithDebInfo"
BCLoad = true

source_dir = nothing
branch = nothing

args = (ARGS...,)
while length(args) > 0
    global args
    global branch
    global source_dir
    global BUILD_TYPE
    global BCLoad
    if length(args) >= 2 && args[1] == "--branch"
        branch = args[2]
        args = (args[3:end]...,)
        continue
    end
    if length(args) >= 1 && args[1] == "--debug"
        BUILD_TYPE = "Debug"
        args = (args[2:end]...,)
        continue
    end
    if length(args) >= 1 && args[1] == "--nobcload"
        BCLoad = false
        args = (args[2:end]...,)
        continue
    end
    if source_dir == nothing
        source_dir = args[1]
        args = (args[2:end]...,)
        continue
    end
    @show args
    @assert length(args) == 0
    break
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
llvm_pkg_version = "$(Base.libllvm_version.major).$(Base.libllvm_version.minor)"
LLVM = if llvm_assertions
    Pkg.add(name="LLVM_full_assert_jll", version=llvm_pkg_version)
    using LLVM_full_assert_jll
    LLVM_full_assert_jll
else
    Pkg.add(name="LLVM_full_jll", version=llvm_pkg_version)
    using LLVM_full_jll
    LLVM_full_jll
end
LLVM_DIR = joinpath(LLVM.artifact_dir, "lib", "cmake", "llvm")
LLVM_VER_MAJOR = Base.libllvm_version.major

# 1. Get a scratch directory
scratch_dir = get_scratch!(Enzyme_jll, "build_$(LLVM_VER_MAJOR)_$(llvm_assertions)")
isdir(scratch_dir) && rm(scratch_dir; recursive=true)


# Build!
@info "Building" source_dir scratch_dir LLVM_DIR BUILD_TYPE
run(`cmake -DLLVM_DIR=$(LLVM_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DENZYME_EXTERNAL_SHARED_LIB=ON -B$(scratch_dir) -S$(source_dir)`)

if BCLoad
  run(`cmake --build $(scratch_dir) --parallel $(Sys.CPU_THREADS) -t Enzyme-$(LLVM_VER_MAJOR) EnzymeBCLoad-$(LLVM_VER_MAJOR)`)
else
  run(`cmake --build $(scratch_dir) --parallel $(Sys.CPU_THREADS) -t Enzyme-$(LLVM_VER_MAJOR)`)
end

# Discover built libraries
built_libs = filter(readdir(joinpath(scratch_dir, "Enzyme"))) do file
    endswith(file, ".$(Libdl.dlext)") && startswith(file, "lib")
end

lib_path = joinpath(scratch_dir, "Enzyme", only(built_libs))
isfile(lib_path) || error("Could not find library $lib_path in build directory")

# Tell Enzyme_jll to load our library instead of the default artifact one
set_preferences!(
    joinpath(dirname(@__DIR__), "LocalPreferences.toml"),
    "Enzyme_jll",
    "libEnzyme_path" => lib_path,
    force=true,
)

if BCLoad
built_libs = filter(readdir(joinpath(scratch_dir, "BCLoad"))) do file
    endswith(file, ".$(Libdl.dlext)") && startswith(file, "lib")
end

libBC_path = joinpath(scratch_dir, "BCLoad", only(built_libs))
isfile(libBC_path) || error("Could not find library $libBC_path in build directory")
# Tell Enzyme_jll to load our library instead of the default artifact one
set_preferences!(
    joinpath(dirname(@__DIR__), "LocalPreferences.toml"),
    "Enzyme_jll",
    "libEnzymeBCLoad_path" => libBC_path;
    force=true,
)
end
