# Invoke with
# `julia --project=deps deps/build_local.jl`

# the pre-built Enzyme_jll might not be loadable on this platform
Enzyme_jll = Base.UUID("7cc45869-7501-5eee-bdea-0790c847d4ef")

using Pkg, Scratch, Preferences, Libdl

# 1. Ensure that an appropriate LLVM_full_jll is installed
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

# 2. Get a scratch directory
scratch_dir = get_scratch!(Enzyme_jll, "build")
isdir(scratch_dir) && rm(scratch_dir; recursive=true)
if length(ARGS) == 1
    source_dir = ARGS[1]
else
    scratch_src_dir = get_scratch!(Enzyme_jll, "src")
    cd(scratch_src_dir) do
        if !isdir("Enzyme")
            run(`git clone https://github.com/wsmoses/Enzyme`)
        end
        run(`git -C Enzyme pull`)
    end
    source_dir = joinpath(scratch_src_dir, "Enzyme", "enzyme")
end

LLVM_VER_MAJOR = Base.libllvm_version.major

# Build!
@info "Building" source_dir scratch_dir LLVM_DIR
run(`cmake -DLLVM_DIR=$(LLVM_DIR) -DENZYME_EXTERNAL_SHARED_LIB=ON -B$(scratch_dir) -S$(source_dir)`)
run(`cmake --build $(scratch_dir) --parallel $(Sys.CPU_THREADS) -t Enzyme-$(LLVM_VER_MAJOR)`)

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
    "libEnzyme_path" => lib_path;
    force=true,
)