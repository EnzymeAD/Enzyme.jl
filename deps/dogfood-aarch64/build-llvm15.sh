#!/usr/bin/env bash
set -euo pipefail
# The libEnzyme commit to build: the `commit` of the dogfood declaration (deps/dogfood_libenzyme.toml).
backend_sha=${KB_DOGFOOD_BACKEND:?set KB_DOGFOOD_BACKEND to the declared libEnzyme commit}
[[ "$backend_sha" =~ ^[0-9a-f]{40}$ ]]
build_state=${KB_JOB_STATE:-$HOME/.local/state/kb-agents/libenzyme-dogfood-build}/enzyme-aarch64
install_root=${KB_LIBENZYME_BUILD_DEST:-$build_state/install}
evidence_dir=${KB_JOB_WORKSPACE:-$PWD}/verification/enzyme-aarch64
mkdir -p "$build_state" "$evidence_dir"
# coreutils on the Linux VM, perl's shasum on macOS; both read and check the same SHA256SUMS format.
sum256() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi
}
exec > >(tee "$evidence_dir/build.log") 2>&1
printf 'BUILD_START %s\n' "$(date -u +%FT%TZ)"
uname -srm
julia --startup-file=no -e 'VERSION >= v"1.10" || error("Julia 1.10+ required"); Base.libllvm_version.major == 15 || error("This build recipe is for LLVM 15"); println("Julia ",VERSION," LLVM ",Base.libllvm_version," machine ",Sys.MACHINE)'
for tool in cmake git; do command -v "$tool"; done
cmake --version
cat >"$build_state/Artifacts.toml" <<'ARTIFACTS'
[[LLVM_full]]
arch = "aarch64"
git-tree-sha1 = "3a0cf4532a9b50215c6976dfa9bbfc0f5a149e5d"
os = "macos"
    [[LLVM_full.download]]
    sha256 = "4b1231b0c3199daade18c30a12bed6625cafe016470c02451ceb27775e7efe03"
    url = "https://github.com/JuliaBinaryWrappers/LLVM_full_jll.jl/releases/download/LLVM_full-v15.0.7+14/LLVM_full.v15.0.7.aarch64-apple-darwin.tar.gz"
[[LLVM_full]]
arch = "aarch64"
cxxstring_abi = "cxx11"
git-tree-sha1 = "7a7c5937994fc65ea9701c9e978ddaad5719d94c"
libc = "glibc"
os = "linux"
    [[LLVM_full.download]]
    sha256 = "6e1a6e76ae642383ea9cdada302cdc5f71bbf4c4d9d3cead72671afb7903a955"
    url = "https://github.com/JuliaBinaryWrappers/LLVM_full_jll.jl/releases/download/LLVM_full-v15.0.7+14/LLVM_full.v15.0.7.aarch64-linux-gnu-cxx11.tar.gz"
ARTIFACTS
# Exact platform entries from LLVM_full_jll 15.0.7+14, using Julia's shared artifact store.
julia --startup-file=no --project=@stdlib -e 'using Pkg.Artifacts, Base.BinaryPlatforms; p=HostPlatform(); Sys.islinux() && (p["cxxstring_abi"]="cxx11"); f=ARGS[1]; ensure_artifact_installed("LLVM_full",f; platform=p); h=artifact_hash("LLVM_full",f; platform=p); h === nothing && error("Unsupported platform"); write(ARGS[2],artifact_path(h)); println("LLVM_ARTIFACT ",h)' "$build_state/Artifacts.toml" "$build_state/llvm-root"
llvm_root=$(cat "$build_state/llvm-root")
# LLVM_full_jll ships llvm-config under tools/, not bin/; cmake needs only the config files.
test -f "$llvm_root/lib/cmake/llvm/LLVMConfig.cmake"
if test -x "$llvm_root/tools/llvm-config"; then "$llvm_root/tools/llvm-config" --version; fi
src="$build_state/source-$backend_sha"
if ! test -d "$src/.git"; then
    git init "$src"
    git -C "$src" fetch --depth=1 https://github.com/nsiccha/Enzyme.git "$backend_sha"
    git -C "$src" checkout --detach FETCH_HEAD
fi
test "$(git -C "$src" rev-parse HEAD)" = "$backend_sha"
test -z "$(git -C "$src" status --porcelain --untracked-files=no)"
build_dir="$build_state/build-$backend_sha-llvm15"
out="$install_root/$backend_sha/llvm15"
# Bound compiler fan-out to the job's CPU share and 5 GiB per compiler within 20 GiB.
jobs=${KB_JOB_CPU_SHARE:-4}
if test "$jobs" -gt 4; then jobs=4; fi
printf 'CONFIGURE jobs=%s source=%s llvm=%s\n' "$jobs" "$src" "$llvm_root"
# EnzymeBCLoad-15 exists only when CMake finds the artifact's Clang (ENZYME_CLANG, the
# default and gordito's lib15 setting); set it explicitly so a stale cache cannot turn it off.
cmake -S "$src/enzyme" -B "$build_dir" -DLLVM_DIR="$llvm_root/lib/cmake/llvm" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENZYME_ENABLE_BENCHMARKS=OFF -DENZYME_CLANG=ON -DENZYME_EXTERNAL_SHARED_LIB=ON
cmake --build "$build_dir" --parallel "$jobs" --target Enzyme-15 EnzymeBCLoad-15
ext=$(julia --startup-file=no --project=@stdlib -e 'using Libdl; print(Libdl.dlext)')
built_enzyme="$build_dir/Enzyme/libEnzyme-15.$ext"
built_bcload="$build_dir/BCLoad/libEnzymeBCLoad-15.$ext"
stripped=none
if test "$(uname -s)" = Linux; then
    # A VM job's outputs tarball is refused above 64 MiB; keep DWARF only when the pair fits.
    gz=$(cat "$built_enzyme" "$built_bcload" | gzip -c | wc -c)
    printf 'LIBRARIES_GZIPPED_BYTES %s\n' "$gz"
    if test "$gz" -gt 62914560; then
        printf 'STRIP_DEBUG the pair exceeds the 60 MiB transport budget\n'
        strip --strip-debug "$built_enzyme" "$built_bcload"
        stripped=debug
    fi
fi
# Create the install directory only after a successful build: the native runner
# refuses an existing one, so a failed compile must not leave it behind.
mkdir -p "$out"
cp "$built_enzyme" "$built_bcload" "$out/"
(cd "$out" && sum256 "libEnzyme-15.$ext" "libEnzymeBCLoad-15.$ext" > SHA256SUMS && sum256 -c SHA256SUMS)
{
    printf 'backend_commit=%s\nllvm_root=%s\ncompiler_jobs=%s\nstripped=%s\n' "$backend_sha" "$llvm_root" "$jobs" "$stripped"
    uname -srm
    julia --startup-file=no -e 'println("julia=",VERSION," llvm=",Base.libllvm_version," machine=",Sys.MACHINE)'
    cmake --version
    git -C "$src" show -s --format=fuller HEAD
    date -u +%FT%TZ
} >"$out/BUILD_INFO"
cp "$out/SHA256SUMS" "$out/BUILD_INFO" "$build_state/Artifacts.toml" "$build_dir/CMakeCache.txt" "$evidence_dir/"
if test "$(uname -s)" = Linux; then
    # Registration of a VM build (POST /mac_jobs/libenzyme) reads a job's collected outputs.
    reg="$evidence_dir/install/$backend_sha/llvm15"
    rm -rf "$evidence_dir/install"
    mkdir -p "$reg"
    cp "$out/libEnzyme-15.$ext" "$out/libEnzymeBCLoad-15.$ext" "$out/SHA256SUMS" "$out/BUILD_INFO" "$reg/"
    (cd "$reg" && sum256 -c SHA256SUMS)
    printf 'BUILD_OUTPUTS %s\n' "$reg"
fi
printf 'BUILD_INSTALL %s\n' "$out"
printf 'BUILD_COMPLETE %s\n' "$(date -u +%FT%TZ)"
