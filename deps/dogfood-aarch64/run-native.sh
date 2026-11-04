#!/usr/bin/env bash
# Invoked only by the reviewed stage-native.sh payload. Bash 3.2 (macOS /bin/bash) compatible.
set -euo pipefail
test "$#" -eq 1
run=$1
recipe="$run/recipe/deps/dogfood-aarch64"
# The dogfood source and its declared backend come from the staged provenance, checked below.
source_sha=$(sed -n 's/^source_commit=//p' "$run/PROVENANCE")
backend_sha=$(sed -n 's/^backend_commit=//p' "$run/PROVENANCE")
[[ "$source_sha" =~ ^[0-9a-f]{40}$ ]]
[[ "$backend_sha" =~ ^[0-9a-f]{40}$ ]]
# Kitware's official CMake 3.28.3 macOS build (the Mac VM's version), pinned by its published SHA-256.
cmake_url=https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-macos-universal.tar.gz
cmake_sha256=d9e2c22fec920a4d1f6b0d0683c035d799475c179c91e41e1a7fbfab610a0305
printf 'NATIVE_HOST %s %s macOS %s bash %s\n' "$(uname -s)" "$(uname -m)" "$(sw_vers -productVersion 2>/dev/null || echo unknown)" "$BASH_VERSION"
test "$(uname -s)" = Darwin
test "$(uname -m)" = arm64

# Report every prerequisite before failing, so one run names everything that is missing.
missing=
for tool in julia git shasum curl tar make c++; do
    if path=$(command -v "$tool"); then
        printf 'TOOL %s %s\n' "$tool" "$path"
    else
        printf 'TOOL_MISSING %s\n' "$tool"
        missing="$missing $tool"
    fi
done
printf 'XCODE_SELECT %s\n' "$(xcode-select -p 2>&1 || true)"
# git, make and c++ in /usr/bin are shims that work only with the Command Line Tools installed.
for tool in git make c++; do
    if command -v "$tool" >/dev/null 2>&1; then
        if out=$("$tool" --version 2>&1); then
            printf 'TOOL_VERSION %s %s\n' "$tool" "$(printf '%s\n' "$out" | head -1)"
        else
            printf 'TOOL_BROKEN %s %s\n' "$tool" "$(printf '%s\n' "$out" | head -1)"
            missing="$missing $tool"
        fi
    fi
done
if sdk=$(xcrun --show-sdk-path 2>/dev/null); then
    printf 'SDK %s\n' "$sdk"
    export SDKROOT="$sdk"
fi

# The aarch64 consumers resolve with Julia 1.10.12, and this recipe builds for its LLVM 15.
mkdir "$run/bin"
selected=
if command -v julia >/dev/null 2>&1; then
    launcher=$(command -v julia)
    for sel in +1.10.12 +1.10 default; do
        if test "$sel" = default; then
            v=$("$launcher" --startup-file=no -e 'print(VERSION, " ", Base.libllvm_version)' 2>/dev/null) || continue
        else
            v=$("$launcher" "$sel" --startup-file=no -e 'print(VERSION, " ", Base.libllvm_version)' 2>/dev/null) || continue
        fi
        printf 'JULIA_CHANNEL %s %s\n' "$sel" "$v"
        case "$v" in
            "1.10."*" 15."*) selected=$sel; break ;;
        esac
    done
fi
if test -z "$selected"; then
    printf 'JULIA_MISSING no Julia 1.10 with LLVM 15 on this Mac\n'
    missing="$missing julia-1.10"
elif test "$selected" = default; then
    printf '#!/bin/sh\nexec "%s" "$@"\n' "$launcher" >"$run/bin/julia"
else
    printf '#!/bin/sh\nexec "%s" %s "$@"\n' "$launcher" "$selected" >"$run/bin/julia"
fi
if test -n "$missing"; then
    printf 'NATIVE_FAILED missing:%s\n' "$missing"
    exit 1
fi
chmod +x "$run/bin/julia"
export PATH="$run/bin:$PATH"

if ! command -v cmake >/dev/null 2>&1; then
    printf 'CMAKE_FETCH %s\n' "$cmake_url"
    curl -sSL --fail -o "$run/cmake.tar.gz" "$cmake_url"
    printf '%s  %s\n' "$cmake_sha256" "$run/cmake.tar.gz" | shasum -a 256 -c -
    tar -xzf "$run/cmake.tar.gz" -C "$run"
    ln -s "$run/cmake-3.28.3-macos-universal/CMake.app/Contents/bin/cmake" "$run/bin/cmake"
fi
printf 'CMAKE %s %s\n' "$(command -v cmake)" "$(cmake --version | head -1)"

export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ENZYME_EXPECT_ARCH=aarch64
unset JULIA_PROJECT KB_LIBENZYME_DOGFOOD_ROOT
julia --startup-file=no --project=@stdlib -e \
    'println("NATIVE_JULIA ", VERSION, " LLVM ", Base.libllvm_version, " ", Sys.MACHINE); Sys.ARCH == :aarch64 || error("Native aarch64 required"); Base.libllvm_version.major == 15 || error("Recipe requires LLVM 15; report this runtime before selecting a new build recipe")'
(cd "$run" && shasum -a 256 -c SHA256SUMS)
cat "$run/PROVENANCE"
test "$(git get-tar-commit-id <"$run/source.tar")" = "$source_sha"
mkdir "$run/source"
tar -xf "$run/source.tar" -C "$run/source"
grep -qx "commit = \"$backend_sha\"" "$run/source/deps/dogfood_libenzyme.toml"
export KB_DOGFOOD_BACKEND="$backend_sha" ENZYME_EXPECT_BACKEND="$backend_sha"
export KB_JOB_STATE="$run/build-state" KB_JOB_WORKSPACE="$run"
export KB_JOB_CPU_SHARE=4
export KB_LIBENZYME_BUILD_DEST="$HOME/.local/state/kb-agents/libenzyme-dogfood"
out="$KB_LIBENZYME_BUILD_DEST/$backend_sha/llvm15"
# Preserve an existing build until its provenance has been reviewed.
test ! -e "$out"
bash "$recipe/build-llvm15.sh"
julia --startup-file=no --project=@stdlib "$recipe/setup.jl" \
    "$run/env" "$run/source" "$run/resolver.jl" | tee "$run/setup.log"
grep -Fx "$(printf 'KB-DOGFOOD-LIBENZYME\t%s\tllvm15\t%s' "$backend_sha" "$out")" "$run/setup.log"
julia --startup-file=no --project="$run/env" "$recipe/canaries.jl" | tee "$run/canaries.log"
grep -Fx CANARIES_COMPLETE "$run/canaries.log"
(cd "$out" && shasum -a 256 -c SHA256SUMS)
printf 'NATIVE_COMPLETE source=%s backend=%s llvm=15\n' "$source_sha" "$backend_sha" | tee "$run/VALIDATED"
test -s "$run/VALIDATED"
