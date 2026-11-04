#!/usr/bin/env bash
# Run on the desktop via a reviewed command-on-GO. No build runs here.
set -euo pipefail
test "$#" -eq 3
repo=$1
recipe_sha=$2
# The dogfood commit whose declared libEnzyme build is built and checked.
source_sha=$3
# The KB checkout and exact commit whose lib-resolve.sh emits the resolver program.
resolver_repo=${KB_RESOLVER_REPO:?set KB_RESOLVER_REPO to the checkout holding lib-resolve.sh}
resolver_sha=${KB_RESOLVER_SHA:?set KB_RESOLVER_SHA to its exact commit}
[[ "$recipe_sha" =~ ^[0-9a-f]{40}$ ]]
[[ "$source_sha" =~ ^[0-9a-f]{40}$ ]]
[[ "$resolver_sha" =~ ^[0-9a-f]{40}$ ]]
backend_sha=$(git -C "$repo" show "$source_sha:deps/dogfood_libenzyme.toml" | sed -n 's/^commit = "\([0-9a-f]\{40\}\)"$/\1/p')
[[ "$backend_sha" =~ ^[0-9a-f]{40}$ ]]

# Check the control link before staging anything. A refused connection needs
# the user to restore maclink; never retry it in a loop.
ssh -n -o BatchMode=yes -o ConnectTimeout=15 mac 'test "$(uname -s)" = Darwin'
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir "$stage/recipe"
git -C "$repo" archive "$source_sha" > "$stage/source.tar"
test "$(git get-tar-commit-id < "$stage/source.tar")" = "$source_sha"
git -C "$repo" archive "$recipe_sha" deps/dogfood-aarch64 | tar -xf - -C "$stage/recipe"
git -C "$resolver_repo" show "$resolver_sha:lib-resolve.sh" > "$stage/lib-resolve.sh"
# Only emit the maintained public resolver program. Supplying its registry
# variables avoids unrelated repository discovery and Bash dependencies.
GITHUB_DIR=/unused LOCAL_PKG_ORDER='' bash -c \
    'set -euo pipefail; source "$1"; dogfood_libenzyme_program' \
    _ "$stage/lib-resolve.sh" > "$stage/resolver.jl"
test -s "$stage/resolver.jl"
printf 'source_commit=%s\nbackend_commit=%s\nrecipe_commit=%s\nresolver_commit=%s\n' \
    "$source_sha" "$backend_sha" "$recipe_sha" "$resolver_sha" > "$stage/PROVENANCE"
(cd "$stage" && shasum -a 256 source.tar resolver.jl PROVENANCE > SHA256SUMS)

# The payload contains only named repository sources and the emitted program.
# The Mac returns build/provenance/canary logs, never application data. Each
# attempt gets its own run directory, and the remote exit is printed explicitly.
tar -cf - -C "$stage" source.tar resolver.jl PROVENANCE SHA256SUMS recipe |
    ssh -o BatchMode=yes -o ConnectTimeout=15 mac \
        'set -eu; export PATH="$HOME/homebrew/bin:$HOME/.juliaup/bin:$PATH"; run="$HOME/.local/state/kb-agents/libenzyme-dogfood-runs/'"$recipe_sha"'-$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$(dirname "$run")"; mkdir "$run"; tar -xf - -C "$run"; echo "NATIVE_RUN_DIR $run"; rc=0; bash "$run/recipe/deps/dogfood-aarch64/run-native.sh" "$run" || rc=$?; echo "NATIVE_REMOTE_EXIT $rc"; exit "$rc"' 2>&1 |
    tee "$stage/native.log"
# Confirm from the log itself, not from a pipeline status: a run that stopped
# early once came back as exit 0 (decision 0o7aqwh).
grep -qx 'NATIVE_REMOTE_EXIT 0' "$stage/native.log" || { echo NATIVE_NOT_CONFIRMED; exit 1; }
grep -q '^NATIVE_COMPLETE ' "$stage/native.log" || { echo NATIVE_NOT_CONFIRMED; exit 1; }
echo NATIVE_CONFIRMED
