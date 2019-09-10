# Enzyme.jl

A Julia package for using [`Enzyme`](https://github.com/wsmoses/Enzyme)

## Installation process
During private pre-release.

### Dependencies
- `LLVM#master`

### Option A: Binary release
1. Download `fetch` from https://github.com/gruntwork-io/fetch/releases
2. Add to `PATH` and make executable
3. Create a Github token and export it as the environment variable `GITHUB_OAUTH_TOKEN`
4. `julia --project=Enzyme -e "using Pkg; pkg"dev BinaryProvider"; pkg"build Enzyme"`

### Option B: Manual build
1. Have a source-build of Julia available
2. Compile `Enzyme` against the LLVM provided by Julia
  - `cmake -DLLVM_DIR=${JULIA_HOME}/usr/lib/cmake/llvm`
3. Set environment variable: `ENZYME_PATH` to directory containing `LLVMEnzyme`

#### Example:
```
git clone https://github.com/JuliaLang/julia
mkdir -p builds/julia
export JULIA_HOME=`pwd`/builds/julia
make -C julia O=${JULIA_HOME} configure
make -C builds/julia -j
git clone https://github.com/wsmoses/Enzyme
mkdir -p builds/enzyme
pushd builds/enzyme
cmake -DLLVM_DIR=${JULIA_HOME}/usr/lib/cmake/llvm ../../Enzyme
make -j
popd
export ENZYME_PATH=`pwd`/builds/enzyme/Enzyme
```
