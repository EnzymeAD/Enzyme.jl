# Enzyme developer documentation

## Development of Enzyme and Enzyme.jl together (recommended)

Normally Enzyme.jl downloads and installs [Enzyme](https://github.com/EnzymeAD/enzyme) for the user automatically since Enzyme needs to be built against
Julia bundled LLVM. In case that you are making updates to Enzyme and want to test them against Enzyme.jl the instructions
below should help you get started.

Start Julia in your development copy of Enzyme.jl and initialize the deps project

```bash
~/s/Enzyme.jl (master)> julia --project=deps
```

```julia-repl
julia> # Hit the `]` key to enter package repl.
(deps) pkg> instantiate
```

We can now build a custom version of Enzyme for use in Enzyme.jl. To build the latest commit on the main branch of Enzyme, run the following.
It may take a few minutes to compile fully.

```bash
~/s/Enzyme.jl (master)> julia --project=deps deps/build_local.jl
```

You will now find a file LocalPreferences.toml which has been generated and contains a path to the new Enzyme\_jll binary you have built.
To use your Enzyme\_jll instead of the default shipped by Enzyme.jl, ensure that this file is at the root of any Julia project you wish
to test it with *and* that the Julia project has Enzyme\_jll as an explicit dependency. Note that an indirect dependency here is not
sufficient (e.g. just because a project depends on Enzyme.jl, which depends on Enzyme\_jll, does not mean that your project will pick up
this file unless you also add a direct dependency to Enzyme\_jll).

To test whether your project found the custom version of Enzyme\_jll, you can inspect the path of the Enzyme\_jll library in use as follows.

```bash
~/my/project.jl (master)> julia --project=.
```

```julia-repl
julia> using Enzyme_jll
julia> Enzyme_jll.libEnzyme_path
"${JULIA_PKG_DEVDIR}/Enzyme_jll/override/lib/LLVMEnzyme-9.so"
```

This should correspond to the path in the LocalPreferences.toml you just generated.

Note that your system can have only one custom built Enzyme\_jll at a time. If you build one version for one version of Enzyme or Julia
and later build a new version of Enzyme, it removes the old build.

Note that Julia versions are tightly coupled and you cannot use an Enzyme\_jll built for one version of Julia for another version of Julia.

The same script can also be used to build Enzyme\_jll for a branch other than main as follows.

```bash
~/s/Enzyme.jl (master)> julia --project=deps deps/build_local.jl --branch mybranch
```

It can also be used to build Enzyme\_jll from a local copy of Enzyme on your machine, which does not need to be committed to git.

```bash
~/s/Enzyme.jl (master)> julia --project=deps deps/build_local.jl ../path/to/Enzyme
```

## Development of Enzyme and Enzyme.jl together (manual)
Start Julia in your development copy of Enzyme.jl

```bash
~/s/Enzyme.jl (master)> julia --project=.
```

Then create a development copy of Enzyme\_jll and activate it within.

```julia-repl
julia> using Enzyme_jll
julia> Enzyme_jll.dev_jll()
[ Info: Enzyme_jll dev'ed out to ${JULIA_PKG_DEVDIR}/Enzyme_jll with pre-populated override directory
(Enzyme) pkg> dev Enzyme_jll
Path `${JULIA_PKG_DEVDIR}/Enzyme_jll` exists and looks like the correct package. Using existing path.
```

After restarting Julia:

```julia-repl
julia> Enzyme_jll.dev_jll()
julia> Enzyme_jll.libEnzyme_path
"${JULIA_PKG_DEVDIR}/Enzyme_jll/override/lib/LLVMEnzyme-9.so"
```

On your machine `${JULIA_PKG_DEVDIR}` most likely corresponds to `~/.julia/dev`.
Now we can inspect `"${JULIA_PKG_DEVDIR}/Enzyme_jll/override/lib` and see that there is a copy of `LLVMEnzyme-9.so`,
which we can replace with a symbolic link or a copy of a version of Enzyme.


## Building Enzyme against Julia's LLVM.

Depending on how you installed Julia the LLVM Julia is using will be different.

1. Download from julialang.org (Recommended)
2. Manual build on your machine
3. Uses a pre-built Julia from your system vendor (Not recommended)

To check what LLVM Julia is using use:
```
julia> Base.libllvm_version_string
"9.0.1jl"
```

If the LLVM version ends in a `jl` you are likely using the private LLVM.

In your source checkout of Enzyme:

```bash
mkdir build-jl
cd build-jl
```

### Prebuilt binary from julialang.org

```
LLVM_MAJOR_VER=`julia -e "print(Base.libllvm_version.major)"`
julia -e "using Pkg; pkg\"add LLVM_full_jll@${LLVM_MAJOR_VER}\""
LLVM_DIR=`julia -e "using LLVM_full_jll; print(LLVM_full_jll.artifact_dir)"`
echo "LLVM_DIR=$LLVM_DIR"
cmake ../enzyme/ -G Ninja -DENZYME_EXTERNAL_SHARED_LIB=ON -DLLVM_DIR=${LLVM_DIR} -DLLVM_EXTERNAL_LIT=${LLVM_DIR}/tools/lit/lit.py
```

### Manual build of Julia
```
cmake ../enzyme/ -G Ninja -DENZYME_EXTERNAL_SHARED_LIB=ON -DLLVM_DIR=${PATH_TO_BUILDDIR_OF_JULIA}/usr/lib/cmake/llvm/
```
