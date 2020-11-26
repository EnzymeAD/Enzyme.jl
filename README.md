# <img src="https://enzyme.mit.edu/logo.svg" width="75" align=left> The Enzyme High-Performance Automatic Differentiator of LLVM

This is a package containing the Julia bindings for [Enzyme](https://github.com/wsmoses/enzyme). This is very much a work in progress and bug reports/discussion is greatly appreciated!

Enzyme is a plugin that performs automatic differentiation (AD) of statically analyzable LLVM. It is highly-efficient and its ability perform AD on optimized code allows Enzyme to meet or exceed the performance of state-of-the-art AD tools.

Enzyme.jl can be installed in the usual way Julia packages are installed
```
] add Enzyme
```

Enzyme.jl can be used by calling `autodiff` on a function to be differentiated as shown below:

```julia
using Enzyme

f1(x) = x*x
@test autodiff(f1, Active(1.0)) ≈ 2.0
```

More information on installing and using Enzyme can be found on our website: [https://enzyme.mit.edu](https://enzyme.mit.edu).

To get involved or if you have questions, please join our [mailing list](https://groups.google.com/d/forum/enzyme-dev).

If using this code in an academic setting, please cite the following paper to appear in NeurIPS 2020

```
@incollection{enzymeNeurips,
title = {Instead of Rewriting Foreign Code for Machine Learning, Automatically Synthesize Fast Gradients},
author = {Moses, William S. and Churavy, Valentin},
booktitle = {Advances in Neural Information Processing Systems 33},
year = {2020},
note = {To appear in},
}
```

## Development of Enzyme and Enzyme.jl together

Normally Enzyme.jl downloads and install Enzyme for the user automatically since Enzyme needs to be built against
Julia bundeled LLVM. In case that you are making updates to Enzyme and want to test them against Enzyme.jl the instructions
below should help you get started.

Start Julia in your development copy of Enzyme.jl

```bash
~/s/Enzyme (master)> julia --project=.
```

Then create a development copy of Enzyme_jll and activate it within.

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
